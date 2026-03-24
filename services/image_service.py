import os
import time
import gc
import torch
import asyncio
import aiohttp
import warnings
from typing import Optional
from core.executors import HEAVY_EXECUTOR
from core.logging_config import get_logger
from services.hardware_service import HardwareService

logger = get_logger(__name__)
_hardware = HardwareService()

try:
    from transformers import BitsAndBytesConfig, Mistral3ForConditionalGeneration
    from diffusers import Flux2Pipeline, Flux2Transformer2DModel, StableDiffusionXLPipeline
except ImportError:
    Flux2Pipeline = None
    Flux2Transformer2DModel = None
    Mistral3ForConditionalGeneration = None
    StableDiffusionXLPipeline = None

FLUX2_REPO = "diffusers/FLUX.2-dev-bnb-4bit"
SDXL_REPO = "stabilityai/stable-diffusion-xl-base-1.0"

class ImageService:
    def __init__(self, **kwargs):
        self.pipeline = None
        self.current_engine = None # Tracks 'FLUX' or 'SDXL'
        self.output_dir = "outputs/images"
        self._last_activity = 0
        self._unload_task = None
        self._generation_lock = asyncio.Lock()
        os.makedirs(self.output_dir, exist_ok=True)

    def _get_vram_usage(self) -> int:
        return _hardware.get_vram_usage_mb()

    def _purge_vram(self):
        """Clears the 3090 Ti completely before swapping engines."""
        if self.pipeline is not None:
            logger.debug("Purging %s from VRAM...", self.current_engine)
            self.pipeline = None
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    def _load_flux(self):
        """Loads the heavy FLUX.2 engine."""
        if self.current_engine == "FLUX" and self.pipeline is not None:
            return
        self._purge_vram()
        logger.debug("Loading FLUX.2 (4-bit) to GPU...")
        
        transformer = Flux2Transformer2DModel.from_pretrained(
            FLUX2_REPO, subfolder="transformer", torch_dtype=torch.bfloat16,
            device_map={"": 0}, low_cpu_mem_usage=True, local_files_only=True
        )
        text_encoder = Mistral3ForConditionalGeneration.from_pretrained(
            FLUX2_REPO, subfolder="text_encoder", torch_dtype=torch.bfloat16,
            device_map={"": 0}, low_cpu_mem_usage=True, local_files_only=True, tie_word_embeddings=False
        )
        self.pipeline = Flux2Pipeline.from_pretrained(
            FLUX2_REPO, transformer=transformer, text_encoder=text_encoder, torch_dtype=torch.bfloat16
        )
        self.pipeline.vae.to("cuda")
        self.current_engine = "FLUX"

    def _load_sdxl(self):
        """Loads the high-speed SDXL engine."""
        if self.current_engine == "SDXL" and self.pipeline is not None:
            return
        self._purge_vram()
        logger.debug("Loading SDXL (FP16) to GPU...")
        
        self.pipeline = StableDiffusionXLPipeline.from_pretrained(
            SDXL_REPO, torch_dtype=torch.float16, variant="fp16", use_safetensors=True
        ).to("cuda")
        self.current_engine = "SDXL"

    async def generate_image(self, prompt: str, progress_callback=None) -> Optional[str]:
        """Entry for !draw (FLUX)"""
        return await self._run_gen(prompt, "FLUX", progress_callback)

    async def generate_sdxl(self, prompt: str, progress_callback=None) -> Optional[str]:
        """Entry for !fast (SDXL)"""
        return await self._run_gen(prompt, "SDXL", progress_callback)

    async def _run_gen(self, prompt, mode, callback):
        filename = f"kiba_{mode.lower()}_{int(time.time())}.png"
        filepath = os.path.join(self.output_dir, filename)
        try:
            loop = asyncio.get_running_loop()
            async with self._generation_lock:
                return await loop.run_in_executor(HEAVY_EXECUTOR, self._generate_sync, prompt, filepath, mode, callback)
        except Exception as e:
            logger.error("%s generation failed: %s", mode, e)
            return None

    def _generate_sync(self, prompt, filepath, mode, callback):
            try: 
                main_loop = asyncio.get_event_loop()
            except Exception:
                main_loop = None

            if mode == "FLUX":
                self._load_flux()
                steps = 14
            else:
                self._load_sdxl()
                steps = 30

            # THREAD-SAFE RATE LIMITING: Only update Discord once per second
            self.last_update_time = 0 

            def pipe_callback(pipe, step, timestep, callback_kwargs):
                if callback:
                    now = time.time()
                    # Throttles updates to prevent Discord Rate Limit (429) errors
                    if now - self.last_update_time >= 1.0 or step == steps:
                        percent = int((step / steps) * 100)
                        vram = round(self._get_vram_usage() / 1024, 1)
                        callback(percent, vram)
                        self.last_update_time = now
                return callback_kwargs

            image = self.pipeline(
                prompt=prompt, 
                num_inference_steps=steps,
                callback_on_step_end=pipe_callback,
                height=1024, 
                width=1024
            ).images[0]
            
            image.save(filepath)
            if main_loop: 
                main_loop.call_soon_threadsafe(self._update_activity)
                
            return filepath

    def _update_activity(self):
        self._last_activity = time.time()
        if self._unload_task: self._unload_task.cancel()
        self._unload_task = asyncio.create_task(self._inactivity_monitor())

    async def _inactivity_monitor(self):
        await asyncio.sleep(300)
        self._purge_vram()
        self.current_engine = None