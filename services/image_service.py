import os
import time
import gc
import torch
import asyncio
import subprocess
import aiohttp
from typing import Optional

# 2026 Updated Imports for FLUX.2 and 4-bit Quantization
try:
    from diffusers import Flux2Pipeline 
    from transformers import BitsAndBytesConfig
except ImportError:
    Flux2Pipeline = None
    BitsAndBytesConfig = None

class ImageService:
    def __init__(self, model_id: str = "black-forest-labs/FLUX.2-dev", **kwargs):
        self.model_id = model_id
        self.pipeline = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.output_dir = "outputs/images"
        os.makedirs(self.output_dir, exist_ok=True)

    def _get_vram_usage(self) -> int:
        """Helper to get current VRAM usage in MB for the 3090 Ti."""
        try:
            cmd = "nvidia-smi --query-gpu=memory.used --format=csv,nounits,noheader"
            result = subprocess.check_output(cmd, shell=True).decode('utf-8').strip()
            return int(result)
        except Exception:
            return 0

    async def _unload_ollama(self):
        """Forces Ollama to unload Qwen3 from VRAM to make room for FLUX.2."""
        try:
            print("[DEBUG] Requesting Ollama to unload models...")
            async with aiohttp.ClientSession() as session:
                # keep_alive: 0 forces immediate ejection from VRAM
                async with session.post(
                    "http://localhost:11434/api/chat",
                    json={"model": "qwen3-coder:7b", "keep_alive": 0}
                ) as resp:
                    if resp.status == 200:
                        print("[DEBUG] Qwen3-Coder unloaded.")
            await asyncio.sleep(2) # Settle time
        except Exception as e:
            print(f"[DEBUG] Could not unload Ollama: {e}")

    def _load_pipeline(self):
        """Optimized FLUX.2 loader for 24GB VRAM using NF4 Quantization."""
        if self.pipeline is None and Flux2Pipeline is not None:
            print("[DEBUG] Initializing FLUX.2 [dev] with 4-bit Quantization...")
            
            # Mandatory config to fit 32B model into 3090 Ti VRAM
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4"
            )

            self.pipeline = Flux2Pipeline.from_pretrained(
                self.model_id,
                torch_dtype=torch.bfloat16,
                quantization_config=quant_config,
                device_map="auto" 
            )
            
            # Essential for preventing OOM crashes during the forward pass
            self.pipeline.enable_sequential_cpu_offload()
            print("[DEBUG] FLUX.2 Primary Engine Online.")

    async def generate_image(self, prompt: str) -> Optional[str]:
        """Async entry point for high-fidelity local generation."""
        await self._unload_ollama()
        
        gc.collect()
        torch.cuda.empty_cache()
        
        current_vram = self._get_vram_usage()
        print(f"[DEBUG] VRAM after Ollama unload: {current_vram}MB. Starting FLUX.2...")

        filename = f"kiba_flux2_{int(time.time())}.png"
        filepath = os.path.join(self.output_dir, filename)

        try:
            return await asyncio.to_thread(self._generate_sync, prompt, filepath)
        except Exception as e:
            print(f"[ERROR] FLUX.2 Generation Failed: {e}")
            return None

    def _generate_sync(self, prompt: str, filepath: str) -> str:
        """Synchronous worker for FLUX.2 generation."""
        self._load_pipeline()
        
        if self.pipeline is None:
            raise RuntimeError("Flux2Pipeline failed to load. Check 'pip install bitsandbytes'.")

        image = self.pipeline(
            prompt,
            height=1024,
            width=1024,
            guidance_scale=3.5,
            num_inference_steps=28,
            max_sequence_length=512,
        ).images[0]
        
        image.save(filepath)
        
        # Clear VRAM so the Dispatcher can reload Qwen3-Coder
        torch.cuda.empty_cache()
        gc.collect()
            
        return filepath