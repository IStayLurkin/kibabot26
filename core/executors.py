from concurrent.futures import ThreadPoolExecutor

# For long-running jobs: image generation, music synthesis
# Capped at 2 to prevent OOM from parallel VRAM allocations
HEAVY_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="kiba-heavy")

# For short blocking ops: nvidia-smi, short HTTP calls
LIGHT_EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="kiba-light")
