"""
One-shot script to download all bot models to D:/ai storage/huggingface_cache
Run from the bot venv: python download_models.py
"""
import os
os.environ["HF_HOME"] = "D:/ai storage/huggingface_cache"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "0"

from huggingface_hub import snapshot_download, login

token = os.environ.get("HF_TOKEN")
if token:
    login(token=token, add_to_git_credential=False)
    print(f"Logged in to HuggingFace.")
else:
    print("WARNING: HF_TOKEN not set — downloads will be slow/rate limited.")

MODELS = [
    # (repo_id, ignore_patterns)
    ("diffusers/FLUX.2-dev-bnb-4bit", None),
    ("stabilityai/stable-diffusion-xl-base-1.0", None),
    ("SG161222/Realistic_Vision_V5.1_noVAE", None),
    ("guoyww/animatediff-motion-adapter-v1-5-2", None),
    ("THUDM/CogVideoX-2b", None),
    ("Wan-AI/Wan2.1-T2V-14B-Diffusers", None),
    ("Systran/faster-whisper-base", None),
]

for repo_id, ignore in MODELS:
    print(f"\n{'='*60}")
    print(f"Downloading: {repo_id}")
    print(f"{'='*60}")
    try:
        path = snapshot_download(
            repo_id=repo_id,
            ignore_patterns=ignore,
        )
        print(f"Done: {path}")
    except Exception as e:
        print(f"FAILED {repo_id}: {e}")

print("\nAll downloads complete.")
