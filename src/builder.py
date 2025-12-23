from huggingface_hub import snapshot_download
import os

def download_model():
    model_id = "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers"
    # Use HF_HOME or default to /models if not set
    local_dir = os.environ.get("HF_HOME", "/models")
    
    # We want nested folder for the specific model if we are using HF_HOME as base
    target_dir = os.path.join(local_dir, "Wan2.1-I2V-14B-720P-Diffusers")
    
    if os.path.exists(target_dir) and any(os.scandir(target_dir)):
        print(f"Model already exists in {target_dir}. Checking for updates/completeness...", flush=True)
    else:
        print(f"Downloading {model_id} to {target_dir} (this may take a while)...", flush=True)
    
    # Enable fast transfer if not already set
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    
    snapshot_download(
        repo_id=model_id,
        local_dir=target_dir,
        ignore_patterns=["*.msgpack", "*.bin", "*.h5", "*.tflite", "*.ot"],
        max_workers=8 # Speed up download if possible
    )
    print("Download complete.", flush=True)

if __name__ == "__main__":
    download_model()
