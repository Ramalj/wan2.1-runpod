from huggingface_hub import snapshot_download
import os

def download_model():
    model_id = "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers"
    local_dir = "/models/Wan2.1-I2V-14B-720P-Diffusers"
    
    if os.path.exists(local_dir):
        print(f"Model directory {local_dir} already exists. checking for completeness...", flush=True)
        # In a real scenario we might want to verify, but for now we assume existence is enough for a cache hit
        # dependent on persistence. If it's empty, huggingface_hub handles correct resuming.
    
    print(f"Downloading {model_id} to {local_dir}...", flush=True)
    
    # Enable fast transfer
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    
    snapshot_download(
        repo_id=model_id,
        local_dir=local_dir,
        ignore_patterns=["*.msgpack", "*.bin", "*.h5", "*.tflite", "*.ot"], 
    )
    print("Download complete.", flush=True)

if __name__ == "__main__":
    download_model()
