from huggingface_hub import snapshot_download
import os

def download_model():
    model_id = "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers"
    local_dir = "/models/Wan2.1-I2V-14B-720P-Diffusers"
    
    print(f"Downloading {model_id} to {local_dir}...")
    snapshot_download(
        repo_id=model_id,
        local_dir=local_dir,
        ignore_patterns=["*.msgpack", "*.bin", "*.h5", "*.tflite", "*.ot"], # Optional: filter non-relevant formats if needed
    )
    print("Download complete.")

if __name__ == "__main__":
    download_model()
