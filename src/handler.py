import runpod
import torch
from diffusers import WanI2VPipeline
from diffusers.utils import export_to_video
import base64
import os
import tempfile
import uuid
import imageio
from io import BytesIO
import requests
from PIL import Image

def download_image(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    return img

def decode_base64_image(base64_string):
    if "base64," in base64_string:
        base64_string = base64_string.split("base64,")[1]
    img = Image.open(BytesIO(base64.b64decode(base64_string)))
    return img

class Handler:
    def __init__(self):
        self.pipe = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._load_model()

    def _load_model(self):
        model_id = "/models/Wan2.1-I2V-14B-720P-Diffusers"
        print(f"Loading model from {model_id}...")
        
        self.pipe = WanI2VPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16
        )
        
        # User requested CPU offloading for T5 to save VRAM
        # This acts similarly to enable_sequential_cpu_offload() but is more specific if needed.
        # However, diffusers' enable_model_cpu_offload() is robust for this.
        self.pipe.enable_model_cpu_offload()
        
        # Enable VAE tiling to save memory during decoding
        # self.pipe.vae.enable_tiling() # Uncomment if OOM on decoding
        
        print("Model loaded successfully.")

    def inference(self, event):
        job_input = event.get("input", {})
        
        image_input = job_input.get("image")
        prompt = job_input.get("prompt", "")
        negative_prompt = job_input.get("negative_prompt", "")
        seed = job_input.get("seed", None)
        num_frames = job_input.get("num_frames", 81) # Default to typical value if needed
        
        if not image_input:
            return {"error": "No image provided"}

        # Load input image
        try:
            if image_input.startswith("http"):
                image = download_image(image_input)
            else:
                image = decode_base64_image(image_input)
        except Exception as e:
            return {"error": f"Failed to load image: {str(e)}"}

        # Set generator if seed provided
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        print("Starting inference...")
        output = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image,
            num_frames=num_frames,
            generator=generator,
        ).frames[0]
        
        # Save to temp MP4
        temp_dir = tempfile.gettempdir()
        output_filename = f"{uuid.uuid4()}.mp4"
        output_path = os.path.join(temp_dir, output_filename)
        
        print(f"Saving video to {output_path}...")
        export_to_video(output, output_path, fps=15)
        
        # Read back as base64
        with open(output_path, "rb") as f:
            video_bytes = f.read()
            video_base64 = base64.b64encode(video_bytes).decode("utf-8")
        
        # Clean up
        os.remove(output_path)
        
        return {
            "video": video_base64,
            "format": "mp4"
        }

handler = Handler()

def run_handler(event):
    try:
        return handler.inference(event)
    except Exception as e:
        print(f"Error during inference: {e}")
        return {"error": str(e)}

runpod.serverless.start({"handler": run_handler})
