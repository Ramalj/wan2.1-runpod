# Wan 2.1 I2V RunPod Serverless Worker

[![Runpod](https://api.runpod.io/badge/Ramalj/wan2.1-runpod)](https://console.runpod.io/hub/Ramalj/wan2.1-runpod)

This is a RunPod Serverless Worker for the **Wan 2.1 Image-to-Video (720p)** model. It supports generation of high-quality videos from images using natural language prompts.

## Features

- **Model**: Wan 2.1 I2V 14B 720P (Diffusers).
- **Optimization**: Uses `torch.bfloat16` and CPU offloading for T5 encoder to fit on A100/A6000 GPUs.
- **Pre-downloaded Weights**: Weights are baked into the container for faster cold starts.
- **Output**: Returns MP4 video as a Base64 string.

## Input Payload

The worker accepts a JSON payload with the following parameters:

```json
{
  "input": {
    "image": "https://example.com/image.jpg",  // URL or Base64 string
    "prompt": "A cinematic drone shot of a futuristic city",
    "negative_prompt": "blurry, low quality",
    "seed": 42,
    "num_frames": 81
  }
}
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `image` | string | **Required** | Input image URL or Base64 encoded string. |
| `prompt` | string | "" | Text description of the desired video. |
| `negative_prompt` | string | "" | Things to avoid in the video. |
| `seed` | int | Random | Random seed for reproducibility. |
| `num_frames` | int | 81 | Number of frames to generate. |

## Output

The worker returns a JSON object containing the base64-encoded generated video:

```json
{
  "video": "<base64_string>",
  "format": "mp4"
}
```

## Local Development & Testing

1.  **Build the Docker image**:
    ```bash
    docker build -t wan2.1-worker .
    ```

2.  **Run the container**:
    ```bash
    docker run --gpus all wan2.1-worker
    ```

3.  **Test with `test_input.json`**:
    You can use the provided `test_input.json` to simulate an event if running the handler locally (requires GPU).

## Deployment

1.  Push your image to a container registry (e.g., Docker Hub):
    ```bash
    docker tag wan2.1-worker yourusername/wan2.1-worker:latest
    docker push yourusername/wan2.1-worker:latest
    ```

2.  Create a Template on RunPod:
    - **Container Image**: `yourusername/wan2.1-worker:latest`
    - **Container Disk**: 50 GB (recommended for model caching if not strictly baked in, though this image has them).
    - **Note**: Ensure you select a GPU instance type with sufficient VRAM (A100 80GB recommended).

## Files

- `src/handler.py`: The entry point for the RunPod worker.
- `builder.py`: Script used during build to pre-download model weights.
- `Dockerfile`: Configuration for specific dependencies and environment.
