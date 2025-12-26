# Base image
FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

# Install system dependencies
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Upgrade pip
RUN pip install --upgrade pip

# Install Python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Create cache directory for models
RUN mkdir -p /models

# Enable HF Transfer and set cache directory
ENV HF_HUB_ENABLE_HF_TRANSFER=1
ENV HF_HOME=/models

# Copy application code
COPY src/ .

# Download models during build (Disabled to avoid timeout, will download on first run)
# RUN python builder.py

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Run the handler
CMD ["python", "-u", "handler.py"]
