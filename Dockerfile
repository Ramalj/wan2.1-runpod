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

# Copy builder script and download weights
COPY builder.py .
RUN python builder.py

# Copy application code
COPY src/ .

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Run the handler
CMD ["python", "-u", "handler.py"]
