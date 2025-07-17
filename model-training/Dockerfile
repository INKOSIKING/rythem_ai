# Rhythm AI: Production Training & Inference Dockerfile

FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Install system dependencies
RUN apt-get update && \
    apt-get install -y python3 python3-pip git libsndfile1 ffmpeg libsm6 libxext6 && \
    rm -rf /var/lib/apt/lists/*

# Set up workdir
WORKDIR /app

# Copy code and requirements
COPY . /app
RUN pip3 install --upgrade pip setuptools wheel
RUN pip3 install -r requirements.txt

# [Optional] Install extra tools for cloud, MLflow, DVC, etc.
RUN pip3 install dvc[s3] mlflow onnxruntime boto3 google-cloud-storage psutil sentencepiece wandb

# Set entrypoint for training (can be changed for inference/serving)
ENTRYPOINT ["python3", "train.py"]