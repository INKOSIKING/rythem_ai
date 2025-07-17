#!/bin/bash
set -euo pipefail

IMAGE_NAME="ghcr.io/rythemai/music-ai-backend:latest"

echo "==> Building Docker image for backend..."
docker build -t $IMAGE_NAME ./music-ai-backend

echo "==> Pushing Docker image..."
docker push $IMAGE_NAME

echo "==> Done."