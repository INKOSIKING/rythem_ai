$ErrorActionPreference = "Stop"
$IMAGE_NAME = "ghcr.io/rythemai/music-ai-backend:latest"

Write-Host "==> Building Docker image for backend..."
docker build -t $IMAGE_NAME .\music-ai-backend

Write-Host "==> Pushing Docker image..."
docker push $IMAGE_NAME

Write-Host "==> Done."