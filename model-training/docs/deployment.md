# Rhythm AI – Cloud/Enterprise Deployment

## Docker

- Use `Dockerfile` for training jobs.
- Use `Dockerfile.inference` for lightweight, fast inference on CPU/GPU.

## Kubernetes

- `k8s/rhythm-ai-train-job.yaml` launches distributed or single-GPU training as a batch job.
- `k8s/rhythm-ai-inference-deployment.yaml` deploys scalable inference pods behind a load balancer.
- `k8s/rhythm-ai-inference-service.yaml` exposes inference API to your cloud VPC or the public internet.

## Persistent Volumes

Attach PVCs for datasets and model checkpoints for portability and recovery.

## Secrets

Use Kubernetes secrets or environment variables for API keys (e.g., WandB, S3, GCS).

## Autoscaling

Add a `HorizontalPodAutoscaler` to automatically scale inference pods based on CPU/RAM usage.

---

## Example: Full Inference Deployment

1. Build and push inference Docker image:
   ```bash
   docker build -f Dockerfile.inference -t ghcr.io/yourorg/rhythm-ai-inference:latest .
   docker push ghcr.io/yourorg/rhythm-ai-inference:latest
   ```

2. Create persistent volumes and secrets.

3. Deploy:
   ```bash
   kubectl apply -f k8s/rhythm-ai-inference-deployment.yaml
   kubectl apply -f k8s/rhythm-ai-inference-service.yaml
   ```

4. Access your model at `http://<cloud-load-balancer>:9000/generate`