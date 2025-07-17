# Rhythm AI – Cost Optimization Guide

## Tips

- Use spot/preemptible GPU VMs for non-critical training
- Autoscale inference pods with HPA to match demand
- Archive old checkpoints and datasets to low-cost storage (S3 Glacier, GCS Nearline)
- Monitor GPU/CPU/Memory utilization with Prometheus & Grafana
- Set resource requests/limits on all K8s containers/jobs
- Use DVC to avoid duplicate storage of large files

## Example: Autoscale Inference

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: rhythm-ai-inference-hpa
spec:
  minReplicas: 2
  maxReplicas: 10
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: rhythm-ai-inference
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 60
```

## Automated Cleanup

- Set up S3/GCS lifecycle rules to delete or archive old objects after X days
- Regularly prune unused Docker images and K8s resources

---

*Review your bills monthly and iterate!*