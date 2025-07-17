# Rhythm AI — Model Registry

## What

A lightweight, file-based model registry for tracking trained models, metadata, and artifacts.

## Why

- Enables end-to-end reproducibility
- Auditable model history and lineage
- Simple integration with CI/CD, MLOps, and deployment

## Usage

```python
from registry.model_registry import register_model, list_models, get_model_info

register_model(
    model_name="melodygen",
    version="1.0.1",
    metadata={
        "accuracy": 0.99,
        "eval_loss": 0.11,
        "training_set": "v2.0"
    },
    artifact_path="models/checkpoints/melodygen_v1.0.1.pt"
)

print(list_models())
print(get_model_info("melodygen", "1.0.1"))
```

> For enterprise scale, integrate with MLflow, SageMaker, or GCP Vertex Model Registry.  
> For most projects, this is fully CI/CD and GitOps compatible.

---

- Register models automatically after training or evaluation (see `registry/register_after_train.py`)
- Registry files are versioned in Git and/or synced to S3/GCS for durability