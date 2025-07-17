# Rhythm AI – Terraform Infrastructure Example

## S3 Bucket for DVC/Checkpoints

```hcl
resource "aws_s3_bucket" "rhythm_ai_dvc" {
  bucket = "rhythm-ai-dvc"
  acl    = "private"
  versioning {
    enabled = true
  }
  lifecycle {
    prevent_destroy = true
  }
}
```

## GKE (Google Kubernetes Engine) Cluster

```hcl
resource "google_container_cluster" "primary" {
  name     = "rhythm-ai-prod"
  location = "us-central1"

  remove_default_node_pool = true
  initial_node_count       = 1

  node_config {
    machine_type = "n1-highmem-8"
    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform"
    ]
    labels = {
      env = "prod"
    }
    guest_accelerator {
      type  = "nvidia-tesla-t4"
      count = 1
    }
  }
}
```

## Secrets (WandB, API keys)

```hcl
resource "kubernetes_secret" "wandb" {
  metadata {
    name = "rhythm-ai-secrets"
  }
  data = {
    wandb_api_key = base64encode(var.wandb_api_key)
  }
}
```

---

**Combine with K8s manifests and GitHub Actions for full GitOps.**