# Rhythm AI – GitOps Automation

## Key Concepts

- **All infra, models, data, and config are versioned in Git**
- **Changes are applied automatically to the cluster using a GitOps tool** (e.g., ArgoCD, Flux)

## Example: ArgoCD Application

```yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: rhythm-ai
spec:
  project: default
  source:
    repoURL: 'https://github.com/yourorg/rhythm-ai'
    targetRevision: HEAD
    path: model-training/k8s
  destination:
    server: 'https://kubernetes.default.svc'
    namespace: rhythm-ai
  syncPolicy:
    automated:
      prune: true
      selfHeal: true
    syncOptions:
      - CreateNamespace=true
```

## Workflow

1. **Push to GitHub** (code, manifests, config, models)
2. **ArgoCD detects change** and applies new manifests to cluster
3. **Cluster self-heals** to match the desired state in Git

## Best Practices

- PRs for all changes (infra, model, data, config)
- Separate `dev`, `staging`, `prod` branches/environments
- Automated validation and policy checks in CI
- Secure GitOps agent with RBAC, secrets management

---

*GitOps ensures your ML infra is reproducible, auditable, and instantly recoverable.*