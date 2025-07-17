import getpass
import sys
from utils.audit_logging import log_event

def audited_register(*args, **kwargs):
    user = kwargs.pop("user", None) or getpass.getuser()
    from registry.model_registry import register_model
    register_model(*args, **kwargs)
    log_event("model_registered", user=user, extra=f"{args[0]}:{args[1]}")

def audited_get(*args, **kwargs):
    user = kwargs.pop("user", None) or getpass.getuser()
    from registry.model_registry import get_model_info
    info = get_model_info(*args, **kwargs)
    log_event("model_info_accessed", user=user, extra=f"{args[0]}:{args[1]}")
    return info

def audited_search(*args, **kwargs):
    user = kwargs.pop("user", None) or getpass.getuser()
    from registry.advanced_search import search_models
    result = search_models(*args, **kwargs)
    log_event("model_search", user=user, extra=str(kwargs))
    return result

# Example use in production
if __name__ == "__main__":
    # Register a model (with audit)
    audited_register(
        model_name="beatgen",
        version="2.0.1",
        metadata={"accuracy": 0.995, "tags": ["prod", "rhythm"], "trained_on": "2025-07-10"},
        artifact_path="models/checkpoints/beatgen_v2.pt",
        user="RythemAI"
    )
    # Search (with audit)
    print(audited_search(metric_gte={"accuracy": 0.99}, tag="prod"))