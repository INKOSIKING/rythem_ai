from utils.audit_logging import log_event
from registry.rollback import rollback_model

def audited_rollback(model_name, version, target_version=None, user=None):
    rollback_model(model_name, version, target_version)
    log_event("model_rollback", user or "system", extra=f"{model_name}:{version} -> {target_version or 'previous'}")

if __name__ == "__main__":
    audited_rollback("melodygen", "1.0.2", user="RythemAI")