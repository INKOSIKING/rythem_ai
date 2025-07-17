# Rhythm AI – Privacy Audit Logging

## Why

- Track all privacy-preserving ML actions (DP, federated, secure agg) for compliance.
- Enable review, reporting, and regulatory audits.

## How

- Every differentially private training run is logged (ε, δ, model, user, timestamp)
- Every federated round and secure aggregation event is logged
- Use `privacy_audit.py` for automated audit

## Example

```python
from privacy.privacy_audit import log_dp_training, log_federated_round, log_secure_agg

log_dp_training("melodygen", "1.0.3", 2.1, 1e-5, user="privacy-bot")
log_federated_round("melodygen", 2, ["client1", "client2"])
log_secure_agg("melodygen", 2)
```

- Use compliance export/reporting from main audit log for privacy audits

---

*Privacy audit = required for enterprise, healthcare, and sensitive data AI.*