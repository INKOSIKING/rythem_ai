# Rhythm AI – Automated Model Rollback & Remediation

## Why

- Ensure safe, fast rollback to a previous model version in case of incident or regression.
- Backups and registry tracking for every rollback event.
- Optionally require human-in-the-loop re-approval after rollback.

## How it Works

1. **Backup**  
   - Before rollback, the current deployed model artifact is backed up to `model_registry/rollback_backups/`
2. **Restore**
   - Previous model artifact replaces the current one on disk.
   - Registry file is updated with a rollback note.
3. **Approval**
   - Optionally, a rollback triggers a new approval workflow to ensure compliance.
4. **Audit**
   - All rollback events should be audit logged (see `utils/audit_logging.py`).

## Example

```python
from registry.rollback import rollback_model
rollback_model("melodygen", "1.0.2")  # Rollback to previous version
```

---

*Automated, auditable rollback is essential for safe production AI.*