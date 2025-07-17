# Rhythm AI – Audit Logging & Compliance

## Purpose

- Track all critical actions:
  - Model registration, deployment, rollback
  - Data access, export, deletion
  - User changes to config, code, infra

## How

- Use `utils/audit_logging.py` to log every event with timestamp, user, and extra details.
- Store audit logs in append-only, versioned storage (e.g., S3 with object lock, GCS, Azure Blob).
- Regularly review logs for compliance and anomaly detection.
- Integrate with SIEM or security monitoring tools for alerting.

## Example Log Entry

```
2025-07-17T18:01:46.123456 | USER: RythemAI | EVENT: model_registered | EXTRA: melodygen v1.0.1
```

---

*Audit logging is required for regulated industries (finance, healthcare, etc) and good practice for all production AI.*