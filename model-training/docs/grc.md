# Rhythm AI – GRC (Governance, Risk, Compliance) Integration

## What

- Automatically send compliance audit exports to GRC tools/systems (e.g., OneTrust, Archer, ServiceNow GRC).
- Supports CSV upload, API key auth, and full audit logging.

## Example

```python
from orchestration.grc_integration import send_compliance_report

send_compliance_report("compliance_exports/audit_log_2025-07-17.csv", "https://grc.example.com/api/upload", api_key="demo-token")
```

- Use in scheduled jobs or after critical events (model approval, retraining, rollback).
- Always verify with your GRC vendor for API details.

---

*Seamless GRC integration = enterprise trust and audit readiness.*