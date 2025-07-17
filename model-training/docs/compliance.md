# Rhythm AI – Compliance & Regulatory Export

## What

- Export full or filtered audit logs for compliance, regulatory, or enterprise review.
- Supports CSV export by default (easy for legal, auditors, or GRC systems).

## How

- All events (training, approval, rollback, privacy, deployment, etc) are audit logged.
- Export tool runs on demand or can be scheduled.

## Example

```python
from orchestration.compliance_export import export_audit_log

export_audit_log()  # Export all logs to CSV
```

- Filter by date range or event type as needed.

---

*Compliance export is essential for regulated and enterprise AI environments.*