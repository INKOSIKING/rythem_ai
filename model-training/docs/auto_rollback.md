# Rhythm AI – Auto-Rollback on SLA Breach

## What

- Constantly monitor live metrics (latency, error rate) from Prometheus.
- If SLA is breached (latency > 2s or error rate > 1%), trigger an automated model rollback.
- All events are audit logged for compliance.

## How it Works

1. **Monitor**
   - Script queries Prometheus for 99th percentile latency and error rate every minute.
2. **Trigger**
   - If SLA is breached, script calls the audited rollback function.
3. **Audit**
   - All breaches and rollbacks are logged with timestamp, user, and details.

## Example

```bash
python registry/auto_rollback_on_sla.py
```

---

*Close the loop: SLA monitoring, alerting, and remediation are now fully automated.*