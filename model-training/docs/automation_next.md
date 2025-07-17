# Rhythm AI – Closed-Loop Retraining Automation

## What

- Monitors for arrival of new labeled data.
- Automatically triggers retraining pipeline (via CI/CD or orchestrator).
- Audit logs all retraining events.
- Supports human-in-the-loop approval after retrain.

## How

1. **Detect new data** (e.g., via filesystem, S3 event, or DVC update)
2. **Trigger retraining** (touch file, CI/CD, or call orchestrator API)
3. **Audit log** the retraining event
4. **Register/re-approve new model after training**

## Example

```bash
python automation/retraining_trigger.py
```

---

*This closes the loop: new data = new model, with full audit and approval!*