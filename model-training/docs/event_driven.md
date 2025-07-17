# Rhythm AI – Event-Driven Orchestration

## What

- Automate workflows based on events (model approval, SLA breach, export ready, etc).
- Supports notifications, compliance export, GRC integration, and more.
- Decouple pipelines from manual triggers.

## How

- Map events to actions (webhooks, exports, compliance uploads, etc).
- Use `event_driven.py` loop or integrate with real event/message bus.

## Example

```python
from orchestration.event_driven import handle_event

handle_event("model_approved", {"model": "melodygen", "version": "1.0.3"})
```

- In production: connect to Kafka, SQS, webhooks, or CI/CD job triggers.
- Add/modify event-action mappings for full automation.

---

*Event-driven MLOps unlocks rapid, robust, and auditable pipelines.*