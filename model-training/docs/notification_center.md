# Rhythm AI – Notification Center

## What

- Unified notification system for Slack, Teams, Email, and SMS (Twilio).
- Use for alerts, approvals, SLA breaches, pipeline status, compliance, and more.
- Audit logs every notification.

## Example

```python
from orchestration.notification_center import notify_slack, notify_email, notify_teams, notify_sms

notify_slack("Model approved!", "https://hooks.slack.com/services/your-url")
notify_email("Model SLA Breach", "Check the pipeline!", "admin@example.com", "smtp.example.com", 465, "robot@example.com", "password")
notify_teams("Model deployed to production.", "https://outlook.office.com/webhook/your-url")
notify_sms("Critical SLA breach!", "+1234567890", "twilio_sid", "twilio_token", "+10987654321")
```

- Securely store credentials (do not hardcode in production!)
- Integrate into event-driven workflows for rapid response

---

*Instant notifications = faster response and better MLOps reliability.*