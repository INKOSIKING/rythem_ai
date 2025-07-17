import time
from orchestration.webhook_trigger import trigger_webhook
from orchestration.compliance_export import export_audit_log
from orchestration.grc_integration import send_compliance_report
from utils.audit_logging import log_event

# Example event-driven workflow mapping
EVENT_HOOKS = {
    "model_approved": [
        lambda payload: trigger_webhook("model_approved", payload, "https://hooks.slack.com/services/demo-url"),
        lambda payload: export_audit_log(),  # Export audit on approval
    ],
    "compliance_export_ready": [
        lambda payload: send_compliance_report(payload["report_file"], "https://grc.example.com/api/upload", api_key="demo-token"),
    ],
    "sla_breach": [
        lambda payload: trigger_webhook("sla_breach", payload, "https://hooks.pagerduty.com/integration/demo-url"),
    ],
}

def handle_event(event_type, payload):
    actions = EVENT_HOOKS.get(event_type, [])
    for action in actions:
        try:
            action(payload)
            log_event("event_action_success", user="system", extra=f"{event_type} {action.__name__}")
        except Exception as e:
            log_event("event_action_failed", user="system", extra=f"{event_type} {action.__name__} {e}")

def event_loop():
    """
    Simulate event-driven orchestration.
    In production, subscribe to real event sources (Kafka, SQS, Pub/Sub, webhooks, etc).
    """
    test_fired = False
    while True:
        if not test_fired:
            print("Simulating event: model_approved")
            handle_event("model_approved", {"model": "melodygen", "version": "1.0.3"})
            test_fired = True
        time.sleep(600)  # Check or poll every 10 min

if __name__ == "__main__":
    event_loop()