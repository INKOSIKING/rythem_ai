import requests
from utils.audit_logging import log_event

def trigger_webhook(event_type, payload, webhook_url):
    """
    Trigger a webhook with event_type and payload dict.
    Audit logs all calls.
    """
    headers = {"Content-Type": "application/json"}
    data = {
        "event_type": event_type,
        "payload": payload
    }
    try:
        resp = requests.post(webhook_url, json=data, timeout=10)
        resp.raise_for_status()
        log_event("webhook_triggered", user="system", extra=f"{event_type} {webhook_url}")
        print(f"Webhook triggered: {webhook_url} ({event_type})")
        return True
    except Exception as e:
        log_event("webhook_failed", user="system", extra=f"{event_type} {webhook_url} {e}")
        print(f"Webhook failed: {webhook_url} ({event_type}) {e}")
        return False

if __name__ == "__main__":
    # Example test webhook
    test_url = "https://webhook.site/your-test-url"
    trigger_webhook("model_deployed", {"model": "melodygen", "version": "1.0.3"}, test_url)