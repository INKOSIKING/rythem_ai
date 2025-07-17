import time
import requests
from registry.rollback_audit import audited_rollback
from utils.audit_logging import log_event

PROMETHEUS_URL = "http://prometheus:9090/api/v1/query"
MODEL_NAME = "melodygen"
DEPLOYED_VERSION = "1.0.2"
SLA_LATENCY_SEC = 2.0
SLA_ERROR_RATE = 0.01
ROLLBACK_TARGET_VERSION = None  # None means rollback to previous version

def get_metric(query):
    try:
        resp = requests.get(PROMETHEUS_URL, params={"query": query}, timeout=5)
        resp.raise_for_status()
        result = resp.json()["data"]["result"]
        if not result:
            return None
        return float(result[0].get("value", [None, None])[1])
    except Exception as e:
        log_event("sla_metric_query_failed", user="system", extra=str(e))
        return None

def monitor_and_auto_rollback():
    while True:
        # 99th percentile latency over 5m
        latency_query = 'histogram_quantile(0.99, sum(rate(response_time_seconds_bucket{job="rhythm-ai-inference"}[5m])) by (le))'
        latency = get_metric(latency_query)
        # Error rate over 5m
        error_query = 'sum(rate(api_request_errors_total{job="rhythm-ai-inference"}[5m])) / sum(rate(api_request_total{job="rhythm-ai-inference"}[5m]))'
        error_rate = get_metric(error_query)
        print(f"Latency (p99): {latency}s | Error rate: {error_rate*100 if error_rate is not None else None}%")
        if latency and latency > SLA_LATENCY_SEC:
            log_event("sla_breach_detected", user="system", extra=f"latency {latency}s > SLA")
            print("SLA latency breach detected — triggering auto-rollback.")
            audited_rollback(MODEL_NAME, DEPLOYED_VERSION, ROLLBACK_TARGET_VERSION, user="auto-sla-rollback")
            break
        if error_rate and error_rate > SLA_ERROR_RATE:
            log_event("sla_breach_detected", user="system", extra=f"error_rate {error_rate} > SLA")
            print("SLA error rate breach detected — triggering auto-rollback.")
            audited_rollback(MODEL_NAME, DEPLOYED_VERSION, ROLLBACK_TARGET_VERSION, user="auto-sla-rollback")
            break
        time.sleep(60)  # Check every minute

if __name__ == "__main__":
    monitor_and_auto_rollback()