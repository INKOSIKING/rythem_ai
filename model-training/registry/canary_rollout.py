import time
import requests
from utils.audit_logging import log_event
from registry.rollback_audit import audited_rollback

PROMETHEUS_URL = "http://prometheus:9090/api/v1/query"
MODEL_NAME = "melodygen"
NEW_VERSION = "1.0.3"
CURRENT_VERSION = "1.0.2"
CANARY_TRAFFIC_PERCENT = 10  # % of traffic to canary
ROLLBACK_ON_SLA_BREACH = True
SLA_LATENCY_SEC = 2.0
SLA_ERROR_RATE = 0.01

def set_traffic_split(canary_percent):
    """
    Stub: Integrate with your deployment/orchestration platform.
    For K8s + Istio, would patch VirtualService weights.
    """
    log_event("canary_traffic_split", user="system", extra=f"{NEW_VERSION}: {canary_percent}%")
    print(f"Set canary rollout: {NEW_VERSION} gets {canary_percent}% of traffic.")

def get_canary_metrics():
    # Example Prometheus label: model_version="1.0.3"
    latency_query = f'histogram_quantile(0.99, sum(rate(response_time_seconds_bucket{{job="rhythm-ai-inference",model_version="{NEW_VERSION}"}}[5m])) by (le))'
    error_query = f'sum(rate(api_request_errors_total{{job="rhythm-ai-inference",model_version="{NEW_VERSION}"}}[5m])) / sum(rate(api_request_total{{job="rhythm-ai-inference",model_version="{NEW_VERSION}"}}[5m]))'
    def get_metric(query):
        try:
            resp = requests.get(PROMETHEUS_URL, params={"query": query}, timeout=5)
            resp.raise_for_status()
            result = resp.json()["data"]["result"]
            if not result:
                return None
            return float(result[0].get("value", [None, None])[1])
        except Exception as e:
            log_event("canary_metric_query_failed", user="system", extra=str(e))
            return None
    latency = get_metric(latency_query)
    error_rate = get_metric(error_query)
    return latency, error_rate

def canary_rollout_monitor():
    set_traffic_split(CANARY_TRAFFIC_PERCENT)
    print("Waiting for canary metrics to stabilize...")
    time.sleep(300)  # Wait 5 min for traffic
    latency, error_rate = get_canary_metrics()
    print(f"Canary {NEW_VERSION} latency (p99): {latency}s | error rate: {error_rate*100 if error_rate is not None else None}%")
    if latency and latency > SLA_LATENCY_SEC:
        log_event("canary_sla_breach", user="system", extra=f"latency {latency}s > SLA")
        print("Canary SLA latency breach — rolling back.")
        if ROLLBACK_ON_SLA_BREACH:
            audited_rollback(MODEL_NAME, NEW_VERSION, CURRENT_VERSION, user="canary-rollback")
        return
    if error_rate and error_rate > SLA_ERROR_RATE:
        log_event("canary_sla_breach", user="system", extra=f"error_rate {error_rate} > SLA")
        print("Canary SLA error rate breach — rolling back.")
        if ROLLBACK_ON_SLA_BREACH:
            audited_rollback(MODEL_NAME, NEW_VERSION, CURRENT_VERSION, user="canary-rollback")
        return
    # If healthy, promote canary to full production
    set_traffic_split(100)
    log_event("canary_promoted", user="system", extra=f"{NEW_VERSION} promoted to 100% traffic")
    print(f"Canary {NEW_VERSION} promoted to full production.")

if __name__ == "__main__":
    canary_rollout_monitor()