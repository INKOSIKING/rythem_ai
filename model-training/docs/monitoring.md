# Rhythm AI – Monitoring & Observability

## Resource Monitoring

- Use `utils/monitoring.py` for logging CPU, RAM, GPU utilization during jobs.
- Forward logs to a centralized log server (e.g., ELK Stack, Datadog).

## API/Inference Monitoring

- Integrate Prometheus with Flask or FastAPI for metrics:
    - Request count
    - Latency/response time
    - Error rates

- Add `/metrics` endpoint for Prometheus scraping.

## Example: Prometheus Metrics Endpoint

```python
from prometheus_client import Counter, Histogram, generate_latest
from flask import Response

REQUEST_COUNT = Counter('request_count', 'Total API Requests')
RESPONSE_TIME = Histogram('response_time_seconds', 'Response time in seconds')

@app.route("/metrics")
def metrics():
    return Response(generate_latest(), mimetype="text/plain")
```

## Visualization

- Use Grafana dashboards for real-time resource and inference metrics.
- Alert on latency spikes, failure rates, or resource exhaustion.