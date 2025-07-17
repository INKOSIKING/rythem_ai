# Rhythm AI – Monitoring, Alerting & Incident Response

## Alertmanager

- Receives alerts from Prometheus (e.g., high latency, OOM, node failures)
- Sends notifications to Slack, email, PagerDuty, etc.
- Configured in `k8s/alertmanager-config.yaml`

## Example Prometheus Alert Rules

```yaml
groups:
- name: rhythm-ai.rules
  rules:
  - alert: HighInferenceLatency
    expr: histogram_quantile(0.95, sum(rate(response_time_seconds_bucket{job="rhythm-ai-inference"}[5m])) by (le)) > 2
    for: 5m
    labels:
      severity: critical
    annotations:
      summary: "High 95th percentile inference latency"
      description: "95th percentile inference latency is > 2s for 5m"
  - alert: PodOOMKilled
    expr: kube_pod_container_status_last_terminated_reason{reason="OOMKilled"} > 0
    for: 1m
    labels:
      severity: warning
    annotations:
      summary: "Pod OOMKilled"
      description: "A Rhythm AI pod was killed due to Out Of Memory."
```

## Slack Integration

- Set your Slack webhook URL in Alertmanager config.
- Alerts go directly to the `#ml-ops-alerts` channel.

## Incident Response

- On alert, check Grafana dashboards for anomalies.
- Investigate root cause in logs (W&B, MLflow, ELK).
- Roll back to previous model/infra version with GitOps if needed.
- Document postmortem in incident log.

---

**Proactive alerting is key for reliable, scalable AI in production.**