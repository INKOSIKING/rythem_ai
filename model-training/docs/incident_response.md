# Rhythm AI – Incident Response Guide

## On-Call Best Practices

- Always have at least one on-call ML/ops engineer during production hours.
- Use Slack or PagerDuty for urgent alerts.

## Typical Incident Process

1. **Receive Alert**  
   (via Slack, PagerDuty, etc.)

2. **Acknowledge in Alertmanager**
   
3. **Triage**  
   - Check Grafana dashboards for health metrics
   - Look for spikes in latency, errors, OOM killed pods, etc.

4. **Investigate**
   - Review logs in W&B, MLflow, or log aggregator
   - Identify root cause (e.g., code bug, bad deploy, hardware failure)

5. **Mitigate**
   - Roll back deployment via GitOps/ArgoCD
   - Scale up/down resources if needed
   - Redeploy previous model if necessary

6. **Postmortem**
   - Document incident, timeline, resolution, and lessons learned
   - Share with team in dedicated incident log

---

**Fast, transparent incident response keeps your AI system reliable and your users happy.**