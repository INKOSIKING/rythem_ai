# Rhythm AI – Canary/Batched Rollout Automation

## What

- Gradually shift traffic to a new model version (canary).
- Monitor canary metrics (latency, error rate) with Prometheus.
- If SLA is breached, auto-rollback to previous version.
- If healthy, promote to full traffic.

## How it Works

1. **Deploy canary model (e.g., 10% of traffic)**
2. **Monitor canary metrics for SLA compliance**
3. **Auto-rollback if breach detected**
4. **Promote to 100% if healthy**

## Example

```bash
python registry/canary_rollout.py
```

> *Integrate `set_traffic_split` with your K8s+Istio, Linkerd, or cloud load balancer for real-world use.*

---

*Canary rollout = safer, more reliable model upgrades in production.*