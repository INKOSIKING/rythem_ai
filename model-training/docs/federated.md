# Rhythm AI – Federated Learning (Privacy by Design)

## What

- Train models across distributed clients/devices without collecting raw data centrally.
- Model updates are aggregated, not the data.
- Enables privacy, compliance, and edge/IoT use cases.

## Example (Simulation)

```python
from privacy.federated_training import federated_training, FakeClient

clients = [FakeClient(data1, ModelClass), FakeClient(data2, ModelClass), ...]
global_model = federated_training(clients, rounds=3)
```

## Best Practices

- Combine with differential privacy for stronger guarantees
- Audit model aggregation and client participation
- Use secure aggregation in production for true privacy

---

*Federated learning = privacy, scalability, and compliance in distributed AI.*