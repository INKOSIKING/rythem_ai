# Rhythm AI – Secure Aggregation for Federated Learning

## What

- Prevents any party (including the central server) from seeing individual client model updates.
- Supports privacy and regulatory compliance for distributed AI.

## How

- Each client adds random noise to their model updates.
- The server aggregates (averages) updates, and the noise cancels out in aggregate.
- Can be combined with differential privacy for extra protection.

## Example

```python
from privacy.secure_aggregation import add_noise_to_update, secure_aggregate

# Each client:
noisy_update = add_noise_to_update(model_update, noise_scale=0.01)

# Server:
global_update = secure_aggregate([noisy1, noisy2, noisy3], noise_scale=0.01)
```

## Production

- Use cryptographic protocols for true secure aggregation (see PySyft, FATE, Flower).
- This module is a simulation for research and prototyping.

---

*Secure aggregation = privacy for all parties in federated learning.*