import numpy as np
from typing import List

def add_noise_to_update(model_update: dict, noise_scale: float) -> dict:
    """
    Add Gaussian noise to each parameter tensor in the model update for secure aggregation.
    """
    noisy_update = {}
    for k, v in model_update.items():
        noise = np.random.normal(0, noise_scale, size=v.shape)
        noisy_update[k] = v + noise
    return noisy_update

def secure_aggregate(updates: List[dict], noise_scale: float = 0.0) -> dict:
    """
    Securely aggregate model updates. Optionally add noise for differential privacy.
    Each update is a dict of numpy arrays (parameter tensors).
    """
    agg = {}
    for k in updates[0]:
        stacked = np.stack([u[k] for u in updates])
        agg[k] = np.mean(stacked, axis=0)
        if noise_scale > 0.0:
            agg[k] += np.random.normal(0, noise_scale, size=agg[k].shape)
    return agg

if __name__ == "__main__":
    print("Secure aggregation simulation for federated learning.")