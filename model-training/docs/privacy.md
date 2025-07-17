# Rhythm AI – Privacy-Preserving ML

## Differential Privacy

- Use Opacus (PyTorch) or TensorFlow Privacy to train models with mathematically guaranteed privacy.
- Protects against memorization of individual data points.
- Essential for sensitive or regulated data.

## Example

```python
from privacy.differential_privacy_training import train_with_dp

# model, dataloader, criterion must be defined
model, (epsilon, delta) = train_with_dp(model, dataloader, criterion, epochs=5)
```

## Best Practices

- Set strict (ε, δ) for regulatory scenarios (e.g., GDPR, HIPAA)
- Audit and log all privacy parameters and training runs

---

*Differential privacy is the gold standard for privacy-preserving AI.*