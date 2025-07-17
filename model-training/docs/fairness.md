# Rhythm AI – Fairness Monitoring & Bias Detection

## Why

- Ensure model predictions are fair across sensitive groups (genre, gender, region, etc).
- Required for ethical and regulatory compliance.

## Metrics

- **Demographic Parity:** P(positive | group1) = P(positive | group2)
- **Equalized Odds:** TPR, FPR equal across groups

## Example

```python
from fairness.fairness_metrics import print_fairness_report

# y_true, y_pred: numpy arrays, sensitive_attr: array of group labels
print_fairness_report(y_true, y_pred, sensitive_attr)
```

## Best Practices

- Log/report fairness metrics after every training run
- Include fairness checks in CI/CD and approval workflow
- Investigate and retrain if bias detected

---

*Fairness is not optional—monitor and improve continuously!*