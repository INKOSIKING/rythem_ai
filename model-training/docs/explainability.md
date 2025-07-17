# Rhythm AI – Model Explainability & Fairness

## Why

- Understand model predictions at feature/input level.
- Build trust with stakeholders (artists, users, auditors).
- Detect bias or unexpected behavior.

## How

- **SHAP for neural nets and tabular models** (`explain_shap.py`, `explain_tabular.py`)
- **Integrated Gradients for audio/sequence models** (`explain_audio.py`)
- Visualize explanations in notebooks, dashboards, or serving UIs.
- Store explanations as artifacts for audit/compliance.

## Example for SHAP

```python
from explainability.explain_shap import explain_model, plot_explanation
shap_values = explain_model(model, x_input, x_background)
plot_explanation(shap_values, x_input)
```

## Example for Audio

```python
from explainability.explain_audio import explain_audio_model
attr = explain_audio_model(model, input_tensor)
```

---

*Explainability is required for responsible, fair, and compliant AI in production.*