import shap
import torch
import numpy as np
from typing import Any

def explain_model(model: torch.nn.Module, input_sample: np.ndarray, background_data: np.ndarray, device="cpu") -> Any:
    """
    Generates SHAP explanations for a PyTorch model.
    - model: Trained PyTorch model.
    - input_sample: A single input to explain (numpy array).
    - background_data: Background dataset for SHAP (numpy array).
    - device: "cpu" or "cuda"
    Returns: SHAP values object.
    """
    model.eval()
    model.to(device)
    e = shap.DeepExplainer(model, torch.tensor(background_data).to(device))
    shap_values = e.shap_values(torch.tensor(input_sample).to(device))
    return shap_values

def plot_explanation(shap_values, input_sample, feature_names=None):
    shap.initjs()
    shap.force_plot(shap_values, matplotlib=True, feature_names=feature_names, data=input_sample)

if __name__ == "__main__":
    # Example usage:
    # Assume model, x_background, x_input are loaded
    # import your_model
    # model = your_model.load(...)
    # x_background = ...
    # x_input = ...
    # shap_values = explain_model(model, x_input, x_background)
    # plot_explanation(shap_values, x_input)
    print("SHAP explainability module. Import and use in your serving pipeline or notebook.")