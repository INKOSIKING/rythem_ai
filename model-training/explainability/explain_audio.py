import torch
import numpy as np
from captum.attr import IntegratedGradients
from typing import Any

def explain_audio_model(model: torch.nn.Module, input_tensor: torch.Tensor, baseline: torch.Tensor = None) -> Any:
    """
    Generates feature attribution for audio models using Integrated Gradients.
    - model: Trained PyTorch model
    - input_tensor: Input audio tensor (batch, features, time)
    - baseline: Baseline tensor (same shape as input)
    Returns: attributions tensor
    """
    model.eval()
    ig = IntegratedGradients(model)
    if baseline is None:
        baseline = torch.zeros_like(input_tensor)
    attributions, _ = ig.attribute(input_tensor, baseline, target=None, return_convergence_delta=True)
    return attributions

if __name__ == "__main__":
    print("Audio model explainability (Integrated Gradients). Import in your workflow.")