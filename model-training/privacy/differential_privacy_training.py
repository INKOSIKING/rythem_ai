import torch
import torch.nn as nn
import torch.optim as optim

from opacus import PrivacyEngine

def train_with_dp(model, dataloader, criterion, epochs=5, lr=1e-3, max_grad_norm=1.0, target_epsilon=2.0, target_delta=1e-5):
    """
    Trains a PyTorch model with differential privacy using Opacus.
    - model: nn.Module
    - dataloader: torch.utils.data.DataLoader
    - criterion: loss function
    Returns: trained model, privacy spent (epsilon, delta)
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    privacy_engine = PrivacyEngine()
    model, optimizer, dataloader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=dataloader,
        target_epsilon=target_epsilon,
        target_delta=target_delta,
        epochs=epochs,
        max_grad_norm=max_grad_norm,
    )
    model.train()
    for epoch in range(epochs):
        for X, y in dataloader:
            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{epochs} complete.")
    epsilon = privacy_engine.get_epsilon(target_delta)
    print(f"Training complete. (ε, δ)=({epsilon:.2f}, {target_delta})")
    return model, (epsilon, target_delta)

if __name__ == "__main__":
    print("Differential privacy training module (Opacus). Import and use in your training pipeline.")