import torch
from models.melodygen.model import MelodyGenModel
from training.dataset_loader import load_dataset
import yaml

with open("training/config.yaml") as f:
    config = yaml.safe_load(f)

train_data, val_data = load_dataset(config["dataset"])

model = MelodyGenModel(**config["model"])
optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["lr"])
loss_fn = torch.nn.CrossEntropyLoss()

for epoch in range(config["training"]["epochs"]):
    model.train()
    for x, y in train_data:
        out = model(x)
        loss = loss_fn(out, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch} complete.")
    # [Validation code here]
torch.save(model.state_dict(), config["training"]["checkpoint_path"])