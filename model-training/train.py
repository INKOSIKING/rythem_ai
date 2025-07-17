import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataloader import get_multimodal_loader
from utils.logging import setup_logging, log_metrics
from utils.distributed import setup_ddp, cleanup_ddp
from models import melodygen, beatgen, vocalsynth, lyricsgen
import yaml
import wandb

def load_model(cfg):
    if cfg["model"]["type"] == "melodygen":
        return melodygen.MelodyGenModel(cfg["model"]).to(cfg["device"])
    elif cfg["model"]["type"] == "beatgen":
        return beatgen.BeatGenModel(cfg["model"]).to(cfg["device"])
    elif cfg["model"]["type"] == "vocalsynth":
        return vocalsynth.VocalSynthModel(cfg["model"]).to(cfg["device"])
    elif cfg["model"]["type"] == "lyricsgen":
        return lyricsgen.LyricsGenModel(cfg["model"]).to(cfg["device"])
    else:
        raise ValueError(f"Unknown model type: {cfg['model']['type']}")

def train_one_epoch(model, loader, optimizer, criterion, device, epoch, cfg):
    model.train()
    total_loss = 0.0
    for batch_idx, batch in enumerate(loader):
        x, y = batch["input"].to(device), batch["target"].to(device)
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if batch_idx % cfg["training"]["log_every"] == 0:
            log_metrics({"batch_loss": loss.item(), "epoch": epoch, "batch_idx": batch_idx})
    avg_loss = total_loss / len(loader)
    log_metrics({"epoch_loss": avg_loss, "epoch": epoch})
    return avg_loss

def validate(model, loader, criterion, device):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            x, y = batch["input"].to(device), batch["target"].to(device)
            output = model(x)
            loss = criterion(output, y)
            val_loss += loss.item()
    avg_val_loss = val_loss / len(loader)
    log_metrics({"val_loss": avg_val_loss})
    return avg_val_loss

def main():
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    setup_logging(cfg)
    wandb.init(project="rhythm-ai-train", config=cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(cfg)
    optimizer = optim.AdamW(model.parameters(), lr=cfg["training"]["lr"])
    criterion = nn.CrossEntropyLoss() # Replace as needed

    train_loader, val_loader = get_multimodal_loader(cfg)
    best_val = float("inf")
    for epoch in range(cfg["training"]["epochs"]):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch, cfg)
        val_loss = validate(model, val_loader, criterion, device)
        wandb.log({"train_loss": train_loss, "val_loss": val_loss, "epoch": epoch})
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), os.path.join(cfg["training"]["checkpoint_path"], f"best_model_epoch{epoch}.pt"))
        if cfg["training"].get("early_stopping") and val_loss > best_val:
            print(f"Early stopping at epoch {epoch}")
            break
    wandb.finish()

if __name__ == "__main__":
    main()