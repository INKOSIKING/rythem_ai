import torch
from torch.utils.data import DataLoader
from dataloader import get_multimodal_loader
from utils.metrics import compute_bleu, compute_mos, compute_accuracy
from models import melodygen, beatgen, vocalsynth, lyricsgen
import yaml
import numpy as np

def load_model(cfg, device):
    # as in train.py
    pass

def evaluate():
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(cfg, device)
    val_loader, _ = get_multimodal_loader(cfg)
    model.eval()
    all_preds, all_targets = [], []
    for batch in val_loader:
        x, y = batch["input"].to(device), batch["target"].to(device)
        with torch.no_grad():
            output = model(x)
        # Save outputs for scoring
        all_preds.append(output.cpu())
        all_targets.append(y.cpu())
    # Example metrics:
    bleu = compute_bleu(all_preds, all_targets)
    mos = compute_mos(all_preds, all_targets)
    acc = compute_accuracy(all_preds, all_targets)
    print(f"BLEU: {bleu:.4f} | MOS: {mos:.4f} | ACC: {acc:.4f}")

if __name__ == "__main__":
    evaluate()