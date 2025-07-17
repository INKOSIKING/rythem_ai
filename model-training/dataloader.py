import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
import librosa
from mido import MidiFile

class MultiModalMusicDataset(Dataset):
    def __init__(self, meta_path, cfg, split="train"):
        with open(meta_path) as f:
            self.meta = json.load(f)
        self.cfg = cfg
        self.split = split

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        item = self.meta[idx]
        # Load audio
        audio, sr = librosa.load(item["audio"], sr=self.cfg["model"]["sample_rate"])
        # Load MIDI
        midi = MidiFile(item["midi"])
        midi_tensor = self.midi_to_tensor(midi)
        # Load text/lyrics/metadata
        text = item.get("caption", "")
        # Prepare target depending on task
        target = self.prepare_target(item)
        return {
            "input": torch.tensor(audio, dtype=torch.float32),
            "midi": midi_tensor,
            "text": text,
            "target": target
        }

    def midi_to_tensor(self, midi):
        # Convert MIDI to tensor (velocity, time, pitch)
        # ... (advanced parsing here)
        return torch.zeros((128, 256)) # placeholder

    def prepare_target(self, item):
        # For classification/regression/generation targets
        return torch.tensor(item.get("target"), dtype=torch.long)

def get_multimodal_loader(cfg):
    train_ds = MultiModalMusicDataset(cfg["datasets"]["metadata"], cfg, split="train")
    val_ds = MultiModalMusicDataset(cfg["datasets"]["metadata"], cfg, split="val")
    train_loader = DataLoader(train_ds, batch_size=cfg["training"]["batch_size"], shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["training"]["batch_size"], shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, val_loader