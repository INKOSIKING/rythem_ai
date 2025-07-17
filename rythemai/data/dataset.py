import torch
from torch.utils.data import Dataset
import pandas as pd
from rythemai.data.midi_tokenizer import MidiTokenizer
from typing import Tuple

class TextToMusicDataset(Dataset):
    def __init__(
        self,
        meta_csv: str,
        midi_dir: str,
        vocab_file: str,
        max_seq_len: int
    ):
        self.df = pd.read_csv(meta_csv)
        self.midi_dir = midi_dir
        self.tokenizer = MidiTokenizer(vocab_file)
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        text_prompt = row["text_prompt"]
        midi_filename = row["midi_filename"]
        text_tokens = self.encode_text(text_prompt)
        midi_tokens = self.tokenizer.midi_to_tokens(f"{self.midi_dir}/{midi_filename}")

        text_tensor = torch.tensor(text_tokens[:self.max_seq_len], dtype=torch.long)
        midi_tensor = torch.tensor(midi_tokens[:self.max_seq_len], dtype=torch.long)
        return text_tensor, midi_tensor

    def encode_text(self, text: str):
        # Simple whitespace tokenization + vocab lookup
        tokens = [self.tokenizer.stoi.get(f"WORD_{w.lower()}", self.tokenizer.stoi["UNK"]) for w in text.split()]
        return tokens