import torch
import torch.nn as nn

class TransformerMusicGenerator(nn.Module):
    """Enterprise-grade music generator using Transformer for MIDI event streams."""
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=12, max_seq=1024):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq, d_model)
        self.transformer = nn.Transformer(
            d_model=d_model, nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt):
        B, S = src.shape
        positions = torch.arange(0, S, device=src.device).unsqueeze(0)
        src = self.tok_emb(src) + self.pos_emb(positions)
        tgt_positions = torch.arange(0, tgt.shape[1], device=tgt.device).unsqueeze(0)
        tgt = self.tok_emb(tgt) + self.pos_emb(tgt_positions)
        out = self.transformer(src, tgt)
        return self.fc(out)