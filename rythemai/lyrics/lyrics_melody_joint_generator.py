import torch
from torch import nn

class LyricsMelodyJointGenerator(nn.Module):
    """
    Generates lyrics and melody jointly, conditioned on prompt and style.
    """
    def __init__(self, vocab_size_lyrics, vocab_size_melody, d_model=512):
        super().__init__()
        self.lyrics_emb = nn.Embedding(vocab_size_lyrics, d_model)
        self.melody_emb = nn.Embedding(vocab_size_melody, d_model)
        self.joint_transformer = nn.Transformer(
            d_model=d_model,
            nhead=8,
            num_encoder_layers=12,
            num_decoder_layers=12,
            batch_first=True
        )
        self.lyrics_head = nn.Linear(d_model, vocab_size_lyrics)
        self.melody_head = nn.Linear(d_model, vocab_size_melody)

    def forward(self, prompt_tokens, style_tokens):
        # Generate jointly
        out = self.joint_transformer(prompt_tokens + style_tokens)
        lyrics_logits = self.lyrics_head(out)
        melody_logits = self.melody_head(out)
        return lyrics_logits, melody_logits