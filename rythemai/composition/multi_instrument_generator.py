import torch
from torch import nn
from rythemai.conditioning.style_encoder import StyleEncoder

class MultiInstrumentMusicGenerator(nn.Module):
    """
    Generates multi-track MIDI arrangements conditioned on style, genre, and instrument config.
    """
    def __init__(
        self,
        vocab_size: int,
        instrument_count: int,
        style_dim: int,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 12,
        device: str = "cuda"
    ):
        super().__init__()
        self.device = device
        self.style_encoder = StyleEncoder(style_dim)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            batch_first=True
        )
        self.instrument_heads = nn.ModuleList([
            nn.Linear(d_model, vocab_size) for _ in range(instrument_count)
        ])
        self.instrument_count = instrument_count

    def forward(self, style_tokens, prompt_tokens, instrument_prompts):
        # style_tokens: [batch, style_dim]
        # prompt_tokens: [batch, seq, d_model]
        # instrument_prompts: List of [batch, seq, d_model], one per instrument
        style_latent = self.style_encoder(style_tokens)
        outputs = []
        for idx in range(self.instrument_count):
            decoded = self.transformer(
                src=prompt_tokens + style_latent.unsqueeze(1),
                tgt=instrument_prompts[idx]
            )
            logits = self.instrument_heads[idx](decoded)
            outputs.append(logits)
        return outputs