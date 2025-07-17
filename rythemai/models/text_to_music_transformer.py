import torch
from torch import nn
from typing import Optional

class TextToMusicTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        max_seq_len: int
    ):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.max_seq_len = max_seq_len

    def forward(
        self,
        src: torch.Tensor,  # [batch, src_len]
        tgt: torch.Tensor,  # [batch, tgt_len]
        src_key_padding_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size, src_len = src.shape
        batch_size, tgt_len = tgt.shape

        src_positions = torch.arange(0, src_len, device=src.device).unsqueeze(0).expand(batch_size, src_len)
        tgt_positions = torch.arange(0, tgt_len, device=tgt.device).unsqueeze(0).expand(batch_size, tgt_len)

        src_emb = self.token_emb(src) + self.pos_emb(src_positions)
        tgt_emb = self.token_emb(tgt) + self.pos_emb(tgt_positions)

        output = self.transformer(
            src=src_emb,
            tgt=tgt_emb,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )
        return self.fc_out(output)