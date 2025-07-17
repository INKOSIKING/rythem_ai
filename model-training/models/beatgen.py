import torch
import torch.nn as nn

class BeatGenModel(nn.Module):
    """
    CNN+Transformer-based drum/beat pattern generator for multi-track rhythm synthesis.
    """

    def __init__(self, config):
        super().__init__()
        self.embedding_dim = config.get("embedding_dim", 512)
        self.n_tracks = config.get("n_tracks", 9)
        self.seq_len = config.get("seq_len", 128)
        self.n_layers = config.get("num_layers", 12)
        self.n_heads = config.get("num_heads", 8)

        self.input_proj = nn.Conv1d(self.n_tracks, self.embedding_dim, kernel_size=3, padding=1)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embedding_dim,
            nhead=self.n_heads,
            dim_feedforward=4 * self.embedding_dim,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.n_layers)
        self.head = nn.Linear(self.embedding_dim, self.n_tracks)

    def forward(self, x):
        """
        x: [batch, n_tracks, seq_len] (binary drum grid)
        """
        x = self.input_proj(x)  # [B, embed_dim, seq_len]
        x = x.permute(0, 2, 1)  # [B, seq_len, embed_dim]
        x = self.transformer(x)
        logits = self.head(x)  # [B, seq_len, n_tracks]
        return logits