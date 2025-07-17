import torch
import torch.nn as nn

class MelodyGenModel(nn.Module):
    """
    Transformer-based model for melody/music generation supporting MIDI, text, and audio prompt conditions.
    """

    def __init__(self, config):
        super().__init__()
        self.embedding_dim = config.get("embedding_dim", 1024)
        self.vocab_size = config.get("vocab_size", 512)
        self.max_seq_len = config.get("max_seq_len", 1024)
        self.n_layers = config.get("num_layers", 24)
        self.n_heads = config.get("num_heads", 16)
        self.dropout = config.get("dropout", 0.1)

        self.token_embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.positional_embedding = nn.Parameter(torch.zeros(1, self.max_seq_len, self.embedding_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embedding_dim,
            nhead=self.n_heads,
            dim_feedforward=4 * self.embedding_dim,
            dropout=self.dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.n_layers)
        self.ln_f = nn.LayerNorm(self.embedding_dim)
        self.head = nn.Linear(self.embedding_dim, self.vocab_size)

    def forward(self, x):
        """
        Args:
            x: [batch, seq_len] integer tokens
        Returns:
            logits: [batch, seq_len, vocab_size]
        """
        B, S = x.shape
        x = self.token_embedding(x) + self.positional_embedding[:, :S, :]
        x = self.transformer(x)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits

    @torch.no_grad()
    def generate(self, x, max_length=512, temperature=1.0, top_k=None):
        """
        Autoregressive sequence generation.
        Args:
            x: [1, seed_len] initial token tensor
        Returns:
            tokens: [1, variable_length]
        """
        tokens = x
        for _ in range(max_length - x.shape[1]):
            logits = self.forward(tokens)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                values, indices = torch.topk(logits, top_k)
                probs = torch.zeros_like(logits).scatter_(1, indices, torch.softmax(values, -1))
            else:
                probs = torch.softmax(logits, -1)
            next_token = torch.multinomial(probs, num_samples=1)
            tokens = torch.cat([tokens, next_token], dim=1)
            if next_token.item() == 0:  # EOS
                break
        return tokens