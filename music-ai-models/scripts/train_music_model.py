import torch
from models.music_generator import TransformerMusicGenerator

def train():
    vocab_size = 512  # Example, adapt to your vocab
    model = TransformerMusicGenerator(vocab_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    criterion = torch.nn.CrossEntropyLoss()
    # Replace this with real DataLoader
    for epoch in range(50):
        # X_src: [B, S], X_tgt: [B, S], y: [B, S]
        X_src = torch.randint(0, vocab_size, (32, 256))
        X_tgt = torch.randint(0, vocab_size, (32, 256))
        y = torch.randint(0, vocab_size, (32, 256))
        out = model(X_src, X_tgt)
        loss = criterion(out.view(-1, vocab_size), y.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        print(f"Epoch {epoch} Loss: {loss.item():.4f}")
    torch.save(model.state_dict(), "music_transformer.pt")

if __name__ == "__main__":
    train()