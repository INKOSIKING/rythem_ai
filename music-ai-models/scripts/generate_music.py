import torch
from models.music_generator import TransformerMusicGenerator

def generate(seed, max_len=256):
    vocab_size = 512
    model = TransformerMusicGenerator(vocab_size)
    model.load_state_dict(torch.load("music_transformer.pt"))
    model.eval()
    src = torch.tensor(seed).unsqueeze(0)
    tgt = src[:, :1]
    for _ in range(max_len-1):
        out = model(src, tgt)
        next_token = out[:, -1, :].argmax(-1, keepdim=True)
        tgt = torch.cat([tgt, next_token], dim=1)
        if next_token.item() == 0:  # EOS
            break
    return tgt.squeeze().tolist()

if __name__ == "__main__":
    seed = [1,2,3,4,5]
    out = generate(seed)
    print(out)