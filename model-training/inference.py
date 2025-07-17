import torch
from models.melodygen import MelodyGenModel
from utils.tokenizer import MusicTokenizer

def generate(prompt, model_path, cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MelodyGenModel(cfg["model"]).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    tokenizer = MusicTokenizer(cfg)
    tokens = tokenizer.text_to_tokens(prompt)
    tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model.generate(tokens, max_length=cfg["inference"]["max_length"])
    return tokenizer.tokens_to_audio(output.cpu().squeeze().tolist())

if __name__ == "__main__":
    prompt = "generate a chill house groove with jazzy chords"
    cfg = ... # load config
    audio = generate(prompt, "models/checkpoints/best_model.pt", cfg)
    # save or play audio