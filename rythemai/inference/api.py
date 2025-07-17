import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import yaml
import tempfile
import os

from rythemai.models.text_to_music_transformer import TextToMusicTransformer
from rythemai.data.midi_tokenizer import MidiTokenizer

app = FastAPI()
config = {}
model = None
tokenizer = None

class Prompt(BaseModel):
    prompt: str

def load_config(config_path: str):
    global config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

def load_model():
    global model, tokenizer
    tokenizer = MidiTokenizer(config["data"]["vocab_file"])
    vocab_size = len(tokenizer.stoi)
    model_inst = TextToMusicTransformer(
        vocab_size=vocab_size,
        d_model=config["model"]["d_model"],
        nhead=config["model"]["nhead"],
        num_layers=config["model"]["num_layers"],
        dim_feedforward=config["model"]["dim_feedforward"],
        dropout=config["model"]["dropout"],
        max_seq_len=config["model"]["max_seq_len"]
    )
    model_inst.load_state_dict(torch.load(config["inference"]["checkpoint"], map_location="cpu"))
    model_inst.eval()
    model = model_inst

@app.on_event("startup")
def startup_event():
    load_config("rythemai/config/config.yaml")
    load_model()

@app.post("/generate/")
def generate_music(prompt: Prompt):
    text_tokens = [tokenizer.stoi.get(f"WORD_{w.lower()}", tokenizer.stoi["UNK"]) for w in prompt.prompt.split()]
    text_tensor = torch.tensor(text_tokens, dtype=torch.long).unsqueeze(0)
    generated = text_tensor.new_full((1, 1), tokenizer.stoi["BOS"])  # start with BOS
    with torch.no_grad():
        for _ in range(config["model"]["max_seq_len"]):
            output = model(text_tensor, generated)
            next_token = output[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            if next_token.item() == tokenizer.stoi["EOS"]:
                break
    midi_tokens = generated.squeeze(0).tolist()
    with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as tmpf:
        out_path = tmpf.name
    tokenizer.tokens_to_midi(midi_tokens, out_path)
    with open(out_path, "rb") as f:
        midi_data = f.read()
    os.remove(out_path)
    return {"midi_base64": midi_data.hex()}