from fastapi import FastAPI, HTTPException, Form
from pydantic import BaseModel
from models.llm_integration import LLMChatWrapper
from models.music_generator import TransformerMusicGenerator
import torch
import tempfile
import os

app = FastAPI()
llm = LLMChatWrapper()
vocab_size = 512
music_model = TransformerMusicGenerator(vocab_size)
music_model.load_state_dict(torch.load("music_transformer.pt", map_location="cpu"))
music_model.eval()

class ChatRequest(BaseModel):
    user_id: str
    message: str

@app.post("/chat/")
async def chat(req: ChatRequest):
    try:
        response = llm.chat(req.message)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-music/")
async def generate_music(seed: str = Form(...)):
    import numpy as np
    import json
    seq = json.loads(seed)
    src = torch.tensor(seq).unsqueeze(0)
    tgt = src[:, :1]
    with torch.no_grad():
        for _ in range(255):
            out = music_model(src, tgt)
            next_token = out[:, -1, :].argmax(-1, keepdim=True)
            tgt = torch.cat([tgt, next_token], dim=1)
            if next_token.item() == 0:
                break
    midi_data = tgt.squeeze().tolist()
    with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as tmp:
        tmp.write(b"MThd...")  # TODO: real MIDI conversion
        midi_url = f"/music/{os.path.basename(tmp.name)}"
    return {"music_url": midi_url, "tokens": midi_data}

@app.get("/music/{fname}")
async def get_music(fname: str):
    path = f"/tmp/{fname}"
    if not os.path.exists(path):
        raise HTTPException(404, detail="File not found")
    with open(path, "rb") as f:
        return f.read()