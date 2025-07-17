from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
from models.llm_integration import chat_with_gpt
from models.music_generator import SimpleMusicGenerator
import torch
import tempfile

app = FastAPI()
model = SimpleMusicGenerator()
model.load_state_dict(torch.load("music_generator.pt"))
model.eval()

class ChatRequest(BaseModel):
    user_id: str
    message: str

@app.post("/chat/")
async def chat(req: ChatRequest):
    response = chat_with_gpt(req.message)
    return {"response": response}

@app.post("/generate-music/")
async def generate_music(seed: list = Form(...)):
    with torch.no_grad():
        out = model(torch.tensor(seed).unsqueeze(0).float())
    # Save to temp MIDI, return file path or bytes (stub)
    with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as tmp:
        tmp.write(b"MThd...")  # Replace with real MIDI bytes
        music_url = f"/music/{tmp.name.split('/')[-1]}"
    return {"music_url": music_url}

@app.get("/music/{fname}")
async def get_music(fname: str):
    # Serve MIDI file by fname
    path = f"/tmp/{fname}"
    with open(path, "rb") as f:
        return f.read()