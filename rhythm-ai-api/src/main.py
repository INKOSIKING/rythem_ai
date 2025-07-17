from fastapi import FastAPI
from src.routes import chat, audio, auth

app = FastAPI()
app.include_router(chat.router)
app.include_router(audio.router)
app.include_router(auth.router)