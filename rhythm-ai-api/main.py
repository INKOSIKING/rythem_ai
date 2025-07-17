from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.security import OAuth2PasswordBearer
from rhythm_ai_core.music_llm import RhythmAIMusicLLM
import logging

app = FastAPI()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
ai = RhythmAIMusicLLM()

@app.post("/chat/")
def chat(prompt: str, token: str = Depends(oauth2_scheme)):
    try:
        resp = ai.generate(prompt)
        return {"result": resp}
    except Exception as e:
        logging.error(f"Error in chat: {e}")
        raise HTTPException(500, str(e))