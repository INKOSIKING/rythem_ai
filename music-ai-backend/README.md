# music-ai-backend

Enterprise-ready FastAPI backend for LLM chat and music generation.

- Secure REST endpoints
- Robust error handling
- Docker/K8s ready

## Usage

```bash
docker build -t music-ai-backend .
docker run -p 8000:8000 --env OPENAI_API_KEY=sk-... music-ai-backend
```