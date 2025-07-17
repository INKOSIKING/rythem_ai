# music-ai-models

Production-level AI models for music generation and LLM chat.

- Transformer-based music generator (MIDI)
- LLM chat integration (OpenAI GPT-4o)
- Training and inference scripts

## Usage

Train:
```bash
python scripts/train_music_model.py
```
Generate:
```bash
python scripts/generate_music.py
```
LLM chat:
```python
from models.llm_integration import LLMChatWrapper
llm = LLMChatWrapper()
print(llm.chat("Write a jazz chord progression for me."))
```