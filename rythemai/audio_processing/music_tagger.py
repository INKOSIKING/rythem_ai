import torch
import torchaudio
from transformers import Wav2Vec2Model

class MusicTagger:
    """
    Auto-tags tracks with mood, instrument, and tempo using a fine-tuned model.
    """
    def __init__(self, model_path: str, device: str = "cpu"):
        self.model = torch.load(model_path, map_location=device)
        self.device = device

    def tag(self, audio_path: str) -> dict:
        waveform, sr = torchaudio.load(audio_path)
        # Preprocessing: Resample, normalize, etc.
        with torch.no_grad():
            features = self.model(waveform.to(self.device))
            tags = self._decode_tags(features)
        return tags

    def _decode_tags(self, features):
        # Map output to tags, e.g. mood, instrument, etc.
        return {
            "mood": "energetic",
            "primary_instrument": "guitar",
            "tempo": 128
        }