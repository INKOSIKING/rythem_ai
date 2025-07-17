import torch
from demucs.pretrained import get_model as get_demucs

class SourceSeparator:
    """
    Uses the Demucs model for high-fidelity source separation.
    """
    def __init__(self, model_name="htdemucs", device="cuda"):
        self.model = get_demucs(model_name).to(device)
        self.device = device

    def separate(self, audio: torch.Tensor, sr: int):
        """
        Splits audio into stems (vocals, drums, bass, others).
        Returns a dict: {stem_name: waveform_tensor}
        """
        self.model.eval()
        with torch.no_grad():
            stems = self.model.separate_audio(audio, sr)
        return stems