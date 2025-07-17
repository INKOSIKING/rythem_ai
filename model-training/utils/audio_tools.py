import librosa
import numpy as np

def audio_to_mel(audio, sr=32000, n_fft=2048, hop_length=512):
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db

def audio_to_spec(audio, sr=32000, n_fft=2048, hop_length=512):
    return librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)

def trim_silence(audio, top_db=20):
    yt, _ = librosa.effects.trim(audio, top_db=top_db)
    return yt