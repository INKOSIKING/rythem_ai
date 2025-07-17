import numpy as np
import librosa
import random

def random_pitch_shift(audio, sr, max_steps=2):
    steps = random.uniform(-max_steps, max_steps)
    return librosa.effects.pitch_shift(audio, sr, steps)

def random_time_stretch(audio, max_rate=0.2):
    rate = random.uniform(1.0 - max_rate, 1.0 + max_rate)
    return librosa.effects.time_stretch(audio, rate)

def random_gain(audio, min_gain=0.8, max_gain=1.2):
    gain = random.uniform(min_gain, max_gain)
    return audio * gain

def add_background_noise(audio, noise_level=0.005):
    noise = np.random.randn(len(audio))
    augmented = audio + noise_level * noise
    return np.clip(augmented, -1.0, 1.0)

def augment_audio(audio, sr):
    if random.random() < 0.3:
        audio = random_pitch_shift(audio, sr)
    if random.random() < 0.3:
        audio = random_time_stretch(audio)
    if random.random() < 0.3:
        audio = add_background_noise(audio)
    if random.random() < 0.3:
        audio = random_gain(audio)
    return audio

def augment_midi(midi_tensor, max_shift=1):
    # Example: Randomly shift note velocity or time
    if random.random() < 0.5:
        midi_tensor = midi_tensor + random.randint(-max_shift, max_shift)
        midi_tensor = np.clip(midi_tensor, 0, 127)
    return midi_tensor