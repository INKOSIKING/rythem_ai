import librosa

class MusicKeyDetector:
    """
    Detects the musical key of an audio or MIDI file.
    """
    def detect_key_audio(self, file_path: str) -> str:
        y, sr = librosa.load(file_path, sr=None, mono=True)
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        key = librosa.key.key_to_degrees(librosa.key.estimate_tuning(chroma))
        return key

    def detect_key_midi(self, midi_path: str) -> str:
        import pretty_midi
        pm = pretty_midi.PrettyMIDI(midi_path)
        # Flatten notes into one big array
        notes = []
        for inst in pm.instruments:
            notes.extend([n.pitch for n in inst.notes])
        if not notes:
            return "Unknown"
        # Use Krumhansl-Schmuckler algorithm or simple histogram
        import numpy as np
        hist = np.zeros(12)
        for n in notes:
            hist[n % 12] += 1
        key_idx = hist.argmax()
        keys = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        return keys[key_idx]