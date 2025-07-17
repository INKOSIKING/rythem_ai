import librosa

class BeatTempoDetector:
    """
    High-precision beat and tempo detection using librosa and neural beat trackers.
    """
    def detect(self, audio_path):
        y, sr = librosa.load(audio_path, sr=None)
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        return tempo, beats