import pretty_midi
import soundfile as sf
import os

class BatchMidiToAudioConverter:
    """
    Converts a collection of MIDI files to audio using built-in synthesizer.
    """
    def convert_folder(self, midi_dir: str, out_dir: str):
        os.makedirs(out_dir, exist_ok=True)
        for fname in os.listdir(midi_dir):
            if fname.endswith(".mid") or fname.endswith(".midi"):
                pm = pretty_midi.PrettyMIDI(os.path.join(midi_dir, fname))
                audio = pm.fluidsynth()
                sf.write(os.path.join(out_dir, fname.replace(".mid", ".wav")), audio, 44100)