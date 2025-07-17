import pretty_midi
import json
from typing import List, Dict

class MidiTokenizer:
    def __init__(self, vocab_file: str):
        with open(vocab_file, "r") as f:
            self.vocab = json.load(f)
        self.itos = {int(v): k for k, v in self.vocab.items()}
        self.stoi = self.vocab

    def midi_to_tokens(self, midi_path: str) -> List[int]:
        midi = pretty_midi.PrettyMIDI(midi_path)
        tokens = []
        for instrument in midi.instruments:
            for note in instrument.notes:
                pitch_token = f"NOTE_{note.pitch}"
                start_token = f"START_{int(note.start * 100)}"
                end_token = f"END_{int(note.end * 100)}"
                tokens.extend([
                    self.stoi.get(pitch_token, self.stoi["UNK"]),
                    self.stoi.get(start_token, self.stoi["UNK"]),
                    self.stoi.get(end_token, self.stoi["UNK"])
                ])
        return tokens

    def tokens_to_midi(self, tokens: List[int], out_path: str):
        midi = pretty_midi.PrettyMIDI()
        instrument = pretty_midi.Instrument(program=0)
        i = 0
        while i + 2 < len(tokens):
            pitch_token = self.itos.get(tokens[i], "NOTE_60")
            start_token = self.itos.get(tokens[i + 1], "START_0")
            end_token = self.itos.get(tokens[i + 2], "END_10")
            if pitch_token.startswith("NOTE_") and start_token.startswith("START_") and end_token.startswith("END_"):
                pitch = int(pitch_token[5:])
                start = int(start_token[6:]) / 100.0
                end = int(end_token[4:]) / 100.0
                note = pretty_midi.Note(velocity=100, pitch=pitch, start=start, end=end)
                instrument.notes.append(note)
            i += 3
        midi.instruments.append(instrument)
        midi.write(out_path)