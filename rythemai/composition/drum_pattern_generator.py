import pretty_midi
from typing import Literal

class DrumPatternGenerator:
    """
    Generates basic MIDI drum loops based on style and tempo.
    """
    def generate(self, style: Literal["rock", "pop", "hiphop"], tempo: int = 120, bars: int = 4) -> pretty_midi.PrettyMIDI:
        pm = pretty_midi.PrettyMIDI()
        drum = pretty_midi.Instrument(program=0, is_drum=True)
        ticks_per_beat = 60.0 / tempo
        for bar in range(bars):
            start = bar * 4 * ticks_per_beat
            # Kick on 1 and 3
            drum.notes.append(pretty_midi.Note(velocity=100, pitch=36, start=start, end=start+0.1))
            drum.notes.append(pretty_midi.Note(velocity=100, pitch=36, start=start+2*ticks_per_beat, end=start+2*ticks_per_beat+0.1))
            # Snare on 2 and 4
            drum.notes.append(pretty_midi.Note(velocity=90, pitch=38, start=start+ticks_per_beat, end=start+ticks_per_beat+0.1))
            drum.notes.append(pretty_midi.Note(velocity=90, pitch=38, start=start+3*ticks_per_beat, end=start+3*ticks_per_beat+0.1))
            # Hi-hat every beat
            for b in range(4):
                drum.notes.append(pretty_midi.Note(velocity=60, pitch=42, start=start+b*ticks_per_beat, end=start+b*ticks_per_beat+0.1))
        pm.instruments.append(drum)
        pm.estimate_tempo = tempo
        return pm