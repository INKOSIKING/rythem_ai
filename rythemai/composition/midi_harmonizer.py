import pretty_midi
from typing import List

class MidiHarmonizer:
    """
    Suggests chord progressions for a given monophonic melody.
    """
    def suggest_chords(self, midi_path: str) -> List[str]:
        pm = pretty_midi.PrettyMIDI(midi_path)
        melody = []
        for inst in pm.instruments:
            if not inst.is_drum:
                melody.extend(inst.notes)
        melody.sort(key=lambda n: n.start)
        # Very basic: Scan notes in groups and suggest triads
        chord_progression = []
        for i in range(0, len(melody), 4):
            notes = melody[i:i+4]
            if notes:
                root = notes[0].pitch % 12
                chord = self._major_minor_guess(root)
                chord_progression.append(chord)
        return chord_progression

    def _major_minor_guess(self, root: int) -> str:
        keys = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        return f"{keys[root]}maj"  # Replace with smarter logic for minor etc.