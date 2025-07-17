import os
import json
from mido import MidiFile
import soundfile as sf

class RhythmAIDataset:
    def __init__(self, root_dir):
        self.audio_dir = os.path.join(root_dir, "audio")
        self.midi_dir = os.path.join(root_dir, "midi")
        self.labels_dir = os.path.join(root_dir, "labels")
        self.song_ids = [os.path.splitext(x)[0] for x in os.listdir(self.audio_dir) if x.endswith(".wav")]

    def __len__(self):
        return len(self.song_ids)

    def __getitem__(self, idx):
        song_id = self.song_ids[idx]
        audio_path = os.path.join(self.audio_dir, f"{song_id}.wav")
        midi_path = os.path.join(self.midi_dir, f"{song_id}.mid")
        label_path = os.path.join(self.labels_dir, f"{song_id}.json")

        # Load audio
        audio, sr = sf.read(audio_path)
        # Load MIDI
        midi = MidiFile(midi_path)
        # Load labels/metadata
        with open(label_path) as f:
            labels = json.load(f)
        return {
            "song_id": song_id,
            "audio": audio,
            "sr": sr,
            "midi": midi,
            "labels": labels
        }

# Example usage
if __name__ == "__main__":
    dataset = RhythmAIDataset("rhythm-ai-dataset")
    for i in range(len(dataset)):
        sample = dataset[i]
        print(f"{sample['song_id']} | audio shape: {sample['audio'].shape} | MIDI tracks: {len(sample['midi'].tracks)} | labels: {sample['labels']}")