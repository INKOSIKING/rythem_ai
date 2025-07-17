import os
import glob
import librosa
import soundfile as sf
import mido
import json

RAW_DIR = "datasets/raw/"
PROC_DIR = "datasets/processed/"
META_PATH = "datasets/metadata.json"

def preprocess_audio(src_path, dst_path, sr=32000, duration=None):
    y, _ = librosa.load(src_path, sr=sr, mono=True, duration=duration)
    y = librosa.util.normalize(y)
    sf.write(dst_path, y, sr)

def preprocess_midi(src_path, dst_path):
    mid = mido.MidiFile(src_path)
    mid.save(dst_path) # could add cleaning/filtering

def run():
    meta = []
    for audio_file in glob.glob(os.path.join(RAW_DIR, "audio", "*.wav")):
        base = os.path.splitext(os.path.basename(audio_file))[0]
        midi_file = os.path.join(RAW_DIR, "midi", f"{base}.mid")
        label_file = os.path.join(RAW_DIR, "labels", f"{base}.json")
        out_audio = os.path.join(PROC_DIR, "audio", f"{base}.wav")
        out_midi = os.path.join(PROC_DIR, "midi", f"{base}.mid")
        out_label = os.path.join(PROC_DIR, "labels", f"{base}.json")
        os.makedirs(os.path.dirname(out_audio), exist_ok=True)
        os.makedirs(os.path.dirname(out_midi), exist_ok=True)
        os.makedirs(os.path.dirname(out_label), exist_ok=True)
        preprocess_audio(audio_file, out_audio)
        preprocess_midi(midi_file, out_midi)
        if os.path.exists(label_file):
            with open(label_file) as f_in, open(out_label, "w") as f_out:
                json.dump(json.load(f_in), f_out)
        meta.append({
            "audio": out_audio,
            "midi": out_midi,
            "label": out_label,
            "caption": f"Track {base}",
            "target": 0 # placeholder
        })
    with open(META_PATH, "w") as fp:
        json.dump(meta, fp, indent=2)

if __name__ == "__main__":
    run()