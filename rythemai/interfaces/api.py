from fastapi import FastAPI, UploadFile, File, Form
from rythemai.audio_processing.key_detection import MusicKeyDetector
from rythemai.composition.midi_harmonizer import MidiHarmonizer
from rythemai.composition.drum_pattern_generator import DrumPatternGenerator
from rythemai.audio_processing.music_tagger import MusicTagger
from rythemai.utils.feedback_store import FeedbackStore

app = FastAPI()
key_detector = MusicKeyDetector()
harmonizer = MidiHarmonizer()
drum_gen = DrumPatternGenerator()
tagger = MusicTagger(model_path="path/to/tagger.pt")
feedback_store = FeedbackStore()

@app.post("/key/")
async def detect_key(file: UploadFile = File(...)):
    if file.filename.endswith(".mid") or file.filename.endswith(".midi"):
        key = key_detector.detect_key_midi(file.file)
    else:
        key = key_detector.detect_key_audio(file.file)
    return {"key": key}

@app.post("/harmonize/")
async def harmonize(file: UploadFile = File(...)):
    chords = harmonizer.suggest_chords(file.file)
    return {"chord_progression": chords}

@app.get("/drum_pattern/")
async def drum_pattern(style: str = "rock", tempo: int = 120, bars: int = 4):
    midi = drum_gen.generate(style, tempo, bars)
    # Return MIDI as bytes for download
    import io
    buf = io.BytesIO()
    midi.write(buf)
    return {"midi_base64": buf.getvalue().hex()}

@app.post("/tag/")
async def tag_audio(file: UploadFile = File(...)):
    tags = tagger.tag(file.file)
    return tags

@app.post("/feedback/")
async def add_feedback(track_id: str = Form(...), rating: int = Form(...), comments: str = Form("")):
    feedback_store.add_feedback(track_id, rating, comments)
    return {"status": "success"}