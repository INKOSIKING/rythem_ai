# Rhythm AI – Super Advanced Training Pipeline

This repo contains everything for enterprise-grade, multi-modal AI music model training and evaluation.

## Features

- Multi-modal: audio, MIDI, lyrics, tags
- Models: Transformer (melodygen, beatgen), neural vocoder (vocalsynth), seq2seq (lyricsgen)
- Distributed/data-parallel training
- Robust logging (wandb, logging), error handling
- Flexible config, modular codebase, rapid prototyping
- Mobile/web export (ONNX)
- LoRA/PEFT support for efficient fine-tuning

## Usage

1. **Preprocess**
   ```bash
   python preprocess.py
   ```

2. **Train**
   ```bash
   python train.py
   ```

3. **Evaluate**
   ```bash
   python evaluate.py
   ```

4. **Inference**
   ```bash
   python inference.py "Make a happy jazz melody"
   ```

5. **Export**
   ```bash
   python convert_to_onnx.py
   ```

---

### Folder Structure

- `models/` — Model definitions (Transformer, CNN, Diffusion, etc.)
- `utils/` — Utilities for audio, metrics, logging, distributed
- `datasets/` — Raw/processed data, metadata
- `checkpoints/` — Model weights
- `train.py`, `evaluate.py`, ... — Main scripts