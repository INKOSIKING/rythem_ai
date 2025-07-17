import numpy as np
from nltk.translate.bleu_score import sentence_bleu

def compute_bleu(preds, refs):
    # expects list of strings
    return np.mean([sentence_bleu([str(r)], str(p)) for p, r in zip(preds, refs)])

def compute_mos(preds, refs):
    # Placeholder for real MOS (subjective or model-based)
    return np.random.uniform(3.0, 5.0)

def compute_accuracy(preds, targets):
    # For classification
    preds = np.concatenate([p.argmax(-1) for p in preds])
    targets = np.concatenate(targets)
    return (preds == targets).mean()