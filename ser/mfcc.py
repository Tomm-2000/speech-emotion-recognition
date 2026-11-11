"""Shared MFCC extraction for historical baseline experiments."""
from pathlib import Path
import numpy as np

SAMPLE_RATE = 16_000
N_MFCC = 40

def extract_mfcc_sequence(path: Path) -> np.ndarray:
    import librosa
    audio, sample_rate = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    sequence = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=N_MFCC,
    )

    if sequence.shape[0] != N_MFCC or sequence.shape[1] == 0:
        raise ValueError(f"Unexpected MFCC shape for {path}: {sequence.shape}")
    if not np.isfinite(sequence).all():
        raise ValueError(f"Invalid MFCC values for {path}")

    return sequence


def extract_mfcc(path: Path) -> np.ndarray:
    sequence = extract_mfcc_sequence(path)
    features = np.concatenate([sequence.mean(axis=1), sequence.std(axis=1)])
    if features.shape != (80,) or not np.isfinite(features).all():
        raise ValueError(f"Invalid MFCC features for {path}")
    return features
