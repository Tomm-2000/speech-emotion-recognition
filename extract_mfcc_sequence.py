import csv
from pathlib import Path

import librosa
import numpy as np

from extract_features import N_MFCC, SAMPLE_RATE

ROOT = Path(__file__).resolve().parent


def extract_mfcc_sequence(path: Path) -> np.ndarray:
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


if __name__ == "__main__":
    with (ROOT / "data" / "manifest.csv").open(encoding="utf-8", newline="") as file:
        recording = next(row for row in csv.DictReader(file) if row["split"] == "train")

    path = ROOT / recording["path"]
    sequence = extract_mfcc_sequence(path)
    print(f"Recording: {path.name}")
    print(f"MFCC sequence shape: {sequence.shape}")