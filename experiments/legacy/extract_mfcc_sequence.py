import csv
from pathlib import Path

import librosa
import numpy as np

from .extract_features import N_MFCC, SAMPLE_RATE

ROOT = Path(__file__).resolve().parents[2]


from ser.mfcc import extract_mfcc_sequence


if __name__ == "__main__":
    with (ROOT / "data" / "manifest.csv").open(encoding="utf-8", newline="") as file:
        recording = next(row for row in csv.DictReader(file) if row["split"] == "train")

    path = ROOT / recording["path"]
    sequence = extract_mfcc_sequence(path)
    print(f"Recording: {path.name}")
    print(f"MFCC sequence shape: {sequence.shape}")
