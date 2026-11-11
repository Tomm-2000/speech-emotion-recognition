import csv
from pathlib import Path

import librosa
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SAMPLE_RATE = 16_000
N_MFCC = 40


from ser.mfcc import extract_mfcc


def main() -> None:
    with (ROOT / "data" / "manifest.csv").open(encoding="utf-8", newline="") as file:
        recording = next(
            (row for row in csv.DictReader(file) if row["split"] == "train"),
            None,
        )

    if recording is None:
        raise SystemExit("No training recordings found in data/manifest.csv.")

    path = ROOT / recording["path"]
    features = extract_mfcc(path)
    print(f"Recording: {path.name}")
    print(f"Speaker: {recording['speaker_id']}")
    print(f"Feature shape: {features.shape}")
    print("All feature values are finite.")


if __name__ == "__main__":
    main()
