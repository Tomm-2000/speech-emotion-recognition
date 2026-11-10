import csv
from pathlib import Path

import librosa
import numpy as np

ROOT = Path(__file__).resolve().parent
SAMPLE_RATE = 16_000
N_MFCC = 40


def extract_mfcc(path: Path) -> np.ndarray:
    audio, sample_rate = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=N_MFCC)

    features = np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)])
    if features.shape != (2 * N_MFCC,) or not np.isfinite(features).all():
        raise ValueError(f"Invalid MFCC features for {path}")

    return features


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