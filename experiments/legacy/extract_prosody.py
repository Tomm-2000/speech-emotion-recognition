import csv
from pathlib import Path

import librosa
import numpy as np

from .extract_features import SAMPLE_RATE

ROOT = Path(__file__).resolve().parents[2]
FRAME_LENGTH = 1024
HOP_LENGTH = 256

FEATURE_NAMES = (
    "pitch_mean_hz",
    "pitch_std_hz",
    "voiced_fraction",
    "level_mean_db",
    "level_std_db",
)


def extract_prosody(path: Path) -> np.ndarray:
    audio, sample_rate = librosa.load(path, sr=SAMPLE_RATE, mono=True)

    pitch, voiced, _ = librosa.pyin(
        audio,
        fmin=65,
        fmax=500,
        sr=sample_rate,
        frame_length=FRAME_LENGTH,
        hop_length=HOP_LENGTH,
    )
    valid_pitch = voiced & np.isfinite(pitch)
    voiced_pitch = pitch[valid_pitch]

    # Use zeros for pitch statistics if no pitch could be estimated.
    pitch_mean = voiced_pitch.mean() if voiced_pitch.size else 0.0
    pitch_std = voiced_pitch.std() if voiced_pitch.size else 0.0

    rms = librosa.feature.rms(
        y=audio,
        frame_length=FRAME_LENGTH,
        hop_length=HOP_LENGTH,
    )[0]
    level_db = 20 * np.log10(np.maximum(rms, 1e-8))

    features = np.array(
        [pitch_mean, pitch_std, valid_pitch.mean(), level_db.mean(), level_db.std()],
        dtype=np.float32,
    )
    if not np.isfinite(features).all():
        raise ValueError(f"Invalid prosody features for {path}")
    return features


if __name__ == "__main__":
    manifest = ROOT / "data" / "manifest.csv"
    with manifest.open(encoding="utf-8", newline="") as file:
        recording = next(row for row in csv.DictReader(file) if row["split"] == "train")

    path = ROOT / recording["path"]
    features = extract_prosody(path)

    print(f"Recording: {path.name}")
    for name, value in zip(FEATURE_NAMES, features):
        print(f"{name}: {value:.3f}")
