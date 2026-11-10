import csv
from pathlib import Path

import numpy as np

from extract_mfcc_sequence import extract_mfcc_sequence

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

with (DATA / "manifest.csv").open(encoding="utf-8", newline="") as file:
    rows = list(csv.DictReader(file))

arrays = {}
max_frames = None

for split, expected_count in (("train", 960), ("validation", 240)):
    recordings = [row for row in rows if row["split"] == split]
    if len(recordings) != expected_count:
        raise SystemExit(f"{split}: expected {expected_count} recordings, got {len(recordings)}")

    sequences = []
    for index, row in enumerate(recordings, start=1):
        sequences.append(extract_mfcc_sequence(ROOT / row["path"]))
        if index % 100 == 0 or index == len(recordings):
            print(f"{split}: processed {index}/{len(recordings)}", flush=True)

    if split == "train":
        max_frames = max(sequence.shape[1] for sequence in sequences)
        print(f"Frame limit from training: {max_frames}")

    features = np.zeros(
        (len(sequences), sequences[0].shape[0], max_frames),
        dtype=np.float32,
    )
    lengths = np.empty(len(sequences), dtype=np.int32)
    truncated = 0

    for index, sequence in enumerate(sequences):
        original_length = sequence.shape[1]
        length = min(original_length, max_frames)
        features[index, :, :length] = sequence[:, :length]
        lengths[index] = length
        truncated += int(original_length > max_frames)

    arrays[f"X_{split}"] = features
    arrays[f"y_{split}"] = np.array([row["emotion_code"] for row in recordings])
    arrays[f"lengths_{split}"] = lengths
    print(f"{split}: shape={features.shape}, truncated={truncated}")

output = DATA / "mfcc_sequences_dev.npz"
np.savez_compressed(output, **arrays)
print(f"Saved {output.relative_to(ROOT).as_posix()}")