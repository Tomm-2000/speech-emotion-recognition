import csv
from pathlib import Path

import numpy as np

from extract_features import extract_mfcc

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
EXPECTED_COUNTS = {"train": 960, "validation": 240}

with (DATA / "manifest.csv").open(encoding="utf-8", newline="") as file:
    rows = list(csv.DictReader(file))

arrays = {}

for split, expected_count in EXPECTED_COUNTS.items():
    recordings = [row for row in rows if row["split"] == split]
    if len(recordings) != expected_count:
        raise SystemExit(f"{split}: expected {expected_count} recordings, got {len(recordings)}")

    features = []
    for index, row in enumerate(recordings, start=1):
        features.append(extract_mfcc(ROOT / row["path"]))
        if index % 100 == 0 or index == len(recordings):
            print(f"{split}: processed {index}/{len(recordings)}", flush=True)

    arrays[f"X_{split}"] = np.stack(features)
    arrays[f"y_{split}"] = np.array([row["emotion_code"] for row in recordings])

output = DATA / "mfcc_dev.npz"
np.savez_compressed(output, **arrays)

print(f"Saved {output.relative_to(ROOT).as_posix()}")
for split in EXPECTED_COUNTS:
    print(f"{split} feature shape: {arrays[f'X_{split}'].shape}")