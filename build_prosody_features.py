import csv
from pathlib import Path

import numpy as np

from extract_prosody import extract_prosody

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
SPLIT_COUNTS = {"train": 960, "validation": 240}


def main():
    with (DATA / "manifest.csv").open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    combined = {}

    with np.load(DATA / "mfcc_dev.npz", allow_pickle=False) as baseline:
        for split, expected_count in SPLIT_COUNTS.items():
            recordings = [row for row in rows if row["split"] == split]
            mfcc = baseline[f"X_{split}"]
            labels = baseline[f"y_{split}"]
            expected_labels = [row["emotion_code"] for row in recordings]

            # The original MFCC builder uses manifest order within each split.
            if (
                len(recordings) != expected_count
                or mfcc.shape != (expected_count, 80)
                or labels.tolist() != expected_labels
            ):
                raise SystemExit(f"Saved MFCC data do not match the {split} manifest.")

            prosody = []
            for index, row in enumerate(recordings, start=1):
                prosody.append(extract_prosody(ROOT / row["path"]))
                if index % 50 == 0 or index == expected_count:
                    print(f"{split}: processed {index}/{expected_count}", flush=True)

            combined[f"X_{split}"] = np.concatenate([mfcc, np.stack(prosody)], axis=1)
            combined[f"y_{split}"] = labels

    output = DATA / "mfcc_prosody_dev.npz"
    np.savez_compressed(output, **combined)

    print(f"Saved {output.relative_to(ROOT).as_posix()}")
    for split in SPLIT_COUNTS:
        print(f"{split} feature shape: {combined[f'X_{split}'].shape}")


if __name__ == "__main__":
    main()