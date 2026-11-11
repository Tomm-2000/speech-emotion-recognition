"""Historical feature builders, consolidated behind one CLI."""
import argparse


def build_mfcc():
    import csv
    from pathlib import Path

    import numpy as np

    from .extract_features import extract_mfcc

    ROOT = Path(__file__).resolve().parents[2]
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


def build_sequences():
    import csv
    from pathlib import Path

    import numpy as np

    from .extract_mfcc_sequence import extract_mfcc_sequence

    ROOT = Path(__file__).resolve().parents[2]
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


import csv
from pathlib import Path

import numpy as np

from .extract_prosody import extract_prosody

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
SPLIT_COUNTS = {"train": 960, "validation": 240}


def build_prosody():
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



def main():
    parser = argparse.ArgumentParser(description="Historical baseline feature builders")
    parser.add_argument("kind", choices=["mfcc", "sequences", "prosody"])
    args = parser.parse_args()
    names = {"mfcc": "mfcc_dev.npz", "sequences": "mfcc_sequences_dev.npz", "prosody": "mfcc_prosody_dev.npz"}
    from ser.paths import ROOT
    output = ROOT / "data" / names[args.kind]
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite historical features: {output}")
    {"mfcc": build_mfcc, "sequences": build_sequences, "prosody": build_prosody}[args.kind]()

if __name__ == "__main__":
    main()
