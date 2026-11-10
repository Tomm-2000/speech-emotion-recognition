import argparse
import csv
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from extract_wav2vec2 import MODEL_ID, SAMPLE_RATE

ROOT = Path(__file__).resolve().parent

CONFIGS = {
    "last": {
        "features": "wav2vec2_dev.npz",
        "output": "wav2vec2_logistic_regression.joblib",
        "pooling": "mean_last_hidden_state",
    },
    "all_transformer_mean": {
        "features": "wav2vec2_all_layers_dev.npz",
        "output": "wav2vec2_all_layers_logistic_regression.joblib",
        "pooling": "all_transformer_mean",
    },
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pooling",
        choices=list(CONFIGS),
        default="last",
    )
    args = parser.parse_args()
    config = CONFIGS[args.pooling]

    output = ROOT / "artifacts" / config["output"]
    if output.exists():
        raise SystemExit(
            f"Model already exists: {output}\n"
            "Existing model was not overwritten."
        )

    with (ROOT / "data" / "manifest.csv").open(
        encoding="utf-8", newline=""
    ) as file:
        rows = list(csv.DictReader(file))

    dev_rows = [
        row
        for split in ("train", "validation")
        for row in rows
        if row["split"] == split
    ]
    dev_speakers = {row["speaker_id"] for row in dev_rows}
    test_speakers = {
        row["speaker_id"] for row in rows if row["split"] == "test"
    }

    if len(dev_rows) != 1200 or len(dev_speakers) != 20:
        raise SystemExit("Expected 1,200 development recordings and 20 speakers.")
    if dev_speakers & test_speakers:
        raise SystemExit("Development and test speakers overlap.")
    if len({row["path"] for row in dev_rows}) != len(dev_rows):
        raise SystemExit("Duplicate development recording paths.")

    feature_path = ROOT / "data" / config["features"]
    with np.load(feature_path, allow_pickle=False) as data:
        features = np.concatenate([
            data["X_train"], data["X_validation"]
        ])
        targets = np.concatenate([
            data["y_train"], data["y_validation"]
        ])

        for key, column in (
            ("speakers", "speaker_id"),
            ("paths", "path"),
        ):
            saved_values = np.concatenate([
                data[f"{key}_train"], data[f"{key}_validation"]
            ])
            if saved_values.tolist() != [row[column] for row in dev_rows]:
                raise SystemExit(f"Saved {column} values do not match the manifest.")

        if args.pooling == "all_transformer_mean":
            expected_metadata = {
                "encoder_id": MODEL_ID,
                "sample_rate": SAMPLE_RATE,
                "pooling": args.pooling,
                "transformer_layer_count": 12,
            }
            for key, expected in expected_metadata.items():
                if key not in data or data[key].item() != expected:
                    raise SystemExit(f"Feature metadata mismatch: {key}")

    if features.shape != (1200, 768) or targets.shape != (1200,):
        raise SystemExit("Unexpected feature or target shape.")
    if not np.isfinite(features).all():
        raise SystemExit("Features contain invalid values.")
    if targets.tolist() != [row["emotion_code"] for row in dev_rows]:
        raise SystemExit("Saved labels do not match the manifest.")
    if set(targets) != {f"{number:02d}" for number in range(1, 9)}:
        raise SystemExit("Expected eight emotion classes.")

    classifier = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000),
    )
    classifier.fit(features, targets)

    artifact = {
        "classifier": classifier,
        "encoder_id": MODEL_ID,
        "sample_rate": SAMPLE_RATE,
        "pooling": config["pooling"],
        "feature_file": config["features"],
        "training_recordings": len(targets),
        "training_speakers": sorted(dev_speakers),
        "training_paths": [row["path"] for row in dev_rows],
    }
    if args.pooling == "all_transformer_mean":
        artifact["transformer_layers"] = list(range(1, 13))

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as file:
        joblib.dump(artifact, file)

    print(f"Pooling: {config['pooling']}")
    print(f"Training recordings: {len(targets)}")
    print(f"Training speakers: {len(dev_speakers)}")
    print(f"Saved: {output.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()