import csv
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from extract_wav2vec2 import (
    MODEL_ID,
    SAMPLE_RATE,
    extract_embedding,
    load_encoder,
)

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
ARTIFACTS = ROOT / "artifacts"

LABELS = ["01", "02", "03", "04", "05", "06", "07", "08"]
EMOTIONS = [
    "neutral", "calm", "happy", "sad",
    "angry", "fearful", "disgust", "surprised",
]


def main():
    model_path = ARTIFACTS / "wav2vec2_logistic_regression.joblib"
    if not model_path.is_file():
        raise SystemExit(f"Saved classifier not found: {model_path}")

    saved = joblib.load(model_path)

    expected_metadata = {
        "encoder_id": MODEL_ID,
        "sample_rate": SAMPLE_RATE,
        "pooling": "mean_last_hidden_state",
    }
    for key, expected in expected_metadata.items():
        if saved.get(key) != expected:
            raise SystemExit(
                f"Model metadata mismatch for {key}: "
                f"expected {expected!r}, found {saved.get(key)!r}"
            )

    classifier = saved["classifier"]
    if set(classifier.classes_) != set(LABELS):
        raise SystemExit("Saved classifier does not contain the expected classes.")

    with (DATA / "manifest.csv").open(
        encoding="utf-8", newline=""
    ) as file:
        rows = list(csv.DictReader(file))

    test_rows = [row for row in rows if row["split"] == "test"]
    dev_rows = [
        row for row in rows
        if row["split"] in {"train", "validation"}
    ]

    if len(test_rows) != 240:
        raise SystemExit(
            f"Expected 240 test recordings, found {len(test_rows)}."
        )

    # RAVDESS filenames end with the actor ID.
    def speaker_id(row):
        return Path(row["path"]).stem.split("-")[-1]

    test_speakers = {speaker_id(row) for row in test_rows}
    dev_speakers = {speaker_id(row) for row in dev_rows}

    if len(test_speakers) != 4:
        raise SystemExit("Expected exactly four test speakers.")
    if test_speakers & dev_speakers:
        raise SystemExit("Speaker overlap between development and test sets.")

    test_paths = [row["path"] for row in test_rows]
    if len(set(test_paths)) != len(test_paths):
        raise SystemExit("Duplicate recording paths in the test set.")

    y_test = np.array([row["emotion_code"] for row in test_rows])
    if set(y_test) != set(LABELS):
        raise SystemExit("Test labels do not match the expected eight classes.")

    print("Loading frozen audio encoder...", flush=True)
    feature_extractor, encoder = load_encoder()

    embeddings = []
    for index, row in enumerate(test_rows, start=1):
        embedding, _ = extract_embedding(
            ROOT / row["path"],
            feature_extractor,
            encoder,
        )
        embeddings.append(embedding)

        if index % 20 == 0 or index == len(test_rows):
            print(
                f"Test features: {index}/{len(test_rows)}",
                flush=True,
            )

    X_test = np.stack(embeddings)
    if X_test.shape != (240, 768) or not np.isfinite(X_test).all():
        raise SystemExit("Invalid test feature matrix.")

    # The saved pipeline applies its training-fitted scaler.
    # No fitting or parameter selection occurs here.
    predictions = classifier.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    macro_f1 = f1_score(
        y_test,
        predictions,
        labels=LABELS,
        average="macro",
        zero_division=0,
    )
    report = classification_report(
        y_test,
        predictions,
        labels=LABELS,
        target_names=EMOTIONS,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y_test, predictions, labels=LABELS)

    print(f"\nTest recordings: {len(y_test)}")
    print(f"Correct predictions: {np.count_nonzero(predictions == y_test)}")
    print(f"Test accuracy: {accuracy:.3f}")
    print(f"Test macro F1: {macro_f1:.3f}")
    print()
    print(
        classification_report(
            y_test,
            predictions,
            labels=LABELS,
            target_names=EMOTIONS,
            zero_division=0,
        )
    )

    results = {
        "model_file": model_path.name,
        **expected_metadata,
        "training_recordings": 1200,
        "test_recordings": len(y_test),
        "test_speakers": sorted(test_speakers),
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "classification_report": report,
        "label_order": LABELS,
        "emotion_order": EMOTIONS,
        "confusion_matrix": matrix.tolist(),
        "evaluation_note": (
            "This test set was previously evaluated with the MFCC baseline. "
            "It is not a previously unseen evaluation set for the project."
        ),
    }
    results_path = ARTIFACTS / "wav2vec2_test_results.json"
    results_path.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    predictions_path = ARTIFACTS / "wav2vec2_test_predictions.csv"
    with predictions_path.open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["path", "speaker", "actual", "predicted"])
        for row, actual, predicted in zip(test_rows, y_test, predictions):
            writer.writerow([
                row["path"], speaker_id(row), actual, predicted,
            ])

    print(f"Saved: {results_path.relative_to(ROOT).as_posix()}")
    print(f"Saved: {predictions_path.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()