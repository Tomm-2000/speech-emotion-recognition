import argparse
import csv
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch import nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

LABELS = [f"{number:02d}" for number in range(1, 9)]

EMOTIONS = [
    "neutral",
    "calm",
    "happy",
    "sad",
    "angry",
    "fearful",
    "disgust",
    "surprised",
]

FEATURE_SETS = {
    "mfcc": ("mfcc_dev.npz", 80),
    "prosody": ("mfcc_prosody_dev.npz", 85),
    "wav2vec2": ("wav2vec2_dev.npz", 768),
    "wav2vec2_all_layers": ("wav2vec2_all_layers_dev.npz", 768),
}

RANDOM_SEED = 42
MAX_EPOCHS = 100
PATIENCE = 12


def macro_f1(targets, predictions):
    return f1_score(
        targets,
        predictions,
        labels=LABELS,
        average="macro",
        zero_division=0,
    )


def check_alignment(data, dev_rows, filename):
    for key, column in (
        ("speakers", "speaker_id"),
        ("paths", "path"),
    ):
        saved = np.concatenate([
            data[f"{key}_train"],
            data[f"{key}_validation"],
        ])

        expected = [row[column] for row in dev_rows]

        if saved.tolist() != expected:
            raise SystemExit(
                f"{filename}: saved {column} does not match the manifest."
            )


def check_labels(targets, dev_rows):
    expected = [row["emotion_code"] for row in dev_rows]

    if targets.tolist() != expected:
        raise SystemExit("Saved labels do not match the manifest.")

    if set(targets) != set(LABELS):
        raise SystemExit("Expected eight emotion classes.")


def load_features(name, dev_rows):
    filename, width = FEATURE_SETS[name]

    with np.load(DATA / filename, allow_pickle=False) as data:
        features = np.concatenate([
            data["X_train"],
            data["X_validation"],
        ])
        targets = np.concatenate([
            data["y_train"],
            data["y_validation"],
        ])

        if name in {"wav2vec2", "wav2vec2_all_layers"}:
            check_alignment(data, dev_rows, filename)

    if features.shape != (1200, width):
        raise SystemExit(
            f"{filename}: expected (1200, {width}), "
            f"found {features.shape}."
        )

    if not np.isfinite(features).all():
        raise SystemExit(f"{filename}: invalid feature values.")

    check_labels(targets, dev_rows)
    return features, targets


def check_inner_split(training, evaluation, inner_validation):
    inner_training = training & ~inner_validation

    if np.any(training & evaluation):
        raise ValueError("Outer training and evaluation overlap.")

    if np.any(inner_validation & ~training):
        raise ValueError("Inner validation is outside outer training.")

    if (
        inner_training.sum() != 720
        or inner_validation.sum() != 240
        or training.sum() != 960
        or evaluation.sum() != 240
    ):
        raise ValueError("Unexpected split sizes.")

    return inner_training


def new_mlp():
    return MLPClassifier(
        hidden_layer_sizes=(128,),
        activation="relu",
        solver="adam",
        alpha=0.001,
        batch_size=32,
        learning_rate_init=0.001,
        random_state=RANDOM_SEED,
        shuffle=True,
        early_stopping=False,
    )


def evaluate_mlp(
    features,
    targets,
    training,
    evaluation,
    inner_validation,
):
    inner_training = check_inner_split(
        training,
        evaluation,
        inner_validation,
    )

    scaler = StandardScaler()
    X_inner = scaler.fit_transform(features[inner_training])
    X_validation = scaler.transform(features[inner_validation])

    model = new_mlp()
    best_score = -1.0
    best_epoch = 0
    stale = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.partial_fit(
            X_inner,
            targets[inner_training],
            classes=np.array(LABELS),
        )

        score = macro_f1(
            targets[inner_validation],
            model.predict(X_validation),
        )

        if score > best_score:
            best_score = score
            best_epoch = epoch
            stale = 0
        else:
            stale += 1

        if stale >= PATIENCE:
            break

    # Refit from scratch on all outer-training recordings.
    final_scaler = StandardScaler()
    X_train = final_scaler.fit_transform(features[training])
    X_evaluation = final_scaler.transform(features[evaluation])

    final_model = new_mlp()

    for _ in range(best_epoch):
        final_model.partial_fit(
            X_train,
            targets[training],
            classes=np.array(LABELS),
        )

    score = macro_f1(
        targets[evaluation],
        final_model.predict(X_evaluation),
    )

    return score, best_epoch, best_score


class LayerMixer(nn.Module):
    def __init__(self, learned):
        super().__init__()

        self.layer_logits = nn.Parameter(
            torch.zeros(12),
            requires_grad=learned,
        )

        self.head = nn.Sequential(
            nn.Linear(768, 128),
            nn.ReLU(),
            nn.Linear(128, 8),
        )

    def forward(self, inputs):
        weights = torch.softmax(self.layer_logits, dim=0)
        mixed = (inputs * weights[None, :, None]).sum(dim=1)
        return self.head(mixed)


def initialize_mixer(learned):
    torch.manual_seed(RANDOM_SEED)

    model = LayerMixer(learned)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
    )
    generator = torch.Generator().manual_seed(RANDOM_SEED)

    return model, optimizer, generator


def train_mixer_epoch(
    model,
    optimizer,
    generator,
    inputs,
    targets,
):
    model.train()
    order = torch.randperm(len(targets), generator=generator)

    for start in range(0, len(targets), 32):
        batch = order[start:start + 32]

        optimizer.zero_grad()

        loss = nn.functional.cross_entropy(
            model(inputs[batch]),
            targets[batch],
        )

        if not torch.isfinite(loss).item():
            raise RuntimeError("Non-finite training loss.")

        loss.backward()
        optimizer.step()


@torch.inference_mode()
def score_mixer(model, inputs, targets):
    model.eval()
    predictions = model(inputs).argmax(dim=1).numpy()

    return f1_score(
        targets.numpy(),
        predictions,
        labels=list(range(8)),
        average="macro",
        zero_division=0,
    )


def prepare_layer_inputs(layers, training, evaluation):
    # Apply the same affine transformation to every layer.
    # Fit only on the relevant training recordings.
    scaler = StandardScaler().fit(
        layers[training].mean(axis=1)
    )

    def transform(mask):
        values = (
            layers[mask] - scaler.mean_[None, None, :]
        ) / scaler.scale_[None, None, :]

        return torch.from_numpy(values.astype(np.float32))

    return transform(training), transform(evaluation)


def compare_layer_mixing(dev_rows, masks):
    path = DATA / "wav2vec2_layerwise_dev.npz"

    with np.load(path, allow_pickle=False) as data:
        layers = np.concatenate([
            data["X_train"],
            data["X_validation"],
        ]).astype(np.float32)

        labels = np.concatenate([
            data["y_train"],
            data["y_validation"],
        ])

        check_alignment(data, dev_rows, path.name)

        expected_metadata = {
            "encoder_id": "facebook/wav2vec2-base",
            "sample_rate": 16000,
            "pooling": "layers",
            "transformer_layer_count": 12,
        }

        for key, expected in expected_metadata.items():
            if key not in data or data[key].item() != expected:
                raise SystemExit(f"Layer metadata mismatch: {key}")

        if not np.array_equal(
            data["layer_indices"],
            np.arange(1, 13),
        ):
            raise SystemExit("Unexpected layer order.")

    if layers.shape != (1200, 12, 768):
        raise SystemExit(
            f"Unexpected layer shape: {layers.shape}"
        )

    if not np.isfinite(layers).all():
        raise SystemExit("Invalid layer feature values.")

    check_labels(labels, dev_rows)

    targets = torch.tensor(
        [int(label) - 1 for label in labels],
        dtype=torch.long,
    )

    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)

    names = list(masks)
    results = {"fixed": [], "learned": []}
    records = []
    started = perf_counter()

    print(
        "Comparison: fixed vs learned layer mixing | metric: macro F1",
        flush=True,
    )
    print(
        f"CPU | seed={RANDOM_SEED} | hidden=128 | "
        f"max_epochs={MAX_EPOCHS} | patience={PATIENCE}",
        flush=True,
    )

    for index, fold_name in enumerate(names):
        evaluation = masks[fold_name]
        training = ~evaluation

        inner_name = names[(index + 1) % len(names)]
        inner_validation = masks[inner_name]

        inner_training = check_inner_split(
            training,
            evaluation,
            inner_validation,
        )

        X_inner, X_validation = prepare_layer_inputs(
            layers,
            inner_training,
            inner_validation,
        )
        X_train, X_evaluation = prepare_layer_inputs(
            layers,
            training,
            evaluation,
        )

        y_inner = targets[inner_training]
        y_validation = targets[inner_validation]
        y_train = targets[training]
        y_evaluation = targets[evaluation]

        for mode in ("fixed", "learned"):
            print(
                f"Running {fold_name}: {mode}...",
                flush=True,
            )
            learned = mode == "learned"

            # Same initial head parameters and batch-order seed.
            model, optimizer, generator = initialize_mixer(learned)

            best_score = -1.0
            best_epoch = 0
            stale = 0

            for epoch in range(1, MAX_EPOCHS + 1):
                train_mixer_epoch(
                    model,
                    optimizer,
                    generator,
                    X_inner,
                    y_inner,
                )

                current = score_mixer(
                    model,
                    X_validation,
                    y_validation,
                )

                if current > best_score:
                    best_score = current
                    best_epoch = epoch
                    stale = 0
                else:
                    stale += 1

                if stale >= PATIENCE:
                    break

            # Discard the inner model and refit from scratch.
            model, optimizer, generator = initialize_mixer(learned)

            for _ in range(best_epoch):
                train_mixer_epoch(
                    model,
                    optimizer,
                    generator,
                    X_train,
                    y_train,
                )

            outer_score = score_mixer(
                model,
                X_evaluation,
                y_evaluation,
            )

            weights = torch.softmax(
                model.layer_logits.detach(),
                dim=0,
            ).tolist()

            results[mode].append(outer_score)
            records.append({
                "fold": fold_name,
                "mode": mode,
                "inner_validation": inner_name,
                "selected_epochs": best_epoch,
                "inner_macro_f1": float(best_score),
                "evaluation_macro_f1": float(outer_score),
                "layer_weights": weights,
            })

            print(
                f"{fold_name}: {mode} | epochs={best_epoch} | "
                f"inner_macro_F1={best_score:.3f} | "
                f"evaluation_macro_F1={outer_score:.3f}",
                flush=True,
            )

    fixed = np.array(results["fixed"])
    learned = np.array(results["learned"])
    differences = learned - fixed

    print("\nPaired development comparison")

    for index, name in enumerate(names):
        print(
            f"{name}: fixed={fixed[index]:.3f} | "
            f"learned={learned[index]:.3f} | "
            f"delta={differences[index]:+.3f}"
        )

    for mode, values in (
        ("fixed", fixed),
        ("learned", learned),
    ):
        print(
            f"{mode}: mean={values.mean():.3f} | "
            f"std={values.std():.3f}"
        )

    print(
        f"Mean paired difference: {differences.mean():+.3f}"
    )

    elapsed = perf_counter() - started
    print(f"Total time: {elapsed:.1f} seconds")

    report = {
        "experiment": "fixed_vs_learned_layer_mixing",
        "feature_file": path.name,
        "seed": RANDOM_SEED,
        "hidden_units": 128,
        "optimizer": "Adam",
        "learning_rate": 0.001,
        "weight_decay": 0.0,
        "batch_size": 32,
        "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "device": "cpu",
        "torch_version": str(torch.__version__),
        "normalization": (
            "shared_scaler_fitted_on_training_layer_mean"
        ),
        "inner_split_rule": "next_fold_cyclically",
        "layer_indices": list(range(1, 13)),
        "folds": {
            name: sorted({
                row["speaker_id"]
                for row, included in zip(dev_rows, mask)
                if included
            })
            for name, mask in masks.items()
        },
        "records": records,
        "fixed_mean": float(fixed.mean()),
        "fixed_std": float(fixed.std()),
        "learned_mean": float(learned.mean()),
        "learned_std": float(learned.std()),
        "mean_paired_difference": float(differences.mean()),
        "elapsed_seconds": elapsed,
        "note": (
            "Exploratory development comparison using one random seed. "
            "No test recordings were used."
        ),
    }

    output = ROOT / "artifacts" / "layer_mixing_comparison.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(f"Saved: {output.relative_to(ROOT).as_posix()}")


def diagnose_best_model(dev_rows, masks):
    features, targets = load_features(
        "wav2vec2_all_layers",
        dev_rows,
    )

    oof_predictions = np.full(
        len(targets),
        "",
        dtype="<U2",
    )
    assignment_count = np.zeros(
        len(targets),
        dtype=int,
    )

    prediction_rows = []
    fold_reports = []
    speaker_reports = []
    started = perf_counter()

    print(
        "Diagnostic: all-layer mean + LogisticRegression(C=1)",
        flush=True,
    )

    for fold_name, evaluation in masks.items():
        training = ~evaluation

        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=1.0,
                max_iter=2000,
            ),
        )
        model.fit(
            features[training],
            targets[training],
        )

        train_predictions = model.predict(features[training])
        predictions = model.predict(features[evaluation])

        train_f1 = macro_f1(
            targets[training],
            train_predictions,
        )
        evaluation_f1 = macro_f1(
            targets[evaluation],
            predictions,
        )

        oof_predictions[evaluation] = predictions
        assignment_count[evaluation] += 1

        fold_reports.append({
            "fold": fold_name,
            "train_macro_f1": float(train_f1),
            "evaluation_macro_f1": float(evaluation_f1),
            "gap": float(train_f1 - evaluation_f1),
            "classification_report": classification_report(
                targets[evaluation],
                predictions,
                labels=LABELS,
                target_names=EMOTIONS,
                output_dict=True,
                zero_division=0,
            ),
            "confusion_matrix": confusion_matrix(
                targets[evaluation],
                predictions,
                labels=LABELS,
            ).tolist(),
        })

        print(
            f"{fold_name}: train={train_f1:.3f} | "
            f"evaluation={evaluation_f1:.3f} | "
            f"gap={train_f1 - evaluation_f1:+.3f}",
            flush=True,
        )

        indices = np.flatnonzero(evaluation)

        for index, predicted in zip(indices, predictions):
            row = dev_rows[index]

            prediction_rows.append({
                "fold": fold_name,
                "path": row["path"],
                "speaker": row["speaker_id"],
                "actual": targets[index],
                "predicted": predicted,
            })

    if not np.all(assignment_count == 1):
        raise RuntimeError(
            "Each development recording must be evaluated exactly once."
        )

    speakers = np.array([
        row["speaker_id"] for row in dev_rows
    ])

    for speaker in sorted(set(speakers)):
        mask = speakers == speaker

        speaker_reports.append({
            "speaker": speaker,
            "recordings": int(mask.sum()),
            "accuracy": float(
                np.mean(
                    targets[mask] == oof_predictions[mask]
                )
            ),
            "macro_f1": float(
                macro_f1(
                    targets[mask],
                    oof_predictions[mask],
                )
            ),
        })

    mean_train = float(np.mean([
        item["train_macro_f1"]
        for item in fold_reports
    ]))
    mean_evaluation = float(np.mean([
        item["evaluation_macro_f1"]
        for item in fold_reports
    ]))
    pooled_f1 = float(
        macro_f1(targets, oof_predictions)
    )

    print(
        f"\nMean training macro F1: {mean_train:.3f}"
    )
    print(
        f"Mean evaluation macro F1: {mean_evaluation:.3f}"
    )
    print(
        f"Pooled out-of-fold macro F1: {pooled_f1:.3f}"
    )

    print("\nOut-of-fold classification report:")
    print(
        classification_report(
            targets,
            oof_predictions,
            labels=LABELS,
            target_names=EMOTIONS,
            zero_division=0,
        )
    )

    print("Speaker results, lowest macro F1 first:")

    for item in sorted(
        speaker_reports,
        key=lambda item: item["macro_f1"],
    ):
        print(
            f"speaker={item['speaker']} | "
            f"macro F1={item['macro_f1']:.3f} | "
            f"accuracy={item['accuracy']:.3f}"
        )

    report = {
        "feature_set": "wav2vec2_all_layers",
        "classifier": "StandardScaler + LogisticRegression",
        "C": 1.0,
        "max_iter": 2000,
        "label_order": LABELS,
        "emotion_order": EMOTIONS,
        "mean_training_macro_f1": mean_train,
        "mean_evaluation_macro_f1": mean_evaluation,
        "pooled_oof_macro_f1": pooled_f1,
        "folds": fold_reports,
        "speakers": speaker_reports,
        "pooled_classification_report": classification_report(
            targets,
            oof_predictions,
            labels=LABELS,
            target_names=EMOTIONS,
            output_dict=True,
            zero_division=0,
        ),
        "pooled_confusion_matrix": confusion_matrix(
            targets,
            oof_predictions,
            labels=LABELS,
        ).tolist(),
        "note": (
            "Development diagnostic. Each OOF prediction comes from "
            "a model trained without that speaker. Training scores "
            "are in-sample. Test recordings are not evaluated."
        ),
    }

    artifacts = ROOT / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)

    report_path = artifacts / "best_model_diagnostics.json"
    predictions_path = (
        artifacts / "best_model_oof_predictions.csv"
    )

    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    with predictions_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "fold",
                "path",
                "speaker",
                "actual",
                "predicted",
            ],
        )
        writer.writeheader()
        writer.writerows(prediction_rows)

    print(
        f"\nSaved: {report_path.relative_to(ROOT).as_posix()}"
    )
    print(
        f"Saved: {predictions_path.relative_to(ROOT).as_posix()}"
    )
    print(
        f"Total time: {perf_counter() - started:.1f} seconds"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate fixed development speaker folds."
    )

    parser.add_argument(
        "features",
        nargs="*",
    )
    parser.add_argument(
        "--classifier",
        choices=("logistic", "svm", "mlp"),
        default="logistic",
    )

    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--compare-layer-mixing",
        action="store_true",
    )
    modes.add_argument(
        "--diagnose-best",
        action="store_true",
    )

    args = parser.parse_args()

    if args.diagnose_best or args.compare_layer_mixing:
        if args.features or args.classifier != "logistic":
            parser.error(
                "Use the selected diagnostic/comparison mode alone."
            )
    else:
        if not args.features:
            parser.error("Specify at least one feature set.")

        unknown = set(args.features) - set(FEATURE_SETS)

        if unknown:
            parser.error(
                f"Unknown feature sets: {sorted(unknown)}"
            )

        if len(set(args.features)) != len(args.features):
            parser.error("Specify each feature set only once.")

    folds = json.loads(
        (ROOT / "speaker_folds.json").read_text(
            encoding="utf-8"
        )
    )

    with (DATA / "manifest.csv").open(
        encoding="utf-8",
        newline="",
    ) as file:
        rows = list(csv.DictReader(file))

    # Preserve feature-builder order within each split.
    dev_rows = [
        row
        for split in ("train", "validation")
        for row in rows
        if row["split"] == split
    ]

    if len(dev_rows) != 1200:
        raise SystemExit(
            "Expected 1,200 development recordings."
        )

    if len({row["path"] for row in dev_rows}) != len(dev_rows):
        raise SystemExit(
            "Duplicate development recording paths."
        )

    speakers = np.array([
        row["speaker_id"] for row in dev_rows
    ])
    test_speakers = {
        row["speaker_id"]
        for row in rows
        if row["split"] == "test"
    }

    if (
        not isinstance(folds, dict)
        or len(folds) != 5
        or any(
            not isinstance(group, list) or len(group) != 4
            for group in folds.values()
        )
    ):
        raise SystemExit(
            "Expected five folds with four speakers each."
        )

    assigned = [
        speaker
        for group in folds.values()
        for speaker in group
    ]

    if (
        len(set(assigned)) != 20
        or set(assigned) != set(speakers)
    ):
        raise SystemExit(
            "Each development speaker must appear in exactly one fold."
        )

    if set(assigned) & test_speakers:
        raise SystemExit(
            "Test speakers overlap with development folds."
        )

    masks = {}

    for name, group in folds.items():
        evaluation = np.isin(speakers, group)

        if (
            evaluation.sum() != 240
            or (~evaluation).sum() != 960
        ):
            raise SystemExit(
                f"Unexpected recording counts for {name}."
            )

        masks[name] = evaluation

    # Dispatch after all split checks have passed.
    if args.diagnose_best:
        diagnose_best_model(dev_rows, masks)
        return

    if args.compare_layer_mixing:
        compare_layer_mixing(dev_rows, masks)
        return

    fold_names = list(masks)
    results = {}
    start = perf_counter()

    print(
        f"Classifier: {args.classifier} | metric: macro F1",
        flush=True,
    )

    if args.classifier == "mlp":
        print(
            f"MLP: hidden=128 | seed={RANDOM_SEED} | "
            f"max_epochs={MAX_EPOCHS} | patience={PATIENCE}",
            flush=True,
        )

    for feature_set in args.features:
        features, targets = load_features(
            feature_set,
            dev_rows,
        )
        scores = []

        for index, fold_name in enumerate(fold_names):
            evaluation = masks[fold_name]
            training = ~evaluation

            if args.classifier == "mlp":
                inner_name = fold_names[
                    (index + 1) % len(fold_names)
                ]

                print(
                    f"Running {feature_set}, {fold_name}...",
                    flush=True,
                )

                score, epochs, inner_score = evaluate_mlp(
                    features,
                    targets,
                    training,
                    evaluation,
                    masks[inner_name],
                )
                scores.append(score)

                print(
                    f"{fold_name}: inner_validation={inner_name} | "
                    f"selected_epochs={epochs} | "
                    f"inner_macro_F1={inner_score:.3f} | "
                    f"evaluation_macro_F1={score:.3f}",
                    flush=True,
                )
                continue

            if args.classifier == "svm":
                classifier = SVC(
                    kernel="rbf",
                    C=1.0,
                    gamma="scale",
                )
            else:
                classifier = LogisticRegression(
                    max_iter=2000
                )

            model = make_pipeline(
                StandardScaler(),
                classifier,
            )
            model.fit(
                features[training],
                targets[training],
            )
            predictions = model.predict(
                features[evaluation]
            )
            scores.append(
                macro_f1(
                    targets[evaluation],
                    predictions,
                )
            )

        results[feature_set] = np.array(scores)

    print("\nDevelopment-fold results")

    for index, name in enumerate(fold_names):
        values = " | ".join(
            f"{key}={scores[index]:.3f}"
            for key, scores in results.items()
        )
        print(f"{name}: {values}")

    for key, scores in results.items():
        print(
            f"{key}: mean={scores.mean():.3f} | "
            f"std={scores.std():.3f}"
        )

    if len(results) == 2:
        (reference, first), (candidate, second) = (
            results.items()
        )
        differences = second - first

        print(
            f"{candidate} minus {reference}: "
            f"{differences.mean():+.3f}"
        )
        print(
            "Fold differences: "
            + ", ".join(
                f"{value:+.3f}" for value in differences
            )
        )

    print(
        f"Total time: {perf_counter() - start:.1f} seconds"
    )


if __name__ == "__main__":
    main()