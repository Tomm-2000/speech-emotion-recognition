import argparse
import csv
import hashlib
import importlib.metadata
import json
import random
import warnings
from pathlib import Path
from time import perf_counter

import numpy as np
import yaml
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .paths import ROOT
LABELS = np.array([f"{number:02d}" for number in range(1, 9)])
PACKAGES = (
    "numpy", "torch", "transformers", "huggingface-hub",
    "librosa", "scikit-learn", "PyYAML",
)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path, value):
    with path.open("x", encoding="utf-8") as file:
        json.dump(value, file, indent=2, allow_nan=False)


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def versions():
    return {
        package: importlib.metadata.version(package)
        for package in PACKAGES
    }


def macro_f1(actual, predicted):
    return float(f1_score(
        actual, predicted, labels=LABELS,
        average="macro", zero_division=0,
    ))


def load_data(config):
    manifest_path = ROOT / config["data"]["manifest"]
    folds_path = ROOT / config["data"]["folds"]

    with manifest_path.open(encoding="utf-8-sig", newline="") as file:
        all_rows = list(csv.DictReader(file))

    rows = [
        row
        for split in ("train", "validation")
        for row in all_rows if row["split"] == split
    ]
    folds = read_json(folds_path)

    if len(rows) != config["data"]["expected_recordings"]:
        raise ValueError("Unexpected development recording count.")

    paths = [row["path"] for row in rows]
    speakers = {row["speaker_id"] for row in rows}
    test_speakers = {
        row["speaker_id"] for row in all_rows if row["split"] == "test"
    }

    if len(paths) != len(set(paths)):
        raise ValueError("Duplicate development paths.")
    if len(speakers) != config["data"]["expected_speakers"]:
        raise ValueError("Unexpected speaker count.")
    if speakers & test_speakers:
        raise ValueError("Development/test speaker overlap.")
    if any(row["emotion_code"] not in LABELS for row in rows):
        raise ValueError("Unexpected label.")

    assigned = [speaker for group in folds.values() for speaker in group]
    if len(folds) != 5 or any(len(group) != 4 for group in folds.values()):
        raise ValueError("Expected five groups of four speakers.")
    if len(assigned) != len(set(assigned)) or set(assigned) != speakers:
        raise ValueError("Invalid fold assignments.")
    for group in folds.values():
        if sum(row["speaker_id"] in group for row in rows) != 240:
            raise ValueError("Expected 240 recordings per evaluation fold.")

    return rows, folds, manifest_path, folds_path


def core_config(config):
    # The ensemble section does not alter an existing encoder experiment.
    return {key: value for key, value in config.items() if key not in {"ensemble", "regularization", "fitting"}}


def verify_source_lock(directory, config, manifest_path, folds_path):
    lock = read_json(directory / "run_lock.json")
    identity = lock["identity"]

    if core_config(identity["config"]) != core_config(config):
        raise ValueError("Source experiment configuration differs.")
    if identity["manifest_sha256"] != file_hash(manifest_path):
        raise ValueError("Manifest changed since feature extraction.")
    if identity["folds_sha256"] != file_hash(folds_path):
        raise ValueError("Fold definitions changed.")
    if identity["versions"] != versions():
        raise ValueError("Package versions changed since the source run.")

    return lock


def prepare_encoder_run(config, manifest_path, folds_path):
    output = ROOT / config["output"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    lock_path = output / "run_lock.json"

    if lock_path.exists():
        lock = verify_source_lock(
            output, config, manifest_path, folds_path
        )
        if lock["identity"]["runner_sha256"] != file_hash(Path(__file__)):
            print(
                "Runner code has changed since extraction. "
                "The source lock is preserved; current code is recorded "
                "with any new evaluation."
            )
    else:
        from huggingface_hub import HfApi
        revision = HfApi().model_info(
            config["encoder"]["model_id"],
            revision=config["encoder"]["revision"],
        ).sha
        if not revision:
            raise ValueError("Could not resolve the model revision.")

        lock = {
            "identity": {
                "config": core_config(config),
                "manifest_sha256": file_hash(manifest_path),
                "folds_sha256": file_hash(folds_path),
                "runner_sha256": file_hash(Path(__file__)),
                "versions": versions(),
            },
            "resolved_revision": revision,
        }
        write_json(lock_path, lock)
        with (output / "config_snapshot.yaml").open(
            "x", encoding="utf-8"
        ) as file:
            yaml.safe_dump(core_config(config), file, sort_keys=False)

    return output, lock["resolved_revision"]


def load_encoder(config, revision):
    from transformers import AutoFeatureExtractor, AutoModel
    settings = config["encoder"]
    processor = AutoFeatureExtractor.from_pretrained(
        settings["model_id"], revision=revision,
        trust_remote_code=False,
    )
    model = AutoModel.from_pretrained(
        settings["model_id"], revision=revision,
        trust_remote_code=False,
    )

    if processor.sampling_rate != settings["sample_rate"]:
        raise ValueError("Unexpected sample rate.")
    if processor.do_normalize != settings["expected_do_normalize"]:
        raise ValueError("Unexpected normalization.")
    if model.config.num_hidden_layers != settings["expected_layers"]:
        raise ValueError("Unexpected layer count.")
    if model.config.hidden_size != settings["expected_width"]:
        raise ValueError("Unexpected feature width.")

    model.to(settings["device"])
    model.eval()
    model.requires_grad_(False)
    print(f"Encoder: {settings['model_id']}")
    print(f"Revision: {revision}")
    print(f"Processor normalization: {processor.do_normalize}")
    return processor, model


def extract_one(row, config, processor, model):
    import torch
    with torch.inference_mode():
        return _extract_one(row, config, processor, model)


def _extract_one(row, config, processor, model):
    import librosa
    import torch
    settings = config["encoder"]
    audio, sample_rate = librosa.load(
        ROOT / row["path"], sr=settings["sample_rate"], mono=True
    )
    if audio.size == 0 or not np.isfinite(audio).all():
        raise ValueError(f"Invalid audio: {row['path']}")

    inputs = processor(
        audio, sampling_rate=sample_rate, return_tensors="pt"
    )
    inputs = {
        key: value.to(settings["device"])
        for key, value in inputs.items()
    }
    outputs = model(
        **inputs, output_hidden_states=True, return_dict=True
    )
    layers = outputs.hidden_states[1:]
    if len(layers) != settings["expected_layers"]:
        raise ValueError("Unexpected hidden states.")

    embedding = torch.stack([
        layer[0].mean(dim=0) for layer in layers
    ]).mean(dim=0).cpu().numpy().astype(np.float32)

    if (
        embedding.shape != (settings["expected_width"],)
        or not np.isfinite(embedding).all()
    ):
        raise ValueError("Invalid embedding.")

    return embedding, int(layers[0].shape[1])


def extract_stage(stage, rows, config, output, revision):
    destination = output / (
        "smoke_results.json" if stage == "smoke" else "features.npz"
    )
    if destination.exists():
        raise FileExistsError(f"Already exists: {destination}")
    if stage == "extract" and not (output / "smoke_results.json").exists():
        raise ValueError("Run smoke first.")

    selected = (
        [row for row in rows if row["split"] == "train"]
        [:config["runtime"]["smoke_recordings"]]
        if stage == "smoke" else rows
    )
    if not selected:
        raise ValueError("No recordings selected.")

    processor, model = load_encoder(config, revision)
    started = perf_counter()
    features, lengths = [], []

    for index, row in enumerate(selected, start=1):
        feature, length = extract_one(row, config, processor, model)
        features.append(feature)
        lengths.append(length)
        if stage == "smoke" or index % 100 == 0 or index == len(selected):
            print(f"Processed {index}/{len(selected)}", flush=True)

    features = np.stack(features)
    seconds = perf_counter() - started

    if stage == "smoke":
        write_json(destination, {
            "recordings": len(selected),
            "paths": [row["path"] for row in selected],
            "shape": list(features.shape),
            "frames": lengths,
            "finite": True,
            "seconds_excluding_model_load": seconds,
            "estimated_full_extraction_minutes":
                seconds / len(selected) * len(rows) / 60,
            "runner_sha256": file_hash(Path(__file__)),
        })
        print(f"Feature shape: {features.shape}")
        print("Smoke check passed. No classifier was trained.")
    else:
        with destination.open("xb") as file:
            np.savez_compressed(
                file,
                X=features,
                y=np.array([row["emotion_code"] for row in rows]),
                speakers=np.array([row["speaker_id"] for row in rows]),
                paths=np.array([row["path"] for row in rows]),
                revision=np.array(revision),
            )

    print(f"Saved: {destination.relative_to(ROOT)}")


def check_features(X, y, speakers, paths, rows):
    for saved, column in (
        (y, "emotion_code"), (speakers, "speaker_id"), (paths, "path")
    ):
        if saved.tolist() != [row[column] for row in rows]:
            raise ValueError(f"Feature metadata mismatch: {column}")

    if X.shape != (len(rows), 768) or not np.isfinite(X).all():
        raise ValueError("Expected finite features with shape (1200, 768).")


def load_candidate(directory, revision, rows):
    with np.load(directory / "features.npz", allow_pickle=False) as data:
        X, y = data["X"], data["y"]
        speakers, paths = data["speakers"], data["paths"]
        if data["revision"].item() != revision:
            raise ValueError("Feature revision mismatch.")
    check_features(X, y, speakers, paths, rows)
    return X, y, speakers, paths


def load_reference(path, rows):
    with np.load(path, allow_pickle=False) as data:
        def combine(key):
            return np.concatenate([
                data[f"{key}_train"], data[f"{key}_validation"]
            ])
        X, y = combine("X"), combine("y")
        speakers, paths = combine("speakers"), combine("paths")
    check_features(X, y, speakers, paths, rows)
    return X


def fit_probabilities(X, y, training, evaluation, config):
    settings = config["classifier"]
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=settings["C"],
            max_iter=settings["max_iter"],
            random_state=config["runtime"]["seed"],
        ),
    )
    # An unconverged fit must not silently enter the comparison.
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(X[training], y[training])

    probabilities = model.predict_proba(X[evaluation])
    classes = model.classes_
    if set(classes) != set(LABELS):
        raise ValueError("Classifier classes do not match all eight labels.")

    order = [int(np.flatnonzero(classes == label)[0]) for label in LABELS]
    probabilities = probabilities[:, order]

    if (
        not np.isfinite(probabilities).all()
        or np.any(probabilities < 0)
        or not np.allclose(probabilities.sum(axis=1), 1.0)
    ):
        raise ValueError("Invalid probabilities.")

    return probabilities


def save_predictions(path, rows, fold_names, predictions):
    with path.open("x", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["mode", "fold", "path", "speaker", "actual", "predicted"])
        for mode, values in predictions.items():
            for index, row in enumerate(rows):
                writer.writerow([
                    mode, fold_names[index], row["path"],
                    row["speaker_id"], row["emotion_code"], values[index],
                ])


def evaluate(rows, folds, config, output, revision):
    report_path = output / "evaluation.json"
    predictions_path = output / "oof_predictions.csv"
    if report_path.exists() or predictions_path.exists():
        raise FileExistsError("Evaluation outputs already exist.")

    X, y, speakers, paths = load_candidate(output, revision, rows)
    predictions = np.empty(len(rows), dtype="<U2")
    fold_names = np.empty(len(rows), dtype=object)
    visits = np.zeros(len(rows), dtype=int)
    scores = {}

    for fold, group in folds.items():
        evaluation = np.isin(speakers, group)
        probabilities = fit_probabilities(
            X, y, ~evaluation, evaluation, config
        )
        predicted = LABELS[probabilities.argmax(axis=1)]
        predictions[evaluation] = predicted
        fold_names[evaluation] = fold
        visits[evaluation] += 1
        scores[fold] = macro_f1(y[evaluation], predicted)
        print(f"{fold}: macro F1={scores[fold]:.3f}", flush=True)

    if not np.all(visits == 1):
        raise ValueError("Invalid evaluation coverage.")

    with predictions_path.open("x", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["fold", "path", "speaker", "actual", "predicted"])
        writer.writerows(zip(fold_names, paths, speakers, y, predictions))

    values = list(scores.values())
    write_json(report_path, {
        "fold_scores": scores,
        "mean_macro_f1": float(np.mean(values)),
        "std_across_folds": float(np.std(values)),
        "pooled_oof_report": classification_report(
            y, predictions, labels=LABELS,
            output_dict=True, zero_division=0,
        ),
        "classifier": config["classifier"],
        "revision": revision,
        "runner_sha256": file_hash(Path(__file__)),
    })
    print(f"Mean macro F1: {np.mean(values):.3f}")
    print(f"Saved: {report_path.relative_to(ROOT)}")


def ensemble(rows, folds, config, manifest_path, folds_path):
    settings = config["ensemble"]
    weights = np.array([
        settings["reference_weight"], settings["candidate_weight"]
    ], dtype=float)

    if not np.array_equal(weights, np.array([0.5, 0.5])):
        raise ValueError("This registered experiment requires equal weights.")
    if config["classifier"]["C"] != 1.0:
        raise ValueError("This experiment requires C=1.")

    source = ROOT / settings["candidate_directory"]
    lock = verify_source_lock(source, config, manifest_path, folds_path)
    revision = lock["resolved_revision"]

    reference_path = ROOT / settings["reference_features"]
    X_ref = load_reference(reference_path, rows)
    X_new, y, speakers, paths = load_candidate(source, revision, rows)
    old_report = read_json(source / "evaluation.json")

    output = ROOT / settings["output_directory"]
    if output.exists():
        raise FileExistsError(
            f"Experiment directory already exists: {output}. "
            "Existing results were not changed."
        )

    modes = ["wav2vec2", "wavlm", "equal_ensemble"]
    probabilities = {
        mode: np.empty((len(rows), len(LABELS)), dtype=np.float64)
        for mode in modes
    }
    scores = {mode: {} for mode in modes}
    fold_names = np.empty(len(rows), dtype=object)
    visits = np.zeros(len(rows), dtype=int)
    started = perf_counter()

    print("Fixed equal-probability ensemble | C=1")
    print("No audio extraction, hyperparameter search or test evaluation.")

    for fold, group in folds.items():
        evaluation = np.isin(speakers, group)
        training = ~evaluation

        first = fit_probabilities(
            X_ref, y, training, evaluation, config
        )
        second = fit_probabilities(
            X_new, y, training, evaluation, config
        )
        combined = 0.5 * first + 0.5 * second

        for mode, values in zip(modes, (first, second, combined)):
            probabilities[mode][evaluation] = values
            predicted = LABELS[values.argmax(axis=1)]
            scores[mode][fold] = macro_f1(y[evaluation], predicted)

        expected = float(old_report["fold_scores"][fold])
        if not np.isclose(scores["wavlm"][fold], expected, atol=1e-10, rtol=0):
            raise ValueError(
                f"{fold}: WavLM does not reproduce its stored score. "
                f"Stored={expected}, new={scores['wavlm'][fold]}. "
                "No ensemble results were saved."
            )

        fold_names[evaluation] = fold
        visits[evaluation] += 1
        print(
            f"{fold}: wav2vec2={scores['wav2vec2'][fold]:.3f} | "
            f"wavlm={scores['wavlm'][fold]:.3f} | "
            f"ensemble={scores['equal_ensemble'][fold]:.3f}",
            flush=True,
        )

    if not np.all(visits == 1):
        raise ValueError("Invalid evaluation coverage.")

    predictions = {
        mode: LABELS[values.argmax(axis=1)]
        for mode, values in probabilities.items()
    }
    summary = {}
    for mode in modes:
        values = list(scores[mode].values())
        summary[mode] = {
            "fold_scores": scores[mode],
            "mean_macro_f1": float(np.mean(values)),
            "std_across_folds": float(np.std(values)),
            "pooled_oof_report": classification_report(
                y, predictions[mode], labels=LABELS,
                output_dict=True, zero_division=0,
            ),
        }

    print("\nDevelopment comparison")
    for mode in modes:
        print(f"{mode}: mean macro F1={summary[mode]['mean_macro_f1']:.3f}")

    for reference in ("wav2vec2", "wavlm"):
        differences = [
            scores["equal_ensemble"][fold] - scores[reference][fold]
            for fold in folds
        ]
        print(
            f"Ensemble minus {reference}: "
            f"{np.mean(differences):+.3f}"
        )

    # Create the new directory only after computation and checks succeed.
    output.mkdir(parents=True, exist_ok=False)
    with (output / "config_snapshot.yaml").open(
        "x", encoding="utf-8"
    ) as file:
        yaml.safe_dump(config, file, sort_keys=False)

    write_json(output / "provenance.json", {
        "runner_sha256": file_hash(Path(__file__)),
        "manifest_sha256": file_hash(manifest_path),
        "folds_sha256": file_hash(folds_path),
        "reference_features_sha256": file_hash(reference_path),
        "candidate_features_sha256": file_hash(source / "features.npz"),
        "candidate_revision": revision,
        "versions": versions(),
        "note": (
            "The historical reference checkpoint revision is not inferred. "
            "The exact reference feature file is identified by its hash."
        ),
    })

    save_predictions(
        output / "oof_predictions.csv", rows, fold_names, predictions
    )
    with (output / "oof_probabilities.npz").open("xb") as file:
        np.savez_compressed(
            file,
            **probabilities,
            labels=LABELS,
            paths=paths,
            actual=y,
            speakers=speakers,
        )
    write_json(output / "evaluation.json", {
        "results": summary,
        "elapsed_seconds": perf_counter() - started,
        "note": (
            "Fixed equal-weight ensemble on development speaker folds. "
            "No weight tuning or test evaluation. "
            "Fold standard deviation is not a confidence interval."
        ),
    })
    print(f"Saved: {output.relative_to(ROOT)}")
