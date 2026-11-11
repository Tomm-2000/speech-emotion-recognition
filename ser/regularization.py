import argparse
import csv
import json
import warnings
from pathlib import Path
from time import perf_counter

import numpy as np
import yaml
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .paths import ROOT
LABELS = np.array([f"{number:02d}" for number in range(1, 9)])
ENCODERS = ("wav2vec2", "wavlm")


def build_classifier(C, config):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=float(C),
            max_iter=config["classifier"]["max_iter"],
            random_state=config["runtime"]["seed"],
        ),
    )


def fit_model(X, y, training, C, config):
    model = build_classifier(C, config)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(X[training], y[training])

    if set(model.classes_) != set(LABELS):
        raise ValueError("The classifier did not learn all eight classes.")
    return model


def predict_probabilities(model, X):
    values = model.predict_proba(X)
    order = [
        int(np.flatnonzero(model.classes_ == label)[0])
        for label in LABELS
    ]
    values = values[:, order]

    if (
        not np.isfinite(values).all()
        or np.any(values < 0)
        or not np.allclose(values.sum(axis=1), 1.0)
    ):
        raise ValueError("Invalid classifier probabilities.")
    return values


def choose_C(
    X, y, speakers, outer_training, inner_folds,
    candidates, config, previous,
):
    search = []

    for C in candidates:
        inner_scores = {}
        for fold, group in inner_folds.items():
            validation = np.isin(speakers, group)
            training = outer_training & ~validation

            if np.any(validation & ~outer_training):
                raise ValueError("Outer evaluation leaked into inner validation.")
            if training.sum() != 720 or validation.sum() != 240:
                raise ValueError("Unexpected inner split sizes.")
            if set(speakers[training]) & set(speakers[validation]):
                raise ValueError("Inner speaker overlap.")

            model = fit_model(X, y, training, C, config)
            probabilities = predict_probabilities(model, X[validation])
            predictions = LABELS[probabilities.argmax(axis=1)]

            inner_scores[fold] = previous.macro_f1(
                y[validation], predictions
            )

        search.append({
            "C": float(C),
            "fold_scores": inner_scores,
            "mean_macro_f1": float(np.mean(list(inner_scores.values()))),
        })

    # Highest inner mean wins; exact ties choose the smaller C.
    chosen = max(
        search,
        key=lambda item: (item["mean_macro_f1"], -item["C"]),
    )
    return chosen["C"], search


def run_regularization(config, previous):
    settings = config["regularization"]
    candidates = sorted(set(float(value) for value in settings["candidates"]))

    if candidates != [0.001, 0.01, 0.1, 1.0]:
        raise ValueError("This registered experiment uses four fixed C values.")
    if float(settings["baseline_C"]) != 1.0:
        raise ValueError("The reference must use C=1.")
    if settings["selection"] != "inner_speaker_folds":
        raise ValueError("Unexpected selection protocol.")
    if settings["tie_break"] != "smaller_C":
        raise ValueError("Unexpected tie-breaking rule.")

    weights = (
        config["ensemble"]["reference_weight"],
        config["ensemble"]["candidate_weight"],
    )
    if weights != (0.5, 0.5):
        raise ValueError("Keep both ensemble weights fixed at 0.5.")

    rows, folds, manifest_path, folds_path = previous.load_data(config)
    source = ROOT / config["ensemble"]["candidate_directory"]
    source_lock = previous.verify_source_lock(
        source, config, manifest_path, folds_path
    )
    revision = source_lock["resolved_revision"]

    X_new, y, speakers, paths = previous.load_candidate(
        source, revision, rows
    )
    reference_path = ROOT / config["ensemble"]["reference_features"]
    X_ref = previous.load_reference(reference_path, rows)
    features = {"wav2vec2": X_ref, "wavlm": X_new}

    old_directory = ROOT / config["ensemble"]["output_directory"]
    old_report = json.loads(
        (old_directory / "evaluation.json").read_text(encoding="utf-8-sig")
    )
    provenance = json.loads(
        (old_directory / "provenance.json").read_text(encoding="utf-8-sig")
    )

    source_checks = {
        "manifest_sha256": previous.file_hash(manifest_path),
        "folds_sha256": previous.file_hash(folds_path),
        "reference_features_sha256": previous.file_hash(reference_path),
        "candidate_features_sha256": previous.file_hash(source / "features.npz"),
    }
    for key, value in source_checks.items():
        if provenance[key] != value:
            raise ValueError(f"Source differs from the original ensemble: {key}")
    if provenance["versions"] != previous.versions():
        raise ValueError("Environment differs from the original ensemble.")

    output = ROOT / settings["output_directory"]
    if output.exists():
        raise FileExistsError(
            f"Output already exists: {output}. No files were changed."
        )

    print("Nested speaker-based regularization experiment")
    print(f"C candidates: {candidates}")
    print("Equal ensemble weights: 0.5 / 0.5")
    print("Each encoder selects C independently using inner folds only.")
    print("Outer evaluation speakers are excluded from selection.")
    print("No audio extraction or test evaluation.\n")

    started = perf_counter()
    records = []
    modes = ("baseline", "selected")
    probabilities_oof = {
        mode: np.empty((len(rows), 8), dtype=np.float64)
        for mode in modes
    }
    fold_names = np.empty(len(rows), dtype=object)
    visits = np.zeros(len(rows), dtype=int)

    for outer_fold, evaluation_speakers in folds.items():
        print(f"Running {outer_fold}...", flush=True)
        evaluation = np.isin(speakers, evaluation_speakers)
        training = ~evaluation
        inner_folds = {
            name: group for name, group in folds.items()
            if name != outer_fold
        }

        selections = {}
        searches = {}
        for encoder in ENCODERS:
            selected_C, search = choose_C(
                features[encoder], y, speakers, training,
                inner_folds, candidates, config, previous,
            )
            selections[encoder] = selected_C
            searches[encoder] = search
            print(
                f"  {encoder}: selected C={selected_C}",
                flush=True,
            )

        result = {
            "fold": outer_fold,
            "selected_C": selections,
            "inner_search": searches,
            "models": {},
        }
        final_probabilities = {}

        for mode in modes:
            train_probabilities = {}
            evaluation_probabilities = {}
            individual_scores = {}

            for encoder in ENCODERS:
                C = 1.0 if mode == "baseline" else selections[encoder]
                model = fit_model(
                    features[encoder], y, training, C, config
                )
                train_probabilities[encoder] = predict_probabilities(
                    model, features[encoder][training]
                )
                evaluation_probabilities[encoder] = predict_probabilities(
                    model, features[encoder][evaluation]
                )

                individual_scores[encoder] = {
                    "C": C,
                    "training_macro_f1": previous.macro_f1(
                        y[training],
                        LABELS[train_probabilities[encoder].argmax(axis=1)],
                    ),
                    "evaluation_macro_f1": previous.macro_f1(
                        y[evaluation],
                        LABELS[evaluation_probabilities[encoder].argmax(axis=1)],
                    ),
                }

                if mode == "baseline":
                    expected = old_report["results"][encoder]["fold_scores"][outer_fold]
                    observed = individual_scores[encoder]["evaluation_macro_f1"]
                    if not np.isclose(observed, expected, atol=1e-10, rtol=0):
                        raise ValueError(
                            f"Baseline reproduction failed: {outer_fold}/{encoder}"
                        )

            train_mean = (
                0.5 * train_probabilities["wav2vec2"]
                + 0.5 * train_probabilities["wavlm"]
            )
            evaluation_mean = (
                0.5 * evaluation_probabilities["wav2vec2"]
                + 0.5 * evaluation_probabilities["wavlm"]
            )

            train_score = previous.macro_f1(
                y[training], LABELS[train_mean.argmax(axis=1)]
            )
            evaluation_score = previous.macro_f1(
                y[evaluation], LABELS[evaluation_mean.argmax(axis=1)]
            )

            if mode == "baseline":
                expected = old_report["results"]["equal_ensemble"]["fold_scores"][outer_fold]
                if not np.isclose(evaluation_score, expected, atol=1e-10, rtol=0):
                    raise ValueError(
                        f"Original ensemble score not reproduced: {outer_fold}"
                    )

            result["models"][mode] = {
                "individual": individual_scores,
                "ensemble_training_macro_f1": train_score,
                "ensemble_evaluation_macro_f1": evaluation_score,
            }
            final_probabilities[mode] = evaluation_mean

            print(
                f"  {mode}: ensemble train={train_score:.3f} | "
                f"evaluation={evaluation_score:.3f}",
                flush=True,
            )
            for encoder, values in individual_scores.items():
                print(
                    f"    {encoder}: train={values['training_macro_f1']:.3f} | "
                    f"evaluation={values['evaluation_macro_f1']:.3f}",
                    flush=True,
                )

        difference = (
            result["models"]["selected"]["ensemble_evaluation_macro_f1"]
            - result["models"]["baseline"]["ensemble_evaluation_macro_f1"]
        )
        result["paired_difference"] = difference
        print(f"  selected minus baseline: {difference:+.3f}\n", flush=True)

        for mode in modes:
            probabilities_oof[mode][evaluation] = final_probabilities[mode]
        fold_names[evaluation] = outer_fold
        visits[evaluation] += 1
        records.append(result)

    if not np.all(visits == 1):
        raise ValueError("Invalid OOF coverage.")

    summary = {}
    print("Development summary")
    for mode in modes:
        train_scores = [
            record["models"][mode]["ensemble_training_macro_f1"]
            for record in records
        ]
        evaluation_scores = [
            record["models"][mode]["ensemble_evaluation_macro_f1"]
            for record in records
        ]
        summary[mode] = {
            "mean_training_macro_f1": float(np.mean(train_scores)),
            "mean_evaluation_macro_f1": float(np.mean(evaluation_scores)),
            "std_evaluation_macro_f1": float(np.std(evaluation_scores)),
        }
        print(
            f"{mode}: train={np.mean(train_scores):.3f} | "
            f"evaluation={np.mean(evaluation_scores):.3f}"
        )

    mean_difference = float(np.mean([
        record["paired_difference"] for record in records
    ]))
    elapsed = perf_counter() - started
    print(f"Mean paired difference: {mean_difference:+.4f}")
    print(f"Total time: {elapsed / 60:.1f} minutes")

    # Save only after all reference and coverage checks succeed.
    output.mkdir(parents=True, exist_ok=False)

    with (output / "config_snapshot.yaml").open("x", encoding="utf-8") as file:
        yaml.safe_dump(config, file, sort_keys=False)

    report = {
        "records": records,
        "summary": summary,
        "mean_paired_difference": mean_difference,
        "elapsed_seconds": elapsed,
        "note": (
            "Development comparison. C selected independently for each encoder "
            "using four inner speaker folds. Ensemble weights fixed at 0.5. "
            "Outer evaluation excluded from selection. No test evaluation."
        ),
    }
    with (output / "evaluation.json").open("x", encoding="utf-8") as file:
        json.dump(report, file, indent=2, allow_nan=False)

    with (output / "oof_predictions.csv").open(
        "x", encoding="utf-8", newline=""
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["mode", "fold", "path", "speaker", "actual", "predicted"])
        for mode in modes:
            predicted = LABELS[probabilities_oof[mode].argmax(axis=1)]
            for index in range(len(rows)):
                writer.writerow([
                    mode, fold_names[index], paths[index],
                    speakers[index], y[index], predicted[index],
                ])

    with (output / "oof_probabilities.npz").open("xb") as file:
        np.savez_compressed(
            file,
            **probabilities_oof,
            paths=paths,
            speakers=speakers,
            actual=y,
            labels=LABELS,
        )

    with (output / "provenance.json").open("x", encoding="utf-8") as file:
        json.dump({
            **source_checks,
            "runner_sha256": previous.file_hash(Path(__file__)),
            "experiments_module_sha256": previous.file_hash(
                Path(__file__).with_name("experiments.py")
            ),
            "versions": previous.versions(),
            "candidate_revision": revision,
        }, file, indent=2)

    print(f"Saved: {output.relative_to(ROOT)}")
    print("Previous models, features and experiment results were preserved.")
