"""Fit and verify the two frozen-encoder classifiers without test-set access."""
import json
from pathlib import Path

import joblib
import numpy as np

from . import experiments as core
from .inference import extract_features, ordered_probabilities
from .paths import ROOT
from .regularization import fit_model, predict_probabilities

ENCODERS = {
    "wav2vec2": {
        "model_id": "facebook/wav2vec2-base",
        "revision": "0b5b8e868dd84f03fd87d01f9c4ff0f080fecfe8",
        "do_normalize": True,
        "use_safetensors": False,
    },
    "wavlm": {
        "model_id": "microsoft/wavlm-base-plus",
        "revision": "4c66d4806a428f2e922ccfa1a962776e232d487b",
        "do_normalize": False,
    },
}


def output_directory(config):
    return ROOT / config.get("fitting", {}).get(
        "output_directory", "artifacts/models/equal_ensemble_refactor_v1"
    )


def source_features(config):
    rows, folds, manifest, fold_file = core.load_data(config)
    source = ROOT / config["ensemble"]["candidate_directory"]
    lock = core.verify_source_lock(source, config, manifest, fold_file)
    if lock["resolved_revision"] != ENCODERS["wavlm"]["revision"]:
        raise ValueError("Unexpected WavLM revision for this registered ensemble.")
    if config["encoder"]["model_id"] != ENCODERS["wavlm"]["model_id"]:
        raise ValueError("Expected the WavLM candidate encoder.")
    candidate, y, speakers, paths = core.load_candidate(source, lock["resolved_revision"], rows)
    ref = ROOT / config["ensemble"]["reference_features"]
    reference = core.load_reference(ref, rows)
    hashes = {"manifest": core.file_hash(manifest), "folds": core.file_hash(fold_file),
              "reference_features": core.file_hash(ref),
              "candidate_features": core.file_hash(source / "features.npz")}
    return rows, {"wav2vec2": reference, "wavlm": candidate}, y, speakers, paths, hashes


def fit(config):
    if config["classifier"]["C"] != 1.0 or (
        config["ensemble"]["reference_weight"], config["ensemble"]["candidate_weight"]
    ) != (0.5, 0.5):
        raise ValueError("This bundle requires C=1 and equal ensemble weights.")
    output = output_directory(config)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite: {output}")
    rows, features, y, speakers, paths, hashes = source_features(config)
    models = {}
    for name, X in features.items():
        print(f"Fitting {name}: {len(y)} development recordings...", flush=True)
        models[name] = fit_model(X, y, np.ones(len(y), dtype=bool), 1.0, config)
    bundle = {
        "classifiers": models, "labels": core.LABELS.tolist(),
        "weights": {"wav2vec2": 0.5, "wavlm": 0.5},
        "pooling": "all_transformer_mean", "sample_rate": 16000,
        "encoders": ENCODERS, "training_recordings": len(y),
        "training_speakers": len(set(speakers)), "training_paths": paths.tolist(),
        "classifier_C": 1.0, "source_hashes": hashes, "versions": core.versions(),
        "note": "Fit on development only. Audio verification is required before inference.",
    }
    output.mkdir(parents=True, exist_ok=False)
    joblib.dump(bundle, output / "ensemble.joblib")
    restored = joblib.load(output / "ensemble.joblib")
    differences = {}
    for name, X in features.items():
        before = predict_probabilities(models[name], X)
        after = predict_probabilities(restored["classifiers"][name], X)
        differences[name] = float(np.max(np.abs(before - after)))
        if not np.array_equal(before, after):
            raise ValueError(f"Serialization changed {name} predictions. Bundle is not verified.")
    selected = [int(np.flatnonzero(speakers == speaker)[0]) for speaker in sorted(set(speakers))]
    arrays = {name: X[selected] for name, X in features.items()}
    arrays["ensemble_probabilities"] = sum(
        0.5 * predict_probabilities(models[name], features[name][selected]) for name in models
    )
    with (output / "inference_reference.npz").open("xb") as handle:
        np.savez_compressed(handle, **arrays, paths=paths[selected], speakers=speakers[selected])
    core.write_json(output / "verification.json", {
        "training_recordings": len(y), "training_speakers": len(set(speakers)),
        "reload_probability_differences": differences, "audio_inference_verified": False,
    })
    core.write_json(output / "config_snapshot.json", config)
    print(f"Save/reload check passed. Saved {output / 'ensemble.joblib'}")
    print("Next: --stage verify. No test evaluation was performed.")


def verify(config, allow_download=False):
    import librosa
    import torch
    output = output_directory(config)
    destination = output / "ensemble_verified.joblib"
    if destination.exists() or (output / "audio_verification.json").exists():
        raise FileExistsError("Verification output exists; existing files were preserved.")
    bundle = joblib.load(output / "ensemble.joblib")  # Trusted project-generated file only.
    rows, _, y, speakers, paths, hashes = source_features(config)
    if hashes != bundle["source_hashes"]:
        raise ValueError("Source data changed since fitting.")
    torch.set_num_threads(config["runtime"]["cpu_threads"])
    with np.load(output / "inference_reference.npz", allow_pickle=False) as data:
        saved = {key: data[key] for key in data.files}
    allowed = set(paths.tolist())
    if len(saved["paths"]) != 20 or not set(saved["paths"]) <= allowed:
        raise ValueError("Verification must use twenty development recordings.")
    probabilities = {}
    differences = {}
    # Verification deliberately uses the same public inference function as predict.py.
    # It loads one model at a time; repeated cache loads are slower but bounded in RAM.
    for name in ENCODERS:
        extracted = []
        for index, path in enumerate(saved["paths"], 1):
            audio, rate = librosa.load(ROOT / str(path), sr=16000, mono=True)
            if not len(audio) or not np.isfinite(audio).all() or not np.any(audio):
                raise ValueError(f"Invalid verification audio: {path}")
            extracted.append(extract_features(audio, rate, bundle["encoders"][name], allow_download)[0])
            print(f"{name}: verified {index}/20", flush=True)
        X = np.stack(extracted)
        differences[name] = float(np.max(np.abs(X - saved[name])))
        if not np.allclose(X, saved[name], rtol=1e-4, atol=1e-5):
            raise ValueError(f"Audio features differ for {name}; no verified bundle was saved.")
        probabilities[name] = predict_probabilities(bundle["classifiers"][name], X)
    combined = 0.5 * probabilities["wav2vec2"] + 0.5 * probabilities["wavlm"]
    expected = saved["ensemble_probabilities"]
    identical = bool(np.array_equal(combined.argmax(1), expected.argmax(1)))
    if not identical or not np.allclose(combined, expected, rtol=0, atol=1e-4):
        raise ValueError("Audio-to-prediction consistency failed.")
    report = {
        "recordings": 20, "paths": saved["paths"].tolist(),
        "maximum_feature_differences": differences,
        "maximum_probability_difference": float(np.max(np.abs(combined - expected))),
        "identical_predicted_labels": identical,
        "note": "Development smoke verification, not an accuracy evaluation.",
    }
    bundle["audio_verification"] = report
    joblib.dump(bundle, destination)
    core.write_json(output / "audio_verification.json", report)
    print(f"Audio-to-prediction consistency passed. Saved: {destination}")


def reference(config):
    """Build pinned Wav2Vec2 development features in the historical NPZ format."""
    import torch
    rows, folds, manifest, fold_file = core.load_data(config)
    destination = ROOT / config["ensemble"]["reference_features"]
    metadata_path = destination.with_suffix(".metadata.json")
    if destination.exists() or metadata_path.exists():
        raise FileExistsError(f"Reference features already exist: {destination}")
    settings = dict(config["encoder"], model_id=ENCODERS["wav2vec2"]["model_id"],
                    expected_do_normalize=True, device="cpu")
    ref_config = dict(config, encoder=settings)
    # Explicit binary choice avoids silently selecting a converted PR revision.
    from transformers import AutoFeatureExtractor, AutoModel
    meta = ENCODERS["wav2vec2"]
    processor = AutoFeatureExtractor.from_pretrained(meta["model_id"], revision=meta["revision"], trust_remote_code=False)
    model = AutoModel.from_pretrained(meta["model_id"], revision=meta["revision"], use_safetensors=False, trust_remote_code=False)
    model.eval(); model.requires_grad_(False)
    if processor.sampling_rate != 16000 or not processor.do_normalize or model.config.num_hidden_layers != 12:
        raise ValueError("Unexpected Wav2Vec2 configuration.")
    torch.set_num_threads(config["runtime"]["cpu_threads"])
    arrays = {}
    for split in ("train", "validation"):
        selected = [row for row in rows if row["split"] == split]
        values = []
        for index, row in enumerate(selected, 1):
            value, _ = core.extract_one(row, ref_config, processor, model)
            values.append(value)
            if index % 100 == 0 or index == len(selected):
                print(f"{split}: {index}/{len(selected)}", flush=True)
        arrays[f"X_{split}"] = np.stack(values)
        for key, column in (("y", "emotion_code"), ("speakers", "speaker_id"), ("paths", "path")):
            arrays[f"{key}_{split}"] = np.array([row[column] for row in selected])
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as handle:
        np.savez_compressed(handle, **arrays)
    core.write_json(metadata_path, {"encoder": meta, "pooling": "all_transformer_mean",
        "manifest_sha256": core.file_hash(manifest), "versions": core.versions()})
    print(f"Saved: {destination}")
