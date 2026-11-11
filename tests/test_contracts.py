"""Small tests with synthetic features; no audio downloads or model training on RAVDESS."""
import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import yaml

from ser import experiments as core
from ser import bundle
from ser.inference import ordered_probabilities
from ser.regularization import fit_model, predict_probabilities
from ser.paths import ROOT


class Contracts(unittest.TestCase):
    def setUp(self):
        self.config = yaml.safe_load((ROOT / "config.yaml").read_text())

    def test_original_reports_are_unchanged(self):
        import hashlib
        hashes = json.loads((ROOT / "docs/original_artifact_hashes.json").read_text())
        for name, expected in hashes.items():
            self.assertEqual(hashlib.sha256((ROOT / name).read_bytes().replace(b"\r\n", b"\n")).hexdigest(), expected, name)

    def test_source_config_excludes_only_nonencoder_stages(self):
        a = core.core_config(self.config)
        b = core.core_config(dict(self.config, fitting={"output_directory": "new"}))
        self.assertEqual(a, b)
        changed = dict(self.config, encoder=dict(self.config["encoder"], pooling="last"))
        self.assertNotEqual(a, core.core_config(changed))

    def test_training_scaler_excludes_evaluation(self):
        rng = np.random.default_rng(42)
        X = rng.normal(size=(96, 12)); X[80:] += 10000
        y = np.tile(core.LABELS, 12)
        train = np.arange(96) < 80
        model = fit_model(X, y, train, 1.0, self.config)
        np.testing.assert_allclose(model[0].mean_, X[train].mean(0))
        self.assertLess(np.abs(model[0].mean_).max(), 1)
        p = predict_probabilities(model, X[~train])
        np.testing.assert_allclose(p.sum(1), 1)

    def test_probability_columns_reordered(self):
        class Fake:
            classes_ = core.LABELS[::-1]
            def predict_proba(self, X):
                return np.array([[.8, .1, .05, .02, .01, .01, .005, .005]])
        p = ordered_probabilities(Fake(), np.zeros((1, 768)), core.LABELS.tolist())
        self.assertEqual(core.LABELS[p.argmax()], "08")

    def test_wrong_labels_rejected(self):
        class Fake:
            classes_ = np.array(["bad"])
        with self.assertRaises(ValueError):
            ordered_probabilities(Fake(), np.zeros((1, 768)), core.LABELS.tolist())

    def test_feature_metadata_permutation_rejected(self):
        rows = [{"emotion_code": "01", "speaker_id": "01", "path": "a"},
                {"emotion_code": "02", "speaker_id": "02", "path": "b"}]
        with self.assertRaises(ValueError):
            core.check_features(np.zeros((2, 768)), np.array(["01", "02"]),
                                np.array(["01", "02"]), np.array(["b", "a"]), rows)

    def test_nonfinite_features_rejected(self):
        rows = [{"emotion_code": "01", "speaker_id": "01", "path": "a"}]
        with self.assertRaises(ValueError):
            core.check_features(np.full((1, 768), np.nan), np.array(["01"]),
                                np.array(["01"]), np.array(["a"]), rows)

    def test_manifest_test_speaker_leakage_rejected(self):
        folds = json.loads((ROOT / "speaker_folds.json").read_text())
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root / "data").mkdir()
            (root / "speaker_folds.json").write_text(json.dumps(folds))
            rows = []
            for group in folds.values():
                for speaker in group:
                    for i in range(60):
                        rows.append(dict(path=f"{speaker}/{i}.wav", speaker_id=speaker,
                            emotion_code=f"{i % 8 + 1:02d}", split="train"))
            rows.append(dict(path="test.wav", speaker_id=rows[0]["speaker_id"], emotion_code="01", split="test"))
            with (root / "data/manifest.csv").open("w", newline="") as f:
                w=csv.DictWriter(f, fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
            with patch.object(core, "ROOT", root), self.assertRaisesRegex(ValueError, "overlap"):
                core.load_data(self.config)

    def test_fit_roundtrip_and_no_overwrite(self):
        rng = np.random.default_rng(7)
        y = np.tile(core.LABELS, 20)
        speakers = np.repeat([f"{i:02d}" for i in range(1, 21)], 8)
        paths = np.array([f"synthetic/{i}.wav" for i in range(160)])
        X = rng.normal(size=(160, 16))
        fixture = ([], {"wav2vec2": X, "wavlm": X * .7}, y, speakers, paths, {"test": "synthetic"})
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "model"
            cfg=dict(self.config, fitting={"output_directory": str(out)})
            with patch.object(bundle, "source_features", return_value=fixture), patch.object(core, "versions", return_value={"test": "synthetic"}):
                bundle.fit(cfg)
                saved = joblib.load(out / "ensemble.joblib")
                self.assertNotIn("audio_verification", saved)
                self.assertEqual(saved["training_recordings"],160)
                self.assertTrue((out / "inference_reference.npz").exists())
                with self.assertRaises(FileExistsError): bundle.fit(cfg)

    def test_published_fold_scores_recompute_from_saved_predictions(self):
        base=ROOT / "artifacts/experiments/wav2vec2_wavlm_equal_ensemble_v1"
        report=json.loads((base / "evaluation.json").read_text())
        with (base / "oof_predictions.csv").open(newline="") as f: rows=list(csv.DictReader(f))
        for mode, result in report["results"].items():
            for fold, score in result["fold_scores"].items():
                selected=[r for r in rows if r["mode"]==mode and r["fold"]==fold]
                self.assertEqual(len(selected),240)
                actual=[r["actual"] for r in selected];pred=[r["predicted"] for r in selected]
                self.assertAlmostEqual(core.macro_f1(actual,pred),score,places=12)


if __name__ == "__main__":
    unittest.main()
