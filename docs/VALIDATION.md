# Refactor validation

## Executed

- Python syntax compilation for all retained and new Python modules.
- `run_experiment.py --help` and `predict.py --help`.
- Ten `unittest` checks: all passed.
- Recomputed all fifteen individual/ensemble fold Macro F1 values from saved
  out-of-fold predictions and matched the original reports to twelve decimal places.
- Verified 39 preserved report fingerprints. Text fingerprints normalize CRLF to
  LF to accommodate Git's Windows checkout conversion; report contents are unchanged.
- Synthetic fitting and joblib roundtrip using both classifiers, plus rejection
  of attempts to replace an existing model output directory.
- Feature/path permutation, nonfinite values, label-column ordering, scaler
  training-only fitting, and development/test speaker overlap checks.
- `git diff --check`.

## Environment used for these checks

Python 3.12.14 on Linux.

- numpy: 2.3.5
- scikit-learn: 1.8.0
- joblib: 1.5.3
- PyYAML: 6.0.3

These package versions are the refactor test environment, not the original
model-training environment. Historical results were not retrained here.

## Not executed here

- Installation of the historical Windows requirements lock.
- Wav2Vec2/WavLM downloads or audio inference: torch, transformers, librosa,
  the original feature caches and trained bundles are absent in this environment.
- Full RAVDESS extraction, retraining, or numerical equivalence of a newly trained
  model on the original audio. The current fit/verify workflow still needs that check.
- The legacy CNN and other audio-heavy experiments after module relocation.
- External CREMA-D inference.

The original selected model's mathematical design has been preserved. Structural
and synthetic tests alone do not establish complete audio-pipeline equivalence.
Use the existing verified model and a known recording for the first Windows check.

## Subsequent Windows verification

The user subsequently ran the refactored pipeline in the original Windows
environment. All ten tests passed, both classifiers fitted on 1,200
development recordings, and serialization checks passed.

Audio verification on twenty development recordings passed:
- Maximum Wav2Vec2 feature difference: 5.364418029785156e-07.
- Maximum WavLM feature difference: 0.0.
- Maximum ensemble probability difference: 1.1920928955078125e-07.
- Predicted labels were identical.

The report is saved at:
artifacts/models/equal_ensemble_refactor_v1/audio_verification.json

These checks establish consistency on the checked recordings.
Clean-environment installation and complete fresh audio extraction
remain unverified. No new accuracy evaluation was performed.
