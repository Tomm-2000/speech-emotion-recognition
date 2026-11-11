# Project status

## Current stage

Repository refactoring is merged into main.
The current ensemble fitting and audio verification stages passed locally
on Windows with Python 3.12.10.

## Selected model

- Frozen Wav2Vec2 and WavLM encoders.
- Mean across twelve transformer layers and time.
- Separate StandardScaler and Logistic Regression classifiers, C=1.
- Equal probability fusion.
- Development mean fold Macro F1: 0.757.
- External CREMA-D sample: 7/30 correct, Macro F1: 0.235.
- These accuracy results predate the refactor.

## Verification completed

- All ten automated tests passed on Windows.
- Existing inference reproduced neutral with ensemble score 0.991.
- Both classifiers fitted on 1,200 development recordings.
- Model save/reload consistency passed.
- Audio verification passed on twenty development recordings.
- Maximum Wav2Vec2 feature difference: 5.364418029785156e-07.
- Maximum WavLM feature difference: 0.0.
- Maximum ensemble probability difference: 1.1920928955078125e-07.
- Predicted labels were identical.
- No new test accuracy evaluation was performed.

## Reproducibility note

The original feature provenance hashes CRLF bytes in speaker_folds.json.
Its original formatting was restored and matched the recorded hash.
.gitattributes disables line-ending conversion for this file.

## Remaining work

- Verify installation and full execution from a clean environment.
- Audit a specialist pretrained emotion model before further experiments.
- Establish independent evaluation data for future development.
