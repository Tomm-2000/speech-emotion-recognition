# Project status

## Current phase

Repository refactor prepared from commit
`93cf2b04870dece3f7eae921e625b32cf5855601`.
Local Windows audio validation of the refactor is pending.

## Selected model and evidence

- Frozen Wav2Vec2 and WavLM, mean of 12 transformer layers and time.
- Separate StandardScaler + Logistic Regression (C=1) pipelines.
- Equal probability fusion with a fixed eight-class order.
- RAVDESS: five speaker-disjoint development folds, 1,200 recordings,
  20 speakers; mean fold Macro F1 = 0.757.
- CREMA-D diagnostic sample: 7/30 correct; six-reference-class Macro F1 = 0.235.
- The old test set has been inspected and is not an untouched final benchmark.
- No new model-quality result was generated during refactoring.

## Decisions

- Keep existing models, feature caches and original reports unchanged.
- Preserve historical experiments in a separate package.
- Use one runner for the current ensemble workflow.
- Do not introduce emotion2vec or change datasets in a structural refactor.
- Train and verify a new bundle in a separate output directory.

## Verified in the refactor environment

- Lightweight unit and integration checks on synthetic data.
- Published ensemble fold scores recomputed from preserved prediction CSV.
- Python syntax and CLI help checks.
- See docs/VALIDATION.md for the exact scope and environment.

## Next action

Apply the patch on a new branch, run the tests in the existing Windows venv,
then verify prediction with the existing audio-verified bundle. After that,
validate the new fit/verify stages on the original local development features.
Clean-environment installation and full audio reproduction remain pending.
