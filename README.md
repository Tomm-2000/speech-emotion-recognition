# Speech Emotion Recognition

An experimental speech emotion recognition project using frozen
Wav2Vec2 and WavLM encoders, lightweight classifiers, and
speaker-disjoint development evaluation.

The current selected system averages the class probabilities of
two independently trained Logistic Regression classifiers.

**Status:** research prototype. Performance on a small external
CREMA-D sample was substantially lower than on RAVDESS development
folds. The project is not validated for production use.

## Supported emotions

- Neutral
- Calm
- Happy
- Sad
- Angry
- Fearful
- Disgust
- Surprised

Predictions describe the model's interpretation of vocal expression.
They are not reliable assessments of a person's internal emotional state.

## Current system

Each recording follows two parallel processing paths:

1. Load mono audio at 16 kHz.
2. Apply the corresponding encoder's audio processor.
3. Extract representations from all 12 transformer blocks.
4. Average across transformer layers and time.
5. Apply an independently fitted StandardScaler and Logistic Regression
   classifier to each encoder's 768-dimensional representation.
6. Align the emotion labels and average the two probability vectors
   with equal weights.
7. Return the emotion with the highest averaged score.

Both audio encoders remain frozen. Each classifier uses `C=1.0`.

The processors have different normalization settings:
Wav2Vec2 uses waveform normalization; WavLM uses its original
processor configuration without waveform normalization.

### Verified encoder revisions

| Encoder | Hugging Face model | Revision |
|---|---|---|
| Wav2Vec2 | `facebook/wav2vec2-base` | `0b5b8e868dd84f03fd87d01f9c4ff0f080fecfe8` |
| WavLM | `microsoft/wavlm-base-plus` | `4c66d4806a428f2e922ccfa1a962776e232d487b` |

These revisions are pinned in the locally verified inference bundle.
Feature consistency was checked against stored development features.

## Data and evaluation

### RAVDESS

The project uses the speech-audio archive:

`Audio_Speech_Actors_01-24.zip`

Source: https://zenodo.org/records/1188976

The dataset contains 1,440 recordings from 24 speakers.

| Original split | Recordings | Speakers |
|---|---:|---:|
| Training | 960 | 16 |
| Validation | 240 | 4 |
| Test | 240 | 4 |

The original training and validation splits form a development pool
of 1,200 recordings from 20 speakers.

Development comparisons use five fixed speaker-disjoint folds from
`speaker_folds.json`. Each fold trains on 16 speakers and evaluates
on four different speakers. StandardScaler is fitted only on the
training portion of each fold.

The primary development metric is mean fold Macro F1.

## Development results

| System | Mean fold Macro F1 |
|---|---:|
| MFCC + Logistic Regression | 0.432 |
| MFCC and prosody + Logistic Regression | 0.455 |
| Wav2Vec2 final-layer mean + Logistic Regression | 0.530 |
| Wav2Vec2 all-layer mean + Logistic Regression | 0.717 |
| WavLM all-layer mean + Logistic Regression | 0.727 |
| Equal-probability ensemble | **0.757** |

### Selected ensemble by fold

| Fold | Wav2Vec2 | WavLM | Equal ensemble |
|---|---:|---:|---:|
| 1 | 0.680 | 0.780 | 0.776 |
| 2 | 0.810 | 0.789 | 0.815 |
| 3 | 0.753 | 0.748 | 0.792 |
| 4 | 0.604 | 0.629 | 0.658 |
| 5 | 0.740 | 0.690 | 0.743 |

The ensemble's pooled development accuracy was 915/1,200, or 76.25%.

These are development results, not an independent final test.
Repeated experimentation on the same folds may make the selected
score optimistic. Fold variation is not a confidence interval.

Additional experiments included neural classifiers, learned layer
mixing, processing before temporal pooling, maximum-probability
fusion, and nested regularization selection.

Maximum-probability fusion achieved mean fold Macro F1 of 0.749.
Independent inner-fold selection of each classifier's regularization
achieved ensemble Macro F1 of 0.745. The original equal ensemble
was retained.

## External evaluation: CREMA-D

Source: https://github.com/CheyneyComputerScience/CREMA-D

A fixed sample was selected before downloading audio and making
predictions:

- 30 recordings from 30 distinct speakers.
- Five recordings for each of six reference emotions.
- Reference labels taken from the intended emotion in filenames.
- No classifier fitting or tuning on this sample.
- All eight model prediction options retained.



Macro F1 is computed over the six reference classes:
angry, disgust, fearful, happy, neutral, and sad.

Predictions of calm or surprised count as incorrect. The ensemble
made 10 such predictions.

This small sample reveals a substantial generalization limitation.
It does not provide a precise performance estimate for all CREMA-D
recordings or real-world speech.

The sample has now been inspected and should be treated as a
diagnostic set for future work.

## Earlier RAVDESS test result

An earlier Wav2Vec2 final-layer classifier achieved:

- Accuracy: 0.708, or 170/240.
- Macro F1: 0.702.

These results do not belong to the current ensemble.

The original test set was inspected during earlier project stages.
It is not an untouched final benchmark for subsequent development.

## Project layout

- `run_experiment.py`: one entry point for the current ensemble workflow.
- `predict.py`: single-file inference, compatible with the existing verified bundle.
- `ser/experiments.py`: data validation, extraction, folds, evaluation, and equal fusion.
- `ser/regularization.py`: nested speaker-based regularization experiment.
- `ser/bundle.py`: pinned reference extraction, final fitting, and audio verification.
- `ser/inference.py`: shared inference implementation.
- `ser/mfcc.py`: shared historical MFCC implementation.
- `experiments/legacy/`: earlier MFCC, CNN, pooling, and layer-mixing experiments.
- `configs/reproduce.yaml`: pinned encoders and fresh output paths for new runs.
- `config.yaml`: original configuration, retained for existing local feature caches.
- `artifacts/`: original experiment evidence, preserved at its existing paths.
- `tests/`: lightweight checks that require no audio or encoder downloads.
- `docs/`: setup, migration, refactor scope, and verification details.

The original results above were produced before the refactor. They are not
claimed as results of a new end-to-end training run.

## Setup and reproduction

See [SETUP.md](docs/SETUP.md) for the complete Windows workflow.

The historical environment used Python 3.12.10 and CPU execution.
`requirements-lock.txt` records that environment with PyYAML reconciled to
6.0.2, the version in the saved experiment provenance. The original snapshot
is preserved in `docs/requirements-original.txt`.

Installation from this historical lock has not been independently verified.
Do not upgrade the existing working environment during migration.

For an existing checkout with its original caches, the new final-fit steps are:

```powershell
.\.venv\Scripts\python.exe .\run_experiment.py --config .\config.yaml --stage fit
.\.venv\Scripts\python.exe .\run_experiment.py --config .\config.yaml --stage verify
```

They write a new bundle directory, leaving the historical fitted model intact:

`artifacts/models/equal_ensemble_refactor_v1/`

The fitting stage does not mark a model as audio-verified. Verification must
succeed before that new bundle can be used by the prediction command.

## Predict

Existing verified bundle (same default path as before):

```powershell
.\.venv\Scripts\python.exe .\predict.py "path\to\recording.wav"
```

Newly fitted and verified bundle:

```powershell
.\.venv\Scripts\python.exe .\predict.py "path\to\recording.wav" --bundle .\artifacts\models\equal_ensemble_refactor_v1\ensemble_verified.joblib
```

Inference uses locally cached pinned encoders by default. Pass
`--allow-download` to permit downloading the specified revisions.
Only load trusted joblib bundles. The repository does not ship trained weights.

## Checks

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Tests cover feature/label alignment, exclusion of evaluation data from scaler
fitting, development/test speaker separation, class-probability ordering,
serialization, and reproduction of published fold scores from saved predictions.

See [VALIDATION.md](docs/VALIDATION.md) for what was actually run and what remains
unverified. Synthetic checks do not establish model quality.

## Research reference and attribution

Pepino, L., Riera, P., and Ferrer, L. (2021).
*Emotion Recognition from Speech Using wav2vec 2.0 Embeddings.*

- Paper: https://www.isca-archive.org/interspeech_2021/pepino21_interspeech.pdf
- Official implementation: https://github.com/habla-liaa/ser-with-w2v2
- Wav2Vec2: https://huggingface.co/facebook/wav2vec2-base
- WavLM: https://huggingface.co/microsoft/wavlm-base-plus

This project is inspired by the research, not an exact replication. Class
mapping, normalization, splits, architecture, and metrics differ. Scores
must not be compared directly with the paper.

Dataset and pretrained-model terms remain applicable to their respective
materials. Raw audio, personal recordings, pretrained weights, and extracted
binary features are excluded from Git. No new license for third-party
materials is implied by this refactor.

## Remaining limitations and next steps

- Run the new fitting/verification stages on the original local data.
- Verify installation and full reproduction from a clean environment.
- Historical scratch-only analyses do not all have permanent reproduction scripts;
  their saved reports are retained as evidence, not claimed as fully reproducible.
- Audit a specialist emotion-recognition model before selecting another experiment.
- Establish independent evaluation data before further model development.
- Improve cross-dataset generalization with controlled, documented comparisons.
