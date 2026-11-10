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

| System | Correct | Accuracy | Macro F1 |
|---|---:|---:|---:|
| Wav2Vec2 | 8/30 | 0.267 | 0.277 |
| WavLM | 6/30 | 0.200 | 0.210 |
| Equal ensemble | 7/30 | 0.233 | 0.235 |

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

## Environment

Development used Windows PowerShell, Python 3.12.10, and CPU execution.

`requirements-lock.txt` contains a previously captured environment
snapshot. It still requires reconciliation with the final environment
and verification in a clean installation.

## Repository organization

- `config.yaml`: experiment settings.
- `run_experiment.py`: experiment entry point.
- `run_experiment_previous.py`: implementation imported by the runner.
- `speaker_splits.json`: original speaker split definitions.
- `speaker_folds.json`: fixed development folds.
- `build_*.py` and `extract_*.py`: data and feature preparation.
- `evaluate_*.py`: evaluation and diagnostic utilities.
- `predict.py`: inference with the verified ensemble bundle.
- `artifacts/`: experiment reports and prediction tables.
- `PROJECT_STATUS.md`: project progress notes.

Some scripts represent earlier experiments and are not the current
inference pipeline.

## Local inference

The current inference command is:

```powershell
.\.venv\Scripts\python.exe .\predict.py "path\to\recording.wav"
```

It requires the locally generated bundle:

```text
artifacts/experiments/wav2vec2_wavlm_equal_ensemble_v1/fitted_model/ensemble_verified.joblib
```

It also requires the encoder files expected by the inference script.

The fitted bundle, audio recordings, environments, and binary feature
files are excluded from Git. Cloning this repository alone does not
currently provide a ready-to-run inference installation.

A complete clean-checkout workflow for producing the verified
ensemble bundle remains to be packaged and documented.

Only load trusted serialized model files.

## Verification completed locally

- Audio readability and dataset-count checks.
- Speaker-disjoint development comparisons.
- Feature alignment and finite-value checks.
- Classifier serialization and reload consistency.
- Audio-to-feature and ensemble-probability consistency on
  20 development recordings.
- Command-line inference.
- External evaluation on the fixed 30-recording sample.

Consistency checks verify implementation behavior; they do not
measure generalization.

## Research reference and attribution

Pepino, L., Riera, P., and Ferrer, L. (2021).
*Emotion Recognition from Speech Using wav2vec 2.0 Embeddings.*

Paper:
https://www.isca-archive.org/interspeech_2021/pepino21_interspeech.pdf

Official implementation:
https://github.com/habla-liaa/ser-with-w2v2

This project is inspired by the research, not an exact replication.
Class definitions, normalization, splits, architecture, and metrics
differ. Published scores are not directly comparable.

Pretrained models:

- https://huggingface.co/facebook/wav2vec2-base
- https://huggingface.co/microsoft/wavlm-base-plus

Consult the original dataset and model licenses before reuse or
redistribution. Raw audio and pretrained weights are not included.

## Next steps

- Package the final ensemble fitting and verification workflow.
- Verify installation and execution from a clean environment.
- Audit a specialist pretrained emotion-recognition model.
- Establish an independent evaluation before further development.
- Improve cross-dataset generalization through controlled experiments.