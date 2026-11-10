# Speech Emotion Recognition

- Current stage: Baseline training complete.
- Data: RAVDESS audio-only speech, 1,440 recordings from 24 speakers.
- Split: 16 train speakers, 4 validation speakers, 4 test speakers.
- Features: 40 MFCC means and 40 MFCC standard deviations per recording.
- Baseline: StandardScaler and Logistic Regression.
- Validation accuracy: 0.450 (108/240).
- Validation macro F1: 0.446.
- Lowest validation F1: sad (0.24).
- Test set: Reserved for final evaluation.
- Next step: Inspect the validation confusion matrix before choosing the next model.


## Final evaluation

- Selected model: StandardScaler + Logistic Regression.
- Selection metric: validation macro F1.
- Validation macro F1: 0.446 for Logistic Regression, 0.427 for the CNN.
- Held-out test set: 240 recordings from 4 unseen speakers.
- Test accuracy: 0.450 (108/240).
- Test macro F1: 0.435.
- Highest test class F1: surprised (0.70).
- Lowest test class F1: happy (0.28).
- Next step: Document setup, reproducibility, results, and limitations in README.md.


## V2 — MFCC and prosody

- Selected development configuration: StandardScaler + Logistic Regression.
- Inputs: 80 MFCC features and 5 pitch, voicing, and level features.
- Evaluation: 5 speaker-disjoint folds over 20 development speakers.
- Mean macro F1: 0.455 vs. 0.432 for the MFCC-only baseline.
- Improvement: +0.023 macro F1, with gains in 4 of 5 folds.
- V1 held-out test macro F1: 0.435. V2 has no independent test result yet.