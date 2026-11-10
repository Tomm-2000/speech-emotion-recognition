from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

root = Path(__file__).resolve().parent
labels = ["01", "02", "03", "04", "05", "06", "07", "08"]
emotion_names = [
    "neutral",
    "calm",
    "happy",
    "sad",
    "angry",
    "fearful",
    "disgust",
    "surprised",
]

with np.load(root / "data" / "mfcc_dev.npz", allow_pickle=False) as data:
    X_train = data["X_train"]
    y_train = data["y_train"]
    X_validation = data["X_validation"]
    y_validation = data["y_validation"]

model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=2000),
)
model.fit(X_train, y_train)
predictions = model.predict(X_validation)

accuracy = accuracy_score(y_validation, predictions)
macro_f1 = f1_score(y_validation, predictions, average="macro")

print(f"Validation accuracy: {accuracy:.3f}")
print(f"Validation macro F1: {macro_f1:.3f}")
print()
print(
    classification_report(
        y_validation,
        predictions,
        labels=labels,
        target_names=emotion_names,
        zero_division=0,
    )
)

matrix = confusion_matrix(y_validation, predictions, labels=labels)

print("Validation confusion matrix (rows: actual, columns: predicted)")
print(f"{'Actual':>10}" + "".join(f"{name:>11}" for name in emotion_names))

for name, row in zip(emotion_names, matrix):
    print(f"{name:>10}" + "".join(f"{count:>11}" for count in row))