from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent


def main():
    feature_file = ROOT / "data" / "mfcc_prosody_dev.npz"

    with np.load(feature_file, allow_pickle=False) as data:
        features = np.concatenate([data["X_train"], data["X_validation"]])
        targets = np.concatenate([data["y_train"], data["y_validation"]])

    if features.shape != (1200, 85) or targets.shape != (1200,):
        raise SystemExit("Expected 1,200 recordings with 85 features each.")
    if not np.isfinite(features).all():
        raise SystemExit("Features contain invalid values.")

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000),
    )
    model.fit(features, targets)

    output = ROOT / "artifacts" / "prosody_logistic_regression.joblib"
    output.parent.mkdir(exist_ok=True)
    joblib.dump(model, output)

    print(f"Training recordings: {features.shape[0]}")
    print(f"Features per recording: {features.shape[1]}")
    print(f"Saved: {output.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()