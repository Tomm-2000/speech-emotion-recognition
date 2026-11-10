import csv
import json

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from cnn_model import EmotionCNN
from train_cnn import ROOT, load_split, normalize, valid_frames

DATA = ROOT / "data"
EPOCHS = 25
SEED = 42


class MeanStdCNN(EmotionCNN):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Linear(64, 8)

    def forward(self, features, lengths):
        valid = valid_frames(features, lengths)[:, None, :]
        hidden = torch.relu(self.conv(features))
        masked = hidden.masked_fill(~valid, 0.0)
        mean = masked.sum(dim=-1) / lengths[:, None]

        centered = (hidden - mean[:, :, None]).masked_fill(~valid, 0.0)
        variance = centered.square().sum(dim=-1) / lengths[:, None]
        std = variance.clamp_min(1e-8).sqrt()

        return self.classifier(torch.cat([mean, std], dim=1))


def train_and_score(model_type, training_data, evaluation_data):
    torch.manual_seed(SEED)
    model = model_type()
    loader = DataLoader(
        training_data,
        batch_size=32,
        shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for _ in range(EPOCHS):
        for features, lengths, targets in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(features, lengths), targets)
            loss.backward()
            optimizer.step()

    features, lengths, targets = evaluation_data
    model.eval()
    with torch.no_grad():
        predictions = model(features, lengths).argmax(dim=1)

    return f1_score(
        targets.numpy(),
        predictions.numpy(),
        labels=list(range(8)),
        average="macro",
        zero_division=0,
    )


def main():
    folds = json.loads((ROOT / "speaker_folds.json").read_text(encoding="utf-8"))
    with (DATA / "manifest.csv").open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    dev_rows = [
        row for split in ("train", "validation")
        for row in rows if row["split"] == split
    ]
    speakers = np.array([row["speaker_id"] for row in dev_rows])

    with np.load(DATA / "mfcc_sequences_dev.npz", allow_pickle=False) as data:
        train = load_split(data, "train")
        validation = load_split(data, "validation")
    features, lengths, targets = (
        torch.cat([train_array, val_array])
        for train_array, val_array in zip(train, validation)
    )

    expected_targets = [int(row["emotion_code"]) - 1 for row in dev_rows]
    if features.shape != (1200, 40, 165) or targets.tolist() != expected_targets:
        raise SystemExit("Saved sequences do not match the development manifest.")

    mean_scores = []
    mean_std_scores = []

    for name, group in folds.items():
        evaluation = torch.from_numpy(np.isin(speakers, group))
        training = ~evaluation
        if training.sum().item() != 960 or evaluation.sum().item() != 240:
            raise SystemExit(f"Unexpected split size for {name}.")

        print(f"Running {name}...", flush=True)
        train_features, train_lengths = features[training], lengths[training]
        real_frames = train_features.transpose(1, 2)[
            valid_frames(train_features, train_lengths)
        ]
        mean = real_frames.mean(dim=0)
        scale = real_frames.std(dim=0, unbiased=False).clamp_min(1e-6)

        training_data = TensorDataset(
            normalize(train_features, train_lengths, mean, scale),
            train_lengths,
            targets[training],
        )
        evaluation_data = (
            normalize(features[evaluation], lengths[evaluation], mean, scale),
            lengths[evaluation],
            targets[evaluation],
        )

        mean_score = train_and_score(EmotionCNN, training_data, evaluation_data)
        mean_std_score = train_and_score(MeanStdCNN, training_data, evaluation_data)
        mean_scores.append(mean_score)
        mean_std_scores.append(mean_std_score)
        print(
            f"{name}: mean={mean_score:.3f}, mean+std={mean_std_score:.3f}, "
            f"difference={mean_std_score - mean_score:+.3f}",
            flush=True,
        )

    print(f"Mean-only CNN average: {np.mean(mean_scores):.3f}")
    print(f"Mean+std CNN average: {np.mean(mean_std_scores):.3f}")


if __name__ == "__main__":
    main()