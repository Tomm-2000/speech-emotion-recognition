from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .cnn_model import EmotionCNN

ROOT = Path(__file__).resolve().parents[2]
BATCH_SIZE = 32
MAX_EPOCHS = 100
PATIENCE = 5


def load_split(data, split):
    features = torch.from_numpy(data[f"X_{split}"]).float()
    lengths = torch.from_numpy(data[f"lengths_{split}"]).long()
    targets = torch.from_numpy(data[f"y_{split}"].astype(np.int64) - 1)
    return features, lengths, targets


def valid_frames(features, lengths):
    frames = torch.arange(features.size(-1))
    return frames[None, :] < lengths[:, None]


def normalize(features, lengths, mean, scale):
    valid = valid_frames(features, lengths)
    normalized = (features - mean[None, :, None]) / scale[None, :, None]
    return normalized.masked_fill(~valid[:, None, :], 0.0)


def main():
    torch.manual_seed(42)

    with np.load(ROOT / "data" / "mfcc_sequences_dev.npz", allow_pickle=False) as data:
        X_train, lengths_train, y_train = load_split(data, "train")
        X_val, lengths_val, y_val = load_split(data, "validation")

    train_frames = X_train.transpose(1, 2)[valid_frames(X_train, lengths_train)]
    mean = train_frames.mean(dim=0)
    scale = train_frames.std(dim=0, unbiased=False).clamp_min(1e-6)

    train_loader = DataLoader(
        TensorDataset(
            normalize(X_train, lengths_train, mean, scale),
            lengths_train,
            y_train,
        ),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    X_val = normalize(X_val, lengths_val, mean, scale)

    model = EmotionCNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    checkpoint = ROOT / "artifacts" / "cnn_best.pt"
    checkpoint.parent.mkdir(exist_ok=True)

    best_f1 = -1.0
    best_epoch = 0
    best_accuracy = 0.0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for features, lengths, targets in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(model(features, lengths), targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * targets.size(0)

        model.eval()
        with torch.no_grad():
            predictions = model(X_val, lengths_val).argmax(dim=1).numpy()

        accuracy = accuracy_score(y_val.numpy(), predictions)
        macro_f1 = f1_score(
            y_val.numpy(), predictions, labels=list(range(8)),
            average="macro", zero_division=0,
        )
        print(
            f"Epoch {epoch:02d} | train loss: {total_loss / len(y_train):.3f} "
            f"| val accuracy: {accuracy:.3f} | val macro F1: {macro_f1:.3f}",
            flush=True,
        )

        if macro_f1 > best_f1:
            best_f1 = macro_f1
            best_epoch = epoch
            best_accuracy = accuracy
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "feature_mean": mean,
                    "feature_scale": scale,
                    "max_frames": X_train.size(-1),
                    "epoch": best_epoch,
                    "validation_macro_f1": best_f1,
                },
                checkpoint,
            )

        if epoch - best_epoch >= PATIENCE:
            break

    print(f"\nBest epoch: {best_epoch}")
    print(f"Best validation accuracy: {best_accuracy:.3f}")
    print(f"Best validation macro F1: {best_f1:.3f}")
    print(f"Saved: {checkpoint.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
