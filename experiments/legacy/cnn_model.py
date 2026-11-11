from pathlib import Path

import numpy as np
import torch
from torch import nn


class EmotionCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv1d(40, 32, kernel_size=5, padding=2)
        self.classifier = nn.Linear(32, 8)

    def forward(
        self, features: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:
        frames = torch.arange(features.size(-1), device=features.device)
        valid = frames[None, :] < lengths[:, None]

        hidden = torch.relu(self.conv(features))
        hidden = hidden.masked_fill(~valid[:, None, :], 0.0)
        pooled = hidden.sum(dim=-1) / lengths[:, None]

        return self.classifier(pooled)


def main() -> None:
    torch.manual_seed(42)
    data_path = Path(__file__).resolve().parents[2] / "data" / "mfcc_sequences_dev.npz"

    with np.load(data_path, allow_pickle=False) as data:
        features = torch.from_numpy(data["X_train"][:8])
        lengths = torch.from_numpy(data["lengths_train"][:8]).long()
        targets = torch.from_numpy(data["y_train"][:8].astype(np.int64) - 1)

    model = EmotionCNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    optimizer.zero_grad()
    scores = model(features, lengths)
    loss = nn.CrossEntropyLoss()(scores, targets)

    if scores.shape != (8, 8) or not torch.isfinite(loss).item():
        raise RuntimeError("Unexpected model output or invalid loss.")

    loss.backward()
    optimizer.step()

    print(f"Input shape: {tuple(features.shape)}")
    print(f"Output shape: {tuple(scores.shape)}")
    print(f"Training loss: {loss.item():.4f}")
    print("One training step completed.")


if __name__ == "__main__":
    main()
