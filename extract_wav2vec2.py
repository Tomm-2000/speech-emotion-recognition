import argparse
import csv
from pathlib import Path
from time import perf_counter

import librosa
import numpy as np
import torch
from transformers import AutoFeatureExtractor, AutoModel

ROOT = Path(__file__).resolve().parent
MODEL_ID = "facebook/wav2vec2-base"
SAMPLE_RATE = 16_000


def load_encoder():
    feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_ID)
    encoder = AutoModel.from_pretrained(MODEL_ID)
    encoder.eval()
    return feature_extractor, encoder


@torch.inference_mode()
def extract_embedding(
    path,
    feature_extractor,
    encoder,
    pooling="last",
):
    if pooling not in {"last", "all_transformer_mean"}:
        raise ValueError(f"Unknown pooling mode: {pooling}")

    audio, sample_rate = librosa.load(
        path,
        sr=SAMPLE_RATE,
        mono=True,
    )
    if audio.size == 0 or not np.isfinite(audio).all():
        raise ValueError(f"Invalid audio: {path}")

    inputs = feature_extractor(
        audio,
        sampling_rate=sample_rate,
        return_tensors="pt",
    )

    use_all_layers = pooling == "all_transformer_mean"
    outputs = encoder(
        **inputs,
        output_hidden_states=use_all_layers,
        return_dict=True,
    )

    if use_all_layers:
        # Exclude hidden_states[0]: it precedes the transformer blocks.
        layers = outputs.hidden_states[1:]

        if len(layers) != encoder.config.num_hidden_layers:
            raise ValueError("Unexpected number of transformer outputs.")

        # Average each layer over time, then average across layers.
        # Equivalent to averaging layers first, then time.
        layer_means = torch.stack(
            [layer[0].mean(dim=0) for layer in layers],
            dim=0,
        )
        embedding_tensor = layer_means.mean(dim=0)
    else:
        embedding_tensor = outputs.last_hidden_state[0].mean(dim=0)

    embedding = embedding_tensor.cpu().numpy()
    frame_count = outputs.last_hidden_state.shape[1]

    if (
        embedding.shape != (encoder.config.hidden_size,)
        or not np.isfinite(embedding).all()
    ):
        raise ValueError(f"Invalid embedding for {path}")

    return embedding, frame_count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pooling",
        choices=["last", "all_transformer_mean"],
        default="last",
    )
    parser.add_argument("--limit", type=int, default=1)
    args = parser.parse_args()

    with (ROOT / "data" / "manifest.csv").open(
        encoding="utf-8", newline=""
    ) as file:
        recordings = [
            row for row in csv.DictReader(file)
            if row["split"] == "train"
        ]

    if not 1 <= args.limit <= len(recordings):
        raise SystemExit(
            f"--limit must be between 1 and {len(recordings)}."
        )

    processor, encoder = load_encoder()
    print(f"Pooling: {args.pooling}")
    print(f"Transformer layers: {encoder.config.num_hidden_layers}")

    start = perf_counter()
    embeddings = []

    for row in recordings[:args.limit]:
        embedding, frame_count = extract_embedding(
            ROOT / row["path"],
            processor,
            encoder,
            pooling=args.pooling,
        )
        embeddings.append(embedding)

    elapsed = perf_counter() - start
    features = np.stack(embeddings)

    print(f"Processed: {len(features)} recordings")
    print(f"Feature shape: {features.shape}")
    print("All embedding values are finite.")
    print(f"Extraction time: {elapsed:.1f} seconds")
    print(
        "Estimated extraction time for 1200: "
        f"{elapsed / len(features) * 1200 / 60:.1f} minutes"
    )
    print("Estimate excludes model loading and file saving.")
    print("No feature files were written.")


if __name__ == "__main__":
    main()