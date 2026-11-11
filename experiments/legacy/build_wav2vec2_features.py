import argparse
import csv
import json
from time import perf_counter

import librosa
import numpy as np
import torch

from .extract_wav2vec2 import (
    MODEL_ID,
    ROOT,
    SAMPLE_RATE,
    extract_embedding,
    load_encoder,
)

DATA = ROOT / "data"
SPLIT_COUNTS = {"train": 960, "validation": 240}

OUTPUT_NAMES = {
    "last": "wav2vec2_dev.npz",
    "all_transformer_mean": "wav2vec2_all_layers_dev.npz",
    "layers": "wav2vec2_layerwise_dev.npz",
}


def read_manifest():
    with (DATA / "manifest.csv").open(
        encoding="utf-8", newline=""
    ) as file:
        rows = list(csv.DictReader(file))

    split_rows = {
        split: [row for row in rows if row["split"] == split]
        for split in SPLIT_COUNTS
    }

    for split, expected in SPLIT_COUNTS.items():
        if len(split_rows[split]) != expected:
            raise SystemExit(f"Unexpected recording count: {split}")

    dev_rows = [
        row
        for recordings in split_rows.values()
        for row in recordings
    ]

    if len({row["path"] for row in dev_rows}) != len(dev_rows):
        raise SystemExit("Duplicate development recording paths.")

    train_speakers = {
        row["speaker_id"] for row in split_rows["train"]
    }
    validation_speakers = {
        row["speaker_id"] for row in split_rows["validation"]
    }
    test_speakers = {
        row["speaker_id"] for row in rows if row["split"] == "test"
    }

    if train_speakers & validation_speakers:
        raise SystemExit("Training/validation speaker overlap.")

    if (train_speakers | validation_speakers) & test_speakers:
        raise SystemExit("Development/test speaker overlap.")

    return split_rows, dev_rows


def load_reference(split_rows):
    reference = {}

    with np.load(
        DATA / "wav2vec2_all_layers_dev.npz",
        allow_pickle=False,
    ) as data:
        for split, recordings in split_rows.items():
            for key, column in (
                ("paths", "path"),
                ("speakers", "speaker_id"),
                ("y", "emotion_code"),
            ):
                expected = [row[column] for row in recordings]
                if data[f"{key}_{split}"].tolist() != expected:
                    raise SystemExit(
                        f"Reference alignment mismatch: {split}/{key}"
                    )

            values = data[f"X_{split}"].copy()
            if (
                values.shape != (len(recordings), 768)
                or not np.isfinite(values).all()
            ):
                raise SystemExit(f"Invalid reference features: {split}")

            reference[split] = values

    return reference


@torch.inference_mode()
def extract_detail(path, processor, encoder, mode):
    audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)

    if audio.size == 0 or not np.isfinite(audio).all():
        raise ValueError(f"Invalid audio: {path}")

    inputs = processor(
        audio,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
    )
    outputs = encoder(
        **inputs,
        output_hidden_states=True,
        return_dict=True,
    )
    layers = outputs.hidden_states[1:]

    if len(layers) != 12:
        raise ValueError("Expected 12 transformer outputs.")

    if mode == "layers":
        result = torch.stack(
            [layer[0].mean(dim=0) for layer in layers],
            dim=0,
        )
    elif mode == "frames":
        result = torch.zeros_like(layers[0][0])
        for layer in layers:
            result.add_(layer[0])
        result.div_(12)
    else:
        raise ValueError(f"Unsupported detail mode: {mode}")

    values = result.cpu().numpy().astype(np.float32, copy=True)

    if (
        values.ndim != 2
        or values.shape[0] == 0
        or values.shape[1] != 768
        or not np.isfinite(values).all()
    ):
        raise ValueError(f"Invalid features: {path}")

    if mode == "layers" and values.shape != (12, 768):
        raise ValueError("Unexpected layer feature shape.")

    return values


def build_frames(split_rows, dev_rows, processor, encoder, reference):
    output = DATA / "wav2vec2_frames_dev"
    temporary = DATA / "wav2vec2_frames_dev.incomplete"

    if output.exists() or temporary.exists():
        raise SystemExit(
            "Frame output or an incomplete build already exists.\n"
            "No existing files were changed."
        )

    temporary.mkdir()
    offsets = []
    lengths = []
    total_frames = 0
    max_difference = 0.0
    started = perf_counter()

    # Each recording occupies consecutive rows in one binary matrix.
    # Index offsets are measured in frames, not bytes.
    with (temporary / "frames.f32").open("xb") as binary:
        for split, recordings in split_rows.items():
            for index, row in enumerate(recordings):
                sequence = extract_detail(
                    ROOT / row["path"], processor, encoder, "frames"
                )
                expected = reference[split][index]
                reconstructed = sequence.mean(axis=0)

                np.testing.assert_allclose(
                    reconstructed,
                    expected,
                    rtol=1e-4,
                    atol=1e-5,
                    err_msg=f"Mean consistency failed: {row['path']}",
                )

                difference = float(
                    np.max(np.abs(reconstructed - expected))
                )
                max_difference = max(max_difference, difference)

                offsets.append(total_frames)
                lengths.append(len(sequence))
                total_frames += len(sequence)

                # Explicit little-endian float32 storage.
                sequence.astype("<f4", copy=False).tofile(binary)

                if (index + 1) % 100 == 0 or index + 1 == len(recordings):
                    print(
                        f"{split}: {index + 1}/{len(recordings)}",
                        flush=True,
                    )

    expected_bytes = total_frames * 768 * 4
    actual_bytes = (temporary / "frames.f32").stat().st_size
    if actual_bytes != expected_bytes:
        raise RuntimeError("Unexpected binary file size.")

    np.savez_compressed(
        temporary / "index.npz",
        offsets=np.array(offsets, dtype=np.int64),
        lengths=np.array(lengths, dtype=np.int64),
        paths=np.array([row["path"] for row in dev_rows]),
        speakers=np.array([row["speaker_id"] for row in dev_rows]),
        labels=np.array([row["emotion_code"] for row in dev_rows]),
        splits=np.array([row["split"] for row in dev_rows]),
    )

    metadata = {
        "encoder_id": MODEL_ID,
        "sample_rate": SAMPLE_RATE,
        "representation": "mean_of_12_transformer_layers_per_frame",
        "layer_indices": list(range(1, 13)),
        "dtype": "<f4",
        "width": 768,
        "recordings": len(dev_rows),
        "total_frames": total_frames,
        "offset_unit": "frames",
        "padding": False,
        "truncation": False,
        "maximum_mean_consistency_difference": max_difference,
        "extraction_seconds": perf_counter() - started,
    }
    (temporary / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    # Check that indexed data can be read back without loading it all.
    mapped = np.memmap(
        temporary / "frames.f32",
        dtype="<f4",
        mode="r",
        shape=(total_frames, 768),
    )
    expected_means = np.concatenate([
        reference[split] for split in SPLIT_COUNTS
    ])

    for index, (offset, length) in enumerate(zip(offsets, lengths)):
        recovered = mapped[offset:offset + length]
        if recovered.shape != (length, 768):
            raise RuntimeError("Invalid indexed recording shape.")

        np.testing.assert_allclose(
            recovered.mean(axis=0),
            expected_means[index],
            rtol=1e-4,
            atol=1e-5,
        )

    # Release the memory map before renaming the directory on Windows.
    del recovered
    del mapped

    temporary.rename(output)

    print("All recording means match the existing reference features.")
    print("Indexed read-back check passed.")
    print(f"Saved: {output.relative_to(ROOT).as_posix()}")
    print(f"Recordings: {len(dev_rows)}")
    print(f"Total frames: {total_frames}")
    print(
        f"Frames per recording: min={min(lengths)}, "
        f"mean={np.mean(lengths):.1f}, max={max(lengths)}"
    )
    print(f"Raw feature size: {actual_bytes / (1024 ** 2):.1f} MiB")
    print(f"Maximum mean difference: {max_difference:.8f}")
    print(f"Total time: {(perf_counter() - started) / 60:.1f} minutes")


def build_vectors(mode, split_rows, processor, encoder, reference):
    output = DATA / OUTPUT_NAMES[mode]

    if output.exists():
        raise SystemExit(f"Output already exists: {output}")

    arrays = {
        "encoder_id": np.array(MODEL_ID),
        "sample_rate": np.array(SAMPLE_RATE),
        "pooling": np.array(mode),
        "transformer_layer_count": np.array(12),
    }
    if mode == "layers":
        arrays["layer_indices"] = np.arange(1, 13)

    started = perf_counter()

    for split, recordings in split_rows.items():
        embeddings = []

        for index, row in enumerate(recordings, start=1):
            path = ROOT / row["path"]

            if mode == "layers":
                embedding = extract_detail(
                    path, processor, encoder, "layers"
                )
            else:
                embedding, _ = extract_embedding(
                    path, processor, encoder, pooling=mode
                )

            embeddings.append(embedding)

            if index % 100 == 0 or index == len(recordings):
                print(f"{split}: {index}/{len(recordings)}", flush=True)

        features = np.stack(embeddings)
        expected_shape = (
            (len(recordings), 12, 768)
            if mode == "layers"
            else (len(recordings), 768)
        )

        if features.shape != expected_shape:
            raise SystemExit(f"Unexpected feature shape: {features.shape}")
        if not np.isfinite(features).all():
            raise SystemExit("Invalid feature values.")

        if mode == "layers":
            np.testing.assert_allclose(
                features.mean(axis=1),
                reference[split],
                rtol=1e-5,
                atol=1e-5,
            )
            print(f"{split}: consistency passed")

        arrays[f"X_{split}"] = features
        arrays[f"y_{split}"] = np.array([
            row["emotion_code"] for row in recordings
        ])
        arrays[f"speakers_{split}"] = np.array([
            row["speaker_id"] for row in recordings
        ])
        arrays[f"paths_{split}"] = np.array([
            row["path"] for row in recordings
        ])

    with output.open("xb") as file:
        np.savez_compressed(file, **arrays)

    print(f"Saved: {output.relative_to(ROOT).as_posix()}")
    for split in SPLIT_COUNTS:
        print(f"{split} feature shape: {arrays[f'X_{split}'].shape}")
    print(
        f"Extraction and save time: "
        f"{(perf_counter() - started) / 60:.1f} minutes"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pooling",
        choices=[*OUTPUT_NAMES, "frames"],
        default="last",
    )
    args = parser.parse_args()

    if args.pooling == "frames":
        if (
            (DATA / "wav2vec2_frames_dev").exists()
            or (DATA / "wav2vec2_frames_dev.incomplete").exists()
        ):
            raise SystemExit(
                "Frame output or incomplete build already exists."
            )
    elif (DATA / OUTPUT_NAMES[args.pooling]).exists():
        raise SystemExit("Output already exists; no files were changed.")

    split_rows, dev_rows = read_manifest()
    reference = (
        load_reference(split_rows)
        if args.pooling in {"layers", "frames"}
        else None
    )

    processor, encoder = load_encoder()
    if encoder.config.num_hidden_layers != 12:
        raise SystemExit("Expected 12 transformer layers.")

    print(f"Pooling: {args.pooling}", flush=True)

    if args.pooling == "frames":
        build_frames(
            split_rows, dev_rows, processor, encoder, reference
        )
    else:
        build_vectors(
            args.pooling, split_rows, processor, encoder, reference
        )


if __name__ == "__main__":
    main()
