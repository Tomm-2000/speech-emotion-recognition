import csv
import json
from collections import Counter
from pathlib import Path

root = Path(__file__).resolve().parent
splits = json.loads((root / "speaker_splits.json").read_text(encoding="utf-8"))

expected_counts = {"train": 960, "validation": 240, "test": 240}
if set(splits) != set(expected_counts):
    raise SystemExit("Expected train, validation, and test splits.")

speaker_to_split = {}
for split, speakers in splits.items():
    for speaker in speakers:
        if speaker in speaker_to_split:
            raise SystemExit(f"Speaker {speaker} is assigned more than once.")
        speaker_to_split[speaker] = split

expected_speakers = {f"{number:02d}" for number in range(1, 25)}
if set(speaker_to_split) != expected_speakers:
    raise SystemExit("The split must cover speakers 01 through 24 exactly once.")

rows = []
for path in sorted((root / "data" / "raw").rglob("*.wav")):
    parts = path.stem.split("-")
    if len(parts) != 7 or parts[-1] not in speaker_to_split:
        raise SystemExit(f"Unexpected recording name: {path.name}")

    speaker = parts[-1]
    rows.append({
        "path": path.relative_to(root).as_posix(),
        "speaker_id": speaker,
        "emotion_code": parts[2],
        "split": speaker_to_split[speaker],
    })

counts = Counter(row["split"] for row in rows)
if counts != expected_counts:
    raise SystemExit(f"Unexpected recording counts: {dict(counts)}")

with (root / "data" / "manifest.csv").open("x", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=["path", "speaker_id", "emotion_code", "split"])
    writer.writeheader()
    writer.writerows(rows)

print("Created data/manifest.csv")
for split in expected_counts:
    print(f"{split}: {counts[split]} recordings")
