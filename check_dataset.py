from collections import Counter, defaultdict
from pathlib import Path
import re
import wave

files = sorted((Path(__file__).resolve().parent / "data" / "raw").rglob("*.wav"))
counts = Counter()
emotions_by_actor = defaultdict(set)
expected_emotions = {f"{n:02d}" for n in range(1, 9)}

for path in files:
    match = re.fullmatch(
        r"03-01-(0[1-8])-0[12]-0[12]-0[12]-(0[1-9]|1[0-9]|2[0-4])",
        path.stem,
    )
    if not match:
        raise SystemExit(f"Invalid filename: {path}")

    emotion, actor = match.groups()
    try:
        with wave.open(str(path), "rb") as audio:
            frames = audio.getnframes()
            frame_size = audio.getnchannels() * audio.getsampwidth()
            if frames == 0 or len(audio.readframes(frames)) != frames * frame_size:
                raise ValueError("empty or truncated audio")
    except (OSError, EOFError, wave.Error, ValueError) as exc:
        raise SystemExit(f"Unreadable WAV: {path} ({exc})")

    counts[actor] += 1
    emotions_by_actor[actor].add(emotion)

if len(files) != 1440 or len(counts) != 24:
    raise SystemExit(f"Unexpected totals: {len(files)} files, {len(counts)} speakers")

for actor in sorted(counts):
    if counts[actor] != 60 or emotions_by_actor[actor] != expected_emotions:
        raise SystemExit(f"Unexpected recordings or emotion classes for speaker {actor}")

print("Readable WAV files: 1440")
print("Speakers: 24")
print("Recordings per speaker: 60")
print("Emotion classes per speaker: 8")
