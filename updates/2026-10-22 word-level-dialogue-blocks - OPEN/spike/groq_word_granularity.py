"""One real Groq verbose_json call with word granularity, against a short
local fixture, to confirm the actual field names before writing Phase 3's
Groq word-extraction code (see the update's roadmap + wiki.md).

Run from the project root: .venv/bin/python "updates/2026-10-22 word-level-dialogue-blocks - OPEN/spike/groq_word_granularity.py"
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.transcription import _compact_mp3_for_upload  # noqa: E402

FIXTURE = Path(__file__).resolve().parents[3] / "testing" / "videos" / "martin_track3_60s.mp3"


def main() -> None:
    os.environ.setdefault("GROQ_API_KEY", "")
    for line in (Path(__file__).resolve().parents[3] / ".env").read_text().splitlines():
        if line.startswith("GROQ_API_KEY="):
            os.environ["GROQ_API_KEY"] = line.split("=", 1)[1].strip()

    from groq import Groq

    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    upload_path = _compact_mp3_for_upload(FIXTURE)
    try:
        with open(upload_path, "rb") as f:
            result = client.audio.transcriptions.create(
                file=(upload_path.name, f.read()),
                model="whisper-large-v3-turbo",
                response_format="verbose_json",
                timestamp_granularities=["word", "segment"],
            )
    finally:
        upload_path.unlink(missing_ok=True)

    segments = getattr(result, "segments", None) or []
    words = getattr(result, "words", None) or []
    print(f"segments: {len(segments)}, words: {len(words)}")
    if segments:
        print("first segment:", json.dumps(_as_dict(segments[0]), indent=2, default=str))
    if words:
        print("first 5 words:", json.dumps([_as_dict(w) for w in words[:5]], indent=2, default=str))


def _as_dict(obj):
    if isinstance(obj, dict):
        return obj
    return {k: getattr(obj, k) for k in dir(obj) if not k.startswith("_") and not callable(getattr(obj, k))}


if __name__ == "__main__":
    main()
