"""Bundle all candidate transcripts for one track into a single review file,
so an LLM-judge can compare them side by side. Config labels are anonymized
(A, B, C...) to avoid biasing the judge toward a known "best" model.

Usage:
    python bundle_for_judge.py <track_substring>   # e.g. tomas__spk1
Writes <work>/judge/<track>__bundle.md and a key file mapping letters->configs.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

WORK = Path(
    "/private/tmp/claude-501/-Users-stage-dev-transcriber/"
    "7a0eee61-b76e-48e9-b9a4-bfca94046233/scratchpad/asr_research"
)
OUT = WORK / "transcripts"
JUDGE = WORK / "judge"
JUDGE.mkdir(exist_ok=True)


def main() -> None:
    track = sys.argv[1]
    results = json.loads((WORK / "results.json").read_text())
    entries = sorted(k for k in results if k.startswith(track))
    key_lines, body_lines = [], []
    for i, k in enumerate(entries):
        letter = chr(ord("A") + i)
        cfg = results[k]["config"]
        txt = (OUT / f"{k}.txt").read_text().strip()
        key_lines.append(f"{letter} = {cfg}  (rep={results[k]['repeat_5gram_frac']}, "
                         f"comp={results[k]['compression_ratio']}, words={results[k]['words']})")
        body_lines.append(f"\n\n===== CANDIDATE {letter} =====\n{txt}")

    (JUDGE / f"{track}__key.txt").write_text("\n".join(key_lines))
    (JUDGE / f"{track}__bundle.md").write_text(
        f"# Candidate transcripts for track: {track}\n" + "".join(body_lines)
    )
    print(f"bundled {len(entries)} candidates for {track}")
    print("\n".join(key_lines))


if __name__ == "__main__":
    main()
