#!/usr/bin/env python
"""One-time setup: convert the Czech-finetuned Whisper large-v3 to MLX.

Downloads `mikr/whisper-large-v3-czech-cv13` (a Hugging Face fp16 finetune) and
converts it to MLX format at the path `src.transcription.CZECH_FT_PATH`, which is
the default "Czech - finetuned large-v3" model in the app. Run once:

    python scripts/convert_czech_model.py

~3 GB download + ~3 GB output. Idempotent: skips if already converted (pass
--force to redo). Uses the vendored mlx-examples whisper converter
(`scripts/mlx_whisper_convert.py`).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.transcription import CZECH_FT_PATH  # noqa: E402

HF_REPO = "mikr/whisper-large-v3-czech-cv13"
CONVERTER = ROOT / "scripts" / "mlx_whisper_convert.py"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="reconvert even if present")
    args = parser.parse_args()

    weights = CZECH_FT_PATH / "weights.safetensors"
    if weights.exists() and not args.force:
        print(f"[skip] already converted at {CZECH_FT_PATH}")
        return

    CZECH_FT_PATH.mkdir(parents=True, exist_ok=True)
    print(f"[convert] {HF_REPO} -> {CZECH_FT_PATH}")
    subprocess.run(
        [sys.executable, str(CONVERTER),
         "--torch-name-or-path", HF_REPO,
         "--mlx-path", str(CZECH_FT_PATH),
         "--dtype", "float16"],
        check=True,
    )

    # The converter writes `model.safetensors`; mlx_whisper's loader looks for
    # `weights.safetensors`. Rename so the app can load it directly.
    produced = CZECH_FT_PATH / "model.safetensors"
    if produced.exists():
        produced.rename(weights)
    if not weights.exists():
        raise SystemExit("conversion did not produce weights.safetensors")
    print(f"[done] {weights} ({weights.stat().st_size / 1e9:.1f} GB)")


if __name__ == "__main__":
    main()
