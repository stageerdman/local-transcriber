from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def check_ffmpeg_installed() -> bool:
    return shutil.which("ffmpeg") is not None


def prepare_mp3(input_file: Path, output_audio_path: Path) -> None:
    output_audio_path.parent.mkdir(parents=True, exist_ok=True)

    if input_file.suffix.lower() == ".mp3":
        shutil.copy2(input_file, output_audio_path)
        return

    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_file),
        "-vn",
        "-codec:a",
        "libmp3lame",
        "-q:a",
        "2",
        str(output_audio_path),
    ]
    subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
