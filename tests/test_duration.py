import shutil
import subprocess
from pathlib import Path

import pytest

from src.duration import probe_duration_seconds

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


def test_probe_duration_seconds_reads_generated_clip(tmp_path: Path) -> None:
    clip = tmp_path / "silence.mp3"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "anullsrc=r=22050:cl=mono",
            "-t",
            "2",
            "-codec:a",
            "libmp3lame",
            str(clip),
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    duration = probe_duration_seconds(clip)

    assert duration == pytest.approx(2.0, abs=0.2)
