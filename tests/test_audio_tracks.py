import shutil
import subprocess
from pathlib import Path

import pytest

from src.audio_tracks import (
    compute_volume_envelope,
    extract_channel_to_mp3,
    extract_track_to_mp3,
    has_multiple_tracks,
    probe_audio_streams,
    safe_probe_audio_streams,
    track_speaker_names,
)

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe not installed",
)


def _make_multitrack_file(path: Path, *, titles: tuple[str | None, str | None] = (None, None)) -> None:
    """Build a real 2-audio-stream file (two distinct sine tones) via ffmpeg lavfi."""
    command = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=220:duration=1.5",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=440:duration=1.5",
        "-map",
        "0:a",
        "-map",
        "1:a",
    ]
    if titles[0]:
        command += ["-metadata:s:a:0", f"title={titles[0]}"]
    if titles[1]:
        command += ["-metadata:s:a:1", f"title={titles[1]}"]
    command += ["-c:a", "libmp3lame", str(path), "-loglevel", "error"]
    subprocess.run(command, check=True)


def _make_single_track_file(path: Path) -> None:
    command = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=220:duration=1.5",
        "-c:a",
        "libmp3lame",
        str(path),
        "-loglevel",
        "error",
    ]
    subprocess.run(command, check=True)


def test_probe_detects_two_streams_in_a_real_multitrack_file(tmp_path: Path) -> None:
    media = tmp_path / "multi.mkv"
    _make_multitrack_file(media)

    streams = probe_audio_streams(media)

    assert len(streams) == 2
    assert has_multiple_tracks(streams)
    assert streams[0].index != streams[1].index


def test_probe_reads_stream_title_tags(tmp_path: Path) -> None:
    media = tmp_path / "multi.mkv"
    _make_multitrack_file(media, titles=("Alice", "Bob"))

    streams = probe_audio_streams(media)

    assert track_speaker_names(streams) == ["Alice", "Bob"]


def test_track_speaker_names_falls_back_to_person_n_without_titles(tmp_path: Path) -> None:
    media = tmp_path / "multi.mkv"
    _make_multitrack_file(media)

    streams = probe_audio_streams(media)

    assert track_speaker_names(streams) == ["Person 1", "Person 2"]


def test_single_track_file_is_not_multi_track(tmp_path: Path) -> None:
    media = tmp_path / "single.mp3"
    _make_single_track_file(media)

    streams = probe_audio_streams(media)

    assert len(streams) == 1
    assert not has_multiple_tracks(streams)


def test_extract_track_to_mp3_produces_independent_correctly_sized_files(tmp_path: Path) -> None:
    media = tmp_path / "multi.mkv"
    _make_multitrack_file(media)
    streams = probe_audio_streams(media)

    out0 = tmp_path / "track0.mp3"
    out1 = tmp_path / "track1.mp3"
    extract_track_to_mp3(media, streams[0].index, out0)
    extract_track_to_mp3(media, streams[1].index, out1)

    assert out0.exists() and out0.stat().st_size > 0
    assert out1.exists() and out1.stat().st_size > 0

    duration0 = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(out0)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert float(duration0) == pytest.approx(1.5, abs=0.2)


def test_safe_probe_returns_empty_list_for_nonexistent_file(tmp_path: Path) -> None:
    assert safe_probe_audio_streams(tmp_path / "does-not-exist.mp4") == []


def test_compute_volume_envelope_returns_requested_bucket_count(tmp_path: Path) -> None:
    media = tmp_path / "single.mp3"
    _make_single_track_file(media)
    streams = probe_audio_streams(media)

    envelope = compute_volume_envelope(media, streams[0].index, num_buckets=12)

    assert len(envelope) == 12
    assert all(0.0 <= level <= 1.0 for level in envelope)
    assert max(envelope) == pytest.approx(1.0)


def test_compute_volume_envelope_is_all_zero_for_silence(tmp_path: Path) -> None:
    media = tmp_path / "silence.mp3"
    subprocess.run(
        [
            "ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono",
            "-t", "1.5", "-c:a", "libmp3lame", str(media), "-loglevel", "error",
        ],
        check=True,
    )
    streams = probe_audio_streams(media)

    envelope = compute_volume_envelope(media, streams[0].index, num_buckets=10)

    assert envelope == [0.0] * 10


def _make_stereo_silent_left_tone_right(path: Path) -> None:
    """A stereo file where the two channels carry different content: left
    is silence, right has a tone - the "these channels are two different
    people" case, as opposed to a mono signal duplicated across both."""
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=1.5",
            "-filter_complex", "[0:a][1:a]join=inputs=2:channel_layout=stereo[a]",
            "-map", "[a]", "-t", "1.5", "-c:a", "pcm_s16le", str(path), "-loglevel", "error",
        ],
        check=True,
    )


def test_compute_volume_envelope_isolates_requested_channel(tmp_path: Path) -> None:
    media = tmp_path / "split.wav"
    _make_stereo_silent_left_tone_right(media)
    streams = probe_audio_streams(media)
    assert streams[0].channels == 2

    left = compute_volume_envelope(media, streams[0].index, num_buckets=10, channel_index=0)
    right = compute_volume_envelope(media, streams[0].index, num_buckets=10, channel_index=1)

    assert max(left) == 0.0
    assert max(right) > 0.0


def test_extract_channel_to_mp3_produces_a_mono_file(tmp_path: Path) -> None:
    media = tmp_path / "split.wav"
    _make_stereo_silent_left_tone_right(media)
    streams = probe_audio_streams(media)

    out = tmp_path / "right_channel.mp3"
    extract_channel_to_mp3(media, streams[0].index, 1, out)

    assert out.exists() and out.stat().st_size > 0
    extracted = probe_audio_streams(out)
    assert extracted[0].channels == 1
