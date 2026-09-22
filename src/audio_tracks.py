from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from src.duration import probe_duration_seconds


@dataclass
class AudioStream:
    """One audio stream inside a media container.

    `index` is the absolute ffmpeg stream index (used with `-map 0:<index>`
    to pull just this stream out), not a 0-based "audio stream number".
    """

    index: int
    channels: int
    title: str | None = None


def probe_audio_streams(path: Path) -> list[AudioStream]:
    """List every audio stream in a media file via ffprobe.

    Most recordings have exactly one. Conferencing tools that save a
    separate audio track per participant (e.g. Zoom's "record a separate
    audio file for each participant") produce more than one - that's the
    signal used elsewhere to switch into per-speaker transcription.
    """
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a",
        "-show_entries",
        "stream=index,channels:stream_tags=title",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    data = json.loads(result.stdout or "{}")

    streams = []
    for entry in data.get("streams", []):
        # Only `title` is used as a speaker-name hint: muxers (ffmpeg's aac
        # encoder, etc.) stamp a generic `handler_name` like "SoundHandler"
        # on nearly every audio stream, so it's not a real participant name.
        title = entry.get("tags", {}).get("title")
        streams.append(
            AudioStream(
                index=int(entry["index"]),
                channels=int(entry.get("channels", 1)),
                title=title.strip() if title else None,
            )
        )
    return streams


def has_multiple_tracks(streams: list[AudioStream]) -> bool:
    return len(streams) > 1


def correlation(a: list[float], b: list[float]) -> float:
    """Pearson correlation between two equal-length series, in [-1, 1].

    Used to spot near-duplicate audio - a mono signal duplicated across
    stereo channels, or the same source routed to more than one track (both
    common in multi-track OBS/conferencing recordings) - so the UI can point
    it out instead of silently showing redundant "people". Returns 0.0 for
    mismatched lengths or a constant (e.g. all-silent) series, where
    correlation is undefined.
    """
    n = len(a)
    if n == 0 or n != len(b):
        return 0.0
    mean_a = sum(a) / n
    mean_b = sum(b) / n
    deviations_a = [x - mean_a for x in a]
    deviations_b = [x - mean_b for x in b]
    numerator = sum(x * y for x, y in zip(deviations_a, deviations_b))
    spread_a = sum(x * x for x in deviations_a) ** 0.5
    spread_b = sum(x * x for x in deviations_b) ** 0.5
    if spread_a == 0 or spread_b == 0:
        return 0.0
    return numerator / (spread_a * spread_b)


def extract_track_to_mp3(source: Path, stream_index: int, output_path: Path) -> None:
    """Extract a single audio stream from `source` into its own mp3 file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(source),
        "-map",
        f"0:{stream_index}",
        "-vn",
        "-codec:a",
        "libmp3lame",
        "-q:a",
        "2",
        str(output_path),
    ]
    subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def track_speaker_names(streams: list[AudioStream]) -> list[str]:
    """Display name per track: the stream's title/handler tag if present, else "Person N"."""
    names = []
    for i, stream in enumerate(streams, start=1):
        names.append(stream.title if stream.title else f"Person {i}")
    return names


def extract_channel_to_mp3(source: Path, stream_index: int, channel_index: int, output_path: Path) -> None:
    """Extract one channel of one audio stream (e.g. left-only out of a
    stereo stream) into its own mono mp3 file.

    Use this instead of `extract_track_to_mp3` when a stream's channels
    don't carry the same content and only one of them should represent this
    track - a full-stream downmix would blend two different signals
    together instead of isolating the one that matters.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(source),
        "-map",
        f"0:{stream_index}",
        "-af",
        f"pan=mono|c0=c{channel_index}",
        "-codec:a",
        "libmp3lame",
        "-q:a",
        "2",
        str(output_path),
    ]
    subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def compute_volume_envelope(
    source: Path, stream_index: int, num_buckets: int = 150, channel_index: int | None = None
) -> list[float]:
    """Per-bucket loudness (0-1, normalized to the track's own loudest bucket)
    for one audio stream across the whole file - drives the "volume over
    time" preview shown next to a track in the UI.

    Uses ffmpeg's `astats` filter to compute RMS level per fixed-size sample
    window directly (no numpy / raw-PCM decoding on the Python side).

    `channel_index` isolates a single channel (e.g. right-only out of a
    stereo stream) instead of downmixing every channel together - needed to
    show each channel's own waveform when they don't carry the same signal.
    """
    duration_seconds = probe_duration_seconds(source)
    if duration_seconds <= 0 or num_buckets <= 0:
        return []

    # The resample has to happen *inside* the filter chain (not via -ar/-ac
    # output options) so `asetnsamples`'s `n` - and therefore the bucket
    # count - is counted against the rate it actually runs at, not the
    # source file's native rate.
    sample_rate = 8000
    samples_per_bucket = max(1, round(sample_rate * duration_seconds / num_buckets))
    channel_select = f"pan=mono|c0=c{channel_index}" if channel_index is not None else "aformat=channel_layouts=mono"
    filter_chain = (
        f"aresample={sample_rate},{channel_select},"
        f"asetnsamples=n={samples_per_bucket}:p=0,"
        "astats=metadata=1:reset=1,"
        "ametadata=print:key=lavfi.astats.Overall.RMS_level:file=-"
    )
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(source),
        "-map",
        f"0:{stream_index}",
        "-af",
        filter_chain,
        "-f",
        "null",
        "-",
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)

    levels_db: list[float] = []
    for line in result.stdout.splitlines():
        if "lavfi.astats.Overall.RMS_level=" not in line:
            continue
        value = line.rsplit("=", 1)[-1].strip()
        try:
            levels_db.append(float(value))
        except ValueError:
            levels_db.append(float("-inf"))

    envelope = [0.0 if db == float("-inf") else 10 ** (db / 20) for db in levels_db]
    peak = max(envelope, default=0.0)
    if peak <= 0:
        return envelope
    return [min(1.0, level / peak) for level in envelope]


def safe_probe_audio_streams(path: Path) -> list[AudioStream]:
    """`probe_audio_streams`, but never raises - falls back to "single track" on any failure."""
    if shutil.which("ffprobe") is None:
        return []
    try:
        return probe_audio_streams(path)
    except (subprocess.CalledProcessError, json.JSONDecodeError, OSError):
        return []
