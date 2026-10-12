from dataclasses import dataclass

import src.transcription as transcription
from app.ui import format_minutes_left, format_timecode
from src.transcription import transcribe_mp3_segments


def test_format_timecode():
    assert format_timecode(None) == "0:00"
    assert format_timecode(0) == "0:00"
    assert format_timecode(9) == "0:09"
    assert format_timecode(90) == "1:30"
    assert format_timecode(3665) == "1:01:05"


def test_format_minutes_left():
    assert format_minutes_left(None) is None
    assert format_minutes_left(-5) is None
    assert format_minutes_left(10) == "<1 min"
    assert format_minutes_left(600) == "10 min"


@dataclass
class _FakeSentence:
    start: float
    end: float
    text: str


class _FakeResult:
    def __init__(self, sentences):
        self.sentences = sentences


class _FakeParakeet:
    """Mimics parakeet-mlx: fires chunk_callback(current, total) samples per
    chunk, then returns aligned sentences."""

    def __init__(self):
        self.callback_args = []

    def transcribe(self, path, *, chunk_duration, overlap_duration, chunk_callback):
        if chunk_callback is not None:
            chunk_callback(48000, 96000)   # halfway (samples)
            chunk_callback(96000, 96000)   # end
        self.callback_args = [chunk_duration, overlap_duration]
        return _FakeResult([_FakeSentence(0.0, 1.0, "hello "), _FakeSentence(1.0, 2.0, "  ")])


def test_parakeet_forwards_progress_as_fraction(monkeypatch):
    fake = _FakeParakeet()
    monkeypatch.setattr(transcription, "_get_parakeet_model", lambda name: fake)

    fractions = []
    segments = transcribe_mp3_segments(
        "mlx-community/parakeet-tdt-0.6b-v3",
        "unused.mp3",
        progress_callback=fractions.append,
    )

    # Samples were mapped to [0, 1] fractions.
    assert fractions == [0.5, 1.0]
    # Empty/whitespace-only sentences are dropped; text is stripped.
    assert [s.text for s in segments] == ["hello"]


def test_parakeet_without_callback_still_works(monkeypatch):
    fake = _FakeParakeet()
    monkeypatch.setattr(transcription, "_get_parakeet_model", lambda name: fake)

    segments = transcribe_mp3_segments("mlx-community/parakeet-tdt-0.6b-v3", "unused.mp3")

    assert [s.text for s in segments] == ["hello"]


def test_whisper_vad_progress_tracks_chunk_samples(monkeypatch):
    """Whisper VAD progress must advance by each chunk's SAMPLE span, not by the
    word-level seconds of its last segment. Regression guard: `_seg_times`
    returns seconds, and shadowing the loop's sample-span `start`/`end` with it
    pinned the fraction (done_speech seconds / total_speech samples) at ~0."""
    # Two 10s speech chunks @ 16 kHz -> 160000 samples each, 320000 total.
    chunks = [(0, 160000), (160000, 320000)]
    monkeypatch.setattr(transcription, "_speech_chunks", lambda audio: chunks)
    monkeypatch.setattr(transcription, "ensure_mlx_whisper_available", lambda: None)

    # Each decode yields one short segment: word times are small (~seconds),
    # which is exactly what used to leak into the progress denominator.
    def fake_transcribe(audio, **kwargs):
        return {"segments": [{"text": "hi", "start": 0.0, "end": 1.0,
                              "words": [{"start": 0.0, "end": 1.0}]}]}

    import sys
    import types

    fake_mlx = types.ModuleType("mlx_whisper")
    fake_mlx.transcribe = fake_transcribe
    fake_audio = types.ModuleType("mlx_whisper.audio")
    fake_audio.load_audio = lambda path: list(range(320000))  # sliceable stand-in
    monkeypatch.setitem(sys.modules, "mlx_whisper", fake_mlx)
    monkeypatch.setitem(sys.modules, "mlx_whisper.audio", fake_audio)

    fractions = []
    transcribe_mp3_segments(
        "mlx-community/whisper-small",
        "unused.mp3",
        progress_callback=fractions.append,
        use_vad=True,
    )

    assert fractions == [0.5, 1.0]
