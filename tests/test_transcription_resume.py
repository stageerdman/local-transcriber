"""chunk_sink / start_chunk resume behavior for the Whisper VAD path.

Mocks out mlx_whisper entirely (we're testing the chunk loop, not the model):
each VAD chunk decodes to one segment whose text is its chunk index.
"""
import sys
import types

import pytest

from src import transcription


@pytest.fixture
def fake_whisper(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(transcription, "ensure_mlx_whisper_available", lambda: None)

    # One "segment" per decode, text = a marker so we can see which chunk it came
    # from. The VAD path slices audio[start:end] per chunk, so we key off len.
    def fake_transcribe(audio, path_or_hf_repo=None, language=None, verbose=False, **opts):
        marker = f"seg@{int(audio[0])}"
        return {"segments": [{"text": marker, "start": 0.0, "end": 1.0, "words": []}]}

    fake_mod = types.ModuleType("mlx_whisper")
    fake_mod.transcribe = fake_transcribe
    audio_mod = types.ModuleType("mlx_whisper.audio")
    audio_mod.load_audio = lambda p: list(range(1000))  # dummy; not really sliced meaningfully
    fake_mod.audio = audio_mod
    monkeypatch.setitem(sys.modules, "mlx_whisper", fake_mod)
    monkeypatch.setitem(sys.modules, "mlx_whisper.audio", audio_mod)

    # Deterministic VAD chunk list: 4 speech regions.
    chunks = [(0, 100), (100, 200), (200, 300), (300, 400)]
    monkeypatch.setattr(transcription, "_speech_chunks", lambda audio: chunks)
    return chunks


def test_chunk_sink_fires_once_per_chunk_with_indices(fake_whisper) -> None:
    seen: list[int] = []
    transcription.transcribe_mp3_segments(
        "mlx-community/whisper-small-mlx",
        audio_path="ignored",
        chunk_sink=lambda idx, segs: seen.append(idx),
    )
    assert seen == [0, 1, 2, 3]


def test_start_chunk_skips_leading_chunks(fake_whisper) -> None:
    seen: list[int] = []
    returned = transcription.transcribe_mp3_segments(
        "mlx-community/whisper-small-mlx",
        audio_path="ignored",
        chunk_sink=lambda idx, segs: seen.append(idx),
        start_chunk=2,
    )
    # Only chunks 2 and 3 run on resume; 0 and 1 are assumed already persisted.
    assert seen == [2, 3]
    assert len(returned) == 2


def test_head_plus_resumed_tail_equals_full_run(fake_whisper) -> None:
    full = transcription.transcribe_mp3_segments(
        "mlx-community/whisper-small-mlx", audio_path="ignored"
    )
    # Simulate a crash after 2 chunks: keep the head, resume the tail.
    head = transcription.transcribe_mp3_segments(
        "mlx-community/whisper-small-mlx", audio_path="ignored", start_chunk=0
    )[:2]
    tail = transcription.transcribe_mp3_segments(
        "mlx-community/whisper-small-mlx", audio_path="ignored", start_chunk=2
    )
    assert [s.text for s in head + tail] == [s.text for s in full]
