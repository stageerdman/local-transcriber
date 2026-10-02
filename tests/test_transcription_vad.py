import pytest

from src import transcription
from src.transcription import CZECH_FT_ID, _group_regions, _resolve_model


def test_group_regions_merges_close_small_gaps() -> None:
    # Two regions 100 samples apart, well under the 1000-sample merge gap and
    # the 10_000-sample max span -> merged into one chunk.
    regions = [(0, 1000), (1100, 2000)]
    assert _group_regions(regions, max_len=10_000, merge_gap=1000) == [(0, 2000)]


def test_group_regions_splits_on_large_gap() -> None:
    # Gap of 5000 > merge_gap -> stays two chunks.
    regions = [(0, 1000), (6000, 7000)]
    assert _group_regions(regions, max_len=10_000, merge_gap=1000) == [(0, 1000), (6000, 7000)]


def test_group_regions_splits_when_span_would_exceed_max() -> None:
    # Close gap, but merging would exceed max_len -> keep separate.
    regions = [(0, 900), (950, 2000)]
    assert _group_regions(regions, max_len=1000, merge_gap=1000) == [(0, 900), (950, 2000)]


def test_group_regions_empty() -> None:
    assert _group_regions([], max_len=10, merge_gap=10) == []


def test_resolve_model_passthrough_for_normal_repo() -> None:
    assert _resolve_model("mlx-community/whisper-large-v3-mlx") == "mlx-community/whisper-large-v3-mlx"


def test_resolve_czech_id_returns_path_when_present(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(transcription, "CZECH_FT_PATH", tmp_path)
    assert _resolve_model(CZECH_FT_ID) == str(tmp_path)


def test_resolve_czech_id_raises_helpful_error_when_missing(tmp_path, monkeypatch) -> None:
    missing = tmp_path / "not-there"
    monkeypatch.setattr(transcription, "CZECH_FT_PATH", missing)
    with pytest.raises(RuntimeError, match="convert_czech_model"):
        _resolve_model(CZECH_FT_ID)
