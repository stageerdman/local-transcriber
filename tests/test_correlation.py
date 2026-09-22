import pytest

from src.audio_tracks import correlation


def test_correlation_is_one_for_identical_series() -> None:
    series = [0.1, 0.5, 0.9, 0.3, 0.0, 0.7]
    assert correlation(series, series) == pytest.approx(1.0)


def test_correlation_is_high_for_scaled_but_shaped_alike_series() -> None:
    a = [0.1, 0.5, 0.9, 0.3, 0.0, 0.7]
    b = [x * 0.5 for x in a]
    assert correlation(a, b) == pytest.approx(1.0)


def test_correlation_is_low_for_unrelated_series() -> None:
    a = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
    b = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
    assert correlation(a, b) == pytest.approx(-1.0)


def test_correlation_is_zero_for_constant_series() -> None:
    assert correlation([0.0, 0.0, 0.0], [0.1, 0.2, 0.3]) == 0.0


def test_correlation_is_zero_for_mismatched_lengths() -> None:
    assert correlation([0.1, 0.2], [0.1, 0.2, 0.3]) == 0.0


def test_correlation_is_zero_for_empty_series() -> None:
    assert correlation([], []) == 0.0
