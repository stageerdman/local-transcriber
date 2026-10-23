from types import SimpleNamespace

from src.transcription import _coalesce_elevenlabs_words


def _word(text: str, start: float, end: float, type_: str = "word") -> SimpleNamespace:
    return SimpleNamespace(text=text, start=start, end=end, type=type_)


def test_coalesce_merges_words_with_small_gaps() -> None:
    words = [_word("Dobrý", 0.0, 0.3), _word("den", 0.35, 0.6)]
    segments = _coalesce_elevenlabs_words(words)
    assert len(segments) == 1
    assert segments[0].start == 0.0
    assert segments[0].end == 0.6
    assert segments[0].text == "Dobrý den"


def test_coalesce_splits_on_long_pause() -> None:
    # Gap of 2s, well over the 0.6s segment-break threshold.
    words = [_word("Ahoj", 0.0, 0.4), _word("jak", 2.4, 2.6)]
    segments = _coalesce_elevenlabs_words(words)
    assert [s.text for s in segments] == ["Ahoj", "jak"]


def test_coalesce_splits_on_sentence_end_even_without_a_pause() -> None:
    words = [_word("Dobrý den.", 0.0, 0.3), _word("Jak", 0.32, 0.5)]
    segments = _coalesce_elevenlabs_words(words)
    assert [s.text for s in segments] == ["Dobrý den.", "Jak"]


def test_coalesce_skips_spacing_and_audio_event_entries() -> None:
    words = [
        _word("Ahoj", 0.0, 0.3),
        _word(" ", 0.3, 0.32, type_="spacing"),
        _word("(laughter)", 0.32, 0.4, type_="audio_event"),
        _word("tam", 0.45, 0.6),
    ]
    segments = _coalesce_elevenlabs_words(words)
    assert len(segments) == 1
    assert segments[0].text == "Ahoj tam"


def test_coalesce_empty_words_returns_no_segments() -> None:
    assert _coalesce_elevenlabs_words([]) == []
