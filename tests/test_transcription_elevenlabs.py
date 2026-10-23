from types import SimpleNamespace

from src.transcription import _coalesce_elevenlabs_words


def _word(text: str, start: float, end: float, type_: str = "word") -> SimpleNamespace:
    return SimpleNamespace(text=text, start=start, end=end, type=type_)


def _spacing(start: float, end: float, text: str = " ") -> SimpleNamespace:
    return _word(text, start, end, type_="spacing")


def test_coalesce_merges_words_with_small_gaps() -> None:
    words = [_word("Dobrý", 0.0, 0.3), _spacing(0.3, 0.35), _word("den", 0.35, 0.6)]
    segments = _coalesce_elevenlabs_words(words)
    assert len(segments) == 1
    assert segments[0].start == 0.0
    assert segments[0].end == 0.6
    assert segments[0].text == "Dobrý den"


def test_coalesce_splits_on_long_pause() -> None:
    # Gap of 2s, well over the 0.6s segment-break threshold.
    words = [_word("Ahoj", 0.0, 0.4), _spacing(0.4, 2.4), _word("jak", 2.4, 2.6)]
    segments = _coalesce_elevenlabs_words(words)
    assert [s.text for s in segments] == ["Ahoj", "jak"]


def test_coalesce_splits_on_sentence_end_even_without_a_pause() -> None:
    words = [_word("Dobrý den.", 0.0, 0.3), _spacing(0.3, 0.32), _word("Jak", 0.32, 0.5)]
    segments = _coalesce_elevenlabs_words(words)
    assert [s.text for s in segments] == ["Dobrý den.", "Jak"]


def test_coalesce_skips_spacing_and_audio_event_entries() -> None:
    words = [
        _word("Ahoj", 0.0, 0.3),
        _spacing(0.3, 0.32),
        _word("(laughter)", 0.32, 0.4, type_="audio_event"),
        _word("tam", 0.45, 0.6),
    ]
    segments = _coalesce_elevenlabs_words(words)
    assert len(segments) == 1
    assert segments[0].text == "Ahoj tam"


def test_coalesce_empty_words_returns_no_segments() -> None:
    assert _coalesce_elevenlabs_words([]) == []


def test_a_word_with_no_spacing_token_before_it_attaches_with_no_space() -> None:
    # No `spacing` entry between "Ahoj" and "," - matches ElevenLabs' own
    # convention for punctuation/contractions that attach to the previous word.
    words = [_word("Ahoj", 0.0, 0.3), _word(",", 0.3, 0.31)]
    segments = _coalesce_elevenlabs_words(words)
    assert len(segments) == 1
    assert segments[0].text == "Ahoj,"


def test_an_empty_text_spacing_token_does_not_force_a_space() -> None:
    words = [_word("Ahoj", 0.0, 0.3), _spacing(0.3, 0.3, text=""), _word("!", 0.3, 0.31)]
    segments = _coalesce_elevenlabs_words(words)
    assert segments[0].text == "Ahoj!"


def test_segments_carry_their_own_word_level_timing() -> None:
    words = [_word("Dobrý", 0.0, 0.3), _spacing(0.3, 0.35), _word("den", 0.35, 0.6)]
    segments = _coalesce_elevenlabs_words(words)
    assert [w.text for w in segments[0].words] == ["Dobrý", "den"]
    assert segments[0].words[0].start == 0.0
    assert segments[0].words[1].end == 0.6
    assert segments[0].words[1].lead_space is True
