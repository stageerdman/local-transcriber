from types import SimpleNamespace

from src.transcription import (
    _bucket_groq_words_by_segment,
    _groq_bucket_words,
    _parakeet_sentence_words,
    _whisper_segment_words,
)


def _token(text: str, start: float, end: float) -> SimpleNamespace:
    return SimpleNamespace(text=text, start=start, end=end)


# -- Parakeet: sub-word tokens -> whole words ---------------------------------

def test_parakeet_groups_subword_tokens_into_one_word() -> None:
    tokens = [_token(" pros", 12.88, 13.04), _token("te", 13.04, 13.2)]
    words = _parakeet_sentence_words(tokens)
    assert len(words) == 1
    assert words[0].text == "proste"
    assert words[0].start == 12.88
    assert words[0].end == 13.2


def test_parakeet_attaches_punctuation_with_no_leading_space() -> None:
    tokens = [
        _token(" r", 13.2, 13.36), _token("ô", 13.36, 13.44),
        _token("z", 13.44, 13.6), _token("ne", 13.6, 13.68), _token(".", 13.68, 14.0),
    ]
    words = _parakeet_sentence_words(tokens)
    assert len(words) == 1
    assert words[0].text == "rôzne."
    assert words[0].end == 14.0


def test_parakeet_separate_tokens_with_leading_spaces_are_separate_words() -> None:
    tokens = [_token(" Ale", 11.6, 11.84), _token(" to", 12.08, 12.24)]
    words = _parakeet_sentence_words(tokens)
    assert [w.text for w in words] == ["Ale", "to"]
    assert words[1].lead_space is True


def test_parakeet_empty_tokens_returns_no_words() -> None:
    assert _parakeet_sentence_words([]) == []


# -- Groq: a flat per-file word list bucketed into segments by word count ---
# (not by timestamp - a real ~60s call showed segment/word timestamps drift
# apart over the file; word count against the segment's own text doesn't.)

def test_groq_buckets_words_by_matching_the_segment_texts_word_count() -> None:
    segments = [{"text": "všetkým ale"}, {"text": "to je"}]
    words = [
        {"word": "všetkým", "start": 0.5, "end": 0.9},
        {"word": "ale", "start": 3.5, "end": 3.8},
        {"word": "to", "start": 4.2, "end": 4.4},
        {"word": "je", "start": 4.4, "end": 4.6},
    ]
    buckets = _bucket_groq_words_by_segment(segments, words)
    assert [w["word"] for w in buckets[0]] == ["všetkým", "ale"]
    assert [w["word"] for w in buckets[1]] == ["to", "je"]


def test_groq_bucketing_is_unaffected_by_timestamp_drift_between_segments_and_words() -> None:
    # Word timestamps that have drifted a whole word "late" relative to the
    # segment boundaries they actually belong to - exactly the real-world
    # failure mode time-based bucketing doesn't survive.
    segments = [{"text": "ahoj"}, {"text": "jak sa"}, {"text": "mas"}]
    words = [
        {"word": "ahoj", "start": 0.0, "end": 0.3},
        {"word": "jak", "start": 4.9, "end": 5.1},  # "belongs" after seg 2's end by time
        {"word": "sa", "start": 5.1, "end": 5.3},
        {"word": "mas", "start": 5.3, "end": 5.5},
    ]
    buckets = _bucket_groq_words_by_segment(segments, words)
    assert [w["word"] for w in buckets[0]] == ["ahoj"]
    assert [w["word"] for w in buckets[1]] == ["jak", "sa"]
    assert [w["word"] for w in buckets[2]] == ["mas"]


def test_groq_bucket_with_no_segments_returns_no_buckets() -> None:
    assert _bucket_groq_words_by_segment([], [{"word": "x", "start": 0.0, "end": 0.1}]) == []


def test_groq_words_from_bucket_are_space_separated() -> None:
    bucket = [{"word": "ahoj", "start": 0.0, "end": 0.3}, {"word": "jak", "start": 0.3, "end": 0.5}]
    words = _groq_bucket_words(bucket)
    assert [w.text for w in words] == ["ahoj", "jak"]
    assert words[1].lead_space is True


# -- Whisper: word dicts, with the VAD path's chunk-relative offset ----------

def test_whisper_segment_words_reads_the_leading_space_convention() -> None:
    seg = {"words": [{"word": " Hello", "start": 1.0, "end": 1.3}, {"word": ",", "start": 1.3, "end": 1.31}]}
    words = _whisper_segment_words(seg)
    assert [w.text for w in words] == ["Hello", ","]
    assert words[1].lead_space is False


def test_whisper_segment_words_applies_the_vad_chunk_offset() -> None:
    seg = {"words": [{"word": " Hi", "start": 1.0, "end": 1.2}]}
    words = _whisper_segment_words(seg, offset=10.0)
    assert words[0].start == 11.0
    assert words[0].end == 11.2


def test_whisper_segment_with_no_words_returns_no_words() -> None:
    assert _whisper_segment_words({"words": []}) == []
