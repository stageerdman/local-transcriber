from src.dialogue import (
    Segment,
    Word,
    _join_words,
    blocks_to_dicts,
    format_timestamp,
    merge_segments_into_blocks,
    realign_words_to_text,
    render_blocks_markdown,
)


def test_merges_consecutive_same_speaker_segments_within_pause() -> None:
    segments = [
        Segment("Person 1", 0.0, 2.0, "Hey there."),
        Segment("Person 1", 2.3, 4.0, "How are you?"),
    ]
    blocks = merge_segments_into_blocks(segments, pause_seconds=0.8)

    assert len(blocks) == 1
    assert blocks[0].text == "Hey there. How are you?"
    assert blocks[0].start == 0.0
    assert blocks[0].end == 4.0


def test_starts_new_block_after_a_long_pause_same_speaker() -> None:
    segments = [
        Segment("Person 1", 0.0, 2.0, "Hey there."),
        Segment("Person 1", 5.0, 6.0, "Anyway, moving on."),
    ]
    blocks = merge_segments_into_blocks(segments, pause_seconds=0.8)

    assert len(blocks) == 2
    assert [b.text for b in blocks] == ["Hey there.", "Anyway, moving on."]


def test_same_speaker_always_merges_across_a_long_pause_when_other_speakers_present() -> None:
    segments = [
        Segment("Person 1", 0.0, 2.0, "Hey there."),
        Segment("Person 1", 5.0, 6.0, "Still me, after a long pause."),
        Segment("Person 2", 6.5, 7.0, "Mhm."),
    ]
    blocks = merge_segments_into_blocks(segments)

    assert len(blocks) == 2
    assert blocks[0].speaker == "Person 1"
    assert blocks[0].text == "Hey there. Still me, after a long pause."
    assert blocks[0].start == 0.0
    assert blocks[0].end == 6.0
    assert blocks[1].speaker == "Person 2"


def test_merged_block_words_carry_their_own_original_timestamps() -> None:
    segments = [
        Segment("Person 1", 0.0, 2.0, "Hey there.", words=[Word(0.0, 2.0, "Hey there.", True)]),
        Segment(
            "Person 1", 5.0, 6.0, "Still talking.",
            words=[Word(5.0, 5.4, "Still", True), Word(5.4, 6.0, "talking.", True)],
        ),
        Segment("Person 2", 6.5, 7.0, "Mhm.", words=[Word(6.5, 7.0, "Mhm.", True)]),
    ]
    blocks = merge_segments_into_blocks(segments)

    assert [w.text for w in blocks[0].words] == ["Hey there.", "Still", "talking."]
    assert [w.start for w in blocks[0].words] == [0.0, 5.0, 5.4]
    assert blocks[0].text == "Hey there. Still talking."


def test_segment_without_words_falls_back_to_one_synthetic_word() -> None:
    segments = [Segment("Person 1", 0.0, 2.0, "Hey there.")]
    blocks = merge_segments_into_blocks(segments)

    assert len(blocks[0].words) == 1
    assert blocks[0].words[0] == Word(0.0, 2.0, "Hey there.", True)


def test_speaker_change_always_starts_a_new_block_even_with_no_gap() -> None:
    segments = [
        Segment("Person 1", 0.0, 2.0, "Hey there."),
        Segment("Person 2", 2.1, 3.0, "Hi!"),
    ]
    blocks = merge_segments_into_blocks(segments, pause_seconds=0.8)

    assert [b.speaker for b in blocks] == ["Person 1", "Person 2"]


def test_overlapping_speech_is_flagged_as_interruption() -> None:
    segments = [
        Segment("Person 1", 0.0, 5.0, "So I wanted to go over the proposal from last week"),
        Segment("Person 2", 3.0, 4.0, "Actually, before that"),
    ]
    blocks = merge_segments_into_blocks(segments)

    assert len(blocks) == 2
    assert blocks[0].interrupting is False
    assert blocks[1].speaker == "Person 2"
    assert blocks[1].interrupting is True


def test_non_overlapping_turn_taking_is_not_flagged_as_interruption() -> None:
    segments = [
        Segment("Person 1", 0.0, 2.0, "Go ahead."),
        Segment("Person 2", 2.5, 4.0, "Thanks."),
    ]
    blocks = merge_segments_into_blocks(segments)

    assert all(not b.interrupting for b in blocks)


def test_segments_are_reordered_by_start_time_regardless_of_input_order() -> None:
    segments = [
        Segment("Person 2", 5.0, 6.0, "Second thing."),
        Segment("Person 1", 0.0, 1.0, "First thing."),
    ]
    blocks = merge_segments_into_blocks(segments)

    assert [b.text for b in blocks] == ["First thing.", "Second thing."]


def test_blank_segments_are_dropped() -> None:
    segments = [
        Segment("Person 1", 0.0, 1.0, "Real text."),
        Segment("Person 1", 1.0, 1.1, "   "),
    ]
    blocks = merge_segments_into_blocks(segments)

    assert len(blocks) == 1
    assert blocks[0].text == "Real text."


def test_format_timestamp_minutes_and_hours() -> None:
    assert format_timestamp(0) == "00:00"
    assert format_timestamp(65) == "01:05"
    assert format_timestamp(3725) == "01:02:05"


def test_render_single_speaker_falls_back_to_plain_paragraphs() -> None:
    segments = [
        Segment("Person 1", 0.0, 2.0, "First paragraph."),
        Segment("Person 1", 5.0, 6.0, "Second paragraph, after a real pause."),
    ]
    blocks = merge_segments_into_blocks(segments, pause_seconds=0.8)
    text = render_blocks_markdown(blocks)

    assert "Person 1" not in text
    assert "[00:00]" not in text
    assert text == "First paragraph.\n\nSecond paragraph, after a real pause.\n"


def test_render_multi_speaker_uses_dialogue_format_with_timestamps_and_interruptions() -> None:
    segments = [
        Segment("Person 1", 0.0, 5.0, "So I wanted to go over the proposal."),
        Segment("Person 2", 3.0, 4.0, "Actually, before that."),
    ]
    blocks = merge_segments_into_blocks(segments)
    text = render_blocks_markdown(blocks)

    assert "**Person 1** [00:00]" in text
    assert "**Person 2** [00:03] (interrupting)" in text
    assert text.index("Person 1") < text.index("Person 2")


def test_render_empty_blocks_is_empty_string() -> None:
    assert render_blocks_markdown([]) == ""


def test_realign_identity_when_polish_changed_nothing() -> None:
    words = [Word(0.0, 0.5, "Hey", True), Word(0.5, 1.0, "there.", True)]
    out = realign_words_to_text(words, "Hey there.")

    assert [(w.start, w.end, w.text) for w in out] == [(0.0, 0.5, "Hey"), (0.5, 1.0, "there.")]


def test_realign_drops_a_filler_word_without_orphaning_a_timestamp() -> None:
    words = [
        Word(0.0, 0.3, "Um,", True), Word(0.3, 0.6, "I", True),
        Word(0.6, 1.0, "think", True), Word(1.0, 1.4, "so.", True),
    ]
    out = realign_words_to_text(words, "I think so.")

    assert [w.text for w in out] == ["I", "think", "so."]
    assert [w.start for w in out] == [0.3, 0.6, 1.0]


def test_realign_keeps_old_timing_for_a_corrected_word() -> None:
    words = [Word(0.0, 0.5, "Hey", True), Word(0.5, 1.0, "tehre", True)]
    out = realign_words_to_text(words, "Hey there")

    assert [w.text for w in out] == ["Hey", "there"]
    assert [w.start for w in out] == [0.0, 0.5]
    assert out[1].end == 1.0


def test_realign_interpolates_a_word_polish_added() -> None:
    words = [Word(0.0, 1.0, "Hello", True), Word(4.0, 5.0, "world", True)]
    out = realign_words_to_text(words, "Hello my dear world")

    assert [w.text for w in out] == ["Hello", "my", "dear", "world"]
    assert out[0].start == 0.0
    assert out[-1].start == 4.0
    assert 1.0 <= out[1].start <= out[2].start <= 4.0


def test_realign_with_no_original_words_still_returns_every_token() -> None:
    out = realign_words_to_text([], "Brand new text")
    assert [w.text for w in out] == ["Brand", "new", "text"]


def test_realign_empty_text_returns_no_words() -> None:
    assert realign_words_to_text([Word(0.0, 1.0, "Hi", True)], "") == []


def test_blocks_to_dicts_serializes_words_and_shape() -> None:
    segments = [
        Segment("Person 1", 0.0, 1.0, "Hey there.", words=[Word(0.0, 1.0, "Hey there.", True)]),
        Segment("Person 2", 2.0, 3.0, "Hi!", words=[Word(2.0, 3.0, "Hi!", True)]),
    ]
    blocks = merge_segments_into_blocks(segments)
    dicts = blocks_to_dicts(blocks)

    assert dicts == [
        {
            "speaker": "Person 1", "start": 0.0, "end": 1.0, "interrupting": False,
            "text": "Hey there.",
            "words": [{"start": 0.0, "end": 1.0, "text": "Hey there.", "lead_space": True}],
        },
        {
            "speaker": "Person 2", "start": 2.0, "end": 3.0, "interrupting": False,
            "text": "Hi!",
            "words": [{"start": 2.0, "end": 3.0, "text": "Hi!", "lead_space": True}],
        },
    ]


def test_a_polished_jobs_turns_words_still_join_back_to_its_text() -> None:
    """Reproduces the exact sequence app.worker._process runs on the polish
    path: realign each block's words to the polished text BEFORE overwriting
    block.text, then serialize - this is the regression Phase 5
    (word-level-dialogue-blocks) exists to prevent: forgetting the realign
    step (or realigning against the wrong text) would leave job.turns with
    words that no longer match what the user actually reads."""
    segments = [Segment("Person 1", 0.0, 1.0, "hey uh there", words=[
        Word(0.0, 0.3, "hey", True), Word(0.3, 0.6, "uh", True), Word(0.6, 1.0, "there", True),
    ])]
    blocks = merge_segments_into_blocks(segments)
    polished_texts = ["Hey there."]  # "uh" dropped, punctuation added - a real polish edit

    for block, text in zip(blocks, polished_texts):
        block.words = realign_words_to_text(block.words, text)
        block.text = text

    turns = blocks_to_dicts(blocks)
    assert turns[0]["text"] == "Hey there."
    assert _join_words([Word(**w) for w in turns[0]["words"]]) == turns[0]["text"]
