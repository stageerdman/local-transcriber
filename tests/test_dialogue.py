from src.dialogue import Segment, format_timestamp, merge_segments_into_blocks, render_blocks_markdown


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
