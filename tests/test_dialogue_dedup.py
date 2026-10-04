"""Cross-speaker echo (mic-bleed) removal — dialogue.drop_cross_speaker_echoes.

The failure this guards against: one person's voice leaks onto a second track,
so the transcriber emits the same utterance twice under two speakers a second
apart. The louder copy (on the speaker's own mic) is the real one; the quiet
echo is dropped.
"""
from src.dialogue import (
    Segment,
    drop_cross_speaker_echoes,
    flag_interruptions,
    merge_segments_into_blocks,
)


def _texts(segments):
    return [s.text for s in segments]


def test_quiet_cross_speaker_echo_is_dropped_louder_copy_kept() -> None:
    segments = [
        Segment("Klient", 10.0, 14.0, "Já se omlouvám, já jsem to nečekal.", energy=0.20),
        Segment("Obchodnik", 10.3, 14.5, "Já se omlouvám, já jsem to nečekal.", energy=0.04),
    ]
    kept = drop_cross_speaker_echoes(segments)
    assert len(kept) == 1
    assert kept[0].speaker == "Klient"  # the loud original


def test_energy_outranks_text_length_when_choosing_the_real_copy() -> None:
    # The bleed copy is often LONGER/garbled (two turns run together); energy,
    # not length, must decide which is the real speaker.
    segments = [
        Segment("Klient", 27.0, 31.0, "A ona mi rikala.", energy=0.25),
        Segment("Obchodnik", 26.5, 32.0, "A ona mi rikala a jeste neco navic.", energy=0.03),
    ]
    kept = drop_cross_speaker_echoes(segments)
    assert [s.speaker for s in kept] == ["Klient"]


def test_distinct_simultaneous_speech_is_kept() -> None:
    segments = [
        Segment("Person 1", 0.0, 5.0, "So I wanted to go over the proposal.", energy=0.2),
        Segment("Person 2", 3.0, 4.0, "Actually, hold on a second there.", energy=0.2),
    ]
    kept = drop_cross_speaker_echoes(segments)
    assert len(kept) == 2


def test_short_overlapping_backchannels_are_not_collapsed() -> None:
    # "jo" / "ano" spoken over each other are real, not bleed.
    segments = [
        Segment("Person 1", 1.0, 1.4, "jo jo", energy=0.2),
        Segment("Person 2", 1.1, 1.5, "jo jo", energy=0.05),
    ]
    kept = drop_cross_speaker_echoes(segments)
    assert len(kept) == 2


def test_same_speaker_repeats_are_never_dropped() -> None:
    segments = [
        Segment("Person 1", 0.0, 3.0, "This is a repeated sentence here.", energy=0.2),
        Segment("Person 1", 3.2, 6.0, "This is a repeated sentence here.", energy=0.2),
    ]
    kept = drop_cross_speaker_echoes(segments)
    assert len(kept) == 2


def test_similar_text_far_apart_in_time_is_not_an_echo() -> None:
    segments = [
        Segment("Person 1", 0.0, 4.0, "Thanks very much for your time today.", energy=0.2),
        Segment("Person 2", 400.0, 404.0, "Thanks very much for your time today.", energy=0.2),
    ]
    kept = drop_cross_speaker_echoes(segments)
    assert len(kept) == 2


def test_one_loud_original_survives_multiple_faint_copies() -> None:
    segments = [
        Segment("A", 10.0, 14.0, "The quarterly numbers look strong this time.", energy=0.30),
        Segment("B", 10.2, 14.2, "The quarterly numbers look strong this time.", energy=0.05),
        Segment("C", 10.4, 14.4, "The quarterly numbers look strong this time.", energy=0.04),
    ]
    kept = drop_cross_speaker_echoes(segments)
    assert [s.speaker for s in kept] == ["A"]


def test_fallback_to_length_when_energy_unknown() -> None:
    # No energy: keep the longer (more complete) copy, deterministic.
    segments = [
        Segment("Person 1", 10.0, 14.0, "We should talk about the budget.", energy=None),
        Segment("Person 2", 10.3, 14.3, "We should talk about the budget today too.", energy=None),
    ]
    kept = drop_cross_speaker_echoes(segments)
    assert len(kept) == 1
    assert kept[0].speaker == "Person 2"


def test_merge_then_dedup_removes_the_bled_speaker_across_interleaved_turns() -> None:
    # The real failure: one voice lands on both tracks, so after merge the output
    # alternates loud-original / quiet-echo turn by turn. Every quiet echo turn
    # must be dropped, leaving only the real speaker's content.
    loud, quiet = "Klient", "Obchodnik"
    segments = [
        Segment(loud, 0.0, 4.0, "Takze mel jsem ruzny zraneni za sebou.", energy=0.22),
        Segment(quiet, 0.3, 4.3, "Takze mel jsem ruzny zraneni za sebou.", energy=0.04),
        Segment(loud, 5.0, 9.0, "A pak jsem skoncil na operaci pateře.", energy=0.21),
        Segment(quiet, 5.3, 9.3, "A pak jsem skoncil na operaci pateře.", energy=0.03),
    ]
    blocks = merge_segments_into_blocks(segments)
    kept = drop_cross_speaker_echoes(blocks)
    assert {b.speaker for b in kept} == {loud}  # no echo turns survive
    assert len(kept) == 2


def test_flag_interruptions_recomputes_after_echo_removal() -> None:
    # A quiet echo that overlapped a real turn falsely marks it interrupting;
    # once the echo is dropped and flags are recomputed, nothing is interrupting.
    segments = [
        Segment("A", 0.0, 5.0, "Let me walk through the whole plan in detail.", energy=0.25),
        Segment("B", 0.4, 5.2, "Let me walk through the whole plan in detail.", energy=0.03),
        Segment("A", 6.0, 9.0, "And here is the second part of it.", energy=0.25),
    ]
    blocks = merge_segments_into_blocks(segments)
    kept = drop_cross_speaker_echoes(blocks)
    flag_interruptions(kept)
    assert all(not b.interrupting for b in kept)


def test_no_echoes_leaves_everything_untouched() -> None:
    segments = [
        Segment("Person 1", 0.0, 2.0, "First distinct point about pricing.", energy=0.2),
        Segment("Person 2", 3.0, 5.0, "A completely different reply about timing.", energy=0.2),
    ]
    kept = drop_cross_speaker_echoes(segments)
    assert len(kept) == 2
