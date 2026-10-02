from src import polish
from src.polish import (
    _accept,
    _chunk_by_lines,
    _containment_ok,
    _strip_fences,
    polish_blocks,
    polish_text,
)


# --- chunking ----------------------------------------------------------------

def test_chunk_by_lines_never_splits_a_line() -> None:
    text = "\n".join(["line one", "line two", "line three"])
    chunks = _chunk_by_lines(text, limit=15)
    assert all(all(ln in text for ln in c.splitlines()) for c in chunks)
    assert "".join(c.replace("\n", "") for c in chunks) == text.replace("\n", "")


# --- output cleanup ----------------------------------------------------------

def test_strip_fences_removes_json_code_block() -> None:
    assert _strip_fences('```json\n["a", "b"]\n```') == '["a", "b"]'
    assert _strip_fences("plain text") == "plain text"


# --- safety guards -----------------------------------------------------------

def test_accept_keeps_valid_cleanup_including_added_diacritics() -> None:
    raw = "chtel jsem se zeptat na tu fakturu prosim"
    polished = "Chtěl jsem se zeptat na tu fakturu, prosím."
    # diacritics are folded away, so the words still count as "already present"
    assert _accept(raw, polished) == polished


def test_accept_length_tripwire_falls_back_to_raw() -> None:
    raw = "krátký text"
    assert _accept(raw, "x" * (len(raw) * 5)) == raw


def test_accept_falls_back_on_empty_or_none() -> None:
    assert _accept("nějaký text tady", "") == "nějaký text tady"
    assert _accept("nějaký text tady", None) == "nějaký text tady"


def test_accept_containment_backstop_rejects_invented_text() -> None:
    raw = "chtel jsem se zeptat na fakturu"
    invented = "Dneska venku krásně svítí slunce a ptáci zpívají."  # mostly new words
    assert not _containment_ok(raw, invented)
    assert _accept(raw, invented) == raw


# --- block batching ----------------------------------------------------------

def test_polish_blocks_happy_path(monkeypatch) -> None:
    def fake(_prompt):
        return '["Dobrý den.", "Jak se dneska máte?"]'

    monkeypatch.setattr(polish, "_run_claude", fake)
    out = polish_blocks(["dobry den tady", "jak se dneska mate"])
    assert out == ["Dobrý den.", "Jak se dneska máte?"]


def test_polish_blocks_length_mismatch_keeps_raw(monkeypatch) -> None:
    monkeypatch.setattr(polish, "_run_claude", lambda _p: '["jen jeden"]')
    raw = ["dobry den tady", "jak se dneska mate"]
    assert polish_blocks(raw) == raw


def test_polish_blocks_invalid_json_keeps_raw(monkeypatch) -> None:
    monkeypatch.setattr(polish, "_run_claude", lambda _p: "to není JSON")
    raw = ["dobry den tady", "jak se dneska mate"]
    assert polish_blocks(raw) == raw


def test_polish_blocks_cli_unavailable_keeps_raw(monkeypatch) -> None:
    monkeypatch.setattr(polish, "_run_claude", lambda _p: None)
    raw = ["dobry den tady", "jak se dneska mate"]
    assert polish_blocks(raw) == raw


def test_polish_blocks_skips_cli_when_all_turns_are_short(monkeypatch) -> None:
    def _boom(_prompt):
        raise AssertionError("CLI must not run when nothing needs polishing")

    monkeypatch.setattr(polish, "_run_claude", _boom)
    assert polish_blocks(["Jo.", "Mhm", "Ne."]) == ["Jo.", "Mhm", "Ne."]


def test_polish_blocks_per_block_guard_falls_back(monkeypatch) -> None:
    # Second block's cleanup balloons -> that block keeps raw, first is accepted.
    def fake(_prompt):
        return '["Dobrý den.", "' + "x" * 200 + '"]'

    monkeypatch.setattr(polish, "_run_claude", fake)
    out = polish_blocks(["dobry den", "jak se dneska mate"])
    assert out[0] == "Dobrý den."
    assert out[1] == "jak se dneska mate"


# --- plain text --------------------------------------------------------------

def test_polish_text_happy_path(monkeypatch) -> None:
    monkeypatch.setattr(polish, "_run_claude", lambda _p: "Dobrý den, jak se máte?")
    assert polish_text("dobry den jak se mate") == "Dobrý den, jak se máte?"


def test_polish_text_short_input_skips_cli(monkeypatch) -> None:
    monkeypatch.setattr(
        polish, "_run_claude", lambda _p: (_ for _ in ()).throw(AssertionError("no CLI"))
    )
    assert polish_text("ok") == "ok"


def test_polish_text_bad_output_falls_back_to_raw(monkeypatch) -> None:
    monkeypatch.setattr(polish, "_run_claude", lambda _p: "x" * 500)
    raw = "tohle je rozumně dlouhý vstupní text k opravě"
    assert polish_text(raw) == raw
