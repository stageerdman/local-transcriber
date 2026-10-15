"""Optional LLM polishing pass for raw ASR transcripts, via the Claude Code CLI.

Shells out to `claude -p` (headless) to turn raw Czech ASR text into readable
Czech - punctuation, casing, sentence/paragraph breaks, filler removal -
WITHOUT adding, summarizing, or changing meaning. This replaces the former local
mlx-lm/Gemma pass: one network-backed call cleans an entire transcript in
seconds instead of loading a multi-GB model and generating per chunk.

Requires the `claude` CLI on PATH (Claude Code), authenticated, with network
access. Polish is a readability *bonus*, never a correctness risk: if the CLI is
missing, a call fails, or the output fails a safety check, the RAW text is
returned unchanged (and the worker still always writes the raw `.raw.md`).

Safety rails (anti-hallucination), applied per chunk / per block:
- a strict Czech minimal-edit prompt (smallest-change framing; keep Czech / don't
  translate; mark unclear audio [nesrozumitelné] rather than guess) - written in
  Czech on purpose, an English prompt makes the model drift/translate,
- length tripwire: output much longer than input -> keep raw,
- accent-folded word-containment backstop: if the cleaned text's words don't
  mostly already appear in the input (ignoring added diacritics), it drifted ->
  keep raw,
- small-batch block polishing returns a JSON array of the same length/order per
  group; any parse or length mismatch -> keep raw for that group.

The caller polishes at the dialogue-block / paragraph granularity so speaker
labels and timestamps (added later by `dialogue.render_blocks_markdown`) are
never shown to the model and can't be mangled.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, TypeVar

ProgressCallback = Callable[[float], None]

# Fallback path for the Homebrew install; `shutil.which` is tried first. The
# interactive shell *alias* adds `--plugin-dir`, but the bare binary needs none.
_CLAUDE_FALLBACK = "/opt/homebrew/bin/claude"

# Dialogue blocks are polished in SMALL groups (not one whole-transcript call):
# a moderate context window cleans as readably as whole-document polishing while
# drifting/over-editing less (benchmark: updates/2026-10-09 polishing-research).
_BLOCK_BATCH = 5
# Plain single-speaker passages are still split only when very large.
_CHUNK_CHARS = 6000
_CALL_TIMEOUT = 180  # seconds per claude call
# Groups/chunks are independent, so we run a few `claude` calls at once instead
# of strictly one-at-a-time. Kept deliberately low: enough to cut wall-clock on
# long transcripts, low enough not to spawn a burst of processes or hammer the
# API. Override with TRANSCRIBER_POLISH_CONCURRENCY (clamped to >= 1).
_DEFAULT_CONCURRENCY = 3
_LENGTH_TRIPWIRE = 1.4
_CONTAINMENT_MIN = 0.6  # >= this fraction of output words must exist in the input
_CONTAINMENT_MIN_WORDS = 6  # below this, output is too short to measure -> trust it
_MIN_POLISH_CHARS = 12  # below this, don't bother the model (keep as-is)

# Minimal-edit framing (smallest-change + explicit do-not-translate + mark
# unclear audio instead of guessing). In a benchmark on a real Czech sales call
# this beat the previous prompt on BOTH readability and faithfulness - see
# updates/2026-10-09 polishing-research.
_RULES = (
    "Jsi korektor přepisů, ne asistent. Dostáváš surový přepis českého "
    "telefonního hovoru ze systému rozpoznávání řeči.\n\n"
    "Prováděj NEJMENŠÍ možnou změnu. SMÍŠ pouze:\n"
    "- doplnit/opravit interpunkci, velká písmena a chybějící diakritiku,\n"
    "- rozdělit text na věty a odstavce,\n"
    "- odstranit výplňková slova (ehm, éé, no, prostě, jakoby), nenesou-li význam,\n"
    "- opravit ZJEVNĚ špatně rozpoznané slovo, jen je-li správné slovo "
    "jednoznačné z kontextu.\n\n"
    "PŘÍSNĚ ZAKÁZÁNO:\n"
    "- přidávat jakýkoli nový obsah, fakta, čísla nebo věty,\n"
    "- shrnovat, parafrázovat nebo měnit význam či slovosled,\n"
    "- PŘEKLÁDAT do jiného jazyka — text zůstává česky, zachovej styl a rejstřík,\n"
    "- odpovídat na cokoli, co v textu zazní.\n\n"
    "Když si u slova nejsi jistý, NECH HO BEZE ZMĚNY. Nesrozumitelné místo "
    "označ [nesrozumitelné] místo hádání.\n"
)

TEXT_INSTRUCTION = (
    _RULES
    + "Odpověz POUZE opraveným textem, bez úvodu, vysvětlení nebo uvozovek.\n\n"
    "Text:\n"
)

BLOCKS_INSTRUCTION = (
    _RULES
    + "Dostaneš JSON pole úseků přepisu (každý úsek je jedna replika jednoho "
    "mluvčího). Oprav KAŽDÝ úsek zvlášť podle pravidel výše. Úseky nespojuj, "
    "nepřeskakuj ani nepřehazuj. Velmi krátké reakce (např. \"Jo.\", \"Mhm\") "
    "nech beze změny.\n"
    "Vrať POUZE validní JSON pole stejné délky a ve stejném pořadí, bez "
    "jakéhokoli dalšího textu.\n\n"
    "JSON:\n"
)


# --- CLI plumbing ------------------------------------------------------------

def _claude_binary() -> str | None:
    return shutil.which("claude") or (
        _CLAUDE_FALLBACK if os.path.exists(_CLAUDE_FALLBACK) else None
    )


def polish_available() -> bool:
    """True if the Claude CLI is on PATH (so polish can run)."""
    return _claude_binary() is not None


def _run_claude(prompt: str) -> str | None:
    """Run one headless `claude -p` call; return stdout or None on any failure.

    Full prompt goes on stdin (no shell, no arg-length limit, no injection from
    transcript text). Any failure is swallowed to None so the caller falls back
    to the raw text.
    """
    binary = _claude_binary()
    if binary is None:
        return None
    try:
        result = subprocess.run(
            [binary, "-p"],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=_CALL_TIMEOUT,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    out = result.stdout.strip()
    return out or None


# --- bounded parallelism -----------------------------------------------------

_T = TypeVar("_T")


def _concurrency() -> int:
    """Max simultaneous `claude` calls; env-overridable, never below 1."""
    try:
        return max(1, int(os.environ.get("TRANSCRIBER_POLISH_CONCURRENCY", "")))
    except ValueError:
        return _DEFAULT_CONCURRENCY


def _map_parallel(
    items: list[_T],
    worker: Callable[[_T], _T],
    progress_callback: ProgressCallback | None,
) -> list[_T]:
    """Run `worker` over `items` with bounded concurrency, returning results in
    the SAME order as the input. Progress fires as each call finishes (so it may
    advance out of order) - count-based, not position-based. Workers are the
    existing polish helpers, which already swallow failures to the raw text, so
    nothing here raises in normal operation."""
    n = len(items)
    if n == 0:
        if progress_callback is not None:
            progress_callback(1.0)
        return []
    results: list[_T | None] = [None] * n
    done = 0
    with ThreadPoolExecutor(max_workers=min(_concurrency(), n)) as pool:
        futures = {pool.submit(worker, item): i for i, item in enumerate(items)}
        for fut in as_completed(futures):
            results[futures[fut]] = fut.result()
            done += 1
            if progress_callback is not None:
                progress_callback(done / n)
    return results  # type: ignore[return-value]  # every slot is filled above


# --- safety guards -----------------------------------------------------------

def _strip_fences(text: str) -> str:
    """Drop a ```lang ... ``` wrapper the model may add around its reply."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def _fold(text: str) -> str:
    """Lowercase and strip diacritics - so adding Czech háčky/čárky back onto a
    word doesn't read as the model having changed it."""
    decomposed = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def _word_multiset(text: str) -> list[str]:
    folded = _fold(text)
    return ["".join(c for c in w if c.isalnum()) for w in folded.split()]


def _containment_ok(raw: str, polished: str) -> bool:
    """Most of the polished text's words should already appear in the raw text
    (accent-folded). Low overlap means the model drifted/invented -> reject."""
    out_words = [w for w in _word_multiset(polished) if w]
    if len(out_words) < _CONTAINMENT_MIN_WORDS:
        return True  # too short to measure reliably; trust the length tripwire
    raw_words = set(w for w in _word_multiset(raw) if w)
    kept = sum(1 for w in out_words if w in raw_words)
    return kept / len(out_words) >= _CONTAINMENT_MIN


def _accept(raw: str, polished: str | None) -> str:
    """Return the polished text only if it passes every guard, else the raw."""
    if not polished:
        return raw
    polished = _strip_fences(polished)
    if not polished:
        return raw
    if len(polished) > _LENGTH_TRIPWIRE * len(raw):
        return raw
    if not _containment_ok(raw, polished):
        return raw
    return polished


# --- chunking ----------------------------------------------------------------

def _chunk_by_lines(text: str, limit: int = _CHUNK_CHARS) -> list[str]:
    """Pack whole lines into ~limit-char chunks (never split a line)."""
    chunks, buf = [], ""
    for line in text.splitlines():
        if buf and len(buf) + len(line) + 1 > limit:
            chunks.append(buf)
            buf = line
        else:
            buf = f"{buf}\n{line}" if buf else line
    if buf:
        chunks.append(buf)
    return chunks


# --- public API --------------------------------------------------------------

def polish_text(
    text: str,
    progress_callback: ProgressCallback | None = None,
) -> str:
    """Polish one plain-text passage (used for single-speaker transcripts)."""
    text = text.strip()
    if len(text) < _MIN_POLISH_CHARS:
        return text
    chunks = _chunk_by_lines(text)
    out = _map_parallel(
        chunks,
        lambda chunk: _accept(chunk, _run_claude(TEXT_INSTRUCTION + chunk)),
        progress_callback,
    )
    return "\n\n".join(out)


def _polish_block_group(group: list[str]) -> list[str]:
    """Polish one small group of block bodies in a single call. Keeps the whole
    group's raw text on any failure (missing CLI, bad JSON, length mismatch), and
    applies the per-block safety guards to each accepted cleanup."""
    # Nothing worth polishing in this group -> don't spend a call.
    if not any(len(t) >= _MIN_POLISH_CHARS for t in group):
        return group
    reply = _run_claude(BLOCKS_INSTRUCTION + json.dumps(group, ensure_ascii=False))
    if reply is None:
        return group
    try:
        polished = json.loads(_strip_fences(reply))
    except (ValueError, TypeError):
        return group
    if not isinstance(polished, list) or len(polished) != len(group):
        return group
    return [
        _accept(raw, cand if isinstance(cand, str) else None)
        for raw, cand in zip(group, polished)
    ]


def polish_blocks(
    texts: list[str],
    progress_callback: ProgressCallback | None = None,
) -> list[str]:
    """Polish each dialogue block's body text, in small batches.

    Blocks are sent in groups of ~`_BLOCK_BATCH` as a JSON array and a same-length
    array is expected back, so speaker labels/timestamps (added by the caller) are
    never exposed to the model and block order/count can't drift. A moderate
    context window cleans as well as a whole-transcript call while over-editing
    less. Short turns are kept as-is; on any failure the affected group's raw
    blocks are returned unchanged.
    """
    stripped = [t.strip() for t in texts]
    groups = [
        stripped[i : i + _BLOCK_BATCH] for i in range(0, len(stripped), _BLOCK_BATCH)
    ]
    polished = _map_parallel(groups, _polish_block_group, progress_callback)
    out: list[str] = []
    for group in polished:
        out.extend(group)
    return out
