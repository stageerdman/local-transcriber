"""Polishing-lab core: parse a transcript, run `claude -p`, measure drift.

This is an isolated research spike (it does NOT import or touch `src/`/`app/`).
It gives every polishing *strategy* the same primitives so results are
comparable:

- `parse_transcript` / `reassemble` — split the speaker-labelled `.md` into
  dialogue blocks (speaker, timestamp, body) and put them back with the EXACT
  original headers, so a strategy only ever rewrites block *bodies* and can
  never mangle labels/timestamps.
- `run_claude` — one headless `claude -p` call (optional `--model`, optional
  system prompt), content on stdin (no shell, no injection).
- metrics — reference-free drift signals (we have no gold transcript): length
  ratio, word-retention, novel-word count/list (the hallucination signal), and
  block-count integrity.

Every strategy returns a list of polished bodies the SAME length/order as the
input blocks, so judging and drift metrics line up block-by-block.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

_CLAUDE_FALLBACK = "/opt/homebrew/bin/claude"
_CALL_TIMEOUT = 300  # seconds per call (whole-transcript calls can be long)

_HEADER_RE = re.compile(
    r"^\*\*(?P<speaker>[^*]+)\*\*\s*\[(?P<ts>\d{1,2}:\d{2})\](?P<tail>.*)$"
)


@dataclass
class Block:
    """One dialogue turn: speaker label, timestamp, optional tail (e.g.
    "(interrupting)"), and the body text we actually clean."""

    speaker: str
    ts: str
    tail: str  # whatever followed the "[mm:ss]" on the header line, verbatim
    body: str

    @property
    def header(self) -> str:
        return f"**{self.speaker}** [{self.ts}]{self.tail}"


def parse_transcript(md_text: str) -> tuple[str, list[Block]]:
    """Return (frontmatter_block, blocks). `frontmatter_block` is the leading
    `---\n...\n---` verbatim (or "" if none)."""
    frontmatter = ""
    body = md_text
    if md_text.startswith("---"):
        end = md_text.find("\n---", 3)
        if end != -1:
            nl = md_text.find("\n", end + 1)
            frontmatter = md_text[: nl + 1 if nl != -1 else len(md_text)]
            body = md_text[nl + 1 :] if nl != -1 else ""

    blocks: list[Block] = []
    cur: Block | None = None
    body_lines: list[str] = []

    def flush() -> None:
        if cur is not None:
            cur.body = "\n".join(body_lines).strip()
            blocks.append(cur)

    for line in body.splitlines():
        m = _HEADER_RE.match(line.strip())
        if m:
            flush()
            cur = Block(
                speaker=m.group("speaker").strip(),
                ts=m.group("ts"),
                tail=m.group("tail"),
                body="",
            )
            body_lines = []
        elif cur is not None:
            body_lines.append(line)
    flush()
    return frontmatter, blocks


def reassemble(frontmatter: str, blocks: list[Block], bodies: list[str]) -> str:
    """Rebuild the markdown with original headers but new bodies."""
    if len(bodies) != len(blocks):
        raise ValueError(f"bodies {len(bodies)} != blocks {len(blocks)}")
    parts = [frontmatter.rstrip("\n"), ""] if frontmatter else []
    for blk, body in zip(blocks, bodies):
        parts.append(blk.header)
        parts.append(body.strip())
        parts.append("")
    return "\n".join(parts).strip() + "\n"


# --- claude -p plumbing ------------------------------------------------------


def claude_binary() -> str | None:
    return shutil.which("claude") or (
        _CLAUDE_FALLBACK if os.path.exists(_CLAUDE_FALLBACK) else None
    )


def run_claude(
    user: str,
    system: str | None = None,
    model: str | None = None,
    timeout: int = _CALL_TIMEOUT,
) -> str | None:
    """One headless `claude -p` call. Content on stdin. Returns stdout or None.

    Calls the bare binary (not the shell alias), so no `--plugin-dir`. `system`,
    when given, fully replaces the default Claude Code system prompt via
    `--system-prompt` (we want only the corrector persona, not the coding agent).
    """
    binary = claude_binary()
    if binary is None:
        return None
    cmd = [binary, "-p"]
    if model:
        cmd += ["--model", model]
    if system:
        cmd += ["--system-prompt", system]
    # Retry transient failures (brief network blips) a few times with backoff,
    # so a dropped wifi packet doesn't poison a whole config with raw fallback.
    for attempt in range(3):
        try:
            result = subprocess.run(
                cmd, input=user, capture_output=True, text=True, timeout=timeout
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            pass
        if attempt < 2:
            time.sleep(3 * (attempt + 1))
    return None


def strip_fences(text: str) -> str:
    """Drop a ```lang ... ``` wrapper the model may add."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


# --- drift metrics (reference-free) ------------------------------------------


def _fold(text: str) -> str:
    """Lowercase + strip diacritics, so adding Czech háčky/čárky back onto a
    word doesn't register as a changed/new word."""
    decomposed = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def _words(text: str, min_len: int = 1) -> list[str]:
    toks = ["".join(c for c in w if c.isalnum()) for w in _fold(text).split()]
    return [w for w in toks if len(w) >= min_len]


def length_ratio(raw: str, polished: str) -> float:
    return len(polished) / max(1, len(raw))


def word_retention(raw: str, polished: str) -> float:
    """Fraction of the polished text's words that already appear in raw
    (accent-folded). Low => the model invented words."""
    out = _words(polished)
    if not out:
        return 1.0
    raw_set = set(_words(raw))
    return sum(1 for w in out if w in raw_set) / len(out)


def novel_words(raw: str, polished: str, min_len: int = 3) -> list[str]:
    """Content words (len >= min_len) in polished but not in raw (folded).
    The headline hallucination signal: invented words the ASR never produced."""
    raw_set = set(_words(raw, min_len))
    seen: set[str] = set()
    out: list[str] = []
    for w in _words(polished, min_len):
        if w not in raw_set and w not in seen:
            seen.add(w)
            out.append(w)
    return out


def drift_report(raw_bodies: list[str], polished_bodies: list[str]) -> dict:
    """Aggregate drift across all blocks, plus whole-transcript figures."""
    raw_all = "\n".join(raw_bodies)
    pol_all = "\n".join(polished_bodies)
    per_block_novel = [
        len(novel_words(r, p)) for r, p in zip(raw_bodies, polished_bodies)
    ]
    return {
        "block_count_raw": len(raw_bodies),
        "block_count_polished": len(polished_bodies),
        "block_integrity": len(raw_bodies) == len(polished_bodies),
        "length_ratio": round(length_ratio(raw_all, pol_all), 3),
        "word_retention": round(word_retention(raw_all, pol_all), 3),
        "novel_words_total": len(novel_words(raw_all, pol_all)),
        "novel_words_per_block_max": max(per_block_novel) if per_block_novel else 0,
        "novel_words_sample": novel_words(raw_all, pol_all)[:40],
    }
