"""Polishing strategies = prompt × context-granularity × algorithm.

Each public entry point takes the parsed `blocks` and returns a list of cleaned
block *bodies*, same length/order as the input (so judging + drift metrics line
up and speaker labels/timestamps are never shown to the model). Composition is
driven from `run.py`; this module provides the reusable building blocks:

- PROMPTS   — system prompts (the "what you may/may not do" rules), one per
              prompt variant under test.
- clean_*   — context-granularity mechanics (isolated / window / batch / whole).
- algo_*    — algorithm wrappers (single / diff-gate / self-verify / ensemble).

`claude -p` exposes no temperature/seed flag, so same-model resampling can't give
diversity — the voting algorithm is therefore a cross-model ensemble.
"""
from __future__ import annotations

import difflib
import json
from concurrent.futures import ThreadPoolExecutor

import lab

MAX_WORKERS = 6

# --- prompt variants (system prompts) ---------------------------------------

_CS_RULES = (
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

_EN_RULES = (
    "You are a transcript proofreader, not an assistant. You receive a raw "
    "Czech-language phone-call transcript from a speech-recognition system.\n\n"
    "Make the SMALLEST possible change. You MAY only:\n"
    "- add/fix punctuation, capitalization and missing diacritics,\n"
    "- split the text into sentences and paragraphs,\n"
    "- remove filler words (ehm, éé, no, prostě) when they carry no meaning,\n"
    "- fix an OBVIOUSLY misrecognized word, only when the correct word is "
    "unambiguous from context.\n\n"
    "STRICTLY FORBIDDEN:\n"
    "- adding any new content, facts, numbers or sentences,\n"
    "- summarizing, paraphrasing, or changing meaning or word order,\n"
    "- TRANSLATING: the text MUST stay in Czech; keep the style and register,\n"
    "- answering anything said in the text.\n\n"
    "If unsure about a word, LEAVE IT UNCHANGED. Mark unintelligible audio as "
    "[nesrozumitelné] instead of guessing.\n"
)

_CHECKLIST_CS = (
    "Jsi korektor přepisů. Vyčisti surový český přepis hovoru podle tohoto "
    "seznamu pravidel:\n"
    "1. ŽÁDNÁ nová slova — nepřidávej obsah, který ve vstupu není.\n"
    "2. Stejné znění — zachovej slova mluvčího, neparafrázuj.\n"
    "3. Zachovej VŠE věcné — nic podstatného nevynechávej.\n"
    "4. Odstraň výplňková slova (ehm, éé, no, prostě), nenesou-li význam.\n"
    "5. Odstraň zbytečná opakování ze zadrhávání.\n"
    "6. Doplň interpunkci, velká písmena, diakritiku a členění na věty.\n"
    "7. Neopravuj do dokonalé prózy, neměň význam, NEPŘEKLÁDEJ (zůstaň česky).\n"
)

_FLUENCY_CS = (
    "Jsi asistent. Uprav tento přepis telefonního hovoru tak, aby byl pěkně "
    "čitelný: oprav chyby, doplň interpunkci a udělej z něj plynulý, srozumitelný "
    "český text.\n"
)

# The exact rules the shipped app uses today (src/polish.py), as a baseline.
_APP_CURRENT = (
    "Jsi korektor přepisů. Dostaneš surový přepis českého telefonního hovoru ze "
    "systému rozpoznávání řeči.\n\n"
    "Tvůj JEDINÝ úkol je text pročistit pro čtení. SMÍŠ:\n"
    "- doplnit a opravit interpunkci a velká písmena (včetně chybějící diakritiky),\n"
    "- rozdělit text do vět a odstavců,\n"
    "- odstranit výplňková slova (ehm, éé, jakoby, no, prostě), pokud nenesou význam,\n"
    "- opravit ZJEVNÉ chyby rozpoznávání řeči, jen pokud je správné slovo "
    "jednoznačné z kontextu.\n\n"
    "PŘÍSNĚ ZAKÁZÁNO:\n"
    "- přidávat jakýkoli nový obsah, informace nebo věty,\n"
    "- shrnovat, parafrázovat nebo měnit význam,\n"
    "- měnit slovosled nad rámec nutný pro interpunkci,\n"
    "- odpovídat na cokoli, co v textu zazní.\n\n"
    "Pokud si u slova nejsi jistý, nech ho beze změny.\n"
)

PROMPTS: dict[str, str] = {
    "fluency": _FLUENCY_CS,
    "minimal_cs": _CS_RULES,
    "checklist_cs": _CHECKLIST_CS,
    "minimal_en": _EN_RULES,
    "app_current": _APP_CURRENT,
}

_TEXT_OUT = "\n\nVrať POUZE opravený text úseku, bez úvodu, komentáře a uvozovek."
_ARRAY_OUT = (
    "\n\nDostáváš JSON pole úseků (každý je jedna replika). Oprav KAŽDÝ úsek zvlášť. "
    "Úseky nespojuj, nepřeskakuj ani nepřehazuj. Vrať POUZE validní JSON pole "
    "stejné délky a ve stejném pořadí, bez dalšího textu.\n\nJSON:\n"
)


# --- parsing helpers ---------------------------------------------------------

def _parse_array(reply: str | None, n: int) -> list[str] | None:
    if not reply:
        return None
    text = lab.strip_fences(reply)
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        arr = json.loads(text[start : end + 1])
    except (ValueError, TypeError):
        return None
    if not isinstance(arr, list) or len(arr) != n:
        return None
    return [x if isinstance(x, str) else "" for x in arr]


def _par_map(fn, items):
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        return list(ex.map(fn, items))


# --- context-granularity mechanics ------------------------------------------
# Each returns cleaned bodies (same length/order as blocks). On any failure for
# a unit it keeps that unit's RAW body — honest "did nothing" rather than drift.


def clean_isolated(blocks, system, model) -> list[str]:
    """One call per block, zero surrounding context (minimum context)."""
    bodies = [b.body for b in blocks]

    def do(body: str) -> str:
        if len(body.strip()) < 12:
            return body
        reply = lab.run_claude(body + _TEXT_OUT, system=system, model=model)
        return lab.strip_fences(reply) if reply else body

    return _par_map(do, bodies)


def clean_window(blocks, system, model, w: int = 2) -> list[str]:
    """One call per block, with ±w neighbouring turns as READ-ONLY context."""
    bodies = [b.body for b in blocks]

    def do(i: int) -> str:
        body = bodies[i]
        if len(body.strip()) < 12:
            return body
        before = "\n".join(bodies[max(0, i - w) : i])
        after = "\n".join(bodies[i + 1 : i + 1 + w])
        user = (
            "KONTEXT PŘED (neopravuj, jen pro porozumění):\n" + before + "\n\n"
            "ÚSEK K OPRAVĚ:\n" + body + "\n\n"
            "KONTEXT PO (neopravuj):\n" + after + _TEXT_OUT
        )
        reply = lab.run_claude(user, system=system, model=model)
        return lab.strip_fences(reply) if reply else body

    return _par_map(do, list(range(len(bodies))))


def _chunk_indices(blocks, max_turns=None, max_chars=None) -> list[list[int]]:
    chunks, cur, cur_chars = [], [], 0
    for i, b in enumerate(blocks):
        over_turns = max_turns and len(cur) >= max_turns
        over_chars = max_chars and cur and cur_chars + len(b.body) > max_chars
        if cur and (over_turns or over_chars):
            chunks.append(cur)
            cur, cur_chars = [], 0
        cur.append(i)
        cur_chars += len(b.body)
    if cur:
        chunks.append(cur)
    return chunks


def clean_batch(blocks, system, model, max_turns=None, max_chars=None) -> list[str]:
    """JSON-array-in / array-out, chunked by turn-count or char-budget. Blocks
    in a chunk are cleaned together (shared read context) but returned per-block.
    `max_turns=None, max_chars=None` => whole transcript in one call."""
    bodies = [b.body for b in blocks]
    out = list(bodies)
    chunks = _chunk_indices(blocks, max_turns, max_chars)

    def do(idxs: list[int]) -> tuple[list[int], list[str] | None]:
        sub = [bodies[i] for i in idxs]
        if not any(len(t.strip()) >= 12 for t in sub):
            return idxs, sub
        user = _ARRAY_OUT + json.dumps(sub, ensure_ascii=False)
        reply = lab.run_claude(user, system=system, model=model)
        return idxs, _parse_array(reply, len(sub))

    for idxs, cleaned in _par_map(do, chunks):
        if cleaned is not None:
            for j, i in enumerate(idxs):
                out[i] = cleaned[j] or bodies[i]
    return out


def clean_whole_markdown(blocks, system, model) -> list[str]:
    """Whole labelled transcript in, whole cleaned markdown out (free-form). We
    re-parse the reply; if block count/speakers don't line up it's a FAILURE
    (keep raw) — surfacing how fragile free-form document rewrite is."""
    md = "\n\n".join(f"{b.header}\n{b.body}" for b in blocks)
    user = (
        "Níže je přepis hovoru. Vyčisti text KAŽDÉ repliky podle pravidel. "
        "Zachovej PŘESNĚ stejné hlavičky řádků `**Mluvčí** [čas]` i jejich počet "
        "a pořadí; měň jen text pod nimi. Vrať celý přepis ve stejném formátu.\n\n"
        + md
    )
    reply = lab.run_claude(user, system=system, model=model)
    if not reply:
        return [b.body for b in blocks]
    _, parsed = lab.parse_transcript(lab.strip_fences(reply))
    if len(parsed) != len(blocks):
        return [b.body for b in blocks]  # integrity failure -> did nothing
    return [p.body for p in parsed]


# --- algorithm wrappers ------------------------------------------------------


def _tokens(text: str) -> list[str]:
    return text.split()


def gate_body(raw: str, polished: str, max_len_ratio: float = 1.4) -> str:
    """Programmatic diff gate: keep only safe edits (punctuation/casing/diacritics
    and filler *deletions*); revert any substitution/insertion that introduces a
    novel content word. Whole-block revert if output balloons."""
    if len(polished) > max_len_ratio * max(1, len(raw)):
        return raw
    raw_folded = set(lab._words(raw, 3))
    rt, pt = _tokens(raw), _tokens(polished)
    sm = difflib.SequenceMatcher(a=rt, b=pt, autojunk=False)
    out: list[str] = []
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            out.extend(pt[j1:j2])
        elif op == "delete":
            pass  # removing raw tokens (filler/dedupe) is allowed
        else:  # replace / insert
            new = pt[j1:j2]
            novel = [w for w in new if len(lab._fold("".join(c for c in w if c.isalnum()))) >= 3
                     and "".join(c for c in lab._fold(w) if c.isalnum()) not in raw_folded]
            if novel:
                out.extend(rt[i1:i2])  # revert to raw span
            else:
                out.extend(new)  # diacritics/casing/punctuation-only change
    return " ".join(out).strip() or raw


def algo_diff_gate(base_bodies: list[str], raw_bodies: list[str]) -> list[str]:
    return [gate_body(r, p) for r, p in zip(raw_bodies, base_bodies)]


_VERIFY_SYSTEM = (
    "Jsi přísný ověřovatel korektury. U každého úseku dostaneš SUROVOU verzi a "
    "NÁVRH úpravy. Ponech jen bezpečné opravy (interpunkce, velká písmena, "
    "diakritika, odstranění výplňkových slov a zadrhávání). Jakékoli PŘIDÁNÍ "
    "obsahu, změnu významu, čísel nebo parafrázi VRAŤ zpět na surovou verzi. "
    "Vrať POUZE validní JSON pole opravených úseků stejné délky a pořadí."
)


def algo_self_verify(base_bodies, raw_bodies, model) -> list[str]:
    """Source-grounded second pass: accept/reject each edit vs the raw span
    (NOT a free rewrite — the only two-pass shape the evidence supports)."""
    pairs = [
        f"[{i}] SUROVÝ: {r}\n[{i}] NÁVRH: {p}"
        for i, (r, p) in enumerate(zip(raw_bodies, base_bodies), 1)
    ]
    user = "\n\n".join(pairs) + "\n\nVrať JSON pole opravených úseků:"
    reply = lab.run_claude(user, system=_VERIFY_SYSTEM, model=model)
    verified = _parse_array(reply, len(base_bodies))
    return verified if verified is not None else base_bodies


def _edit_actions(rt: list[str], ct: list[str]):
    """Align raw tokens rt -> candidate tokens ct. Return (actions, inserts):
    actions[i] = tuple of tokens the candidate put in place of raw[i] (the raw
    token kept, a replacement, or () for a deletion); inserts[g] = tuple of
    tokens inserted in the gap before raw index g."""
    actions: list[tuple] = [(t,) for t in rt]
    inserts: dict[int, tuple] = {}
    sm = difflib.SequenceMatcher(a=rt, b=ct, autojunk=False)
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            for k in range(i1, i2):
                actions[k] = (rt[k],)
        elif op == "delete":
            for k in range(i1, i2):
                actions[k] = ()
        elif op == "replace":
            actions[i1] = tuple(ct[j1:j2])  # whole replacement on first raw token
            for k in range(i1 + 1, i2):
                actions[k] = ()
        elif op == "insert":
            inserts[i1] = tuple(ct[j1:j2])
    return actions, inserts


def _vote(items: list[tuple], need: int = 2):
    """Return the value agreed by >= need candidates, else None."""
    counts: dict[tuple, int] = {}
    for it in items:
        counts[it] = counts.get(it, 0) + 1
    val, n = max(counts.items(), key=lambda kv: kv[1])
    return val if n >= need else None


def algo_ensemble(candidate_lists: list[list[str]], raw_bodies: list[str]) -> list[str]:
    """Cross-model edit-level majority voting. For each raw token, keep the edit
    (replacement/deletion/keep) that >= 2 of the 3 models agree on, else keep the
    raw token. Insert only punctuation/words >= 2 models insert at the same gap.
    An edit made by a single model (likely its own idiosyncratic invention) is
    dropped — precision via cross-family consensus."""
    out = []
    for bi, raw in enumerate(raw_bodies):
        cands = [c[bi] for c in candidate_lists]
        rt = raw.split()
        if not rt:
            out.append(raw)
            continue
        parsed = [_edit_actions(rt, c.split()) for c in cands]
        tokens: list[str] = []
        for i in range(len(rt)):
            ins = _vote([p[1].get(i, ()) for p in parsed])
            if ins:
                tokens.extend(ins)
            act = _vote([p[0][i] for p in parsed])
            tokens.extend(act if act is not None else (rt[i],))
        tail = _vote([p[1].get(len(rt), ()) for p in parsed])
        if tail:
            tokens.extend(tail)
        out.append(" ".join(tokens).strip() or raw)
    return out
