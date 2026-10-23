"""Reference-free rater for polished transcripts — faithfulness gate + quality.

Protocol (from the judging-research brief, see wiki.md §Judging):
- We have NO gold transcript, so everything is *source-grounded*: compare the
  polished candidate against the RAW ASR as the source of truth.
- Two isolated LLM passes, each prompted in Czech, output strict JSON:
    1. FAITHFULNESS pass (the gate): decompose the polished text into atomic
       claims, label each supported/unsupported/contradicted vs the raw, and
       list added content, changed meaning, and substantive omissions.
       -> hallucination_count = unsupported + contradicted.
    2. QUALITY pass: 1-5 scores for ASR-error correction, Czech fluency,
       readability/structure, and conciseness, plus a residual-ASR-error count.
  Faithfulness is kept OUT of the quality score so fluency can't mask a
  hallucination.
- Each pass is run N times (median for counts, mean for scores, report spread) —
  single LLM-judge runs are too unstable to trust (Rating Roulette).
- Deterministic cross-checks (number retention, novel-word rate) run too; they
  are family-agnostic, so they backstop the fact that our judge is also Claude
  (self-preference risk).

The judge model should be a STRONG frontier model, ideally a different family
than the generator — we can't get non-Claude via the CLI offline, so we lean on
the deterministic gate and multiple runs instead.
"""
from __future__ import annotations

import json
import re
import statistics
from concurrent.futures import ThreadPoolExecutor

import lab

JUDGE_MODEL_DEFAULT = "claude-sonnet-5"

# --- deterministic cross-checks ---------------------------------------------

_NUM_RE = re.compile(r"\d[\d\s.,]*\d|\d")


def _numbers(text: str) -> list[str]:
    """Digit sequences, normalised (spaces/commas/dots stripped). Catches
    fabricated/altered *numeric* prices, counts, dates written as digits."""
    out = []
    for m in _NUM_RE.findall(text):
        norm = re.sub(r"[\s.,]", "", m)
        if norm:
            out.append(norm)
    return out


def number_gate(raw: str, polished: str) -> list[str]:
    """Numeric tokens present in polished but absent from raw (a near-1:1
    fabrication signal for a sales call: invented prices/dates)."""
    raw_nums = set(_numbers(raw))
    return [n for n in _numbers(polished) if n not in raw_nums]


# --- LLM judge passes --------------------------------------------------------

_FAITH_SYSTEM = (
    "Jsi přísný kontrolor věrnosti přepisu. Dostaneš SUROVÝ přepis z rozpoznávání "
    "řeči (zdroj pravdy) a jeho UPRAVENOU verzi. Tvůj úkol je najít, kde se "
    "upravená verze odchýlila od zdroje: cokoli PŘIDANÉHO (nový obsah, fakta, "
    "čísla, jména), ZMĚNĚNÉHO VÝZNAMU (obrácená negace, jiné číslo/cena, jiný "
    "smysl) nebo VÝZNAMNĚ VYPUŠTĚNÉHO (zahozený věcný obsah; odstranění výplňkových "
    "slov jako 'ehm, no, prostě' NENÍ chyba).\n\n"
    "Oprava diakritiky, interpunkce, velkých písmen a zjevných přepisových "
    "překlepů NENÍ odchylka. Jde jen o obsah a význam.\n\n"
    "Postupuj takto: rozlož upravený text na jednotlivá tvrzení (atomic claims). "
    "U každého rozhodni: 'supported' (plyne ze zdroje), 'unsupported' (ve zdroji "
    "není) nebo 'contradicted' (odporuje zdroji).\n\n"
    "Vrať POUZE validní JSON tohoto tvaru, bez jakéhokoli dalšího textu:\n"
    "{\n"
    '  "added_content": [stručné řetězce přidaného obsahu],\n'
    '  "changed_meaning": [stručné řetězce změněného významu],\n'
    '  "substantive_omissions": [stručné řetězce vypuštěného věcného obsahu],\n'
    '  "n_claims_total": <int>,\n'
    '  "n_supported": <int>,\n'
    '  "n_unsupported": <int>,\n'
    '  "n_contradicted": <int>\n'
    "}\n"
)

_QUALITY_SYSTEM = (
    "Jsi hodnotitel čtivosti českého přepisu telefonního hovoru. Dostaneš surový "
    "přepis z rozpoznávání řeči a jeho upravenou verzi. Hodnoť POUZE kvalitu "
    "úpravy, NE věrnost obsahu (tu řeší jiný kontrolor).\n\n"
    "Oznámkuj 1–5 (5 = nejlepší) tyto nezávislé osy:\n"
    "- asr_correction: do jaké míry jsou opraveny zjevně špatně rozpoznaná / "
    "zkomolená slova na správná česká slova (podle kontextu).\n"
    "- fluency: gramatická a jazyková správnost češtiny.\n"
    "- readability: interpunkce, velká písmena, členění na věty/odstavce, čitelnost.\n"
    "- conciseness: odstranění výplňkových slov BEZ ztráty obsahu.\n\n"
    "Spočítej také residual_asr_errors: počet zjevně zkomolených/nesmyslných slov, "
    "která v upraveném textu ZŮSTALA neopravená.\n\n"
    "Vrať POUZE validní JSON, bez dalšího textu:\n"
    "{\n"
    '  "asr_correction": <1-5>,\n'
    '  "fluency": <1-5>,\n'
    '  "readability": <1-5>,\n'
    '  "conciseness": <1-5>,\n'
    '  "residual_asr_errors": <int>\n'
    "}\n"
)


def _aligned_payload(raw_bodies: list[str], polished_bodies: list[str]) -> str:
    """Raw vs polished, block-by-block, so the judge can compare in place."""
    rows = []
    for i, (r, p) in enumerate(zip(raw_bodies, polished_bodies), 1):
        rows.append(f"[{i}] SUROVÝ: {r}\n[{i}] UPRAVENÝ: {p}")
    return "\n\n".join(rows)


def _parse_json(reply: str | None) -> dict | None:
    if not reply:
        return None
    text = lab.strip_fences(reply)
    # tolerate leading/trailing prose: grab the outermost {...}
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start : end + 1]
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except (ValueError, TypeError):
        return None


def _judge_once(system: str, payload: str, model: str) -> dict | None:
    return _parse_json(lab.run_claude(payload, system=system, model=model))


def _run_n(system: str, payload: str, model: str, n: int) -> list[dict]:
    with ThreadPoolExecutor(max_workers=min(n, 6)) as ex:
        results = list(ex.map(lambda _: _judge_once(system, payload, model), range(n)))
    return [r for r in results if r is not None]


def _median(vals: list[float]) -> float | None:
    return round(statistics.median(vals), 2) if vals else None


def _mean(vals: list[float]) -> float | None:
    return round(statistics.mean(vals), 2) if vals else None


def judge_variant(
    raw_bodies: list[str],
    polished_bodies: list[str],
    model: str = JUDGE_MODEL_DEFAULT,
    n: int = 3,
) -> dict:
    """Run both passes N times and aggregate. Returns a scorecard dict."""
    payload = _aligned_payload(raw_bodies, polished_bodies)
    faith = _run_n(_FAITH_SYSTEM, payload, model, n)
    qual = _run_n(_QUALITY_SYSTEM, payload, model, n)

    def col(rows: list[dict], key: str) -> list[float]:
        return [float(r[key]) for r in rows if isinstance(r.get(key), (int, float))]

    halluc_counts = [
        float(r.get("n_unsupported", 0)) + float(r.get("n_contradicted", 0))
        for r in faith
    ]
    faith_ratios = [
        (float(r["n_supported"]) / float(r["n_claims_total"]))
        for r in faith
        if r.get("n_claims_total")
    ]

    def examples(key: str) -> list[str]:
        out: list[str] = []
        for r in faith[:2]:
            out.extend((r.get(key) or [])[:3])
        return out

    raw_all, pol_all = "\n".join(raw_bodies), "\n".join(polished_bodies)
    return {
        "judge_runs_faith": len(faith),
        "judge_runs_quality": len(qual),
        # --- faithfulness / hallucination (the gate) ---
        "hallucination_count_median": _median(halluc_counts),
        "hallucination_count_spread": [min(halluc_counts), max(halluc_counts)]
        if halluc_counts
        else None,
        "faithfulness_ratio_mean": _mean(faith_ratios),
        "added_content_examples": examples("added_content"),
        "changed_meaning_examples": examples("changed_meaning"),
        "omission_examples": examples("substantive_omissions"),
        # --- quality ---
        "asr_correction_mean": _mean(col(qual, "asr_correction")),
        "fluency_mean": _mean(col(qual, "fluency")),
        "readability_mean": _mean(col(qual, "readability")),
        "conciseness_mean": _mean(col(qual, "conciseness")),
        "residual_asr_errors_median": _median(col(qual, "residual_asr_errors")),
        # --- deterministic cross-checks (family-agnostic hard gate) ---
        "fabricated_numbers": number_gate(raw_all, pol_all),
        "novel_words_total": len(lab.novel_words(raw_all, pol_all)),
        "length_ratio": lab.length_ratio(raw_all, pol_all),
        "word_retention": round(lab.word_retention(raw_all, pol_all), 3),
    }


def composite_rank(scorecards: dict[str, dict]) -> list[tuple[str, float, bool]]:
    """Rank variants. Hard gate first: fabricated numbers or median
    hallucination_count > 0 => gated (can't outrank a clean variant). Among the
    rest, rank by faithfulness_ratio then mean quality.
    """
    def quality(sc: dict) -> float:
        vals = [
            sc.get(k)
            for k in ("asr_correction_mean", "fluency_mean", "readability_mean", "conciseness_mean")
            if sc.get(k) is not None
        ]
        return sum(vals) / len(vals) if vals else 0.0

    def gated(sc: dict) -> bool:
        return bool(sc.get("fabricated_numbers")) or (sc.get("hallucination_count_median") or 0) > 0

    ranked = []
    for name, sc in scorecards.items():
        ranked.append((name, sc, gated(sc)))
    ranked.sort(
        key=lambda t: (
            t[2],  # non-gated first
            -(t[1].get("faithfulness_ratio_mean") or 0),
            -quality(t[1]),
        )
    )
    return [(name, round(quality(sc), 2), g) for name, sc, g in ranked]
