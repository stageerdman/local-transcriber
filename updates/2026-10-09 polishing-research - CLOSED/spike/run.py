"""Orchestrate the polishing tournament, then the full-transcript final.

Design (see wiki.md §3): a greedy staged sweep where each experiment sweeps ONE
axis and holds the others at the evidence-predicted optimum (small-batch
granularity · minimal-edit Czech prompt · Sonnet · single-pass). Runs the broad
tournament on a representative EXCERPT (per-block strategies cost one call each,
and the CLI has a ~16 s/call floor), then runs the winner + the current-app
baseline on the FULL transcript with N=5 judging.

Everything is cached to results/ (outputs + scorecards keyed by config name), so
the run is resumable and inspectable mid-flight.

Usage:
    python run.py tournament      # excerpt sweep -> results/scorecard.md
    python run.py final <name>    # one config on the full transcript, N=5
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import judge
import lab
import strategies as S

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "marek_p2_raw.md"
RES = ROOT / "results"
OUT = RES / "outputs"
SCO = RES / "scores"
for d in (OUT, SCO):
    d.mkdir(parents=True, exist_ok=True)

HOLD_MODEL = "sonnet"
ENSEMBLE_MODELS = ["claude-haiku-4-5-20251001", "sonnet", "opus"]
TOURNAMENT_JUDGE = "sonnet"  # fast; relative ranking + deterministic gate. Opus for the final.
TOURNAMENT_N = 3


def load_blocks():
    fm, blocks = lab.parse_transcript(RAW.read_text(encoding="utf-8"))
    return fm, blocks


def excerpt(blocks):
    """Contiguous representative slice that INCLUDES the longest (pitch) block
    plus ~20 preceding turns of varied length (so window context is valid)."""
    n = len(blocks)
    longest = max(range(n), key=lambda i: len(blocks[i].body))
    start = max(0, longest - 22)
    end = min(n, longest + 2)
    return blocks[start:end]


# --- config -> cleaned bodies (cached) --------------------------------------

def _base_single(blocks, granularity, prompt, model):
    """One single-pass clean for a (granularity, prompt, model) triple."""
    system = S.PROMPTS[prompt]
    if granularity == "isolated":
        return S.clean_isolated(blocks, system, model)
    if granularity == "window":
        return S.clean_window(blocks, system, model, w=2)
    if granularity == "batch_small":
        return S.clean_batch(blocks, system, model, max_turns=5)
    if granularity == "batch_large":
        return S.clean_batch(blocks, system, model, max_chars=1500)
    if granularity == "whole_array":
        return S.clean_batch(blocks, system, model)
    if granularity == "whole_md":
        return S.clean_whole_markdown(blocks, system, model)
    raise ValueError(granularity)


def _looks_failed(raw_bodies, bodies) -> bool:
    """A long raw block left byte-identical means that chunk's call failed
    (network blip) and fell back to raw — don't trust/cache such a result."""
    return any(
        len(r) > 120 and b == r for r, b in zip(raw_bodies, bodies)
    )


def produce(cfg, blocks) -> list[str]:
    """Return cleaned bodies for a config dict, caching to results/outputs.
    Refuses to cache a result that shows signs of a mid-run network failure."""
    cache = OUT / f"{cfg['name']}.json"
    if cache.exists():
        return json.loads(cache.read_text())["bodies"]

    raw_bodies = [b.body for b in blocks]
    algo = cfg.get("algo", "single")
    gran, prompt, model = cfg["granularity"], cfg["prompt"], cfg.get("model", HOLD_MODEL)

    if algo == "single":
        bodies = _base_single(blocks, gran, prompt, model)
    elif algo == "diff_gate":
        base = _base_single(blocks, gran, prompt, model)
        bodies = S.algo_diff_gate(base, raw_bodies)
    elif algo == "self_verify":
        base = _base_single(blocks, gran, prompt, model)
        bodies = S.algo_self_verify(base, raw_bodies, model)
    elif algo == "ensemble":
        cands = [_base_single(blocks, gran, prompt, m) for m in ENSEMBLE_MODELS]
        bodies = S.algo_ensemble(cands, raw_bodies)
    else:
        raise ValueError(algo)

    if not _looks_failed(raw_bodies, bodies):
        cache.write_text(json.dumps({"cfg": cfg, "bodies": bodies}, ensure_ascii=False, indent=1))
    return bodies


def score(cfg, blocks, bodies, judge_model, n) -> dict:
    cache = SCO / f"{cfg['name']}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    raw_bodies = [b.body for b in blocks]
    sc = judge.judge_variant(raw_bodies, bodies, model=judge_model, n=n)
    sc["cfg"] = cfg
    sc.update(lab.drift_report(raw_bodies, bodies))
    # Only cache a fully-successful judging; a short run means calls failed.
    if sc.get("judge_runs_faith", 0) >= n and sc.get("judge_runs_quality", 0) >= n:
        cache.write_text(json.dumps(sc, ensure_ascii=False, indent=1))
    return sc


# --- experiment matrix -------------------------------------------------------

def tournament_configs() -> list[dict]:
    cfgs: list[dict] = []
    # Exp 1 — context granularity (prompt=minimal_cs, algo=single)
    for g in ("isolated", "window", "batch_small", "batch_large", "whole_array", "whole_md"):
        cfgs.append({"name": f"g_{g}", "exp": "granularity",
                     "granularity": g, "prompt": "minimal_cs", "algo": "single"})
    # Exp 2 — prompt (granularity=batch_small, algo=single)
    for p in ("fluency", "minimal_cs", "checklist_cs", "minimal_en", "app_current"):
        cfgs.append({"name": f"p_{p}", "exp": "prompt",
                     "granularity": "batch_small", "prompt": p, "algo": "single"})
    # Exp 3 — algorithm (granularity=batch_small, prompt=minimal_cs)
    for a in ("single", "diff_gate", "self_verify", "ensemble"):
        cfgs.append({"name": f"a_{a}", "exp": "algorithm",
                     "granularity": "batch_small", "prompt": "minimal_cs", "algo": a})
    # Exp 4 — model (granularity=batch_small, prompt=minimal_cs, algo=diff_gate)
    for m, alias in (("haiku", "claude-haiku-4-5-20251001"), ("sonnet", "sonnet"), ("opus", "opus")):
        cfgs.append({"name": f"m_{m}", "exp": "model", "granularity": "batch_small",
                     "prompt": "minimal_cs", "algo": "diff_gate", "model": alias})
    # de-dup configs that are byte-identical (same cache key would collide anyway)
    seen, uniq = set(), []
    for c in cfgs:
        key = (c["granularity"], c["prompt"], c["algo"], c.get("model", HOLD_MODEL))
        if key in seen:
            continue
        seen.add(key)
        uniq.append(c)
    return uniq


def write_scorecard(scorecards: dict[str, dict], path: Path, title: str):
    ranking = judge.composite_rank(scorecards)
    lines = [f"# {title}", "",
             "Hard gate: a fabricated number or median hallucination>0 disqualifies "
             "(⛔), no matter how fluent.", "",
             "| rank | config | exp | gate | halluc(med)[spread] | faith | ASRfix | "
             "fluency | read | concise | resid.ASR | len | w.ret | novelW | fab.# |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for rank, (name, qual, gated) in enumerate(ranking, 1):
        sc = scorecards[name]
        cfg = sc.get("cfg", {})
        spread = sc.get("hallucination_count_spread")
        lines.append(
            f"| {rank} | {name} | {cfg.get('exp','')} | {'⛔' if gated else '✅'} | "
            f"{sc.get('hallucination_count_median')}{spread or ''} | "
            f"{sc.get('faithfulness_ratio_mean')} | {sc.get('asr_correction_mean')} | "
            f"{sc.get('fluency_mean')} | {sc.get('readability_mean')} | "
            f"{sc.get('conciseness_mean')} | {sc.get('residual_asr_errors_median')} | "
            f"{sc.get('length_ratio')} | {sc.get('word_retention')} | "
            f"{sc.get('novel_words_total')} | {len(sc.get('fabricated_numbers') or [])} |")
    path.write_text("\n".join(lines) + "\n")
    return ranking


def run_tournament():
    fm, blocks = load_blocks()
    ex = excerpt(blocks)
    print(f"excerpt: {len(ex)} blocks (of {len(blocks)}); "
          f"longest body {max(len(b.body) for b in ex)} chars", flush=True)
    configs = tournament_configs()
    scorecards = {}
    for cfg in configs:
        print(f"[produce] {cfg['name']} ...", flush=True)
        bodies = produce(cfg, ex)
        (OUT / f"{cfg['name']}.md").write_text(lab.reassemble(fm, ex, bodies))
        print(f"[judge]   {cfg['name']} ...", flush=True)
        scorecards[cfg["name"]] = score(cfg, ex, bodies, TOURNAMENT_JUDGE, TOURNAMENT_N)
    ranking = write_scorecard(scorecards, RES / "scorecard.md", "Polishing tournament (excerpt)")
    print("\n=== RANKING (non-gated first) ===", flush=True)
    for rank, (name, qual, gated) in enumerate(ranking, 1):
        print(f"  {rank:2d}. {'⛔' if gated else '✅'} {name:16s} quality={qual}", flush=True)


# Finalists to confirm on the FULL transcript (chosen from the tournament's
# deterministic Pareto frontier + the current-app baseline).
FINAL_CONFIGS = ["a_ensemble", "g_whole_md", "g_batch_small", "p_app_current"]
FINAL_JUDGE = "opus"
FINAL_N = 3


def run_finals():
    fm, blocks = load_blocks()
    print(f"[FINALS] full transcript, {len(blocks)} blocks", flush=True)
    scorecards = {}
    for base in FINAL_CONFIGS:
        src = next((c for c in tournament_configs() if c["name"] == base), None)
        cfg = {**src, "name": f"F_{base}"}
        print(f"[produce] {cfg['name']} ...", flush=True)
        bodies = produce(cfg, blocks)
        (OUT / f"{cfg['name']}.md").write_text(lab.reassemble(fm, blocks, bodies))
        print(f"[judge]   {cfg['name']} ...", flush=True)
        scorecards[cfg["name"]] = score(cfg, blocks, bodies, FINAL_JUDGE, FINAL_N)
    write_scorecard(scorecards, RES / "scorecard_final.md",
                    "Full-transcript finals (126 blocks)")
    print("done -> results/scorecard_final.md", flush=True)


def run_final(name: str):
    fm, blocks = load_blocks()
    # rebuild the config from its name via the tournament matrix
    cfg = next((c for c in tournament_configs() if c["name"] == name), None)
    if cfg is None:
        raise SystemExit(f"unknown config {name}")
    cfg = {**cfg, "name": f"FINAL_{name}"}
    print(f"[FINAL] {name} on full {len(blocks)} blocks ...", flush=True)
    bodies = produce(cfg, blocks)
    (OUT / f"{cfg['name']}.md").write_text(lab.reassemble(fm, blocks, bodies))
    sc = score(cfg, blocks, bodies, "opus", 5)
    print(json.dumps({k: sc[k] for k in (
        "hallucination_count_median", "hallucination_count_spread",
        "faithfulness_ratio_mean", "asr_correction_mean", "fluency_mean",
        "readability_mean", "conciseness_mean", "residual_asr_errors_median",
        "fabricated_numbers", "novel_words_total", "length_ratio")}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "tournament"
    if cmd == "tournament":
        run_tournament()
    elif cmd == "finals":
        run_finals()
    elif cmd == "final":
        run_final(sys.argv[2])
    else:
        raise SystemExit(__doc__)
