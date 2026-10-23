# Wiki — Polishing Research (durable findings)

Scope: find the polishing **prompt + context-granularity + algorithm** that yields
the most readable Czech transcript **without hallucination**, tested on the real
Marek Pavel P2 sales call. Isolated spike — does NOT touch `src/`/`app/`.

## 1. SOTA for LLM post-editing of raw ASR (2024–2026)

**The one-line truth:** a generic "clean up this transcript" LLM pass
**over-edits, paraphrases, and injects content by default**, and on
already-decent or low-resource (Czech-grade) material it can make transcripts
*worse*. Everything that reliably helps points one way: **minimal-edit framing,
extractive/copy behavior, reasoning OFF, moderate chunking, and a programmatic
diff/revert guard.** Treat "did it stay faithful?" as the primary metric, not
"does it read nicely."

### What helps (build these)
- **Minimal-edit framing.** LLMs default to fluency (free paraphrase, precision
  <45%). Fix = explicit *smallest-change* directive + **forbidden list** (don't
  rephrase / add / remove / translate) + **"leave unchanged" escape hatch** +
  **output-only discipline**. GEC scores on F₀.₅ (precision 2× recall) for
  exactly this reason: an invented edit is worse than a missed one.
- **Tool/role frame** ("You are a correction tool") constrains more than
  "assistant". Mark unclear audio `[nesrozumitelné]` rather than guessing (the
  single highest-impact anti-hallucination lever).
- **Moderate chunk size.** Document-level beats sentence-level for cross-sentence
  consistency, but the benefit **saturates within ~2 neighboring turns**.
  Whole-document *hurts* faithfulness ("lost in the middle" / "context rot":
  silent omission/truncation on long inputs). And **larger batches reduce
  overcorrection** (untouched-rate 43.8%→80.7% as batch grew) — a regularizer.
  → sweet spot is a **small window / moderate batch**, split at speaker turns,
  read-only overlap.
- **Extractive/copy prompting** ("reuse the speaker's exact words") measurably
  cuts invention.
- **Programmatic diff gate / revert-unsupported.** Diff input→output, reject any
  edit outside allowed categories (punctuation, casing, diacritics, filler
  removal, dedupe), cap total edit distance. Mechanical, cheap, doesn't trust
  the prompt. (The app already has a weak version: length tripwire + word
  containment.)
- **Edit-level majority voting** (N samples, keep edits in a majority). Precision
  via consensus. Caveat: entrenches *systematic* mishears.
- **Reasoning OFF.** Reasoning/thinking *hurts* faithfulness tasks (translation
  scored higher with reasoning off across every model tested; CoT increases
  overcorrection) — the model "decides it could be better" and rewrites.

### What's overhyped (be skeptical)
- **Full Generative Error Correction (GER/H2T)** needs N-best hypotheses +
  acoustic features + in-domain fine-tuning. We have only the final text → out of
  scope. Zero/few-shot LLM correction *regressed* WER catastrophically in
  controlled tests (meaning reversals, world-knowledge injection, 3–12%
  hallucination in low-error regimes).
- **Few-shot** is NOT a reliable win for restraint — examples tend to *increase*
  edit aggressiveness / import style drift. Default to a well-specified zero-shot.
- **Naive two-pass self-critique** can *increase* hallucination (1.55→2.13 per
  summary in one study). Only a **source-grounded accept/reject-per-edit** second
  pass helps — never a free "improve this" rewrite.
- **Confidence-gated correction** (only touch low-confidence spans) is the one
  real GER lesson we could use — but it needs ASR logprobs the finished `.md`
  doesn't carry. Out of scope here; worth revisiting at the engine level.

### Czech / low-resource specifics
- For Czech GEC, fine-tuned task-specific models still beat general LLMs. Czech
  levers: morphology/agreement, diacritics restoration, overcorrection
  avoidance. Hallucination risk rises as resource level drops (Czech =
  mid-resource: elevated, not catastrophic).
- **Instruction language (Czech vs English) is genuinely unsettled** — no clean
  A/B exists. Risk for non-English: the model "helpfully" translates/anglicizes.
  → test it as a variable; keep a hard "do not translate, keep Czech" clause.

## 2. SOTA for reference-free quality judging + hallucination counting

We have **no gold transcript**, so judging is *source-grounded*: compare the
polished candidate against the RAW ASR. Protocol adopted (see `spike/judge.py`):

- **Pointwise, structured-JSON, reasoning-before-count** (G-Eval lineage). Two
  **isolated** passes so fluency can't mask a hallucination:
  1. **Faithfulness pass (the gate):** decompose polished text into atomic
     claims; label each supported / unsupported / contradicted vs raw; list
     added content, changed meaning, substantive omissions.
     `hallucination_count = unsupported + contradicted` (FActScore/RAGAS
     "faithfulness ratio" framing — exactly the "count the mistakes" ask).
  2. **Quality pass:** 1–5 on ASR-error correction, Czech fluency, readability,
     conciseness + residual-ASR-error count.
- **Stability:** single judge runs are unreliable ("Rating Roulette": majority
  vote over 3 beat any single run; temp≈0 *reduced* accuracy). → run **N times**
  (3 tournament / 5 final), **median** counts, **mean** scores, **report spread**.
- **Known biases:** position (anonymize/pointwise), verbosity ("longer ≠
  better"), **self-preference** — our judge is also Claude, a real risk. Mitigate
  with deterministic, family-agnostic cross-checks:
  - **number retention** (digit tokens in polished absent from raw — a ~1:1
    fabrication signal for a sales call's prices/dates),
  - **novel-content-word rate**, length ratio, word-retention.
  These form a **hard gate**: a fabricated number or median hallucination>0
  disqualifies a variant no matter how fluent.
- **Caveats:** LLM claim-counts are themselves noisy (report ranges, spot-check);
  non-English judging is weaker (use a strong judge, prompt in Czech);
  embedding/BERTScore-style metrics miss flipped numbers/negations — never the
  faithfulness metric.

## 3. Experiment design (this spike)

Greedy staged sweep on a fixed **representative excerpt** (affordable: per-block
strategies cost one `claude -p` call per block; the CLI has a ~16 s/call floor),
then the single winner + the current-app baseline on the **full 126-block
transcript** with N=5 judging. Greedy ignores cross-axis interactions — a
documented limitation.

- **Exp 1 — context granularity** (prompt + algo fixed): per-block-isolated ·
  windowed(±2, read-only) · small-batch(~5 turns) · large-batch(~1.5k chars) ·
  whole-transcript-array · whole-transcript-markdown. → pick G*.
- **Exp 2 — prompt** (G* fixed): fluency-control · minimal-edit-CS ·
  checklist-CS (TalkLess) · minimal-edit-EN · current-app-prompt. → pick P*.
- **Exp 3 — algorithm** (G*,P* fixed): single-pass · edit-level-vote(N=5) ·
  diff-gate/revert · source-grounded-self-verify. → pick A*.
- **Exp 4 — model/reasoning** (G*,P*,A* fixed): haiku · sonnet · opus; reasoning
  off vs on.
- **Final:** best config vs current-app baseline on the full transcript, N=5.

Every strategy returns a body list the same length/order as the input blocks, so
judging and drift metrics line up and speaker labels/timestamps are never exposed
to the model (added back verbatim on reassembly).

## 4. Results

Excerpt tournament: `results/scorecard.md`. Full-transcript finals (126 blocks,
Opus judge N=3): `results/scorecard_final.md`. Per-config outputs:
`results/outputs/*.md`.

### 4a. Methodological finding (important)
- **The LLM judge alone is NOT reliable here.** In the tournament the Sonnet
  judge rated `p_fluency` — a generic "make it readable" prompt that invented 43
  novel words and dropped 9% of content — as **0 hallucinations, faith 0.99, best
  quality**. Fluent paraphrase fools the judge (verbosity + self-preference
  bias). The **deterministic drift metrics (novel-word rate, word-retention)**
  and **direct human/Opus reading** caught it. → never gate on the LLM count
  alone.
- **But `novelW` over-counts on this transcript.** Because the raw ASR is so
  garbled, almost every *legitimate* correction (`zlyhavate`→`zlyháváte`,
  `vyskrtnout`) registers as a "novel" word. So a higher novelW here is partly
  *more correction* (good), not only drift (bad). The honest faithfulness gate is
  therefore: deterministic **number/entity retention** (hard) + **word-retention
  trend** + **direct reading of the highest-risk blocks**, not raw novelW.

### 4b. Faithfulness verdict (direct read of the risk blocks)
Spot-read the price block ("sto tisíc", "dvacet tisíc"), the 14-day-guarantee
block, the health negations, and the long pitch block across all four finalists:
**no finalist fabricated a number, price, claim, or flipped a negation.** All
four preserved the prices byte-identically (deterministic `fab# = 0` confirmed by
eye). The metric-flagged drift was overwhelmingly legitimate Czech ASR/grammar
correction. Anti-hallucination is essentially a *solved* problem once the prompt
is minimal-edit + "mark `[nesrozumitelné]` instead of guessing" — the real axis
of difference is **readability vs. conservatism**.

### 4c. Full-transcript finals (126 blocks)
| config | prompt×context×algo | drift novelW | word-ret | readability (resid↓) | reads | faithful? |
|---|---|---|---|---|---|---|
| **g_batch_small** | minimal-CS · small-batch(~5 turns) · single | 43 | 0.967 | **12** | read 4 / flu 3.3 / asrfix 3.7 | ✅ no fabrication |
| g_whole_md | minimal-CS · whole-doc markdown · single | 64 | 0.962 | **10** | read 4 / flu 3.7 / asrfix 4 | ✅ but most liberal (restructures, drops words) |
| a_ensemble | minimal-CS · small-batch · 3-model edit-vote | **18** | **0.989** | 30 | read 3.7 / flu 3 | ✅ safest, but under-corrects (leaves garbled words) |
| p_app_current | **current app prompt** · small-batch · single | 47 | 0.975 | 28 | read 4 / flu 3 | ✅ no fabrication, but… |

### 4d. Conclusions
1. **Winner: `g_batch_small` — minimal-edit Czech prompt + moderate (small-batch,
   ~5 turns) context + single pass.** It reaches ~all of whole-doc's readability
   (read 4, resid 12) with less liberal restructuring, and is faithful. Matches
   the research prediction (moderate chunk = sweet spot).
2. **The current app prompt is dominated.** Same context/model, only the prompt
   differs: `g_batch_small` (minimal-edit) beats `p_app_current` on BOTH
   faithfulness trend (word-ret 0.967 vs… comparable, fewer liberal edits) and
   **readability (resid 12 vs 28)**. The shipped prompt drifts *without* buying
   readability. Swapping in the minimal-edit prompt is a clear, low-risk win.
3. **Whole-document is viable at this length** (no block omission on 126 blocks)
   and reads marginally best, but is the most liberal (novelW 64, drops pronouns,
   restructures) → slightly higher meaning-change risk. Pick it only if max
   readability is wanted and light word-order cleanup is acceptable.
4. **The cross-model edit-vote ensemble is the safety champion** (word-ret 0.989,
   marks `[nesrozumitelné]`) but under-corrects the hard Slovak-garbled blocks and
   costs 3× calls — overkill unless faithfulness must be guaranteed above
   readability.
5. **ASR quality still caps the ceiling.** Residual errors are high everywhere
   because the client's Slovak-tinged, badly-recognized speech *cannot* be fixed
   without guessing — and every faithful config correctly refuses to guess.
   Polishing improves readability; it cannot recover true ASR word errors.

### 4e. Recommendation for `src/polish.py` (future update — NOT applied here)
- **Swap the prompt** to the minimal-edit variant (`strategies.PROMPTS["minimal_cs"]`):
  smallest-change framing + explicit "do not translate / keep Czech" + "mark
  `[nesrozumitelné]` instead of guessing". This beat the current prompt on both
  axes with no extra cost.
- **Keep batched context but shrink the batch** from "all blocks in one call" to
  ~5-turn chunks (moderate context regularizes against over-editing).
- **Keep the raw `.raw.md` always** (already done) and the length/containment
  guards (cheap deterministic backstop).
- Optional high-assurance mode: the 3-model edit-vote ensemble for users who want
  maximum faithfulness over readability.
- Out of scope but promising: confidence-gated correction at the ENGINE level
  (needs ASR logprobs) — only touch low-confidence spans.

## Sources
See the two research briefs archived in `spike/sources.md`.
