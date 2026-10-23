# Research sources (SOTA briefs, 2024–2026)

Two web-research agents gathered these; distilled conclusions live in
`../wiki.md` §1–2. Load-bearing findings are multiply-sourced; several 2025/2026
preprints are directional (abstract-level) — flagged in the wiki.

## A. LLM post-editing of ASR / conservative correction

**Generative Error Correction (GER):** HyPoradise https://arxiv.org/abs/2309.15701 ·
Whispering LLaMA https://arxiv.org/abs/2310.06434 · RobustGER https://arxiv.org/abs/2401.10446 ·
ClozeGER https://arxiv.org/pdf/2405.10025 · MMGER https://arxiv.org/pdf/2405.03152 ·
**Revisiting ASR EC w/ Specialized Models (failure modes: WER regression, meaning
reversals, world-knowledge injection)** https://arxiv.org/html/2405.15216 ·
Confidence-Guided EC https://arxiv.org/pdf/2509.25048 · West Frisian
contamination study https://arxiv.org/html/2605.19711

**Conservative prompting / minimal-edit:** OpenAI Whisper cookbook
https://developers.openai.com/cookbook/examples/whisper_processing_guide ·
https://developers.openai.com/cookbook/examples/whisper_correct_misspelling ·
Katinskaia & Yangarber (smallest-change prompt) https://arxiv.org/abs/2401.07702 ·
Is ChatGPT a Fluent GEC System https://arxiv.org/abs/2304.01746 ·
**Larger Context Window, Fewer Overcorrections** https://arxiv.org/html/2609.10810v1 ·
Minimal-edit GEC https://arxiv.org/pdf/2506.13148 ·
LREC 2024 prompting https://aclanthology.org/2024.lrec-main.569/ ·
TalkLess (allowed/forbidden checklist) https://arxiv.org/abs/2507.15202

**Context / chunk size:** Document-level GEC (Yuan & Bryant)
https://aclanthology.org/2021.bea-1.8/ · Cross-Sentence GEC
https://aclanthology.org/P19-1042.pdf · Lost in the Middle
https://arxiv.org/html/2307.03172v1 · Context Rot (Chroma)
https://www.trychroma.com/research/context-rot

**Hallucination mitigation:** CopyPasteLLM (extractive)
https://arxiv.org/html/2510.00508 · Edit-level Majority Voting
https://arxiv.org/pdf/2605.13624 · CLEME2.0 https://arxiv.org/pdf/2407.00934 ·
Factored Verification (self-correction backfire 1.55→2.13)
https://arxiv.org/abs/2310.10627 · Anthropic reduce-hallucinations
https://platform.claude.com/docs/en/docs/build-with-claude/reduce-hallucinations

**Reasoning hurts faithfulness:** Overthinking of o1-like LLMs
https://arxiv.org/pdf/2412.21187 · Stop Overthinking survey
https://arxiv.org/pdf/2503.16419 · Reasoning makes translations worse
https://www.zansara.dev/posts/2026-07-05-reasoning-makes-translations-worse/

**Czech / low-resource:** Refining Czech GEC https://arxiv.org/abs/2506.22402 ·
GECCC https://aclanthology.org/2022.tacl-1.26/ · AKCES-GEC
https://arxiv.org/pdf/1910.00353

### Quotable prompt templates (tested in the harness)
- OpenAI Whisper cookbook: *"Preserve the original words and only insert
  necessary punctuation... Use only the context provided."*
- TalkLess checklist: *"1. No New Words. 2. Same Wording. 3. Include Everything.
  4. Remove Filler Words. 5. Preserve Style. 6. Remove Repetitions."*
- Katinskaia & Yangarber: *"Make the smallest possible change... Do not change
  the meaning by adding or removing information. If already correct, output the
  original unchanged."*

## B. Reference-free judging + hallucination counting

**LLM-as-judge:** G-Eval https://arxiv.org/pdf/2303.16634 · Pairwise vs Pointwise
bias https://arxiv.org/abs/2504.14716 · Comparative Trap
https://arxiv.org/pdf/2406.12319 · Justice or Prejudice (bias audit)
https://arxiv.org/pdf/2410.02736 · Self-preference bias
https://arxiv.org/pdf/2410.21819 · Reliability without Validity
https://arxiv.org/pdf/2606.19544

**Stability / counting:** Rating Roulette (majority-vote > single run)
https://arxiv.org/html/2510.27106v1 · Temperature in LLM-as-judge
https://arxiv.org/html/2603.28304v1

**Reference-free faithfulness / hallucination:** FActScore (atomic facts)
https://arxiv.org/abs/2305.14251 · Claim decomposition
https://arxiv.org/pdf/2403.11903 · RAGAS (faithfulness ratio)
https://arxiv.org/pdf/2309.15217 · AlignScore https://arxiv.org/abs/2305.16739 ·
Czech AlignScoreCS https://arxiv.org/html/2512.06586 ,
https://huggingface.co/krotima1/AlignScoreCS · SelfCheckGPT
https://arxiv.org/abs/2303.08896 · Fine-grained span-level hallucination
https://arxiv.org/pdf/2509.22582

**Metric limits / multilingual:** Semantic Illusion (embeddings miss flipped
numbers) https://arxiv.org/html/2512.15068v1 · Multilingual judge reliability
https://arxiv.org/pdf/2505.12201 , https://arxiv.org/html/2607.02235
