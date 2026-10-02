# Wiki — Claude-CLI polish (durable findings)

## Why we switched off the local model
The previous polish (see `2026-10-01 transcription-quality-research`) used Gemma
3 12B QAT 4-bit via mlx-lm. Correct and hallucination-safe, but: loads ~6.7 GB
before generating, then generates per chunk/block on an M3/16 GB — minutes of
wall-clock on top of ASR. On real use it dominated job time. Claude via the CLI
cleans an entire transcript in one call in seconds, and is a stronger model.

## CLI invocation (decided)
- Resolve the binary with `shutil.which("claude")` (falls back to
  `/opt/homebrew/bin/claude`). The interactive shell *alias* adds `--plugin-dir`;
  the bare binary does not need it and must not be relied on in a subprocess.
- Call: `subprocess.run([claude, "-p"], input=<full prompt>, text=True,
  capture_output=True, timeout=...)`. Full prompt (instruction + delimited text)
  goes on **stdin** — no shell, no arg-length limit, no injection from transcript
  text. Default output format is plain assistant text.
- Startup+round-trip overhead ≈ 4–8 s per call, so **batch**: one call per whole
  text passage / per block-list, never one call per block.

## Prompt design (research-backed)
Sources: OpenAI cookbook (whisper processing / correct-misspelling),
github.com/meain/ojut, github.com/darian-gajgic/local-wisprflow PR#7,
huggingface danielrosehill Basic-STT-Transcript-Cleanup, theneuralbase
post-processing-with-llm, dev.to masonwritescode video-chapters.

Recurring techniques adopted:
- **Write the prompt in the target language.** English instructions make the
  model complete/translate non-English input. So the Czech prompt is in Czech.
- Enumerate only the allowed edits (punctuation, casing incl. diacritics, filler
  removal, obvious-and-unambiguous ASR fixes); explicitly forbid adding,
  summarizing, paraphrasing, answering the content.
- **Echo uncertain words verbatim** ("nech beze změny") — key anti-hallucination.
- **Output-only discipline** — "return only the cleaned text, no preamble".
- Batch segments as a numbered/JSON structure and require the same count back,
  so segment order/count can't drift.

## Anti-hallucination guards (in `src/polish.py`)
Applied per chunk and per block; on any failure keep the RAW text:
1. **Length tripwire** — output > 1.4× input chars ⇒ keep raw (ballooning is a
   hallucination signature). Kept from the previous implementation.
2. **Accent-folded word-containment backstop** (from local-wisprflow): fraction
   of output words that already appear in the input (both accent-folded and
   lowercased, so added Czech diacritics don't count as drift) must be ≥ 0.6;
   below that the model drifted/invented ⇒ keep raw. Very short outputs are
   exempt (too few words to measure). "A false positive costs polish, never
   correctness."
3. **JSON array contract** — block batching must return a parseable array of the
   same length; any parse/length mismatch ⇒ keep raw for the affected blocks.
   Stray ```json fences are stripped before parsing.

## Progress model (decided)
Polish is folded into the same overall bar as a time-estimated **tail**:
- `estimated_polish_seconds` ≈ `POLISH_BASE_SECONDS + audio_duration *
  POLISH_RTF` (audio duration as a cheap proxy for transcript length).
- `polish_share = est_polish / (est_transcribe + est_polish)`, clamped to
  [0.05, 0.40]. Transcription fills `[0, 1 - polish_share]`; polish animates
  `[1 - polish_share, 1]` from its own elapsed/estimate.
- ETA shown during transcription already includes the polish estimate, so the
  "~X left" number accounts for the whole job. The bar never resets to 0 and
  never freezes during polish (previously "polishing" had no UI branch and fell
  through to value=0).
</content>
