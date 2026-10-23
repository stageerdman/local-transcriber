# Groq cloud backend — durable decisions & lessons

## Durable decisions

- **Model-id convention: `groq:<model>` prefix.** Keeps dispatch a cheap
  string check (matches the existing `"parakeet" in name` pattern) and keeps
  the catalog id visually distinct from on-device `mlx-community/...` repo
  ids. `GROQ_ID` in `src/transcription.py` is the one catalog entry actually
  wired up (`large-v3-turbo`); the prefix supports more if ever needed.

- **`verbose_json`, not `text`.** Groq's default/simplest response format is
  plain text with no timestamps — useless here, since `dialogue`
  blocking/merging and multi-track interruption flagging are built entirely on
  segment start/end times. Always request `verbose_json`.

- **No mid-call progress, no mid-file resume.** One blocking HTTP call = the
  whole file. Shaped exactly like the Parakeet backend: emits everything as
  chunk 0, `progress_callback` fires once at completion (not incrementally),
  and a stop/resume mid-Cloud-call just re-runs the whole unit. Acceptable
  because the call is already seconds, not minutes.

- **`.env` loaded in `app/__init__.py`, not a new dependency.** The file is
  flat `KEY=VALUE`; a ~15-line manual parser avoids adding `python-dotenv` for
  one secret. Existing environment variables win over `.env` values (standard
  convention, in case someone exports `GROQ_API_KEY` in their shell instead).

- **Cloud is never the default or auto-restored tier.** `Bridge.get_settings()`
  maps any persisted `Cloud` selection back to `Best` at read time — Cloud
  must be re-picked every launch. Keeps the "offline by default" identity
  honest: no user is ever silently initialized into the one tier that sends
  audio off the device, even if they picked it last session.

- **turbo over plain large-v3.** In same-sample testing on real (colloquial,
  code-switched Czech/Slovak) call audio, plain `large-v3` hallucinated a
  repeated-phrase loop on a quiet opening — the known Whisper silence-loop
  failure mode the local pipeline's VAD + `condition_on_previous_text=False`
  exists specifically to avoid. Groq's raw API gets none of those local decode
  protections, so `turbo` (which didn't loop on this sample) was the safer
  pick, and it's also faster/cheaper.

## UX decisions (from the dedicated UX pass)

- Button label: **"Cloud"** — not "Groq." The backend is an implementation
  detail; the user-facing concept is "this one uses the internet."
- The disclosure that audio leaves the device is **inline in the existing
  `#qeta` label**, not a tooltip or modal: `~20 sec · sends audio online`.
  Always visible the instant Cloud is picked — no hover, no extra element.
- Cloud's ETA is a **flat `~20 sec`**, not duration-scaled like the other two
  tiers — intentional, since Groq's whole value proposition is "basically
  instant" regardless of call length.
- Error copy for all three Cloud failure modes (missing key, no network,
  generic API failure) ends with the same actionable fallback — "switch to
  Faster or Best" — since that's the one thing every failure mode shares.

## Known limitation to watch

The raw Groq API skips the local pipeline's VAD-based chunking and tuned
decode options (`condition_on_previous_text=False`, temperature ladder, etc.)
entirely — see `WHISPER_DECODE_OPTS` in `src/transcription.py`. Cloud quality
is not a strict upgrade over the local Czech-finetune; it's a speed/cost
tradeoff. If hallucination on Cloud turns out to be a recurring problem (not
just the one sample tested this session), the fix would be pre-trimming
silence client-side before upload, not something Groq's API exposes directly.
Confirmed in real use: Groq's stock `large-v3-turbo` did produce the classic
Whisper fansub-credit hallucination ("Titulky vytvořil...") on a real Czech
call, and Groq's API accepts none of `WHISPER_DECODE_OPTS` as request
params (checked against their docs) — there's no way to suppress it
server-side. This is what motivated the ElevenLabs addition below, rather
than trying to patch Groq's quality further.

## Cloud-model-research spike (ElevenLabs Scribe v2)

Searched for a Czech-accuracy upgrade over Groq's stock Whisper. Findings
(web search, October 2026 — re-verify if this gets revisited much later):

- **ElevenLabs Scribe v2**: 3.1-5.5% WER on Czech (FLEURS / Common Voice)
  — vs. Deepgram Nova-2's 20.5% WER on the same FLEURS test. $0.22/audio-hr.
  Chosen over the alternative below for two reasons: (1) better or
  comparable accuracy, (2) **native word-level timestamps**, which fit this
  codebase's `_seg_times`-style "anchor to word starts, not segment starts"
  philosophy better than Groq's segment-level timestamps do.
- **Speechmatics** (Melia-1/Enhanced): markets a dedicated Czech dialect
  model at ~96% accuracy, comparable word-timestamp support. Close second,
  not implemented — no reason found to prefer it over Scribe v2, and two
  cloud backends was already the agreed scope.
- Scribe v2's **5GB file-size limit** is generous enough (vs. Groq's much
  tighter cap, which is what `_compact_mp3_for_upload` exists to work
  around) that the ElevenLabs path skips re-encoding entirely and uploads
  the source file as-is — simpler, and no lossy downsample.

### Durable decisions (ElevenLabs backend)

- **Model-id convention: `elevenlabs:<model>` prefix**, mirroring Groq's.
  `ELEVENLABS_ID` = `elevenlabs:scribe_v2`.
- **No `segments` from Scribe's API — only a flat per-word list.** Unlike
  Groq/Whisper's `verbose_json`, there's no sentence/phrase grouping in the
  response at all (confirmed against the installed `elevenlabs` SDK's
  `SpeechToTextWordResponseModel`/`SpeechToTextChunkResponseModel` types).
  `_coalesce_elevenlabs_words` in `src/transcription.py` reconstructs
  segment-shaped output so the rest of the pipeline (`dialogue` blocking,
  multi-track merging) doesn't need to know the difference: a new segment
  starts on a pause > 0.6s or right after sentence-ending punctuation. 0.6s
  was picked to sit just under `dialogue.DEFAULT_PAUSE_SECONDS` (0.8s), so a
  single speaker's natural mid-sentence breath still gets re-merged into one
  block downstream instead of staying split into two short ones.
- **Kept alongside Groq, not swapped in.** Explicit user decision — cost/
  speed (Groq) and Czech accuracy (ElevenLabs) should both stay one click
  away, not a tradeoff baked into a single fixed "Cloud" choice.
- **`cloud_provider` is a separately-persisted setting, exempt from the
  Cloud opt-in gate.** The "never auto-restore into Cloud" privacy rule
  exists so a user never gets silently re-enrolled into sending audio off
  the device. The *provider* sub-choice carries no such risk on its own — it
  only takes effect once Cloud is independently re-picked each session — so
  unlike `quality`, it's fine to remember across launches. Default: `groq`.
- **ElevenLabs' real-time factor in `catalog.DEFAULT_REAL_TIME_FACTOR` is a
  placeholder (0.15).** First real measurement (60s Czech/Slovak call,
  single sample): ~7.5s, i.e. ~0.125 — close to the placeholder, left
  unchanged since one short sample doesn't separate per-second decode cost
  from fixed HTTP/network overhead. Revisit with a longer or second sample
  if the ETA display drifts noticeably off in practice.

### UX decisions (from the dedicated UX pass)

- The provider picker is a **small, visually quieter sub-choice (`.qsub`)**
  next to the `.qseg` Faster/Best/Cloud control, not a fourth peer button —
  shown only when Cloud is selected. Reusing the bold `.qseg`/`.qpick` pill
  styling for it would have made it read as a second primary choice, which
  undermines "Cloud" staying one backend-agnostic concept at the top level.
- **Plain provider names** ("Groq" / "ElevenLabs"), not benefit-oriented
  labels ("Fast" / "Accurate Czech"). The backend-agnostic principle governs
  the top-level Cloud button; once a user has already opted into "uses the
  internet," naming *which* internet service is honest and won't go stale
  as speed/accuracy tradeoffs shift later. The tradeoff itself is stated in
  real units via the ETA line instead.
- `#qeta`'s `sends audio online` disclosure is **unconditional across both
  providers** — only the time estimate before it changes (`~20 sec` for
  Groq, a duration-scaled placeholder for ElevenLabs). Never make the
  disclosure clause itself provider-specific wording.
- Default provider on first Cloud pick: **Groq** (the already-proven,
  zero-config option; ElevenLabs is the deliberate opt-in for the Czech win).
