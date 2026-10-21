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
