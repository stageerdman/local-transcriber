# Web-UI rewrite — update v1 (OPEN)

**Status: OPEN — building (Phases 1–4 shipped 2026-10-04; pending real-app verify)**

## Goal
Replace the Tkinter UI with a from-scratch, beautiful native front-end — the
design the user signed off on (prototype in `scratchpad/transcriber-redesign.html`,
published artifact). Keep the entire Python transcription engine (`src/`); only
the UI layer changes. Thesis: **a call is a dialogue — present it as a typeset
two-voice interview**; the common path is drop → Transcribe; all mechanics are
silent or one quiet step away.

## Why not Tkinter
Tkinter cannot be made to look modern; the user rejected reskins. New UI is
HTML/CSS/JS rendered in a native WebKit window via **pywebview** (+ pyobjc),
wired to Python through a `Bridge` (js_api). Fully offline, no browser.

## Design (signed off, after many UX passes — see scratchpad + artifact)
- Transcript is the hero: New York serif, ~68ch column, two speakers by warm/cool
  ink, a voices legend (rename inline · ⇄ swap · Fix speakers), a calm player.
- Home: drop zone (one or many files) + recent calls.
- Multi-call: sidebar is a live queue (working/queued/done); read finished while
  others run.
- "Who's on this call?" (Fix speakers): the **waveform is the control surface** —
  a bled-in voice shows as a struck ghost you click to remove; a language token;
  no jargon, no drawers.
- Settings as one in-flow step: Quality (Faster/Best + ETA) · Tidy · (language
  global). No model names.

## Architecture
- `app/web/index.html` — the front-end (generated from the approved prototype).
- `app/webview_main.py` — creates the pywebview window, loads the UI, wires `Bridge`.
- `app/bridge.py` — js_api exposed as `window.pywebview.api.*`; pushes events to
  the UI via `window.appEvent(...)`. Reuses `src/` (transcription, track_separation,
  dialogue, md_writer, history_db, polish, catalog).
- Old Tkinter (`app/main.py`, `app/ui.py`) stays until the new app reaches parity,
  then is retired.

## Roadmap (phases)
1. **Native shell** — pywebview window renders the approved UI; bridge stub. ✅ DONE.
   Run: `.venv/bin/python -m app.webview_main`.
2. ✅ **History + settings** — populate recent calls from `history_db`; load/save
   quality/tidy/language via settings.json. Replace the JS mock data with bridge calls.
3. ✅ **Add + detect** — native file pick/drop → probe tracks → detect speakers
   (track_separation) → "Who's on this call" from real data.
4. ✅ **Transcribe** — run the engine (reuse worker logic), stream real progress to
   the queue, render the finished `.md` as the two-voice transcript.
5. **Fix speakers** — wire the ghost-remove / language / cleanup to real
   subtractions; re-run non-destructively.
6. **Parity + cutover** — audio preview (playback.py), export, polish, drag-drop;
   retire Tkinter; tests; verify by running the real app.

## Notes
- pywebview 6.2.1 + pyobjc WebKit install cleanly from PyPI (GitHub is blocked
  here, PyPI is not). Added to `requirements.txt`.
- The front-end JS currently uses mock data (sample call, fake queue) so the shell
  is demoable before the bridge is wired; Phase 2+ swaps mocks for bridge calls.
