# Web-UI rewrite — update v1 (CLOSED)

**Status: CLOSED (2026-10-23)** — user confirmed it works in real use.
Originally "OPEN — building (Phases 1–4 + playback/export/cutover shipped;
Fix-speakers override + drag-drop remain; pending full real-app verify)".

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

---

## ▶ RESUME HERE (paused 2026-10-05)

The Tkinter UI has been replaced by a native pywebview app and is the default
entry. Core loop works end to end (verified headless + via the user's interactive
runs); several rounds of real-run bug reports have been fixed.

**Run:** `.venv/bin/python -m app.main`  (old Tkinter still at `app.main_tk`)

### Architecture (recap)
- `app/web/index.html` — single-file front-end (HTML/CSS/JS), from the approved
  prototype (`scratchpad/transcriber-redesign.html`). Falls back to a mock demo
  when opened without the bridge.
- `app/webview_main.py` — opens the WebKit window, wires `Bridge`, sets the dock
  icon, attaches the OS drag-drop handler, `debug=True` (devtools on).
- `app/bridge.py` — js_api over the engine: get/set_settings, recent_calls,
  open_call, pick_files+detect, transcribe (via `TranscriptionWorker`),
  play_source/stop_audio/playback_position, reveal, handle_drop. Emits `job` and
  `dropped` events to the UI. Reuses all of `src/` unchanged.

### DONE
- Native shell; real recent calls + settings; drop/pick → detect speakers →
  "Who's on this call"; transcribe for real with live queue progress; open a
  finished transcript as the two-voice script.
- Playback: play/pause, click-a-line-to-play, live progress bar + speaker label +
  current-line highlight + auto-scroll; previews start at first SUSTAINED speech.
- Audio lifecycle: one source at a time; stops on navigation / open+close Fix;
  killed on quit (shutdown + atexit) — no more orphaned ffplay.
- Export (reveal .md in Finder); full-window drag-and-drop; brand app icon;
  sequential speaker names; Fix speakers editing surface (waveform per person,
  rename, Don't-transcribe, Remove-a-voice picker, add back a set-aside track) →
  drives the worker via speakers + subtractions. Cutover: `app.main` = new UI.

### PENDING / NEXT (pick up here)
1. **Full real-app verify**: a complete GUI transcribe of a real call confirming
   queue→progress→finished transcript. (User has been running interactively;
   no end-to-end "transcribe to completion" confirmation captured yet.)
2. **Fix speakers from a FINISHED transcript**: the legend "Fix speakers…" opens
   the editor but a post-transcribe structural change needs a NON-DESTRUCTIVE
   RE-RUN (keep old transcript until the new one is ready). Only the pre-transcribe
   path is wired today. `renderFix` uses `currentDetect` (the last detected call);
   from a finished transcript it may be empty.
3. **Per-person language**: the `CS` token in the UI cycles but is NOT sent to the
   worker; only the global default language (settings) is applied. Wire
   `track_languages` through transcribe if wanted.
4. **Multiple files**: pick/drop returns all paths but `openDetect` only handles
   `paths[0]`; batch (detect+queue each) is not wired.
5. **Release hygiene**: turn off `debug=True` in `app/webview_main.py`; then
   formally retire `app/main_tk.py` + the Tkinter `app/ui.py` once confident.
6. Dead CSS from the prototype (settings/choice/.p-adjust/.ribbon etc.) can be
   pruned from `app/web/index.html`.

All code committed on `main`. If GitHub was unreachable at pause time, push when
back online.

## 2026-10-23 — Closed
User confirmed in real use: it works. Closing. Items 2-6 above (fix-speakers
re-run, per-person language, multi-file batch, Tkinter retirement, dead CSS)
were not done this session — carried to `issues.txt` as optional follow-ups
rather than blocking this close.
