# Local Transcriber — working agreement

Compiled from the global AI Control modules (WORKFLOW, STRUCTURE, CODING, UX)
for this project. See `.project` for the module list; rebuild with the
`rebuild-claude-md` routine when the modules change.

Local Transcriber is an **offline Mac desktop app** (Python + `mlx-whisper` /
`parakeet-mlx`, ffmpeg, native UI). It transcribes audio/video on-device and
writes `.md` transcripts, with multi-track speaker-labeled dialogue blocks. A
companion `testing/` harness benchmarks transcription backends.

## Structure
- Standard AI Control scaffold lives at the project root: `.project`,
  `CLAUDE.md`, `.gitignore` (covers `.env`), `.env` (secrets, never committed),
  `updates/`, `issues.txt`.
- Keep the layout **flat and idiomatic to Python** — `src/` (core logic), `app/`
  (UI/worker/jobs), `tests/`, `scripts/`. Add folders only when the project
  genuinely grows into them; don't nest prematurely.
- `testing/` is a subordinate benchmark harness, not a separate project. It
  shares the app's venv and mirrors the app's model list.
- **Updates** go in `updates/YYYY-MM-DD NAME - OPEN|CLOSED/` folders, each with
  `update vX.md` (goal + phased roadmap + live status) and `wiki.md` (durable
  decisions and lessons). Reopen an update by flipping `CLOSED` → `OPEN`.

## Coding
- **Modular, isolatable.** Each `src/` module (transcription, audio_tracks,
  dialogue, history_db, md_writer, media_finder, converter, duration) should be
  understandable and fixable on its own, with a small explicit interface and low
  coupling. A bug should have one obvious home.
- **Minimal.** Build only what's needed. Prefer reuse over duplication; don't
  add abstraction on speculation. (History: a v1 single-track speaker-guessing
  heuristic was cut for being unreliable — see the multi-speaker update. Keep
  that bias: cut what doesn't earn its place.)
- **Idiomatic Python.** Match the surrounding code and standard tooling; read
  the neighbors before writing.
- **Test what matters.** Core logic and risky paths — audio-track detection,
  dialogue blocking/overlap, history DB, markdown output. Skip trivial glue.
  `pytest` from the project root; most roadmap phases end with tests that lock
  in what they built.

## UX
- It's a user-facing desktop app — for any change to the surface, **launch a
  dedicated UX-expert agent** (several in parallel for multi-part surfaces) and
  synthesize their thinking; don't improvise the experience.
- **Cut everything unnecessary** (Steve-Jobs default: does this element earn its
  place?). Fewest screens, controls, and steps that do the job well.
- Minimal visual system: a small deliberate scale of fonts/colors/spacing.
  Build from reusable, isolated components. Responsive and accessible by
  default — real contrast, keyboard reachability, sensible focus, honest error
  states over decoration. Fast feedback (live progress, time-remaining).

## Workflow
- **Commit after every logical change** — small, focused, one unit per commit.
  Plain, honest messages. Never commit secrets (`.env` stays ignored).
- **Everything lives on GitHub:** https://github.com/stageerdman/local-transcriber.
  Push regularly; nothing important stays local-only.
- **Roadmaps in phases** for any non-trivial work — design phases first, record
  them in the update's `update vX.md`, track status as you go, most phases end
  with tests.
- **Research spikes** for new APIs/libraries/designs run *outside* the main code,
  inside the update folder, in isolation; capture findings in the update's
  `wiki.md`.
- **Orchestrate:** prefer delegating to focused agents over doing everything
  inline; decompose, fan out, synthesize.
- **Verify by running the actual app**, not just green tests, before moving on.
