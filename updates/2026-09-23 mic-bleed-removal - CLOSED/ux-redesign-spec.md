# Track panel UX redesign — synthesized spec (2026-10-04)

Synthesis of four parallel UX-expert passes (mental-model/IA, trust-flow,
speaker-identity/naming, layout/states/Start-flow). Driven by three verbatim user
complaints on the Petr Sochůrek call:
1. "it unselected both tracks and only kept one track, that doesn't make sense."
2. "it says that track Person 3 is the same as track Person 1" (confusing).
3. "very confusing — we need a huge UX upgrade, make everything clear."

## Root-cause bug behind complaint #1 (verified in code + on real audio)
The two real speakers come from a *specific* track combination: keep the clean
track (3) + isolate a mix (1) using track 3 as reference. If the user hand-picks
"2 tracks for 2 people" without the clean reference track, `resolve_shared_voice`
can only recover one speaker and collapses to one. Measured:
`select [1,2]→keeps [1]`, `[2,3]→keeps [3]`, `[1,3]→keeps [1,3]`, `[1,2,3]→keeps [1,3]`.
The track→speaker mapping is counterintuitive and invisible. **Fix: the user
picks/confirms PEOPLE, not raw tracks; the app owns the track mapping.**

## The reframe (all four experts agree): Recordings in → People out
Lead with the RESULT. The panel answers one question: *"who will I get, and
where did they come from?"* Tracks become evidence, demoted behind disclosure.

## Convergent design decisions
1. **Header states speakers, not tracks.** `2 speakers will be transcribed · 1 track skipped`
   (not "3 detected, 2 included"). This alone defuses "it only kept one track."
2. **Kill `Person N`.** It's a raw ffmpeg index and the source of complaint #2.
   Replace with first-speech-ordered `Speaker N` ghost/placeholder text; use the
   embedded ffmpeg stream **title** metadata when present (solid, "from file").
   Add recent-name suggestions (MRU in settings) so "Klient"/"Obchodník" stick.
3. **Nothing vanishes silently.** A skipped/dropped track stays as a dimmed row
   *in place* with a plain reason + one-click `Transcribe anyway`. Never flip a
   checkbox with no trace. (Fixes complaint #1's "it broke my selection.")
4. **Directional, named, zero jargon.** Never "X is the same as Y". Say
   "Client's voice also bled into the Salesperson's track — we'll transcribe
   Client once and skip the copy." Ban isolate/subtract/drop/mix/component from
   the UI. Each track role → a human sentence (keep / isolated / duplicate).
5. **Calm, already-handled framing** (auto-apply is now the worker default): a
   quiet green `✓ Set up to transcribe each person once` card, not a bold ⚠
   alarm. Amber only when genuinely unresolved OR the user re-created doubling.
6. **Confidence-aware.** High-confidence match → auto-applied (green). Marginal →
   amber *proposed* card, nothing mutated until the user chooses.
7. **Verify by ear.** `Listen: Track 1 ⟷ Track 3` A/B preview on the card and
   per row (reuse the detection window + TrackPlayer). Makes the claim checkable.
8. **Delete the askyesnocancel modal.** The card is the pre-flight. Start just
   runs; if the user deliberately left doubling, an inline amber strip next to a
   dynamic CTA (`▶ Transcribe 2 speakers`) offers `Fix it` / `Transcribe anyway`.
9. **Progressive disclosure.** Default = the answer (the people). Mechanics
   (language, noise, remove-a-voice, channel) live behind the per-row ▸ drawer.
10. **Accessibility.** State carried by glyph + word, never color alone; grid
    row layout so columns align; keyboard-reachable include/preview; honest
    error state.

## Listen→label rhythm (naming becomes ~3s/track)
▶ seeks to first speech (not 0.0); pressing ▶ focuses that row's name field;
Enter commits + auto-advances ▶ to the next unnamed track. `Listen to each →` in
the header kicks it off.

## Per-track states (one enum, from existing data)
clean · isolated (`◐ Cleaned`) · skipped-duplicate (`⊘ Skipped`) ·
skipped-shared (`⊘ Skipped`) · excluded-by-user (`○ Off`) · conflict (`⚠ Same voice`) · playing.

## Code map (app/ui.py)
- `_update_tracks_header` → speaker-count summary.
- `_render_tracks_section` → grid rows + chip col + reason line; skipped rows
  rendered dimmed in place (don't just uncheck).
- new `track_state(job, stream)` helper; fold in `_track_status_summary` +
  `_shared_voice_partner_name` (delete the symmetric "same voice as" path).
- `_render_shared_voice_banner` → plan **card** (green/amber, `Listen ⟷`, `Undo`).
- `_shared_voice_plan_text` → named, directional, jargon-free sentences.
- duplicate side-effect → render as skipped row, not a silent uncheck.
- `_start_job` → inline pre-flight strip + `allow_doubling` latch; **delete
  `_confirm_shared_voice_start`** (the modal).
- bottom primary CTA `▶ Transcribe N speakers`.
- `src/audio_tracks.track_speaker_names` → `Speaker N` (first-speech order) +
  title metadata; naming MRU in settings.

## Highest-leverage first slice (if phasing)
Kill `Person N` (→ named/`Speaker N`) + named directional language in the
card/rows + speakers-count header + skipped-in-place (no silent uncheck). That
subset alone makes every sentence intelligible and the outcome predictable —
the literal complaints — before the richer card/preview/CTA work.

---

## Implementation status (2026-10-04)

SHIPPED (committed, verified via headless Tk render + interaction smoke + 112 tests):
- Ph1: default names Person N -> Speaker N (src/audio_tracks + worker + ui).
- Ph2: people-first header ("N speakers will be transcribed · M skipped");
  per-track state model (clean/isolated/skipped_dup/excluded/conflict) driving a
  chip + plain-language reason line + inline Include/Undo links; skipped tracks
  dimmed in place (never silently removed).
- Ph3: auto-apply the keep/isolate/skip plan on detection; calm GREEN plan card
  ("✓ Set up to transcribe each person once — You'll get 2 speakers: …" +
  "Transcribe every track instead"); AMBER only when the user reverts and a voice
  really would double. Directional, jargon-free copy throughout.
- Ph4: deleted the askyesnocancel Start modal; no-expand Start auto-applies too.
- Ph5: recent-name suggestions (editable combobox, MRU in settings).
- Ph6: previewing a track focuses its name field (listen->label).

DEFERRED polish (low-risk, documented for a later pass):
- First-speech ordering of default Speaker-N labels; ghost-text placeholder
  styling; "from file" provenance marker when a stream carries a title tag.
- Dedicated A/B "Listen: Track 1 ⟷ Track 3" button on the card (per-row ▶ covers
  verification for now).
- Dynamic bottom CTA ("▶ Transcribe 2 speakers"); grid-aligned row columns
  (kept pack layout + checkbox per expert-2's recommendation).
- Enter-commits-and-advances-to-next-unnamed-track rhythm.

VERIFY GATE (per CLAUDE.md "run the actual app"): headless render + interaction
smoke pass; a human run of `python -m app.main` on the Petr Sochůrek call is the
final confirmation before closing.

---

## v2 — RADICAL simplification (2026-10-04, after "be Steve Jobs" feedback)

The v1 redesign was still too noisy (chips + reason lines + card = text clutter)
and a real bug collapsed 2 speakers to 1. Three ruthless-simplicity UX passes →
rebuilt the default view from zero.

SHIPPED:
- **Bug fix (load-bearing):** the envelope-correlation `duplicate_of` check was
  auto-unselecting tracks; when one loud voice dominates all tracks their
  envelopes match, collapsing a 2-speaker call to 1 ("it only selects one track").
  Removed the auto-unselect; the waveform-accurate shared-voice plan is now the
  sole authority on what to drop (keeps both speakers).
- **Default view = PEOPLE, not tracks:** one row per resulting speaker (preview
  button + editable name, pre-filled from recent names). No header, no plan card,
  no checkboxes, no chips, no reason lines, no waveforms, no cleanup text.
  Cleanup is silent-correct.
- **One escape hatch:** "Not these speakers? Edit tracks" → the full raw-track
  editor (the entire v1 detailed view) in place, "‹ Back to speakers" to return.
  All mechanics (language, noise, remove-voice, channels, include/exclude, the
  plan card) live there, two clicks deep.
- Single combined recording → no panel. Preview selects the name for instant
  rename. No cleanup toggle (always on; reversible in the editor).

Verified: headless render smoke across states (speakers view / editor / clean /
reverted) + interaction harness + 112 tests. Final gate: human run of the app.
