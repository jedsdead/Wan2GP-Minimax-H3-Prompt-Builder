# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.6.0] — 2026-08-13

### Fixed

- **WanGP compatibility requirement was blocking installs on unrelated
  versions.** `wan2gp_version` in `plugin_info.json` had been set to the exact
  WanGP release current at packaging time (12.452), but WanGP's plugin manager
  treats that field as a *minimum required version*
  (`is_wangp_compatible`), not a "built against" label. The plugin uses no
  version-specific APIs - nothing beyond `insert_after`, `request_component`
  and `request_global`, stable throughout this project - so there was never a
  genuine reason to require any particular release.
- Clearing the field to an empty string did not fix this: WanGP's catalog
  merge (`_merge_entry_fields`) treats a blank local value as *absent* and
  refills it from a cached or remote copy of the plugin's metadata, which can
  still hold the old requirement. The field is now `"1.0"` - a real, non-blank
  value low enough to be satisfied by any realistic WanGP release, so it wins
  the merge outright rather than falling through to a stale cache.

### Notes

If this is still reported after updating, WanGP may be reading a cached entry
from `plugins_local.json` in the WanGP root rather than this file. Closing
WanGP, deleting that plugin's entry from `plugins_local.json` (or the whole
file, which regenerates), and restarting clears it.

## [1.5.0] — 2026-08-08

### Added

- **Character creator**, per cast/subject entry. A "Use character creator"
  checkbox above the Description field reveals name plus eight dropdowns —
  ethnicity, gender, age range, height, build, hairstyle, hair colour, eye
  colour, all typeable — and an **Add to description** button that composes
  and writes a physical-description sentence:

  > John, an Asian male in his mid-40s, six feet tall with a muscular build,
  > long straight black hair and brown eyes.

  Every field is optional and the grammar adjusts to whatever is filled in -
  dropping fields removes them cleanly rather than leaving a gap, a stray
  comma, or a missing connector. `bald` and `shaved head` become noun phrases
  ("a bald head") so they read as list items alongside build and eyes rather
  than a stray adjective; non-binary and androgynous genders get "person"
  appended since they function as adjectives, unlike male/female. The button
  overwrites the Description field.
- **Speaker names in the beat "Who" dropdown.** Once a character creator name
  is set, every beat's Who dropdown shows `Subject 1 (John)` instead of the
  bare `Subject 1`, purely as a memory aid when juggling several subjects. The
  stored value and the final prompt are unaffected — it's still
  `Subject N` / `(Sx)` throughout.

## [1.4.1] — 2026-08-08

### Fixed

- **The panel appeared for every model.** The visibility wiring was bound to a
  component that could never work: WanGP's visible model dropdown is created
  *after* it snapshots its locals for the plugin component registry, so it was
  never requestable and the handler silently never attached. It now binds to
  the two hidden trigger components that do fire on a model change, and reads
  `model_type` out of `state`.

### Notes

Frame injection was investigated and is out of scope. It's a generation
parameter — a `frames_positions` field of frame numbers, or `L` for the end of
the current sliding-window segment — that injects from the reference images.
Nothing about it touches the prompt text.

## [1.4.0] — 2026-08-08

### Added

- **Start image / End image checkboxes**, beside the reference-mode switch.
  They write the keyframe instruction line the guide requires, reproduced
  verbatim — including its own inconsistency, where the start-and-end form
  uses plain `Picture 1` and `Shot 1` while the other two use `<Picture 1>`
  and `[Shot 1]`. The shot index tracks your actual final shot and the
  duration is formatted to two decimal places. End-only correctly uses
  `<Picture 1>`, not Picture 2.
- **Insert as sliding window.** Appends the built prompt below whatever is
  already in the prompt box, separated by a blank line — build one window,
  insert it, write the next. Since each assembled prompt has no blank lines of
  its own, the separator is unambiguous.
- **Clear shots and beats**, leaving cast, scene, audio and summary intact,
  since those usually carry over between windows while the action does not.

### Changed

- **Duration** is labelled as the duration of *this window*, because
  sliding-window timing restarts at zero for each window.
- The keyframe note no longer explains where to describe start and end images,
  since the instruction line now states their alignment. It says instead that
  referring to `<Picture 1>` inside your own descriptions is up to you.
- `wan2gp_version` corrected to 12.44.

### Notes

**Insert as sliding window** needs *How to Process each Line of the Text
Prompt* set to the paragraph-per-sliding-window option. On the default queue
setting each window becomes a separate job instead.

## [1.3.2] — 2026-08-08

### Added

- **Worked example in every text field.** The greyed-out placeholders now form
  one coherent prompt end to end — a fishmonger gutting a fish, across two
  subjects, two shots and two beats each, with one spoken and one silent beat
  per shot. Copying them through produces a valid prompt, which makes the
  expected grammar and structure readable without leaving the panel. Slots
  beyond the example get generic hints.

### Fixed

- The **accent** field appended the word "accent" unconditionally, so "a faint
  West Country accent" came out doubled. It now recognises the word and common
  synonyms — lilt, brogue, drawl, twang, burr, inflection, cadence, dialect —
  and leaves those alone, while a bare "West Country" still gets it added.

### Notes

Dropdowns can't show placeholder text, so style, location, lighting, grading,
camera and rig aren't covered by the example. It assumes a documentary style in
a covered market hall under harsh fluorescent light, shot on Super 35mm.

## [1.3.1] — 2026-08-07

### Changed

- **Removed the "No non-diegetic music" checkbox.** The section now derives
  from its own fields: `N/A` when the presets and free text are both empty, and
  whatever you entered otherwise. The checkbox defaulted to ticked, so filling
  in the music field without noticing it produced `N/A` anyway.

## [1.3.0] — 2026-08-07

### Changed

- **Base modes now write the three-field schema.** With reference mode off the
  output is `integrated_multimodal_description`, `overall_soundscape` and
  `non_diegetic_music` — no `subject_definitions`, `summary` or
  `retention_analysis`, matching MiniMax's base guide. Reference mode still
  writes the six-section format.
- **Speaker identity is written inline in base modes.** Without
  `subject_definitions` there is nowhere to define a subject, so a speaker's
  description appears at their first mention and `(S1)` is used after that.
  Entries with no speaker have no ID to fall back on, so their description is
  emitted inline each time — the shot anchor is usually the better home for
  scenery in base modes.
- **The summary is a reference-mode section** and hides with the switch. Base
  modes have no summary field.

### Fixed

- An inline subject starting a sentence wasn't capitalised, giving *"The camera
  pushes in. a private detective…"*. Bracketed IDs like `(S1)` are left alone.

### Notes

Interface labels still read "Subject 1", "Subject 2" in the cast list and the
beat **Who** dropdown regardless of mode. These are row identifiers rather than
output labels — in base modes the word never reaches the prompt.

## [1.2.1] — 2026-08-07

### Added

- **Colour grade** dropdown in the Scene row, written into the opening line.
- **Score genres and screen-music styles** at the top of the music presets —
  orchestral, thriller, horror, synth, western, noir jazz, period, war, heist,
  spy, superhero, documentary, sitcom, anime, silent film and more, ahead of
  the texture entries. 72 presets in total.
- A standing note above the buttons, and a line in the status message, asking
  you to read the prompt for grammar before generating — fields are stitched
  into sentences from templates, so your wording may not agree with the
  phrasing around it (*"a woman faces the door"* where *facing* was needed).

### Changed

- **One reference-mode switch.** The per-entry "Is a referenced subject"
  checkbox is replaced by a single **Use reference mode (Ref2VA)** checkbox at
  the top, which reveals the reference block on every entry along with the
  Reference task section. Unticked, none of it is written.
- **Source video moved inside Reference task**, so it appears and disappears
  with the rest of the reference options.
- The task-type reminder now fires only when reference mode is on, rather than
  whenever a description happened to look like a reference.

### Fixed

- A beat with no time set no longer emits `At 00:00.000`. Empty number inputs
  can come back as `None`, `""` or `0`, and a beat at zero is the start of the
  clip, which the shot already implies — all three now count as unset.
- The **audio reference fields** for soundscape and music were the last
  reference controls left permanently visible. They now follow the reference
  switch like everything else, and are ignored by the assembly when it's off.
- Those fields sat *above* their own textboxes; they now sit beneath them.
- The **Draft summary** notes no longer mention task types or source videos
  when reference mode is off, since neither applies there.
- The end-image note said "the final beat of the last shot", which doesn't help
  if that shot has no beats. It now says the end of the last shot — its final
  beat, or its anchor.

### Changed

- The soundscape and music free-text fields are labelled **custom / additional
  information** rather than "added after the presets".

## [1.2.0] — 2026-08-07

### Added

- **Audio and video reference linking.** Each cast/subject entry gained **Voice
  from** and **Motion from**, naming a reference audio or video slot directly.
  These produce the spec's own phrasing — `<Audio 1> is the voice-timbre
  reference for <Subject 1> (S1).` — along with a matching retention line.
- **Audio references for soundscape and score.** The Audio section gained
  **Ambience from** and **Music from**, each with a retention marker, and the
  music reference names what it controls: style, beat and rhythm,
  instrumentation or mood.
- **One sentence per asset.** When a single reference serves several roles, the
  roles are gathered and written as one natural sentence rather than a
  subsection each, as the guide requires.
- **Is a referenced subject (Ref2VA only)** checkbox on each entry. Reference
  fields stay collapsed until it's ticked, and values left in a collapsed block
  are ignored rather than leaking into the prompt.
- **Source asset is a dropdown** offering `Picture 1`–`Picture 9`, still
  typeable, and accepting several at once.
- Warnings on insert when audio is referenced without an audio task type, and
  when both the source video's soundtrack and standalone audio slots are set —
  WanGP's single Audio References dropdown makes those alternatives.

### Fixed

- **Reference tags now get angle brackets.** H3 only treats `<Picture 1>` as a
  reference; plain `Picture 1` is read as ordinary words, so source assets were
  silently not registering. Already-bracketed input is left alone.

### Changed

- **Summary moved to the bottom**, directly above the Insert button.
- **Removed the "No retention analysis" checkbox.** The section now derives
  entirely from the data: retention lines appear when a referenced subject has
  both a source and a marker, and `N/A` falls out on its own otherwise. The
  checkbox could previously override real retention data to `N/A`.
- Reference video and audio slots limited to **two each**, matching what
  WanGP's selectors offer rather than the model's ceiling of three.

## [1.1.1] — 2026-08-02

### Added

- **Language per speaker**, defaulting to English and accepting typed values.
  The beat-level language became an override.
- **Rig, as a third camera axis** alongside framing and motion — tripod,
  handheld, steadicam, gimbal, dolly track, crane, jib, drone, cable cam, car
  mounts, slider, motion control, underwater housing. It reads as a trailing
  clause: *"The camera pushes in slowly, on a steadicam."*
- **A few movements the guide's table lacks** — full 360 orbit, spiral, dolly
  zoom, crash and snap zooms, whip pan, rack focus, trail and lead. Moves named
  after rigs are deliberately absent, since a dolly in is Push In and a 360 is
  an Arc Shot.
- **Many more style presets**, covering live-action genres, movements and eras
  alongside animation and illustrative styles.
- Expanded presets throughout: framings, cut verbs, locations, lighting,
  atmosphere, soundscape, music, voice timbre, rate and age.

### Changed

- The first beat of a shot takes a timestamp again. A shot cutting at 4s can
  have its first action at 4.5s.

## [1.1.0] — 2026-08-02

First public release.

### Changed

- **One form, no modes.** Keyframe images are attached in the generator and
  associate with the prompt positionally, so there was nothing for the builder
  to configure — only guidance on where each gets described.
- **Cast and Reference subjects merged** into a single list. Everything that
  appears is one kind of entry; only a description is required.
- **Subject definitions populate whether or not references are used**, and the
  shot text refers to the label instead of repeating the description.
- **Beats attribute to subjects** and accept several at once, with speaker IDs
  appended automatically.
- Output uses the six-section shape throughout.

### Added

- **Explicit speaker assignment**, set per entry rather than derived from list
  position.
- **Per-beat timestamps** for events inside a shot.
- **Draft summary from fields** button.
- **Retention (audio)** on the source video, separate from picture.
- **Scene fields** — location, lighting, atmosphere — plus camera body, written
  as a single opening sentence before `[Shot 1]`.
- **Per-shot lens**, since lenses commonly change at a cut.
- **Preset dropdowns** for soundscape and non-diegetic music.
- **Duration** stated in the opening line.
- **Validation warnings** on insert.
- **Add and remove buttons** rendered beneath the last visible item.
- Model-based visibility, hiding the panel for non-MiniMax models.

### Fixed

- Speaker IDs were dropped on action beats.
- The **Speaks** checkbox on reference subjects was collected but never used.
- Task type and summary were reachable only by opening the Reference subjects
  section.
- Entries with no description and no source produced a bare `<Subject 1>.`
- Several dropdowns had a default value absent from their own choice list.
- **Clear all fields** reset counts but left slot panels visible.
- The number input stepped inconsistently.
- Blank lines between sections split the prompt into separate generations.
- Assorted grammar in generated prose.
