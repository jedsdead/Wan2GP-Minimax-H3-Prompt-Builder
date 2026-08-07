# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
