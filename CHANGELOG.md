# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] — 2026-08-02

First public release.

### Changed

- **One form, no modes.** The mode selector has been removed. Keyframe images
  are attached in the generator and associate with the prompt positionally, so
  there was nothing for the builder to configure — only guidance on where each
  one gets described: a start image in Shot 1's anchor, an end image in the
  final beat of the last shot.
- **Cast and Reference subjects merged** into a single *Cast & subjects* list.
  Everything that appears — people, animals, places, props — is one kind of
  entry. Only a description is required; **Speaker** and **Source asset** are
  optional, and each reveals its own fields.
- **Subject definitions populate whether or not references are used.** The
  description drives `subject_definitions`; the source asset drives
  `retention_analysis`. A subject invented from description appears in the
  definitions and contributes nothing to retention.
- **Shot text refers to labels rather than repeating descriptions**, so a
  subject reads identically in every shot. Identity is written once in the
  definitions instead of inline at first appearance.
- **Beats attribute to subjects**, not just speakers, and accept several at
  once. Speaker IDs are appended automatically where they apply, producing
  `<Subject 1> and <Subject 3> (S1,S2)`.
- **Retention analysis** draws only on entries with a source asset, so invented
  subjects correctly yield `N/A` without needing the override.
- Output always uses the six-section shape, so the description field is written
  as `detailed_description` rather than the base schema's
  `integrated_multimodal_description`.

### Added

- **Explicit speaker assignment.** Speaker numbers are set per entry rather than
  derived from list position — if entry 2 is a car and entry 3 talks, entry 3
  can be S2.
- **Per-beat timestamps** for events inside a shot: `At 00:04.000, they clash in
  the centre.` Not offered on a shot's first beat, whose time is already stated
  by the shot's cut time.
- **Draft summary from fields** button, composing a first pass from the shot
  anchor, subjects, dialogue, shot count and source video, written into the
  summary box so it can be edited.
- **Retention (audio)** on the source video, separate from picture and using the
  correct marker set (`fully_copy`, `partially_copy`, `reference`,
  `weak_reference`). Emits its own `<Audio N>` label.
- **No retention analysis** checkbox, writing `N/A` for cases where nothing is
  preserved from an asset.
- **Scene fields** — location, lighting, atmosphere — plus camera body, written
  as a single opening sentence before `[Shot 1]`.
- **Per-shot lens**, since lenses commonly change at a cut.
- **Preset dropdowns** for soundscape and non-diegetic music, combining with
  free text.
- **Duration** stated in the opening line: *"across a 10-second duration"*.
- **Validation warnings** on insert for an empty summary, entries with no
  description or source, or no anchor on any shot. The task-type warning fires
  only when reference assets are genuinely in use.
- **Add and remove buttons** for shots, beats and subjects, rendered beneath the
  last visible item rather than at the top of the section.
- Model-based visibility: the panel hides itself when a non-MiniMax model is
  selected, where the selector component can be found.

### Fixed

- Speaker IDs were dropped on action beats, so `(S1) turns and looks` lost its
  ID. Dialogue is optional; an ID is emitted whenever a speaker is selected.
- The **Speaks** checkbox on reference subjects was collected but never used.
- Task type and summary were reachable only by opening the Reference subjects
  section, despite being needed independently of it.
- Entries with no description and no source produced a bare `<Subject 1>.` and a
  retention marker for content the model was never shown. They are now skipped
  and reported.
- Several dropdowns had a default value absent from their own choice list, which
  raised an error on insert and on **Clear all fields**.
- **Clear all fields** reset the counts but left slot panels visible, because a
  programmatic value change doesn't re-fire a component's own change handler.
- `weak_reference` appeared twice in the combined retention list.
- The number input stepped inconsistently, because HTML anchors its step grid to
  `min` and the two were misaligned.
- Blank lines between sections split the prompt into separate generations in
  WanGP. Sections are now written on consecutive lines, and blank lines typed
  into any field are stripped.
- Assorted grammar in generated prose: articles before vowel sounds, a doubled
  "shot" when the framing already contained the word, mid-sentence
  capitalisation after a speaker, and prepositions that only suited interiors.

### Known limitations

- Beats can't be reordered after the fact; remove and re-add to change the
  sequence.
- Beat timestamps aren't validated against the clip duration or against each
  other.
- The panel hides on model change only — loading WanGP with another model
  already selected leaves it visible until you switch.
- Reference labels are typed by hand and aren't checked against the generator's
  actual reference slots.
- Voiceover has required phrasing in the spec, including a follow-up clause
  about the speaker's lips, which isn't generated yet.
- There's no way to keep a source video's spoken words while replacing the
  voice.
