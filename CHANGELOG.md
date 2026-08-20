# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [3.0.0] — 2026-08-19

The shots and beats panel is gone. In its place is one Action field and a
toolbar that writes into it.

The old panel pre-created `MAX_SHOTS × MAX_BEATS` Gradio components and hid
most of them, then assembled the prompt from templates. That worked, but it
made every sentence a fill-in-the-blanks exercise: phrase something the
template did not expect and you got prose that read as though it had been
generated, because it had. Editing the result meant editing the prompt box
after the fact, and the next Insert threw the edit away.

Now the buttons write the same correctly formatted text into a field you can
edit in place, and the build reads that text back. The formatting the guides
fix — `[Shot N]`, `<d>` blocks, `<scenetrans>` at both connecting points,
`<cutoff>`, the voiceover phrasing and its closed-lips clause — is still
written for you. Everything around it is now yours.

The flat input list shrank from 506 values to 165, and the six construction
sites that had to be kept in step became four.

### Added

- **The Action field and its toolbar.** One text box holds the whole action.
  **Shot** starts a new line, everything else appends to the line you are on,
  so a shot is built left to right and then tidied by hand.

  Nothing is remembered between presses. Shot numbering, whether a camera
  press is a cut or a move within the shot, and whether a carried line is
  waiting to be picked up are all worked out by reading the field back. Delete
  a shot by hand and the next press agrees with what is left, which a counter
  could not do.

- **Late speaker binding.** The buttons always write the bare `(S1)`. Which
  form it takes in the finished prompt — an inline description in base modes,
  a `<Subject 1>` label in reference mode — is decided at build time, so the
  reference switch keeps working after the action is written.

- **A validation pass on Insert.** Warnings only, never a block; a freeform
  field is allowed to be a work in progress. It catches unbalanced *and*
  nested `<d>` tags, `<d>` blocks with no `[Language]` tag, repeated or
  out-of-order or missing shot numbers, empty shots, timestamps past the clip
  duration or out of order, timestamps written in a form nothing can read
  (`At 3s,`), a carried line with no pick-up, `<cutoff>` anywhere but the very
  end, unclosed quotes around visible text, and speaker IDs with no entry in
  Cast & subjects.

  That last one is new territory: a `(S7)` with no Subject 7 used to survive
  into the finished prompt as a bare `(S7)`, meaning nothing to the model and
  invisible to read past.

- **The draft is saved to disk as you work.** WanGP can go down mid-build, and
  a half-written prompt is an hour of work. The whole form is written to
  `h3_draft.json` beside `plugin.py` after every press that changes something,
  and on a 20-second timer for the typing in between. **Restore last draft**
  puts it back; **Save draft now** forces a write.

  An automatic save never replaces a draft with an empty form — a restart
  brings the panel up with every field at its default, and the timer would
  otherwise write that over the draft twenty seconds later. Every write also
  keeps the draft it replaced as `h3_draft.prev.json`, which covers the case
  where you start typing before remembering to restore.

  The file holds the flat list positionally and records its length. Naming
  every field would be another construction site to keep in step, and getting
  it wrong there would restore your lens into your anchor — so a draft written
  by a different version of the plugin is refused outright rather than shifted
  into the wrong slots.

- **A Clear all fields button at the top of the panel**, alongside the draft
  controls, so starting over doesn't mean scrolling to the bottom. The one at
  the bottom stays. Clearing is saved too, so a crash after it can't
  resurrect the old form.

- **Every section folds.** Mode & keyframes, Scene, Cast & subjects, Action,
  Audio, Reference task and Summary are all accordions, as are the Camera,
  Dialogue and Visible text groups inside Action and each cast entry on its
  own heading.

- **Time of day.** Its own field, separate from lighting — lighting is how the
  scene is lit, this is when it happens. It reaches the opening sentence and
  the audio digest, where dawn and midnight imply very different sound.

- **Keep the enhancer loaded.** The `ENHANCER_KEEP_LOADED` constant is now a
  checkbox. It still only governs a copy the plugin loaded itself; an enhancer
  borrowed from WanGP is WanGP's to release.

- **Undo last insert**, and a **Clear the action** that preserves an open
  carry. If a line was still carrying when the action was cleared for the next
  sliding window, the pick-up half is left behind as ordinary visible text —
  the window boundary is itself a cut, and its emitting half is in the window
  before. It is left where it can be read and deleted rather than remembered
  in a flag.

### Changed

- **The opening shot takes no transition verb.** There is nothing before
  `[Shot 1]` to cut from, so leaving **Transition** blank there now writes
  `A medium shot of the detective at his desk, on an 85mm lens.` rather than
  putting a camera cut in front of it. Blank still picks "cuts to" for a later
  shot and "moves to" for a second camera press within one, and an explicitly
  chosen verb is always used as picked.

- **The audio suggestion is two buttons.** Asking for the soundscape and the
  score in one request made the enhancer hold two jobs in mind at once, and the
  weaker ones drifted between them — score notes in the soundscape, or the
  reverse. One field per request costs a second press and reliably answers the
  question that was asked.

- **The audio digest reads the action.** It pulls the anchor back out of each
  camera sentence and drops the rest, since framing and lens say nothing about
  sound, and it strips continuity prose, the closed-lips clause and the spoken
  words themselves.

- **"Appears in shots" is derived, not typed.** Which shots a subject appears
  in is read out of the action by speaker ID, so it cannot go stale when a shot
  is renumbered or deleted.

- **Location presets no longer carry time-of-day tails.** "a farmhouse kitchen
  at dawn" is now "a farmhouse kitchen", so picking both fields cannot produce
  "a rooftop at dusk at night". Two launderette entries collapsed into one.

- **"Clear shots and beats" is now "Clear the action."**

### Removed

- The fixed shot and beat slots, `MAX_SHOTS`, `MAX_BEATS`, `shot_count`, the
  per-shot and per-beat field groups, and the `_shot_text` assembler.

### Fixed

- **Add dialogue raised a Gradio error** once a subject had a name typed into
  the character creator. The handler that shows that name next to the Subject
  number returned its update wrapped in a list, which was correct when there
  were thirty per-beat dropdowns to update and wrong once there was one.
  Gradio reads a returned list as the component's *value*, and a
  `CheckboxGroup` value is a list, so the update dict was stored as a selected
  item and failed when the field was next read back.

- A subject's inline description is capitalised when it opens a sentence. The
  old assembler capitalised as it built; substituting into freeform text, the
  binding step has to do it.

## [2.0.0] — 2026-08-19

The first release where the plugin writes prose it wasn't told to. The Audio
section can now hand the scene to WanGP's own Prompt Enhancer and have it fill
in the soundscape and score, which is a different kind of feature to everything
before it: the rest of the panel assembles what you typed, this asks a model
what you left out.

Alongside it, three pieces of the spec that the builder had been leaving to you
are now written properly — voiceover phrasing, `<scenetrans>` on both sides of
a cut, and `<cutoff>`.

### Added

- **Suggest soundscape and music from the scene.** A button in the Audio
  section reads what you have built and asks WanGP's Prompt Enhancer what it
  should sound like, writing the two custom text boxes and leaving the preset
  dropdowns alone — so a suggestion is always undone by clearing one field.

  The soundscape is drawn from location, atmosphere, style, every shot anchor
  and the action in every beat, plus any soundscape presets and notes already
  set. The score is drawn from style, music presets and music notes. Where a
  choice is already made the model is told to build on it and fill the gaps
  around it; where none is, it works one out from the scene.

  Camera work is withheld deliberately. Framing, lens, motion and rig say
  nothing about what a scene sounds like, and feeding them in pulls the model
  towards describing the shot instead. Dialogue is reported as "someone speaks
  aloud here" without the words, so there is nothing for it to echo back into a
  field that must not contain any.

  Requires the Prompt Enhancer enabled and set to a Qwen 3.5 variant; the Llama
  3.2 and JoyCaption options are captioning models and will not follow the
  instruction. The bridge borrows the enhancer WanGP already holds where it
  can, and loads one at the configured level where it cannot, releasing it
  afterwards. `ENHANCER_KEEP_LOADED` in `plugin.py` keeps a self-loaded copy
  resident between presses, trading VRAM for speed.

- **Voiceover beats.** A third beat type alongside action and dialogue, and the
  only one that changes the output rather than labelling it. The spec fixes
  both the clause and its follow-up, and both are now written:

  > The man (S1) says in an off-screen voiceover: `<d>`[English] I still
  > remember that road.`</d>` while his lips remain completely closed.

  Anything in the delivery box is kept as a preceding action, with a trailing
  "says" stripped so it cannot collide with the required phrase and produce
  "says quietly and says in an off-screen voiceover". The pronoun comes from
  the entry's voice gender, defaulting to "their"; several speakers sharing a
  line take the plural verb, matching the guide's group-speech examples. A
  speaker whose **Presence** is set to off-screen gets no lips clause, since
  the guide's wording is about the on-screen character and there is no visible
  face to describe.

- **`<cutoff>`.** A per-beat checkbox for speech the clip ends in the middle
  of. Without it the model treats the line as something that must complete
  inside the duration, so it either rushes the delivery or lands on an
  unnatural silence. Works on dialogue and voiceover alike, and is worth having
  with sliding windows, where a line deliberately runs past the end of one
  window and picks up in the next.

- **On-screen text**, per shot. A carrier dropdown — sign, neon sign, banner,
  subtitle, phone screen, licence plate, departure board and so on — and a text
  field. The words are quoted verbatim in English double quotation marks and
  never translated, per the guide's on-screen text rule. Quotes typed around
  the text are stripped rather than doubled up.

- **Clothing in the character creator.** A preset list plus free text, added
  last so the sentence reads as a person first and an outfit second:

  > John, an Asian male in his mid-40s, six feet tall with a muscular build,
  > long straight black hair and brown eyes, wearing a rumpled trenchcoat.

  Presets are phrased to follow "wearing", which the composer supplies;
  anything typed that already carries its own verb ("dressed in a long evening
  gown") is used as written. Clothing alone yields "A person, wearing a police
  uniform."

### Changed

- **`<scenetrans>` is written at both connecting points.** The guide asks for
  the tag on both halves of a line crossing a cut *and* for the continuity to
  be stated in words; the builder emitted a bare tag on one side only, which is
  half of it. The originating beat now carries the tag plus one of the guide's
  three forward-looking phrases, chosen from a **How it carries** dropdown, and
  the receiving shot writes its own matching half automatically. "Carries over
  from the previous shot" is reserved for the receiving side and so is not
  offered in the dropdown.
- The music instruction sent to the enhancer follows guide §4.7:
  instrumentation, speed, rhythm and dynamics, with mood words and
  explanations of the score's emotional function ruled out.

### Fixed

- **Audio suggestions worked once and then never again.** The Qwen enhancers
  reason before answering, and once the scene grew past a certain size the
  reply was truncated mid-`<think>` — no closing tag, no JSON, nothing to
  parse. Four changes: the token budget is up from 320 to 1024; thinking is
  suppressed three ways (the keyword argument, the flag set on the model
  object and restored after, and `/no_think` appended to the text, because a
  model borrowed from the offload pipe carries the flag WanGP set at load time
  rather than the one passed in); an unclosed reasoning block is now handled by
  dropping everything from it onward; and a reply that parses to nothing
  retries once with a blunter prompt that forbids reasoning outright.
- **A well-formed reply that ignored the requested format parsed as nothing.**
  An unlabelled answer is now read as one line for the soundscape, or two for
  the soundscape and the score — guarded so that refusals and preamble are
  rejected rather than written into the fields as sound design.
- **Parse failures were opaque.** The full reply now goes to the console and
  the first 300 characters are quoted in the panel, which is the difference
  between diagnosing this and guessing at it.
- **The enhancer was reported as not loaded when it plainly was.** It is
  registered in mmgp's offload pipe under `prompt_enhancer_llm_model` — the
  "Hooked to model" line in the console — rather than held in a `wgp.py`
  global, which is where the first implementation looked. It is now found there
  first, with the module global and an on-demand load as fallbacks, and the
  status line ends with a probe report saying which routes were open.

### Notes

Two known limitations are retired: the voiceover clause the spec requires after
`</d>` is emitted, and `_beat_text` now has a slot after the speech for it.

The audio suggestion depends on WanGP internals that are not a documented API.
Every step is probed and every failure surfaces as text in the panel rather
than a traceback, but a WanGP update could still break it.

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
