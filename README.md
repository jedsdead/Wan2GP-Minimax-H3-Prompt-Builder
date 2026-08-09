# MiniMax H3 Prompt Builder — a WanGP plugin

MiniMax H3 doesn't take a free-form prompt. It expects a structured document
with named fields, speaker IDs, timestamped shots, tagged dialogue, and
angle-bracket labels for every reference you attach. Getting that right by hand
is fiddly and easy to break in ways that fail quietly.

This plugin adds a panel beneath the prompt box in WanGP that builds the whole
thing — fill in shots, cast, references and audio, and it writes the fields,
tags, timestamps and boilerplate in the format the model expects.

Format follows MiniMax's own guides:
[base modes](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md)
and [full reference](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md).

> The prose templates are tuned but not exhaustive. Read what it produces
> before committing to a long generation.

---

## Installation

**From the plugin manager**

1. Open WanGP → plugin manager
2. Paste this repo's URL into the install-from-URL field
3. Enable **MiniMax H3 Prompt Builder** in the plugin list
4. Restart WanGP

**Manual**

Clone into your WanGP `plugins/` folder so you have:

```
Wan2GP/plugins/wan2gp-h3-prompt-builder/
├── plugin.py
└── plugin_info.json
```

Enable it in the plugin list and restart. No extra dependencies — the plugin
only uses `re` and `gradio`.

---

## What it writes

**Base modes** — three fields:

```
integrated_multimodal_description: A film noir scene, set in an office at night, lit by shafts of light through blinds, with a high-contrast black and white grade, shot on Super 35mm film, across a 10-second duration.
[Shot 1] A medium shot on an 85mm portrait lens frames the detective at his desk. The camera pushes in. A private detective in a rumpled trenchcoat, a middle-aged male with a low, gravelly, measured voice (S1) sits back and says: <d>[English] She walked in like trouble.</d> At 00:05.000, (S1) turns to the window.
overall_soundscape: The scene carries rain on a tin roof.
non_diegetic_music: A noir jazz score with muted trumpet and brushed drums.
```

**Reference mode** — six sections, with `N/A` in any that doesn't apply:

```
subject_definitions:
<Subject 1> (S1) is a private detective in a rumpled trenchcoat, a middle-aged male with a low, gravelly, measured voice, from <Picture 1>.
<Audio 1> is the voice-timbre reference for <Subject 1> (S1).
summary:
[reference generation + audio reference] The target video shows the detective at his desk, featuring <Subject 1>. It runs 10 seconds in a film noir style.
retention_analysis:
<Subject 1> (appears in [Shot 1]): fully_preserved - the trenchcoat and hat are retained.
<Audio 1>: reference - the voice-timbre reference for <Subject 1> (S1).
detailed_description:
A film noir scene, set in an office at night, lit by shafts of light through blinds, with a high-contrast black and white grade, shot on Super 35mm film, across a 10-second duration.
[Shot 1] A medium shot on an 85mm portrait lens frames the detective at his desk. <Subject 1> (S1) sits back and says: <d>[English] She walked in like trouble.</d>
overall_soundscape:
The scene carries rain on a tin roof.
non_diegetic_music:
A noir jazz score with muted trumpet and brushed drums.
```

In reference mode descriptions live in `subject_definitions` and the shot text
refers to the label, so a subject reads identically in every shot.

## One switch, two schemas

There's no mode selector. A single **Use reference mode (Ref2VA)** checkbox at
the top decides which schema the prompt uses and reveals every reference
control.

**Off** — the base three-field schema:
`integrated_multimodal_description`, `overall_soundscape`, `non_diegetic_music`.
Speakers carry their description inline at first mention and `(S1)` after that.

**On** — the six-section reference schema: `subject_definitions`, `summary`,
`retention_analysis`, `detailed_description`, `overall_soundscape`,
`non_diegetic_music`. Entries become `<Subject N>` labels and the shot text
refers to the label rather than repeating the description.

Keyframe images are attached in the generator; **Start image** and **End
image** checkboxes at the top tell the builder which instruction line to write:

| Ticked | Instruction line |
|---|---|
| Start | `<Picture 1>` is fully referenced at 0.00 seconds |
| End | `<Picture 1>` aligns with the S.SS-second mark of the final shot |
| Both | `Picture 1` at 0.00 seconds, `Picture 2` at the S.SS-second mark |

The shot index tracks your actual final shot, and the duration is formatted to
two decimals. Referring to `<Picture 1>` inside your own descriptions is up to
you — the guide's examples do it in the shot text as well.

With both a start and an end image, a single shot usually works best so the
model can interpolate between them.

## Sliding windows

**Insert into prompt** replaces the prompt box. **Insert as sliding window**
appends the built prompt below what's already there, separated by a blank line
— so you build one window, insert it, then write the next and append that.
Since each assembled prompt has no blank lines of its own, the separator is
unambiguous.

Two things to know. **Duration** is the length of *that window*, since
sliding-window timing restarts at zero for each one. And WanGP's *How to
Process each Line of the Text Prompt* setting must be on the
paragraph-per-sliding-window option — on the default queue setting each window
becomes a separate job instead.

**Clear shots and beats** resets only the action, leaving cast, scene, audio
and summary in place for the next window.

---

## Sections

### Scene

Style, colour grade, location, lighting, atmosphere and camera body — the
things that hold for the whole clip. These become a single opening sentence
before `[Shot 1]`.

Lens and rig live with each shot instead, since both commonly change at a cut.

### Cast & subjects

One list for everything that appears — people, animals, places, props. Each
entry becomes a `<Subject N>` definition, and the shot text refers to the label
rather than repeating the description.

Only the description is required. **Speaker** is assigned by you, not derived
from position: if entry 2 is a car and entry 3 talks, entry 3 can be S2.

Tick **Use reference mode (Ref2VA)** at the top of the panel to reveal the
reference fields on every entry, along with the Reference task section. They
stay collapsed otherwise, and values left in a collapsed block are ignored
entirely rather than leaking into the prompt:

- **Source asset** — which reference image the subject comes from, chosen from
  `Picture 1`–`Picture 9` or typed
- **Retention** and **What is retained** — how much of it carries over
- **Voice from** — a reference audio supplying this speaker's timbre
- **Motion from** — a reference video supplying movement or performance

### Shots

**Add shot** appends one. Shot 1 is the opening and takes no timestamp; later
shots take a cut time and a transition. Each shot has framing, a lens, a camera
triple (motion, amplitude, speed), a rig and an anchor.

Framing, motion and rig are three separate axes: where the camera is, what it
does, and how it's mounted. A dolly in is Push In on a dolly track; a 360 is an
Arc Shot; a drone shot is any motion at all, on a drone.

### Beats

Within each shot, **Add beat** builds an ordered sequence. A beat is:

- **unattributed** — leave **Who** blank: `A bus pulls away.`
- **attributed action** — pick subjects, leave the speech blank
- **dialogue** — subjects, their action and delivery, and the spoken words

Action and delivery go *outside* the `<d>` tag; only the language tag and the
words go inside. Beats appear in the order you add them.

Each beat takes an optional **At (seconds)**, timing an event inside the shot —
`At 00:04.000, they clash in the centre.` Tick **carries across the next cut**
to emit `<scenetrans>`.

### Audio

Two fields, and which one a sound belongs in depends on a single question:
**can the characters hear it?**

- **Soundscape** — ambience, physical sounds, breathing, laughter
- **Non-diegetic** — score only the audience hears

Music playing on-screen is diegetic and belongs in a beat. Both fields have
preset dropdowns — the music presets lead with score genres and screen-music
styles — and a free-text box for anything else.

In reference mode each gains an audio reference beneath it with its own
retention marker; the music reference also names what it controls (style, beat
and rhythm, instrumentation, mood).

### Reference task

Task type, which becomes the bracketed prefix on the summary, and the **Source
video** section nested beneath it. Both appear only in reference mode.

### Summary

A reference-mode section, sitting just above the Insert button and hidden when
the switch is off — base modes have no summary field. **Draft summary from
fields** produces a first pass you can edit. The `[task type]` prefix is added
when the prompt is built, so it won't appear in the box itself.

---

## Three things worth knowing

**Tags need angle brackets.** H3 only treats `<Picture 1>` as a reference —
plain `Picture 1` is read as ordinary words. The plugin adds the brackets on
output, so either form works in the fields.

**Locked dropdowns.** Camera motion, amplitude, speed, transitions and retention
markers come from fixed vocabularies in the spec. Everything descriptive accepts
whatever you type.

**No blank lines.** WanGP treats an empty line as a prompt separator, so every
section sits on consecutive lines and blank lines typed into any field are
stripped.

**Read the output before generating.** Fields are stitched into sentences from
templates, so wording you enter may not agree grammatically with the phrasing
around it — *"the target video shows a woman faces the door"* rather than
*facing*. The prompt box is editable.

**The placeholders are a worked example.** Every text field's greyed-out text
belongs to one coherent prompt — a fishmonger gutting a fish across two shots.
Read them in order to see the expected phrasing and structure; they vanish as
soon as you type.

---

## Reference slots

The model accepts up to 9 images, 3 videos and 3 audio clips, but WanGP's
selectors offer **two** videos and **two** audio references, so the dropdowns
stop there. Slots follow upload order — reordering your uploads silently
reassigns them.

WanGP's Audio References selector is a single dropdown, so **Use
Reference-Video Soundtrack(s)** and standalone audio clips are alternatives,
not additions. The plugin warns if you describe both.

---

## Customising

Vocabulary lists are plain Python lists at the top of `plugin.py` —
`LOCATIONS`, `SCENE_LIGHTING`, `MOTION_TYPES`, `RIGS`, `SOUNDSCAPE_PRESETS`,
`MUSIC_PRESETS` and so on. Edit freely, but keep entries phrased to read
naturally mid-sentence.

`MAX_SHOTS`, `MAX_BEATS` and `MAX_ENTRIES` set the ceilings. Slots are
pre-created and hidden rather than built on demand, so raising them adds
components at load time.

---

## Known limitations

- Beats can't be reordered after the fact; remove and re-add to change the
  sequence.
- Beat timestamps aren't validated against the clip duration or against each
  other.
- The panel hides on model change. If WanGP starts with a non-MiniMax model
  already selected, it may stay visible until you switch models once.
- Slot numbers aren't checked against what's actually loaded in the generator.
- Voiceover has required phrasing in the spec, including a follow-up clause
  about the speaker's lips, which isn't generated yet.
- There's no way to keep a source video's spoken words while replacing the
  voice.
- Interface labels read "Subject 1", "Subject 2" in the cast list and the beat
  **Who** dropdown whatever the mode. They're row identifiers, not output
  labels — in base modes the word never reaches the prompt.
- The voiceover clause the spec requires after `</d>` ("while his lips remain
  completely closed") can't be emitted; there's no field after the speech.

---

## Changelog

See [CHANGELOG.md](CHANGELOG.md).

## Licence

MIT
