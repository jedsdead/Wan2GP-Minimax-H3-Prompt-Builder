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

Six named sections, with `N/A` in any that doesn't apply:

```
subject_definitions:
<Subject 1> (S1) is the fishmonger, heavy apron and forearms wet to the elbow, a middle-aged male with a low, weathered, measured voice, from <Picture 1>.
<Subject 2> is the covered market hall, iron roof beams and crushed-ice counters, from <Picture 1>.
<Audio 1> is the voice-timbre reference for <Subject 1> (S1).
<Video 1> is the motion and performance reference for <Subject 1> (S1).
summary:
[reference generation + audio reference] The target video shows rows of crushed-ice counters, featuring <Subject 1> and <Subject 2>. It runs 10 seconds in a live-action, cinematic style.
retention_analysis:
<Subject 1> (appears in [Shot 1]): fully_preserved - the apron and wet forearms are retained.
<Audio 1>: reference - the voice-timbre reference for <Subject 1> (S1).
detailed_description:
A live-action, cinematic scene, set in a covered market hall, lit by harsh fluorescent light, shot on Super 35mm film, across a 10-second duration.
[Shot 1] A wide shot on a 35mm lens frames rows of crushed-ice counters. The camera pushes in, on a steadicam. <Subject 1> (S1) lays fish across the ice. At 00:04.000, <Subject 1> (S1) calls out: <d>[English] First batch of the morning.</d>
overall_soundscape:
The scene carries market noise and trolley wheels.
non_diegetic_music:
N/A
```

Descriptions live in `subject_definitions` and the shot text refers to the
label, so a subject reads identically in every shot.

---

## One form, no modes

There are no modes. Every section is always available and the output always
uses the six-section shape — a character is a subject whether or not a
reference asset backs it, so defining one as `<Subject 1>` and referring to the
label is worth doing even for text-to-video.

Keyframe images are attached in the generator, not here, and the model
associates them with the prompt positionally:

- **Start image** — describe it in **Shot 1's anchor**
- **End image** — describe it in the **final beat of the last shot**
- **Reference images** — `<Picture 1>`, `<Picture 2>` resolve to the images
  loaded above, in order

With both a start and an end image, a single shot usually works best so the
model can interpolate between them.

---

## Sections

### Scene

Style, location, lighting, atmosphere and camera body — the things that hold
for the whole clip. These become a single opening sentence before `[Shot 1]`.

Lens and rig live with each shot instead, since both commonly change at a cut.

### Cast & subjects

One list for everything that appears — people, animals, places, props. Each
entry becomes a `<Subject N>` definition, and the shot text refers to the label
rather than repeating the description.

Only the description is required. **Speaker** is assigned by you, not derived
from position: if entry 2 is a car and entry 3 talks, entry 3 can be S2.

Tick **Is a referenced subject (Ref2VA only)** to reveal the reference fields —
they stay collapsed otherwise, and values left in a collapsed block are ignored
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
preset dropdowns and an optional audio reference with its own retention marker;
the music reference also names what it controls (style, beat and rhythm,
instrumentation, mood).

### Reference task

Task type, which becomes the bracketed prefix on the summary. Only needed when
reference assets are involved.

### Summary

Always written, and sits just above the Insert button. **Draft summary from
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
- The panel hides on model change only — loading WanGP with another model
  already selected leaves it visible until you switch.
- Slot numbers aren't checked against what's actually loaded in the generator.
- Voiceover has required phrasing in the spec, including a follow-up clause
  about the speaker's lips, which isn't generated yet.
- There's no way to keep a source video's spoken words while replacing the
  voice.
- The description field is written as `detailed_description`, the reference
  schema's name. The base schema calls the equivalent
  `integrated_multimodal_description`.

---

## Changelog

See [CHANGELOG.md](CHANGELOG.md).

## Licence

MIT
