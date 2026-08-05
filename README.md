# MiniMax H3 Prompt Builder — a WanGP plugin

MiniMax H3 doesn't take a free-form prompt. It expects a structured document
with named fields, speaker IDs, timestamped shots and tagged dialogue. Getting
that right by hand is fiddly and easy to break in ways that fail quietly.

This plugin adds a panel beneath the prompt box in WanGP that builds the whole
thing for you — fill in shots, speakers and audio, and it writes the fields,
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

Base modes produce three named fields:

```
For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.
integrated_multimodal_description: [Shot 1] Live-action, cinematic, a wide shot frames rows of crushed-ice counters. The setting is a covered market hall, lit by harsh fluorescent light, with rising steam. The camera trucks right with small amplitude at slow speed. The fishmonger, a middle-aged male with a low, weathered, measured voice (S1) says: <d>[English] First batch of the morning.</d>
overall_soundscape: The scene carries market noise and trolley wheels. Ice shifts under moving fish.
non_diegetic_music: A walking double bass with brushed drums.
```

Reference mode produces six: `subject_definitions`, `summary`,
`retention_analysis`, `detailed_description`, `overall_soundscape` and
`non_diegetic_music`.

---

## Modes

Pick the mode matching what you've attached in the generator above. The panel
shows only the sections that mode can use.

| Mode | Use when | Adds |
|---|---|---|
| **T2VA** | text only | — |
| **I2VA** | start image | instruction line anchoring it at 0.00 s |
| **FL2VA** | start and end image | instruction line with both alignment marks |
| **L2VA** | end image only | instruction line for the closing frame |
| **Ref2VA** | reference assets | subjects, retention markers, task types |

A **source video** section appears for FL2VA (continuing a clip) and Ref2VA
(where it becomes a `<Video N>` reference).

---

## Sections

### Scene

Style, location, lighting, atmosphere and camera body — the things that hold for
the whole clip. These become a single opening sentence written before `[Shot 1]`:

```
A live-action, cinematic scene, set in a busy New York street at dusk, lit by golden hour light, shot on Super 35mm film.
```

Lens lives with each shot instead, since it commonly changes at a cut. It reads
as `a wide shot on a 24mm wide-angle lens frames ...`.

### Cast

Add anyone you want to refer to by a stable ID. Each gets `(S1)`, `(S2)` and so
on, and their description is written **once** at first appearance, then
referenced by ID after that.

Dialogue is optional. Select a speaker on any beat to attribute it to them —
`(S1) turns and looks` is a perfectly good action beat. The ID is emitted
whenever a speaker is selected, whether or not the beat has spoken words.

### Shots

**Add shot** appends one. Shot 1 is the opening and takes no timestamp; later
shots take a cut time and a transition, written as `[Shot 2] At 00:04.000, the
shot cuts to …`.

Each shot has framing, a lens, a camera triple (motion, amplitude, speed) and an
anchor describing what's in frame. Each shot starts on its own line in the
output.

### Beats

Within each shot, **Add beat** builds an ordered sequence. A beat is either:

- **unattributed** — leave the speaker blank: `A bus pulls away.`
- **attributed action** — pick a speaker, leave the speech blank:
  `(S1) turns and looks.`
- **dialogue** — a speaker, their action and delivery, and the spoken words

The Type dropdown is a label for your own benefit; what actually decides the
output is whether a speaker is selected and whether there are spoken words.

The split matters: action and delivery go *outside* the `<d>` tag, only the
language tag and the words themselves go inside. Beats appear in the order you
add them, so ordering them is how you order the timeline.

Leave the speech blank for a non-verbal event like a shared laugh — it keeps
the ID but emits no `<d>` tag. Tick **carries across the next cut** to emit
`<scenetrans>`.

### Audio

Two fields, and which one a sound belongs in depends on a single question:
**can the characters hear it?**

- **Soundscape** — ambience, physical sounds, breathing, laughter
- **Non-diegetic** — score only the audience hears

Music playing on-screen from a radio, an instrument or a phone is diegetic and
belongs in a beat, not in the music field. Both fields have preset dropdowns
that combine with anything you type.

### Reference subjects (Ref2VA)

One entry per thing a reference contributes. A location is a **Subject**, same
as a person — the label covers any reusable visible content. An image used only
to define a subject needs no entry of its own; cite it in the subject's source
field instead.

Retention markers are fixed values: `fully_preserved`, `partially_preserved`,
`attribute_transfer`, `weak_reference` for visuals, and `fully_copy`,
`partially_copy`, `reference`, `weak_reference` for audio.

---

## Two things worth knowing

**Locked dropdowns.** Camera motion, amplitude, speed, transitions and
retention markers come from fixed vocabularies in the spec and can't be typed
into. The model was trained on those exact expressions. Everything descriptive —
style, location, voice qualities, actions — accepts whatever you type.

**No blank lines.** WanGP treats an empty line as a prompt separator and would
split the output into several generations, so every section sits on consecutive
lines. The guide asks for a blank line after the instruction line; that's
deliberately omitted here. Blank lines typed into any field are stripped too.

---

## Customising

Vocabulary lists are plain Python lists at the top of `plugin.py` — `LOCATIONS`,
`SCENE_LIGHTING`, `SOUNDSCAPE_PRESETS`, `MUSIC_PRESETS` and so on. Edit freely,
but keep entries phrased to read naturally mid-sentence.

`MAX_SHOTS`, `MAX_BEATS`, `MAX_SPEAKERS` and `MAX_SUBJECTS` set the ceilings.
Slots are pre-created and hidden rather than built on demand, so raising them
adds components at load time — lower them if the panel feels slow to open.

---

## Known limitations

- Mode is selected manually. Reading it from the generator's attached images
  isn't implemented yet, so the builder can describe an image the model wasn't
  given.
- The panel hides itself when you switch to a non-MiniMax model, but only once
  a model change fires — if you load WanGP with another model already selected,
  it stays visible until you switch models. If the selector component isn't
  found, the builder stays visible for every model and says so in the console.
- Reference labels are typed by hand and aren't checked against the generator's
  actual reference slots — `Picture 1` must really be in slot 1.
- Voiceover has required phrasing in the spec, including a follow-up clause
  about the speaker's lips, which isn't generated yet.
- The `summary` section in reference mode is the weakest template. Treat it as
  a draft and edit it.
- Beats can't be reordered after the fact; remove and re-add to change the
  sequence.

---

## Licence

MIT
