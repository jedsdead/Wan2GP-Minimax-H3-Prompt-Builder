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

Six named sections, with `N/A` in any that doesn't apply:

```
subject_definitions:
<Subject 1> (S1) is the fishmonger, heavy apron and forearms wet to the elbow, a middle-aged male with a low, weathered, measured voice.
<Subject 2> is the covered market hall, iron roof beams and crushed-ice counters.
summary:
The target video shows rows of crushed-ice counters, featuring <Subject 1> and <Subject 2>. It runs 10 seconds in a live-action, cinematic style.
retention_analysis:
N/A
detailed_description:
A live-action, cinematic scene, set in a covered market hall, lit by harsh fluorescent light, shot on Super 35mm film, across a 10-second duration.
[Shot 1] A wide shot on a 35mm lens frames rows of crushed-ice counters. <Subject 1> (S1) lays fish across the ice. At 00:04.000, <Subject 1> (S1) calls out: <d>[English] First batch of the morning.</d>
overall_soundscape:
The scene carries market noise and trolley wheels.
non_diegetic_music:
N/A
```

Descriptions live in `subject_definitions` and the shot text refers to the
label, so a subject reads identically in every shot.

> **On field names.** `detailed_description` is the reference schema's name for
> this field; the base schema calls the equivalent `integrated_multimodal_
> description`. The builder uses one shape for everything, so it always writes
> the former. If plain text-to-video output seems worse than it used to be,
> that's the first thing to test.

## One form, no modes

There are no modes. Every section is always available and the output always
uses the six-section shape, with `N/A` in any section that doesn't apply — a
character is a subject whether or not a reference asset backs it, so defining
one as `<Subject 1>` and referring to the label is worth doing either way.

Keyframe images are attached in the generator, not here, and the model
associates them with the prompt positionally:

- **Start image** — describe it in **Shot 1's anchor**
- **End image** — describe it in the **final beat of the last shot**
- **Reference images** — `<Picture 1>`, `<Picture 2>` resolve to the images
  loaded above, in order

With both a start and an end image, a single shot usually works best so the
model can interpolate between them.

## Sections

### Scene

Style, location, lighting, atmosphere and camera body — the things that hold for
the whole clip. These become a single opening sentence written before `[Shot 1]`:

```
A live-action, cinematic scene, set in a busy New York street at dusk, lit by golden hour light, shot on Super 35mm film.
```

Lens lives with each shot instead, since it commonly changes at a cut. It reads
as `a wide shot on a 24mm wide-angle lens frames ...`.

### Cast & subjects

One list for everything that appears — people, animals, places, props. Each
entry becomes a `<Subject N>` definition, and the shot text refers to the label
rather than repeating the description:

```
subject_definitions:
<Subject 1> (S1) is a skilled Jedi in dark hooded robes holding an ignited blue lightsaber, a young male with a low, clear, measured voice.
<Subject 2> is an Imperial Lambda-class shuttle with three folded wings, parked on the right of the pad.
<Subject 3> (S2) is Darth Vader in black armour with a flowing cape, wielding an ignited red lightsaber.
```

Only the description is required. The rest is optional:

- **Speaker** — assigned by you, not derived from position. If entry 2 is a car
  and entry 3 talks, entry 3 can be S2. Setting it makes the voice fields
  meaningful.
- **Source asset** — only for entries backed by a reference picture. Setting it
  along with a retention marker adds a line to `retention_analysis`; entries
  invented from description contribute nothing there.

Because descriptions live in the definitions, a subject reads identically in
every shot — which is the main thing that keeps them consistent.

### Shots

**Add shot** appends one. Shot 1 is the opening and takes no timestamp; later
shots take a cut time and a transition, written as `[Shot 2] At 00:04.000, the
shot cuts to …`.

Each shot has framing, a lens, a camera triple (motion, amplitude, speed), a rig
and an anchor describing what's in frame.

Framing, motion and rig are three separate axes: where the camera is, what it
does, and how it's mounted. A dolly in is Push In on a dolly track; a 360 is an
Arc Shot; a drone shot is any motion at all, on a drone. Each shot starts on its own line in the
output.

### Beats

Within each shot, **Add beat** builds an ordered sequence. A beat is either:

- **unattributed** — leave **Who** blank: `A bus pulls away.`
- **attributed action** — pick one or more subjects, leave the speech blank:
  `<Subject 2> rests on the right of the platform.`
- **dialogue** — subjects, their action and delivery, and the spoken words

Speaker IDs are added automatically where they apply, so picking two subjects
that both speak gives `<Subject 1> and <Subject 3> (S1,S2)`. The Type dropdown
is a label for your own benefit; what decides the output is who is selected and
whether there are spoken words.

The split matters: action and delivery go *outside* the `<d>` tag, only the
language tag and the words themselves go inside. Beats appear in the order you
add them, so ordering them is how you order the timeline.

Leave the speech blank for a non-verbal event like a shared laugh — it keeps
the ID but emits no `<d>` tag.

Each beat also takes an optional **At (seconds)**, which times an event *inside*
the shot: `At 00:04.000, they clash in the centre.` That's how a single
continuous take gets internal timing, and it's worth using for FL2VA, where one
shot is usually preferred. Tick **carries across the next cut** to emit
`<scenetrans>`.

### Audio

Two fields, and which one a sound belongs in depends on a single question:
**can the characters hear it?**

- **Soundscape** — ambience, physical sounds, breathing, laughter
- **Non-diegetic** — score only the audience hears

Music playing on-screen from a radio, an instrument or a phone is diegetic and
belongs in a beat, not in the music field. Both fields have preset dropdowns
that combine with anything you type.

### Summary

Always available, always written. Yours to compose — **Draft summary from
fields** produces a first pass from what you've entered, which you can then
edit. The `[task type]` prefix is added when the prompt is built, so it won't
appear in the box itself.

### Reference task

Task type only, and only needed when reference assets are involved. It becomes
the bracketed prefix on the summary.

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

- The panel hides itself when you switch to a non-MiniMax model, but only once
  a model change fires — if you load WanGP with another model already selected,
  it stays visible until you switch models. If the selector component isn't
  found, the builder stays visible for every model and says so in the console.
- Reference labels are typed by hand and aren't checked against the generator's
  actual reference slots — `Picture 1` must really be in slot 1.
- Voiceover has required phrasing in the spec, including a follow-up clause
  about the speaker's lips, which isn't generated yet.
- The `summary` section in reference mode is written by you, not generated —
  the other five sections have enough structure to template reliably, this one
  doesn't.
- There's no way to keep the source video's spoken words while replacing the
  voice. `audio reuse` brings the original voice along with the words, and
  `audio reference` supplies a voice for words you type. Type the line.
- Beats can't be reordered after the fact; remove and re-add to change the
  sequence.
- Beat timestamps aren't validated against the clip duration or against each
  other, so nothing stops you writing times out of order or past the end.

---

## Changelog

See [CHANGELOG.md](CHANGELOG.md).

## Licence

MIT
