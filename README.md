# MiniMax H3 Prompt Builder — a WanGP plugin

MiniMax H3 doesn't take a free-form prompt. It expects a structured document
with named fields, speaker IDs, timestamped shots, tagged dialogue, and
angle-bracket labels for every reference you attach. Getting that right by hand
is fiddly and easy to break in ways that fail quietly.

This plugin adds a panel beneath the prompt box in WanGP. You write the action
in one field and press buttons to insert the parts that have to be exact —
shot markers, timestamps, camera sentences, dialogue tags, continuity marks —
and it writes the fields, tags and boilerplate in the format the model expects.

Format follows MiniMax's own guides:
[base modes](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md)
and [full reference](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md).

> Everything the buttons write is ordinary text you can edit in place. Read
> what it produces before committing to a long generation.

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
only uses `re` and `gradio`. The enhancer buttons additionally talk to WanGP's
own Prompt Enhancer, but that's already part of WanGP; nothing extra is
installed for it.

---

## What it writes

**Base modes** — three fields:

```
integrated_multimodal_description: A film noir scene, set in an office at night, lit by shafts of light through blinds, with drifting cigarette smoke, graded with high-contrast black and white, shot on Super 35mm film, across a 10-second duration.
[Shot 1] The camera cuts to a medium shot of the detective at his desk, on an 85mm portrait lens, on a dolly track. The camera pushes in with small amplitude at slow speed. A neon sign reading "OPEN ALL NIGHT" is visible in the frame. A private detective in a rumpled trenchcoat, a middle-aged male with a low, gravelly, measured voice (S1) sits back and says: <d>[English] She walked in like trouble</d><scenetrans> The speech continues seamlessly across the cut.
[Shot 2] At 00:05.000, the camera cuts to a close-up of the rain-streaked window, on a 35mm lens, on a tripod. <scenetrans>The speech carries over from the previous shot. (S1) watches the glass and says in an off-screen voiceover: <d>[English] I should have shown her the door</d> while his lips remain completely closed.<cutoff>
overall_soundscape: The scene carries rain on a tin roof.
non_diegetic_music: A noir jazz score.
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
A film noir scene, set in an office at night, lit by shafts of light through blinds, graded with high-contrast black and white, shot on Super 35mm film, across a 10-second duration.
[Shot 1] The camera cuts to a medium shot of the detective at his desk, on an 85mm portrait lens. <Subject 1> (S1) sits back and says: <d>[English] She walked in like trouble</d>
overall_soundscape:
The scene carries rain on a tin roof.
non_diegetic_music:
A noir jazz score.
```

In reference mode descriptions live in `subject_definitions` and the shot text
refers to the label, so a subject reads identically in every shot.

Note that the action text is the same in both. You write `(S1)` and the
builder decides what it becomes — an inline description in base modes, a
`<Subject 1>` label in reference mode — when the prompt is built. That's why
flipping the switch works on an action you wrote an hour ago.

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

The shot index tracks your actual final shot — counted out of the action text,
so deleting a shot by hand keeps it right — and the duration is formatted to
two decimals. Referring to `<Picture 1>` inside your own descriptions is up to
you; the guide's examples do it in the shot text as well.

With both a start and an end image, a single shot usually works best so the
model can interpolate between them.

## The draft

WanGP goes down sometimes, and a half-written prompt is an hour of work. The
whole form is written to `h3_draft.json` beside `plugin.py` — after every press
that changes something, and on a 20-second timer for the typing in between.

Three controls sit at the top of the panel, outside every fold, because after
a crash they're the first thing wanted:

- **Restore last draft** puts the saved form back, reopening any cast entries
  and reference blocks that were showing when it was saved
- **Save draft now** forces a write
- **Clear all fields** starts over (there's a second one at the bottom, by the
  build controls)

**Restore before you start typing.** The line under the buttons tells you
what's on disk; once there's anything in the form, autosave writes over the
top of it.

Two rules protect the draft from the panel itself. An automatic save never
replaces a draft with an empty form — otherwise a restart would blank it
before you could restore — and every write keeps the draft it replaced as
`h3_draft.prev.json`. If you type a little before realising you should have
restored, the big draft is still in the backup: close WanGP and rename it over
`h3_draft.json`.

Clearing goes to disk deliberately rather than through autosave, so a crash
straight after it can't resurrect the old form. Restoring straight after a
clear offers back the draft the clear replaced, since restoring an empty one
onto an empty form would do nothing.

The file holds the field values positionally and records how many there are.
If you update the plugin and the field count has changed, restoring is refused
rather than attempted — a draft shifted by one field would put your lens in
your anchor, which is exactly the failure the rest of the design works to
avoid. Delete the file to be rid of it, or press Clear.

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

**Clear the action** resets only the action, leaving cast, scene, audio and
summary in place for the next window.

One exception: if a line was still carrying when you cleared, the pick-up half
is left behind as ordinary text —

```
[Shot 1] <scenetrans>The speech carries over from the previous shot.
```

— because the window boundary is itself a cut and its other half is in the
window you just wrote. Delete the line if the next window doesn't follow
straight on. **Undo last insert** still holds the whole cleared window.

---

## Sections

Every section is an accordion — click its heading to fold it away. The panel
is long, and folding what you've finished with beats scrolling past it.

### Scene

Style, colour grade, location, time of day, lighting, atmosphere and camera
body — the things that hold for the whole clip. These become a single opening
sentence before `[Shot 1]`.

**Time of day** is separate from lighting on purpose: lighting is how the
scene is lit, this is when it happens. The hour rides along with the setting
in the opening sentence ("set in an office at night"), and it reaches the
audio suggestion too, where dawn and midnight imply very different sound in
the same room.

Lens and rig live with the camera button instead, since both commonly change
at a cut.

### Cast & subjects

One list for everything that appears — people, animals, places, props. Each
entry becomes a `<Subject N>` definition, and the shot text refers to the label
rather than repeating the description. Each entry folds on its own heading —
eight open at once is most of a screen.

Only the description is required. **Use character creator** reveals a small
casting sheet — name, ethnicity, gender, age range, height, build, hairstyle,
hair colour, eye colour, clothing — with an **Add to description** button that
composes a sentence like:

> John, an Asian male in his mid-40s, six feet tall with a muscular build,
> long straight black hair and brown eyes, wearing a rumpled trenchcoat.

Every field is optional; leave any of them blank and the grammar adjusts
rather than leaving a gap. Clothing goes last, so the sentence reads as a
person first and an outfit second — presets are phrased to follow "wearing",
which the composer supplies, but anything you type that already carries its
own verb ("dressed in a long evening gown") is used as written.

The button overwrites the Description field, so type there directly if you'd
rather skip the creator. If a name is set, it also shows up next to the Subject
number in the dialogue **Who** boxes — `Subject 1 (John)` — purely to help
track several subjects; the prompt itself still reads `Subject N` / `(Sx)`.

**Speaker** is assigned by you, not derived from position: if entry 2 is a
car and entry 3 talks, entry 3 can be S2. It's also what ties an entry to the
action text, so an entry that speaks or gets a retention marker needs one.

**Presence** matters for voiceover: a speaker marked off-screen gets no
closed-lips clause, since there's no visible face to describe.

Which shots a subject appears in isn't typed anywhere. It's read out of the
action by speaker ID when the prompt is built, so it can't go stale when you
renumber or delete a shot.

Tick **Use reference mode (Ref2VA)** at the top of the panel to reveal the
reference fields on every entry, along with the Reference task section. They
stay collapsed otherwise, and values left in a collapsed block are ignored
entirely rather than leaking into the prompt:

- **Source asset** — which reference image the subject comes from, chosen from
  `Picture 1`–`Picture 9` or typed
- **Retention** and **What is retained** — how much of it carries over
- **Voice from** — a reference audio supplying this speaker's timbre
- **Motion from** — a reference video supplying movement or performance

### Action

One text field holds the whole action, and a toolbar of buttons writes into it.
**Shot** starts a new line; everything else appends to the line you're on, so
you build a shot left to right and then tidy it by hand.

Gradio can't tell Python where your cursor is, so an insert always lands at the
end. That's the one real constraint, and the reason the workflow is "compose,
then edit" rather than "click into the middle".

**Shot** adds the next `[Shot N]`. The number comes from reading the field, not
from a counter — delete a shot by hand and the next press agrees with what's
left.

**Time** writes a timestamp in the form the model reads: `At 00:03.000,`

**Camera** composes a sentence from framing, lens, motion, amplitude, speed and
rig. The **Of — what is in frame** box goes in the middle, where the grammar
wants it:

> The camera cuts to a medium shot of the detective at his desk, on an 85mm
> portrait lens, on a dolly track.

Leave **Transition** blank and it picks for you, and it has three cases:

| Where you are | What it writes |
|---|---|
| The opening shot | nothing — `A medium shot of the desk, on a 35mm lens.` |
| A later shot | `The camera cuts to …` |
| A second camera press in the same shot | `The camera moves to …` |

There is nothing before `[Shot 1]` to cut from, so it doesn't pretend
otherwise. Pick a verb explicitly and it's used as picked, wherever you are.

Framing, motion and rig are three separate axes: where the camera is, what it
does, and how it's mounted. A dolly in is Push In on a dolly track; a 360 is an
Arc Shot; a drone shot is any motion at all, on a drone.

**Dialogue** writes attributed speech. Tick who's speaking, type the action and
delivery, and type the words. Action and delivery go *outside* the `<d>` tag;
only the language tag and the words go inside. Leave the words blank and you
get an attributed action instead; leave **Who** blank and type into the field
directly for something unattributed like `A bus pulls away.`

The button always writes the bare `(S1)`, never the description — see [What it
writes](#what-it-writes) above.

**Voiceover** is the one type that changes the output rather than just
labelling it. The spec requires an exact clause and a specific follow-up, and
both are written for you:

> (S1) says in an off-screen voiceover: `<d>`[English] I still remember that
> road.`</d>` while his lips remain completely closed.

Anything in the delivery box is kept as a preceding action, with a trailing
"says" dropped so it can't collide with the required phrase. The pronoun comes
from the entry's voice gender; several speakers sharing a line take the plural
verb.

Two checkboxes handle audio that doesn't respect shot boundaries:

- **Line carries across the next cut** writes `<scenetrans>` and states the
  continuity in words. **How it carries** picks the wording for the outgoing
  half from the guide's sanctioned phrases; the next **Shot** press writes the
  receiving half automatically, because it can see the carry is still open.
- **Speech runs past the end** writes `<cutoff>`, telling the model the clip
  ends mid-line rather than waiting for the speaker to finish. Useful with
  sliding windows, where a line deliberately continues into the next one.

**Visible text** covers anything actually readable on screen — a sign, banner,
subtitle, phone screen, licence plate. Pick what's carrying it and type the
words; they're quoted verbatim in double quotation marks and never translated,
which is what the guide asks for:

> A neon sign reading "OPEN ALL NIGHT" is visible in the frame.

**Camera**, **Dialogue** and **Visible text** each fold independently, so the
toolbar can be cut down to the buttons you're actually using.

**Undo last insert** reverts the last button press, including a clear.

### Audio

Two fields, and which one a sound belongs in depends on a single question:
**can the characters hear it?**

- **Soundscape** — ambience, physical sounds, breathing, laughter
- **Non-diegetic** — score only the audience hears

Music playing on-screen is diegetic and belongs in the action. Both fields have
preset dropdowns — the music presets lead with score genres and screen-music
styles — and a free-text box for anything else.

**Suggest a soundscape** and **Suggest a score** hand what you've built to
WanGP's own Prompt Enhancer. See below.

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

## What Insert checks

Insert never blocks. A freeform field is allowed to be a work in progress, and
refusing to build because a tag is half-typed would be worse than the tag. But
it reads the action back and says what it found, in the status line under the
prompt box:

- `<d>` tags that are unbalanced **or nested** — `<d>a<d>b</d></d>` balances if
  you only count, but the model reads the inner pair as the spoken words
- `<d>` blocks with no `[Language]` tag
- shot numbers repeated, out of order, or with gaps
- empty shots — a marker with nothing after it spends a cut doing nothing
- timestamps past the clip duration, or out of order
- timestamps in a form nothing reads: `At 3s,` and `At 00:03,` are invisible to
  the ordering check, the duration check *and* the model
- a carried line with no pick-up, or the reverse
- `<cutoff>` anywhere but the very end — it means the clip ends mid-line, so
  there's one and it's last
- an odd number of double quotes, meaning some visible text is left unclosed
- speaker IDs with no entry in Cast & subjects — a stray `(S7)` reaches the
  finished prompt as a bare `(S7)`, meaning nothing to the model and easy to
  read straight past

---

## Suggesting the audio

Two buttons hand the scene to WanGP's Prompt Enhancer and write the reply
back. Both are a first pass, not a finished field.

The enhancer is used for audio only. The action is yours to write, and the
summary is composed from the fields you have already filled in, so neither
needs a model guessing at it.

**Suggest a soundscape** and **Suggest a score** are separate requests on
purpose. Asked for both at once, the weaker enhancers drift between the two
jobs — score notes landing in the soundscape, or the reverse. One field per
request costs a second press and reliably answers the question that was asked.
Each writes its own text box and leaves the preset dropdowns alone, so a
suggestion is always undone by clearing one field.

**What they read.** Style, location, time of day, atmosphere, and the action
reduced to what a sound editor can use, plus whatever presets and notes you've
already set. Where you've already chosen something the model is told to build
on it and fill the gaps rather than restate it.

**What they deliberately don't read.** Camera work — framing, lens, motion,
rig. None of it says anything about sound, and feeding it in pulls the model
towards describing the shot instead. The anchor is pulled back out of each
camera sentence and the rest is dropped. Dialogue is reported as *"someone
speaks aloud here"* without the words, so there's nothing to echo back into a
field that must not contain any. Continuity prose and the closed-lips clause go
too — they describe how a cut or a mouth behaves, not how anything sounds.

**Requirements.** The Prompt Enhancer must be enabled in the Configuration tab
and set to one of the Qwen 3.5 variants — the Llama 3.2 and JoyCaption options
are captioning models and won't follow the instruction. The buttons borrow the
enhancer WanGP already holds where they can, and load one at your configured
level where they can't. **Keep the enhancer loaded between presses** trades
VRAM for speed on the second press; it only governs a copy the plugin loaded
itself, since a borrowed enhancer is WanGP's to release.

**When it fails.** The status line ends with a probe report saying what it
could and couldn't find, and a reply that won't parse is printed in full to the
console and quoted in the panel.

---

## Things worth knowing

**Tags need angle brackets.** H3 only treats `<Picture 1>` as a reference —
plain `Picture 1` is read as ordinary words. The plugin adds the brackets on
output, so either form works in the fields.

**Locked dropdowns.** Camera motion, amplitude, speed, transitions and retention
markers come from fixed vocabularies in the spec. Everything descriptive accepts
whatever you type.

**No blank lines.** WanGP treats an empty line as a prompt separator, so every
section sits on consecutive lines and blank lines typed into any field are
stripped.

**Read the output before generating.** The action is yours, but the sentences
around it — the opening line, subject definitions, the summary — are still
assembled from fields, so wording you enter may not agree grammatically with
the phrasing around it. The prompt box is editable.

**The placeholders are a worked example.** Every text field's greyed-out text
belongs to one coherent prompt — a fishmonger gutting a fish across two shots.
The Action field's placeholder shows what the buttons produce when you use them
in order; they all vanish as soon as you type.

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
`LOCATIONS`, `TIMES_OF_DAY`, `SCENE_LIGHTING`, `MOTION_TYPES`, `RIGS`,
`CUT_VERBS`, `CONTINUE_VERBS`, `SOUNDSCAPE_PRESETS`, `MUSIC_PRESETS`,
`CHAR_CLOTHING`, `SCREEN_TEXT_KINDS`, `CARRY_PHRASES` and so on. Edit freely,
but keep entries phrased to read naturally mid-sentence.

`LOCATIONS` entries no longer carry their own time-of-day tails, since that's
its own field now — if you add your own, leave the hour off it.

`MAX_ENTRIES` sets the cast ceiling. There's no shot or beat ceiling any more;
the action is one text field and it holds as many shots as you write.

The enhancer instructions are near the top of `plugin.py`:
`SOUNDSCAPE_ONLY_PROMPT`, `MUSIC_ONLY_PROMPT`, and `AUDIO_RETRY_PROMPT` as a
blunter second pass when the first reply won't parse. `AUTOSAVE_SECONDS` sets
the draft timer.

---

## Known limitations

- Inserts always land at the end of the field, because Gradio doesn't hand
  Python the cursor position. Build a shot left to right, then edit.
- The panel hides on model change. If WanGP starts with a non-MiniMax model
  already selected, it may stay visible until you switch models once.
- Slot numbers aren't checked against what's actually loaded in the generator.
- There's no way to keep a source video's spoken words while replacing the
  voice.
- `<scenetrans>` assumes the line carries into the *next* shot. A line spanning
  more than one cut needs editing by hand.
- Interface labels read "Subject 1", "Subject 2" in the cast list and the
  dialogue **Who** boxes whatever the mode. They're row identifiers, not output
  labels — in base modes the word never reaches the prompt.
- Validation reads the text, so it can only catch what's written in the
  expected shapes. A hand-written line that avoids them is invisible to it.
- The enhancer buttons depend on WanGP internals that aren't a documented API.
  They degrade to a message in the panel rather than an error, but a WanGP
  update could still break them.
- The draft is one file plus one backup, so it holds one prompt and the one
  before it. Building a third without losing the first means copying the
  prompt box somewhere yourself.
- Autosave needs `gr.Timer` (Gradio 4.x and later). On older Gradio the button
  presses still save, but typing between them isn't covered.

---

## Changelog

See [CHANGELOG.md](CHANGELOG.md).

If WanGP reports needing a specific version to install this plugin after updating, it may be reading a cached entry in `plugins_local.json` rather than this repo's current `plugin_info.json`. Delete that plugin's entry (or the whole file, which regenerates) and restart WanGP.

## Licence

MIT
