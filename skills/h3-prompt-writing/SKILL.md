---
name: h3-prompt-writing
description: Write MiniMax H3 video generation prompts for T2VA, I2VA, FL2VA, L2VA, and Ref2VA. Use when rewriting multimodal requests into H3 prompt structures, composing integrated_multimodal_description, overall_soundscape, and non_diegetic_music, aligning keyframes, or defining reference labels for images, videos, and audio. Also covers weapon-combat choreography — duels, swordplay, spear and polearm work, crowd-clearing and mounted action, executions — including action phrasing, initiative, weapon stability, combat camera angles, visible speed, and impact sound. Also covers delivering multi-window prompt sequences into Wan2GP/WanGP sliding windows, including [/duration=...] and [/overlap=...] window commands.
compatibility: Portable to any agent that can read local files — no external API calls, MiniMax Hub tools, or proprietary runtime required. The agents/openai.yaml file only adds optional ChatGPT/Codex UI metadata; it does not restrict the skill to OpenAI agents.
---

# H3 Prompt Writing

## Workflow

1. Identify the input mode: T2VA, I2VA, FL2VA, L2VA, or full-reference Ref2VA.
2. For base text/keyframe modes, read `references/base-en.txt` and follow its final prompt structure.
3. For full-reference mode, read `references/ref-en.txt` and follow its six-section rewrite format.
4. If the video contains weapon combat of any kind, also read
   `references/weapon-combat-en.txt` before writing the shot timeline, and plan
   the choreography from it. It governs the fight content; the mode guide above
   still governs notation and field format. See "Weapon Combat" below.
5. Preserve the exact field names, section order, labels, and timing notation from the selected guide.

## Base Modes

- T2VA: build the full audiovisual timeline from text.
- I2VA: start from the first frame and develop forward from it.
- FL2VA: describe the continuous path between the first and last frames.
- L2VA: infer a plausible opening and converge to the supplied last frame.

Use `integrated_multimodal_description`, `overall_soundscape`, and `non_diegetic_music` in the order shown in `references/base-en.txt`.

## Full-Reference Mode

Ref2VA rewrites use `subject_definitions`, `summary`, `retention_analysis`, `detailed_description`, `overall_soundscape`, and `non_diegetic_music` in that order. Reference labels stay consistent across all sections.

Read `references/ref-en.txt` for label rules, retention analysis, and complete examples.

## Weapon Combat

`references/weapon-combat-en.txt` is the authority on what the fighters actually
do: linked action phrases instead of trade-off exchanges, action density per
duration, spatial anchors and travel, initiative ownership per phrase, positive
weapon-stability language, per-weapon technique lists, combat camera angles,
visible evidence of speed, normal-duel versus execution resolution, transition
types, and per-impact sound. Its decision process (XVII) and quality check (XIX)
are worth walking before writing an action timeline.

Where it and the mode guides disagree, the mode guide wins on form and the
combat guide wins on content:

- **Shot notation.** The combat guide's output format shows
  `[Shot 1 | time range]`. Do not use it. H3 notation is `[Shot 1]` with no
  timestamp and `[Shot N] At MM:SS.mmm, the camera cuts to ...` per base-en 4.2.
- **Six sections.** Its section list matches ref-en's, so Ref2VA work needs no
  adjustment; for T2VA and keyframe modes the main field is still
  `integrated_multimodal_description`.
- **Music.** Its default of `non_diegetic_music: N/A` holds only until the user
  asks for a score. A requested score is then written per base-en 4.7.
- **Endings, in a sliding-window sequence.** Its transition list (weapon wipe,
  dust wipe, energy bloom, whip-pan) is written for standalone segments. A
  frame-filling blade, a dust cloud or a white flare is exactly the
  underdetermined ending described below, so inside a continuous multi-window
  sequence keep those transitions mid-window and close on a specific framing
  with a subject in it. Its rule against static locked-weapons endings still
  applies: end on motion, just legible motion.
- **User instruction beats both.** When the user asks for something the combat
  guide advises against — strictly simultaneous parries, stationary combat, no
  wide shots — follow the user and use the guide's other tools (phrase themes,
  inherited weapon positions, initiative pressure, spatial travel) to keep the
  fight from reading as mechanical turn-taking.

## Reference Files

The guides are bundled in `references/`. Read the one matching the mode before
writing; they are the authority on notation and override anything remembered.

`references/base-en.txt` (T2VA / I2VA / FL2VA / L2VA) — 1 task overview,
2 final prompt structure, 3 keyframe handling per mode, 4.1 timeline,
**4.2 shots and cuts**, **4.3 camera motion vocabulary**, 4.4 speakers and
dialogue, 4.5 on-screen text, 4.6 soundscape, 4.7 music, 5 four worked cases.

`references/weapon-combat-en.txt` (weapon combat, any mode) — I core
choreography philosophy, II action density per duration, III spatial movement,
IV initiative and set-piece phrasing, V weapon stability language, **VI
per-weapon technique lists**, **VII camera rules for long weapons**, VIII
visible speed evidence, IX effects attached to real weapon motion, X normal
duel resolution, XI execution mode, XII transitions, XIII impact sound, XIV
prompt style, XVII decision process, XVIII failure patterns, XIX quality check,
**XXI ten good-versus-bad worked examples**, XXIII a good 10-second duel.

`references/wan2gp-prompts.md` (Wan2GP/WanGP delivery, not H3 format) — line
processing modes, sliding windows, `[/...]` window commands, multiple images,
prompt enhancer, `@`/`@@` syntax, Think mode, macros. Read only when the target
runtime is Wan2GP; see the delivery section at the end of this file.

`references/ref-en.txt` (Ref2VA) — 1 section order, 2 reference labels,
3 summary and task-type prefixes, 4 retention markers, 5 detailed_description,
6 audio sections, 7 a complete worked example. Section 5.1 defers to
`base-en.txt` for shots, camera, speakers and dialogue, so Ref2VA work usually
needs both files.

## Points Most Often Got Wrong

Check these against the guides rather than from memory:

- `[Shot 1]` carries no timestamp. Every later shot opens with a strictly
  increasing cut time: `[Shot 2] At 00:03.500, the camera cuts to ...`.
  Timestamps also mark beats inside a shot, not only its start.
- In a sliding-window sequence, `[Shot 1]` of every window after the first
  opens with "The camera cuts to ..." or "The camera continues to ...", still
  with no timestamp. A bare description there reads as ambiguous against the
  overlap conditioning and produces a morph instead of a cut. See "Every window
  after the first opens with a continuity verb".
- A cut must introduce new information about subject, space, state, viewpoint
  or time. If only distance or a slight angle changes, use camera motion
  instead — this is what makes adjacent shots read as a jump.
- Never end a window on a fade, an emptying frame, or a static hold on
  low-detail material, and never let consecutive windows close on visually
  convergent imagery. See "Never end a window on an underdetermined frame" -
  this is the first thing to check when content from one window shows up in
  another.
- Cuts within a scene are written on the timeline, never expressed by splitting
  the work across runtime windows. If the target is Wan2GP, see "Two different
  kinds of cut" at the end of this file: `[/new_shot]` zeroes the overlap and
  destroys continuity, and is only for an actual scene change.
- Camera motion uses the fixed vocabulary in base-en 4.3 (Push In, Truck,
  Pedestal, Arc Shot, Tracking Shot, Static Shot and so on), optionally with
  `with small/large amplitude` and `at slow/fast speed`, written as natural
  English inside the shot rather than stacked as labels.
- Speaker IDs `(S1)`, `(S2)` belong only to characters who actually vocalize,
  numbered by order of vocal events. Never write them in `retention_analysis`.
- `retention_analysis` lists the shots each subject appears in:
  `<Subject 1> (appears in [Shot 1], [Shot 3]): partially_preserved - ...`.
- Delivery, tone and speaker identity go outside `<d>`; inside `<d>` there is
  only the language tag and the spoken words, verbatim.
- `detailed_description` is normally 350-500 English words for generation
  tasks. Action-dense timelines may run over rather than drop timed beats.

## Output Rules

- Write rewrite sections in English; preserve dialogue, lyrics, and visible scene text in their original language.
- Describe each shot by composition, subjects, environment, actions, camera, sound, and the exact point where referenced content appears.
- Avoid plot summaries, unresolved reference labels, and timing that does not match the requested duration.
## Tips for Better Results
- Always match the total duration of the description to the requested video length (4–15 seconds).
- Keep reference labels consistent (e.g. `<Picture 1>`, `<Video 1>`, `<Audio 1>`) across every section.
- Prefer concrete visual and audio details over abstract words like "cinematic" or "beautiful".
- When using keyframes (I2VA / FL2VA / L2VA), clearly state how the first and/or last frame connects to the timeline.

## Delivering to Wan2GP Sliding Windows

Optional; only relevant when prompts are pasted into Wan2GP (WanGP) as one
sliding-window sequence. `references/wan2gp-prompts.md` is Wan2GP's own prompt
guide — read it before writing a multi-window sequence. Most relevant sections:
"How To Process Each Line Of The Text Prompt", "Each Line Is Used For A New
Sliding Window", and "Optional `[/...]` Window Commands".

### Pick the right line-processing mode

The `How to Process each Line of the Text Prompt` dropdown decides how the text
is split. **A structured H3 prompt spans many lines, so it needs a paragraph
mode, not a line mode.** Choose:

`Each Paragraph Separated by an Empty line will be used for a new Sliding
Window of the same Video Generation`

Note that `references/wan2gp-prompts.md` documents only three of the five
options and omits both paragraph modes. `get_multi_prompts_gen_choices` in
`shared/utils/prompt_parser.py` is the authority: `G`, `PG`, `W`, `PW`, `FG`.
Picking the documented `Each Line Will be used for a new Sliding Window` splits
a structured prompt into one window per line.

### Format for the parser, not for reading

In paragraph mode `split_prompt_units` treats **every blank line as the start of
a new window**. So:

- keep every line and labeled section of one window adjacent, single newlines
- put exactly one blank line between complete windows
- use no separators such as `---`, and no blank lines between sections

A file formatted for readability parses as many times more windows than
intended. `validate_sliding_window_prompt_boundaries` will not catch it: it only
warns once it can confirm the intended count, and stray fragments such as a bare
`summary:` line defeat that check, so it returns silently.

Verify before handing the sequence over, without needing torch or a GPU:

```python
import sys; sys.path.insert(0, "<Wan2GP>/shared/utils")
import prompt_parser
windows = prompt_parser.split_prompt_units(open("sequence.txt").read(), "PW")
print(len(windows))          # must equal the intended window count
```

### Window commands

Each window may open with a bracketed command; Wan2GP strips it before the text
reaches the model. Brackets not starting with `/` are left alone, so H3's own
`[Shot 1]` markers pass through untouched. Combine with commas:
`[/duration=10s,/overlap=18]`.

- `[/duration=121]`, `[/duration=5s]`, `[/duration=20%]` — output frames this
  window contributes. Without one, Wan2GP uses the remaining frame count capped
  by Sliding Window Size.
- `[/overlap=N]` — overlap frames, rounded to the model's step. These are
  generated *in addition* to the window's duration and condition the transition
  rather than counting as committed output.
- `[/new_shot]` (alias for `[/overlap=0]`) — see "Two different kinds of cut"
  below before using it.
- `[/loras_mult=...]` — override LoRA multipliers for this window only.

Unknown slash commands are rejected at validation. Only one Start Image is
supported in sliding-window mode.

### Every window after the first opens with a continuity verb

Base-en's notation is written for a standalone segment, where `[Shot 1]` opens
cold with no timestamp and no transition verb because there is nothing before
it. A sliding-window sequence breaks that assumption and the rule has to be
extended.

Window 2's `[Shot 1]` is not the start of a video. It lands mid-stream, with
the previous window's final frames sitting in the history block as
conditioning. Open it with a bare description — "A medium close-up of the
pilot's hands on the yoke" — and the prompt has said nothing about how this
frame relates to the one before it. H3 resolves that ambiguity in favour of
the conditioning, because the conditioning is the only thing in context making
a claim about continuity. The result is not a cut: it is a morph. The camera
drifts out of the old framing toward the new description, subjects deform
across the boundary, and the join reads as a glitch rather than an edit.

So in a sequence, every window except the first opens `[Shot 1]` with an
explicit continuity verb, and the choice of verb states the intent:

| The window opens on | Write |
|-|-|
| a new shot — a cut falls on the boundary | `[Shot 1] The camera cuts to a low three-quarter shot of ...` |
| the previous shot, still running | `[Shot 1] The camera continues to push in ... ` |

Constraints:

- **No timestamp.** The opening shot is at 00:00.000 by definition, so
  base-en's "`[Shot 1]` carries no timestamp" still holds. Write "The camera
  cuts to ...", never "At 00:00.000, the camera cuts to ...".
- **Shot numbering still restarts** at `[Shot 1]` in every window. Each window
  is its own H3 generation with its own timeline; the continuity verb describes
  the relationship to the previous window, it does not continue its numbering.
- **"Continues" must name what continues** — the move, the framing, the action
  — not merely assert continuity. "The camera continues to push in with small
  amplitude at slow speed on the same low three-quarter framing as he brings
  the blaster up into line" gives the model something to match. "The camera
  continues the previous shot" gives it nothing and drifts exactly like a bare
  description.
- **`[/new_shot]` windows are the exception.** Zero overlap means no history
  block, so there is nothing to cut away from or continue; that window opens
  cold like a first window.

This is the opening-frame counterpart to the section below. Both come from the
same mechanism: conditioning wins wherever the prompt stops making claims. At
the end of a window that means keeping the last frame specific; at the start of
one it means naming the relationship to what came before. A cut on a boundary
is a cut executed *against* conditioning that disagrees with it, so make the
opening emphatic and fully specified — and keep the default overlap rather than
reaching for `/new_shot` to force the cut, which throws away wardrobe, grade
and geography to buy something the verb already achieves.

### Never end a window on an underdetermined frame

Where a window *ends* needs more specificity than anywhere else in it, not
less. This is the opposite of the natural instinct, which is to let a window
trail off and hand over to the next one.

The reason is how the conditioning is attended. The history block sits before
target frame 0, so it is tempting to assume its influence is concentrated at
the start of the window. It is not: condition rows are attended by *every*
target row, so the previous window's imagery is available across the whole
window. It loses at the opening because the prompt is emphatic there - "At
00:00.000, the shot cuts to a wide shot of a moonlit glade" leaves nothing to
infer. It wins wherever the prompt stops specifying, and the last shot is
usually where that happens.

So these endings are the dangerous ones:

- fades to black, to white, to empty frame
- "the mist closes over them", "the figures are swallowed", "the dust settles"
- anything that removes the subject from frame before the final frame
- a static camera on a low-detail, low-contrast field (drifting snow, fog, ash)
- "the shot ends wide with both men still engaged" - generic, and repeated
  verbatim across windows

Each one asks the model to render a frame with almost no specified content,
while a fully specified frame from the previous scene sits in its attention
context. The previous scene wins, and it reads as the end of one window
reappearing inside the next.

Worse, two consecutive windows that both end this way converge on each other
even without any bleed, because "cold, static, particulate, nothing in frame"
describes the same image whichever scene it belongs to. A real case, two
consecutive windows of one sequence:

```
              property | window A                           | window B
                camera | static shot                        | static shot
                 grade | cold slate blue, almost monochrome | cold blue-grey, crushed blacks
         single accent | single red accent                  | only bright element
  airborne particulate | fine snow drifts across frame      | mist lifts to swallow
      subject obscured | half-buried in snow                | swallowed by mist
             end state | no moving subject                  | no subject at all
```

Those two shots share six properties and the verbatim sentence "The camera
holds a static shot". They were written as a battlefield aftermath and a
moonlit ravine, and they resolve to nearly the same picture.

What to do instead:

- keep a specific subject in frame through the final frame
- prefer a slow camera move over a static hold at the end of a window, since a
  move keeps supplying information
- if the scene really must dissolve, put the dissolve *mid-window* and end on
  something concrete afterwards
- check the closing shot of every window against the closing shot of the
  window before it, and change size, angle, or subject if they match
- never repeat a closing sentence across windows

Check this before anything else when content from one window appears in
another. It is a more common cause than any coordinate or conditioning fault,
it explains an artefact that lands at the *end* of a window (which a
conditioning error cannot), and it is far cheaper to fix.

### Two different kinds of cut

`[/new_shot]` is misleadingly named. It does not mean "a new camera shot" in
the filmmaking sense — it sets the window's overlap to zero, which removes all
visual conditioning from the previous window. In `wgp.py`, `new_shot` forces
`prefix_frames_count = 0`; the H3 pipeline then takes neither branch of
`if continuation_count:`, so the window gets no history block and no anchor on
the previous window's final frame. Nothing carries over: not the framing, not
the lighting, not the subjects' positions, not the grade.

That is the right behaviour for a **scene change** — a new location, a jump in
time, a genuinely unrelated shot. It is the wrong tool for a cut *within* a
continuous scene, even though a cut is exactly what the word suggests.

Cuts inside a scene belong in the prompt, on H3's own timeline:

```text
[Shot 2] At 00:03.500, the camera cuts to a wide shot of ...
```

H3 generates the whole window as one continuous piece of footage and places the
cut inside it, so subjects, wardrobe, environment, lighting and grade stay
consistent across the cut. Reaching for `[/new_shot]` to get the same camera
change throws away every one of those, and the join will read as a
discontinuity rather than an edit.

The rule of thumb:

| Change | Where it belongs |
|-|-|
| New angle, size or camera move, same scene | `[Shot N] At MM:SS.mmm, the camera cuts to ...` |
| Same scene continuing across a window boundary | `[/overlap=N]`, default overlap, `[Shot 1] The camera continues to ...` |
| A cut that falls on a window boundary | `[/overlap=N]`, default overlap, `[Shot 1] The camera cuts to ...` |
| New scene, location or time | `[/new_shot]` on that window |

A ten-second window can hold four or five in-prompt cuts. Window boundaries are
a generation-length constraint, not an editorial one, and should not be used to
express cutting.
