---
name: "h3-prompt-writing"
description: "Write MiniMax H3 video prompts (T2VA, I2VA, FL2VA, L2VA, Ref2VA): structures, soundscape, music, keyframes, reference labels, 360 panoramas. Covers weapon, unarmed and one-vs-many fights, Wan2GP sliding-window sequences, the Ref2VA/RefMod subject format, and automatic per-fight-type choice of Wan2GP combat LoRAs (BUNNY Weapon Combat, Fight Flow, Combat Base V2): triggers, weights, per-window multipliers."
---

# H3 Prompt Writing

## Workflow

1. Identify the input mode: T2VA, I2VA, FL2VA, L2VA, or full-reference Ref2VA.
2. For base text/keyframe modes, read `references/base-en.txt` and follow its final prompt structure.
3. For full-reference mode, read `references/ref-en.txt` and follow its six-section rewrite format,
   with subject definitions, retention analysis and dialogue written as in
   "Reference Prompts: Default Format" below.
4. If the video contains weapon combat of any kind, also read
   `references/weapon-combat-en.txt` before writing the shot timeline, and plan
   the choreography from it. It governs the fight content; the mode guide above
   still governs notation and field format. See "Weapon Combat" below.
5. If the video contains unarmed hand-to-hand combat, also read
   `references/unarmed-action-director-zh.txt`. If one character fights
   several opponents (armed or not), also read
   `references/multi-opponent-combat-en.txt`. Both govern the fight content;
   see "Unarmed and Multi-Opponent Combat" below for how they fit H3 notation.
6. If any reference image is a 360-degree or equirectangular panorama of a location,
   also read `references/panorama-360-en.txt` before planning shots, and build its
   direction map first.
7. If the user says the prompt is a "RefMod in FL2VA" prompt, or the characters
   come from RefMods (the MiniMax H3 RefMods plugin for Wan2GP), use the
   six-section layout and "Reference Prompts: Default Format" below, including
   its RefMods notes. Where it differs from `references/base-en.txt` or
   `references/ref-en.txt`, it wins.
8. Preserve the exact field names, section order, labels, and timing notation from the selected guide.
9. Whenever the video contains any fighting, apply "Combat LoRAs in Wan2GP"
   below automatically, without waiting to be asked: classify the fight type
   (weapons, hand to hand, gun-fu, one-versus-many, mixed) per window, pick the
   LoRAs, triggers, reference guides and per-window multipliers from its
   selection table, and tell the user in one line which LoRAs to load and in
   what order.
10. For any multi-window Wan2GP sequence, apply "The user's standing rules for
    sequences" below without being asked, and run its pre-delivery checklist
    before handing the file over.

## The user's standing rules for sequences

Rules the user has set for every Wan2GP sliding-window sequence. They hold
unless the user says otherwise for a particular sequence.

- **Format.** Paragraph mode, no blank lines inside a window, exactly one blank
  line between windows, each window opening `[/duration=15s,/overlap=18]`
  (plus `/loras_mult` only when combat LoRAs are in use; see below). The
  six-section layout with text-defined subjects, as in "Prompt layout tested
  with these LoRAs".
- **First window opens on a wide shot.** Every window except the last ends on a
  wide shot with the characters still in frame and still doing something
  specific.
- **Window openings.** `[Shot 1]` of every later window opens with "The camera
  continues to ..." on the previous window's last shot (see "Continuations run
  2 seconds before cutting" below), and the first cut comes at 00:02.000 or
  later. When it cuts, use an angle clearly different from the previous
  window's last shot (not the same side angle a little closer).
- **Nothing is described at 00:15.000** (or at the window's last frame).
- **Music** is `N/A` unless the user asks for a score; Suno scores are written
  separately.
- **One fixed lighting description per location**, repeated identically in
  every window set there.
- **Voices: tone only, never voice type.** Never describe what kind of voice a
  character has (male or female, light, deep, husky, velvety, tenor). Give only
  the tone of each line, fitting its emotion and context, and only on the line
  where that character speaks: `<Subject 2> (S1), with an anxious, quick tone,
  says`. Describing the voice type ("a young, light, soft-spoken male voice",
  repeated in shots where he did not speak) made Davin speak in Siren's voice
  from one line onward; removing the voice description fixed it. Later lines may
  still tie back with "in the same voice as his line at 00:01.200". A requested
  accent is the one exception: add it to every line of that character ("with a
  smug tone, in an English accent").
- **Close-ups at three-quarter angles.** With reference pictures, a straight
  front-on close-up, especially against a plain background, reproduced the
  reference portrait itself instead of the scene. Frame close-ups and medium
  close-ups at a three-quarter angle, from slightly to the side, with the
  location visible behind the face.
- **Dialogue window joins.** End each window on the scene's master wide shot and
  open the next by continuing that wide ("The camera continues on the same wide
  side view ... both as before, breathing, lips closed"), then cut to the
  close-ups at 00:02.000 or later (see the next rule). Keep the
  master's wording identical every time, and give it a new variant only when
  positions change (seated, standing, facing away).
- **Continuations run 2 seconds before cutting.** In every continuation
  prompt (each sliding window after the first, and any prompt that continues a
  video, in Wan2GP or the Video Editor), `[Shot 1]` continues the previous
  shot and runs for at least 2 seconds: no cut and no spoken line before
  00:02.000. Apply it automatically; the user should not have to ask. The
  reason: a continuation's timestamps start at the first frame of the join,
  not the first new frame, and the Video Editor's masked joins lock the
  previous clip's last frames and their sound inside that start (18 frames
  between windows, often 22 between clips). A cut at 00:01.000 left about 2
  frames of new wide shot, the model cut straight after the held frames, and
  the line meant for 00:01.200 started inside them. Open `[Shot 1]` with "The
  camera continues ..." on the previous window's last framing and action (lips
  closed), give it one small new beat around 00:01.000 to 00:01.500 (a breath,
  a step, a glance), then cut at 00:02.000 or later.
- **Positive phrasing only**, including in the opening description and the
  summary. "There is no fighting", "there is no dialogue", "do not engage",
  "not impressed" all count as negatives and can produce the thing they deny.
  Write what is seen and heard instead: "a calm, quiet conversation; both
  lightsaber hilts stay unlit", "Siren stays silent, her lips closed", "stay
  hidden", "finds him unimpressive". Spoken lines inside `<d>` are left as the
  user wrote them.
- **The opening description is style only.** The text between
  `detailed_description:` and `[Shot 1]` holds only the triggers and what stays
  the same for the whole window: style and genre (live-action space opera,
  black-and-white film noir), camera and format (35mm anamorphic film, 8K
  digital, handheld, Steadicam, a single unbroken take), grade and lighting,
  and elements that stay consistent across the sequence. It carries no
  actions, timestamps, camera moves, exchange patterns, screen positions or
  speaking order. That text has no timeline, so the model fits whatever it
  describes in somewhere and it competes with the shots: in a single-take
  running duel a long rules paragraph caused a cut on a window join, and
  stripping it back to the triggers fixed it. Everything that happens goes in
  the shot, in its first sentence or at the beat where it applies.
- **Summaries are one or two plain sentences**: who, where, and the overall
  situation ("Glory confronts Siren in the middle of a busy market street").
  Never retell a spoken line's content, and never describe transitions, timed
  events or the order of shots there. The summary has no timeline, so the
  model plays whatever it describes at a moment of its own choosing and then
  again where the shots put it: a paraphrased line was spoken as well as (or
  instead of) the real one, a wipe described in the summary flashed up early
  and then played again, and a vendor described there returned in a later
  shot. Every event and every word belongs in the shots only.
- **Soundscape: sounds only, no voices.** `overall_soundscape` holds ambience
  and specific synchronised sounds (a ramp's clunk and hiss, a brazier's
  crackle, dripping water, robes rustling), never who speaks or how a voice
  sounds. Like the summary it has no timeline, so a voice named there gets
  placed somewhere: a soundscape describing both women's voices, in a window
  where only Glory spoke, gave Siren dialogue she never had. Each line carries
  its own tone in the shot ("with an anxious, quick tone"), and later lines tie
  back with "in the same voice as her line at ...". Avoid sound words that
  suggest speech, such as "whispering" or "murmuring".
- **Fights.** Windows end mid-fight; fighters stop only while someone is
  speaking.
- **No line during a sustained action.** H3 will not play a spoken line and a
  continuing action (Sith lightning, a lightsaber bind, an exchange of strikes)
  at the same moment. It plays them in turn: the action stops, the line plays,
  then the action starts again, so the action appears twice. Glory's line over
  continuous lightning came out as line, lightning, line, lightning again.
  Finish the action before the line or start it after, and say its state at
  the line: "the lightning has stopped; Glory stands over her and says".
- **No action across a window boundary.** Each window is generated as its own
  short scene, so an action already under way when a window opens gets
  replayed from its beginning (the lightning starting again, the duel's
  opening charge again). Put window boundaries at pauses: finish each action
  before its window ends and open the next window on a new beat. If an action
  really must cross, open the next window with "The camera continues to ..."
  on the same framing and describe the action already in full flow at the
  first frame, never its start.
- **Lightsabers switch off** the instant the hilt leaves its owner's hand, or
  when its owner is defeated, unless the user says otherwise. Write it as a
  rule only in the windows where a disarm happens, and describe the switch-off
  in the shot itself.

### Pre-delivery checklist

Run these over the finished file (a short script is fine) and fix anything
that fails before sending it:

1. Window count matches the plan; no blank lines inside a window.
2. Timestamps inside the `[Shot N]` lines increase strictly and stay below the
   window's last frame.
3. `retention_analysis` lists for each character exactly the shots that name
   them (see "Shot lists, not 'all shots'" below).
4. Speaker IDs restart per window in order of who speaks first in that window.
5. Every speaker's line fits its shot: estimate about 2.5 words per second, and
   leave at least a second between a line's estimated end and the next cut to
   anyone else.
6. No two consecutive shots show the same subject from the same angle at only a
   different size, inside a window or across a window boundary.
7. No negatives outside `<d>`.
8. No base-text rule describes an event that does not happen in that window.
9. Each window's location description covers only where that window happens.
10. Every prop that changes hands or is put away is accounted for in later
    windows' subject definitions.
11. Each window is at most about 7,000 characters (the official H3 prompt
    limit, so the length the model was built around). Trim repetition before
    trimming timed beats.
12. No ellipses, emotion tags or stage directions inside `<d>`.
13. No stillness phrases ("holds completely still", "goes still", "freezes")
    on a whole frame; give the subject a breath, a weight shift or an eyeline
    change, or keep the camera fixed while something in frame moves.
14. The characters present at the start of each window are the ones present at
    the end of the window before; anyone arriving or leaving does so mid-window.
15. Every summary is one or two sentences of who, where and situation, with no
    spoken content, transitions or event sequence.
16. No spoken line falls inside a sustained action (lightning, a bind, a run of
    exchanges); the action ends before the line or starts after it.
17. No action is in progress at a window boundary, unless the next window opens
    with "The camera continues to ..." on the same framing.
18. The opening description (before `[Shot 1]`) holds only triggers, style,
    camera and format, grade and lighting, and consistent elements: no
    actions, timestamps, camera moves, screen positions or speaking order.
19. In a single-take window: no shot-size nouns after the opening framing, no
    "cut" or "cuts" for any action, no "every shot", "side views" or second
    `[Shot 1]` marker, and every big camera move spread over about two seconds.
20. A window that continues the previous window's shot repeats its location
    and summary word for word and opens where its action actually was, with
    any move into a new area a second or two in.
21. No soundscape mentions a voice, speech, dialogue, whispering or who speaks;
    it holds ambience and specific sounds only.
22. No line describes what kind of voice a character has; each line carries only
    its tone (and an accent where the user asked for one).
23. Every close-up and medium close-up is at a three-quarter angle with the
    location behind, never straight front-on, when reference pictures are used.
24. The "Only one of them speaks at a time" sentence sits in the first shot that
    has a line, never in a silent opening shot.
25. Every continuation prompt (each window after the first, and any prompt that
    continues a video) opens with "The camera continues ..." and has no cut and
    no spoken line before 00:02.000.

## Base Modes

- T2VA: build the full audiovisual timeline from text.
- I2VA: start from the first frame and develop forward from it.
- FL2VA: describe the continuous path between the first and last frames.
- L2VA: infer a plausible opening and converge to the supplied last frame.

Use `integrated_multimodal_description`, `overall_soundscape`, and `non_diegetic_music` in the order shown in `references/base-en.txt`.

## Full-Reference Mode

Ref2VA rewrites use `subject_definitions`, `summary`, `retention_analysis`, `detailed_description`, `overall_soundscape`, and `non_diegetic_music` in that order. Reference labels stay consistent across all sections.

Read `references/ref-en.txt` for label rules, summary prefixes, and complete examples. Subject definitions, retention analysis and dialogue tagging follow "Reference Prompts: Default Format" below, which overrides ref-en on those three.

## Reference Prompts: Default Format

This is the default format for every prompt with reference subjects: Ref2VA,
and FL2VA with RefMods ("RefMod in FL2VA"). It has given the best adherence in
testing in both modes, so it overrides the base-en and ref-en rules on
`subject_definitions`, `retention_analysis` and dialogue tagging where they
differ. Read ref-en for everything else (summary prefixes, shots, audio
sections).

In FL2VA with RefMods the prompt still uses this six-section layout, not
base-en's `integrated_multimodal_description` structure.

### Subject definitions

- **One line per subject, this exact wording**, with the picture sentence
  followed by the audio sentence on the same line:
  `<Subject N>: <Picture N> is the source picture providing the visual identity, facial features and body type of <Subject N>. <Audio N> is the audio source providing the specific timbre, pitch and vocal character of <Subject N> (SN).`
- A subject with no voice reference gets only the picture sentence.
- For a reference that is not a person (a location, a prop, a style), keep
  the same shape and name what the picture provides:
  `<Subject N>: <Picture N> is the source picture providing the layout, materials and lighting of <Subject N>.`
- **Use `<Picture N>` for every visual reference, even a video or a video
  mod.** This has given better adherence than `<Video N>`. The exception is a
  real Ref2VA video edit or continuation, where ref-en's whole-video `<Video N>`
  role still applies.
- **Pictures plus descriptions (the user's preferred form for recurring
  characters).** Keep the full physical description, and say exactly which
  features come from the picture: `<Picture 2> is the source picture providing
  the facial features, skin, eyes, hair and burn scar of <Subject 2>.`, then in
  the description name those same features as coming "from <Picture 2>,
  exactly as they appear there", and everything else (clothing, hood,
  demeanor) "from the visual description". Retention is then
  `partially_preserved`, listing what is kept from the picture and saying the
  clothing follows the description. Leave out of the picture's list anything
  hidden in that window (a bald head under a raised hood).

### Retention analysis

- One line per reference label, keyed by `<Picture N>` and `<Audio N>`, not
  ref-en's `<Subject N> (appears in [Shot 1] ...)` form.
- A picture is `partially_preserved` when the costume or clothing changes:
  say what is kept (face, facial features, hair, skin tone, body type) and
  what replaces the clothing. It is `fully_preserved` when nothing changes.
- An audio is `reference`, and its line limits the voice to that one speaker:
  `<Audio N>: reference. The voice timbre, pitch and vocal character are used only for the lines spoken by <Subject N> (SN).`
- Keep `summary` and `retention_analysis`. Whether they help is not yet
  certain, but they do no harm.

### Dialogue

- **Every line names the speaker, their ID and their audio**:
  `<Subject N> (SN), in the voice following <Audio N>, ...` then the delivery,
  then `<d>[English] ...</d>`. Repeat it on every line, not just the first.
- **One speaker at a time.** With two or more speakers, say so once in the
  first sentence of the shot where they talk ("Only one of them speaks at a
  time; while one speaks, the other keeps her lips closed"), and after each
  line say the other subject stays silent with lips closed.
- **Keep the speaker's face the only face talking.** When a line falls in a
  shot framed on someone else, voices can swap from that line onward. Give
  later lines their own shot: a medium close-up of the speaker facing the
  camera, the other subject behind with their back turned.
- **Tie later lines back to the first.** From a speaker's second line on, write
  `in the same voice following <Audio N> as her line at 00:01.800`.

### Formatting and timing

- Section tags exactly as in the example: `<subject_definitions>` ...
  `</subject_definitions>` and so on, in ref-en's section order.
- **No empty lines anywhere in the prompt.** Every section and every line sits
  directly under the one before it. Wan2GP needs it.
- **Use the user's timings and lines as given.** Fix only punctuation in the
  dialogue. Add a final beat after the last line so the clip ends on something
  specific, and state the total length the clip needs so the user can set the
  frame count.

### RefMods (Wan2GP RefMods plugin)

When the subjects come from RefMods, stored references loaded from the RefMods
panel in Wan2GP instead of attached files:

- **Number pictures in panel order.** `<Picture 1>` is the first visual mod
  loaded, `<Picture 2>` the second, and so on. Video mods are still tagged
  `<Picture N>`.
- **Continuing a video.** When the user continues an existing video in Wan2GP,
  the continued video takes the Picture 1 slot for the whole run, so every
  window's picture tags move up one (Siren `<Picture 2>`, Davin `<Picture 3>`).
- **Unused pictures still load.** References are loaded for the whole run, so a
  character whose picture is loaded can appear in a window that never names
  them. When that happens, generate that window on its own with only the
  pictures it needs.
- **Audio is matched by load order only.** The text encoder cannot hear the
  audio, so `<Audio 1>` is simply the first audio mod loaded. Remind the user
  which voice to load first.
- **Summary prefix** is `[reference generation + audio reference]` when there
  are audio mods, `[reference generation]` when there are none.

### Example

A two-character dialogue with two pictures and two voices. This is the format
to follow:

```text
<subject_definitions>
<Subject 1>: <Picture 1> is the source picture providing the visual identity, facial features and body type of <Subject 1>. <Audio 1> is the audio source providing the specific timbre, pitch and vocal character of <Subject 1> (S1).
<Subject 2>: <Picture 2> is the source picture providing the visual identity, facial features and body type of <Subject 2>. <Audio 2> is the audio source providing the specific timbre, pitch and vocal character of <Subject 2> (S2).
</subject_definitions>
<summary>
[reference generation + audio reference] A locked-off medium-wide shot of <Subject 1> in a Black Widow costume and <Subject 2> in a Scarlet Witch costume standing side by side against a plain, softly lit wall. They trade a short comic exchange about the multiverse and Doomsday. The joke lands on a deliberate slip of the tongue, they both turn to look at the camera, and the scene cuts to black.
</summary>
<retention_analysis>
<Picture 1>: partially_preserved. The face, facial features, hair, skin tone and body type of <Subject 1> are kept. The original clothing is replaced with a black Black Widow tactical bodysuit.
<Picture 2>: partially_preserved. The face, facial features, hair, skin tone and body type of <Subject 2> are kept. The original clothing is replaced with a red Scarlet Witch costume with a long coat and headpiece.
<Audio 1>: reference. The voice timbre, pitch and vocal character are used only for the lines spoken by <Subject 1> (S1).
<Audio 2>: reference. The voice timbre, pitch and vocal character are used only for the lines spoken by <Subject 2> (S2).
</retention_analysis>
<detailed_description>
[Shot 1] A locked-off medium-wide shot frames <Subject 1> wearing a Black Widow costume on the left and <Subject 2> wearing a Scarlet Witch costume on the right, standing side by side in front of a plain, softly lit wall. The camera does not move. Both stand relaxed, arms loose at their sides, facing slightly toward the camera. Only one of them speaks at a time; while one speaks, the other keeps her lips closed.
At 00:00.800, <Subject 1> (S1), in the voice following <Audio 1>, turns slightly toward <Subject 2> and asks with mock puzzlement, eyebrows raised and one hand lifting in an open gesture: <d>[English] I know my character is dead, but it's the multiverse. Surely some version of me could be in Doomsday?</d> <Subject 2> stays silent, lips closed.
At 00:07.000, <Subject 2> (S2), in the voice following <Audio 2>, turns her head toward <Subject 1> and replies knowingly, with a small smirk: <d>[English] Oh, I'm pretty sure we're Doom's bitches.</d> <Subject 1> stays silent, lips closed.
At 00:09.800, <Subject 1> (S1), in the same voice following <Audio 1> as her line at 00:00.800, frowns, tilts her head and says: <d>[English] I think you mean witches.</d> <Subject 2> stays silent, lips closed.
At 00:11.500, <Subject 2> (S2), in the same voice following <Audio 2> as her line at 00:07.000, shrugs, keeping the smirk, and says: <d>[English] I know what I said.</d> <Subject 1> stays silent, lips closed.
At 00:12.800, both women slowly turn their heads and look straight into the camera with deadpan expressions, holding still.
At 00:14.000, the scene cuts to black.
</detailed_description>
<overall_soundscape>
A quiet indoor room tone with a faint, even ambience. Small sounds of costume fabric rustle as the women turn and shrug. There is a brief silent beat as they stare at the camera, then the cut to black lands in silence.
</overall_soundscape>
<non_diegetic_music>
N/A
</non_diegetic_music>
```

This clip needs about 14 seconds of frames.

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

## Unarmed and Multi-Opponent Combat

Two content guides sit alongside the weapon-combat guide:

- `references/unarmed-action-director-zh.txt` (written in Chinese) — five laws
  for one-on-one hand-to-hand fights: micro-split the timeline into 2-3 second
  beats of transferred momentum; zero idle, so each hit reaction (doubling over,
  slamming into a wall) becomes the starting momentum of the next attack or
  pursuit; anatomically specific verbs (forearm check, elbow strike, low calf
  kick, palm strike, wrist-trap) in place of "fight", "boxing" or "kung fu";
  explicit bare hands and 1.0x real-time speed to stop the model adding gloves
  and slow motion; and camera movement that follows the fighters' momentum,
  with sound matched to the material struck.
- `references/multi-opponent-combat-en.txt` — Jackie Chan-style one-versus-many:
  the protagonist is the camera's anchor; pressure overlaps instead of taking
  turns; only 1-2 attackers engage at any moment while the rest flank, close
  in, recover or enter partly through frame edges; the protagonist's path
  through the space is planned first and the environment funnels attackers
  into chokepoints; the protagonist takes glancing hits and setbacks; most
  attackers are repelled rather than knocked out, and the fallen become
  obstacles; brief breathing beats keep it readable; grounded Foley and
  off-screen footsteps sell the encirclement.

Use both for an unarmed brawl against several attackers. If anyone holds a
weapon, `references/weapon-combat-en.txt` also applies, and the multi-opponent
guide extends its crowd-clearing material.

As with the weapon guide, the mode guide wins on form and these win on content:

- **Section structure.** Both files show the six-section Ref2VA layout. That is
  correct only for Ref2VA and RefMod in FL2VA, with subject definitions and
  retention analysis written as in "Reference Prompts: Default Format". For
  T2VA and keyframe modes use base-en's
  `integrated_multimodal_description` / `overall_soundscape` /
  `non_diegetic_music`, and drop the `[Text-to-Video generation]` summary
  prefix and the `retention_analysis: N/A` line from the Action Director
  template.
- **Timing notation.** The Action Director template writes beats as
  `- From 00:00.000 to 00:03.000: ...` bullets. Write the same 2-3 second
  beats as prose inside the shot, with point timestamps: `At 00:03.000, ...`.
  `[Shot 1]` still takes no timestamp, and later shots open with
  `[Shot N] At MM:SS.mmm, the camera cuts to ...`.
- **Positive phrasing.** The multi-opponent guide keeps instructions and
  negatives out of the final prompt; the Action Director asks for "no gloves"
  and "zero slow-motion". Keep the intent in descriptive form: "bare,
  ungloved fists with scraped knuckles", "in 1.0x real-time speed with high
  shutter-speed clarity". If gloves or slow motion still appear in output, the
  Action Director's literal tokens are the tested fix and can go in.
- **Zero idle versus breathing beats.** These do not conflict. What the Action
  Director bans is the reset: strike, stop, return to a guard pose. What the
  multi-opponent guide adds is a micro-adjustment that is itself a reaction —
  a stumble, a sharp breath, a glance toward the next attacker — and it flows
  straight into the next movement.
- **Standalone by default.** The multi-opponent guide treats every request as a
  new scene unless the user asks to continue one. In a sliding-window sequence
  continuation is the point, so the Wan2GP rules at the end of this file apply.

## Combat LoRAs in Wan2GP

Tested with the user's own Wan2GP setup. The LoRA authors' pages and their
BUNNY SelfLift ComfyUI workflow are the source for triggers and intent; the
weights below are what actually worked in Wan2GP.

### Automatic selection (do this for every fight)

Classify each window by what the fighting in it is, then apply the matching
row. Do this per window: a sequence can move from a duel to a brawl to a
dialogue scene.

| Fight in the window | Content guides to read | LoRAs active | Triggers at the start of the action text |
|-|-|-|-|
| Weapons (swords, lightsabers, spears, axes, staffs) | weapon-combat-en | Fight Flow + Weapon Combat (BUNNY) | `Melee weapon interaction. BUNNY.` |
| Hand to hand, one on one | unarmed-action-director-zh | Fight Flow + Combat Base V2 | `Unarmed combat interaction.` plus V2: none by default, `prfight2` for a fierce fight, `prfight2, prfin1` in the window with a knockout or heavy finisher |
| One versus many (armed or not) | multi-opponent-combat-en, plus weapon-combat-en or unarmed-action-director-zh | Fight Flow + Combat Base V2 (+ Weapon Combat if blades dominate) | the matching Fight Flow trigger, plus the V2 trigger as above (and `BUNNY.` if Weapon Combat is on) |
| Guns mixed with close combat (gun-fu) | unarmed-action-director-zh, multi-opponent-combat-en if several attackers | Fight Flow + Combat Base V2 | `Gun-fu interaction.` plus the V2 trigger as above |
| Mixed: one armed, one unarmed | weapon-combat-en and unarmed-action-director-zh | Fight Flow + Weapon Combat + Combat Base V2, each at 0.2 because three stacked LoRAs warp sooner | `Melee weapon interaction. BUNNY.` |
| No fighting (dialogue, aftermath, travel) | none | Combat Base V2 only (it adds body movement and breath to dialogue), or none | no combat triggers |

Fixed load order, so per-window multipliers can be written without asking:
**1 Fight Flow, 2 Weapon Combat (BUNNY), 3 Combat Base V2.** Tell the user to
load all three in that order once. Then give each window its own multipliers
with `[/loras_mult=...]` in that order, keeping each active LoRA at 0.25 and an
inactive one at 0; with Two Phases add `;0` to each, for example:

- weapons: `[/duration=15s,/overlap=18,/loras_mult=0.25;0 0.25;0 0]`
- hand to hand: `[/duration=15s,/overlap=18,/loras_mult=0.25;0 0 0.25;0]`
- no fighting: `[/duration=15s,/overlap=18,/loras_mult=0 0 0.25;0]` (or all `0`)

If the user has only some of these LoRAs loaded, or a different order, follow
what they say and drop the multipliers for missing ones. When a whole sequence
is one fight type, a single multiplier setting in the UI is enough and the
per-window override can be left out.

**The user's current setup** (October 2026): only Fight Flow and Weapon Combat,
loaded in that order, one-phase generation. Duel windows get
`[/duration=15s,/overlap=18,/loras_mult=0.25 0.25]`, windows with no fighting
get `/loras_mult=0 0`. Fight Flow at 0.4 was tried and the user went back to
0.25. A sequence with no fighting at all gets no `/loras_mult` and no combat
triggers.

Fight-type writing rules that go with the table:

- **Weapons:** blade-on-blade contact rules in "Lessons from tested
  sequences" below; weapon-combat-en for phrasing and camera.
- **Hand to hand:** bare, ungloved fists; anatomically specific strikes and
  where each lands; Combat Base V2's persistent, accumulating hit reactions
  building to the finisher; ACTION then DIALOGUE inside a shot.
- **One versus many:** staggered attacker entrances, 1-2 active attackers at a
  time, the protagonist as the camera's anchor.

### Model and sampling

- Best results so far: the community hybrid checkpoint
  `minimax_h3_hybrid_fl2va_ref2va_b25-49` (FL2VA with blocks 25-49 taken from
  Ref2VA), added to Wan2GP as a finetune of `minimax_h3_fl2va_pruned`. Wan2GP
  detects a hybrid from `hybrid_fl2va_ref2va_b25-49` in the file name, so the
  name must be kept.
- Fight Flow was trained on Ref2VA (`ss_base_model_version: minimax_h3_ref2va`);
  on plain FL2VA it warps. Combat Base V2 has only been fully tested by its
  author on FL2VA. The hybrid suits both.
- Euler, 8 steps, CFG 1, 24 fps, the Turbo/Lightx2v profile.

### Weights

- At the author's weights (0.65 / 0.5) the stacked LoRAs warp the image in
  Wan2GP. Working setting: each combat LoRA at **0.25**.
- With Two Phases: `0.25;0` per LoRA (phase 1 only). Wan2GP adds its own Turbo
  in phase 2 automatically.
- In the author's workflow, `0.65 / 0.25` was two separate LoRAs (Fight Flow at
  0.65, Motion Repair at 0.25), not two phases.

### Triggers

Use exactly as written, once, at the start of the action text in
`detailed_description` (or `integrated_multimodal_description`). Never invent
one. Drop combat triggers from windows with no fighting.

- **Weapon Combat (BUNNY):** `BUNNY.`
- **Fight Flow:** `Melee weapon interaction` (weapons), `Unarmed combat
  interaction` (hand to hand), `Gun-fu interaction.` (gun-fu).
- **Combat Base V2:** no trigger for the most stable anatomy and motion;
  `prfight2` for higher intensity; `prfight2, prfin1` for heavy finishers and
  stunt impacts. The V1 triggers (`prfight1`, `prfin1`, `prslow1`) belong to
  the older version. `prslow1` (slow-motion finisher) only works when the
  prompt also says the decisive strike gets a brief slow-motion moment at
  impact before normal speed resumes.

### Prompt layout tested with these LoRAs

- The author's template uses the six-section Ref2VA layout even on the
  FL2VA-based hybrid, and it has worked well there. With no reference images,
  define each subject from the text, e.g. `<Subject 1> is <name>, whose facial
  identity, ... and demeanor come from the visual description.`, and write
  `retention_analysis` per subject:
  `<Subject 1> (appears in [Shot 1], [Shot 3]): fully_preserved - ...`. The
  summary prefix is `[reference generation]`. Section headers are plain lines
  (`subject_definitions:` and so on), with no blank lines inside a window.
- **Shot lists, not "all shots".** Write `(appears in all shots)` only when the
  character really is in every shot. "All shots" tells the model to put them in
  every shot: a Jedi hidden in a crop field for three shots and then bursting
  out appeared twice, once in the hiding shots and once on his entrance. Name
  the character (`<Subject N>`) in every shot they are in, and list exactly
  those shots. The location and style subjects are "all shots" only if every
  shot is set there (a window that wipes from a market to space lists the
  market only for the market shots).
- **Each window's subject definition reflects that window's state**: lit or
  unlit weapon, robe on or off, hood up or down, a fresh scar, empty hands. Do
  not describe a hidden feature next to the thing hiding it ("hood raised over
  her bald head" risks the hood coming down); describe what the camera sees.
- With real reference pictures or RefMods, "Reference Prompts: Default Format"
  still applies.

### Combat Base V2 writing rules

- Within a shot, ACTION then DIALOGUE: finish the strike completely before a
  line starts, or motion falls back to the base model's slow behaviour.
- The LoRA does not choreograph. Say who attacks first, what they do, where it
  lands, how the opponent dodges or reacts, and why the next attack follows.
- Write toward its strengths: continuous chains (misses carry momentum, blocks
  redirect smoothly, contact points separate cleanly); hit reactions that
  persist and accumulate (sustained torso bend, lowered guard, impaired
  movement) and build to a heavy finisher; continuous spinning kicks; sustained
  grappling and takedowns; three-stage environmental collisions (body, then
  environment, then the material's reaction); multi-phase falls; one-vs-many
  with staggered entrances, lane control and protagonist-centred tracking.
- V2 also adds body movement, breathing and vocal dynamics to dialogue scenes.

### Lessons from tested sequences

- **Blade contact.** Blades connect when fighters stay within arm's reach and
  every strike is written as the defender's blade rising or dropping to catch
  it, at a named height, with an arm jolt or recoil, one contact per sentence.
  Phrasing a cut as "carries into the next cut" reads as swinging past.
- **Timed exchanges (best lightsaber contact so far).** Give every exchange its
  own timestamp, with both fighters' actions in that one sentence:
  `At 00:01.750, <Subject 2> swings high and <Subject 1> blocks, the red and
  green lightsabers meeting in a bright white flash and knocking apart.` State
  the pattern once, in the first sentence of the fight shot: each timestamp is
  one complete exchange, the swing and the block happen in the same moment, the impact
  knocks both lightsabers apart, and the next exchange begins straight away.
  Spacing: about 0.875-0.9 s between exchanges held contact; 1.0 s worked but
  felt slow; 0.75 s lost contact. Moving fights can go to about 1.0 s. Results
  vary a lot by seed, so judge a change on two or three seeds.
- **What did not help lightsaber contact:** a fast flurry written as a list;
  aiming strikes at the opponent's guarding blade; "They fight." with no
  choreography; strikes "toward" a body part with the block as a separate
  idea. Long, detailed descriptions of where each block sits did not beat the
  short "swings high and blocks" form inside the timed structure.
- **Blades hold together only in a bind.** The constant white flare goes in the
  bind shot only. Written as a general rule ("burns for as long as they stay
  locked") it turns every clash into a lock; arm jolts and bent elbows in
  every clash read as a pause after each one.
- **Lightsaber wording and sound.** Say "lightsaber" and "lightsaber beam",
  never "blade", and drop "metal hilt" and "clash": they pull the audio toward
  sword clangs. Describe the sound as electrical: a deep steady hum that rises
  and falls in pitch with each swing, and "a sharp burst of crackling, sizzling
  electrical energy like high-voltage arcing" timed to each white flash. The
  Weapon Combat LoRA may still push toward metal; lowering it is the test.
- **Lightsaber hands.** Hands wrap around the hilt below the glowing beam; a
  single beam is more reliable than a double-bladed saber, which spun and
  warped.
- **Fight rules in a window that opens with dialogue.** When the opening
  description carried the duel rules (footwork, exchanges, tracking camera)
  and the window started with talking, the model started the fight under the
  dialogue and then replayed the scripted start. Keep the duel rules out of
  the opening description, and in the shot state the time the fight starts:
  "the two stand still facing each other and only talk, their
  lightsabers held still, until 00:08.600, when the duel begins". For a pause
  mid-fight, end the exchanges first and separate the fighters before the line
  ("they break apart and step back, lightsabers lowered; from 00:02.600 to
  00:07.400 he speaks"), never a line spoken over a bind or a running fight.
- **Event rules only where the event happens.** A general rule in the opening
  description that describes an event ("cut stalks fall away", "a beam switches
  off when the hilt leaves the hand") makes the event appear at random, then
  again where it is scripted. Put such rules only in the windows where the
  event happens.
- **Slow motion of one character only.** "Frozen", "almost to a stop" and "hangs
  in the air" produce a statue. Describe continuous motion instead: "extreme
  slow motion, like a 200-frames-per-second high-speed shot slowed right down:
  his fist keeps travelling forward in one slow, smooth, continuous motion",
  with small moving details (shoulder rolling, cloth flowing, sweat drifting),
  and say the other character "alone moves at normal 1.0x speed".
- **Force pulls and other object moves.** Keep the whole move in one shot that
  shows both the person and the object, keep the person upright with the arm
  outstretched throughout, and time each stage: the object lifts, flies the
  whole distance on its own, smacks into the palm. A cut straight after the
  move, or a short move split across shots, produced the character bending
  down to pick it up.
- **Movement.** Fighters stand still unless every exchange includes footwork
  (attacker steps forward on each strike, defender steps back on each block)
  and the camera tracks along. "Feet planted" freezes them.
- **Moving vehicles.** Make the vehicle exaggerated and simple (e.g. a train so
  long neither end is ever visible), describe it identically in every window,
  fix its direction of travel on screen, and keep the camera riding with the
  fighters, steady and level.
- **Props that leave a fight** should leave the frame (a disarmed sword goes
  over the rail), and every later window's subject definition must describe the
  new state (unarmed, empty hand), or the prop reappears. The same goes for any
  prop picked up outside a fight (a fruit taken from a stall is slipped into a
  robe pocket, and the hands are empty from then on).
- **Voices.** See the standing rule: tone only, never voice type. If a voice
  still swaps at a window join, open the window with a short line from the
  character whose voice is bleeding in, so theirs is the first voice heard (an
  incidental "Go on." or "Mm." at 00:01.000, in their close-up); the user cuts
  it in editing.
- **Silent opening shots.** "Only one of them speaks at a time" in an opening
  shot with no dialogue produced gibberish speech there. Put that sentence in
  the first shot that has a line, and write silent shots as "walking in silence,
  lips closed".
- **Holograms and screens that speak.** A line from a small hologram, with the
  listener's face also in frame, was spoken by the listener. Frame the hologram's
  lines over the listener's shoulder from behind, the listener's face turned
  away, so the hologram's face is the only face on screen, and frame the
  listener's own lines with the hologram just below the frame. A hologram
  character can still use a reference picture for her face; describe the rest
  as translucent blue light with scan lines.
- **Pronunciation.** Respell a name the model gets wrong phonetically inside
  `<d>` only, and keep the respelling identical every time: "Nar Sha-daa",
  "Alder-on", "Cor-us-ont", "Dah-vin". Narration and descriptions keep the real
  spelling.
- **Accents.** An audio reference did not carry a French accent, and IPA
  spellings inside `<d>` forced one but sounded jarring, even softened. Use plain
  dialogue, adding "in an X accent" to the line only when the user asks.
- **Nervous, fast talkers.** A nervous character can be timed at about 3.2 words
  per second with "talking quickly, his words tumbling out" in the line's tone,
  which fits more lines into a window.
- **On-screen writing.** A shop sign "written in Aurebesh, spelling ZORAKS TECH"
  did not render; describe signs as "a glowing sign of angular alien glyphs".
- **Hidden reveals.** A location described in full was shown early and walked
  past. To keep it hidden until a beat, shoot the approach side-on along the
  street and say in the location that the place "comes into view only when they
  stop in front of it"; open the next window with them already stopped there.
- **Props kept from another character.** Something a character hides (a vial)
  must be hidden in every shot: say how her body hides it, keep it out of the
  wide shots ("her hands hidden in front of her at the counter"), and take it out
  only at the beat where the user wants it.
- **Location mechanics are events.** A cargo hold described as having "a ramp
  that lowers ... and rises to close it" replayed the ramp closing in every wide
  shot. After the event, describe only its result ("a solid wall of heavy black
  metal plates"), keeping the rest of the location word for word.
- **A line must finish before the cut to someone else.** A 25-word speech given
  8 seconds, followed by a cut to the listener, put the rest of the line in the
  listener's mouth: the voice stayed right but the listener lip-synced it.
  Budget about 2.5 words per second, split long speeches across two or three
  shots of the same speaker from different angles (medium close-up, then over
  the listener's shoulder), start the listener's reply at least a second after
  the estimated end, and say at the start of the shot who speaks when: "Glory
  is the only one who speaks in this video; Siren listens silently with her
  lips closed throughout", or "Siren's lips stay closed until her own single
  line at 00:11.800".
- **Speaker IDs restart per window**, in order of who speaks first in that
  window. A character who speaks alone in a window is `(S1)` there, even if she
  was `(S2)` elsewhere.
- **Location per window.** When one location description covers several areas
  (farmhouse, garden, yard, crop field), the model starts each window in the
  first area described and then replays the action where the shots put it: a
  fight meant for the field played in front of the house first. Give each
  window a location definition limited to where it happens ("the inside of a
  vast, dense crop field ... the farmhouse lies far away, out of sight") and
  say where the action is in the first sentence of the shot.
- **Jump cuts.** A cut that only changes size from the same angle reads as a
  jump. Merge it into one shot with a camera move ("rises to his feet, and the
  camera pushes in to a medium close-up"; "the close-up of his hands tilts up
  the beam to his face"). Check window joins too: a window ending on a wide side
  shot and the next opening on a medium side shot is the same jump.
- **Single-take windows.** In a window that is one unbroken shot, avoid
  anything that reads as a cut. Naming a new framing mid-shot ("pushes in to a
  medium shot", "into a close-up") reads as a new shot; describe distance
  instead ("drifts closer, travelling alongside them at arm's length", "eases
  back a few paces"). A big reposition in one beat is easier to render as a
  cut, so spread it over about two seconds ("a slow, smooth arc that carries it
  around behind her left shoulder over two seconds"). Whip-like verbs ("fast",
  "swings around", "sweeps") suggest a whip-pan transition; use "glides",
  "arcs", "drifts". An over-the-shoulder reveal ("to frame his face") is
  shot-reverse-shot coverage; let an orbit carry on past instead. The word
  "cuts" for any action ("her lightsaber cuts through the railing") can cut
  the camera; write "slices". "Every shot", "in side views" and a second
  `[Shot 1]` marker all imply several shots. And never ask for geography the
  fighters cannot cover in one beat (off a walkway and into the middle of a
  plaza in one second): the model cuts to get there.
- **Window joins in a continuous shot.** A window continuing the previous
  window's shot repeats that window's location and summary word for word; a
  new location or summary reads as a new scene and starts with a cut. If the
  action reaches a new area, name it at the end of that shared location
  description ("at its right-hand end the walkway opens into a wide plaza").
  Open where the previous window's action actually was, not where its last
  beat was heading: renders often stop short of a goal set in the final second
  (reaching the end of the walkway), and a window that assumes it was reached
  cuts to get there. Move into the new area a second or two into the window.
- **Wipes and transitions on request.** base-en 4.2 allows a wipe when the user
  asks. Put it mid-window, never on a boundary, and specify the edge and its
  path ("a straight, sharp-edged diagonal line enters at the top-left corner
  and sweeps down and across to the bottom-right over one second"), what is
  revealed behind it, and where the new subject sits (the planet centred where
  the face was). Add the new location and subjects to that window's
  definitions, listed only for the shots after the wipe. If the model fudges
  it, the fallback is to end the window on the first image, generate the
  second as its own clip, and wipe in an editor.
- **Invented languages.** For lines in a fictional language (Huttese), write the
  words in `<d>` with a `[Huttese]` tag; if the model garbles them or the mouth
  stays still, switch the tag to `[English]` with the same words.
- **Lighting.** A time of day that changes across windows (sunset into night)
  darkens shots unevenly; keep one fixed lighting description per location.
- **Skin colour covers the whole body.** "Crimson red skin" was read as face
  paint: the face came out red and the neck did not. Write "crimson red skin
  covering her entire body, with her face, ears, neck, throat and hands all
  the same even crimson red", and frame reference portraits down to the
  collarbones so the neck shows.
- **Positive phrasing only**, especially for anything that must not be seen
  (nudity, wrong props): describe what the camera does see.

### Findings from public H3 guides and tools (checked October 2026)

Gathered from MiniMax's official repository and API docs, the ComfyUI H3
prompt guide, and community tools with render-tested notes (sources listed at
the end of this subsection). Official sources outrank community ones; anything
marked "to test" is not yet confirmed on the user's setup.

- **Prompt length.** The official API caps an H3 prompt at 7,000 characters,
  and the guides target 350-500 words of `detailed_description`. Long windows
  (8,000+ characters) give each instruction less weight. Keep each window under
  about 7,000 characters: say a rule once, in the first shot it applies to,
  rather than in every shot, and keep subject definitions to what the camera needs.
- **Contradictions render as unions.** At CFG 1 there is no negative branch,
  so when the prompt describes a different arrangement of people or props than
  the conditioning frames show, the model renders both: extra people, doubled
  props. In Wan2GP's native H3 continuation, target frame 0 of a window is the
  previous window's last overlap frame (`pipeline.py`), so a window's opening
  must agree with who and what was in the previous window's last frame. Bring
  characters in or take them out mid-window, never at the boundary. This
  matches the doubled Jedi and the returning Rodian vendor.
- **Stillness phrases freeze the frame.** "Holds completely still", "goes
  still" and "does not move" can render as a stalled video. Hold the camera,
  not the performer: a breath, a weight shift, an eyeline change, drifting
  dust, a flickering light.
- **Every window changes something physical.** A window whose action could be
  swapped with the previous one's comes back as a near-copy. Leave the world in
  a state it was not in before (a sheet torn off, a robe on the ground, a scar).
- **Descriptions verbatim.** Repeat each character's and location's description
  word for word in every window; rewording between windows is the most common
  cause of a face changing mid-scene.
- **One causal chain per window.** H3 follows one continuous subject/state
  chain far better than several independent mechanisms that can each succeed
  or fail. Fast, dense action is a known weak point ("very fast movement gives
  the model less room to resolve clean geometry"), which matches lightsaber
  contact failing below about 0.75 s per exchange. Reduce competing mechanisms
  in a window before cutting words.
- **Beats hand over state.** Inside a shot, give each timed beat one main
  observable change, and start each beat in the state the last one ended in
  (pose, hands and props, gaze, screen position, camera side and distance).
  The last state of a window is the first state of the next.
- **Anchor spatial words.** "Behind", "around" and "to the side" need an
  anchor: "from behind Siren's left shoulder", "around the rear of the ship".
  Distinguish a physical push in or pull back from an optical zoom.
- **Faces need size.** Identity drifts when a face is small in frame. Put
  identity-critical beats (dialogue, reactions, a reveal) in medium or closer
  shots.
- **Dialogue text.** Ellipses inside `<d>` often hurt intelligibility; use a
  comma or full stop and put the pause in the description outside the tag.
  Emotion words inside `<d>` (such as "(nervous)") get spoken aloud; delivery
  always goes outside. Tie lines to visible events as well as timestamps.
- **Sound placement.** MiniMax staff advice: mention an important synchronised
  sound once beside its action in the shot and again in `overall_soundscape`,
  and describe mix balance in relative words ("the hum sits under the
  dialogue"), not dB or percentages. If unwanted speech appears, fill both
  audio fields explicitly rather than leaving either thin. Voices are the
  exception: they stay on the line in the shot and out of the soundscape (see
  the standing rule on soundscapes).
- **Endings that lock early.** Tests showed final states arriving early and the
  last seconds going static. Keep the final beat in motion up to the end of the
  window.
- **Frame rate and length.** Keep 24 fps; other rates shift voice accents. H3
  frame counts snap to 17k+5 (a 15 s window is 362 frames, 15.08 s).
- **Confirmed: a short "airlock" at window starts.** A render-tested chaining
  tool found that opening each new segment on the previous segment's exact
  arrangement for a moment, then cutting, avoided union artefacts and audio
  clipping at joins. On the user's setup this works with the master wide held
  to 00:02.000; see the standing rule that continuations run 2 seconds before
  cutting.

Sources: MiniMax-H3 official repository and its `h3-prompt-writing` skill
(github.com/MiniMax-AI/MiniMax-H3), the official MMX CLI H3 skill
(github.com/MiniMax-AI/cli), the ComfyUI H3 prompt guide
(docs.comfy.org/tutorials/video/minimax/minimax-h3-prompt-guide), ComfyUI-H3-
Multishot's PROMPTING.md (github.com/jlucasmcrell/ComfyUI-H3-Multishot), the
minimax-h3-prompting validated findings
(github.com/q2522879285-source/minimax-h3-prompting-skill-public), the
minimax-h3-reference-prompt skill (github.com/dzesen/minimax-h3-reference-prompt),
and the fal.ai, Kapwing and InstaSD H3 prompting guides.

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

`references/unarmed-action-director-zh.txt` (unarmed one-on-one combat, in
Chinese) — five action-design laws and an output template; read it for the
laws and the verb examples, and take notation from the mode guide.

`references/multi-opponent-combat-en.txt` (one-versus-many, armed or unarmed) —
highest-priority rules on standalone generation and positive phrasing, then
thirteen directorial principles: protagonist as anchor, overlapping pressure,
1-2 active attackers, partial framing, **spatial blocking**, **environment as
funnel**, body over props, setbacks, repelled rather than knocked out,
breathing beats, camera language, cause-and-effect chains, Foley.

`references/wan2gp-prompts.md` (Wan2GP/WanGP delivery, not H3 format) — line
processing modes, sliding windows, `[/...]` window commands, multiple images,
prompt enhancer, `@`/`@@` syntax, Think mode, macros. Read only when the target
runtime is Wan2GP; see the delivery section at the end of this file.

`references/panorama-360-en.txt` (360 panorama location references) — 1 how H3
treats a panorama, **2 the dewarp sentence**, 3 building a direction map,
4 anchoring shots by direction and image region, 5 degree-based camera moves,
**6 never opening on an unframed wide**, 7 light direction, 8 moving elements and
background figures, 9 sliding windows, 10 quality check.

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
- With real reference pictures or RefMods, `retention_analysis` has one line
  per `<Picture N>` and `<Audio N>`, in the form shown in "Reference Prompts:
  Default Format", not ref-en's `<Subject N> (appears in ...)` form. With
  text-defined subjects (no references) it is the `<Subject N> (appears in
  [Shot 1], [Shot 3])` form, listing only the shots each subject is in.
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
| a new shot — a cut falls on the boundary (not for the user's sequences; see the 2-second rule) | `[Shot 1] The camera cuts to a low three-quarter shot of ...` |
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
| A cut that falls on a window boundary | not for the user's sequences: continue for 2 seconds, then cut (`[Shot 2] At 00:02.000, the camera cuts to ...`) |
| New scene, location or time | `[/new_shot]` on that window |

A ten-second window can hold four or five in-prompt cuts. Window boundaries are
a generation-length constraint, not an editorial one, and should not be used to
express cutting.