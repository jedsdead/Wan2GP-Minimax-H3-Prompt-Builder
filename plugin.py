"""
MiniMax H3 Prompt Builder - a WanGP plugin
------------------------------------------
Builds MiniMax H3 prompts in the format described by MiniMax's own guides:

  base modes
    https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md
  full reference
    https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md

There is one form rather than a set of modes. The output always uses the
six-section reference shape, with N/A in any section that does not apply - a
character is a subject whether or not a reference asset backs it.

Keyframe images are attached in the generator, not here, and the model
associates them with the prompt positionally: a start image is described in
Shot 1's anchor, an end image at the end of the last shot - its final
beat, or its anchor if it has no beats.

The builder owns the mechanical parts: angle-bracket reference tags, shot
timestamps, intra-shot beat timing, speaker IDs, <d> wrapping, retention
markers, task-type prefixes and N/A placeholders.

REFERENCE LINKING
  Each cast/subject entry has an opt-in reference block. Ticking "Is a
  referenced subject" reveals a source asset, retention marker, and links to a
  reference audio (voice timbre) and reference video (motion). Values left in
  a collapsed block are ignored, so they cannot leak into the prompt.

  An asset serving several roles is described in one sentence per the guide,
  so roles are gathered per label and written once.

KNOWN SHAPE
  - Shots, beats and entries use fixed slots revealed by add and remove
    buttons rather than components created on demand. Gradio makes true
    dynamic lists awkward inside an injected plugin layout.
  - Prose templates are a first pass; read the output before committing to a
    long generation.
"""

import re

import gradio as gr

from shared.utils.plugins import WAN2GPPlugin


# =============================================================================
# Limits
# =============================================================================

MAX_ENTRIES = 8      # cast and subjects are one list
MAX_SPEAKERS = 6     # how many speaker slots the Speaker dropdown offers
MAX_SHOTS = 6
MAX_BEATS = 5

# Substrings that mark a model as MiniMax H3. Checked case-insensitively
# against the model type reported by on_model_change.
H3_MODEL_HINTS = ("minimax", "h3")


# =============================================================================
# Vocabulary
# =============================================================================

# Guidance only. Keyframe images are attached in the generator, and the model
# associates them with the prompt positionally - so there is nothing for the
# builder to configure, only somewhere to say where each one gets described.
KEYFRAME_NOTE = (
    "Keyframe images are attached in the generator, not here. If you are "
    "starting from an image, describe it in **Shot 1's anchor**. If you have "
    "an end image, describe it at the **end of the last shot** - its final "
    "beat, or its anchor if that shot has no beats. "
    "Reference labels like `<Picture 1>` resolve to the images loaded above, "
    "in order."
)

STYLES = [
    "",
    # live action, general
    "Live-action, cinematic", "Live-action, documentary",
    "Observational documentary", "Handheld, naturalistic",
    "Multi-camera sitcom", "Music-video", "Vintage film",
    "Home-video footage", "Found-footage", "Mockumentary",
    "Security-camera footage", "Newsreel",
    # live action, genre
    "Neo-noir", "Film noir", "Western", "Spaghetti western",
    "War epic", "Historical epic", "Period drama", "Courtroom drama",
    "Heist thriller", "Spy thriller", "Psychological thriller",
    "Slasher horror", "Cosmic horror", "Body horror", "Gothic horror",
    "Science-fiction epic", "Cyberpunk", "Steampunk", "Post-apocalyptic",
    "Dystopian", "Space opera", "Creature feature", "Disaster movie",
    "Superhero blockbuster", "Action blockbuster", "Martial-arts film",
    "Romantic comedy", "Screwball comedy", "Coming-of-age drama",
    "Road movie", "Sports drama", "Biopic", "Musical", "Fantasy epic",
    "French New Wave", "Italian neorealism", "Kitchen-sink realism",
    "German expressionism", "Silent film", "Technicolor melodrama",
    "1970s New Hollywood", "1980s VHS aesthetic", "Surrealist film",
    # animation
    "2D-animated", "Hand-painted anime film", "Cel-shaded anime",
    "Modern 3D animated feature", "1990s 2D animated feature",
    "1930s rubber-hose cartoon", "Saturday-morning cartoon",
    "Adult animated comedy", "Stop-motion", "Claymation",
    "Puppet animation", "Papercraft animation", "Rotoscoped animation",
    "Motion comic", "Pixel-art animation", "Silhouette animation",
    # illustrative and rendered
    "Watercolour", "Oil-painted", "Charcoal sketch", "Ink and wash",
    "Comic-book panel", "Graphic-novel", "Storyboard sketch",
    "3D CG", "Photorealistic render", "Low-poly render", "Wireframe render",
]

FRAMINGS = [
    "", "an extreme close-up", "a close-up", "a medium close-up",
    "a medium shot", "a medium-wide shot", "a wide shot",
    "an extreme wide shot", "an establishing shot", "a two-shot",
    "a three-shot", "an over-the-shoulder shot", "a low-angle shot",
    "a high-angle shot", "an overhead shot", "a worm's-eye view",
    "a dutch-angle shot", "a point-of-view shot", "a profile shot",
    "a silhouette shot", "an insert shot", "a cutaway", "a master shot",
    "a reflection shot", "a through-the-window shot",
]

# Fixed vocabulary from the guide's camera table. Not user-extendable:
# the model was trained on these exact expressions.
# The guide's own camera table - the expressions the model was trained on -
# followed by a short group of movements the table lacks. Rigs live in RIGS
# below: a dolly, crane, drone or steadicam is how the camera is mounted, not
# what it does, and several "moves" named after rigs are just spec moves under
# another name (a dolly in is Push In, a dolly left is Truck Left, a 360 is an
# Arc Shot).
MOTION_TYPES = [
    "", "Zoom In", "Zoom Out", "Push In", "Pull Out", "Pan Left", "Pan Right",
    "Truck Left", "Truck Right", "Tilt Up", "Tilt Down", "Pedestal Up",
    "Pedestal Down", "Arc Shot", "Tracking Shot", "Static Shot",
    "Shake Slightly", "Shake Strongly", "POV", "Roll Clockwise",
    "Roll Counterclockwise",
    # --- beyond the guide's table ---
    "Full 360 Orbit", "Spiral Around Subject", "Dolly Zoom",
    "Crash Zoom In", "Snap Zoom Out", "Whip Pan", "Rack Focus",
    "Trail Behind Subject", "Lead Subject Backwards", "Locked Off",
]

# How each motion type reads as a verb phrase after "The camera ...".
MOTION_VERBS = {
    "Zoom In": "zooms in", "Zoom Out": "zooms out",
    "Push In": "pushes in", "Pull Out": "pulls out",
    "Pan Left": "pans left", "Pan Right": "pans right",
    "Truck Left": "trucks left", "Truck Right": "trucks right",
    "Tilt Up": "tilts up", "Tilt Down": "tilts down",
    "Pedestal Up": "rises on the pedestal",
    "Pedestal Down": "lowers on the pedestal",
    "Arc Shot": "moves in an arc around the subject",
    "Tracking Shot": "tracks the subject",
    "Static Shot": "holds a static shot",
    "Shake Slightly": "shakes slightly", "Shake Strongly": "shakes strongly",
    "POV": "takes the subject's point of view",
    "Roll Clockwise": "rolls clockwise",
    "Roll Counterclockwise": "rolls counterclockwise",
    # --- beyond the guide's table ---
    "Full 360 Orbit": "makes a full 360-degree orbit around the subject",
    "Spiral Around Subject": "spirals around the subject",
    "Dolly Zoom": "performs a dolly zoom",
    "Crash Zoom In": "crash zooms in", "Snap Zoom Out": "snap zooms out",
    "Whip Pan": "whip pans", "Rack Focus": "racks focus",
    "Trail Behind Subject": "trails behind the subject",
    "Lead Subject Backwards": "leads the subject backwards",
    "Locked Off": "holds a locked-off frame",
}

# How the camera is mounted. Written as a trailing clause on the camera
# sentence: "The camera pushes in slowly, mounted on a drone."
RIGS = [
    "", "on a tripod", "handheld", "on a shoulder rig", "on a steadicam",
    "on a gimbal", "on a dolly track", "on a crane", "on a jib arm",
    "on a drone", "on a cable cam", "on a car mount", "on a hood mount",
    "on a slider", "on a motion-control rig", "in an underwater housing",
    "body-mounted to the subject", "mounted to a moving vehicle",
    "hand-passed between operators", "on a rickshaw rig",
]

AMPLITUDES = ["", "with small amplitude", "with large amplitude"]
SPEEDS = ["", "at slow speed", "at fast speed"]

CUT_VERBS = [
    "the camera cuts to", "the shot cuts to", "the shot hard-cuts to",
    "the shot smash-cuts to", "the shot match-cuts to",
    "the shot jump-cuts to", "the shot transitions to",
    "the shot changes to", "the shot switches to",
    "the shot cross-dissolves to", "the shot dissolves to",
    "the shot fades to", "the shot wipes to", "the shot whip-pans to",
    "the shot irises to", "the shot cuts away to",
]

LANGUAGES = [
    "English", "Mandarin Chinese", "Cantonese", "Japanese", "Korean",
    "Spanish", "French", "German", "Italian", "Portuguese", "Dutch",
    "Russian", "Polish", "Ukrainian", "Czech", "Greek", "Turkish",
    "Arabic", "Hebrew", "Farsi", "Hindi", "Urdu", "Bengali", "Tamil",
    "Thai", "Vietnamese", "Indonesian", "Tagalog", "Swahili", "Yoruba",
    "Swedish", "Norwegian", "Danish", "Finnish", "Icelandic",
    "Irish", "Welsh", "Scottish Gaelic", "Latin",
]

VOICE_AGES = ["", "child", "teenage", "young", "young adult", "middle-aged",
              "older", "elderly", "ageless"]
VOICE_GENDERS = ["", "female", "male", "androgynous"]
VOICE_PITCH = ["", "low", "medium", "high"]
VOICE_TIMBRE = [
    "", "clear", "raspy", "breathy", "warm", "nasal", "gravelly", "bright",
    "weathered", "smooth", "resonant", "reedy", "husky", "thin", "rich",
    "hoarse", "silky", "grainy", "booming", "wavering", "metallic",
]
VOICE_RATE = [
    "", "slow", "measured", "unhurried", "deliberate", "quick", "rapid",
    "clipped", "halting", "breathless", "drawling", "steady", "urgent",
]

# Reference mode -----------------------------------------------------------

ASSET_KINDS = ["Subject", "Picture", "Video", "Audio"]

# The model itself accepts up to three reference videos and three reference
# audios, but WanGP's Ref2VA panel exposes two of each ("Use One" / "Use Two"),
# so these stop at two. The label names the actual slot rather than being
# auto-numbered, and slots follow upload order - reordering your uploads
# silently reassigns them.
REF_VIDEO_SLOTS = ["", "Video 1", "Video 2"]
REF_AUDIO_SLOTS = ["", "Audio 1", "Audio 2"]

# Ref2VA takes up to nine reference images. Offered as plain text because
# _ref_tags() adds the angle brackets on the way out.
REF_PICTURE_SLOTS = [f"Picture {n + 1}" for n in range(9)]

GRADING = [
    "", "a vibrant colour grade", "a muted colour grade",
    "a high-contrast grade", "a low-contrast grade",
    "a teal-and-orange grade", "a bleach-bypass grade",
    "a warm vintage film grade", "a cool desaturated grade",
    "a pastel low-contrast grade", "a faded retro grade",
    "a rich saturated technicolor grade", "a sepia-toned grade",
    "a cross-processed grade", "a neon cyberpunk grade",
    "a sun-bleached grade", "a cold blue-grey grade",
    "a moody green-tinted grade", "a monochrome grade",
    "a high-key bright grade", "a crushed-blacks grade",
    "an amber golden grade", "a silver-halide grade",
]

# What an audio reference controls. The guide treats these as distinct roles:
# a voice reference names its speaker, a music-style reference belongs to the
# music layer, and a beat reference controls timing without instrumentation.
MUSIC_REF_ROLES = [
    "style", "beat and rhythm", "style and rhythm",
    "instrumentation", "mood",
]

VISUAL_RETENTION = ["fully_preserved", "partially_preserved",
                    "attribute_transfer", "weak_reference"]

AUDIO_RETENTION = ["fully_copy", "partially_copy", "reference", "weak_reference"]

# Combined for the subject dropdown, deduped - weak_reference is in both.
ALL_RETENTION = list(dict.fromkeys(VISUAL_RETENTION + AUDIO_RETENTION))

TASK_TYPES = ["keyframe completion", "reference generation", "video editing",
              "video continuation", "audio reuse", "audio reference"]



# -- Scene ------------------------------------------------------------------
# Woven into the opening shot rather than emitted as separate fields: H3 keeps
# all visual description inside integrated_multimodal_description.

LOCATIONS = [
    "a rain-slicked alley behind a nightclub", "a farmhouse kitchen at dawn",
    "an airport departure lounge", "a lighthouse gallery in a storm",
    "a disused swimming pool", "a rooftop garden above the city",
    "an antique shop crowded with clocks", "a motorway service station at night",
    "a boat deck on open water", "a stone chapel lit by candles",
    "a launderette at closing time", "a records archive in a basement",
    "a ski lift above treeline", "a bustling New York street at dusk",
    "a covered market hall", "a late-night launderette", "a tiled underpass",
    "a rain-soaked city street", "a suburban kitchen", "a hotel corridor",
    "a crowded subway platform", "a quiet library reading room",
    "an empty car park at night", "a coastal fishing dock",
    "a pine forest clearing", "a desert highway", "a rooftop at dusk",
    "a hospital waiting room", "a school classroom", "a dive bar",
    "an office at night", "a country lane", "a snowbound cabin",
    "a train carriage", "a cathedral interior", "a warehouse floor",
    "a greenhouse", "a mountain ridge", "a riverbank at dawn",
]

SCENE_LIGHTING = [
    "a single bare bulb", "car headlights sweeping across",
    "television glow in a dark room", "shafts of light through blinds",
    "the cold blue of a screen", "stage lighting from above",
    "a red safelight", "lightning flashes", "sodium streetlight",
    "sunrise backlight through haze",
    "golden hour light", "blue hour light", "harsh midday sun",
    "soft overcast light", "moody low-key lighting", "high-key lighting",
    "flickering candlelight", "neon glow", "a backlit silhouette",
    "practical lamps only", "dramatic hard shadows", "rim lighting",
    "dappled sunlight through leaves", "harsh fluorescent light",
    "moonlight", "firelight", "diffused window light",
    "strobing club lighting", "underwater caustics", "streetlight sodium glow",
]

SCENE_ATMOSPHERE = [
    "torrential rain", "sea spray", "ash falling", "petals drifting",
    "insects circling a lamp", "condensation on every surface",
    "a low ground mist", "sunbeams cutting through dust",
    "thick fog", "light mist", "drizzling rain", "heavy rain",
    "drifting dust", "smoke haze", "floating particles", "still, calm air",
    "gusting wind", "humid haze", "crisp clear air", "falling snow",
    "swirling sand", "rising steam", "heat shimmer", "drifting embers",
    "industrial smog", "morning dew",
]

# -- Camera body and lens ---------------------------------------------------
# Production constants: emitted once in the opening shot, not per shot.

CAMERA_TYPES = [
    "an IMAX camera", "IMAX 70mm film", "an Arri Alexa digital cinema camera",
    "a RED digital cinema camera", "a Panavision camera",
    "a Blackmagic cinema camera", "Super 35mm film", "Super 16mm film",
    "16mm film", "8mm film", "35mm film", "a 4K digital camera",
    "an 8K digital camera", "a DSLR", "a mirrorless camera",
    "a GoPro action camera", "a drone camera", "a vintage VHS camcorder",
    "a phone camera", "a security camera", "a Polaroid instant camera",
    "black-and-white film stock", "expired film stock",
]

LENS_TYPES = [
    "a 14mm ultra-wide lens", "a 24mm wide-angle lens", "a 35mm lens",
    "a 50mm standard lens", "an 85mm portrait lens", "a 100mm macro lens",
    "a 135mm telephoto lens", "a 200mm telephoto lens", "an anamorphic lens",
    "a fisheye lens", "a tilt-shift lens", "a vintage soft-focus lens",
    "a prime lens", "a zoom lens",
    "a large-format lens with shallow depth of field",
    "a probe lens", "a split-diopter lens",
]

# -- Audio presets ----------------------------------------------------------

SOUNDSCAPE_PRESETS = [
    "wind across an open field", "rain on a tin roof",
    "a kettle coming to the boil", "a fridge humming in a quiet kitchen",
    "seagulls over a harbour", "an aircraft passing overhead",
    "keyboard tapping and chair creaks", "a crowd in a stadium",
    "hooves on cobblestones", "a clock chiming the hour",
    "surf dragging over shingle", "a generator running outside",
    "room tone and distant traffic", "rain against windows",
    "wind through trees", "birdsong and rustling leaves",
    "waves against a shoreline", "a crackling fire",
    "crowd chatter and clinking glasses", "market noise and trolley wheels",
    "fluorescent hum and footsteps on tile", "machinery hum",
    "a ticking clock in a quiet room", "distant sirens",
    "train rumble through a tunnel", "church bells",
    "cicadas in the heat", "creaking timber", "coins and paper shifting",
    "breathing and shifting fabric", "footsteps on gravel",
    "a washing drum tumbling", "near silence with faint air movement",
]

MUSIC_PRESETS = [
    # --- score genres and screen-music styles ---
    "a sweeping orchestral adventure score",
    "a brooding orchestral drama score",
    "a soaring romantic string score",
    "a tense thriller score with pulsing ostinato strings",
    "a horror score with dissonant strings and sudden stingers",
    "a slasher-film synth score",
    "a stalking analogue synth pulse",
    "an 1980s retro-synthwave score",
    "a cyberpunk industrial synth score",
    "a science-fiction score with choir and low brass",
    "a fantasy score with harp, horns and choir",
    "a western score with lone guitar, whistle and trumpet",
    "a spaghetti-western score with twanging guitar and vocal wails",
    "a noir jazz score with muted trumpet and brushed drums",
    "a smoky late-night saxophone score",
    "a French New Wave score with light jazz piano",
    "an Italian neorealist score with mournful accordion",
    "a period-drama chamber score with strings and piano",
    "a baroque harpsichord score",
    "a war-film score with martial snare and low brass",
    "a heist score with cool bass groove and hi-hats",
    "a spy-thriller score with surf guitar and brass stabs",
    "a superhero score with heroic brass fanfare",
    "a disaster-movie score with rising brass and timpani",
    "a courtroom-drama score with restrained piano",
    "a sports-montage score with driving drums and brass",
    "a coming-of-age score with warm indie guitars",
    "an indie-film score with fingerpicked acoustic guitar",
    "a mumblecore score with lo-fi bedroom pop",
    "a road-movie score with slide guitar and harmonica",
    "a documentary score with minimal piano and strings",
    "a nature-documentary score with wide orchestral awe",
    "a true-crime score with cold synth drones",
    "a sitcom score with bright brassy stings",
    "a soap-opera score with lush melodrama strings",
    "a children's-animation score with playful woodwinds",
    "an anime score with emotive piano and strings",
    "a magical-girl anime score with bells and bright synths",
    "a video-game cinematic score with hybrid orchestra and percussion",
    "a silent-film score with rollicking piano",
    "a musical-theatre score with full pit orchestra",
    "a Bollywood score with tabla, strings and vocals",
    "a Nordic-noir score with icy ambient textures",
    "a pastoral animation score with piano and woodwinds",
    "a minimalist score with repeating arpeggios",
    "a post-rock crescendo with building guitars and drums",
    "a trailer score with braams and rising percussion",
    "an end-credits score with reflective piano and strings",
    # --- textures and small ensembles ---
    "a lone cello line held under the scene",
    "pulsing synth arpeggios building slowly",
    "a plaintive solo violin",
    "muted horns over a slow drum shuffle",
    "distant church organ",
    "a music-box melody, slightly out of tune",
    "heavy orchestral brass swells",
    "fingerpicked banjo at a walking pace",
    "an ominous low drone with occasional percussion hits",
    "bright pizzicato strings",
    "sparse solo piano at a slow tempo",
    "a low sustained string drone",
    "a walking double bass with brushed drums",
    "warm analogue synth pads",
    "a lone acoustic guitar figure",
    "slow-building orchestral strings",
    "a muted trumpet over light percussion",
    "minimal electronic pulse",
    "a distant choir held under the scene",
    "plucked harp over sustained cello",
    "an upright piano with heavy room reverb",
    "tense staccato strings",
    "a slow waltz on accordion",
    "ambient drone with no clear pulse",
]

# =============================================================================
# Plugin
# =============================================================================

class H3PromptBuilderPlugin(WAN2GPPlugin):
    """Injects an H3 Prompt Builder accordion after the prompt box."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_model_type = ""

    # -- lifecycle ---------------------------------------------------------

    def setup_ui(self) -> None:
        self.request_component("prompt")
        # The sample plugin uses this id as the model selector, so it should
        # exist. Requested here so it is populated as self.model_choice_target
        # before create_ui runs; the wiring degrades gracefully if not.
        self.request_component("model_choice_target")
        self.insert_after("prompt", self.create_ui)

    def on_model_change(self, state, model_type) -> None:
        # Notification only - the dispatcher discards whatever this returns,
        # so it cannot toggle visibility. It records the model so the insert
        # action can warn when the wrong one is selected.
        self.current_model_type = model_type or ""

    def is_h3_model(self) -> bool:
        m = (self.current_model_type or "").lower()
        return any(h in m for h in H3_MODEL_HINTS)

    # -- UI ----------------------------------------------------------------

    def create_ui(self):
        """
        Called by PluginManager inside the prompt component's live parent
        context. Everything must be created inside the single accordion
        below: insert_after does parent.children.pop(-1) and assumes the
        constructor added exactly one top-level child.
        """
        entries, shots = [], []

        def dd(choices, label, **kw):
            return gr.Dropdown(choices, label=label, value="",
                               allow_custom_value=True, **kw)

        def locked_dd(choices, label, **kw):
            # Spec-enumerated values - typing a custom one produces language
            # the model was not trained to read as camera direction. A blank
            # entry is prepended so "unset" is a valid choice and the field
            # can be cleared.
            options = list(choices)
            if "" not in options:
                options = [""] + options
            return gr.Dropdown(options, label=label, value="", **kw)

        with gr.Accordion("MiniMax H3 Prompt Builder", open=False,
                          elem_id="h3_prompt_builder") as root:

            model_warning = gr.Markdown(visible=False)

            gr.Markdown(KEYFRAME_NOTE)

            ref_mode = gr.Checkbox(
                label="Use reference mode (Ref2VA)",
                value=False,
                info="Reveals reference subjects, the source video and the "
                     "task type. Leave unticked and none of it is written.",
            )

            with gr.Row():
                duration = gr.Number(label="Duration (seconds)", value=8.0,
                                     minimum=0.5, step=0.5)
                style = dd(STYLES, "Style")
                grading = dd(GRADING, "Colour grade")

            with gr.Row():
                location = dd(LOCATIONS, "Location")
                lighting = dd(SCENE_LIGHTING, "Lighting")
                atmosphere = dd(SCENE_ATMOSPHERE, "Atmosphere")

            with gr.Row():
                camera_type = dd(CAMERA_TYPES, "Camera / stock")

            # ---- source video (FL2VA continue / Ref2VA video reference) ----
            # ---- cast and subjects (one list) ------------------------------
            with gr.Accordion("Cast & subjects", open=True):
                gr.Markdown(
                    "Everything that appears - people, animals, places, props. "
                    "Each entry becomes a `<Subject N>` definition, and the "
                    "shot text refers to the label rather than repeating the "
                    "description.\n\n"
                    "**Speaker** is assigned by you, not derived from position: "
                    "if entry 2 is a car and entry 3 talks, entry 3 can be S2. "
                    "**Source asset** is only for entries that come from a "
                    "reference picture."
                )

                entry_count = gr.State(0)

                for i in range(MAX_ENTRIES):
                    with gr.Group(visible=False) as grp:
                        gr.Markdown(f"**Subject {i + 1}**")
                        e_desc = gr.Textbox(
                            label="Description",
                            placeholder="the fishmonger, heavy apron, forearms wet to the elbow",
                        )
                        with gr.Row():
                            e_kind = gr.Dropdown(ASSET_KINDS, label="Label",
                                                 value="Subject")
                            e_speaker = gr.Dropdown(
                                [""] + [f"S{n + 1}" for n in range(MAX_SPEAKERS)],
                                label="Speaker", value="",
                                info="Leave blank if it never vocalises",
                            )
                            e_onscreen = gr.Dropdown(
                                ["", "on-screen", "off-screen"],
                                label="Presence", value="",
                            )
                        with gr.Row():
                            e_age = dd(VOICE_AGES, "Voice age")
                            e_gender = dd(VOICE_GENDERS, "Voice gender")
                            e_pitch = dd(VOICE_PITCH, "Pitch")
                        with gr.Row():
                            e_timbre = dd(VOICE_TIMBRE, "Timbre")
                            e_rate = dd(VOICE_RATE, "Rate")
                            e_accent = gr.Textbox(label="Accent (optional)")
                        with gr.Row():
                            e_lang = gr.Dropdown(
                                LANGUAGES, label="Language", value="English",
                                allow_custom_value=True,
                                info="Used for this speaker's dialogue tags",
                            )
                        with gr.Group(visible=False) as e_ref_block:
                            with gr.Row():
                                e_source = gr.Dropdown(
                                    REF_PICTURE_SLOTS, label="Source asset",
                                    value=[], multiselect=True,
                                    allow_custom_value=True,
                                    info="Which reference image this comes "
                                         "from - type another tag if needed",
                                )
                                e_retention = gr.Dropdown(
                                    [""] + ALL_RETENTION, label="Retention",
                                    value="",
                                )
                            with gr.Row():
                                e_note = gr.Textbox(
                                    label="What is retained",
                                    placeholder="the apron and wet forearms are retained",
                                )
                                e_shots = gr.Textbox(label="Appears in shots",
                                                     placeholder="1, 2")
                            with gr.Row():
                                e_voice_from = gr.Dropdown(
                                    REF_AUDIO_SLOTS, label="Voice from",
                                    value="",
                                    info="A reference audio supplying this "
                                         "speaker's timbre",
                                )
                                e_motion_from = gr.Dropdown(
                                    REF_VIDEO_SLOTS, label="Motion from",
                                    value="",
                                    info="A reference video supplying this "
                                         "subject's movement or performance",
                                )

                    entries.append({
                        "group": grp, "kind": e_kind, "desc": e_desc,
                        "speaker": e_speaker, "onscreen": e_onscreen,
                        "age": e_age, "gender": e_gender, "pitch": e_pitch,
                        "timbre": e_timbre, "rate": e_rate, "accent": e_accent,
                        "lang": e_lang,
                        "ref_block": e_ref_block,
                        "source": e_source, "retention": e_retention,
                        "note": e_note, "shots": e_shots,
                        "voice_from": e_voice_from,
                        "motion_from": e_motion_from,
                    })

                with gr.Row():
                    add_entry = gr.Button("Add subject", size="sm")
                    rm_entry = gr.Button("Remove last subject", size="sm")

                _entry_out = [entry_count] + [e["group"] for e in entries]
                add_entry.click(
                    fn=lambda n: self._step_count(n, +1, MAX_ENTRIES),
                    inputs=[entry_count], outputs=_entry_out,
                )
                rm_entry.click(
                    fn=lambda n: self._step_count(n, -1, MAX_ENTRIES),
                    inputs=[entry_count], outputs=_entry_out,
                )

            with gr.Accordion("Reference task", open=False,
                              visible=False) as ref_task_section:
                gr.Markdown(
                    "Only needed when reference assets are involved. The task "
                    "type tells the model what kind of job this is, and is "
                    "written as a prefix on the summary."
                )
                task_types = gr.CheckboxGroup(
                    TASK_TYPES, label="Task type - combined with + in summary",
                )

                with gr.Accordion("Source video", open=False) as video_section:
                    video_role = gr.Radio(
                        ["none", "continue from it", "edit it",
                         "reference its camera and cutting only"],
                        label="Role of an attached video", value="none",
                    )
                    video_desc = gr.Textbox(
                        label="What the video contributes",
                        placeholder="handheld camera movement and cutting rhythm",
                    )
                    video_retention = locked_dd(VISUAL_RETENTION,
                                                "Retention (picture)")

                    gr.Markdown(
                        "If the video's **audio** is being reused or referenced, "
                        "set it below - audio uses a different marker set from "
                        "picture, and gets its own <Audio N> label. Tick the "
                        "matching task type (audio reuse, or audio reference).\n\n"
                        "WanGP's Audio References selector is one dropdown, so "
                        "**Use Reference-Video Soundtrack(s)** and standalone "
                        "audio references are mutually exclusive - use this "
                        "section or the audio slots, not both."
                    )
                    with gr.Row():
                        video_audio = locked_dd(AUDIO_RETENTION,
                                                "Retention (audio)")
                        video_audio_desc = gr.Textbox(
                            label="What the audio contributes",
                            placeholder="the original spoken dialogue",
                        )


            with gr.Accordion("Shots", open=True):
                shot_count = gr.State(1)
                gr.Markdown(
                    "Shot 1 opens the clip and takes no cut time. If you are "
                    "working from a start and an end image, a single shot "
                    "usually works best so the model can interpolate between "
                    "them."
                )

                for si in range(MAX_SHOTS):
                    with gr.Group(visible=(si == 0)) as sgrp:
                        gr.Markdown(f"**Shot {si + 1}**")
                        with gr.Row():
                            if si == 0:
                                sh_cut = gr.Textbox(
                                    label="Cut time", value="opening",
                                    interactive=False,
                                )
                                sh_cutverb = gr.Textbox(
                                    label="Transition", value="-",
                                    interactive=False,
                                )
                            else:
                                sh_cut = gr.Number(label="Cut at (seconds)",
                                                   value=None, minimum=0, step=0.5)
                                sh_cutverb = locked_dd(CUT_VERBS, "Transition")
                            sh_framing = dd(FRAMINGS, "Framing")
                            sh_lens = dd(LENS_TYPES, "Lens")
                        with gr.Row():
                            sh_motion = locked_dd(MOTION_TYPES, "Camera motion")
                            sh_ampl = locked_dd(AMPLITUDES, "Amplitude")
                            sh_speed = locked_dd(SPEEDS, "Speed")
                            sh_rig = dd(RIGS, "Rig")
                        sh_anchor = gr.Textbox(
                            label="Anchor - composition and what is in frame",
                            placeholder="a fishmonger behind a crushed-ice counter in a covered market hall",
                        )

                        beats = []
                        beat_count = gr.State(0)
                        for bi in range(MAX_BEATS):
                            with gr.Group(visible=False) as bgrp:
                                with gr.Row():
                                    b_type = gr.Dropdown(
                                        ["action", "dialogue"], label="Type",
                                        value="action",
                                        info="Label only - a beat becomes "
                                             "dialogue when it has spoken words",
                                    )
                                    b_speaker = gr.Dropdown(
                                        [f"Subject {n + 1}" for n in range(MAX_ENTRIES)],
                                        label="Who", value=[],
                                        multiselect=True,
                                        info="Speaker IDs are added "
                                             "automatically where they apply",
                                    )
                                    b_lang = gr.Dropdown(
                                        [""] + LANGUAGES,
                                        label="Language override",
                                        value="", allow_custom_value=True,
                                        info="Blank uses the speaker's language",
                                    )
                                b_action = gr.Textbox(
                                    label="Action / delivery (outside <d>)",
                                    placeholder="turns her head and says",
                                )
                                b_speech = gr.Textbox(
                                    label="Spoken words (inside <d>) - leave blank for non-verbal",
                                )
                                with gr.Row():
                                    b_at = gr.Number(
                                        label="At (seconds, optional)",
                                        value=None, minimum=0, step=0.5,
                                        info="When this beat happens. A shot "
                                             "cutting at 4s can have its first "
                                             "beat at 4.5s",
                                    )
                                    b_carries = gr.Checkbox(
                                        label="Line carries across the next cut",
                                        value=False,
                                    )
                            beats.append({
                                "group": bgrp, "type": b_type,
                                "speaker": b_speaker, "lang": b_lang,
                                "action": b_action, "speech": b_speech,
                                "at": b_at, "carries": b_carries,
                            })

                        with gr.Row():
                            add_beat = gr.Button("Add beat", size="sm")
                            rm_beat = gr.Button("Remove last beat", size="sm")

                        _beat_out = [beat_count] + [b["group"] for b in beats]
                        add_beat.click(
                            fn=lambda n: self._step_count(n, +1, MAX_BEATS),
                            inputs=[beat_count], outputs=_beat_out,
                        )
                        rm_beat.click(
                            fn=lambda n: self._step_count(n, -1, MAX_BEATS),
                            inputs=[beat_count], outputs=_beat_out,
                        )

                    shots.append({
                        "group": sgrp, "cut": sh_cut, "cutverb": sh_cutverb,
                        "framing": sh_framing, "lens": sh_lens,
                        "motion": sh_motion, "rig": sh_rig,
                        "ampl": sh_ampl, "speed": sh_speed,
                        "anchor": sh_anchor, "beat_count": beat_count,
                        "beats": beats,
                    })

                with gr.Row():
                    add_shot = gr.Button("Add shot", size="sm")
                    rm_shot = gr.Button("Remove last shot", size="sm")

                _shot_out = [shot_count] + [s["group"] for s in shots]
                add_shot.click(
                    fn=lambda n: self._step_count(n, +1, MAX_SHOTS, minimum=1),
                    inputs=[shot_count], outputs=_shot_out,
                )
                rm_shot.click(
                    fn=lambda n: self._step_count(n, -1, MAX_SHOTS, minimum=1),
                    inputs=[shot_count], outputs=_shot_out,
                )

            # ---- audio -----------------------------------------------------
            with gr.Accordion("Audio", open=False):
                soundscape_presets = gr.Dropdown(
                    SOUNDSCAPE_PRESETS, label="Soundscape presets",
                    value=[], multiselect=True, allow_custom_value=True,
                )
                soundscape = gr.Textbox(
                    label="Soundscape - custom / additional information",
                    lines=2,
                    placeholder="A wide market-hall echo carries trolley wheels and distant haggling.",
                )
                # Reference fields sit under their own field and follow the
                # Ref2VA switch, like every other reference control.
                with gr.Group(visible=False) as ambience_ref_block:
                    with gr.Row():
                        ambience_from = gr.Dropdown(
                            REF_AUDIO_SLOTS, label="Ambience from", value="",
                            info="A reference audio supplying the ambient bed",
                        )
                        ambience_retention = gr.Dropdown(
                            [""] + AUDIO_RETENTION, label="Retention", value="",
                        )
                gr.Markdown(
                    "Dialogue, singing and music the characters can hear "
                    "belong in a beat, not here."
                )
                music_presets = gr.Dropdown(
                    MUSIC_PRESETS, label="Music presets",
                    value=[], multiselect=True, allow_custom_value=True,
                )
                music = gr.Textbox(
                    label="Non-diegetic music - custom / additional information",
                    lines=2,
                )
                with gr.Group(visible=False) as music_ref_block:
                    with gr.Row():
                        music_from = gr.Dropdown(
                            REF_AUDIO_SLOTS, label="Music from", value="",
                            info="A reference audio supplying the score",
                        )
                        music_role = gr.Dropdown(
                            MUSIC_REF_ROLES, label="It controls", value="style",
                        )
                        music_retention = gr.Dropdown(
                            [""] + AUDIO_RETENTION, label="Retention", value="",
                        )

            status = gr.Markdown("")

            with gr.Group(visible=False) as summary_block:
                gr.Markdown(
                    "The summary is a reference-mode section. Base modes have "
                    "no summary field, so it is not written when reference "
                    "mode is off."
                )
                summary_text = gr.Textbox(
                    label="Summary",
                    lines=3,
                    placeholder="A fishmonger works the counter of a covered market hall as a schoolboy stops to ask about the fish.",
                )
                with gr.Row():
                    draft_summary_btn = gr.Button("Draft summary from fields",
                                                  size="sm")
                summary_status = gr.Markdown("")

            gr.Markdown(
                "**Read the prompt before you generate.** Fields are stitched "
                "into sentences from templates, so wording you enter may not "
                "agree grammatically with the phrasing around it - "
                "*\"the target video shows a woman faces the door\"* rather "
                "than *facing*. The prompt box is editable; fixing it there "
                "takes seconds and the model reads what you leave."
            )

            with gr.Row():
                insert_btn = gr.Button("Insert into prompt", variant="primary")
                clear_btn = gr.Button("Clear all fields")

        # ---- wiring -------------------------------------------------------

        flat = [ref_mode, duration, style, grading, location, lighting, atmosphere,
                camera_type,
                video_role, video_desc, video_retention,
                video_audio, video_audio_desc, entry_count]
        for e in entries:
            flat += [e["kind"], e["desc"], e["speaker"], e["onscreen"],
                     e["age"], e["gender"], e["pitch"], e["timbre"],
                     e["rate"], e["accent"], e["lang"],
                     e["source"], e["retention"], e["note"], e["shots"],
                     e["voice_from"], e["motion_from"]]
        flat += [task_types, summary_text, shot_count]
        for s in shots:
            flat += [s["cut"], s["cutverb"], s["framing"], s["lens"],
                     s["motion"], s["ampl"], s["speed"], s["rig"],
                     s["anchor"], s["beat_count"]]
            for b in s["beats"]:
                flat += [b["type"], b["speaker"], b["lang"], b["action"],
                         b["speech"], b["at"], b["carries"]]
        flat += [ambience_from, ambience_retention,
                 soundscape_presets, soundscape,
                 music_from, music_role, music_retention,
                 music_presets, music]

        insert_btn.click(fn=self._build, inputs=flat,
                         outputs=[self.prompt, status])

        # One switch for everything Ref2VA. Values in a hidden block are
        # ignored by the assembly too, so nothing can leak in.
        ref_blocks = ([ref_task_section]
                      + [e["ref_block"] for e in entries]
                      + [ambience_ref_block, music_ref_block,
                         summary_block])
        ref_mode.change(
            fn=lambda on: [gr.update(visible=bool(on))] * len(ref_blocks),
            inputs=[ref_mode], outputs=ref_blocks,
        )

        draft_summary_btn.click(
            fn=self._draft_summary, inputs=flat,
            outputs=[summary_text, summary_status],
        )
        all_groups = (
            [e["group"] for e in entries]
            + [e["ref_block"] for e in entries]
            + [ambience_ref_block, music_ref_block, summary_block]
            + [s["group"] for s in shots]
        )
        for s in shots:
            all_groups += [b["group"] for b in s["beats"]]

        clear_btn.click(fn=self._clear, inputs=[], outputs=flat + all_groups)

        self._wire_model_visibility(root, model_warning)

    def _wire_model_visibility(self, root, warning):
        """
        Try to hide the builder when a non-H3 model is selected.

        on_model_change cannot do this: PluginManager's dispatcher discards
        the return value, so it can only record state. Binding our own
        handler to the model selector works, but the component id is a guess
        and varies between WanGP versions - so a failure here degrades to
        an always-visible builder rather than breaking the plugin.
        """
        try:
            selector = getattr(self, "model_choice_target", None)
            if selector is None:
                print("[MiniMaxH3PromptBuilder] no model selector found; "
                      "the builder will stay visible for every model.")
                return

            def _toggle(model_type):
                m = str(model_type or "").lower()
                is_h3 = any(h in m for h in H3_MODEL_HINTS)
                return gr.update(visible=is_h3), gr.update(visible=False)

            selector.change(fn=_toggle, inputs=[selector],
                            outputs=[root, warning])
        except Exception as exc:
            print(f"[H3PromptBuilder] model visibility not wired: {exc}")

    # =====================================================================
    # Assembly
    # =====================================================================

    @staticmethod
    def _s(value):
        if value is None:
            return ""
        if isinstance(value, (list, tuple)):
            return ", ".join(str(v).strip() for v in value if v and str(v).strip())
        return str(value).strip()

    @classmethod
    def _scene_clause(cls, location, lighting, atmosphere):
        """
        Location, lighting and atmosphere are visual description, so they are
        woven into the opening shot rather than emitted as their own fields.
        """
        # "The setting is X" avoids the preposition trap: "in a rooftop at
        # dusk" is wrong, but "the setting is a rooftop at dusk" works for
        # interiors and exteriors alike.
        if location:
            clause = f"The setting is {location}"
            if lighting:
                clause += f", lit by {lighting}"
            if atmosphere:
                clause += f", with {atmosphere}"
            return clause + "."

        if lighting:
            clause = f"The scene is lit by {lighting}"
            if atmosphere:
                clause += f", with {atmosphere}"
            return clause + "."

        if atmosphere:
            return f"The scene is filled with {atmosphere}."

        return ""

    @classmethod
    def _opening_clause(cls, style, location, lighting, atmosphere,
                        camera_type, duration=None, grading=""):
        """
        One sentence establishing the whole clip, written before [Shot 1]:
        style, setting, light, air and camera body.
        """
        head = f"A {style[0].lower() + style[1:]} scene" if style else "A scene"

        parts = [head]
        if location:
            parts.append(f"set in {location}")
        if lighting:
            parts.append(f"lit by {lighting}")
        if atmosphere:
            parts.append(f"with {atmosphere}")
        if grading:
            parts.append(f"graded with {grading}"
                         if not grading.startswith(("a ", "an ", "the "))
                         else f"with {grading}")
        if camera_type:
            parts.append(f"shot on {camera_type}")
        if duration:
            try:
                secs = float(duration)
                secs = int(secs) if float(secs).is_integer() else secs
                article = "an" if str(secs)[0] in "8" else "a"
                parts.append(f"across {article} {secs}-second duration")
            except (TypeError, ValueError):
                pass

        if len(parts) == 1:
            return "" if not style else head + "."

        # Style entries carry internal commas ("Live-action, cinematic"), so
        # they need a comma before the next clause. A bare "A scene" doesn't.
        if style:
            return ", ".join(parts) + "."
        return parts[0] + " " + ", ".join(parts[1:]) + "."

    @staticmethod
    def _gear_clause(camera_type, lens_type):
        """
        Camera body and lens are production constants, so they are stated once
        in the opening shot rather than repeated per shot. Vocabulary entries
        already carry their own article.
        """
        if camera_type and lens_type:
            return f"Shot on {camera_type} using {lens_type}."
        if camera_type:
            return f"Shot on {camera_type}."
        if lens_type:
            return f"Shot using {lens_type}."
        return ""

    @classmethod
    def _merge_audio(cls, presets, freetext, lead="{}."):
        """Presets first as one sentence, then whatever was typed."""
        presets = cls._s(presets)
        freetext = cls._s(freetext)
        out = []
        if presets:
            sentence = lead.format(presets)
            out.append(sentence[0].upper() + sentence[1:])
        if freetext:
            freetext = freetext[0].upper() + freetext[1:]
            out.append(freetext if freetext.endswith(".") else freetext + ".")
        return " ".join(out)

    @staticmethod
    def _step_count(current, delta, maximum, minimum=0):
        """
        Shared handler for every add/remove button pair. Returns the new
        count followed by one visibility update per slot.

        Slots are pre-created and hidden rather than built on demand:
        creating components after render would need gr.render, which is a
        poor fit for a plugin injected into someone else's layout.
        """
        try:
            n = int(current)
        except (TypeError, ValueError):
            n = minimum
        n = max(minimum, min(maximum, n + delta))
        return [n] + [gr.update(visible=(i < n)) for i in range(maximum)]

    @classmethod
    def _ref_tags(cls, text):
        """
        Normalise a source-asset field into angle-bracket tags.

        H3 only treats <Picture 1> as a reference; plain "Picture 1" is read
        as ordinary words. Entries already wrapped are left alone, so typing
        either form works.
        """
        raw = cls._s(text)
        if not raw:
            return ""
        parts = [p.strip() for p in raw.split(",") if p.strip()]
        tagged = []
        for p in parts:
            if p.startswith("<") and p.endswith(">"):
                tagged.append(p)
            else:
                tagged.append(f"<{p}>")
        if len(tagged) == 1:
            return tagged[0]
        return ", ".join(tagged[:-1]) + f" and {tagged[-1]}"

    @staticmethod
    def _no_blank_lines(text):
        """
        Collapse any empty lines out of the finished prompt. WanGP treats a
        blank line as a prompt separator, so one typed into a description
        field would silently split the job into two generations.
        """
        return "\n".join(l for l in (text or "").split("\n") if l.strip())

    @staticmethod
    def _timecode(seconds):
        try:
            total = float(seconds)
        except (TypeError, ValueError):
            return None
        minutes = int(total // 60)
        rest = total - minutes * 60
        return f"{minutes:02d}:{rest:06.3f}"

    @classmethod
    def _voice_phrase(cls, e):
        """
        Voice description for an entry that speaks. Written into its subject
        definition rather than inline in the shot text, so it appears once.
        """
        if not cls._s(e.get("speaker")):
            return ""
        person = " ".join(b for b in [cls._s(e.get("age")),
                                      cls._s(e.get("gender"))] if b)
        bits = [cls._s(e.get(k)) for k in ("pitch", "timbre", "rate")]
        bits = [b for b in bits if b]
        parts = []
        if person:
            parts.append(f"a {person} voice" if not bits else f"a {person}")
        if bits:
            parts.append("with a " + ", ".join(bits) + " voice")
        accent = cls._s(e.get("accent"))
        if accent:
            parts.append(f"with {accent} accent" if not bits
                         else f"and {accent} accent")
        presence = cls._s(e.get("onscreen"))
        if presence == "off-screen":
            parts.append("heard off-screen")
        return " ".join(parts)

    @classmethod
    def _camera_clause(cls, motion, ampl, speed, rig=None):
        motion = cls._s(motion)
        rig = cls._s(rig)
        if not motion and not rig:
            return ""

        if motion:
            verb = MOTION_VERBS.get(motion, motion.lower())
            clause = f"The camera {verb}"
            extras = [e for e in [cls._s(ampl), cls._s(speed)] if e]
            if extras:
                clause += " " + " ".join(extras)
            if rig:
                clause += f", {rig}"
        else:
            # A rig with no stated movement still says something useful.
            clause = f"The camera is {rig}"
        return clause + "."

    @classmethod
    def _beat_text(cls, beat, label_for, lang_for=None, lang_default="English"):
        """
        A beat refers to subjects by label. Descriptions live in
        subject_definitions, so the shot text stays short and the same
        subject reads identically in every shot.
        """
        action = cls._s(beat["action"])
        speech = cls._s(beat["speech"])
        who = beat["speaker"]
        picked = who if isinstance(who, (list, tuple)) else ([who] if who else [])

        # The beat's language overrides; otherwise the first speaker's own
        # language is used, so a bilingual cast needs no per-line fiddling.
        lang = cls._s(beat["lang"])
        if not lang:
            lang_for = lang_for or {}
            for p in picked:
                lang = lang_for.get(cls._s(p), "")
                if lang:
                    break
        lang = lang or lang_default
        labels = [label_for.get(cls._s(p), "") for p in picked]
        labels = [l for l in labels if l]

        # An empty Number can come back as None, "" or 0. A beat at 0.000 is
        # the start of the clip, which the shot already implies, so all three
        # count as unset and no timestamp is written.
        raw_at = beat.get("at")
        try:
            unset = raw_at in (None, "") or float(raw_at) <= 0
        except (TypeError, ValueError):
            unset = True
        at = None if unset else cls._timecode(raw_at)
        stamp = f"At {at}, " if at else ""

        if not labels:
            if not action:
                return ""
            text = action[0].upper() + action[1:] if not stamp else action[0].lower() + action[1:]
            return stamp + text + ("" if text.endswith(".") else ".")

        # Several subjects vocalising together share one compound ID.
        ids = re.findall(r"\((S\d+)\)", " ".join(labels))
        if len(labels) > 1:
            bare = [re.sub(r"\s*\(S\d+\)", "", l) for l in labels]
            joined = ", ".join(bare[:-1]) + f" and {bare[-1]}"
            subject_text = joined + (f" ({','.join(ids)})" if ids else "")
        else:
            subject_text = labels[0]

        # A beat starts a sentence, so the subject is capitalised unless a
        # timestamp precedes it. Bracketed IDs like "(S1)" are left alone.
        if stamp:
            lead = stamp + (subject_text[0].lower() + subject_text[1:]
                            if subject_text[:1].isalpha() else subject_text)
        else:
            lead = (subject_text[0].upper() + subject_text[1:]
                    if subject_text[:1].isalpha() else subject_text)
        if action:
            lead += " " + action[0].lower() + action[1:]

        if speech:
            sentence = f"{lead}: <d>[{lang}] {speech}</d>"
            if beat.get("carries"):
                sentence += "<scenetrans>"
            return sentence

        return lead + ("" if lead.endswith(".") else ".")

    @classmethod
    def _shot_text(cls, idx, shot, label_for, lang_for=None,
                   label_after=None):
        anchor = cls._s(shot["anchor"])
        framing = cls._s(shot["framing"])
        head = f"[Shot {idx + 1}]"

        lens = cls._s(shot["lens"])

        if idx == 0:
            # Framing plus lens read naturally together: "a wide shot on a
            # 24mm lens frames ...".
            lead = framing
            if lead and lens:
                lead += f" on {lens}"
            if lead and anchor:
                lead += " frames"
            if anchor:
                lead = f"{lead} {anchor}" if lead else anchor
            if lead:
                lead = lead[0].upper() + lead[1:]
            body = [lead.rstrip(".") + "." if lead else ""]
        else:
            tc = cls._timecode(shot["cut"])
            verb = cls._s(shot["cutverb"]) or "the camera cuts to"
            target = framing or "the next shot"
            if lens:
                target += f" on {lens}"
            lead = f"At {tc}, {verb} {target}" if tc else f"{verb.capitalize()} {target}"
            if anchor:
                lead += f" of {anchor}"
            body = [lead.rstrip(".") + "."]

        cam = cls._camera_clause(shot["motion"], shot["ampl"], shot["speed"],
                                 shot.get("rig"))
        if cam:
            body.append(cam)

        for beat in shot["beats"]:
            text = cls._beat_text(beat, label_for, lang_for or {})
            if text:
                body.append(text)
                # Identity is written once; later mentions use the short form.
                if label_after:
                    who = beat["speaker"]
                    picked = who if isinstance(who, (list, tuple)) else [who]
                    for pk in picked:
                        pk = cls._s(pk)
                        if pk in label_after:
                            label_for[pk] = label_after[pk]

        return head + " " + " ".join(b for b in body if b)

    # -- top level ---------------------------------------------------------

    @classmethod
    def _unpack(cls, values):
        """
        Walk the flat input list into a dict. Shared by _build and
        _draft_summary so the two can never drift out of step.
        """
        vals = list(values)
        i = 0

        def take(n=1):
            nonlocal i
            out = vals[i:i + n]
            i += n
            return out[0] if n == 1 else out

        ref_mode = bool(take())
        duration = take()
        style = cls._s(take())
        grading = cls._s(take())
        location = cls._s(take())
        lighting = cls._s(take())
        atmosphere = cls._s(take())
        camera_type = cls._s(take())
        video_role = cls._s(take())
        video_desc = cls._s(take())
        video_retention = cls._s(take())
        video_audio = cls._s(take())
        video_audio_desc = cls._s(take())

        entry_count = int(cls._s(take()) or 0)
        entries = []
        for _ in range(MAX_ENTRIES):
            entries.append({
                "kind": take(), "desc": take(), "speaker": take(),
                "onscreen": take(), "age": take(), "gender": take(),
                "pitch": take(), "timbre": take(), "rate": take(),
                "accent": take(), "lang": take(), "source": take(), "retention": take(), "note": take(),
                "shots": take(), "voice_from": take(), "motion_from": take(),
            })

        task_types = take()
        summary_text = cls._s(take())
        shot_count = int(cls._s(take()) or 1)

        shots = []
        for _ in range(MAX_SHOTS):
            shot = {
                "cut": take(), "cutverb": take(), "framing": take(),
                "lens": take(), "motion": take(), "ampl": take(),
                "speed": take(), "rig": take(), "anchor": take(),
                "beat_count": take(),
                "beats": [],
            }
            for _ in range(MAX_BEATS):
                shot["beats"].append({
                    "type": take(), "speaker": take(), "lang": take(),
                    "action": take(), "speech": take(), "at": take(),
                    "carries": take(),
                })
            shot["beats"] = shot["beats"][:int(cls._s(shot["beat_count"]) or 0)]
            shots.append(shot)

        ambience_from = cls._s(take())
        ambience_retention = cls._s(take())
        soundscape_presets = cls._s(take())
        soundscape = cls._s(take())
        music_from = cls._s(take())
        music_role = cls._s(take())
        music_retention = cls._s(take())
        music_presets = cls._s(take())
        music = cls._s(take())

        return dict(
            ref_mode=ref_mode, duration=duration, style=style,
            grading=grading, location=location,
            lighting=lighting, atmosphere=atmosphere, camera_type=camera_type,
            video_role=video_role, video_desc=video_desc,
            video_retention=video_retention, video_audio=video_audio,
            video_audio_desc=video_audio_desc,
            entry_count=entry_count, entries=entries,
            task_types=task_types, summary_text=summary_text,
            shot_count=shot_count, shots=shots,
            ambience_from=ambience_from, ambience_retention=ambience_retention,
            soundscape_presets=soundscape_presets, soundscape=soundscape,
            music_from=music_from, music_role=music_role,
            music_retention=music_retention,
            music_presets=music_presets, music=music,
        )

    @classmethod
    def _draft_summary(cls, *values):
        """
        Compose a first-pass summary from what has been entered, for the user
        to edit. Deliberately not run at build time: the summary should read
        as a short human overview, and a generated one usually needs a pass.
        """
        d = cls._unpack(values)

        prefix = cls._s(d["task_types"])
        prefix = " + ".join(p.strip() for p in prefix.split(",") if p.strip())

        shots = d["shots"][:max(1, d["shot_count"])]
        entries = d["entries"][:d["entry_count"]]

        named = []
        counters = {k: 0 for k in ASSET_KINDS}
        for e in entries:
            if not cls._s(e["desc"]) and not cls._s(e["source"]):
                continue
            kind = cls._s(e["kind"]) or "Subject"
            counters[kind] += 1
            desc = cls._s(e["desc"])
            named.append(f"<{kind} {counters[kind]}>"
                         + (f", {desc}," if desc else ""))

        # Shaped to read like MiniMax's own examples: one lead sentence
        # naming what the video is, then the assets it draws on.
        style = cls._s(d["style"])
        duration = d["duration"]
        opening = cls._s(shots[0]["anchor"]) if shots else ""

        # "shows" reads better than "is" here, but a duration and style
        # don't fit inside it, so they get their own short sentence after.
        lead_bits = []
        if opening:
            lead_bits.append("The target video shows "
                             + opening[0].lower() + opening[1:])
        if named:
            bare = [n.split(",")[0] for n in named]
            joined = (", ".join(bare[:-1]) + f" and {bare[-1]}"
                      if len(bare) > 1 else bare[0])
            if lead_bits:
                lead_bits.append(f", featuring {joined}")
            else:
                lead_bits.append(f"The target video features {joined}")

        sentences = []
        if lead_bits:
            lead = "".join(lead_bits).replace(" ,", ",")
            sentences.append(lead.rstrip(".") + ".")

        descriptor = []
        try:
            secs = float(duration)
            secs = int(secs) if float(secs).is_integer() else secs
            descriptor.append(f"{secs} seconds")
        except (TypeError, ValueError):
            pass
        if style:
            descriptor.append(f"in a {style[0].lower() + style[1:]} style")
        if descriptor and sentences:
            sentences.append("It runs " + " ".join(descriptor) + ".")

        voices = set()
        for shot in shots:
            for beat in shot["beats"]:
                if cls._s(beat["speech"]):
                    who = beat["speaker"]
                    picked = who if isinstance(who, (list, tuple)) else [who]
                    voices.update(cls._s(p) for p in picked if cls._s(p))
        if voices:
            label = "one speaker" if len(voices) == 1 else f"{len(voices)} speakers"
            sentences.append(f"There is dialogue from {label}.")

        if len(shots) > 1:
            sentences.append(f"It runs to {len(shots)} shots.")

        role = cls._s(d["video_role"])
        if role not in ("", "none"):
            vdesc = cls._s(d["video_desc"])
            phrasing = {
                "continue from it": "The source video is continued",
                "edit it": "The source video is edited",
                "reference its camera and cutting only":
                    "The source video guides camera movement and cutting",
            }.get(role, f"The source video is used to {role}")
            sentences.append(phrasing
                             + (f", providing {vdesc}" if vdesc else "") + ".")

        if not opening and not named and role in ("", "none"):
            sentences = []

        if not sentences:
            # A source video is only a draftable input in reference mode, so
            # only offer it as a suggestion there.
            missing = ("a shot anchor, a subject, or a source video"
                       if d["ref_mode"] else "a shot anchor or a subject")
            return "", f"Nothing to summarise yet - add {missing} first."

        draft = " ".join(sentences)
        # The task-type prefix only applies in reference mode, so only
        # mention it there.
        note = ("Draft written - edit it so it reads as your own overview."
                if prefix or not d["ref_mode"] else
                "Draft written, but no **task type** is ticked, so the summary "
                "will have no prefix.")
        return draft, note

    @classmethod
    def _build(cls, *values):
        d = cls._unpack(values)
        ref_mode = d["ref_mode"]; duration = d["duration"]
        style = d["style"]; grading = d["grading"]
        location = d["location"]; lighting = d["lighting"]
        atmosphere = d["atmosphere"]; camera_type = d["camera_type"]
        video_role = d["video_role"]; video_desc = d["video_desc"]
        video_retention = d["video_retention"]; video_audio = d["video_audio"]
        video_audio_desc = d["video_audio_desc"]
        entry_count = d["entry_count"]; entries = d["entries"]
        task_types = d["task_types"]
        summary_text = d["summary_text"]
        shot_count = d["shot_count"]; shots = d["shots"]
        ambience_from = d["ambience_from"]
        ambience_retention = d["ambience_retention"]
        soundscape_presets = d["soundscape_presets"]; soundscape = d["soundscape"]
        music_from = d["music_from"]; music_role = d["music_role"]
        music_retention = d["music_retention"]
        music_presets = d["music_presets"]; music = d["music"]

        shots = shots[:max(1, shot_count)]

        # One output shape for every case. Characters are subjects whether or
        # not they come from a reference asset, so subject_definitions is
        # always written and unused sections say N/A.
        #
        # Note this is the reference schema. The base guide specifies three
        # fields for text and keyframe prompts; if plain text-to-video results
        # get worse, this is the first thing to test.
        active = entries[:entry_count]
        counters = {k: 0 for k in ASSET_KINDS}
        defs, retention = [], []
        label_for = {}          # first mention
        label_after = {}        # every mention after that
        lang_for = {}           # "Subject 3" -> "Japanese"
        skipped = 0
        uses_reference_assets = False

        # An asset can serve several roles. The guide asks for one natural
        # sentence per label rather than a subsection each, so roles are
        # gathered here and written once.
        asset_roles = {}     # "Audio 1" -> [role phrase, ...]
        asset_kind = {}      # "Audio 1" -> "audio" | "video"
        asset_marker = {}    # "Audio 1" -> retention marker

        for idx, e in enumerate(active):
            desc = cls._s(e["desc"])
            # Reference fields only count when the entry is marked as a
            # referenced subject, so a stray value in a collapsed block
            # cannot leak into the prompt.
            is_ref = ref_mode
            source = cls._ref_tags(e["source"]) if is_ref else ""
            if not desc and not source:
                skipped += 1
                continue

            kind = cls._s(e["kind"]) or "Subject"
            counters[kind] = counters.get(kind, 0) + 1
            label = f"<{kind} {counters[kind]}>"

            speaker = cls._s(e["speaker"])
            def_label = f"{label} ({speaker})" if speaker else label
            key = f"Subject {idx + 1}"

            if ref_mode:
                # Labels are stable: the description lives in
                # subject_definitions, so every mention is just the label.
                label_for[key] = def_label
                label_after[key] = def_label
            else:
                # Base modes have no subject_definitions, so identity is
                # written inline at first appearance and referenced by
                # speaker ID after that. An entry with no speaker has no ID
                # to fall back on, so its description repeats.
                voice = cls._voice_phrase(e)
                inline = ", ".join(p for p in [desc, voice] if p)
                if speaker:
                    label_for[key] = (f"{inline} ({speaker})" if inline
                                      else f"({speaker})")
                    label_after[key] = f"({speaker})"
                else:
                    label_for[key] = inline
                    label_after[key] = inline
            lang_for[f"Subject {idx + 1}"] = cls._s(e["lang"]) or "English"

            # Reference assets attached to this entry. The label names the
            # actual slot, so it is never renumbered.
            vfrom = cls._s(e["voice_from"]) if is_ref else ""
            if vfrom:
                asset_roles.setdefault(vfrom, []).append(
                    f"the voice-timbre reference for {def_label}")
                asset_kind[vfrom] = "audio"
            mfrom = cls._s(e["motion_from"]) if is_ref else ""
            if mfrom:
                asset_roles.setdefault(mfrom, []).append(
                    f"the motion and performance reference for {def_label}")
                asset_kind[mfrom] = "video"

            # The description lives here, so the shot text can just use the
            # label - which is what keeps a subject consistent across shots.
            line = f"{def_label} is {desc}" if desc else def_label
            voice = cls._voice_phrase(e)
            if voice:
                line += f", {voice}"
            if source:
                line += f", from {source}"
            defs.append(line.rstrip(".") + ".")

            # Only reference-backed entries have anything to retain.
            if source:
                uses_reference_assets = True
            marker = cls._s(e["retention"])
            if source and marker:
                note = cls._s(e["note"])
                where = cls._s(e["shots"])
                scope = ""
                if where:
                    shot_list = ", ".join(f"[Shot {p.strip()}]"
                                          for p in where.split(",") if p.strip())
                    scope = f" (appears in {shot_list})"
                entry = f"{label}{scope}: {marker}"
                if note:
                    entry += f" - {note}"
                retention.append(entry.rstrip(".") + ".")

        # Global scene block, written once before [Shot 1]. Style, location,
        # lighting, atmosphere and camera body hold for the whole clip, so
        # they belong here rather than inside the opening shot.
        opening = cls._opening_clause(style, location, lighting,
                                      atmosphere, camera_type, duration,
                                      grading)

        shot_lines = [cls._shot_text(n, s, label_for, lang_for, label_after)
                      for n, s in enumerate(shots)]
        body = "\n".join(l for l in ([opening] + shot_lines) if l.strip())

        sound_field = cls._merge_audio(
            soundscape_presets, soundscape,
            lead="The scene carries {}.",
        ) or "N/A"

        # Empty means no score, which is what N/A says. No separate switch.
        music_field = cls._merge_audio(music_presets, music, lead="{}.") or "N/A"

        video_label = ""
        if video_role != "none" and video_desc:
            counters["Video"] += 1
            video_label = f"<Video {counters['Video']}>"
            defs.append(f"{video_label} provides {video_desc}.")
            retention.append(
                f"{video_label}: {video_retention or 'weak_reference'} - {video_desc}."
            )

        # Audio from the source video gets its own <Audio N> label and uses
        # the audio marker set, which differs from the picture one.
        if video_audio or video_audio_desc:
            counters["Audio"] += 1
            alabel = f"<Audio {counters['Audio']}>"
            what = video_audio_desc or "the original audio"
            source = f" from {video_label}" if video_label else ""
            defs.append(f"{alabel} is {what}{source}.")
            retention.append(
                f"{alabel}: {video_audio or 'reference'} - {what}."
            )

        # Audio references for the ambient bed and the score. Ignored unless
        # reference mode is on, so a value left in a hidden block cannot leak
        # into the prompt.
        if not ref_mode:
            ambience_from = music_from = ""
        if ambience_from:
            asset_roles.setdefault(ambience_from, []).append(
                "the ambient-sound reference for the scene")
            asset_kind[ambience_from] = "audio"
            if ambience_retention:
                asset_marker[ambience_from] = ambience_retention
        if music_from:
            role = music_role or "style"
            asset_roles.setdefault(music_from, []).append(
                f"the {role} reference for the non-diegetic score")
            asset_kind[music_from] = "audio"
            if music_retention:
                asset_marker[music_from] = music_retention

        # One line per label, however many roles it serves.
        for slot in sorted(asset_roles):
            roles = asset_roles[slot]
            joined = (", ".join(roles[:-1]) + f" and {roles[-1]}"
                      if len(roles) > 1 else roles[0])
            defs.append(f"<{slot}> is {joined}.")
            uses_reference_assets = True

            default = "reference" if asset_kind.get(slot) == "audio" else "weak_reference"
            marker = asset_marker.get(slot, default)
            retention.append(f"<{slot}>: {marker} - {joined}.")

        prefix = cls._s(task_types)
        prefix = " + ".join(p.strip() for p in prefix.split(",") if p.strip())
        summary = f"[{prefix}] {summary_text}".strip() if prefix else summary_text

        # The opening clause already carries the style, so no separate
        # style line here.
        detailed = body

        if not ref_mode:
            # Base modes use the three core fields. No subject_definitions,
            # no summary, no retention_analysis - speakers carry their own
            # identity inline and (S1) is the only label.
            lines = [f"integrated_multimodal_description: {detailed}",
                     f"overall_soundscape: {sound_field}",
                     f"non_diegetic_music: {music_field}"]
            warnings = []
            if not any(cls._s(s["anchor"]) for s in shots):
                warnings.append("no **anchor** on any shot - nothing "
                                "describes what is in frame")
            check = "Read it through for grammar before generating."
            status = (f"Prompt written. {check}" if not warnings
                      else "Written, but: " + "; ".join(warnings) + f". {check}")
            return cls._no_blank_lines("\n".join(lines)), status

        sections = [
            "subject_definitions:\n" + "\n".join(defs) if defs
            else "subject_definitions:\nN/A",
            "summary:\n" + summary if summary else "summary:\nN/A",
            # Written only when something is actually being preserved from a
            # reference asset; otherwise N/A falls out on its own.
            "retention_analysis:\n" + "\n".join(retention) if retention
            else "retention_analysis:\nN/A",
            "detailed_description:\n" + detailed,
            "overall_soundscape:\n" + sound_field,
            "non_diegetic_music:\n" + music_field,
        ]
        # Single newlines only - a blank line would make WanGP treat what
        # follows as a separate generation.
        # Only nag about a task type when something is actually referenced.
        # Subjects invented from description are not reference assets, and
        # most prompts have no task type at all.
        warnings = []
        # Only a reminder once reference mode is actually on.
        if ref_mode and not prefix:
            warnings.append("reference mode is on but no **task type** is "
                            "ticked")
        if not summary_text:
            warnings.append("**summary** is empty")
        if skipped:
            warnings.append(f"{skipped} reference entr"
                            f"{'y was' if skipped == 1 else 'ies were'} "
                            "skipped for having no description or source")
        if not any(cls._s(s["anchor"]) for s in shots):
            warnings.append("no **anchor** on any shot - nothing describes "
                            "what is in frame")

        # Audio references imply an audio task type.
        audio_slots = {s for s in asset_roles if asset_kind.get(s) == "audio"}
        if (audio_slots or video_audio or video_audio_desc) and \
                "audio" not in prefix.lower():
            warnings.append("audio is referenced but no **audio task type** "
                            "(audio reference / audio reuse) is ticked")

        # WanGP's Audio References selector is one dropdown, so the video's
        # own soundtrack and standalone audio clips cannot both be used.
        if audio_slots and (video_audio or video_audio_desc):
            warnings.append("both the **source video's audio** and standalone "
                            "**audio slots** are set - WanGP lets you pick one "
                            "or the other")

        check = "Read it through for grammar before generating."
        status = (f"Prompt written. {check}"
                  if not warnings
                  else "Written, but: " + "; ".join(warnings) + f". {check}")
        return cls._no_blank_lines("\n".join(sections)), status

    @staticmethod
    def _clear():
        # mode, duration, style, location, lighting, atmosphere,
        # camera_type, video_role, video_desc, video_retention,
        # video_audio, video_audio_desc
        out = [False, 8.0, "", "", "", "", "", "", "none", "", "", "", ""]
        out.append(0)                                 # entry_count
        for _ in range(MAX_ENTRIES):
            # kind, desc, speaker, presence, age, gender, pitch, timbre,
            # rate, accent, lang, source, retention, note, shots
            # kind, desc, speaker, presence, age, gender, pitch, timbre,
            # rate, accent, lang, is_ref, source, retention, note, shots,
            # voice_from, motion_from
            out += ["Subject", "", "", "", "", "", "", "", "", "",
                    "English", [], "", "", "", "", ""]
        # task_types, summary, shot_count
        out += [[], "", 1]
        for si in range(MAX_SHOTS):
            # cut, cutverb, framing, lens, motion, ampl, speed, rig,
            # anchor, beats
            out += ["opening" if si == 0 else None, "-" if si == 0 else "",
                    "", "", "", "", "", "", "", 0]
            out += ["action", [], "", "", "", None, False] * MAX_BEATS
        # ambience_from, ambience_retention, soundscape_presets, soundscape,
        # music_from, music_role, music_retention, music_presets, music
        out += ["", "", [], "", "", "style", "", [], ""]

        # Re-hide every slot: entries, shots, then beats.
        # Shot 1 stays visible because a prompt always has at least one.
        out += [gr.update(visible=False)] * MAX_ENTRIES   # entry groups
        out += [gr.update(visible=False)] * MAX_ENTRIES   # reference blocks
        out += [gr.update(visible=False)] * 3   # audio refs + summary block
        out += [gr.update(visible=(i == 0)) for i in range(MAX_SHOTS)]
        out += [gr.update(visible=False)] * (MAX_SHOTS * MAX_BEATS)
        return out


Plugin = H3PromptBuilderPlugin
