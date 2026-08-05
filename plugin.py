"""
MiniMax H3 Prompt Builder - a WanGP plugin
------------------------------------------
Builds MiniMax H3 prompts in the format described by MiniMax's own guides:

  base modes (T2VA / I2VA / FL2VA / L2VA)
    https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md
  full-reference mode (Ref2VA)
    https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md

Base modes emit three named fields; reference mode emits six. The builder owns
the mechanical parts - mode instruction lines, shot timestamps, speaker IDs,
<d> wrapping, retention markers, task-type prefixes and N/A placeholders.

DRAFT STATUS (v0.1)
  - Shots and beats use fixed slots revealed by count dropdowns, not the
    drag-reorderable list from the design mockups. Gradio makes true dynamic
    lists awkward; the show/hide pattern is the one already proven to work in
    this plugin API.
  - Mode is chosen manually. Auto-detection is attempted but cannot be relied
    on - see _wire_model_visibility() for why.
  - Prose templates are a first pass. Expect to tune the sentence shapes after
    reading real output.
"""

import re

import gradio as gr

from shared.utils.plugins import WAN2GPPlugin


# =============================================================================
# Limits
# =============================================================================

MAX_SPEAKERS = 6
MAX_SUBJECTS = 8
MAX_SHOTS = 6
MAX_BEATS = 5

# Substrings that mark a model as MiniMax H3. Checked case-insensitively
# against the model type reported by on_model_change.
H3_MODEL_HINTS = ("minimax", "h3")


# =============================================================================
# Vocabulary
# =============================================================================

MODES = [
    "T2VA - text only",
    "I2VA - start image",
    "FL2VA - start and end image",
    "L2VA - end image only",
    "Ref2VA - full reference",
]

STYLES = [
    "", "Live-action, cinematic", "Live-action, documentary",
    "Observational documentary", "2D-animated", "3D CG", "Claymation",
    "Watercolour", "Vintage film", "Music-video", "Multi-camera sitcom",
    "Handheld, naturalistic", "Stop-motion",
]

FRAMINGS = [
    "", "an extreme close-up", "a close-up", "a medium close-up",
    "a medium shot", "a medium-wide shot", "a wide shot",
    "an establishing shot", "a two-shot", "an over-the-shoulder shot",
    "a low-angle shot", "a high-angle shot", "an overhead shot",
    "a point-of-view shot", "a profile shot",
]

# Fixed vocabulary from the guide's camera table. Not user-extendable:
# the model was trained on these exact expressions.
MOTION_TYPES = [
    "", "Zoom In", "Zoom Out", "Push In", "Pull Out", "Pan Left", "Pan Right",
    "Truck Left", "Truck Right", "Tilt Up", "Tilt Down", "Pedestal Up",
    "Pedestal Down", "Arc Shot", "Tracking Shot", "Static Shot",
    "Shake Slightly", "Shake Strongly", "POV", "Roll Clockwise",
    "Roll Counterclockwise",
]

# How each motion type reads as a verb phrase after "The camera ...".
MOTION_VERBS = {
    "Zoom In": "zooms in", "Zoom Out": "zooms out",
    "Push In": "pushes in", "Pull Out": "pulls out",
    "Pan Left": "pans left", "Pan Right": "pans right",
    "Truck Left": "trucks left", "Truck Right": "trucks right",
    "Tilt Up": "tilts up", "Tilt Down": "tilts down",
    "Pedestal Up": "rises on the pedestal", "Pedestal Down": "lowers on the pedestal",
    "Arc Shot": "moves in an arc around the subject",
    "Tracking Shot": "tracks the subject",
    "Static Shot": "holds a static shot",
    "Shake Slightly": "shakes slightly", "Shake Strongly": "shakes strongly",
    "POV": "takes the subject's point of view",
    "Roll Clockwise": "rolls clockwise",
    "Roll Counterclockwise": "rolls counterclockwise",
}

AMPLITUDES = ["", "with small amplitude", "with large amplitude"]
SPEEDS = ["", "at slow speed", "at fast speed"]

CUT_VERBS = [
    "the camera cuts to", "the shot cuts to", "the shot transitions to",
    "the shot changes to", "the shot switches to",
    "the shot cross-dissolves to", "the shot fades to", "the shot wipes to",
]

LANGUAGES = ["English", "Chinese", "Japanese", "Korean", "Spanish", "French",
             "German", "Italian", "Portuguese", "Russian", "Arabic", "Hindi"]

VOICE_AGES = ["", "child", "teenage", "young", "middle-aged", "older", "elderly"]
VOICE_GENDERS = ["", "female", "male", "androgynous"]
VOICE_PITCH = ["", "low", "medium", "high"]
VOICE_TIMBRE = ["", "clear", "raspy", "breathy", "warm", "nasal", "gravelly",
                "bright", "weathered", "smooth"]
VOICE_RATE = ["", "slow", "measured", "unhurried", "quick", "clipped", "halting"]

# Reference mode -----------------------------------------------------------

ASSET_KINDS = ["Subject", "Picture", "Video", "Audio"]

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
    "golden hour light", "blue hour light", "harsh midday sun",
    "soft overcast light", "moody low-key lighting", "high-key lighting",
    "flickering candlelight", "neon glow", "a backlit silhouette",
    "practical lamps only", "dramatic hard shadows", "rim lighting",
    "dappled sunlight through leaves", "harsh fluorescent light",
    "moonlight", "firelight", "diffused window light",
    "strobing club lighting", "underwater caustics", "streetlight sodium glow",
]

SCENE_ATMOSPHERE = [
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
        speakers, subjects, shots = [], [], []

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

            mode = gr.Radio(MODES, label="Mode", value=MODES[0])
            gr.Markdown(
                "Pick the mode that matches what you have attached in the "
                "generator above. Reference mode emits six sections instead "
                "of three."
            )

            with gr.Row():
                duration = gr.Number(label="Duration (seconds)", value=8.0,
                                     minimum=0.5, step=0.5)
                style = dd(STYLES, "Style")

            with gr.Row():
                location = dd(LOCATIONS, "Location")
                lighting = dd(SCENE_LIGHTING, "Lighting")
                atmosphere = dd(SCENE_ATMOSPHERE, "Atmosphere")

            with gr.Row():
                camera_type = dd(CAMERA_TYPES, "Camera / stock")

            # ---- source video (FL2VA continue / Ref2VA video reference) ----
            with gr.Accordion("Source video", open=False, visible=False) as video_section:
                video_role = gr.Radio(
                    ["none", "continue from it", "edit it",
                     "reference its camera and cutting only"],
                    label="Role of an attached video", value="none",
                )
                video_desc = gr.Textbox(
                    label="What the video contributes",
                    placeholder="handheld camera movement and cutting rhythm",
                )
                video_retention = locked_dd(VISUAL_RETENTION, "Retention")

            # ---- cast ------------------------------------------------------
            with gr.Accordion("Cast", open=False):
                gr.Markdown(
                    "Add anyone you want to refer to by a stable ID. Their "
                    "description is written once, at their first appearance, "
                    "and they are referenced as (S1), (S2) after that. Select "
                    "them on a beat to attribute an action or a line to them - "
                    "dialogue is optional."
                )
                speaker_count = gr.State(0)
                for i in range(MAX_SPEAKERS):
                    with gr.Group(visible=False) as grp:
                        gr.Markdown(f"**(S{i + 1})**")
                        with gr.Row():
                            s_name = gr.Textbox(label="Who they are",
                                                placeholder="the fishmonger in the heavy apron")
                            s_onscreen = gr.Dropdown(
                                ["on-screen", "off-screen"], label="Presence",
                                value="on-screen",
                            )
                        with gr.Row():
                            s_age = dd(VOICE_AGES, "Age")
                            s_gender = dd(VOICE_GENDERS, "Gender")
                            s_pitch = dd(VOICE_PITCH, "Pitch")
                        with gr.Row():
                            s_timbre = dd(VOICE_TIMBRE, "Timbre")
                            s_rate = dd(VOICE_RATE, "Rate")
                            s_accent = gr.Textbox(label="Accent (optional)")
                    speakers.append({
                        "group": grp, "name": s_name, "onscreen": s_onscreen,
                        "age": s_age, "gender": s_gender, "pitch": s_pitch,
                        "timbre": s_timbre, "rate": s_rate, "accent": s_accent,
                    })

                # Created after the slots so the buttons always render
                # directly beneath the last visible speaker - hidden groups
                # take no vertical space.
                with gr.Row():
                    add_speaker = gr.Button("Add speaker", size="sm")
                    rm_speaker = gr.Button("Remove last speaker", size="sm")

                _spk_out = [speaker_count] + [s["group"] for s in speakers]
                add_speaker.click(
                    fn=lambda n: self._step_count(n, +1, MAX_SPEAKERS),
                    inputs=[speaker_count], outputs=_spk_out,
                )
                rm_speaker.click(
                    fn=lambda n: self._step_count(n, -1, MAX_SPEAKERS),
                    inputs=[speaker_count], outputs=_spk_out,
                )

            # ---- reference subjects ---------------------------------------
            with gr.Accordion("Reference subjects", open=False, visible=False) as ref_section:
                gr.Markdown(
                    "One image can supply several subjects, and one subject "
                    "can draw on several images. An image used only to define "
                    "a subject needs no entry of its own."
                )
                subject_count = gr.State(0)
                for i in range(MAX_SUBJECTS):
                    with gr.Group(visible=False) as grp:
                        with gr.Row():
                            r_kind = gr.Dropdown(ASSET_KINDS, label="Label",
                                                 value="Subject")
                            r_source = gr.Textbox(
                                label="Source asset(s)",
                                placeholder="Picture 1",
                            )
                            r_speaks = gr.Checkbox(label="Speaks", value=False)
                        r_desc = gr.Textbox(
                            label="What it is",
                            placeholder="the fishmonger, heavy apron, forearms wet to the elbow",
                        )
                        with gr.Row():
                            r_retention = gr.Dropdown(
                                ALL_RETENTION,
                                label="Retention", value="fully_preserved",
                            )
                            r_retention_note = gr.Textbox(
                                label="What is retained",
                                placeholder="the heavy apron and wet forearms are retained",
                            )
                            r_shots = gr.Textbox(label="Appears in shots",
                                                 placeholder="1, 2, 3")
                    subjects.append({
                        "group": grp, "kind": r_kind, "source": r_source,
                        "speaks": r_speaks, "desc": r_desc,
                        "retention": r_retention, "note": r_retention_note,
                        "shots": r_shots,
                    })

                with gr.Row():
                    add_subject = gr.Button("Add reference entry", size="sm")
                    rm_subject = gr.Button("Remove last entry", size="sm")

                _sub_out = [subject_count] + [s["group"] for s in subjects]
                add_subject.click(
                    fn=lambda n: self._step_count(n, +1, MAX_SUBJECTS),
                    inputs=[subject_count], outputs=_sub_out,
                )
                rm_subject.click(
                    fn=lambda n: self._step_count(n, -1, MAX_SUBJECTS),
                    inputs=[subject_count], outputs=_sub_out,
                )

                task_types = gr.CheckboxGroup(
                    TASK_TYPES, label="Task type - combined with + in summary",
                )
                summary_text = gr.Textbox(
                    label="Summary (draft it yourself, or edit after inserting)",
                    lines=2,
                    placeholder="A fishmonger works the counter while a schoolboy stops to ask about the fish.",
                )

            # ---- shots -----------------------------------------------------
            with gr.Accordion("Shots", open=True):
                shot_count = gr.State(1)
                shot_hint = gr.Markdown("")

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
                                        [""] + [f"S{n + 1}" for n in range(MAX_SPEAKERS)]
                                        + ["S1,S2", "S1,S2,S3"],
                                        label="Speaker", value="",
                                    )
                                    b_lang = gr.Dropdown(LANGUAGES,
                                                         label="Language",
                                                         value="English")
                                b_action = gr.Textbox(
                                    label="Action / delivery (outside <d>)",
                                    placeholder="turns her head and says",
                                )
                                b_speech = gr.Textbox(
                                    label="Spoken words (inside <d>) - leave blank for non-verbal",
                                )
                                b_carries = gr.Checkbox(
                                    label="Line carries across the next cut",
                                    value=False,
                                )
                            beats.append({
                                "group": bgrp, "type": b_type,
                                "speaker": b_speaker, "lang": b_lang,
                                "action": b_action, "speech": b_speech,
                                "carries": b_carries,
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
                        "motion": sh_motion,
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
                    label="Soundscape - added after the presets",
                    lines=2,
                    placeholder="A wide market-hall echo carries trolley wheels and distant haggling.",
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
                    label="Non-diegetic music - added after the presets",
                    lines=2,
                )
                music_none = gr.Checkbox(label="No non-diegetic music (writes N/A)",
                                         value=True)

            status = gr.Markdown("")

            with gr.Row():
                insert_btn = gr.Button("Insert into prompt", variant="primary")
                clear_btn = gr.Button("Clear all fields")

        # ---- wiring -------------------------------------------------------

        flat = [mode, duration, style, location, lighting, atmosphere,
                camera_type,
                video_role, video_desc, video_retention, speaker_count]
        for s in speakers:
            flat += [s["name"], s["onscreen"], s["age"], s["gender"],
                     s["pitch"], s["timbre"], s["rate"], s["accent"]]
        flat += [subject_count]
        for s in subjects:
            flat += [s["kind"], s["source"], s["speaks"], s["desc"],
                     s["retention"], s["note"], s["shots"]]
        flat += [task_types, summary_text, shot_count]
        for s in shots:
            flat += [s["cut"], s["cutverb"], s["framing"], s["lens"],
                     s["motion"], s["ampl"], s["speed"], s["anchor"],
                     s["beat_count"]]
            for b in s["beats"]:
                flat += [b["type"], b["speaker"], b["lang"], b["action"],
                         b["speech"], b["carries"]]
        flat += [soundscape_presets, soundscape,
                 music_presets, music, music_none]

        insert_btn.click(fn=self._build, inputs=flat,
                         outputs=[self.prompt, status])
        all_groups = (
            [s["group"] for s in speakers]
            + [s["group"] for s in subjects]
            + [s["group"] for s in shots]
        )
        for s in shots:
            all_groups += [b["group"] for b in s["beats"]]

        clear_btn.click(fn=self._clear, inputs=[], outputs=flat + all_groups)

        mode.change(
            fn=self._apply_mode,
            inputs=[mode],
            outputs=[ref_section, video_section, shot_hint],
        )

        self._wire_model_visibility(root, model_warning)

    @staticmethod
    def _apply_mode(mode):
        """
        Show only the sections a mode can actually use.

        Reference subjects are Ref2VA-only. A source video is meaningful for
        FL2VA (continuing an existing clip) and for Ref2VA (where it becomes
        a <Video N> reference), and meaningless for the other three modes.
        """
        mode = mode or ""
        is_ref = mode.startswith("Ref2VA")
        is_fl = mode.startswith("FL2VA")

        if is_ref:
            hint = ("Reference mode writes six sections. Subject labels are "
                    "numbered in the order you list them.")
        elif is_fl:
            hint = ("With a start and an end image, a single shot usually "
                    "works best so the model can interpolate between them. "
                    "The end image is assumed to land at the full duration.")
        elif mode.startswith("L2VA"):
            hint = ("Describe a plausible earlier state that converges on the "
                    "reference image by the end of the clip.")
        elif mode.startswith("I2VA"):
            hint = ("Shot 1 should restate what is already in the start image "
                    "before describing what changes.")
        else:
            hint = ""

        return (
            gr.update(visible=is_ref),
            gr.update(visible=is_ref or is_fl),
            gr.update(value=hint),
        )

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
    def _opening_clause(cls, style, location, lighting, atmosphere, camera_type):
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
        if camera_type:
            parts.append(f"shot on {camera_type}")

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
    def _instruction(cls, mode, duration):
        """The exact boilerplate each keyframe mode requires."""
        secs = f"{float(duration):.2f}" if duration else "0.00"
        if mode.startswith("I2VA"):
            return ("For the target video, at 0.00 seconds into the target "
                    "video, <Picture 1> (from [Shot 1]) is fully referenced.")
        if mode.startswith("FL2VA"):
            return ("How the reference pictures align with the target video - "
                    "Picture 1 (from Shot 1) aligns with the 0.00-second mark "
                    f"of the target video; Picture 2 (from Shot 1) aligns with "
                    f"the {secs}-second mark of the target video.")
        if mode.startswith("L2VA"):
            return ("How the reference pictures align with the target video - "
                    "<Picture 1> (from [Shot 1]) aligns with the "
                    f"{secs}-second mark of the target video.")
        return ""

    @classmethod
    def _speaker_intro(cls, sp):
        """
        Identity phrase emitted at a speaker's first appearance. Age and
        gender describe the person; pitch, timbre and rate describe the
        voice - stacking all five into one adjective list reads badly.
        """
        name = cls._s(sp["name"])
        person = " ".join(b for b in [cls._s(sp["age"]), cls._s(sp["gender"])] if b)
        voice_bits = [cls._s(sp[k]) for k in ("pitch", "timbre", "rate")]
        voice_bits = [b for b in voice_bits if b]
        accent = cls._s(sp["accent"])

        clause = ""
        if person:
            clause = f"a {person}" if not person[0].lower() in "aeiou" else f"an {person}"
        if voice_bits:
            voice = "with a " + ", ".join(voice_bits) + " voice"
            clause = f"{clause} {voice}" if clause else voice
        if accent:
            clause = (f"{clause} and {accent} accent" if clause
                      else f"with {accent} accent")

        if cls._s(sp["onscreen"]) == "off-screen":
            clause = (clause + ", off-screen") if clause else "off-screen"

        parts = [p for p in [name, clause] if p]
        return ", ".join(parts)

    @classmethod
    def _camera_clause(cls, motion, ampl, speed):
        motion = cls._s(motion)
        if not motion:
            return ""
        verb = MOTION_VERBS.get(motion, motion.lower())
        extras = [cls._s(ampl), cls._s(speed)]
        extras = [e for e in extras if e]
        clause = f"The camera {verb}"
        if extras:
            clause += " " + " ".join(extras)
        return clause + "."

    @classmethod
    def _beat_text(cls, beat, intro_used, lang_default="English"):
        btype = cls._s(beat["type"])
        action = cls._s(beat["action"])
        speech = cls._s(beat["speech"])
        speaker = cls._s(beat["speaker"])
        lang = cls._s(beat["lang"]) or lang_default

        # A speaker is emitted whenever one is selected, whether or not the
        # beat has dialogue: "(S1) turns and looks" is a valid action beat
        # attributed to a known character.
        if not speaker:
            if not action:
                return ""
            return action[0].upper() + action[1:] + ("" if action.endswith(".") else ".")

        who = intro_used.get(speaker, "")
        lead = f"{who} ({speaker})" if who else f"({speaker})"
        lead = lead[0].upper() + lead[1:]
        if action:
            lead += f" {action}"

        if speech:
            sentence = f"{lead}: <d>[{lang}] {speech}</d>"
            if beat.get("carries"):
                sentence += "<scenetrans>"
            return sentence

        return lead + ("" if lead.endswith(".") else ".")

    @classmethod
    def _shot_text(cls, idx, shot, speakers_map, intro_used):
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

        cam = cls._camera_clause(shot["motion"], shot["ampl"], shot["speed"])
        if cam:
            body.append(cam)

        for beat in shot["beats"]:
            text = cls._beat_text(beat, intro_used)
            if not text:
                continue
            body.append(text)
            sp = cls._s(beat["speaker"])
            if sp and sp in intro_used:
                intro_used[sp] = ""  # identity written once

        return head + " " + " ".join(b for b in body if b)

    # -- top level ---------------------------------------------------------

    @classmethod
    def _build(cls, *values):
        vals = list(values)
        i = 0

        def take(n=1):
            nonlocal i
            out = vals[i:i + n]
            i += n
            return out[0] if n == 1 else out

        mode = cls._s(take())
        duration = take()
        style = cls._s(take())
        location = cls._s(take())
        lighting = cls._s(take())
        atmosphere = cls._s(take())
        camera_type = cls._s(take())
        video_role = cls._s(take())
        video_desc = cls._s(take())
        video_retention = cls._s(take())

        speaker_count = int(cls._s(take()) or 0)
        speakers = []
        for _ in range(MAX_SPEAKERS):
            speakers.append({
                "name": take(), "onscreen": take(), "age": take(),
                "gender": take(), "pitch": take(), "timbre": take(),
                "rate": take(), "accent": take(),
            })

        subject_count = int(cls._s(take()) or 0)
        subjects = []
        for _ in range(MAX_SUBJECTS):
            subjects.append({
                "kind": take(), "source": take(), "speaks": take(),
                "desc": take(), "retention": take(), "note": take(),
                "shots": take(),
            })

        task_types = take()
        summary_text = cls._s(take())
        shot_count = int(cls._s(take()) or 1)

        shots = []
        for _ in range(MAX_SHOTS):
            shot = {
                "cut": take(), "cutverb": take(), "framing": take(),
                "lens": take(), "motion": take(), "ampl": take(),
                "speed": take(), "anchor": take(), "beat_count": take(),
                "beats": [],
            }
            for _ in range(MAX_BEATS):
                shot["beats"].append({
                    "type": take(), "speaker": take(), "lang": take(),
                    "action": take(), "speech": take(), "carries": take(),
                })
            shot["beats"] = shot["beats"][:int(cls._s(shot["beat_count"]) or 0)]
            shots.append(shot)

        soundscape_presets = cls._s(take())
        soundscape = cls._s(take())
        music_presets = cls._s(take())
        music = cls._s(take())
        music_none = take()

        is_ref = mode.startswith("Ref2VA")
        shots = shots[:max(1, shot_count)]

        # speaker identity phrases, consumed on first use
        intro_used = {}
        for n in range(min(speaker_count, MAX_SPEAKERS)):
            intro = cls._speaker_intro(speakers[n])
            if intro:
                intro_used[f"S{n + 1}"] = intro

        # Global scene block, written once before [Shot 1]. Style, location,
        # lighting, atmosphere and camera body hold for the whole clip, so
        # they belong here rather than inside the opening shot.
        opening = cls._opening_clause(style, location, lighting,
                                      atmosphere, camera_type)

        shot_lines = [cls._shot_text(n, s, speakers, intro_used)
                      for n, s in enumerate(shots)]
        body = "\n".join(l for l in ([opening] + shot_lines) if l.strip())

        sound_field = cls._merge_audio(
            soundscape_presets, soundscape,
            lead="The scene carries {}.",
        ) or "N/A"

        if music_none:
            music_field = "N/A"
        else:
            music_field = cls._merge_audio(
                music_presets, music, lead="{}.",
            ) or "N/A"

        if not is_ref:
            # No blank lines anywhere: WanGP splits a prompt into separate
            # generations at an empty line. The guide asks for a blank line
            # after the instruction line, but that would break the prompt
            # into two jobs, so sections are packed onto consecutive lines.
            lines = []
            instruction = cls._instruction(mode, duration)
            if instruction:
                lines.append(instruction)
            lines.append(f"integrated_multimodal_description: {body}")
            lines.append(f"overall_soundscape: {sound_field}")
            lines.append(f"non_diegetic_music: {music_field}")
            return cls._no_blank_lines("\n".join(lines)), "Prompt written."

        # ---- reference mode: six sections ----
        active = subjects[:subject_count]
        counters = {k: 0 for k in ASSET_KINDS}
        defs, retention = [], []

        for s in active:
            kind = cls._s(s["kind"]) or "Subject"
            counters[kind] = counters.get(kind, 0) + 1
            label = f"<{kind} {counters[kind]}>"
            src = cls._s(s["source"])
            desc = cls._s(s["desc"])

            line = f"{label} is {desc}" if desc else f"{label}"
            if src:
                line += f", from {src}"
            defs.append(line.rstrip(".") + ".")

            marker = cls._s(s["retention"])
            note = cls._s(s["note"])
            where = cls._s(s["shots"])
            scope = ""
            if where and kind in ("Subject", "Picture"):
                shot_list = ", ".join(f"[Shot {p.strip()}]"
                                      for p in where.split(",") if p.strip())
                scope = f" (appears in {shot_list})"
            entry = f"{label}{scope}: {marker}"
            if note:
                entry += f" - {note}"
            retention.append(entry.rstrip(".") + ".")

        if video_role != "none" and video_desc:
            counters["Video"] += 1
            vlabel = f"<Video {counters['Video']}>"
            defs.append(f"{vlabel} provides {video_desc}.")
            retention.append(
                f"{vlabel}: {video_retention or 'weak_reference'} - {video_desc}."
            )

        prefix = cls._s(task_types)
        prefix = " + ".join(p.strip() for p in prefix.split(",") if p.strip())
        summary = f"[{prefix}] {summary_text}".strip() if prefix else summary_text

        # The opening clause already carries the style, so no separate
        # style line here.
        detailed = body

        sections = [
            "subject_definitions:\n" + "\n".join(defs) if defs else "subject_definitions:",
            "summary:\n" + summary if summary else "summary:",
            "retention_analysis:\n" + "\n".join(retention) if retention else "retention_analysis:",
            "detailed_description:\n" + detailed,
            "overall_soundscape:\n" + sound_field,
            "non_diegetic_music:\n" + music_field,
        ]
        # Single newlines only - a blank line would make WanGP treat what
        # follows as a separate generation.
        return (cls._no_blank_lines("\n".join(sections)),
                "Reference prompt written - check the summary reads naturally.")

    @staticmethod
    def _clear():
        # mode, duration, style, location, lighting, atmosphere,
        # camera_type, video_role, video_desc, video_retention
        out = [MODES[0], 8.0, "", "", "", "", "", "none", "", ""]
        out.append(0)                                 # speaker_count
        for _ in range(MAX_SPEAKERS):
            # name, presence, age, gender, pitch, timbre, rate, accent.
            # Presence has no blank choice, so it resets to its default.
            out += ["", "on-screen", "", "", "", "", "", ""]
        out.append(0)                                 # subject_count
        for _ in range(MAX_SUBJECTS):
            out += ["Subject", "", False, "", "fully_preserved", "", ""]
        out += [[], "", 1]                            # task_types, summary, shot_count
        for si in range(MAX_SHOTS):
            # cut, cutverb, framing, lens, motion, ampl, speed, anchor, beats
            out += ["opening" if si == 0 else None, "-" if si == 0 else "",
                    "", "", "", "", "", "", 0]
            out += ["action", "", "English", "", "", False] * MAX_BEATS
        # soundscape_presets, soundscape, music_presets, music, music_none
        out += [[], "", [], "", True]

        # Re-hide every slot: speakers, subjects, shots, then beats.
        # Shot 1 stays visible because a prompt always has at least one.
        out += [gr.update(visible=False)] * MAX_SPEAKERS
        out += [gr.update(visible=False)] * MAX_SUBJECTS
        out += [gr.update(visible=(i == 0)) for i in range(MAX_SHOTS)]
        out += [gr.update(visible=False)] * (MAX_SHOTS * MAX_BEATS)
        return out


Plugin = H3PromptBuilderPlugin
