"""The optional generator accordion at the foot of the Prompt Builder.

It adds no fields of its own beyond an idea box. Everything else — the mode,
the duration, the scheduling switch, the style and scene, the cast and their
voices, the reference slots, the soundscape and score — is read from the
builder's form through its own ``_unpack``, so there is one place to fill in.

The builder works exactly as before if this accordion is never opened, and a
failure in here is caught so it can never take the builder down with it.
"""

from __future__ import annotations

import time

import gradio as gr

from shared.gradio.progress import WangpProgress

from . import lint
from . import spec as spec_mod
from .modes import Target
from .spec import Character, Scene, WindowPlan

BRACKET = ("Picture", "Video", "Audio", "Subject")


def _s(value) -> str:
    return str(value or "").strip()


def _label(value: str) -> str:
    """'Video 1' or '<Video 1>' from the builder's slot fields -> '<Video 1>'."""
    text = _s(value)
    if not text:
        return ""
    if text.startswith("<") and text.endswith(">"):
        return text
    return f"<{text}>" if any(text.startswith(k) for k in BRACKET) else text


def _first_label(value) -> str:
    if isinstance(value, (list, tuple)):
        for item in value:
            label = _label(item)
            if label:
                return label
        return ""
    return _label(value)


def _all_labels(value) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [l for l in (_label(v) for v in value) if l]
    label = _label(value)
    return [label] if label else []


def family_of(form: dict) -> str:
    return "ref2va" if form.get("ref_mode") else "fl2va"


def mode_of(form: dict) -> str:
    if form.get("ref_mode"):
        return "Ref2VA"
    start, end = bool(form.get("start_image")), bool(form.get("end_image"))
    if start and end:
        return "FL2VA"
    if end:
        return "L2VA"
    if start:
        return "I2VA"
    return "T2VA"


def cast_of(form: dict) -> list[Character]:
    """The builder's cast entries and reference slots as characters."""
    cast: list[Character] = []
    entries = form.get("entries") or []
    for entry in entries[:int(form.get("entry_count") or 0)]:
        if not any(_s(entry.get(k)) for k in ("desc", "speaker", "source", "retention")):
            continue
        sources = _all_labels(entry.get("source"))
        character = Character(
            kind=_s(entry.get("kind")) or "Subject",
            desc=_s(entry.get("desc")),
            use_creator=False,                      # the builder already wrote the sentence
            onscreen=_s(entry.get("onscreen")) or "on-screen",
            speaker=_s(entry.get("speaker")),
            lang=_s(entry.get("lang")) or "English",
            age=_s(entry.get("age")), gender=_s(entry.get("gender")),
            pitch=_s(entry.get("pitch")), timbre=_s(entry.get("timbre")),
            rate=_s(entry.get("rate")), accent=_s(entry.get("accent")),
            source=sources[0] if sources else "",
            retention=_s(entry.get("retention")),
            retention_note=_s(entry.get("note")),
            voice_from=_first_label(entry.get("voice_from")),
            motion_from=_first_label(entry.get("motion_from")),
        )
        if len(sources) > 1:
            character.desc = (character.desc.rstrip(".") + ". Also drawn from " + ", ".join(sources[1:]) + ".").strip()
        cast.append(character)

    # Reference slots: assets declared once in the builder, with their own role.
    for ref in (form.get("refs") or [])[:int(form.get("ref_count") or 0)]:
        slot = _label(ref.get("slot"))
        if not slot:
            continue
        kind, _, number = slot.strip("<>").partition(" ")
        role, desc = _s(ref.get("role")), _s(ref.get("desc"))
        cast.append(Character(
            kind=kind or "Video", label_index=int(number) if number.isdigit() else 0,
            use_creator=False, onscreen="non-speaking",
            desc=" ".join(p for p in (desc, f"Used as {role}." if role else "") if p).strip(),
            retention=_s(ref.get("retention")), retention_note=_s(ref.get("from")),
            retention_on="source" if kind in ("Video", "Audio", "Picture") else "subject",
            source=slot,
        ))
    return cast


def scene_of(form: dict) -> Scene:
    return Scene(
        style=_s(form.get("style")), location=_s(form.get("location")),
        time_of_day=_s(form.get("time_of_day")), lighting=_s(form.get("lighting")),
        atmosphere=_s(form.get("atmosphere")), camera=_s(form.get("camera_type")),
        grade=_s(form.get("grading")), soundscape=_s(form.get("soundscape")),
        music=_s(form.get("music")),
        task_types=list(form.get("task_types") or []),
        notes=_s(form.get("summary_text")),
    )


def score_rule(form: dict) -> str:
    # Only a written score or a named audio source means there is music: the builder's
    # role and retention fields carry defaults even when no score is wanted.
    music, role = _s(form.get("music")), _s(form.get("music_role"))
    source = _label(form.get("music_from"))
    if music or source:
        detail = " ".join(p for p in (music, f"Role: {role}." if role else "",
                                      f"It comes from {source}." if source else "") if p)
        return f"SCORE: the video has non-diegetic music. Write it in non_diegetic_music. {detail}".strip()
    return "SCORE: there is no non-diegetic music. non_diegetic_music must be exactly N/A."


def extras(form: dict) -> str:
    """Builder fields that are instructions rather than description."""
    lines = []
    ambience_from, ambience_retention = _label(form.get("ambience_from")), _s(form.get("ambience_retention"))
    if ambience_from:
        lines.append(f"AMBIENCE: the room tone comes from {ambience_from}"
                     + (f" ({ambience_retention})" if ambience_retention else "") + ".")
    if _s(form.get("music_retention")):
        lines.append(f"SCORE RETENTION: {_s(form.get('music_retention'))}.")
    return "\n".join(lines)


class GeneratorPanel:
    """Created inside the builder's accordion; wired once the flat list exists."""

    def __init__(self, plugin):
        self.plugin = plugin
        self.studio = plugin.h3_studio
        self.flat_length = None

    # ------------------------------------------------------------------ UI
    def build(self):
        with gr.Accordion("Write it with the prompt enhancer (optional)", open=False):
            gr.Markdown(
                "Turns the form above into a finished H3 prompt using **the prompt enhancer configured in "
                "Configuration** — whichever model and settings you already use there.\n\n"
                "Everything is taken from the fields above: mode and keyframes, duration, the scheduling switch, "
                "style and scene, the cast with their voices and reference slots, soundscape and score. Add an idea "
                "below to say what actually happens, or leave the form empty and write only the idea. One paragraph "
                "per sliding window, separated by a blank line."
            )
            self.idea = gr.Textbox(lines=5, label="The idea (one paragraph per window)",
                                   placeholder="She wakes in the chair and listens.\n\n"
                                               "She crosses to the window and calls out.")
            with gr.Row():
                self.use_form = gr.Checkbox(True, label="Use the fields above")
                self.use_action = gr.Checkbox(True, label="Keep the action I already wrote")
                self.think = gr.Checkbox(False, label="Think (Qwen enhancers)")
            self.progress = WangpProgress.component()
            with gr.Row():
                self.generate_btn = gr.Button("Write the prompt", variant="primary")
            self.output = gr.Textbox(lines=12, label="The written prompt (editable)", show_copy_button=True)
            self.report = gr.Markdown()
            with gr.Row():
                self.insert_btn = gr.Button("Insert into prompt", variant="primary")
                self.append_btn = gr.Button("Append to prompt")
                self.check_btn = gr.Button("Check it")
        return self

    # --------------------------------------------------------------- logic
    def _form(self, values):
        try:
            return self.plugin._unpack(values)
        except Exception as exc:
            raise gr.Error(f"The builder's fields could not be read: {exc}")

    def _assemble(self, state, values, idea, use_form, use_action):
        form = self._form(values) if use_form else {}
        family = family_of(form) if use_form else (self.studio.current(state).get("family") or "fl2va")
        cast = cast_of(form) if use_form else []
        scene = scene_of(form) if use_form else Scene()
        blocks = []
        if use_form:
            block = spec_mod.spec_block(scene, cast, family)
            if block:
                blocks.append(block)
            blocks.append(score_rule(form))
            more = extras(form)
            if more:
                blocks.append(more)
            action = _s(form.get("action"))
            if action and use_action:
                blocks.append("ALREADY WRITTEN in the builder — keep these shots, their order and their timings, and "
                              "improve only the wording and the missing detail:\n" + action)
        spec_text = "\n\n".join(b for b in blocks if b)
        windows = lint.split_windows(idea or "") or [(idea or "").strip()]
        if not any(w.strip() for w in windows):
            action = _s(form.get("action"))
            if not action:
                raise gr.Error("Write an idea below, or fill in the action in the form above.")
            windows = [action]
        duration = float(form.get("duration") or 0) or None
        plans = [WindowPlan(index=n, seconds=duration or 0, join="continue",
                            overlap=spec_mod.H3_OVERLAP_DEFAULT, idea=w)
                 for n, w in enumerate(windows, 1)]
        return form, family, cast, plans, spec_text

    def generate(self, *args, progress=WangpProgress()):
        state, values = args[0], args[1:-4]
        idea, use_form, use_action, think = args[-4], args[-3], args[-2], args[-1]
        form, family, cast, plans, spec_text = self._assemble(state, values, idea, use_form, use_action)
        mode = mode_of(form) if use_form else "Auto"
        canonical = spec_mod.subject_definitions(cast, family)
        retention = spec_mod.retention_map(cast, family)
        try:
            outputs, targets, briefs = self.studio.write(
                state, "\n\n".join(p.idea for p in plans), [], family,
                plans[0].seconds or None, mode, "auto", progress, bool(think),
                spec_text=spec_text, plans=plans,
                commands_enabled=bool(form.get("window_commands")),
                force_definitions=canonical, force_retention=retention)
        except RuntimeError as exc:
            raise gr.Error(str(exc))
        text = "\n\n".join(outputs)
        expected = "Ref2VA" if family == "ref2va" else "auto"
        report = lint.check(text, expected, targets[0].duration if len(targets) == 1 else None, len(outputs))
        notes = spec_mod.window_warnings(plans)
        repaired = self.studio.last_run.get("repaired") or []
        if repaired:
            notes.insert(0, "window " + ", ".join(str(n) for n in repaired) + " came back unusable and was written again")
        summary = lint.format_report(report) + ("\n" + "\n".join(f"- ⚠️ {n}" for n in notes) if notes else "")
        return text, summary

    def check(self, text, *values):
        form = self._form(values) if values else {}
        duration = float(form.get("duration") or 0) or None
        windows = lint.check(text or "")["windows"]
        expected = "Ref2VA" if family_of(form) == "ref2va" else "auto"
        return lint.format_report(lint.check(text or "", expected, duration if windows == 1 else None,
                                             0 if windows > 1 else 1))

    @staticmethod
    def insert(text):
        if not _s(text):
            raise gr.Error("Write the prompt first.")
        return _s(text), "Prompt replaced. Set *How to Process each Line of the Text Prompt* to the " \
                         "paragraph-per-sliding-window option for a multi-window prompt."

    @staticmethod
    def append(current, text):
        if not _s(text):
            raise gr.Error("Write the prompt first.")
        joined = (_s(current) + "\n\n" + _s(text)).strip() if _s(current) else _s(text)
        return joined, "Appended below what was already in the box."

    # -------------------------------------------------------------- wiring
    def wire(self, flat):
        self.flat_length = len(flat)
        status = self.report
        WangpProgress.bind(self.generate_btn.click, self.generate,
                           inputs=[self.plugin.state] + list(flat) + [self.idea, self.use_form, self.use_action, self.think],
                           outputs=[self.output, self.report], component=self.progress)
        self.check_btn.click(self.check, [self.output] + list(flat), [status])
        self.insert_btn.click(self.insert, [self.output], [self.plugin.prompt, status])
        self.append_btn.click(self.append, [self.plugin.prompt, self.output], [self.plugin.prompt, status])
        return self


def attach_generator_ui(plugin):
    """Called inside the builder's accordion. Returns None if anything is missing."""
    try:
        return GeneratorPanel(plugin).build()
    except Exception as exc:                                  # never break the builder
        print(f"[MiniMax H3 Prompt Builder] prompt-enhancer panel unavailable: {exc}")
        return None
