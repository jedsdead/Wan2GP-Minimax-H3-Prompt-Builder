"""Decide which H3 task a request is, and describe it to the writing model."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

BASE_MODES = ("T2VA", "I2VA", "FL2VA", "L2VA")
ALL_MODES = BASE_MODES + ("Ref2VA",)

WINDOW_COMMAND = re.compile(r"\[\s*/[^\]]*\]")

COMBAT_TERMS = (
    "sword", "swords", "swordplay", "swordsman", "swordswoman", "blade", "blades", "saber", "sabre",
    "katana", "longsword", "greatsword", "rapier", "scimitar", "dao", "jian", "spear", "spears",
    "polearm", "glaive", "halberd", "guandao", "naginata", "lance", "pike", "axe", "axes", "warhammer",
    "mace", "dagger", "daggers", "knife fight", "duel", "duels", "dueling", "duelling", "fencing",
    "parry", "parries", "parrying", "riposte", "lightsaber", "energy blade", "execution", "executes",
    "behead", "decapitate", "cavalry charge", "mounted charge", "staff fight", "bo staff", "weapon",
    "weapons", "clash of blades", "slash", "slashes", "thrust", "cleave",
)
_COMBAT_RE = re.compile(r"\b(" + "|".join(re.escape(t) for t in sorted(COMBAT_TERMS, key=len, reverse=True)) + r")\b", re.I)


@dataclass
class Target:
    mode: str = "T2VA"
    has_image: bool = False            # an image is shown to the writing model
    duration: float | None = None      # seconds, when known
    window_index: int = 1              # 1-based
    window_count: int = 1
    new_shot: bool = False             # this window has [/new_shot] or [/overlap=0]
    combat: bool = False
    reasons: list[str] = field(default_factory=list)

    @property
    def sliding(self) -> bool:
        return self.window_count > 1

    @property
    def continuation(self) -> bool:
        return self.sliding and self.window_index > 1 and not self.new_shot


def detect_combat(text: str, setting: str = "auto") -> bool:
    setting = (setting or "auto").lower()
    if setting in ("on", "true", "yes", "1"):
        return True
    if setting in ("off", "false", "no", "0"):
        return False
    return bool(_COMBAT_RE.search(text or ""))


def mode_from_labels(labels: list[str] | None, reference_model: bool, images_visible: bool) -> str:
    if reference_model:
        return "Ref2VA"
    if not images_visible or not labels:
        return "T2VA"
    lowered = [str(label).lower() for label in labels]
    start = any(label.startswith("start image") for label in lowered)
    end = any(label.startswith("end image") for label in lowered)
    if start and end:
        return "FL2VA"
    if end:
        return "L2VA"
    if start:
        return "I2VA"
    return "T2VA"


def mode_from_image_prompt_type(image_prompt_type: str | None, reference_model: bool, images_visible: bool) -> str:
    if reference_model:
        return "Ref2VA"
    if not images_visible:
        return "T2VA"
    flags = image_prompt_type or ""
    start = "S" in flags or "L" in flags or "V" in flags
    end = "E" in flags
    if start and end:
        return "FL2VA"
    if end:
        return "L2VA"
    if start:
        return "I2VA"
    return "T2VA"


def guess_mode_from_text(text: str) -> str:
    """Used by the Deepy tools and the preview when no Wan2GP inputs exist."""
    t = (text or "").lower()
    if re.search(r"<(subject|video|audio) \d+>|subject_definitions|reference (image|video|audio)|ref2va", t):
        return "Ref2VA"
    first = bool(re.search(r"first[- ]frame|start(ing)? (image|frame)|i2va|from (this|the) image", t))
    last = bool(re.search(r"last[- ]frame|end(ing)? (image|frame)|l2va|lands? on", t))
    if "fl2va" in t or (first and last):
        return "FL2VA"
    if last:
        return "L2VA"
    if first:
        return "I2VA"
    return "T2VA"


def window_commands(text: str) -> tuple[str, list[str]]:
    commands = WINDOW_COMMAND.findall(text or "")
    return WINDOW_COMMAND.sub("", text or "").strip(), commands


def commands_force_new_shot(commands: list[str]) -> bool:
    joined = " ".join(commands).lower().replace(" ", "")
    return "/new_shot" in joined or bool(re.search(r"/overlap=0(?![\d.])", joined))


def command_duration(commands: list[str], fps: float | None = None) -> float | None:
    joined = " ".join(commands).lower().replace(" ", "")
    match = re.search(r"/duration=([\d.]+)(s|%)?", joined)
    if not match:
        return None
    value, unit = float(match.group(1)), match.group(2)
    if unit == "s":
        return value
    if unit is None and fps:
        return value / fps
    return None


def _fmt_seconds(value: float) -> str:
    return f"{value:.2f}".rstrip("0").rstrip(".")


MODE_SUMMARY = {
    "T2VA": "text to video+audio. Build the whole audiovisual timeline from the text. No reference labels and no picture-alignment line.",
    "I2VA": "first frame to video+audio. The supplied image is <Picture 1>, the exact first frame of [Shot 1] at 0.00 s. Start from what it shows and develop forward.",
    "FL2VA": "first and last frame to video+audio. <Picture 1> is the first frame; <Picture 2> is the last frame. Describe one continuous path from the first frame to the last.",
    "L2VA": "last frame to video+audio. The supplied image is <Picture 1>, the final frame. Infer a plausible opening and converge onto it at the end.",
    "Ref2VA": "full-reference video+audio. Use the six-section rewrite format with stable reference labels.",
}


def instruction_line(target: Target) -> str | None:
    """Exact first line required by base-en 2.1 for keyframe modes."""
    d = _fmt_seconds_two(target.duration) if target.duration else "S.SS"
    if target.mode == "I2VA":
        return "For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced."
    if target.mode == "FL2VA":
        return f"How the reference pictures align with the target video — Picture 1 (from Shot 1) aligns with the 0.00-second mark of the target video; Picture 2 (from Shot N) aligns with the {d}-second mark of the target video."
    if target.mode == "L2VA":
        return f"How the reference pictures align with the target video — <Picture 1> (from [Shot N]) aligns with the {d}-second mark of the target video."
    return None


def _fmt_seconds_two(value: float | None) -> str:
    return "S.SS" if value is None else f"{value:.2f}"


def target_block(target: Target) -> str:
    lines = [f"TARGET TASK: {target.mode} — {MODE_SUMMARY[target.mode]}"]
    if target.mode == "Ref2VA":
        if target.has_image:
            lines.append("The supplied image is <Picture 1>, a general reference asset, not the first frame, unless the user explicitly asks for a keyframe.")
        else:
            lines.append("No reference media is visible to you. Use only labels and asset facts stated in the user's text; never invent what unseen references contain.")
    line = instruction_line(target)
    if line:
        lines.append("The first line of the output must be exactly (replace N with the index of the final shot):")
        lines.append(line)
    if target.duration:
        lines.append(f"Duration: {_fmt_seconds(target.duration)} seconds. Every timestamp must fall inside it and the timeline must fill it.")
    if target.sliding:
        if target.window_index >= 1:
            lines.append(f"This is window {target.window_index} of {target.window_count} of one continuous Wan2GP sliding-window sequence. Write only this window.")
        else:
            lines.append(f"This prompt is one window of a {target.window_count}-window Wan2GP sliding-window sequence. Write only this window. "
                         'If it is not the first window, [Shot 1] has no timestamp and opens with "The camera cuts to ..." or "The camera continues to ...".')
        if target.new_shot:
            lines.append("This window starts a new scene with no visual carry-over, so [Shot 1] opens cold like a first window.")
        elif target.continuation:
            lines.append('[Shot 1] of this window has no timestamp and must open with "The camera cuts to ..." or "The camera continues to ..." naming exactly what continues.')
        lines.append("Keep every section on consecutive lines with single line breaks. Never output a blank line: a blank line splits the window in Wan2GP's paragraph mode.")
        lines.append("End the window on a specific subject in frame, preferably with slow camera motion; never on a fade, an empty frame, or a static low-detail hold.")
    if target.combat:
        lines.append("The request involves weapon combat: plan the choreography from the combat excerpts, but keep H3 shot notation from the mode guide.")
    return "\n".join(lines)


def checklist(target: Target) -> str:
    items = []
    if target.mode == "Ref2VA":
        items.append("sections in order: subject_definitions, summary, retention_analysis, detailed_description, overall_soundscape, non_diegetic_music")
        items.append("retention_analysis lines read `<Subject N> (appears in [Shot 1], ...): marker - ...` with no speaker IDs")
    else:
        items.append("fields in order: integrated_multimodal_description, overall_soundscape, non_diegetic_music")
    if instruction_line(target):
        items.append("the alignment instruction is the first line")
    items.append("[Shot 1] has no timestamp; later shots start `[Shot N] At MM:SS.mmm, the camera cuts to ...` with strictly increasing times")
    items.append("every cut adds new subject, space, state, viewpoint or time information")
    items.append("spoken words only inside <d>[Language] ...</d>, speaker IDs (S1), (S2) only for characters who vocalize")
    items.append("non_diegetic_music is N/A unless a score was requested")
    items.append("no Markdown, no code fence, no commentary")
    return "Before answering, verify: " + "; ".join(items) + "."
