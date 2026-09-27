"""Structured input for the generator: a cast, a scene, and a window plan.

The cast and scene are written once and then handed to the writing model with
every window, so a character keeps the same wording, label and speaker ID from
the first window to the last. The window plan owns the `[/duration=...]`,
`[/overlap=...]` and `[/new_shot]` commands Wan2GP reads at the head of each
paragraph.

Character fields use the same names as the Prompt Builder's saved subjects, so
a subject JSON saved there loads here unchanged.
"""

from __future__ import annotations

import json
import os
import re
import uuid
from dataclasses import asdict, dataclass, field

from .vocab import (H3_FPS, H3_OVERLAP_DEFAULT, H3_WINDOW_DOC_SECONDS, H3_WINDOW_FRAMES_MAX,
                    H3_WINDOW_FRAMES_MIN)

MAX_SPEAKERS = 6
# Wan2GP's H3 reference slots: Reference Images 1-9, Reference Video 1-3, Audio Reference 1-3.
REF_SLOT_COUNTS = {"Picture": 9, "Video": 3, "Audio": 3}


def slot_labels(kind: str) -> list[str]:
    return [f"<{kind} {n}>" for n in range(1, REF_SLOT_COUNTS.get(kind, 1) + 1)]


def source_choices() -> list[str]:
    return [""] + slot_labels("Picture") + slot_labels("Video") + slot_labels("Audio")
JOIN_CHOICES = [("Continue from the previous window (overlap)", "continue"),
                ("Cut to a new scene (/new_shot)", "cut")]
ACCENT_MARKERS = ("accent", "lilt", "brogue", "drawl", "twang", "burr", "inflection", "cadence", "dialect", "intonation")
FIELD_NAMES = ["kind", "desc", "onscreen", "age", "gender", "pitch", "timbre", "rate", "accent", "lang",
               "use_creator", "char_name", "char_ethnicity", "char_gender", "char_age", "char_height",
               "char_build", "char_hairstyle", "char_haircolor", "char_eyecolor", "char_clothing"]


def _s(value) -> str:
    return str(value or "").strip()


@dataclass
class Character:
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    kind: str = "Subject"                # Subject | Picture | Video | Audio
    label_index: int = 0                 # 0 = numbered automatically, else <Kind N> is fixed to this N
    desc: str = ""                       # free description, used when the creator fields are empty
    onscreen: str = "on-screen"          # on-screen | off-screen | non-speaking
    age: str = ""
    gender: str = ""
    pitch: str = ""
    timbre: str = ""
    rate: str = ""
    accent: str = ""
    lang: str = "English"
    use_creator: bool = True
    char_name: str = ""
    char_ethnicity: str = ""
    char_gender: str = ""
    char_age: str = ""
    char_height: str = ""
    char_build: str = ""
    char_hairstyle: str = ""
    char_haircolor: str = ""
    char_eyecolor: str = ""
    char_clothing: str = ""
    speaker: str = ""                    # S1 ... S6, blank when the character does not speak
    source: str = ""                     # <Picture 1>, <Video 1> ... the asset this character comes from
    retention: str = ""                  # fully_preserved, attribute_transfer ...
    retention_note: str = ""
    retention_on: str = "subject"        # which label carries it: subject | source | both
    voice_from: str = ""                 # <Audio 1> ... the voice this character is given
    motion_from: str = ""                # <Video 1> ... the movement this character copies

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(data: dict) -> "Character":
        kw = {k: data[k] for k in Character.__dataclass_fields__ if k in data}
        kw.setdefault("id", uuid.uuid4().hex[:8])
        return Character(**kw)

    # ---------------------------------------------------------------- text
    @property
    def name(self) -> str:
        return _s(self.char_name) or (_s(self.desc).split(",")[0] if self.desc else "") or "Unnamed"

    def appearance(self) -> str:
        """One sentence describing the character, from the creator fields or the free field."""
        if not self.use_creator:
            return _s(self.desc)
        name, parts = _s(self.char_name), []
        person = " ".join(p for p in (_s(self.char_ethnicity), _s(self.char_gender)) if p)
        age = _s(self.char_age)
        if person or age:
            parts.append(" ".join(p for p in (f"a {person}" if person else "a person", age) if p).strip())
        height, build = _s(self.char_height), _s(self.char_build)
        if height and build:
            parts.append(f"{height} with a {build} build")
        elif height or build:
            parts.append(height or f"a {build} build")
        hair = " ".join(p for p in (_s(self.char_hairstyle), _s(self.char_haircolor)) if p)
        eyes = _s(self.char_eyecolor)
        look = ""
        if hair:
            look = f"{hair} hair"
        if eyes:
            look = f"{look} and {eyes} eyes" if look else f"{eyes} eyes"
        if look:
            parts.append(look if (height or build) else "with " + look)
        if _s(self.char_clothing):
            parts.append(f"wearing {_s(self.char_clothing)}")
        body = ", ".join(p for p in parts if p)
        extra = _s(self.desc)
        sentence = f"{name}, {body}" if name and body else (name or body)
        sentence = sentence.rstrip(".")
        if sentence and extra:
            return f"{sentence}. {extra.rstrip('.')}."
        return (sentence + ".") if sentence else (extra.rstrip(".") + "." if extra else "")

    def voice(self) -> str:
        """Voice description, written once so it does not drift between windows."""
        if not _s(self.speaker):
            return ""
        person = " ".join(p for p in (_s(self.age), _s(self.gender)) if p)
        bits = [b for b in (_s(self.pitch), _s(self.timbre), _s(self.rate)) if b]
        parts = []
        if person:
            parts.append(f"a {person} voice")
        elif bits:
            parts.append("a voice")
        if bits:
            qualities = bits[0] if len(bits) == 1 else ", ".join(bits[:-1]) + " and " + bits[-1]
            parts.append(qualities)
        if _s(self.accent):
            low = _s(self.accent).lower()
            phrase = self.accent if any(m in low for m in ACCENT_MARKERS) else f"a {self.accent} accent"
            parts.append(f"with {phrase}")
        if _s(self.onscreen) == "off-screen":
            parts.append("heard off-screen")
        return ", ".join(p for p in parts if p)


@dataclass
class Scene:
    style: str = ""
    location: str = ""
    time_of_day: str = ""
    lighting: str = ""
    atmosphere: str = ""
    camera: str = ""
    lens: str = ""
    grade: str = ""
    framing: str = ""
    motion: str = ""
    rig: str = ""
    soundscape: str = ""
    music: str = ""
    task_types: list = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(data: dict) -> "Scene":
        return Scene(**{k: data[k] for k in Scene.__dataclass_fields__ if k in data})

    def lines(self) -> list[str]:
        rows = [("Visual style", self.style), ("Location", self.location), ("Time of day", self.time_of_day),
                ("Lighting", self.lighting), ("Atmosphere", self.atmosphere), ("Camera", self.camera),
                ("Lens", self.lens), ("Colour grade", self.grade), ("Preferred framing", self.framing),
                ("Preferred camera motion", " ".join(p for p in (self.motion, self.rig and f"on a {self.rig}") if p)),
                ("Soundscape direction", self.soundscape), ("Score direction", self.music),
                ("Other notes", self.notes)]
        return [f"{label}: {_s(value)}" for label, value in rows if _s(value)]

    @property
    def empty(self) -> bool:
        return not self.lines() and not self.task_types


def assign_speakers(cast: list[Character]) -> None:
    """Number speaking characters S1, S2 ... in list order, leaving non-speakers blank."""
    n = 0
    for c in cast:
        if _s(c.onscreen) == "non-speaking" or (not c.use_creator and not _s(c.desc) and not _s(c.char_name)):
            if _s(c.onscreen) == "non-speaking":
                c.speaker = ""
            continue
        if _s(c.speaker) or _s(c.onscreen) in ("on-screen", "off-screen"):
            n += 1
            c.speaker = f"S{min(n, MAX_SPEAKERS)}"


def labels(cast: list[Character], family: str) -> dict[str, str]:
    """`<Kind N>` per cast entry for Ref2VA; base modes describe characters in words instead.

    A chosen number is honoured; the rest are filled with the lowest free number
    for their kind, so adding a subject never renumbers the ones already set.
    """
    if family != "ref2va":
        return {}
    out: dict[str, str] = {}
    used: dict[str, set] = {}
    for c in cast:
        kind = _s(c.kind) or "Subject"
        index = int(c.label_index or 0)
        if index and index not in used.setdefault(kind, set()):
            used[kind].add(index)
            out[c.id] = f"<{kind} {index}>"
    for c in cast:
        if c.id in out:
            continue
        kind = _s(c.kind) or "Subject"
        taken = used.setdefault(kind, set())
        index = 1
        while index in taken:
            index += 1
        taken.add(index)
        out[c.id] = f"<{kind} {index}>"
    return out


def label_conflicts(cast: list[Character], family: str) -> list[str]:
    """Report numbers asked for twice; the second one is renumbered automatically."""
    if family != "ref2va":
        return []
    marks = labels(cast, family)
    notes, seen = [], {}
    for c in cast:
        index = int(c.label_index or 0)
        if not index:
            continue
        key = (_s(c.kind) or "Subject", index)
        if key in seen:
            notes.append(f"{c.name or 'a subject'} also asks for <{key[0]} {index}>, already taken by "
                         f"{seen[key] or 'another entry'} — it was given {marks.get(c.id, '?')} instead.")
        else:
            seen[key] = c.name
    return notes


def max_label_index(kind: str = "Subject") -> int:
    return REF_SLOT_COUNTS.get(kind, 9)


def subject_definitions(cast: list[Character], family: str) -> str:
    """The canonical `subject_definitions:` body, so every window says the same thing."""
    marks = labels(cast, family)
    lines = []
    for c in cast:
        appearance = c.appearance()
        if not appearance and not _s(c.source):
            continue
        label = marks.get(c.id)
        source = _s(c.source)
        voice = c.voice()
        extras = ""
        if _s(c.voice_from):
            extras += f" Voice reference: {_s(c.voice_from)}."
        if _s(c.motion_from):
            extras += f" Movement reference: {_s(c.motion_from)}."
        name, body = _s(c.char_name), appearance.rstrip(".")
        if source and name and body.startswith(name):
            rest = body[len(name):].lstrip(", ")
            body = f"{name} from {source}" + (f", {rest}" if rest else "")
        elif source:
            body = f"{body} from {source}"
        if label and re.match(r"^(A|An|The)\b", body):
            body = body[0].lower() + body[1:]
        text = f"{label} is {body}." if label else f"{body}."
        if voice:
            speaker = f" ({c.speaker})" if _s(c.speaker) else ""
            text += f" Speaks{speaker} with {voice}."
        lines.append(text + extras)
    return "\n".join(lines)


def retention_lines(cast: list[Character], family: str, shots: str = "[Shot 1]") -> str:
    """One line per reference label, from the same map the generator pins with."""
    out = []
    for label, (marker, note) in retention_map(cast, family).items():
        head = f"{label}" if label.startswith("<Audio") else f"{label} (appears in {shots})"
        out.append(f"{head}: {marker}" + (f" - {note}" if note else ""))
    return "\n".join(out)


def retention_map(cast: list[Character], family: str) -> dict[str, tuple[str, str]]:
    """label -> (marker, note): the retention every window must repeat for that asset."""
    if family != "ref2va":
        return {}          # base modes carry no reference labels at all
    marks = labels(cast, family)
    out: dict[str, tuple[str, str]] = {}
    for c in cast:
        label, source = marks.get(c.id), _s(c.source)
        if _s(c.retention):
            note, where = _s(c.retention_note), (_s(c.retention_on) or "subject")
            if label and where in ("subject", "both"):
                out[label] = (_s(c.retention), note)
            if source and source != label and where in ("source", "both"):
                who = label or c.name or "the subject"
                out[source] = (_s(c.retention), note or f"supplies {who}")
        if _s(c.voice_from):
            out[_s(c.voice_from)] = ("reference", "guides this character's voice; source wording is not copied")
        if _s(c.motion_from):
            out[_s(c.motion_from)] = ("attribute_transfer", "its movement and timing guide this character; appearance is not copied")
    return out


def spec_block(scene: Scene, cast: list[Character], family: str) -> str:
    """The block appended to every window's request so the writer stays consistent."""
    parts = []
    scene_lines = scene.lines()
    if scene.task_types:
        scene_lines.insert(0, "Task type for the summary line: [" + " + ".join(scene.task_types) + "]")
    if scene_lines:
        parts.append("SCENE SPEC (use these choices; do not replace them with your own):\n" + "\n".join(f"- {l}" for l in scene_lines))
    marks = labels(cast, family)
    rows = []
    for c in cast:
        appearance = c.appearance()
        if not appearance and not _s(c.source):
            continue
        head = marks.get(c.id) or c.name
        bits = [appearance.rstrip(".") + "."]
        if _s(c.speaker):
            bits.append(f"Speaker ID ({c.speaker}), {_s(c.onscreen) or 'on-screen'}, speaking {_s(c.lang) or 'English'}.")
        else:
            bits.append("Does not speak.")
        if c.voice():
            bits.append(f"Voice: {c.voice()}.")
        if _s(c.source):
            bits.append(f"Comes from {_s(c.source)}.")
        if _s(c.retention):
            bits.append(f"Retention: {_s(c.retention)}" + (f" — {_s(c.retention_note)}" if _s(c.retention_note) else "") + ".")
        if _s(c.voice_from):
            bits.append(f"Voice reference: {_s(c.voice_from)}.")
        if _s(c.motion_from):
            bits.append(f"Movement reference: {_s(c.motion_from)} — copy its movement and timing, not its appearance.")
        rows.append(f"- {head}: " + " ".join(bits))
    fixed = retention_map(cast, family)
    if fixed:
        parts.append("RETENTION (fixed for the whole sequence — every window's retention_analysis must give each label exactly "
                     "this marker, never a different one; only the shot list changes from window to window):\n"
                     + "\n".join(f"- {label}: {marker}" + (f" - {note}" if note else "") for label, (marker, note) in fixed.items()))
    if rows:
        parts.append("CAST (identical in every window — keep each label, speaker ID, name and appearance wording exactly as given; "
                     "never renumber, rename or re-describe them, and never introduce a character that is not listed):\n" + "\n".join(rows))
    return "\n\n".join(parts)


# ---------------------------------------------------------------- windows
@dataclass
class WindowPlan:
    index: int = 1
    seconds: float = 0.0
    join: str = "continue"          # continue | cut  (ignored on window 1)
    overlap: int = H3_OVERLAP_DEFAULT
    idea: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(data: dict) -> "WindowPlan":
        return WindowPlan(**{k: data[k] for k in WindowPlan.__dataclass_fields__ if k in data})

    @property
    def new_shot(self) -> bool:
        return self.index > 1 and self.join == "cut"


def seconds_text(value: float) -> str:
    return f"{float(value):.3f}".rstrip("0").rstrip(".")


def window_command(plan: WindowPlan, enabled: bool = True) -> str:
    """`[/duration=5s,/overlap=18]` or `[/duration=5s,/new_shot]`.

    Written on every window or none: Wan2GP pads the video by repeating the last
    paragraph only when no window carries a duration. The first window takes no
    join, because its overlap is whatever a start image or continued video gives it.
    """
    if not enabled or not plan.seconds or float(plan.seconds) <= 0:
        return ""
    parts = [f"/duration={seconds_text(plan.seconds)}s"]
    if plan.index > 1:
        parts.append("/new_shot" if plan.join == "cut" else f"/overlap={max(0, int(plan.overlap or 0))}")
    return "[" + ",".join(parts) + "]"


def window_warnings(plans: list[WindowPlan]) -> list[str]:
    out, low, high = [], *H3_WINDOW_DOC_SECONDS
    for plan in plans:
        secs = float(plan.seconds or 0)
        if secs <= 0:
            continue
        frames = int(round(secs * H3_FPS))
        if frames > H3_WINDOW_FRAMES_MAX:
            out.append(f"Window {plan.index}: {seconds_text(secs)}s is {frames} frames, past the {H3_WINDOW_FRAMES_MAX} frames "
                       f"({seconds_text(H3_WINDOW_FRAMES_MAX / H3_FPS)}s) the Sliding Window Size slider allows.")
        elif secs > high:
            out.append(f"Window {plan.index}: {seconds_text(secs)}s is {frames} frames, past the {seconds_text(high)}s MiniMax documents "
                       "for one window.")
        elif secs < low:
            out.append(f"Window {plan.index}: {seconds_text(secs)}s is {frames} frames, under the {seconds_text(low)}s MiniMax documents — "
                       f"H3 generates {H3_WINDOW_FRAMES_MIN} frames minimum and may raise the overlap to fill them.")
    return out


def sync_plans(plans: list[WindowPlan], ideas: list[str], default_seconds: float, default_join: str,
               default_overlap: int) -> list[WindowPlan]:
    """Keep one plan row per window of the idea text, preserving what is already set."""
    out = []
    for index, idea in enumerate(ideas, 1):
        old = plans[index - 1] if index <= len(plans) else None
        out.append(WindowPlan(index=index,
                              seconds=float(old.seconds) if old and old.seconds else float(default_seconds or 0),
                              join=(old.join if old else default_join) or default_join,
                              overlap=int(old.overlap) if old and old.overlap is not None else int(default_overlap),
                              idea=idea))
    return out


def plans_to_rows(plans: list[WindowPlan], commands_enabled: bool) -> list[list]:
    return [[p.index, float(p.seconds or 0), "—" if p.index == 1 else p.join,
             "—" if p.index == 1 or p.join == "cut" else int(p.overlap),
             window_command(p, commands_enabled) or "—",
             (p.idea[:60] + "…") if len(p.idea) > 60 else p.idea] for p in plans]


def table_rows(table) -> list:
    """Rows out of a Gradio Dataframe value, which may be a DataFrame, a dict or a plain list."""
    if table is None:
        return []
    if isinstance(table, dict):
        return list(table.get("data") or [])
    values = getattr(table, "values", None)
    if values is not None and hasattr(values, "tolist"):
        return values.tolist()
    try:
        return list(table)
    except TypeError:
        return []


def rows_to_plans(rows, plans: list[WindowPlan]) -> list[WindowPlan]:
    """Read the editable seconds / join / overlap columns back out of the table."""
    out = []
    values = table_rows(rows)
    for index, plan in enumerate(plans, 1):
        row = values[index - 1] if index <= len(values) else None
        if row:
            try:
                plan.seconds = float(row[1] or 0)
            except (TypeError, ValueError):
                pass
            join = _s(row[2]).lower()
            if join in ("continue", "cut"):
                plan.join = join
            try:
                overlap = int(float(row[3]))
                plan.overlap = max(0, overlap)
            except (TypeError, ValueError):
                pass
        out.append(plan)
    return out


# ------------------------------------------------------------- cast library
class CastLibrary:
    """Characters saved as named JSON, compatible with the Prompt Builder's subjects."""

    def __init__(self, root: str):
        self.dir = os.path.join(root, "cast")
        os.makedirs(self.dir, exist_ok=True)

    def names(self) -> list[str]:
        return sorted(os.path.splitext(f)[0] for f in os.listdir(self.dir) if f.lower().endswith(".json"))

    def save(self, character: Character, name: str = "") -> str:
        name = re.sub(r"[^A-Za-z0-9 ._-]+", "_", _s(name) or character.name).strip() or "character"
        data = {k: v for k, v in character.to_dict().items() if k in FIELD_NAMES}
        with open(os.path.join(self.dir, name + ".json"), "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2)
        return name

    def load(self, name: str) -> Character:
        with open(os.path.join(self.dir, name + ".json"), "r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict):
            raise ValueError(f"{name}.json does not hold a single saved character.")
        return Character.from_dict({k: v for k, v in data.items() if k in FIELD_NAMES})

    def import_file(self, path: str) -> str:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict) or not any(k in data for k in FIELD_NAMES):
            raise ValueError(f"{os.path.basename(path)} is not a saved character (a Prompt Builder subject JSON works here).")
        character = Character.from_dict(data)
        return self.save(character, os.path.splitext(os.path.basename(path))[0])
