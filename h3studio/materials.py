"""User-supplied materials for one prompt: images, videos and audio with roles.

The role decides how the writer may use an item:

* ``first_frame`` / ``last_frame``  timeline anchors (keyframes)
* ``reference``                     attached to the H3 generation, cited as <Picture N>/<Video N>/<Audio N>
* ``describe``                      NOT attached to the generation. The writer sees only the text
                                    description and must describe the content in words, never cite a label.

Label numbering follows Wan2GP's Ref2VA attachment order: Start Image, End Image,
then Reference Images for <Picture N>; Control/Reference Video 1 and 2 for
<Video N>; Audio Reference 1 and 2 for <Audio N>. Numbering is per type.
"""

from __future__ import annotations

import os
import uuid
from dataclasses import asdict, dataclass

KINDS = ("image", "video", "audio")

ROLE_CHOICES = [
    ("First frame (keyframe at 0 s)", "first_frame"),
    ("Last frame (keyframe at the end)", "last_frame"),
    ("Reference attached to the generation", "reference"),
    ("Description only (not attached — helps the writer understand)", "describe"),
]
ROLE_LABELS = {value: label for label, value in ROLE_CHOICES}

IMAGE_EXT = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif")
VIDEO_EXT = (".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v")
AUDIO_EXT = (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac")


@dataclass
class Material:
    id: str
    kind: str                 # image | video | audio
    path: str
    role: str = "describe"
    note: str = ""            # what the user wants from it: "identity only", "camera motion", "voice timbre"...
    description: str = ""     # analysis result, editable
    source: str = "upload"    # upload | main form slot name
    order: int = 0            # attachment order within its main-form slot

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(data: dict) -> "Material":
        kw = {k: data[k] for k in Material.__dataclass_fields__ if k in data}
        kw.setdefault("id", uuid.uuid4().hex[:8])
        kw.setdefault("kind", "image")
        kw.setdefault("path", "")
        return Material(**kw)


def kind_of(path: str) -> str | None:
    ext = os.path.splitext(str(path or ""))[1].lower()
    if ext in IMAGE_EXT:
        return "image"
    if ext in VIDEO_EXT:
        return "video"
    if ext in AUDIO_EXT:
        return "audio"
    return None


def new_material(path: str, kind: str | None = None, role: str | None = None, source: str = "upload", order: int = 0) -> Material:
    kind = kind or kind_of(path) or "image"
    if role is None:
        role = "describe" if kind == "image" else "reference"
    return Material(uuid.uuid4().hex[:8], kind, path, role, source=source, order=order)


# Slot priority for Ref2VA picture numbering: anchors come before general references.
_SLOT_RANK = {"image_start": 0, "image_end": 1, "image_refs": 2, "upload": 3,
              "video_guide": 0, "video_guide2": 1, "video_guide3": 2,
              "audio_guide": 0, "audio_guide2": 1, "audio_guide3": 2}


def assign_labels(materials: list[Material], family: str) -> dict[str, str]:
    """Return material id -> H3 label for items the generation will actually receive."""
    labels: dict[str, str] = {}
    attached = [m for m in materials if m.role != "describe"]
    if family == "fl2va":
        first = next((m for m in attached if m.kind == "image" and m.role == "first_frame"), None)
        last = next((m for m in attached if m.kind == "image" and m.role == "last_frame"), None)
        if first:
            labels[first.id] = "<Picture 1>"
        if last:
            labels[last.id] = "<Picture 2>" if first else "<Picture 1>"
        return labels
    for kind, tag in (("image", "Picture"), ("video", "Video"), ("audio", "Audio")):
        items = [m for m in attached if m.kind == kind]
        role_rank = {"first_frame": 0, "last_frame": 1, "reference": 2}
        items.sort(key=lambda m: (_SLOT_RANK.get(m.source, 3), role_rank.get(m.role, 3), m.order))
        for n, m in enumerate(items, 1):
            labels[m.id] = f"<{tag} {n}>"
    return labels


def mode_for(materials: list[Material], family: str) -> str:
    if family == "ref2va":
        return "Ref2VA"
    roles = {m.role for m in materials if m.kind == "image"}
    first, last = "first_frame" in roles, "last_frame" in roles
    if first and last:
        return "FL2VA"
    if last:
        return "L2VA"
    if first:
        return "I2VA"
    return "T2VA"


def warnings_for(materials: list[Material], family: str) -> list[str]:
    out = []
    if family == "fl2va":
        for m in materials:
            if m.role == "reference":
                out.append(f"{os.path.basename(m.path)}: FL2VA models take no reference assets, so it is treated as description only.")
            if m.kind != "image" and m.role in ("first_frame", "last_frame"):
                out.append(f"{os.path.basename(m.path)}: only images can be keyframes.")
        if sum(m.role == "first_frame" for m in materials) > 1 or sum(m.role == "last_frame" for m in materials) > 1:
            out.append("More than one first or last frame is set; only the first of each is used.")
    else:
        pictures = [m for m in materials if m.kind == "image" and m.role != "describe"]
        videos = [m for m in materials if m.kind == "video" and m.role != "describe"]
        audios = [m for m in materials if m.kind == "audio" and m.role != "describe"]
        from .spec import REF_SLOT_COUNTS
        for items, kind, what in ((pictures, "Picture", "reference images"), (videos, "Video", "reference videos"),
                                  (audios, "Audio", "audio references")):
            limit = REF_SLOT_COUNTS[kind]
            if len(items) > limit:
                out.append(f"Ref2VA accepts at most {limit} {what}.")
        if any(m.role != "describe" and m.source == "upload" for m in materials):
            out.append("Some attached references were uploaded here, not imported from the main form. Attach them in the main "
                       "form in the order shown by their labels, or the labels will point at the wrong files.")
    missing = [os.path.basename(m.path) for m in materials if not m.description.strip() and m.kind != "audio"]
    if missing:
        out.append("Not analysed yet (the writer will know only the note): " + ", ".join(missing))
    return out


def materials_block(materials: list[Material], family: str) -> str:
    """Text appended to the user's request so any enhancer (text-only included) knows the assets."""
    if not materials:
        return ""
    labels = assign_labels(materials, family)
    attached, described = [], []
    for m in materials:
        label = labels.get(m.id)
        desc = m.description.strip() or "(no analysis available)"
        note = f" Intended use: {m.note.strip()}." if m.note.strip() else ""
        if label:
            role = {"first_frame": "first frame of the target video", "last_frame": "last frame of the target video",
                    "reference": f"reference {m.kind} attached to the generation"}[m.role]
            attached.append(f"- {label}: {role}.{note} Contents: {desc}")
        else:
            described.append(f"- A {m.kind} the user showed you for understanding only.{note} Contents: {desc}")
    parts = []
    if attached:
        parts.append("ATTACHED ASSETS (cite them with exactly these labels; describe their visible content consistently with these notes):\n" + "\n".join(attached))
    if described:
        parts.append("DESCRIPTION-ONLY MATERIAL (NOT attached to the generation. Never give it a <Picture>/<Video>/<Audio> label and "
                     "never add a subject_definitions or retention entry for it. Use it only to describe appearance, place or sound in words):\n"
                     + "\n".join(described))
    return "\n\n".join(parts)
