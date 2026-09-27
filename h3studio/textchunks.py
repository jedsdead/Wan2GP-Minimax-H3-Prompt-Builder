"""Split knowledge documents into addressable sections.

Two heading styles are recognised:

* Markdown headings (``#`` .. ``######``), ignored inside fenced code blocks.
* Banner headings, as used by the weapon-combat guide::

      ==========
      VI. WEAPON-SPECIFIC CHOREOGRAPHY
      ==========

A document without headings becomes paragraph-group chunks so that plain
notes dropped into ``knowledge/`` are still retrievable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

MD_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
BANNER_RULE = re.compile(r"^\s*={5,}\s*$")
FENCE = re.compile(r"^\s*(```|~~~)")
FRONT_MATTER = re.compile(r"\A---\s*\n.*?\n---\s*\n", re.S)

MAX_CHUNK_CHARS = 6000
PLAIN_CHUNK_CHARS = 1800


@dataclass
class Section:
    doc: str                      # document key, e.g. "h3-prompt-writing/references/base-en.txt"
    index: int                    # position in the document
    level: int                    # 0 = preamble, 1..6 heading depth
    title: str
    heading_line: str             # heading exactly as written (empty for preamble)
    body: str                     # text under the heading, excluding child sections
    parent: int | None = None
    children: list[int] = field(default_factory=list)
    part: int = 0                 # >0 when an oversized section was split

    @property
    def text(self) -> str:
        head = self.heading_line
        if self.part > 1:
            head = f"{head} (continued)" if head else ""
        return f"{head}\n{self.body}".strip() if head else self.body.strip()

    @property
    def chars(self) -> int:
        return len(self.text)


def strip_front_matter(text: str) -> tuple[str, str]:
    match = FRONT_MATTER.match(text)
    if not match:
        return "", text
    return match.group(0), text[match.end():]


def parse_sections(doc: str, text: str) -> list[Section]:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = text.split("\n")
    headings: list[tuple[int, int, str, str, int]] = []  # (line_no, level, title, heading_line, span)
    in_fence = False
    i = 0
    while i < len(lines):
        line = lines[i]
        if FENCE.match(line):
            in_fence = not in_fence
            i += 1
            continue
        if not in_fence:
            md = MD_HEADING.match(line)
            if md:
                headings.append((i, len(md.group(1)), md.group(2).strip(), line, 1))
                i += 1
                continue
            if (BANNER_RULE.match(line) and i + 2 < len(lines) and lines[i + 1].strip()
                    and BANNER_RULE.match(lines[i + 2]) and not BANNER_RULE.match(lines[i + 1])):
                title = lines[i + 1].strip()
                headings.append((i, 1, title, title, 3))
                i += 3
                continue
        i += 1

    if not headings:
        return _plain_chunks(doc, text)

    sections: list[Section] = []
    first_line = headings[0][0]
    preamble = "\n".join(lines[:first_line]).strip()
    if preamble:
        sections.append(Section(doc, 0, 0, "(introduction)", "", preamble))

    stack: list[int] = []
    for n, (line_no, level, title, heading_line, span) in enumerate(headings):
        end = headings[n + 1][0] if n + 1 < len(headings) else len(lines)
        body = "\n".join(lines[line_no + span:end]).strip("\n")
        while stack and sections[stack[-1]].level >= level:
            stack.pop()
        parent = stack[-1] if stack else None
        section = Section(doc, len(sections), level, title, heading_line, body, parent)
        sections.append(section)
        if parent is not None:
            sections[parent].children.append(section.index)
        stack.append(section.index)

    return _split_oversized(sections)


def _plain_chunks(doc: str, text: str) -> list[Section]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks, current = [], []
    for paragraph in paragraphs:
        if current and sum(len(p) for p in current) + len(paragraph) > PLAIN_CHUNK_CHARS:
            chunks.append("\n\n".join(current))
            current = []
        current.append(paragraph)
    if current:
        chunks.append("\n\n".join(current))
    return [Section(doc, n, 1, f"part {n + 1}", "", body) for n, body in enumerate(chunks)]


def _split_oversized(sections: list[Section]) -> list[Section]:
    """Split very long leaf bodies at paragraph boundaries, keeping indices stable."""
    if not any(len(s.body) > MAX_CHUNK_CHARS for s in sections):
        return sections
    result: list[Section] = []
    remap: dict[int, int] = {}
    extra_children: dict[int, list[Section]] = {}
    for section in sections:
        remap[section.index] = len(result)
        if len(section.body) <= MAX_CHUNK_CHARS:
            result.append(section)
            continue
        paragraphs = re.split(r"(\n\s*\n)", section.body)
        parts, current = [], ""
        for piece in paragraphs:
            if current and len(current) + len(piece) > MAX_CHUNK_CHARS and piece.strip():
                parts.append(current.strip("\n"))
                current = ""
            current += piece
        if current.strip():
            parts.append(current.strip("\n"))
        section.body = parts[0]
        section.part = 1
        result.append(section)
        extra_children[section.index] = []
        for k, body in enumerate(parts[1:], start=2):
            sibling = Section(section.doc, -1, section.level, section.title, section.heading_line, body, section.index, part=k)
            result.append(sibling)
            extra_children[section.index].append(sibling)
    for new_index, section in enumerate(result):
        old_parent = section.parent
        section.parent = remap.get(old_parent, old_parent) if old_parent is not None else None
        section.index = new_index
    for section in result:
        section.children = [remap[c] for c in section.children if c in remap]
    for old_index, siblings in extra_children.items():
        owner = result[remap[old_index]]
        owner.children = [s.index for s in siblings] + owner.children
    return result


def selector_matches(title: str, selector: str) -> bool:
    """Match a section title against a selector such as ``4.2``, ``XVII`` or ``Points Most``.

    The selector must be a case-insensitive prefix ending on a boundary: ``4``
    matches ``4. How to Write`` but not ``4.1 Develop``; ``V`` matches
    ``V. WEAPON STABILITY`` but not ``VI. ...``.
    """
    t = title.strip().lower().strip("`")
    s = selector.strip().lower()
    if not s or not t.startswith(s):
        return False
    rest = t[len(s):]
    if not rest:
        return True
    if rest[0].isalnum():
        return False
    if rest[0] == "." and len(rest) > 1 and rest[1].isdigit():
        return False
    return True


def descendants(sections: list[Section], index: int) -> list[int]:
    out = [index]
    for child in sections[index].children:
        out.extend(descendants(sections, child))
    return sorted(set(out))
