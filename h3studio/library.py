"""Skill and knowledge library for the H3 prompt writer.

Layout inside the plugin folder::

    skills/<name>/SKILL.md              Claude-style skill (front matter optional)
    skills/<name>/references/...        reference documents
    skills/<name>/wan2gp_skill.json     optional manifest: which sections to use when
    knowledge/always/*.md|*.txt         always sent to the writer (house style, bible...)
    knowledge/**/*.md|*.txt             retrieved by relevance to each request

Nothing here needs torch or Gradio, so it can be unit-tested and used from
Deepy tools running on Deepy's worker thread.
"""

from __future__ import annotations

import json
import math
import os
import re
import threading
from collections import Counter
from dataclasses import dataclass, field

from .modes import BASE_MODES, Target, checklist, target_block
from .textchunks import Section, descendants, parse_sections, selector_matches, strip_front_matter

TEXT_EXTENSIONS = (".md", ".txt", ".markdown")
CHARS_PER_TOKEN = 3.8

TIER_REQUIRED = 1   # manifest "include" / untargeted SKILL.md body
TIER_ALWAYS = 2     # knowledge/always
TIER_OPTIONAL = 3   # manifest "optional", other references, user knowledge

STOPWORDS = set("""a an and are as at be but by for from has have he her his i if in into is it its of on or
our she so that the their them then there these they this to was we were what when which while who will with
you your shot shots video camera scene one two three""".split())


def estimate_tokens(text: str) -> int:
    return int(math.ceil(len(text) / CHARS_PER_TOKEN))


def tokenize(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9][a-z0-9'-]{2,}", (text or "").lower()) if w not in STOPWORDS]


@dataclass
class Document:
    key: str                  # "<source>/<relative path>"
    source: str               # skill name or "knowledge"
    rel_path: str
    sections: list[Section]
    always: bool = False


@dataclass
class Skill:
    name: str
    path: str
    description: str = ""
    manifest: dict = field(default_factory=dict)
    documents: dict[str, Document] = field(default_factory=dict)   # rel_path -> Document


@dataclass
class Pick:
    doc: Document
    section: Section
    tier: int
    order: float              # lower = earlier in the required list
    score: float = 0.0
    reason: str = ""


@dataclass
class Brief:
    text: str
    target: Target
    tokens: int
    included: list[str]
    dropped: list[str]
    warnings: list[str]


class Library:
    def __init__(self, root: str):
        self.root = root
        self.skills_dir = os.path.join(root, "skills")
        self.knowledge_dir = os.path.join(root, "knowledge")
        self.skills: dict[str, Skill] = {}
        self.knowledge: dict[str, Document] = {}
        self.load_errors: list[str] = []
        self._lock = threading.RLock()
        self.reload()

    # ------------------------------------------------------------------ loading
    def reload(self) -> None:
        with self._lock:
            self.skills, self.knowledge, self.load_errors = {}, {}, []
            os.makedirs(self.skills_dir, exist_ok=True)
            os.makedirs(os.path.join(self.knowledge_dir, "always"), exist_ok=True)
            for name in sorted(os.listdir(self.skills_dir)):
                path = os.path.join(self.skills_dir, name)
                if os.path.isdir(path) and os.path.isfile(os.path.join(path, "SKILL.md")):
                    try:
                        self.skills[name] = self._load_skill(name, path)
                    except Exception as exc:  # a broken skill must not break the others
                        self.load_errors.append(f"skill {name}: {exc}")
            for dirpath, _dirs, files in os.walk(self.knowledge_dir):
                for filename in sorted(files):
                    if not filename.lower().endswith(TEXT_EXTENSIONS) or filename.lower() == "readme.md":
                        continue
                    full = os.path.join(dirpath, filename)
                    rel = os.path.relpath(full, self.knowledge_dir).replace(os.sep, "/")
                    try:
                        doc = self._load_document("knowledge", rel, full)
                        doc.always = rel.startswith("always/")
                        self.knowledge[rel] = doc
                    except Exception as exc:
                        self.load_errors.append(f"knowledge {rel}: {exc}")

    def _load_skill(self, name: str, path: str) -> Skill:
        skill = Skill(name=name, path=path)
        manifest_path = os.path.join(path, "wan2gp_skill.json")
        if os.path.isfile(manifest_path):
            with open(manifest_path, "r", encoding="utf-8") as handle:
                skill.manifest = json.load(handle)
        for dirpath, _dirs, files in os.walk(path):
            for filename in sorted(files):
                if not filename.lower().endswith(TEXT_EXTENSIONS):
                    continue
                full = os.path.join(dirpath, filename)
                rel = os.path.relpath(full, path).replace(os.sep, "/")
                skill.documents[rel] = self._load_document(name, rel, full)
        front, _ = strip_front_matter(open(os.path.join(path, "SKILL.md"), encoding="utf-8").read())
        match = re.search(r"^description:\s*(.+)$", front, re.M)
        skill.description = match.group(1).strip() if match else ""
        return skill

    @staticmethod
    def _load_document(source: str, rel: str, full: str) -> Document:
        with open(full, "r", encoding="utf-8", errors="replace") as handle:
            text = handle.read()
        if rel.endswith("SKILL.md"):
            _front, text = strip_front_matter(text)
        key = f"{source}/{rel}"
        return Document(key, source, rel, parse_sections(key, text))

    # ------------------------------------------------------------ inventory
    def inventory(self, enabled_skills: list[str] | None = None) -> list[dict]:
        rows = []
        for name, skill in self.skills.items():
            active = enabled_skills is None or name in enabled_skills
            for rel, doc in skill.documents.items():
                chars = sum(s.chars for s in doc.sections)
                rows.append({"source": f"skill:{name}", "file": rel, "sections": len(doc.sections),
                             "tokens": estimate_tokens("x" * chars), "enabled": active,
                             "manifest": bool(skill.manifest)})
        for rel, doc in self.knowledge.items():
            chars = sum(s.chars for s in doc.sections)
            rows.append({"source": "knowledge", "file": rel, "sections": len(doc.sections),
                         "tokens": estimate_tokens("x" * chars), "enabled": True,
                         "manifest": False, "always": doc.always})
        return rows

    # ------------------------------------------------------------ selection
    def _resolve(self, skill: Skill, spec: str) -> list[tuple[Document, Section]]:
        rel, _, selector = spec.partition("#")
        doc = skill.documents.get(rel.strip())
        if doc is None:
            return []
        if not selector:
            return [(doc, s) for s in doc.sections]
        out = []
        for section in doc.sections:
            if section.part <= 1 and selector_matches(section.title, selector):
                out.extend((doc, doc.sections[i]) for i in descendants(doc.sections, section.index))
        return out

    @staticmethod
    def _condition_holds(when: dict, target: Target, request: str) -> bool:
        if not when:
            return True
        if "mode" in when:
            modes = when["mode"] if isinstance(when["mode"], list) else [when["mode"]]
            expanded = set()
            for m in modes:
                expanded.update(BASE_MODES if m == "base" else [m])
            if target.mode not in expanded:
                return False
        if "combat" in when and bool(when["combat"]) != target.combat:
            return False
        if "sliding_window" in when and bool(when["sliding_window"]) != target.sliding:
            return False
        if "continuation" in when and bool(when["continuation"]) != target.continuation:
            return False
        if "has_image" in when and bool(when["has_image"]) != target.has_image:
            return False
        if "keywords" in when:
            words = when["keywords"] if isinstance(when["keywords"], list) else [when["keywords"]]
            low = (request or "").lower()
            if not any(re.search(r"\b" + re.escape(w.lower()) + r"\b", low) for w in words):
                return False
        return True

    def _skill_picks(self, skill: Skill, target: Target, request: str, purpose: str) -> tuple[list[Pick], list[Pick], set[str]]:
        """Return (required, optional, excluded section ids) for one skill."""
        required: list[Pick] = []
        optional: list[Pick] = []
        manifest = skill.manifest
        excluded: set[str] = set()
        exclude_key = "enhancer_exclude" if purpose == "enhancer" else "deepy_exclude"
        for spec in manifest.get("exclude", []) + manifest.get(exclude_key, []):
            excluded.update(self._sid(d, s) for d, s in self._resolve(skill, spec))

        if not manifest:
            # No manifest: SKILL.md is the always-on core, everything else is a retrieval pool.
            doc = skill.documents.get("SKILL.md")
            if doc:
                required += [Pick(doc, s, TIER_REQUIRED, s.index) for s in doc.sections]
            for rel, doc in skill.documents.items():
                if rel != "SKILL.md":
                    optional += [Pick(doc, s, TIER_OPTIONAL, 0) for s in doc.sections]
            return required, optional, excluded

        order = 0.0
        rules = [{"include": manifest.get("always", [])}] + manifest.get("rules", [])
        for rule in rules:
            if not self._condition_holds(rule.get("when", {}), target, request):
                continue
            for spec in rule.get("include", []):
                for doc, section in self._resolve(skill, spec):
                    required.append(Pick(doc, section, TIER_REQUIRED, order, reason=spec))
                    order += 0.001
                order = math.floor(order) + 1
            for spec in rule.get("optional", []):
                boost = float(rule.get("boost", 2.0))
                for doc, section in self._resolve(skill, spec):
                    optional.append(Pick(doc, section, TIER_OPTIONAL, 0, score=boost, reason=spec))
        pool = manifest.get("retrieval_pool", [])
        for spec in pool:
            for doc, section in self._resolve(skill, spec):
                optional.append(Pick(doc, section, TIER_OPTIONAL, 0, reason="pool"))
        return required, optional, excluded

    @staticmethod
    def _unit(pick: Pick) -> tuple[tuple[str, int], list[Section]]:
        """A section split into parts is one unit: all parts, in order, or none."""
        doc, section = pick.doc, pick.section
        root = doc.sections[section.parent] if section.part > 1 and section.parent is not None else section
        if root.part == 0:
            return (doc.key, root.index), [root]
        members = [root] + [doc.sections[c] for c in root.children if doc.sections[c].part > 1]
        return (doc.key, root.index), members

    @staticmethod
    def _sid(doc: Document, section: Section) -> str:
        return f"{doc.key}::{section.index}"

    def _bm25(self, query: str, picks: list[Pick]) -> None:
        q = tokenize(query)
        if not q or not picks:
            return
        docs = [Counter(tokenize(p.section.title + " " + p.section.body)) for p in picks]
        n = len(docs)
        avg = sum(sum(d.values()) for d in docs) / max(1, n)
        df = Counter(term for d in docs for term in set(d))
        for pick, counts in zip(picks, docs):
            length = sum(counts.values()) or 1
            score = 0.0
            for term in set(q):
                tf = counts.get(term, 0)
                if not tf:
                    continue
                idf = math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5))
                score += idf * tf * 2.2 / (tf + 1.2 * (0.25 + 0.75 * length / avg))
            title_hits = len(set(q) & set(tokenize(pick.section.title)))
            pick.score += score + 1.5 * title_hits

    def search(self, query: str, limit: int = 5, enabled_skills: list[str] | None = None) -> list[dict]:
        picks = []
        for name, skill in self.skills.items():
            if enabled_skills is not None and name not in enabled_skills:
                continue
            for doc in skill.documents.values():
                picks += [Pick(doc, s, TIER_OPTIONAL, 0) for s in doc.sections]
        for doc in self.knowledge.values():
            picks += [Pick(doc, s, TIER_OPTIONAL, 0) for s in doc.sections]
        self._bm25(query, picks)
        best = sorted((p for p in picks if p.score > 0), key=lambda p: -p.score)[:max(1, limit)]
        return [{"source": p.doc.key, "section": self._label(p), "score": round(p.score, 2), "text": p.section.text} for p in best]

    @staticmethod
    def _label(pick: Pick) -> str:
        return f"{pick.doc.key} § {pick.section.title}" + (f" (part {pick.section.part})" if pick.section.part > 1 else "")

    # --------------------------------------------------------------- briefs
    def build_brief(self, request: str, target: Target, *, budget_tokens: int, enabled_skills: list[str] | None = None,
                    contract: str = "", purpose: str = "enhancer", role: str | None = None) -> Brief:
        with self._lock:
            return self._build_brief(request, target, budget_tokens, enabled_skills, contract, purpose, role)

    def _build_brief(self, request, target, budget_tokens, enabled_skills, contract, purpose, role) -> Brief:
        warnings: list[str] = []
        required: list[Pick] = []
        optional: list[Pick] = []
        excluded: set[str] = set()
        active = [s for n, s in self.skills.items() if enabled_skills is None or n in enabled_skills]
        if not active:
            warnings.append("No skill is enabled; only the stock contract and knowledge documents are used.")
        for skill in active:
            r, o, x = self._skill_picks(skill, target, request, purpose)
            required += r
            optional += o
            excluded |= x
        always = [Pick(doc, s, TIER_ALWAYS, n) for n, doc in enumerate(d for d in self.knowledge.values() if d.always) for s in doc.sections]
        optional += [Pick(doc, s, TIER_OPTIONAL, 0) for doc in self.knowledge.values() if not doc.always for s in doc.sections]

        chosen: dict[str, Pick] = {}
        for pick in sorted(required, key=lambda p: p.order) + always:
            sid = self._sid(pick.doc, pick.section)
            if sid not in excluded and sid not in chosen:
                chosen[sid] = pick
        pool = [p for p in optional if self._sid(p.doc, p.section) not in excluded and self._sid(p.doc, p.section) not in chosen]
        self._bm25(request, pool)
        seen: set[str] = set()
        ranked = []
        for pick in sorted(pool, key=lambda p: -p.score):
            sid = self._sid(pick.doc, pick.section)
            if sid not in seen and pick.score > 0.5:
                seen.add(sid)
                ranked.append(pick)

        head = self._head(target, contract, role, purpose)
        tail = checklist(target)
        budget_chars = int(budget_tokens * CHARS_PER_TOKEN) - len(head) - len(tail) - 200
        if budget_chars < 0:
            warnings.append("Budget is smaller than the mode contract itself; raise the token budget.")
            budget_chars = 0

        used, dropped = 0, []
        final: list[Pick] = []
        placed: set[tuple[str, int]] = set()

        def place(pick: Pick, required_unit: bool) -> None:
            nonlocal used
            key, members = self._unit(pick)
            if key in placed:
                return
            placed.add(key)
            members = [m for m in members if self._sid(pick.doc, m) not in excluded]
            cost = sum(m.chars + 2 for m in members)
            if used + cost <= budget_chars:
                final.extend(Pick(pick.doc, m, pick.tier, pick.order, pick.score, pick.reason) for m in members)
                used += cost
            elif required_unit or pick.score >= 2.0:
                dropped.append(self._label(Pick(pick.doc, members[0], pick.tier, 0)))

        for pick in chosen.values():
            place(pick, True)
        if dropped:
            warnings.append(f"{len(dropped)} required section(s) did not fit the token budget: " + "; ".join(dropped))
        for pick in ranked:
            place(pick, False)

        body = self._render(final)
        included = [self._label(p) for p in final]
        text = f"{head}\n\n{body}\n\n{tail}".strip() if body else f"{head}\n\n{tail}"
        return Brief(text, target, estimate_tokens(text), included, dropped, warnings + self.load_errors)

    @staticmethod
    def _head(target: Target, contract: str, role: str | None, purpose: str) -> str:
        parts = []
        if role:
            parts.append(role.strip())
        if contract:
            parts.append(contract.strip())
        parts.append(target_block(target))
        parts.append(
            "REFERENCE KNOWLEDGE follows. It is authoritative on field names, section order, labels and timing "
            "notation, and overrides anything remembered. Mentions of files such as references/base-en.txt point to "
            "the excerpts below; if an excerpt is absent, apply the rule as stated. Never mention files or this brief "
            "in the output. When sources conflict: the user's explicit request wins, then the mode guide on format, "
            "then the specialist guide on content." if purpose == "enhancer" else
            "REFERENCE KNOWLEDGE for writing this H3 prompt follows. It is authoritative on notation and overrides memory. "
            "Mentions of files point to the excerpts below."
        )
        return "\n\n".join(parts)

    def _render(self, picks: list[Pick]) -> str:
        # Keep document order within each document; documents in first-use order.
        by_doc: dict[str, list[Pick]] = {}
        for pick in picks:
            by_doc.setdefault(pick.doc.key, []).append(pick)
        blocks = []
        for key, items in by_doc.items():
            items.sort(key=lambda p: p.section.index)
            doc = items[0].doc
            included_ids = {p.section.index for p in items}
            chunks = []
            for pick in items:
                crumb = self._breadcrumb(doc, pick.section, included_ids)
                chunks.append((crumb + "\n" if crumb else "") + pick.section.text)
            blocks.append(f"===== {key} =====\n" + "\n\n".join(chunks))
        return "\n\n".join(blocks)

    @staticmethod
    def _breadcrumb(doc: Document, section: Section, included_ids: set[int]) -> str:
        trail, parent = [], section.parent
        while parent is not None:
            if parent in included_ids:
                return ""
            trail.append(doc.sections[parent].title)
            parent = doc.sections[parent].parent
        return f"[in: {' > '.join(reversed(trail))}]" if trail and section.part <= 1 else ""

    # ------------------------------------------------------------- uploads
    def add_knowledge_file(self, src_path: str, always: bool = False) -> str:
        name = os.path.basename(src_path)
        stem, ext = os.path.splitext(name)
        if ext.lower() not in TEXT_EXTENSIONS:
            raise ValueError(f"{name}: only {', '.join(TEXT_EXTENSIONS)} files can be added as knowledge.")
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._") or "document"
        dest_dir = os.path.join(self.knowledge_dir, "always" if always else "")
        os.makedirs(dest_dir, exist_ok=True)
        dest = os.path.join(dest_dir, safe + ext.lower())
        with open(src_path, "rb") as fin, open(dest, "wb") as fout:
            fout.write(fin.read())
        self.reload()
        return os.path.relpath(dest, self.knowledge_dir).replace(os.sep, "/")

    def remove_knowledge_file(self, rel: str) -> None:
        full = os.path.normpath(os.path.join(self.knowledge_dir, rel))
        if not full.startswith(os.path.normpath(self.knowledge_dir) + os.sep):
            raise ValueError("Path is outside the knowledge folder.")
        if os.path.isfile(full):
            os.remove(full)
        self.reload()
