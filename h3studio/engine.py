"""Everything that touches wgp.py lives here.

wgp.py globals are read from the running main module at call time rather than
cached, because Wan2GP rebinds some of them (models_def after a finetune
refresh, for example).
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import traceback
import uuid

from . import lint
from .library import Library
from .materials import Material, assign_labels, materials_block, mode_for, new_material
from .spec import window_command as spec_window_command
from .modes import (Target, command_duration, commands_force_new_shot, detect_combat, mode_from_image_prompt_type,
                    mode_from_labels, window_commands)

PROFILE = "8"                      # enhancer profile digit used for H3 Prompt Generator modes (T8 / TI8)
STUDIO_MARK = "_minimax_h3_promptgen_briefs"   # model_def key carrying precomputed per-prompt system prompts
DEEPY_MARK = "[Minimax H3 Prompt Generator]"

DEFAULT_SETTINGS = {
    "enabled_skills": None,          # None = every skill found in skills/
    "enhancer_budget_tokens": 10000,
    "deepy_budget_tokens": 6000,
    "max_new_tokens_base": 2048,
    "max_new_tokens_ref": 3072,
    "analysis_max_tokens": 450,
    "combat": "auto",
    "main_dropdown": True,
    "use_stock_contract": True,
    "console_lint": True,
    "video_analysis_frames": 3,
}

ANALYSIS_SYSTEM_PROMPT = """You analyse reference media for a video prompt writer who cannot see it. Describe only what is visible, in plain English prose, 80-170 words, no Markdown, no lists, no preamble.
For people: apparent age range, build, face shape and features, skin tone, hair (colour, length, style), clothing layer by layer with colours, materials and fit, footwear, accessories, distinctive marks, pose and expression. Never guess names or identities.
For creatures and objects: shape, size cues, colours, materials, surface wear, distinctive parts.
Always add: setting, lighting direction and quality, colour palette, visual style (photograph, 3D render, anime, painting), and camera framing and angle.
Follow the role and note given in the user message: a first or last frame needs composition, positions and camera angle; a character reference needs appearance detail; a video frame needs the action in progress and the camera position."""


def _main():
    for name in ("__main__", "wgp"):
        module = sys.modules.get(name)
        if module is not None and hasattr(module, "process_prompt_enhancer"):
            return module
    return None


def wgp(name, default=None):
    module = _main()
    return getattr(module, name, default) if module is not None else default


def family_of(model_def: dict | None) -> str | None:
    arch = str((model_def or {}).get("architecture", ""))
    if arch.startswith("minimax_h3_ref2va"):
        return "ref2va"
    if arch.startswith("minimax_h3_fl2va"):
        return "fl2va"
    return None   # includes minimax_h3_tts_ref2va_pruned (audio only) and viggle


FIELD_START = r"^(summary|retention_analysis|detailed_description|overall_soundscape|non_diegetic_music|integrated_multimodal_description):"


def apply_canonical_definitions(text: str, canonical: str) -> str:
    """Replace the model's subject_definitions with the cast's own wording, so it cannot drift."""
    pattern = re.compile(r"(?ms)^subject_definitions:[ \t]*\n?.*?(?=" + FIELD_START + ")")
    if not canonical.strip() or not pattern.search(text or ""):
        return text
    return pattern.sub(lambda _m: "subject_definitions:\n" + canonical.strip() + "\n", text, count=1)


RETENTION_LINE = re.compile(r"^(<(?:Subject|Picture|Video|Audio) \d+>)\s*(\([^)]*\))?\s*:\s*([A-Za-z_]+)\s*(?:-\s*(.*))?$")


def _shots_for_label(body: str, label: str) -> str:
    """Which shots of this window mention the label, for the `(appears in ...)` part."""
    shots, current = [], None
    for piece in re.split(r"(\[Shot \d+\])", body):
        if re.fullmatch(r"\[Shot \d+\]", piece):
            current = piece
        elif current and label in piece and current not in shots:
            shots.append(current)
    return ", ".join(shots) or "[Shot 1]"


def apply_canonical_retention(text: str, fixed: dict) -> str:
    """Pin each label's retention marker across windows, keeping this window's shot list."""
    if not fixed:
        return text
    pattern = re.compile(r"(?ms)^retention_analysis:[ \t]*\n?(.*?)(?=" + FIELD_START + ")")
    match = pattern.search(text or "")
    if not match:
        return text
    body = text[match.end():]
    seen, lines = set(), []
    for line in match.group(1).strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        hit = RETENTION_LINE.match(line)
        if hit and hit.group(1) in fixed:
            label = hit.group(1)
            marker, note = fixed[label]
            if label.startswith("<Audio"):
                where = hit.group(2) or ""
            else:
                where = hit.group(2) or f"(appears in {_shots_for_label(body, label)})"
            lines.append(f"{label} {where}".rstrip() + f": {marker}" + (f" - {note or (hit.group(4) or '').strip()}".rstrip() if (note or hit.group(4)) else ""))
            seen.add(label)
        else:
            lines.append(line)
            if hit:
                seen.add(hit.group(1))
    for label, (marker, note) in fixed.items():
        if label not in seen and label in (text or ""):
            head = label if label.startswith("<Audio") else f"{label} (appears in {_shots_for_label(body, label)})"
            lines.append(f"{head}: {marker}" + (f" - {note}" if note else ""))
    return pattern.sub(lambda _m: "retention_analysis:\n" + "\n".join(lines) + "\n", text, count=1)


STRUCTURAL = ("missing field", "not a usable", "fields are out of order", "fields not used", "no `[Shot 1]`")


def needs_repair(report: dict) -> list[str]:
    """Errors worth a second attempt: the window came back structurally unusable."""
    return [e for e in report.get("errors", []) if any(marker in e for marker in STRUCTURAL)]


def repair_note(problems: list[str]) -> str:
    return ("\n\nSECOND ATTEMPT. The previous answer was rejected because: " + " ".join(problems[:3])
            + " Write the complete prompt again, every required field present, in order, each on its own line, "
              "nothing repeated, no commentary.")


def collapse_blank_lines(text: str) -> str:
    return re.sub(r"\n\s*\n+", "\n", text or "").strip()


def clean_output(text: str) -> str:
    text = re.sub(r"(?s)<think>.*?</think>", "", str(text or ""))
    text = re.sub(r"(?m)^\s*```[a-zA-Z]*\s*$", "", text)
    text = re.sub(r"^\s*(Here is|Here's)[^\n]*:\s*\n", "", text, flags=re.I)
    return text.strip()


class Studio:
    """Shared state for the tab, the enhancer hooks and the Deepy tools."""

    def __init__(self, root: str):
        self.root = root
        self.settings_path = os.path.join(root, "settings.json")
        self.cache_dir = os.path.join(root, "cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        self.settings = dict(DEFAULT_SETTINGS)
        self.load_settings()
        self.library = Library(root)
        self.last_run: dict = {}
        self._originals: dict = {}

    # ------------------------------------------------------------ settings
    def load_settings(self) -> None:
        try:
            with open(self.settings_path, "r", encoding="utf-8") as handle:
                self.settings.update(json.load(handle))
        except FileNotFoundError:
            pass
        except Exception as exc:
            print(f"[Minimax H3 Prompt Generator] settings.json unreadable, using defaults: {exc}")

    def save_settings(self) -> None:
        with open(self.settings_path, "w", encoding="utf-8") as handle:
            json.dump(self.settings, handle, indent=2)

    @property
    def enabled_skills(self):
        return self.settings.get("enabled_skills")

    # --------------------------------------------------------- introspection
    def current(self, state) -> dict:
        info = {"model_type": "", "model_def": {}, "family": None, "name": "", "settings": {}}
        try:
            model_type = wgp("get_state_model_type")(state)
            model_def = wgp("get_model_def")(model_type) or {}
            settings = wgp("get_model_settings")(state, model_type) or {}
            info.update(model_type=model_type, model_def=model_def, family=family_of(model_def),
                        name=model_def.get("name", model_type), settings=settings)
        except Exception as exc:
            info["error"] = str(exc)
        return info

    def enhancer_description(self) -> str:
        server_config = wgp("server_config", {}) or {}
        try:
            from shared.remote_llm import is_remote_engine, resolve_role_engine
            engine = resolve_role_engine(server_config, "prompt_enhancer")
            if is_remote_engine(engine):
                return f"remote engine `{engine}`"
        except Exception:
            pass
        code = server_config.get("enhancer_enabled", 0)
        if not code:
            return "none — enable a Prompt Enhancer in Configuration"
        quant = server_config.get("prompt_enhancer_quantization", "")
        vision = "vision-capable" if code in (3, 4, 5) else "caption-based images"
        return f"local enhancer #{code}" + (f" ({quant})" if quant else "") + f", {vision}"

    def enhancer_available(self) -> bool:
        server_config = wgp("server_config", {}) or {}
        if server_config.get("enhancer_enabled", 0):
            return True
        try:
            from shared.remote_llm import is_remote_engine, resolve_role_engine
            return is_remote_engine(resolve_role_engine(server_config, "prompt_enhancer"))
        except Exception:
            return False

    def form_duration(self, info: dict, windows: int) -> float | None:
        s = info.get("settings") or {}
        try:
            fps = wgp("get_computed_fps")(s.get("force_fps", ""), info["model_def"].get("architecture", info["model_type"]),
                                         s.get("video_guide"), s.get("video_source"))
        except Exception:
            fps = 24
        try:
            length = int(s.get("video_length", 0))
            window = int(s.get("sliding_window_size", 0) or 0)
            frames = min(length, window) if (windows > 1 or length > window) and window else length
            return round(frames / float(fps), 2) if frames and fps else None
        except Exception:
            return None

    def contract(self, family: str) -> str:
        if not self.settings.get("use_stock_contract", True):
            return ""
        try:
            from models.minimax_h3 import prompt_enhancer as stock
            return (stock._REF2VA_SHARED_RULES if family == "ref2va" else stock._FL2VA_SHARED_RULES).strip()
        except Exception:
            return "Output only the finished H3 prompt, with no commentary, Markdown, or code fence."

    def max_tokens(self, family: str) -> int:
        return int(self.settings["max_new_tokens_ref" if family == "ref2va" else "max_new_tokens_base"])

    # ------------------------------------------------------------ targets
    def brief_for(self, request: str, target: Target, family: str, purpose: str = "enhancer"):
        budget = self.settings["deepy_budget_tokens" if purpose == "deepy" else "enhancer_budget_tokens"]
        return self.library.build_brief(request, target, budget_tokens=int(budget), enabled_skills=self.enabled_skills,
                                        contract=self.contract(family), purpose=purpose)

    def studio_targets(self, windows: list[str], materials: list[Material], family: str, duration: float | None,
                       mode_override: str = "Auto", combat: str | None = None, plans: list | None = None) -> list[Target]:
        n = len(windows)
        combat_setting = combat or self.settings.get("combat", "auto")
        full_text = "\n".join(windows) + "\n" + "\n".join(m.note + " " + m.description for m in materials)
        base_mode = mode_for(materials, family)
        has_first = base_mode in ("I2VA", "FL2VA")
        has_last = base_mode in ("L2VA", "FL2VA")
        targets = []
        for i, window in enumerate(windows, 1):
            text, commands = window_commands(window)
            plan = plans[i - 1] if plans and i <= len(plans) else None
            if mode_override and mode_override != "Auto":
                mode = mode_override
            elif family == "ref2va":
                mode = "Ref2VA"
            else:
                first, last = has_first and i == 1, has_last and i == n
                mode = "FL2VA" if first and last else "I2VA" if first else "L2VA" if last else "T2VA"
            w_duration = (float(plan.seconds) if plan and plan.seconds else None) or command_duration(commands) or duration
            new_shot = bool(plan.new_shot) if plan else commands_force_new_shot(commands)
            targets.append(Target(mode=mode, has_image=False, duration=w_duration, window_index=i, window_count=n,
                                  new_shot=new_shot,
                                  combat=detect_combat(text if combat_setting == "auto" else full_text, combat_setting) or
                                  (combat_setting == "auto" and detect_combat(full_text))))
        return targets

    # ----------------------------------------------------- enhancer runs
    def _run_enhancer(self, state, prompts: list[str], briefs: dict[str, str], images: list | None, max_tokens: int,
                      progress, think: bool) -> list[str]:
        if not self.enhancer_available():
            raise RuntimeError("No prompt enhancer is configured. Enable one in Configuration (local model or remote LLM).")
        model_type = wgp("get_state_model_type")(state)
        model_def = dict(wgp("get_model_def")(model_type) or {})
        model_def[STUDIO_MARK] = briefs
        for prefix in ("text", "video", "image"):
            model_def[f"{prefix}_prompt_enhancer_max_tokens{PROFILE}"] = int(max_tokens)
        mode = ("TI" if images else "T") + PROFILE + ("K" if think else "")
        EnhancementProgress = wgp("EnhancementProgress")
        if EnhancementProgress is None:
            from shared.prompt_enhancer.progress import EnhancementProgress
        kwargs = {"image_prompt_type": "S" if images else "", "video_prompt_type": "", "audio_prompt_type": "",
                  "generation_callbacks": {"stop_requested": lambda: progress.gen["abort"],
                                           "enhancement_progress": EnhancementProgress(progress, progress.stream_tokens)}}
        image_start = images if images else [None] * len(prompts)
        result = wgp("exec_prompt_enhancer_engine")(state, model_type, model_def, mode, prompts, image_start, None,
                                                    False, False, -1, progress, -1, enhancer_kwargs=kwargs)
        flat = []
        for item in result or []:
            flat.extend(item if isinstance(item, list) else [item])
        return [clean_output(x) for x in flat]

    def analyse(self, state, materials: list[Material], ids: list[str] | None, progress, think: bool = False) -> list[Material]:
        from PIL import Image
        jobs = []   # (material, prompt, image, time)
        for m in materials:
            if ids and m.id not in ids:
                continue
            role = {"first_frame": "first frame of the target video", "last_frame": "last frame of the target video",
                    "reference": f"reference {m.kind} attached to the generation",
                    "describe": "material shown only so the writer understands the subject"}[m.role]
            note = f" User note: {m.note.strip()}." if m.note.strip() else ""
            if m.kind == "image":
                jobs.append((m, f"[{m.id}] Role: {role}.{note} Describe this image.", Image.open(m.path).convert("RGB"), None))
            elif m.kind == "video":
                for t, frame in self._video_frames(m.path, int(self.settings.get("video_analysis_frames", 3))):
                    jobs.append((m, f"[{m.id}] Role: {role}.{note} This is the frame at {t:.1f} s of the video. Describe it.", frame, t))
        if not jobs:
            return materials
        prompts = [j[1] for j in jobs]
        outputs = self._run_enhancer(state, prompts, {p: ANALYSIS_SYSTEM_PROMPT for p in prompts}, [j[2] for j in jobs],
                                     int(self.settings.get("analysis_max_tokens", 450)), progress, think)
        grouped: dict[str, list[str]] = {}
        for (m, _p, _img, t), text in zip(jobs, outputs):
            grouped.setdefault(m.id, []).append(f"At {t:.1f} s: {text}" if t is not None else text)
        for m in materials:
            if m.id in grouped:
                m.description = "\n".join(grouped[m.id])
        return materials

    @staticmethod
    def _video_frames(path: str, count: int):
        import cv2
        from PIL import Image
        cap = cv2.VideoCapture(path)
        try:
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 24.0)
            if total <= 0:
                raise RuntimeError(f"Cannot read frames from {os.path.basename(path)}")
            positions = [int(total * (k + 0.5) / max(1, count)) for k in range(max(1, count))]
            for pos in positions:
                cap.set(cv2.CAP_PROP_POS_FRAMES, min(pos, total - 1))
                ok, frame = cap.read()
                if ok:
                    yield pos / fps, Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        finally:
            cap.release()

    def write(self, state, request: str, materials: list[Material], family: str, duration: float | None,
              mode_override: str, combat: str, progress, think: bool, spec_text: str = "", plans: list | None = None,
              commands_enabled: bool = True, force_definitions: str = "",
              force_retention: dict | None = None) -> tuple[list[str], list[Target], list]:
        if plans:
            windows = [p.idea for p in plans]
        else:
            windows = lint.split_windows(request) or [request.strip()]
        targets = self.studio_targets(windows, materials, family, duration, mode_override, combat, plans)
        block = materials_block(materials, family).replace("@", "(at)")
        spec_text = (spec_text or "").replace("@", "(at)")
        prompts, briefs, prefixes, brief_objs = [], {}, [], []
        for i, (window, target) in enumerate(zip(windows, targets), 1):
            text, commands = window_commands(window)
            plan = plans[i - 1] if plans and i <= len(plans) else None
            prefixes.append(spec_window_command(plan, commands_enabled) if plan else " ".join(commands))
            parts = []
            if len(windows) > 1:
                plan_text = "\n".join(f"{k}. {window_commands(w)[0]}" for k, w in enumerate(windows, 1)).replace("@", "(at)")
                parts.append(f"Write window {i} of {len(windows)}.\nWhole sequence (context only, write just window {i}):\n{plan_text}")
            parts.append(f"Request for this {'window' if len(windows) > 1 else 'video'}:\n{text}")
            if spec_text:
                parts.append(spec_text)
            if block:
                parts.append(block)
            prompt = "\n\n".join(parts)
            brief = self.brief_for(text + "\n" + spec_text + "\n" + block, target, family)
            prompts.append(prompt)
            briefs[prompt] = brief.text
            brief_objs.append(brief)
        expected_mode = "Ref2VA" if family == "ref2va" else "auto"
        outputs = self._run_enhancer(state, prompts, briefs, None, self.max_tokens(family), progress, think)
        outputs, repaired = self._repair(state, prompts, briefs, outputs, targets, expected_mode, progress, think,
                                         self.max_tokens(family))
        if family == "ref2va":
            if force_definitions:
                outputs = [apply_canonical_definitions(o, force_definitions) for o in outputs]
            if force_retention:
                outputs = [apply_canonical_retention(o, force_retention) for o in outputs]
        if len(windows) > 1:
            outputs = [collapse_blank_lines(o) for o in outputs]
        outputs = [f"{c}\n{o}" if c else o for c, o in zip(prefixes, outputs)]
        self.last_run = {"source": "H3 Prompt Generator tab", "time": time.strftime("%H:%M:%S"), "targets": targets,
                         "briefs": brief_objs, "outputs": outputs, "repaired": repaired}
        return outputs, targets, brief_objs

    def _repair(self, state, prompts, briefs, outputs, targets, expected_mode, progress, think, max_tokens):
        """Write any structurally unusable window once more, and keep the better attempt."""
        failed = []
        for index, text in enumerate(outputs):
            duration = targets[index].duration if index < len(targets) else None
            problems = needs_repair(lint.check(text, expected_mode, duration, 1))
            if problems:
                failed.append((index, problems))
        if not failed:
            return outputs, []
        retry_prompts, retry_briefs = [], {}
        for index, problems in failed:
            prompt = prompts[index]
            retry_prompts.append(prompt)
            retry_briefs[prompt] = briefs[prompt] + repair_note(problems)
        print(f"[Minimax H3 Prompt Generator] rewriting window(s) {', '.join(str(i + 1) for i, _p in failed)} "
              "— the first attempt was unusable.")
        try:
            again = self._run_enhancer(state, retry_prompts, retry_briefs, None, max_tokens, progress, think)
        except Exception as exc:
            print(f"[Minimax H3 Prompt Generator] the rewrite failed, keeping the first attempt: {exc}")
            return outputs, []
        repaired = []
        for (index, problems), candidate in zip(failed, again):
            duration = targets[index].duration if index < len(targets) else None
            if len(needs_repair(lint.check(candidate, expected_mode, duration, 1))) < len(problems):
                outputs[index] = candidate
                repaired.append(index + 1)
        return outputs, repaired

    # ------------------------------------------------------- main-form I/O
    def import_from_form(self, info: dict, existing: list[Material]) -> tuple[list[Material], list[str]]:
        s, family = info.get("settings") or {}, info.get("family")
        notes, found = [], []
        image_flags, video_flags, audio_flags = s.get("image_prompt_type", "") or "", s.get("video_prompt_type", "") or "", s.get("audio_prompt_type", "") or ""
        slots = []
        if "S" in image_flags:
            slots.append(("image_start", s.get("image_start"), "image", "first_frame"))
        if "E" in image_flags:
            slots.append(("image_end", s.get("image_end"), "image", "last_frame"))
        if "I" in video_flags:
            slots.append(("image_refs", s.get("image_refs"), "image", "reference" if family == "ref2va" else "describe"))
        if family == "ref2va":
            # Reference Video 1-3 ("+" enables the second, "*" the third).
            for key in ("video_guide", "video_guide2", "video_guide3"):
                slots.append((key, s.get(key), "video", "reference"))
        # Audio Reference 1-3, flagged A, B and D in audio_prompt_type.
        for key, flag in (("audio_guide", "A"), ("audio_guide2", "B"), ("audio_guide3", "D")):
            if flag in audio_flags or s.get(key):
                slots.append((key, s.get(key), "audio", "reference"))
        by_path = {m.path: m for m in existing}
        for slot, value, kind, role in slots:
            for order, item in enumerate(self._as_list(value)):
                path = self._to_path(item, kind)
                if not path:
                    continue
                m = by_path.get(path) or new_material(path, kind, role, source=slot, order=order)
                m.role, m.source, m.order = role, slot, order
                found.append(m)
        kept = [m for m in existing if m.source == "upload" and m not in found]
        if not found:
            notes.append("No images, videos or audio are active in the main form for this model.")
        return found + kept, notes

    @staticmethod
    def _as_list(value):
        if value is None:
            return []
        return list(value) if isinstance(value, (list, tuple)) and not (len(value) == 2 and isinstance(value[1], (str, type(None))) and not isinstance(value[0], (list, tuple))) else [value]

    def _to_path(self, item, kind: str) -> str | None:
        if isinstance(item, (list, tuple)):
            item = item[0] if item else None
        if isinstance(item, dict):
            item = item.get("path") or item.get("name")
        if item is None:
            return None
        if isinstance(item, str):
            return item if os.path.isfile(item) else None
        if kind == "image" and hasattr(item, "save"):
            path = os.path.join(self.cache_dir, f"form_{uuid.uuid4().hex[:10]}.png")
            item.save(path)
            return path
        return None

    def export_to_form(self, state, text: str, windows: int) -> str:
        info = self.current(state)
        settings = wgp("get_current_model_settings")(state)
        mode = settings.get("multi_prompts_gen_type", "")
        if windows > 1:
            settings["multi_prompts_gen_type"] = "PW"
        elif "P" in str(mode or ""):
            text = re.sub(r"\n\s*\n+", "\n", text)   # a blank line would start a second window/queue item
        settings["prompt"] = text
        return info.get("name", "")

    # --------------------------------------------------------------- hooks
    def install_hooks(self, set_global=None) -> bool:
        main = _main()
        if main is None:
            return False
        ppe, choices = getattr(main, "process_prompt_enhancer"), getattr(main, "get_prompt_enhancer_choices")
        if getattr(ppe, "_minimax_h3_promptgen", False):
            return True
        self._originals = {"ppe": ppe, "choices": choices}
        wrapped_ppe, wrapped_choices = self._make_ppe(ppe), self._make_choices(choices)
        for name, fn in (("process_prompt_enhancer", wrapped_ppe), ("get_prompt_enhancer_choices", wrapped_choices)):
            if set_global is not None:
                set_global(name, fn)
            if getattr(main, name) is not fn:
                setattr(main, name, fn)
        print("[Minimax H3 Prompt Generator] prompt enhancer hooks installed.")
        return True

    def _make_choices(self, original):
        studio = self

        def get_prompt_enhancer_choices(model_def, *args, **kwargs):
            family = family_of(model_def)
            if family and studio.settings.get("main_dropdown", True) and model_def.get("prompt_enhancer_def"):
                pe_def = dict(model_def["prompt_enhancer_def"])
                labels = dict(pe_def.get("labels", {}))
                kind = "Reference Prompt" if family == "ref2va" else "Prompt"
                labels[f"T{PROFILE}V"] = f"H3 Prompt Generator: Skill-Guided {kind} from Text"
                labels[f"TI{PROFILE}V"] = f"H3 Prompt Generator: Skill-Guided {kind} from Text + {{image_inputs}}"
                pe_def["labels"] = labels
                pe_def["selection"] = list(pe_def.get("selection", [])) + [f"T{PROFILE}", f"TI{PROFILE}"]
                model_def = {**model_def, "prompt_enhancer_def": pe_def}
            return original(model_def, *args, **kwargs)

        get_prompt_enhancer_choices._minimax_h3_promptgen = True
        return get_prompt_enhancer_choices

    def _make_ppe(self, original):
        studio = self

        def process_prompt_enhancer(model_type, model_def, prompt_enhancer, original_prompts, image_start, original_image_refs,
                                    is_image, audio_only, seed, prompt_enhancer_instructions=None, text_encoder_max_tokens=512,
                                    enhancer_kwargs=None, batch_images=False):
            args = (image_start, original_image_refs, is_image, audio_only, seed)
            extra = dict(prompt_enhancer_instructions=prompt_enhancer_instructions, text_encoder_max_tokens=text_encoder_max_tokens,
                         batch_images=batch_images)
            mode = str(prompt_enhancer or "")
            model_def = model_def or {}
            try:
                if STUDIO_MARK in model_def:          # precomputed by the tab
                    return studio._call_with_briefs(original, model_type, model_def, mode, original_prompts, args, extra, enhancer_kwargs)
                if PROFILE in mode and family_of(model_def) and not is_image and not audio_only:
                    return studio._main_tab_run(original, model_type, model_def, mode, original_prompts, args, extra, enhancer_kwargs)
            except (InterruptedError, KeyboardInterrupt):
                raise
            except Exception as exc:
                if type(exc).__name__ in ("DownloadCancelled", "LoadingCancelled"):
                    raise
                print("[Minimax H3 Prompt Generator] falling back to the stock H3 enhancer after an error:")
                traceback.print_exc()
                mode = mode.replace(PROFILE, "")
            return original(model_type, model_def, mode, original_prompts, *args, enhancer_kwargs=enhancer_kwargs, **extra)

        process_prompt_enhancer._minimax_h3_promptgen = True
        return process_prompt_enhancer

    @staticmethod
    def _with_instructions(model_def: dict, instructions: str, max_tokens: int | None = None) -> dict:
        md = {k: v for k, v in model_def.items() if k != STUDIO_MARK}
        for prefix in ("text", "video", "image"):
            md[f"{prefix}_prompt_enhancer_instructions{PROFILE}"] = instructions
            if max_tokens:
                md[f"{prefix}_prompt_enhancer_max_tokens{PROFILE}"] = int(max_tokens)
        return md

    def _call_with_briefs(self, original, model_type, model_def, mode, prompts, args, extra, kwargs):
        briefs = model_def[STUDIO_MARK]
        outputs = []
        for index, prompt in enumerate(prompts):
            instructions = briefs.get(prompt) or next(iter(briefs.values()))
            kw = self._slice_kwargs(kwargs, index, len(prompts))
            image_start = args[0]
            if isinstance(image_start, list) and len(prompts) > 1:
                image_start = image_start[index:index + 1] if index < len(image_start) else None
            outputs += original(model_type, self._with_instructions(model_def, instructions), mode, [prompt], image_start,
                                *args[1:], enhancer_kwargs=kw, **extra)
        return outputs

    @staticmethod
    def _slice_kwargs(kwargs, index, count):
        if kwargs is None or count <= 1:
            return kwargs
        try:
            from shared.prompt_enhancer import images as pe_images
            return pe_images.batch_kwargs(kwargs, index, False, count)
        except Exception:
            return kwargs

    def _main_tab_run(self, original, model_type, model_def, mode, prompts, args, extra, kwargs):
        family = family_of(model_def)
        kwargs = kwargs or {}
        images_visible = "I" in mode
        contexts = kwargs.get("image_contexts")
        n = len(prompts)
        split = n > 1 and (contexts is None or len(contexts) == n)
        max_tokens = self.max_tokens(family)
        outputs, briefs = [], []
        units = list(enumerate(prompts, 1)) if split or n == 1 else [(0, p) for p in prompts]
        for index, prompt in units:
            text, commands = window_commands(prompt)
            ctx = contexts[index - 1] if contexts is not None and index >= 1 and len(contexts) >= index else None
            if ctx is not None:
                w_mode = mode_from_labels(list(ctx.labels), family == "ref2va", images_visible)
                duration = ctx.duration_seconds
            else:
                w_mode = mode_from_image_prompt_type(kwargs.get("image_prompt_type"), family == "ref2va", images_visible)
                duration = kwargs.get("video_duration_seconds")
            if n > 1 and w_mode in ("I2VA", "FL2VA") and ctx is None and index > 1:
                w_mode = "T2VA"
            target = Target(mode=w_mode, has_image=images_visible, duration=command_duration(commands) or duration,
                            window_index=max(index, 0), window_count=n, new_shot=commands_force_new_shot(commands),
                            combat=detect_combat(text, self.settings.get("combat", "auto")))
            brief = self.brief_for(text, target, family)
            briefs.append(brief)
            md = self._with_instructions(model_def, brief.text, max_tokens)
            if index >= 1 and n > 1:
                kw = self._slice_kwargs(kwargs, index - 1, n)
                outputs += original(model_type, md, mode, [prompt], *args, enhancer_kwargs=kw, **extra)
            else:
                outputs += original(model_type, md, mode, [prompt] if n == 1 else prompts, *args, enhancer_kwargs=kwargs, **extra)
                if n > 1:
                    break
        outputs = [self._clean_keep_commands(o) for o in outputs]
        if len(outputs) > 1:
            # One paragraph per window: a blank line inside a window would make Wan2GP
            # split it into two windows at generation time.
            outputs = [collapse_blank_lines(o) for o in outputs]
        expected_mode = "Ref2VA" if family == "ref2va" else "auto"
        repaired = []
        for index, text in enumerate(outputs):
            brief = briefs[index] if index < len(briefs) else briefs[-1]
            problems = needs_repair(lint.check(text, expected_mode, brief.target.duration, 1))
            if not problems or index >= len(prompts):
                continue
            md = self._with_instructions(model_def, brief.text + repair_note(problems), max_tokens)
            kw = self._slice_kwargs(kwargs, index, n) if n > 1 else kwargs
            try:
                again = original(model_type, md, mode, [prompts[index]], *args, enhancer_kwargs=kw, **extra)
            except Exception as exc:
                print(f"[Minimax H3 Prompt Generator] rewrite of window {index + 1} failed, keeping the first attempt: {exc}")
                continue
            candidate = self._clean_keep_commands(again[0]) if again else ""
            if len(outputs) > 1:
                candidate = collapse_blank_lines(candidate)
            if candidate and len(needs_repair(lint.check(candidate, expected_mode, brief.target.duration, 1))) < len(problems):
                outputs[index] = candidate
                repaired.append(index + 1)
        if repaired:
            print(f"[Minimax H3 Prompt Generator] window(s) {', '.join(str(r) for r in repaired)} came back unusable and were "
                  "written again.")
        report = (lint.check("\n\n".join(outputs), expected_mode, expected_windows=len(outputs)) if len(outputs) > 1
                  else lint.check(outputs[0] if outputs else "", expected_mode, expected_windows=1))
        self.last_run = {"source": "main tab Write/auto enhancer", "time": time.strftime("%H:%M:%S"),
                         "targets": [b.target for b in briefs], "briefs": briefs, "outputs": outputs, "lint": report,
                         "repaired": repaired}
        if self.settings.get("console_lint", True) and (report["errors"] or report["warnings"]):
            print("[Minimax H3 Prompt Generator] check of the enhanced prompt:\n" + lint.format_report(report))
        return outputs

    @staticmethod
    def _clean_keep_commands(text: str) -> str:
        lead = re.match(r"^\s*((?:\[\s*/[^\]]*\]\s*)+)", text or "")
        head = lead.group(1).strip() if lead else ""
        body = clean_output(text[lead.end():] if lead else text)
        return f"{head}\n{body}" if head else body

    # ------------------------------------------------------ Deepy pointer
    def annotate_deepy_infos(self) -> int:
        models_def = wgp("models_def", {}) or {}
        count = 0
        for model_def in models_def.values():
            if not isinstance(model_def, dict) or not family_of(model_def):
                continue
            infos = str(model_def.get("deepy_prompt_infos", ""))
            if DEEPY_MARK not in infos:
                model_def["deepy_prompt_infos"] = (infos + f"\n{DEEPY_MARK} Before writing an H3 prompt, call minimax_h3_prompt_guide with the request, "
                                                   "mode, duration and window position, and follow its guidance. After writing, call "
                                                   "minimax_h3_check_prompt and fix every reported error.").strip()
                count += 1
        return count
