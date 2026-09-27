"""Deterministic checks for H3 prompts, following the skill's 'Points Most Often Got Wrong'."""

from __future__ import annotations

import re

from .modes import WINDOW_COMMAND, commands_force_new_shot

BASE_FIELDS = ("integrated_multimodal_description", "overall_soundscape", "non_diegetic_music")
REF_FIELDS = ("subject_definitions", "summary", "retention_analysis", "detailed_description", "overall_soundscape", "non_diegetic_music")
KNOWN_COMMANDS = ("duration", "overlap", "new_shot", "loras_mult")

I2VA_LINE = re.compile(r"^For the target video, at 0\.00 seconds into the target video, <Picture 1> \(from \[Shot 1\]\) is fully referenced\.\s*$")
FL2VA_LINE = re.compile(r"^How the reference pictures align with the target video — Picture 1 \(from Shot 1\) aligns with the 0\.00-second mark of the target video; Picture 2 \(from Shot (\d+)\) aligns with the (\d+\.\d{2})-second mark of the target video\.\s*$")
L2VA_LINE = re.compile(r"^How the reference pictures align with the target video — <Picture 1> \(from \[Shot (\d+)\]\) aligns with the (\d+\.\d{2})-second mark of the target video\.\s*$")
SHOT = re.compile(r"\[Shot (\d+)\]")
CUT_TIME = re.compile(r"^\s*At (\d{2}):(\d{2})\.(\d{3})\b")
ANY_TIMESTAMP_START = re.compile(r"^\s*At \d")
LABEL = re.compile(r"<(Subject|Picture|Video|Audio) (\d+)>")
RETENTION_LINE = re.compile(
    r"^<(Subject|Picture|Video|Audio) \d+>(?: \((appears in|guides|heard in)[^)]*\))?:\s*"
    r"(fully_preserved|partially_preserved|attribute_transfer|weak_reference|fully_copy|partially_copy|reference)\b")
UNDERDETERMINED = ("fade to black", "fades to black", "fade out", "fades out", "fade to white", "fades to white",
                   "cut to black", "empty frame", "frame empties", "swallowed", "dust settles", "mist closes",
                   "fog closes", "disappears into", "white flare fills", "screen goes")


def split_windows(text: str) -> list[str]:
    """Paragraph split exactly like Wan2GP's PW mode (comments dropped, blank line = new window)."""
    lines = [line.rstrip() for line in text.replace("\r\n", "\n").split("\n") if not line.strip().startswith("#")]
    windows, current = [], []
    for line in lines:
        if not line.strip():
            if current:
                windows.append("\n".join(current).strip())
                current = []
            continue
        current.append(line)
    if current:
        windows.append("\n".join(current).strip())
    return windows


def _field_spans(text: str, names: tuple[str, ...]) -> dict[str, tuple[int, int]]:
    hits = []
    for name in set(BASE_FIELDS + REF_FIELDS):
        for m in re.finditer(rf"(?m)^\s*{name}\s*:", text):
            hits.append((m.start(), m.end(), name))
    hits.sort()
    spans = {}
    for i, (start, end, name) in enumerate(hits):
        stop = hits[i + 1][0] if i + 1 < len(hits) else len(text)
        spans.setdefault(name, (end, stop))
    return spans


def _field_order(text: str) -> list[str]:
    found = []
    for name in set(BASE_FIELDS + REF_FIELDS):
        for m in re.finditer(rf"(?m)^\s*{name}\s*:", text):
            found.append((m.start(), name))
    return [name for _pos, name in sorted(found)]


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


def _jaccard(a: str, b: str) -> float:
    x, y = set(re.findall(r"[a-z]+", a.lower())), set(re.findall(r"[a-z]+", b.lower()))
    return len(x & y) / max(1, len(x | y))


def check(prompt: str, mode: str = "auto", duration: float | None = None, expected_windows: int = 0) -> dict:
    errors: list[str] = []
    warnings: list[str] = []
    text = (prompt or "").strip()
    if not text:
        return {"ok": False, "windows": 0, "modes": [], "errors": ["The prompt is empty."], "warnings": []}

    if expected_windows == 1:
        windows = [text]
    else:
        windows = split_windows(text)
        # A keyframe instruction line followed by a blank line belongs to the next paragraph.
        merged = []
        for chunk in windows:
            if merged and (I2VA_LINE.match(merged[-1]) or merged[-1].startswith("How the reference pictures align")) and "\n" not in merged[-1]:
                merged[-1] = merged[-1] + "\n" + chunk
            else:
                merged.append(chunk)
        if len(merged) != len(windows) and len(merged) > 1:
            warnings.append("A blank line follows the keyframe instruction line. In paragraph (sliding-window) mode Wan2GP "
                            "would split it into its own window; use a single line break there.")
        windows = merged
        fragments = 0
        joined = []
        respect_count = bool(expected_windows) and len(windows) == expected_windows
        for chunk in windows:
            # A window opens with its first field (or a command / keyframe line). A paragraph holding
            # only a later field is a section of the window above, split off by a stray blank line.
            starts_window = bool(re.match(r"(?ms)^\s*(\[\s*/|For the target video|How the reference pictures align"
                                          r"|integrated_multimodal_description\s*:|subject_definitions\s*:)", chunk))
            if joined and not starts_window and not respect_count:
                joined[-1] = joined[-1] + "\n" + chunk
                fragments += 1
            else:
                joined.append(chunk)
        if fragments:
            warnings.append(f"{fragments} paragraph(s) carry no field of their own, so they are read as part of the window above. "
                            "In paragraph mode Wan2GP would treat each as a separate window — remove the blank lines inside a window.")
        windows = joined
        if expected_windows and len(windows) != expected_windows:
            errors.append(f"Wan2GP paragraph mode would split this into {len(split_windows(text))} window(s), "
                          f"but {expected_windows} were expected. Remove blank lines inside windows and keep exactly one between windows.")
    if re.search(r"(?m)^\s*(-{3,}|\*{3,}|_{3,})\s*$", text):
        errors.append("Separator lines such as '---' are not allowed; use exactly one blank line between windows.")
    if re.search(r"(?m)^\s*```", text):
        errors.append("Remove Markdown code fences from the prompt.")
    if re.search(r"(?m)^\s*#{1,6}\s|\*\*[^*]+\*\*", text):
        warnings.append("Markdown headings or bold markers found; H3 fields are plain `name:` lines.")

    sliding = len(windows) > 1
    closings: list[str] = []
    modes: list[str] = []
    markers: dict[str, dict[str, list[int]]] = {}
    definitions: dict[int, str] = {}
    for w_index, window in enumerate(windows, 1):
        tag = f"Window {w_index}: " if sliding else ""
        body, commands = window, []
        lead = re.match(r"^\s*((?:\[\s*/[^\]]*\]\s*)+)", window)
        if lead:
            commands = WINDOW_COMMAND.findall(lead.group(1))
            body = window[lead.end():].strip()
        for command in WINDOW_COMMAND.findall(window):
            for part in command.strip("[] ").split(","):
                name = part.strip().lstrip("/").split("=")[0].strip()
                if name and name not in KNOWN_COMMANDS:
                    errors.append(f"{tag}unknown window command `/{name}` (Wan2GP rejects it). Known: " + ", ".join("/" + k for k in KNOWN_COMMANDS))
        if WINDOW_COMMAND.search(body):
            warnings.append(f"{tag}window commands should open the window, before any prompt text.")
        new_shot = commands_force_new_shot(commands)
        if new_shot and sliding and w_index > 1:
            warnings.append(f"{tag}[/new_shot] removes all visual carry-over (wardrobe, grade, geography). Use it only for a real scene change; "
                            "cuts inside a scene belong on the timeline as `[Shot N] At MM:SS.mmm, the camera cuts to ...`.")

        w_mode = mode
        first_line = body.split("\n", 1)[0].strip()
        if w_mode in ("auto", "", None):
            if re.search(r"(?m)^\s*subject_definitions\s*:", body):
                w_mode = "Ref2VA"
            elif I2VA_LINE.match(first_line):
                w_mode = "I2VA"
            elif FL2VA_LINE.match(first_line) or "Picture 2 (from Shot" in first_line:
                w_mode = "FL2VA"
            elif first_line.startswith("How the reference pictures align"):
                w_mode = "L2VA"
            else:
                w_mode = "T2VA"
        modes.append(w_mode)
        expected = REF_FIELDS if w_mode == "Ref2VA" else BASE_FIELDS
        order = _field_order(body)
        if not any(name in order for name in expected):
            shots_seen = len(SHOT.findall(body))
            errors.append(f"{tag}this window is not a usable {w_mode} prompt: none of its fields "
                          f"({', '.join(expected[:2])} ...) are present"
                          + (f", and `[Shot 1]` appears {SHOT.findall(body).count('1')} times, which usually means the writer "
                             "looped or was cut off" if SHOT.findall(body).count("1") > 1 else "")
                          + ". Generate this window again, or shorten the request.")
            modes[-1] = w_mode
            closings.append("")
            continue
        for name in expected:
            count = order.count(name)
            if count == 0:
                errors.append(f"{tag}missing field `{name}:`.")
            elif count > 1:
                errors.append(f"{tag}field `{name}:` appears {count} times.")
        foreign = [n for n in dict.fromkeys(order) if n not in expected]
        if foreign:
            errors.append(f"{tag}fields not used in {w_mode}: " + ", ".join(f"`{n}`" for n in foreign))
        present = [n for n in order if n in expected]
        if present != [n for n in expected if n in present]:
            errors.append(f"{tag}fields are out of order; expected " + ", ".join(expected) + ".")
        spans = _field_spans(body, expected)

        # Keyframe instruction line
        shot_numbers = [int(n) for n in SHOT.findall(body)]
        last_shot = max(shot_numbers) if shot_numbers else None
        if w_mode in ("I2VA", "FL2VA", "L2VA"):
            line_re = {"I2VA": I2VA_LINE, "FL2VA": FL2VA_LINE, "L2VA": L2VA_LINE}[w_mode]
            m = line_re.match(first_line)
            if not m:
                errors.append(f"{tag}the first line must be the exact {w_mode} alignment instruction from base-en 2.1.")
            elif w_mode in ("FL2VA", "L2VA"):
                shot_n, seconds = int(m.group(1)), float(m.group(2))
                if last_shot is not None and shot_n != last_shot:
                    errors.append(f"{tag}the alignment line names Shot {shot_n}, but the final shot is [Shot {last_shot}].")
                if duration and abs(seconds - duration) > 0.05:
                    errors.append(f"{tag}the alignment line says {seconds:.2f} s but the duration is {duration:.2f} s.")
        elif re.match(r"^(For the target video|How the reference pictures align)", first_line) and w_mode == "T2VA":
            errors.append(f"{tag}T2VA has no picture-alignment line.")

        # Shots and cut times
        main_field = "detailed_description" if w_mode == "Ref2VA" else "integrated_multimodal_description"
        main_text = body[slice(*spans[main_field])] if main_field in spans else body
        shots = list(SHOT.finditer(main_text))
        if not shots:
            errors.append(f"{tag}no `[Shot 1]` marker in {main_field}.")
        numbers = [int(s.group(1)) for s in shots]
        if numbers and numbers != list(range(1, len(numbers) + 1)):
            errors.append(f"{tag}shot numbers must run 1, 2, 3 ... in order; found {numbers}.")
        last_time = 0.0
        for s in shots:
            after = main_text[s.end():s.end() + 160]
            n = int(s.group(1))
            if n == 1:
                if ANY_TIMESTAMP_START.match(after):
                    errors.append(f"{tag}[Shot 1] must not carry a timestamp.")
                if sliding and w_index > 1 and not new_shot:
                    if not re.match(r"^\s*The camera (cuts to|continues to|continues)\b", after):
                        errors.append(f"{tag}[Shot 1] of a later window must open with 'The camera cuts to ...' or "
                                      "'The camera continues to ...'; a bare description morphs instead of cutting.")
                    elif re.match(r"^\s*The camera continues (the previous shot|as before)\b", after):
                        warnings.append(f"{tag}'continues' should name what continues (the move, framing or action).")
            else:
                m = CUT_TIME.match(after)
                if not m:
                    errors.append(f"{tag}[Shot {n}] must start with `At MM:SS.mmm, ...`.")
                    continue
                t = int(m.group(1)) * 60 + int(m.group(2)) + int(m.group(3)) / 1000
                if t <= last_time:
                    errors.append(f"{tag}cut time of [Shot {n}] ({t:.3f} s) is not after the previous one ({last_time:.3f} s).")
                if duration and t >= duration:
                    errors.append(f"{tag}[Shot {n}] cuts at {t:.3f} s, outside the {duration:g} s duration.")
                last_time = t

        # Dialogue
        opens, closes = body.count("<d>"), body.count("</d>")
        if opens != closes:
            errors.append(f"{tag}unbalanced <d> tags ({opens} open, {closes} close).")
        for m in re.finditer(r"<d>(.*?)</d>", body, re.S):
            if not re.match(r"^\s*\[[^\]]+\]\s*\S", m.group(1)):
                errors.append(f"{tag}dialogue `<d>{m.group(1)[:30]}...` must start with a language tag such as [English].")
        speakers = []
        for sid in re.findall(r"\(S(\d+)\)", body):
            if sid not in speakers:
                speakers.append(sid)
        speaker_numbers = [int(x) for x in speakers]
        if speaker_numbers != sorted(speaker_numbers):
            warnings.append(f"{tag}speaker IDs should rise with first vocal appearance; found order "
                            + ", ".join(f"(S{n})" for n in speaker_numbers) + ".")

        # Ref2VA sections
        if w_mode == "Ref2VA":
            if "retention_analysis" in spans:
                block = body[slice(*spans["retention_analysis"])]
                if re.search(r"\(S\d+\)", block):
                    errors.append(f"{tag}speaker IDs must not appear in retention_analysis.")
                for line in [l.strip() for l in block.strip().split("\n") if l.strip()]:
                    seen_marker = re.match(r"^(<(?:Subject|Picture|Video|Audio) \d+>)[^:]*:\s*([A-Za-z_]+)", line)
                    if seen_marker:
                        markers.setdefault(seen_marker.group(1), {}).setdefault(seen_marker.group(2), []).append(w_index)
                    if not RETENTION_LINE.match(line):
                        warnings.append(f"{tag}retention line not in the form `<Subject N> (appears in [Shot 1], ...): marker - ...`: {line[:70]}")
            if "retention_analysis" in spans:
                block = body[slice(*spans["retention_analysis"])]
                covered = {m.group(1) for m in re.finditer(r"(?m)^(<(?:Subject|Picture|Video|Audio) \d+>)", block.strip())}
                # Per ref-en 4, subjects and audio always get a line; a picture or video that only
                # sources a subject does not (see the guide's own complete example).
                used_labels = {m.group(0) for m in LABEL.finditer(body)
                               if m.group(1) in ("Subject", "Audio")}
                missing_lines = sorted(used_labels - covered)
                if missing_lines:
                    warnings.append(f"{tag}retention_analysis has no line for " + ", ".join(missing_lines)
                                    + "; the guide wants one line per reference label.")
            if "subject_definitions" in spans:
                defs = body[slice(*spans["subject_definitions"])]
                definitions[w_index] = re.sub(r"\s+", " ", defs).strip()
                defined = {m.group(0) for m in LABEL.finditer(defs)}
                used = {m.group(0) for m in LABEL.finditer(body)}
                missing = sorted(used - defined)
                if missing:
                    errors.append(f"{tag}labels used but never defined in subject_definitions: " + ", ".join(missing))
            if "summary" in spans and not re.match(r"^\s*\[[a-z +]+\]", body[slice(*spans["summary"])]):
                errors.append(f"{tag}summary must begin with the bracketed task type, e.g. [reference generation].")
            if main_field in spans:
                words = len(main_text.split())
                if words < 250:
                    warnings.append(f"{tag}detailed_description has {words} words; 350-500 is normal for generation tasks.")
        else:
            stray = sorted({m.group(0) for m in LABEL.finditer(body)} - {"<Picture 1>"} if w_mode in ("I2VA", "L2VA") else
                           {m.group(0) for m in LABEL.finditer(body)} - ({"<Picture 1>", "<Picture 2>"} if w_mode == "FL2VA" else set()))
            if stray:
                errors.append(f"{tag}{w_mode} does not use these reference labels: " + ", ".join(stray))

        if "non_diegetic_music" in spans and not body[slice(*spans["non_diegetic_music"])].strip():
            errors.append(f"{tag}non_diegetic_music is empty; write N/A when there is no score.")

        # Ending
        if main_text and shots:
            last = main_text[shots[-1].start():]
            tail = " ".join(_sentences(last)[-2:]).lower()
            hits = [p for p in UNDERDETERMINED if p in tail]
            if hits:
                warnings.append(f"{tag}the window ends on an underdetermined frame ({', '.join(hits)}). Keep a specific subject in frame "
                                "to the last frame; move any dissolve mid-window.")
            sentences = _sentences(last)
            if sentences:
                closings.append(sentences[-1])

    for label, found in markers.items():
        if len(found) > 1:
            detail = "; ".join(f"{marker} in window{'s' if len(where) > 1 else ''} "
                               + ", ".join(str(w) for w in where) for marker, where in found.items())
            errors.append(f"{label} changes retention marker between windows ({detail}). One asset keeps one marker for the whole "
                          "sequence; only the shot list changes.")
    if len(definitions) > 1:
        first_index, first_text = sorted(definitions.items())[0]
        for index, text_ in sorted(definitions.items())[1:]:
            if text_ != first_text:
                warnings.append(f"subject_definitions in window {index} is worded differently from window {first_index}; "
                                "repeat the same definitions verbatim so identities do not drift.")
                break

    for i in range(1, len(closings)):
        if closings[i] == closings[i - 1]:
            errors.append(f"Windows {i} and {i + 1} end on the same sentence; consecutive windows must close on different framings.")
        elif _jaccard(closings[i], closings[i - 1]) > 0.7:
            warnings.append(f"Windows {i} and {i + 1} close on very similar imagery; change size, angle or subject of one closing shot.")

    errors = list(dict.fromkeys(errors))
    warnings = list(dict.fromkeys(warnings))
    return {"ok": not errors, "windows": len(windows), "modes": modes, "errors": errors, "warnings": warnings}


def format_report(result: dict) -> str:
    head = ("✅ No errors" if result["ok"] else f"❌ {len(result['errors'])} error(s)") + \
        f" · {len(result['warnings'])} warning(s) · {result['windows']} window(s)" + \
        (f" · modes: {', '.join(result['modes'])}" if result.get("modes") else "")
    lines = [head]
    lines += [f"- ❌ {e}" for e in result["errors"]]
    lines += [f"- ⚠️ {w}" for w in result["warnings"]]
    return "\n".join(lines)
