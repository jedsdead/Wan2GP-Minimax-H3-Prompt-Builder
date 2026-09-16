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
markers, task-type prefixes and N/A placeholders. It also writes the phrasing
the spec fixes rather than leaves open - the voiceover clause and its
closed-lips follow-up, <scenetrans> at both connecting points with the
continuity stated in words, <cutoff> for speech the clip ends in the middle
of, and on-screen text quoted verbatim.

AUDIO SUGGESTIONS
  The Audio section can hand the scene to WanGP's own Prompt Enhancer and
  have it write the soundscape and score. See the bridge below the imports;
  it borrows the enhancer WanGP already holds where it can, and loads one at
  the configured level where it cannot.

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

import gc
import json
import re
import sys
import time
from pathlib import Path

import gradio as gr

from shared.utils.plugins import WAN2GPPlugin


# =============================================================================
# Prompt Enhancer bridge
# =============================================================================
#
# The button borrows WanGP's Prompt Enhancer. Three ways to reach it, tried in
# order, because where the model lives depends on the WanGP version and on
# whether an enhancement has been run yet this session:
#
#   1. mmgp's offload pipe. When the enhancer is enabled, WanGP registers it
#      alongside the video model under "prompt_enhancer_llm_model" - this is
#      the "Hooked to model 'prompt_enhancer_llm_model'" line in the console.
#      Weights reached this way are already managed, so calling generate()
#      pulls them onto the GPU and the profile puts them back afterwards.
#   2. A PromptEnhancerRuntime sitting in a wgp.py global.
#   3. Loading it here, on demand, at the level set in the Configuration tab,
#      downloading the weights first if they are missing. This copy is ours:
#      it is not part of the offload profile, so it is moved to the GPU for
#      the call and released after unless ENHANCER_KEEP_LOADED is set.
#
# None of this is documented API. Every step is probed, nothing raises into
# the console, and a failure reports what it did and did not find so the
# problem is visible from the panel rather than only in a terminal.

# Generous, because the Qwen enhancers may reason before answering and a
# reply truncated mid-<think> leaves nothing to parse.
ENHANCER_MAX_NEW_TOKENS = 1024

# Keep a self-loaded enhancer resident between presses. Much faster on the
# second press, but it holds several GB the video model may want. A borrowed
# enhancer is unaffected either way - this only governs our own copy.
ENHANCER_KEEP_LOADED = False

# Which Prompt Enhancer a level selects. WanGP owns this mapping - it lives
# in get_qwen35_prompt_enhancer_variant() - so it is read from WanGP first
# and this table is only the fallback for a build that will not import.
#
# 1 (Llama 3.2) and 2 (Llama Joy) are captioners and will not reliably return
# the JSON these prompts ask for. Everything in the Qwen family will: 3 and 4
# are Qwen3.5 Abliterated 4B and 9B, 5 is Qwen3.8-27B Uncensored. Each has
# its own quantizations - Int8 or GGUF for Qwen3.5, GGUF Q4 or Q2 for
# Qwen3.8 - which change the weights, not the instruction-following, so the
# builder passes the configured backend through and does not care which.
ENHANCER_LEVEL_NAMES = {
    0: "Disabled",
    1: "Llama 3.2",
    2: "Llama Joy",
    3: "Qwen3.5-4B Abliterated",
    4: "Qwen3.5-9B Abliterated",
    5: "Qwen3.8-27B Uncensored",
}

# Captioners. Anything not listed here is assumed to be an instruct model
# and given the benefit of the doubt, so a level WanGP adds after this was
# written is tried rather than refused.
ENHANCER_CAPTIONING_LEVELS = (1, 2)

# Where the enhancer's language model is registered in the offload pipe.
ENHANCER_PIPE_KEY = "prompt_enhancer_llm_model"

# Used only when the first reply parses to nothing. Shorter, blunter, and it
# forgoes the reasoning that caused the truncation in the first place.
AUDIO_RETRY_PROMPT = """Read the scene below and answer with one JSON object and nothing else. No preamble, no explanation, no reasoning.

{"soundscape": "<one sentence of diegetic sound: ambience, surfaces, weather, footsteps, breath>", "music": "<one sentence naming instruments, tempo and rhythm for the score, or the single word none>"}

No dialogue. No camera. No mood words in the music."""

# Asking for one field at a time. The combined prompt has to hold two jobs in
# mind at once, and the weaker enhancers drift between them - a score note
# turning up in the soundscape, or the reverse. One field per call costs a
# second press and reliably gets an answer about the thing that was asked.
SOUNDSCAPE_ONLY_PROMPT = """You are a supervising sound editor. You will be given three blocks: SCENE, SOUNDSCAPE DIRECTION and SCORE DIRECTION.

Write one thing only: the soundscape - the diegetic sound bed, meaning everything a microphone standing in that scene would pick up: room tone, weather, surfaces, machinery, footsteps, cloth, breath, crowd. Base it on the SCENE block and on SOUNDSCAPE DIRECTION only. Where direction is already given, build on it and fill the gaps around it rather than restating it or contradicting it. Where none is given, work it out from the setting, the atmosphere and what happens.

Rules:
- Do not transcribe or invent dialogue. Spoken words are handled elsewhere.
- Say nothing about the score. Non-diegetic music is handled elsewhere, and the SCORE DIRECTION block is context only.
- Music the characters can hear on screen is handled elsewhere. Leave it out.
- Say nothing about camera, framing, lens, movement, colour, editing or performance.
- One or two sentences, present tense, no line breaks.
- Use only what the blocks support. Do not add locations, objects or events that are not in them.

Return only this JSON object and nothing else:
{"soundscape": "..."}"""

MUSIC_ONLY_PROMPT = """You are a film composer. You will be given three blocks: SCENE, SOUNDSCAPE DIRECTION and SCORE DIRECTION.

Write one thing only: the music - the non-diegetic score, meaning the instrumentation, speed, rhythm and changes in volume of music only the audience hears. Base it on the SCORE DIRECTION block, using SCENE for context. Where a style or a choice is already given, develop it into specific instrumentation and tempo. Where none is given, work out a score that suits the scene as described. If the scene plainly wants no score, write exactly: none

Rules:
- Do not transcribe or invent dialogue.
- Say nothing about diegetic sound - ambience, weather and footsteps are handled elsewhere.
- Music the characters can hear on screen is handled elsewhere. Leave it out.
- Name instruments, tempo, rhythm and dynamics. Do not use mood words and do not explain what the score does for the audience emotionally. Write "sparse piano at a slow tempo, joined by sustained low strings that swell and fade", not "a tense, melancholy piano theme".
- One or two sentences, present tense, no line breaks.

Return only this JSON object and nothing else:
{"music": "..."}"""

_ENH_FENCE_RE = re.compile(r"^```[a-zA-Z]*\s*|\s*```$")
_ENH_THINK_RE = re.compile(r"<think>.*?</think>", re.S)
_ENH_TAG_RE = re.compile(r"<[^<>\n]{0,60}>")
_ENH_LABEL_RE = re.compile(
    r'^\s*[-*"\']?\s*(overall_soundscape|non_diegetic_music|soundscape|music|score|ambience)'
    r'"?\s*[:\-]\s*(.+)$',
    re.I,
)

# Our own copy, if we ended up loading one.
_ENHANCER_OWNED = {"model": None, "tokenizer": None}

# -- the action field -------------------------------------------------------
#
# Shot markers are the spine of the field: numbering, camera continuation,
# retention scope and validation all read them back rather than trusting a
# counter, so deleting a shot by hand stays correct.
_ACTION_SHOT_RE = re.compile(r"\[Shot\s+(\d+)\]", re.I)
_ACTION_CAMERA_RE = re.compile(r"\bthe camera\b", re.I)

# Whether a shot already has a camera sentence in it. "The camera" alone is
# not enough: the opening shot takes no transition verb, so its camera
# sentence leads with the framing instead and never names the camera at all.
# Matched against the shot with its [Shot N] marker removed, or every tail
# would match on the word "Shot".
_ACTION_FRAMING_RE = re.compile(
    r"\b(?:shot|close-?up|cutaway|view|frame holds|the frame)\b", re.I)
_ACTION_SPEAKER_RE = re.compile(r"\((S\d+(?:\s*,\s*S\d+)*)\)")
_ACTION_TRAILING_ID_RE = re.compile(r"\s*\(S\d+(?:\s*,\s*S\d+)*\)\s*$")
# (S1) is how the buttons name a speaker, and {Subject 1} how 3.3.0's
# Insert subject named an entry. Both are resolved in one pass, so a first
# mention is a first mention whichever form it took. The brace form is no
# longer written - it is kept so drafts saved by 3.3.0 still build.
# A written label in the action, for checking it still points at a live
# definition. Any kind, since the cast names its own.
_ACTION_LABEL_RE = re.compile(r"<([A-Z][A-Za-z]*)\s+(\d+)>")
_ACTION_BIND_RE = re.compile(
    r"\((?P<ids>S\d+(?:\s*,\s*S\d+)*)\)"
    r"|\{\s*[Ss]ubject\s+(?P<num>\d+)\s*\}")
_ACTION_TRANS_RE = re.compile(r"<scenetrans>")
_ACTION_RECEIVE_RE = re.compile(r"<scenetrans>\s*The speech carries over", re.I)
_ACTION_TIME_RE = re.compile(r"\bAt\s+(\d{2}):(\d{2}\.\d{3})")

# A timestamp the buttons would never write, so it escapes both the ordering
# check and the duration check. "At 3s," and "At 00:03," are the common ones.
_ACTION_LOOSE_TIME_RE = re.compile(r"\bAt\s+(?!\d{2}:\d{2}\.\d{3})(\d[^,\n]{0,15}),")

# Dialogue tags, walked left to right rather than counted, so an interleaved
# pair is caught as well as an unbalanced one.
_ACTION_D_TAG_RE = re.compile(r"</?d>")
_ACTION_D_BLOCK_RE = re.compile(r"<d>(.*?)</d>", re.S)
_ACTION_D_LANG_RE = re.compile(r"^\s*\[[^\]\n]+\]")

# Continuity prose and the closed-lips clause, which the audio digest strips.
# Both describe how a cut or a mouth behaves rather than how anything sounds,
# and neither is a sentence of its own - they hang off a tag or a </d>.
_ACTION_CARRY_SENTENCE_RE = re.compile(r"<scenetrans>\s*The speech[^.]*\.", re.I)
_ACTION_LIPS_RE = re.compile(
    r"\s*while (?:his|her|their|its) lips remain[^.]*\.", re.I)

# The receiving half a Shot press writes, used when the action is cleared
# for a new sliding window and a carried line was still open.
CARRY_RECEIVE_TEXT = "The speech carries over from the previous shot."


class EnhancerUnavailable(Exception):
    """Raised with a sentence fit to show the user."""


def _wgp_module():
    """The running wgp.py, whatever name it ended up imported under."""
    for name in ("wgp", "__main__"):
        mod = sys.modules.get(name)
        if mod is not None and hasattr(mod, "server_config"):
            return mod
    for mod in list(sys.modules.values()):
        try:
            if getattr(mod, "__name__", "").rsplit(".", 1)[-1] == "wgp" \
                    and hasattr(mod, "server_config"):
                return mod
        except Exception:
            continue
    return None


def _server_config():
    mod = _wgp_module()
    cfg = getattr(mod, "server_config", None) if mod is not None else None
    return cfg if isinstance(cfg, dict) else {}


def _config_int(keys, default=None):
    cfg = _server_config()
    for key in keys:
        if key in cfg:
            try:
                return int(cfg[key])
            except (TypeError, ValueError):
                pass
    return default


def _config_str(keys, default=""):
    cfg = _server_config()
    for key in keys:
        value = cfg.get(key)
        if isinstance(value, str) and value:
            return value
    return default


def _enhancer_level():
    """The configured Prompt Enhancer level, or None if it can't be read."""
    return _config_int(("enhancer_enabled", "prompt_enhancer", "enhancer",
                        "enhancer_mode", "prompt_enhancer_enabled"))


def _enhancer_variants():
    """
    Which levels WanGP itself considers part of the Qwen family, asked of
    WanGP rather than assumed.

    get_qwen35_prompt_enhancer_variant() raises KeyError for anything it does
    not know, so probing it is how a build that has gained a level is
    discovered without this file being edited.
    """
    try:
        from shared.prompt_enhancer.qwen35_vl import (
            get_qwen35_prompt_enhancer_variant,
        )
    except Exception:                                 # noqa: BLE001
        return {}
    found = {}
    for level in range(1, 12):
        try:
            found[level] = str(get_qwen35_prompt_enhancer_variant(level))
        except Exception:                             # noqa: BLE001
            continue
    return found


def _enhancer_name(level):
    """A human name for a level, preferring what WanGP calls it."""
    if level is None:
        return "unreadable"
    variant = _enhancer_variants().get(level)
    if variant:
        return variant
    return ENHANCER_LEVEL_NAMES.get(level, f"level {level}")


def _enhancer_is_capable(level):
    """
    Whether a level can be asked for JSON.

    Anything WanGP maps to a Qwen variant qualifies. So does an unknown
    level, on the grounds that refusing a model that would have worked is
    worse than trying one that will not - a model that ignores the format is
    caught by the reply parser and reported, whereas a wrong refusal leaves
    no way through at all.
    """
    if level is None or level <= 0:
        return False
    if level in _enhancer_variants():
        return True
    return level not in ENHANCER_CAPTIONING_LEVELS


def _offload_pipe():
    """
    The dict mmgp keeps its managed models in. Looked up by attribute name
    first so nothing unexpected gets touched; the sweep is a fallback and
    only inspects objects whose class name marks them as an offload object.
    """
    mod = _wgp_module()
    if mod is None:
        return {}
    found = {}
    for name in ("offloadobj", "offload_obj", "offloadobject",
                 "offload_object", "offloadobj_prompt_enhancer"):
        pipe = getattr(getattr(mod, name, None), "pipe", None)
        if isinstance(pipe, dict):
            found.update(pipe)
    if found:
        return found
    try:
        for value in list(vars(mod).values()):
            if type(value).__name__.lower().startswith("offload"):
                pipe = getattr(value, "pipe", None)
                if isinstance(pipe, dict):
                    found.update(pipe)
    except Exception:
        pass
    return found


def _tokenizer_for(model):
    """
    The Qwen loader hangs the tokenizer off the model, which is what
    makes a model borrowed straight from the pipe usable on its own.
    """
    for attr in ("_prompt_enhancer_tokenizer", "tokenizer"):
        tok = getattr(model, attr, None)
        if tok is not None:
            return tok
    return None


def _from_pipe():
    model = _offload_pipe().get(ENHANCER_PIPE_KEY)
    if model is None:
        return None
    tok = _tokenizer_for(model)
    # generate_messages carries its own tokenizer; the plain path needs one.
    if tok is None and not hasattr(model, "generate_messages"):
        return None
    return model, tok, "borrowed from WanGP"


def _from_runtime_global():
    try:
        from shared.prompt_enhancer.loader import PromptEnhancerRuntime
    except Exception:
        return None
    mod = _wgp_module()
    if mod is None:
        return None

    def usable(value):
        return (isinstance(value, PromptEnhancerRuntime)
                and getattr(value, "llm_model", None) is not None)

    candidates = [getattr(mod, name, None) for name in
                  ("prompt_enhancer_runtime", "prompt_enhancer",
                   "enhancer_runtime")]
    try:
        candidates.extend(vars(mod).values())
    except Exception:
        pass
    for value in candidates:
        if usable(value):
            tok = value.llm_tokenizer or _tokenizer_for(value.llm_model)
            return value.llm_model, tok, "borrowed from WanGP"
    return None


def _load_our_own(level):
    """
    Load the enhancer ourselves, fetching the weights if they are missing.
    Slower than borrowing - seconds to a minute the first time - but it means
    the button works without having run an enhancement first.
    """
    if _ENHANCER_OWNED["model"] is not None:
        return (_ENHANCER_OWNED["model"], _ENHANCER_OWNED["tokenizer"],
                "loaded by the builder")

    try:
        from shared.prompt_enhancer.loader import (
            download_prompt_enhancer_assets,
            load_prompt_enhancer_runtime,
        )
    except Exception as exc:
        raise EnhancerUnavailable(
            f"this build's prompt_enhancer package looks different ({exc})")

    # WanGP's own key is prompt_enhancer_quantization; the rest are older or
    # forked spellings. Getting this wrong is not harmless: the variant spec
    # only overrides the backend it is given, so a missed key means Qwen3.5
    # downloads Int8 weights when GGUF was chosen, and Qwen3.8 downloads Q4
    # when Q2 was.
    backend = _config_str(("prompt_enhancer_quantization",
                           "enhancer_quantization", "prompt_enhancer_backend",
                           "qwen_backend"), "quanto_int8")
    engine = _config_str(("lm_decoder_engine", "prompt_enhancer_lm_engine",
                          "llm_engine"), "")
    # Qwen3.8-27B ships MTP draft weights and WanGP turns speculative
    # decoding on for it by default. Passing the setting through keeps a
    # self-loaded copy the same speed as WanGP's own.
    speculative = _config_int(("prompt_enhancer_speculative_decoding",))

    def _download(**kwargs):
        try:
            return download_prompt_enhancer_assets(
                level, qwen_backend=backend, speculative_decoding=speculative,
                **kwargs)
        except TypeError:
            # Older builds take neither speculative_decoding nor a backend.
            try:
                return download_prompt_enhancer_assets(
                    level, qwen_backend=backend, **kwargs)
            except TypeError:
                return download_prompt_enhancer_assets(level, **kwargs)

    try:
        _download()
    except EnhancerUnavailable:
        raise
    except Exception as exc:
        raise EnhancerUnavailable(
            f"the {_enhancer_name(level)} weights could not be fetched "
            f"({exc})")

    def _no_download(**_kw):
        return None            # assets ensured just above

    try:
        runtime = load_prompt_enhancer_runtime(
            _no_download,
            enhancer_enabled=level,
            lm_decoder_engine=engine,
            qwen_backend=backend,
            speculative_decoding=speculative,
        )
    except TypeError:
        # A build without speculative decoding, then one without the backend
        # arguments at all.
        try:
            runtime = load_prompt_enhancer_runtime(
                _no_download, enhancer_enabled=level,
                lm_decoder_engine=engine, qwen_backend=backend)
        except TypeError:
            try:
                runtime = load_prompt_enhancer_runtime(_no_download, level)
            except Exception as exc:
                raise EnhancerUnavailable(
                    f"the enhancer would not load "
                    f"({type(exc).__name__}: {exc})")
        except Exception as exc:
            raise EnhancerUnavailable(
                f"the enhancer would not load ({type(exc).__name__}: {exc})")
    except Exception as exc:
        raise EnhancerUnavailable(
            f"the enhancer would not load ({type(exc).__name__}: {exc})")

    model = getattr(runtime, "llm_model", None)
    if model is None:
        raise EnhancerUnavailable("the enhancer loaded without a language model")
    tok = getattr(runtime, "llm_tokenizer", None) or _tokenizer_for(model)

    # Ours is not in the offload profile, so nothing else will move it.
    try:
        import torch
        if torch.cuda.is_available() and hasattr(model, "to"):
            model.to("cuda")
    except Exception:
        pass

    _ENHANCER_OWNED["model"] = model
    _ENHANCER_OWNED["tokenizer"] = tok
    return model, tok, "loaded by the builder"


def _release_our_own():
    """Give our copy's VRAM back. A borrowed model is never touched."""
    model = _ENHANCER_OWNED["model"]
    if model is None:
        return
    _ENHANCER_OWNED["model"] = None
    _ENHANCER_OWNED["tokenizer"] = None
    try:
        from shared.prompt_enhancer.loader import unload_prompt_enhancer_models
        unload_prompt_enhancer_models(model)
    except Exception:
        pass
    try:
        if hasattr(model, "to"):
            model.to("cpu")
    except Exception:
        pass
    del model
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _probe_report():
    """What the resolver can and cannot see, for when it comes back empty."""
    mod = _wgp_module()
    pipe = _offload_pipe()
    enhancer_keys = [k for k in pipe if "enhancer" in str(k).lower()]
    level = _enhancer_level()
    return (
        f"wgp module {'found' if mod is not None else 'not found'}; "
        f"enhancer level {level if level is not None else 'unreadable'}"
        + (f" ({_enhancer_name(level)})" if level is not None else "") + "; "
        f"offload pipe holds {len(pipe)} model(s)"
        + (f", enhancer entries {enhancer_keys}" if enhancer_keys
           else ", no enhancer entry")
        + f"; own copy {'held' if _ENHANCER_OWNED['model'] is not None else 'none'}"
    )


def _get_enhancer():
    """(model, tokenizer, how_it_was_obtained), or raise EnhancerUnavailable."""
    level = _enhancer_level()
    if level is not None and not _enhancer_is_capable(level):
        if level <= 0:
            raise EnhancerUnavailable(
                "the **Prompt Enhancer** is switched off - turn it on in the "
                "Configuration tab and pick a Qwen one")
        raise EnhancerUnavailable(
            f"the selected **Prompt Enhancer** ({_enhancer_name(level)}) is a "
            "captioning model and will not follow this instruction - switch "
            "to a Qwen enhancer in the Configuration tab")

    for finder in (_from_pipe, _from_runtime_global):
        try:
            found = finder()
        except Exception:
            found = None
        if found:
            return found

    if level is None:
        raise EnhancerUnavailable(
            "nothing is loaded and the Prompt Enhancer setting could not be "
            "read, so there is nothing to load either - " + _probe_report())
    return _load_our_own(level)


def _run_enhancer(system_prompt, user_text,
                  max_new_tokens=ENHANCER_MAX_NEW_TOKENS,
                  keep_loaded=None):
    """
    One text-only completion through the Prompt Enhancer, with our own system
    prompt in place of its cinematic one.

    keep_loaded overrides the ENHANCER_KEEP_LOADED default for this call, so
    the checkbox in the UI can decide per press. It only governs a copy we
    loaded ourselves - a borrowed enhancer is WanGP's to release.

    Returns (text, source) on success or (None, reason) on failure.
    """
    if keep_loaded is None:
        keep_loaded = ENHANCER_KEEP_LOADED
    try:
        model, tokenizer, source = _get_enhancer()
    except EnhancerUnavailable as exc:
        return None, str(exc)
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}. {_probe_report()}"

    try:
        from shared.prompt_enhancer.prompt_enhance_utils import (
            generate_cinematic_prompt,
        )
    except Exception as exc:
        if not keep_loaded:
            _release_our_own()
        return None, f"could not reach the Prompt Enhancer ({exc})"

    # Thinking is the main way this fails: the reasoning eats the token
    # budget, generation stops before the answer, and there is nothing to
    # parse. The keyword argument below is not honoured by every build - a
    # model borrowed from the pipe carries the flag WanGP set at load time -
    # so the flag is also set on the object, and /no_think appended to the
    # text, which the Qwen 3 chat templates understand on their own.
    thinking_attr = "_prompt_enhancer_enable_thinking"
    had_thinking = getattr(model, thinking_attr, None)
    try:
        setattr(model, thinking_attr, False)
    except Exception:
        had_thinking = None
    if not user_text.rstrip().endswith("/no_think"):
        user_text = user_text.rstrip() + "\n\n/no_think"

    # images=None takes the text-only path, so no vision tower is touched.
    call = dict(
        images=None,
        max_new_tokens=max_new_tokens,
        prompt_enhancer_instructions=system_prompt,
        do_sample=False,
        thinking_enabled=False,
    )
    trimmed = {k: v for k, v in call.items()
               if k not in ("thinking_enabled", "do_sample")}

    out, error = None, None
    for attempt in (call, trimmed):
        try:
            out = generate_cinematic_prompt(
                None, None, model, tokenizer, user_text, **attempt)
            error = None
            break
        except TypeError as exc:
            error = f"this build's enhancer takes different arguments ({exc})"
            continue
        except Exception as exc:
            error = f"the enhancer raised {type(exc).__name__}: {exc}"
            break

    if had_thinking is not None:
        try:
            setattr(model, thinking_attr, had_thinking)
        except Exception:
            pass
    if not keep_loaded:
        _release_our_own()

    if error:
        return None, error
    if not out:
        return None, "the enhancer returned nothing"
    return (out[0] or "").strip(), source


def _strip_thinking(text):
    """
    Remove a reasoning block. A closed <think>...</think> is cut out; an
    unclosed one means generation was truncated mid-thought, so everything
    from it onward is dropped and whatever came before is kept.
    """
    text = _ENH_THINK_RE.sub("", text or "")
    open_at = text.find("<think>")
    if open_at != -1:
        text = text[:open_at]
    return text.strip()


def _clean_audio_line(text):
    """One clean line, safe to drop into a prompt field."""
    text = _ENH_TAG_RE.sub("", str(text or ""))
    text = " ".join(text.split())
    text = text.strip().strip('"').strip("'").strip()
    m = _ENH_LABEL_RE.match(text)
    if m:
        text = m.group(2).strip().strip('"').strip("'").strip()
    return text


def _parse_audio_reply(raw):
    """Pull (soundscape, music) out of the reply. JSON first, prose second."""
    text = _strip_thinking(raw)
    text = _ENH_FENCE_RE.sub("", text).strip()

    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            blob = json.loads(text[start:end + 1])
        except ValueError:
            blob = None
        if isinstance(blob, dict):
            lower = {str(k).lower(): v for k, v in blob.items()}
            return (_clean_audio_line(lower.get("soundscape")
                                      or lower.get("overall_soundscape")),
                    _clean_audio_line(lower.get("music")
                                      or lower.get("non_diegetic_music")))

    found = {"soundscape": "", "music": ""}
    for line in text.split("\n"):
        m = _ENH_LABEL_RE.match(line)
        if not m:
            continue
        tag = m.group(1).lower()
        key = ("music" if tag in ("music", "score", "non_diegetic_music")
               else "soundscape")
        if not found[key]:
            found[key] = _clean_audio_line(m.group(2))
    if found["soundscape"] or found["music"]:
        return found["soundscape"], found["music"]

    # Last resort: a well-formed answer that simply ignored the format. One
    # line is the soundscape, two lines are the soundscape and the score.
    # Anything that reads as the model talking to the user rather than
    # describing sound is rejected - a refusal is not a soundscape.
    chatter = ("i ", "i'", "sorry", "as an", "here", "sure", "okay", "ok,",
               "certainly", "of course", "note", "please", "unfortunately")

    def usable(line):
        low = line.lower()
        return (15 <= len(line) < 400
                and not low.startswith(chatter))

    lines = [_clean_audio_line(l) for l in text.split("\n")]
    lines = [l for l in lines if l]
    if len(lines) == 1 and usable(lines[0]):
        return lines[0], ""
    if len(lines) == 2 and all(usable(l) for l in lines):
        return lines[0], lines[1]
    return "", ""


# =============================================================================
# Limits
# =============================================================================

MAX_ENTRIES = 4      # cast and subjects are one list

# Words the Of field can end on that are plainly waiting for a noun, so
# Insert subject continues the phrase instead of starting a new item.
ANCHOR_CONNECTIVES = frozenset({
    "of", "and", "with", "beside", "behind", "before", "between",
    "near", "next", "to", "at", "on", "in", "over", "under", "above",
    "below", "facing", "opposite", "toward", "towards", "across",
    "around", "past", "through", "the", "a", "an",
})
MAX_SPEAKERS = 6     # how many speaker slots the Speaker dropdown offers

# How many reference-source rows the Reference sources section offers.
# Duties a subject owns are declared on the subject, so this only has to
# cover whole-asset ones - a continued video, a composition anchor, a
# soundtrack. Four is comfortably more than a normal prompt uses.
MAX_REFS = 4

# Visibility updates _clear and _restore_draft return after the field values:
# one per cast entry, one per entry's reference block, one per
# reference-source row, then the two audio reference blocks and the summary
# block.
CLEAR_GROUP_UPDATES = MAX_ENTRIES * 2 + MAX_REFS + 3

# How many saved prompts the Load dropdown offers. It is a shortcut, not a
# file manager - the rest of the folder is reached through Load from file.
PROMPT_LIST_LIMIT = 10

# What a saved subject holds, by name.
#
# Named rather than positional, unlike the draft. The draft is one rolling
# autosave of 107 fields and naming them would be another construction site
# to keep in step; a subject is twenty fields written once, and naming them
# means a character saved today still loads after the plugin gains a field.
# For a library you build up over months that matters more than it does for
# an autosave that is replaced every twenty seconds.
#
# Three things are deliberately absent. The Ref2VA reference fields - source,
# retention, note, voice_from, motion_from - name assets attached to one
# project, and carrying them into another would point at slots that hold
# something else entirely. The speaker ID describes this subject's part in
# this prompt's action rather than anything about the character, so a subject
# loaded into slot 3 keeps slot 3's existing ID instead of dragging (S1) in
# and colliding with whoever already has it.
SUBJECT_FIELDS = [
    "kind", "desc", "onscreen", "age", "gender", "pitch", "timbre",
    "rate", "accent", "lang",
    "use_creator", "char_name", "char_ethnicity", "char_gender", "char_age",
    "char_height", "char_build", "char_hairstyle", "char_haircolor",
    "char_eyecolor", "char_clothing",
]

# The same shortcut cap as prompts, for the same reason.
SUBJECT_LIST_LIMIT = 10

# How often the form is written to disk while you work, in seconds. Only
# used when the installed Gradio has gr.Timer; otherwise the draft is saved
# on every button press instead.
AUTOSAVE_SECONDS = 20

# The last payload written, so an idle timer costs a comparison rather than
# a file write and a UI update.
_DRAFT_CACHE = {"values": None}

# Substrings that mark a model as MiniMax H3. Checked case-insensitively
# against the model type reported by on_model_change.
# Substrings that mark a model as MiniMax H3, checked case-insensitively
# against the model type WanGP reports.
#
# Deliberately the full "minimax_h3" rather than either half. Every H3
# architecture is minimax_h3_ref2va or minimax_h3_fl2va, with _pruned and
# _pdd variants, so the prefix catches all of them - while "minimax" alone
# also matched minimax_music3, a MiniMax model this builder has nothing to
# say about, and "h3" alone is loose enough to catch anything.
H3_MODEL_HINTS = ("minimax_h3",)

# H3's sliding-window figures, read out of WanGP rather than guessed at.
# fps, frames_minimum and frames_steps come from the model definition in
# models/minimax_h3/minimax_h3_handler.py; window_max and overlap_default
# come from its sliding_window_defaults; the 4-15 second band is what
# MiniMax's own documentation gives for a single window.
#
# These are advisory. Nothing here blocks an insert - a longer window may
# degrade or run out of VRAM, but that is the user's call to make, and a
# later model may widen the band or remove the problem entirely. When that
# happens these five numbers are the only edit: every warning and every
# emitted command reads them from here.
H3_FPS = 24
H3_WINDOW_FRAMES_MIN = 107            # 4.46s - one window cannot be shorter
H3_WINDOW_FRAMES_MAX = 481            # 20.04s - the Sliding Window Size cap
H3_WINDOW_DOC_SECONDS = (4.0, 15.0)   # the documented band
H3_OVERLAP_DEFAULT = 18               # overlap_default; the ladder is 17k+1


# =============================================================================
# Vocabulary
# =============================================================================

# Worked example used for the greyed-out placeholder text. Read end to end,
# the fields below form one coherent prompt: a fishmonger gutting a fish,
# across two shots of two beats each. Placeholders vanish as soon as you type.

EXAMPLE_ENTRIES = [
    {
        "desc": "a fishmonger in his fifties, heavy rubber apron, forearms wet to the elbow",
        "accent": "a faint West Country accent",   # or just "West Country"
        "note": "the rubber apron and wet forearms are retained",
        "shots": "1, 2",
    },
    {
        "desc": "the marble counter of a covered market stall, crushed ice banked along its length",
        "accent": "",
        "note": "the marble surface and banked ice are retained",
        "shots": "1, 2",
    },
]

EXAMPLE_ENTRY_FALLBACK = {
    "desc": "another subject - a person, an animal, a place or a prop",
    "accent": "any accent worth stating",
    "note": "which of its features carry over",
    "shots": "1, 2",
}

# The worked example, now that the action lives in one field. Same fishmonger
# as the rest of the placeholders, written the way the buttons would write it.
EXAMPLE_ACTION = (
    "[Shot 1] The camera cuts to a medium shot of the fishmonger behind the "
    "counter, a whole fish laid on the ice, on a 35mm lens. A chalkboard "
    'reading "TODAY: SEA BASS" is visible in the frame. (S1) lifts the fish '
    "onto the board and says: <d>[English] This one came off the boat this "
    "morning.</d> At 00:03.000, (S1) sets the blade against the belly and "
    "steadies the fish.\n"
    "[Shot 2] At 00:05.000, the camera cuts to a close-up of his hands and "
    "the board, on a 50mm macro lens. (S1) draws the knife along the spine "
    "in one unbroken movement."
)


# Keyframe images are attached in the generator; the checkboxes only decide
# which instruction line the prompt needs.
KEYFRAME_NOTE = (
    "Tick these to match the keyframe images attached in the generator - they "
    "write the instruction line the model expects. Referring to "
    "`<Picture 1>` inside your own descriptions is up to you; the guide's "
    "examples do it in the shot text as well."
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

# Framing as a bare adjective. The word "shot" is added once, after the rig,
# so "medium" and "crane" compose into "a medium crane shot" rather than the
# two of them being bolted together with a preposition.
#
# Two- and three-shots are deliberately absent. They only read correctly with
# the number last - "a medium handheld three shot" - which the framing-then-
# rig order cannot produce, and there are far too many variations of them to
# enumerate anyway. The field takes typed values, so the whole phrase goes in
# here instead: type "medium handheld three" and leave Rig alone.
FRAMINGS = [
    "", "extreme close-up", "close-up", "medium close-up", "medium",
    "medium-wide", "wide", "extreme wide", "establishing",
    "over-the-shoulder", "low-angle", "high-angle", "overhead", "worm's-eye",
    "dutch-angle", "point-of-view", "profile", "silhouette", "insert",
    "cutaway", "master", "reflection", "through-the-window",
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
# Rigs as adjectives too, for the same reason. "A medium shot, on a crane"
# put a crane in the frame - the phrase had no stated subject, so the model
# attached it to the scene. "A medium crane shot" cannot be read that way,
# because the rig is describing the shot rather than sitting in it.
RIGS = [
    "", "tripod", "handheld", "shoulder-rig", "steadicam", "gimbal", "dolly",
    "crane", "jib", "drone", "cable-cam", "car-mounted", "hood-mounted",
    "slider", "motion-control", "underwater", "body-mounted",
    "vehicle-mounted", "rickshaw",
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

# Used when the camera changes within a shot rather than at a cut. The
# insert button picks from these instead of CUT_VERBS when a camera sentence
# has already been written since the last [Shot N].
CONTINUE_VERBS = [
    "the camera moves to",
    "the camera continues into",
    "the camera reframes to",
    "the camera settles into",
    "the camera drifts to",
    "the camera swings round to",
]

LANGUAGES = [
    "English", "Mandarin Chinese", "Cantonese", "Japanese", "Korean",
    "Spanish", "French", "German", "Italian", "Portuguese", "Russian",
    "Polish", "Greek", "Turkish", "Arabic", "Hebrew", "Hindi", "Bengali",
    "Thai", "Vietnamese", "Indonesian", "Tagalog", "Swahili",
    "Swedish", "Norwegian", "Danish", "Finnish", "Irish", "Welsh",
]

VOICE_AGES = ["", "child", "teenage", "young", "young adult", "middle-aged",
              "older", "elderly", "ageless"]
VOICE_GENDERS = ["", "female", "male", "androgynous"]
VOICE_PITCH = ["", "low", "medium", "high"]
VOICE_TIMBRE = [
    "", "clear", "raspy", "breathy", "warm", "nasal", "gravelly", "bright",
    "weathered", "smooth", "resonant", "husky", "rich", "booming",
]
VOICE_RATE = [
    "", "slow", "measured", "unhurried", "quick", "clipped", "halting",
    "breathless", "drawling", "steady", "urgent",
]

# Character creator ---------------------------------------------------------
# Physical description only - voice already has its own fields above. All
# typeable; entries left blank are simply omitted from the composed sentence.

CHAR_ETHNICITIES = [
    "", "Asian", "East Asian", "South Asian", "Black", "White",
    "Hispanic", "Middle Eastern", "Native American", "Pacific Islander",
    "Mixed ethnicity",
]

CHAR_GENDERS = ["", "male", "female", "non-binary", "androgynous"]

CHAR_AGE_RANGES = [
    "", "childhood", "the teens", "early 20s", "late 20s", "early 30s",
    "mid-30s", "early 40s", "mid-40s", "the 50s", "the 60s", "the 70s",
    "old age",
]

CHAR_HEIGHTS = [
    "", "short", "petite", "average height", "five foot four",
    "five foot eight", "six feet", "six foot two", "tall",
]

CHAR_BUILDS = [
    "", "slender", "athletic", "toned", "muscular", "stocky", "heavyset",
    "broad-shouldered", "wiry", "curvy", "average build", "petite build",
]

CHAR_HAIRSTYLES = [
    "", "long straight", "long wavy", "long curly", "shoulder-length",
    "short cropped", "buzz cut", "shaved head", "braided", "dreadlocked",
    "afro", "slicked-back", "messy tousled", "ponytailed",
]

CHAR_HAIR_COLORS = [
    "", "black", "dark brown", "light brown", "blonde", "auburn", "red",
    "grey", "white", "salt-and-pepper", "dyed vibrant",
]

CHAR_EYE_COLORS = [
    "", "brown", "dark brown", "blue", "light blue", "green", "hazel",
    "grey", "amber", "black",
]

# Phrased to follow "wearing", so anything typed here should read the same
# way - "a rumpled trenchcoat", not "trenchcoat" or "he wears a trenchcoat".
# An entry that already carries its own verb ("dressed in ...") is used as
# written instead.
CHAR_CLOTHING = [
    "", "a plain white t-shirt and jeans", "a rumpled trenchcoat",
    "a tailored black suit", "a floral summer dress", "a long evening gown",
    "a hooded sweatshirt and joggers", "a leather biker jacket",
    "a wool overcoat and scarf", "a white lab coat over scrubs",
    "a stained apron over a work shirt",
    "a high-visibility jacket and work boots", "a police uniform",
    "a military field uniform", "a school uniform",
    "worn workwear, patched at the knees", "traditional formal dress",
]


# The guide sanctions four wordings for audio crossing a cut. Three look
# forward from the shot that starts the line; the fourth looks back, and is
# written automatically on the receiving side, so it is not offered here.
CARRY_PHRASES = [
    "continues seamlessly across the cut",
    "continues uninterrupted into the next shot",
    "remains audible across the transition",
]

# What is carrying the on-screen text. Phrased with its article so the
# sentence reads "A neon sign reading ... is visible in the frame."
SCREEN_TEXT_KINDS = [
    "a sign", "a neon sign", "a shopfront sign", "a street sign",
    "a banner", "a poster", "a label", "a subtitle", "a caption",
    "a screen", "a phone screen", "a handwritten note", "a letter",
    "a newspaper headline", "a book cover", "a badge", "a licence plate",
    "a printed T-shirt", "a chalkboard", "a departure board",
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

# Every slot an asset can occupy, for the Reference sources section. A label
# belongs to the *asset*, not to the job it does: <Video 1> means "the first
# video I attached", whatever duties it ends up serving. Naming the same slot
# twice is how you say one file does two jobs, and the assembly merges those
# duties into one definition and one retention line rather than treating them
# as two assets.
REF_ALL_SLOTS = ([""] + REF_PICTURE_SLOTS
                 + REF_VIDEO_SLOTS[1:] + REF_AUDIO_SLOTS[1:])

# Whether a reference video's soundtrack takes the audio slot matching its
# own number - <Video 2>'s soundtrack being <Audio 2>.
#
# Read out of WanGP's H3 validator, which in soundtrack mode sets
# audio_count to the number of reference videos and walks them in order,
# requiring each to have an audio track. Its selector also offers standalone
# audio references and video soundtracks as separate choices rather than a
# combination, so today the mapping is one-to-one and positional.
#
# A default, not a rule. The underlying field is a letter set that could
# hold a combination the dropdown does not currently offer, and this plugin
# writes prompt text - it is in no position to enforce anything about what
# gets selected in WanGP. A Reference sources row that names a different
# slot wins, silently: whoever is looking at the panel knows better than
# this constant does.
SOUNDTRACK_FOLLOWS_VIDEO_ORDER = True

# What a subject's voice can be taken from. A voice reference is always an
# <Audio N> - the guide gives that label to audio referenced for voice, and
# numbers it independently of video - but a reference video's soundtrack is
# a perfectly ordinary source for one, and WanGP spends an audio-reference
# slot on it. Picking a video here means "the soundtrack of that video", and
# the assembly resolves it to whichever audio slot a Reference sources row
# says it occupies.
VOICE_SOURCE_SLOTS = REF_AUDIO_SLOTS + REF_VIDEO_SLOTS[1:]

# What a subject's appearance can be taken from. Pictures, and now videos:
# the guide treats a video used only for appearance as reference generation
# rather than editing, and says visible content taken from a video still
# receives <Subject N> labels.
SUBJECT_SOURCE_SLOTS = REF_PICTURE_SLOTS + REF_VIDEO_SLOTS[1:]

# Whole-asset duties, for the Reference sources section. Duties a *subject*
# owns - its voice, its motion, its appearance - stay in the subject's own
# block, because they belong to the subject rather than to the file. Both
# routes feed the same registry, so a slot named in both merges.
#
# Each entry maps the visible label to the kind of slot it belongs on and the
# phrase the definition line uses.
REF_ROLE_PHRASES = {
    "continued from":               ("Video", "the video this one continues from"),
    "edited":                       ("Video", "the video being edited"),
    "camera and cutting structure": ("Video", "the camera and cutting-structure reference"),
    "motion and rhythm":            ("Video", "the motion and rhythm reference"),
    "appearance and content":       ("Video", "the appearance and content reference"),
    "keyframe":                     ("Picture", "a keyframe"),
    "composition anchor":           ("Picture", "the composition anchor"),
    "storyboard frame":             ("Picture", "a storyboard frame"),
    "edited frame":                 ("Picture", "the frame being edited"),
    "appearance reference":         ("Picture", "an appearance reference"),
    "soundtrack":                   ("Audio", "the soundtrack reference"),
    "dialogue":                     ("Audio", "the dialogue reference"),
    "sound effects":                ("Audio", "the sound-effects reference"),
}

REF_ROLES = [""] + list(REF_ROLE_PHRASES)

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

# What to assume when a subject is drawn from a video but its own retention
# marker is left blank. The general fallback for an asset is weak_reference,
# which is right for something that only nudges the generation - a camera
# reference, say - and wrong for a video a character's face and build are
# being lifted from. partially_preserved says the character carries over
# while the surroundings do not, which is what that role means.
APPEARANCE_RETENTION = "partially_preserved"

TASK_TYPES = ["keyframe completion", "reference generation", "video editing",
              "video continuation", "audio reuse", "audio reference"]



# -- Scene ------------------------------------------------------------------
# Woven into the opening shot rather than emitted as separate fields: H3 keeps
# all visual description inside integrated_multimodal_description.

LOCATIONS = [
    "a rain-slicked alley behind a nightclub", "a farmhouse kitchen",
    "an airport departure lounge", "a lighthouse gallery in a storm",
    "a disused swimming pool", "a rooftop garden above the city",
    "an antique shop crowded with clocks", "a motorway service station",
    "a boat deck on open water", "a stone chapel lit by candles",
    "a launderette", "a records archive in a basement",
    "a ski lift above treeline", "a bustling New York street",
    "a covered market hall", "a tiled underpass",
    "a rain-soaked city street", "a suburban kitchen", "a hotel corridor",
    "a crowded subway platform", "a quiet library reading room",
    "an empty car park", "a coastal fishing dock",
    "a pine forest clearing", "a desert highway", "a rooftop",
    "a hospital waiting room", "a school classroom", "a dive bar",
    "an office", "a country lane", "a snowbound cabin",
    "a train carriage", "a cathedral interior", "a warehouse floor",
    "a greenhouse", "a mountain ridge", "a riverbank",
]

# Time of day sits apart from lighting: lighting is how the scene is lit,
# this is when it happens. It reaches the opening sentence and the audio
# digest, where dawn and midnight imply very different sound.
TIMES_OF_DAY = [
    "", "at dawn", "in the early morning", "in the morning", "at midday",
    "in the afternoon", "in the late afternoon", "at golden hour",
    "at sunset", "at dusk", "in the evening", "at night", "late at night",
    "at midnight", "in the small hours",
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

# The guide asks for instrumentation, tempo and dynamic development, and its
# own examples read that way - "a restrained solo-piano score at a slow tempo,
# with sustained low cello underneath and no swell". Genre and mood labels are
# what it steers away from.
#
# Three lists rather than one, because they are three separate choices and a
# single mixed dropdown makes you hunt for which entries belong to which axis.
# The Add button composes one sentence across them.
MUSIC_INSTRUMENTS = [
    "", "a restrained solo piano",
    "sustained low strings",
    "a lone cello line",
    "a full string section",
    "brushed drums and upright bass",
    "a muted trumpet over brushed drums",
    "fingerpicked acoustic guitar",
    "electric guitar with long reverb",
    "warm analogue synth pads",
    "a pulsing synth arpeggio",
    "harp and woodwinds",
    "low brass and timpani",
    "a heroic brass fanfare",
    "sparse percussion and hand claps",
]

MUSIC_TEMPOS = [
    "", "at a slow tempo", "at a moderate tempo", "at a brisk tempo",
    "at a driving tempo", "on a steady pulse", "rubato, with no fixed pulse",
]

MUSIC_DYNAMICS = [
    "", "held quietly under the scene",
    "swelling gradually, then falling away",
    "building steadily to a peak",
    "entering late and fading at the end",
    "with no swell",
    "dropping out abruptly at the cut",
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
        # state carries "model_type"; the two trigger components are what
        # actually fire when the model changes. The visible model dropdown is
        # created after WanGP snapshots its locals, so it is not requestable.
        self.request_component("state")
        self.request_component("refresh_form_trigger")
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
        entries, char_names = [], []

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

        # Hidden from the start unless WanGP is opening on an H3 model. The
        # old default was visible, so the builder showed under whatever model
        # was selected at startup until the first switch fired the toggle.
        start_visible = self._starts_visible()

        with gr.Accordion("MiniMax H3 Prompt Builder", open=False,
                          visible=start_visible,
                          elem_id="h3_prompt_builder") as root:

            model_warning = gr.Markdown(visible=False)

            # Kept at the top and outside every accordion: after a crash this
            # is the first thing wanted, and it should not be behind a fold.
            with gr.Row():
                restore_draft_btn = gr.Button("Restore last draft", size="sm")
                save_draft_btn = gr.Button("Save draft now", size="sm")
                clear_top_btn = gr.Button("Clear all fields", size="sm")
            draft_status = gr.Markdown(self._draft_note())

            # The draft is one rolling autosave answering "what was I just
            # doing". This answers "where is that market scene from last
            # week" - named, kept, and never written over unasked.
            with gr.Accordion("Saved prompts", open=False):
                gr.Markdown(
                    "Saved into a `prompts` folder beside the plugin, one "
                    "file each. The dropdown lists the "
                    f"{PROMPT_LIST_LIMIT} most recent; **Load from file** "
                    "reaches the rest of the folder, and anything someone "
                    "sends you."
                )
                with gr.Row():
                    prompt_name = gr.Textbox(
                        label="Name", placeholder="fishmonger, two shots",
                        scale=3,
                    )
                    save_prompt_btn = gr.Button("Save prompt", size="sm",
                                                variant="primary", scale=1)
                with gr.Row():
                    prompt_picker = gr.Dropdown(
                        self._prompt_choices(), label="Saved prompts",
                        value=None, scale=3,
                    )
                    load_prompt_btn = gr.Button("Load", size="sm", scale=1)
                    refresh_prompts_btn = gr.Button("Refresh", size="sm",
                                                    scale=1)
                with gr.Row():
                    load_recent_btn = gr.Button("Load most recent", size="sm")
                prompt_file = gr.File(
                    label="Load from file", file_types=[".json"],
                    file_count="single",
                )
                prompt_status = gr.Markdown("")

            # Every section below is an accordion so a long form can be
            # folded down to the part being worked on. Open states here are
            # only the defaults - Gradio remembers nothing between sessions,
            # so they are set to what a first build needs rather than to
            # what is used most.
            with gr.Accordion("Mode & keyframes", open=False):
                gr.Markdown(KEYFRAME_NOTE)

                with gr.Row():
                    start_image = gr.Checkbox(label="Start image", value=False)
                    end_image = gr.Checkbox(label="End image", value=False)

                ref_mode = gr.Checkbox(
                    label="Use reference mode (Ref2VA)",
                    value=False,
                    info="Reveals reference subjects, the source video and "
                         "the task type. Leave unticked and none of it is "
                         "written.",
                )

            with gr.Accordion("Scene", open=False):
                gr.Markdown(
                    "What holds for the whole clip. These become the single "
                    "opening sentence before `[Shot 1]`. Lens and rig are "
                    "not here - they belong to the camera button in Action, "
                    "since both commonly change at a cut."
                )
                with gr.Row():
                    style = dd(STYLES, "Style")
                    grading = dd(GRADING, "Colour grade")

                with gr.Row():
                    location = dd(LOCATIONS, "Location")
                    # Location presets used to carry their own "at dawn"
                    # tails. They are stripped now that the hour is its own
                    # field, so picking both cannot produce "a rooftop at
                    # dusk at night".
                    time_of_day = dd(TIMES_OF_DAY, "Time of day")
                with gr.Row():
                    lighting = dd(SCENE_LIGHTING, "Lighting")
                    atmosphere = dd(SCENE_ATMOSPHERE, "Atmosphere")

                with gr.Row():
                    camera_type = dd(CAMERA_TYPES, "Camera / stock")

            # ---- source video (FL2VA continue / Ref2VA video reference) ----
            # ---- cast and subjects (one list) ------------------------------
            with gr.Accordion("Cast & subjects", open=False):
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
                    ex = (EXAMPLE_ENTRIES[i] if i < len(EXAMPLE_ENTRIES)
                          else EXAMPLE_ENTRY_FALLBACK)
                    # One accordion per entry: eight open at once is most of
                    # a screen, and only the one being edited needs to be.
                    with gr.Accordion(f"Subject {i + 1}", open=True,
                                      visible=False) as grp:
                        # A character is worth more than the prompt it was
                        # written for. Saved by name, and never carrying the
                        # reference assets or the speaker ID - those belong
                        # to the project rather than to the character.
                        with gr.Row():
                            e_lib_name = gr.Textbox(
                                label="Save this subject as", scale=2,
                                placeholder="Marcus, the fishmonger",
                            )
                            e_lib_save = gr.Button("Save subject", size="sm")
                        with gr.Row():
                            e_lib_pick = gr.Dropdown(
                                self._subject_choices(), label="Load a subject",
                                value=None, allow_custom_value=True, scale=2,
                            )
                            e_lib_load = gr.Button("Load subject", size="sm")
                            e_lib_refresh = gr.Button("Refresh subjects", size="sm")
                        e_lib_status = gr.Markdown("")

                        e_use_creator = gr.Checkbox(
                            label="Use character creator", value=False,
                        )
                        with gr.Group(visible=False) as e_creator_block:
                            e_char_name = gr.Textbox(
                                label="Name (optional)",
                                placeholder="John",
                            )
                            with gr.Row():
                                e_char_ethnicity = dd(CHAR_ETHNICITIES, "Ethnicity")
                                e_char_gender = dd(CHAR_GENDERS, "Gender")
                                e_char_age = dd(CHAR_AGE_RANGES, "Age range")
                            with gr.Row():
                                e_char_height = dd(CHAR_HEIGHTS, "Height")
                                e_char_build = dd(CHAR_BUILDS, "Build")
                            with gr.Row():
                                e_char_hairstyle = dd(CHAR_HAIRSTYLES, "Hairstyle")
                                e_char_haircolor = dd(CHAR_HAIR_COLORS, "Hair colour")
                                e_char_eyecolor = dd(CHAR_EYE_COLORS, "Eye colour")
                            with gr.Row():
                                e_char_clothing = dd(CHAR_CLOTHING, "Clothing")
                            e_add_to_desc = gr.Button(
                                "Add to description", size="sm",
                            )
                        e_use_creator.change(
                            fn=lambda t: gr.update(visible=bool(t)),
                            inputs=[e_use_creator], outputs=[e_creator_block],
                        )
                        e_desc = gr.Textbox(
                            label="Description",
                            placeholder=ex["desc"],
                        )
                        e_add_to_desc.click(
                            fn=self._character_description,
                            inputs=[e_char_name, e_char_ethnicity, e_char_gender,
                                    e_char_age, e_char_height, e_char_build,
                                    e_char_hairstyle, e_char_haircolor,
                                    e_char_eyecolor, e_char_clothing],
                            outputs=[e_desc],
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
                            e_accent = gr.Textbox(label="Accent (optional)",
                                                  placeholder=ex["accent"])
                        with gr.Row():
                            e_lang = gr.Dropdown(
                                LANGUAGES, label="Language", value="English",
                                allow_custom_value=True,
                                info="Used for this speaker's dialogue tags",
                            )
                        with gr.Group(visible=False) as e_ref_block:
                            with gr.Row():
                                e_source = gr.Dropdown(
                                    SUBJECT_SOURCE_SLOTS, label="Source asset",
                                    value=[], multiselect=True,
                                    allow_custom_value=True,
                                    info="Which reference image or video this "
                                         "subject comes from - type another "
                                         "tag if needed",
                                )
                                e_retention = gr.Dropdown(
                                    [""] + ALL_RETENTION, label="Retention",
                                    value="",
                                    info="How closely to hold to the "
                                         "source - or to the description "
                                         "itself, if there is no asset",
                                )
                            with gr.Row():
                                e_note = gr.Textbox(
                                    label="What is retained",
                                    placeholder=ex["note"],
                                )
                                # "Appears in shots" used to be typed here.
                                # It is read out of the action by speaker ID
                                # now, so it cannot go stale when a shot is
                                # renumbered or deleted by hand.
                            with gr.Row():
                                e_voice_from = gr.Dropdown(
                                    VOICE_SOURCE_SLOTS, label="Voice from",
                                    value="", allow_custom_value=True,
                                    info="A reference audio supplying this "
                                         "speaker's timbre - or a video, if "
                                         "its soundtrack is declared in "
                                         "Reference sources",
                                )
                                e_motion_from = gr.Dropdown(
                                    REF_VIDEO_SLOTS, label="Motion from",
                                    value="",
                                    info="A reference video supplying this "
                                         "subject's movement or performance",
                                )

                    char_names.append(e_char_name)
                    entries.append({
                        "group": grp, "kind": e_kind, "desc": e_desc,
                        "speaker": e_speaker, "onscreen": e_onscreen,
                        "use_creator": e_use_creator,
                        "creator_block": e_creator_block,
                        "char_name": e_char_name,
                        "char_ethnicity": e_char_ethnicity,
                        "char_gender": e_char_gender,
                        "char_age": e_char_age,
                        "char_height": e_char_height,
                        "char_build": e_char_build,
                        "char_hairstyle": e_char_hairstyle,
                        "char_haircolor": e_char_haircolor,
                        "char_eyecolor": e_char_eyecolor,
                        "char_clothing": e_char_clothing,
                        "age": e_age, "gender": e_gender, "pitch": e_pitch,
                        "timbre": e_timbre, "rate": e_rate, "accent": e_accent,
                        "lang": e_lang,
                        "ref_block": e_ref_block,
                        "source": e_source, "retention": e_retention,
                        "note": e_note,
                        "voice_from": e_voice_from,
                        "motion_from": e_motion_from,
                        "lib_name": e_lib_name, "lib_save": e_lib_save,
                        "lib_pick": e_lib_pick, "lib_load": e_lib_load,
                        "lib_refresh": e_lib_refresh,
                        "lib_status": e_lib_status,
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

                with gr.Accordion("Reference sources", open=False) as video_section:
                    gr.Markdown(
                        "Declare each file you attached in WanGP, once, and "
                        "say what job it does. A label belongs to the "
                        "**asset**, not to the job - `<Video 1>` means *the "
                        "first video I attached*, whatever it ends up doing. "
                        "Name the same slot on two rows and its duties merge "
                        "into one definition and one retention line, which is "
                        "how you say one file does two jobs.\n\n"
                        "Duties a **subject** owns - its voice, its motion, "
                        "its appearance - go on the subject instead, in Cast "
                        "& subjects. Those feed the same registry, so a slot "
                        "named in both merges too.\n\n"
                        "**Taken from** is how a video's soundtrack is "
                        "declared: `Audio 1`, taken from `Video 1`. WanGP's "
                        "Audio References selector offers one or two "
                        "standalone audio references *or* reference-video "
                        "soundtracks, never a mix."
                    )
                    with gr.Row():
                        ref_count = gr.Number(
                            label="Reference sources", value=0, precision=0,
                            minimum=0, maximum=MAX_REFS, interactive=False,
                        )
                        add_ref = gr.Button("Add source", size="sm")
                        rm_ref = gr.Button("Remove last source", size="sm")

                    refs = []
                    for i in range(MAX_REFS):
                        with gr.Group(visible=False) as ref_group:
                            with gr.Row():
                                r_slot = gr.Dropdown(
                                    REF_ALL_SLOTS, label=f"Source {i + 1}",
                                    value="", allow_custom_value=True,
                                    info="Which attached file this is",
                                )
                                r_role = gr.Dropdown(
                                    REF_ROLES, label="What it does",
                                    value="", allow_custom_value=True,
                                )
                            with gr.Row():
                                r_desc = gr.Textbox(
                                    label="What it contributes",
                                    placeholder="the original camera movement "
                                                "and the pace of the cuts",
                                )
                            with gr.Row():
                                r_retention = gr.Dropdown(
                                    [""] + ALL_RETENTION, label="Retention",
                                    value="",
                                    info="Picture markers for pictures and "
                                         "video, audio markers for audio",
                                )
                                r_from = gr.Dropdown(
                                    REF_ALL_SLOTS, label="Taken from (optional)",
                                    value="", allow_custom_value=True,
                                    info="For a soundtrack or a frame lifted "
                                         "out of another asset",
                                )
                        refs.append({
                            "group": ref_group, "slot": r_slot,
                            "role": r_role, "desc": r_desc,
                            "retention": r_retention, "from": r_from,
                        })

                    _ref_out = [ref_count] + [r["group"] for r in refs]
                    add_ref.click(
                        fn=lambda n: self._step_count(n, +1, MAX_REFS),
                        inputs=[ref_count], outputs=_ref_out,
                    )
                    rm_ref.click(
                        fn=lambda n: self._step_count(n, -1, MAX_REFS),
                        inputs=[ref_count], outputs=_ref_out,
                    )

            # The subject library, one set of controls per entry. Driven off
            # SUBJECT_FIELDS so save, load and this wiring cannot disagree
            # about the order - the failure mode the flat list already
            # taught us to design out.
            for e in entries:
                fields = [e[name] for name in SUBJECT_FIELDS]
                e["lib_save"].click(
                    fn=self._save_subject,
                    inputs=[e["lib_name"]] + fields,
                    outputs=[e["lib_pick"], e["lib_status"]],
                )
                e["lib_load"].click(
                    fn=self._load_subject, inputs=[e["lib_pick"]],
                    outputs=fields + [e["creator_block"], e["lib_status"]],
                )
                e["lib_refresh"].click(
                    fn=self._refresh_subjects, inputs=[],
                    outputs=[e["lib_pick"], e["lib_status"]],
                )


            with gr.Accordion("Action", open=True):
                gr.Markdown(
                    "Write the action here. The buttons insert labels, "
                    "timestamps, camera sentences and dialogue in the format "
                    "H3 expects - everything they write is ordinary text you "
                    "can edit afterwards.\n\n"
                    "**Shot** starts a new line; everything else is appended "
                    "to the line you are on, so build a shot left to right "
                    "and then tidy it up."
                )
                action_text = gr.Textbox(
                    label="Action", lines=14, max_lines=40,
                    placeholder=EXAMPLE_ACTION,
                )
                # Holds the field as it was before the last insert, so Undo
                # needs no history and cannot drift out of step with the text.
                action_prev = gr.State("")
                action_status = gr.Markdown("")

                with gr.Row():
                    ins_shot = gr.Button("Shot", size="sm", variant="primary")
                    at_seconds = gr.Number(
                        label="At (seconds)", value=None, minimum=0, step=0.5,
                    )
                    ins_time = gr.Button("Time", size="sm")

                with gr.Accordion("Camera", open=True):
                    with gr.Row():
                        cam_verb = gr.Dropdown(
                            [""] + CUT_VERBS + CONTINUE_VERBS,
                            label="Transition", value="",
                            allow_custom_value=True,
                            info="Leave blank and it picks 'cuts to' for a "
                                 "new shot, 'moves to' within one",
                        )
                        cam_framing = dd(FRAMINGS, "Framing")
                        cam_lens = dd(LENS_TYPES, "Lens")
                    with gr.Row():
                        cam_motion = locked_dd(MOTION_TYPES, "Camera motion")
                        cam_ampl = locked_dd(AMPLITUDES, "Amplitude")
                        cam_speed = locked_dd(SPEEDS, "Speed")
                        cam_rig = dd(RIGS, "Rig")
                    cam_anchor = gr.Textbox(
                        label="Of - what is in frame",
                        placeholder="the fishmonger behind the counter, a "
                                    "whole fish laid on the ice",
                        info="Goes in the middle of the sentence, where the "
                             "grammar wants it. Leave blank for a camera "
                             "sentence that stands on its own",
                    )
                    with gr.Row():
                        cam_subject = gr.Dropdown(
                            [f"Subject {n + 1}" for n in range(MAX_ENTRIES)],
                            label="Subject", value=None, scale=2,
                            info="Appends the subject's <Subject N> tag, numbered "
                                 "as the finished prompt will number it",
                        )
                        ins_anchor_subject = gr.Button(
                            "Insert subject", size="sm")
                    ins_camera = gr.Button("Add camera", size="sm")

                with gr.Accordion("Dialogue", open=True):
                    with gr.Row():
                        dl_who = gr.CheckboxGroup(
                            [f"Subject {n + 1}" for n in range(MAX_ENTRIES)],
                            label="Who", value=[],
                        )
                    with gr.Row():
                        dl_type = gr.Dropdown(
                            ["dialogue", "voiceover"], label="Type",
                            value="dialogue",
                            info="voiceover writes the spec's required "
                                 "phrasing and the closed-lips clause",
                        )
                        dl_lang = dd(LANGUAGES, "Language")
                    dl_delivery = gr.Textbox(
                        label="Action and delivery",
                        placeholder="lifts the fish onto the board and says",
                        info="Goes outside the <d> tag. For a voiceover a "
                             "trailing 'says' is dropped, since the spec "
                             "fixes that wording",
                    )
                    dl_speech = gr.Textbox(
                        label="Spoken words",
                        placeholder="This one came off the boat this morning.",
                    )
                    with gr.Row():
                        dl_carries = gr.Checkbox(
                            label="Line carries across the next cut",
                            value=False,
                        )
                        dl_carry_phrase = gr.Dropdown(
                            CARRY_PHRASES, label="How it carries",
                            value=CARRY_PHRASES[0], allow_custom_value=True,
                            info="The next Shot writes its matching half",
                        )
                        dl_cutoff = gr.Checkbox(
                            label="Speech runs past the end", value=False,
                            info="Writes <cutoff>",
                        )
                    ins_dialogue = gr.Button("Add dialogue", size="sm")

                with gr.Accordion("Visible text", open=False):
                    with gr.Row():
                        st_kind = gr.Dropdown(
                            [""] + SCREEN_TEXT_KINDS,
                            label="Visible text is on", value="",
                            allow_custom_value=True,
                        )
                        st_text = gr.Textbox(
                            label="Visible text", placeholder="TODAY: SEA BASS",
                            info="Quoted verbatim, never translated",
                        )
                    ins_screen = gr.Button("Add visible text", size="sm")

                with gr.Row():
                    undo_action = gr.Button("Undo last insert", size="sm")

            # ---- audio -----------------------------------------------------
            with gr.Accordion("Audio", open=False):
                # The box is the field. The presets are a way of filling it
                # in, not a second input that gets merged at build time -
                # which also means what you read here is exactly what the
                # prompt will say, and what the enhancer is shown.
                soundscape = gr.Textbox(
                    label="Soundscape",
                    lines=3,
                    placeholder="Ice shifts under the fish and the blade taps the board. Market chatter carries from further down the hall.",
                )
                with gr.Row():
                    soundscape_presets = gr.Dropdown(
                        SOUNDSCAPE_PRESETS, label="Presets",
                        value=[], multiselect=True, allow_custom_value=True,
                        scale=4,
                    )
                    add_soundscape_btn = gr.Button("Add", size="sm", scale=1)
                soundscape_status = gr.Markdown("")
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
                music = gr.Textbox(
                    label="Non-diegetic music",
                    lines=3,
                    placeholder="A slow upright bass with brushed drums, thinning out as the second shot begins.",
                )
                # Three axes rather than one mixed list, because the guide
                # asks for all three and a single dropdown makes you hunt for
                # which entries belong to which. Add composes them in the
                # guide's order.
                with gr.Row():
                    music_instruments = dd(MUSIC_INSTRUMENTS, "Instrumentation")
                    music_tempo = dd(MUSIC_TEMPOS, "Tempo")
                    music_dynamics = dd(MUSIC_DYNAMICS, "Development")
                    add_music_btn = gr.Button("Add", size="sm", scale=1)
                music_status = gr.Markdown("")
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
                with gr.Row():
                    suggest_sound_btn = gr.Button("Suggest a soundscape",
                                                  size="sm")
                    suggest_music_btn = gr.Button("Suggest a score", size="sm")
                audio_status = gr.Markdown("")
                gr.Markdown(
                    "Reads the scene and the action above and asks WanGP's "
                    "**Prompt Enhancer** what this should sound like. Each "
                    "button writes its own box and leaves the presets alone. "
                    "They ask separately because one field per request keeps "
                    "the answer on the field that was asked for. Needs the "
                    "enhancer switched on and loaded - any of the Qwen "
                    "options, at whichever quantization you have chosen. "
                    "The Llama captioning enhancers will not follow the "
                    "instruction."
                )

            status = gr.Markdown("")

            with gr.Accordion("Summary", open=False,
                              visible=False) as summary_block:
                gr.Markdown(
                    "The summary is a reference-mode section. Base modes have "
                    "no summary field, so it is not written when reference "
                    "mode is off."
                )
                summary_text = gr.Textbox(
                    label="Summary",
                    lines=3,
                    placeholder="The target video shows a fishmonger preparing and gutting a whole fish at his market counter, featuring <Subject 1> and <Subject 2>. It runs 10 seconds in a live-action, documentary style.",
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

            # Duration lives down here rather than up in Scene, which holds
            # what is true of the whole clip - this is per-window, and you
            # rarely know it until the action is written. Where a field is
            # built has nothing to do with where it sits in the flat input
            # list below, so moving it here costs nothing: the flat list,
            # _unpack, _clear and _restore_draft are all untouched.
            with gr.Row():
                duration = gr.Number(
                    label="Length of this window (seconds)",
                    value=8.0, minimum=0.5, step=0.5,
                )
                window_commands = gr.Checkbox(
                    label="Write scheduling into the prompt",
                    value=False,
                    info="Adds /duration, and /overlap or /new_shot on the "
                         "appends. These override Sliding Window Size and "
                         "Number of frames. Leave off to drive it all from "
                         "WanGP's own sliders.",
                )

            with gr.Row():
                insert_btn = gr.Button("Insert into prompt", variant="primary")
            with gr.Row():
                append_cont_btn = gr.Button("Append continuing window")
                append_cut_btn = gr.Button("Append as new shot")
            with gr.Row():
                clear_action_btn = gr.Button("Clear the action")
                clear_btn = gr.Button("Clear all fields")

            gr.Markdown(
                "**Insert into prompt** replaces the box. The two **Append** "
                "buttons add this window below what is already there, "
                "separated by a blank line - build one window, append it, "
                "then write the next. Which one you press decides how the "
                "windows meet: **continuing** carries the closing frames and "
                "audio of the previous window over as conditioning, for when "
                "the action runs straight on; **new shot** carries nothing, "
                "for a hard cut to somewhere else. This is separate from a "
                "cut *inside* a window, which is what a second `[Shot 2]` "
                "with a timestamp already does.\n\n"
                "For any of it to work, set *How to Process each Line of the "
                "Text Prompt* to the paragraph-per-sliding-window option; on "
                "the default queue setting each window becomes a separate job."
            )

        # ---- wiring -------------------------------------------------------

        flat = [start_image, end_image,
                ref_mode, duration, window_commands, style, grading,
                location, time_of_day, lighting, atmosphere,
                camera_type,
                ref_count]
        for r in refs:
            flat += [r["slot"], r["role"], r["desc"], r["retention"],
                     r["from"]]
        flat += [entry_count]
        for e in entries:
            flat += [e["kind"], e["desc"], e["speaker"], e["onscreen"],
                     e["age"], e["gender"], e["pitch"], e["timbre"],
                     e["rate"], e["accent"], e["lang"],
                     e["source"], e["retention"], e["note"],
                     e["voice_from"], e["motion_from"]]
        # The action is one field, so the list stops growing here - no
        # shot_count, no MAX_SHOTS x MAX_BEATS block behind it.
        flat += [task_types, summary_text, action_text]
        flat += [ambience_from, ambience_retention, soundscape,
                 music_from, music_role, music_retention, music]

        # Every insert button reads the whole form, so it can resolve a
        # speaker ID or a subject's voice without separate wiring. Its own
        # controls ride on the end, which is why the handlers slice from the
        # back rather than the front.
        action_out = [action_text, action_prev, action_status]

        ins_shot.click(fn=self._insert_shot, inputs=flat, outputs=action_out)
        ins_time.click(fn=self._insert_time, inputs=flat + [at_seconds],
                       outputs=action_out)
        ins_camera.click(
            fn=self._insert_camera,
            inputs=flat + [cam_verb, cam_framing, cam_lens, cam_motion,
                           cam_ampl, cam_speed, cam_rig, cam_anchor],
            outputs=action_out,
        )
        # Writes into the Of box rather than the action: it is building the
        # phrase Add camera will then use, so it has to land before the
        # sentence is assembled.
        # Reads the whole form like the other buttons do - it needs the cast
        # to know which label number the entry will be given.
        ins_anchor_subject.click(
            fn=self._insert_anchor_subject,
            inputs=flat + [cam_anchor, cam_subject],
            outputs=[cam_anchor, action_status],
        )
        ins_dialogue.click(
            fn=self._insert_dialogue,
            inputs=flat + [dl_who, dl_type, dl_lang, dl_delivery, dl_speech,
                           dl_carries, dl_carry_phrase, dl_cutoff],
            outputs=action_out,
        )
        ins_screen.click(fn=self._insert_screen_text,
                         inputs=flat + [st_kind, st_text], outputs=action_out)
        undo_action.click(fn=self._undo_action, inputs=[action_prev],
                          outputs=action_out)
        clear_action_btn.click(fn=self._clear_action, inputs=[action_text],
                               outputs=action_out)

        # The enhancer switch rides on the end of the list, like every other
        # button's own controls.
        suggest_sound_btn.click(
            fn=self._suggest_soundscape, inputs=flat,
            outputs=[soundscape, audio_status],
        )
        suggest_music_btn.click(
            fn=self._suggest_music, inputs=flat,
            outputs=[music, audio_status],
        )

        insert_btn.click(fn=self._build, inputs=flat,
                         outputs=[self.prompt, status])

        # Two buttons rather than a join field: they stay out of the flat
        # list, so the count and the four sites that read it do not move.
        append_cont_btn.click(fn=self._append_continuing,
                              inputs=[self.prompt] + flat,
                              outputs=[self.prompt, status])
        append_cut_btn.click(fn=self._append_cut,
                             inputs=[self.prompt] + flat,
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
        # Order matters: this has to match the order _clear and _apply_saved
        # return their visibility updates in.
        all_groups = (
            [e["group"] for e in entries]
            + [e["ref_block"] for e in entries]
            + [r["group"] for r in refs]
            + [ambience_ref_block, music_ref_block, summary_block]
        )

        # Reference task is a tab people never open if they are working only
        # inside the subject blocks, and an unticked task type is invisible
        # until the summary comes out without its prefix. So every control
        # that implies one keeps it in step.
        #
        # Deliberately not wired to task_types itself: the handler writes to
        # that box, and a change event on it would feed straight back.
        # What the last sync ticked, so a later one can untick exactly what
        # it put there and nothing else. Session plumbing, not a field: it
        # never enters the flat list or the draft.
        derived_tasks = gr.State([])

        task_triggers = [start_image, end_image, ref_mode, ref_count]
        for r in refs:
            task_triggers += [r["slot"], r["role"], r["retention"]]
        for e in entries:
            task_triggers += [e["source"], e["voice_from"], e["motion_from"]]

        # A description now implies the task type too, so it has to be a
        # trigger - but on blur rather than change. The rest of these are
        # dropdowns that fire once; a textbox fires on every keystroke, and
        # the sync reads the whole form. Leaving the box is the moment the
        # description is worth acting on.
        for e in entries:
            box = e["desc"]
            if hasattr(box, "blur"):
                box.blur(fn=self._sync_tasks,
                         inputs=[derived_tasks] + flat,
                         outputs=[task_types, derived_tasks])
        task_triggers += [ambience_from, ambience_retention,
                          music_from, music_retention]
        for control in task_triggers:
            control.change(fn=self._sync_tasks,
                           inputs=[derived_tasks] + flat,
                           outputs=[task_types, derived_tasks])

        # The presets fill the boxes and are not read anywhere else, so they
        # stay out of the flat list - like the camera and dialogue controls,
        # they ride on the end of their own button.
        add_soundscape_btn.click(
            fn=self._add_soundscape_preset,
            inputs=[soundscape, soundscape_presets],
            outputs=[soundscape, soundscape_status],
        )
        add_music_btn.click(
            fn=self._add_music_preset,
            inputs=[music, music_instruments, music_tempo, music_dynamics],
            outputs=[music, music_status],
        )

        # ---- the draft ----------------------------------------------------
        #
        # Saved after every press that changes something, which covers the
        # moments most work has just been done, and on a timer for the typing
        # in between. gr.Timer arrived in Gradio 4.x; without it the button
        # presses alone still cover the expensive-to-retype parts.
        save_draft_btn.click(fn=self._save_draft, inputs=flat,
                             outputs=[draft_status])
        restore_draft_btn.click(fn=self._restore_draft, inputs=[],
                                outputs=flat + all_groups + [draft_status])

        for button in (ins_shot, ins_time, ins_camera, ins_dialogue,
                       ins_screen, undo_action, clear_action_btn):
            button.click(fn=self._autosave, inputs=flat,
                         outputs=[draft_status])

        if hasattr(gr, "Timer"):
            autosave = gr.Timer(AUTOSAVE_SECONDS)
            autosave.tick(fn=self._autosave, inputs=flat,
                          outputs=[draft_status])

        # Loading writes the whole form, exactly as restoring a draft does.
        form_out = flat + all_groups + [prompt_status]
        save_prompt_btn.click(fn=self._save_prompt,
                              inputs=[prompt_name] + flat,
                              outputs=[prompt_picker, prompt_status])
        load_prompt_btn.click(fn=self._load_prompt, inputs=[prompt_picker],
                              outputs=form_out)
        load_recent_btn.click(fn=self._load_recent_prompt, inputs=[],
                              outputs=form_out)
        prompt_file.change(fn=self._load_prompt_file, inputs=[prompt_file],
                           outputs=form_out)
        refresh_prompts_btn.click(fn=self._refresh_prompts, inputs=[],
                                  outputs=[prompt_picker, prompt_status])

        # Both Clear buttons do the same thing - one at the top for starting
        # over, one at the bottom where the build controls are. Clearing is
        # saved too, so a crash after it cannot resurrect the old form.
        for button in (clear_btn, clear_top_btn):
            button.click(fn=self._clear, inputs=[],
                         outputs=flat + all_groups)
            button.click(fn=self._save_draft, inputs=flat,
                         outputs=[draft_status])

        for button in (insert_btn, append_cont_btn, append_cut_btn):
            button.click(fn=self._autosave, inputs=flat,
                         outputs=[draft_status])

        # Character-creator fields aren't part of _build/_clear's flat list
        # at all - they only ever feed the per-entry "Add to description"
        # button, so they need their own reset here.
        char_fields, char_blocks = [], []
        for e in entries:
            char_fields += [e["use_creator"], e["char_name"],
                            e["char_ethnicity"], e["char_gender"],
                            e["char_age"], e["char_height"], e["char_build"],
                            e["char_hairstyle"], e["char_haircolor"],
                            e["char_eyecolor"], e["char_clothing"]]
            char_blocks.append(e["creator_block"])

        def _reset_char_fields():
            per_entry = [False, "", "", "", "", "", "", "", "", "", ""]
            return per_entry * len(entries) + [gr.update(visible=False)] * len(char_blocks)

        for button in (clear_btn, clear_top_btn):
            button.click(fn=_reset_char_fields, inputs=[],
                         outputs=char_fields + char_blocks)

        # Show each entry's name next to its Subject number in the dialogue
        # Who boxes, purely as a memory aid when juggling several subjects -
        # the stored value stays "Subject N" and the final prompt is
        # unaffected.
        #
        # The update is returned bare, not wrapped in a list. There is one
        # output now that the per-beat dropdowns are gone, and Gradio reads a
        # returned list as that component's value - which for a CheckboxGroup
        # is exactly what a value looks like, so the update dict lands in the
        # selection instead of the choices and the next press fails to
        # preprocess.
        if char_names:
            def _sync_who_labels(*names):
                choices = []
                for n in range(MAX_ENTRIES):
                    nm = (names[n] or "").strip() if n < len(names) else ""
                    label = f"Subject {n + 1} ({nm})" if nm else f"Subject {n + 1}"
                    choices.append((label, f"Subject {n + 1}"))
                # Two outputs now, so a list is read as one update each -
                # the ambiguity the note above warns about only arises when
                # there is a single output to land in.
                return [gr.update(choices=choices)] * 2

            for name_field in char_names:
                name_field.change(
                    fn=_sync_who_labels, inputs=char_names,
                    outputs=[dl_who, cam_subject],
                )

        self._wire_model_visibility(root, model_warning)

    @staticmethod
    def _startup_model_type():
        """
        The model WanGP is about to open with, read from its own settings.

        create_ui runs while the Blocks are being built, and on_model_change
        does not fire until the page has loaded, so at build time the plugin
        has not been told which model is selected. Starting hidden on that
        basis meant the builder was missing whenever WanGP opened straight
        onto an H3 model, and only appeared once you switched away and back.

        server_config.json holds last_model_type, which is exactly what
        WanGP restores on startup. Read from disk rather than by importing
        wgp: it is a settings file rather than an internal, and the plugin
        is in no position to reach into the host's globals.

        Returns "" if it cannot be found, and the caller treats that as
        "show the builder" - being unable to tell is not a reason to hide.
        """
        for folder in Path(__file__).resolve().parents:
            candidate = folder / "server_config.json"
            if candidate.is_file():
                try:
                    blob = json.loads(candidate.read_text(encoding="utf-8"))
                    return str(blob.get("last_model_type") or "")
                except Exception:                     # noqa: BLE001
                    return ""
        return ""

    def _starts_visible(self):
        """
        Whether the builder should be on screen before anything has fired.

        Hidden only when the startup model is known and is not H3. If the
        model cannot be determined, or visibility cannot be wired at all,
        it stays visible - a builder that never appears is worse than one
        that appears where it is not needed, and the toggle corrects the
        latter on the first model change.
        """
        if not self._can_wire_visibility():
            return True
        startup = self._startup_model_type() or self.current_model_type
        if not startup:
            return True
        return any(h in startup.lower() for h in H3_MODEL_HINTS)

    def _can_wire_visibility(self):
        """Whether the pieces the visibility toggle needs are all present."""
        return bool(getattr(self, "state", None)) and any(
            getattr(self, name, None) is not None
            for name in ("refresh_form_trigger", "model_choice_target"))

    def _wire_model_visibility(self, root, warning):
        """
        Hide the builder unless a MiniMax H3 model is selected.

        Two things make this awkward. on_model_change() is notification-only,
        because PluginManager's dispatcher discards whatever it returns. And
        the visible model dropdown is created after WanGP snapshots its
        locals for the plugin component registry, so it cannot be requested.

        What is available is `state`, whose "model_type" is set immediately
        before the model-change notification, plus two hidden trigger
        components that fire when the form refreshes or a model switch is
        requested. Binding to both and reading the model out of state covers
        either path. A failure here leaves the builder visible rather than
        breaking the plugin.
        """
        state = getattr(self, "state", None)
        triggers = [t for t in (getattr(self, "refresh_form_trigger", None),
                                getattr(self, "model_choice_target", None))
                    if t is not None]

        if state is None or not triggers:
            print("[H3PromptBuilder] could not wire model visibility; "
                  "the builder will stay visible for every model.")
            return

        def _toggle(fired, st):
            # model_choice_target holds the model being switched *to*, and
            # fires before the switch happens - at which point state still
            # holds the model being switched *from*. Reading state there got
            # the answer exactly backwards: switching to H3 saw the old
            # model and hid the builder, switching away from H3 saw H3 and
            # showed it. So the trigger's own value wins when it carries a
            # model id.
            #
            # refresh_form_trigger carries a counter rather than a name,
            # which is what the letter test sorts out; that path falls
            # through to state, which by then has been updated.
            seen = str(fired or "")
            model_type = seen if any(c.isalpha() for c in seen) else ""
            if not model_type and isinstance(st, dict):
                model_type = str(st.get("model_type", "") or "")
            model_type = model_type or self.current_model_type
            is_h3 = any(h in model_type.lower() for h in H3_MODEL_HINTS)
            return gr.update(visible=is_h3), gr.update(visible=False)

        for trigger in triggers:
            try:
                trigger.change(fn=_toggle, inputs=[trigger, state],
                               outputs=[root, warning])
            except Exception as exc:
                print(f"[H3PromptBuilder] visibility trigger not wired: {exc}")

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
    def _scene_clause(cls, location, lighting, atmosphere, time_of_day=""):
        """
        Location, lighting and atmosphere are visual description, so they are
        woven into the opening shot rather than emitted as their own fields.
        """
        # "The setting is X" avoids the preposition trap: "in a rooftop at
        # dusk" is wrong, but "the setting is a rooftop at dusk" works for
        # interiors and exteriors alike.
        if location:
            clause = f"The setting is {location} {time_of_day}".rstrip()
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
                        camera_type, duration=None, grading="",
                        time_of_day=""):
        """
        One sentence establishing the whole clip, written before [Shot 1]:
        style, setting, hour, light, air and camera body.
        """
        head = f"A {style[0].lower() + style[1:]} scene" if style else "A scene"

        parts = [head]
        if location:
            # The hour rides along with the setting rather than taking a
            # clause of its own: "set in an office at night", not "set in an
            # office, at night".
            parts.append(f"set in {location} {time_of_day}".rstrip())
        elif time_of_day:
            parts.append(f"taking place {time_of_day}"
                         if not time_of_day.startswith(("at ", "in "))
                         else time_of_day)
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
    def _sentence_case(cls, text):
        """Stand a field's text up as one or more finished sentences."""
        text = cls._s(text)
        if not text:
            return ""
        text = text[0].upper() + text[1:]
        return text if text.endswith((".", "!", "?")) else text + "."

    @staticmethod
    def _compose_music(instruments, tempo, dynamics):
        """
        One sentence across the guide's three axes, in its own order:
        instrumentation, then tempo, then dynamic development.
        """
        parts = [p.strip() for p in (instruments, tempo, dynamics)
                 if isinstance(p, str) and p.strip()]
        if not parts:
            return ""
        head, rest = parts[0], parts[1:]
        if not rest:
            sentence = head
        elif len(rest) == 1:
            sentence = f"{head} {rest[0]}"
        else:
            # "A solo piano at a slow tempo, swelling gradually" - the
            # development clause is an aside, so it takes a comma.
            sentence = f"{head} {rest[0]}, {rest[1]}"
        sentence = sentence[0].upper() + sentence[1:]
        return sentence if sentence.endswith(".") else sentence + "."

    @classmethod
    def _append_sentence(cls, existing, addition):
        """
        Add a composed sentence to whatever is already in the box.

        The box is the field, not a footnote to the dropdowns, so pressing
        Add twice builds it up rather than replacing what is there - and
        adding the same sentence twice is a no-op rather than a repetition.
        """
        existing = (existing or "").strip()
        if not addition:
            return existing, False
        if addition.rstrip(".").lower() in existing.rstrip(".").lower():
            return existing, False
        return (f"{existing} {addition}".strip() if existing else addition), True

    @classmethod
    def _add_music_preset(cls, existing, instruments, tempo, dynamics):
        """Compose the three dropdowns into the score box."""
        sentence = cls._compose_music(instruments, tempo, dynamics)
        if not sentence:
            return gr.update(), "Pick an instrument, a tempo or a development."
        text, added = cls._append_sentence(existing, sentence)
        if not added:
            return gr.update(), "That is already in the box."
        return text, "Added to the score."

    @classmethod
    def _add_soundscape_preset(cls, existing, presets):
        """Compose the chosen ambience presets into the soundscape box."""
        chosen = [p for p in (presets or []) if cls._s(p)]
        if not chosen:
            return gr.update(), "Pick a preset first."
        sentence = f"The scene carries {cls._oxford_join(chosen)}."
        text, added = cls._append_sentence(existing, sentence)
        if not added:
            return gr.update(), "That is already in the box."
        return text, "Added to the soundscape."

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
    def _slot_list(cls, value):
        """
        A multiselect source field as a list of bare slot names.

        The field accepts a list from the dropdown or a typed string, and
        entries may already be wrapped in angle brackets, so all three forms
        come back the same way.
        """
        if isinstance(value, (list, tuple)):
            parts = [str(v) for v in value]
        else:
            parts = (cls._s(value) or "").split(",")
        return [p.strip().strip("<>") for p in parts if p and p.strip()]

    @staticmethod
    def _slot_kind(slot):
        """
        Which asset family a slot name belongs to.

        Read off the name rather than tracked, because the name is what the
        user picked and what the prompt will say. An unrecognised name is
        treated as a picture, which is the marker set most references use.
        """
        head = (slot or "").strip("<>").split()[0].lower() if slot else ""
        return {"video": "video", "audio": "audio",
                "picture": "picture"}.get(head, "picture")

    @classmethod
    def _slot_order(cls, slot):
        """
        Sort assets by family, then by number rather than by string.

        Plain sorting puts "Picture 10" between "Picture 1" and "Picture 2",
        which reads as a mistake in a finished prompt.
        """
        kind = cls._slot_kind(slot)
        rank = {"picture": 0, "video": 1, "audio": 2}.get(kind, 3)
        digits = "".join(c for c in (slot or "") if c.isdigit())
        return (rank, int(digits) if digits else 0, slot or "")

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

    @staticmethod
    def _window_seconds(duration):
        """The duration field as a positive float, or None if unusable."""
        try:
            secs = float(duration)
        except (TypeError, ValueError):
            return None
        return secs if secs > 0 else None

    @staticmethod
    def _secs_text(secs):
        """8.0 -> '8', 7.5 -> '7.5'. WanGP parses the seconds form as a
        float, so there is nothing to round to."""
        return f"{secs:.3f}".rstrip("0").rstrip(".")

    @classmethod
    def _window_command(cls, duration, join=None, enabled=True):
        """
        The `[/...]` block WanGP reads at the head of a window's paragraph.

        WanGP strips the block before the model sees it, so this is
        scheduling instruction rather than prompt text. Three things about
        it are worth knowing:

        - It goes on every window, including a single one. WanGP pads the
          video out to Number of frames by repeating the last paragraph
          only when *no* window carries a `/duration`, so all or none are
          the only safe choices and all is the one that honours the field.
        - It overrides both Sliding Window Size and Number of frames for
          the generation, and is capped by neither.
        - `join` is left off the first window. Its overlap is clamped to
          whatever a start image or Continue Video source provides, which
          on plain text-to-video is nothing, so a command there would
          imply a control the prompt does not have.

        The overlap is written out as a number rather than as a bare
        `[/overlap]`, which would mean "whatever the slider says" and
        leave a saved prompt at the mercy of a setting changed months
        later.
        """
        # Off by default. These commands override WanGP's own sliders, and
        # taking that over should be something asked for rather than
        # something that happens because the plugin was used.
        if not enabled:
            return ""
        secs = cls._window_seconds(duration)
        if secs is None:
            return ""
        parts = [f"/duration={cls._secs_text(secs)}s"]
        if join == "continue":
            parts.append(f"/overlap={H3_OVERLAP_DEFAULT}")
        elif join == "cut":
            # /new_shot is an alias for /overlap=0. A cut with overlap is
            # not expressible, and should not be: the overlap frames are
            # the previous window's closing frames handed to the model as
            # picture and audio conditioning, so asking it to cut away
            # from them at the same time is a contradiction.
            parts.append("/new_shot")
        return "[" + ",".join(parts) + "]"

    @classmethod
    def _shot_boundaries(cls, action):
        """
        Timestamps that open a `[Shot N]`, offered as candidate split
        points when a window is long.

        Where a split actually goes is a creative decision, and rebasing
        the timestamps, renumbering the shots and deciding whether the new
        join is a continuation or a cut all belong to whoever is writing
        it. So this points at the boundaries and stops there.
        """
        text = action or ""
        found = []
        for m in _ACTION_SHOT_RE.finditer(text):
            stamp = _ACTION_TIME_RE.match(text[m.end():m.end() + 40].lstrip())
            if stamp:
                mark = f"{stamp.group(1)}:{stamp.group(2)}"
                if mark not in found:
                    found.append(mark)
        return found

    @classmethod
    def _window_warnings(cls, duration, action=""):
        """
        What this window's length means for H3, with the arithmetic shown.

        Advisory, like everything else here. These are the current model's
        limits, not the parser's - `/duration` has no ceiling at all - so a
        long window is a decision about quality and VRAM rather than
        something that can be got wrong. Saying so with the numbers
        attached beats refusing.
        """
        secs = cls._window_seconds(duration)
        if secs is None:
            return []
        frames = int(round(secs * H3_FPS))
        low, high = H3_WINDOW_DOC_SECONDS
        shown = cls._secs_text(secs)
        out = []

        if secs > high:
            if frames > H3_WINDOW_FRAMES_MAX:
                note = (f"**{shown}s is {frames} frames**, past the "
                        f"{H3_WINDOW_FRAMES_MAX} "
                        f"({cls._secs_text(H3_WINDOW_FRAMES_MAX / H3_FPS)}s) "
                        "the Sliding Window Size slider allows at all")
            else:
                note = (f"**{shown}s is {frames} frames**, past the "
                        f"{cls._secs_text(high)}s MiniMax documents for one "
                        "window")
            marks = cls._shot_boundaries(action)
            if marks:
                note += (" - shot boundaries at "
                         + cls._oxford_join([f"`{m}`" for m in marks])
                         + " if you would rather split it")
            out.append(note)
        elif secs < low:
            out.append(
                f"**{shown}s is {frames} frames**, under the "
                f"{cls._secs_text(low)}s MiniMax documents - H3 generates "
                f"{H3_WINDOW_FRAMES_MIN} frames minimum, so it may raise the "
                "overlap above what is asked for to fill them"
            )
        return out

    @classmethod
    def _instruction(cls, start_image, end_image, duration, shot_count):
        """
        The keyframe instruction line, reproduced verbatim from the guide.

        Note the guide's own inconsistency: the start-and-end form uses plain
        "Picture 1" and "Shot 1", while the other two use <Picture 1> and
        [Shot 1]. That is copied as written rather than tidied, since this is
        boilerplate the model was trained on. N is the actual final shot and
        S.SS is the duration to exactly two decimal places.
        """
        try:
            secs = f"{float(duration):.2f}"
        except (TypeError, ValueError):
            secs = "0.00"
        last = max(1, int(shot_count or 1))

        if start_image and end_image:
            return ("How the reference pictures align with the target video "
                    "\u2014 Picture 1 (from Shot 1) aligns with the 0.00-second "
                    "mark of the target video; Picture 2 (from Shot "
                    f"{last}) aligns with the {secs}-second mark of the "
                    "target video.")
        if start_image:
            return ("For the target video, at 0.00 seconds into the target "
                    "video, <Picture 1> (from [Shot 1]) is fully referenced.")
        if end_image:
            return ("How the reference pictures align with the target video "
                    f"\u2014 <Picture 1> (from [Shot {last}]) aligns with the "
                    f"{secs}-second mark of the target video.")
        return ""

    @staticmethod
    def _article(phrase):
        """'an' or 'a' based on the phrase's first letter sound."""
        phrase = (phrase or "").strip()
        return "an" if phrase and phrase[0].lower() in "aeiou" else "a"

    @staticmethod
    def _oxford_join(items):
        """'A' / 'A and B' / 'A, B and C' - no serial comma, matching the
        style used elsewhere in the assembled prompt."""
        items = [i for i in items if i]
        if not items:
            return ""
        if len(items) == 1:
            return items[0]
        return ", ".join(items[:-1]) + f" and {items[-1]}"

    @classmethod
    def _character_description(cls, name, ethnicity, gender, age_range,
                               height, build, hairstyle, haircolor, eyecolor,
                               clothing=""):
        """
        Compose a physical-description sentence from the character-creator
        fields. Every part is optional and the grammar adjusts to whatever is
        filled in - dropping a field removes it cleanly rather than leaving a
        gap or a stray comma.
        """
        name = cls._s(name)
        ethnicity = cls._s(ethnicity)
        gender = cls._s(gender)
        age_range = cls._s(age_range)
        height = cls._s(height)
        build = cls._s(build)
        hairstyle = cls._s(hairstyle)
        haircolor = cls._s(haircolor)
        eyecolor = cls._s(eyecolor)
        clothing = cls._s(clothing)

        # "male"/"female" already work as bare nouns ("a male"). Anything
        # else in the gender field - non-binary, androgynous - is really an
        # adjective and needs "person" to attach to.
        if gender.lower() in ("male", "female"):
            noun = gender
        elif gender:
            noun = f"{gender} person"
        else:
            noun = "person"
        core = f"{ethnicity} {noun}" if ethnicity else noun
        core_phrase = f"{cls._article(core)} {core}"

        # Age, with a pronoun derived from gender rather than typed by hand.
        if age_range:
            low = gender.lower()
            pronoun = "his" if low == "male" else "her" if low == "female" else "their"
            head = f"{core_phrase} in {pronoun} {age_range}"
        else:
            head = core_phrase

        # Hair. "bald"/"shaved head" become noun phrases ("a bald head") so
        # they sit in the list alongside "a muscular build" and "brown eyes"
        # as parallel items, rather than reading as a stray adjective.
        hair_phrase = ""
        if hairstyle or haircolor:
            low_style = hairstyle.lower()
            if low_style == "bald":
                hair_phrase = "a bald head"
            elif low_style == "shaved head":
                hair_phrase = "a shaved head"
            else:
                bits = [b for b in [hairstyle, haircolor] if b]
                if bits:
                    hair_phrase = " ".join(bits) + " hair"

        # Physical attributes as one list, each a bare noun phrase. "with" is
        # added once, in front of the whole list, so it is correct whichever
        # subset is present - build alone, eyes alone, or all three.
        attr_items = []
        if build:
            b_phrase = build if "build" in build.lower() else f"{build} build"
            attr_items.append(f"{cls._article(b_phrase)} {b_phrase}")
        if hair_phrase:
            attr_items.append(hair_phrase)
        if eyecolor:
            attr_items.append(f"{eyecolor} eyes")
        attributes_clause = (f"with {cls._oxford_join(attr_items)}"
                             if attr_items else "")

        # Height. Words like "tall" or "petite" already stand alone; a bare
        # measurement ("six feet") needs "tall" appended.
        if height:
            markers = ("tall", "short", "petite", "height", "statuesque")
            height_clause = (height if any(mk in height.lower() for mk in markers)
                             else f"{height} tall")
        else:
            height_clause = ""

        trailing = " ".join(p for p in [height_clause, attributes_clause] if p)

        # Clothing goes last, after the physical description, so the
        # sentence reads as a person first and an outfit second. A value
        # that already carries its own verb is used as written.
        if clothing:
            worn = ("wearing", "dressed", "clad", "in ")
            clothing_clause = (clothing
                               if clothing.lower().startswith(worn)
                               else f"wearing {clothing}")
        else:
            clothing_clause = ""

        parts = [p for p in [head, trailing, clothing_clause] if p]
        body = ", ".join(parts)

        if not body and not name:
            return ""

        if name:
            sentence = f"{name}, {body}" if body else name
        else:
            sentence = body[0].upper() + body[1:]

        sentence = sentence.rstrip(".")
        return sentence + "." if sentence else ""

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
            # People write this field several ways: bare ("West Country"),
            # with the word ("a West Country accent"), or with a synonym
            # ("a soft Irish lilt"). Only add "accent" when nothing already
            # names the kind of thing it is.
            markers = ("accent", "lilt", "brogue", "drawl", "twang", "burr",
                       "inflection", "cadence", "dialect", "intonation")
            low = accent.lower()
            phrase = accent if any(mk in low for mk in markers) else f"{accent} accent"
            parts.append(f"with {phrase}" if not bits else f"and {phrase}")
        presence = cls._s(e.get("onscreen"))
        if presence == "off-screen":
            parts.append("heard off-screen")
        return " ".join(parts)

    @staticmethod
    def _article(phrase):
        """"a" or "an" for a phrase we are assembling rather than storing."""
        return "an" if (phrase or "")[:1].lower() in "aeiou" else "a"

    @classmethod
    def _shot_phrase(cls, framing, rig):
        """
        Framing and rig as one noun phrase: "a medium crane shot".

        Both are adjectives and "shot" is added once, wherever the stack
        ends - after the framing alone, after the rig alone, or after both.
        """
        words = [w for w in (cls._s(framing), cls._s(rig)) if w]
        if not words:
            return ""
        phrase = " ".join(words) + " shot"
        return f"{cls._article(phrase)} {phrase}"

    @classmethod
    def _camera_clause(cls, motion, ampl, speed, rig=None):
        """
        The movement, as its own sentence.

        The rig is no longer written here. It belongs to the shot phrase now
        - "a dolly shot" - so carrying it into the movement as well would
        say it twice.
        """
        motion = cls._s(motion)
        if not motion:
            return ""
        verb = MOTION_VERBS.get(motion, motion.lower())
        clause = f"The camera {verb}"
        extras = [e for e in [cls._s(ampl), cls._s(speed)] if e]
        if extras:
            clause += " " + " ".join(extras)
        return clause + "."

    @classmethod
    def _beat_text(cls, beat, label_for, lang_for=None, lang_default="English",
                   speaker_info=None):
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
        # Voiceover has required phrasing in the spec: the exact clause
        # "says in an off-screen voiceover", and a statement immediately
        # after the </d> block that the on-screen character's lips stay shut.
        voiceover = (cls._s(beat.get("type")).lower() == "voiceover"
                     and bool(speech))
        if voiceover:
            # Whatever delivery was typed is kept as a preceding action, but
            # a trailing "says" is dropped so it cannot collide with the
            # required phrase and produce "says quietly and says in an ...".
            trimmed = re.sub(r"[\s,]*\b(?:and\s+)?says?\b[\s,]*$", "",
                             action, flags=re.I).strip(" ,")
            if trimmed:
                lead += " " + trimmed[0].lower() + trimmed[1:] + " and"
            # The spec's phrase is "says in an off-screen voiceover"; several
            # speakers sharing one line take the plural verb, matching the
            # guide's own group-speech examples.
            verb = "say" if len(picked) > 1 else "says"
            lead += f" {verb} in an off-screen voiceover"
        elif action:
            lead += " " + action[0].lower() + action[1:]

        if speech:
            sentence = f"{lead}: <d>[{lang}] {speech}</d>"
            if voiceover:
                info = {}
                for p in picked:
                    info = (speaker_info or {}).get(cls._s(p), {})
                    if info:
                        break
                # The guide's clause is about the on-screen character. A
                # speaker explicitly marked off-screen has no visible face,
                # so writing about their lips would describe nothing.
                if info.get("onscreen") != "off-screen":
                    if len(picked) > 1:
                        pronoun = "their"
                    else:
                        g = (info.get("gender") or "").lower()
                        pronoun = ("his" if g == "male"
                                   else "her" if g == "female" else "their")
                    sentence += (f" while {pronoun} lips remain completely "
                                 "closed.")
            # The lips clause is specified as immediately following the <d>
            # block, so the continuity tags sit after it.
            if beat.get("cutoff"):
                sentence += "<cutoff>"
            if beat.get("carries"):
                # Tag at the connecting point, then the continuity stated in
                # words. The receiving shot writes its own half.
                phrase = (cls._s(beat.get("carry_phrase"))
                          or CARRY_PHRASES[0])
                sentence += f"<scenetrans> The speech {phrase.rstrip('.')}."
            return sentence

        return lead + ("" if lead.endswith(".") else ".")

    @classmethod
    def _screen_text_clause(cls, kind, screen_text):
        """
        Text actually readable on screen. The guide asks for it verbatim
        inside English double quotes and never translated, so whatever is
        typed is quoted as written - any quotes already around it are
        stripped first rather than doubled up.
        """
        screen_text = cls._s(screen_text).strip().strip('"').strip("'").strip()
        if not screen_text:
            return ""
        kind = cls._s(kind) or "a sign"
        kind = kind[0].upper() + kind[1:]
        return f'{kind} reading "{screen_text}" is visible in the frame.'

    @classmethod
    def _scan_action(cls, text):
        """What the action field currently says about its own structure."""
        text = text or ""
        marks = [(m.start(), int(m.group(1)))
                 for m in _ACTION_SHOT_RE.finditer(text)]
        numbers = [n for _, n in marks]
        tail = text[marks[-1][0]:] if marks else ""

        # An emitting <scenetrans> is one that is not the receiving half
        # written by a previous Shot press.
        emitting = False
        for m in _ACTION_TRANS_RE.finditer(tail):
            following = tail[m.end():m.end() + 48].lstrip()
            if not following.lower().startswith("the speech carries over"):
                emitting = True
                break

        body = _ACTION_SHOT_RE.sub("", tail)
        return {
            "numbers": numbers,
            "count": len(marks),
            "next": (max(numbers) + 1) if numbers else 1,
            "tail": tail,
            "camera_in_shot": bool(_ACTION_CAMERA_RE.search(body)
                                   or _ACTION_FRAMING_RE.search(body)),
            "carry_pending": emitting,
        }

    @staticmethod
    def _append_to_line(text, addition):
        """Everything except Shot continues the line already being written."""
        text = (text or "").rstrip()
        if not text:
            return addition
        return f"{text} {addition}"

    @classmethod
    def _speaker_labels(cls, d):
        """
        Subject N -> (Sx), plus the voice details voiceover needs.

        The buttons always write the bare speaker ID. Which form it takes in
        the finished prompt - an inline description in base modes, a
        <Subject N> label in reference mode - is decided at build time, so
        the reference switch keeps working after the action is written.
        """
        labels, info, missing = {}, {}, []
        for idx, e in enumerate(d["entries"][:d["entry_count"]]):
            key = f"Subject {idx + 1}"
            speaker = cls._s(e["speaker"])
            if speaker:
                labels[key] = f"({speaker})"
            elif cls._s(e["desc"]):
                missing.append(key)
            info[key] = {
                "gender": cls._s(e["gender"]),
                "onscreen": cls._s(e["onscreen"]),
            }
        return labels, info, missing

    @classmethod
    def _entry_labels(cls, d):
        """
        The label the build will give each entry, worked out the same way.

        The numbers are not the entry numbers. An entry with neither a
        description nor a source asset never gets a definition, so it is
        skipped and everything below it moves up - and the count is per
        kind, so a <Location 1> does not consume a Subject number. Insert
        subject and the warnings both read the cast through here rather
        than assuming the two numbering schemes agree.

        Returns {"Subject 2": "<Subject 1>"} - entry key to written label.
        """
        labels, counters = {}, {}
        for idx, e in enumerate(d["entries"][:d["entry_count"]]):
            desc = cls._s(e["desc"])
            source = cls._ref_tags(e["source"]) if d["ref_mode"] else ""
            if not desc and not source:
                continue
            kind = cls._s(e["kind"]) or "Subject"
            counters[kind] = counters.get(kind, 0) + 1
            labels[f"Subject {idx + 1}"] = f"<{kind} {counters[kind]}>"
        return labels

    @classmethod
    def _known_speaker_ids(cls, d):
        """The speaker IDs the cast actually defines, for validation."""
        ids = []
        for e in d["entries"][:d["entry_count"]]:
            speaker = cls._s(e["speaker"])
            if speaker and speaker not in ids:
                ids.append(speaker)
        return ids

    # -- insert buttons -----------------------------------------------------

    @classmethod
    def _insert_shot(cls, *values):
        d = cls._unpack(values)
        text = (d["action"] or "").rstrip()
        info = cls._scan_action(text)

        line = f"[Shot {info['next']}]"
        note = ""
        if info["carry_pending"]:
            # The other half of a line crossing the cut. The guide asks for
            # the tag at both connecting points and the continuity stated in
            # words, and the previous shot only wrote its own half.
            line += f" <scenetrans>{CARRY_RECEIVE_TEXT}"
            note = " The carried line was picked up."

        new = f"{text}\n{line}" if text else line
        return new, text, f"Added Shot {info['next']}.{note}"

    @classmethod
    def _insert_time(cls, *values):
        d = cls._unpack(values)
        seconds = values[-1]
        text = (d["action"] or "").rstrip()
        if seconds is None or cls._s(str(seconds)) == "":
            return gr.update(), gr.update(), "Set **At (seconds)** first."
        try:
            stamp = f"At {cls._timecode(float(seconds))},"
        except (TypeError, ValueError):
            return gr.update(), gr.update(), "That is not a number of seconds."
        return cls._append_to_line(text, stamp), text, f"Added {stamp[:-1]}."

    @classmethod
    def _insert_camera(cls, *values):
        d = cls._unpack(values)
        (verb, framing, lens, motion, ampl, speed, rig,
         anchor) = values[-8:]
        text = (d["action"] or "").rstrip()
        info = cls._scan_action(text)

        verb = cls._s(verb)
        if not verb:
            # Blank lets the field decide, and it has three cases. A second
            # camera press inside the same shot is a move, not a cut. A new
            # shot after an earlier one is a cut. The opening shot is neither
            # - there is nothing before it to cut from, so it takes no
            # transition verb at all and simply states what is on screen.
            if info["camera_in_shot"]:
                verb = CONTINUE_VERBS[0]
            elif info["count"] > 1:
                verb = CUT_VERBS[0]
            else:
                verb = ""
        framing = cls._s(framing)
        lens = cls._s(lens)
        rig = cls._s(rig)
        # A full stop typed at the end of the anchor would leave the lens
        # and rig hanging off a finished sentence - "...without a scratch.,
        # on a crane." - which reads as a fragment and makes the dangling
        # worse.
        anchor = cls._s(anchor).rstrip().rstrip(".").rstrip()

        sentences = []
        shot = cls._shot_phrase(framing, rig)
        if shot or anchor:
            # "The camera cuts to a medium hood-mounted shot on a 35mm lens
            # of a car driving at speed." The lens sits between the shot and
            # what it is pointed at, so both read as describing the shot
            # rather than as things in the scene.
            if verb and shot:
                head = f"{verb} {shot}"
            elif shot:
                # No verb: the shot phrase opens the sentence itself.
                head = shot
            elif verb:
                head = f"{verb} {anchor}"
                anchor = ""
            else:
                # An anchor with nothing framing it still needs a verb, or
                # the sentence is a bare noun phrase.
                head = f"the frame holds {anchor}"
                anchor = ""
            if lens:
                head += f" on {lens}"
            if anchor:
                head += f" of {anchor}"
            sentences.append(head[0].upper() + head[1:] + ".")
        elif lens:
            sentences.append(f"Shot on {lens}.")

        camera = cls._camera_clause(motion, ampl, speed)
        if camera:
            sentences.append(camera)

        if not sentences:
            return (gr.update(), gr.update(),
                    "Pick a framing, lens, rig or motion first.")
        return (cls._append_to_line(text, " ".join(sentences)), text,
                "Added the camera.")

    @classmethod
    def _insert_anchor_subject(cls, *values):
        """
        Append a subject label to the camera's Of field.

        Writes the finished <Subject N> tag, which is what H3 reads. The
        number is the one the build will use rather than the entry number
        on the accordion: _entry_labels skips the entries that never get a
        definition and counts per kind, so picking the third accordion can
        correctly write <Subject 2>.

        That label is fixed once written. Rearranging the cast afterwards
        can leave it pointing at the wrong definition, so _action_warnings
        checks the labels in the action against the cast at build time and
        says so rather than letting it through quietly.
        """
        d = cls._unpack(values)
        anchor, subject = values[-2:]
        subject = cls._s(subject)
        anchor = (anchor or "").rstrip()
        if not subject:
            return gr.update(), "Pick a subject first."

        if not d["ref_mode"]:
            # Base modes write no subject_definitions, so there is no
            # definition for a label to point at - H3 reads <Subject 1> as
            # literal text. Identity is written inline there, which is what
            # the speaker ID already does.
            return (gr.update(),
                    "Subject labels only exist in reference mode. Turn on "
                    "**Use reference assets**, or describe the subject in "
                    "the Of box and let the speaker ID carry it.")

        label = cls._entry_labels(d).get(subject)
        if not label:
            return (gr.update(),
                    f"**{subject}** has no description or source asset "
                    "yet, so it gets no definition to point at.")

        if not anchor:
            return label, f"Added {label}."
        # Half-written phrasing - "a wide shot of", "the table, and" - is
        # waiting for the subject, so it takes a space. Anything else reads
        # as a finished item in the list the field holds, so it takes the
        # comma that separates them.
        if anchor.endswith((",", ":", ";")):
            joined = f"{anchor} {label}"
        else:
            tail = anchor.rsplit(" ", 1)[-1].lower()
            joined = (f"{anchor} {label}" if tail in ANCHOR_CONNECTIVES
                      else f"{anchor}, {label}")
        return joined, f"Added {label}."

    @classmethod
    def _insert_dialogue(cls, *values):
        d = cls._unpack(values)
        (who, dtype, lang, delivery, speech, carries, carry_phrase,
         cutoff) = values[-8:]
        text = (d["action"] or "").rstrip()

        who = [w for w in (who or []) if cls._s(w)]
        speech = cls._s(speech)
        delivery = cls._s(delivery)
        if not (speech or delivery):
            return (gr.update(), gr.update(),
                    "Write some speech or a delivery first.")

        labels, info, missing = cls._speaker_labels(d)
        unassigned = [w for w in who if w not in labels]
        if unassigned:
            names = ", ".join(unassigned)
            return (gr.update(), gr.update(),
                    f"{names} has no **Speaker** ID. Assign one in Cast & "
                    "subjects, since the line needs something to attribute "
                    "it to.")

        beat = {
            "type": cls._s(dtype) or "dialogue",
            "speaker": who, "lang": cls._s(lang), "action": delivery,
            "speech": speech, "at": None,
            "carries": bool(carries) and bool(speech),
            "cutoff": bool(cutoff) and bool(speech),
            "carry_phrase": cls._s(carry_phrase),
        }
        line = cls._beat_text(beat, labels, {}, speaker_info=info)
        if not line:
            return gr.update(), gr.update(), "Nothing to add."
        return cls._append_to_line(text, line), text, "Added the dialogue."

    @classmethod
    def _insert_screen_text(cls, *values):
        d = cls._unpack(values)
        kind, screen_text = values[-2:]
        text = (d["action"] or "").rstrip()
        clause = cls._screen_text_clause(kind, screen_text)
        if not clause:
            return gr.update(), gr.update(), "Type the visible text first."
        return cls._append_to_line(text, clause), text, "Added the visible text."

    @classmethod
    def _undo_action(cls, previous):
        if not previous:
            return gr.update(), "", "Nothing to undo."
        return previous, "", "Reverted the last insert."

    @classmethod
    def _clear_action(cls, action):
        """
        Clear the action for the next sliding window, leaving cast, scene,
        audio and summary alone - those usually carry over, the action does
        not.

        A line still carrying at the end of the window is the one thing that
        should survive, because the window boundary is itself a cut and the
        guide wants the tag at both connecting points. Rather than remember
        a flag - the one piece of hidden state this design refuses - the
        pick-up half is derived here and written back as ordinary visible
        text, where it can be read and deleted like anything else.
        """
        previous = (action or "").rstrip()
        if not previous:
            return "", "", "The action is already empty."

        if cls._scan_action(previous)["carry_pending"]:
            return (f"[Shot 1] <scenetrans>{CARRY_RECEIVE_TEXT}", previous,
                    "Cleared. A line was still carrying, so its pick-up half "
                    "was left in place - delete it if the next window does "
                    "not follow straight on.")
        return "", previous, "Cleared the action."

    # -- validation ---------------------------------------------------------

    @classmethod
    def _d_tag_problem(cls, text):
        """
        Walk the <d> tags left to right rather than counting them. Counting
        is not enough: "<d>a<d>b</d></d>" balances but nests, and the model
        reads the inner pair as the spoken words. Returns a problem in words,
        or "" when the tags are sound.

        Also used to check the enhancer's rewrite of a shot, which is the
        other place a </d> goes missing.
        """
        depth = 0
        for m in _ACTION_D_TAG_RE.finditer(text or ""):
            if m.group(0) == "<d>":
                depth += 1
                if depth > 1:
                    return "a `<d>` opened inside another `<d>`"
            else:
                depth -= 1
                if depth < 0:
                    return "a `</d>` with no `<d>` opening it"
        if depth > 0:
            return f"{depth} unclosed `<d>`"
        return ""

    @classmethod
    def _action_warnings(cls, action, duration, known_ids=None,
                         known_labels=None):
        """
        Things that will generate but not do what was meant. Warnings only -
        a freeform field is allowed to be a work in progress, and blocking
        an insert because a tag is half-typed would be worse than the tag.
        """
        action = action or ""
        problems = []

        tags = cls._d_tag_problem(action)
        if tags:
            problems.append(tags)

        # The spec wants the language named inside the tag. The buttons
        # always write it; a hand-written line often will not.
        untagged = [b for b in _ACTION_D_BLOCK_RE.findall(action)
                    if not _ACTION_D_LANG_RE.match(b)]
        if untagged:
            problems.append(f"{len(untagged)} `<d>` block(s) with no "
                            "`[Language]` tag")

        info = cls._scan_action(action)
        numbers = info["numbers"]
        if not numbers:
            problems.append("no `[Shot 1]` marker")
        else:
            duplicates = sorted({n for n in numbers if numbers.count(n) > 1})
            if duplicates:
                problems.append("repeated shot numbers "
                                + ", ".join(str(n) for n in duplicates))
            if numbers != sorted(numbers):
                problems.append("shot numbers out of order")
            expected = list(range(1, len(numbers) + 1))
            if sorted(numbers) != expected and not duplicates:
                problems.append("gaps in the shot numbering")

        # A shot marker with nothing after it generates nothing and spends a
        # cut doing it.
        marks = [m for m in _ACTION_SHOT_RE.finditer(action)]
        empty = []
        for n, m in enumerate(marks):
            end = marks[n + 1].start() if n + 1 < len(marks) else len(action)
            if not action[m.end():end].strip():
                empty.append(m.group(1))
        if empty:
            problems.append("empty shot " + ", ".join(empty))

        stamps = [float(m.group(1)) * 60 + float(m.group(2))
                  for m in _ACTION_TIME_RE.finditer(action)]
        try:
            limit = float(duration or 0)
        except (TypeError, ValueError):
            limit = 0
        if limit and any(s > limit for s in stamps):
            problems.append("a timestamp past the clip duration")
        if stamps != sorted(stamps):
            problems.append("timestamps out of order")

        # A stamp in the wrong shape is read by nothing - not the ordering
        # check above, not the duration check, and not the model.
        loose = []
        for m in _ACTION_LOOSE_TIME_RE.finditer(action):
            found = m.group(0).rstrip(",")
            if found not in loose:
                loose.append(found)
        if loose:
            problems.append("`" + "`, `".join(loose) + "` is not in the "
                            "`At 00:00.000,` form the model reads")

        # A carried line needs its other half. The receiving text is written
        # by the Shot button, so a missing one means the cut was never made.
        emits, receives, inherited = 0, 0, 0
        for m in _ACTION_TRANS_RE.finditer(action):
            following = action[m.end():m.end() + 48].lstrip().lower()
            if following.startswith("the speech carries over"):
                # A pick-up before anything has been carried belongs to the
                # window before this one - Clear the action leaves it there
                # on purpose, and its emitting half is in the previous
                # window, where this field cannot see it.
                if emits == 0 and receives == 0:
                    inherited = 1
                receives += 1
            else:
                emits += 1
        if emits != receives - inherited:
            problems.append(f"{emits} carried line(s) but "
                            f"{receives - inherited} pick-up(s)")

        # <cutoff> means the clip ends mid-line, so there is one and it is
        # last. Anywhere else it is describing something that cannot happen.
        cutoffs = action.count("<cutoff>")
        if cutoffs > 1:
            problems.append(f"{cutoffs} `<cutoff>` marks, but only the end "
                            "of the clip can cut a line off")
        elif cutoffs == 1 and not action.rstrip().endswith("<cutoff>"):
            problems.append("`<cutoff>` is not at the end of the action")

        # On-screen text is quoted verbatim, so an odd quote count means one
        # is left open and the quotation swallows what follows.
        if action.count('"') % 2:
            problems.append("an odd number of double quotes - some visible "
                            "text is left unclosed")

        # Retention analysis reads the action to find where each subject
        # appears, so a subject written in by hand without its speaker ID
        # will be missed. An ID with no cast entry is the reverse: it
        # survives into the prompt as a bare (Sx), meaning nothing.
        if known_ids is not None:
            unknown = []
            for m in _ACTION_SPEAKER_RE.finditer(action):
                for one in [p.strip() for p in m.group(1).split(",")]:
                    if one and one not in known_ids and one not in unknown:
                        unknown.append(one)
            if unknown:
                problems.append(", ".join(f"`({u})`" for u in unknown)
                                + " has no entry in Cast & subjects, so it "
                                "is written through as-is")

        # A label written into the action by Insert subject is fixed, and
        # the cast can be edited after it. A tag with no definition behind
        # it points at nothing, which is worth saying before a generation
        # rather than after one.
        if known_labels is not None:
            dangling = []
            for m in _ACTION_LABEL_RE.finditer(action):
                tag = f"<{m.group(1)} {m.group(2)}>"
                if tag not in known_labels and tag not in dangling:
                    dangling.append(tag)
            if dangling:
                problems.append(", ".join(f"`{t}`" for t in dangling)
                                + " has no definition behind it - "
                                "subject_definitions is only written in "
                                "reference mode, and an edit to the cast "
                                "can leave an inserted label stranded")

        return problems

    @classmethod
    def _bind_speakers(cls, text, label_for, label_after, key_by_id):
        """
        The buttons write the bare speaker ID. The finished prompt needs the
        subject's identity at first mention and a short reference after, and
        which form that takes depends on the mode - an inline description in
        base modes, a <Subject N> label in reference mode.

        Resolving it here rather than baking it in at insert time is what
        lets the reference switch keep working after the action is written.
        """
        seen = set()

        def swap(match):
            num = match.group("num")
            if num:
                # A {Subject N} mention names the entry directly, so there
                # is no ID to look up - but an entry that was left blank
                # never got a label, and writing the token through is
                # better than writing a label that belongs to someone else.
                ids, keys = [], [f"Subject {num}"]
                if keys[0] not in label_for:
                    return match.group(0)
            else:
                ids = [p.strip() for p in match.group("ids").split(",")
                       if p.strip()]
                keys = [key_by_id.get(i) for i in ids]
            if not keys or any(k is None for k in keys):
                return match.group(0)
            unseen = [k for k in keys if k not in seen]
            table = label_for if unseen else label_after
            parts = [table.get(k) or match.group(0) for k in keys]
            seen.update(keys)
            if len(parts) == 1:
                out = parts[0]
            else:
                # Several speakers on one line share one compound ID, so the
                # trailing (Sx) is stripped off each and written once at the
                # end.
                joined = ",".join(ids)
                bare = []
                for p in parts:
                    stripped = _ACTION_TRAILING_ID_RE.sub("", p).strip()
                    if stripped:
                        bare.append(stripped)
                out = (f"({joined})" if not bare
                       else f"{cls._oxford_join(bare)} ({joined})")

            # "(S1) sits back" is how the button writes it, and in base modes
            # that (S1) becomes a description which then opens the sentence.
            # The fixed-slot version had _beat_text to capitalise for it;
            # substituting into freeform text, nothing else will.
            if out and cls._opens_a_sentence(match.string, match.start()):
                out = out[0].upper() + out[1:]
            return out

        return _ACTION_BIND_RE.sub(swap, text or "")

    @staticmethod
    def _opens_a_sentence(text, pos):
        """Whether position `pos` is the start of a sentence."""
        before = (text or "")[:pos].rstrip()
        if not before:
            return True
        before = before.rstrip('"\u201d\u2019\')')
        if not before:
            return True
        # ">" covers a preceding tag: <scenetrans> and </d> both end a
        # sentence's worth of markup.
        return before.endswith((".", "!", "?", ":", ";", ">", "\n"))

    @classmethod
    def _subject_shots(cls, action, speaker_id, label=None):
        """
        Which shots a subject appears in, for retention_analysis.

        A subject can be in shot without speaking in it - framed by the
        camera, or named in the action - so the speaker ID is not the only
        way in. The written label counts too, which is what lets the Of
        field's Insert subject earn its place in the scope line.
        """
        patterns = []
        if speaker_id:
            patterns.append(re.compile(r"\((?:S\d+\s*,\s*)*"
                                       + re.escape(speaker_id)
                                       + r"(?:\s*,\s*S\d+)*\)"))
        if cls._s(label):
            patterns.append(re.compile(re.escape(cls._s(label)),
                                       re.I))
        if not patterns:
            return []
        found, current = [], None
        for line in (action or "").split("\n"):
            m = _ACTION_SHOT_RE.search(line)
            if m:
                current = int(m.group(1))
            if not current or current in found:
                continue
            if any(p.search(line) for p in patterns):
                found.append(current)
        return found

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

        start_image = bool(take())
        end_image = bool(take())
        ref_mode = bool(take())
        duration = take()
        window_commands = bool(take())
        style = cls._s(take())
        grading = cls._s(take())
        location = cls._s(take())
        time_of_day = cls._s(take())
        lighting = cls._s(take())
        atmosphere = cls._s(take())
        camera_type = cls._s(take())

        ref_count = int(cls._s(take()) or 0)
        refs = []
        for _ in range(MAX_REFS):
            refs.append({
                "slot": cls._s(take()), "role": cls._s(take()),
                "desc": cls._s(take()), "retention": cls._s(take()),
                "from": cls._s(take()),
            })

        entry_count = int(cls._s(take()) or 0)
        entries = []
        for _ in range(MAX_ENTRIES):
            entries.append({
                "kind": take(), "desc": take(), "speaker": take(),
                "onscreen": take(), "age": take(), "gender": take(),
                "pitch": take(), "timbre": take(), "rate": take(),
                "accent": take(), "lang": take(), "source": take(), "retention": take(), "note": take(),
                "voice_from": take(), "motion_from": take(),
            })

        task_types = take()
        summary_text = cls._s(take())
        # One freeform field for the whole action. There is no shot_count to
        # read: the number of shots is counted out of the text itself, by the
        # same scan the Shot button uses to pick its next number.
        action = cls._s(take())

        ambience_from = cls._s(take())
        ambience_retention = cls._s(take())
        soundscape = cls._s(take())
        music_from = cls._s(take())
        music_role = cls._s(take())
        music_retention = cls._s(take())
        music = cls._s(take())

        return dict(
            start_image=start_image, end_image=end_image,
            ref_mode=ref_mode, duration=duration,
            window_commands=window_commands, style=style,
            grading=grading, location=location, time_of_day=time_of_day,
            lighting=lighting, atmosphere=atmosphere, camera_type=camera_type,
            ref_count=ref_count, refs=refs,
            entry_count=entry_count, entries=entries,
            task_types=task_types, summary_text=summary_text,
            action=action,
            ambience_from=ambience_from, ambience_retention=ambience_retention,
            soundscape=soundscape,
            music_from=music_from, music_role=music_role,
            music_retention=music_retention,
            music=music,
        )

    @classmethod
    def _append_window(cls, existing, *values, join):
        """
        Append this prompt below whatever is already in the box, separated by
        one blank line. That blank line is the window separator when the
        prompt-processing mode is set to paragraph-per-sliding-window, and
        each assembled prompt has no blank lines of its own, so the boundary
        is unambiguous.

        join is "continue" or "cut", from whichever button was pressed. It
        is dropped when the box is empty: the first window's overlap is
        clamped to whatever a start image or Continue Video source
        provides, so there is nothing for a join to decide there.
        """
        first = not (existing or "").strip()
        built, status = cls._build(*values, join=None if first else join)
        if not built.strip():
            return existing, "Nothing to append yet."

        existing = (existing or "").rstrip()
        if first:
            return built, status.replace("Prompt written.",
                                         "First window written.")

        extra = cls._join_warnings(existing, values, join)
        combined = existing + "\n\n" + built
        windows = combined.count("\n\n") + 1
        head = ("Appended as window "
                f"{windows}, {'continuing' if join == 'continue' else 'cutting'}"
                " from the one before. ")
        tail = status.split(". ", 1)[-1]
        if extra:
            # The build's own warnings, if it had any, are already in tail.
            joined = "; ".join(extra)
            tail = (f"Also: {joined}. {tail}" if tail.startswith("Read ")
                    else f"{tail} Also: {joined}.")
        return combined, head + tail

    @classmethod
    def _join_warnings(cls, previous, values, join):
        """
        Problems visible only at the seam between two windows.

        The continuation signal is one-way. A `<cutoff>` in the window
        before means its last line is still running, which proves the two
        windows are connected; nothing proves the opposite, since a shot
        can carry on in silence. So this warns about cutting away from an
        unfinished line and stays quiet otherwise.
        """
        out = []
        if join == "cut" and "<cutoff>" in (previous or ""):
            out.append("the window before ends on a `<cutoff>`, so its line "
                       "is still running - a new shot cuts away from it "
                       "mid-sentence")

        # Picture numbering restarts inside every window, and from the
        # second one the carried frame has already taken <Picture 1>.
        # Whether an end image is actually attached is generator state this
        # cannot see, so this says what changes rather than rewriting it.
        d = cls._unpack(values)
        if d["start_image"] or d["end_image"]:
            if join == "continue":
                out.append("from window 2 the overlap frame is `<Picture 1>`, "
                           "so an end image here is `<Picture 2>` - check the "
                           "instruction line reads the way you meant")
            else:
                out.append("picture numbering restarts in each window - after "
                           "a `/new_shot` there is no carried frame, so an "
                           "end image here is `<Picture 1>`")
        return out

    @classmethod
    def _append_continuing(cls, existing, *values):
        return cls._append_window(existing, *values, join="continue")

    @classmethod
    def _append_cut(cls, existing, *values):
        return cls._append_window(existing, *values, join="cut")

    @classmethod
    def _speaking_ids(cls, action):
        """Speaker IDs attached to a <d> block, for the summary draft."""
        text = action or ""
        found = []
        for m in _ACTION_D_BLOCK_RE.finditer(text):
            # The attribution sits before the tag, in the same sentence.
            head = text[:m.start()]
            cut = max(head.rfind(". "), head.rfind("\n"))
            for sm in _ACTION_SPEAKER_RE.finditer(head[cut + 1:]):
                for one in [p.strip() for p in sm.group(1).split(",")]:
                    if one and one not in found:
                        found.append(one)
        return found

    @classmethod
    def _camera_sentence_anchor(cls, sentence):
        """
        What was in frame, pulled back out of a camera sentence.

        The button wrote "{verb} {framing} of {anchor}, on {lens}, {rig}.",
        so the anchor is what sits between "of" and the lens or rig clause -
        and it is the only part of a camera sentence that says anything at
        all about what a scene sounds like.
        """
        body = sentence.strip().rstrip(".")
        low = body.lower()
        if " of " not in low:
            return ""
        body = body[low.index(" of ") + 4:]
        idx = body.lower().find(", on ")
        if idx > 0:
            body = body[:idx]
        keep = []
        rigs = {r.lower() for r in RIGS if r}
        for part in [p.strip() for p in body.split(",")]:
            if part.lower() in rigs:
                break
            keep.append(part)
        return ", ".join(p for p in keep if p).strip()

    @classmethod
    def _action_sound_lines(cls, action):
        """
        The action reduced to what a sound editor can use: what is in frame,
        what happens, and where someone speaks.

        Camera work is dropped on purpose - framing, lens, motion and rig say
        nothing about what a scene sounds like, and feeding them in pulls the
        model towards describing the shot instead of the sound. The spoken
        words go too: quoting them invites the model to echo dialogue back
        into a field that must not contain any.
        """
        marks = list(_ACTION_SHOT_RE.finditer(action or ""))
        if not marks:
            chunks = [(1, action or "")]
        else:
            chunks = []
            for n, m in enumerate(marks):
                end = marks[n + 1].start() if n + 1 < len(marks) else len(action)
                chunks.append((int(m.group(1)), action[m.end():end]))

        out = []
        for number, chunk in chunks:
            speaks = bool(_ACTION_D_BLOCK_RE.search(chunk))
            chunk = _ACTION_CARRY_SENTENCE_RE.sub(" ", chunk)
            chunk = _ACTION_LIPS_RE.sub(".", chunk)
            chunk = _ACTION_D_BLOCK_RE.sub(" ", chunk)
            chunk = _ENH_TAG_RE.sub(" ", chunk)
            chunk = _ACTION_SPEAKER_RE.sub(" ", chunk)
            # Removing a trailing clause can leave the colon that introduced
            # it stranded against the full stop.
            chunk = re.sub(r"\s*[:;,]\s*\.", ".", chunk)

            lines = []
            for raw in re.split(r"(?<=[.!?])\s+", chunk):
                sentence = " ".join(raw.split()).strip()
                if not sentence:
                    continue
                if (_ACTION_CAMERA_RE.search(sentence)
                        or sentence.lower().startswith("the shot is")):
                    sentence = cls._camera_sentence_anchor(sentence)
                    if not sentence:
                        continue
                # Continuity prose and the closed-lips clause describe how
                # the cut and the mouth behave, not what anything sounds
                # like, and the model will happily score them if left in.
                low = sentence.lower()
                if low.startswith("the speech "):
                    continue
                if low.startswith("while ") and "lips" in low:
                    continue
                sentence = sentence.strip(" ,:;")
                # A stamp is timing, not sound.
                sentence = _ACTION_TIME_RE.sub("", sentence).strip(" ,")
                if not sentence:
                    continue
                sentence = sentence[0].upper() + sentence[1:]
                if not sentence.endswith((".", "!", "?")):
                    sentence += "."
                lines.append(sentence)
            if speaks:
                lines.append("Someone speaks aloud here.")
            if lines:
                out.append((number, " ".join(lines)))
        return out

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

        action = d["action"]
        shot_count = max(1, cls._scan_action(action)["count"])
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
        # The anchor is no longer a field of its own, so the lead comes from
        # the setting instead - which is what a human overview opens with
        # anyway.
        opening = cls._s(d["location"])
        if opening and cls._s(d["time_of_day"]):
            opening += f" {cls._s(d['time_of_day'])}"

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

        voices = cls._speaking_ids(action)
        if voices:
            label = "one speaker" if len(voices) == 1 else f"{len(voices)} speakers"
            sentences.append(f"There is dialogue from {label}.")

        if shot_count > 1:
            sentences.append(f"It runs to {shot_count} shots.")

        used = []
        for r in d["refs"][:d["ref_count"]]:
            slot = cls._s(r["slot"]).strip("<>")
            if not slot:
                continue
            role = cls._s(r["role"])
            what = cls._s(r["desc"])
            phrase = f"<{slot}>"
            if role:
                phrase += f" as {role}"
            if what:
                phrase += f", providing {what}"
            used.append(phrase)
        if used:
            sentences.append("It draws on " + cls._oxford_join(used) + ".")

        if not opening and not named and not used:
            sentences = []

        if not sentences:
            # A source video is only a draftable input in reference mode, so
            # only offer it as a suggestion there.
            missing = ("a location, a subject, or a source video"
                       if d["ref_mode"] else "a location or a subject")
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
    def _audio_digest(cls, d):
        """
        What the enhancer reads. Two labelled blocks, because the two fields
        answer different questions: a soundscape follows from where the scene
        is and what happens in it, a score follows from style and from
        whatever score direction has already been given.

        Camera work is left out on purpose. Framing, lens, motion and rig say
        nothing about what a scene sounds like, and feeding them in pulls the
        model towards describing the shot instead of the sound.
        """

        style = cls._s(d["style"])
        location = cls._s(d["location"])
        time_of_day = cls._s(d["time_of_day"])
        atmosphere = cls._s(d["atmosphere"])

        scene = []
        if style:
            scene.append(f"Style: {style}.")
        if location:
            scene.append(f"Setting: {location}.")
        # Dawn and midnight imply very different sound in the same room, so
        # the hour is worth stating even though the setting already is.
        if time_of_day:
            scene.append(f"Time: {time_of_day}.")
        if atmosphere:
            scene.append(f"Atmosphere: {atmosphere}.")

        shot_lines = cls._action_sound_lines(d["action"])
        for number, text in shot_lines:
            scene.append(f"Shot {number}: {text}")

        # Nothing about where it is or what happens means nothing to listen
        # for, and a style on its own would only get invention back.
        if not (location or atmosphere or shot_lines):
            return ""

        # Only what is in the boxes. The preset dropdowns write into them
        # rather than sitting alongside, so there is one place a choice can
        # live and the model is shown exactly what the prompt will say.
        sound_note = cls._s(d["soundscape"])
        sound = []
        if sound_note:
            sound.append(f"Already written: {sound_note.rstrip('.')}.")
        else:
            sound.append("Nothing chosen. Work it out from the scene.")

        music_note = cls._s(d["music"])
        music = []
        if style:
            music.append(f"Style: {style}.")
        if music_note:
            music.append(f"Already written: {music_note.rstrip('.')}.")
        else:
            music.append("Nothing chosen. Work out a score that suits the scene.")

        return ("SCENE\n" + "\n".join(scene)
                + "\n\nSOUNDSCAPE DIRECTION\n" + "\n".join(sound)
                + "\n\nSCORE DIRECTION\n" + "\n".join(music))

    @classmethod
    def _ask_for_audio_field(cls, values, which):
        """
        One field at a time. The combined prompt asked the enhancer to hold
        two jobs in mind at once, and the weaker ones drift between them -
        score notes turning up in the soundscape, or the reverse. Asking
        separately costs a second press and reliably answers the question.

        Returns (text, status). text is "" when nothing usable came back.
        """
        d = cls._unpack(values)
        digest = cls._audio_digest(d)
        if not digest:
            return "", ("Nothing to work from yet - add a **location**, an "
                        "**atmosphere** or some **action** first.")

        prompt = (SOUNDSCAPE_ONLY_PROMPT if which == "soundscape"
                  else MUSIC_ONLY_PROMPT)
        raw, note = _run_enhancer(prompt, digest)
        if raw is None:
            # The probe line is worth showing: it says which of the three
            # routes to the enhancer were open, which is the only way to tell
            # a configuration problem from a version one without a terminal.
            return "", f"No suggestion: {note}.\n\n*{_probe_report()}*"

        sound, music = _parse_audio_reply(raw)
        got = sound if which == "soundscape" else music
        if not got:
            # A single-field prompt often gets a bare sentence back with no
            # label at all, which the prose fallback files under soundscape
            # whichever field was asked for.
            got = sound or music
        if not got:
            retry, retry_note = _run_enhancer(AUDIO_RETRY_PROMPT, digest,
                                              max_new_tokens=512)
            if retry:
                raw, note = retry, retry_note
                sound, music = _parse_audio_reply(raw)
                got = (sound if which == "soundscape" else music) or sound

        if which == "music" and got.strip().lower().rstrip(".") in (
                "none", "no score", "no music", "n/a", "silence"):
            return "", ("The enhancer suggested no score for this scene, so "
                        "the field is untouched.")

        if not got:
            # Showing the reply is the whole point here - a parse failure is
            # otherwise indistinguishable from the enhancer misbehaving, and
            # the raw text says immediately which it was.
            print("[H3 Prompt Builder] audio suggestion did not parse. "
                  "Raw reply:\n" + (raw or "<empty>"))
            excerpt = " ".join((raw or "").split())[:300] or "<empty reply>"
            return "", ("The enhancer replied but nothing usable parsed out "
                        "of it. The full reply is in the console; it starts:"
                        f"\n\n> {excerpt}")

        return got, (f"Wrote the {which} ({note}). Presets were left alone - "
                     "read it before you build, and edit freely.")

    @classmethod
    def _suggest_soundscape(cls, *values):
        text, status = cls._ask_for_audio_field(values, "soundscape")
        return (text if text else gr.update()), status

    @classmethod
    def _suggest_music(cls, *values):
        text, status = cls._ask_for_audio_field(values, "music")
        return (text if text else gr.update()), status

    # Roles that mean something other than "referenced". Everything else in
    # REF_ROLE_PHRASES is a reference of one kind or another.
    TASK_ROLE_HINTS = {
        "continued from": "video continuation",
        "edited": "video editing",
        "edited frame": "video editing",
        "keyframe": "keyframe completion",
    }

    # Audio markers that mean the clip itself is carried into the output
    # rather than guiding what gets generated.
    AUDIO_REUSE_MARKERS = ("fully_copy", "partially_copy")

    @classmethod
    def _soundtrack_slot(cls, d, video):
        """
        The audio slot carrying a given video's soundtrack.

        A declared Reference sources row wins - an audio row whose *taken
        from* names this video. You can see the WanGP panel and the plugin
        cannot, so if you have said which slot it occupies, that is the
        answer.

        Otherwise fall back to the positional default described at
        SOUNDTRACK_FOLLOWS_VIDEO_ORDER. It is a default rather than a rule:
        nothing here blocks a pairing that disagrees with it, because the
        behaviour it is read from could change and a plugin that writes
        prompt text is in no position to enforce anything anyway.
        """
        want = (video or "").strip("<>").lower()
        for r in d["refs"][:d["ref_count"]]:
            slot = (r["slot"] or "").strip("<>")
            origin = (r["from"] or "").strip("<>")
            if slot and origin.lower() == want and "audio" in slot.lower():
                return slot

        if SOUNDTRACK_FOLLOWS_VIDEO_ORDER:
            digits = "".join(c for c in (video or "") if c.isdigit())
            if digits:
                return f"Audio {int(digits)}"
        return ""

    @classmethod
    def _implied_tasks(cls, values):
        """
        The task types the rest of the form already implies.

        Reference task is a tab people never open if they are only working
        inside the subject blocks, and an unticked task type is invisible
        until the summary comes out without its prefix. Everything it asks
        for is already stated somewhere else, so it is derived rather than
        remembered - the same reasoning as the shot count and the speaker
        list.
        """
        try:
            d = cls._unpack(values)
        except Exception:                             # noqa: BLE001
            return []
        if not d["ref_mode"]:
            return []

        found = set()
        if d["start_image"] or d["end_image"]:
            found.add("keyframe completion")

        for e in d["entries"][:d["entry_count"]]:
            # A description is enough. The schema is what asks for the
            # prefix, and it is the right one whether the subject is drawn
            # from an asset or invented from the description - the hybrid
            # models read the definitions either way.
            if cls._s(e["desc"]) or cls._slot_list(e["source"]):
                found.add("reference generation")
            if cls._s(e["motion_from"]):
                found.add("reference generation")
            if cls._s(e["voice_from"]):
                # True whether or not the soundtrack row exists yet: the
                # intent is an audio reference either way.
                found.add("audio reference")

        for slot, marker in ((cls._s(d["ambience_from"]),
                              cls._s(d["ambience_retention"])),
                             (cls._s(d["music_from"]),
                              cls._s(d["music_retention"]))):
            if not slot:
                continue
            found.add("audio reuse" if marker in cls.AUDIO_REUSE_MARKERS
                      else "audio reference")

        for r in d["refs"][:d["ref_count"]]:
            slot = cls._s(r["slot"])
            if not slot:
                continue
            role = cls._s(r["role"])
            hint = cls.TASK_ROLE_HINTS.get(role)
            if hint:
                found.add(hint)
            elif cls._slot_kind(slot) == "audio":
                found.add("audio reuse"
                          if cls._s(r["retention"]) in cls.AUDIO_REUSE_MARKERS
                          else "audio reference")
            else:
                found.add("reference generation")

        return [t for t in TASK_TYPES if t in found]

    @classmethod
    def _task_list(cls, value):
        """
        The task field as a list, whichever form it arrives in.

        The checkbox group hands back a list, but a saved form or a restored
        draft can carry the comma-joined string the rest of the assembly
        reads. Treating a string as empty would silently drop ticks that
        were already there, which is the one thing an add-only merge must
        not do.
        """
        if isinstance(value, (list, tuple)):
            items = [str(v) for v in value]
        else:
            items = (cls._s(value) or "").split(",")
        return [p.strip() for p in items if p and p.strip()]

    @classmethod
    def _sync_tasks(cls, derived, *values):
        """
        Bring the task boxes in step with the rest of the form.

        Ticks what the form now implies, and unticks only what an earlier
        sync had ticked for a reason that has since gone away - pick
        Picture 1 on a subject and reference generation appears, remove it
        again and it goes. A box ticked by hand was never in the derived
        set and is never touched, so "video continuation" for a source
        video attached on WanGP's side survives every later edit.

        `derived` is what the last sync was responsible for, carried in a
        gr.State. That is session plumbing rather than a form field: not in
        the flat list, not written to the draft, not visible or editable.
        The rule about the form remembering nothing between presses is
        about the form, and is intact.

        After a reload the set starts empty, so the first sync can only
        add. A tick left stale across that reload would survive until the
        field that implied it is touched, which the "no task type" warning
        still covers.
        """
        try:
            d = cls._unpack(values)
        except Exception:                             # noqa: BLE001
            return gr.update(), cls._task_list(derived)

        previous = cls._task_list(derived)
        current = cls._task_list(d["task_types"])
        implied = cls._implied_tasks(values)

        stale = [t for t in previous if t not in implied]
        kept = [t for t in current if t not in stale]
        merged = kept + [t for t in implied if t not in kept]
        ordered = [t for t in TASK_TYPES if t in merged]

        if ordered == current:
            return gr.update(), implied
        return gr.update(value=ordered), implied

    @classmethod
    def _build(cls, *values, join=None):
        """
        Assemble the prompt for one window.

        join says how this window meets the one before it - None for a
        window with nothing before it, "continue" for an overlap, "cut"
        for a hard cut. It only reaches the `[/...]` block at the head of
        the output; nothing else in the assembly reads it. Gradio calls
        this with the flat list alone, so a plain Insert gets None.
        """
        d = cls._unpack(values)
        start_image = d["start_image"]; end_image = d["end_image"]
        ref_mode = d["ref_mode"]; duration = d["duration"]
        style = d["style"]; grading = d["grading"]
        location = d["location"]; time_of_day = d["time_of_day"]
        lighting = d["lighting"]
        atmosphere = d["atmosphere"]; camera_type = d["camera_type"]
        entry_count = d["entry_count"]; entries = d["entries"]
        task_types = d["task_types"]
        summary_text = d["summary_text"]
        action = d["action"]
        ambience_from = d["ambience_from"]
        ambience_retention = d["ambience_retention"]
        soundscape = d["soundscape"]
        music_from = d["music_from"]; music_role = d["music_role"]
        music_retention = d["music_retention"]
        music = d["music"]

        # The action is one field. Its structure - how many shots, where
        # each subject appears - is read back out of the text rather than
        # tracked alongside it.
        scan = cls._scan_action(action)
        shot_count = max(1, scan["count"])

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
        # Voiceover needs a pronoun for the closed-lips clause and needs to
        # know whether the speaker is visible at all.
        speaker_info = {}       # "Subject 3" -> {"gender": .., "onscreen": ..}
        key_by_id = {}          # "S1" -> "Subject 1", to bind the action text
        skipped = 0
        uses_reference_assets = False

        # An asset can serve several roles. The guide asks for one natural
        # sentence per label rather than a subsection each, so roles are
        # gathered here and written once.
        # The asset registry. Keyed on the slot the user named, because a
        # label belongs to the file rather than to the job: <Video 1> is the
        # first video attached, whatever duties it ends up serving. Two
        # routes write into this - a subject naming an asset, and a
        # Reference sources row - and a slot named by both merges into one
        # definition and one retention line.
        asset_roles = {}     # "Audio 1" -> [role phrase, ...]
        asset_kind = {}      # "Audio 1" -> "audio" | "video" | "picture"
        asset_marker = {}    # "Audio 1" -> retention marker
        asset_desc = {}      # "Audio 1" -> [what it contributes, ...]
        asset_from = {}      # "Audio 1" -> "Video 1", for a soundtrack
        unset_retention = []     # subjects that named an asset but no marker

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
            if speaker:
                key_by_id[speaker] = key
            lang_for[f"Subject {idx + 1}"] = cls._s(e["lang"]) or "English"
            speaker_info[key] = {
                "gender": cls._s(e["gender"]),
                "onscreen": cls._s(e["onscreen"]),
            }

            # Reference assets attached to this entry. The label names the
            # actual slot, so it is never renumbered.
            #
            # A picture named here is not given its own definition: the guide
            # only wants a standalone <Picture N> entry when it is a keyframe,
            # composition anchor, edited frame or storyboard, which is what
            # the Reference sources section is for. A video is different -
            # the guide treats <Video N> as a whole-asset role in its own
            # right, so it earns a line.
            for one in cls._slot_list(e["source"]) if is_ref else []:
                if cls._slot_kind(one) != "video":
                    continue
                asset_roles.setdefault(one, []).append(
                    f"the appearance and content reference for {def_label}")
                asset_kind[one] = "video"
                # A video a character's face and build are lifted from is
                # about the strongest claim there is, so the generic
                # weak_reference fallback - which means loose inspiration -
                # is wrong here. The subject's own marker wins when set.
                asset_marker.setdefault(
                    one, cls._s(e["retention"]) or APPEARANCE_RETENTION)
                if not cls._s(e["retention"]):
                    unset_retention.append(def_label)

            # A voice reference is always an <Audio N>: the guide gives that
            # label to audio referenced for voice, and numbers it
            # independently of video. So a video picked here is resolved to
            # whichever audio slot carries its soundtrack - which is what
            # WanGP does too, spending an audio-reference slot on it. If no
            # row declares that, the warning below says so rather than
            # writing a label the model will not read as audio.
            vfrom = cls._s(e["voice_from"]) if is_ref else ""
            if vfrom and cls._slot_kind(vfrom) == "video":
                origin, vfrom = vfrom, cls._soundtrack_slot(d, vfrom)
                if vfrom:
                    # Say where it came from. The reader and the model both
                    # benefit from knowing this audio is the video's own
                    # track rather than a separate clip that happens to sit
                    # in the next slot.
                    asset_from.setdefault(vfrom, origin.strip("<>"))
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

            if source:
                uses_reference_assets = True
            # A marker is worth writing whether or not an asset is behind
            # it. With one, it says how closely to hold to the asset. With
            # only a description, it says how closely to hold to the
            # definition - which is what the hybrid Ref2VA models read it
            # as, and why a text-only prompt still wants the section.
            marker = cls._s(e["retention"])
            if marker:
                note = cls._s(e["note"])
                # Which shots the subject appears in is read out of the
                # action by its speaker ID, so it cannot go stale when a
                # shot is renumbered or deleted by hand.
                where = cls._subject_shots(action, speaker, label)
                scope = ""
                if where:
                    shot_list = ", ".join(f"[Shot {n}]" for n in where)
                    scope = f" (appears in {shot_list})"
                entry = f"{label}{scope}: {marker}"
                if not note and not source:
                    # Nothing to name as the origin, so the definition is
                    # the thing being held to. Saying so keeps the line a
                    # sentence rather than a bare marker.
                    note = "the appearance and identity given in "
                    note += "subject_definitions"
                if note:
                    entry += f" - {note}"
                retention.append(entry.rstrip(".") + ".")

        # What each entry was labelled, for the warnings below - the same
        # table Insert subject wrote its tags from. Base modes write no
        # definitions at all, so nothing there is a live label and every
        # tag in the action is one to ask about.
        known_labels = set(cls._entry_labels(d).values()) if ref_mode else set()

        # Global scene block, written once before [Shot 1]. Style, location,
        # lighting, atmosphere and camera body hold for the whole clip, so
        # they belong here rather than inside the opening shot.
        opening = cls._opening_clause(style, location, lighting,
                                      atmosphere, camera_type, duration,
                                      grading, time_of_day)

        # The action already reads as finished prose - the buttons wrote it
        # that way. All that is left is to bind the bare (Sx) IDs to the
        # labels this mode calls for, which is what lets the same text serve
        # both schemas.
        bound = cls._bind_speakers(action, label_for, label_after, key_by_id)
        body = "\n".join(l for l in ([opening] + bound.split("\n"))
                          if l.strip())

        # The boxes are the fields now - the presets wrote into them, so
        # there is nothing left to merge here.
        sound_field = cls._sentence_case(soundscape) or "N/A"

        # Empty means no score, which is what N/A says. No separate switch.
        music_field = cls._sentence_case(music) or "N/A"

        if ref_mode:
            for r in d["refs"][:d["ref_count"]]:
                slot = cls._s(r["slot"])
                if not slot:
                    continue
                slot = slot.strip("<>")
                role = cls._s(r["role"])
                phrase = REF_ROLE_PHRASES.get(role, (None, None))[1]
                if not phrase:
                    # An unknown role is written through as typed rather than
                    # dropped - the dropdown takes custom values.
                    phrase = f"the {role} reference" if role else "referenced"
                asset_roles.setdefault(slot, []).append(phrase)
                asset_kind[slot] = cls._slot_kind(slot)
                if cls._s(r["retention"]):
                    asset_marker[slot] = cls._s(r["retention"])
                if cls._s(r["desc"]):
                    asset_desc.setdefault(slot, []).append(cls._s(r["desc"]))
                origin = cls._s(r["from"]).strip("<>")
                if origin:
                    asset_from[slot] = origin

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

        # One line per label, however many roles it serve. A slot named by a
        # subject and by a reference row is one asset doing two jobs, not two
        # assets, so its duties are joined rather than written twice.
        for slot in sorted(asset_roles, key=cls._slot_order):
            # The same duty can arrive by two routes - a subject naming an
            # asset and a Reference sources row saying the same thing - and
            # that is one claim, not two. The two phrasings are not always
            # identical: the subject route adds "for <Subject 1> (S1)" to
            # the same opening words. So a role that another role merely
            # extends is dropped in favour of the more specific one.
            roles = list(dict.fromkeys(asset_roles[slot]))
            roles = [r for r in roles
                     if not any(o != r and o.startswith(r) for o in roles)]
            joined = (", ".join(roles[:-1]) + f" and {roles[-1]}"
                      if len(roles) > 1 else roles[0])
            line = f"<{slot}> is {joined}"
            origin = asset_from.get(slot)
            if origin:
                line += f", taken from <{origin}>"
            what = asset_desc.get(slot)
            if what:
                line += ", supplying " + cls._oxford_join(what)
            defs.append(line.rstrip(".") + ".")
            uses_reference_assets = True

            default = ("reference" if asset_kind.get(slot) == "audio"
                       else "weak_reference")
            marker = asset_marker.get(slot, default)
            detail = cls._oxford_join(what) if what else joined
            retention.append(f"<{slot}>: {marker} - {detail}.")

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
            lines = []
            # First line of the paragraph. WanGP substitutes it out and
            # strips what is left, so the section below moves up to the top
            # before the model ever sees it.
            command = cls._window_command(duration, join,
                                          d["window_commands"])
            if command:
                lines.append(command)
            instruction = cls._instruction(start_image, end_image, duration,
                                           shot_count)
            if instruction:
                lines.append(instruction)
            lines += [f"integrated_multimodal_description: {detailed}",
                      f"overall_soundscape: {sound_field}",
                      f"non_diegetic_music: {music_field}"]
            warnings = cls._action_warnings(action, duration,
                                            cls._known_speaker_ids(d),
                                            known_labels)
            warnings += cls._window_warnings(duration, action)
            if not cls._s(action):
                warnings.append("the **action** is empty")
            check = "Read it through for grammar before generating."
            status = (f"Prompt written. {check}" if not warnings
                      else "Written, but: " + "; ".join(warnings) + f". {check}")
            return cls._no_blank_lines("\n".join(lines)), status

        sections = []
        command = cls._window_command(duration, join,
                                      d["window_commands"])
        if command:
            sections.append(command)
        instruction = cls._instruction(start_image, end_image, duration,
                                       shot_count)
        if instruction:
            sections.append(instruction)

        sections += [
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
        warnings = cls._action_warnings(action, duration,
                                        cls._known_speaker_ids(d),
                                        known_labels)
        warnings += cls._window_warnings(duration, action)
        if not cls._s(action):
            warnings.append("the **action** is empty")
        if unset_retention:
            who = cls._oxford_join([f"**{n}**" for n in
                                    dict.fromkeys(unset_retention)])
            warnings.append(
                f"{who} draws on a reference asset with **what is retained** "
                f"left blank, so `{APPEARANCE_RETENTION}` was assumed - set "
                "it if you meant something else")
        # The Ref2VA schema is worth having without a single asset attached:
        # the hybrid models read subject_definitions and retention_analysis
        # as adherence instructions, so a text-only prompt built here gets
        # tighter character consistency than the three-field base schema.
        # That only works if the markers are actually set, and an empty
        # section looks like a deliberate choice rather than an oversight.
        if defs and not retention and not uses_reference_assets:
            warnings.append("no **retention** markers are set, so "
                            "`retention_analysis` is N/A - setting them "
                            "is what holds the subjects steady when "
                            "there are no reference assets")
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
        # Audio references imply an audio task type.
        audio_slots = {s for s in asset_roles if asset_kind.get(s) == "audio"}
        if audio_slots and "audio" not in prefix.lower():
            warnings.append("audio is referenced but no **audio task type** "
                            "(audio reference / audio reuse) is ticked")

        # WanGP's Audio References selector offers standalone references or
        # reference-video soundtracks, never a mix - so an audio slot lifted
        # from a video and one that stands on its own cannot both be used.
        from_video = {s for s in audio_slots if asset_from.get(s)}
        if from_video and (audio_slots - from_video):
            warnings.append("some audio is taken from a video and some "
                            "stands alone - WanGP's Audio References "
                            "selector lets you pick one or the other")

        check = "Read it through for grammar before generating."
        status = (f"Prompt written. {check}"
                  if not warnings
                  else "Written, but: " + "; ".join(warnings) + f". {check}")
        return cls._no_blank_lines("\n".join(sections)), status

    # -- the draft ----------------------------------------------------------
    #
    # WanGP can go down mid-build, and a half-written prompt is an hour of
    # work. The whole form is written to one JSON file beside plugin.py so it
    # survives a crash, a restart, or a browser reload.
    #
    # The file holds the flat list positionally rather than by field name.
    # Naming every field would be a seventh construction site to keep in step,
    # and getting it wrong there would restore your lens into your anchor -
    # the exact failure the flat list is already prone to. Instead the length
    # is recorded and checked: a draft written by a different version of the
    # plugin is refused outright rather than restored into the wrong slots.

    @staticmethod
    def _draft_path():
        return Path(__file__).resolve().parent / "h3_draft.json"

    @staticmethod
    def _draft_backup_path():
        """The draft the current one replaced, kept one deep."""
        return Path(__file__).resolve().parent / "h3_draft.prev.json"

    @classmethod
    def _flat_len(cls):
        """How many values the flat list carries, derived from _clear."""
        return len(cls._clear()) - CLEAR_GROUP_UPDATES

    @classmethod
    def _form_has_content(cls, values):
        """
        Whether the form holds anything worth keeping.

        Deliberately semantic rather than a comparison against the cleared
        defaults: what a freshly built panel hands back is not guaranteed to
        equal what _clear returns, and getting that wrong in the safe
        direction means silently declining to save real work.
        """
        try:
            d = cls._unpack(values)
        except Exception:                             # noqa: BLE001
            return True          # unreadable is not the same as empty
        if cls._s(d["action"]) or cls._s(d["summary_text"]):
            return True
        for field in ("style", "grading", "location", "time_of_day",
                      "lighting", "atmosphere", "camera_type",
                      "soundscape", "music"):
            if cls._s(d[field]):
                return True
        if cls._s(d["soundscape"]) or cls._s(d["music"]):
            return True
        return any(cls._s(e["desc"]) or cls._s(e["speaker"])
                   for e in d["entries"][:d["entry_count"]])

    @classmethod
    def _autosave(cls, *values):
        """
        The timer and the insert buttons come through here rather than
        straight to _save_draft.

        An empty form is never written. WanGP restarts with every field at its
        default, and the timer would otherwise tick twenty seconds later and
        replace the draft with that blank form before it could be restored -
        which is exactly what it did, once.

        Clearing is still saved, because Clear calls _save_draft directly.
        Emptiness only stops the automatic writes.
        """
        if not cls._form_has_content(values):
            return gr.update()
        return cls._save_draft(*values)

    @classmethod
    def _save_draft(cls, *values):
        """
        Write the whole form to disk. Returns a status line.

        Saving is skipped when nothing has changed since the last write, so a
        timer ticking against an idle form costs one comparison rather than a
        file write and a UI update.
        """
        payload = list(values)
        if payload == _DRAFT_CACHE.get("values"):
            return gr.update()

        stamp = time.strftime("%H:%M:%S")
        blob = {"flat_len": len(payload), "saved": stamp, "values": payload}
        path = cls._draft_path()
        try:
            # The draft that is about to be replaced is kept as .prev.json.
            # Autosave will not overwrite an empty form, but it will happily
            # overwrite a large draft with a small one if you start typing
            # instead of restoring, and one rotation is enough to get it back.
            if path.exists():
                path.replace(cls._draft_backup_path())
            # Written beside the target and moved into place, so a crash
            # during the write cannot leave a half-file where the draft was.
            temp = path.with_suffix(".json.tmp")
            temp.write_text(json.dumps(blob), encoding="utf-8")
            temp.replace(path)
        except Exception as exc:                      # noqa: BLE001
            return (f"Could not save the draft ({type(exc).__name__}). "
                    f"Tried `{path}`.")

        _DRAFT_CACHE["values"] = payload
        return f"Draft saved at {stamp}."

    @classmethod
    def _load_draft(cls, previous=False):
        """The saved flat list, or None when there is nothing usable."""
        path = cls._draft_backup_path() if previous else cls._draft_path()
        try:
            blob = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except Exception:                             # noqa: BLE001
            return None
        if not isinstance(blob, dict) or not isinstance(blob.get("values"), list):
            return None
        return blob

    @classmethod
    def _draft_note(cls):
        """
        What to say about the draft on the line under the buttons.

        This is read at panel construction, so it is also the line someone
        sees straight after a crash - which is the moment to say restore
        first rather than to describe the feature.
        """
        blob = cls._load_draft()
        backup = cls._load_draft(previous=True)
        if not blob:
            if backup:
                return ("No current draft, but an earlier one from "
                        f"{backup.get('saved', 'a previous session')} is on "
                        "disk. **Restore last draft** will offer it.")
            return ("No saved draft yet. The form is written to disk as you "
                    "work, so a WanGP crash doesn't cost the prompt.")
        if blob.get("flat_len") != cls._flat_len():
            return ("A draft is on disk but it was saved by a different "
                    "version of this plugin, so restoring it would put values "
                    "in the wrong fields. **Clear all fields** overwrites it.")
        return (f"**Draft on disk from {blob.get('saved', 'earlier')}.** "
                "Restore it before you start typing - once there is something "
                "in the form it gets saved over the top of this one.")

    @classmethod
    @classmethod
    def _unchanged_form(cls, note):
        """Write nothing anywhere and say why - a refusal, not an error."""
        return ([gr.update()] * (cls._flat_len() + CLEAR_GROUP_UPDATES)
                + [note])

    # -- older layouts ------------------------------------------------------
    #
    # The flat list is positional, so a form saved under a different layout
    # cannot be poured into the current one - every value after the first
    # difference lands in the wrong field. Refusing is the safe default and
    # what happens to anything unrecognised.
    #
    # But a layout change is not always a content change. 3.2 replaced the
    # Source video block with the Reference sources section, and everything
    # the old block held has somewhere to go in the new one, so those saves
    # can be carried forward rather than lost. Each migration is one step
    # between adjacent layouts; a save two versions old runs through both.

    # The 3.1 flat list: eleven scene fields, then the five Source video
    # fields, then entry_count and everything after it - none of which moved.
    FLAT_LEN_3_1 = 91
    HEAD_LEN_3_1 = 11   # eleven scene fields; 3.2 inserts one more
    VIDEO_ROLES_3_1 = ("none", "continue from it", "edit it",
                       "reference its camera and cutting only")

    @classmethod
    def _migrate_3_1(cls, values):
        """
        3.1's Source video block into 3.2's Reference sources rows.

        The old block described one video and, separately, that video's
        audio. Both become rows: the video in its own slot, its audio in an
        audio slot that names the video it came from - which is what the new
        model calls the same thing.
        """
        head = list(values[:cls.HEAD_LEN_3_1])
        # 3.1.1 never wrote scheduling commands, so a save from
        # it keeps the behaviour it had.
        head.insert(4, False)
        role, vdesc, vret, aret, adesc = values[cls.HEAD_LEN_3_1:
                                                cls.HEAD_LEN_3_1 + 5]
        rest = list(values[cls.HEAD_LEN_3_1 + 5:])   # entry_count onward

        role_map = {
            "continue from it": "continued from",
            "edit it": "edited",
            "reference its camera and cutting only":
                "camera and cutting structure",
        }
        rows = []
        if cls._s(role) not in ("", "none") or cls._s(vdesc) or cls._s(vret):
            rows.append(["Video 1", role_map.get(cls._s(role), cls._s(role)),
                         cls._s(vdesc), cls._s(vret), ""])
        if cls._s(aret) or cls._s(adesc):
            rows.append(["Audio 1", "soundtrack",
                         cls._s(adesc) or "the original audio", cls._s(aret),
                         "Video 1" if rows else ""])

        out = head + [len(rows)]
        for i in range(MAX_REFS):
            out += rows[i] if i < len(rows) else ["", "", "", "", ""]
        return out + rest

    @classmethod
    def _migrate_values(cls, values):
        """
        Bring a saved form up to the current layout.

        Returns (values, note). A layout this cannot recognise comes back as
        (None, "") so the caller refuses rather than guessing - a wrong guess
        here is silent, and puts a lens in an anchor.
        """
        values = list(values)
        if len(values) == cls._flat_len():
            return values, ""

        if (len(values) == cls.FLAT_LEN_3_1
                and cls._s(values[cls.HEAD_LEN_3_1]) in cls.VIDEO_ROLES_3_1):
            upgraded = cls._migrate_3_1(values)
            if len(upgraded) == cls._flat_len():
                return upgraded, (
                    " Upgraded from the 3.1 layout - the Source video block "
                    "is now rows in **Reference sources**, so check them "
                    "before you build.")
        return None, ""

    @classmethod
    def _apply_saved(cls, blob, describe):
        """
        Put a saved form back into every field.

        Shared by the draft and the prompt library, because the shape is the
        awkward part: one value per flat input, then the group visibility
        updates, then the status line. Two copies of that arithmetic would be
        two things to keep in step with the flat list.
        """
        expected = cls._flat_len()
        values = blob.get("values") or []
        values, upgrade_note = cls._migrate_values(values)
        if values is None:
            saved_len = len(blob.get("values") or [])
            return cls._unchanged_form(
                f"That was saved with {saved_len} fields and this version "
                f"expects {expected}, so loading it would shift every value "
                "after the difference. Left the form alone.")
        describe += upgrade_note

        # Slots hidden at save time have to be reopened, or the restored
        # values sit in components nobody can see.
        d = cls._unpack(values)
        groups = [gr.update(visible=(i < d["entry_count"]))
                  for i in range(MAX_ENTRIES)]
        groups += [gr.update(visible=d["ref_mode"])] * MAX_ENTRIES
        groups += [gr.update(visible=(i < d["ref_count"]))
                   for i in range(MAX_REFS)]
        groups += [gr.update(visible=d["ref_mode"])] * 3

        _DRAFT_CACHE["values"] = list(values)
        return list(values) + groups + [describe]

    @classmethod
    def _restore_draft(cls):
        """
        Put the autosaved draft back into every field.

        Returns the same shape as _clear plus a status line. Refusing is a
        no-op rather than an error.
        """
        def unchanged(note):
            return cls._unchanged_form(note)

        # The draft that was replaced is worth offering when the current one
        # holds nothing - that is what a blank autosave over real work looks
        # like from here, and the rotation exists to make it recoverable.
        blob, source = cls._load_draft(), ""
        if blob and not cls._form_has_content(blob.get("values") or []):
            older = cls._load_draft(previous=True)
            if older and cls._form_has_content(older.get("values") or []):
                blob, source = older, " (the one before it was empty)"

        if not blob:
            blob = cls._load_draft(previous=True)
            source = " from the previous session"
        if not blob:
            return unchanged("No saved draft to restore.")

        return cls._apply_saved(
            blob, f"Restored the draft saved at "
                  f"{blob.get('saved', 'earlier')}{source}.")

    # -- the prompt library -------------------------------------------------
    #
    # The draft is one rolling autosave: it answers "what was I just doing".
    # This answers "where is that market scene I built last week". Same file
    # format, same length check, but named by you and never written over
    # without being asked.

    @classmethod
    def _prompts_dir(cls):
        """The prompts folder, created on first use."""
        path = Path(__file__).resolve().parent / "prompts"
        try:
            path.mkdir(parents=True, exist_ok=True)
        except Exception:                             # noqa: BLE001
            pass
        return path

    @classmethod
    def _subjects_dir(cls):
        """The subjects folder, created on first use."""
        path = Path(__file__).resolve().parent / "subjects"
        try:
            path.mkdir(parents=True, exist_ok=True)
        except Exception:                             # noqa: BLE001
            pass
        return path

    @classmethod
    def _subject_files(cls):
        """Every saved subject, newest first."""
        try:
            files = [p for p in cls._subjects_dir().glob("*.json")
                     if p.is_file()]
        except Exception:                             # noqa: BLE001
            return []
        return sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)

    @classmethod
    def _subject_choices(cls, limit=SUBJECT_LIST_LIMIT):
        return [p.stem for p in cls._subject_files()[:limit]]

    @classmethod
    def _save_subject(cls, name, *values):
        """
        Write one entry's character fields to subjects/<n>.json.

        Takes the entry's own controls in SUBJECT_FIELDS order, so the same
        list drives saving, loading and the wiring - there is no second place
        to keep in step.
        """
        filename = cls._prompt_filename(name)
        if not filename:
            return gr.update(), ("Give the subject a name first - letters, "
                                 "numbers, spaces, dashes.")
        fields = dict(zip(SUBJECT_FIELDS, values))
        if not (cls._s(fields.get("desc"))
                or cls._s(fields.get("char_name"))):
            return gr.update(), ("There is nothing to save yet - give the "
                                 "subject a description or a name in the "
                                 "creator.")

        path = cls._subjects_dir() / f"{filename}.json"
        existed = path.exists()
        blob = {
            "kind": "subject",
            "saved": time.strftime("%Y-%m-%d %H:%M"),
            "name": filename,
            "fields": {k: fields.get(k) for k in SUBJECT_FIELDS},
        }
        try:
            temp = path.with_suffix(".json.tmp")
            temp.write_text(json.dumps(blob), encoding="utf-8")
            temp.replace(path)
        except Exception as exc:                      # noqa: BLE001
            return gr.update(), (f"Could not save the subject "
                                 f"({type(exc).__name__}). Tried `{path}`.")

        note = (("Replaced" if existed else "Saved") + f" **{filename}**. "
                "Reference assets and the speaker ID are not saved - they "
                "belong to the project, not the character.")
        return gr.update(choices=cls._subject_choices(), value=filename), note

    @classmethod
    def _load_subject(cls, name):
        """
        Put a saved subject into this entry's fields.

        Returns one value per SUBJECT_FIELDS, then the creator block's
        visibility, then the status line. A field the saved file does not
        have keeps whatever the schema defaults to rather than blanking, so
        an older save loads into a newer plugin without wiping new fields.
        """
        def unchanged(note):
            return [gr.update()] * len(SUBJECT_FIELDS) + [gr.update(), note]

        filename = cls._prompt_filename(name)
        if not filename:
            return unchanged("Pick a subject to load first.")
        path = cls._subjects_dir() / f"{filename}.json"
        try:
            blob = json.loads(path.read_text(encoding="utf-8"))
        except Exception:                             # noqa: BLE001
            return unchanged(f"Could not read **{filename}** - it may have "
                             "been moved or written by something else.")
        fields = blob.get("fields") if isinstance(blob, dict) else None
        if not isinstance(fields, dict):
            return unchanged(f"`{filename}` is not a subject this plugin "
                             "wrote.")

        out = [fields[k] if k in fields else gr.update()
               for k in SUBJECT_FIELDS]
        # The creator block is hidden by default, so a subject built in it
        # would load into fields nobody can see.
        out.append(gr.update(visible=bool(fields.get("use_creator"))))
        missing = [k for k in SUBJECT_FIELDS if k not in fields]
        note = f"Loaded **{filename}**, saved {blob.get('saved', 'earlier')}."
        if missing:
            note += (f" {len(missing)} field(s) were not in that save and "
                     "were left as they were.")
        return out + [note]

    @classmethod
    def _refresh_subjects(cls):
        choices = cls._subject_choices()
        note = (f"{len(choices)} subject(s) in `subjects`."
                if choices else "No saved subjects yet.")
        return gr.update(choices=choices), note

    @staticmethod
    def _prompt_filename(name):
        """
        A file name from whatever was typed.

        Anything that could walk out of the prompts folder is dropped rather
        than escaped - a prompt called "../../wgp" should be a file called
        "wgp", not a way to write one anywhere on the disk.
        """
        name = (name or "").strip()
        keep = [c for c in name if c.isalnum() or c in " -_.,()"]
        cleaned = "".join(keep).strip(" .")
        while ".." in cleaned:
            cleaned = cleaned.replace("..", ".")
        return cleaned[:80].strip(" .")

    @classmethod
    def _prompt_files(cls):
        """Every saved prompt, newest first."""
        try:
            files = [p for p in cls._prompts_dir().glob("*.json") if p.is_file()]
        except Exception:                             # noqa: BLE001
            return []
        return sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)

    @classmethod
    def _prompt_choices(cls, limit=PROMPT_LIST_LIMIT):
        """
        The most recent saved prompts, for the dropdown.

        Capped because the dropdown is a shortcut, not a file manager - the
        rest of the folder is reachable through Load from file.
        """
        return [p.stem for p in cls._prompt_files()[:limit]]

    @classmethod
    def _read_prompt(cls, path):
        try:
            blob = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception:                             # noqa: BLE001
            return None
        if not isinstance(blob, dict) or not isinstance(blob.get("values"), list):
            return None
        return blob

    @classmethod
    def _save_prompt(cls, name, *values):
        """
        Write the form to prompts/<name>.json.

        Returns (dropdown update, status). An existing prompt of that name is
        replaced, which is what saving under a name you already used means -
        but it is said out loud, because a prompt is not a draft and losing
        one is not recoverable from a rotation.
        """
        filename = cls._prompt_filename(name)
        if not filename:
            return gr.update(), ("Give the prompt a name first - letters, "
                                 "numbers, spaces, dashes.")
        if not cls._form_has_content(values):
            return gr.update(), ("There is nothing in the form to save yet.")

        path = cls._prompts_dir() / f"{filename}.json"
        existed = path.exists()
        blob = {
            "flat_len": len(values),
            "saved": time.strftime("%Y-%m-%d %H:%M"),
            "name": filename,
            "values": list(values),
        }
        try:
            temp = path.with_suffix(".json.tmp")
            temp.write_text(json.dumps(blob), encoding="utf-8")
            temp.replace(path)
        except Exception as exc:                      # noqa: BLE001
            return gr.update(), (f"Could not save the prompt "
                                 f"({type(exc).__name__}). Tried `{path}`.")

        note = ("Replaced" if existed else "Saved") + f" **{filename}**."
        return (gr.update(choices=cls._prompt_choices(), value=filename), note)

    @classmethod
    def _load_prompt(cls, name):
        """Load a named prompt from the dropdown."""
        filename = cls._prompt_filename(name)
        if not filename:
            return cls._unchanged_form("Pick a prompt to load first.")
        path = cls._prompts_dir() / f"{filename}.json"
        blob = cls._read_prompt(path)
        if not blob:
            return cls._unchanged_form(
                f"Could not read **{filename}** - it may have been moved or "
                "written by something else.")
        return cls._apply_saved(
            blob, f"Loaded **{filename}**, saved {blob.get('saved', 'earlier')}.")

    @classmethod
    def _load_recent_prompt(cls):
        """Load whichever prompt was saved last, without picking it first."""
        files = cls._prompt_files()
        if not files:
            return cls._unchanged_form(
                "No saved prompts yet. **Save prompt** writes one into the "
                "`prompts` folder beside the plugin.")
        return cls._load_prompt(files[0].stem)

    @classmethod
    def _load_prompt_file(cls, upload):
        """
        Load a prompt from a file chosen on disk.

        The dropdown only offers the most recent few, so this is how the rest
        of the folder is reached - and how a prompt someone sent you is
        opened without copying it in first.
        """
        if not upload:
            return cls._unchanged_form("Choose a file to load first.")
        path = getattr(upload, "name", upload)
        blob = cls._read_prompt(path)
        if not blob:
            return cls._unchanged_form(
                f"`{Path(path).name}` is not a prompt this plugin wrote.")
        label = blob.get("name") or Path(path).stem
        return cls._apply_saved(
            blob, f"Loaded **{label}** from file, saved "
                  f"{blob.get('saved', 'earlier')}.")

    @classmethod
    def _refresh_prompts(cls):
        """Re-read the folder, for prompts added or removed outside the UI."""
        choices = cls._prompt_choices()
        note = (f"{len(choices)} prompt(s) in `prompts`."
                if choices else "No saved prompts yet.")
        return gr.update(choices=choices), note

    @staticmethod
    def _clear():
        # start_image, end_image, ref_mode, duration, window_commands,
        # style, grading, location, time_of_day, lighting, atmosphere,
        # camera_type
        out = [False, False, False, 8.0, False,
               "", "", "", "", "", "", ""]
        out.append(0)                                 # ref_count
        for _ in range(MAX_REFS):
            # slot, role, desc, retention, from
            out += ["", "", "", "", ""]
        out.append(0)                                 # entry_count
        for _ in range(MAX_ENTRIES):
            # kind, desc, speaker, onscreen, age, gender, pitch, timbre,
            # rate, accent, lang, source, retention, note,
            # voice_from, motion_from
            out += ["Subject", "", "", "", "", "", "", "", "", "",
                    "English", [], "", "", "", ""]
        # task_types, summary, action
        out += [[], "", ""]
        # ambience_from, ambience_retention, soundscape,
        # music_from, music_role, music_retention, music
        out += ["", "", "", "", "style", "", ""]

        # Re-hide every slot. Shots and beats had their own groups; the
        # action is one always-visible field, so only the cast entries, the
        # reference blocks and the reference-source rows are left to hide.
        out += [gr.update(visible=False)] * MAX_ENTRIES   # entry groups
        out += [gr.update(visible=False)] * MAX_ENTRIES   # reference blocks
        out += [gr.update(visible=False)] * MAX_REFS      # source rows
        out += [gr.update(visible=False)] * 3   # audio refs + summary block
        assert len(out) - CLEAR_GROUP_UPDATES > 0
        return out


Plugin = H3PromptBuilderPlugin
