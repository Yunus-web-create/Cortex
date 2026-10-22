"""Model routing: pick the best available model for the job.

Arena-style behaviour in miniature — the user either pins a model or lets the
router choose per message. Rankings are curated, but every candidate is
validated against the live catalogue, and each task has a pattern fallback so
new model generations keep working without a code change.
"""
from __future__ import annotations

import re

from . import gateway

# Ordered preference per task. The first available id wins; the rest become
# fallbacks if the provider errors.
PREFERENCE = {
    "reasoning": [
        "nvidia/nemotron-3-ultra-550b-a55b:free", "thinkingmachines/inkling:free",
        "minimax/minimax-m3.1-flash:free",
        "openai/gpt-6.1-sol", "anthropic/claude-opus-5.5", "openai/gpt-6-sol",
        "google/gemini-3.6-flash", "google/gemini-3.5-flash", "google/gemini-4-argon",
        "openai/gpt-5.6-sol", "spacexai/grok-4.20-reasoning",
        "deepseek/deepseek-v4.1-flash", "anthropic/claude-sonnet-5.5",
    ],
    "code": [
        "cohere/north-mini-code:free", "nvidia/nemotron-3-ultra-550b-a55b:free",
        "minimax/minimax-m3.1-flash:free",
        "openai/gpt-5.3-codex", "openai/gpt-5.1-codex-max", "openai/gpt-5.2-codex",
        "anthropic/claude-sonnet-5.5", "anthropic/claude-opus-5.5",
        "alibaba/qwen3-coder-next", "mistral/codestral",
        "google/gemini-3.6-flash", "google/gemini-3.5-flash", "google/gemini-3.8-flash",
    ],
    "general": [
        "nvidia/nemotron-3-ultra-550b-a55b:free", "minimax/minimax-m3.1-flash:free",
        "thinkingmachines/inkling:free", "openrouter/free",
        "anthropic/claude-sonnet-5.5", "openai/gpt-5.5", "google/gemini-3.6-flash",
        "google/gemini-3.5-flash", "spacexai/grok-4.3", "deepseek/deepseek-v4-pro",
        "google/gemini-3.8-flash", "openai/gpt-5.4-mini",
    ],
    "creative": [
        "thinkingmachines/inkling:free", "nvidia/nemotron-3-ultra-550b-a55b:free",
        "anthropic/claude-opus-5.5", "openai/gpt-6.1-sol", "anthropic/claude-sonnet-5.5",
        "google/gemini-3.6-flash", "openai/gpt-5.5", "google/gemini-3.5-flash",
    ],
    "fast": [
        "minimax/minimax-m3.1-flash:free", "nvidia/nemotron-3.5-lightning:free",
        "google/gemini-3.5-flash-lite", "openai/gpt-5.4-nano", "anthropic/claude-haiku-5.5",
        "deepseek/deepseek-v4.1-flash", "google/gemini-3.6-flash", "google/gemini-3.5-flash",
    ],
    "vision": [
        "nvidia/nemotron-3.5-super-vl:free", "thinkingmachines/inkling:free",
        "google/gemini-3.6-flash", "google/gemini-3.5-flash", "openai/gpt-5.5",
        "anthropic/claude-opus-5.5", "spacexai/grok-4.3",
    ],
    "image": [
        "openai/gpt-image-2.5-sunburst", "openai/gpt-image-2.5-flare", "bfl/flux-3-image",
        "google/gemini-3.1-flash-image", "recraft/recraft-v4.1-pro",
        "bytedance/seedream-5.0-pro", "spacexai/grok-imagine-image-2.0",
    ],
    "speech": [
        "openai/tts-1-hd", "google/gemini-3.8-flash-tts", "microsoft/mai-voice-2.1",
        "openai/tts-1", "microsoft/mai-voice-2",
    ],
    "video": [
        "google/veo-3.1-generate-001", "klingai/kling-v3.0-t2v", "bytedance/seedance-2.5",
        "alibaba/wan-v3.0-video", "spacexai/grok-imagine-video-1.5", "google/veo-3.0-generate-001",
    ],
    "transcribe": [
        "openai/gpt-4o-transcribe", "google/gemini-3.5-transcribe", "openai/whisper-1",
        "mistral/voxtral-small-transcribe-3",
    ],
}

# Pattern fallback per task, used when no curated id exists in the catalogue.
PATTERN = {
    "reasoning": r"(^|/)(.*(sol|opus|pro|argon|thinking|multi-agent).*)$",
    "code": r"(codex|coder|codestral|sonnet)",
    "general": r"(sonnet|gpt-5|gemini-3|grok-4)",
    "creative": r"(opus|sol|sonnet)",
    "fast": r"(flash-lite|nano|haiku|mini|flash)",
    "vision": r"(argon|gpt-5|opus|gemini)",
    "image": r"(gpt-image|flux|seedream|recraft|imagine-image|gemini.*image)",
    "speech": r"(tts|voice)",
    "video": r"(veo|kling|seedance|wan|imagine-video)",
    "transcribe": r"(transcribe|whisper)",
}

TYPE_FOR = {"image": "image", "speech": "speech", "video": "video", "transcribe": "transcription"}

LABELS = {
    "auto": "Auto — best model per task", "reasoning": "Deep reasoning", "code": "Coding",
    "general": "Everyday", "creative": "Creative writing", "fast": "Fast & cheap",
    "vision": "Vision", "image": "Image generation", "speech": "Speech", "video": "Video",
}

MEDIA_TASKS = {"image": "image", "speech": "voice", "video": "video"}


def free_renderer(task: str) -> str:
    """The local, keyless generator that handles a medium the provider can't."""
    return f"free {MEDIA_TASKS.get(task, task)} renderer"

WHY = {
    "reasoning": "hardest reasoning", "code": "best at code", "general": "best all-rounder",
    "creative": "best long-form writer", "fast": "cheapest capable model", "vision": "strongest vision",
    "image": "best image quality", "speech": "most natural voice", "video": "best video model",
    "transcribe": "accurate transcription",
}

CODE_CUES = re.compile(
    # an action on a piece of software, not merely a language name: asking
    # "what is the latest Python version" is a lookup, "write a Python script" is code
    r"\b(?:write|build|implement|refactor|debug|generate|create|fix|optimi[sz]e|rewrite)\b"
    r"[^.!?]{0,48}\b(?:code|script|program|function|class|app|component|endpoint|api|pipeline|"
    r"game|website|cli|tool|query|regex|test|dockerfile)\b|"
    r"\bstack ?trace\b|\btraceback\b|\brefactor(?:ing)?\b|\bunit tests?\b|\bdockerfile\b|"
    r"\bsyntax error\b|\bcompile error\b|\bsegfault\b|\bapi endpoint\b|\bsql query\b|"
    r"\bwhy (?:does|is) (?:my|this) code\b|\bcode review\b", re.I)
CREATE_CUES = re.compile(
    r"\b(write (?:me )?(?:a|an) (?:story|poem|song|essay|script|speech|email|letter|blog)|"
    r"brainstorm|tagline|slogan|copy for|draft (?:a|an)|creative|narrative|screenplay)\b", re.I)
IMAGE_CUES = re.compile(
    r"\b(draw|paint|sketch|illustrate|generate (?:an? )?(?:image|picture|photo|logo|artwork|poster)|"
    r"image of|picture of|photo of|logo for|render (?:a|an)|design (?:a|an) (?:logo|poster|cover))\b", re.I)
VIDEO_CUES = re.compile(
    r"\b(generate (?:a |an )?video|make (?:a |an )?video|animate|video of|clip of|text[- ]to[- ]video)\b", re.I)
SPEECH_CUES = re.compile(
    r"\b(read (?:this|that) aloud|text[- ]to[- ]speech|voice ?over|narrate|say (?:this|it) out loud|generate (?:audio|speech))\b", re.I)
REASON_CUES = re.compile(
    r"\b(prove|derive|analy[sz]e|compare|trade-?offs?|architecture|strategy|plan (?:a|the)|evaluate|"
    r"step[- ]by[- ]step|root cause|why (?:does|do|is)|optimi[sz]e|design (?:a|the) system)\b", re.I)


def classify(question: str, has_image: bool = False) -> str:
    q = question or ""
    if has_image:
        return "vision"
    if VIDEO_CUES.search(q):
        return "video"
    if SPEECH_CUES.search(q):
        return "speech"
    if IMAGE_CUES.search(q):
        return "image"
    if CODE_CUES.search(q):
        return "code"
    if CREATE_CUES.search(q):
        return "creative"
    if REASON_CUES.search(q) or len(q) > 320:
        return "reasoning"
    looks_like_question = q.rstrip().endswith("?") or re.match(
        r"^\s*(who|what|when|where|why|how|is|are|can|does|do|should|which)\b", q, re.I)
    if len(q) < 40 and not looks_like_question:
        return "fast"
    return "general"


def _type_ok(model: dict, task: str) -> bool:
    want = TYPE_FOR.get(task, "language")
    return (model.get("type") or "language") == want


def _clean(models: list[dict]) -> list[dict]:
    """Drop duplicates nobody wants to chat with — and honour a $0 account.

    On a key with no credit (OpenRouter's free tier) the *only* usable models are
    the zero-price ones, so the rule flips: keep those and drop the rest.
    """
    if gateway.free_only():
        free = [m for m in (models or []) if m.get("free")]
        if free:
            return free
    return [m for m in (models or [])
            if not re.search(r"(-batch|:free|preview-|-\d{4}$)", m["id"])]


def candidates(task: str, models: list[dict], limit: int = 4) -> list[str]:
    if task in MEDIA_TASKS and not gateway.can(task):
        return []                       # provider has no such endpoint (Gemini-compat is chat only)
    usable = _clean(models or [])
    parked = gateway.model_cooldowns()
    if parked:                          # models this key just got a 429/503 from
        live = [m for m in usable if m["id"].split("/")[-1] not in parked]
        usable = live or usable
    pool = [m["id"] for m in usable if _type_ok(m, task)]
    if not pool:
        return []
    ordered: list[str] = []
    by_short = {}
    for pid in pool:                       # "google/gemini-3.8-flash" and "gemini-3.8-flash" are the same model
        by_short.setdefault(pid.split("/")[-1], pid)
    for wanted in PREFERENCE.get(task, []):
        hit = wanted if wanted in pool else by_short.get(wanted.split("/")[-1])
        if hit and hit not in ordered:
            ordered.append(hit)
    if len(ordered) < limit:
        pat = re.compile(PATTERN.get(task, r".*"), re.I)
        rest = sorted([m for m in usable if m["id"] in pool and m["id"] not in ordered],
                      key=lambda m: (-(m.get("released") or 0), -(m.get("context") or 0)))
        for m in rest:
            if pat.search(m["id"]) and m["id"] not in ordered:
                ordered.append(m["id"])
            if len(ordered) >= limit:
                break
    if len(ordered) < limit:
        for m in usable:
            if m["id"] in pool and m["id"] not in ordered:
                ordered.append(m["id"])
            if len(ordered) >= limit:
                break
    return ordered[:limit]


def route_preview(task: str, models: list[dict], limit: int = 3) -> list[str]:
    """What the UI should show for this task, including free renderers."""
    if task in MEDIA_TASKS and not gateway.can(task):
        return [free_renderer(task)]
    return candidates(task, models, limit=limit)


def pick(task: str, models: list[dict], override: str | None = None) -> dict:
    """Return {model, task, why, fallbacks} for one message."""
    if override and override not in ("auto", ""):
        return {"model": override, "task": task, "why": "your pick",
                "fallbacks": candidates(task, models)[:3]}
    ordered = candidates(task, models)
    if task in MEDIA_TASKS and not gateway.can(task):
        ordered = []                      # this provider has no such endpoint (Gemini-compat = chat only)
    if not ordered and task in MEDIA_TASKS:
        # No model in the catalogue generates this medium, so the turn runs on a
        # language model and the free renderer does the image/voice/video.
        chat = candidates("general", models, limit=3)
        return {"model": chat[0] if chat else "", "task": task,
                "why": free_renderer(task), "fallbacks": chat[1:]}
    if not ordered:
        return {"model": "", "task": task, "why": "no model available", "fallbacks": []}
    return {"model": ordered[0], "task": task, "why": WHY.get(task, "best fit"),
            "fallbacks": ordered[1:]}


def helper_model(models: list[dict], task: str = "fast") -> str:
    """Cheap model for titles, query rewrites, follow-ups."""
    ids = candidates(task, models, limit=3)
    return ids[0] if ids else ""


def option_list(models: list[dict]) -> list[dict]:
    """What the UI shows in the model picker."""
    if not _clean(models):
        return []
    out = [{"id": "auto", "label": LABELS["auto"], "group": "Routing",
            "hint": "Cortex picks per message"}]
    for task in ("reasoning", "code", "general", "fast", "vision", "creative"):
        best = candidates(task, models, limit=1)
        if best:
            out.append({"id": f"task:{task}", "label": LABELS[task], "group": "Routing",
                        "hint": f"always {best[0].split('/')[-1]}"})
    seen = set()
    for m in sorted((m for m in _clean(models) if (m.get("type") or "language") == "language"),
                    key=lambda m: (-(m.get("released") or 0), m["id"])):
        if m["id"] in seen:
            continue
        seen.add(m["id"])
        out.append({"id": m["id"], "label": m.get("name") or m["id"], "group": m.get("vendor") or "Models",
                    "hint": _short_hint(m)})
    return out


def _short_hint(m: dict) -> str:
    bits = []
    ctx = m.get("context") or 0
    if ctx >= 1_000_000:
        bits.append(f"{ctx // 1_000_000}M ctx")
    elif ctx:
        bits.append(f"{ctx // 1000}K ctx")
    for tag in ("reasoning", "vision", "tool-use", "web-search"):
        if tag in (m.get("tags") or []):
            bits.append(tag)
    return " · ".join(bits[:3])


async def resolve(question: str, *, override: str | None = None, has_image: bool = False,
                  catalog: list[dict] | None = None) -> dict:
    models = catalog if catalog is not None else await gateway.models()
    task = "auto"
    wanted = (override or "auto").strip()
    if wanted.startswith("task:"):
        wanted, task = "auto", wanted.split(":", 1)[1]
    if wanted not in ("auto", ""):
        task = classify(question, has_image)
        kinds = {m["id"]: (m.get("type") or "language") for m in _clean(models)}
        if kinds.get(wanted, "language") != "language" and wanted not in ("", "auto"):
            # a pinned image/voice/video model cannot answer text; keep it for the
            # renderer and let a chat model drive the turn
            chat = candidates("general", models, limit=2)
            return {"model": chat[0] if chat else "", "task": task,
                    "why": f"your pick ({wanted.split('/')[-1]})", "fallbacks": chat[1:],
                    "requested": wanted}
        return {"model": wanted, "task": task, "why": "your pick",
                "fallbacks": [], "requested": wanted}
    if task == "auto":
        task = classify(question, has_image)
    decision = pick(task, models)
    decision["requested"] = "auto"
    return decision
