"""Resolve WHICH scene object a free-form task acts on (the placement target).

Given a task string ("turn on the faucet") and the objects a spatial index found,
this picks the single :class:`SceneObject` the robot should position itself for.
That choice is the hinge of the dynamic-placement pipeline: nothing downstream is
hardcoded, so if we point the placement solver at the wrong object the robot stands
in the wrong place.

Two strategies, in priority order:

  * :func:`resolve_target` asks a VLM (Gemini, the SAME integration
    ``scene_semantics.py`` / ``render_visual_qc.py`` use) to read the object list +
    task and return strict JSON ``{"target_id": "..."}``. The actual model call is
    INJECTED (``generate``) so the whole module imports and unit-tests with NO
    google-genai, NO network, NO GPU; the default ``generate`` lazily imports the
    SDK and mirrors the repo pattern (file/env ``GOOGLE_GENAI_API_KEY``, model
    cascade, ``response_mime_type=application/json``, thinking-part filtering — and
    deliberately NO ``thinking_config``, which silently fails per repo notes).

  * :func:`resolve_target_by_label` is a pure, dependency-free fuzzy fallback
    (label contains + a small synonym table: faucet->tap/spout/sink, ...). It runs
    when no VLM is available and as a safety net when the VLM names an id that is
    not in the scene.

WHY a VLM at all: object labels are noisy ("kitchen_faucet_01", "spout", "tap"),
tasks are free-form, and the mapping from verb+noun to the right object is exactly
what a language model is good at. The label fallback keeps the package usable —
and testable — with no model in the loop.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Callable, List, Optional, Sequence

from .types import SceneObject

# Mirror scene_semantics.py / render_visual_qc.py: gemini-3-flash-preview first
# (fast, correct JSON output per repo notes), then progressively heavier models.
DEFAULT_MODEL_CASCADE = (
    "gemini-3-flash-preview",
    "gemini-3-pro-preview",
    "gemini-2.5-flash",
    "gemini-2.5-pro",
)

# ``generate(prompt: str) -> raw_text``. Text-only (no image), injected so callers
# (and tests) can swap in a fake without google-genai. Default = real Gemini call.
GenerateFn = Callable[[str], str]

# Coarse synonym GROUPS for the pure label fallback. Each tuple is a set of words
# that name the same fixture (or an acceptable proxy for it). Matching is
# bidirectional and group-based: a task noun and a label fragment match as synonyms
# if they fall in the same group, so "tap" (task) finds a "faucet" (label) and vice
# versa. The FIRST word of a group that contains a faucet-like proxy stays last in
# its group so a direct token hit is always preferred over a proxy (see
# ``_label_match_rank``). Kept small + obvious — the VLM path is the real matcher.
_SYNONYM_GROUPS: tuple[tuple[str, ...], ...] = (
    ("faucet", "tap", "spout", "mixer"),
    ("sink", "basin", "washbasin"),
    ("stove", "cooktop", "range", "burner", "hob"),
    ("oven",),
    ("fridge", "refrigerator", "freezer"),
    ("microwave",),
    ("dishwasher",),
    ("kettle",),
    ("cabinet", "cupboard", "cabinetry"),
    ("drawer",),
    ("door",),
    ("window",),
    ("light", "lamp", "lightswitch"),
    ("switch", "toggle", "lightswitch"),
    ("trash", "bin", "garbage", "waste", "wastebasket"),
    ("toilet", "lavatory", "wc"),
    ("shower",),
    ("counter", "countertop", "worktop"),
    ("table", "desk"),
    ("handle", "knob", "lever", "pull"),
)

# A faucet sits on a sink, so a faucet task can fall back to the sink as a proxy
# target (and vice versa). These cross-group proxies are weaker than a synonym
# match, so they rank below both direct + synonym hits.
_PROXY_GROUPS: tuple[tuple[str, ...], ...] = (
    ("faucet", "tap", "spout", "mixer", "sink", "basin", "washbasin"),
    ("stove", "cooktop", "range", "burner", "hob", "oven"),
)
_OPENABLE_TARGET_GROUPS = frozenset(
    {
        "fridge",
        "oven",
        "microwave",
        "dishwasher",
        "cabinet",
        "drawer",
        "door",
    }
)


def _synonyms_of(token: str) -> set[str]:
    """All words sharing a synonym group with ``token`` (includes the token)."""
    out: set[str] = {token}
    for group in _SYNONYM_GROUPS:
        if token in group:
            out.update(group)
    return out


def _canonical_group_for_token(token: str) -> str | None:
    """Canonical synonym-group name for a token, or None when it is not a known target noun."""
    token = (token or "").strip().lower()
    if not token:
        return None
    for group in _SYNONYM_GROUPS:
        if token in group:
            return group[0]
    return None


def _proxies_of(token: str) -> set[str]:
    """Words that are an acceptable cross-fixture proxy for ``token`` (weakest tie)."""
    out: set[str] = set()
    for group in _PROXY_GROUPS:
        if token in group:
            out.update(group)
    return out

# Common task verbs we strip when guessing the intent noun in the label fallback,
# so "turn on the faucet" reduces to "faucet" rather than matching "on"/"turn".
# NOTE: "switch", "toggle", and "pull" are deliberately NOT stopwords — they are
# also object nouns (a light switch, a drawer pull) and have their own synonym
# groups, so "flip the switch" must keep "switch" as the intent noun rather than
# discard it and resolve to nothing.
_TASK_STOPWORDS = frozenset(
    {
        "turn", "on", "off", "the", "a", "an", "open", "close", "press", "push",
        "flip", "to", "of", "at", "in", "with", "and",
        "please", "go", "move", "walk", "stand", "near", "by", "use", "operate",
        "activate", "start", "stop", "robot", "pick", "up", "grab", "take", "put",
        "place", "set", "get", "reach", "for", "into", "onto",
        # Room / location qualifiers — these describe WHERE, never the object the task
        # acts on, yet they often appear as a prim/label substring (e.g. a "kitchen_box"
        # or "KitchenRoom" wrapper). Without dropping them, longest-first token ordering
        # tries "kitchen" before "faucet"/"sink" and resolves to the wrong object.
        "kitchen", "kitchenette", "bathroom", "bedroom", "garage", "office",
        "hallway", "pantry", "basement", "closet", "room", "scene", "area", "here",
        "there", "side",
    }
)


# ----------------------------- prompt -----------------------------

def build_target_prompt(task: str, objects: Sequence[SceneObject]) -> str:
    """Build the strict-JSON prompt listing ``{id, label, centroid}`` per object.

    Centroids are included so the model can disambiguate duplicate labels by
    position if it has any spatial context, but the contract it must honor is
    simple: return the ``id`` of the one object the task acts on.
    """
    task_text = (task or "").strip() or "interact with an object"
    lines = []
    for obj in objects:
        cx, cy, cz = obj.centroid
        label = obj.label or obj.id
        # json.dumps each string so a label/id with a quote or brace (perception
        # labels are SAM3-supplied and pass through verbatim) cannot break the
        # block structure or inject into the prompt — quotes/braces get escaped.
        id_json = json.dumps(str(obj.id))
        label_json = json.dumps(str(label))
        cat = f", category={json.dumps(str(obj.category))}" if obj.category else ""
        lines.append(
            f'  {{"id": {id_json}, "label": {label_json}{cat}, '
            f'"centroid": [{cx:.3f}, {cy:.3f}, {cz:.3f}]}}'
        )
    object_block = "\n".join(lines) if lines else "  (no objects)"
    return (
        "You are positioning a robot to perform a task in a 3D scene. You are given "
        "the task and a list of scene objects (each with an id, a human label, and a "
        "world-space centroid [x, y, z]).\n\n"
        f"TASK: {task_text}\n\n"
        "SCENE OBJECTS:\n"
        f"{object_block}\n\n"
        "Pick the SINGLE object the task most directly acts on (the thing the robot "
        "must stand near and face). If the exact object is not present, pick the "
        "closest functional proxy (e.g. a 'sink' for a faucet task). If nothing fits, "
        'return null for target_id.\n\n'
        "Respond with STRICT JSON only, no prose, exactly this shape:\n"
        '{"target_id": "<one id from the list, or null>"}'
    )


# ----------------------------- JSON parsing -----------------------------

def _extract_json_object(text: str) -> dict:
    """``json.loads`` else the first ``{...}`` block.

    Mirrors ``scene_semantics._extract_json_object`` / ``render_visual_qc`` so a
    reply with stray prose or code fences still parses.
    """
    try:
        payload = json.loads(text)
        if isinstance(payload, dict):
            return payload
    except Exception:
        pass
    match = re.search(r"\{.*\}", text or "", re.DOTALL)
    if not match:
        return {}
    try:
        payload = json.loads(match.group(0))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _extract_response_text(response: Any) -> str:
    """Final answer text from a google-genai response, SKIPPING thinking parts.

    Thinking parts (``thought=True``) carry reasoning, not the answer; including
    them corrupts the JSON. Mirrors ``scene_semantics._extract_response_text``.
    """
    text = str(getattr(response, "text", "") or "").strip()
    if text:
        return text
    candidates = getattr(response, "candidates", None)
    if not isinstance(candidates, list):
        return ""
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        parts = getattr(content, "parts", None)
        if not isinstance(parts, list):
            continue
        for part in parts:
            if getattr(part, "thought", False):
                continue
            part_text = str(getattr(part, "text", "") or "").strip()
            if part_text:
                return part_text
    return ""


# ----------------------------- gemini call (injectable) -----------------------------

def _gemini_resolve_text(prompt: str, *, models: Sequence[str] = DEFAULT_MODEL_CASCADE) -> str:
    """Default ``generate``: send the text prompt to Gemini, return the raw model text.

    Lazily imports google-genai so the module imports without it. Mirrors
    ``render_visual_qc._gemini_review_image`` but text-only: file/env
    ``GOOGLE_GENAI_API_KEY``, model cascade, ``response_mime_type=application/json``,
    thinking-part filtering, and NO ``thinking_config`` (silent-fail per repo notes).
    """
    api_key = (os.getenv("GOOGLE_GENAI_API_KEY") or os.getenv("GEMINI_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError("missing_GOOGLE_GENAI_API_KEY")
    from google import genai  # type: ignore

    client = genai.Client(api_key=api_key)
    override = (os.getenv("BLUEPRINT_TARGET_RESOLVER_GEMINI_MODEL") or "").strip()
    models_to_try = [override] if override else list(models)
    last_exc: Optional[Exception] = None
    for model in models_to_try:
        try:
            response = client.models.generate_content(
                model=model,
                contents=[prompt],
                config={"response_mime_type": "application/json"},  # no thinking_config (silent fail)
            )
            text = _extract_response_text(response)
            if text:
                return text
        except Exception as exc:  # noqa: BLE001 - try the next model in the cascade
            last_exc = exc
            continue
    if last_exc is not None:
        raise last_exc
    return ""


# ----------------------------- resolvers -----------------------------

def _index_by_id(objects: Sequence[SceneObject]) -> dict[str, SceneObject]:
    return {obj.id: obj for obj in objects}


def resolve_target(
    task: str,
    objects: List[SceneObject],
    *,
    generate: Optional[GenerateFn] = None,
) -> Optional[SceneObject]:
    """Resolve the target object for ``task`` via the VLM, with a label fallback.

    ``generate(prompt) -> raw_text`` is injectable; it defaults to the real Gemini
    text call. The reply must be ``{"target_id": "<id>"}``; the id is validated
    against the actual object list (a model can hallucinate an id). If the VLM
    yields nothing usable — empty list, no/blank/unknown id, parse failure, or the
    call raises — we fall back to :func:`resolve_target_by_label` so the pipeline
    still produces an answer rather than crashing.

    Returns the chosen :class:`SceneObject`, or ``None`` if nothing matches.
    """
    if not objects:
        return None
    by_id = _index_by_id(objects)
    gen = generate or _gemini_resolve_text
    prompt = build_target_prompt(task, objects)
    try:
        raw = gen(prompt)
    except Exception:  # noqa: BLE001 - a failed VLM call degrades to the label net
        return resolve_target_by_label(task, objects)
    obj = _extract_json_object(raw or "")
    target_id = obj.get("target_id")
    if isinstance(target_id, str):
        target_id = target_id.strip()
        if target_id and target_id in by_id:
            return by_id[target_id]
    # VLM gave us nothing we can trust (null, blank, or a hallucinated id).
    return resolve_target_by_label(task, objects)


def _task_intent_tokens(task: str) -> List[str]:
    """Content words of ``task`` (verbs/articles stripped), longest first.

    Longest-first so a multi-word intent like "light switch" is tried before its
    parts, which makes the synonym lookup prefer the most specific noun.
    """
    words = re.findall(r"[a-z0-9]+", (task or "").lower())
    content = [w for w in words if w not in _TASK_STOPWORDS]
    # de-dupe while preserving order, then sort longest-first (stable)
    seen: set[str] = set()
    uniq = [w for w in content if not (w in seen or seen.add(w))]
    return sorted(uniq, key=len, reverse=True)


def task_target_groups(task: str) -> list[str]:
    """Distinct fixture groups named by a task, preserving first mention order.

    ``faucet`` + ``tap`` is one group; ``faucet`` + ``stove`` is a multi-target
    task. This is a diagnostic contract, not a claim that all targets can be
    executed in one render.
    """
    words = re.findall(r"[a-z0-9]+", (task or "").lower())
    groups: list[str] = []
    seen: set[str] = set()
    for word in words:
        if word in _TASK_STOPWORDS:
            continue
        group = _canonical_group_for_token(word)
        if group and group not in seen:
            seen.add(group)
            groups.append(group)
    return groups


def detect_multi_target(task: str) -> bool:
    """True when a task names two or more distinct fixture groups."""
    return len(task_target_groups(task)) >= 2


def classify_target_kind(obj: SceneObject) -> str:
    """Advisory label-derived target kind: ``openable`` or ``static``.

    This is not capture truth. It is a conservative placement hint for articulated
    fixtures such as refrigerator doors, drawers, cabinets, and dishwashers.
    """
    text = f"{obj.label} {obj.category} {obj.id}".lower()
    tokens = re.findall(r"[a-z0-9]+", text)
    groups = {_canonical_group_for_token(token) for token in tokens}
    return "openable" if any(group in _OPENABLE_TARGET_GROUPS for group in groups) else "static"


def is_openable_target(obj: SceneObject) -> bool:
    return classify_target_kind(obj) == "openable"


# Match strength, lower == better. A label that literally contains the task token
# beats a synonym proxy ("faucet" label over a "sink" proxy), which beats a weaker
# cross-fixture proxy. Used to rank candidate objects for one intent token.
_RANK_DIRECT = 0
_RANK_SYNONYM = 1
_RANK_PROXY = 2
_RANK_NONE = 3


def _label_match_rank(token: str, label: str) -> int:
    """How well ``label`` matches intent ``token``: direct < synonym < proxy < none.

    Substring (not equality) because labels are noisy — "kitchen_faucet_01" should
    match the intent "faucet". Ranking lets the fallback prefer the object whose
    name most directly states the intent over a mere proxy that happens to share a
    group (e.g. a "sink" standing in for a faucet).
    """
    label = (label or "").lower()
    if not token or not label:
        return _RANK_NONE
    if token in label:
        return _RANK_DIRECT
    if any(s in label for s in _synonyms_of(token)):
        return _RANK_SYNONYM
    if any(p in label for p in _proxies_of(token)):
        return _RANK_PROXY
    return _RANK_NONE


def resolve_target_by_label(
    task: str,
    objects: List[SceneObject],
) -> Optional[SceneObject]:
    """Pure, dependency-free fuzzy resolver — the no-VLM fallback.

    Reduces the task to its content nouns (verbs/articles dropped), then for each
    noun (most specific first) ranks objects by how directly their label names that
    noun: a literal substring hit beats a synonym (faucet -> tap/spout), which beats
    a cross-fixture proxy (faucet -> sink). The first noun that matches anything
    wins; among its matches the best rank wins, and the shortest label breaks ties
    toward the most direct name ("faucet" over "kitchen_faucet_handle"). Returns
    ``None`` when nothing matches.
    """
    if not objects:
        return None
    for token in _task_intent_tokens(task):
        ranked = [
            (rank, len(obj.label or ""), obj.label or "", obj)
            for obj in objects
            for rank in (_label_match_rank(token, obj.label),)
            if rank != _RANK_NONE
        ]
        if ranked:
            ranked.sort(key=lambda t: (t[0], t[1], t[2]))
            return ranked[0][3]
    return None


def resolve_targets_by_label(task: str, objects: List[SceneObject]) -> list[SceneObject]:
    """Resolve every distinct target group mentioned by ``task`` using the label fallback.

    Results are ordered by first mention in the task. The existing single-target
    ``resolve_target_by_label`` API remains the primary/first target.
    """
    if not objects:
        return []
    out: list[SceneObject] = []
    seen_ids: set[str] = set()
    for group in task_target_groups(task):
        ranked = [
            (rank, len(obj.label or ""), obj.label or "", obj)
            for obj in objects
            for token in _synonyms_of(group)
            for rank in (_label_match_rank(token, obj.label),)
            if rank != _RANK_NONE
        ]
        if not ranked:
            continue
        ranked.sort(key=lambda t: (t[0], t[1], t[2]))
        selected = ranked[0][3]
        if selected.id not in seen_ids:
            seen_ids.add(selected.id)
            out.append(selected)
    return out
