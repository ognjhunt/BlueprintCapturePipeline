"""VLM visual sanity-QC gate for rendered frames + WAM outputs.

Catches anomalies a human would flag at a glance — an incongruous background (a cityscape through
a window in a scene that should read as an enclosed kitchen), large dark/void regions, a missing or
duplicated robot, broken/melted geometry, the wrong room — by asking a VLM (Gemini, the SAME
integration ``scene_semantics.py`` uses) to review sampled frames against a rubric and return
structured verdicts. It is a trust/quality gate that sits between "rendered" and "trusted as an
OSCAR seed / shipped" — NOT a blocker on the product core.

Design:
  * Pure, GPU-free, dependency-light. The actual Gemini call is injected (``generate``) so the whole
    module is unit-testable without ``google-genai`` or any network; the default ``generate`` lazily
    imports the SDK and mirrors scene_semantics (file/env key, model cascade, JSON mime, thinking-part
    filtering — no ``thinking_config`` which silently fails).
  * Sample frames (don't review every one) to keep it cheap.
  * Verdicts are normalized + aggregated; the gate "flags" on any anomaly at/above a severity floor,
    a not-coherent frame, or a large dark region.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

# scene_semantics.py model cascade — gemini-3-flash-preview first (fast, correct JSON), per repo notes
DEFAULT_MODEL_CASCADE = (
    "gemini-3-flash-preview",
    "gemini-3.1-pro-preview",
    "gemini-2.5-flash",
    "gemini-2.5-pro",
)

_SEVERITY_RANK = {"none": 0, "low": 1, "medium": 2, "high": 3}
# at/above this severity a frame is "flagged" for human attention
DEFAULT_SEVERITY_FLOOR = "medium"
# a frame more than this fraction dark is itself an anomaly (the under-lit-basin / void case)
DEFAULT_DARK_REGION_FLOOR = 0.30

_ANOMALY_CATEGORIES = (
    "incongruous_background",   # e.g. a city/outdoors where an enclosed room is expected
    "dark_or_void_region",      # large black/under-lit area
    "robot_missing_or_wrong",   # robot absent, duplicated, or malformed where expected
    "broken_geometry",          # melted/distorted/floating geometry
    "wrong_scene",              # wrong room / wrong objects for the task
    "other",
)


# ----------------------------- prompt -----------------------------

def build_qc_prompt(task_description: str, *, scene_context: str = "") -> str:
    """Rubric prompt asking the VLM to flag the anomalies a human would catch, as strict JSON."""
    task = (task_description or "a robot manipulation scene").strip()
    ctx = f"\nExpected scene context: {scene_context.strip()}" if scene_context.strip() else ""
    cats = ", ".join(_ANOMALY_CATEGORIES)
    return (
        "You are a strict visual QC reviewer for rendered robotics scenes. You are shown ONE frame "
        f"that is meant to depict: {task}.{ctx}\n\n"
        "Judge it the way an attentive human would at a glance and report ONLY problems that a human "
        "would obviously notice. In particular check:\n"
        "  - Background coherence: does anything visible (e.g. through a window) contradict the "
        "expected setting? A cityscape/outdoors behind a scene that should read as an enclosed room "
        "is an anomaly.\n"
        "  - Large dark/black/void regions that look unlit or like missing geometry.\n"
        "  - Is the expected robot present, single, and structurally plausible (not duplicated, "
        "melted, or absent)?\n"
        "  - Broken/distorted/floating geometry.\n"
        "  - Is this even the right kind of scene/room for the task?\n\n"
        "Respond with STRICT JSON only, no prose, exactly this shape:\n"
        "{\n"
        '  "coherent": true,\n'
        '  "robot_visible": true,\n'
        '  "background_consistent": true,\n'
        '  "dark_region_fraction": 0.0,\n'
        '  "overall_severity": "none",\n'
        '  "anomalies": [\n'
        '    {"category": "one of: ' + cats + '", "description": "what a human would notice", '
        '"severity": "low|medium|high"}\n'
        "  ],\n"
        '  "summary": "one sentence"\n'
        "}\n"
        "Use an empty anomalies list and overall_severity \"none\" when the frame looks correct."
    )


# ----------------------------- parsing / normalization -----------------------------

def _extract_json_object(text: str) -> dict:
    """json.loads, else the first {...} block (mirrors scene_semantics._extract_json_object)."""
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


def extract_model_text(response: Any) -> str:
    """Final answer text from a google-genai response, SKIPPING thinking parts (thought=True),
    which contain reasoning, not the answer (mirrors scene_semantics._extract_response_text)."""
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


def _as_bool(value: Any, default: bool = True) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() not in {"false", "no", "0", ""}
    if value is None:
        return default
    return bool(value)


def _as_fraction(value: Any) -> float:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, f))


def _norm_severity(value: Any) -> str:
    s = str(value or "").strip().lower()
    return s if s in _SEVERITY_RANK else ("none" if not s else "low")


def parse_qc_verdict(raw_text: str) -> dict:
    """Normalize a model JSON reply into a stable verdict dict. Robust to junk/missing fields:
    a malformed reply yields ``parsed=False`` (treated as inconclusive, not as 'clean')."""
    obj = _extract_json_object(raw_text or "")
    if not obj:
        return {
            "parsed": False, "coherent": None, "robot_visible": None,
            "background_consistent": None, "dark_region_fraction": 0.0,
            "overall_severity": "none", "anomalies": [], "summary": "",
            "raw_text": (raw_text or "")[:500],
        }
    anomalies = []
    for a in obj.get("anomalies") or []:
        if not isinstance(a, Mapping):
            continue
        cat = str(a.get("category") or "other").strip().lower()
        anomalies.append({
            "category": cat if cat in _ANOMALY_CATEGORIES else "other",
            "description": str(a.get("description") or "").strip(),
            "severity": _norm_severity(a.get("severity")),
        })
    return {
        "parsed": True,
        "coherent": _as_bool(obj.get("coherent"), True),
        "robot_visible": _as_bool(obj.get("robot_visible"), True),
        "background_consistent": _as_bool(obj.get("background_consistent"), True),
        "dark_region_fraction": _as_fraction(obj.get("dark_region_fraction")),
        "overall_severity": _norm_severity(obj.get("overall_severity")),
        "anomalies": anomalies,
        "summary": str(obj.get("summary") or "").strip(),
        "raw_text": (raw_text or "")[:500],
    }


def verdict_is_flagged(verdict: Mapping[str, Any], *, severity_floor: str = DEFAULT_SEVERITY_FLOOR,
                       dark_region_floor: float = DEFAULT_DARK_REGION_FLOOR) -> bool:
    """A frame is flagged for human attention when: it failed to parse, the model called it
    not-coherent / robot-missing / background-inconsistent, the dark region is too large, or any
    anomaly meets the severity floor."""
    if not verdict.get("parsed", False):
        return True
    floor = _SEVERITY_RANK.get(severity_floor, 2)
    if verdict.get("coherent") is False:
        return True
    if verdict.get("robot_visible") is False:
        return True
    if verdict.get("background_consistent") is False:
        return True
    if float(verdict.get("dark_region_fraction") or 0.0) >= dark_region_floor:
        return True
    if _SEVERITY_RANK.get(verdict.get("overall_severity", "none"), 0) >= floor:
        return True
    return any(_SEVERITY_RANK.get(a.get("severity", "none"), 0) >= floor
               for a in verdict.get("anomalies") or [])


def worst_severity(verdicts: Sequence[Mapping[str, Any]]) -> str:
    rank = 0
    for v in verdicts:
        rank = max(rank, _SEVERITY_RANK.get(v.get("overall_severity", "none"), 0))
        for a in v.get("anomalies") or []:
            rank = max(rank, _SEVERITY_RANK.get(a.get("severity", "none"), 0))
    for name, r in _SEVERITY_RANK.items():
        if r == rank:
            return name
    return "none"


# ----------------------------- frame sampling -----------------------------

def sample_frame_paths(frame_paths: Sequence[Any], sample_n: int) -> list:
    """Evenly-spaced sample of up to ``sample_n`` frames (always includes first + last). Sorting is
    the caller's responsibility for ordered frames; this preserves the given order."""
    paths = list(frame_paths)
    n = len(paths)
    if sample_n <= 0 or n <= sample_n:
        return paths
    if sample_n == 1:
        return [paths[0]]
    idxs = sorted({round(i * (n - 1) / (sample_n - 1)) for i in range(sample_n)})
    return [paths[i] for i in idxs]


# ----------------------------- gemini call (injectable) -----------------------------

def _gemini_review_image(image_bytes: bytes, prompt: str, *,
                         models: Sequence[str] = DEFAULT_MODEL_CASCADE,
                         mime_type: str = "image/png") -> str:
    """Default ``generate``: send the image + prompt to Gemini and return the raw model text.
    Lazily imports google-genai so the module imports without it. Mirrors scene_semantics:
    file/env key, model cascade, response_mime_type=application/json, thinking-part filtering."""
    api_key = (os.getenv("GOOGLE_GENAI_API_KEY") or os.getenv("GEMINI_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError("missing_GOOGLE_GENAI_API_KEY")
    from google import genai  # type: ignore

    client = genai.Client(api_key=api_key)
    override = (os.getenv("BLUEPRINT_RENDER_QC_GEMINI_MODEL") or "").strip()
    models_to_try = [override] if override else list(models)
    image_part = genai.types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
    last_exc: Exception | None = None
    for model in models_to_try:
        try:
            response = client.models.generate_content(
                model=model,
                contents=[image_part, prompt],
                config={"response_mime_type": "application/json"},  # no thinking_config (silent fail)
            )
            text = extract_model_text(response)
            if text:
                return text
        except Exception as exc:  # noqa: BLE001 - try the next model in the cascade
            last_exc = exc
            continue
    if last_exc is not None:
        raise last_exc
    return ""


GenerateFn = Callable[[bytes, str], str]


def review_render_frame(image: Any, task_description: str, *,
                        generate: GenerateFn | None = None, scene_context: str = "",
                        frame_label: str = "") -> dict:
    """Review a single frame (path / Path / raw bytes) and return a normalized verdict dict
    (plus ``frame`` label and ``flagged``). ``generate(image_bytes, prompt) -> raw_text`` is
    injectable; defaults to the real Gemini call."""
    gen = generate or _gemini_review_image
    if isinstance(image, (bytes, bytearray)):
        image_bytes = bytes(image)
        label = frame_label or "<bytes>"
    else:
        p = Path(image)
        image_bytes = p.read_bytes()
        label = frame_label or p.name
    prompt = build_qc_prompt(task_description, scene_context=scene_context)
    try:
        raw = gen(image_bytes, prompt)
        verdict = parse_qc_verdict(raw)
        verdict["error"] = None
    except Exception as exc:  # noqa: BLE001 - a failed review is inconclusive, surfaced as flagged
        verdict = parse_qc_verdict("")
        verdict["error"] = repr(exc)[:300]
    verdict["frame"] = label
    verdict["flagged"] = verdict_is_flagged(verdict)
    return verdict


@dataclass
class RenderQCReport:
    task_description: str
    frames_reviewed: int
    flagged: bool
    worst_severity: str
    anomalies: list = field(default_factory=list)   # flat [{frame, category, description, severity}]
    per_frame: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "schema_version": "render_visual_qc.v1",
            "task_description": self.task_description,
            "frames_reviewed": self.frames_reviewed,
            "flagged": self.flagged,
            "worst_severity": self.worst_severity,
            "anomalies": self.anomalies,
            "per_frame": self.per_frame,
        }


def qc_render_frames(frame_paths: Sequence[Any], task_description: str, *, sample_n: int = 3,
                     generate: GenerateFn | None = None, scene_context: str = "") -> RenderQCReport:
    """Sample frames, review each, and aggregate into a single gate report. ``flagged`` is true if
    ANY reviewed frame is flagged — the signal a human would act on."""
    sampled = sample_frame_paths(frame_paths, sample_n)
    per_frame = [review_render_frame(p, task_description, generate=generate, scene_context=scene_context)
                 for p in sampled]
    anomalies = []
    for v in per_frame:
        for a in v.get("anomalies") or []:
            anomalies.append({"frame": v.get("frame"), **a})
    return RenderQCReport(
        task_description=task_description,
        frames_reviewed=len(per_frame),
        flagged=any(v.get("flagged") for v in per_frame),
        worst_severity=worst_severity(per_frame),
        anomalies=anomalies,
        per_frame=per_frame,
    )


def qc_render_output_dir(render_out_dir: Any, task_description: str, *, sample_n: int = 3,
                         generate: GenerateFn | None = None, scene_context: str = "",
                         glob: str = "**/robot_pov_*.png") -> RenderQCReport:
    """Convenience: find the robot-POV frames under a render_output dir and QC a sample of them."""
    frames = sorted(Path(render_out_dir).glob(glob))
    return qc_render_frames(frames, task_description, sample_n=sample_n,
                            generate=generate, scene_context=scene_context)
