"""Whole-view agreement review of generated views before a refined splat.

World Labs' posedImages2Splats takes ordinary whole frames (image, depth and
camera only), and its splat guidance asks that generated views be compared with
overlapping views before splatting: objects keep their position, orientation,
shape and appearance, or the splat ghosts and duplicates surfaces. The
completion review judges the generated set for the edit, not each whole view
against the photographs, and records no per-view verdict. This is one bounded
call for every candidate at once, retained so a restart never buys it again.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .atlas_refined_splat_forms import REVIEW_RECORD as RECORD
from .common import write_json

REFERENCES_PER_VIEW = 3
MINIMUM_OVERLAP = 0.02
MAX_OUTPUT_TOKENS = 8192
PROMPT = (
    "Each CANDIDATE image is a generated view of a real space from a new camera. Each REFERENCE image is a "
    "photograph of the same space from a nearby camera. All of them will be combined into one 3D "
    "reconstruction, where any disagreement between views becomes ghosted or duplicated surfaces. For "
    "every candidate, compare everything it shares with the references listed for it. Admit it whole only "
    "if every shared object keeps its position, orientation, shape and appearance; nothing is invented, "
    "duplicated, substituted or missing where a reference shows the same surface; and the layout of the "
    "floor, walls and supports agrees. New surfaces no reference shows are acceptable when they continue "
    "the visible layout plausibly. Minor blur, seams and lighting differences are acceptable. If a "
    "candidate shares nothing you can compare, do not admit it. Return JSON: {\"views\": [{\"view_id\": "
    "exact candidate id, \"admit_whole\": boolean, \"reason\": short}]} with one entry per candidate. "
    "This is visual agreement review, not geometry, metric or physical validation."
)


def _digest(rgb) -> str:
    return "sha256:" + hashlib.sha256(np.ascontiguousarray(rgb).tobytes()).hexdigest()


def _overlap(reference, candidate) -> float:
    """The share of a reference's sampled depth that lands inside the candidate camera."""
    from .atlas_depth_splat import _rotation_matrix
    from .atlas_scene_preservation import _project, _world

    sampled = np.zeros(reference["depth"].shape, bool)
    sampled[::8, ::8] = True
    sampled &= np.isfinite(reference["depth"]) & (reference["depth"] > 0)
    if not sampled.any():
        return 0.0
    posed = {**reference, "rotation": _rotation_matrix(reference["extrinsics"]["quaternion"]),
             "position": np.asarray(reference["extrinsics"]["position"], float)}
    points, _, _ = _world(posed, sampled)
    return float(_project(points, {"intrinsics": candidate["intrinsics"],
                                   "extrinsics": candidate["extrinsics"]})[3].mean())


def _ask(*, prompt, pairs, output_root: Path, binding_digest: str) -> dict:
    from google import genai
    from google.genai import types
    from .clean_plate_removal_analysis_gemini import DEFAULT_MODEL, _api_key
    from .core.handoff_lease import handoff_lease_checkpoint
    from .website_image_completion import _review_contents

    with genai.Client(api_key=_api_key()[0], http_options=types.HttpOptions(
            timeout=300_000, retry_options=types.HttpRetryOptions(attempts=1))) as client:
        with _review_contents(client=client, types=types, pairs=pairs, prompt=prompt, output_root=output_root,
                              binding_digest=binding_digest) as contents:
            handoff_lease_checkpoint("gemini_whole_view_review")
            response = client.models.generate_content(model=DEFAULT_MODEL, contents=contents,
                config=types.GenerateContentConfig(response_mime_type="application/json",
                                                   max_output_tokens=MAX_OUTPUT_TOKENS,
                                                   media_resolution="MEDIA_RESOLUTION_HIGH"))
    finished = bool(response.candidates) and response.candidates[0].finish_reason == "STOP"
    try:
        answer = json.loads(response.text) if finished else None
    except ValueError:
        answer = None
    # An incomplete answer is retained too: it admits nothing and is never bought again.
    return {"complete": answer is not None, "answer": answer}


def review_whole_views(*, views, root: Path, task_context, policy, source_video_digest, checkpoint=None,
                       retained_only=False) -> dict:
    """Per-view ``admit_whole`` verdicts for the partly admitted (generated) merged views.

    ``views`` are ``merge_duplicate_views`` rows. Real frames are references; each
    candidate is compared with the real frames that overlap it most. Priced under
    the reviewed review cap; over it, or after an uncertain earlier call, nothing
    is admitted and nothing more is bought. The verdicts are kept in the root.
    ``retained_only`` (export replay and retained finalization) adopts a
    completed review and refuses before any reservation instead of buying one.
    """
    from .clean_plate_removal_analysis_gemini import DEFAULT_MODEL, _api_key
    from .decision_evidence_contracts import canonical_digest
    from .website_gemini_receipts import gemini_quote, retained_gemini_call

    candidates = [(base, view) for base, view, _ in views
                  if view.get("keep_mask") is not None and not view["keep_mask"].all() and view["keep_mask"].any()]
    whole = [(base, view) for base, view, _ in views if view.get("keep_mask") is None or view["keep_mask"].all()]
    value = {"schema_version": "atlas_whole_view_review.v1", "binding_digest": None,
             "status": "no_overlapping_real_frame" if candidates else "not_required",
             "verdicts": [{"position": base, "frame_id": view["frame_id"], "admit_whole": False,
                           "reason": "not_reviewed"} for base, view in candidates]}
    pairs = {}
    for base, view in candidates:
        ranked = sorted(((overlap, ref) for ref, real in whole
                         if (overlap := _overlap(real, view)) >= MINIMUM_OVERLAP), reverse=True)
        pairs[base] = [ref for _, ref in ranked[:REFERENCES_PER_VIEW]]
    for verdict in value["verdicts"]:
        if not pairs[verdict["position"]]:
            verdict["reason"] = "no_overlapping_real_frame"  # Nothing to compare it with.
    reviewable = [(base, view) for base, view in candidates if pairs[base]]
    if reviewable:
        frames = dict(whole)
        labels = [(f"CANDIDATE candidate-{base:04d}; compare with: "
                   + ", ".join(f"reference-{ref:04d}" for ref in pairs[base]), base, view) for base, view in reviewable]
        labels += [(f"REFERENCE reference-{ref:04d}", ref, frames[ref])
                   for ref in sorted({ref for base, _ in reviewable for ref in pairs[base]})]
        binding = {"kind": "whole_view_agreement_review", "revision": 1, "model": DEFAULT_MODEL, "prompt": PROMPT,
                   "max_output_tokens": MAX_OUTPUT_TOKENS, "media_resolution": "MEDIA_RESOLUTION_HIGH",
                   "images": [{"label": label, "frame_id": view["frame_id"], "rgb_sha256": _digest(view["rgb"])}
                              for label, _, view in labels]}
        # HIGH media is at most 1120 tokens per image; bytes bound the text.
        input_tokens = len(PROMPT.encode()) + 1024 + sum(len(label.encode()) + 32 + 1120 for label, _, _ in labels)
        quote = gemini_quote(model=DEFAULT_MODEL, input_tokens=input_tokens, max_output_tokens=MAX_OUTPUT_TOKENS)
        cap = policy.get("maximum_review_cost_usd")
        value["binding_digest"] = canonical_digest(binding)
        if not isinstance(cap, (int, float)) or quote > cap:
            value["status"] = "over_review_cap"
        else:
            if checkpoint:
                checkpoint("refined_splat_whole_view_review")

            def preflight():
                if retained_only:
                    raise ValueError("atlas_export_replay_cached_stage_missing")
                if not _api_key()[0]:
                    raise ValueError("atlas_whole_view_review_key_missing")
                from google import genai  # noqa: F401

            def invoke():
                from PIL import Image
                from .local_reconstruction_adapters import _sha256_file
                media = root / "whole_view_review"
                media.mkdir(parents=True, exist_ok=True)
                items = []
                for label, base, view in labels:
                    path = media / f"{base:04d}.png"
                    Image.fromarray(view["rgb"]).save(path)
                    items.append((label, {"image_path": str(path), "image_digest": _sha256_file(path)}))
                return _ask(prompt=PROMPT, pairs=items, output_root=media, binding_digest=value["binding_digest"])
            try:
                result = retained_gemini_call(output_root=root / "whole_view_review/receipts", binding=binding,
                    task_context=task_context, maximum_cost_usd=quote, preflight=preflight, invoke=invoke,
                    source_video_digest=source_video_digest)
            except ValueError as error:
                if str(error) != "website_gemini_requires_reconciliation":
                    raise
                result = None  # An uncertain earlier call never authorizes another.
            answer = (result or {}).get("answer")
            rows = answer.get("views") if isinstance(answer, dict) else None
            answers = {row.get("view_id"): row for row in (rows if isinstance(rows, list) else [])
                       if isinstance(row, dict)}
            value["status"] = "reviewed" if result is not None else "unresolved"
            for verdict in value["verdicts"]:
                if pairs[verdict["position"]]:
                    row = answers.get(f"candidate-{verdict['position']:04d}") or {}
                    verdict.update(admit_whole=row.get("admit_whole") is True,
                                   reason=str(row.get("reason") or "no_verdict")[:300])
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / RECORD, value)
    return value
