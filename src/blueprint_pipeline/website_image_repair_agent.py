"""Bounded repair of rejected website image edits (ADP-009B, public_scene_day_14).

After the independent review rejects the prepared views, one Agents SDK call
reads the review's reasons with each original and edited view and proposes at
most ``MAX_REPAIRS`` targeted re-edits. The agent never spends, never edits and
never approves: deterministic code performs each re-edit through the existing
paid image lane (reservation, receipt, settlement), and the independent review
runs again and alone decides. Off unless ``BLUEPRINT_WEBSITE_IMAGE_REPAIR_AGENT``
is set. A retained plan is reused on restart; an uncertain one is never re-bought.
"""
from __future__ import annotations

import base64
import fcntl
import json
import os
from io import BytesIO
from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

from .common import write_json
from .decision_evidence_contracts import canonical_digest
from .local_reconstruction_adapters import _sha256_file

ENABLE_ENV = "BLUEPRINT_WEBSITE_IMAGE_REPAIR_AGENT"
MODEL = "gpt-6-sol"
MAX_REPAIRS = 3
MAX_OUTPUT_TOKENS = 2000
MAX_INPUT_TOKENS = 32_000
MAX_COST_USD = 0.5
PREVIEW_LONG_SIDE = 768
CAPABILITY = "website_image_repair_planner"
INSTRUCTIONS = (
    "You plan repairs for rejected image edits of ONE room. An independent review rejected the prepared "
    "views; its reasons are below. For each view you see the ORIGINAL and the EDITED image. The edits must "
    "remove the listed task objects and every person or body part, keep the empty bay of a removed built-in "
    "object open to its full depth, keep every other object exactly as it is, and look consistent across "
    "views. Choose at most {limit} views whose problem a single new edit of that view from its ORIGINAL "
    "could fix, and for each give one short, concrete instruction naming what to keep, restore or leave "
    "open, for example 'Keep the paper towel package on the island exactly as in the original'. Never ask "
    "to add new objects, text or people. Never judge whether the set passes; the independent review does "
    "that. Return an empty list when no targeted re-edit would fix the rejection."
)


class RepairRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    frame_id: str = Field(min_length=1, max_length=200)
    instruction: str = Field(min_length=1, max_length=400)


class RepairPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    repairs: list[RepairRequest] = Field(max_length=MAX_REPAIRS)
    summary: str = Field(max_length=600)


def image_repair_enabled() -> bool:
    return os.getenv(ENABLE_ENV, "").strip().casefold() in {"1", "true", "yes", "on"}


def _preview(path: Path) -> str:
    with Image.open(path) as image:
        image = image.convert("RGB")
        image.thumbnail((PREVIEW_LONG_SIDE, PREVIEW_LONG_SIDE))
        stream = BytesIO()
        image.save(stream, format="PNG")
    return "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode()


def _agent_input(*, frames: Sequence[Mapping[str, Any]], originals: Mapping[str, Mapping[str, Any]],
                 review: Mapping[str, Any], targets: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [{"type": "input_text", "text": (
        "Review reasons (data): " + json.dumps(review.get("review") or {}, sort_keys=True)
        + " Targets (data): " + json.dumps([{key: target.get(key) for key in (
            "target_id", "semantic_label", "target_class", "task_effect", "disposition")}
            for target in targets], sort_keys=True))}]
    for frame in frames:
        original = originals[frame["frame_id"]]
        for label, item in (("ORIGINAL", original), ("EDITED", frame)):
            path = Path(item["image_path"])
            if _sha256_file(path) != item["image_digest"]:
                raise ValueError("website_image_repair_source_changed")
            content.append({"type": "input_text", "text": f"{label} view {frame['frame_id']}"})
            content.append({"type": "input_image", "image_url": _preview(path), "detail": "high"})
    return [{"role": "user", "content": content}]


def _default_invoker():
    from .task_evaluation_supervisor.agents_sdk import OpenAIAgentsSDKConfig, OpenAIAgentsSDKInvoker
    return OpenAIAgentsSDKInvoker(OpenAIAgentsSDKConfig(
        model=MODEL, max_turns=1, max_output_tokens=MAX_OUTPUT_TOKENS, max_input_tokens=MAX_INPUT_TOKENS,
        allow_live_invocation=True, tracing_disabled=True, max_inference_cost_usd=MAX_COST_USD))


def plan_image_repairs(*, frames: Sequence[Mapping[str, Any]], originals: Mapping[str, Mapping[str, Any]],
                       failed_review: Mapping[str, Any], targets: Sequence[Mapping[str, Any]],
                       task_context: Mapping[str, Any], output_root: Path, invoker: Any = None) -> dict[str, Any]:
    """Buy at most one retained repair plan for one failed review; never an approval."""
    from .website_task_context import reserve_website_preparation_spend, website_webapp_request

    binding = {"kind": "website_image_repair_plan", "revision": 1, "model": MODEL,
               "max_output_tokens": MAX_OUTPUT_TOKENS, "max_repairs": MAX_REPAIRS,
               "instructions": INSTRUCTIONS.format(limit=MAX_REPAIRS),
               "failed_review_digest": canonical_digest(dict(failed_review.get("review") or {})),
               "failed_review_request_digest": failed_review.get("request_digest"),
               "frames": [{"frame_id": frame["frame_id"], "image_digest": frame["image_digest"],
                           "original_image_digest": originals[frame["frame_id"]]["image_digest"]}
                          for frame in frames],
               "task_context_digest": task_context["context_digest"]}
    digest = canonical_digest(binding)
    root = output_root / "repair_agent"
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"plan-{digest[7:]}.json"
    with (root / f"plan-{digest[7:]}.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("website_image_repair_in_progress") from exc
        if path.is_file():
            receipt = json.loads(path.read_text())
            if receipt.get("status") != "completed":
                # An uncertain plan call is never bought again.
                raise ValueError("website_image_repair_requires_reconciliation")
            if (receipt.get("binding_digest") != digest
                    or receipt.get("plan_digest") != canonical_digest(receipt.get("plan"))):
                raise ValueError("website_image_repair_receipt_invalid")
        else:
            receipt = _buy_plan(path=path, root=root, digest=digest, binding=binding, frames=frames,
                                originals=originals, failed_review=failed_review, targets=targets,
                                task_context=task_context, invoker=invoker,
                                reserve=reserve_website_preparation_spend)
    settlement_path = root / f"plan-{digest[7:]}.settlement.json"
    if not settlement_path.is_file():
        # Release the unused quote; the request count is never replenished.
        command = {"task_context_digest": task_context["context_digest"], "allocation_binding_digest": digest,
                   "provider": "openai", "completed_request_count": 1,
                   "provider_charge_amount_usd": round(min(float(receipt["cost_usd"]), MAX_COST_USD), 6),
                   "usage_receipt_digest": canonical_digest({"usage": receipt["usage"]})}
        settlement = website_webapp_request(capture_id=task_context["capture_id"],
            operation="preparation-settlement", payload={"request_id": task_context["request_id"],
                "scene_id": task_context["scene_id"], "settlement": command})
        if settlement.get("status") != "settled":
            raise ValueError("website_image_repair_settlement_invalid")
        write_json(settlement_path, settlement)
    return receipt


def _buy_plan(*, path: Path, root: Path, digest: str, binding: Mapping[str, Any],
              frames: Sequence[Mapping[str, Any]], originals: Mapping[str, Mapping[str, Any]],
              failed_review: Mapping[str, Any], targets: Sequence[Mapping[str, Any]],
              task_context: Mapping[str, Any], invoker: Any, reserve: Any) -> dict[str, Any]:
    input_value = _agent_input(frames=frames, originals=originals, review=failed_review, targets=targets)
    admission, _grant = reserve(
        task_context=task_context, binding_digest=digest, maximum_cost_usd=MAX_COST_USD, request_count=1,
        resource_class="openai_api_candidate", provider="openai")
    with path.open("x") as stream:
        json.dump({"status": "submitting", "binding_digest": digest, "binding": binding,
                   "admission": admission}, stream)
        stream.flush()
        os.fsync(stream.fileno())
    from .task_evaluation_supervisor.agents_sdk import AgentsSDKAgentSpec
    from .task_evaluation_supervisor.inference_reservations import InferenceReservationAudit
    selected_invoker = invoker if invoker is not None else _default_invoker()
    audit = InferenceReservationAudit(run_root=root, run_id=digest[7:23])
    selected_invoker.configure_reservation_audit(record_reservation=audit.record_reservation,
        record_completion=audit.record_completion, restored_reserved_cost_usd=0.0)
    spec = AgentsSDKAgentSpec(run_id=digest[7:23], capability=CAPABILITY,
        name="Blueprint Website Image Repair Planner", instructions=binding["instructions"],
        model=MODEL, max_turns=1, max_output_tokens=MAX_OUTPUT_TOKENS, max_input_tokens=MAX_INPUT_TOKENS,
        output_type=RepairPlan)
    try:
        invocation = selected_invoker.invoke(spec, input_value)
    finally:
        audit.write_manifest()
    plan = RepairPlan.model_validate(invocation.output).model_dump(mode="json")
    frame_ids = [frame["frame_id"] for frame in frames]
    requested = [row["frame_id"] for row in plan["repairs"]]
    if len(set(requested)) != len(requested) or any(frame_id not in frame_ids for frame_id in requested):
        raise ValueError("website_image_repair_plan_frames_invalid")
    cost = float(invocation.cost_usd)
    receipt = {"status": "completed", "binding_digest": digest, "binding": binding, "admission": admission,
               "plan": plan, "plan_digest": canonical_digest(plan), "model": invocation.model,
               "usage": invocation.usage, "cost_usd": cost, "basis": "model_repair_plan",
               "approves_views": False}
    temporary = path.with_suffix(".tmp")
    write_json(temporary, receipt)
    os.replace(temporary, path)
    return receipt


def repair_rejected_views(*, selected: Sequence[Mapping[str, Any]], object_removal_frames: Sequence[Mapping[str, Any]],
                          original_frames: Sequence[Mapping[str, Any]], plan: Mapping[str, Any],
                          failed_review: Mapping[str, Any], output_root: Path, task_context: Mapping[str, Any],
                          invoker: Any = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Plan, perform and independently re-review at most MAX_REPAIRS targeted re-edits."""
    from .website_image_completion import complete_background_images, verify_completed_background

    sources = {frame["frame_id"]: frame for frame in object_removal_frames}
    originals = {}
    for frame in selected:
        source = sources.get(frame["frame_id"])
        if source is None:
            raise ValueError("website_image_repair_source_missing")
        originals[frame["frame_id"]] = {"image_path": source["original_image_path"],
                                        "image_digest": source["original_image_digest"]}
    receipt = plan_image_repairs(frames=selected, originals=originals, failed_review=failed_review,
                                 targets=plan["targets"], task_context=task_context,
                                 output_root=output_root, invoker=invoker)
    repairs = receipt["plan"]["repairs"]
    if not repairs:
        return list(map(dict, selected)), {**failed_review, "repair_plan": receipt["plan"]}
    repaired_ids = {row["frame_id"] for row in repairs}
    # Every repaired view copies the revealed space of one accepted edited view.
    anchor = next((frame for frame in sorted(selected, key=lambda row: -int(row.get("generated_pixel_count") or 0))
                   if frame.get("generated_pixels_present") and frame["frame_id"] not in repaired_ids), None)
    result = [dict(frame) for frame in selected]
    for row in repairs:
        index = next(i for i, frame in enumerate(result) if frame["frame_id"] == row["frame_id"])
        source = sources[row["frame_id"]]
        edited = complete_background_images(
            frames=[source], task_digest=plan["task_context_sha256"], output_root=output_root / "repairs",
            admission={}, token=os.getenv("OPENAI_API_KEY", ""), targets=plan["targets"],
            task_context=task_context, repair_instruction=row["instruction"],
            repair_reference_path=Path(anchor["image_path"]) if anchor is not None else None)[0]
        result[index] = {**edited, "repair_instruction": row["instruction"]}
    review = verify_completed_background(frames=result, original_frames=original_frames, plan=plan,
                                         output_root=output_root, task_context=task_context)
    return result, {**review, "prior_failed_review": failed_review, "repair_plan": receipt["plan"],
                    "repair_plan_digest": receipt["plan_digest"], "repaired_frame_ids": sorted(repaired_ids)}
