"""Seal independent AI review of one production ArtiFixer scene edit.

The ArtiFixer runtime deliberately emits an unreviewed generated candidate.
This module is the fail-closed join between that candidate and a separately
retained structured vision-review execution.  It grants appearance-review
authority only; it never grants collision, physics, simulator, or physical
truth authority.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .common import write_json
from .decision_evidence_contracts import canonical_digest, canonical_json
from .openai_official_cost_gate import build_openai_official_cost_run_gate
from .task_evaluation_supervisor.agents_sdk import (
    AgentsSDKAgentSpec,
    OpenAIAgentsSDKConfig,
    OpenAIAgentsSDKInvoker,
)
from .task_evaluation_supervisor.inference_reservations import (
    InferenceReservationAudit,
)


EXECUTION_SCHEMA_VERSION = "task_evaluation_artifixer_ai_visual_review_execution.v1"
RECEIPT_SCHEMA_VERSION = "task_evaluation_artifixer_ai_visual_review.v1"
FINAL_COMPOSITE_SCHEMA_VERSION = "public_scene_artifixer3d_final_composite.v1"
DUAL_TARGET_REVIEW_SCHEMA_VERSION = "task_evaluation_artifixer3d_dual_target_review_input.v1"
_SHA256 = re.compile(r"sha256:[0-9a-f]{64}\Z")
RIGHTS_SCHEMA_VERSION = "task_evaluation_artifixer_ai_visual_review_rights.v1"
AI_REVIEW_MODEL = "gpt-5.6-terra"
AI_REVIEWER_ID = "artifixer-independent-vision-reviewer-v1"
AI_REVIEW_MAX_COST_USD = 0.75
AI_REVIEW_MAX_INPUT_TOKENS = 80_000
AI_REVIEW_MAX_OUTPUT_TOKENS = 8_000
AI_REVIEW_MAX_FRAMES = 32
_PROMPT = (
    "Independently review every digest-identified final ArtiFixer frame. The "
    "target source object must be absent, the locally generated replacement "
    "surface must be visually plausible, and all non-target content must remain "
    "unchanged. Then decide whether the set is mutually consistent across "
    "cameras. Return every task/camera exactly once. Reject uncertainty. This is "
    "appearance review only, never collision, physics, or physical-world proof."
)


class TaskEvaluationArtifixerAIVisualReviewError(ValueError):
    """The candidate or retained independent review is not admissible."""


class ArtifixerFrameReviewDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str = Field(min_length=1, max_length=200)
    camera_id: str = Field(min_length=1, max_length=200)
    frame_sha256: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    source_object_absent: bool
    repair_is_locally_plausible: bool
    preserves_non_target_content: bool
    decision: Literal["accepted", "rejected"]
    rationale: str = Field(min_length=1, max_length=1_000)


class ArtifixerAIVisualReviewOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision: Literal["accepted", "rejected"]
    semantic_object_absence_review_passed: bool
    multiview_consistency_review_passed: bool
    summary: str = Field(min_length=1, max_length=4_000)
    frames: list[ArtifixerFrameReviewDecision] = Field(
        min_length=1, max_length=AI_REVIEW_MAX_FRAMES
    )


def _read(path: Path, *, code: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TaskEvaluationArtifixerAIVisualReviewError(code) from exc
    if path.is_symlink() or not isinstance(value, Mapping):
        raise TaskEvaluationArtifixerAIVisualReviewError(code)
    return dict(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _verified_frame(path: Path, record: Mapping[str, Any]) -> dict[str, Any]:
    raw = Path(str(record.get("path") or "")).expanduser()
    candidate = raw if raw.is_absolute() else path.parent / raw
    resolved = candidate.resolve()
    if (
        candidate.is_symlink()
        or not resolved.is_file()
        or resolved.stat().st_size != record.get("size_bytes")
        or _sha256(resolved) != record.get("sha256")
    ):
        raise TaskEvaluationArtifixerAIVisualReviewError("artifixer_ai_review_frame_bytes_invalid")
    return {
        "frame_index": record["frame_index"],
        "camera_id": record["camera_id"],
        "sha256": record["sha256"],
        "size_bytes": record["size_bytes"],
    }


def _frame_inventory(
    *, final_path: Path, final: Mapping[str, Any]
) -> tuple[str, list[dict[str, Any]]]:
    tasks = final.get("tasks")
    schema = final.get("schema_version")
    common_invalid = (
        final.get("receipt_digest") != canonical_digest(final, digest_field="receipt_digest")
        or final.get("semantic_object_absence_review_passed") is not False
        or final.get("multiview_consistency_review_passed") is not False
        or final.get("appearance_repair_qualified") is not False
        or final.get("generated_output_is_capture_or_physical_evidence") is not False
        or not isinstance(tasks, list)
        or len(tasks) != 1
        or not isinstance(tasks[0], Mapping)
    )
    final_composite_invalid = schema == FINAL_COMPOSITE_SCHEMA_VERSION and (
        final.get("status") != "final_composite_materialized_pending_human_multiview_review"
        or final.get("outside_support_invariance_proven") is not True
        or final.get("outside_support_changed_pixels_total") != 0
    )
    dual_target_invalid = schema == DUAL_TARGET_REVIEW_SCHEMA_VERSION and (
        final.get("status") != "paired_target_frames_pending_independent_visual_review"
        or final.get("review_scope")
        != "source_anchor_exact_mask_and_generated_full_frame_comparison"
        or final.get("outside_support_invariance_proven") is not False
        or final.get("outside_support_invariance_claimed") is not False
    )
    if (
        schema not in {FINAL_COMPOSITE_SCHEMA_VERSION, DUAL_TARGET_REVIEW_SCHEMA_VERSION}
        or common_invalid
        or final_composite_invalid
        or dual_target_invalid
    ):
        raise TaskEvaluationArtifixerAIVisualReviewError(
            "artifixer_ai_review_final_composite_invalid"
        )
    task = tasks[0]
    frames = task.get("frames")
    task_id = str(task.get("task_id") or "")
    if (
        not task_id
        or not isinstance(frames, list)
        or not frames
        or task.get("physical_camera_count") != len(frames)
    ):
        raise TaskEvaluationArtifixerAIVisualReviewError(
            "artifixer_ai_review_frame_inventory_invalid"
        )
    inventory: list[dict[str, Any]] = []
    seen_cameras: set[str] = set()
    for index, row in enumerate(frames):
        dual_target = schema == DUAL_TARGET_REVIEW_SCHEMA_VERSION
        if (
            not isinstance(row, Mapping)
            or row.get("frame_index") != index
            or not str(row.get("camera_id") or "")
            or row["camera_id"] in seen_cameras
            or not isinstance(row.get("final_frame"), Mapping)
            or (not dual_target and row.get("outside_support_changed_pixels") != 0)
            or (
                dual_target
                and (
                    not isinstance(row.get("source_frame"), Mapping)
                    or not isinstance(row.get("exact_repair_mask"), Mapping)
                )
            )
        ):
            raise TaskEvaluationArtifixerAIVisualReviewError(
                "artifixer_ai_review_frame_inventory_invalid"
            )
        sealed = _verified_frame(
            final_path, {**row["final_frame"], "frame_index": index, "camera_id": row["camera_id"]}
        )
        inventory.append(sealed)
        if dual_target:
            _verified_frame(
                final_path,
                {
                    **row["source_frame"],
                    "frame_index": index,
                    "camera_id": row["camera_id"],
                },
            )
            _verified_frame(
                final_path,
                {
                    **row["exact_repair_mask"],
                    "frame_index": index,
                    "camera_id": row["camera_id"],
                },
            )
        seen_cameras.add(str(row["camera_id"]))
    return task_id, inventory


def build_artifixer_ai_visual_review_input(
    *, final_composite_receipt_path: str | Path
) -> tuple[list[dict[str, Any]], str, list[dict[str, Any]], dict[str, Any]]:
    """Build exact digest-rehashed multimodal input for the fixed reviewer."""

    final_path = Path(final_composite_receipt_path).expanduser().resolve()
    final = _read(final_path, code="artifixer_ai_review_final_composite_invalid")
    task_id, inventory = _frame_inventory(final_path=final_path, final=final)
    if len(inventory) > AI_REVIEW_MAX_FRAMES:
        raise TaskEvaluationArtifixerAIVisualReviewError(
            "artifixer_ai_review_frame_count_exceeds_cap"
        )
    frame_rows = final["tasks"][0]["frames"]
    content: list[dict[str, Any]] = [{"type": "input_text", "text": _PROMPT}]
    for sealed, row in zip(inventory, frame_rows, strict=True):
        final_record = row["final_frame"]
        raw = Path(str(final_record.get("path") or "")).expanduser()
        frame_path = (raw if raw.is_absolute() else final_path.parent / raw).resolve()
        if _sha256(frame_path) != sealed["sha256"]:
            raise TaskEvaluationArtifixerAIVisualReviewError(
                "artifixer_ai_review_frame_changed_before_transport"
            )
        content.append(
            {
                "type": "input_text",
                "text": canonical_json(
                    {
                        "task_id": task_id,
                        "camera_id": sealed["camera_id"],
                        "frame_sha256": sealed["sha256"],
                        "publisher_scene_id": final.get("publisher_scene_id"),
                        "comparison_kind": (
                            "source_anchor_exact_mask_then_generated_candidate"
                            if final.get("schema_version") == DUAL_TARGET_REVIEW_SCHEMA_VERSION
                            else "final_exact_support_composite"
                        ),
                    }
                ),
            }
        )
        if final.get("schema_version") == DUAL_TARGET_REVIEW_SCHEMA_VERSION:
            for label, field in (
                ("source_anchor", "source_frame"),
                ("exact_repair_mask", "exact_repair_mask"),
            ):
                record = row[field]
                unresolved = Path(str(record.get("path") or "")).expanduser()
                comparison_path = (
                    unresolved if unresolved.is_absolute() else final_path.parent / unresolved
                ).resolve()
                if _sha256(comparison_path) != record.get("sha256"):
                    raise TaskEvaluationArtifixerAIVisualReviewError(
                        "artifixer_ai_review_frame_changed_before_transport"
                    )
                content.extend(
                    [
                        {"type": "input_text", "text": label},
                        {
                            "type": "input_image",
                            "image_url": (
                                "data:image/png;base64,"
                                + base64.b64encode(comparison_path.read_bytes()).decode("ascii")
                            ),
                            "detail": "high",
                        },
                    ]
                )
            content.append({"type": "input_text", "text": "generated_candidate"})
        content.append(
            {
                "type": "input_image",
                "image_url": (
                    "data:image/png;base64,"
                    + base64.b64encode(frame_path.read_bytes()).decode("ascii")
                ),
                "detail": "high",
            }
        )
    return [{"role": "user", "content": content}], task_id, inventory, final


def validate_artifixer_ai_visual_review_rights(
    *,
    rights_attestation_path: str | Path,
    configuration_run_id: str,
) -> tuple[Path, dict[str, Any]]:
    """Require human-issued scope authority for private derived-frame review."""

    path = Path(rights_attestation_path).expanduser().resolve()
    rights = _read(path, code="artifixer_ai_review_rights_invalid")
    if (
        rights.get("schema_version") != RIGHTS_SCHEMA_VERSION
        or rights.get("status") != "accepted_for_private_derived_visual_review"
        or rights.get("program_id") != "arm-decision-proof-v1"
        or rights.get("configuration_run_id") != configuration_run_id
        or rights.get("provider_id") != "openai"
        or rights.get("runtime") != "openai_agents_sdk"
        or rights.get("model") != AI_REVIEW_MODEL
        or rights.get("private_derived_frame_disclosure_authorized") is not True
        or rights.get("raw_interiorgs_bytes_disclosure_authorized") is not False
        or rights.get("provider_training_authorized") is not False
        or rights.get("public_redistribution_authorized") is not False
        or rights.get("max_inference_spend_usd") != AI_REVIEW_MAX_COST_USD
        or not _SHA256.fullmatch(str(rights.get("source_scene_rights_admission_digest") or ""))
        or rights.get("attestation_digest")
        != canonical_digest(rights, digest_field="attestation_digest")
    ):
        raise TaskEvaluationArtifixerAIVisualReviewError("artifixer_ai_review_rights_invalid")
    return path, rights


def materialize_artifixer_ai_visual_review_rights(
    *,
    configuration_run_id: str,
    source_scene_rights_admission_digest: str,
    accepted_by: str,
    accepted_on: str,
    human_authority_reference: str,
    output_path: str | Path,
) -> dict[str, Any]:
    """Seal scope authority; exact generated frames are bound at execution."""

    if (
        not configuration_run_id.strip()
        or not _SHA256.fullmatch(source_scene_rights_admission_digest)
        or not accepted_by.strip()
        or not accepted_on.strip()
        or not human_authority_reference.strip()
    ):
        raise TaskEvaluationArtifixerAIVisualReviewError(
            "artifixer_ai_review_rights_request_invalid"
        )
    value: dict[str, Any] = {
        "schema_version": RIGHTS_SCHEMA_VERSION,
        "status": "accepted_for_private_derived_visual_review",
        "program_id": "arm-decision-proof-v1",
        "configuration_run_id": configuration_run_id,
        "source_scene_rights_admission_digest": (source_scene_rights_admission_digest),
        "provider_id": "openai",
        "runtime": "openai_agents_sdk",
        "model": AI_REVIEW_MODEL,
        "private_derived_frame_disclosure_authorized": True,
        "raw_interiorgs_bytes_disclosure_authorized": False,
        "provider_training_authorized": False,
        "public_redistribution_authorized": False,
        "max_inference_spend_usd": AI_REVIEW_MAX_COST_USD,
        "accepted_by": accepted_by.strip(),
        "accepted_on": accepted_on.strip(),
        "human_authority_reference": human_authority_reference.strip(),
        "generated_frame_bytes_unknown_until_production_execution": True,
        "exact_frame_inventory_bound_by_execution_receipt": True,
        "attestation_digest": "",
    }
    value["attestation_digest"] = canonical_digest(value, digest_field="attestation_digest")
    destination = Path(output_path).expanduser().resolve()
    if destination.exists() or destination.is_symlink():
        raise TaskEvaluationArtifixerAIVisualReviewError("artifixer_ai_review_rights_output_exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(canonical_json(value) + "\n", encoding="utf-8")
    validate_artifixer_ai_visual_review_rights(
        rights_attestation_path=destination,
        configuration_run_id=configuration_run_id,
    )
    return value


def _validate_decisions(
    *, execution: Mapping[str, Any], task_id: str, inventory: Sequence[Mapping[str, Any]]
) -> bool:
    rows = execution.get("frames")
    if not isinstance(rows, list) or len(rows) != len(inventory):
        return False
    expected = {(str(row["camera_id"]), str(row["sha256"])) for row in inventory}
    observed: set[tuple[str, str]] = set()
    for row in rows:
        if (
            not isinstance(row, Mapping)
            or row.get("task_id") != task_id
            or not str(row.get("camera_id") or "")
            or not _SHA256.fullmatch(str(row.get("frame_sha256") or ""))
            or row.get("source_object_absent") is not True
            or row.get("repair_is_locally_plausible") is not True
            or row.get("preserves_non_target_content") is not True
            or row.get("decision") != "accepted"
            or not str(row.get("rationale") or "").strip()
        ):
            return False
        identity = (str(row["camera_id"]), str(row["frame_sha256"]))
        if identity in observed:
            return False
        observed.add(identity)
    return observed == expected


def _record(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": _sha256(resolved),
    }


def run_artifixer_ai_visual_review(
    *,
    final_composite_receipt_path: str | Path,
    rights_attestation_path: str | Path,
    configuration_run_id: str,
    publisher_instance_id: str,
    minimum_review_frames: int,
    output_root: str | Path,
    openai_cost_scope_attestation_path: str | Path,
    openai_admin_api_key_file: str | Path,
    openai_project_id: str,
    openai_api_key_id: str,
    model: str = AI_REVIEW_MODEL,
    max_cost_usd: float = AI_REVIEW_MAX_COST_USD,
    openai_cost_transport: Any | None = None,
    wall_clock: Any = lambda: datetime.now(timezone.utc),
) -> dict[str, Any]:
    """Execute the fixed structured reviewer and seal an accepted decision."""

    if model != AI_REVIEW_MODEL or max_cost_usd != AI_REVIEW_MAX_COST_USD:
        raise TaskEvaluationArtifixerAIVisualReviewError(
            "artifixer_ai_review_fixed_model_or_cost_invalid"
        )
    if (
        not configuration_run_id.strip()
        or not openai_project_id.strip()
        or not openai_api_key_id.strip()
    ):
        raise TaskEvaluationArtifixerAIVisualReviewError(
            "artifixer_ai_review_runtime_configuration_missing"
        )
    rights_path, rights = validate_artifixer_ai_visual_review_rights(
        rights_attestation_path=rights_attestation_path,
        configuration_run_id=configuration_run_id,
    )
    final_path = Path(final_composite_receipt_path).expanduser().resolve()
    input_value, task_id, inventory, final = build_artifixer_ai_visual_review_input(
        final_composite_receipt_path=final_path
    )
    if len(inventory) < minimum_review_frames:
        raise TaskEvaluationArtifixerAIVisualReviewError(
            "artifixer_ai_review_frame_count_below_configuration"
        )
    destination = Path(output_root).expanduser().resolve()
    if destination.is_symlink() or (destination.exists() and any(destination.iterdir())):
        raise TaskEvaluationArtifixerAIVisualReviewError("artifixer_ai_review_output_not_empty")
    destination.mkdir(parents=True, exist_ok=True)
    input_digest = canonical_digest({"input": input_value})
    run_id = "artifixer-ai-review-" + final["receipt_digest"].removeprefix("sha256:")[:16]
    cost_gate = build_openai_official_cost_run_gate(
        scope_attestation_path=openai_cost_scope_attestation_path,
        admin_api_key_file=openai_admin_api_key_file,
        project_id=openai_project_id,
        api_key_id=openai_api_key_id,
        lane_id="task_evaluation_artifixer_ai_visual_review",
        run_id=run_id,
        request_digest=input_digest,
        candidate_digest=final["receipt_digest"],
        authorization_receipt_digest=rights["attestation_digest"],
        max_cost_usd=max_cost_usd,
        output_root=destination / "official_openai_cost",
        provider_id="openai",
        paid_resource_class="task_evaluation_artifixer_ai_visual_review",
        transport=openai_cost_transport,
        wall_clock=wall_clock,
    )
    cost_reservation = cost_gate.reserve()
    audit = InferenceReservationAudit(run_root=destination, run_id=run_id)
    invoker = OpenAIAgentsSDKInvoker(
        OpenAIAgentsSDKConfig(
            model=model,
            max_turns=1,
            max_output_tokens=AI_REVIEW_MAX_OUTPUT_TOKENS,
            allow_live_invocation=True,
            tracing_disabled=True,
            max_inference_cost_usd=max_cost_usd,
        )
    )
    invoker.configure_reservation_audit(
        record_reservation=audit.record_reservation,
        record_completion=audit.record_completion,
        restored_reserved_cost_usd=0.0,
    )
    spec = AgentsSDKAgentSpec(
        run_id=run_id,
        capability="task_evaluation_artifixer_ai_visual_review",
        name="Blueprint ArtiFixer Independent Visual Reviewer",
        instructions=_PROMPT,
        model=model,
        max_turns=1,
        max_output_tokens=AI_REVIEW_MAX_OUTPUT_TOKENS,
        max_input_tokens=AI_REVIEW_MAX_INPUT_TOKENS,
        output_type=ArtifixerAIVisualReviewOutput,
    )
    try:
        invocation = invoker.invoke(spec, input_value)
    except Exception as exc:
        audit.write_manifest()
        cost_gate.complete(
            provider_call_performed=True,
            runtime_result_digest=None,
            runtime_exception_type=type(exc).__name__,
        )
        raise
    reservation_manifest = audit.write_manifest()
    if invocation.provider != "openai" or invocation.model != model:
        raise TaskEvaluationArtifixerAIVisualReviewError(
            "artifixer_ai_review_provider_identity_invalid"
        )
    output = ArtifixerAIVisualReviewOutput.model_validate(invocation.output)
    structured = output.model_dump(mode="json")
    structured_digest = canonical_digest(structured)
    cost_completion = cost_gate.complete(
        provider_call_performed=True,
        runtime_result_digest=structured_digest,
        runtime_exception_type=None,
    )
    accepted = (
        output.decision == "accepted"
        and output.semantic_object_absence_review_passed
        and output.multiview_consistency_review_passed
        and all(
            row.decision == "accepted"
            and row.source_object_absent
            and row.repair_is_locally_plausible
            and row.preserves_non_target_content
            for row in output.frames
        )
    )
    execution: dict[str, Any] = {
        "schema_version": EXECUTION_SCHEMA_VERSION,
        "status": "completed",
        "configuration_run_id": configuration_run_id,
        "publisher_instance_id": publisher_instance_id,
        "task_id": task_id,
        "decision": "accepted" if accepted else "rejected",
        "final_composite_receipt_digest": final["receipt_digest"],
        "review_frame_inventory_digest": canonical_digest({"frames": inventory}),
        "review_frame_count": len(inventory),
        "reviewer": {
            "kind": "ai",
            "identity": AI_REVIEWER_ID,
            "runtime": "openai_agents_sdk",
            "model": invocation.model,
            "model_version": invocation.model,
            "sdk_version": invocation.sdk_version,
        },
        "frames": structured["frames"],
        "summary": structured["summary"],
        "semantic_object_absence_review_passed": (
            accepted and output.semantic_object_absence_review_passed
        ),
        "multiview_consistency_review_passed": (
            accepted and output.multiview_consistency_review_passed
        ),
        "input_digest": input_digest,
        "structured_output_digest": structured_digest,
        "provider_called": True,
        "provider": invocation.provider,
        "response_store": False,
        "tracing_disabled": True,
        "raw_secret_values_recorded": False,
        "rights_attestation_digest": rights["attestation_digest"],
        "rights_attestation": _record(rights_path),
        "usage": dict(invocation.usage),
        "cost_usd": invocation.cost_usd,
        "cost_status": invocation.cost_status,
        "official_openai_cost_reservation": cost_reservation,
        "official_openai_cost_completion": cost_completion,
        "inference_reservation_manifest": reservation_manifest,
        "generated_output_is_capture_or_physical_evidence": False,
        "physics_or_collision_authority_granted": False,
        "execution_digest": "",
    }
    execution["execution_digest"] = canonical_digest(execution, digest_field="execution_digest")
    execution_path = destination / f"{EXECUTION_SCHEMA_VERSION}.json"
    write_json(execution_path, execution)
    review: dict[str, Any] | None = None
    if accepted:
        review = seal_artifixer_ai_visual_review(
            final_composite_receipt_path=final_path,
            review_execution_receipt_path=execution_path,
            publisher_instance_id=publisher_instance_id,
            minimum_review_frames=minimum_review_frames,
            output_path=destination / f"{RECEIPT_SCHEMA_VERSION}.json",
        )
    return {
        "status": (
            "artifixer_ai_visual_review_accepted"
            if accepted
            else "artifixer_ai_visual_review_rejected"
        ),
        "decision": execution["decision"],
        "execution_receipt": {
            **_record(execution_path),
            "execution_digest": execution["execution_digest"],
        },
        "review_receipt": (
            {
                **_record(destination / f"{RECEIPT_SCHEMA_VERSION}.json"),
                "receipt_digest": review["receipt_digest"],
            }
            if review is not None
            else None
        ),
    }


def seal_artifixer_ai_visual_review(
    *,
    final_composite_receipt_path: str | Path,
    review_execution_receipt_path: str | Path,
    publisher_instance_id: str,
    minimum_review_frames: int,
    output_path: str | Path,
) -> dict[str, Any]:
    """Bind an accepted structured review to every exact final composite frame."""

    if (
        not str(publisher_instance_id).strip()
        or isinstance(minimum_review_frames, bool)
        or not isinstance(minimum_review_frames, int)
        or minimum_review_frames < 1
    ):
        raise TaskEvaluationArtifixerAIVisualReviewError("artifixer_ai_review_request_invalid")
    final_path = Path(final_composite_receipt_path).expanduser().resolve()
    execution_path = Path(review_execution_receipt_path).expanduser().resolve()
    final = _read(final_path, code="artifixer_ai_review_final_composite_invalid")
    execution = _read(execution_path, code="artifixer_ai_review_execution_receipt_invalid")
    task_id, inventory = _frame_inventory(final_path=final_path, final=final)
    reviewer = execution.get("reviewer")
    accepted = _validate_decisions(execution=execution, task_id=task_id, inventory=inventory)
    if (
        len(inventory) < minimum_review_frames
        or execution.get("schema_version") != EXECUTION_SCHEMA_VERSION
        or execution.get("status") != "completed"
        or execution.get("decision") != "accepted"
        or execution.get("publisher_instance_id") != publisher_instance_id
        or execution.get("task_id") != task_id
        or execution.get("final_composite_receipt_digest") != final.get("receipt_digest")
        or execution.get("review_frame_inventory_digest") != canonical_digest({"frames": inventory})
        or execution.get("provider_called") is not True
        or execution.get("response_store") is not False
        or execution.get("tracing_disabled") is not True
        or execution.get("raw_secret_values_recorded") is not False
        or execution.get("semantic_object_absence_review_passed") is not True
        or execution.get("multiview_consistency_review_passed") is not True
        or not _SHA256.fullmatch(str(execution.get("rights_attestation_digest") or ""))
        or not isinstance(reviewer, Mapping)
        or reviewer.get("kind") != "ai"
        or not str(reviewer.get("identity") or "")
        or not str(reviewer.get("runtime") or "")
        or not str(reviewer.get("model") or "")
        or execution.get("execution_digest")
        != canonical_digest(execution, digest_field="execution_digest")
        or not accepted
    ):
        raise TaskEvaluationArtifixerAIVisualReviewError(
            "artifixer_ai_review_execution_not_acceptable"
        )
    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "status": "accepted",
        "publisher_instance_id": publisher_instance_id,
        "task_id": task_id,
        "decision": "accepted",
        "semantic_object_absence_review_passed": True,
        "multiview_consistency_review_passed": True,
        "review_frame_count": len(inventory),
        "review_frame_inventory_digest": canonical_digest({"frames": inventory}),
        "all_review_frames_digest_bound": True,
        "reviewer": dict(reviewer),
        "review_execution_receipt": {
            "sha256": _sha256(execution_path),
            "size_bytes": execution_path.stat().st_size,
            "execution_digest": execution["execution_digest"],
        },
        "final_composite_receipt": {
            "sha256": _sha256(final_path),
            "size_bytes": final_path.stat().st_size,
            "receipt_digest": final["receipt_digest"],
        },
        "rights_attestation_digest": execution["rights_attestation_digest"],
        "ai_visual_review_completed": True,
        "human_review_completed": False,
        "generated_output_is_capture_or_physical_evidence": False,
        "physics_or_collision_authority_granted": False,
        "claim_boundary": ("independent_ai_review_of_digest_bound_generated_appearance_only"),
        "receipt_digest": "",
    }
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    destination = Path(output_path).expanduser().resolve()
    if destination.exists() or destination.is_symlink():
        raise TaskEvaluationArtifixerAIVisualReviewError("artifixer_ai_review_output_exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(canonical_json(receipt) + "\n", encoding="utf-8")
    return receipt


__all__ = [
    "AI_REVIEW_MAX_COST_USD",
    "AI_REVIEW_MODEL",
    "ArtifixerAIVisualReviewOutput",
    "ArtifixerFrameReviewDecision",
    "EXECUTION_SCHEMA_VERSION",
    "RECEIPT_SCHEMA_VERSION",
    "TaskEvaluationArtifixerAIVisualReviewError",
    "build_artifixer_ai_visual_review_input",
    "materialize_artifixer_ai_visual_review_rights",
    "run_artifixer_ai_visual_review",
    "seal_artifixer_ai_visual_review",
    "validate_artifixer_ai_visual_review_rights",
]
