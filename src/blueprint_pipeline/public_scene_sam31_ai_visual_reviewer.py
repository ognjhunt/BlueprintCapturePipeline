"""Run production SAM overlay review through the OpenAI Agents SDK.

The runtime transports every digest-verified PNG overlay to one structured-output
vision agent, retains the inference reservation/completion evidence, and seals the
existing AI selection receipt without a Codex, Claude, or human decision seam.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
import hashlib
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .common import write_json
from .decision_evidence_contracts import canonical_digest, canonical_json
from .public_scene_sam31_track_selection_review import (
    AI_EXECUTION_SCHEMA_VERSION,
    AI_REVIEW_CAPABILITY,
    AI_REVIEW_FRAME_COUNT,
    AI_REVIEW_INPUT_TOKEN_CEILING,
    AI_REVIEW_MAX_COST_USD,
    AI_REVIEW_METHOD,
    AI_REVIEW_MODEL,
    AI_REVIEWER_ID,
    build_sam31_ai_visual_review_input,
    load_validated_sam31_track_selection_review_candidate,
    seal_sam31_track_selection_ai_review,
    validate_sam31_ai_structured_decision,
    validate_sam31_ai_visual_review_rights,
)
from .scene_placement.semantic_gaussian_lifting import canonical_json_digest
from .openai_official_cost_gate import (
    build_openai_official_cost_run_gate,
)
from .task_evaluation_supervisor.agents_sdk import (
    AgentsSDKAgentSpec,
    OpenAIAgentsSDKConfig,
    OpenAIAgentsSDKInvoker,
)
from .task_evaluation_supervisor.inference_reservations import InferenceReservationAudit


EXECUTION_RECEIPT_NAME = "public_scene_sam31_ai_visual_review_execution.v1.json"
AI_REVIEW_RECEIPT_NAME = "public_scene_sam31_track_selection_ai_visual_review.v1.json"
DEFAULT_MAX_INPUT_TOKENS = AI_REVIEW_INPUT_TOKEN_CEILING
DEFAULT_MAX_OUTPUT_TOKENS = 3_000
DEFAULT_MAX_COST_USD = AI_REVIEW_MAX_COST_USD
OPENAI_COST_PROVIDER_ID = "openai"
OPENAI_COST_RESOURCE_CLASS = "sam31_ai_visual_review"


class Sam31AIVisualReviewError(RuntimeError):
    """The production visual review could not produce a valid bound decision."""


class Sam31FrameVisualDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str = Field(min_length=1, max_length=200)
    camera_id: str = Field(min_length=1, max_length=200)
    target_visibility: Literal["visible_or_partially_visible", "absent_or_fully_occluded"]
    selected_mask_matches_target: bool
    decision: Literal["accepted", "rejected"]
    rationale: str = Field(min_length=1, max_length=1_000)


class Sam31AIVisualReviewOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision: Literal["accepted", "rejected"]
    summary: str = Field(min_length=1, max_length=4_000)
    frames: list[Sam31FrameVisualDecision] = Field(min_length=1, max_length=100)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _record(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": "sha256:" + hashlib.sha256(resolved.read_bytes()).hexdigest(),
    }


def run_sam31_ai_visual_review(
    *,
    candidate_path: str | Path,
    rights_attestation_path: str | Path,
    output_root: str | Path,
    model: str = AI_REVIEW_MODEL,
    max_input_tokens: int = DEFAULT_MAX_INPUT_TOKENS,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    max_cost_usd: float = DEFAULT_MAX_COST_USD,
    openai_cost_scope_attestation_path: str | Path | None = None,
    openai_admin_api_key_file: str | Path | None = None,
    openai_project_id: str = "",
    openai_api_key_id: str = "",
    openai_cost_transport: Callable[..., Mapping[str, Any]] | None = None,
    openai_cost_wall_clock: Callable[[], datetime] = lambda: datetime.now(
        timezone.utc
    ),
) -> dict[str, Any]:
    """Execute and retain one production, digest-bound AI visual review."""

    if max_input_tokens != DEFAULT_MAX_INPUT_TOKENS:
        raise Sam31AIVisualReviewError("sam31_ai_review_input_reservation_must_be_fixed")
    if max_cost_usd != DEFAULT_MAX_COST_USD:
        raise Sam31AIVisualReviewError("sam31_ai_review_cost_cap_must_be_fixed")
    if model != AI_REVIEW_MODEL:
        raise Sam31AIVisualReviewError("sam31_ai_review_model_must_be_fixed")
    if (
        openai_cost_scope_attestation_path is None
        or openai_admin_api_key_file is None
        or not openai_project_id.strip()
        or not openai_api_key_id.strip()
    ):
        raise Sam31AIVisualReviewError(
            "sam31_ai_review_official_cost_config_missing"
        )
    candidate_file, candidate = load_validated_sam31_track_selection_review_candidate(
        candidate_path
    )
    rights_path, rights = validate_sam31_ai_visual_review_rights(
        candidate_path=candidate_file,
        rights_attestation_path=rights_attestation_path,
    )
    destination = Path(output_root).expanduser().resolve()
    if destination.is_symlink() or (destination.exists() and any(destination.iterdir())):
        raise Sam31AIVisualReviewError("sam31_ai_review_output_not_empty")
    destination.mkdir(parents=True, exist_ok=True)
    input_value, frame_inventory = build_sam31_ai_visual_review_input(
        candidate_path=candidate_file,
    )
    if len(frame_inventory) != AI_REVIEW_FRAME_COUNT:
        raise Sam31AIVisualReviewError("sam31_ai_review_requires_exactly_16_overlays")
    run_id = f"sam31-ai-visual-review-{candidate['candidate_digest'].removeprefix('sha256:')[:16]}"
    input_digest = canonical_digest({"input": input_value})
    cost_gate = build_openai_official_cost_run_gate(
        scope_attestation_path=openai_cost_scope_attestation_path,
        admin_api_key_file=openai_admin_api_key_file,
        project_id=openai_project_id,
        api_key_id=openai_api_key_id,
        lane_id=AI_REVIEW_CAPABILITY,
        run_id=run_id,
        request_digest=input_digest,
        candidate_digest=candidate["candidate_digest"],
        authorization_receipt_digest=rights["attestation_digest"],
        max_cost_usd=max_cost_usd,
        output_root=destination / "official_openai_cost",
        provider_id=OPENAI_COST_PROVIDER_ID,
        paid_resource_class=OPENAI_COST_RESOURCE_CLASS,
        transport=openai_cost_transport,
        wall_clock=openai_cost_wall_clock,
    )
    cost_reservation = cost_gate.reserve()
    audit = InferenceReservationAudit(run_root=destination, run_id=run_id)
    selected_invoker = OpenAIAgentsSDKInvoker(
        OpenAIAgentsSDKConfig(
            model=model,
            max_turns=1,
            max_output_tokens=max_output_tokens,
            allow_live_invocation=True,
            tracing_disabled=True,
            max_inference_cost_usd=max_cost_usd,
        )
    )
    selected_invoker.configure_reservation_audit(
        record_reservation=audit.record_reservation,
        record_completion=audit.record_completion,
        restored_reserved_cost_usd=0.0,
    )
    spec = AgentsSDKAgentSpec(
        run_id=run_id,
        capability=AI_REVIEW_CAPABILITY,
        name="Blueprint SAM Track Selection Visual Reviewer",
        instructions=(
            "Independently inspect every digest-bound overlay supplied by Blueprint. Follow the "
            "per-frame acceptance rules exactly, return every requested task/camera row once, "
            "and fail closed when object identity or mask coverage is uncertain."
        ),
        model=model,
        max_turns=1,
        max_output_tokens=max_output_tokens,
        max_input_tokens=max_input_tokens,
        output_type=Sam31AIVisualReviewOutput,
    )
    try:
        invocation = selected_invoker.invoke(spec, input_value)
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
        raise Sam31AIVisualReviewError("sam31_ai_review_provider_identity_invalid")
    output = Sam31AIVisualReviewOutput.model_validate(invocation.output)
    structured_output = output.model_dump(mode="json")
    structured_output_digest = canonical_digest(structured_output)
    cost_completion = cost_gate.complete(
        provider_call_performed=True,
        runtime_result_digest=structured_output_digest,
        runtime_exception_type=None,
    )
    decision, blockers = validate_sam31_ai_structured_decision(
        structured_output=structured_output,
        frame_inventory=frame_inventory,
    )
    reservation_manifest_path = destination / "inference_reservations" / "manifest.json"
    timestamp = _utc_now()
    execution: dict[str, Any] = {
        "schema_version": AI_EXECUTION_SCHEMA_VERSION,
        "status": (
            "ai_visual_review_execution_completed"
            if output.decision == decision
            else "ai_visual_review_execution_invalid"
        ),
        "candidate": {
            **_record(candidate_file),
            "candidate_digest": candidate["candidate_digest"],
        },
        "rights_attestation": {
            **_record(rights_path),
            "attestation_digest": rights["attestation_digest"],
        },
        "reviewer": {
            "kind": "ai",
            "identity": AI_REVIEWER_ID,
            "runtime": "openai_agents_sdk",
            "model": invocation.model,
            "model_version": invocation.model,
            "sdk_version": invocation.sdk_version,
            "method": AI_REVIEW_METHOD,
        },
        "reviewed_at": timestamp,
        "run_id": run_id,
        "capability": AI_REVIEW_CAPABILITY,
        "decision": decision,
        "review_media_digest": canonical_json_digest(candidate["review_media"]),
        "review_frame_count": len(frame_inventory),
        "frame_inventory": frame_inventory,
        "structured_output": structured_output,
        "structured_output_digest": structured_output_digest,
        "input_digest": input_digest,
        "input_transport": "digest_rehashed_png_data_urls",
        "provider_called": True,
        "provider": invocation.provider,
        "response_store": False,
        "tracing_disabled": True,
        "trace_sensitive_data_included": False,
        "usage": dict(invocation.usage),
        "cost_usd": invocation.cost_usd,
        "cost_status": invocation.cost_status,
        "official_openai_cost_reservation": cost_reservation,
        "official_openai_cost_reservation_record": _record(
            cost_gate.reservation_path
        ),
        "official_openai_cost_completion": cost_completion,
        "official_openai_cost_completion_record": _record(
            cost_gate.completion_path
        ),
        "official_openai_billing_status": cost_completion["status"],
        "strict_official_billing_satisfied": False,
        "inference_reservation_manifest": reservation_manifest,
        "inference_reservation_manifest_record": _record(reservation_manifest_path),
        "blockers": blockers,
        "raw_secret_values_recorded": False,
        "claim_boundary": {
            "visual_track_selection_review_only": True,
            "human_review_completed": False,
            "per_view_segmentation_completeness_qualified": False,
            "geometry_qualified": False,
            "physical_evidence": False,
        },
        "execution_receipt_digest": "",
    }
    execution["execution_receipt_digest"] = canonical_digest(
        execution, digest_field="execution_receipt_digest"
    )
    execution_path = destination / EXECUTION_RECEIPT_NAME
    write_json(execution_path, execution)
    if execution["status"] != "ai_visual_review_execution_completed":
        raise Sam31AIVisualReviewError("sam31_ai_review_structured_output_invalid")
    review_path = destination / AI_REVIEW_RECEIPT_NAME
    review = seal_sam31_track_selection_ai_review(
        candidate_path=candidate_file,
        review_execution_receipt_path=execution_path,
        output_path=review_path,
    )
    result = {
        "status": f"ai_visual_review_{decision}",
        "decision": decision,
        "execution_receipt": {
            **_record(execution_path),
            "execution_receipt_digest": execution["execution_receipt_digest"],
        },
        "review_receipt": {
            **_record(review_path),
            "receipt_digest": review["receipt_digest"],
        },
        "blockers": blockers,
    }
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--rights-attestation", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--max-output-tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS)
    parser.add_argument("--openai-cost-scope-attestation", required=True)
    parser.add_argument("--openai-admin-api-key-file", required=True)
    parser.add_argument("--openai-project-id", required=True)
    parser.add_argument("--openai-api-key-id", required=True)
    args = parser.parse_args(argv)
    result = run_sam31_ai_visual_review(
        candidate_path=args.candidate,
        rights_attestation_path=args.rights_attestation,
        output_root=args.output_root,
        max_output_tokens=args.max_output_tokens,
        openai_cost_scope_attestation_path=args.openai_cost_scope_attestation,
        openai_admin_api_key_file=args.openai_admin_api_key_file,
        openai_project_id=args.openai_project_id,
        openai_api_key_id=args.openai_api_key_id,
    )
    print(canonical_json(result))
    return 0 if result["decision"] == "accepted" else 2


__all__ = [
    "AI_REVIEW_RECEIPT_NAME",
    "EXECUTION_RECEIPT_NAME",
    "Sam31AIVisualReviewError",
    "Sam31AIVisualReviewOutput",
    "Sam31FrameVisualDecision",
    "run_sam31_ai_visual_review",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
