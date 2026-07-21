"""Normalize model-runtime receipts into evaluator-neutral evidence rows.

Runtime completion is only one input to an evaluation row.  This module binds
the exact runtime output, model artifact, generated output, adapter, provider
execution, and backend manifest while leaving task criteria, authoritative
episode completion, and evaluator outcomes as separate evidence states.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .evaluator_evidence_profiles import (
    EVALUATOR_BACKEND_MANIFEST_SCHEMA_VERSION,
    canonical_evaluator_backend_manifest_sha256,
    validate_evaluator_evidence,
)


EVALUATOR_RUNTIME_RECEIPT_SCHEMA_VERSION = "evaluator_runtime_receipt.v1"
EVALUATOR_RUNTIME_RECEIPT_VALIDATION_SCHEMA_VERSION = "evaluator_runtime_receipt_validation.v1"
EVALUATOR_RUNTIME_NORMALIZATION_REQUEST_SCHEMA_VERSION = (
    "evaluator_runtime_normalization_request.v1"
)
EVALUATOR_RUNTIME_NORMALIZATION_SCHEMA_VERSION = "evaluator_runtime_normalization.v1"

_SHA256_RE = re.compile(r"^(?:sha256:)?[0-9a-f]{64}$")
_PROVIDER_EXECUTION_FIELDS = (
    "schema_version",
    "status",
    "execution_id",
    "runtime_id",
    "provider_id",
    "runtime_output_sha256",
    "model_artifact_sha256",
    "adapter_code_sha256",
    "runtime_manifest_sha256",
    "provider_is_evaluator_identity",
)
_SENSITIVE_FIELD_MARKERS = (
    "api_key",
    "api_token",
    "authorization",
    "auth_token",
    "access_token",
    "password",
    "credential",
    "private_key",
    "signed_url",
    "raw_response",
)
_SUPPORTED_WAM_OUTPUTS = {
    "cosmos3_wam_command_adapter.v1": {
        "adapter_id": "blueprint_cosmos3_nano_wam_command_adapter",
        "subprocess_field": "cosmos3_subprocess",
        "model_family": "cosmos3",
        "model_version_path": ("model_provenance", "model"),
    },
    "oscar_wam_command_adapter.v1": {
        "adapter_id": "blueprint_oscar_wam_command_adapter",
        "subprocess_field": "oscar_subprocess",
        "model_family": "oscar",
        "model_version_path": ("official_oscar_release", "model_name"),
    },
}


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _strict_rows(value: Any) -> tuple[list[dict[str, Any]], bool]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return [], False
    if any(not isinstance(row, Mapping) for row in value):
        return [], False
    return [dict(row) for row in value], True


def _normalized_digest(value: Any) -> str:
    digest = str(value or "").strip().lower()
    return digest.removeprefix("sha256:") if _SHA256_RE.fullmatch(digest) else ""


def _digest(value: Any) -> bool:
    return bool(_normalized_digest(value))


def _nested_text(payload: Mapping[str, Any], path: Sequence[str]) -> str:
    value: Any = payload
    for field in path:
        if not isinstance(value, Mapping):
            return ""
        value = value.get(field)
    return str(value or "").strip()


def _provider_execution_record(value: Any) -> tuple[dict[str, Any], bool]:
    raw = _mapping(value)
    return (
        {field: raw.get(field) for field in _PROVIDER_EXECUTION_FIELDS},
        set(raw).issubset(_PROVIDER_EXECUTION_FIELDS),
    )


def _sensitive_field_paths(value: Any, path: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for raw_key, nested in value.items():
            key = str(raw_key)
            nested_path = f"{path}.{key}" if path else key
            normalized_key = key.lower().replace("-", "_")
            if any(marker in normalized_key for marker in _SENSITIVE_FIELD_MARKERS):
                found.append(nested_path)
            else:
                found.extend(_sensitive_field_paths(nested, nested_path))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, nested in enumerate(value):
            found.extend(_sensitive_field_paths(nested, f"{path}[{index}]"))
    return found


def canonical_json_sha256(payload: Mapping[str, Any]) -> str:
    """Return a canonical SHA-256 for one JSON mapping, or ``""`` if invalid."""

    try:
        serialized = json.dumps(
            dict(payload),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError):
        return ""
    return f"sha256:{hashlib.sha256(serialized.encode('utf-8')).hexdigest()}"


def validate_evaluator_runtime_receipt(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the provider-neutral receipt emitted after a real model run."""

    blockers: list[str] = []
    if receipt.get("schema_version") != EVALUATOR_RUNTIME_RECEIPT_SCHEMA_VERSION:
        blockers.append("runtime_receipt_schema_missing_or_unsupported")
    if "status" in receipt and receipt.get("status") != "validated":
        blockers.append("runtime_receipt_declared_status_not_validated")
    if receipt.get("blockers") not in (None, []):
        blockers.append("runtime_receipt_declares_blockers")
    if receipt.get("source_runtime_blockers") not in (None, []):
        blockers.append("runtime_receipt_declares_source_runtime_blockers")
    for field in (
        "runtime_id",
        "runtime_adapter_id",
        "runtime_adapter_version",
        "backend_id",
        "model_family",
        "model_version",
        "provider_id",
    ):
        if not str(receipt.get(field) or "").strip():
            blockers.append(f"runtime_receipt_identity_missing:{field}")
    for field in (
        "runtime_output_sha256",
        "model_artifact_sha256",
        "adapter_code_sha256",
        "runtime_manifest_sha256",
        "license_manifest_sha256",
        "provider_execution_sha256",
    ):
        if not _digest(receipt.get(field)):
            blockers.append(f"runtime_receipt_digest_missing_or_invalid:{field}")
    provider_execution, provider_execution_fields_valid = _provider_execution_record(
        receipt.get("provider_execution")
    )
    if not provider_execution_fields_valid:
        blockers.append("runtime_receipt_provider_execution_fields_invalid")
    provider_execution_sha256 = canonical_json_sha256(provider_execution)
    if provider_execution.get("schema_version") != "evaluator_provider_execution.v1":
        blockers.append("runtime_receipt_provider_execution_schema_invalid")
    if provider_execution.get("status") != "succeeded":
        blockers.append("runtime_receipt_provider_execution_not_succeeded")
    for field in ("runtime_id", "provider_id"):
        if (
            str(provider_execution.get(field) or "").strip()
            != str(receipt.get(field) or "").strip()
        ):
            blockers.append(f"runtime_receipt_provider_execution_identity_mismatch:{field}")
    for field in (
        "runtime_output_sha256",
        "model_artifact_sha256",
        "adapter_code_sha256",
        "runtime_manifest_sha256",
    ):
        if _normalized_digest(provider_execution.get(field)) != _normalized_digest(
            receipt.get(field)
        ):
            blockers.append(f"runtime_receipt_provider_execution_digest_mismatch:{field}")
    if _normalized_digest(receipt.get("provider_execution_sha256")) != _normalized_digest(
        provider_execution_sha256
    ):
        blockers.append("runtime_receipt_provider_execution_manifest_digest_mismatch")
    if provider_execution.get("provider_is_evaluator_identity") is not False:
        blockers.append("runtime_receipt_provider_must_not_be_evaluator_identity")
    if receipt.get("runtime_status") != "completed":
        blockers.append("runtime_receipt_not_completed")
    if receipt.get("infrastructure_status") != "succeeded":
        blockers.append("runtime_receipt_infrastructure_not_succeeded")
    if receipt.get("fresh_model_execution_proven") is not True:
        blockers.append("runtime_receipt_fresh_model_execution_not_proven")
    run_steps = receipt.get("fresh_model_run_steps")
    if isinstance(run_steps, bool) or not isinstance(run_steps, int) or run_steps <= 0:
        blockers.append("runtime_receipt_fresh_model_run_steps_missing_or_invalid")
    if receipt.get("backend_is_compute_provider") is not False:
        blockers.append("runtime_receipt_backend_must_not_be_compute_provider")
    for flag in (
        "fixture_or_proxy_model_output_used",
        "fallback_model_output_used",
        "stale_model_output_used",
    ):
        if receipt.get(flag) is not False:
            blockers.append(f"runtime_receipt_forbidden_or_unproven:{flag}")

    outputs, outputs_valid = _strict_rows(receipt.get("model_outputs"))
    if not outputs_valid:
        blockers.append("runtime_receipt_model_outputs_payload_invalid")
    if not outputs:
        blockers.append("runtime_receipt_model_outputs_missing")
    output_ids: set[str] = set()
    output_digests: set[str] = set()
    for index, output in enumerate(outputs):
        output_id = str(output.get("output_id") or "").strip()
        output_digest = _normalized_digest(output.get("model_output_sha256"))
        if not output_id or output_id in output_ids:
            blockers.append(f"runtime_receipt_model_output_id_missing_or_duplicate:{index}")
        else:
            output_ids.add(output_id)
        if not output_digest or output_digest in output_digests:
            blockers.append(f"runtime_receipt_model_output_digest_invalid_or_duplicate:{index}")
        else:
            output_digests.add(output_digest)
        if output.get("model_output_status") != "completed":
            blockers.append(f"runtime_receipt_model_output_not_completed:{index}")
    if isinstance(run_steps, int) and not isinstance(run_steps, bool) and run_steps != len(outputs):
        blockers.append("runtime_receipt_fresh_model_run_steps_do_not_match_outputs")

    blockers = sorted(set(blockers))
    return {
        "schema_version": EVALUATOR_RUNTIME_RECEIPT_VALIDATION_SCHEMA_VERSION,
        "status": "validated" if not blockers else "blocked",
        "runtime_id": receipt.get("runtime_id"),
        "model_output_count": len(outputs),
        "blockers": blockers,
        "claim_boundary": {
            "runtime_completion_is_not_criterion_success": True,
            "runtime_completion_is_not_authoritative_episode_completion": True,
            "compute_provider_is_not_evaluator_identity": True,
            "runtime_receipt_is_not_real_world_correlation": True,
        },
    }


def build_wam_runtime_receipt(request: Mapping[str, Any]) -> dict[str, Any]:
    """Convert an exact OSCAR/Cosmos WAM output into the neutral receipt schema.

    Provider, adapter, runtime, and license digests are explicit request inputs;
    the model execution state and output digests are derived from the exact WAM
    output.  Missing execution proof blocks the receipt instead of being filled
    by a fixture or inferred from a generated-video path.
    """

    blockers: list[str] = []
    if request.get("infrastructure_status") != "succeeded":
        blockers.append("wam_runtime_infrastructure_not_succeeded")
    runtime_output = _mapping(request.get("runtime_output"))
    runtime_output_sha256 = canonical_json_sha256(runtime_output)
    if not runtime_output_sha256:
        blockers.append("wam_runtime_output_payload_invalid")
    if _normalized_digest(request.get("runtime_output_sha256")) != _normalized_digest(
        runtime_output_sha256
    ):
        blockers.append("wam_runtime_output_digest_mismatch")

    runtime_schema = str(runtime_output.get("schema_version") or "").strip()
    runtime_profile = _SUPPORTED_WAM_OUTPUTS.get(runtime_schema)
    if runtime_profile is None:
        blockers.append("wam_runtime_output_schema_missing_or_unsupported")
        runtime_profile = {}
    expected_adapter_id = str(runtime_profile.get("adapter_id") or "")
    if expected_adapter_id and runtime_output.get("adapter_id") != expected_adapter_id:
        blockers.append("wam_runtime_adapter_identity_mismatch")
    if runtime_output.get("status") != "completed":
        blockers.append("wam_runtime_output_not_completed")
    if runtime_output.get("blockers") not in ([], None):
        blockers.append("wam_runtime_output_declares_blockers")
    if runtime_output.get("learned_wam_model_ran") is not True:
        blockers.append("wam_runtime_learned_model_execution_not_proven")
    if runtime_output.get("fresh_model_command_executed_this_invocation") is not True:
        blockers.append("wam_runtime_fresh_command_execution_not_proven")
    if runtime_output.get("fresh_model_run_claimed") is not True:
        blockers.append("wam_runtime_fresh_model_run_not_claimed")
    fresh_steps = runtime_output.get("fresh_model_run_steps")
    if isinstance(fresh_steps, bool) or not isinstance(fresh_steps, int) or fresh_steps <= 0:
        blockers.append("wam_runtime_fresh_model_run_steps_missing_or_invalid")
    configured_inference_steps = runtime_output.get("configured_inference_steps_per_model_run")
    if (
        isinstance(configured_inference_steps, bool)
        or not isinstance(configured_inference_steps, int)
        or configured_inference_steps <= 0
    ):
        blockers.append("wam_runtime_configured_inference_steps_missing_or_invalid")

    subprocess_field = str(runtime_profile.get("subprocess_field") or "")
    subprocess_result = _mapping(runtime_output.get(subprocess_field))
    if subprocess_field and subprocess_result.get("status") != "completed":
        blockers.append("wam_runtime_subprocess_not_completed")
    truth = _mapping(runtime_output.get("truth_boundary"))
    if truth.get("generated_video_is_model_output") is not True:
        blockers.append("wam_runtime_generated_output_not_proven_as_model_output")

    rollouts, rollouts_valid = _strict_rows(runtime_output.get("rollouts"))
    if not rollouts_valid:
        blockers.append("wam_runtime_rollouts_payload_invalid")
    model_outputs: list[dict[str, Any]] = []
    for index, rollout in enumerate(rollouts):
        output_id = str(rollout.get("rollout_id") or "").strip()
        output_digest = rollout.get("generated_video_sha256")
        validation = _mapping(rollout.get("generated_video_review_validation"))
        if not output_id:
            blockers.append(f"wam_runtime_rollout_id_missing:{index}")
        if not _digest(output_digest):
            blockers.append(f"wam_runtime_model_output_digest_missing_or_invalid:{index}")
        if validation.get("status") != "completed":
            blockers.append(f"wam_runtime_model_output_media_not_valid:{index}")
        if output_id and _digest(output_digest) and validation.get("status") == "completed":
            model_outputs.append(
                {
                    "output_id": output_id,
                    "model_output_sha256": output_digest,
                    "model_output_status": "completed",
                }
            )
    if not model_outputs:
        blockers.append("wam_runtime_no_admissible_model_outputs")
    if (
        isinstance(fresh_steps, int)
        and not isinstance(fresh_steps, bool)
        and fresh_steps != len(model_outputs)
    ):
        blockers.append("wam_runtime_fresh_model_run_steps_do_not_match_outputs")

    backend = _mapping(request.get("evaluator_backend"))
    if backend.get("schema_version") != EVALUATOR_BACKEND_MANIFEST_SCHEMA_VERSION:
        blockers.append("wam_runtime_backend_manifest_missing_or_unsupported")
    if backend.get("backend_is_compute_provider") is not False:
        blockers.append("wam_runtime_backend_must_not_be_compute_provider")
    expected_model_family = str(runtime_profile.get("model_family") or "")
    if expected_model_family and str(backend.get("model_family") or "").strip().lower() != (
        expected_model_family
    ):
        blockers.append("wam_runtime_backend_model_family_mismatch")
    expected_model_version = _nested_text(
        runtime_output,
        tuple(runtime_profile.get("model_version_path") or ()),
    )
    if not expected_model_version:
        blockers.append("wam_runtime_model_version_not_proven")
    elif str(backend.get("model_version") or "").strip() != expected_model_version:
        blockers.append("wam_runtime_backend_model_version_mismatch")
    runtime_identity = _mapping(request.get("runtime_identity"))
    for field in (
        "runtime_id",
        "provider_id",
        "runtime_adapter_version",
    ):
        if not str(runtime_identity.get(field) or "").strip():
            blockers.append(f"wam_runtime_identity_missing:{field}")
    for field in (
        "adapter_code_sha256",
        "runtime_manifest_sha256",
        "license_manifest_sha256",
    ):
        if not _digest(request.get(field)):
            blockers.append(f"wam_runtime_binding_digest_missing_or_invalid:{field}")

    provider_execution, provider_execution_fields_valid = _provider_execution_record(
        request.get("provider_execution")
    )
    if not provider_execution_fields_valid:
        blockers.append("wam_provider_execution_fields_invalid")
    provider_execution_sha256 = canonical_json_sha256(provider_execution)
    if provider_execution.get("schema_version") != "evaluator_provider_execution.v1":
        blockers.append("wam_provider_execution_schema_missing_or_unsupported")
    if provider_execution.get("status") != "succeeded":
        blockers.append("wam_provider_execution_not_succeeded")
    for field in ("execution_id", "runtime_id", "provider_id"):
        if not str(provider_execution.get(field) or "").strip():
            blockers.append(f"wam_provider_execution_identity_missing:{field}")
    if provider_execution.get("runtime_id") != runtime_identity.get("runtime_id"):
        blockers.append("wam_provider_execution_runtime_identity_mismatch")
    if provider_execution.get("provider_id") != runtime_identity.get("provider_id"):
        blockers.append("wam_provider_execution_provider_identity_mismatch")
    provider_digest_bindings = {
        "runtime_output_sha256": runtime_output_sha256,
        "model_artifact_sha256": backend.get("model_artifact_sha256"),
        "adapter_code_sha256": request.get("adapter_code_sha256"),
        "runtime_manifest_sha256": request.get("runtime_manifest_sha256"),
    }
    for field, expected in provider_digest_bindings.items():
        if _normalized_digest(provider_execution.get(field)) != _normalized_digest(expected):
            blockers.append(f"wam_provider_execution_digest_binding_mismatch:{field}")
    if _normalized_digest(request.get("provider_execution_sha256")) != _normalized_digest(
        provider_execution_sha256
    ):
        blockers.append("wam_provider_execution_digest_mismatch")
    if provider_execution.get("provider_is_evaluator_identity") is not False:
        blockers.append("wam_provider_execution_must_not_be_evaluator_identity")

    receipt = {
        "schema_version": EVALUATOR_RUNTIME_RECEIPT_SCHEMA_VERSION,
        "runtime_id": runtime_identity.get("runtime_id"),
        "runtime_adapter_id": runtime_output.get("adapter_id"),
        "runtime_adapter_version": runtime_identity.get("runtime_adapter_version"),
        "runtime_output_schema_version": runtime_schema,
        "runtime_output_sha256": runtime_output_sha256,
        "backend_id": backend.get("backend_id"),
        "model_family": backend.get("model_family"),
        "model_version": backend.get("model_version"),
        "model_artifact_sha256": backend.get("model_artifact_sha256"),
        "adapter_code_sha256": request.get("adapter_code_sha256"),
        "runtime_manifest_sha256": request.get("runtime_manifest_sha256"),
        "license_manifest_sha256": request.get("license_manifest_sha256"),
        "provider_id": runtime_identity.get("provider_id"),
        "provider_execution": provider_execution,
        "provider_execution_sha256": provider_execution_sha256,
        "runtime_status": "completed" if not blockers else "blocked",
        "infrastructure_status": (
            "succeeded" if not blockers else str(request.get("infrastructure_status") or "failed")
        ),
        "fresh_model_execution_proven": not blockers,
        "fresh_model_run_steps": fresh_steps if not blockers else 0,
        "configured_inference_steps_per_model_run": (
            configured_inference_steps if not blockers else 0
        ),
        "backend_is_compute_provider": backend.get("backend_is_compute_provider"),
        "model_outputs": model_outputs,
        "fixture_or_proxy_model_output_used": request.get("fixture_or_proxy_model_output_used"),
        "fallback_model_output_used": request.get("fallback_model_output_used"),
        "stale_model_output_used": request.get("stale_model_output_used"),
        "source_runtime_blockers": sorted(set(blockers)),
    }
    validation = validate_evaluator_runtime_receipt(receipt)
    return {
        **receipt,
        "status": validation["status"],
        "blockers": sorted(set([*blockers, *validation["blockers"]])),
    }


def normalize_evaluator_runtime_evidence(request: Mapping[str, Any]) -> dict[str, Any]:
    """Bind one validated runtime receipt to one evaluator evidence row."""

    blockers: list[str] = []
    if request.get("schema_version") != EVALUATOR_RUNTIME_NORMALIZATION_REQUEST_SCHEMA_VERSION:
        blockers.append("runtime_normalization_request_schema_missing_or_unsupported")
    receipt = _mapping(request.get("runtime_receipt"))
    receipt_validation = validate_evaluator_runtime_receipt(receipt)
    blockers.extend(f"runtime_receipt:{item}" for item in receipt_validation["blockers"])
    receipt_sha256 = canonical_json_sha256(receipt)
    if _normalized_digest(request.get("runtime_receipt_sha256")) != _normalized_digest(
        receipt_sha256
    ):
        blockers.append("runtime_receipt_digest_mismatch")

    output_id = str(request.get("model_output_id") or "").strip()
    outputs, _ = _strict_rows(receipt.get("model_outputs"))
    selected_outputs = [row for row in outputs if str(row.get("output_id") or "") == output_id]
    if len(selected_outputs) != 1:
        blockers.append("runtime_model_output_selection_missing_or_ambiguous")
        selected_output: dict[str, Any] = {}
    else:
        selected_output = selected_outputs[0]

    row = _mapping(request.get("evaluator_row"))
    if _sensitive_field_paths(row):
        blockers.append("runtime_evaluator_row_contains_sensitive_fields")
    backend = _mapping(row.get("evaluator_backend"))
    expected_bindings = {
        "evaluator_checkpoint_sha256": receipt.get("model_artifact_sha256"),
        "evaluator_runtime_output_sha256": receipt.get("runtime_output_sha256"),
        "model_output_sha256": selected_output.get("model_output_sha256"),
        "provider_execution_sha256": receipt.get("provider_execution_sha256"),
    }
    for field, expected in expected_bindings.items():
        if _normalized_digest(row.get(field)) != _normalized_digest(expected):
            blockers.append(f"runtime_evaluator_row_digest_binding_mismatch:{field}")
    if _normalized_digest(row.get("policy_runtime_output_sha256")) == _normalized_digest(
        receipt.get("runtime_output_sha256")
    ):
        blockers.append("runtime_policy_and_evaluator_output_digests_must_be_distinct")
    for field in ("backend_id", "model_family", "model_version"):
        if str(backend.get(field) or "").strip() != str(receipt.get(field) or "").strip():
            blockers.append(f"runtime_evaluator_backend_identity_mismatch:{field}")
    for field in (
        "model_artifact_sha256",
        "adapter_code_sha256",
        "runtime_manifest_sha256",
        "license_manifest_sha256",
    ):
        if _normalized_digest(backend.get(field)) != _normalized_digest(receipt.get(field)):
            blockers.append(f"runtime_evaluator_backend_digest_mismatch:{field}")
    if _normalized_digest(row.get("evaluator_backend_manifest_sha256")) != _normalized_digest(
        canonical_evaluator_backend_manifest_sha256(backend)
    ):
        blockers.append("runtime_evaluator_backend_manifest_digest_mismatch")
    if row.get("fresh_evaluator_model_execution_proven") is not True:
        blockers.append("runtime_evaluator_row_fresh_execution_not_proven")
    if row.get("fresh_evaluator_model_run_steps") != receipt.get("fresh_model_run_steps"):
        blockers.append("runtime_evaluator_row_fresh_steps_mismatch")
    if row.get("infrastructure_status") != receipt.get("infrastructure_status"):
        blockers.append("runtime_evaluator_row_infrastructure_status_mismatch")
    if row.get("evaluator_identity_is_compute_provider") is not False:
        blockers.append("runtime_evaluator_identity_must_not_be_compute_provider")

    evaluator_validation = validate_evaluator_evidence(row)
    blockers.extend(f"evaluator_evidence:{item}" for item in evaluator_validation["blockers"])
    blockers = sorted(set(blockers))
    return {
        "schema_version": EVALUATOR_RUNTIME_NORMALIZATION_SCHEMA_VERSION,
        "status": "normalized" if not blockers else "blocked",
        "decision_grade_row_admitted": not blockers,
        "runtime_id": receipt.get("runtime_id"),
        "provider_id": receipt.get("provider_id"),
        "evaluator_backend_id": evaluator_validation.get("evaluator_backend_id"),
        "evaluator_model_family": evaluator_validation.get("evaluator_model_family"),
        "evaluator_profile_id": evaluator_validation.get("evaluator_profile_id"),
        "model_output_id": output_id or None,
        "model_abstained": evaluator_validation.get("model_abstained", False),
        "blockers": blockers,
        "evaluator_row": row if not blockers else None,
        "claim_boundary": {
            "runtime_completion_does_not_override_authoritative_manifest": True,
            "runtime_model_output_is_not_task_success": True,
            "fresh_model_run_steps_count_outputs_not_sampler_iterations": True,
            "provider_identity_is_separate_from_evaluator_identity": True,
            "oscar_cosmos_and_future_models_share_the_receipt_contract": True,
            "normalized_row_is_not_real_world_correlation": True,
        },
    }


def _read_json_mapping(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("runtime_evidence_request_must_be_json_object")
    return dict(payload)


def main(argv: Sequence[str] | None = None) -> int:
    """Build or normalize a runtime receipt from an immutable JSON request."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "operation",
        choices=("build-wam-receipt", "normalize-evaluator-row"),
    )
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    request = _read_json_mapping(args.request.expanduser().resolve())
    result = (
        build_wam_runtime_receipt(request)
        if args.operation == "build-wam-receipt"
        else normalize_evaluator_runtime_evidence(request)
    )
    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    accepted = result.get("status") in {"validated", "normalized"}
    print(
        json.dumps(
            {
                "schema_version": "evaluator_runtime_evidence_cli_result.v1",
                "status": result.get("status"),
                "output": str(output_path),
            },
            sort_keys=True,
        )
    )
    return 0 if accepted else 2


if __name__ == "__main__":
    raise SystemExit(main())
