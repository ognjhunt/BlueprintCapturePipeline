"""Digest-bound planning, authorization, and local reconstruction execution.

The control plane resolves source bytes from Pipeline-owned capture-upload
receipts. Clients may request claims and authorize an exact planned adapter,
but they cannot choose filesystem paths, commands, provider credentials, or
upgrade a derived result's claim ceiling.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from .capture_intake import CaptureIntakeError, validate_capture_intake_envelope
from .core.security_controls import strict_identifier
from .decision_evidence_contracts import canonical_digest, canonical_json
from .local_reconstruction_adapters import (
    LOCAL_ARKIT_METRIC_SCAFFOLD_ADAPTER,
    LOCAL_DECODED_OBSERVATION_ADAPTER,
    LocalReconstructionAdapterError,
    arkit_metric_scaffold_method_profile,
    authorized_local_reconstruction_adapter_registry,
    decoded_observation_method_profile,
)
from .reconstruction_capability import (
    ReconstructionContractError,
    normalize_reconstruction_result,
    plan_reconstruction_methods,
)


class ReconstructionControlPlaneError(ValueError):
    def __init__(self, code: str, *, status_code: int = 422) -> None:
        self.code = code
        self.status_code = status_code
        super().__init__(code)


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _strings(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return sorted({str(item).strip() for item in value if str(item).strip()})


def _read_object(path: Path, *, code: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReconstructionControlPlaneError(code, status_code=404) from exc
    if not isinstance(value, Mapping):
        raise ReconstructionControlPlaneError(code, status_code=422)
    return dict(value)


def _safe_child(root: Path, relative_path: str, *, code: str) -> Path:
    relative = PurePosixPath(str(relative_path).replace("\\", "/"))
    if (
        not relative_path
        or relative.is_absolute()
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise ReconstructionControlPlaneError(code)
    resolved_root = root.expanduser().resolve()
    candidate = (resolved_root / Path(*relative.parts)).resolve()
    if candidate != resolved_root and resolved_root not in candidate.parents:
        raise ReconstructionControlPlaneError(code)
    return candidate


def _write_immutable(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    normalized = json.loads(canonical_json(dict(value)))
    payload = (canonical_json(normalized) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        existing = _read_object(path, code=f"immutable_artifact_invalid:{path.name}")
        if canonical_json(existing) != canonical_json(normalized):
            raise ReconstructionControlPlaneError(
                f"immutable_artifact_conflict:{path.name}", status_code=409
            )
        return existing
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            existing = _read_object(path, code=f"immutable_artifact_invalid:{path.name}")
            if canonical_json(existing) != canonical_json(normalized):
                raise ReconstructionControlPlaneError(
                    f"immutable_artifact_conflict:{path.name}", status_code=409
                )
            return existing
    finally:
        temporary.unlink(missing_ok=True)
    return normalized


def _receipt_path(store_root: Path, capture_session_id: str, intake_id: str) -> Path:
    key = hashlib.sha256(f"{capture_session_id}\0{intake_id}".encode("utf-8")).hexdigest()
    return store_root / "transfer_receipts" / f"{key}.json"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _source_binding(
    *, capture_store_root: Path, capture_session_id: str, intake_id: str
) -> dict[str, Any]:
    receipt = _read_object(
        _receipt_path(capture_store_root, capture_session_id, intake_id),
        code="capture_upload_receipt_not_found",
    )
    if receipt.get("capture_session_id") != capture_session_id or receipt.get("intake_id") != intake_id:
        raise ReconstructionControlPlaneError("capture_upload_receipt_binding_mismatch")
    if receipt.get("admission_status") != "accepted" or receipt.get("state") != "capture_accepted":
        raise ReconstructionControlPlaneError("capture_not_accepted")
    qa = _mapping(receipt.get("capture_qa_report"))
    if qa.get("status") != "accepted" or qa.get("state") != "capture_accepted":
        raise ReconstructionControlPlaneError("capture_qa_not_accepted")
    artifact_reference = _mapping(receipt.get("artifact_reference"))
    artifact_root = _safe_child(
        capture_store_root,
        str(artifact_reference.get("uri") or ""),
        code="capture_artifact_reference_unsafe",
    )
    envelope = _read_object(
        artifact_root / "capture_intake_envelope.json",
        code="capture_intake_envelope_not_found",
    )
    try:
        envelope = validate_capture_intake_envelope(envelope)
    except CaptureIntakeError as exc:
        raise ReconstructionControlPlaneError("capture_intake_envelope_invalid") from exc
    artifact_qa = _read_object(
        artifact_root / "capture_qa_report.json",
        code="capture_qa_report_not_found",
    )
    if (
        canonical_json(artifact_qa) != canonical_json(qa)
        or artifact_qa.get("qa_report_digest")
        != canonical_digest(artifact_qa, digest_field="qa_report_digest")
    ):
        raise ReconstructionControlPlaneError("capture_qa_report_digest_mismatch")
    object_manifest = _read_object(
        artifact_root / "capture_intake_object_manifest.json",
        code="capture_object_manifest_not_found",
    )
    if (
        envelope.get("intake_id") != intake_id
        or envelope.get("envelope_digest") != receipt.get("envelope_digest")
        or object_manifest.get("envelope_digest") != receipt.get("envelope_digest")
    ):
        raise ReconstructionControlPlaneError("capture_artifact_binding_mismatch")
    if object_manifest.get("manifest_digest") != canonical_digest(
        object_manifest, digest_field="manifest_digest"
    ):
        raise ReconstructionControlPlaneError("capture_object_manifest_digest_mismatch")
    objects = [dict(row) for row in object_manifest.get("objects", []) if isinstance(row, Mapping)]
    matching = [row for row in objects if row.get("sha256") == receipt.get("capture_digest")]
    if len(objects) != 1 or len(matching) != 1:
        raise ReconstructionControlPlaneError("capture_object_binding_ambiguous")
    object_path = _safe_child(
        capture_store_root,
        str(matching[0].get("object_path") or ""),
        code="capture_object_reference_unsafe",
    )
    if not object_path.is_file():
        raise ReconstructionControlPlaneError("capture_object_not_found", status_code=404)
    try:
        expected_size = int(matching[0].get("size_bytes"))
    except (TypeError, ValueError) as exc:
        raise ReconstructionControlPlaneError("capture_object_size_invalid") from exc
    if object_path.stat().st_size != expected_size or _sha256_file(object_path) != matching[0].get(
        "sha256"
    ):
        raise ReconstructionControlPlaneError("capture_object_digest_or_size_mismatch", status_code=409)
    return {
        "receipt": receipt,
        "qa": qa,
        "envelope": envelope,
        "object_manifest": object_manifest,
        "artifact_root": artifact_root,
        "object_path": object_path,
        "object_relative_path": str(object_path.relative_to(capture_store_root.resolve())),
    }


def _run_root(state_root: str | Path, plan_id: str) -> Path:
    return Path(state_root).expanduser().resolve() / "plans" / plan_id


def prepare_reconstruction_plan(
    *,
    state_root: str | Path,
    capture_store_root: str | Path,
    capture_session_id: str,
    intake_id: str,
    requested_claim_types: Sequence[str],
    idempotency_key: str,
) -> dict[str, Any]:
    try:
        session = strict_identifier(capture_session_id, field="capture_session_id", max_length=192)
        intake = strict_identifier(intake_id, field="intake_id", max_length=192)
        key = strict_identifier(idempotency_key, field="idempotency_key", max_length=192)
    except ValueError as exc:
        raise ReconstructionControlPlaneError(str(exc)) from exc
    claims = sorted({str(item).strip() for item in requested_claim_types if str(item).strip()})
    if not claims:
        raise ReconstructionControlPlaneError("requested_claim_types:missing")
    source = _source_binding(
        capture_store_root=Path(capture_store_root).expanduser().resolve(),
        capture_session_id=session,
        intake_id=intake,
    )
    envelope = source["envelope"]
    receipt = source["receipt"]
    provider_values = _strings(envelope.get("permitted_reconstruction_providers"))
    local_permitted = not provider_values or bool({"local", "local_only"}.intersection(provider_values))
    profiles = [decoded_observation_method_profile(execution_authorized=local_permitted)]
    # A single-file Web upload is not enough to execute the V3.2 ARKit bundle
    # adapter. It remains available through the bundle-native lane, never by
    # treating an MP4 as poses/depth authority.
    profile = str(envelope.get("capture_authority_profile") or "")
    if profile == "iphone_arkit_lidar" and (source["artifact_root"] / "capture_bundle/manifest.json").is_file():
        profiles.append(arkit_metric_scaffold_method_profile(execution_authorized=local_permitted))
    plan = plan_reconstruction_methods(
        intake_id=intake,
        capture_digest=str(receipt.get("capture_digest") or ""),
        capture_authority_profile=profile,
        claim_ceiling=_mapping(receipt.get("claim_ceiling")),
        requested_claim_types=claims,
        permitted_provider_identities=["local"] if local_permitted else [],
        method_profiles=profiles,
    )
    plan_id = f"reconstruction-{plan['reconstruction_plan_digest'][7:31]}"
    root = _run_root(state_root, plan_id)
    context = {
        "schema_version": "reconstruction_control_plane_context.v1",
        "plan_id": plan_id,
        "capture_session_id": session,
        "intake_id": intake,
        "capture_digest": receipt["capture_digest"],
        "envelope_digest": receipt["envelope_digest"],
        "qa_report_digest": source["qa"].get("qa_report_digest"),
        "capture_authority_profile": profile,
        "object_manifest_digest": source["object_manifest"].get("manifest_digest"),
        "object_relative_path": source["object_relative_path"],
        "capture_artifact_reference": source["receipt"]["artifact_reference"],
        "rights_and_retention": _mapping(envelope.get("governance")),
        "idempotency_key": key,
    }
    context["context_digest"] = canonical_digest(context, digest_field="context_digest")
    stored_context = _write_immutable(root / "artifacts" / "context.json", context)
    stored_plan = _write_immutable(root / "artifacts" / "reconstruction_plan.json", plan)
    state = "authorization_required" if stored_plan["selected_methods"] else "abstained"
    return {
        "schema_version": "reconstruction_control_plane_plan_result.v1",
        "plan_id": plan_id,
        "state": state,
        "context_digest": stored_context["context_digest"],
        "reconstruction_plan": stored_plan,
        "authorization_candidates": [
            {
                "method_id": row["method_id"],
                "method_profile_digest": row["method_profile_digest"],
                "adapter_reference": row.get("adapter_reference"),
                "execution_authorized": False,
            }
            for row in stored_plan["selected_methods"]
        ],
        "next_cheapest_experiments": [
            row["next_cheapest_experiment"] for row in stored_plan["missing_representations"]
        ],
        "proof_boundary": {
            "plan_is_execution_authorization": False,
            "derived_reconstruction_upgrades_raw_capture": False,
            "physical_task_success_established": False,
            "comparative_policy_ranking_verdict": "thesis_not_supported",
        },
    }


def authorize_reconstruction_plan(
    *,
    state_root: str | Path,
    plan_id: str,
    reconstruction_plan_digest: str,
    authorized_adapter_references: Sequence[str],
    actor: Mapping[str, Any],
    idempotency_key: str,
) -> dict[str, Any]:
    try:
        plan_identifier = strict_identifier(plan_id, field="plan_id", max_length=192)
        key = strict_identifier(idempotency_key, field="idempotency_key", max_length=192)
    except ValueError as exc:
        raise ReconstructionControlPlaneError(str(exc)) from exc
    root = _run_root(state_root, plan_identifier)
    plan = _read_object(root / "artifacts" / "reconstruction_plan.json", code="reconstruction_plan_not_found")
    if plan.get("reconstruction_plan_digest") != reconstruction_plan_digest:
        raise ReconstructionControlPlaneError("authorization_plan_digest_mismatch", status_code=409)
    planned = {
        str(row.get("adapter_reference") or "")
        for row in plan.get("selected_methods", [])
        if isinstance(row, Mapping) and row.get("adapter_reference")
    }
    registry = authorized_local_reconstruction_adapter_registry(authorized_adapter_references)
    authorized = sorted(registry)
    if not authorized:
        raise ReconstructionControlPlaneError("authorization_adapter_missing")
    if set(authorized) - planned:
        raise ReconstructionControlPlaneError("authorization_adapter_not_planned")
    actor_value = dict(actor)
    forbidden = {
        str(name)
        for name, value in actor_value.items()
        if value not in (None, "", [], {})
        and (
            str(name).lower() in {"authorization", "credential", "credentials", "password", "secret", "token"}
            or str(name).lower().endswith(("_token", "_secret", "_password"))
        )
    }
    if forbidden:
        raise ReconstructionControlPlaneError("authorization_actor_secret_forbidden")
    context = _read_object(root / "artifacts" / "context.json", code="reconstruction_context_not_found")
    authorization = {
        "schema_version": "reconstruction_execution_authorization.v1",
        "plan_id": plan_identifier,
        "reconstruction_plan_digest": reconstruction_plan_digest,
        "context_digest": context["context_digest"],
        "authorized_adapter_references": authorized,
        "actor": actor_value,
        "idempotency_key": key,
        "live_provider_execution": False,
        "paid_compute_authorized": False,
        "physical_robot_run_authorized": False,
        "proof_boundary": {
            "authorization_is_method_qualification": False,
            "simulation_is_physical_success": False,
            "comparative_policy_ranking_verdict": "thesis_not_supported",
        },
    }
    authorization["authorization_digest"] = canonical_digest(
        authorization, digest_field="authorization_digest"
    )
    return _write_immutable(root / "artifacts" / "execution_authorization.json", authorization)


def execute_reconstruction_plan(
    *, state_root: str | Path, capture_store_root: str | Path, plan_id: str
) -> dict[str, Any]:
    try:
        plan_identifier = strict_identifier(plan_id, field="plan_id", max_length=192)
    except ValueError as exc:
        raise ReconstructionControlPlaneError(str(exc)) from exc
    root = _run_root(state_root, plan_identifier)
    execution_path = root / "artifacts" / "execution_result.json"
    if execution_path.is_file():
        return {**_read_object(execution_path, code="reconstruction_execution_result_invalid"), "already_exists": True}
    plan = _read_object(root / "artifacts" / "reconstruction_plan.json", code="reconstruction_plan_not_found")
    context = _read_object(root / "artifacts" / "context.json", code="reconstruction_context_not_found")
    authorization = _read_object(
        root / "artifacts" / "execution_authorization.json",
        code="reconstruction_authorization_not_found",
    )
    if (
        authorization.get("reconstruction_plan_digest") != plan.get("reconstruction_plan_digest")
        or authorization.get("context_digest") != context.get("context_digest")
    ):
        raise ReconstructionControlPlaneError("reconstruction_authorization_stale", status_code=409)
    store_root = Path(capture_store_root).expanduser().resolve()
    source = _source_binding(
        capture_store_root=store_root,
        capture_session_id=str(context["capture_session_id"]),
        intake_id=str(context["intake_id"]),
    )
    if (
        source["receipt"].get("capture_digest") != context.get("capture_digest")
        or source["receipt"].get("envelope_digest") != context.get("envelope_digest")
        or source["object_manifest"].get("manifest_digest") != context.get("object_manifest_digest")
        or source["object_relative_path"] != context.get("object_relative_path")
    ):
        raise ReconstructionControlPlaneError("reconstruction_source_stale", status_code=409)
    registry = authorized_local_reconstruction_adapter_registry(
        authorization.get("authorized_adapter_references", [])
    )
    selected = {
        str(row.get("adapter_reference") or ""): dict(row)
        for row in plan.get("selected_methods", [])
        if isinstance(row, Mapping) and row.get("adapter_reference")
    }
    results: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    output_root = root / "derived"
    for reference, adapter in sorted(registry.items()):
        if reference not in selected:
            raise ReconstructionControlPlaneError("execution_adapter_not_planned")
        try:
            if reference == LOCAL_DECODED_OBSERVATION_ADAPTER:
                result = adapter.execute(
                    intake_id=str(context["intake_id"]),
                    capture_digest=str(context["capture_digest"]),
                    capture_authority_profile=str(context["capture_authority_profile"]),
                    capture_root=store_root,
                    video_relative_path=str(context["object_relative_path"]),
                    output_root=output_root,
                    rights_and_retention=_mapping(context.get("rights_and_retention")),
                )
            elif reference == LOCAL_ARKIT_METRIC_SCAFFOLD_ADAPTER:
                result = adapter.execute(
                    intake_id=str(context["intake_id"]),
                    capture_digest=str(context["capture_digest"]),
                    capture_root=source["artifact_root"] / "capture_bundle",
                    output_root=output_root,
                    rights_and_retention=_mapping(context.get("rights_and_retention")),
                )
            else:  # registry is fail-closed; retain an explicit guard.
                raise ReconstructionControlPlaneError("execution_adapter_not_supported")
            normalized = normalize_reconstruction_result(result)
            if (
                normalized["capture_digest"] != context["capture_digest"]
                or normalized["intake_id"] != context["intake_id"]
                or normalized["method_profile_digest"] != selected[reference]["method_profile_digest"]
            ):
                raise ReconstructionControlPlaneError("reconstruction_result_binding_mismatch")
            results.append(normalized)
        except (LocalReconstructionAdapterError, ReconstructionContractError) as exc:
            errors.append(
                {
                    "adapter_reference": reference,
                    "blockers": sorted(set(getattr(exc, "errors", [str(exc)]))),
                    "next_cheapest_experiment": "provide a supported retained capture or the missing source evidence",
                }
            )
    outputs = {output for result in results for output in result.get("outputs", [])}
    required = set(plan.get("required_representations", []))
    if required and required.issubset(outputs) and not errors:
        state = "completed"
    elif results:
        state = "partial"
    else:
        state = "abstained"
    execution = {
        "schema_version": "reconstruction_control_plane_execution_result.v1",
        "plan_id": plan_identifier,
        "state": state,
        "reconstruction_plan_digest": plan["reconstruction_plan_digest"],
        "authorization_digest": authorization["authorization_digest"],
        "context_digest": context["context_digest"],
        "results": sorted(results, key=lambda row: row["reconstruction_result_digest"]),
        "errors": errors,
        "missing_representations": sorted(required - outputs),
        "next_cheapest_experiments": sorted(
            {
                *(
                    str(row.get("next_cheapest_experiment") or "")
                    for row in plan.get("missing_representations", [])
                    if isinstance(row, Mapping)
                ),
                *(str(row["next_cheapest_experiment"]) for row in errors),
            }
            - {""}
        ),
        "cost_usd": round(sum(float(row.get("cost_usd") or 0.0) for row in results), 6),
        "proof_boundary": {
            "execution_was_local_and_explicitly_authorized": True,
            "derived_reconstruction_upgrades_raw_capture": False,
            "physical_task_success_established": False,
            "deployment_or_safety_approved": False,
            "comparative_policy_ranking_verdict": "thesis_not_supported",
        },
        "already_exists": False,
    }
    execution["execution_result_digest"] = canonical_digest(
        execution, digest_field="execution_result_digest"
    )
    return _write_immutable(execution_path, execution)


def inspect_reconstruction_plan(*, state_root: str | Path, plan_id: str) -> dict[str, Any]:
    try:
        identifier = strict_identifier(plan_id, field="plan_id", max_length=192)
    except ValueError as exc:
        raise ReconstructionControlPlaneError(str(exc)) from exc
    artifacts = _run_root(state_root, identifier) / "artifacts"
    plan = _read_object(artifacts / "reconstruction_plan.json", code="reconstruction_plan_not_found")
    context = _read_object(artifacts / "context.json", code="reconstruction_context_not_found")
    authorization = (
        _read_object(artifacts / "execution_authorization.json", code="reconstruction_authorization_invalid")
        if (artifacts / "execution_authorization.json").is_file()
        else None
    )
    execution = (
        _read_object(artifacts / "execution_result.json", code="reconstruction_execution_result_invalid")
        if (artifacts / "execution_result.json").is_file()
        else None
    )
    state = execution["state"] if execution else ("authorization_required" if plan["selected_methods"] else "abstained")
    return {
        "schema_version": "reconstruction_control_plane_inspection.v1",
        "plan_id": identifier,
        "state": state,
        "source_binding": {
            key: context.get(key)
            for key in (
                "capture_session_id",
                "intake_id",
                "capture_digest",
                "envelope_digest",
                "qa_report_digest",
                "object_manifest_digest",
                "context_digest",
            )
        },
        "reconstruction_plan": plan,
        "execution_authorization": authorization,
        "execution_result": execution,
        "proof_boundary": {
            "inspection_recomputes_scientific_truth": False,
            "physical_task_success_established": False,
            "comparative_policy_ranking_verdict": "thesis_not_supported",
        },
    }


def load_reconstruction_compilation_inputs(
    *,
    state_root: str | Path,
    capture_store_root: str | Path,
    plan_id: str,
    execution_result_digest: str,
) -> dict[str, Any]:
    """Load exact Pipeline-owned inputs for immutable testbed compilation."""

    inspection = inspect_reconstruction_plan(state_root=state_root, plan_id=plan_id)
    execution = inspection.get("execution_result")
    if not isinstance(execution, Mapping):
        raise ReconstructionControlPlaneError("reconstruction_execution_required")
    if execution.get("execution_result_digest") != execution_result_digest:
        raise ReconstructionControlPlaneError(
            "reconstruction_execution_digest_mismatch", status_code=409
        )
    if execution.get("state") not in {"completed", "partial", "abstained"}:
        raise ReconstructionControlPlaneError("reconstruction_execution_not_terminal")
    context = _read_object(
        _run_root(state_root, plan_id) / "artifacts" / "context.json",
        code="reconstruction_context_not_found",
    )
    source = _source_binding(
        capture_store_root=Path(capture_store_root).expanduser().resolve(),
        capture_session_id=str(context["capture_session_id"]),
        intake_id=str(context["intake_id"]),
    )
    if (
        source["receipt"].get("capture_digest") != context.get("capture_digest")
        or source["receipt"].get("envelope_digest") != context.get("envelope_digest")
    ):
        raise ReconstructionControlPlaneError("reconstruction_source_stale", status_code=409)
    return {
        "context": context,
        "capture_intake_envelope": source["envelope"],
        "capture_qa_report": source["qa"],
        "reconstruction_plan": inspection["reconstruction_plan"],
        "reconstruction_results": [
            dict(row) for row in execution.get("results", []) if isinstance(row, Mapping)
        ],
        "execution_result": dict(execution),
    }


__all__ = [
    "ReconstructionControlPlaneError",
    "authorize_reconstruction_plan",
    "execute_reconstruction_plan",
    "inspect_reconstruction_plan",
    "load_reconstruction_compilation_inputs",
    "prepare_reconstruction_plan",
]
