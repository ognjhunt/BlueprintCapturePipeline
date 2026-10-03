"""Fail-closed visual diagnosis for configured-scene native controls.

The vision model at this seam is a proposal author, never an evaluator.  It may
suggest a small, typed change, but this module cannot apply that change and can
never produce controls qualification.  Native construction readback, collision
and reachability probes, and deterministic zero-action/scripted-positive traces
remain the only authorities consumed by the next controls run.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .common import write_json
from .decision_evidence_contracts import canonical_digest


REQUEST_SCHEMA_VERSION = "task_evaluation_controls_visual_supervisor_request.v1"
RESULT_SCHEMA_VERSION = "task_evaluation_controls_visual_supervisor_result.v1"
AUTHORIZATION_SCHEMA_VERSION = "task_evaluation_controls_visual_supervisor_authorization.v1"
COST_SCOPE_SCHEMA_VERSION = "task_evaluation_controls_visual_supervisor_cost_scope.v1"
EXACT_FRAME_COUNT = 8
HARD_MAX_ATTEMPTS = 3
HARD_MAX_REVISIONS_PER_ATTEMPT = 5
HARD_MAX_COST_USD = 0.25
HARD_MAX_JSON_ARTIFACT_BYTES = 512 * 1024
HARD_MAX_JSON_INPUT_BYTES = 2 * 1024 * 1024
HARD_MAX_RENDER_BYTES = 16 * 1024 * 1024
HARD_MAX_RENDER_INPUT_BYTES = 64 * 1024 * 1024

_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,199}\Z")
_SECRET_KEYS = re.compile(
    r"(?:api[_-]?key|access[_-]?token|password|passwd|secret|authorization|cookie|private[_-]?key)",
    re.IGNORECASE,
)
_UNSAFE_TEXT = re.compile(
    r"(?:bearer\s+|sk-[A-Za-z0-9]|-----BEGIN [A-Z ]*PRIVATE KEY-----|"
    r"(?:file|s3|gs)://|/(?:private|var|tmp|home|Users)/)",
    re.IGNORECASE,
)

ArtifactInvoker = Callable[
    [Mapping[str, Any], Sequence[bytes], str, str], Mapping[str, Any]
]


class ControlsVisualSupervisorError(RuntimeError):
    """The diagnostic request cannot safely reach the visual-model seam."""


class BoundedRevision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["camera", "base", "contact", "approach", "task"]
    target_id: str = Field(min_length=1, max_length=200)
    parameters: dict[str, Any]
    rationale: str = Field(min_length=1, max_length=1_000)


class VisualDiagnosis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    disposition: Literal["propose_bounded_revision", "abstain"]
    diagnoses: list[str] = Field(min_length=1, max_length=8)
    revisions: list[BoundedRevision] = Field(max_length=HARD_MAX_REVISIONS_PER_ATTEMPT)
    uncertainty: str = Field(min_length=1, max_length=2_000)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _safe_text(value: Any) -> bool:
    return (
        isinstance(value, str)
        and bool(value.strip())
        and not _UNSAFE_TEXT.search(value)
        and not _SECRET_KEYS.search(value)
        and all(ord(character) >= 32 or character in "\n\t" for character in value)
    )


def _secret_safe_json(value: Any) -> bool:
    if isinstance(value, Mapping):
        return all(
            isinstance(key, str)
            and not _SECRET_KEYS.search(key)
            and _secret_safe_json(child)
            for key, child in value.items()
        )
    if isinstance(value, list):
        return all(_secret_safe_json(child) for child in value)
    if isinstance(value, str):
        return _UNSAFE_TEXT.search(value) is None
    if isinstance(value, float):
        return math.isfinite(value)
    return value is None or isinstance(value, (bool, int))


def _artifact(
    value: Any, *, base: Path, code: str, json_required: bool
) -> tuple[dict[str, Any], Path, Any | None]:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256", "size_bytes"}:
        raise ControlsVisualSupervisorError(code)
    unresolved = Path(str(value.get("path") or "")).expanduser()
    candidate = unresolved if unresolved.is_absolute() else base / unresolved
    path = candidate.resolve()
    if (
        candidate.is_symlink()
        or path.is_symlink()
        or not path.is_file()
        or isinstance(value.get("size_bytes"), bool)
        or value.get("size_bytes") != path.stat().st_size
        or not _DIGEST.fullmatch(str(value.get("sha256") or ""))
        or _sha256(path) != value.get("sha256")
        or (json_required and path.stat().st_size > HARD_MAX_JSON_ARTIFACT_BYTES)
    ):
        raise ControlsVisualSupervisorError(code)
    document = None
    if json_required:
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ControlsVisualSupervisorError(code) from exc
        if not isinstance(document, Mapping) or not _secret_safe_json(document):
            raise ControlsVisualSupervisorError(f"{code}:secret_unsafe_or_invalid_json")
    retained = {"sha256": value["sha256"], "size_bytes": value["size_bytes"]}
    return retained, path, document


def _credential(path_value: str | Path) -> str:
    path = Path(path_value).expanduser()
    try:
        mode = path.stat().st_mode & 0o777
        value = path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError) as exc:
        raise ControlsVisualSupervisorError("controls_visual_supervisor_credential_invalid") from exc
    if path.is_symlink() or not path.is_file() or mode != 0o600 or not value:
        raise ControlsVisualSupervisorError("controls_visual_supervisor_credential_invalid")
    return value


def _authorization(value: Mapping[str, Any], *, request: Mapping[str, Any]) -> None:
    cost = request["cost_scope"]
    if (
        set(value)
        != {
            "schema_version",
            "status",
            "program_id",
            "run_id",
            "provider",
            "model",
            "private_derived_renders_disclosure_authorized",
            "configured_manifests_disclosure_authorized",
            "native_readback_and_deterministic_traces_disclosure_authorized",
            "raw_capture_or_splat_disclosure_authorized",
            "provider_training_authorized",
            "revision_proposal_only",
            "vlm_may_grade_controls",
            "issued_by_agent",
            "authorized_by",
            "authorization_digest",
        }
        or value.get("schema_version") != AUTHORIZATION_SCHEMA_VERSION
        or value.get("status") != "authorized"
        or value.get("program_id") != "arm-decision-proof-v1"
        or value.get("run_id") != request.get("run_id")
        or value.get("provider") != cost.get("provider")
        or value.get("model") != cost.get("model")
        or value.get("private_derived_renders_disclosure_authorized") is not True
        or value.get("configured_manifests_disclosure_authorized") is not True
        or value.get("native_readback_and_deterministic_traces_disclosure_authorized")
        is not True
        or value.get("raw_capture_or_splat_disclosure_authorized") is not False
        or value.get("provider_training_authorized") is not False
        or value.get("revision_proposal_only") is not True
        or value.get("vlm_may_grade_controls") is not False
        or value.get("issued_by_agent") is not False
        or value.get("authorization_digest")
        != canonical_digest(value, digest_field="authorization_digest")
    ):
        raise ControlsVisualSupervisorError("controls_visual_supervisor_authorization_invalid")


def _cost_scope(value: Mapping[str, Any], *, request: Mapping[str, Any]) -> None:
    cap = value.get("maximum_cost_usd")
    if (
        set(value)
        != {
            "schema_version",
            "status",
            "run_id",
            "attempt",
            "provider",
            "model",
            "exclusive_scope",
            "zero_cost_baseline_confirmed",
            "maximum_cost_usd",
            "cost_scope_digest",
        }
        or
        value.get("schema_version") != COST_SCOPE_SCHEMA_VERSION
        or value.get("status") != "reserved_before_vlm_call"
        or value.get("run_id") != request.get("run_id")
        or value.get("attempt") != request.get("attempt")
        or not _SAFE_ID.fullmatch(str(value.get("provider") or ""))
        or not _SAFE_ID.fullmatch(str(value.get("model") or ""))
        or value.get("exclusive_scope") is not True
        or value.get("zero_cost_baseline_confirmed") is not True
        or isinstance(cap, bool)
        or not isinstance(cap, (int, float))
        or not math.isfinite(float(cap))
        or not 0 < float(cap) <= HARD_MAX_COST_USD
        or value.get("cost_scope_digest")
        != canonical_digest(value, digest_field="cost_scope_digest")
    ):
        raise ControlsVisualSupervisorError("controls_visual_supervisor_cost_scope_invalid")


def _vector(value: Any, *, limit: float) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 3
        and all(
            isinstance(part, (int, float))
            and not isinstance(part, bool)
            and math.isfinite(float(part))
            and abs(float(part)) <= limit
            for part in value
        )
    )


def _bounded_revisions(
    diagnosis: VisualDiagnosis, *, scope: Mapping[str, Any]
) -> list[dict[str, Any]]:
    if diagnosis.disposition == "abstain":
        if diagnosis.revisions:
            raise ControlsVisualSupervisorError("controls_visual_supervisor_abstention_has_revisions")
        return []
    if not diagnosis.revisions:
        raise ControlsVisualSupervisorError("controls_visual_supervisor_proposal_empty")
    allowed_kinds = scope.get("allowed_kinds")
    targets = scope.get("allowed_target_ids")
    if not isinstance(allowed_kinds, list) or not isinstance(targets, Mapping):
        raise ControlsVisualSupervisorError("controls_visual_supervisor_revision_scope_invalid")
    expected_parameters = {
        "camera": {"translation_delta_m", "rotation_delta_rad"},
        "base": {"translation_delta_m", "rotation_delta_rad"},
        "contact": {"translation_delta_m", "rotation_delta_rad"},
        "approach": {"translation_delta_m", "rotation_delta_rad"},
        "task": {"start_position_delta_m", "goal_position_delta_m"},
    }
    limits = {
        "camera": (0.05, 0.1745329252),
        "base": (0.05, 0.1745329252),
        "contact": (0.02, 0.0872664626),
        "approach": (0.05, 0.0872664626),
        "task": (0.05, 0.05),
    }
    retained: list[dict[str, Any]] = []
    identities: set[tuple[str, str]] = set()
    for row in diagnosis.revisions:
        target_ids = targets.get(row.kind)
        parameter_names = {
            "camera": ("translation_delta_m", "rotation_delta_rad"),
            "base": ("translation_delta_m", "rotation_delta_rad"),
            "contact": ("translation_delta_m", "rotation_delta_rad"),
            "approach": ("translation_delta_m", "rotation_delta_rad"),
            "task": ("start_position_delta_m", "goal_position_delta_m"),
        }
        first, second = parameter_names[row.kind]
        valid = (
            row.kind in allowed_kinds
            and isinstance(target_ids, list)
            and row.target_id in target_ids
            and set(row.parameters) == expected_parameters[row.kind]
            and _vector(row.parameters.get(first), limit=limits[row.kind][0])
            and _vector(row.parameters.get(second), limit=limits[row.kind][1])
            and _safe_text(row.rationale)
            and (row.kind, row.target_id) not in identities
        )
        if not valid:
            raise ControlsVisualSupervisorError("controls_visual_supervisor_revision_out_of_bounds")
        identities.add((row.kind, row.target_id))
        retained.append(row.model_dump(mode="json"))
    return retained


def _validate_prior_receipts(
    rows: Any, *, base: Path, run_id: str, attempt: int
) -> list[dict[str, Any]]:
    if not isinstance(rows, list) or len(rows) != attempt - 1:
        raise ControlsVisualSupervisorError("controls_visual_supervisor_attempt_chain_invalid")
    retained: list[dict[str, Any]] = []
    for expected_attempt, row in enumerate(rows, start=1):
        record, _path, document = _artifact(
            row,
            base=base,
            code="controls_visual_supervisor_prior_receipt_invalid",
            json_required=True,
        )
        if (
            document.get("schema_version") != RESULT_SCHEMA_VERSION
            or document.get("run_id") != run_id
            or document.get("attempt") != expected_attempt
            or document.get("controls_qualified") is not False
            or document.get("result_digest")
            != canonical_digest(document, digest_field="result_digest")
        ):
            raise ControlsVisualSupervisorError("controls_visual_supervisor_prior_receipt_invalid")
        retained.append({**record, "attempt": expected_attempt, "result_digest": document["result_digest"]})
    return retained


def _existing(
    path: Path,
    *,
    request_digest: str,
    run_id: str,
    attempt: int,
    max_attempts: int,
) -> dict[str, Any] | None:
    if not path.exists() and not path.is_symlink():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ControlsVisualSupervisorError("controls_visual_supervisor_existing_result_invalid") from exc
    if (
        path.is_symlink()
        or not isinstance(value, Mapping)
        or value.get("schema_version") != RESULT_SCHEMA_VERSION
        or value.get("source_request_digest") != request_digest
        or value.get("run_id") != run_id
        or value.get("attempt") != attempt
        or value.get("max_attempts") != max_attempts
        or value.get("controls_qualified") is not False
        or value.get("vlm_may_grade_success") is not False
        or value.get("revision_applied") is not False
        or value.get("candidate_policy_queried") is not False
        or value.get("proof_effect") != "none"
        or value.get("claim_ceiling")
        != "development_only_controls_diagnostic_proposal"
        or value.get("result_digest") != canonical_digest(value, digest_field="result_digest")
    ):
        raise ControlsVisualSupervisorError("controls_visual_supervisor_existing_result_conflict")
    return dict(value)


def run_controls_visual_supervisor(
    *,
    request_path: str | Path,
    output_root: str | Path,
    credential_file: str | Path,
    invoker: ArtifactInvoker,
) -> dict[str, Any]:
    """Validate, diagnose once, and retain a non-authoritative bounded proposal."""

    source_candidate = Path(request_path).expanduser()
    if source_candidate.is_symlink():
        raise ControlsVisualSupervisorError("controls_visual_supervisor_request_invalid")
    source = source_candidate.resolve()
    try:
        request = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ControlsVisualSupervisorError("controls_visual_supervisor_request_invalid") from exc
    if source.is_symlink() or not isinstance(request, Mapping):
        raise ControlsVisualSupervisorError("controls_visual_supervisor_request_invalid")
    attempt = request.get("attempt")
    maximum = request.get("max_attempts")
    if (
        set(request)
        != {
            "schema_version",
            "run_id",
            "attempt",
            "max_attempts",
            "renders",
            "inputs",
            "revision_scope",
            "vlm_authorization",
            "cost_scope",
            "prior_attempt_receipts",
            "request_digest",
        }
        or
        request.get("schema_version") != REQUEST_SCHEMA_VERSION
        or not _SAFE_ID.fullmatch(str(request.get("run_id") or ""))
        or isinstance(attempt, bool)
        or not isinstance(attempt, int)
        or isinstance(maximum, bool)
        or not isinstance(maximum, int)
        or not 1 <= attempt <= maximum <= HARD_MAX_ATTEMPTS
        or request.get("request_digest")
        != canonical_digest(request, digest_field="request_digest")
    ):
        raise ControlsVisualSupervisorError("controls_visual_supervisor_request_invalid")

    destination_candidate = Path(output_root).expanduser()
    if destination_candidate.is_symlink():
        raise ControlsVisualSupervisorError("controls_visual_supervisor_output_invalid")
    destination = destination_candidate.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    result_path = destination / f"controls_visual_supervisor_attempt_{attempt}.v1.json"
    base = source.parent
    frames = request.get("renders")
    if not isinstance(frames, list) or len(frames) != EXACT_FRAME_COUNT:
        raise ControlsVisualSupervisorError("controls_visual_supervisor_requires_exactly_8_renders")
    frame_inventory: list[dict[str, Any]] = []
    frame_bytes: list[bytes] = []
    cameras: set[str] = set()
    for index, row in enumerate(frames):
        if not isinstance(row, Mapping) or set(row) != {"camera_id", "artifact"}:
            raise ControlsVisualSupervisorError("controls_visual_supervisor_render_invalid")
        camera_id = str(row.get("camera_id") or "")
        if not _SAFE_ID.fullmatch(camera_id) or camera_id in cameras:
            raise ControlsVisualSupervisorError("controls_visual_supervisor_render_invalid")
        record, path, _document = _artifact(
            row["artifact"], base=base, code="controls_visual_supervisor_render_invalid", json_required=False
        )
        data = path.read_bytes()
        if (
            not 8 < len(data) <= HARD_MAX_RENDER_BYTES
            or not (data.startswith(b"\x89PNG\r\n\x1a\n") or data.startswith(b"\xff\xd8\xff"))
            or record["sha256"] in {existing["sha256"] for existing in frame_inventory}
        ):
            raise ControlsVisualSupervisorError("controls_visual_supervisor_render_invalid")
        cameras.add(camera_id)
        frame_inventory.append({"frame_index": index, "camera_id": camera_id, **record})
        frame_bytes.append(data)
    if sum(len(data) for data in frame_bytes) > HARD_MAX_RENDER_INPUT_BYTES:
        raise ControlsVisualSupervisorError("controls_visual_supervisor_render_input_too_large")

    inputs = request.get("inputs")
    names = (
        "configured_usd_manifest",
        "task_manifest",
        "robot_manifest",
        "camera_manifest",
        "native_construction_readback",
        "collision_result",
        "reachability_result",
        "zero_action_trace",
        "scripted_positive_trace",
    )
    if not isinstance(inputs, Mapping) or set(inputs) != set(names):
        raise ControlsVisualSupervisorError("controls_visual_supervisor_inputs_invalid")
    inventory: dict[str, Any] = {}
    documents: dict[str, Any] = {}
    for name in names:
        inventory[name], _path, documents[name] = _artifact(
            inputs[name], base=base, code=f"controls_visual_supervisor_input_invalid:{name}", json_required=True
        )
    if sum(row["size_bytes"] for row in inventory.values()) > HARD_MAX_JSON_INPUT_BYTES:
        raise ControlsVisualSupervisorError("controls_visual_supervisor_json_input_too_large")
    if (
        documents["zero_action_trace"].get("control_selection") != "zero_action_negative"
        or documents["scripted_positive_trace"].get("control_selection")
        != "deterministic_scripted_positive"
        or any(
            document.get("candidate_policy_queried") is not False
            for document in (
                documents["zero_action_trace"],
                documents["scripted_positive_trace"],
            )
        )
    ):
        raise ControlsVisualSupervisorError("controls_visual_supervisor_control_trace_invalid")

    cost = request.get("cost_scope")
    authorization = request.get("vlm_authorization")
    if not isinstance(cost, Mapping) or not isinstance(authorization, Mapping):
        raise ControlsVisualSupervisorError("controls_visual_supervisor_authority_invalid")
    _cost_scope(cost, request=request)
    _authorization(authorization, request=request)
    revision_scope = request.get("revision_scope")
    if (
        not isinstance(revision_scope, Mapping)
        or set(revision_scope) != {"allowed_kinds", "allowed_target_ids"}
        or not isinstance(revision_scope.get("allowed_kinds"), list)
        or not revision_scope["allowed_kinds"]
        or len(revision_scope["allowed_kinds"])
        != len(set(revision_scope["allowed_kinds"]))
        or any(
            kind not in {"camera", "base", "contact", "approach", "task"}
            for kind in revision_scope["allowed_kinds"]
        )
        or not isinstance(revision_scope.get("allowed_target_ids"), Mapping)
        or set(revision_scope["allowed_target_ids"]) != set(revision_scope["allowed_kinds"])
        or any(
            not isinstance(target_ids, list)
            or not target_ids
            or len(target_ids) != len(set(target_ids))
            or any(not _SAFE_ID.fullmatch(str(target_id or "")) for target_id in target_ids)
            for target_ids in revision_scope["allowed_target_ids"].values()
        )
    ):
        raise ControlsVisualSupervisorError("controls_visual_supervisor_revision_scope_invalid")
    prior = _validate_prior_receipts(
        request.get("prior_attempt_receipts"), base=base, run_id=request["run_id"], attempt=attempt
    )
    replay = _existing(
        result_path,
        request_digest=request["request_digest"],
        run_id=request["run_id"],
        attempt=attempt,
        max_attempts=maximum,
    )
    if replay is not None:
        return replay
    secret = _credential(credential_file)
    idempotency_key = request["request_digest"]
    prompt = {
        "instruction": (
            "Diagnose only. Propose at most five bounded camera, base, contact, approach, or task "
            "parameter deltas, or abstain. Never grade success, relax a tolerance, claim controls "
            "qualification, or replace deterministic simulator/contact/collision/readback authority."
        ),
        "run_id": request["run_id"],
        "attempt": attempt,
        "authoritative_documents": documents,
        "render_inventory": frame_inventory,
        "revision_scope": revision_scope,
    }

    blockers: list[str] = []
    provider_called = False
    diagnosis_value: dict[str, Any] | None = None
    revisions: list[dict[str, Any]] = []
    provider_request_id: str | None = None
    reported_cost: float | None = None
    try:
        provider_called = True
        envelope = dict(invoker(prompt, frame_bytes, secret, idempotency_key))
        if (
            envelope.get("provider") != cost["provider"]
            or envelope.get("model") != cost["model"]
            or envelope.get("response_store") is not False
            or envelope.get("tracing_disabled") is not True
        ):
            raise ControlsVisualSupervisorError("controls_visual_supervisor_provider_identity_invalid")
        provider_request_id = str(envelope.get("provider_request_id") or "")
        reported_cost = envelope.get("reported_cost_usd")
        if (
            not _SAFE_ID.fullmatch(provider_request_id)
            or isinstance(reported_cost, bool)
            or not isinstance(reported_cost, (int, float))
            or not math.isfinite(float(reported_cost))
            or not 0 <= float(reported_cost) <= float(cost["maximum_cost_usd"])
        ):
            raise ControlsVisualSupervisorError("controls_visual_supervisor_provider_receipt_invalid")
        diagnosis = VisualDiagnosis.model_validate(envelope.get("response"))
        if not all(_safe_text(text) for text in [*diagnosis.diagnoses, diagnosis.uncertainty]):
            raise ControlsVisualSupervisorError("controls_visual_supervisor_output_secret_unsafe")
        revisions = _bounded_revisions(diagnosis, scope=revision_scope)
        diagnosis_value = diagnosis.model_dump(mode="json")
        if diagnosis.disposition == "abstain":
            blockers = ["vlm_diagnosis_abstained"]
    except (ControlsVisualSupervisorError, ValidationError, TypeError, ValueError) as exc:
        blockers = [f"vlm_diagnostic_abstained:{type(exc).__name__}"]
        diagnosis_value = None
        revisions = []
    except Exception as exc:  # provider errors are retained by type, never message
        blockers = [f"vlm_transport_abstained:{type(exc).__name__}"]

    proposed = bool(diagnosis_value and diagnosis_value["disposition"] == "propose_bounded_revision")
    result: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "bounded_revision_proposed" if proposed else "abstained_fail_closed",
        "run_id": request["run_id"],
        "attempt": attempt,
        "max_attempts": maximum,
        "source_request_digest": request["request_digest"],
        "render_inventory": frame_inventory,
        "authoritative_input_inventory": inventory,
        "prior_attempt_receipts": prior,
        "vlm": {
            "provider": cost["provider"],
            "model": cost["model"],
            "provider_called": provider_called,
            "provider_request_id": provider_request_id,
            "reported_cost_usd": float(reported_cost) if reported_cost is not None else None,
            "maximum_cost_usd": float(cost["maximum_cost_usd"]),
            "cost_scope_digest": cost["cost_scope_digest"],
            "authorization_digest": authorization["authorization_digest"],
            "credential_source": "secret_file",
            "credential_path_recorded": False,
            "raw_credential_recorded": False,
            "response_store": False,
            "tracing_disabled": True,
            "provider_reported_cost_authoritative": False,
            "strict_official_billing_satisfied": False,
        },
        "diagnosis": diagnosis_value,
        "bounded_revisions": revisions,
        "blockers": blockers,
        "revision_applied": False,
        "requires_deterministic_rerun": proposed,
        "deterministic_simulator_state_authoritative": True,
        "native_contact_collision_readback_authoritative": True,
        "scripted_controls_authoritative": True,
        "vlm_diagnosis_advisory_only": True,
        "vlm_may_grade_success": False,
        "controls_qualified": False,
        "candidate_policy_queried": False,
        "proof_effect": "none",
        "claim_ceiling": "development_only_controls_diagnostic_proposal",
        "result_digest": "",
    }
    result["result_digest"] = canonical_digest(result, digest_field="result_digest")
    write_json(result_path, result)
    return result


__all__ = [
    "AUTHORIZATION_SCHEMA_VERSION",
    "COST_SCOPE_SCHEMA_VERSION",
    "EXACT_FRAME_COUNT",
    "HARD_MAX_ATTEMPTS",
    "REQUEST_SCHEMA_VERSION",
    "RESULT_SCHEMA_VERSION",
    "ControlsVisualSupervisorError",
    "run_controls_visual_supervisor",
]
