"""Trusted reconstruction of worker proof rows from collected leaf evidence.

The host never accepts a worker-supplied ``passed`` boolean, never repairs a
missing or mismatched worker identity, and only counts leaf artifacts whose
bytes, hashes, schemas, identity bindings, and pinned-key attestations all
verify against the immutable attempt identity.
"""

from __future__ import annotations

import base64
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .g1_kitchen_attempt_closure import IDENTITY_FIELDS

SCHEMA_VERSION = "g1_kitchen_trusted_proof_row_validation.v1"
ATTESTATION_PINS_SCHEMA_VERSION = "g1_kitchen_attestation_public_key_pins.v1"
ATTESTATION_PINS_FILE_ENV = "BLUEPRINT_G1_ATTESTATION_PUBLIC_KEY_PINS_FILE"

_TRANSITION_SCHEMAS = (
    "task_transition_measurement.v1",
    "isaac_manipulation_success_evaluator_results.v1",
    "g1_kitchen_terminal_horizon.v1",
)

WORKER_PROOF_ROW_SPECS: dict[str, dict[str, Any]] = {
    "startup": {
        "leaf_schema_versions": ("groot_oscar_same_allocation_startup_gates.v1",),
        "attestation_role": "startup",
    },
    "fast_canary": {
        "leaf_schema_versions": ("isaac_worker_runtime_preflight.v1",),
        "attestation_role": "startup",
    },
    "review_canary": {
        "leaf_schema_versions": ("isaac_review_renderer_canary.v1",),
        "attestation_role": "startup",
    },
    "asset_gate": {
        "leaf_schema_versions": ("kitchen_asset_startup_gate.v1",),
        "attestation_role": "startup",
    },
    "scene_load": {
        "leaf_schema_versions": ("task_transition_measurement.v1",),
        "attestation_role": "task_transition",
    },
    "target": {
        "leaf_schema_versions": ("task_transition_measurement.v1",),
        "attestation_role": "task_transition",
    },
    "stance": {
        "leaf_schema_versions": ("g1_kitchen_live_stance_validation.v1",),
        "attestation_role": "geometry",
    },
    "collision": {
        "leaf_schema_versions": ("g1_kitchen_live_collision_validation.v1",),
        "attestation_role": "geometry",
    },
    "controller_fk": {
        "leaf_schema_versions": (
            "gear_sonic_controller_fk_execution.v1",
            "g1_kitchen_policy_action_sequence.v1",
        ),
        "attestation_roles_by_schema": {
            "gear_sonic_controller_fk_execution.v1": "controller",
            "g1_kitchen_policy_action_sequence.v1": "policy",
        },
    },
    "persistent_simulator_transition": {
        "leaf_schema_versions": _TRANSITION_SCHEMAS,
        "attestation_role": "task_transition",
    },
    "forward_consistency": {
        "leaf_schema_versions": ("strict_action_aware_consistency_contract.v1",),
        "attestation_role": "scorer",
    },
    "inverse_consistency": {
        "leaf_schema_versions": ("strict_action_aware_consistency_contract.v1",),
        "attestation_role": "scorer",
    },
}


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _sequence(value: Any) -> list[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return list(value)
    return []


def load_attestation_pins(path: str | Path) -> dict[str, Any] | None:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    pins = _mapping(value)
    if pins.get("schema_version") != ATTESTATION_PINS_SCHEMA_VERSION:
        return None
    return pins


def _pinned_public_key(
    pins: Mapping[str, Any], *, role: str, fingerprint: str
) -> bytes | None:
    if str(pins.get("algorithm") or "") != "ed25519":
        return None
    allowed = _sequence(_mapping(pins.get("roles")).get(role))
    if fingerprint not in {str(item) for item in allowed}:
        return None
    encoded = _mapping(pins.get("public_keys")).get(fingerprint)
    if not encoded:
        return None
    try:
        raw = base64.b64decode(str(encoded), validate=True)
    except (ValueError, TypeError):
        return None
    if hashlib.sha256(raw).hexdigest() != fingerprint:
        return None
    return raw


def verify_leaf_attestation(
    *,
    data: bytes,
    attestation: Mapping[str, Any],
    expected_role: str,
    pins: Mapping[str, Any] | None,
) -> list[str]:
    if pins is None:
        return ["attestation_public_key_pins_missing"]
    detail = _mapping(attestation)
    if detail.get("algorithm") != "ed25519":
        return ["leaf_artifact_attestation_invalid:algorithm"]
    if str(detail.get("role") or "") != expected_role:
        return ["leaf_artifact_attestation_invalid:role"]
    fingerprint = str(detail.get("public_key_fingerprint") or "").lower()
    raw_key = _pinned_public_key(pins, role=expected_role, fingerprint=fingerprint)
    if raw_key is None:
        return ["leaf_artifact_attestation_invalid:fingerprint_not_pinned"]
    try:
        signature = base64.b64decode(str(detail.get("signature_b64") or ""), validate=True)
    except (ValueError, TypeError):
        return ["leaf_artifact_attestation_invalid:signature_encoding"]
    try:
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    except ImportError:
        return ["leaf_artifact_attestation_invalid:cryptography_unavailable"]
    try:
        Ed25519PublicKey.from_public_bytes(raw_key).verify(signature, data)
    except (InvalidSignature, ValueError):
        return ["leaf_artifact_attestation_invalid:signature"]
    return []


def _identity_blockers(
    prefix: str, observed: Mapping[str, Any], identity: Mapping[str, Any]
) -> list[str]:
    blockers: list[str] = []
    for field in IDENTITY_FIELDS:
        if field not in observed or not str(observed.get(field) or "").strip():
            blockers.append(f"{prefix}_missing:{field}")
        elif str(observed.get(field) or "") != str(identity.get(field) or ""):
            blockers.append(f"{prefix}_mismatch:{field}")
    return blockers


def _validate_leaf(
    *,
    ref: Mapping[str, Any],
    spec: Mapping[str, Any],
    collected_root: Path,
    identity: Mapping[str, Any],
    pins: Mapping[str, Any] | None,
) -> tuple[dict[str, Any] | None, dict[str, Any], list[str]]:
    """Return (payload, verified_ref, blockers) for one declared leaf artifact."""
    detail = _mapping(ref)
    relative = str(detail.get("path") or "")
    blockers: list[str] = []
    verified = {"path": relative, "sha256": None, "schema_version": None}
    if not relative:
        return None, verified, ["leaf_artifact_path_missing"]
    root = collected_root.resolve()
    candidate = (root / relative).resolve()
    if Path(relative).is_absolute() or not candidate.is_relative_to(root):
        return None, verified, [f"leaf_artifact_path_escape:{relative}"]
    if not candidate.is_file():
        return None, verified, [f"leaf_artifact_missing:{relative}"]
    data = candidate.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    verified["sha256"] = digest
    if digest != str(detail.get("sha256") or "").lower():
        blockers.append(f"leaf_artifact_sha256_mismatch:{relative}")
    try:
        declared_size = int(detail.get("size_bytes"))
    except (TypeError, ValueError):
        declared_size = -1
    if declared_size != len(data):
        blockers.append(f"leaf_artifact_size_mismatch:{relative}")
    declared_schema = str(detail.get("schema_version") or "")
    verified["schema_version"] = declared_schema
    if declared_schema not in tuple(spec["leaf_schema_versions"]):
        blockers.append(f"leaf_artifact_schema_unknown:{declared_schema or 'missing'}")
    try:
        payload = _mapping(json.loads(data.decode("utf-8")))
    except (json.JSONDecodeError, UnicodeDecodeError):
        blockers.append(f"leaf_artifact_not_json:{relative}")
        payload = None
    if payload is not None:
        if str(payload.get("schema_version") or "") != declared_schema:
            blockers.append(f"leaf_artifact_schema_version_mismatch:{relative}")
        blockers.extend(
            _identity_blockers(
                "leaf_identity", _mapping(payload.get("identity_binding")), identity
            )
        )
    blockers.extend(
        verify_leaf_attestation(
            data=data,
            attestation=_mapping(detail.get("attestation")),
            expected_role=str(
                _mapping(spec.get("attestation_roles_by_schema")).get(declared_schema)
                or spec.get("attestation_role")
                or ""
            ),
            pins=pins,
        )
    )
    return (payload if not blockers else None), verified, blockers


def _payloads_of(
    payloads: Sequence[Mapping[str, Any]], schema_version: str
) -> list[dict[str, Any]]:
    return [
        dict(payload)
        for payload in payloads
        if str(payload.get("schema_version") or "") == schema_version
    ]


def _measurement_chronology_blockers(
    measurements: Sequence[Mapping[str, Any]],
) -> list[str]:
    if not measurements:
        return ["transition_measurement_leafs_missing"]
    blockers: list[str] = []
    try:
        ordered = sorted(measurements, key=lambda row: int(row.get("source_step_index")))
        indices = [int(row.get("source_step_index")) for row in ordered]
    except (TypeError, ValueError):
        return ["transition_action_chronology_broken:step_indices_invalid"]
    if indices != list(range(len(indices))):
        blockers.append(
            "transition_action_chronology_broken:step_indices_not_contiguous_from_zero"
        )
    previous_after: int | None = None
    for row in ordered:
        try:
            before = int(str(row.get("before_timestamp")))
            after = int(str(row.get("after_timestamp")))
        except (TypeError, ValueError):
            blockers.append("transition_action_chronology_broken:timestamps_invalid")
            continue
        if after <= before:
            blockers.append("transition_action_chronology_broken:step_not_forward_in_time")
        if previous_after is not None and before < previous_after:
            blockers.append("transition_action_chronology_broken:steps_overlap_in_time")
        previous_after = after
    if len(
        {str(row.get("simulator_session_id") or "") for row in ordered}
    ) != 1 or len({str(row.get("stage_id") or "") for row in ordered}) != 1:
        blockers.append("transition_stage_or_session_identity_not_single")
    return sorted(set(blockers))


def _verdict_scene_load(payloads: Sequence[Mapping[str, Any]]) -> list[str]:
    measurements = _payloads_of(payloads, "task_transition_measurement.v1")
    return _measurement_chronology_blockers(measurements)


def _verdict_target(payloads: Sequence[Mapping[str, Any]]) -> list[str]:
    measurements = _payloads_of(payloads, "task_transition_measurement.v1")
    if not measurements:
        return ["transition_measurement_leafs_missing"]
    prims = {str(row.get("articulation_prim_path") or "") for row in measurements}
    blockers: list[str] = []
    if len(prims) != 1:
        blockers.append("target_prim_path_not_single")
    if not all(prim.startswith("/") for prim in prims):
        blockers.append("target_prim_path_not_absolute")
    return blockers


def _verdict_controller_fk(payloads: Sequence[Mapping[str, Any]]) -> list[str]:
    executions = _payloads_of(payloads, "gear_sonic_controller_fk_execution.v1")
    if not executions:
        return ["controller_fk_execution_leafs_missing"]
    policies = _payloads_of(payloads, "g1_kitchen_policy_action_sequence.v1")
    if len(policies) != 1:
        return ["policy_action_sequence_leaf_not_single"]
    blockers: list[str] = []
    for row in executions:
        if row.get("status") != "completed":
            blockers.append("controller_fk_execution_not_completed")
        if row.get("official_controller_action_applied") is not True:
            blockers.append("controller_fk_official_action_not_applied")
        if not str(row.get("source_action_sha256") or ""):
            blockers.append("controller_fk_source_action_sha256_missing")
    return sorted(set(blockers))


def _verdict_transition(payloads: Sequence[Mapping[str, Any]]) -> list[str]:
    measurements = _payloads_of(payloads, "task_transition_measurement.v1")
    blockers = _measurement_chronology_blockers(measurements)
    for measurement in measurements:
        from .task_episode_baseline import verify_task_episode_baseline

        baseline = _mapping(measurement.get("episode_baseline"))
        identity = _mapping(measurement.get("identity_binding"))
        expected = {
            "baseline_digest": measurement.get("episode_baseline_digest"),
            "attempt_id": identity.get("attempt_id"),
            "launch_nonce": identity.get("launch_nonce"),
            "simulator_session_id": measurement.get("simulator_session_id"),
            "stage_id": measurement.get("stage_id"),
            "articulation_prim_path": measurement.get("articulation_prim_path"),
            "task_contract_sha256": identity.get("task_contract_sha256"),
        }
        if not baseline or any(
            str(baseline.get(field) or "") != str(value or "")
            for field, value in expected.items()
        ):
            blockers.append("task_episode_baseline_binding_mismatch")
        blockers.extend(
            verify_task_episode_baseline(
                baseline,
                simulator_session_id=str(measurement.get("simulator_session_id") or ""),
                stage_id=str(measurement.get("stage_id") or ""),
                articulation_prim_path=str(
                    measurement.get("articulation_prim_path") or ""
                ),
                task_contract_sha256=str(identity.get("task_contract_sha256") or ""),
                attempt_id=str(identity.get("attempt_id") or ""),
                launch_nonce=str(identity.get("launch_nonce") or ""),
            )
        )
        if not _mapping(measurement.get("episode_baseline_attestation")):
            blockers.append("task_episode_baseline_attestation_missing")
    judges = _payloads_of(payloads, "isaac_manipulation_success_evaluator_results.v1")
    if len(judges) != 1:
        blockers.append("manipulation_success_judge_leaf_not_single")
    for judge in judges:
        if judge.get("manipulation_success_proven") is not True:
            blockers.append("manipulation_success_not_proven_by_leaf_evidence")
        if judge.get("did_target_manipulation_succeed") is not True:
            blockers.append("target_manipulation_not_proven_by_leaf_evidence")
    horizons = _payloads_of(payloads, "g1_kitchen_terminal_horizon.v1")
    if len(horizons) != 1:
        blockers.append("terminal_horizon_leaf_not_single")
    for horizon in horizons:
        measurements = _payloads_of(payloads, "task_transition_measurement.v1")
        executed = horizon.get("executed_step_count")
        terminal = horizon.get("terminal_step_index")
        if (
            isinstance(executed, bool)
            or not isinstance(executed, int)
            or executed != len(measurements)
            or isinstance(terminal, bool)
            or not isinstance(terminal, int)
            or terminal != executed - 1
        ):
            blockers.append("terminal_horizon_execution_count_mismatch")
        planned = horizon.get("planned_max_steps")
        scenario_count = horizon.get("scenario_count")
        if (
            isinstance(planned, bool)
            or not isinstance(planned, int)
            or not isinstance(executed, int)
            or isinstance(executed, bool)
            or (isinstance(planned, int) and isinstance(executed, int) and planned < executed)
            or isinstance(scenario_count, bool)
            or not isinstance(scenario_count, int)
            or (isinstance(scenario_count, int) and scenario_count < 1)
            or not isinstance(horizon.get("task_completed"), bool)
        ):
            blockers.append("terminal_horizon_contract_fields_invalid")
        if not str(horizon.get("termination_reason") or ""):
            blockers.append("terminal_horizon_termination_reason_missing")
    return sorted(set(blockers))


def _verdict_consistency(field: str):
    def verdict(payloads: Sequence[Mapping[str, Any]]) -> list[str]:
        results = _payloads_of(
            payloads, "strict_action_aware_consistency_contract.v1"
        )
        if not results:
            return ["strict_consistency_result_leafs_missing"]
        if any(row.get(field) is not True for row in results):
            return [f"{field}_not_proven_by_leaf_evidence"]
        return []

    return verdict


def _verdict_strict_true(schema_version: str, *fields: str):
    def verdict(payloads: Sequence[Mapping[str, Any]]) -> list[str]:
        results = _payloads_of(payloads, schema_version)
        if not results:
            return [f"leaf_evidence_missing:{schema_version}"]
        blockers = [
            f"{field}_not_proven_by_leaf_evidence"
            for row in results
            for field in fields
            if row.get(field) is not True
        ]
        return sorted(set(blockers))

    return verdict


def _verdict_status(schema_version: str, expected: str):
    def verdict(payloads: Sequence[Mapping[str, Any]]) -> list[str]:
        results = _payloads_of(payloads, schema_version)
        if len(results) != 1:
            return [f"leaf_evidence_not_single:{schema_version}"]
        return [] if results[0].get("status") == expected else [
            f"leaf_status_not_{expected}:{schema_version}"
        ]

    return verdict


_ROW_VERDICTS = {
    "startup": _verdict_status("groot_oscar_same_allocation_startup_gates.v1", "passed"),
    "fast_canary": _verdict_status("isaac_worker_runtime_preflight.v1", "passed"),
    "review_canary": _verdict_status("isaac_review_renderer_canary.v1", "passed"),
    "asset_gate": _verdict_status("kitchen_asset_startup_gate.v1", "completed"),
    "scene_load": _verdict_scene_load,
    "target": _verdict_target,
    "stance": _verdict_strict_true(
        "g1_kitchen_live_stance_validation.v1",
        "stance_valid",
        "reach_valid",
        "facing_valid",
    ),
    "collision": _verdict_strict_true(
        "g1_kitchen_live_collision_validation.v1",
        "collision_free",
        "clearance_valid",
    ),
    "controller_fk": _verdict_controller_fk,
    "persistent_simulator_transition": _verdict_transition,
    "forward_consistency": _verdict_consistency("forward_consistency_proven"),
    "inverse_consistency": _verdict_consistency("inverse_consistency_proven"),
}


def transition_step_bindings(
    rows: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Project per-step media bindings from the validated transition row."""
    row = _mapping(rows.get("persistent_simulator_transition"))
    if row.get("status") != "passed":
        return []
    payloads = _sequence(_mapping(row.get("evidence")).get("verified_leaf_payloads"))
    measurements = _payloads_of(
        [_mapping(item) for item in payloads], "task_transition_measurement.v1"
    )
    return [
        {
            "step_index": int(row.get("source_step_index") or 0),
            "source_action_sha256": str(row.get("source_action_sha256") or ""),
            "stage_id": str(row.get("stage_id") or ""),
            "simulator_session_id": str(row.get("simulator_session_id") or ""),
            "before_timestamp": str(row.get("before_timestamp") or ""),
            "after_timestamp": str(row.get("after_timestamp") or ""),
        }
        for row in sorted(
            measurements, key=lambda item: int(item.get("source_step_index") or 0)
        )
    ]


def transition_terminal_horizon(
    rows: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any] | None:
    row = _mapping(rows.get("persistent_simulator_transition"))
    if row.get("status") != "passed":
        return None
    payloads = [
        _mapping(item)
        for item in _sequence(_mapping(row.get("evidence")).get("verified_leaf_payloads"))
    ]
    horizons = _payloads_of(payloads, "g1_kitchen_terminal_horizon.v1")
    return horizons[0] if len(horizons) == 1 else None


def _cross_row_blockers(rows: Mapping[str, Mapping[str, Any]]) -> list[str]:
    def payloads(row_id: str, schema: str) -> list[dict[str, Any]]:
        evidence = _mapping(_mapping(rows.get(row_id)).get("evidence"))
        return _payloads_of(
            [_mapping(item) for item in _sequence(evidence.get("verified_leaf_payloads"))],
            schema,
        )

    measurements = payloads(
        "persistent_simulator_transition", "task_transition_measurement.v1"
    )
    controllers = payloads("controller_fk", "gear_sonic_controller_fk_execution.v1")
    expected_actions = [str(row.get("source_action_sha256") or "") for row in measurements]
    controller_actions = [str(row.get("source_action_sha256") or "") for row in controllers]
    blockers: list[str] = []
    if not expected_actions or controller_actions != expected_actions:
        blockers.append("cross_row_action_sequence_mismatch:controller_transition")
    policies = payloads("controller_fk", "g1_kitchen_policy_action_sequence.v1")
    policy_actions = (
        [str(item) for item in _sequence(policies[0].get("source_action_sha256s"))]
        if len(policies) == 1
        else []
    )
    if policy_actions != expected_actions:
        blockers.append("cross_row_action_sequence_mismatch:policy_transition")
    for row_id in ("forward_consistency", "inverse_consistency"):
        scorers = payloads(row_id, "strict_action_aware_consistency_contract.v1")
        if len(scorers) != 1 or [
            str(item) for item in _sequence(scorers[0].get("source_action_sha256s"))
        ] != expected_actions:
            blockers.append(f"cross_row_action_sequence_mismatch:{row_id}")
    baseline_digests = {
        str(row.get("episode_baseline_digest") or "") for row in measurements
    }
    baseline_values = {row.get("episode_initial_value") for row in measurements}
    sessions = {str(row.get("simulator_session_id") or "") for row in measurements}
    stages = {str(row.get("stage_id") or "") for row in measurements}
    if (
        len(baseline_digests) != 1
        or "" in baseline_digests
        or len(baseline_values) != 1
        or len(sessions) != 1
        or "" in sessions
        or len(stages) != 1
        or "" in stages
    ):
        blockers.append("cross_row_episode_baseline_or_runtime_identity_mismatch")
    horizons = payloads(
        "persistent_simulator_transition", "g1_kitchen_terminal_horizon.v1"
    )
    if len(horizons) == 1:
        horizon_actions = [
            str(item) for item in _sequence(horizons[0].get("source_action_sha256s"))
        ]
        if horizon_actions != expected_actions:
            blockers.append("cross_row_action_sequence_mismatch:terminal_horizon")
        if (
            str(horizons[0].get("simulator_session_id") or "") not in sessions
            or str(horizons[0].get("stage_id") or "") not in stages
        ):
            blockers.append("terminal_horizon_runtime_identity_mismatch")
    return sorted(set(blockers))


def validate_worker_proof_rows(
    *,
    worker_rows: Mapping[str, Any],
    worker_manifest_path: str | Path,
    collected_root: str | Path,
    identity: Mapping[str, Any],
    attestation_pins: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Reconstruct every worker proof row from validated leaf artifacts."""
    root = Path(collected_root)
    manifest_path = Path(worker_manifest_path)
    top_blockers: list[str] = []
    manifest_sha256: str | None = None
    if manifest_path.is_file():
        manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    else:
        top_blockers.append("collected_worker_manifest_missing")
    rows: dict[str, dict[str, Any]] = {}
    for row_id, spec in WORKER_PROOF_ROW_SPECS.items():
        raw = _mapping(_mapping(worker_rows).get(row_id))
        blockers: list[str] = list(top_blockers)
        if not raw:
            blockers.append(f"{row_id}:worker_row_missing")
        binding = _mapping(raw.get("identity_binding"))
        blockers.extend(
            _identity_blockers("worker_identity_binding", binding, identity)
        )
        leaf_refs = _sequence(raw.get("leaf_artifacts"))
        if not leaf_refs:
            blockers.append(f"{row_id}:leaf_artifacts_missing")
        payloads: list[dict[str, Any]] = []
        verified_refs: list[dict[str, Any]] = []
        for ref in leaf_refs:
            payload, verified, leaf_blockers = _validate_leaf(
                ref=_mapping(ref),
                spec=spec,
                collected_root=root,
                identity=identity,
                pins=attestation_pins,
            )
            verified_refs.append(verified)
            blockers.extend(leaf_blockers)
            if payload is not None:
                payloads.append(payload)
        blockers.extend(_ROW_VERDICTS[row_id](payloads))
        status = "passed" if not blockers else "blocked"
        if str(raw.get("status") or "") == "passed" and status == "blocked":
            blockers.append(f"{row_id}:worker_status_contradicts_leaf_evidence")
        rows[row_id] = {
            "status": status,
            "identity_binding": binding,
            "blockers": sorted(set(blockers)),
            "evidence": {
                "verified_leaf_artifacts": verified_refs,
                "verified_leaf_payloads": payloads if status == "passed" else [],
                "worker_manifest_sha256": manifest_sha256,
                "worker_manifest_path": str(manifest_path),
                "worker_claimed_status": raw.get("status"),
            },
            "artifact_refs": [
                str(root / str(item.get("path")))
                for item in verified_refs
                if item.get("path")
            ],
        }
    cross_blockers = _cross_row_blockers(rows)
    if cross_blockers:
        for row_id in (
            "controller_fk",
            "persistent_simulator_transition",
            "forward_consistency",
            "inverse_consistency",
        ):
            row = rows[row_id]
            row["status"] = "blocked"
            row["blockers"] = sorted(set([*row["blockers"], *cross_blockers]))
            row["evidence"]["verified_leaf_payloads"] = []
    return {
        "schema_version": SCHEMA_VERSION,
        "rows": rows,
        "worker_manifest_sha256": manifest_sha256,
        "worker_manifest_path": str(manifest_path),
        "blockers": sorted(set([*top_blockers, *cross_blockers])),
    }
