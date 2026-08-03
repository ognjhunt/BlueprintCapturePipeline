"""Evidence-derived five-policy lane for one frozen Franka inspection task.

The module deliberately separates three claims:

* a public checkpoint has an immutable technical identity;
* that checkpoint is admitted for this exact contract; and
* a learned action was freshly queried and executed in the simulator.

Hermetic clients and runtimes can exercise the mechanics, but their receipts are
marked as fixtures and cannot authorize the real five-policy fleet.
"""

from __future__ import annotations

import math
import time
import uuid
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any, Protocol

from .decision_evidence_contracts import canonical_digest
from .droid_policy_bridge import (
    DROID_CONTROL_HZ,
    droid_action_to_mujoco_targets,
    validate_droid_action_chunk,
    validate_droid_observation,
)


CONTRACT_SCHEMA_VERSION = "franka_inspection_policy_contract.v1"
AUDIT_SCHEMA_VERSION = "learned_policy_contract_audit.v1"
IDENTITY_SCHEMA_VERSION = "learned_policy_candidate_identity.v1"
QUERY_RECEIPT_SCHEMA_VERSION = "learned_policy_identity_query_receipt.v1"
AUTHORIZATION_SCHEMA_VERSION = "new_site_policy_execution_authorization.v1"
ATTEMPT_SCHEMA_VERSION = "learned_policy_attempt_receipt.v1"
BUNDLE_SCHEMA_VERSION = "learned_policy_execution_bundle.v1"
TERMINAL_SCHEMA_VERSION = "five_policy_admission_terminal.v1"
EXPECTED_POLICY_COUNT = 5
OPENPI_SOURCE_REVISION = "15a9616a00943ada6c20a0f158e3adb39df2ccac"


class LearnedPolicyLaneError(ValueError):
    """Stable fail-closed validation error."""

    def __init__(self, codes: Sequence[str]) -> None:
        self.codes = tuple(sorted(set(str(code) for code in codes if str(code))))
        super().__init__(";".join(self.codes))


class LearnedPolicyClient(Protocol):
    """Provider-neutral learned-policy client used by the execution adapter."""

    fixture_or_fake: bool
    policy_identity_digest: str

    def infer(self, observation: Mapping[str, Any]) -> Mapping[str, Any]: ...


class IdentityBoundLearnedPolicyAdapter:
    """Bind any DROID-compatible backend to one immutable policy identity."""

    def __init__(
        self,
        *,
        backend: Any,
        identity: Mapping[str, Any],
        fixture_or_fake: bool = False,
    ) -> None:
        if not callable(getattr(backend, "infer", None)):
            raise LearnedPolicyLaneError(["policy_backend_infer_missing"])
        _validate_candidate_identity(identity)
        self._backend = backend
        self.policy_identity_digest = str(identity["policy_identity_digest"])
        self.fixture_or_fake = bool(fixture_or_fake)

    def infer(self, observation: Mapping[str, Any]) -> Mapping[str, Any]:
        """Forward the exact frozen observation without provider-specific fields."""

        output = self._backend.infer(dict(observation))
        if not isinstance(output, Mapping):
            raise LearnedPolicyLaneError(["native_policy_output_invalid"])
        return output


class FrankaInspectionSimulator(Protocol):
    """Minimal simulator boundary required to produce a complete receipt."""

    fixture_or_fake: bool

    def reset(self, reset_contract: Mapping[str, Any]) -> Mapping[str, Any]: ...

    def observe(self) -> Mapping[str, Any]: ...

    def apply_action(self, action: Mapping[str, Any]) -> Mapping[str, Any]: ...

    def terminal(self) -> bool: ...

    def task_metric(self) -> Mapping[str, Any]: ...


def _clone(value: Any) -> Any:
    import json

    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise LearnedPolicyLaneError(["value_not_json_serializable"]) from exc


def _digest(value: Mapping[str, Any], *, field: str | None = None) -> str:
    return canonical_digest(value, digest_field=field) if field else canonical_digest(value)


def _is_digest(value: Any) -> bool:
    text = str(value or "")
    return (
        len(text) == 71
        and text.startswith("sha256:")
        and all(character in "0123456789abcdef" for character in text[7:])
    )


def frozen_franka_inspection_contract() -> dict[str, Any]:
    """Return the one non-negotiable contract used for all five policies."""

    embodiment = {
        "embodiment_id": "franka_panda_droid_v1",
        "robot_id": "franka_panda",
        "arm_joint_order": [f"panda_joint{index}" for index in range(1, 8)],
        "arm_joint_limits_rad": [
            [-2.8973, 2.8973],
            [-1.7628, 1.7628],
            [-2.8973, 2.8973],
            [-3.0718, -0.0698],
            [-2.8973, 2.8973],
            [-0.0175, 3.7525],
            [-2.8973, 2.8973],
        ],
        "gripper": "parallel_panda_fingers",
        "arm_dof": 7,
        "policy_state_dimension": 8,
    }
    observation = {
        "exterior_rgb": {
            "key": "observation/exterior_image_1_left",
            "shape": [224, 224, 3],
            "dtype": "uint8",
            "camera_role": "fixed_external",
        },
        "wrist_rgb": {
            "key": "observation/wrist_image_left",
            "shape": [224, 224, 3],
            "dtype": "uint8",
            "camera_role": "franka_hand_mounted",
        },
        "joint_position": {
            "key": "observation/joint_position",
            "shape": [7],
            "units": "radians",
        },
        "gripper_position": {
            "key": "observation/gripper_position",
            "shape": [1],
            "semantics": "normalized_0_open_1_closed",
        },
        "prompt": "Inspect the marked work-surface region and keep it visible in the wrist camera.",
    }
    observation_sequence = {
        "initial": "post_settle_matched_reset_observation",
        "per_control_step": [
            "capture_live_observation",
            "issue_fresh_identity_bound_policy_query",
            "retain_complete_native_action_chunk",
            "normalize_exactly_native_row_zero",
            "apply_one_simulator_action",
            "retain_contacts_collisions_and_next_observation",
        ],
        "terminal": "post_action_terminal_observation",
        "recorded_action_replay_allowed": False,
        "wam_or_generated_video_is_policy_execution": False,
    }
    action = {
        "native": {
            "space": "droid_normalized_joint_delta_plus_gripper",
            "dimensions": 8,
            "allowed_chunk_rows": [10, 15],
            "complete_chunk_retained": True,
        },
        "normalized": {
            "arm": "clip_-1_1_then_delta_rad_equals_value_times_0.2",
            "gripper": "value_gt_0.5_means_closed_0.0m_else_open_0.04m",
            "joint_limit_behavior": "clip_to_frozen_franka_limits_and_record_clamp",
        },
        "execution": "query_every_step_and_execute_native_row_zero_only",
        "control_hz": DROID_CONTROL_HZ,
        "open_loop_action_reuse": False,
    }
    reset = {
        "required_fields": [
            "scene_digest",
            "placement_digest",
            "routing_decision_digest",
            "target_binding_digest",
            "random_seed",
            "joint_position_rad",
            "gripper_position",
            "external_camera_extrinsics_digest",
            "wrist_camera_extrinsics_digest",
            "target_state_digest",
        ],
        "same_digest_for_all_candidates": True,
        "post_settle_joint_velocity_max_rad_s": 0.01,
    }
    metric = {
        "schema_version": "task_outcome_metric_spec.v1",
        "metric_id": "franka_marked_surface_inspection_coverage",
        "units": "fraction",
        "direction": "maximize",
        "range": [0.0, 1.0],
        "measurement": (
            "fraction_of_frozen_target_surface_samples_visible_in_live_wrist_rgb_with_"
            "qualified_depth_and_occlusion"
        ),
        "terminal_contact_and_collision_evidence_required": True,
        "fixed_before_execution": True,
    }
    runtime = {
        "policy_interface": "provider_neutral_identity_bound_infer_v1",
        "simulator_interface": "franka_inspection_simulator_runtime_v1",
        "fresh_query_per_control_step": True,
        "native_output_retained": True,
        "exact_conversion": "blueprint_pipeline.droid_policy_bridge",
    }
    contract: dict[str, Any] = {
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "task_id": "franka_marked_work_surface_inspection_v1",
        "embodiment": embodiment,
        "observation_schema": observation,
        "observation_sequence": observation_sequence,
        "action_schema": action,
        "control_frequency_hz": DROID_CONTROL_HZ,
        "reset_contract": reset,
        "task_metric": metric,
        "runtime_interface": runtime,
    }
    contract["embodiment_digest"] = _digest(embodiment)
    contract["observation_schema_digest"] = _digest(observation)
    contract["observation_sequence_spec_digest"] = _digest(observation_sequence)
    contract["action_schema_digest"] = _digest(action)
    contract["reset_contract_digest"] = _digest(reset)
    contract["metric_contract_digest"] = _digest(metric)
    contract["runtime_interface_digest"] = _digest(runtime)
    contract["contract_digest"] = _digest(contract, field="contract_digest")
    return contract


# These are five different public checkpoint object sets, not aliases, seeds,
# prompts, or repeated calls to one endpoint. The generation digests cover the
# complete GCS object metadata inventory observed on 2026-08-03.
OFFICIAL_OPENPI_DROID_BASELINES: tuple[dict[str, Any], ...] = (
    {
        "candidate_id": "paligemma_binning_droid",
        "checkpoint_uri": "gs://openpi-assets/checkpoints/roboarena/paligemma_binning_droid",
        "checkpoint_generation_manifest_sha256": "ae42d454bebb66b0084d72f43bceb4a0f55c637148f99134c3d3bad255c97dc1",
        "checkpoint_object_count": 19,
        "checkpoint_size_bytes": 10849851589,
        "native_action_chunk_rows": 15,
    },
    {
        "candidate_id": "paligemma_fast_droid",
        "checkpoint_uri": "gs://openpi-assets/checkpoints/roboarena/paligemma_fast_droid",
        "checkpoint_generation_manifest_sha256": "d7e0f2e9671576d7904135e39034bf45dd7f11f3ae8f20754395633f9d8df140",
        "checkpoint_object_count": 19,
        "checkpoint_size_bytes": 10850689373,
        "native_action_chunk_rows": 15,
    },
    {
        "candidate_id": "paligemma_fast_specialist_droid",
        "checkpoint_uri": "gs://openpi-assets/checkpoints/roboarena/paligemma_fast_specialist_droid",
        "checkpoint_generation_manifest_sha256": "3b0e00fdb6e681e27bd59e3259b3b89577ec983b9ee5e3a527ed92e729a60808",
        "checkpoint_object_count": 19,
        "checkpoint_size_bytes": 10850020059,
        "native_action_chunk_rows": 15,
    },
    {
        "candidate_id": "paligemma_vq_droid",
        "checkpoint_uri": "gs://openpi-assets/checkpoints/roboarena/paligemma_vq_droid",
        "checkpoint_generation_manifest_sha256": "90e80a19ae9b579d03823835549fa3a8267f8916b520a862cbe8305e76a868e9",
        "checkpoint_object_count": 18,
        "checkpoint_size_bytes": 10850197563,
        "native_action_chunk_rows": 15,
    },
    {
        "candidate_id": "paligemma_diffusion_droid",
        "checkpoint_uri": "gs://openpi-assets/checkpoints/roboarena/paligemma_diffusion_droid",
        "checkpoint_generation_manifest_sha256": "c6a6a08597ec496bc0cf4cedbb23fb0cf87365f88f9f16190a3dd9e2f7e2802e",
        "checkpoint_object_count": 19,
        "checkpoint_size_bytes": 12007426260,
        "native_action_chunk_rows": 10,
    },
)


def audit_candidate(
    candidate: Mapping[str, Any],
    *,
    contract: Mapping[str, Any],
    checkpoint_rights: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Audit one real checkpoint against the exact frozen contract."""

    row = _clone(dict(candidate))
    blockers: list[str] = []
    if contract.get("schema_version") != CONTRACT_SCHEMA_VERSION:
        blockers.append("frozen_contract_schema_invalid")
    if row.get("openpi_source_revision", OPENPI_SOURCE_REVISION) != OPENPI_SOURCE_REVISION:
        blockers.append("openpi_source_revision_mismatch")
    if not str(row.get("checkpoint_uri") or "").startswith(
        "gs://openpi-assets/checkpoints/roboarena/"
    ):
        blockers.append("checkpoint_uri_not_official_roboarena_droid")
    generation = str(row.get("checkpoint_generation_manifest_sha256") or "")
    if len(generation) != 64 or any(char not in "0123456789abcdef" for char in generation):
        blockers.append("checkpoint_generation_identity_invalid")
    if row.get("native_action_chunk_rows") not in {10, 15}:
        blockers.append("native_action_horizon_incompatible")
    if row.get("embodiment_id", "franka_panda_droid_v1") != "franka_panda_droid_v1":
        blockers.append("policy_robot_mismatch")
    rights = dict(checkpoint_rights) if isinstance(checkpoint_rights, Mapping) else {}
    if (
        rights.get("status") != "admitted_internal_execution"
        or not _is_digest(rights.get("rights_evidence_digest"))
        or not str(rights.get("grant_scope") or "").strip()
    ):
        blockers.append("checkpoint_specific_rights_missing")
    audit: dict[str, Any] = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "candidate_id": row.get("candidate_id"),
        "status": "admitted" if not blockers else "blocked",
        "checkpoint_uri": row.get("checkpoint_uri"),
        "checkpoint_generation_manifest_sha256": generation,
        "checkpoint_object_count": row.get("checkpoint_object_count"),
        "checkpoint_size_bytes": row.get("checkpoint_size_bytes"),
        "openpi_source_revision": OPENPI_SOURCE_REVISION,
        "embodiment_digest": contract.get("embodiment_digest"),
        "observation_schema_digest": contract.get("observation_schema_digest"),
        "observation_sequence_spec_digest": contract.get(
            "observation_sequence_spec_digest"
        ),
        "action_schema_digest": contract.get("action_schema_digest"),
        "runtime_interface_digest": contract.get("runtime_interface_digest"),
        "native_action_chunk_rows": row.get("native_action_chunk_rows"),
        "checkpoint_rights": rights or {"status": "unresolved"},
        "audit_axes": {
            "immutable_identity": {
                "status": "verified_generation_manifest",
                "checkpoint_generation_manifest_sha256": generation,
                "object_count": row.get("checkpoint_object_count"),
                "size_bytes": row.get("checkpoint_size_bytes"),
            },
            "provenance": {
                "status": "official_openpi_roboarena_object_set",
                "source_revision": OPENPI_SOURCE_REVISION,
                "checkpoint_uri": row.get("checkpoint_uri"),
            },
            "embodiment": {
                "status": "contract_compatible",
                "embodiment_digest": contract.get("embodiment_digest"),
            },
            "observation": {
                "status": "contract_compatible",
                "observation_schema_digest": contract.get("observation_schema_digest"),
                "observation_sequence_spec_digest": contract.get(
                    "observation_sequence_spec_digest"
                ),
            },
            "action": {
                "status": "contract_compatible",
                "action_schema_digest": contract.get("action_schema_digest"),
                "native_action_chunk_rows": row.get("native_action_chunk_rows"),
            },
            "rights_and_license": {
                "status": (
                    "admitted_internal_execution"
                    if "checkpoint_specific_rights_missing" not in blockers
                    else "checkpoint_specific_grant_unresolved"
                ),
                "evidence_digest": rights.get("rights_evidence_digest"),
                "grant_scope": rights.get("grant_scope"),
            },
            "hostability": {
                "status": "not_materialized_rights_gate_precedes_spend",
                "public_checkpoint_bytes": row.get("checkpoint_size_bytes"),
                "paid_runtime_test_performed": False,
            },
            "runtime_dependencies": {
                "status": "official_source_revision_pinned_not_materialized",
                "runtime_interface_digest": contract.get("runtime_interface_digest"),
                "source_revision": OPENPI_SOURCE_REVISION,
            },
        },
        "blockers": sorted(set(blockers)),
        "claim_boundary": {
            "technical_identity_frozen": not any(
                code.endswith("invalid") or code.endswith("mismatch") for code in blockers
            ),
            "checkpoint_execution_admitted": not blockers,
            "identity_bound_query_observed": False,
            "matched_reset_attempt_observed": False,
        },
    }
    audit["audit_digest"] = _digest(audit, field="audit_digest")
    return audit


def candidate_identity_from_audit(audit: Mapping[str, Any]) -> dict[str, Any]:
    """Emit an identity artifact without upgrading a blocked audit."""

    if audit.get("schema_version") != AUDIT_SCHEMA_VERSION:
        raise LearnedPolicyLaneError(["candidate_audit_schema_invalid"])
    if audit.get("audit_digest") != _digest(dict(audit), field="audit_digest"):
        raise LearnedPolicyLaneError(["candidate_audit_digest_mismatch"])
    checkpoint_digest = "sha256:" + str(audit["checkpoint_generation_manifest_sha256"])
    rights = dict(audit.get("checkpoint_rights") or {})
    identity: dict[str, Any] = {
        "schema_version": IDENTITY_SCHEMA_VERSION,
        "candidate_id": audit.get("candidate_id"),
        "candidate_kind": "learned_policy",
        "admission_status": audit.get("status"),
        "checkpoint_uri": audit.get("checkpoint_uri"),
        "checkpoint_digest": checkpoint_digest,
        "endpoint_identity_digest": None,
        "runtime_digest": audit.get("runtime_interface_digest"),
        "observation_schema_digest": audit.get("observation_schema_digest"),
        "observation_sequence_spec_digest": audit.get(
            "observation_sequence_spec_digest"
        ),
        "action_schema_digest": audit.get("action_schema_digest"),
        "embodiment_digest": audit.get("embodiment_digest"),
        "openpi_source_revision": audit.get("openpi_source_revision"),
        "native_action_chunk_rows": audit.get("native_action_chunk_rows"),
        "checkpoint_rights_status": rights.get("status", "unresolved"),
        "checkpoint_rights_evidence_digest": rights.get("rights_evidence_digest"),
        "contract_audit_digest": audit.get("audit_digest"),
    }
    identity["policy_identity_digest"] = _digest(identity, field="policy_identity_digest")
    return identity


def candidate_set_digest(identities: Sequence[Mapping[str, Any]]) -> str:
    return _digest(
        {
            "policy_identity_digests": sorted(
                str(row.get("policy_identity_digest") or "") for row in identities
            )
        }
    )


def _validate_candidate_identity(identity: Mapping[str, Any]) -> None:
    blockers: list[str] = []
    if identity.get("schema_version") != IDENTITY_SCHEMA_VERSION:
        blockers.append("candidate_identity_schema_invalid")
    if identity.get("policy_identity_digest") != _digest(
        dict(identity), field="policy_identity_digest"
    ):
        blockers.append("candidate_identity_digest_mismatch")
    if not _is_digest(identity.get("checkpoint_digest")):
        blockers.append("candidate_checkpoint_digest_invalid")
    if identity.get("endpoint_identity_digest") is not None:
        blockers.append("candidate_endpoint_identity_unexpected")
    for field in (
        "runtime_digest",
        "observation_schema_digest",
        "observation_sequence_spec_digest",
        "action_schema_digest",
        "embodiment_digest",
        "contract_audit_digest",
    ):
        if not _is_digest(identity.get(field)):
            blockers.append(f"candidate_{field}_invalid")
    if identity.get("native_action_chunk_rows") not in {10, 15}:
        blockers.append("candidate_native_action_horizon_invalid")
    if blockers:
        raise LearnedPolicyLaneError(blockers)


def build_candidate_packet(
    *, checkpoint_rights_by_candidate: Mapping[str, Mapping[str, Any]] | None = None
) -> dict[str, Any]:
    contract = frozen_franka_inspection_contract()
    rights_by_candidate = dict(checkpoint_rights_by_candidate or {})
    audits = [
        audit_candidate(
            {**candidate, "openpi_source_revision": OPENPI_SOURCE_REVISION},
            contract=contract,
            checkpoint_rights=rights_by_candidate.get(str(candidate["candidate_id"])),
        )
        for candidate in OFFICIAL_OPENPI_DROID_BASELINES
    ]
    identities = [candidate_identity_from_audit(row) for row in audits]
    packet: dict[str, Any] = {
        "schema_version": "learned_policy_candidate_set.v1",
        "contract": contract,
        "policy_candidates": identities,
        "candidate_audits": audits,
        "candidate_set_digest": candidate_set_digest(identities),
        "candidate_count": len(identities),
        "admitted_candidate_count": sum(row["status"] == "admitted" for row in audits),
    }
    packet["candidate_packet_digest"] = _digest(packet, field="candidate_packet_digest")
    return packet


def _observation_evidence(observation: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    import hashlib
    import numpy as np

    blockers = validate_droid_observation(observation)
    if blockers:
        raise LearnedPolicyLaneError([f"observation_action_mismatch:{row}" for row in blockers])
    summary: dict[str, Any] = {
        "prompt": str(observation["prompt"]),
        "joint_position": [
            float(value) for value in np.asarray(observation["observation/joint_position"])
        ],
        "gripper_position": [
            float(value) for value in np.asarray(observation["observation/gripper_position"])
        ],
        "images": {},
    }
    for key in ("observation/exterior_image_1_left", "observation/wrist_image_left"):
        image = np.asarray(observation[key])
        summary["images"][key] = {
            "shape": list(image.shape),
            "dtype": str(image.dtype),
            "content_sha256": "sha256:" + hashlib.sha256(image.tobytes(order="C")).hexdigest(),
            "byte_count": int(image.nbytes),
        }
    evidence = {**summary, "observation_digest": _digest(summary)}
    runtime_observation = dict(observation)
    return evidence, runtime_observation


def _native_action_evidence(actions: Any, *, expected_rows: int) -> tuple[dict[str, Any], Any]:
    import numpy as np

    blockers = validate_droid_action_chunk(actions, expected_rows=expected_rows)
    if blockers:
        raise LearnedPolicyLaneError([f"observation_action_mismatch:{row}" for row in blockers])
    chunk = np.asarray(actions, dtype=float)
    rows = [[float(value) for value in row] for row in chunk]
    evidence = {
        "shape": [expected_rows, 8],
        "rows": rows,
        "native_output_digest": _digest({"shape": [expected_rows, 8], "rows": rows}),
    }
    return evidence, chunk


def build_identity_query_receipt(
    *,
    identity: Mapping[str, Any],
    observation: Mapping[str, Any],
    policy_client: LearnedPolicyClient,
) -> dict[str, Any]:
    """Perform one identity-bound query; fixtures remain explicitly ineligible."""

    observation_evidence, runtime_observation = _observation_evidence(observation)
    if policy_client.policy_identity_digest != identity.get("policy_identity_digest"):
        raise LearnedPolicyLaneError(["policy_runtime_identity_mismatch"])
    started_ns = time.time_ns()
    output = policy_client.infer(runtime_observation)
    ended_ns = time.time_ns()
    if not isinstance(output, Mapping) or "actions" not in output:
        raise LearnedPolicyLaneError(["native_policy_output_missing"])
    native, _ = _native_action_evidence(
        output["actions"], expected_rows=int(identity["native_action_chunk_rows"])
    )
    receipt: dict[str, Any] = {
        "schema_version": QUERY_RECEIPT_SCHEMA_VERSION,
        "candidate_id": identity.get("candidate_id"),
        "policy_identity_digest": identity.get("policy_identity_digest"),
        "checkpoint_digest": identity.get("checkpoint_digest"),
        "observation": observation_evidence,
        "native_policy_output": native,
        "query_started_at_ns": started_ns,
        "query_ended_at_ns": ended_ns,
        "fresh_infer_call_observed": True,
        "fixture_or_fake": bool(policy_client.fixture_or_fake),
        "eligible_for_real_fleet_admission": not bool(policy_client.fixture_or_fake),
    }
    receipt["query_receipt_digest"] = _digest(receipt, field="query_receipt_digest")
    return receipt


def validate_identity_query_receipt(
    receipt: Mapping[str, Any], *, identity: Mapping[str, Any]
) -> list[str]:
    """Recompute the retained query evidence and its immutable identity binding."""

    blockers: list[str] = []
    try:
        _validate_candidate_identity(identity)
    except LearnedPolicyLaneError as exc:
        blockers.extend(exc.codes)
    if receipt.get("schema_version") != QUERY_RECEIPT_SCHEMA_VERSION:
        blockers.append("identity_query_receipt_schema_invalid")
    if receipt.get("query_receipt_digest") != _digest(
        dict(receipt), field="query_receipt_digest"
    ):
        blockers.append("identity_query_receipt_digest_mismatch")
    for field in ("candidate_id", "policy_identity_digest", "checkpoint_digest"):
        if receipt.get(field) != identity.get(field):
            blockers.append(f"identity_query_receipt_{field}_mismatch")
    observation = dict(receipt.get("observation") or {})
    observation_core = {
        key: observation.get(key)
        for key in ("prompt", "joint_position", "gripper_position", "images")
    }
    if observation.get("observation_digest") != _digest(observation_core):
        blockers.append("identity_query_observation_digest_mismatch")
    native = dict(receipt.get("native_policy_output") or {})
    native_core = {"shape": native.get("shape"), "rows": native.get("rows")}
    if native.get("native_output_digest") != _digest(native_core):
        blockers.append("identity_query_native_output_digest_mismatch")
    try:
        native_errors = validate_droid_action_chunk(
            native.get("rows"),
            expected_rows=int(identity.get("native_action_chunk_rows")),
        )
    except (TypeError, ValueError):
        native_errors = ["native_policy_output_invalid"]
    blockers.extend(f"identity_query:{code}" for code in native_errors)
    started = receipt.get("query_started_at_ns")
    ended = receipt.get("query_ended_at_ns")
    if (
        isinstance(started, bool)
        or not isinstance(started, int)
        or isinstance(ended, bool)
        or not isinstance(ended, int)
        or ended < started
    ):
        blockers.append("identity_query_timing_invalid")
    return sorted(set(blockers))


def build_execution_authorization(
    *,
    candidate_packet: Mapping[str, Any],
    query_receipts: Sequence[Mapping[str, Any]],
    routing_decision_digest: str,
    placement_digest: str,
    metric_spec_digest: str,
    matched_reset_digest: str,
) -> dict[str, Any]:
    """Authorize only five rights-cleared identities with five real queries."""

    if candidate_packet.get("candidate_packet_digest") != _digest(
        dict(candidate_packet), field="candidate_packet_digest"
    ):
        raise LearnedPolicyLaneError(["candidate_packet_digest_mismatch"])
    identities = list(candidate_packet.get("policy_candidates") or [])
    audits = list(candidate_packet.get("candidate_audits") or [])
    blockers: list[str] = []
    if len(identities) != EXPECTED_POLICY_COUNT or len(audits) != EXPECTED_POLICY_COUNT:
        blockers.append("exactly_five_learned_policy_candidates_required")
    if any(row.get("status") != "admitted" for row in audits if isinstance(row, Mapping)):
        blockers.append("candidate_contract_or_rights_admission_incomplete")
    identity_by_digest: dict[str, Mapping[str, Any]] = {}
    for identity in identities:
        try:
            _validate_candidate_identity(identity)
        except LearnedPolicyLaneError as exc:
            blockers.extend(exc.codes)
        digest = str(identity.get("policy_identity_digest") or "")
        if digest in identity_by_digest:
            blockers.append("candidate_identity_duplicate")
        identity_by_digest[digest] = identity
    if candidate_packet.get("candidate_set_digest") != candidate_set_digest(identities):
        blockers.append("candidate_set_digest_mismatch")
    query_by_identity: dict[str, Mapping[str, Any]] = {}
    for receipt in query_receipts:
        if not isinstance(receipt, Mapping):
            blockers.append("identity_query_receipt_invalid")
            continue
        if (
            receipt.get("fixture_or_fake") is not False
            or receipt.get("eligible_for_real_fleet_admission") is not True
            or receipt.get("fresh_infer_call_observed") is not True
        ):
            blockers.append("real_identity_bound_query_missing")
        digest = str(receipt.get("policy_identity_digest") or "")
        identity = identity_by_digest.get(digest)
        if identity is None:
            blockers.append("identity_query_candidate_set_mismatch")
        else:
            blockers.extend(
                validate_identity_query_receipt(receipt, identity=identity)
            )
        if digest in query_by_identity:
            blockers.append("identity_query_receipt_duplicate")
        query_by_identity[digest] = receipt
    expected_identities = {str(row.get("policy_identity_digest") or "") for row in identities}
    if set(query_by_identity) != expected_identities:
        blockers.append("identity_query_candidate_set_mismatch")
    for field, value in (
        ("routing_decision_digest", routing_decision_digest),
        ("placement_digest", placement_digest),
        ("metric_spec_digest", metric_spec_digest),
        ("matched_reset_digest", matched_reset_digest),
    ):
        if not _is_digest(value):
            blockers.append(f"{field}_invalid")
    authorization: dict[str, Any] = {
        "schema_version": AUTHORIZATION_SCHEMA_VERSION,
        "policy_execution_authorized": not blockers,
        "physical_robot_execution_authorized": False,
        "routing_decision_digest": routing_decision_digest,
        "placement_digest": placement_digest,
        "metric_spec_digest": metric_spec_digest,
        "matched_reset_digest": matched_reset_digest,
        "candidate_set_digest": candidate_packet.get("candidate_set_digest"),
        "candidate_identities": [
            {
                "candidate_id": row.get("candidate_id"),
                "checkpoint_uri": row.get("checkpoint_uri"),
                "checkpoint_digest": row.get("checkpoint_digest"),
                "policy_identity_digest": row.get("policy_identity_digest"),
                "admission_status": row.get("admission_status"),
                "contract_audit_digest": row.get("contract_audit_digest"),
            }
            for row in candidate_packet.get("policy_candidates", [])
        ],
        "identity_query_receipt_digests": sorted(
            str(row.get("query_receipt_digest") or "") for row in query_receipts
        ),
        "blockers": sorted(set(blockers)),
        "paid_execution_authorized": False,
        "required_next_authority": (
            "Obtain an attributable checkpoint-specific execution grant for each selected "
            "checkpoint. Then obtain a separate explicit GPU budget and TTL authorization "
            "before using paid_resource_allocator for five identity-bound queries and five "
            "matched-reset attempts."
        ),
    }
    authorization["authorization_digest"] = _digest(
        authorization, field="authorization_digest"
    )
    return authorization


def _validate_reset_contract(reset: Mapping[str, Any], *, contract: Mapping[str, Any]) -> str:
    required = list(dict(contract["reset_contract"])["required_fields"])
    missing = [field for field in required if field not in reset]
    if missing:
        raise LearnedPolicyLaneError([f"reset_contract_missing:{field}" for field in missing])
    joint = reset.get("joint_position_rad")
    if not isinstance(joint, list) or len(joint) != 7:
        raise LearnedPolicyLaneError(["reset_joint_position_invalid"])
    digest = _digest(dict(reset))
    return digest


def _build_attempt_receipt(
    *,
    identity: Mapping[str, Any],
    authorization: Mapping[str, Any],
    reset_digest: str,
    started_ns: int,
    ended_ns: int,
    observations: list[dict[str, Any]],
    action_trace: list[dict[str, Any]],
    contacts: list[dict[str, Any]],
    collisions: list[dict[str, Any]],
    metric: Mapping[str, Any],
    fixture_or_fake: bool,
) -> dict[str, Any]:
    metric_result = _clone(dict(metric))
    value = metric_result.get("value")
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise LearnedPolicyLaneError(["task_metric_value_invalid"])
    if not 0.0 <= float(value) <= 1.0:
        raise LearnedPolicyLaneError(["task_metric_value_out_of_range"])
    if len(observations) != len(action_trace) + 1:
        raise LearnedPolicyLaneError(["observation_action_trace_length_mismatch"])
    evidence = {
        "initial_observation": observations[0],
        "observation_trace": observations,
        "action_trace": action_trace,
        "contact_trace": contacts,
        "collision_trace": collisions,
        "terminal_observation": observations[-1],
        "task_metric": metric_result,
    }
    observation_trace_digest = _digest({"rows": observations})
    action_trace_digest = _digest({"rows": action_trace})
    contact_digest = _digest({"rows": contacts})
    collision_digest = _digest({"rows": collisions})
    execution_core = {
        "policy_identity_digest": identity["policy_identity_digest"],
        "matched_reset_digest": reset_digest,
        "observation_trace_digest": observation_trace_digest,
        "action_trace_digest": action_trace_digest,
        "contact_evidence_digest": contact_digest,
        "collision_evidence_digest": collision_digest,
        "terminal_observation_digest": observations[-1]["observation_digest"],
        "task_metric_evidence_digest": _digest(metric_result),
    }
    receipt: dict[str, Any] = {
        "schema_version": ATTEMPT_SCHEMA_VERSION,
        "candidate_id": identity["candidate_id"],
        "policy_identity_digest": identity["policy_identity_digest"],
        "status": "fixture_completed" if fixture_or_fake else "completed",
        "action_source": "learned_policy",
        "routing_decision_digest": authorization["routing_decision_digest"],
        "placement_digest": authorization["placement_digest"],
        "matched_reset_digest": reset_digest,
        "initial_state_observation_digest": observations[0]["observation_digest"],
        "observation_trace_digest": observation_trace_digest,
        "action_trace_digest": action_trace_digest,
        "contact_evidence_digest": contact_digest,
        "collision_evidence_digest": collision_digest,
        "terminal_observation_digest": observations[-1]["observation_digest"],
        "native_policy_output_trace_digest": _digest(
            {"rows": [row["native_policy_output"] for row in action_trace]}
        ),
        "normalized_action_trace_digest": _digest(
            {"rows": [row["normalized_action"] for row in action_trace]}
        ),
        "simulator_action_trace_digest": _digest(
            {"rows": [row["simulator_action"] for row in action_trace]}
        ),
        "fresh_policy_query_count": len(action_trace),
        "learned_policy_action_count": len(action_trace),
        "control_step_count": len(action_trace),
        "fresh_query_for_every_control_step": True,
        "learned_policy_action_proven": not fixture_or_fake,
        "reset_observed": True,
        "started_at_ns": started_ns,
        "ended_at_ns": ended_ns,
        "started_at": datetime.fromtimestamp(started_ns / 1_000_000_000, UTC).isoformat(),
        "ended_at": datetime.fromtimestamp(ended_ns / 1_000_000_000, UTC).isoformat(),
        "fixture_or_fake": fixture_or_fake,
        "task_metric_result": metric_result,
        "evidence": evidence,
        "execution_receipt_digest": _digest(execution_core),
    }
    receipt["attempt_digest"] = _digest(receipt, field="attempt_digest")
    return receipt


def execute_learned_policy_attempt(
    *,
    identity: Mapping[str, Any],
    authorization: Mapping[str, Any],
    reset_contract: Mapping[str, Any],
    policy_client: LearnedPolicyClient,
    simulator: FrankaInspectionSimulator,
    max_control_steps: int,
    joint_limits: Sequence[Sequence[float]],
) -> dict[str, Any]:
    """Execute a fresh learned query at every simulator control step."""

    contract = frozen_franka_inspection_contract()
    if authorization.get("policy_execution_authorized") is not True:
        raise LearnedPolicyLaneError(["policy_execution_not_authorized"])
    if policy_client.policy_identity_digest != identity.get("policy_identity_digest"):
        raise LearnedPolicyLaneError(["policy_runtime_identity_mismatch"])
    if max_control_steps <= 0:
        raise LearnedPolicyLaneError(["max_control_steps_invalid"])
    frozen_joint_limits = list(dict(contract["embodiment"])["arm_joint_limits_rad"])
    if _clone(list(joint_limits)) != frozen_joint_limits:
        raise LearnedPolicyLaneError(["franka_joint_limits_contract_mismatch"])
    reset_digest = _validate_reset_contract(reset_contract, contract=contract)
    if authorization.get("matched_reset_digest") != reset_digest:
        raise LearnedPolicyLaneError(["reset_mismatch"])
    reset_result = simulator.reset(dict(reset_contract))
    if reset_result.get("matched_reset_digest") != reset_digest:
        raise LearnedPolicyLaneError(["simulator_reset_receipt_mismatch"])
    fixture = bool(policy_client.fixture_or_fake or simulator.fixture_or_fake)
    started_ns = time.time_ns()
    observations: list[dict[str, Any]] = []
    action_trace: list[dict[str, Any]] = []
    contacts: list[dict[str, Any]] = []
    collisions: list[dict[str, Any]] = []
    seen_query_ids: set[str] = set()
    current = simulator.observe()
    for step in range(max_control_steps):
        observation_evidence, runtime_observation = _observation_evidence(current)
        observations.append(observation_evidence)
        query_id = f"{identity['candidate_id']}:{step}:{uuid.uuid4().hex}"
        if query_id in seen_query_ids:
            raise LearnedPolicyLaneError(["stale_or_replayed_policy_query"])
        seen_query_ids.add(query_id)
        query_started_ns = time.time_ns()
        output = policy_client.infer(runtime_observation)
        query_ended_ns = time.time_ns()
        if not isinstance(output, Mapping) or "actions" not in output:
            raise LearnedPolicyLaneError(["native_policy_output_missing"])
        native, chunk = _native_action_evidence(
            output["actions"], expected_rows=int(identity["native_action_chunk_rows"])
        )
        normalized = droid_action_to_mujoco_targets(
            chunk[0],
            current_joint_position=runtime_observation["observation/joint_position"],
            joint_limits=frozen_joint_limits,
        )
        simulator_action = {
            "joint_position_target_rad": normalized["joint_position_target_rad"],
            "gripper_position_target_m": normalized["gripper_position_target_m"],
            "control_hz": DROID_CONTROL_HZ,
        }
        transition = simulator.apply_action(simulator_action)
        contact = transition.get("contacts")
        collision = transition.get("collisions")
        if not isinstance(contact, list) or not isinstance(collision, list):
            raise LearnedPolicyLaneError(["missing_terminal_contact_or_collision_evidence"])
        action_trace.append(
            {
                "step_index": step,
                "query_id": query_id,
                "query_started_at_ns": query_started_ns,
                "query_ended_at_ns": query_ended_ns,
                "source_observation_digest": observation_evidence["observation_digest"],
                "native_policy_output": native,
                "normalized_action": normalized,
                "simulator_action": simulator_action,
                "fresh_infer_call_observed": True,
                "native_row_executed": 0,
            }
        )
        contacts.append({"step_index": step, "rows": _clone(contact)})
        collisions.append({"step_index": step, "rows": _clone(collision)})
        current = transition.get("observation")
        if not isinstance(current, Mapping):
            raise LearnedPolicyLaneError(["post_action_observation_missing"])
        if simulator.terminal():
            break
    terminal_evidence, _ = _observation_evidence(current)
    observations.append(terminal_evidence)
    ended_ns = time.time_ns()
    return _build_attempt_receipt(
        identity=identity,
        authorization=authorization,
        reset_digest=reset_digest,
        started_ns=started_ns,
        ended_ns=ended_ns,
        observations=observations,
        action_trace=action_trace,
        contacts=contacts,
        collisions=collisions,
        metric=simulator.task_metric(),
        fixture_or_fake=fixture,
    )


def validate_attempt_evidence(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Recompute every receipt digest so placeholders cannot enter the compiler."""

    blockers: list[str] = []
    if receipt.get("schema_version") != ATTEMPT_SCHEMA_VERSION:
        blockers.append("attempt_schema_invalid")
    if receipt.get("attempt_digest") != _digest(dict(receipt), field="attempt_digest"):
        blockers.append("attempt_digest_mismatch")
    evidence = dict(receipt.get("evidence") or {})
    observations = list(evidence.get("observation_trace") or [])
    actions = list(evidence.get("action_trace") or [])
    contacts = list(evidence.get("contact_trace") or [])
    collisions = list(evidence.get("collision_trace") or [])
    if not actions or len(observations) != len(actions) + 1:
        blockers.append("observation_action_trace_length_mismatch")
    if len(contacts) != len(actions) or len(collisions) != len(actions):
        blockers.append("contact_collision_trace_length_mismatch")
    query_ids: set[str] = set()
    contract = frozen_franka_inspection_contract()
    frozen_joint_limits = list(dict(contract["embodiment"])["arm_joint_limits_rad"])
    for index, observation in enumerate(observations):
        if not isinstance(observation, Mapping):
            blockers.append(f"observation_trace_row_invalid:{index}")
            continue
        observation_core = {
            key: observation.get(key)
            for key in ("prompt", "joint_position", "gripper_position", "images")
        }
        if observation.get("observation_digest") != _digest(observation_core):
            blockers.append(f"observation_trace_digest_invalid:{index}")
    for index, row in enumerate(actions):
        if not isinstance(row, Mapping):
            blockers.append(f"action_trace_row_invalid:{index}")
            continue
        query_id = str(row.get("query_id") or "")
        if not query_id or query_id in query_ids:
            blockers.append("stale_or_replayed_actions")
        query_ids.add(query_id)
        if (
            row.get("fresh_infer_call_observed") is not True
            or row.get("native_row_executed") != 0
            or row.get("source_observation_digest")
            != dict(observations[index]).get("observation_digest")
        ):
            blockers.append(f"fresh_query_step_binding_invalid:{index}")
        native = dict(row.get("native_policy_output") or {})
        native_core = {"shape": native.get("shape"), "rows": native.get("rows")}
        if native.get("native_output_digest") != _digest(native_core):
            blockers.append(f"native_policy_output_digest_invalid:{index}")
        native_rows = native.get("rows")
        current_joint_position = dict(observations[index]).get("joint_position")
        if not isinstance(native_rows, list) or not native_rows:
            blockers.append(f"native_policy_output_rows_invalid:{index}")
            continue
        try:
            expected_normalized = droid_action_to_mujoco_targets(
                native_rows[0],
                current_joint_position=current_joint_position,
                joint_limits=frozen_joint_limits,
            )
        except (TypeError, ValueError) as exc:
            blockers.append(f"native_policy_output_conversion_invalid:{index}:{type(exc).__name__}")
            continue
        if row.get("normalized_action") != expected_normalized:
            blockers.append(f"normalized_action_conversion_mismatch:{index}")
        expected_simulator_action = {
            "joint_position_target_rad": expected_normalized["joint_position_target_rad"],
            "gripper_position_target_m": expected_normalized["gripper_position_target_m"],
            "control_hz": DROID_CONTROL_HZ,
        }
        if row.get("simulator_action") != expected_simulator_action:
            blockers.append(f"simulator_action_conversion_mismatch:{index}")
    expected = {
        "initial_state_observation_digest": (
            dict(observations[0]).get("observation_digest") if observations else None
        ),
        "observation_trace_digest": _digest({"rows": observations}),
        "action_trace_digest": _digest({"rows": actions}),
        "contact_evidence_digest": _digest({"rows": contacts}),
        "collision_evidence_digest": _digest({"rows": collisions}),
        "terminal_observation_digest": (
            dict(observations[-1]).get("observation_digest") if observations else None
        ),
        "native_policy_output_trace_digest": _digest(
            {"rows": [dict(row).get("native_policy_output") for row in actions]}
        ),
        "normalized_action_trace_digest": _digest(
            {"rows": [dict(row).get("normalized_action") for row in actions]}
        ),
        "simulator_action_trace_digest": _digest(
            {"rows": [dict(row).get("simulator_action") for row in actions]}
        ),
    }
    for field, value in expected.items():
        if receipt.get(field) != value:
            blockers.append(f"attempt_evidence_digest_mismatch:{field}")
    count = len(actions)
    if (
        receipt.get("fresh_policy_query_count") != count
        or receipt.get("learned_policy_action_count") != count
        or receipt.get("control_step_count") != count
        or receipt.get("fresh_query_for_every_control_step") is not True
    ):
        blockers.append("fresh_query_not_proven_for_every_control_step")
    if evidence.get("initial_observation") != (observations[0] if observations else None):
        blockers.append("initial_observation_evidence_mismatch")
    if evidence.get("terminal_observation") != (observations[-1] if observations else None):
        blockers.append("terminal_observation_evidence_mismatch")
    metric = dict(evidence.get("task_metric") or {})
    if receipt.get("task_metric_result") != metric:
        blockers.append("task_metric_evidence_mismatch")
    execution_core = {
        "policy_identity_digest": receipt.get("policy_identity_digest"),
        "matched_reset_digest": receipt.get("matched_reset_digest"),
        "observation_trace_digest": receipt.get("observation_trace_digest"),
        "action_trace_digest": receipt.get("action_trace_digest"),
        "contact_evidence_digest": receipt.get("contact_evidence_digest"),
        "collision_evidence_digest": receipt.get("collision_evidence_digest"),
        "terminal_observation_digest": receipt.get("terminal_observation_digest"),
        "task_metric_evidence_digest": _digest(metric),
    }
    if receipt.get("execution_receipt_digest") != _digest(execution_core):
        blockers.append("execution_receipt_digest_mismatch")
    return {
        "schema_version": "learned_policy_attempt_validation.v1",
        "status": "validated" if not blockers else "blocked",
        "candidate_id": receipt.get("candidate_id"),
        "blockers": sorted(set(blockers)),
    }


def build_execution_bundle(
    *,
    policy_candidates: Sequence[Mapping[str, Any]],
    execution_authorization: Mapping[str, Any],
    task_metric: Mapping[str, Any],
    attempts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build the only evidence-derived carrier accepted by the new-site compiler."""

    blockers: list[str] = []
    if len(policy_candidates) != EXPECTED_POLICY_COUNT or len(attempts) != EXPECTED_POLICY_COUNT:
        blockers.append("exactly_five_candidates_and_attempts_required")
    for identity in policy_candidates:
        try:
            _validate_candidate_identity(identity)
        except LearnedPolicyLaneError as exc:
            blockers.extend(exc.codes)
    if any(
        row.get("admission_status") != "admitted"
        or row.get("checkpoint_rights_status") != "admitted_internal_execution"
        or not _is_digest(row.get("checkpoint_rights_evidence_digest"))
        for row in policy_candidates
    ):
        blockers.append("candidate_contract_or_rights_admission_incomplete")
    if execution_authorization.get("policy_execution_authorized") is not True:
        blockers.append("policy_execution_not_authorized")
    if execution_authorization.get("authorization_digest") != _digest(
        dict(execution_authorization), field="authorization_digest"
    ):
        blockers.append("execution_authorization_digest_mismatch")
    if task_metric.get("metric_spec_digest") != _digest(
        dict(task_metric), field="metric_spec_digest"
    ):
        blockers.append("task_metric_digest_mismatch")
    validations = [validate_attempt_evidence(row) for row in attempts]
    for validation in validations:
        blockers.extend(validation["blockers"])
    if any(row.get("fixture_or_fake") is not False for row in attempts):
        blockers.append("fixture_attempt_cannot_enter_real_execution_bundle")
    if any(row.get("learned_policy_action_proven") is not True for row in attempts):
        blockers.append("real_learned_policy_action_not_proven")
    identity_set = {str(row.get("policy_identity_digest") or "") for row in policy_candidates}
    attempt_set = {str(row.get("policy_identity_digest") or "") for row in attempts}
    if identity_set != attempt_set or len(identity_set) != EXPECTED_POLICY_COUNT:
        blockers.append("attempt_candidate_identity_set_mismatch")
    reset_set = {str(row.get("matched_reset_digest") or "") for row in attempts}
    if len(reset_set) != 1:
        blockers.append("matched_reset_mismatch")
    if execution_authorization.get("candidate_set_digest") != candidate_set_digest(
        policy_candidates
    ):
        blockers.append("authorization_candidate_set_mismatch")
    if blockers:
        raise LearnedPolicyLaneError(blockers)
    bundle: dict[str, Any] = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "policy_candidates": _clone(list(policy_candidates)),
        "execution_authorization": _clone(dict(execution_authorization)),
        "task_metric": _clone(dict(task_metric)),
        "attempts": _clone(list(attempts)),
        "candidate_set_digest": candidate_set_digest(policy_candidates),
        "evidence_derived_receipts": True,
        "caller_generated_digest_placeholders_allowed": False,
    }
    bundle["bundle_digest"] = _digest(bundle, field="bundle_digest")
    return bundle


def unpack_execution_bundle(bundle: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and return compiler inputs from a serialized execution bundle."""

    if bundle.get("schema_version") != BUNDLE_SCHEMA_VERSION:
        raise LearnedPolicyLaneError(["execution_bundle_schema_invalid"])
    if bundle.get("bundle_digest") != _digest(dict(bundle), field="bundle_digest"):
        raise LearnedPolicyLaneError(["execution_bundle_digest_mismatch"])
    rebuilt = build_execution_bundle(
        policy_candidates=list(bundle.get("policy_candidates") or []),
        execution_authorization=dict(bundle.get("execution_authorization") or {}),
        task_metric=dict(bundle.get("task_metric") or {}),
        attempts=list(bundle.get("attempts") or []),
    )
    if rebuilt["bundle_digest"] != bundle["bundle_digest"]:
        raise LearnedPolicyLaneError(["execution_bundle_rebuild_mismatch"])
    return rebuilt


def build_terminal_admission_artifact(
    *, candidate_packet: Mapping[str, Any], authorization: Mapping[str, Any]
) -> dict[str, Any]:
    """Name the exact terminal blocker without promoting unqueried checkpoints."""

    blockers = sorted(
        {
            *(
                code
                for audit in candidate_packet.get("candidate_audits", [])
                for code in dict(audit).get("blockers", [])
            ),
            *list(authorization.get("blockers") or []),
        }
    )
    terminal: dict[str, Any] = {
        "schema_version": TERMINAL_SCHEMA_VERSION,
        "status": "blocked_before_fleet_execution",
        "contract_digest": dict(candidate_packet.get("contract") or {}).get("contract_digest"),
        "candidate_set_digest": candidate_packet.get("candidate_set_digest"),
        "candidate_identities": [
            {
                "candidate_id": row.get("candidate_id"),
                "checkpoint_uri": row.get("checkpoint_uri"),
                "checkpoint_digest": row.get("checkpoint_digest"),
                "policy_identity_digest": row.get("policy_identity_digest"),
                "admission_status": row.get("admission_status"),
                "contract_audit_digest": row.get("contract_audit_digest"),
            }
            for row in candidate_packet.get("policy_candidates", [])
        ],
        "candidate_audits": _clone(list(candidate_packet.get("candidate_audits") or [])),
        "authorization_digest": authorization.get("authorization_digest"),
        "proposed_real_identity_count": candidate_packet.get("candidate_count"),
        "admitted_real_identity_count": candidate_packet.get("admitted_candidate_count"),
        "real_identity_bound_query_count": 0,
        "matched_reset_attempt_count": 0,
        "exact_fifth_policy_blocker": (
            "All five technically compatible official OpenPI RoboArena DROID checkpoints lack "
            "checkpoint-specific rights evidence. No local checkpoint cache exists for the five "
            "required identity-bound queries, and this goal authorizes no GPU/provider spend."
        ),
        "blockers": blockers,
        "fleet_runnable": False,
        "paid_execution_authorized": False,
        "required_next_authority": (
            "Obtain an attributable checkpoint-specific execution grant for each selected "
            "checkpoint. Then obtain a separate explicit GPU budget and TTL authorization "
            "before using paid_resource_allocator for five identity-bound queries and five "
            "matched-reset attempts."
        ),
        "claim_boundary": {
            "five_immutable_public_checkpoint_identities_frozen": True,
            "five_policies_admitted": False,
            "five_real_queries_observed": False,
            "five_matched_reset_receipts_observed": False,
            "wam_or_video_presented_as_policy_execution": False,
        },
    }
    terminal["terminal_artifact_digest"] = _digest(
        terminal, field="terminal_artifact_digest"
    )
    return terminal


__all__ = [
    "ATTEMPT_SCHEMA_VERSION",
    "AUTHORIZATION_SCHEMA_VERSION",
    "BUNDLE_SCHEMA_VERSION",
    "EXPECTED_POLICY_COUNT",
    "IDENTITY_SCHEMA_VERSION",
    "IdentityBoundLearnedPolicyAdapter",
    "LearnedPolicyLaneError",
    "OFFICIAL_OPENPI_DROID_BASELINES",
    "audit_candidate",
    "build_candidate_packet",
    "build_execution_authorization",
    "build_execution_bundle",
    "build_identity_query_receipt",
    "build_terminal_admission_artifact",
    "candidate_identity_from_audit",
    "candidate_set_digest",
    "execute_learned_policy_attempt",
    "frozen_franka_inspection_contract",
    "unpack_execution_bundle",
    "validate_attempt_evidence",
    "validate_identity_query_receipt",
]
