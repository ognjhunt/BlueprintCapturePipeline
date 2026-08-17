"""Adapt one governed dual-task freeze to the optional Joint Agent lane.

The Joint Agent is a topology candidate generator, not the authority for the
already frozen task graph.  This adapter binds the exact task and SAGE-derived
source receipt while keeping that direction of authority explicit.  An
articulated task may admit one commanded target joint inside a bounded assembly;
a rigid task whose authored hinge is locked receives a typed inapplicability
receipt and can never be turned into a paid Joint Agent launch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .decision_evidence_contracts import canonical_digest, canonical_json
from .dual_task_rehearsal_contract import (
    DualTaskRehearsalContractError,
    validate_task_freeze,
)


SCHEMA_VERSION = "dual_task_joint_agent_admission.v1"
READY_STATUS = "ready_for_optional_joint_agent_topology_candidate"
INAPPLICABLE_STATUS = "inapplicable_locked_preexisting_articulation"
MAXIMUM_ASSEMBLY_JOINT_COUNT = 5
NON_TASK_MODE = "exclude_non_task_candidates_without_behavior_claim"


class DualTaskJointAgentAdmissionError(ValueError):
    """Stable fail-closed dual-task Joint Agent admission errors."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(sorted(set(str(error) for error in errors if str(error))))
        super().__init__(";".join(self.errors))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _digest(value: Any) -> bool:
    text = str(value or "")
    return len(text) == 71 and text.startswith("sha256:") and all(
        character in "0123456789abcdef" for character in text[7:]
    )


def _clone(value: Mapping[str, Any], *, code: str) -> dict[str, Any]:
    try:
        cloned = json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise DualTaskJointAgentAdmissionError([code]) from exc
    if not isinstance(cloned, dict):
        raise DualTaskJointAgentAdmissionError([code])
    return cloned


def _validated_source_receipt(value: Mapping[str, Any]) -> dict[str, Any]:
    receipt = _clone(value, code="joint_agent_source_receipt_invalid")
    errors: list[str] = []
    if receipt.get("schema_version") != "articulated_source_asset.v1":
        errors.append("joint_agent_source_receipt_schema_invalid")
    if receipt.get("status") != "materialized":
        errors.append("joint_agent_source_receipt_status_invalid")
    if receipt.get("receipt_digest") != canonical_digest(
        receipt, digest_field="receipt_digest"
    ):
        errors.append("joint_agent_source_receipt_digest_invalid")
    output = receipt.get("output_asset")
    if (
        not isinstance(output, Mapping)
        or not _digest(output.get("sha256"))
        or isinstance(output.get("size_bytes"), bool)
        or not isinstance(output.get("size_bytes"), int)
        or output.get("size_bytes", 0) <= 0
    ):
        errors.append("joint_agent_source_asset_record_invalid")
    components = receipt.get("connected_components")
    count = receipt.get("connected_component_count")
    if (
        isinstance(count, bool)
        or not isinstance(count, int)
        or count < 1
        or not isinstance(components, list)
        or len(components) != count
    ):
        errors.append("joint_agent_source_components_invalid")
    claim = receipt.get("claim_boundary")
    if (
        not isinstance(claim, Mapping)
        or claim.get("connected_components_are_not_rigid_links") is not True
        or claim.get("joint_topology_inferred") is not False
        or claim.get("simready_qualified") is not False
        or claim.get("physical_equivalence_proven") is not False
    ):
        errors.append("joint_agent_source_claim_boundary_invalid")
    readiness = receipt.get("joint_agent_0_5_2_input")
    if (
        not isinstance(readiness, Mapping)
        or readiness.get("usd_path_ready") is not True
        or readiness.get("default_prim_valid") is not True
        or readiness.get("connected_component_geom_subsets_authored") is not True
        or readiness.get("predicted_split_prim_count") != count
        or readiness.get("topology_inference_executed") is not False
    ):
        errors.append("joint_agent_source_readiness_invalid")
    if errors:
        raise DualTaskJointAgentAdmissionError(errors)
    return receipt


def _source_component_extent(
    receipt: Mapping[str, Any], axis: Sequence[float]
) -> tuple[float, float]:
    lower = math.inf
    upper = -math.inf
    for index, component in enumerate(receipt.get("connected_components") or []):
        if not isinstance(component, Mapping):
            raise DualTaskJointAgentAdmissionError(
                [f"joint_agent_source_component_invalid:{index}"]
            )
        minimum = component.get("aabb_min_asset_m")
        maximum = component.get("aabb_max_asset_m")
        if (
            not isinstance(minimum, list)
            or not isinstance(maximum, list)
            or len(minimum) != 3
            or len(maximum) != 3
            or any(
                isinstance(item, bool)
                or not isinstance(item, (int, float))
                or not math.isfinite(float(item))
                for item in [*minimum, *maximum]
            )
            or any(float(low) > float(high) for low, high in zip(minimum, maximum))
        ):
            raise DualTaskJointAgentAdmissionError(
                [f"joint_agent_source_component_invalid:{index}"]
            )
        center = [(float(a) + float(b)) * 0.5 for a, b in zip(minimum, maximum)]
        half = [(float(b) - float(a)) * 0.5 for a, b in zip(minimum, maximum)]
        projection = sum(value * direction for value, direction in zip(center, axis))
        radius = sum(value * abs(direction) for value, direction in zip(half, axis))
        lower = min(lower, projection - radius)
        upper = max(upper, projection + radius)
    if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
        raise DualTaskJointAgentAdmissionError(
            ["joint_agent_source_projection_extent_invalid"]
        )
    return lower, upper


def _normalized_axis(value: Any) -> list[float]:
    if (
        not isinstance(value, list)
        or len(value) != 3
        or any(
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
            for item in value
        )
    ):
        raise DualTaskJointAgentAdmissionError(["joint_agent_target_axis_invalid"])
    norm = math.sqrt(sum(float(item) ** 2 for item in value))
    if norm <= 0:
        raise DualTaskJointAgentAdmissionError(["joint_agent_target_axis_invalid"])
    return [float(item) / norm for item in value]


def _source_binding(
    *, publisher_scene_id: str, freeze: Mapping[str, Any], receipt: Mapping[str, Any]
) -> dict[str, Any]:
    source = freeze.get("source_object") or {}
    target = receipt.get("target") or {}
    source_files = receipt.get("source_files") or {}
    collision = source_files.get("sage_collision_usd") or {}
    output = receipt.get("output_asset") or {}
    if (
        target.get("interiorgs_instance_id") != source.get("instance_id")
        or target.get("semantic_label") != source.get("semantic_label")
        or receipt.get("source_collision_prim_path")
        != (freeze.get("removal_plan") or {}).get("source_collider_prim_path")
        or receipt.get("source_collision_identity_receipt_digest")
        != source.get("collision_identity_receipt_digest")
        or collision.get("path") != f"{publisher_scene_id}_collision.usd"
        or not _digest(collision.get("sha256"))
    ):
        raise DualTaskJointAgentAdmissionError(
            ["joint_agent_dual_task_source_binding_invalid"]
        )
    return {
        "source_receipt_digest": receipt["receipt_digest"],
        "source_asset_sha256": output["sha256"],
        "source_asset_size_bytes": output["size_bytes"],
        "connected_component_count": receipt["connected_component_count"],
        "source_collision_identity_receipt_digest": receipt[
            "source_collision_identity_receipt_digest"
        ],
        "source_collision_prim_path": receipt["source_collision_prim_path"],
        "sage_collision_usd_sha256": collision.get("sha256"),
        "target_instance_id": target.get("interiorgs_instance_id"),
        "target_semantic_label": target.get("semantic_label"),
    }


def build_dual_task_joint_agent_admission(
    *,
    publisher_scene_id: str,
    task_freeze: Mapping[str, Any],
    source_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a paid-ready Task A adapter or typed Task B inapplicability."""

    scene_id = str(publisher_scene_id).strip()
    if not scene_id or not scene_id.isdecimal():
        raise DualTaskJointAgentAdmissionError(
            ["joint_agent_publisher_scene_id_invalid"]
        )
    try:
        freeze = validate_task_freeze(task_freeze)
    except DualTaskRehearsalContractError as exc:
        raise DualTaskJointAgentAdmissionError(exc.errors) from exc
    receipt = _validated_source_receipt(source_receipt)
    source = _source_binding(
        publisher_scene_id=scene_id, freeze=freeze, receipt=receipt
    )

    task_summary = {
        "publisher_scene_id": scene_id,
        "task_id": freeze["task_id"],
        "task_kind": freeze["task_kind"],
        "task_freeze_digest": freeze["task_freeze_digest"],
        "scene_freeze_digest": freeze["scene_freeze_digest"],
        "source_instance_id": (freeze.get("source_object") or {}).get("instance_id"),
    }
    graph = freeze.get("articulation_graph") or {}
    joints = graph.get("joints") or []
    links = graph.get("links") or []
    if freeze["task_kind"] != "articulated_interaction":
        if joints and all(
            isinstance(joint, Mapping) and joint.get("role") == "locked"
            for joint in joints
        ):
            payload: dict[str, Any] = {
                "schema_version": SCHEMA_VERSION,
                "status": INAPPLICABLE_STATUS,
                "task_freeze": freeze,
                "source_receipt": receipt,
                "task": task_summary,
                "source": source,
                "reason": "rigid_task_with_only_locked_preexisting_joints",
                "paid_joint_agent_execution_permitted": False,
                "claim_boundary": {
                    "locked_joint_exercised": False,
                    "joint_agent_inference_executed": False,
                    "joint_topology_qualified": False,
                    "simready_qualified": False,
                    "physical_equivalence_proven": False,
                },
                "admission_digest": "",
            }
            payload["admission_digest"] = canonical_digest(
                payload, digest_field="admission_digest"
            )
            return payload
        raise DualTaskJointAgentAdmissionError(
            ["joint_agent_rigid_task_not_safely_inapplicable"]
        )

    if not 1 <= len(joints) <= MAXIMUM_ASSEMBLY_JOINT_COUNT:
        raise DualTaskJointAgentAdmissionError(
            ["joint_agent_assembly_joint_count_out_of_range"]
        )
    roots = [link for link in links if isinstance(link, Mapping) and link.get("is_root") is True]
    target_ids = list((freeze.get("target_configuration") or {}).get("target_joint_ids") or [])
    targets = [
        joint
        for joint in joints
        if isinstance(joint, Mapping) and joint.get("joint_id") in target_ids
    ]
    if (
        len(roots) != 1
        or len(target_ids) != 1
        or len(targets) != 1
        or targets[0].get("role") != "target"
        or targets[0].get("joint_type") not in {"revolute", "prismatic"}
    ):
        raise DualTaskJointAgentAdmissionError(
            ["joint_agent_commanded_task_joint_scope_invalid"]
        )
    target = targets[0]
    axis = _normalized_axis(target.get("axis"))
    projection_interval = _source_component_extent(receipt, axis)
    reset_tolerances = [
        float(joint.get("reset_tolerance"))
        for joint in joints
        if isinstance(joint, Mapping)
        and isinstance(joint.get("reset_tolerance"), (int, float))
        and not isinstance(joint.get("reset_tolerance"), bool)
    ]
    articulation_graph_digest = canonical_digest(graph)
    normalized_freeze: dict[str, Any] = {
        "schema_version": "joint_agent_dual_task_freeze_adapter.v1",
        "scene": {"publisher_scene_id": scene_id},
        "task_freeze_digest": freeze["task_freeze_digest"],
        "articulation_graph_digest": articulation_graph_digest,
        "source_receipt_digest": source["source_receipt_digest"],
        "task_spec": {
            "task_id": freeze["task_id"],
            "task_kind": freeze["task_kind"],
            "target_joint_id": target["joint_id"],
            "target_joint_type": target["joint_type"],
            "non_task_joint_motion_tolerance_rad": max(reset_tolerances or [0.0]),
        },
        "member_geometry_observation": {
            "coordinate_frame": "articulated_source_asset",
            "joint_axis_world": axis,
            "target_joint_type": target["joint_type"],
            "target_member_projection_constraints": [
                {
                    "axis_world": axis,
                    "interval_m": list(projection_interval),
                    "minimum_overlap_fraction": 0.5,
                }
            ],
            "selector_basis": "frozen_whole_source_object_extent_candidate_only",
        },
        "freeze_digest": "",
    }
    normalized_freeze["freeze_digest"] = canonical_digest(
        normalized_freeze, digest_field="freeze_digest"
    )
    scope: dict[str, Any] = {
        "schema_version": "joint_agent_dual_task_scope.v1",
        "task_family": "one_commanded_joint_in_bounded_multi_joint_articulated_assembly",
        "task_freeze_digest": freeze["task_freeze_digest"],
        "joint_scope": {
            "minimum_assembly_joint_count": 1,
            "maximum_assembly_joint_count": MAXIMUM_ASSEMBLY_JOINT_COUNT,
            "frozen_assembly_joint_count": len(joints),
            "commanded_task_joint_count": 1,
            "required_articulation_root_count": 1,
            "non_task_joint_mode": NON_TASK_MODE,
            "non_task_joint_motion_tolerance": max(reset_tolerances or [0.0]),
            "non_task_joint_roles": sorted(
                {
                    str(joint.get("role"))
                    for joint in joints
                    if isinstance(joint, Mapping)
                    and joint.get("joint_id") != target["joint_id"]
                }
            ),
        },
        "amendment_digest": "",
    }
    scope["amendment_digest"] = canonical_digest(
        scope, digest_field="amendment_digest"
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": READY_STATUS,
        "task_freeze": freeze,
        "source_receipt": receipt,
        "task": {
            **task_summary,
            "articulation_graph_digest": articulation_graph_digest,
            "frozen_assembly_joint_count": len(joints),
            "target_joint_id": target["joint_id"],
        },
        "source": source,
        "normalized_freeze": normalized_freeze,
        "scope_amendment": scope,
        "paid_joint_agent_execution_permitted": True,
        "claim_boundary": {
            "frozen_graph_is_prior_task_authority": True,
            "joint_agent_output_is_optional_topology_candidate": True,
            "non_task_joint_behavior_exercised": False,
            "connected_components_are_not_rigid_links": True,
            "target_selector_is_candidate_only": True,
            "joint_topology_qualified": False,
            "simready_qualified": False,
            "physical_equivalence_proven": False,
        },
        "admission_digest": "",
    }
    payload["admission_digest"] = canonical_digest(
        payload, digest_field="admission_digest"
    )
    return payload


def validate_dual_task_joint_agent_admission(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    admission = _clone(value, code="joint_agent_dual_task_admission_invalid")
    if admission.get("schema_version") != SCHEMA_VERSION or admission.get(
        "admission_digest"
    ) != canonical_digest(admission, digest_field="admission_digest"):
        raise DualTaskJointAgentAdmissionError(
            ["joint_agent_dual_task_admission_invalid"]
        )
    rebuilt = build_dual_task_joint_agent_admission(
        publisher_scene_id=str((admission.get("task") or {}).get("publisher_scene_id") or ""),
        task_freeze=admission.get("task_freeze") or {},
        source_receipt=admission.get("source_receipt") or {},
    )
    if rebuilt != admission:
        raise DualTaskJointAgentAdmissionError(
            ["joint_agent_dual_task_admission_rebuild_mismatch"]
        )
    return admission


def validate_dual_task_joint_agent_source_binding(
    admission: Mapping[str, Any], source_receipt: Mapping[str, Any]
) -> dict[str, Any]:
    """Rebuild an admission from the complete retained source receipt."""

    value = _clone(admission, code="joint_agent_dual_task_admission_invalid")
    if value.get("schema_version") != SCHEMA_VERSION or value.get(
        "admission_digest"
    ) != canonical_digest(value, digest_field="admission_digest"):
        raise DualTaskJointAgentAdmissionError(
            ["joint_agent_dual_task_admission_invalid"]
        )
    rebuilt = build_dual_task_joint_agent_admission(
        publisher_scene_id=str((value.get("task") or {}).get("publisher_scene_id") or ""),
        task_freeze=value.get("task_freeze") or {},
        source_receipt=source_receipt,
    )
    if rebuilt != value:
        raise DualTaskJointAgentAdmissionError(
            ["joint_agent_dual_task_admission_rebuild_mismatch"]
        )
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publisher-scene-id", required=True)
    parser.add_argument("--task-freeze", required=True)
    parser.add_argument("--source-receipt", required=True)
    parser.add_argument("--source-asset", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        freeze_path = Path(args.task_freeze).expanduser().resolve()
        receipt_path = Path(args.source_receipt).expanduser().resolve()
        source_asset = Path(args.source_asset).expanduser().resolve()
        freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if not isinstance(freeze, Mapping) or not isinstance(receipt, Mapping):
            raise DualTaskJointAgentAdmissionError(
                ["joint_agent_dual_task_input_invalid"]
            )
        output_record = receipt.get("output_asset") or {}
        if (
            source_asset.is_symlink()
            or not source_asset.is_file()
            or source_asset.stat().st_size != output_record.get("size_bytes")
            or _sha256(source_asset) != output_record.get("sha256")
        ):
            raise DualTaskJointAgentAdmissionError(
                ["joint_agent_source_asset_bytes_invalid"]
            )
        admission = build_dual_task_joint_agent_admission(
            publisher_scene_id=args.publisher_scene_id,
            task_freeze=freeze,
            source_receipt=receipt,
        )
        output = Path(args.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(canonical_json(admission) + "\n", encoding="utf-8")
    except (
        OSError,
        json.JSONDecodeError,
        DualTaskJointAgentAdmissionError,
    ) as exc:
        print(
            json.dumps(
                {
                    "status": "blocked",
                    "blockers": [str(exc)],
                    "provider_mutation_performed": False,
                },
                sort_keys=True,
            )
        )
        return 2
    print(
        json.dumps(
            {
                "status": admission["status"],
                "admission_digest": admission["admission_digest"],
                "output": str(output),
                "provider_mutation_performed": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "DualTaskJointAgentAdmissionError",
    "INAPPLICABLE_STATUS",
    "MAXIMUM_ASSEMBLY_JOINT_COUNT",
    "NON_TASK_MODE",
    "READY_STATUS",
    "SCHEMA_VERSION",
    "build_dual_task_joint_agent_admission",
    "validate_dual_task_joint_agent_admission",
    "validate_dual_task_joint_agent_source_binding",
]
