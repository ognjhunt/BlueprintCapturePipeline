"""Derive a graph-bound articulated contact path from registered OpenUSD.

This deterministic candidate samples the frozen reset-to-target joint path and
evaluates the selected contact link using exact USD joint/body transforms.  It
does not author geometry, solve robot IK, or claim native reach/contact.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .articulation_graph_contract import validate_articulation_graph
from .decision_evidence_contracts import canonical_digest
from .dual_task_rehearsal_contract import validate_task_freeze


SCHEMA_VERSION = "paired_target_articulated_kinematic_path.v1"


class PairedTargetArticulatedKinematicPathError(ValueError):
    """Stable deterministic articulated-path error."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _read(path: str | Path, code: str) -> tuple[Path, dict[str, Any]]:
    source = Path(path).expanduser().resolve()
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PairedTargetArticulatedKinematicPathError(code) from exc
    if source.is_symlink() or not isinstance(value, dict):
        raise PairedTargetArticulatedKinematicPathError(code)
    return source, value


def _record(path: Path, **extra: Any) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        raise PairedTargetArticulatedKinematicPathError(
            "paired_target_kinematic_path_source_invalid"
        )
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
        **extra,
    }


def _unit(value: Sequence[float], code: str) -> list[float]:
    vector = [float(item) for item in value]
    norm = math.sqrt(sum(item * item for item in vector))
    if len(vector) != 3 or not math.isfinite(norm) or norm <= 1.0e-12:
        raise PairedTargetArticulatedKinematicPathError(code)
    return [item / norm for item in vector]


def materialize_paired_target_articulated_kinematic_path(
    *,
    task_freeze_path: str | Path,
    interaction_affordance_path: str | Path,
    output_path: str | Path,
    waypoint_count: int = 5,
) -> dict[str, Any]:
    """Sample one complete graph path and exact contact-link transforms."""

    if isinstance(waypoint_count, bool) or not 2 <= waypoint_count <= 64:
        raise PairedTargetArticulatedKinematicPathError(
            "paired_target_kinematic_path_waypoint_count_invalid"
        )
    freeze_path, raw_freeze = _read(
        task_freeze_path, "paired_target_kinematic_path_task_freeze_invalid"
    )
    freeze = validate_task_freeze(raw_freeze)
    affordance_path, affordance = _read(
        interaction_affordance_path,
        "paired_target_kinematic_path_affordance_invalid",
    )
    candidate = affordance.get("candidate")
    registered = affordance.get("registered_usd")
    if (
        affordance.get("schema_version")
        != "paired_target_interaction_affordance_candidate.v1"
        or affordance.get("receipt_digest")
        != canonical_digest(affordance, digest_field="receipt_digest")
        or affordance.get("task_id") != freeze.get("task_id")
        or affordance.get("task_kind") != "articulated_interaction"
        or not isinstance(candidate, Mapping)
        or not isinstance(registered, Mapping)
    ):
        raise PairedTargetArticulatedKinematicPathError(
            "paired_target_kinematic_path_affordance_invalid"
        )
    usd_path = Path(str(registered.get("path") or "")).expanduser().resolve()
    if (
        usd_path.is_symlink()
        or not usd_path.is_file()
        or usd_path.stat().st_size != registered.get("size_bytes")
        or _sha256(usd_path) != registered.get("sha256")
    ):
        raise PairedTargetArticulatedKinematicPathError(
            "paired_target_kinematic_path_usd_mismatch"
        )
    graph = validate_articulation_graph(freeze["articulation_graph"])
    target_ids = [row["joint_id"] for row in graph["joints"] if row["role"] == "target"]
    if len(target_ids) != 1:
        raise PairedTargetArticulatedKinematicPathError(
            "paired_target_kinematic_path_target_joint_invalid"
        )
    target_id = target_ids[0]
    target_interval = graph["success_predicate"]["joint_intervals"][target_id]
    target_value = (float(target_interval[0]) + float(target_interval[1])) / 2.0
    contact_link_id = str(candidate.get("link_id") or "")
    contact_local = [float(item) for item in candidate.get("contact_point_link_m") or []]
    approach_world = _unit(
        candidate.get("approach_unit_registered_stage") or [],
        "paired_target_kinematic_path_approach_invalid",
    )
    if len(contact_local) != 3:
        raise PairedTargetArticulatedKinematicPathError(
            "paired_target_kinematic_path_contact_invalid"
        )
    try:
        from pxr import Gf, Usd, UsdGeom, UsdPhysics
    except ImportError as exc:  # pragma: no cover
        raise PairedTargetArticulatedKinematicPathError(
            "paired_target_kinematic_path_openusd_missing"
        ) from exc
    stage = Usd.Stage.Open(str(usd_path))
    root = stage.GetDefaultPrim() if stage is not None else None
    if root is None or not root.IsValid():
        raise PairedTargetArticulatedKinematicPathError(
            "paired_target_kinematic_path_usd_invalid"
        )
    root_path = str(root.GetPath())
    links = {row["link_id"]: row for row in graph["links"]}
    joints = {row["joint_id"]: row for row in graph["joints"]}
    child_joint = {row["child_link_id"]: row for row in graph["joints"]}
    if contact_link_id not in links:
        raise PairedTargetArticulatedKinematicPathError(
            "paired_target_kinematic_path_contact_link_invalid"
        )

    def joint_positions(target: float) -> dict[str, float]:
        values: dict[str, float] = {}
        for joint_id, row in joints.items():
            if row["role"] == "target":
                values[joint_id] = target
            elif row["role"] == "dependent":
                dependency = row["dependency"]
                values[joint_id] = (
                    target * float(dependency["multiplier"])
                    + float(dependency["offset"])
                )
            else:
                values[joint_id] = float(row["reset_position"])
        return values

    cache = UsdGeom.XformCache()

    def matrix(value: Any) -> np.ndarray:
        # Gf stores row-vector matrices; transpose into conventional
        # column-vector homogeneous form for explicit FK multiplication.
        return np.asarray(value, dtype=np.float64).T

    root_world = matrix(cache.GetLocalToWorldTransform(root))
    world_from_root = np.linalg.inv(root_world)
    approach = _unit(
        world_from_root[:3, :3] @ np.asarray(approach_world),
        "paired_target_kinematic_path_approach_invalid",
    )

    def joint_frame(position: Any, rotation: Any) -> np.ndarray:
        value = Gf.Matrix4d(1.0)
        value.SetRotate(Gf.Rotation(rotation or Gf.Quatf(1.0)))
        value.SetTranslateOnly(Gf.Vec3d(*(position or Gf.Vec3f())))
        return matrix(value)

    root_links = [
        link_id
        for link_id, row in links.items()
        if row.get("is_root") is True
    ]
    if len(root_links) != 1:
        raise PairedTargetArticulatedKinematicPathError(
            "paired_target_kinematic_path_root_link_invalid"
        )
    root_link_prim = stage.GetPrimAtPath(f"{root_path}/links/{root_links[0]}")
    if not root_link_prim.IsValid():
        raise PairedTargetArticulatedKinematicPathError(
            "paired_target_kinematic_path_link_missing"
        )
    root_link_transform = world_from_root @ matrix(
        cache.GetLocalToWorldTransform(root_link_prim)
    )

    def all_link_transforms(positions: Mapping[str, float]) -> dict[str, np.ndarray]:
        transforms = {root_links[0]: root_link_transform}
        remaining = dict(child_joint)
        while remaining:
            progress = False
            for child_id, row in list(remaining.items()):
                parent_id = str(row["parent_link_id"])
                if parent_id not in transforms:
                    continue
                joint_prim = stage.GetPrimAtPath(
                    f"{root_path}/joints/{row['joint_id']}"
                )
                child_prim = stage.GetPrimAtPath(f"{root_path}/links/{child_id}")
                if not joint_prim.IsValid() or not child_prim.IsValid():
                    raise PairedTargetArticulatedKinematicPathError(
                        "paired_target_kinematic_path_joint_missing"
                    )
                joint = UsdPhysics.Joint(joint_prim)
                frame0 = joint_frame(
                    joint.GetLocalPos0Attr().Get(),
                    joint.GetLocalRot0Attr().Get(),
                )
                frame1 = joint_frame(
                    joint.GetLocalPos1Attr().Get(),
                    joint.GetLocalRot1Attr().Get(),
                )
                motion = np.eye(4, dtype=np.float64)
                value = float(positions[row["joint_id"]])
                axis_token = str(joint_prim.GetAttribute("physics:axis").Get() or "")
                axis_index = {"X": 0, "Y": 1, "Z": 2}.get(axis_token)
                if row["joint_type"] in {"revolute", "continuous"}:
                    if axis_index is None:
                        raise PairedTargetArticulatedKinematicPathError(
                            "paired_target_kinematic_path_joint_axis_invalid"
                        )
                    axis = np.zeros(3, dtype=np.float64)
                    axis[axis_index] = 1.0
                    x, y, z = axis
                    cosine, sine = math.cos(value), math.sin(value)
                    cross = np.array(
                        [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]]
                    )
                    motion[:3, :3] = (
                        cosine * np.eye(3)
                        + (1.0 - cosine) * np.outer(axis, axis)
                        + sine * cross
                    )
                elif row["joint_type"] == "prismatic":
                    if axis_index is None:
                        raise PairedTargetArticulatedKinematicPathError(
                            "paired_target_kinematic_path_joint_axis_invalid"
                        )
                    motion[axis_index, 3] = value
                transforms[child_id] = (
                    transforms[parent_id]
                    @ frame0
                    @ motion
                    @ np.linalg.inv(frame1)
                )
                del remaining[child_id]
                progress = True
            if not progress:
                raise PairedTargetArticulatedKinematicPathError(
                    "paired_target_kinematic_path_graph_disconnected"
                )
        return transforms

    reset_positions = joint_positions(float(joints[target_id]["reset_position"]))
    reset_transforms = all_link_transforms(reset_positions)
    for link_id, derived in reset_transforms.items():
        authored_prim = stage.GetPrimAtPath(f"{root_path}/links/{link_id}")
        authored = world_from_root @ matrix(cache.GetLocalToWorldTransform(authored_prim))
        if not np.allclose(derived, authored, rtol=0.0, atol=1.0e-6):
            raise PairedTargetArticulatedKinematicPathError(
                f"paired_target_kinematic_path_reset_fk_mismatch:{link_id}"
            )

    rows = []
    for index in range(waypoint_count):
        fraction = index / (waypoint_count - 1)
        target = float(joints[target_id]["reset_position"]) + fraction * (
            target_value - float(joints[target_id]["reset_position"])
        )
        positions = joint_positions(target)
        transform = all_link_transforms(positions)[contact_link_id]
        point = transform @ np.asarray([*contact_local, 1.0], dtype=np.float64)
        gf_transform = Gf.Matrix4d(*transform.T.reshape(-1).tolist())
        rotation = gf_transform.ExtractRotationQuat()
        rows.append(
            {
                "waypoint_id": f"mechanism_path_{index:02d}",
                "joint_positions": positions,
                "contact_pose_asset_root": {
                    "position_m": [float(item) for item in point[:3]],
                    "orientation_xyzw": [
                        float(rotation.GetImaginary()[0]),
                        float(rotation.GetImaginary()[1]),
                        float(rotation.GetImaginary()[2]),
                        float(rotation.GetReal()),
                    ],
                },
                "clearance_unit_asset_root": approach,
            }
        )
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "deterministic_registered_usd_kinematic_candidate",
        "task_id": freeze["task_id"],
        "asset_id": affordance["asset_id"],
        "task_freeze": _record(
            freeze_path, task_freeze_digest=freeze["task_freeze_digest"]
        ),
        "interaction_affordance": _record(
            affordance_path, receipt_digest=affordance["receipt_digest"]
        ),
        "registered_usd": _record(usd_path),
        "articulation_graph_digest": canonical_digest(graph),
        "target_joint_id": target_id,
        "target_value_rad": target_value,
        "waypoint_count": waypoint_count,
        "joint_contact_path": rows,
        "native_ik_or_contact_executed": False,
        "claim_boundary": (
            "deterministic_openusd_joint_fk_candidate_only;not_native_ik_"
            "reach_contact_clearance_task_success_policy_or_physical_evidence"
        ),
        "receipt_digest": "",
    }
    payload["receipt_digest"] = canonical_digest(payload, digest_field="receipt_digest")
    destination = Path(output_path).expanduser().resolve()
    if destination.exists() or destination.is_symlink():
        raise PairedTargetArticulatedKinematicPathError(
            "paired_target_kinematic_path_output_exists"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


__all__ = [
    "PairedTargetArticulatedKinematicPathError",
    "SCHEMA_VERSION",
    "materialize_paired_target_articulated_kinematic_path",
]
