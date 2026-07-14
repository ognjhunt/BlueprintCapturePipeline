"""One-process Isaac backend for controller application and task measurement.

Imports are intentionally lazy: this module is shipped in the worker image and
is not importable as an Isaac runtime on ordinary CI hosts.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .g1_proprioception_map import (
    G1_PROPRIOCEPTION_MAP_SCHEMA_VERSION,
    resolve_g1_proprioception_map,
)
from .gear_sonic_joint_order_contract import (
    JOINT_ORDER_SCHEMA_VERSION,
    PROTOCOL_V4_FULL_JOINT_ORDER,
    PROTOCOL_V4_MAPPING_DIGEST,
    validate_full_joint_order,
)
from .task_episode_baseline import (
    build_task_episode_baseline,
    canonical_task_contract_sha256,
    verify_task_episode_baseline,
)


def load_robot_start_pose(route_file: str | Path) -> tuple[list[float], float]:
    """Load the attempt-bound stance used by the proven kitchen runner."""
    path = Path(route_file).expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    points = list(payload.get("route_points") or [])
    if not points or len(points[0]) != 3:
        raise RuntimeError("persistent_isaac_route_start_pose_missing")
    pose = [float(value) for value in points[0]]
    yaw = float(payload.get("accepted_stance_yaw_rad"))
    if not all(math.isfinite(value) for value in [*pose, yaw]):
        raise RuntimeError("persistent_isaac_route_start_pose_invalid")
    return pose, yaw


def compose_g1_for_episode(
    stage: Any,
    *,
    robot_prim_path: str,
    g1_usd_path: str | Path,
    route_file: str | Path,
) -> dict[str, Any]:
    """Compose and place G1 exactly once when the raw kitchen lacks it."""
    from pxr import Gf, UsdGeom, UsdPhysics  # type: ignore

    asset = Path(g1_usd_path).expanduser().resolve()
    if not asset.is_file():
        raise RuntimeError("persistent_isaac_g1_asset_missing")
    existing = stage.GetPrimAtPath(robot_prim_path)
    existing_valid = bool(existing and existing.IsValid())
    if not existing_valid:
        robot = stage.DefinePrim(robot_prim_path, "Xform")
        robot.GetReferences().AddReference(str(asset))
        stage.Load(robot_prim_path)
    robot = stage.GetPrimAtPath(robot_prim_path)
    if not robot or not robot.IsValid():
        raise RuntimeError("persistent_isaac_g1_composition_failed")

    pose, yaw = load_robot_start_pose(route_file)
    xform = UsdGeom.Xformable(robot)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(*pose))
    xform.AddRotateZOp().Set(math.degrees(yaw))

    articulation_roots = [
        str(prim.GetPath())
        for prim in stage.Traverse()
        if str(prim.GetPath()).startswith(robot_prim_path)
        and prim.HasAPI(UsdPhysics.ArticulationRootAPI)
    ]
    if not articulation_roots:
        raise RuntimeError("persistent_isaac_g1_articulation_missing_after_composition")
    return {
        "schema_version": "persistent_isaac_g1_composition.v1",
        "status": "passed",
        "robot_prim_path": robot_prim_path,
        "g1_usd_path": str(asset),
        "route_file": str(Path(route_file).expanduser().resolve()),
        "start_pose_xyz": pose,
        "start_yaw_rad": yaw,
        "robot_was_already_present": existing_valid,
        "articulation_root_paths": articulation_roots,
        "claim_boundary": (
            "Composition proves a controllable G1 is present at the attempt-bound stance; "
            "it does not prove policy actions or task success."
        ),
    }


class IsaacPersistentTaskBackend:
    def __init__(
        self,
        *,
        stage_path: str,
        robot_prim_path: str,
        evidence_dir: str | Path,
        g1_usd_path: str | Path,
        route_file: str | Path,
        headless: bool = True,
    ) -> None:
        from isaacsim import SimulationApp  # type: ignore

        self.app = SimulationApp({"headless": bool(headless)})
        import omni.timeline  # type: ignore
        import omni.usd  # type: ignore

        self.timeline = omni.timeline.get_timeline_interface()
        self.stage_path = str(Path(stage_path).expanduser().resolve())
        self.robot_prim_path = str(robot_prim_path)
        self.evidence_dir = Path(evidence_dir)
        self.evidence_dir.mkdir(parents=True, exist_ok=True)
        self.session_id = f"isaac-task-session-{uuid.uuid4().hex}"
        self.episode_baseline: dict[str, Any] | None = None
        self.episode_baseline_attestation: dict[str, Any] | None = None
        self.live_geometry_results: dict[str, dict[str, Any]] = {}
        self.attempt_id = ""
        self.launch_nonce = ""
        self.stage_id = hashlib.sha256(Path(self.stage_path).read_bytes()).hexdigest()
        omni.usd.get_context().open_stage(self.stage_path)
        self.stage = omni.usd.get_context().get_stage()
        self.robot_composition = compose_g1_for_episode(
            self.stage,
            robot_prim_path=self.robot_prim_path,
            g1_usd_path=g1_usd_path,
            route_file=route_file,
        )
        (self.evidence_dir / "robot_stage_composition.json").write_text(
            json.dumps(self.robot_composition, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        self.timeline.play()
        for _ in range(8):
            self.app.update()
        self._articulations: dict[str, Any] = {}
        self.robot = self._articulation(self.robot_prim_path)
        if not bool(getattr(self.robot, "handles_initialized", False)):
            raise RuntimeError("persistent_isaac_robot_articulation_not_found")
        from .isaac_task_review_renderer import IsaacTaskReviewRenderer

        self.review_renderer = IsaacTaskReviewRenderer(
            stage=self.stage,
            app=self.app,
            robot_prim_path=self.robot_prim_path,
            output_dir=self.evidence_dir,
        )

    def _articulation(self, prim_path: str):
        cached = self._articulations.get(prim_path)
        if cached is not None:
            return cached
        from isaacsim.core.prims import SingleArticulation  # type: ignore

        articulation = SingleArticulation(
            prim_path=prim_path,
            name=f"blueprint_articulation_{len(self._articulations)}",
        )
        articulation.initialize()
        if not bool(getattr(articulation, "handles_initialized", False)):
            raise RuntimeError(f"persistent_isaac_articulation_not_initialized:{prim_path}")
        self._articulations[prim_path] = articulation
        return articulation

    def _articulation_and_dof(self, prim_path: str):
        parts = [part for part in str(prim_path).split("/") if part]
        dof_name = parts[-1]
        candidates = ["/" + "/".join(parts[:index]) for index in range(len(parts) - 1, 0, -1)]
        for root in candidates:
            try:
                articulation = self._articulation(root)
                dof_index = int(articulation.get_dof_index(dof_name))
            except Exception:  # noqa: BLE001 - candidate roots are probed fail-closed
                continue
            if dof_index >= 0:
                return articulation, dof_index
        raise RuntimeError(f"persistent_isaac_task_dof_not_found:{prim_path}")

    @staticmethod
    def _dof_position(articulation: Any, dof_index: int) -> float:
        positions = articulation.get_joint_positions(joint_indices=[int(dof_index)])
        if positions is None or len(positions) != 1:
            raise RuntimeError("persistent_isaac_joint_position_unavailable")
        return float(positions[0])

    def _resolve_task_prim(self, criterion: Mapping[str, Any]) -> str:
        exact = str(criterion.get("articulation_prim_path") or "").strip()
        if exact:
            return exact
        resolution = dict(criterion.get("articulation_prim_path_resolution") or {})
        root_term = str(resolution.get("required_target_root") or "").lower()
        terms = [str(item).lower() for item in resolution.get("required_affordance_terms") or []]
        from pxr import UsdPhysics  # type: ignore
        import omni.usd  # type: ignore

        stage = omni.usd.get_context().get_stage()
        matches = []
        for prim in stage.Traverse():
            path = str(prim.GetPath())
            lower = path.lower()
            if root_term and root_term not in lower:
                continue
            if terms and not any(term in lower for term in terms):
                continue
            if prim.IsA(UsdPhysics.RevoluteJoint) or prim.IsA(UsdPhysics.PrismaticJoint):
                matches.append(path)
        if len(matches) != 1:
            raise RuntimeError(
                "persistent_isaac_task_prim_resolution_not_unique:" + ",".join(matches)
            )
        return matches[0]

    def _apply_controller_state(self, state: Mapping[str, Any]) -> None:
        names = [str(item) for item in state.get("joint_names") or []]
        positions = [float(item) for item in state.get("joint_positions") or []]
        if (
            str(state.get("joint_order_schema_version") or "")
            != JOINT_ORDER_SCHEMA_VERSION
            or str(state.get("mapping_digest") or "") != PROTOCOL_V4_MAPPING_DIGEST
        ):
            raise RuntimeError("persistent_isaac_controller_joint_mapping_invalid")
        try:
            validate_full_joint_order(names, source="persistent_isaac_controller")
        except ValueError as exc:
            raise RuntimeError("persistent_isaac_controller_joint_mapping_invalid") from exc
        if len(names) != len(positions) or len(names) != len(PROTOCOL_V4_FULL_JOINT_ORDER):
            raise RuntimeError("persistent_isaac_controller_joint_state_invalid")
        joint_indices = []
        for name in names:
            try:
                joint_index = int(self.robot.get_dof_index(name))
            except Exception as exc:  # noqa: BLE001 - normalize Isaac lookup errors
                raise RuntimeError(f"persistent_isaac_robot_dof_missing:{name}") from exc
            if joint_index < 0:
                raise RuntimeError(f"persistent_isaac_robot_dof_missing:{name}")
            joint_indices.append(joint_index)
        import numpy as np
        from isaacsim.core.utils.types import ArticulationAction  # type: ignore

        self.robot.apply_action(
            ArticulationAction(
                joint_positions=np.asarray(positions, dtype=np.float32),
                joint_indices=np.asarray(joint_indices, dtype=np.int64),
            )
        )

    def _live_projected_gravity(self) -> list[float]:
        """Measure base orientation and express world gravity in the base frame."""
        _, rotation = self.robot.get_world_pose()
        w, x, y, z = (float(value) for value in rotation)
        norm = math.sqrt(x * x + y * y + z * z + w * w)
        if not math.isfinite(norm) or norm <= 0:
            raise RuntimeError("persistent_isaac_base_orientation_invalid")
        x, y, z, w = x / norm, y / norm, z / norm, w / norm
        projected = [
            -(2.0 * x * z - 2.0 * y * w),
            -(2.0 * y * z + 2.0 * x * w),
            -(1.0 - 2.0 * x * x - 2.0 * y * y),
        ]
        if not all(math.isfinite(value) for value in projected):
            raise RuntimeError("persistent_isaac_projected_gravity_invalid")
        return projected

    def _single_registered_criterion(self, contract: Mapping[str, Any]) -> dict[str, Any]:
        criteria = [
            dict(item)
            for item in contract.get("registered_criteria") or contract.get("criteria") or []
        ]
        if len(criteria) != 1:
            raise RuntimeError("persistent_isaac_requires_one_registered_criterion")
        return criteria[0]

    def capture_episode_baseline(
        self,
        *,
        task_success_contract: Mapping[str, Any],
        attempt_id: str,
        launch_nonce: str,
        settle_steps: int = 8,
    ) -> dict[str, Any]:
        """Capture the immutable episode baseline after settle, before action zero."""
        if getattr(self, "episode_baseline", None) is not None:
            raise RuntimeError("persistent_isaac_episode_baseline_already_captured")
        contract = dict(task_success_contract or {})
        criterion = self._single_registered_criterion(contract)
        prim_path = self._resolve_task_prim(criterion)
        task_articulation, task_dof = self._articulation_and_dof(prim_path)
        for _ in range(max(1, int(settle_steps))):
            self.app.update()
        baseline = build_task_episode_baseline(
            episode_initial_value=self._dof_position(task_articulation, task_dof),
            attempt_id=str(attempt_id),
            launch_nonce=str(launch_nonce),
            simulator_session_id=self.session_id,
            stage_id=self.stage_id,
            articulation_prim_path=prim_path,
            task_contract_sha256=canonical_task_contract_sha256(contract),
            criterion_id=str(criterion.get("criterion_id") or ""),
            unit=str(criterion.get("unit") or ""),
            captured_timestamp=str(time.time_ns()),
        )
        artifact = self.evidence_dir / "task_episode_baseline.json"
        try:
            with artifact.open("x", encoding="utf-8") as handle:
                handle.write(json.dumps(baseline, indent=2, sort_keys=True) + "\n")
        except FileExistsError as exc:
            raise RuntimeError("persistent_isaac_episode_baseline_artifact_already_exists") from exc
        self.episode_baseline = dict(baseline)
        self.attempt_id = str(attempt_id)
        self.launch_nonce = str(launch_nonce)
        self.episode_baseline_artifact = {
            "path": str(artifact),
            "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        }
        self.live_geometry_results = self._measure_live_geometry(
            target_prim_path=prim_path,
            task_success_contract=contract,
        )
        return dict(baseline)

    def _measure_live_geometry(
        self, *, target_prim_path: str, task_success_contract: Mapping[str, Any]
    ) -> dict[str, dict[str, Any]]:
        from .isaac_live_geometry_validation import build_live_geometry_results

        try:
            position, orientation = self.robot.get_world_pose()
            robot_xyz = [float(value) for value in position]
            # Isaac Core returns WXYZ; the geometry validator consumes XYZW.
            quat = [
                float(orientation[1]),
                float(orientation[2]),
                float(orientation[3]),
                float(orientation[0]),
            ]
            renderer = self.review_renderer
            target_xyz = renderer._center(target_prim_path)
            import omni.physx  # type: ignore

            overlaps: list[str] = []

            def on_hit(hit):
                path = str(hit.get("rigid_body") or hit.get("collision") or "")
                if path:
                    overlaps.append(path)
                return True

            query = omni.physx.get_physx_scene_query_interface()
            query.overlap_box(
                (0.45, 0.45, 0.9),
                tuple(robot_xyz),
                tuple(quat),
                on_hit,
                False,
            )
            max_reach = float(task_success_contract.get("max_reach_distance_m") or 1.5)
            return build_live_geometry_results(
                robot_xyz=robot_xyz,
                robot_quaternion_xyzw=quat,
                target_xyz=target_xyz,
                overlapping_prim_paths=overlaps,
                robot_prim_path=self.robot_prim_path,
                max_reach_distance_m=max_reach,
            )
        except Exception as exc:  # noqa: BLE001 - unsupported runtime must fail closed
            blocker = f"live_isaac_geometry_measurement_failed:{type(exc).__name__}"
            return {
                "stance": {
                    "schema_version": "g1_kitchen_live_stance_validation.v1",
                    "stance_valid": False,
                    "reach_valid": False,
                    "facing_valid": False,
                    "blockers": [blocker],
                },
                "collision": {
                    "schema_version": "g1_kitchen_live_collision_validation.v1",
                    "collision_free": False,
                    "clearance_valid": False,
                    "blockers": [blocker],
                },
            }

    def install_episode_baseline_attestation(self, attestation: Mapping[str, Any]) -> None:
        if self.episode_baseline is None:
            raise RuntimeError("persistent_isaac_episode_baseline_missing")
        if getattr(self, "episode_baseline_attestation", None) is not None:
            raise RuntimeError("persistent_isaac_episode_baseline_attestation_already_installed")
        self.episode_baseline_attestation = dict(attestation)

    def apply_and_measure(self, request: Mapping[str, Any]) -> dict[str, Any]:
        action = dict(request.get("action") or {})
        wam_output = dict(request.get("wam_output") or {})
        state = dict(wam_output.get("generated_robot_state") or {})
        source_action_sha = hashlib.sha256(
            json.dumps(action, sort_keys=True, separators=(",", ":"), default=str).encode()
        ).hexdigest()
        if str(state.get("source_action_sha256") or "") != source_action_sha:
            raise RuntimeError("persistent_isaac_controller_state_action_mismatch")
        contract = dict(request.get("task_success_contract") or {})
        criterion = self._single_registered_criterion(contract)
        prim_path = self._resolve_task_prim(criterion)
        task_articulation, task_dof = self._articulation_and_dof(prim_path)
        baseline = getattr(self, "episode_baseline", None)
        baseline_blockers = verify_task_episode_baseline(
            baseline,
            simulator_session_id=self.session_id,
            stage_id=self.stage_id,
            articulation_prim_path=prim_path,
            task_contract_sha256=canonical_task_contract_sha256(contract),
            attempt_id=getattr(self, "attempt_id", ""),
            launch_nonce=getattr(self, "launch_nonce", ""),
        )
        if baseline_blockers:
            raise RuntimeError(
                "persistent_isaac_episode_baseline_invalid:" + ",".join(baseline_blockers)
            )
        if not getattr(self, "episode_baseline_attestation", None):
            raise RuntimeError("persistent_isaac_episode_baseline_attestation_missing")
        episode_initial = float(baseline["episode_initial_value"])
        before_timestamp = time.time_ns()
        before = self._dof_position(task_articulation, task_dof)
        self._apply_controller_state(state)
        for _ in range(max(1, int(request.get("physics_steps_per_action") or 4))):
            self.app.update()
        after = self._dof_position(task_articulation, task_dof)
        after_timestamp = time.time_ns()
        step = int(request.get("step_index") or 0)
        evidence_step = int(request.get("evidence_step_index", step))
        renderer = getattr(self, "review_renderer", None)
        if renderer is None:
            raise RuntimeError("persistent_isaac_attempt_review_renderer_missing")
        review_frames = renderer.render(
            step_index=evidence_step,
            target_prim_path=prim_path,
        )
        measurement = {
            "schema_version": "task_transition_measurement.v1",
            "criterion_id": criterion.get("criterion_id"),
            "observable_transition": criterion.get("observable_transition"),
            "before_value": before,
            "after_value": after,
            "episode_initial_value": episode_initial,
            "step_before": before,
            "step_after": after,
            "step_delta": after - before,
            "episode_delta": after - episode_initial,
            "episode_baseline_digest": str(baseline["baseline_digest"]),
            "episode_baseline": dict(baseline),
            "episode_baseline_artifact": dict(self.episode_baseline_artifact),
            "episode_baseline_attestation": dict(self.episode_baseline_attestation or {}),
            "attempt_id": getattr(self, "attempt_id", ""),
            "launch_nonce": getattr(self, "launch_nonce", ""),
            "unit": criterion.get("unit"),
            "source_step_index": step,
            "evidence_step_index": evidence_step,
            "source_action_sha256": source_action_sha,
            "articulation_prim_path": prim_path,
            "simulator_session_id": self.session_id,
            "stage_id": self.stage_id,
            "before_timestamp": str(before_timestamp),
            "after_timestamp": str(after_timestamp),
        }
        artifact = self.evidence_dir / f"task_measurement_{step:04d}.json"
        artifact.write_text(json.dumps(measurement, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        artifact_sha = hashlib.sha256(artifact.read_bytes()).hexdigest()
        for frame in review_frames:
            frame.update(
                {
                    "source_action_sha256": source_action_sha,
                    "simulator_session_id": self.session_id,
                    "stage_id": self.stage_id,
                    "before_timestamp": str(before_timestamp),
                    "after_timestamp": str(after_timestamp),
                    "attempt_id": getattr(self, "attempt_id", ""),
                    "launch_nonce": getattr(self, "launch_nonce", ""),
                }
            )
        from .isaac_review_media import record_frame_step_bindings

        record_frame_step_bindings(
            frames_dir=getattr(renderer, "frames_dir", self.evidence_dir / "frames"),
            artifacts=review_frames,
        )
        return {
            **measurement,
            "runtime_result_id": f"{self.session_id}-step-{step:04d}",
            "persistent_simulator_state_applied": True,
            "official_controller_action_applied": True,
            "simulator_backend": "isaac",
            "evidence_artifacts": [
                {"path": str(artifact), "sha256": artifact_sha},
                *review_frames,
            ],
            "review_frames": review_frames,
            "live_stance_validation": dict(
                getattr(self, "live_geometry_results", {}).get("stance") or {}
            ),
            "live_collision_validation": dict(
                getattr(self, "live_geometry_results", {}).get("collision") or {}
            ),
        }

    def initial_policy_state(self) -> dict[str, Any]:
        """Return attempt-bound proprioception measured from the live articulation."""
        names = list(self.robot.dof_names or [])
        positions = self.robot.get_joint_positions()
        if positions is None or len(names) != len(positions):
            raise RuntimeError("persistent_isaac_initial_proprioception_unavailable")
        observed = [
            {"name": str(name), "position": float(position)}
            for name, position in zip(names, positions, strict=True)
        ]
        resolution = resolve_g1_proprioception_map(observed, require_hands=True)
        if resolution["status"] != "passed":
            raise RuntimeError(
                "persistent_isaac_initial_proprio_mapping_blocked:"
                + ",".join(resolution["blockers"])
            )
        return {
            **resolution["group_values"],
            "projected_gravity": self._live_projected_gravity(),
            "proprioception_mapping": {
                "schema_version": G1_PROPRIOCEPTION_MAP_SCHEMA_VERSION,
                "observed_dof_inventory": resolution["observed_dof_inventory"],
                "resolved_map": resolution["resolved_map"],
                "dimensions": resolution["dimensions"],
                "unmapped_observed_dofs": resolution["unmapped_observed_dofs"],
                "mapping_digest": resolution["mapping_digest"],
            },
            "measurement": {
                "simulator_session_id": self.session_id,
                "stage_id": self.stage_id,
                "source": "live_isaac_articulation_dof_positions_and_base_orientation",
                "surrogate": False,
                "mapping_digest": resolution["mapping_digest"],
            },
        }

    def close(self) -> None:
        self.timeline.stop()
        self.app.close()


def create_backend(**kwargs) -> IsaacPersistentTaskBackend:
    return IsaacPersistentTaskBackend(**kwargs)
