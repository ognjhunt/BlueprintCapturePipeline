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


class IsaacPersistentTaskBackend:
    def __init__(
        self,
        *,
        stage_path: str,
        robot_prim_path: str,
        evidence_dir: str | Path,
        headless: bool = True,
    ) -> None:
        from isaacsim import SimulationApp  # type: ignore

        self.app = SimulationApp({"headless": bool(headless)})
        import omni.timeline  # type: ignore
        import omni.usd  # type: ignore
        from omni.isaac.dynamic_control import _dynamic_control  # type: ignore

        self.dc = _dynamic_control.acquire_dynamic_control_interface()
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
        self.timeline.play()
        for _ in range(8):
            self.app.update()
        self.robot_handle = self.dc.get_articulation(self.robot_prim_path)
        if not self.robot_handle:
            raise RuntimeError("persistent_isaac_robot_articulation_not_found")
        from .isaac_task_review_renderer import IsaacTaskReviewRenderer

        self.review_renderer = IsaacTaskReviewRenderer(
            stage=self.stage,
            app=self.app,
            robot_prim_path=self.robot_prim_path,
            output_dir=self.evidence_dir,
        )

    def _articulation_and_dof(self, prim_path: str):
        parts = [part for part in str(prim_path).split("/") if part]
        dof_name = parts[-1]
        candidates = ["/" + "/".join(parts[:index]) for index in range(len(parts) - 1, 0, -1)]
        for root in candidates:
            articulation = self.dc.get_articulation(root)
            if not articulation:
                continue
            dof = self.dc.find_articulation_dof(articulation, dof_name)
            if dof:
                return articulation, dof
        raise RuntimeError(f"persistent_isaac_task_dof_not_found:{prim_path}")

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
        for name, position in zip(names, positions):
            dof = self.dc.find_articulation_dof(self.robot_handle, name)
            if not dof:
                raise RuntimeError(f"persistent_isaac_robot_dof_missing:{name}")
            self.dc.set_dof_position_target(dof, position)

    def _live_projected_gravity(self) -> list[float]:
        """Measure base orientation and express world gravity in the base frame."""
        root_body = self.dc.get_articulation_root_body(self.robot_handle)
        pose = self.dc.get_rigid_body_pose(root_body)
        rotation = pose.r
        x, y, z, w = (
            float(rotation.x),
            float(rotation.y),
            float(rotation.z),
            float(rotation.w),
        )
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
        _, task_dof = self._articulation_and_dof(prim_path)
        for _ in range(max(1, int(settle_steps))):
            self.app.update()
        baseline = build_task_episode_baseline(
            episode_initial_value=float(self.dc.get_dof_position(task_dof)),
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
            root_body = self.dc.get_articulation_root_body(self.robot_handle)
            pose = self.dc.get_rigid_body_pose(root_body)
            robot_xyz = [float(pose.p.x), float(pose.p.y), float(pose.p.z)]
            quat = [float(pose.r.x), float(pose.r.y), float(pose.r.z), float(pose.r.w)]
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
        _, task_dof = self._articulation_and_dof(prim_path)
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
        before = float(self.dc.get_dof_position(task_dof))
        self._apply_controller_state(state)
        for _ in range(max(1, int(request.get("physics_steps_per_action") or 4))):
            self.app.update()
        after = float(self.dc.get_dof_position(task_dof))
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
        count = int(self.dc.get_articulation_dof_count(self.robot_handle))
        observed: list[dict[str, Any]] = []
        for index in range(count):
            dof = self.dc.get_articulation_dof(self.robot_handle, index)
            observed.append(
                {
                    "name": str(self.dc.get_dof_name(dof)),
                    "position": float(self.dc.get_dof_position(dof)),
                }
            )
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
