"""One-process Isaac backend for controller application and task measurement.

Imports are intentionally lazy: this module is shipped in the worker image and
is not importable as an Isaac runtime on ordinary CI hosts.
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any


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
        if len(names) != len(positions) or not names:
            raise RuntimeError("persistent_isaac_controller_joint_state_invalid")
        for name, position in zip(names, positions):
            dof = self.dc.find_articulation_dof(self.robot_handle, name)
            if not dof:
                raise RuntimeError(f"persistent_isaac_robot_dof_missing:{name}")
            self.dc.set_dof_position_target(dof, position)

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
        criteria = [dict(item) for item in contract.get("registered_criteria") or contract.get("criteria") or []]
        if len(criteria) != 1:
            raise RuntimeError("persistent_isaac_requires_one_registered_criterion")
        criterion = criteria[0]
        prim_path = self._resolve_task_prim(criterion)
        _, task_dof = self._articulation_and_dof(prim_path)
        before_timestamp = time.time_ns()
        before = float(self.dc.get_dof_position(task_dof))
        self._apply_controller_state(state)
        for _ in range(max(1, int(request.get("physics_steps_per_action") or 4))):
            self.app.update()
        after = float(self.dc.get_dof_position(task_dof))
        after_timestamp = time.time_ns()
        step = int(request.get("step_index") or 0)
        renderer = getattr(self, "review_renderer", None)
        if renderer is None:
            raise RuntimeError("persistent_isaac_attempt_review_renderer_missing")
        review_frames = renderer.render(
            step_index=step,
            target_prim_path=prim_path,
        )
        measurement = {
            "schema_version": "task_transition_measurement.v1",
            "criterion_id": criterion.get("criterion_id"),
            "observable_transition": criterion.get("observable_transition"),
            "before_value": before,
            "after_value": after,
            "unit": criterion.get("unit"),
            "source_step_index": step,
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
        }

    def initial_policy_state(self) -> dict[str, Any]:
        """Return attempt-bound proprioception measured from the live articulation."""
        count = int(self.dc.get_articulation_dof_count(self.robot_handle))
        named: dict[str, float] = {}
        for index in range(count):
            dof = self.dc.get_articulation_dof(self.robot_handle, index)
            named[str(self.dc.get_dof_name(dof)).lower()] = float(
                self.dc.get_dof_position(dof)
            )

        def group(*tokens: str, limit: int) -> list[float]:
            values = [
                value
                for name, value in sorted(named.items())
                if all(token in name for token in tokens)
            ]
            if len(values) != limit:
                raise RuntimeError(
                    "persistent_isaac_initial_proprio_group_dimension_mismatch:"
                    + "_".join(tokens)
                )
            return values

        return {
            "left_leg": group("left", limit=6),
            "right_leg": group("right", limit=6),
            "waist": group("waist", limit=3),
            "left_arm": group("left", "shoulder", limit=3)
            + group("left", "elbow", limit=1)
            + group("left", "wrist", limit=3),
            "right_arm": group("right", "shoulder", limit=3)
            + group("right", "elbow", limit=1)
            + group("right", "wrist", limit=3),
            "left_hand": group("left", "hand", limit=7),
            "right_hand": group("right", "hand", limit=7),
            "projected_gravity": [0.0, 0.0, -1.0],
            "measurement": {
                "simulator_session_id": self.session_id,
                "stage_id": self.stage_id,
                "source": "live_isaac_articulation_dof_positions",
                "surrogate": False,
            },
        }

    def close(self) -> None:
        self.timeline.stop()
        self.app.close()


def create_backend(**kwargs) -> IsaacPersistentTaskBackend:
    return IsaacPersistentTaskBackend(**kwargs)
