"""Execute placement candidates on one retained native Task Arena worker.

This is the concrete caller for the ``PlacementExecutor`` seam.  Each model
proposal is compiled into a new digest-bound construction packet, bundled from
the exact current commit, admitted through the canonical allocator, and attached
to the same independently watched Vast instance.  Native failures and selected
frames return to the placement agent for its next bounded round.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import mimetypes
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .common import write_json
from .decision_evidence_contracts import canonical_digest
from .native_task_arena_construction_bundle import (
    build_native_task_arena_construction_bundle,
)
from .native_task_arena_warm_authority import (
    materialize_native_task_arena_warm_attempt_authority,
)
from .task_evaluation_diagnostic_native_arena_compiler import (
    SCHEMA_VERSION as DIAGNOSTIC_COMPILER_OUTPUT_SCHEMA_VERSION,
    compile_diagnostic_native_arena_packet,
)
from .native_task_construction_plan import (
    materialize_native_task_construction_phase_plan,
)
from .task_evaluation_robot_placement_trajectory import (
    placement_trajectory_from_native_plan,
    validate_robot_placement_trajectory,
)


CONFIG_SCHEMA_VERSION = "task_evaluation_robot_placement_warm_native_loop.v1"
ATTEMPT_SCHEMA_VERSION = "task_evaluation_robot_placement_native_attempt.v1"


class WarmRobotPlacementExecutorError(RuntimeError):
    """A warm native round could not produce authoritative scientific feedback."""


def _trajectory_content_matches(
    expected: Mapping[str, Any], observed_plan: Mapping[str, Any]
) -> bool:
    """Compare the immutable trajectory while allowing its scene envelope to change."""

    observed = placement_trajectory_from_native_plan(observed_plan)
    expected_content = dict(expected)
    observed_content = dict(observed)
    for value in (expected_content, observed_content):
        value.pop("source_plan_digest", None)
        value.pop("trajectory_digest", None)
    return expected_content == observed_content


def _read_mapping(path: str | Path, *, blocker: str) -> dict[str, Any]:
    unresolved = Path(path).expanduser()
    resolved = unresolved.resolve()
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise WarmRobotPlacementExecutorError(blocker) from exc
    if unresolved.is_symlink() or not isinstance(value, Mapping):
        raise WarmRobotPlacementExecutorError(blocker)
    return dict(value)


def _droid_profile_reference(path: str | Path) -> dict[str, Any]:
    """Open either the direct reference or its sealed compiler envelope.

    The diagnostic compiler is the canonical producer of the DROID profile
    reference used by later warm rounds.  Its durable artifact is the compiler
    output receipt, not a second standalone reference file.  Accepting that
    receipt here keeps the consumer bound to the exact produced bytes while
    preserving the historical direct-reference input for existing callers.
    """

    value = _read_mapping(
        path, blocker="robot_placement_droid_profile_reference_invalid"
    )
    nested = value.get("droid_profile_reference")
    if nested is None:
        return value
    if (
        value.get("schema_version") != DIAGNOSTIC_COMPILER_OUTPUT_SCHEMA_VERSION
        or value.get("status") != "completed_development_only"
        or value.get("compiler_output_digest")
        != canonical_digest(value, digest_field="compiler_output_digest")
        or not isinstance(nested, Mapping)
    ):
        raise WarmRobotPlacementExecutorError(
            "robot_placement_droid_profile_reference_invalid"
        )
    return dict(nested)


def _absolute_file(value: object, *, blocker: str) -> Path:
    unresolved = Path(str(value or "")).expanduser()
    resolved = unresolved.resolve()
    if unresolved.is_symlink() or not resolved.is_file():
        raise WarmRobotPlacementExecutorError(blocker)
    return resolved


def _image_input(path: Path, *, label: str) -> dict[str, Any]:
    payload = path.read_bytes()
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    if not mime.startswith("image/"):
        raise WarmRobotPlacementExecutorError(
            "robot_placement_native_feedback_image_invalid"
        )
    return {
        "label": label,
        "digest": "sha256:" + hashlib.sha256(payload).hexdigest(),
        "image_url": f"data:{mime};base64," + base64.b64encode(payload).decode("ascii"),
        "detail": "high",
    }


def _selected_feedback_images(execution_root: Path) -> list[dict[str, Any]]:
    selected = (
        ("external", "precontact"),
        ("overview", "precontact"),
        ("wrist", "precontact"),
        ("external", "push_contact"),
        ("wrist", "push_contact"),
        ("overview", "recovery"),
    )
    images = []
    for camera, phase in selected:
        path = execution_root / "construction_frames" / camera / f"{phase}.png"
        if path.is_file() and not path.is_symlink():
            images.append(_image_input(path, label=f"native_{camera}_{phase}"))
    return images


def _native_feedback_summary(result: Mapping[str, Any]) -> dict[str, Any]:
    phases = []
    for raw in result.get("phase_results") or []:
        if not isinstance(raw, Mapping):
            continue
        phases.append(
            {
                key: raw.get(key)
                for key in (
                    "phase_id",
                    "steps",
                    "target_position_world_m",
                    "terminal_position_world_m",
                    "terminal_position_error_m",
                    "terminal_orientation_error_rad",
                    "target_reached",
                    "position_error_m",
                    "orientation_error_rad",
                    "ik_reached",
                    "arrival_reached",
                    "blockers",
                )
                if key in raw
            }
        )
    cameras = {}
    for camera, raw in (result.get("camera_gates") or {}).items():
        if not isinstance(raw, Mapping):
            continue
        best = raw.get("best_observability") or {}
        cameras[str(camera)] = {
            "passed": raw.get("passed"),
            "best_snapshot_id": raw.get("best_snapshot_id"),
            "pixel_fraction": best.get("pixel_fraction"),
            "centroid_xy_fraction": best.get("centroid_xy_fraction"),
            "blockers": best.get("blockers"),
        }
    return {
        "initial_robot_root_pose_world": (
            (result.get("initial_readback") or {}).get("robot_root_pose_world")
            if isinstance(result.get("initial_readback"), Mapping)
            else None
        ),
        "franka_pose_binding": result.get("franka_pose_binding"),
        "rigid_construction_gates": result.get("rigid_construction_gates"),
        "phase_results": phases,
        "camera_gates": cameras,
    }


def native_attempt_from_warm_result(
    *, allocator_result: Mapping[str, Any], native_result_path: str | Path
) -> dict[str, Any]:
    """Convert one canonical warm result into agent feedback without regrading it."""

    result_path = _absolute_file(
        native_result_path, blocker="robot_placement_native_result_missing"
    )
    native = _read_mapping(
        result_path, blocker="robot_placement_native_result_invalid"
    )
    observed_digest = canonical_digest(native, digest_field="result_digest")
    if (
        native.get("schema_version")
        != "native_task_arena_construction_result.v1"
        or native.get("result_digest") != observed_digest
    ):
        raise WarmRobotPlacementExecutorError(
            "robot_placement_native_result_invalid"
        )
    passed = bool(
        allocator_result.get("status") == "completed"
        and native.get("status") == "completed"
        and native.get("construction_gate_qualified") is True
        and not native.get("blockers")
    )
    blockers = sorted(
        set(
            str(item)
            for item in [
                *(allocator_result.get("blockers") or []),
                *(native.get("blockers") or []),
            ]
            if str(item)
        )
    )
    if not passed and not blockers:
        blockers = ["robot_placement_native_construction_not_qualified"]
    execution_root = result_path.parent
    feedback_images = _selected_feedback_images(execution_root)
    if not passed and not feedback_images:
        raise WarmRobotPlacementExecutorError(
            "robot_placement_native_feedback_images_missing"
        )
    attempt: dict[str, Any] = {
        "schema_version": ATTEMPT_SCHEMA_VERSION,
        "status": "passed" if passed else "rejected",
        "phase_reached": str(native.get("phase_reached") or "construction"),
        "blockers": blockers,
        "native_result_digest": native["result_digest"],
        "provider_instance_id": allocator_result.get("provider_instance_id"),
        "provider_allocations_performed": allocator_result.get(
            "provider_allocations_performed"
        ),
        "runtime_seconds": allocator_result.get("runtime_seconds"),
        "incremental_cost_upper_bound_usd": allocator_result.get(
            "incremental_cost_upper_bound_usd"
        ),
        "native_feedback": _native_feedback_summary(native),
        "native_attempt_digest": "",
    }
    attempt["native_attempt_digest"] = canonical_digest(
        attempt, digest_field="native_attempt_digest"
    )
    attempt["feedback_images"] = feedback_images
    return attempt


class WarmNativePlacementExecutor:
    """Callable placement executor backed by the canonical warm Vast allocator."""

    def __init__(
        self,
        *,
        config: Mapping[str, Any],
        task_trajectory: Mapping[str, Any],
        output_root: str | Path,
        allocator_main: Callable[[Sequence[str] | None], int] | None = None,
    ) -> None:
        value = json.loads(json.dumps(dict(config), allow_nan=False))
        if value.get("schema_version") != CONFIG_SCHEMA_VERSION:
            raise WarmRobotPlacementExecutorError(
                "robot_placement_native_loop_config_invalid"
            )
        self._task_trajectory = validate_robot_placement_trajectory(task_trajectory)
        self._diagnostic_controls_input_path = _absolute_file(
            value.get("diagnostic_controls_input_path"),
            blocker="robot_placement_native_loop_config_invalid",
        )
        self._droid_profile_path = _absolute_file(
            value.get("droid_profile_path"),
            blocker="robot_placement_native_loop_config_invalid",
        )
        self._droid_profile_reference = _droid_profile_reference(
            _absolute_file(
                value.get("droid_profile_reference_path"),
                blocker="robot_placement_native_loop_config_invalid",
            )
        )
        self._runtime_source_packet_path = _absolute_file(
            value.get("runtime_source_packet_receipt_path"),
            blocker="robot_placement_native_loop_config_invalid",
        )
        self._warm_session_path = _absolute_file(
            value.get("warm_session_path"),
            blocker="robot_placement_native_loop_config_invalid",
        )
        self._warm_session = _read_mapping(
            self._warm_session_path,
            blocker="robot_placement_native_loop_config_invalid",
        )
        self._implementation_commit = str(value.get("implementation_commit") or "")
        self._authorization_reference = str(
            value.get("authorization_reference") or ""
        )
        self._authorized_by = str(value.get("authorized_by") or "")
        self._authorized_on = str(value.get("authorized_on") or "")
        try:
            self._max_hourly_rate_usd = float(value.get("max_hourly_rate_usd"))
            self._hard_cap_usd = float(value.get("hard_cap_usd"))
            self._hard_ttl_seconds = int(value.get("hard_ttl_seconds"))
        except (TypeError, ValueError) as exc:
            raise WarmRobotPlacementExecutorError(
                "robot_placement_native_loop_config_invalid"
            ) from exc
        if (
            len(self._implementation_commit) != 40
            or any(ch not in "0123456789abcdef" for ch in self._implementation_commit)
            or not self._authorization_reference
            or not self._authorized_by
            or not self._authorized_on
            or not math.isfinite(self._max_hourly_rate_usd)
            or not 0 < self._max_hourly_rate_usd <= 10
            or not math.isfinite(self._hard_cap_usd)
            or not 0 < self._hard_cap_usd <= 2
            or not 1_800 <= self._hard_ttl_seconds <= 14_400
            or not isinstance(self._warm_session.get("instance_id"), int)
        ):
            raise WarmRobotPlacementExecutorError(
                "robot_placement_native_loop_config_invalid"
            )
        self._output_root = Path(output_root).expanduser().resolve()
        if self._output_root.exists() or self._output_root.is_symlink():
            raise WarmRobotPlacementExecutorError(
                "robot_placement_native_loop_output_exists"
            )
        if allocator_main is None:
            from .paid_resource_allocator import main as allocator_main_impl

            allocator_main = allocator_main_impl
        self._allocator_main = allocator_main

    def __call__(
        self,
        proposal: Mapping[str, Any],
        provisional_receipt: Mapping[str, Any],
        round_index: int,
    ) -> dict[str, Any]:
        self._output_root.mkdir(parents=True, exist_ok=True)
        round_root = self._output_root / f"round-{int(round_index):02d}"
        round_root.mkdir(parents=False, exist_ok=False)
        write_json(round_root / "proposal.v1.json", dict(proposal))
        write_json(
            round_root / "provisional_robot_placement_receipt.v1.json",
            dict(provisional_receipt),
        )
        compiler = compile_diagnostic_native_arena_packet(
            diagnostic_controls_input=_read_mapping(
                self._diagnostic_controls_input_path,
                blocker="robot_placement_diagnostic_controls_input_invalid",
            ),
            droid_profile_path=self._droid_profile_path,
            droid_profile_reference=self._droid_profile_reference,
            output_root=round_root / "compiled",
            robot_placement_receipt=provisional_receipt,
            task_trajectory=self._task_trajectory,
        )
        packet_dir = Path(compiler["packet_receipt_path"]).parent
        scene_plan = _read_mapping(
            packet_dir / "native_task_arena_scene_plan.v1.json",
            blocker="robot_placement_native_scene_plan_invalid",
        )
        observed_plan = materialize_native_task_construction_phase_plan(scene_plan)
        if not _trajectory_content_matches(self._task_trajectory, observed_plan):
            raise WarmRobotPlacementExecutorError(
                "robot_placement_native_trajectory_binding_mismatch"
            )
        bundle_root = round_root / "bundle"
        prepared = build_native_task_arena_construction_bundle(
            job_dir=bundle_root,
            packet_dir=packet_dir,
            runtime_source_packet_receipt=self._runtime_source_packet_path,
            implementation_commit=self._implementation_commit,
        )
        receipt_path = bundle_root / "native_task_arena_provider_bundle_receipt.v1.json"
        authority_path = round_root / "native_task_arena_warm_attempt_authority.v1.json"
        materialize_native_task_arena_warm_attempt_authority(
            warm_session_path=self._warm_session_path,
            bundle_receipt_path=receipt_path,
            prepared_bundle=prepared,
            authorization_reference=self._authorization_reference,
            authorized_by=self._authorized_by,
            authorized_on=self._authorized_on,
            output_path=authority_path,
        )
        execution_root = round_root / "execution"
        admission_path = round_root / "admission.json"
        adapter_path = round_root / "allocator-result.json"
        exit_code = self._allocator_main(
            [
                "gpu-canary",
                "--provider",
                "vast",
                "--probe-kind",
                "native-task-arena-construction",
                "--native-task-arena-packet",
                str(packet_dir),
                "--native-task-arena-runtime-source-packet",
                str(self._runtime_source_packet_path),
                "--native-task-arena-bundle-receipt",
                str(receipt_path),
                "--native-task-arena-attempt-authority",
                str(authority_path),
                "--native-task-arena-warm-session",
                str(self._warm_session_path),
                "--adp-job-dir",
                str(execution_root),
                "--adp-max-hourly-rate-usd",
                str(self._max_hourly_rate_usd),
                "--adp-max-spend-usd",
                str(self._hard_cap_usd),
                "--adp-hard-ttl-seconds",
                str(self._hard_ttl_seconds),
                "--adp-allowed-active-vast-instance-id",
                str(self._warm_session["instance_id"]),
                "--admission-out",
                str(admission_path),
                "--adapter-output",
                str(adapter_path),
                "--execute",
            ]
        )
        if not adapter_path.is_file():
            raise WarmRobotPlacementExecutorError(
                "robot_placement_native_allocator_output_missing"
            )
        allocator_result = _read_mapping(
            adapter_path, blocker="robot_placement_native_allocator_output_invalid"
        )
        native_result_path = allocator_result.get("native_construction_result_path")
        if not native_result_path:
            blockers = allocator_result.get("blockers") or []
            raise WarmRobotPlacementExecutorError(
                "robot_placement_native_infrastructure_blocked:"
                + ",".join(sorted(str(item) for item in blockers))
            )
        # A scientifically rejected construction returns allocator exit 2 by design.
        # Any other nonzero exit still has to carry an exact native result above.
        if exit_code not in {0, 2}:
            raise WarmRobotPlacementExecutorError(
                "robot_placement_native_allocator_exit_invalid"
            )
        attempt = native_attempt_from_warm_result(
            allocator_result=allocator_result,
            native_result_path=str(native_result_path),
        )
        write_json(round_root / "native_attempt.v1.json", attempt)
        return attempt


__all__ = [
    "CONFIG_SCHEMA_VERSION",
    "WarmNativePlacementExecutor",
    "WarmRobotPlacementExecutorError",
    "native_attempt_from_warm_result",
]
