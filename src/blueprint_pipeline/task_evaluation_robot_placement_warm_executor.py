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
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .common import write_json
from .decision_evidence_contracts import canonical_digest
from .native_task_arena_construction_bundle import (
    build_native_task_arena_construction_bundle,
)
from .native_task_arena_controls_bundle import (
    build_native_task_arena_controls_bundle,
)
from .native_task_arena_packet import (
    materialize_native_task_arena_packet,
    validate_native_task_arena_packet_request,
)
from .native_task_arena_warm_authority import (
    materialize_native_task_arena_warm_attempt_authority,
)
from .native_task_arena_warm_vast import close_native_task_arena_warm_instance
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
from .task_evaluation_native_construction_feedback_controller import (
    CONTROLS_CONTINUATION_SCHEMA_VERSION,
    CompositeCandidateGenerator,
    bind_warm_native_construction_execution,
    build_next_native_construction_inventory,
    construction_phase_plan_for_candidate,
    run_native_construction_feedback_controller,
    summarize_native_construction_feedback,
    validate_native_construction_candidate,
)
from .task_evaluation_robot_placement_agent import (
    robot_placement_agents_sdk_config,
)
from .task_evaluation_supervisor.agents_sdk import OpenAIAgentsSDKInvoker
from .task_evaluation_curobo_candidate_generator import (
    CUROBO_BACKEND_IDENTITY,
    RemoteCuroboCandidateGenerator,
)
from .task_evaluation_curobo_context import materialize_remote_curobo_context


CONFIG_SCHEMA_VERSION = "task_evaluation_robot_placement_warm_native_loop.v1"
FEEDBACK_EXECUTOR_CONFIG_SCHEMA_VERSION = (
    "task_evaluation_native_construction_feedback_warm_executor.v1"
)
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
        # The historical summary is retained for receipt compatibility.  The
        # controller-facing form adds collision/contact/task-pose/camera
        # measurements and its own digest, so the next deterministic inventory
        # does not need to reverse-engineer a paid result.
        "scientific_construction_feedback": (
            summarize_native_construction_feedback(result)
        ),
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


class WarmNativeConstructionFeedbackExecutor:
    """Execute exact controller candidates on one already-owned Arena worker."""

    def __init__(
        self,
        *,
        config: Mapping[str, Any],
        output_root: str | Path,
        allocator_main: Callable[[Sequence[str] | None], int] | None = None,
    ) -> None:
        value = json.loads(json.dumps(dict(config), allow_nan=False))
        if value.get("schema_version") != FEEDBACK_EXECUTOR_CONFIG_SCHEMA_VERSION:
            raise WarmRobotPlacementExecutorError(
                "native_construction_feedback_executor_config_invalid"
            )
        self._base_request = validate_native_task_arena_packet_request(
            _read_mapping(
                _absolute_file(
                    value.get("base_packet_request_path"),
                    blocker="native_construction_feedback_executor_config_invalid",
                ),
                blocker="native_construction_feedback_executor_config_invalid",
            )
        )
        evidence_root = Path(str(value.get("evidence_root") or "")).expanduser()
        self._evidence_root = evidence_root.resolve()
        if evidence_root.is_symlink() or not self._evidence_root.is_dir():
            raise WarmRobotPlacementExecutorError(
                "native_construction_feedback_executor_config_invalid"
            )
        self._runtime_source_packet_path = _absolute_file(
            value.get("runtime_source_packet_receipt_path"),
            blocker="native_construction_feedback_executor_config_invalid",
        )
        self._warm_session_path = _absolute_file(
            value.get("warm_session_path"),
            blocker="native_construction_feedback_executor_config_invalid",
        )
        self._warm_session = _read_mapping(
            self._warm_session_path,
            blocker="native_construction_feedback_executor_config_invalid",
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
                "native_construction_feedback_executor_config_invalid"
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
            or not 0 < self._hard_cap_usd <= 10
            or not 1_800 <= self._hard_ttl_seconds <= 14_400
            or not isinstance(self._warm_session.get("instance_id"), int)
        ):
            raise WarmRobotPlacementExecutorError(
                "native_construction_feedback_executor_config_invalid"
            )
        self._output_root = Path(output_root).expanduser().resolve()
        if self._output_root.exists() or self._output_root.is_symlink():
            raise WarmRobotPlacementExecutorError(
                "native_construction_feedback_executor_output_exists"
            )
        if allocator_main is None:
            from .paid_resource_allocator import main as allocator_main_impl

            allocator_main = allocator_main_impl
        self._allocator_main = allocator_main
        self._qualified_context: dict[str, Any] | None = None

    def __call__(
        self,
        candidate: Mapping[str, Any],
        execution_binding: Mapping[str, Any],
    ) -> dict[str, Any]:
        selected = validate_native_construction_candidate(candidate)
        binding = json.loads(json.dumps(dict(execution_binding), allow_nan=False))
        if (
            binding.get("candidate_digest") != selected["candidate_digest"]
            or binding.get("expected_provider_instance_id")
            != self._warm_session["instance_id"]
            or binding.get("warm_session_digest")
            != self._warm_session.get("session_digest")
        ):
            raise WarmRobotPlacementExecutorError(
                "native_construction_feedback_executor_binding_invalid"
            )
        self._output_root.mkdir(parents=True, exist_ok=True)
        round_root = self._output_root / f"round-{int(binding['round_index']):02d}"
        round_root.mkdir(parents=False, exist_ok=False)
        write_json(round_root / "candidate.v1.json", selected)
        write_json(round_root / "execution-binding.v1.json", binding)

        request = json.loads(json.dumps(self._base_request, allow_nan=False))
        request["robot_base_pose_world"] = selected["robot_base_pose_world"]
        request["robot_joint_reset_positions_rad"] = selected["reset_variant"][
            "robot_joint_reset_positions_rad"
        ]
        request["cameras"] = selected["camera_variant"]["cameras"]
        request["construction_feedback_candidate_binding"] = {
            "run_id": binding["run_id"],
            "round_index": binding["round_index"],
            "inventory_digest": binding["inventory_digest"],
            "candidate_digest": selected["candidate_digest"],
            "reset_variant_digest": selected["reset_variant"][
                "reset_variant_digest"
            ],
            "entry_trajectory_variant_digest": selected[
                "entry_trajectory_variant"
            ]["entry_trajectory_variant_digest"],
            "camera_variant_digest": selected["camera_variant"][
                "camera_variant_digest"
            ],
            "gates_or_thresholds_modified": False,
        }
        request["request_digest"] = ""
        request["request_digest"] = canonical_digest(
            request, digest_field="request_digest"
        )
        packet_dir = round_root / "native-task-packet"
        materialize_native_task_arena_packet(
            request=request,
            evidence_root=self._evidence_root,
            output_dir=packet_dir,
        )
        scene_plan = _read_mapping(
            packet_dir / "native_task_arena_scene_plan.v1.json",
            blocker="native_construction_feedback_scene_plan_invalid",
        )
        phase_plan = construction_phase_plan_for_candidate(
            scene_plan=scene_plan,
            candidate=selected,
        )
        bundle_root = round_root / "bundle"
        prepared = build_native_task_arena_construction_bundle(
            job_dir=bundle_root,
            packet_dir=packet_dir,
            runtime_source_packet_receipt=self._runtime_source_packet_path,
            implementation_commit=self._implementation_commit,
            construction_phase_plan_override=phase_plan,
        )
        receipt_path = (
            bundle_root / "native_task_arena_provider_bundle_receipt.v1.json"
        )
        authority_path = (
            round_root / "native_task_arena_warm_attempt_authority.v1.json"
        )
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
                str(round_root / "admission.json"),
                "--adapter-output",
                str(adapter_path),
                "--execute",
            ]
        )
        if exit_code not in {0, 2} or not adapter_path.is_file():
            raise WarmRobotPlacementExecutorError(
                "native_construction_feedback_allocator_invalid"
            )
        allocator_result = _read_mapping(
            adapter_path,
            blocker="native_construction_feedback_allocator_invalid",
        )
        native_path = allocator_result.get("native_construction_result_path")
        if not native_path:
            raise WarmRobotPlacementExecutorError(
                "native_construction_feedback_native_result_missing"
            )
        native_result = _read_mapping(
            native_path,
            blocker="native_construction_feedback_native_result_invalid",
        )
        execution = bind_warm_native_construction_execution(
            candidate=selected,
            inventory_digest=str(binding["inventory_digest"]),
            allocator_result=allocator_result,
            native_result=native_result,
        )
        write_json(round_root / "controller-execution.v1.json", execution)
        if execution["status"] == "passed":
            self._qualified_context = {
                "candidate_digest": selected["candidate_digest"],
                "packet_dir": str(packet_dir),
                "construction_result_path": str(Path(native_path).resolve()),
            }
        return execution

    def continue_to_controls(
        self, controller_receipt: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Run the canonical control pair, then close the retained worker."""

        receipt = json.loads(json.dumps(dict(controller_receipt), allow_nan=False))
        context = self._qualified_context
        if (
            context is None
            or receipt.get("qualified_candidate_digest")
            != context["candidate_digest"]
            or receipt.get("provider_instance_id")
            != self._warm_session["instance_id"]
        ):
            raise WarmRobotPlacementExecutorError(
                "native_construction_feedback_controls_binding_invalid"
            )
        root = self._output_root / "controls"
        root.mkdir(parents=False, exist_ok=False)
        packet_dir = Path(context["packet_dir"])
        construction_result_path = Path(context["construction_result_path"])
        prepared = build_native_task_arena_controls_bundle(
            job_dir=root / "bundle",
            packet_dir=packet_dir,
            construction_result_path=construction_result_path,
            runtime_source_packet_receipt=self._runtime_source_packet_path,
            implementation_commit=self._implementation_commit,
        )
        bundle_receipt = (
            root / "bundle" / "native_task_arena_provider_bundle_receipt.v1.json"
        )
        authority_path = root / "native_task_arena_warm_attempt_authority.v1.json"
        materialize_native_task_arena_warm_attempt_authority(
            warm_session_path=self._warm_session_path,
            bundle_receipt_path=bundle_receipt,
            prepared_bundle=prepared,
            authorization_reference=self._authorization_reference,
            authorized_by=self._authorized_by,
            authorized_on=self._authorized_on,
            output_path=authority_path,
        )
        adapter_path = root / "allocator-result.json"
        exit_code = self._allocator_main(
            [
                "gpu-canary",
                "--provider",
                "vast",
                "--probe-kind",
                "native-task-arena-controls",
                "--native-task-arena-packet",
                str(packet_dir),
                "--native-task-arena-runtime-source-packet",
                str(self._runtime_source_packet_path),
                "--native-task-arena-construction-result",
                str(construction_result_path),
                "--native-task-arena-bundle-receipt",
                str(bundle_receipt),
                "--native-task-arena-attempt-authority",
                str(authority_path),
                "--native-task-arena-warm-session",
                str(self._warm_session_path),
                "--adp-job-dir",
                str(root / "execution"),
                "--adp-max-hourly-rate-usd",
                str(self._max_hourly_rate_usd),
                "--adp-max-spend-usd",
                str(self._hard_cap_usd),
                "--adp-hard-ttl-seconds",
                str(self._hard_ttl_seconds),
                "--adp-allowed-active-vast-instance-id",
                str(self._warm_session["instance_id"]),
                "--admission-out",
                str(root / "admission.json"),
                "--adapter-output",
                str(adapter_path),
                "--execute",
            ]
        )
        if exit_code != 0 or not adapter_path.is_file():
            raise WarmRobotPlacementExecutorError(
                "native_construction_feedback_controls_allocator_invalid"
            )
        allocator = _read_mapping(
            adapter_path,
            blocker="native_construction_feedback_controls_allocator_invalid",
        )
        result_path = Path(
            str(allocator.get("native_control_result_path") or "")
        ).expanduser()
        control = _read_mapping(
            result_path,
            blocker="native_construction_feedback_controls_result_invalid",
        )
        if (
            allocator.get("status") != "completed"
            or allocator.get("provider_instance_id")
            != self._warm_session["instance_id"]
            or allocator.get("provider_allocations_performed") != 0
            or allocator.get("continuing_spend_from_this_run") is not False
            or control.get("schema_version")
            != "native_task_arena_control_result.v1"
            or control.get("status") != "completed"
            or control.get("controls_qualified") is not True
            or control.get("blockers") not in ([], ())
            or control.get("result_digest")
            != canonical_digest(control, digest_field="result_digest")
        ):
            raise WarmRobotPlacementExecutorError(
                "native_construction_feedback_controls_result_invalid"
            )
        continuation: dict[str, Any] = {
            "schema_version": CONTROLS_CONTINUATION_SCHEMA_VERSION,
            "status": "completed",
            "run_id": receipt["run_id"],
            "construction_qualification_digest": receipt[
                "construction_qualification_digest"
            ],
            "qualified_candidate_digest": context["candidate_digest"],
            "provider_instance_id": self._warm_session["instance_id"],
            "provider_allocations_performed": 0,
            "continuing_spend_from_this_run": False,
            "native_control_result_digest": control["result_digest"],
            "native_control_result_path": str(result_path.resolve()),
            "zero_action_and_scripted_positive_qualified": True,
            "controls_continuation_digest": "",
        }
        continuation["controls_continuation_digest"] = canonical_digest(
            continuation, digest_field="controls_continuation_digest"
        )
        write_json(root / "controls-continuation.v1.json", continuation)
        return continuation


def _run_retained_native_construction_feedback(
    *,
    cold_allocator_result: Mapping[str, Any],
    packet_dir: str | Path,
    runtime_source_packet_receipt_path: str | Path,
    implementation_commit: str,
    output_root: str | Path,
    authorization_reference: str,
    authorized_by: str,
    authorized_on: str,
    max_hourly_rate_usd: float,
    hard_cap_usd: float,
    hard_ttl_seconds: int,
    max_inference_cost_usd: float = 0.64,
    maximum_rounds: int = 4,
    invoker: Any | None = None,
    allocator_main: Callable[[Sequence[str] | None], int] | None = None,
    candidate_generator: Any | None = None,
    search_ledger: Any | None = None,
) -> dict[str, Any]:
    """Continue one retained cold construction through variants and controls.

    The production episode compiler embeds the CPU-built candidate universe in
    the packet request.  This callsite reopens those exact bytes only after a
    scientifically rejected cold construction retained its one provider
    instance.  It allocates nothing: every subsequent construction and the
    control pair use canonical warm-attach allocator calls.
    """

    cold = json.loads(json.dumps(dict(cold_allocator_result), allow_nan=False))
    native_path = Path(str(cold.get("native_control_result_path") or "")).expanduser()
    warm_session_path = Path(
        str(cold.get("warm_session_receipt_path") or "")
    ).expanduser()
    native = _read_mapping(
        native_path,
        blocker="native_construction_feedback_cold_result_invalid",
    )
    feedback = summarize_native_construction_feedback(native)
    warm_session = _read_mapping(
        warm_session_path,
        blocker="native_construction_feedback_warm_session_invalid",
    )
    packet = Path(packet_dir).expanduser().resolve()
    request_path = packet / "native_task_arena_packet_request.v1.json"
    request = validate_native_task_arena_packet_request(
        _read_mapping(
            request_path,
            blocker="native_construction_feedback_packet_request_invalid",
        )
    )
    embedded = request.get("native_construction_feedback")
    universe = (
        embedded.get("candidate_universe")
        if isinstance(embedded, Mapping)
        else None
    )
    generator_authority = (
        embedded.get("candidate_generator_authority")
        if isinstance(embedded, Mapping)
        else None
    )
    if (
        feedback["passed"]
        or not isinstance(universe, Mapping)
        or embedded.get("allocator_retry_cap") != 0
        or embedded.get("native_gates_unchanged") is not True
        or not isinstance(generator_authority, Mapping)
        or generator_authority.get("generator")
        != "remote_curobo_v2_motion_generation"
        or generator_authority.get("package_version") != "0.8.0"
        or generator_authority.get("source_revision")
        != CUROBO_BACKEND_IDENTITY["source_revision"]
        or generator_authority.get("required_on_retained_gpu") is not True
        or generator_authority.get("deterministic_cpu_prefilter_required") is not True
        or generator_authority.get("silent_fallback_permitted") is not False
        or warm_session.get("status") != "ready"
        or warm_session.get("continuing_spend") is not True
        or warm_session.get("remote_work_dir")
        not in {"/workspace", "/tmp/blueprint_vast_work"}
        or cold.get("continuing_spend_from_this_run") is not True
        or cold.get("retry_cap") != 0
    ):
        raise WarmRobotPlacementExecutorError(
            "native_construction_feedback_cold_continuation_invalid"
        )
    run_id = str(universe.get("run_id") or "")
    initial_inventory = build_next_native_construction_inventory(
        run_id=run_id,
        round_index=0,
        source_native_feedback=feedback,
        prior_history=(),
        candidate_universe=universe.get("candidates") or [],
        maximum_candidates=min(
            int(universe.get("maximum_candidates_per_round") or 64), 64
        ),
    )
    try:
        deadline = float(warm_session["watchdog_deadline_epoch"])
        cold_cost = float(cold.get("estimated_cost_usd") or 0.0)
    except (KeyError, TypeError, ValueError) as exc:
        raise WarmRobotPlacementExecutorError(
            "native_construction_feedback_warm_session_invalid"
        ) from exc
    remaining_cost = min(float(hard_cap_usd) - cold_cost, 2.0)
    if not math.isfinite(remaining_cost) or remaining_cost <= 0.0:
        raise WarmRobotPlacementExecutorError(
            "native_construction_feedback_cost_authority_exhausted"
        )
    authority: dict[str, Any] = {
        "schema_version": (
            "task_evaluation_native_construction_feedback_authority.v1"
        ),
        "run_id": run_id,
        "expected_provider_instance_id": int(warm_session["instance_id"]),
        "warm_session_digest": warm_session["session_digest"],
        "allocator_retry_cap": 0,
        "maximum_rounds": min(int(maximum_rounds), 8),
        "maximum_candidates_per_round": min(
            len(initial_inventory["candidates"]), 64
        ),
        # The outer provider authority remains the ultimate cap.  Candidate
        # ceilings and the controller aggregate refuse any larger sequence.
        "maximum_incremental_cost_usd": remaining_cost,
        "deadline_unix_s": deadline,
        "authority_digest": "",
    }
    authority["authority_digest"] = canonical_digest(
        authority, digest_field="authority_digest"
    )
    executor = WarmNativeConstructionFeedbackExecutor(
        config={
            "schema_version": FEEDBACK_EXECUTOR_CONFIG_SCHEMA_VERSION,
            "base_packet_request_path": str(request_path),
            "evidence_root": str(packet.parent),
            "runtime_source_packet_receipt_path": str(
                runtime_source_packet_receipt_path
            ),
            "warm_session_path": str(warm_session_path),
            "implementation_commit": implementation_commit,
            "authorization_reference": authorization_reference,
            "authorized_by": authorized_by,
            "authorized_on": authorized_on,
            "max_hourly_rate_usd": float(max_hourly_rate_usd),
            "hard_cap_usd": float(hard_cap_usd),
            "hard_ttl_seconds": int(hard_ttl_seconds),
        },
        output_root=output_root,
        allocator_main=allocator_main,
    )
    if candidate_generator is None:
        context, remote_package_root = materialize_remote_curobo_context(
            packet_dir=packet,
            universe=universe,
            output_root=Path(output_root).with_name(
                Path(output_root).name + "-curobo-context"
            ),
            commit=implementation_commit,
            maximum_incremental_cost_usd=min(remaining_cost, 0.2),
            maximum_runtime_seconds=min(
                max(30.0, deadline - time.time() - 120.0), 300.0
            ),
            warm_session=warm_session,
        )
        candidate_generator = RemoteCuroboCandidateGenerator(
            context=context,
            warm_session=warm_session,
            local_transport_root=Path(output_root).with_name(
                Path(output_root).name + "-curobo-transport"
            ),
            remote_python_package_root=remote_package_root,
        )
    if search_ledger is None:
        from .task_evaluation_native_construction_optuna_ledger import (
            NativeConstructionOptunaSearchLedger,
        )

        search_ledger = NativeConstructionOptunaSearchLedger(
            root=Path(output_root) / "search-ledger",
            run_id=run_id,
        )
    if invoker is None:
        invoker = OpenAIAgentsSDKInvoker(
            robot_placement_agents_sdk_config(
                max_inference_cost_usd=float(max_inference_cost_usd),
                allow_live_invocation=True,
                tracing_disabled=True,
            )
        )

    class DeterministicUniverseGenerator:
        def generate(
            self,
            *,
            source_native_feedback,
            prior_history,
            round_index,
            maximum_candidates,
        ):
            return build_next_native_construction_inventory(
                run_id=run_id,
                round_index=round_index,
                source_native_feedback=source_native_feedback,
                prior_history=prior_history,
                candidate_universe=universe["candidates"],
                maximum_candidates=min(
                    maximum_candidates, len(universe["candidates"]), 64
                ),
            )

    composite_generator = CompositeCandidateGenerator(
        generators=(() if candidate_generator is None else (candidate_generator,)),
        deterministic_fallback=DeterministicUniverseGenerator(),
        fallback_on_generator_unavailable=False,
    )

    controller_result = run_native_construction_feedback_controller(
        invoker=invoker,
        authority=authority,
        initial_inventory=initial_inventory,
        produce_next_inventory=None,
        candidate_generator=composite_generator,
        search_ledger=search_ledger,
        execute_candidate=executor,
        continue_to_controls=executor.continue_to_controls,
        initial_native_feedback=feedback,
    )
    if controller_result.get("status") != "controls_completed":
        closeout = close_native_task_arena_warm_instance(
            warm_session=warm_session
        )
        controller_result["warm_session_closeout"] = closeout
        controller_result["continuing_spend_from_this_run"] = closeout.get(
            "continuing_spend_from_this_run"
        )
        controller_result["receipt_digest"] = ""
        controller_result["receipt_digest"] = canonical_digest(
            controller_result, digest_field="receipt_digest"
        )
    return controller_result


def run_retained_native_construction_feedback(**kwargs: Any) -> dict[str, Any]:
    """Run feedback and guarantee retained-instance closeout on any refusal."""

    try:
        return _run_retained_native_construction_feedback(**kwargs)
    except Exception as exc:
        cold = dict(kwargs.get("cold_allocator_result") or {})
        warm_session_path = Path(
            str(cold.get("warm_session_receipt_path") or "")
        ).expanduser()
        closeout: dict[str, Any] = {
            "status": "closeout_not_attempted",
            "continuing_spend_from_this_run": True,
        }
        try:
            session = _read_mapping(
                warm_session_path,
                blocker="native_construction_feedback_warm_session_invalid",
            )
            closeout = close_native_task_arena_warm_instance(
                warm_session=session
            )
        except Exception as close_exc:
            closeout = {
                "status": "closeout_failed",
                "blockers": [
                    "native_construction_feedback_closeout_failed:"
                    + type(close_exc).__name__
                ],
                "continuing_spend_from_this_run": True,
            }
        output_root = Path(str(kwargs.get("output_root") or "")).expanduser()
        if output_root.is_absolute():
            output_root.mkdir(parents=True, exist_ok=True)
            failure = {
                "schema_version": (
                    "task_evaluation_native_construction_feedback_failure.v1"
                ),
                "status": "blocked",
                "blockers": [
                    "native_construction_feedback_failed:" + type(exc).__name__
                ],
                "warm_session_closeout": closeout,
                "continuing_spend_from_this_run": closeout.get(
                    "continuing_spend_from_this_run"
                ),
                "failure_digest": "",
            }
            failure["failure_digest"] = canonical_digest(
                failure, digest_field="failure_digest"
            )
            write_json(output_root / "feedback-failure.v1.json", failure)
        raise WarmRobotPlacementExecutorError(
            "native_construction_feedback_failed:" + type(exc).__name__
        ) from exc


__all__ = [
    "CONFIG_SCHEMA_VERSION",
    "FEEDBACK_EXECUTOR_CONFIG_SCHEMA_VERSION",
    "WarmNativeConstructionFeedbackExecutor",
    "WarmNativePlacementExecutor",
    "WarmRobotPlacementExecutorError",
    "native_attempt_from_warm_result",
    "run_retained_native_construction_feedback",
]
