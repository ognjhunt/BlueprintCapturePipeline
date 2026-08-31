"""Fail-closed process boundary for collision-aware construction candidates.

The native-construction feedback controller intentionally knows nothing about
cuRobo or MoveIt Task Constructor.  Both planners implement this boundary and
return the controller's one canonical, digest-bound candidate inventory.  A
planner solution remains generation evidence only: Isaac construction and the
existing controls gates are still unresolved and authoritative.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .decision_evidence_contracts import canonical_digest, canonical_json


REQUEST_SCHEMA_VERSION = (
    "task_evaluation_collision_aware_candidate_generation_request.v1"
)
RESULT_SCHEMA_VERSION = (
    "task_evaluation_collision_aware_candidate_generation_result.v1"
)
INVENTORY_SCHEMA_VERSION = "task_evaluation_native_construction_candidate_inventory.v1"
CANDIDATE_SCHEMA_VERSION = "task_evaluation_native_construction_candidate.v1"
RUNTIME_PROBE_SCHEMA_VERSION = "task_evaluation_candidate_generator_runtime_probe.v1"

REQUIRED_STAGE_KINDS = ("entry", "approach", "contact", "release", "retreat")
_SHA256 = re.compile(r"sha256:[0-9a-f]{64}")
_COMMIT = re.compile(r"[0-9a-f]{40}")
_FORBIDDEN_CANDIDATE_KEYS = frozenset(
    {
        "arrival_tolerance_m",
        "arrival_orientation_tolerance_rad",
        "collision_failure_minimum_force_n",
        "contact_failure_minimum_force_n",
        "gate",
        "gates",
        "gate_ids",
        "passed",
        "qualified",
        "score",
        "status",
        "success",
        "threshold",
        "thresholds",
    }
)


class CollisionAwareCandidateGenerationError(ValueError):
    """The planner runtime or its result could not be trusted."""


class CandidateGenerator(Protocol):
    """Stable producer interface consumed by the feedback controller."""

    def generate(
        self,
        *,
        source_native_feedback: Mapping[str, Any] | None,
        prior_history: Sequence[Mapping[str, Any]],
        round_index: int,
        maximum_candidates: int,
    ) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class CandidateGeneratorContext:
    """Immutable inputs and budget shared by one configured generator."""

    run_id: str
    expected_production_commit: str
    robot_configuration: Mapping[str, Any]
    world_configuration: Mapping[str, Any]
    task_trajectory: Mapping[str, Any]
    analytic_candidate_inventory: Mapping[str, Any]
    maximum_incremental_cost_usd: float
    maximum_runtime_seconds: float


CommandRunner = Callable[..., subprocess.CompletedProcess[str]]


def _copy(value: Mapping[str, Any]) -> dict[str, Any]:
    try:
        return json.loads(json.dumps(dict(value), allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise CollisionAwareCandidateGenerationError(
            "candidate_generation_value_invalid"
        ) from exc


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def validate_sealed_file_reference(
    value: Mapping[str, Any], *, role: str
) -> dict[str, Any]:
    """Reopen one local staged input and verify every declared byte."""

    reference = _copy(value)
    path_text = str(reference.get("path") or "")
    digest = str(reference.get("digest") or "")
    size = reference.get("size_bytes")
    unresolved_path = Path(path_text).expanduser()
    try:
        path = unresolved_path.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise CollisionAwareCandidateGenerationError(
            f"candidate_generation_{role}_unavailable"
        ) from exc
    if (
        reference.get("role") != role
        or unresolved_path.is_symlink()
        or not path.is_file()
        or not isinstance(size, int)
        or size < 1
        or not _SHA256.fullmatch(digest)
        or path.stat().st_size != size
        or _sha256_file(path) != digest
    ):
        raise CollisionAwareCandidateGenerationError(
            f"candidate_generation_{role}_invalid"
        )
    reference["path"] = str(path)
    attachments = reference.get("attachments")
    if attachments is not None:
        if (
            not isinstance(attachments, list)
            or not attachments
            or any(not isinstance(row, Mapping) for row in attachments)
        ):
            raise CollisionAwareCandidateGenerationError(
                f"candidate_generation_{role}_attachments_invalid"
            )
        validated_attachments = []
        for row in attachments:
            attachment_role = str(row.get("role") or "")
            if not attachment_role or row.get("attachments") is not None:
                raise CollisionAwareCandidateGenerationError(
                    f"candidate_generation_{role}_attachments_invalid"
                )
            validated_attachments.append(
                validate_sealed_file_reference(row, role=attachment_role)
            )
        reference["attachments"] = validated_attachments
    return reference


def _finite(value: object, *, blocker: str, positive: bool = False) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise CollisionAwareCandidateGenerationError(blocker) from exc
    if not math.isfinite(result) or (positive and result <= 0.0):
        raise CollisionAwareCandidateGenerationError(blocker)
    return result


def _contains_forbidden_candidate_key(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(
            str(key).lower() in _FORBIDDEN_CANDIDATE_KEYS
            or _contains_forbidden_candidate_key(child)
            for key, child in value.items()
        )
    if isinstance(value, list):
        return any(_contains_forbidden_candidate_key(child) for child in value)
    return False


def _feedback_codes(value: Mapping[str, Any] | None) -> list[str]:
    if value is None:
        return []
    raw = value.get("feedback_codes")
    if raw is None:
        raw = value.get("native_blockers") or value.get("blockers")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raw = []
    codes = {
        str(row)
        for row in raw
        if isinstance(row, str) and 0 < len(row) <= 240
    }
    failed = str(value.get("first_failed_phase") or "")
    if failed:
        codes.add(f"phase_unreached:{failed}")
    collision = value.get("first_collision")
    if isinstance(collision, Mapping):
        phase_id = str(collision.get("phase_id") or "")
        channel = str(collision.get("channel") or "")
        if phase_id and channel:
            codes.add(f"collision:{phase_id}:{channel}")
    for role, camera in (value.get("camera_measurements") or {}).items():
        if isinstance(camera, Mapping) and camera.get("passed") is not True:
            codes.add(f"camera_failed:{role}")
            if camera.get("site_rendered") is False:
                codes.add(f"site_not_rendered:{role}")
    return sorted(
        {
            row for row in codes if 0 < len(row) <= 240
        }
    )


def build_candidate_generation_request(
    *,
    context: CandidateGeneratorContext,
    backend_identity: Mapping[str, Any],
    source_native_feedback: Mapping[str, Any] | None,
    prior_history: Sequence[Mapping[str, Any]],
    round_index: int,
    maximum_candidates: int,
) -> dict[str, Any]:
    if (
        not context.run_id
        or not _COMMIT.fullmatch(context.expected_production_commit)
        or not isinstance(round_index, int)
        or round_index < 0
        or not isinstance(maximum_candidates, int)
        or maximum_candidates < 1
    ):
        raise CollisionAwareCandidateGenerationError(
            "candidate_generation_context_invalid"
        )
    feedback = _copy(source_native_feedback) if source_native_feedback is not None else None
    feedback_digest = None
    if feedback is not None:
        feedback_digest = feedback.get("feedback_digest") or feedback.get("result_digest")
        if not _SHA256.fullmatch(str(feedback_digest or "")):
            raise CollisionAwareCandidateGenerationError(
                "candidate_generation_feedback_digest_invalid"
            )
    history_digests: list[str] = []
    for row in prior_history:
        execution = row.get("execution")
        candidate = row.get("candidate")
        digest = (
            row.get("execution_digest")
            or row.get("candidate_digest")
            or (
                execution.get("execution_result_digest")
                if isinstance(execution, Mapping)
                else None
            )
            or (
                candidate.get("candidate_digest")
                if isinstance(candidate, Mapping)
                else None
            )
        )
        if not _SHA256.fullmatch(str(digest or "")):
            raise CollisionAwareCandidateGenerationError(
                "candidate_generation_history_digest_invalid"
            )
        history_digests.append(str(digest))
    request = {
        "schema_version": REQUEST_SCHEMA_VERSION,
        "run_id": context.run_id,
        "round_index": round_index,
        "expected_production_commit": context.expected_production_commit,
        "backend_identity": _copy(backend_identity),
        "source_native_feedback_digest": feedback_digest,
        "addressable_feedback_codes": _feedback_codes(feedback),
        "prior_execution_digests": history_digests,
        "maximum_candidates": maximum_candidates,
        "maximum_incremental_cost_usd": _finite(
            context.maximum_incremental_cost_usd,
            blocker="candidate_generation_budget_invalid",
            positive=True,
        ),
        "maximum_runtime_seconds": _finite(
            context.maximum_runtime_seconds,
            blocker="candidate_generation_budget_invalid",
            positive=True,
        ),
        "robot_configuration": validate_sealed_file_reference(
            context.robot_configuration, role="robot_configuration"
        ),
        "world_configuration": validate_sealed_file_reference(
            context.world_configuration, role="world_configuration"
        ),
        "task_trajectory": validate_sealed_file_reference(
            context.task_trajectory, role="task_trajectory"
        ),
        "analytic_candidate_inventory": validate_sealed_file_reference(
            context.analytic_candidate_inventory,
            role="analytic_candidate_inventory",
        ),
        "required_stage_kinds": list(REQUIRED_STAGE_KINDS),
        "request_digest": "",
    }
    request["request_digest"] = canonical_digest(
        request, digest_field="request_digest"
    )
    return request


def _validate_backend_identity(
    value: object, *, expected: Mapping[str, Any]
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or _copy(value) != _copy(expected):
        raise CollisionAwareCandidateGenerationError(
            "candidate_generation_backend_identity_mismatch"
        )
    return _copy(value)


def validate_runtime_probe(
    value: Mapping[str, Any], *, expected_backend_identity: Mapping[str, Any]
) -> dict[str, Any]:
    probe = _copy(value)
    if (
        probe.get("schema_version") != RUNTIME_PROBE_SCHEMA_VERSION
        or probe.get("runtime_ready") is not True
        or probe.get("probe_digest")
        != canonical_digest(probe, digest_field="probe_digest")
    ):
        raise CollisionAwareCandidateGenerationError(
            "candidate_generator_runtime_unavailable"
        )
    _validate_backend_identity(
        probe.get("backend_identity"), expected=expected_backend_identity
    )
    return probe


def _ordered_required_stages(stages: Sequence[Mapping[str, Any]]) -> bool:
    cursor = 0
    for row in stages:
        if cursor < len(REQUIRED_STAGE_KINDS) and row.get("stage_kind") == REQUIRED_STAGE_KINDS[cursor]:
            cursor += 1
    return cursor == len(REQUIRED_STAGE_KINDS)


def validate_candidate_generation_result(
    value: Mapping[str, Any],
    *,
    request: Mapping[str, Any],
    expected_backend_identity: Mapping[str, Any],
) -> dict[str, Any]:
    result = _copy(value)
    raw_solutions = result.get("solutions")
    if (
        result.get("schema_version") != RESULT_SCHEMA_VERSION
        or result.get("request_digest") != request.get("request_digest")
        or result.get("result_digest")
        != canonical_digest(result, digest_field="result_digest")
        or not isinstance(raw_solutions, list)
        or not 1 <= len(raw_solutions) <= int(request["maximum_candidates"])
    ):
        raise CollisionAwareCandidateGenerationError(
            "candidate_generation_result_invalid"
        )
    _validate_backend_identity(
        result.get("backend_identity"), expected=expected_backend_identity
    )
    solution_ids: list[str] = []
    for solution in raw_solutions:
        if not isinstance(solution, Mapping):
            raise CollisionAwareCandidateGenerationError(
                "candidate_generation_solution_invalid"
            )
        row = dict(solution)
        stages = row.get("stages")
        pose = row.get("robot_base_pose_world")
        reset = row.get("robot_joint_reset_positions_rad")
        cameras = row.get("cameras")
        addressed = row.get("addressed_feedback_codes")
        if (
            not str(row.get("solution_id") or "")
            or not isinstance(row.get("deterministic_rank"), int)
            or int(row["deterministic_rank"]) < 0
            or not isinstance(stages, list)
            or not stages
            or any(not isinstance(stage, Mapping) for stage in stages)
            or not _ordered_required_stages(stages)
            or row.get("solution_digest")
            != canonical_digest(row, digest_field="solution_digest")
            or not isinstance(pose, Mapping)
            or not isinstance(reset, Mapping)
            or not reset
            or not isinstance(cameras, list)
            or not cameras
            or any(not isinstance(camera, Mapping) for camera in cameras)
            or not isinstance(addressed, list)
            or any(code not in request["addressable_feedback_codes"] for code in addressed)
            or not str(row.get("support_surface_id") or "")
            or not str(row.get("joins_authored_phase_id") or "")
            or row.get("joint_limit_compliance_observed") is not True
            or row.get("collision_aware_motion_generated") is not True
        ):
            raise CollisionAwareCandidateGenerationError(
                "candidate_generation_solution_invalid"
            )
        for stage in stages:
            waypoints = stage.get("waypoints")
            if (
                not str(stage.get("stage_id") or "")
                or stage.get("stage_kind") not in REQUIRED_STAGE_KINDS
                or not isinstance(waypoints, list)
                or not waypoints
                or any(not isinstance(item, Mapping) for item in waypoints)
            ):
                raise CollisionAwareCandidateGenerationError(
                    "candidate_generation_solution_stage_invalid"
                )
        try:
            position = [float(value) for value in pose.get("position_world_m")]
            orientation = [float(value) for value in pose.get("orientation_xyzw")]
            reset_values = [float(value) for value in reset.values()]
        except (TypeError, ValueError) as exc:
            raise CollisionAwareCandidateGenerationError(
                "candidate_generation_solution_invalid"
            ) from exc
        if (
            len(position) != 3
            or len(orientation) != 4
            or not all(math.isfinite(value) for value in [*position, *orientation, *reset_values])
            or not math.isclose(
                math.sqrt(math.fsum(value * value for value in orientation)),
                1.0,
                rel_tol=0.0,
                abs_tol=1.0e-4,
            )
            or not all(str(name) for name in reset)
        ):
            raise CollisionAwareCandidateGenerationError(
                "candidate_generation_solution_invalid"
            )
        solution_ids.append(str(row["solution_id"]))
    if len(set(solution_ids)) != len(solution_ids):
        raise CollisionAwareCandidateGenerationError(
            "candidate_generation_solution_duplicate"
        )
    return result


def _sealed(value: dict[str, Any], *, field: str) -> dict[str, Any]:
    value[field] = canonical_digest(value, digest_field=field)
    return value


def _candidate_from_solution(
    *,
    solution: Mapping[str, Any],
    request: Mapping[str, Any],
    backend_identity: Mapping[str, Any],
) -> dict[str, Any]:
    reset = _sealed(
        {
            "schema_version": "task_evaluation_native_robot_reset_variant.v1",
            "robot_joint_reset_positions_rad": dict(
                solution["robot_joint_reset_positions_rad"]
            ),
            "reset_variant_digest": "",
        },
        field="reset_variant_digest",
    )
    waypoints = []
    for stage in solution["stages"]:
        for waypoint in stage["waypoints"]:
            row = _copy(waypoint)
            row["stage_id"] = str(stage["stage_id"])
            row["stage_kind"] = str(stage["stage_kind"])
            waypoints.append(row)
    entry = _sealed(
        {
            "schema_version": "task_evaluation_native_entry_trajectory_variant.v1",
            "joins_authored_phase_id": str(solution["joins_authored_phase_id"]),
            "waypoints": waypoints,
            "entry_trajectory_variant_digest": "",
        },
        field="entry_trajectory_variant_digest",
    )
    camera = _sealed(
        {
            "schema_version": "task_evaluation_native_camera_variant.v1",
            "cameras": [dict(row) for row in solution["cameras"]],
            "camera_variant_digest": "",
        },
        field="camera_variant_digest",
    )
    evidence = {
        "backend_identity": _copy(backend_identity),
        "generation_request_digest": request["request_digest"],
        "solver_solution_id": solution["solution_id"],
        "solver_solution_digest": solution["solution_digest"],
        "minimum_world_clearance_m": _finite(
            solution["minimum_world_clearance_m"],
            blocker="candidate_generation_solution_evidence_invalid",
        ),
        "minimum_self_clearance_m": _finite(
            solution["minimum_self_clearance_m"],
            blocker="candidate_generation_solution_evidence_invalid",
        ),
        "joint_limit_compliance_observed": solution.get(
            "joint_limit_compliance_observed"
        )
        is True,
        "collision_aware_motion_generated": solution.get(
            "collision_aware_motion_generated"
        )
        is True,
        "native_requirements_unresolved": [
            "orientation_execution",
            "collision_and_contact_readback",
            "camera_observability",
            "task_execution",
        ],
    }
    evidence["solver_evidence_digest"] = canonical_digest(evidence)
    candidate = {
        "schema_version": CANDIDATE_SCHEMA_VERSION,
        "candidate_id": (
            f"{backend_identity['backend_id']}-r{request['round_index']}-"
            f"{solution['solution_id']}"
        ),
        "deterministic_rank": int(solution["deterministic_rank"]),
        "robot_base_pose_world": _copy(solution["robot_base_pose_world"]),
        "support_surface_id": str(solution["support_surface_id"]),
        "reset_variant": reset,
        "entry_trajectory_variant": entry,
        "camera_variant": camera,
        "generation_evidence": evidence,
        "maximum_incremental_cost_usd": request["maximum_incremental_cost_usd"],
        "maximum_runtime_seconds": request["maximum_runtime_seconds"],
        "addressed_feedback_codes": list(solution["addressed_feedback_codes"]),
        "candidate_digest": "",
    }
    if _contains_forbidden_candidate_key(candidate):
        raise CollisionAwareCandidateGenerationError(
            "candidate_generation_native_criteria_mutation_forbidden"
        )
    return _sealed(candidate, field="candidate_digest")


def build_native_candidate_inventory(
    *,
    result: Mapping[str, Any],
    request: Mapping[str, Any],
    backend_identity: Mapping[str, Any],
) -> dict[str, Any]:
    validated = validate_candidate_generation_result(
        result,
        request=request,
        expected_backend_identity=backend_identity,
    )
    candidates = [
        _candidate_from_solution(
            solution=row,
            request=request,
            backend_identity=backend_identity,
        )
        for row in validated["solutions"]
    ]
    candidates.sort(key=lambda row: (row["deterministic_rank"], row["candidate_id"]))
    inventory = {
        "schema_version": INVENTORY_SCHEMA_VERSION,
        "run_id": request["run_id"],
        "round_index": request["round_index"],
        "source_native_feedback_digest": request["source_native_feedback_digest"],
        "model_authored_candidates": False,
        "generator_backend": _copy(backend_identity),
        "generation_request_digest": request["request_digest"],
        "solver_result_digest": validated["result_digest"],
        "native_requirements_unresolved": [
            "orientation_execution",
            "collision_and_contact_readback",
            "camera_observability",
            "task_execution",
        ],
        "candidates": candidates,
        "inventory_digest": "",
    }
    return _sealed(inventory, field="inventory_digest")


class JsonProcessCandidateGenerator:
    """Invoke one pinned backend process and validate its complete result."""

    def __init__(
        self,
        *,
        context: CandidateGeneratorContext,
        backend_identity: Mapping[str, Any],
        command: Sequence[str],
        require_cuda: bool,
        environment: Mapping[str, str] | None = None,
        runner: CommandRunner = subprocess.run,
    ) -> None:
        self._context = context
        self._backend_identity = _copy(backend_identity)
        self._command = tuple(str(item) for item in command)
        self._require_cuda = require_cuda
        self._environment = {
            str(key): str(value) for key, value in (environment or {}).items()
        }
        self._runner = runner
        if not self._command or not os.path.isabs(self._command[0]):
            raise CollisionAwareCandidateGenerationError(
                "candidate_generator_command_invalid"
            )

    def _invoke(self, *, request: Mapping[str, Any] | None) -> dict[str, Any]:
        with tempfile.TemporaryDirectory(prefix="blueprint-candidate-generator-") as temp:
            root = Path(temp)
            output_path = root / "output.json"
            argv = [*self._command]
            if request is None:
                argv.extend(("--probe", "--result-json", str(output_path)))
            else:
                request_path = root / "request.json"
                request_path.write_text(canonical_json(dict(request)) + "\n", encoding="utf-8")
                argv.extend(
                    (
                        "--request-json",
                        str(request_path),
                        "--result-json",
                        str(output_path),
                    )
                )
            try:
                completed = self._runner(
                    argv,
                    check=False,
                    text=True,
                    capture_output=True,
                    timeout=max(30.0, self._context.maximum_runtime_seconds + 30.0),
                    cwd="/tmp",
                    env={**os.environ, **self._environment},
                )
            except (OSError, subprocess.SubprocessError) as exc:
                raise CollisionAwareCandidateGenerationError(
                    "candidate_generator_runtime_unavailable"
                ) from exc
            if completed.returncode != 0 or not output_path.is_file() or output_path.is_symlink():
                raise CollisionAwareCandidateGenerationError(
                    "candidate_generator_process_failed"
                )
            try:
                value = json.loads(output_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise CollisionAwareCandidateGenerationError(
                    "candidate_generator_process_result_invalid"
                ) from exc
            if not isinstance(value, Mapping):
                raise CollisionAwareCandidateGenerationError(
                    "candidate_generator_process_result_invalid"
                )
            return dict(value)

    def generate(
        self,
        *,
        source_native_feedback: Mapping[str, Any] | None,
        prior_history: Sequence[Mapping[str, Any]],
        round_index: int,
        maximum_candidates: int,
    ) -> Mapping[str, Any]:
        probe = validate_runtime_probe(
            self._invoke(request=None),
            expected_backend_identity=self._backend_identity,
        )
        if self._require_cuda and (
            probe.get("cuda_available") is not True
            or int(probe.get("cuda_device_count") or 0) < 1
        ):
            raise CollisionAwareCandidateGenerationError(
                "candidate_generator_cuda_unavailable"
            )
        request = build_candidate_generation_request(
            context=self._context,
            backend_identity=self._backend_identity,
            source_native_feedback=source_native_feedback,
            prior_history=prior_history,
            round_index=round_index,
            maximum_candidates=maximum_candidates,
        )
        result = self._invoke(request=request)
        return build_native_candidate_inventory(
            result=result,
            request=request,
            backend_identity=self._backend_identity,
        )


__all__ = [
    "CANDIDATE_SCHEMA_VERSION",
    "CandidateGenerator",
    "CandidateGeneratorContext",
    "CollisionAwareCandidateGenerationError",
    "INVENTORY_SCHEMA_VERSION",
    "JsonProcessCandidateGenerator",
    "REQUEST_SCHEMA_VERSION",
    "REQUIRED_STAGE_KINDS",
    "RESULT_SCHEMA_VERSION",
    "RUNTIME_PROBE_SCHEMA_VERSION",
    "build_candidate_generation_request",
    "build_native_candidate_inventory",
    "validate_candidate_generation_result",
    "validate_runtime_probe",
    "validate_sealed_file_reference",
]
