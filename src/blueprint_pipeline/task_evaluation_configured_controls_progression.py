"""Fail-closed progression from one configured scene to native controls.

This module owns the narrow joins between existing production queues.  It does
not execute a provider, issue authority, invent robot registration, or grade a
control result.  Each transition validates the exact predecessor and stages at
most one immutable successor request.  Repeating the same transition is safe
because the existing preparation and activation queues enforce immutable
identities.

The progression intentionally has two paid launches after configuration:
native scene construction first, then the canonical zero-action plus
deterministic-scripted-positive control pair.  Controls cannot borrow the
configuration GPU's robot-neutral result as construction evidence.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest
from .task_evaluation_configured_scene_revision import (
    TaskEvaluationConfiguredSceneRevisionError,
    validate_configured_scene_revision,
)
from .task_evaluation_launch_activation_contract import (
    validate_launch_activation_request,
)
from .task_evaluation_launch_preparation_contract import (
    launch_preparation_request_digest,
    validate_launch_preparation_request,
)


PROGRESSION_SCHEMA_VERSION = "task_evaluation_configured_controls_progression.v1"
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
_COMMIT = re.compile(r"[0-9a-f]{40}")
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,191}")

ReferencePublisher = Callable[..., Mapping[str, Any]]
PreparationStager = Callable[..., Mapping[str, Any]]
ActivationStager = Callable[..., Mapping[str, Any]]
LaunchSubmitter = Callable[[Mapping[str, Any]], Mapping[str, Any]]
ReadinessMaterializer = Callable[..., Mapping[str, Any]]


class TaskEvaluationConfiguredControlsProgressionError(RuntimeError):
    """The configured-scene controls progression could not advance safely."""


def _copy(value: Mapping[str, Any], *, blocker: str) -> dict[str, Any]:
    try:
        return json.loads(json.dumps(dict(value), allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise TaskEvaluationConfiguredControlsProgressionError(blocker) from exc


def _reference(value: Any, *, blocker: str) -> dict[str, Any]:
    reference = dict(value) if isinstance(value, Mapping) else {}
    if (
        set(reference) != {"uri", "digest", "size_bytes"}
        or not str(reference.get("uri") or "").startswith(("s3://", "gs://", "https://"))
        or _DIGEST.fullmatch(str(reference.get("digest") or "")) is None
        or not isinstance(reference.get("size_bytes"), int)
        or isinstance(reference.get("size_bytes"), bool)
        or reference["size_bytes"] < 1
    ):
        raise TaskEvaluationConfiguredControlsProgressionError(blocker)
    return reference


def _sha256_and_size(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return "sha256:" + digest.hexdigest(), size


def _sealed(value: dict[str, Any], *, field: str) -> dict[str, Any]:
    value[field] = ""
    value[field] = canonical_digest(value, digest_field=field)
    return value


def _validate_configuration_predecessor(
    *,
    terminal_result: Mapping[str, Any],
    publication_result: Mapping[str, Any],
    configured_revision: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    terminal = _copy(
        terminal_result,
        blocker="configured_controls_progression_terminal_invalid",
    )
    publication = _copy(
        publication_result,
        blocker="configured_controls_progression_publication_invalid",
    )
    try:
        revision = validate_configured_scene_revision(configured_revision)
    except TaskEvaluationConfiguredSceneRevisionError as exc:
        raise TaskEvaluationConfiguredControlsProgressionError(
            "configured_controls_progression_revision_invalid"
        ) from exc
    offering = publication.get("configured_scene_offering")
    binding = offering.get("evaluation_preparation_binding") if isinstance(offering, Mapping) else None
    admission = offering.get("evaluation_admission") if isinstance(offering, Mapping) else None
    finalization = terminal.get("scene_construction_queue_finalization")
    if (
        terminal.get("schema_version")
        != "task_evaluation_scene_configuration_vast_result.v1"
        or terminal.get("status") != "completed"
        or terminal.get("configuration_completed") is not True
        or terminal.get("configured_scene_published") is not True
        or terminal.get("full_byte_service_account_readback_passed") is not True
        or terminal.get("continuing_spend_from_this_run") is not False
        or terminal.get("provider_mutations_performed") != 1
        or terminal.get("retry_cap") != 0
        or terminal.get("evaluation_episode_executed") is not False
        or terminal.get("candidate_policy_queried") is not False
        or terminal.get("blockers") not in ([], ())
        or terminal.get("result_digest")
        != canonical_digest(terminal, digest_field="result_digest")
        or not isinstance(finalization, Mapping)
        or finalization.get("status") != "completed"
        or finalization.get("queue_state") != "completed"
        or finalization.get("finalization_performed") is not True
        or finalization.get("result_digest")
        != canonical_digest(finalization, digest_field="result_digest")
    ):
        raise TaskEvaluationConfiguredControlsProgressionError(
            "configured_controls_progression_qualifying_configuration_missing"
        )
    if (
        publication.get("schema_version")
        != "task_evaluation_scene_configuration_publication.v1"
        or publication.get("status") != "configured_scene_published"
        or publication.get("full_byte_service_account_readback_passed") is not True
        or publication.get("provider_mutation_performed") is not False
        or publication.get("paid_execution_requested") is not False
        or publication.get("result_digest")
        != canonical_digest(publication, digest_field="result_digest")
        or publication.get("configured_scene_revision_digest") != revision["revision_digest"]
        or terminal.get("configured_scene_revision_digest") != revision["revision_digest"]
        or terminal.get("publication_result_digest") != publication["result_digest"]
        or terminal.get("run_id") != revision["configuration_run_id"]
        or revision.get("status") != "configured"
        or revision.get("evaluation_admission")
        != {
            "zero_action_required": True,
            "scripted_positive_required": True,
            "learned_policy_admitted": False,
        }
        or not isinstance(offering, Mapping)
        or offering.get("schema_version")
        != "task_evaluation_configured_scene_offering.v1"
        or offering.get("status") != "configured_controls_pending"
        or offering.get("configuration_run_id") != revision["configuration_run_id"]
        or offering.get("team_namespace") != revision["team_namespace"]
        or offering.get("offering_digest")
        != canonical_digest(offering, digest_field="offering_digest")
        or not isinstance(binding, Mapping)
        or binding.get("configured_scene_revision_digest") != revision["revision_digest"]
        or binding.get("configured_scene_bundle") != revision["configured_scene_bundle"]
        or binding.get("configured_scene_revision")
        != publication.get("configured_scene_revision_reference")
        or admission
        != {
            "zero_action_required": True,
            "scripted_positive_required": True,
            "learned_policy_evaluation_admitted": False,
        }
    ):
        raise TaskEvaluationConfiguredControlsProgressionError(
            "configured_controls_progression_publication_binding_invalid"
        )
    _reference(
        publication.get("configured_scene_revision_reference"),
        blocker="configured_controls_progression_revision_reference_invalid",
    )
    return terminal, publication, revision


def _default_readiness_materializer(**kwargs: Any) -> Mapping[str, Any]:
    # Kept inside the call so this PR may be reviewed independently while the
    # canonical Franka input materializer lands in its prerequisite PR.
    from .task_evaluation_franka_robotiq_readiness_inputs import (
        materialize_franka_robotiq_readiness_inputs,
    )

    return materialize_franka_robotiq_readiness_inputs(**kwargs)


def _publish_materialized_file(
    *,
    path: Path,
    object_name: str,
    publisher: ReferencePublisher,
) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise TaskEvaluationConfiguredControlsProgressionError(
            "configured_controls_progression_materialized_file_invalid"
        )
    expected_digest, expected_size = _sha256_and_size(path)
    observed = dict(publisher(path=path, object_name=object_name))
    reference = {
        key: observed.get(key) for key in ("uri", "digest", "size_bytes")
    }
    if (
        _reference(
            reference,
            blocker="configured_controls_progression_publication_readback_invalid",
        )
        != reference
        or reference["digest"] != expected_digest
        or reference["size_bytes"] != expected_size
        or observed.get("full_byte_service_account_readback_passed") is not True
        or observed.get("readback_digest") != expected_digest
        or observed.get("readback_size_bytes") != expected_size
    ):
        raise TaskEvaluationConfiguredControlsProgressionError(
            "configured_controls_progression_publication_readback_invalid"
        )
    return reference


def stage_configured_controls_episode_preparation(
    *,
    terminal_result: Mapping[str, Any],
    publication_result: Mapping[str, Any],
    configured_revision: Mapping[str, Any],
    expected_production_commit: str,
    robot_mount_interface_path: str | Path,
    scene_camera_calibration_path: str | Path,
    base_pose_candidate: Mapping[str, Any],
    cameras: Sequence[Mapping[str, Any]],
    runtime_binding: Mapping[str, Any],
    output_root: str | Path,
    publisher: ReferencePublisher,
    queue_root: str | Path,
    submitted_by: str,
    readiness_materializer: ReadinessMaterializer = _default_readiness_materializer,
    preparation_stager: PreparationStager | None = None,
) -> dict[str, Any]:
    """Materialize canonical inputs and queue one no-spend episode compilation."""

    _, publication, revision = _validate_configuration_predecessor(
        terminal_result=terminal_result,
        publication_result=publication_result,
        configured_revision=configured_revision,
    )
    if _COMMIT.fullmatch(expected_production_commit) is None:
        raise TaskEvaluationConfiguredControlsProgressionError(
            "configured_controls_progression_evaluator_commit_invalid"
        )
    runtime = _copy(
        runtime_binding,
        blocker="configured_controls_progression_runtime_binding_invalid",
    )
    if set(runtime) != {"runtime", "execution_adapter", "spend"}:
        raise TaskEvaluationConfiguredControlsProgressionError(
            "configured_controls_progression_runtime_binding_invalid"
        )
    progression_input_digest = canonical_digest(
        {
            "configuration_terminal_result_digest": terminal_result.get(
                "result_digest"
            ),
            "publication_result_digest": publication["result_digest"],
            "configured_scene_revision_digest": revision["revision_digest"],
            "expected_production_commit": expected_production_commit,
            "base_pose_candidate": _copy(
                base_pose_candidate,
                blocker="configured_controls_progression_base_pose_invalid",
            ),
            "cameras": _copy(
                {"rows": list(cameras)},
                blocker="configured_controls_progression_cameras_invalid",
            )["rows"],
            "runtime_binding": runtime,
        }
    )
    namespace = (
        f"{revision['configuration_run_id']}-franka-controls-"
        f"{expected_production_commit[:12]}"
    )
    if _IDENTIFIER.fullmatch(namespace) is None:
        raise TaskEvaluationConfiguredControlsProgressionError(
            "configured_controls_progression_identity_invalid"
        )
    root = Path(output_root).expanduser()
    if root.is_symlink():
        raise TaskEvaluationConfiguredControlsProgressionError(
            "configured_controls_progression_output_root_unsafe"
        )
    root.mkdir(parents=True, exist_ok=True, mode=0o750)
    root = root.resolve()
    receipt_path = root / "configured_controls_progression.v1.json"
    if receipt_path.is_file() and not receipt_path.is_symlink():
        try:
            existing = json.loads(receipt_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise TaskEvaluationConfiguredControlsProgressionError(
                "configured_controls_progression_existing_receipt_invalid"
            ) from exc
        if (
            not isinstance(existing, Mapping)
            or existing.get("schema_version") != PROGRESSION_SCHEMA_VERSION
            or existing.get("status") != "episode_preparation_queued"
            or existing.get("progression_input_digest")
            != progression_input_digest
            or existing.get("progression_digest")
            != canonical_digest(existing, digest_field="progression_digest")
        ):
            raise TaskEvaluationConfiguredControlsProgressionError(
                "configured_controls_progression_immutable_conflict"
            )
        return dict(existing)
    if any(root.iterdir()):
        raise TaskEvaluationConfiguredControlsProgressionError(
            "configured_controls_progression_output_root_not_empty"
        )
    controller_identity = {
        "id": "canonical-planar-push-control-pair",
        "version": "zero-action-plus-scripted-positive-v1",
    }
    materialized = dict(
        readiness_materializer(
            configured_revision=revision,
            robot_mount_interface_path=robot_mount_interface_path,
            scene_camera_calibration_path=scene_camera_calibration_path,
            base_pose_candidate=base_pose_candidate,
            cameras=cameras,
            controller_identity=controller_identity,
            # The positive controller is the actuator-bearing controller.  The
            # native controls lane independently injects the zero-action
            # negative and requires both outcomes from the same packet.
            controller_kind="deterministic_scripted",
            output_root=root,
        )
    )
    files = materialized.get("files")
    robot_identity = materialized.get("robot_identity")
    if (
        materialized.get("status")
        != "materialized_candidate_pending_native_construction_readback"
        or materialized.get("configured_scene_revision_digest")
        != revision["revision_digest"]
        or materialized.get("robot_base_qualified") is not False
        or materialized.get("camera_configuration_qualified") is not False
        or materialized.get("native_construction_readback_required") is not True
        or materialized.get("candidate_policy_queried") is not False
        or not isinstance(files, Mapping)
        or not isinstance(robot_identity, Mapping)
    ):
        raise TaskEvaluationConfiguredControlsProgressionError(
            "configured_controls_progression_readiness_inputs_invalid"
        )
    expected_roles = {
        "robot_configuration",
        "robot_kinematics",
        "robot_joint_bounds",
        "robot_base_registration",
        "controller_configuration",
        "sensor_configuration",
    }
    if set(files) != expected_roles:
        raise TaskEvaluationConfiguredControlsProgressionError(
            "configured_controls_progression_readiness_inputs_invalid"
        )
    published: dict[str, dict[str, Any]] = {}
    for role in sorted(expected_roles):
        record = files[role]
        path = Path(str(record.get("path") if isinstance(record, Mapping) else ""))
        if (
            not isinstance(record, Mapping)
            or _sha256_and_size(path) != (record.get("digest"), record.get("size_bytes"))
        ):
            raise TaskEvaluationConfiguredControlsProgressionError(
                "configured_controls_progression_readiness_inputs_invalid"
            )
        published[role] = _publish_materialized_file(
            path=path,
            object_name=f"{namespace}/readiness/{path.name}",
            publisher=publisher,
        )
    offering = publication["configured_scene_offering"]
    task = offering["task"]
    request: dict[str, Any] = {
        "schema_version": "task_evaluation_launch_preparation_request.v1",
        "run_mode": "episode_evaluation",
        "expected_production_commit": expected_production_commit,
        "preparation_id": namespace + "-preparation",
        "team_namespace": revision["team_namespace"],
        "run_id": namespace + "-episode",
        "scene": {
            "mode": "reuse_configured_revision",
            "identity": dict(revision["scene_identity"]),
            "configured_revision": dict(
                publication["configured_scene_revision_reference"]
            ),
        },
        "construction": {"mode": "reuse_configured_scene"},
        "robot": {
            "identity": dict(robot_identity),
            "configuration": published["robot_configuration"],
            "kinematics": published["robot_kinematics"],
            "joint_bounds": published["robot_joint_bounds"],
            "base_registration": published["robot_base_registration"],
            "controller_configuration": published["controller_configuration"],
        },
        "controller": {
            "identity": controller_identity,
            "kind": "deterministic_scripted",
            "configuration": published["controller_configuration"],
        },
        "task": {
            "identity": dict(task["identity"]),
            "binding_mode": "reuse_configured_template",
            "kind": task["kind"],
            "strategy": task["strategy"],
            "configured_scene_revision_digest": revision["revision_digest"],
            "subject": {
                "mode": "configured_scene_object",
                "identity": dict(task["subject_identity"]),
                "physics_authority": "configured_scene_revision",
            },
        },
        "sensors": {"configuration": published["sensor_configuration"]},
        "runtime": runtime["runtime"],
        "execution_adapter": runtime["execution_adapter"],
        "publication": {
            "input_namespace": namespace,
            "service_account_readback_required": True,
        },
        "spend": runtime["spend"],
    }
    request = validate_launch_preparation_request(request)
    if preparation_stager is None:
        from .task_evaluation_launch_preparation_queue import (
            stage_launch_preparation_request,
        )

        preparation_stager = stage_launch_preparation_request
    intake = dict(
        preparation_stager(
            value=request,
            queue_root=queue_root,
            submitted_by=submitted_by,
        )
    )
    if (
        intake.get("status") != "queued_for_no_spend_preparation"
        or intake.get("accepted") is not True
        or intake.get("preparation_id") != request["preparation_id"]
        or intake.get("request_digest") != launch_preparation_request_digest(request)
        or intake.get("provider_mutation_performed_inside_http_request") is not False
        or intake.get("catalog_mutation_performed_inside_http_request") is not False
        or intake.get("paid_execution_requested") is not False
    ):
        raise TaskEvaluationConfiguredControlsProgressionError(
            "configured_controls_progression_preparation_intake_invalid"
        )
    result = {
        "schema_version": PROGRESSION_SCHEMA_VERSION,
        "status": "episode_preparation_queued",
        "configuration_run_id": revision["configuration_run_id"],
        "configured_scene_revision_digest": revision["revision_digest"],
        "configured_scene_offering_digest": offering["offering_digest"],
        "progression_input_digest": progression_input_digest,
        "expected_production_commit": expected_production_commit,
        "episode_preparation_request": request,
        "episode_preparation_request_digest": launch_preparation_request_digest(request),
        "episode_preparation_intake_receipt_digest": intake["receipt_digest"],
        "robot_base_qualified": False,
        "camera_configuration_qualified": False,
        "native_construction_readback_required": True,
        "zero_action_required": True,
        "scripted_positive_required": True,
        "candidate_policy_queried": False,
        "provider_mutation_performed": False,
        "paid_execution_requested": False,
        "progression_digest": "",
    }
    result = _sealed(result, field="progression_digest")
    try:
        with receipt_path.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, sort_keys=True, separators=(",", ":"))
            stream.write("\n")
        receipt_path.chmod(0o440)
    except FileExistsError as exc:
        raise TaskEvaluationConfiguredControlsProgressionError(
            "configured_controls_progression_immutable_conflict"
        ) from exc
    return result


def _activation_request(
    *,
    lane: str,
    expected_production_commit: str,
    activation_id: str,
    team_namespace: str,
    preparation: Mapping[str, Any],
    release_window: Mapping[str, Any],
    lineage: Mapping[str, Any],
    authorization: Mapping[str, Any],
) -> dict[str, Any]:
    result = {
        "schema_version": "task_evaluation_launch_activation_request.v1",
        "expected_production_commit": expected_production_commit,
        "activation_id": activation_id,
        "team_namespace": team_namespace,
        "lane": lane,
        "preparation": dict(preparation),
        "release_window": _reference(
            release_window,
            blocker="configured_controls_progression_release_window_invalid",
        ),
        "lineage": _copy(
            lineage,
            blocker="configured_controls_progression_lineage_invalid",
        ),
        "authorization": _copy(
            authorization,
            blocker="configured_controls_progression_authorization_invalid",
        ),
        "requested_mutations": {
            "profile_publication": True,
            "catalog_synchronization": True,
            "standing_authorization": True,
        },
    }
    return validate_launch_activation_request(result)


def _validate_controls_predecessor_artifacts(
    *,
    lineage: Mapping[str, Any],
    artifact_paths: Mapping[str, str | Path] | None,
) -> None:
    required = {
        "prior_authority",
        "prior_result",
        "prior_launch_receipt",
        "prior_webapp_sync",
        "prior_provider_zero",
        "prior_spend_reconciliation",
        "construction_result",
    }
    if artifact_paths is None or set(artifact_paths) != required:
        raise TaskEvaluationConfiguredControlsProgressionError(
            "configured_controls_progression_controls_predecessor_missing"
        )
    values: dict[str, dict[str, Any]] = {}
    for name in sorted(required):
        reference = _reference(
            lineage.get(name),
            blocker="configured_controls_progression_controls_predecessor_invalid",
        )
        path = Path(artifact_paths[name]).expanduser()
        if (
            path.is_symlink()
            or not path.is_file()
            or _sha256_and_size(path)
            != (reference["digest"], reference["size_bytes"])
        ):
            raise TaskEvaluationConfiguredControlsProgressionError(
                "configured_controls_progression_controls_predecessor_invalid"
            )
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise TaskEvaluationConfiguredControlsProgressionError(
                "configured_controls_progression_controls_predecessor_invalid"
            ) from exc
        if not isinstance(value, Mapping):
            raise TaskEvaluationConfiguredControlsProgressionError(
                "configured_controls_progression_controls_predecessor_invalid"
            )
        values[name] = dict(value)
    construction = values["construction_result"]
    if (
        construction.get("schema_version")
        != "native_task_arena_construction_result.v1"
        or construction.get("status") != "completed"
        or construction.get("construction_gate_qualified") is not True
        or construction.get("blockers") not in ([], ())
        or construction.get("candidate_policy_queried") is not False
        or construction.get("result_digest")
        != canonical_digest(construction, digest_field="result_digest")
    ):
        raise TaskEvaluationConfiguredControlsProgressionError(
            "configured_controls_progression_construction_not_qualified"
        )


def build_configured_controls_activation_request(
    *,
    progression: Mapping[str, Any],
    preparation_result: Mapping[str, Any],
    release_window: Mapping[str, Any],
    lineage: Mapping[str, Any],
    authorization: Mapping[str, Any],
    lane: str,
    lineage_artifact_paths: Mapping[str, str | Path] | None = None,
) -> dict[str, Any]:
    """Build the exact request so coordinator authority can bind its intent."""

    state = _copy(
        progression,
        blocker="configured_controls_progression_state_invalid",
    )
    preparation = _copy(
        preparation_result,
        blocker="configured_controls_progression_preparation_result_invalid",
    )
    if (
        state.get("schema_version") != PROGRESSION_SCHEMA_VERSION
        or state.get("status") != "episode_preparation_queued"
        or state.get("progression_digest")
        != canonical_digest(state, digest_field="progression_digest")
        or preparation.get("schema_version")
        != "task_evaluation_launch_preparation_result.v1"
        or preparation.get("status") != "queued_for_production_episode_compilation"
        or preparation.get("run_mode") != "episode_evaluation"
        or preparation.get("configured_scene_revision_digest")
        != state.get("configured_scene_revision_digest")
        or preparation.get("automatic_progression_required") is not True
        or preparation.get("provider_mutation_performed") is not False
        or preparation.get("paid_execution_requested") is not False
        or preparation.get("result_digest")
        != canonical_digest(preparation, digest_field="result_digest")
        or lane not in {"native_task_arena_construction", "native_task_arena_controls"}
    ):
        raise TaskEvaluationConfiguredControlsProgressionError(
            "configured_controls_progression_preparation_result_invalid"
        )
    expected_lineage_kind = "initial_project" if lane == "native_task_arena_construction" else "predecessor"
    if lineage.get("kind") != expected_lineage_kind:
        raise TaskEvaluationConfiguredControlsProgressionError(
            "configured_controls_progression_lineage_invalid"
        )
    if lane == "native_task_arena_controls":
        _validate_controls_predecessor_artifacts(
            lineage=lineage,
            artifact_paths=lineage_artifact_paths,
        )
    preparation_request = state["episode_preparation_request"]
    return _activation_request(
        lane=lane,
        expected_production_commit=state["expected_production_commit"],
        activation_id=(
            preparation_request["run_id"]
            + ("-construction" if lane == "native_task_arena_construction" else "-controls")
        ),
        team_namespace=preparation_request["team_namespace"],
        preparation={
            "preparation_id": preparation_request["preparation_id"],
            "request_digest": state["episode_preparation_request_digest"],
            "result_digest": preparation["result_digest"],
        },
        release_window=release_window,
        lineage=lineage,
        authorization=authorization,
    )


def stage_configured_controls_activation(
    *,
    progression: Mapping[str, Any],
    preparation_result: Mapping[str, Any],
    release_window: Mapping[str, Any],
    lineage: Mapping[str, Any],
    authorization: Mapping[str, Any],
    lane: str,
    queue_root: str | Path,
    submitted_by: str,
    lineage_artifact_paths: Mapping[str, str | Path] | None = None,
    activation_stager: ActivationStager | None = None,
) -> dict[str, Any]:
    """Queue construction or the combined control pair; never execute either."""

    state = _copy(
        progression,
        blocker="configured_controls_progression_state_invalid",
    )
    preparation = _copy(
        preparation_result,
        blocker="configured_controls_progression_preparation_result_invalid",
    )
    request = build_configured_controls_activation_request(
        progression=progression,
        preparation_result=preparation_result,
        release_window=release_window,
        lineage=lineage,
        authorization=authorization,
        lane=lane,
        lineage_artifact_paths=lineage_artifact_paths,
    )
    if activation_stager is None:
        from .task_evaluation_launch_activation_queue import (
            stage_launch_activation_request,
        )

        activation_stager = stage_launch_activation_request
    intake = dict(
        activation_stager(
            value=request,
            queue_root=queue_root,
            submitted_by=submitted_by,
        )
    )
    if (
        intake.get("status") != "queued_for_authority_gated_activation"
        or intake.get("accepted") is not True
        or intake.get("activation_id") != request["activation_id"]
        or intake.get("lane") != lane
        or intake.get("provider_mutation_performed_inside_http_request") is not False
        or intake.get("paid_execution_requested") is not False
    ):
        raise TaskEvaluationConfiguredControlsProgressionError(
            "configured_controls_progression_activation_intake_invalid"
        )
    result = {
        "schema_version": PROGRESSION_SCHEMA_VERSION,
        "status": (
            "construction_activation_queued"
            if lane == "native_task_arena_construction"
            else "controls_activation_queued"
        ),
        "base_progression_digest": state["progression_digest"],
        "configured_scene_revision_digest": state["configured_scene_revision_digest"],
        "expected_production_commit": state["expected_production_commit"],
        "episode_preparation_request_digest": state[
            "episode_preparation_request_digest"
        ],
        "preparation_result_digest": preparation["result_digest"],
        "lane": lane,
        "activation_request": request,
        "activation_intake_receipt_digest": intake["receipt_digest"],
        "activation_executed_provider": False,
        "provider_mutation_performed": False,
        "paid_execution_requested": False,
        "progression_digest": "",
    }
    return _sealed(result, field="progression_digest")


def build_authorized_webapp_launch_request(
    *,
    activation_progression: Mapping[str, Any],
    activation_result: Mapping[str, Any],
    profile: Mapping[str, Any],
    launch_authority: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the exact launch-only request for canonical WebApp submission."""

    state = _copy(
        activation_progression,
        blocker="configured_controls_progression_activation_state_invalid",
    )
    activation = _copy(
        activation_result,
        blocker="configured_controls_progression_activation_result_invalid",
    )
    launch_profile = _copy(
        profile,
        blocker="configured_controls_progression_profile_invalid",
    )
    authority = _copy(
        launch_authority,
        blocker="configured_controls_progression_launch_authority_invalid",
    )
    if (
        state.get("status")
        not in {"construction_activation_queued", "controls_activation_queued"}
        or state.get("progression_digest")
        != canonical_digest(state, digest_field="progression_digest")
        or activation.get("schema_version")
        != "task_evaluation_launch_activation_result.v1"
        or activation.get("status")
        != "profile_authority_materialized_no_execution"
        or activation.get("activation_id")
        != state.get("activation_request", {}).get("activation_id")
        or activation.get("lane") != state.get("lane")
        or activation.get("source_commit") != state.get("expected_production_commit")
        or activation.get("profile_id") != launch_profile.get("profile_id")
        or activation.get("profile_digest") != launch_profile.get("profile_digest")
        or activation.get("provider_mutation_performed") is not False
        or activation.get("paid_execution_requested") is not False
        or activation.get("blockers") not in ([], ())
        or activation.get("result_digest")
        != canonical_digest(activation, digest_field="result_digest")
        or launch_profile.get("schema_version") != "task_evaluation_launch_profile.v1"
        or launch_profile.get("source_commit") != state.get("expected_production_commit")
        or launch_profile.get("profile_digest")
        != canonical_digest(launch_profile, digest_field="profile_digest")
        or set(authority)
        != {"rights_scope", "rights_evidence", "max_spend_usd", "expires_at"}
    ):
        raise TaskEvaluationConfiguredControlsProgressionError(
            "configured_controls_progression_activation_result_invalid"
        )
    rights = _reference(
        authority["rights_evidence"],
        blocker="configured_controls_progression_launch_authority_invalid",
    )
    max_spend = authority["max_spend_usd"]
    profile_cap = (launch_profile.get("allocator") or {}).get("max_spend_usd")
    if (
        not isinstance(max_spend, (int, float))
        or isinstance(max_spend, bool)
        or not 0 < float(max_spend) <= float(profile_cap or 0)
        or not str(authority["rights_scope"]).strip()
        or not str(authority["expires_at"]).strip()
    ):
        raise TaskEvaluationConfiguredControlsProgressionError(
            "configured_controls_progression_launch_authority_invalid"
        )
    lane = state["lane"]
    launch_id = state["activation_request"]["activation_id"] + "-launch"
    return {
        "confirm_execution": True,
        "launch_id": launch_id,
        "run_id": launch_id,
        "profile_id": launch_profile["profile_id"],
        "profile_digest": launch_profile["profile_digest"],
        "rights": {
            "scope": authority["rights_scope"],
            "evidence": {"uri": rights["uri"], "digest": rights["digest"]},
        },
        "spend": {
            "max_spend_usd": float(max_spend),
            "expires_at": authority["expires_at"],
        },
        "progression": {
            "lane": lane,
            "configured_scene_revision_digest": state[
                "configured_scene_revision_digest"
            ],
            "activation_result_digest": activation["result_digest"],
            "zero_action_required": lane == "native_task_arena_controls",
            "scripted_positive_required": lane == "native_task_arena_controls",
            "provider_mutation_performed_before_webapp_submission": False,
        },
    }


def submit_authorized_progression_launch(
    *,
    activation_progression: Mapping[str, Any],
    activation_result: Mapping[str, Any],
    profile: Mapping[str, Any],
    launch_authority: Mapping[str, Any],
    submitter: LaunchSubmitter,
) -> dict[str, Any]:
    """Submit once through WebApp; the submitter owns signing and transport."""

    request = build_authorized_webapp_launch_request(
        activation_progression=activation_progression,
        activation_result=activation_result,
        profile=profile,
        launch_authority=launch_authority,
    )
    # ``progression`` is retained locally and is not part of the WebApp's
    # intentionally small request contract.
    outbound = {key: value for key, value in request.items() if key != "progression"}
    response = dict(submitter(outbound))
    if (
        response.get("status") not in {"submitted", "accepted", "queued"}
        or response.get("launch_id") != outbound["launch_id"]
        or response.get("provider_mutation_performed_inside_web_request") is not False
    ):
        raise TaskEvaluationConfiguredControlsProgressionError(
            "configured_controls_progression_webapp_submission_invalid"
        )
    result = {
        "schema_version": PROGRESSION_SCHEMA_VERSION,
        "status": (
            "construction_launch_queued"
            if request["progression"]["lane"] == "native_task_arena_construction"
            else "controls_pair_launch_queued"
        ),
        "configured_scene_revision_digest": request["progression"][
            "configured_scene_revision_digest"
        ],
        "activation_result_digest": request["progression"][
            "activation_result_digest"
        ],
        "launch_id": outbound["launch_id"],
        "profile_id": outbound["profile_id"],
        "profile_digest": outbound["profile_digest"],
        "zero_action_required": request["progression"]["zero_action_required"],
        "scripted_positive_required": request["progression"][
            "scripted_positive_required"
        ],
        "submitted_through_webapp": True,
        "provider_mutation_performed_inside_progression": False,
        "paid_execution_requested": True,
        "progression_digest": "",
    }
    return _sealed(result, field="progression_digest")


__all__ = [
    "PROGRESSION_SCHEMA_VERSION",
    "TaskEvaluationConfiguredControlsProgressionError",
    "build_authorized_webapp_launch_request",
    "stage_configured_controls_activation",
    "stage_configured_controls_episode_preparation",
    "submit_authorized_progression_launch",
]
