"""Production-owned automatic progression from configured scenes to controls.

The launch reconciler remains observation-only.  This separate timer worker
consumes an operator-authored immutable plan, a qualifying terminal launch, a
successful WebApp sync, and the reconciler's post-teardown global provider-zero
receipt.  It advances existing no-spend preparation/activation queues and uses
the canonical WebApp-only client for paid launch submission.  It never invokes
an allocator or provider directly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess  # nosec B404 - fixed repository-owned launch-only client
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest, cross_runtime_canonical_digest
from .task_evaluation_configured_controls_progression import (
    PROGRESSION_SCHEMA_VERSION,
    TaskEvaluationConfiguredControlsProgressionError,
    stage_configured_controls_activation,
    stage_configured_controls_episode_preparation,
    submit_authorized_progression_launch,
)
from .task_evaluation_configured_scene_object_store import (
    configured_scene_object_store_publisher,
)
from .task_evaluation_launch_dispatcher import LAUNCH_RECEIPT_DIGEST_CANONICALIZATION
from .task_evaluation_launch_reconciler import validated_succeeded_webapp_sync_row


PLAN_SCHEMA_VERSION = "task_evaluation_configured_controls_progression_plan.v1"
WORKER_RESULT_SCHEMA_VERSION = "task_evaluation_configured_controls_progression_worker.v1"
CONFIGURED_CONTROLS_KEY_PREFIX = (
    "task-evaluation/production-inputs/configured-controls"
)
Submitter = Callable[[Mapping[str, Any]], Mapping[str, Any]]
PublisherFactory = Callable[[], Callable[..., Mapping[str, Any]]]


class TaskEvaluationConfiguredControlsProgressionWorkerError(RuntimeError):
    """The automatic progression worker refused an unsafe transition."""


def configured_controls_object_store_publisher() -> Callable[..., Mapping[str, Any]]:
    """Publish readiness inputs inside the preparation worker's admitted prefix."""

    return configured_scene_object_store_publisher(
        key_prefix=CONFIGURED_CONTROLS_KEY_PREFIX
    )


def _load(path: Path, *, blocker: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TaskEvaluationConfiguredControlsProgressionWorkerError(blocker) from exc
    if path.is_symlink() or not isinstance(value, Mapping):
        raise TaskEvaluationConfiguredControlsProgressionWorkerError(blocker)
    return dict(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _write_immutable(path: Path, value: Mapping[str, Any]) -> None:
    payload = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o750)
    try:
        with path.open("xb") as stream:
            stream.write(payload)
        path.chmod(0o440)
    except FileExistsError:
        if path.is_symlink() or path.read_bytes() != payload:
            raise TaskEvaluationConfiguredControlsProgressionWorkerError(
                "configured_controls_worker_immutable_conflict"
            )


def _sealed_progression(path: Path, *, statuses: set[str]) -> dict[str, Any] | None:
    if not path.exists():
        return None
    value = _load(path, blocker="configured_controls_worker_state_invalid")
    if (
        value.get("schema_version") != PROGRESSION_SCHEMA_VERSION
        or value.get("status") not in statuses
        or value.get("progression_digest")
        != canonical_digest(value, digest_field="progression_digest")
    ):
        raise TaskEvaluationConfiguredControlsProgressionWorkerError(
            "configured_controls_worker_state_invalid"
        )
    return value


def _plan(path: Path) -> dict[str, Any]:
    value = _load(path, blocker="configured_controls_worker_plan_invalid")
    if (
        value.get("schema_version") != PLAN_SCHEMA_VERSION
        or value.get("enabled") is not True
        or value.get("plan_digest") != canonical_digest(value, digest_field="plan_digest")
        or not str(value.get("source_launch_id") or "").strip()
        or not str(value.get("submitted_by") or "").strip()
        or set(value.get("phases") or {}) != {"construction", "controls"}
    ):
        raise TaskEvaluationConfiguredControlsProgressionWorkerError(
            "configured_controls_worker_plan_invalid"
        )
    return value


def _input(path_value: Any, *, blocker: str) -> dict[str, Any]:
    return _load(Path(str(path_value)).expanduser(), blocker=blocker)


def _validate_source(run_root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    receipt = _load(
        run_root / "launch_receipt.json", blocker="configured_controls_worker_launch_receipt_invalid"
    )
    expected = (
        cross_runtime_canonical_digest(receipt, digest_field="receipt_digest")
        if receipt.get("receipt_digest_canonicalization")
        == LAUNCH_RECEIPT_DIGEST_CANONICALIZATION
        else canonical_digest(receipt, digest_field="receipt_digest")
    )
    terminal = receipt.get("terminal_evidence")
    result_artifact = terminal.get("result") if isinstance(terminal, Mapping) else None
    if (
        receipt.get("schema_version") != "task_evaluation_launch_receipt.v1"
        or receipt.get("status") != "completed"
        or receipt.get("receipt_digest") != expected
        or not isinstance(terminal, Mapping)
        or terminal.get("status") != "passed"
        or not isinstance(terminal.get("scene_configuration"), Mapping)
        or not isinstance(result_artifact, Mapping)
        or result_artifact.get("exists") is not True
    ):
        raise TaskEvaluationConfiguredControlsProgressionWorkerError(
            "configured_controls_worker_qualifying_terminal_missing"
        )
    result_path = Path(str(result_artifact.get("path") or "")).expanduser()
    if (
        result_path.is_symlink()
        or not result_path.is_file()
        or _sha256(result_path) != result_artifact.get("digest")
    ):
        raise TaskEvaluationConfiguredControlsProgressionWorkerError(
            "configured_controls_worker_terminal_artifact_invalid"
        )
    sync = _load(
        run_root / "webapp_sync_succeeded.json",
        blocker="configured_controls_worker_webapp_sync_missing",
    )
    try:
        validated_succeeded_webapp_sync_row(receipt=receipt, attempt=sync)
    except Exception as exc:
        raise TaskEvaluationConfiguredControlsProgressionWorkerError(
            "configured_controls_worker_webapp_sync_invalid"
        ) from exc
    zero = _load(
        run_root / "post_teardown_provider_zero_receipt.json",
        blocker="configured_controls_worker_post_teardown_provider_zero_missing",
    )
    if (
        zero.get("schema_version") != "task_evaluation_post_teardown_provider_zero.v1"
        or zero.get("status") != "provider_zero_confirmed"
        or zero.get("provider_zero_verified") is not True
        or zero.get("continuing_spend_from_this_run") is not False
        or zero.get("allocator_invoked") is not False
        or zero.get("provider_mutation_performed") is not False
        or zero.get("automatic_retry_performed") is not False
        or zero.get("blockers") != []
        or zero.get("provider_zero_receipt_digest")
        != canonical_digest(zero, digest_field="provider_zero_receipt_digest")
        or any(
            zero.get(field) != receipt.get(field)
            for field in ("launch_id", "run_id", "request_digest", "receipt_digest", "launch_profile_digest")
        )
    ):
        raise TaskEvaluationConfiguredControlsProgressionWorkerError(
            "configured_controls_worker_post_teardown_provider_zero_invalid"
        )
    return _load(result_path, blocker="configured_controls_worker_terminal_result_invalid"), receipt, zero


def _phase(plan: Mapping[str, Any], name: str) -> dict[str, Any]:
    value = plan["phases"].get(name)
    if not isinstance(value, Mapping):
        raise TaskEvaluationConfiguredControlsProgressionWorkerError(
            "configured_controls_worker_phase_invalid"
        )
    return dict(value)


def _queue_result(queue_root: Path, preparation_id: str) -> dict[str, Any] | None:
    identity = queue_root / "identities" / f"{preparation_id}.json"
    if not identity.exists():
        return None
    matches = list((queue_root / "results").glob(f"{preparation_id}-*.json"))
    if not matches:
        return None
    if len(matches) != 1:
        raise TaskEvaluationConfiguredControlsProgressionWorkerError(
            "configured_controls_worker_preparation_result_ambiguous"
        )
    return _load(matches[0], blocker="configured_controls_worker_preparation_result_invalid")


def _production_submitter(
    *, repo_root: Path, secret_file: Path, endpoint: str, state_root: Path
) -> Submitter:
    def submit(request: Mapping[str, Any]) -> Mapping[str, Any]:
        launch_id = str(request["launch_id"])
        request_path = state_root / f"{launch_id}.webapp-request.json"
        receipt_path = state_root / f"{launch_id}.webapp-submission.json"
        _write_immutable(request_path, request)
        if not receipt_path.exists():
            completed = subprocess.run(  # nosec B603 - fixed Python and repository script
                [
                    sys.executable,
                    str(repo_root / "scripts" / "submit_task_evaluation_launch_via_webapp.py"),
                    "--request", str(request_path),
                    "--secret-file", str(secret_file),
                    "--receipt-out", str(receipt_path),
                    "--endpoint", endpoint,
                ],
                cwd=repo_root,
                check=False,
                capture_output=True,
                text=True,
                timeout=60,
            )
            if completed.returncode != 0:
                raise TaskEvaluationConfiguredControlsProgressionWorkerError(
                    "configured_controls_worker_webapp_submission_failed"
                )
        evidence = _load(receipt_path, blocker="configured_controls_worker_webapp_receipt_invalid")
        web = evidence.get("webapp_receipt")
        if (
            evidence.get("status") not in {"submitted", "replayed"}
            or evidence.get("launch_id") != launch_id
            or not isinstance(web, Mapping)
            or web.get("provider_mutation_performed_inside_web_request") is not False
        ):
            raise TaskEvaluationConfiguredControlsProgressionWorkerError(
                "configured_controls_worker_webapp_receipt_invalid"
            )
        return {
            "status": "submitted" if evidence["status"] == "submitted" else "accepted",
            "launch_id": launch_id,
            "provider_mutation_performed_inside_web_request": False,
        }

    return submit


def advance_configured_controls_plan(
    *,
    plan_path: str | Path,
    launch_state_root: str | Path,
    progression_root: str | Path,
    preparation_queue_root: str | Path,
    activation_queue_root: str | Path,
    publisher_factory: PublisherFactory = configured_controls_object_store_publisher,
    submitter: Submitter | None = None,
    repo_root: str | Path | None = None,
    webapp_secret_file: str | Path | None = None,
    webapp_endpoint: str = "https://tryblueprint.io/api/internal/task-evaluation-launch-submissions",
) -> dict[str, Any]:
    """Advance at most one transition for one immutable progression plan."""

    plan = _plan(Path(plan_path).expanduser())
    run_root = Path(launch_state_root).expanduser() / plan["source_launch_id"]
    state = Path(progression_root).expanduser() / plan["source_launch_id"]
    state.mkdir(parents=True, exist_ok=True, mode=0o750)
    base_path = state / "configured_controls_progression.v1.json"
    base = _sealed_progression(base_path, statuses={"episode_preparation_queued"})
    if base is None:
        terminal, _, _ = _validate_source(run_root)
        publication = _input(
            terminal.get("publication_result_path"), blocker="configured_controls_worker_publication_missing"
        )
        revision = _input(
            terminal.get("configured_scene_revision_path"), blocker="configured_controls_worker_revision_missing"
        )
        base_pose = _input(plan.get("base_pose_candidate_path"), blocker="configured_controls_worker_base_pose_missing")
        cameras = _input(plan.get("cameras_path"), blocker="configured_controls_worker_cameras_missing")
        runtime = _input(plan.get("runtime_binding_path"), blocker="configured_controls_worker_runtime_missing")
        rows = cameras.get("cameras")
        if not isinstance(rows, list):
            raise TaskEvaluationConfiguredControlsProgressionWorkerError(
                "configured_controls_worker_cameras_invalid"
            )
        result = stage_configured_controls_episode_preparation(
            terminal_result=terminal,
            publication_result=publication,
            configured_revision=revision,
            expected_production_commit=plan["expected_production_commit"],
            robot_mount_interface_path=plan["robot_mount_interface_path"],
            scene_camera_calibration_path=plan["scene_camera_calibration_path"],
            base_pose_candidate=base_pose,
            cameras=rows,
            runtime_binding=runtime,
            output_root=state,
            publisher=publisher_factory(),
            queue_root=preparation_queue_root,
            submitted_by=plan["submitted_by"],
        )
        return {"status": result["status"], "source_launch_id": plan["source_launch_id"]}

    prep_id = base["episode_preparation_request"]["preparation_id"]
    preparation = _queue_result(Path(preparation_queue_root), prep_id)
    if preparation is None:
        return {"status": "awaiting_episode_preparation", "source_launch_id": plan["source_launch_id"]}

    construction_phase = _phase(plan, "construction")
    construction_activation_path = state / "construction_activation_progression.json"
    construction_activation = _sealed_progression(
        construction_activation_path, statuses={"construction_activation_queued"}
    )
    if construction_activation is None:
        result = stage_configured_controls_activation(
            progression=base,
            preparation_result=preparation,
            release_window=_input(construction_phase.get("release_window_path"), blocker="configured_controls_worker_release_window_missing"),
            lineage=_input(construction_phase.get("lineage_path"), blocker="configured_controls_worker_lineage_missing"),
            authorization=_input(construction_phase.get("authorization_path"), blocker="configured_controls_worker_authorization_missing"),
            lane="native_task_arena_construction",
            queue_root=activation_queue_root,
            submitted_by=plan["submitted_by"],
        )
        _write_immutable(construction_activation_path, result)
        return {"status": result["status"], "source_launch_id": plan["source_launch_id"]}

    construction_launch_path = state / "construction_launch_progression.json"
    construction_launch = _sealed_progression(
        construction_launch_path, statuses={"construction_launch_queued"}
    )
    if construction_launch is None:
        activation_result_path = Path(str(construction_phase.get("activation_result_path") or ""))
        profile_path = Path(str(construction_phase.get("profile_path") or ""))
        if not activation_result_path.is_file() or not profile_path.is_file():
            return {"status": "awaiting_construction_activation", "source_launch_id": plan["source_launch_id"]}
        if submitter is None:
            if repo_root is None or webapp_secret_file is None:
                raise TaskEvaluationConfiguredControlsProgressionWorkerError(
                    "configured_controls_worker_webapp_configuration_missing"
                )
            submitter = _production_submitter(
                repo_root=Path(repo_root), secret_file=Path(webapp_secret_file),
                endpoint=webapp_endpoint, state_root=state,
            )
        result = submit_authorized_progression_launch(
            activation_progression=construction_activation,
            activation_result=_input(activation_result_path, blocker="configured_controls_worker_activation_result_invalid"),
            profile=_input(profile_path, blocker="configured_controls_worker_profile_invalid"),
            launch_authority=_input(construction_phase.get("launch_authority_path"), blocker="configured_controls_worker_launch_authority_missing"),
            submitter=submitter,
        )
        _write_immutable(construction_launch_path, result)
        return {"status": result["status"], "source_launch_id": plan["source_launch_id"]}

    controls_phase = _phase(plan, "controls")
    controls_activation_path = state / "controls_activation_progression.json"
    controls_activation = _sealed_progression(
        controls_activation_path, statuses={"controls_activation_queued"}
    )
    if controls_activation is None:
        artifact_paths = controls_phase.get("lineage_artifact_paths")
        if not isinstance(artifact_paths, Mapping) or any(
            not Path(str(path)).is_file() for path in artifact_paths.values()
        ):
            return {"status": "awaiting_qualified_construction", "source_launch_id": plan["source_launch_id"]}
        result = stage_configured_controls_activation(
            progression=base,
            preparation_result=preparation,
            release_window=_input(controls_phase.get("release_window_path"), blocker="configured_controls_worker_release_window_missing"),
            lineage=_input(controls_phase.get("lineage_path"), blocker="configured_controls_worker_lineage_missing"),
            authorization=_input(controls_phase.get("authorization_path"), blocker="configured_controls_worker_authorization_missing"),
            lane="native_task_arena_controls",
            queue_root=activation_queue_root,
            submitted_by=plan["submitted_by"],
            lineage_artifact_paths={str(key): str(value) for key, value in artifact_paths.items()},
        )
        _write_immutable(controls_activation_path, result)
        return {"status": result["status"], "source_launch_id": plan["source_launch_id"]}

    controls_launch_path = state / "controls_launch_progression.json"
    controls_launch = _sealed_progression(controls_launch_path, statuses={"controls_pair_launch_queued"})
    if controls_launch is not None:
        return {"status": "controls_pair_launch_queued", "source_launch_id": plan["source_launch_id"]}
    activation_result_path = Path(str(controls_phase.get("activation_result_path") or ""))
    profile_path = Path(str(controls_phase.get("profile_path") or ""))
    if not activation_result_path.is_file() or not profile_path.is_file():
        return {"status": "awaiting_controls_activation", "source_launch_id": plan["source_launch_id"]}
    if submitter is None:
        if repo_root is None or webapp_secret_file is None:
            raise TaskEvaluationConfiguredControlsProgressionWorkerError(
                "configured_controls_worker_webapp_configuration_missing"
            )
        submitter = _production_submitter(
            repo_root=Path(repo_root), secret_file=Path(webapp_secret_file),
            endpoint=webapp_endpoint, state_root=state,
        )
    result = submit_authorized_progression_launch(
        activation_progression=controls_activation,
        activation_result=_input(activation_result_path, blocker="configured_controls_worker_activation_result_invalid"),
        profile=_input(profile_path, blocker="configured_controls_worker_profile_invalid"),
        launch_authority=_input(controls_phase.get("launch_authority_path"), blocker="configured_controls_worker_launch_authority_missing"),
        submitter=submitter,
    )
    _write_immutable(controls_launch_path, result)
    return {"status": result["status"], "source_launch_id": plan["source_launch_id"]}


def process_plans(**kwargs: Any) -> dict[str, Any]:
    plan_root = Path(kwargs.pop("plan_root")).expanduser()
    rows: list[dict[str, Any]] = []
    for path in sorted(plan_root.glob("*.json")) if plan_root.is_dir() else []:
        try:
            rows.append(advance_configured_controls_plan(plan_path=path, **kwargs))
        except (TaskEvaluationConfiguredControlsProgressionError, TaskEvaluationConfiguredControlsProgressionWorkerError) as exc:
            rows.append({"status": "blocked", "plan": path.name, "blockers": [str(exc)]})
    return {
        "schema_version": WORKER_RESULT_SCHEMA_VERSION,
        "status": "blocked" if any(row["status"] == "blocked" for row in rows) else "completed",
        "rows": rows,
        "provider_mutation_performed": False,
        "allocator_invoked": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-root", required=True)
    parser.add_argument("--launch-state-root", required=True)
    parser.add_argument("--progression-root", required=True)
    parser.add_argument("--preparation-queue-root", required=True)
    parser.add_argument("--activation-queue-root", required=True)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--webapp-secret-file", required=True)
    parser.add_argument("--webapp-endpoint", default="https://tryblueprint.io/api/internal/task-evaluation-launch-submissions")
    args = parser.parse_args(argv)
    report = process_plans(**vars(args))
    print(json.dumps(report, sort_keys=True))
    return 0 if report["status"] == "completed" else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
