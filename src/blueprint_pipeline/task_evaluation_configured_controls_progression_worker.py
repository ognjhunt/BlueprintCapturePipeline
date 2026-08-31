"""Production-owned automatic progression from configured scenes to controls.

The launch reconciler remains observation-only.  This separate timer worker
consumes an immutable CPU-materialized plan, a qualifying terminal launch, a
successful WebApp sync, and the reconciler's post-teardown global provider-zero
receipt.  It advances existing no-spend preparation/activation queues and uses
the canonical WebApp-only client for paid launch submission.  It never invokes
an allocator or provider directly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import stat
import subprocess  # nosec B404 - fixed repository-owned launch-only client
import sys
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest, cross_runtime_canonical_digest
from .task_evaluation_configured_controls_progression import (
    PROGRESSION_SCHEMA_VERSION,
    TaskEvaluationConfiguredControlsProgressionError,
    _publish_materialized_file,
    build_configured_controls_activation_request,
    stage_configured_controls_activation,
    stage_configured_controls_episode_preparation,
    submit_authorized_progression_launch,
)
from .task_evaluation_launch_activation_contract import (
    launch_activation_intent_digest,
)
from .task_evaluation_shared_mutation_window import (
    TaskEvaluationSharedMutationWindowError,
    materialize_shared_mutation_window,
    validate_shared_mutation_window,
    validate_shared_mutation_window_template,
)
from .task_evaluation_configured_scene_object_store import (
    configured_scene_object_store_publisher,
)
from .task_evaluation_launch_dispatcher import LAUNCH_RECEIPT_DIGEST_CANONICALIZATION
from .task_evaluation_launch_reconciler import validated_succeeded_webapp_sync_row


PLAN_SCHEMA_VERSION = "task_evaluation_configured_controls_progression_plan.v2"
_COMMIT = re.compile(r"[0-9a-f]{40}")
WORKER_RESULT_SCHEMA_VERSION = "task_evaluation_configured_controls_progression_worker.v1"
CONFIGURED_CONTROLS_KEY_PREFIX = (
    "task-evaluation/production-inputs/configured-controls"
)
CONFIGURED_CONTROLS_RELEASE_WINDOW_KEY_PREFIX = (
    "task-evaluation/production-inputs/coordinator-release-windows"
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


def configured_controls_release_window_publisher() -> Callable[..., Mapping[str, Any]]:
    """Publish coordinator authority under the activation worker's exact prefix."""

    return configured_scene_object_store_publisher(
        key_prefix=CONFIGURED_CONTROLS_RELEASE_WINDOW_KEY_PREFIX
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
        or not str(value.get("source_launch_receipt_digest") or "").startswith(
            "sha256:"
        )
        or _COMMIT.fullmatch(
            str(value.get("source_configuration_commit") or "")
        )
        is None
        or _COMMIT.fullmatch(str(value.get("expected_production_commit") or ""))
        is None
        or not str(value.get("submitted_by") or "").strip()
        or set(value.get("phases") or {}) != {"construction", "controls"}
        or set(value["phases"].get("construction") or {})
        != {
            "release_window_template_path",
            "lineage_path",
            "authorization_path",
            "launch_authority_path",
        }
        or set(value["phases"].get("controls") or {})
        != {
            "release_window_template_path",
            "authorization_path",
            "launch_authority_path",
        }
        or not Path(str(value.get("profile_dir") or "")).is_absolute()
        or Path(str(value.get("profile_dir") or "")).is_symlink()
        or not Path(str(value.get("profile_dir") or "")).is_dir()
    ):
        raise TaskEvaluationConfiguredControlsProgressionWorkerError(
            "configured_controls_worker_plan_invalid"
        )
    inventory = value.get("artifact_inventory")
    declared_paths: set[str] = set()

    def collect_paths(row: Any, key: str = "") -> None:
        if isinstance(row, Mapping):
            for child_key, child in row.items():
                if child_key.endswith("_path") and isinstance(child, str):
                    declared_paths.add(child)
                elif child_key == "lineage_artifact_paths" and isinstance(
                    child, Mapping
                ):
                    declared_paths.update(str(item) for item in child.values())
                elif child_key not in {"artifact_inventory", "future_outputs"}:
                    collect_paths(child, child_key)

    collect_paths(value)
    if not isinstance(inventory, Mapping) or not inventory:
        raise TaskEvaluationConfiguredControlsProgressionWorkerError(
            "configured_controls_worker_plan_inventory_invalid"
        )
    inventory_paths: set[str] = set()
    for row in inventory.values():
        if not isinstance(row, Mapping) or set(row) != {
            "path",
            "digest",
            "size_bytes",
            "mode",
        }:
            raise TaskEvaluationConfiguredControlsProgressionWorkerError(
                "configured_controls_worker_plan_inventory_invalid"
            )
        artifact = Path(str(row.get("path") or ""))
        try:
            metadata = artifact.stat()
        except OSError as exc:
            raise TaskEvaluationConfiguredControlsProgressionWorkerError(
                "configured_controls_worker_plan_inventory_invalid"
            ) from exc
        if (
            not artifact.is_absolute()
            or artifact.is_symlink()
            or not artifact.is_file()
            or _sha256(artifact) != row.get("digest")
            or metadata.st_size != row.get("size_bytes")
            or f"{stat.S_IMODE(metadata.st_mode):04o}" != row.get("mode")
        ):
            raise TaskEvaluationConfiguredControlsProgressionWorkerError(
                "configured_controls_worker_plan_inventory_invalid"
            )
        inventory_paths.add(str(artifact))
    future = value.get("future_outputs")
    if not isinstance(future, Mapping) or set(future) != {"construction", "controls"}:
        raise TaskEvaluationConfiguredControlsProgressionWorkerError(
            "configured_controls_worker_plan_future_outputs_invalid"
        )
    for phase in ("construction", "controls"):
        row = future.get(phase)
        if (
            not isinstance(row, Mapping)
            or set(row) != {"expected_activation_id"}
            or not str(row.get("expected_activation_id") or "")
        ):
            raise TaskEvaluationConfiguredControlsProgressionWorkerError(
                "configured_controls_worker_plan_future_outputs_invalid"
            )
    if inventory_paths != declared_paths:
        raise TaskEvaluationConfiguredControlsProgressionWorkerError(
            "configured_controls_worker_plan_inventory_invalid"
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


def _materialize_phase_release_window(
    *,
    state: Mapping[str, Any],
    preparation: Mapping[str, Any],
    phase: Mapping[str, Any],
    lineage: Mapping[str, Any],
    authorization: Mapping[str, Any],
    lane: str,
    root: Path,
    publisher: Callable[..., Mapping[str, Any]],
    lineage_artifact_paths: Mapping[str, str | Path] | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Publish a current window bound to the now-complete activation intent."""

    placeholder = {
        "uri": "https://tryblueprint.io/internal/release-window-placeholder",
        "digest": "sha256:" + "0" * 64,
        "size_bytes": 1,
    }
    request = build_configured_controls_activation_request(
        progression=state,
        preparation_result=preparation,
        release_window=placeholder,
        lineage=lineage,
        authorization=authorization,
        lane=lane,
        lineage_artifact_paths=lineage_artifact_paths,
    )
    template = _input(
        phase.get("release_window_template_path"),
        blocker="configured_controls_worker_release_window_template_missing",
    )
    template = validate_shared_mutation_window_template(
        template,
        team_namespace=str(request["team_namespace"]),
        expected_production_commit=str(request["expected_production_commit"]),
    )
    provider_allowlist = list(
        state["episode_preparation_request"]["spend"]["provider_allowlist"]
    )
    hard_cap_usd = float(
        state["episode_preparation_request"]["spend"]["hard_cap_usd"]
    )
    observed_now = now or datetime.now(timezone.utc)
    intent_digest = launch_activation_intent_digest(request)
    attempts = (
        root
        / "release-window-attempts"
        / lane
        / (
            intent_digest.removeprefix("sha256:")
            + "-"
            + str(template["template_digest"]).removeprefix("sha256:")
        )
    )
    attempts.mkdir(parents=True, exist_ok=True, mode=0o750)
    for existing_path in sorted(attempts.glob("window-*.json"), reverse=True):
        existing = _input(
            existing_path,
            blocker="configured_controls_worker_release_window_checkpoint_invalid",
        )
        try:
            window = validate_shared_mutation_window(
                existing,
                activation_id=str(request["activation_id"]),
                activation_intent_digest=intent_digest,
                team_namespace=str(request["team_namespace"]),
                expected_production_commit=str(
                    request["expected_production_commit"]
                ),
                provider_allowlist=provider_allowlist,
                hard_cap_usd=hard_cap_usd,
                now=observed_now,
            )
        except TaskEvaluationSharedMutationWindowError as exc:
            if str(exc) == "shared_mutation_window_not_current":
                continue
            raise TaskEvaluationConfiguredControlsProgressionWorkerError(
                "configured_controls_worker_release_window_checkpoint_invalid"
            ) from exc
        return _publish_materialized_file(
            path=existing_path,
            object_name=(
                f"release-windows/{request['activation_id']}/"
                f"{window['window_digest'].removeprefix('sha256:')}.json"
            ),
            publisher=publisher,
        )
    window = materialize_shared_mutation_window(
        template,
        activation_request=request,
        provider_allowlist=provider_allowlist,
        hard_cap_usd=hard_cap_usd,
        now=observed_now,
    )
    window_path = attempts / (
        f"window-{window['window_digest'].removeprefix('sha256:')}.json"
    )
    _write_immutable(window_path, window)
    return _publish_materialized_file(
        path=window_path,
        object_name=(
            f"release-windows/{request['activation_id']}/"
            f"{window['window_digest'].removeprefix('sha256:')}.json"
        ),
        publisher=publisher,
    )


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


def _activation_authority(
    *, activation_queue_root: Path, profile_dir: Path, activation_id: str
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    matches = list((activation_queue_root / "results").glob(f"{activation_id}-*.json"))
    if not matches:
        return None
    if len(matches) != 1:
        raise TaskEvaluationConfiguredControlsProgressionWorkerError(
            "configured_controls_worker_activation_result_ambiguous"
        )
    activation = _load(
        matches[0], blocker="configured_controls_worker_activation_result_invalid"
    )
    profile_id = str(activation.get("profile_id") or "")
    profile_path = profile_dir / f"{profile_id}.json"
    if (
        activation.get("activation_id") != activation_id
        or not profile_id
        or profile_path.parent != profile_dir
        or not profile_path.is_file()
    ):
        raise TaskEvaluationConfiguredControlsProgressionWorkerError(
            "configured_controls_worker_activation_result_invalid"
        )
    profile = _load(
        profile_path, blocker="configured_controls_worker_profile_invalid"
    )
    if profile.get("profile_digest") != activation.get("profile_digest"):
        raise TaskEvaluationConfiguredControlsProgressionWorkerError(
            "configured_controls_worker_profile_invalid"
        )
    return activation, profile


def _construction_predecessor(
    *,
    launch_state_root: Path,
    construction_launch_id: str,
    publisher: Callable[..., Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, str]] | None:
    """Resolve and publish the exact terminal construction evidence set."""

    run_root = launch_state_root / construction_launch_id
    receipt_path = run_root / "launch_receipt.json"
    if not receipt_path.exists():
        return None
    receipt = _load(
        receipt_path,
        blocker="configured_controls_worker_construction_launch_receipt_invalid",
    )
    expected_receipt_digest = (
        cross_runtime_canonical_digest(receipt, digest_field="receipt_digest")
        if receipt.get("receipt_digest_canonicalization")
        == LAUNCH_RECEIPT_DIGEST_CANONICALIZATION
        else canonical_digest(receipt, digest_field="receipt_digest")
    )
    terminal = receipt.get("terminal_evidence")
    terminal_artifact = (
        terminal.get("result") if isinstance(terminal, Mapping) else None
    )
    if (
        receipt.get("schema_version") != "task_evaluation_launch_receipt.v1"
        or receipt.get("status") != "completed"
        or receipt.get("launch_id") != construction_launch_id
        or receipt.get("run_id") != construction_launch_id
        or receipt.get("receipt_digest") != expected_receipt_digest
        or not isinstance(terminal, Mapping)
        or terminal.get("status") != "passed"
        or not isinstance(terminal_artifact, Mapping)
        or terminal_artifact.get("exists") is not True
    ):
        raise TaskEvaluationConfiguredControlsProgressionWorkerError(
            "configured_controls_worker_construction_launch_receipt_invalid"
        )
    result_path = Path(str(terminal_artifact.get("path") or "")).expanduser()
    if (
        result_path.is_symlink()
        or not result_path.is_file()
        or _sha256(result_path) != terminal_artifact.get("digest")
    ):
        raise TaskEvaluationConfiguredControlsProgressionWorkerError(
            "configured_controls_worker_construction_terminal_artifact_invalid"
        )
    result = _load(
        result_path,
        blocker="configured_controls_worker_construction_terminal_result_invalid",
    )
    construction_result_path = Path(
        str(result.get("native_control_result_path") or "")
    ).expanduser()
    construction_result = _load(
        construction_result_path,
        blocker="configured_controls_worker_construction_result_invalid",
    )
    if (
        result.get("schema_version") != "native_task_arena_vast_run.v1"
        or result.get("status") != "completed"
        or result.get("blockers") not in ([], ())
        or result.get("native_control_result_digest")
        != construction_result.get("result_digest")
        or construction_result.get("schema_version")
        != "native_task_arena_construction_result.v1"
        or construction_result.get("status") != "completed"
        or construction_result.get("construction_gate_qualified") is not True
        or construction_result.get("candidate_policy_queried") is not False
        or construction_result.get("blockers") not in ([], ())
        or construction_result.get("result_digest")
        != canonical_digest(construction_result, digest_field="result_digest")
    ):
        raise TaskEvaluationConfiguredControlsProgressionWorkerError(
            "configured_controls_worker_construction_result_invalid"
        )
    sync_path = run_root / "webapp_sync_succeeded.json"
    sync = _load(
        sync_path,
        blocker="configured_controls_worker_construction_webapp_sync_missing",
    )
    try:
        validated_succeeded_webapp_sync_row(receipt=receipt, attempt=sync)
    except Exception as exc:
        raise TaskEvaluationConfiguredControlsProgressionWorkerError(
            "configured_controls_worker_construction_webapp_sync_invalid"
        ) from exc
    zero_path = run_root / "post_teardown_provider_zero_receipt.json"
    zero = _load(
        zero_path,
        blocker="configured_controls_worker_construction_provider_zero_missing",
    )
    if (
        zero.get("schema_version")
        != "task_evaluation_post_teardown_provider_zero.v1"
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
            for field in (
                "launch_id",
                "run_id",
                "request_digest",
                "receipt_digest",
                "launch_profile_digest",
            )
        )
    ):
        raise TaskEvaluationConfiguredControlsProgressionWorkerError(
            "configured_controls_worker_construction_provider_zero_invalid"
        )
    profile_path = run_root / "launch_profile.json"
    profile = _load(
        profile_path,
        blocker="configured_controls_worker_construction_profile_invalid",
    )
    if (
        profile.get("schema_version") != "task_evaluation_launch_profile.v1"
        or profile.get("profile_digest") != receipt.get("launch_profile_digest")
        or profile.get("profile_digest")
        != canonical_digest(profile, digest_field="profile_digest")
        or not isinstance(profile.get("immutable_inputs"), list)
    ):
        raise TaskEvaluationConfiguredControlsProgressionWorkerError(
            "configured_controls_worker_construction_profile_invalid"
        )
    input_names = {
        "prior_authority": "native_task_arena_attempt_authority",
        "prior_spend_reconciliation": (
            "native_task_arena_attempt_authority_prior_spend_reconciliation"
        ),
    }
    artifact_paths: dict[str, Path] = {
        "prior_result": result_path,
        "prior_launch_receipt": receipt_path,
        "prior_webapp_sync": sync_path,
        "prior_provider_zero": zero_path,
        "construction_result": construction_result_path,
    }
    for role, expected_name in input_names.items():
        matches = [
            row
            for row in profile["immutable_inputs"]
            if isinstance(row, Mapping) and row.get("name") == expected_name
        ]
        if len(matches) != 1:
            raise TaskEvaluationConfiguredControlsProgressionWorkerError(
                "configured_controls_worker_construction_profile_invalid"
            )
        row = matches[0]
        path = Path(str(row.get("path") or "")).expanduser()
        if (
            path.is_symlink()
            or not path.is_file()
            or _sha256(path) != row.get("digest")
        ):
            raise TaskEvaluationConfiguredControlsProgressionWorkerError(
                "configured_controls_worker_construction_profile_input_invalid"
            )
        artifact_paths[role] = path
    lineage: dict[str, Any] = {"kind": "predecessor"}
    published_paths: dict[str, str] = {}
    for role, path in sorted(artifact_paths.items()):
        lineage[role] = _publish_materialized_file(
            path=path,
            object_name=(
                f"construction-predecessor/{construction_launch_id}/{role}.json"
            ),
            publisher=publisher,
        )
        published_paths[role] = str(path)
    return lineage, published_paths


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
    release_window_publisher_factory: PublisherFactory = (
        configured_controls_release_window_publisher
    ),
    submitter: Submitter | None = None,
    repo_root: str | Path | None = None,
    webapp_secret_file: str | Path | None = None,
    webapp_endpoint: str = "https://tryblueprint.io/api/internal/task-evaluation-launch-submissions",
) -> dict[str, Any]:
    """Advance at most one transition for one immutable progression plan."""

    plan = _plan(Path(plan_path).expanduser())
    run_root = Path(launch_state_root).expanduser() / plan["source_launch_id"]
    # Scope progression state to the production commit. The sealed receipt
    # carries the episode preparation id, and preparation results are
    # immutable, so sharing one directory across commits makes a launch that
    # blocked under an earlier commit unanswerable under its successor.
    state = (
        Path(progression_root).expanduser()
        / plan["source_launch_id"]
        / f"franka-controls-{plan['expected_production_commit'][:12]}"
    )
    state.mkdir(parents=True, exist_ok=True, mode=0o750)
    base_path = state / "configured_controls_progression.v1.json"
    base = _sealed_progression(base_path, statuses={"episode_preparation_queued"})
    if base is None:
        terminal, receipt, _ = _validate_source(run_root)
        if (
            receipt.get("receipt_digest")
            != plan["source_launch_receipt_digest"]
            or receipt.get("source_commit")
            != plan["source_configuration_commit"]
        ):
            raise TaskEvaluationConfiguredControlsProgressionWorkerError(
                "configured_controls_worker_source_receipt_mismatch"
            )
        namespace = (
            f"{terminal['run_id']}-franka-controls-"
            f"{plan['expected_production_commit'][:12]}-episode"
        )
        if any(
            plan["future_outputs"][phase]["expected_activation_id"]
            != f"{namespace}-{phase}"
            for phase in ("construction", "controls")
        ):
            raise TaskEvaluationConfiguredControlsProgressionWorkerError(
                "configured_controls_worker_future_activation_identity_mismatch"
            )
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
        lineage = _input(construction_phase.get("lineage_path"), blocker="configured_controls_worker_lineage_missing")
        authorization = _input(construction_phase.get("authorization_path"), blocker="configured_controls_worker_authorization_missing")
        release_window = _materialize_phase_release_window(
            state=base,
            preparation=preparation,
            phase=construction_phase,
            lineage=lineage,
            authorization=authorization,
            lane="native_task_arena_construction",
            root=state,
            publisher=release_window_publisher_factory(),
        )
        result = stage_configured_controls_activation(
            progression=base,
            preparation_result=preparation,
            release_window=release_window,
            lineage=lineage,
            authorization=authorization,
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
        authority = _activation_authority(
            activation_queue_root=Path(activation_queue_root),
            profile_dir=Path(plan["profile_dir"]),
            activation_id=plan["future_outputs"]["construction"][
                "expected_activation_id"
            ],
        )
        if authority is None:
            return {"status": "awaiting_construction_activation", "source_launch_id": plan["source_launch_id"]}
        activation_result, profile = authority
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
            activation_result=activation_result,
            profile=profile,
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
        predecessor = _construction_predecessor(
            launch_state_root=Path(launch_state_root),
            construction_launch_id=str(construction_launch["launch_id"]),
            publisher=publisher_factory(),
        )
        if predecessor is None:
            return {
                "status": "awaiting_qualified_construction",
                "source_launch_id": plan["source_launch_id"],
            }
        lineage, artifact_paths = predecessor
        authorization = _input(controls_phase.get("authorization_path"), blocker="configured_controls_worker_authorization_missing")
        release_window = _materialize_phase_release_window(
            state=base,
            preparation=preparation,
            phase=controls_phase,
            lineage=lineage,
            authorization=authorization,
            lane="native_task_arena_controls",
            root=state,
            publisher=release_window_publisher_factory(),
            lineage_artifact_paths=artifact_paths,
        )
        result = stage_configured_controls_activation(
            progression=base,
            preparation_result=preparation,
            release_window=release_window,
            lineage=lineage,
            authorization=authorization,
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
    authority = _activation_authority(
        activation_queue_root=Path(activation_queue_root),
        profile_dir=Path(plan["profile_dir"]),
        activation_id=plan["future_outputs"]["controls"]["expected_activation_id"],
    )
    if authority is None:
        return {"status": "awaiting_controls_activation", "source_launch_id": plan["source_launch_id"]}
    activation_result, profile = authority
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
        activation_result=activation_result,
        profile=profile,
        launch_authority=_input(controls_phase.get("launch_authority_path"), blocker="configured_controls_worker_launch_authority_missing"),
        submitter=submitter,
    )
    _write_immutable(controls_launch_path, result)
    return {"status": result["status"], "source_launch_id": plan["source_launch_id"]}


def process_plans(**kwargs: Any) -> dict[str, Any]:
    plan_root = Path(kwargs.pop("plan_root")).expanduser()
    intent_root_value = kwargs.pop("autostart_intent_root", None)
    intent_root = (
        Path(intent_root_value).expanduser()
        if intent_root_value is not None
        else None
    )
    launch_state_root = Path(kwargs["launch_state_root"]).expanduser()
    rows: list[dict[str, Any]] = []
    # Configuration profiles carrying the required autostart intent need no
    # operator-written plan.  The no-spend materializer reopens the completed
    # revision, runs the full CPU placement inventory/trajectory gate, and
    # writes the same immutable plan consumed below.  A profile without the
    # intent remains untouched for backwards-compatible observation.
    from .task_evaluation_configured_controls_autostart import (
        TaskEvaluationConfiguredControlsAutostartError,
        materialize_configured_controls_autostart,
    )

    for run_root in sorted(launch_state_root.iterdir()) if launch_state_root.is_dir() else []:
        if not run_root.is_dir() or run_root.is_symlink():
            continue
        profile_path = run_root / "launch_profile.json"
        receipt_path = run_root / "launch_receipt.json"
        if not profile_path.is_file() or not receipt_path.is_file():
            continue
        try:
            profile = _load(
                profile_path, blocker="configured_controls_autostart_profile_invalid"
            )
        except TaskEvaluationConfiguredControlsProgressionWorkerError as exc:
            rows.append({"status": "blocked", "source_launch_id": run_root.name, "blockers": [str(exc)]})
            continue
        has_intent = any(
            isinstance(item, Mapping)
            and item.get("name") == "configured_controls_autostart_intent"
            for item in profile.get("immutable_inputs") or []
        )
        intent_path_override: Path | None = None
        if not has_intent:
            if (
                intent_root is None
                or not intent_root.is_absolute()
                or intent_root.is_symlink()
                or not intent_root.is_dir()
            ):
                continue
            task_run = profile.get("task_evaluation_run")
            if not isinstance(task_run, Mapping):
                continue
            from .task_evaluation_configured_controls_autostart import (
                configured_controls_autostart_adoption_registry_name,
                validate_configured_controls_autostart_intent,
            )

            candidate = intent_root / configured_controls_autostart_adoption_registry_name(
                team_namespace=str(task_run.get("team_namespace") or ""),
                scene_id=str(task_run.get("scene_id") or ""),
                task_id=str(task_run.get("task_id") or ""),
                source_launch_id=run_root.name,
            )
            if not candidate.is_file() or candidate.is_symlink():
                continue
            try:
                registry_intent = validate_configured_controls_autostart_intent(
                    _load(
                        candidate,
                        blocker="configured_controls_autostart_adoption_intent_invalid",
                    )
                )
            except (
                TaskEvaluationConfiguredControlsAutostartError,
                TaskEvaluationConfiguredControlsProgressionWorkerError,
            ) as exc:
                rows.append(
                    {
                        "status": "blocked",
                        "source_launch_id": run_root.name,
                        "blockers": [str(exc)],
                    }
                )
                continue
            adoption = registry_intent.get("configuration_adoption")
            if (
                not isinstance(adoption, Mapping)
                or adoption.get("mode") != "explicit_terminal_adoption"
                or adoption.get("source_launch_id") != run_root.name
            ):
                continue
            intent_path_override = candidate
        try:
            receipt = _load(
                receipt_path,
                blocker="configured_controls_autostart_launch_receipt_invalid",
            )
        except TaskEvaluationConfiguredControlsProgressionWorkerError as exc:
            rows.append({"status": "blocked", "source_launch_id": run_root.name, "blockers": [str(exc)]})
            continue
        if receipt.get("status") != "completed":
            rows.append({"status": "awaiting_configuration_terminal", "source_launch_id": run_root.name})
            continue
        if not (run_root / "webapp_sync_succeeded.json").is_file():
            rows.append({"status": "awaiting_configuration_webapp_sync", "source_launch_id": run_root.name})
            continue
        if not (run_root / "post_teardown_provider_zero_receipt.json").is_file():
            rows.append({"status": "awaiting_configuration_provider_zero", "source_launch_id": run_root.name})
            continue
        try:
            auto = materialize_configured_controls_autostart(
                source_launch_id=run_root.name,
                launch_state_root=launch_state_root,
                progression_root=kwargs["progression_root"],
                plan_root=plan_root,
                intent_path_override=intent_path_override,
            )
            rows.append({
                "status": auto["status"],
                "source_launch_id": run_root.name,
                "selected_candidate_id": auto["selected_candidate_id"],
                "plan_digest": auto["plan_digest"],
            })
        except (
            TaskEvaluationConfiguredControlsAutostartError,
            TaskEvaluationConfiguredControlsProgressionError,
            TaskEvaluationConfiguredControlsProgressionWorkerError,
            OSError,
            ValueError,
        ) as exc:
            rows.append({"status": "blocked", "source_launch_id": run_root.name, "blockers": [str(exc)]})
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
    parser.add_argument("--autostart-intent-root", required=True)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--webapp-secret-file", required=True)
    parser.add_argument("--webapp-endpoint", default="https://tryblueprint.io/api/internal/task-evaluation-launch-submissions")
    args = parser.parse_args(argv)
    report = process_plans(**vars(args))
    print(json.dumps(report, sort_keys=True))
    return 0 if report["status"] == "completed" else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
