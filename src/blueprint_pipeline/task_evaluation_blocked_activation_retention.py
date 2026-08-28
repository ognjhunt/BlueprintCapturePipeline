"""Reclaim rebuildable bundle bytes from one terminal pre-authority activation.

The activation envelope, terminal result, preparation receipt, references,
logs, context and bundle receipt are evidence and are never removed.  The only
eligible payloads are the exact provider ZIP named by the receipt and the
byte-identical staging tree from which it was produced.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import stat
import zipfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from .decision_evidence_contracts import canonical_digest
from .task_evaluation_launch_activation_queue import (
    ENVELOPE_SCHEMA_VERSION,
    RESULT_SCHEMA_VERSION,
)
from .task_evaluation_release_reference_lock import release_reference_lock
from .task_evaluation_release_retention import _write_exclusive
from .task_evaluation_scene_configuration_bundle import (
    BUNDLE_SCHEMA_VERSION,
    load_scene_configuration_provider_bundle_receipt,
)


SCHEMA_VERSION = "task_evaluation_blocked_activation_retention_plan.v1"
APPLY_SCHEMA_VERSION = "task_evaluation_blocked_activation_retention_apply.v1"
APPLY_ACKNOWLEDGEMENT = "reap-blocked-activation-rebuildables"
DEFAULT_ACTIVATION_BASE_ROOT = Path("/var/lib/blueprint/task-evaluation-inputs/launch-activations")
DEFAULT_STATE_ROOT = Path("/var/lib/blueprint/pipeline-control-plane")
DEFAULT_ACTIVATION_QUEUE_ROOT = DEFAULT_STATE_ROOT / "task-evaluation-launch-activations"
DEFAULT_PROFILE_DIR = Path("/etc/blueprint/task-evaluation-launch-profiles")
DEFAULT_PUBLIC_CATALOG = DEFAULT_STATE_ROOT / "task-evaluation-launch-profile-catalog.json"
DEFAULT_STANDING_AUTHORIZATION_DIR = DEFAULT_STATE_ROOT / "standing-authorizations"
DEFAULT_LIVE_REFERENCE_ROOTS = tuple(
    DEFAULT_STATE_ROOT / name / state
    for name in ("task-evaluation-launches", "task-evaluation-terminal-resource-releases")
    for state in ("pending", "processing")
)

BundleValidator = Callable[..., Mapping[str, Any]]


class BlockedActivationRetentionError(ValueError):
    """The activation was not proven safe for rebuildable-byte reclamation."""


def _absolute(value: str | Path, *, field: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise BlockedActivationRetentionError(
            f"blocked_activation_retention_{field}_must_be_absolute"
        )
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _file_record(path: Path, *, blocker: str) -> dict[str, Any]:
    try:
        info = path.lstat()
    except OSError as exc:
        raise BlockedActivationRetentionError(blocker) from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise BlockedActivationRetentionError(blocker)
    return {
        "path": str(path),
        "device": int(info.st_dev),
        "inode": int(info.st_ino),
        "mode": stat.S_IMODE(info.st_mode),
        "size_bytes": int(info.st_size),
        "mtime_ns": int(info.st_mtime_ns),
        "sha256": _sha256(path),
    }


def _read_sealed(
    path: Path, *, schema_version: str, digest_field: str, blocker: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    record = _file_record(path, blocker=blocker)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BlockedActivationRetentionError(blocker) from exc
    if (
        not isinstance(value, Mapping)
        or value.get("schema_version") != schema_version
        or value.get(digest_field) != canonical_digest(value, digest_field=digest_field)
    ):
        raise BlockedActivationRetentionError(blocker)
    return dict(value), record


def _assert_directory(path: Path, *, blocker: str) -> Path:
    if path.is_symlink() or not path.is_dir():
        raise BlockedActivationRetentionError(blocker)
    return path.resolve(strict=True)


def _tree_and_archive_snapshot(stage: Path, bundle: Path) -> dict[str, Any]:
    """Prove the removable tree is exactly the ZIP payload, without storing it."""

    stage = _assert_directory(stage, blocker="blocked_activation_retention_stage_tree_invalid")
    stage_info = stage.lstat()
    rows: list[dict[str, Any]] = []
    stage_names: set[str] = set()
    for path in sorted(stage.rglob("*"), key=lambda item: item.as_posix()):
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode):
            raise BlockedActivationRetentionError("blocked_activation_retention_stage_tree_symlink")
        if stat.S_ISDIR(info.st_mode):
            continue
        if not stat.S_ISREG(info.st_mode):
            raise BlockedActivationRetentionError("blocked_activation_retention_stage_tree_invalid")
        relative = path.relative_to(stage).as_posix()
        stage_names.add(relative)
        rows.append(
            {
                "path": relative,
                "mode": stat.S_IMODE(info.st_mode),
                "size_bytes": int(info.st_size),
                "sha256": _sha256(path),
            }
        )
    try:
        with zipfile.ZipFile(bundle) as archive:
            members = [row for row in archive.infolist() if not row.is_dir()]
            if len(members) != len({row.filename for row in members}):
                raise BlockedActivationRetentionError(
                    "blocked_activation_retention_bundle_archive_invalid"
                )
            archive_names = {row.filename for row in members}
            if archive_names != stage_names or any(
                PurePosixPath(row.filename).is_absolute()
                or ".." in PurePosixPath(row.filename).parts
                for row in members
            ):
                raise BlockedActivationRetentionError(
                    "blocked_activation_retention_stage_archive_mismatch"
                )
            rows_by_name = {row["path"]: row for row in rows}
            for member in members:
                digest_builder = hashlib.sha256()
                with archive.open(member) as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        digest_builder.update(chunk)
                digest = "sha256:" + digest_builder.hexdigest()
                mode = (member.external_attr >> 16) & 0o777
                source = rows_by_name[member.filename]
                if (
                    digest != source["sha256"]
                    or member.file_size != source["size_bytes"]
                    or mode != source["mode"]
                ):
                    raise BlockedActivationRetentionError(
                        "blocked_activation_retention_stage_archive_mismatch"
                    )
    except (OSError, zipfile.BadZipFile) as exc:
        raise BlockedActivationRetentionError(
            "blocked_activation_retention_bundle_archive_invalid"
        ) from exc
    return {
        "path": str(stage),
        "device": int(stage_info.st_dev),
        "inode": int(stage_info.st_ino),
        "mtime_ns": int(stage_info.st_mtime_ns),
        "file_count": len(rows),
        "size_bytes": sum(row["size_bytes"] for row in rows),
        "tree_digest": canonical_digest({"files": rows}),
        "archive_byte_identity_proven": True,
        "symlinks_followed": False,
    }


def _json_reference_paths(root: Path, *, blocker: str) -> list[Path]:
    root = _assert_directory(root, blocker=blocker)
    paths: list[Path] = []
    for path in sorted(root.rglob("*.json")):
        if path.is_symlink() or not path.is_file():
            raise BlockedActivationRetentionError(blocker)
        paths.append(path)
    return paths


def _assert_no_references(*, paths: Sequence[Path], needles: Sequence[str], blocker: str) -> None:
    for path in paths:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise BlockedActivationRetentionError(blocker + "_invalid") from exc
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"))
        if any(needle and needle in encoded for needle in needles):
            raise BlockedActivationRetentionError(blocker)


def _assert_no_forbidden_activation_artifacts(
    *, activation_root: Path, bundle_root: Path, allowed_step_logs: Sequence[Path]
) -> None:
    allowed_logs = {path.resolve() for path in allowed_step_logs}
    forbidden = (
        "paid_authority",
        "attempt-authority",
        "allocator_dry_run",
        "admission.json",
        "bound-request.json",
        "live_profile",
        "profile_publication_receipt",
        "standing_authorization",
        "terminal_rehearsal",
        "provider_execution",
        "adapter_result",
        "teardown",
        "billing",
    )
    for path in activation_root.rglob("*"):
        if path == bundle_root or bundle_root in path.parents:
            continue
        if path.is_symlink():
            raise BlockedActivationRetentionError("blocked_activation_retention_activation_symlink")
        if path.is_file() and path.resolve() in allowed_logs:
            continue
        if path.is_file() and any(token in path.name for token in forbidden):
            raise BlockedActivationRetentionError(
                "blocked_activation_retention_authority_or_execution_artifact_present"
            )


def _validate_step_logs(
    *,
    preparation: Mapping[str, Any],
    completed_step_ids: Sequence[str],
    bundle_receipt: Path,
    launch_set: Path,
) -> list[dict[str, Any]]:
    expected_attempted_steps = (
        ["provider_bundle", "immutable_manifest"]
        if list(completed_step_ids) == ["provider_bundle"]
        else ["provider_bundle", "immutable_manifest", "paid_authority"]
    )
    produces = {
        "provider_bundle": bundle_receipt,
        "immutable_manifest": launch_set / "manifest_publication_receipt.v1.json",
        "paid_authority": (
            launch_set / "task_evaluation_scene_configuration_paid_authority.v1.json"
        ),
    }
    rows = preparation.get("step_logs")
    if (
        not isinstance(rows, list)
        or not all(isinstance(row, Mapping) for row in rows)
        or [row.get("step_id") for row in rows] != expected_attempted_steps
    ):
        raise BlockedActivationRetentionError("blocked_activation_retention_step_logs_invalid")
    validated: list[dict[str, Any]] = []
    for row in rows:
        step_id = str(row["step_id"])
        base = produces[step_id]
        expected_stdout = base.with_name(f"{base.name}.{step_id}.stdout.log")
        expected_stderr = base.with_name(f"{base.name}.{step_id}.stderr.log")
        stdout = Path(str(row.get("stdout_path") or ""))
        stderr = Path(str(row.get("stderr_path") or ""))
        if (
            not stdout.is_absolute()
            or not stderr.is_absolute()
            or stdout.resolve() != expected_stdout
            or stderr.resolve() != expected_stderr
            or row.get("credential_redaction_applied") is not True
        ):
            raise BlockedActivationRetentionError("blocked_activation_retention_step_logs_invalid")
        stdout_record = _file_record(
            stdout, blocker="blocked_activation_retention_step_logs_invalid"
        )
        stderr_record = _file_record(
            stderr, blocker="blocked_activation_retention_step_logs_invalid"
        )
        if (
            row.get("stdout_sha256") != stdout_record["sha256"]
            or row.get("stderr_sha256") != stderr_record["sha256"]
        ):
            raise BlockedActivationRetentionError("blocked_activation_retention_step_logs_invalid")
        validated.append(
            {
                "step_id": step_id,
                "stdout": stdout_record,
                "stderr": stderr_record,
                "credential_redaction_applied": True,
            }
        )
    return validated


def build_blocked_activation_retention_plan(
    *,
    activation_root: str | Path,
    activation_base_root: str | Path = DEFAULT_ACTIVATION_BASE_ROOT,
    state_root: str | Path = DEFAULT_STATE_ROOT,
    activation_queue_root: str | Path = DEFAULT_ACTIVATION_QUEUE_ROOT,
    profile_dir: str | Path = DEFAULT_PROFILE_DIR,
    public_catalog: str | Path = DEFAULT_PUBLIC_CATALOG,
    standing_authorization_dir: str | Path = DEFAULT_STANDING_AUTHORIZATION_DIR,
    live_reference_roots: Sequence[str | Path] = DEFAULT_LIVE_REFERENCE_ROOTS,
    bundle_validator: BundleValidator = load_scene_configuration_provider_bundle_receipt,
) -> dict[str, Any]:
    """Return a deterministic plan only for a terminal, pre-authority activation."""

    base = _assert_directory(
        _absolute(activation_base_root, field="activation_base_root"),
        blocker="blocked_activation_retention_activation_base_invalid",
    )
    unresolved_activation = _absolute(activation_root, field="activation_root")
    activation = _assert_directory(
        unresolved_activation,
        blocker="blocked_activation_retention_activation_root_invalid",
    )
    if unresolved_activation.is_symlink() or activation.parent != base:
        raise BlockedActivationRetentionError(
            "blocked_activation_retention_activation_root_unmanaged"
        )
    activation_id = activation.name
    state = _assert_directory(
        _absolute(state_root, field="state_root"),
        blocker="blocked_activation_retention_state_root_invalid",
    )
    queue = _assert_directory(
        _absolute(activation_queue_root, field="activation_queue_root"),
        blocker="blocked_activation_retention_activation_queue_invalid",
    )
    if queue.parent != state:
        raise BlockedActivationRetentionError(
            "blocked_activation_retention_activation_queue_unmanaged"
        )
    matches = [
        (queue / queue_state / path.name)
        for queue_state in ("pending", "processing", "prepared", "blocked")
        for path in (queue / queue_state).glob(f"{activation_id}-*.json")
    ]
    if len(matches) != 1 or matches[0].parent.name != "blocked":
        raise BlockedActivationRetentionError(
            "blocked_activation_retention_activation_not_terminal_blocked"
        )
    envelope_path = matches[0]
    result_path = queue / "results" / envelope_path.name
    envelope, envelope_record = _read_sealed(
        envelope_path,
        schema_version=ENVELOPE_SCHEMA_VERSION,
        digest_field="envelope_digest",
        blocker="blocked_activation_retention_envelope_invalid",
    )
    result, result_record = _read_sealed(
        result_path,
        schema_version=RESULT_SCHEMA_VERSION,
        digest_field="result_digest",
        blocker="blocked_activation_retention_result_invalid",
    )
    request = envelope.get("request")
    if (
        not isinstance(request, Mapping)
        or request.get("activation_id") != activation_id
        or result.get("activation_id") != activation_id
        or result.get("status") != "blocked"
        or result.get("provider_mutation_performed") is not False
        or result.get("paid_execution_requested") is not False
        or not result.get("blockers")
    ):
        raise BlockedActivationRetentionError("blocked_activation_retention_terminal_claim_invalid")

    preparation_path = activation / "paid_lane_launch_preparation.v1.json"
    preparation_record = _file_record(
        preparation_path,
        blocker="blocked_activation_retention_preparation_receipt_invalid",
    )
    try:
        preparation = json.loads(preparation_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BlockedActivationRetentionError(
            "blocked_activation_retention_preparation_receipt_invalid"
        ) from exc
    completed = preparation.get("completed_steps") if isinstance(preparation, Mapping) else None
    completed_step_ids = (
        [row.get("step_id") for row in completed]
        if isinstance(completed, list) and all(isinstance(row, Mapping) for row in completed)
        else []
    )
    if (
        preparation.get("schema_version") != "paid_lane_launch_preparation.v1"
        or preparation.get("lane") != "task_evaluation_scene_configuration"
        or preparation.get("status") != "blocked"
        or preparation.get("provider_allocation_performed") is not False
        or preparation.get("paid_inference_performed") is not False
        or completed_step_ids
        not in (
            ["provider_bundle"],
            ["provider_bundle", "immutable_manifest"],
        )
    ):
        raise BlockedActivationRetentionError(
            "blocked_activation_retention_preparation_receipt_invalid"
        )
    bundle_receipt = Path(str(completed[0].get("artifact_path") or ""))
    if not bundle_receipt.is_absolute():
        raise BlockedActivationRetentionError("blocked_activation_retention_bundle_receipt_invalid")
    bundle_receipt = bundle_receipt.resolve()
    bundle_root = activation / "launch-set" / "bundle"
    if bundle_receipt.parent != bundle_root or bundle_receipt.name != (
        f"{BUNDLE_SCHEMA_VERSION}.receipt.json"
    ):
        raise BlockedActivationRetentionError("blocked_activation_retention_bundle_receipt_invalid")
    receipt_record = _file_record(
        bundle_receipt,
        blocker="blocked_activation_retention_bundle_receipt_invalid",
    )
    if completed[0].get("artifact_sha256") != receipt_record["sha256"]:
        raise BlockedActivationRetentionError("blocked_activation_retention_bundle_receipt_invalid")
    immutable_manifest_record: dict[str, Any] | None = None
    if completed_step_ids == ["provider_bundle", "immutable_manifest"]:
        immutable_manifest = activation / "launch-set" / "manifest_publication_receipt.v1.json"
        declared_manifest = Path(str(completed[1].get("artifact_path") or ""))
        if not declared_manifest.is_absolute() or declared_manifest.resolve() != immutable_manifest:
            raise BlockedActivationRetentionError(
                "blocked_activation_retention_immutable_manifest_invalid"
            )
        immutable_manifest_record = _file_record(
            immutable_manifest,
            blocker="blocked_activation_retention_immutable_manifest_invalid",
        )
        if completed[1].get("artifact_sha256") != immutable_manifest_record["sha256"]:
            raise BlockedActivationRetentionError(
                "blocked_activation_retention_immutable_manifest_invalid"
            )
    step_log_records = _validate_step_logs(
        preparation=preparation,
        completed_step_ids=completed_step_ids,
        bundle_receipt=bundle_receipt,
        launch_set=activation / "launch-set",
    )
    try:
        receipt = dict(bundle_validator(bundle_receipt))
    except (OSError, TypeError, ValueError) as exc:
        raise BlockedActivationRetentionError(
            "blocked_activation_retention_bundle_receipt_invalid"
        ) from exc
    bundle = Path(str(receipt.get("bundle_path") or ""))
    if not bundle.is_absolute():
        raise BlockedActivationRetentionError("blocked_activation_retention_bundle_receipt_invalid")
    bundle = bundle.resolve()
    if bundle.parent != bundle_root or bundle.name != (
        "task_evaluation_scene_configuration_provider_bundle.zip"
    ):
        raise BlockedActivationRetentionError("blocked_activation_retention_bundle_receipt_invalid")
    bundle_record = _file_record(bundle, blocker="blocked_activation_retention_bundle_invalid")
    if (
        receipt.get("schema_version") != BUNDLE_SCHEMA_VERSION
        or receipt.get("status") != "ready"
        or receipt.get("bundle_sha256") != bundle_record["sha256"]
        or receipt.get("bundle_size_bytes") != bundle_record["size_bytes"]
        or receipt.get("nested_provider_mutations_performed") != 0
        or receipt.get("evaluation_episode_executed") is not False
    ):
        raise BlockedActivationRetentionError("blocked_activation_retention_bundle_receipt_invalid")
    stage_record = _tree_and_archive_snapshot(bundle_root / "stage", bundle)
    _assert_no_forbidden_activation_artifacts(
        activation_root=activation,
        bundle_root=bundle_root,
        allowed_step_logs=[
            Path(record[stream]["path"])
            for record in step_log_records
            for stream in ("stdout", "stderr")
        ],
    )

    profiles = _absolute(profile_dir, field="profile_dir")
    standing = _absolute(standing_authorization_dir, field="standing_authorization_dir")
    catalog = _absolute(public_catalog, field="public_catalog")
    catalog_record = _file_record(
        catalog, blocker="blocked_activation_retention_public_catalog_invalid"
    )
    reference_paths = [catalog]
    reference_paths.extend(
        _json_reference_paths(profiles, blocker="blocked_activation_retention_profile_dir_invalid")
    )
    reference_paths.extend(
        _json_reference_paths(
            standing,
            blocker="blocked_activation_retention_standing_authorization_dir_invalid",
        )
    )
    live_roots = tuple(
        _absolute(path, field="live_reference_root") for path in live_reference_roots
    )
    for live_root in live_roots:
        reference_paths.extend(
            _json_reference_paths(
                live_root,
                blocker="blocked_activation_retention_live_reference_root_invalid",
            )
        )
    _assert_no_references(
        paths=reference_paths,
        needles=(activation_id, str(bundle), str(bundle_receipt)),
        blocker="blocked_activation_retention_live_reference_present",
    )

    plan: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "dry_run",
        "activation_id": activation_id,
        "activation_root": str(activation),
        "activation_base_root": str(base),
        "state_root": str(state),
        "activation_queue_root": str(queue),
        "profile_dir": str(profiles.resolve()),
        "public_catalog": str(catalog.resolve()),
        "standing_authorization_dir": str(standing.resolve()),
        "live_reference_roots": sorted(str(path.resolve()) for path in live_roots),
        "terminal_envelope": envelope_record,
        "terminal_envelope_digest": envelope["envelope_digest"],
        "terminal_result": result_record,
        "terminal_result_digest": result["result_digest"],
        "preparation_receipt": preparation_record,
        "immutable_manifest": immutable_manifest_record,
        "step_logs": step_log_records,
        "bundle_receipt": receipt_record,
        "public_catalog_snapshot": catalog_record,
        "removable_bundle": bundle_record,
        "removable_stage_tree": stage_record,
        "predicted_removed_bytes": bundle_record["size_bytes"] + stage_record["size_bytes"],
        "completed_preparation_steps": completed_step_ids,
        "profile_or_standing_authorization_observed": False,
        "live_queue_reference_observed": False,
        "paid_authority_observed": False,
        "provider_execution_observed": False,
        "provider_mutation_performed": False,
        "evidence_artifacts_removed": False,
        "preserved_evidence": [
            str(envelope_path),
            str(result_path),
            str(preparation_path),
            str(bundle_receipt),
            *(
                [str(immutable_manifest_record["path"])]
                if immutable_manifest_record is not None
                else []
            ),
            *[
                str(record[stream]["path"])
                for record in step_log_records
                for stream in ("stdout", "stderr")
            ],
            str(activation / "references"),
        ],
        "plan_digest": "",
    }
    plan["plan_digest"] = canonical_digest(plan, digest_field="plan_digest")
    return plan


def _read_plan(path: Path) -> dict[str, Any]:
    record = _file_record(path, blocker="blocked_activation_retention_plan_invalid")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BlockedActivationRetentionError("blocked_activation_retention_plan_invalid") from exc
    if (
        not isinstance(value, Mapping)
        or value.get("schema_version") != SCHEMA_VERSION
        or value.get("status") != "dry_run"
        or value.get("plan_digest") != canonical_digest(value, digest_field="plan_digest")
    ):
        raise BlockedActivationRetentionError("blocked_activation_retention_plan_invalid")
    return {**dict(value), "_plan_file_record": record}


def _remove_stage_tree(stage: Path, snapshot: Mapping[str, Any]) -> None:
    info = stage.lstat()
    if (
        stage.is_symlink()
        or not stage.is_dir()
        or info.st_dev != snapshot.get("device")
        or info.st_ino != snapshot.get("inode")
    ):
        raise BlockedActivationRetentionError("blocked_activation_retention_stage_tree_changed")
    for path in sorted(stage.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode):
            raise BlockedActivationRetentionError("blocked_activation_retention_stage_tree_changed")
        if stat.S_ISDIR(info.st_mode):
            path.rmdir()
        elif stat.S_ISREG(info.st_mode):
            path.unlink()
        else:
            raise BlockedActivationRetentionError("blocked_activation_retention_stage_tree_changed")
    stage.rmdir()


def apply_blocked_activation_retention_plan(
    *,
    dry_run_plan_path: str | Path,
    acknowledgement: str,
    receipt_out: str | Path,
    bundle_validator: BundleValidator = load_scene_configuration_provider_bundle_receipt,
) -> dict[str, Any]:
    """Take the global lock, reproduce the plan, then remove only its two targets."""

    if acknowledgement != APPLY_ACKNOWLEDGEMENT:
        raise BlockedActivationRetentionError(
            "blocked_activation_retention_acknowledgement_missing"
        )
    plan_path = _absolute(dry_run_plan_path, field="dry_run_plan")
    output = _absolute(receipt_out, field="receipt_out")
    if output.resolve() == plan_path.resolve():
        raise BlockedActivationRetentionError("blocked_activation_retention_receipt_overlaps_plan")
    plan = _read_plan(plan_path)
    state_root = _absolute(str(plan.get("state_root") or ""), field="state_root")
    with release_reference_lock(state_root, exclusive=True):
        current = build_blocked_activation_retention_plan(
            activation_root=str(plan.get("activation_root") or ""),
            activation_base_root=str(plan.get("activation_base_root") or ""),
            state_root=str(plan.get("state_root") or ""),
            activation_queue_root=str(plan.get("activation_queue_root") or ""),
            profile_dir=str(plan.get("profile_dir") or ""),
            public_catalog=str(plan.get("public_catalog") or ""),
            standing_authorization_dir=str(plan.get("standing_authorization_dir") or ""),
            live_reference_roots=tuple(plan.get("live_reference_roots") or ()),
            bundle_validator=bundle_validator,
        )
        if current.get("plan_digest") != plan.get("plan_digest"):
            raise BlockedActivationRetentionError("blocked_activation_retention_plan_changed")
        bundle = Path(str(current["removable_bundle"]["path"]))
        stage = Path(str(current["removable_stage_tree"]["path"]))
        bundle.unlink()
        _remove_stage_tree(stage, current["removable_stage_tree"])
        result: dict[str, Any] = {
            "schema_version": APPLY_SCHEMA_VERSION,
            "status": "applied",
            "activation_id": current["activation_id"],
            "dry_run_plan_path": str(plan_path.resolve()),
            "dry_run_plan_sha256": plan["_plan_file_record"]["sha256"],
            "dry_run_plan_digest": current["plan_digest"],
            "removed": [
                {
                    "path": str(bundle),
                    "kind": "provider_bundle_zip",
                    "sha256": current["removable_bundle"]["sha256"],
                    "removed_bytes": current["removable_bundle"]["size_bytes"],
                },
                {
                    "path": str(stage),
                    "kind": "provider_bundle_stage_tree",
                    "tree_digest": current["removable_stage_tree"]["tree_digest"],
                    "removed_bytes": current["removable_stage_tree"]["size_bytes"],
                },
            ],
            "removed_bytes": current["predicted_removed_bytes"],
            "predicted_removed_bytes": current["predicted_removed_bytes"],
            "bundle_receipt_preserved": current["bundle_receipt"],
            "terminal_envelope_digest": current["terminal_envelope_digest"],
            "terminal_result_digest": current["terminal_result_digest"],
            "evidence_artifacts_removed": False,
            "provider_mutation_performed": False,
            "symlinks_followed": False,
            "receipt_digest": "",
        }
        result["receipt_digest"] = canonical_digest(result, digest_field="receipt_digest")
        _write_exclusive(output, result)
        return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activation-root")
    parser.add_argument("--activation-base-root", default=str(DEFAULT_ACTIVATION_BASE_ROOT))
    parser.add_argument("--state-root", default=str(DEFAULT_STATE_ROOT))
    parser.add_argument("--activation-queue-root", default=str(DEFAULT_ACTIVATION_QUEUE_ROOT))
    parser.add_argument("--profile-dir", default=str(DEFAULT_PROFILE_DIR))
    parser.add_argument("--public-catalog", default=str(DEFAULT_PUBLIC_CATALOG))
    parser.add_argument(
        "--standing-authorization-dir",
        default=str(DEFAULT_STANDING_AUTHORIZATION_DIR),
    )
    parser.add_argument("--live-reference-root", action="append")
    parser.add_argument("--receipt-out", required=True)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--dry-run-plan")
    parser.add_argument("--ack")
    args = parser.parse_args(argv)
    try:
        if args.apply:
            if args.activation_root:
                raise BlockedActivationRetentionError(
                    "blocked_activation_retention_apply_parameters_must_come_from_plan"
                )
            result = apply_blocked_activation_retention_plan(
                dry_run_plan_path=str(args.dry_run_plan or ""),
                acknowledgement=str(args.ack or ""),
                receipt_out=args.receipt_out,
            )
        else:
            if not args.activation_root or args.dry_run_plan or args.ack:
                raise BlockedActivationRetentionError(
                    "blocked_activation_retention_cli_arguments_invalid"
                )
            result = build_blocked_activation_retention_plan(
                activation_root=args.activation_root,
                activation_base_root=args.activation_base_root,
                state_root=args.state_root,
                activation_queue_root=args.activation_queue_root,
                profile_dir=args.profile_dir,
                public_catalog=args.public_catalog,
                standing_authorization_dir=args.standing_authorization_dir,
                live_reference_roots=(
                    tuple(args.live_reference_root)
                    if args.live_reference_root
                    else DEFAULT_LIVE_REFERENCE_ROOTS
                ),
            )
            _write_exclusive(_absolute(args.receipt_out, field="receipt_out"), result)
    except (OSError, TypeError, ValueError) as exc:
        print(json.dumps({"status": "blocked", "blockers": [str(exc)]}, sort_keys=True))
        return 2
    print(
        json.dumps(
            {
                "status": result["status"],
                "removed_bytes": result.get("removed_bytes", 0),
                "predicted_removed_bytes": result.get("predicted_removed_bytes", 0),
                "provider_mutation_performed": False,
            },
            sort_keys=True,
        )
    )
    return 0


__all__ = [
    "APPLY_ACKNOWLEDGEMENT",
    "BlockedActivationRetentionError",
    "apply_blocked_activation_retention_plan",
    "build_blocked_activation_retention_plan",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
