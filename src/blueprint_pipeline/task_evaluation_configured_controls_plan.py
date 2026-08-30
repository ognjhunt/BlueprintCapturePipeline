"""Materialize one immutable configured-scene to controls progression plan."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest


PLAN_SCHEMA_VERSION = "task_evaluation_configured_controls_progression_plan.v2"
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,255}")
_COMMIT = re.compile(r"[0-9a-f]{40}")
_TOP_LEVEL_PATHS = {
    "robot_mount_interface_path",
    "scene_camera_calibration_path",
    "base_pose_candidate_path",
    "cameras_path",
    "runtime_binding_path",
}
_COMMON_PHASE_PATHS = {
    "release_window_template_path",
    "authorization_path",
    "launch_authority_path",
}


class TaskEvaluationConfiguredControlsPlanError(ValueError):
    """A plan input was absent, mutable, or disagreed with its source launch."""


def _payload(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


def _file_identity(path: Path) -> dict[str, Any]:
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise TaskEvaluationConfiguredControlsPlanError(
            "configured_controls_plan_artifact_invalid"
        )
    digest = hashlib.sha256()
    size = 0
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
                size += len(chunk)
        value = json.loads(path.read_text(encoding="utf-8"))
        metadata = path.stat()
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TaskEvaluationConfiguredControlsPlanError(
            "configured_controls_plan_artifact_invalid"
        ) from exc
    if not isinstance(value, Mapping):
        raise TaskEvaluationConfiguredControlsPlanError(
            "configured_controls_plan_artifact_invalid"
        )
    return {
        "path": str(path),
        "digest": "sha256:" + digest.hexdigest(),
        "size_bytes": size,
        "mode": f"{stat.S_IMODE(metadata.st_mode):04o}",
        "value": dict(value),
    }


def _path(value: Any) -> Path:
    path = Path(str(value or "")).expanduser()
    if not path.is_absolute():
        raise TaskEvaluationConfiguredControlsPlanError(
            "configured_controls_plan_artifact_path_not_absolute"
        )
    return path


def _flatten_bindings(bindings: Mapping[str, Any]) -> dict[str, Path]:
    if set(bindings) != _TOP_LEVEL_PATHS | {"phases"}:
        raise TaskEvaluationConfiguredControlsPlanError(
            "configured_controls_plan_bindings_invalid"
        )
    phases = bindings.get("phases")
    if not isinstance(phases, Mapping) or set(phases) != {"construction", "controls"}:
        raise TaskEvaluationConfiguredControlsPlanError(
            "configured_controls_plan_bindings_invalid"
        )
    flattened = {name: _path(bindings[name]) for name in sorted(_TOP_LEVEL_PATHS)}
    for phase_name in ("construction", "controls"):
        phase = phases.get(phase_name)
        expected = _COMMON_PHASE_PATHS | (
            {"lineage_path"} if phase_name == "construction" else set()
        )
        if not isinstance(phase, Mapping) or set(phase) != expected:
            raise TaskEvaluationConfiguredControlsPlanError(
                "configured_controls_plan_bindings_invalid"
            )
        for name in sorted(expected):
            flattened[f"phases.{phase_name}.{name}"] = _path(phase[name])
    return flattened


def _validate_commit_fields(value: Mapping[str, Any], expected_commit: str) -> None:
    for field in ("source_commit", "expected_production_commit"):
        observed = value.get(field)
        if observed is not None and observed != expected_commit:
            raise TaskEvaluationConfiguredControlsPlanError(
                "configured_controls_plan_artifact_commit_mismatch"
            )


def materialize_configured_controls_plan(
    *,
    source_launch_id: str,
    launch_state_root: str | Path,
    expected_production_commit: str,
    submitted_by: str,
    bindings: Mapping[str, Any],
    plan_root: str | Path,
    profile_dir: str | Path,
) -> dict[str, Any]:
    """Validate exact source/authority bytes and write one idempotent 0440 plan."""

    if (
        _IDENTIFIER.fullmatch(source_launch_id) is None
        or _COMMIT.fullmatch(expected_production_commit) is None
        or not submitted_by.strip()
    ):
        raise TaskEvaluationConfiguredControlsPlanError(
            "configured_controls_plan_identity_invalid"
        )
    launch_root = Path(launch_state_root).expanduser()
    if not launch_root.is_absolute() or launch_root.is_symlink():
        raise TaskEvaluationConfiguredControlsPlanError(
            "configured_controls_plan_launch_root_invalid"
        )
    from .task_evaluation_configured_controls_progression_worker import (
        _validate_source,
    )

    terminal, receipt, _ = _validate_source(launch_root / source_launch_id)
    source_configuration_commit = receipt.get("source_commit")
    if (
        receipt.get("launch_id") != source_launch_id
        or not isinstance(source_configuration_commit, str)
        or _COMMIT.fullmatch(source_configuration_commit) is None
    ):
        raise TaskEvaluationConfiguredControlsPlanError(
            "configured_controls_plan_source_launch_mismatch"
        )
    paths = _flatten_bindings(bindings)
    inventory: dict[str, dict[str, Any]] = {}
    for name, path in paths.items():
        row = _file_identity(path)
        # Mount and calibration bytes are products of the sealed configuration
        # release. Every downstream input belongs to the separately qualified
        # construction release. Do not blur those two provenance boundaries.
        artifact_commit = (
            source_configuration_commit
            if name
            in {
                "robot_mount_interface_path",
                "scene_camera_calibration_path",
            }
            else expected_production_commit
        )
        _validate_commit_fields(row["value"], artifact_commit)
        inventory[name] = row
    configuration_run_id = str(terminal.get("run_id") or "")
    namespace = (
        f"{configuration_run_id}-franka-controls-"
        f"{expected_production_commit[:12]}"
    )
    expected_activation_ids = {
        phase: f"{namespace}-episode-{phase}"
        for phase in ("construction", "controls")
    }
    profiles = Path(profile_dir).expanduser()
    if not profiles.is_absolute() or profiles.is_symlink() or not profiles.is_dir():
        raise TaskEvaluationConfiguredControlsPlanError(
            "configured_controls_plan_profile_dir_invalid"
        )
    plan = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "enabled": True,
        "source_launch_id": source_launch_id,
        "source_launch_receipt_digest": receipt["receipt_digest"],
        "source_configuration_commit": source_configuration_commit,
        "expected_production_commit": expected_production_commit,
        "submitted_by": submitted_by,
        "profile_dir": str(profiles),
        **{name: str(paths[name]) for name in _TOP_LEVEL_PATHS},
        "phases": {
            phase: {
                **{
                    name: str(paths[f"phases.{phase}.{name}"])
                    for name in _COMMON_PHASE_PATHS
                },
                **(
                    {
                        "lineage_path": str(
                            paths["phases.construction.lineage_path"]
                        )
                    }
                    if phase == "construction"
                    else {}
                ),
            }
            for phase in ("construction", "controls")
        },
        "artifact_inventory": {
            name: {key: value for key, value in row.items() if key != "value"}
            for name, row in sorted(inventory.items())
        },
        "future_outputs": {
            phase: {
                "expected_activation_id": expected_activation_ids[phase],
            }
            for phase in ("construction", "controls")
        },
        "provider_mutation_performed": False,
        "paid_execution_requested": False,
        "plan_digest": "",
    }
    plan["plan_digest"] = canonical_digest(plan, digest_field="plan_digest")
    root = Path(plan_root).expanduser()
    if not root.is_absolute() or root.is_symlink():
        raise TaskEvaluationConfiguredControlsPlanError(
            "configured_controls_plan_root_invalid"
        )
    root.mkdir(parents=True, exist_ok=True, mode=0o750)
    if not root.is_dir():
        raise TaskEvaluationConfiguredControlsPlanError(
            "configured_controls_plan_root_invalid"
        )
    # Scope the plan to its production commit. A redeploy authors a plan
    # whose bytes differ only by commit, and a launch-id-only filename turns
    # that into configured_controls_plan_immutable_conflict forever.
    destination = root / f"{source_launch_id}-{expected_production_commit[:12]}.json"
    payload = _payload(plan)
    status = "materialized"
    try:
        with destination.open("xb") as stream:
            stream.write(payload)
        root_metadata = root.stat()
        destination_metadata = destination.stat()
        if (
            destination_metadata.st_uid != root_metadata.st_uid
            or destination_metadata.st_gid != root_metadata.st_gid
        ):
            os.chown(destination, root_metadata.st_uid, root_metadata.st_gid)
        destination.chmod(0o440)
    except FileExistsError:
        if destination.is_symlink() or destination.read_bytes() != payload:
            raise TaskEvaluationConfiguredControlsPlanError(
                "configured_controls_plan_immutable_conflict"
            ) from None
        status = "replayed"
    readback = destination.stat()
    root_metadata = root.stat()
    if (
        destination.read_bytes() != payload
        or readback.st_uid != root_metadata.st_uid
        or readback.st_gid != root_metadata.st_gid
        or stat.S_IMODE(readback.st_mode) != 0o440
    ):
        raise TaskEvaluationConfiguredControlsPlanError(
            "configured_controls_plan_readback_mismatch"
        )
    return {
        "status": status,
        "plan_path": str(destination),
        "plan_digest": plan["plan_digest"],
        "source_launch_id": source_launch_id,
        "provider_mutation_performed": False,
        "paid_execution_requested": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-launch-id", required=True)
    parser.add_argument("--launch-state-root", required=True)
    parser.add_argument("--expected-production-commit", required=True)
    parser.add_argument("--submitted-by", required=True)
    parser.add_argument("--bindings", required=True)
    parser.add_argument("--plan-root", required=True)
    parser.add_argument("--profile-dir", required=True)
    args = parser.parse_args(argv)
    try:
        bindings = json.loads(Path(args.bindings).read_text(encoding="utf-8"))
        if not isinstance(bindings, Mapping):
            raise TaskEvaluationConfiguredControlsPlanError(
                "configured_controls_plan_bindings_invalid"
            )
        result = materialize_configured_controls_plan(
            source_launch_id=args.source_launch_id,
            launch_state_root=args.launch_state_root,
            expected_production_commit=args.expected_production_commit,
            submitted_by=args.submitted_by,
            bindings=bindings,
            plan_root=args.plan_root,
            profile_dir=args.profile_dir,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, TaskEvaluationConfiguredControlsPlanError) as exc:
        print(json.dumps({"status": "blocked", "blockers": [str(exc)]}, sort_keys=True))
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
