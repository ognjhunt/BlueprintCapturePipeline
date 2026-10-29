"""Operator-selected default robot bytes using the existing USD closure path.

Beta: bind the Blueprint-operated robot at qualification, staging and closeout
(AUT-06). Selecting/staging a model grants no qualification or paid authority.
The caller must pass the path actually opened by the execution runtime.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from collections.abc import Mapping
from contextlib import ExitStack
from pathlib import Path, PurePosixPath
from typing import Any

from .decision_evidence_contracts import cross_runtime_canonical_digest as digest
from .droid_policy_canary_embodiment import DROID_POLICY_CANARY_PRESET_ID
from .team_robot_model_delivery import (
    _anchored_directory_fd, _walk_directory_fd, copy_prepared_closure,
)

CONFIG_SCHEMA = "blueprint.default_robot_model.v1"
DELIVERY_SCHEMA = "blueprint.default_robot_model_delivery.v1"
_SHA = re.compile(r"sha256:[0-9a-f]{64}")


class BlueprintRobotModelError(ValueError):
    """Fixed errors without operator filesystem or credential content."""


def _require(ok: Any, code: str) -> None:
    if not ok:
        raise BlueprintRobotModelError("blueprint_robot_model_" + code)


def _selected(profile: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    from .deployment_embodiment_asset_import import verify_prepared_asset_directory

    config = profile.get("default_robot_model")
    _require(isinstance(config, Mapping) and set(config) == {
        "schema_version", "embodiment_preset_id", "directory", "model_sha256", "dependency_closure_digest"},
        "operator_binding_missing")
    _require(config["schema_version"] == CONFIG_SCHEMA
             and config["embodiment_preset_id"] == DROID_POLICY_CANARY_PRESET_ID
             and all(_SHA.fullmatch(str(config[k])) for k in ("model_sha256", "dependency_closure_digest")),
             "operator_binding_invalid")
    path = Path(str(config["directory"]))
    try:
        status = path.lstat()
        _require(path.is_absolute() and stat.S_ISDIR(status.st_mode) and not stat.S_ISLNK(status.st_mode)
                 and not status.st_mode & 0o022 and status.st_uid in {0, os.getuid()}, "not_operator_owned")
        imported = verify_prepared_asset_directory(path)
    except (OSError, ValueError):
        raise BlueprintRobotModelError("blueprint_robot_model_prepared_asset_invalid") from None
    _require(imported["model"]["sha256"] == config["model_sha256"]
             and imported["dependency_closure"]["closure_digest"] == config["dependency_closure_digest"],
             "operator_binding_changed")
    return dict(config), imported


def _verify_runtime_closure(directory: Path, closure: Mapping[str, Any]) -> None:
    """Authenticate each opened regular file with directory descriptors held.

    The closure itself comes from the already verified operator import, rather
    than a job-controlled manifest. This uses the same anchored traversal as
    the canonical team model copier, including across path replacement races.
    """
    try:
        with ExitStack() as stack:
            root = _anchored_directory_fd(directory, create=False)
            stack.callback(os.close, root)
            for row in closure["files"]:
                parts = PurePosixPath(row["path"]).parts
                parent = _walk_directory_fd(root, parts[:-1], create=False)
                try:
                    fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
                finally:
                    os.close(parent)
                with os.fdopen(fd, "rb") as stream:
                    status = os.fstat(stream.fileno())
                    _require(stat.S_ISREG(status.st_mode) and status.st_size == row["size_bytes"], "runtime_bytes_changed")
                    sha, count = hashlib.sha256(), 0
                    while data := stream.read(1024 * 1024):
                        count += len(data)
                        _require(count <= row["size_bytes"], "runtime_bytes_changed")
                        sha.update(data)
                    _require(count == row["size_bytes"] and "sha256:" + sha.hexdigest() == row["sha256"],
                             "runtime_bytes_changed")
    except BlueprintRobotModelError:
        raise
    except OSError:
        raise BlueprintRobotModelError("blueprint_robot_model_runtime_bytes_unavailable") from None


def require_default_robot_model(*, profile: Mapping[str, Any], model_path: Path,
                                expected_model_sha256: str, expected_closure_digest: str) -> dict[str, Any]:
    """Recheck operator preparation and every byte the runtime is expected to open."""
    config, imported = _selected(profile)
    closure = imported["dependency_closure"]
    _require(config["model_sha256"] == expected_model_sha256
             and closure["closure_digest"] == expected_closure_digest
             and Path(model_path).name == closure["model_path"], "execution_binding_mismatch")
    _verify_runtime_closure(Path(model_path).parent, closure)
    return {"schema_version": DELIVERY_SCHEMA, "embodiment_preset_id": DROID_POLICY_CANARY_PRESET_ID,
            "model_sha256": config["model_sha256"], "dependency_closure": closure,
            "source_import_digest": imported["import_digest"],
            "claim_ceiling": "source_asset_not_qualified", "execution_authorized": False}


def stage_default_robot_model(*, profile: Mapping[str, Any], target_directory: Path) -> dict[str, Any]:
    """Copy the exact closed asset once; restart only adopts matching retained bytes."""
    config, imported = _selected(profile)
    closure = imported["dependency_closure"]
    target = Path(target_directory)
    if target.exists():
        _verify_runtime_closure(target, closure)
    else:
        copy_prepared_closure(source=Path(config["directory"]), destination=target, closure=closure)
    return require_default_robot_model(profile=profile, model_path=target / closure["model_path"],
        expected_model_sha256=config["model_sha256"], expected_closure_digest=config["dependency_closure_digest"])


def delivery_digest(receipt: Mapping[str, Any]) -> str:
    """Portable identity; deliberately excludes the operator's local source path."""
    return digest(json.loads(json.dumps(receipt, allow_nan=False)))
