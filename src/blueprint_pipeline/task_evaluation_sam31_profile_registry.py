"""Resolve server-owned SAM profiles by content, without per-scene unit edits."""

from __future__ import annotations

import os
import re
import secrets
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .task_evaluation_scene_configuration_sam31_plan import PROFILE_ENV, PROFILE_SCHEMA
from .task_evaluation_scene_configuration_submission_inputs import read, require, sha

REGISTRY_ENV = "BLUEPRINT_TASK_EVALUATION_SAM31_PREPARATION_PROFILE_DIR"

#: One fixed, scene-independent content-addressed registry every control-plane
#: unit resolves from (``REGISTRY_ENV`` in the base unit files) and the
#: scene-progression factory / provisioning write into. Placing it under the
#: shared task-evaluation input root lets the deploy's unit-sandbox installer
#: create it blueprint-owned and 0750 without a per-scene unit edit, so the
#: look-ahead admission replay -- which runs in a unit that never received a
#: per-scene ``PROFILE_ENV`` drop-in -- can still resolve the profile by digest.
DEFAULT_PROFILE_REGISTRY_ROOT = (
    "/var/lib/blueprint/task-evaluation-inputs/sam31-profile-registry"
)


def resolve_sam31_profile(plan: Mapping[str, Any]) -> Path:
    digest = plan.get("server_profile_sha256")
    require(isinstance(digest, str) and re.fullmatch(r"sha256:[0-9a-f]{64}", digest) is not None,
            "sam31_server_profile_digest_invalid")
    registry = os.getenv(REGISTRY_ENV, "")
    if registry:
        root = Path(registry)
        require(root.is_absolute() and not any(p.is_symlink() for p in (root, *root.parents)),
                "sam31_server_profile_registry_unsafe")
        candidate = root / (digest.removeprefix("sha256:") + ".json")
        if candidate.exists() or candidate.is_symlink():
            require(sha(candidate) == digest, "sam31_server_profile_changed")
            return candidate
    # Compatibility for existing immutable plans. No caller-supplied path is
    # accepted; even the legacy service binding must match the plan's digest.
    configured = os.getenv(PROFILE_ENV, "")
    require(bool(configured), "sam31_server_profile_missing")
    candidate = Path(configured)
    require(candidate.is_absolute() and sha(candidate) == digest, "sam31_server_profile_changed")
    return candidate


def register_sam31_profile(*, profile_path: str | Path, registry_root: str | Path) -> dict[str, Any]:
    source = Path(profile_path)
    profile = read(source, digest_field="profile_digest")
    require(profile.get("schema_version") == PROFILE_SCHEMA
            and re.fullmatch(r"[0-9a-f]{40}", str(profile.get("source_commit"))) is not None,
            "sam31_server_profile_invalid")
    root = Path(registry_root)
    require(root.is_absolute() and not any(p.is_symlink() for p in (root, *root.parents)),
            "sam31_server_profile_registry_unsafe")
    root.mkdir(parents=True, exist_ok=True, mode=0o750)
    require(root.stat().st_mode & 0o002 == 0, "sam31_server_profile_registry_world_writable")
    digest = sha(source)
    destination = root / (digest.removeprefix("sha256:") + ".json")
    payload = source.read_bytes()
    temporary = root / (".profile-" + secrets.token_hex(12))
    try:
        descriptor = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o440)
        with os.fdopen(descriptor, "wb") as stream:
            os.fchown(stream.fileno(), root.stat().st_uid, root.stat().st_gid)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        require(sha(temporary) == digest, "sam31_server_profile_changed")
        try:
            os.link(temporary, destination, follow_symlinks=False)
        except FileExistsError:
            require(sha(destination) == digest, "sam31_server_profile_registry_conflict")
        directory = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)
    return {"path": str(destination), "sha256": digest, "size_bytes": len(payload),
            "profile_digest": profile["profile_digest"], "source_commit": profile["source_commit"]}
