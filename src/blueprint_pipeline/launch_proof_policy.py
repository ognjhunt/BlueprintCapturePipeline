"""Launch-proof mode policy shared by production-facing pipeline stages."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping

from .common import parse_bool


PRODUCTION_MODE = "production"


def launch_proof_mode(env: Mapping[str, str] | None = None) -> str:
    source = os.environ if env is None else env
    return str(source.get("BLUEPRINT_LAUNCH_PROOF_MODE") or "").strip().lower()


def production_launch_mode(env: Mapping[str, str] | None = None) -> bool:
    return launch_proof_mode(env) == PRODUCTION_MODE


def production_forces_true(name: str, *, default: bool = False, env: Mapping[str, str] | None = None) -> bool:
    if production_launch_mode(env):
        return True
    source = os.environ if env is None else env
    return parse_bool(source.get(name), default=default)


def production_forces_false(name: str, *, default: bool = False, env: Mapping[str, str] | None = None) -> bool:
    if production_launch_mode(env):
        return False
    source = os.environ if env is None else env
    return parse_bool(source.get(name), default=default)


def runtime_required(env: Mapping[str, str] | None = None) -> bool:
    return production_launch_mode(env) or parse_bool(
        (os.environ if env is None else env).get("SITE_WORLD_RUNTIME_REQUIRED"),
        default=False,
    )


def runtime_url_present(env: Mapping[str, str] | None = None) -> bool:
    source = os.environ if env is None else env
    return bool(str(source.get("SITE_WORLD_RUNTIME_SERVICE_URL") or "").strip())


def fallback_geometry_launchable_allowed(env: Mapping[str, str] | None = None) -> bool:
    if production_launch_mode(env):
        return False
    source = os.environ if env is None else env
    return parse_bool(source.get("BLUEPRINT_ALLOW_FALLBACK_GEOMETRY_LAUNCH"), default=True)


def buyer_access_required(env: Mapping[str, str] | None = None) -> bool:
    return production_launch_mode(env) or parse_bool(
        (os.environ if env is None else env).get("PIPELINE_SYNC_BUYER_ACCESS_REQUIRED"),
        default=False,
    )


def relative_artifact_checksum(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
