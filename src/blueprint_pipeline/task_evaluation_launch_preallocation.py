"""Prove from disk that a started launch never reached its live allocator.

A launch profile that carries ``standing_launch_authorization`` admits a live
(``--execute``) allocator call only after the dispatcher has recorded the
launch in that authorization's consumption ledger,
``<standing dir>/consumed/<profile_id>/<launch_id>.json``
(``consume_standing_authorization_once``, written before the allocator runs).
A live request whose authorization is not admitted, or whose consumption is not
recorded, is refused by the dispatcher before any allocator argv runs. A dry run
without ``--execute`` cannot mutate a provider.

So a run directory whose profile requires a one-use standing authorization,
whose authorization is present in the directory read here (proof that this is
the dispatcher's directory), and whose launch has no consumption record, no
allocator output and no launch receipt, never invoked the paid allocator.

capture-447ae4f4, 2026-11-01 02:58Z: the dispatcher raised after
``launch_started.json`` and before the allocator (#3224). Its lease kept the
profile's 7.5 h hard TTL and its scene-configuration row kept a full $20 hold
that no evidence could release.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

CONSUMED_DIRECTORY = "consumed"
_ALLOCATOR_OUTPUT = "allocator"


def _file(path: Path) -> dict[str, Any]:
    return {"path": str(path), "digest": "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()}


def _sealed_json(path: Path, field: str) -> dict[str, Any] | None:
    from .task_evaluation_launch_dispatcher import canonical_digest

    if path.is_symlink() or not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(value, dict) or value.get(field) != canonical_digest(value, digest_field=field):
        return None
    return value


def _identifier(value: Any) -> bool:
    return isinstance(value, str) and bool(value) and Path(value).name == value and value not in {".", ".."}


def preallocation_stop_evidence(
    *, run_root: str | Path, launch_id: str, state_root: str | Path,
) -> dict[str, Any] | None:
    """Evidence that this run stopped before its live allocator, else ``None``.

    ``None`` whenever anything is missing, unreadable or points the other way:
    the caller then keeps its existing, more conservative handling.
    """
    from .task_evaluation_launch_dispatcher import standing_authorization_directory

    run = Path(run_root)
    if not _identifier(launch_id) or run.name != launch_id or run.is_symlink() or not run.is_dir():
        return None
    for name in ("launch_receipt.json", _ALLOCATOR_OUTPUT):
        if (run / name).exists() or (run / name).is_symlink():
            return None
    profile = _sealed_json(run / "launch_profile.json", "profile_digest")
    started = _sealed_json(run / "launch_started.json", "started_digest")
    if profile is None or started is None or started.get("launch_id") != launch_id:
        return None
    requirement = profile.get("standing_launch_authorization")
    profile_id = profile.get("profile_id")
    if not isinstance(requirement, Mapping) or not requirement or not _identifier(profile_id):
        return None
    directory = Path(standing_authorization_directory(state_root))
    authorization = directory / f"{profile_id}.json"
    if authorization.is_symlink() or not authorization.is_file():
        return None  # Not provably the dispatcher's directory: an absent record would prove nothing.
    consumption = directory / CONSUMED_DIRECTORY / str(profile_id) / f"{launch_id}.json"
    if consumption.exists() or consumption.is_symlink():
        return None
    return {
        "basis": "standing_authorization_unconsumed",
        "launch_profile": _file(run / "launch_profile.json"),
        "launch_started": _file(run / "launch_started.json"),
        "standing_authorization_path": str(authorization),
        "consumption_record_path": str(consumption),
    }


def preallocation_evidence_still_holds(evidence: Mapping[str, Any], *, run_root: str | Path) -> bool:
    """Re-check retained evidence: no run or consumption evidence appeared since.

    The authorization file itself is not re-required here: it may be retired
    later, and its absence would not make a recorded consumption appear.
    """
    run = Path(run_root)
    try:
        if evidence.get("basis") != "standing_authorization_unconsumed":
            return False
        for name in ("launch_receipt.json", _ALLOCATOR_OUTPUT):
            if (run / name).exists() or (run / name).is_symlink():
                return False
        for key, name in (("launch_profile", "launch_profile.json"), ("launch_started", "launch_started.json")):
            ref = evidence.get(key) or {}
            if ref.get("path") != str(run / name) or _file(run / name) != dict(ref):
                return False
        consumption = Path(str(evidence.get("consumption_record_path") or ""))
        return (consumption.name == run.name + ".json"
                and consumption.parent.parent.name == CONSUMED_DIRECTORY
                and not consumption.exists() and not consumption.is_symlink())
    except (OSError, TypeError, ValueError, AttributeError):
        return False


__all__ = ["preallocation_evidence_still_holds", "preallocation_stop_evidence"]
