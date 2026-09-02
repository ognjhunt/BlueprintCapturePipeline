"""Explicit pins that keep derived directories alive while a launch still needs them.

Derived inputs (a preparation's materialized references, a compiled episode,
an activation launch set) are reproducible, so they belong to the ``cache``
storage class and may be reclaimed.  What must never happen is reclaiming one
while a launch that depends on it is still in flight.  Rather than have the
reaper reconstruct that dependency graph by scanning every queue and run root,
the producers write a pin when they create a directory and the terminal step
releases it.  A pin also carries a TTL so that a release which never arrives
cannot protect bytes forever.

Pins are files under one root: ``<pins_root>/<kind>/<owner_id>.json``.  The
first writer wins; a pin is never rewritten except to record its release.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import time
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "control_plane_storage_pin.v1"
DEFAULT_PINS_ROOT = Path("/var/lib/blueprint/pipeline-control-plane/storage-pins")
PINS_ROOT_ENV = "BLUEPRINT_CONTROL_PLANE_STORAGE_PINS_ROOT"
DEFAULT_PIN_TTL_SECONDS = 30 * 24 * 60 * 60
PIN_KINDS = ("preparation", "compilation", "activation")
_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,191}\Z")


class ControlPlaneStoragePinError(RuntimeError):
    """A pin could not be written or read safely."""


def pins_root_from_environment(environ: Mapping[str, str] = os.environ) -> Path | None:
    raw = str(environ.get(PINS_ROOT_ENV) or "").strip()
    return Path(raw).expanduser() if raw else None


def _validated_owner(kind: str, owner_id: str) -> tuple[str, str]:
    if kind not in PIN_KINDS:
        raise ControlPlaneStoragePinError(f"control_plane_storage_pin_kind_invalid:{kind}")
    if not isinstance(owner_id, str) or _ID_RE.fullmatch(owner_id) is None:
        raise ControlPlaneStoragePinError("control_plane_storage_pin_owner_invalid")
    return kind, owner_id


def pin_path(pins_root: str | Path, kind: str, owner_id: str) -> Path:
    kind, owner_id = _validated_owner(kind, owner_id)
    return Path(pins_root).expanduser() / kind / f"{owner_id}.json"


def _write_atomic(path: Path, payload: Mapping[str, Any], *, exclusive: bool) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o750)
    descriptor, temporary_name = tempfile.mkstemp(prefix=".pin-", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, sort_keys=True, separators=(",", ":"))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.chmod(0o640)
        if exclusive:
            try:
                os.link(temporary, path)
            except FileExistsError:
                return False
        else:
            os.replace(temporary, path)
        return True
    finally:
        temporary.unlink(missing_ok=True)


def _load(path: Path) -> dict[str, Any] | None:
    if path.is_symlink() or not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(value, Mapping) or value.get("schema_version") != SCHEMA_VERSION:
        return None
    return dict(value)


def write_storage_pin(
    *,
    pins_root: str | Path,
    kind: str,
    owner_id: str,
    paths: Iterable[str | Path],
    depends_on: Sequence[Mapping[str, str]] = (),
    ttl_seconds: int = DEFAULT_PIN_TTL_SECONDS,
    now: Any = time.time,
) -> dict[str, Any]:
    """Pin ``paths`` for ``owner_id``; an existing pin is returned unchanged."""

    kind, owner_id = _validated_owner(kind, owner_id)
    if not isinstance(ttl_seconds, int) or isinstance(ttl_seconds, bool) or ttl_seconds <= 0:
        raise ControlPlaneStoragePinError("control_plane_storage_pin_ttl_invalid")
    pinned: list[str] = []
    for raw in paths:
        path = Path(raw)
        if not path.is_absolute():
            raise ControlPlaneStoragePinError("control_plane_storage_pin_path_not_absolute")
        pinned.append(str(path))
    dependencies: list[dict[str, str]] = []
    for row in depends_on:
        dependency_kind, dependency_id = _validated_owner(
            str(row.get("kind") or ""), str(row.get("owner_id") or "")
        )
        dependencies.append({"kind": dependency_kind, "owner_id": dependency_id})
    observed_at = float(now())
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": kind,
        "owner_id": owner_id,
        "paths": sorted(set(pinned)),
        "depends_on": sorted(dependencies, key=lambda row: (row["kind"], row["owner_id"])),
        "created_at_epoch": observed_at,
        "expires_at_epoch": observed_at + ttl_seconds,
        "released_at_epoch": None,
    }
    path = pin_path(pins_root, kind, owner_id)
    if _write_atomic(path, payload, exclusive=True):
        return payload
    existing = _load(path)
    if existing is None:
        raise ControlPlaneStoragePinError("control_plane_storage_pin_existing_unreadable")
    return existing


def pin_activation_best_effort(
    request: Any,
    activation_root: str | Path,
    *,
    pins_root: str | Path | None = None,
    environ: Mapping[str, str] = os.environ,
) -> dict[str, Any] | None:
    """Pin one activation's launch set plus its preparation and compilation.

    Called by the activation worker after a successful activation; the pins
    root comes from the unit environment unless given.  Never raises: a
    missing ledger must not disturb a sealed activation result.
    """

    root = Path(pins_root) if pins_root is not None else pins_root_from_environment(environ)
    if root is None or not isinstance(request, Mapping):
        return None
    try:
        activation_id = str(request["activation_id"])
        preparation_id = str(request["preparation"]["preparation_id"])
        return write_storage_pin(
            pins_root=root,
            kind="activation",
            owner_id=activation_id,
            paths=[Path(activation_root) / activation_id],
            depends_on=[
                {"kind": "preparation", "owner_id": preparation_id},
                {"kind": "compilation", "owner_id": preparation_id},
            ],
        )
    except (ControlPlaneStoragePinError, OSError, KeyError, TypeError):
        return None


def pin_status(pin: Mapping[str, Any], *, now: float) -> str:
    if pin.get("released_at_epoch") is not None:
        return "released"
    try:
        expires = float(pin.get("expires_at_epoch", 0))
    except (TypeError, ValueError):
        return "released"
    return "live" if expires > now else "expired"


def load_storage_pins(pins_root: str | Path, *, now: Any = time.time) -> list[dict[str, Any]]:
    root = Path(pins_root).expanduser()
    observed_at = float(now())
    pins: list[dict[str, Any]] = []
    if not root.is_dir():
        return pins
    for kind in PIN_KINDS:
        directory = root / kind
        if not directory.is_dir() or directory.is_symlink():
            continue
        for path in sorted(directory.glob("*.json")):
            pin = _load(path)
            if pin is None or pin.get("kind") != kind:
                continue
            pin["status"] = pin_status(pin, now=observed_at)
            pins.append(pin)
    return pins


def live_pinned_paths(pins_root: str | Path, *, now: Any = time.time) -> set[str]:
    return {
        str(path)
        for pin in load_storage_pins(pins_root, now=now)
        if pin["status"] == "live"
        for path in pin.get("paths") or []
    }


def release_storage_pin(
    *, pins_root: str | Path, kind: str, owner_id: str, now: Any = time.time
) -> dict[str, Any]:
    """Release one pin and every dependency no other live pin still needs."""

    kind, owner_id = _validated_owner(kind, owner_id)
    observed_at = float(now())
    released: list[dict[str, str]] = []
    pending = [(kind, owner_id)]
    seen: set[tuple[str, str]] = set()
    while pending:
        current_kind, current_id = pending.pop()
        if (current_kind, current_id) in seen:
            continue
        seen.add((current_kind, current_id))
        path = pin_path(pins_root, current_kind, current_id)
        pin = _load(path)
        if pin is None:
            continue
        if pin.get("released_at_epoch") is None:
            pin["released_at_epoch"] = observed_at
            _write_atomic(path, pin, exclusive=False)
            released.append({"kind": current_kind, "owner_id": current_id})
        for dependency in pin.get("depends_on") or []:
            dependency_kind = str(dependency.get("kind") or "")
            dependency_id = str(dependency.get("owner_id") or "")
            if _still_needed(
                pins_root,
                kind=dependency_kind,
                owner_id=dependency_id,
                now=observed_at,
                excluding=seen,
            ):
                continue
            pending.append((dependency_kind, dependency_id))
    return {
        "schema_version": "control_plane_storage_pin_release.v1",
        "kind": kind,
        "owner_id": owner_id,
        "released": released,
        "released_at_epoch": observed_at,
    }


def _still_needed(
    pins_root: str | Path,
    *,
    kind: str,
    owner_id: str,
    now: float,
    excluding: set[tuple[str, str]],
) -> bool:
    for pin in load_storage_pins(pins_root, now=lambda: now):
        if pin["status"] != "live":
            continue
        if (str(pin["kind"]), str(pin["owner_id"])) in excluding:
            continue
        for dependency in pin.get("depends_on") or []:
            if dependency.get("kind") == kind and dependency.get("owner_id") == owner_id:
                return True
    return False


__all__ = [
    "ControlPlaneStoragePinError",
    "DEFAULT_PINS_ROOT",
    "DEFAULT_PIN_TTL_SECONDS",
    "PINS_ROOT_ENV",
    "PIN_KINDS",
    "SCHEMA_VERSION",
    "live_pinned_paths",
    "load_storage_pins",
    "pin_activation_best_effort",
    "pin_path",
    "pin_status",
    "pins_root_from_environment",
    "release_storage_pin",
    "write_storage_pin",
]
