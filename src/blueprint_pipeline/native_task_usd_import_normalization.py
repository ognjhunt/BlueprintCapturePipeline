"""Make leaf USD assets safe to import into one environment-owned physics scene.

Standalone SimReady assets often carry a ``PhysicsScene`` so they can be
opened and exercised by themselves.  Referencing that same asset into an Arena
environment composes a second physics scene beside the environment scene.
PhysX then has two lifecycle/device owners; the first symptom can be a vague
step-rate warning or CPU tensors returned to a CUDA runtime.

This module keeps the source asset immutable and derives an import copy with
every active embedded ``PhysicsScene`` removed or deactivated.  The Arena
environment remains the sole owner of simulation cadence and device.  It is a
task-neutral USD composition rule, not a refrigerator repair.
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest


SCHEMA_VERSION = "native_task_usd_import_normalization.v1"


class NativeTaskUsdImportNormalizationError(ValueError):
    """Stable failures while auditing or deriving an import-safe USD."""

    def __init__(self, errors: list[str]):
        self.errors = tuple(sorted(set(str(error) for error in errors if str(error))))
        super().__init__(";".join(self.errors))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _usd_modules() -> tuple[Any, Any]:
    try:
        from pxr import Usd, UsdPhysics
    except ImportError as exc:  # pragma: no cover - exercised by deployment preflight
        raise NativeTaskUsdImportNormalizationError(
            ["native_task_usd_import_normalization_runtime_missing"]
        ) from exc
    return Usd, UsdPhysics


def _active_physics_scene_paths(stage: Any, *, usd_physics: Any) -> list[str]:
    return sorted(
        str(prim.GetPath())
        for prim in stage.TraverseAll()
        if prim.IsActive()
        and (
            prim.GetTypeName() == "PhysicsScene"
            or prim.IsA(usd_physics.Scene)
        )
    )


def inspect_environment_import_usd(
    path: str | Path, *, semantic_role: str
) -> dict[str, Any]:
    """Return a digest-bound audit of active embedded physics scenes."""

    source = Path(path).expanduser().resolve()
    role = str(semantic_role or "")
    if not role or not source.is_file() or source.is_symlink():
        raise NativeTaskUsdImportNormalizationError(
            [f"native_task_usd_import_asset_invalid:{role or 'unknown'}"]
        )
    Usd, UsdPhysics = _usd_modules()
    try:
        stage = Usd.Stage.Open(str(source), load=Usd.Stage.LoadAll)
    except Exception as exc:
        raise NativeTaskUsdImportNormalizationError(
            [f"native_task_usd_import_asset_unreadable:{role}"]
        ) from exc
    if stage is None:
        raise NativeTaskUsdImportNormalizationError(
            [f"native_task_usd_import_asset_unreadable:{role}"]
        )
    paths = _active_physics_scene_paths(stage, usd_physics=UsdPhysics)
    receipt: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "semantic_role": role,
        "filename": source.name,
        "size_bytes": source.stat().st_size,
        "sha256": _sha256(source),
        "active_embedded_physics_scene_paths": paths,
        "active_embedded_physics_scene_count": len(paths),
        "environment_import_scene_free": not paths,
        "environment_owns_physics_scene": True,
        "audit_digest": "",
    }
    receipt["audit_digest"] = canonical_digest(
        receipt, digest_field="audit_digest"
    )
    return receipt


def normalize_environment_import_usd(
    source_path: str | Path,
    destination_path: str | Path,
    *,
    semantic_role: str,
) -> dict[str, Any]:
    """Derive one scene-free leaf USD while preserving the source exactly."""

    source = Path(source_path).expanduser().resolve()
    destination = Path(destination_path).expanduser().resolve()
    role = str(semantic_role or "")
    if (
        not role
        or not source.is_file()
        or source.is_symlink()
        or source == destination
        or destination.exists()
    ):
        raise NativeTaskUsdImportNormalizationError(
            [f"native_task_usd_import_normalization_input_invalid:{role or 'unknown'}"]
        )
    source_size = source.stat().st_size
    source_digest = _sha256(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)

    before = inspect_environment_import_usd(destination, semantic_role=role)
    removed_paths: list[str] = []
    deactivated_paths: list[str] = []
    if before["active_embedded_physics_scene_paths"]:
        Usd, _ = _usd_modules()
        try:
            stage = Usd.Stage.Open(str(destination), load=Usd.Stage.LoadAll)
        except Exception as exc:  # defensive; inspect above already opened these bytes
            raise NativeTaskUsdImportNormalizationError(
                [f"native_task_usd_import_asset_unreadable:{role}"]
            ) from exc
        if stage is None:  # defensive; inspect above already opened these bytes
            raise NativeTaskUsdImportNormalizationError(
                [f"native_task_usd_import_asset_unreadable:{role}"]
            )
        root_identifier = stage.GetRootLayer().identifier
        for path in before["active_embedded_physics_scene_paths"]:
            prim = stage.GetPrimAtPath(path)
            authored_in_root = any(
                spec.layer.identifier == root_identifier for spec in prim.GetPrimStack()
            )
            if authored_in_root and stage.RemovePrim(path):
                removed_paths.append(path)
            else:
                # A scene supplied by a referenced/sublayered dependency must not
                # cause us to mutate that dependency.  An inactive override in the
                # derived root layer excludes it from the composed environment.
                prim.SetActive(False)
                deactivated_paths.append(path)
        stage.GetRootLayer().Save()

    after = inspect_environment_import_usd(destination, semantic_role=role)
    if not after["environment_import_scene_free"]:
        destination.unlink(missing_ok=True)
        raise NativeTaskUsdImportNormalizationError(
            [f"native_task_usd_import_embedded_physics_scene_retained:{role}"]
        )
    if source.stat().st_size != source_size or _sha256(source) != source_digest:
        destination.unlink(missing_ok=True)
        raise NativeTaskUsdImportNormalizationError(
            [f"native_task_usd_import_source_mutated:{role}"]
        )

    staged_digest = _sha256(destination)
    receipt: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "semantic_role": role,
        "filename": destination.name,
        "source_size_bytes": source_size,
        "source_sha256": source_digest,
        "staged_size_bytes": destination.stat().st_size,
        "staged_sha256": staged_digest,
        "source_bytes_mutated": False,
        "staged_bytes_derived": staged_digest != source_digest,
        "source_and_staged_bytes_identical": staged_digest == source_digest,
        "active_embedded_physics_scene_paths_before": before[
            "active_embedded_physics_scene_paths"
        ],
        "removed_physics_scene_paths": removed_paths,
        "deactivated_physics_scene_paths": deactivated_paths,
        "active_embedded_physics_scene_paths_after": after[
            "active_embedded_physics_scene_paths"
        ],
        "environment_import_scene_free": True,
        "environment_owns_physics_scene": True,
        "normalization_digest": "",
    }
    receipt["normalization_digest"] = canonical_digest(
        receipt, digest_field="normalization_digest"
    )
    return receipt


__all__ = [
    "NativeTaskUsdImportNormalizationError",
    "SCHEMA_VERSION",
    "inspect_environment_import_usd",
    "normalize_environment_import_usd",
]
