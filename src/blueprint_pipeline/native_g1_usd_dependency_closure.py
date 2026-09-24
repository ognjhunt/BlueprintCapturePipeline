"""Resolve the local, relative OpenUSD closure of a G1 robot asset.

The packet may be moved to another host. A digest of its root USD alone does
not bind referenced layers or textures, so every external byte must be staged
and checked with it. Absolute, remote, escaping, and unresolved references
are refused before the packet is published.
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import NamedTuple

G1_KIT_RUNTIME_ASSETS = frozenset({"OmniPBR.mdl"})


class RobotUsdDependencyClosure(NamedTuple):
    sources: list[tuple[Path, PurePosixPath]]
    runtime_assets: tuple[str, ...]


def robot_usd_dependency_sources(
    source: Path, *, allowed_runtime_assets: frozenset[str] = frozenset()
) -> RobotUsdDependencyClosure:
    try:
        from pxr import UsdUtils

        layers, assets, unresolved = UsdUtils.ComputeAllDependencies(str(source))
    except Exception as exc:  # noqa: BLE001 - unavailable USD inspection fails closed
        raise ValueError("native_g1_usd_dependency_inspection_failed") from exc
    unresolved_names = {str(value) for value in unresolved}
    if unresolved_names - allowed_runtime_assets:
        raise ValueError("native_g1_usd_dependency_unresolved")

    root = source.parent.resolve()
    dependencies: dict[str, tuple[Path, PurePosixPath]] = {}

    def admit(raw: str, *, owner: Path) -> None:
        if not raw:
            return
        path = PurePosixPath(raw)
        if path.is_absolute() or ".." in path.parts or ":" in raw or "\\" in raw:
            raise ValueError("native_g1_usd_dependency_nonportable")
        candidate = owner.parent.joinpath(*path.parts)
        resolved = candidate.resolve()
        if (
            candidate.is_symlink()
            or root not in resolved.parents
            or not resolved.is_file()
            or any(parent.is_symlink() for parent in (candidate, *candidate.parents) if parent != root and root in parent.parents)
        ):
            raise ValueError("native_g1_usd_dependency_outside_source")
        relative = PurePosixPath(resolved.relative_to(root).as_posix())
        if resolved != source.resolve():
            dependencies[relative.as_posix()] = (resolved, relative)

    for layer in layers:
        identifier = str(getattr(layer, "realPath", "") or getattr(layer, "identifier", ""))
        # Entries inside one packaged USDZ are carried by its root file.
        if identifier.startswith(str(source.resolve()) + "["):
            continue
        owner = Path(identifier)
        if owner != source.resolve():
            if owner.is_symlink() or root not in owner.parents or not owner.is_file():
                raise ValueError("native_g1_usd_dependency_outside_source")
            admit(owner.relative_to(root).as_posix(), owner=source)
        for raw in (*layer.GetExternalReferences(), *layer.GetExternalAssetDependencies()):
            admit(str(raw), owner=owner)
    for asset in assets:
        raw = str(asset)
        if raw.startswith(str(source.resolve()) + "["):
            continue
        path = Path(raw)
        if path.is_absolute():
            if root not in path.parents or not path.is_file() or path.is_symlink():
                raise ValueError("native_g1_usd_dependency_outside_source")
            relative = PurePosixPath(path.relative_to(root).as_posix())
            dependencies[relative.as_posix()] = (path, relative)
        else:
            admit(raw, owner=source)
    return RobotUsdDependencyClosure(
        [dependencies[key] for key in sorted(dependencies)],
        tuple(sorted(unresolved_names)),
    )
