"""Deterministic, self-contained OpenUSD packaging for reconstruction layers.

The packager composes an already-qualified appearance asset and collider asset
into one meter/Z-up stage, packages the dependency closure as USDZ, normalizes
ZIP metadata for byte-stable replay, and inspects the exact resulting package.
It does not qualify either input and cannot promote collision or task claims.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import struct
import tempfile
import time
from typing import Any, Mapping
import zipfile

from .reconstruction_geometry_contracts import (
    ReconstructionGeometryContractError,
    build_nurec_openusd_packaging_request,
    build_nurec_openusd_packaging_result,
)


PACKAGER_IMPLEMENTATION_VERSION = "blueprint_openusd_packager.v1"
MAX_SOURCE_ASSET_BYTES = 2_000_000_000
MAX_USDZ_MEMBER_BYTES = 2_000_000_000
MAX_USDZ_TOTAL_BYTES = 4_000_000_000
MAX_USDZ_MEMBER_COUNT = 20_000
_USD_SUFFIXES = {".usd", ".usda", ".usdc", ".usdz"}


class NuRecOpenUSDPackagingError(ValueError):
    def __init__(self, codes: list[str] | tuple[str, ...]) -> None:
        self.codes = tuple(sorted(set(str(code) for code in codes if str(code))))
        super().__init__("; ".join(self.codes))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _safe_source(root: Path, binding: Mapping[str, Any], name: str) -> Path:
    text = str(binding.get("relative_path") or "").replace("\\", "/")
    relative = PurePosixPath(text)
    if (
        not text
        or relative.is_absolute()
        or any(part in {"", ".", ".."} for part in relative.parts)
        or ":" in relative.parts[0]
    ):
        raise NuRecOpenUSDPackagingError([f"{name}_relative_path_unsafe"])
    candidate = root.joinpath(*relative.parts)
    if candidate.is_symlink():
        raise NuRecOpenUSDPackagingError([f"{name}_symlink_forbidden"])
    resolved = candidate.resolve()
    if resolved != root and root not in resolved.parents:
        raise NuRecOpenUSDPackagingError([f"{name}_path_escape"])
    if resolved.is_symlink() or not resolved.is_file():
        raise NuRecOpenUSDPackagingError([f"{name}_missing"])
    if resolved.suffix.lower() not in _USD_SUFFIXES:
        raise NuRecOpenUSDPackagingError([f"{name}_format_unsupported"])
    if resolved.stat().st_size > MAX_SOURCE_ASSET_BYTES:
        raise NuRecOpenUSDPackagingError([f"{name}_oversized"])
    if _sha256_file(resolved) != binding.get("digest"):
        raise NuRecOpenUSDPackagingError([f"{name}_digest_mismatch"])
    if resolved.suffix.lower() == ".usdz":
        _validate_source_usdz(resolved, name)
    return resolved


def _subtree_count(root_prim, predicate) -> int:
    from pxr import Usd  # type: ignore

    return sum(1 for prim in Usd.PrimRange(root_prim) if predicate(prim))


def _inspect_source_stage(path: Path, prim_path: str, *, appearance: bool) -> dict[str, Any]:
    from pxr import Usd, UsdGeom, UsdPhysics, UsdUtils  # type: ignore

    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise NuRecOpenUSDPackagingError(["openusd_source_stage_open_failed"])
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        raise NuRecOpenUSDPackagingError(["openusd_source_prim_missing"])
    meters = float(UsdGeom.GetStageMetersPerUnit(stage))
    up_axis = str(UsdGeom.GetStageUpAxis(stage))
    if meters != 1.0 or up_axis != "Z":
        raise NuRecOpenUSDPackagingError(["openusd_source_units_or_up_axis_invalid"])
    try:
        layers, assets, unresolved = UsdUtils.ComputeAllDependencies(str(path))
    except Exception as exc:  # noqa: BLE001
        raise NuRecOpenUSDPackagingError(["openusd_source_dependency_inspection_failed"]) from exc
    source_text = str(path.resolve())
    external_dependencies = []
    for layer in layers:
        layer_path = str(getattr(layer, "realPath", "") or getattr(layer, "identifier", ""))
        if layer_path and layer_path != source_text and not layer_path.startswith(source_text + "["):
            external_dependencies.append(layer_path)
    for asset in assets:
        asset_path = str(asset)
        if asset_path and asset_path != source_text and not asset_path.startswith(source_text + "["):
            external_dependencies.append(asset_path)
    if unresolved:
        raise NuRecOpenUSDPackagingError(["openusd_source_dependency_unresolved"])
    if external_dependencies:
        raise NuRecOpenUSDPackagingError(["openusd_source_external_dependency_unbound"])
    particlefields = _subtree_count(
        prim, lambda item: str(item.GetTypeName()) == "ParticleField3DGaussianSplat"
    )
    collisions = _subtree_count(prim, lambda item: item.HasAPI(UsdPhysics.CollisionAPI))
    if appearance and particlefields < 1:
        raise NuRecOpenUSDPackagingError(["appearance_particlefield_prim_missing"])
    if not appearance and collisions < 1:
        raise NuRecOpenUSDPackagingError(["collider_collision_api_missing"])
    return {
        "meters_per_unit": meters,
        "up_axis": up_axis,
        "particlefield_prim_count": particlefields,
        "collision_api_prim_count": collisions,
        "external_dependency_count": 0,
    }


def _validate_zip_member(name: str) -> None:
    path = PurePosixPath(name)
    if (
        not name
        or path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
        or "\\" in name
    ):
        raise NuRecOpenUSDPackagingError(["usdz_member_path_unsafe"])


def _zip_member_is_symlink(info: zipfile.ZipInfo) -> bool:
    mode = (info.external_attr >> 16) & 0xFFFF
    return info.create_system == 3 and stat.S_ISLNK(mode)


def _validate_source_usdz(path: Path, name: str) -> None:
    try:
        with zipfile.ZipFile(path, "r") as archive:
            infos = archive.infolist()
            if not infos or len(infos) > MAX_USDZ_MEMBER_COUNT:
                raise NuRecOpenUSDPackagingError([f"{name}_usdz_member_count_invalid"])
            total = 0
            for info in infos:
                _validate_zip_member(info.filename)
                if info.is_dir() or _zip_member_is_symlink(info):
                    raise NuRecOpenUSDPackagingError([f"{name}_usdz_nonregular_member"])
                if info.flag_bits & 0x1:
                    raise NuRecOpenUSDPackagingError([f"{name}_usdz_encrypted_member"])
                if info.compress_type != zipfile.ZIP_STORED:
                    raise NuRecOpenUSDPackagingError([f"{name}_usdz_compressed_member"])
                if info.file_size > MAX_USDZ_MEMBER_BYTES:
                    raise NuRecOpenUSDPackagingError([f"{name}_usdz_member_oversized"])
                total += info.file_size
            if total > MAX_USDZ_TOTAL_BYTES:
                raise NuRecOpenUSDPackagingError([f"{name}_usdz_package_oversized"])
            if archive.testzip() is not None:
                raise NuRecOpenUSDPackagingError([f"{name}_usdz_crc_invalid"])
    except zipfile.BadZipFile as exc:
        raise NuRecOpenUSDPackagingError([f"{name}_usdz_corrupt"]) from exc


def validate_safe_usdz_archive(path: Path, name: str = "source") -> None:
    """Validate a USDZ as a bounded, self-contained, uncompressed archive.

    This public boundary is shared by local importers that must inspect a USDZ
    before copying it into a trusted artifact root. It intentionally performs
    archive-safety checks only; scientific or Isaac qualification remains a
    separate deterministic gate.
    """

    _validate_source_usdz(path, name)


def _alignment_extra(offset: int, filename: str) -> bytes:
    base = offset + 30 + len(filename.encode("utf-8"))
    if base % 64 == 0:
        return b""
    padding = (-(base + 4)) % 64
    return struct.pack("<HH", 0x1986, padding) + (b"\0" * padding)


def _normalize_usdz(source_path: Path, output_path: Path) -> dict[str, Any]:
    """Rewrite a USD-created package with stable timestamps/order and 64-byte alignment."""
    with zipfile.ZipFile(source_path, "r") as source:
        infos = source.infolist()
        if not infos or len(infos) > MAX_USDZ_MEMBER_COUNT:
            raise NuRecOpenUSDPackagingError(["usdz_package_empty"])
        names = [item.filename for item in infos]
        if len(names) != len(set(names)):
            raise NuRecOpenUSDPackagingError(["usdz_duplicate_member"])
        for name in names:
            _validate_zip_member(name)
        if any(item.is_dir() or _zip_member_is_symlink(item) for item in infos):
            raise NuRecOpenUSDPackagingError(["usdz_nonregular_member"])
        if any(item.file_size > MAX_USDZ_MEMBER_BYTES for item in infos):
            raise NuRecOpenUSDPackagingError(["usdz_member_oversized"])
        if sum(item.file_size for item in infos) > MAX_USDZ_TOTAL_BYTES:
            raise NuRecOpenUSDPackagingError(["usdz_package_oversized"])
        root_name = infos[0].filename
        by_name = {item.filename: item for item in infos}
        ordered_names = [root_name, *sorted(name for name in names if name != root_name)]
        with output_path.open("wb") as raw_output:
            with zipfile.ZipFile(raw_output, "w", compression=zipfile.ZIP_STORED) as target:
                for name in ordered_names:
                    source_info = by_name[name]
                    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                    info.compress_type = zipfile.ZIP_STORED
                    info.create_system = 3
                    info.external_attr = 0o100644 << 16
                    info.file_size = source_info.file_size
                    info.extra = _alignment_extra(raw_output.tell(), name)
                    with source.open(source_info, "r") as input_stream:
                        with target.open(info, "w", force_zip64=False) as output_stream:
                            shutil.copyfileobj(input_stream, output_stream, length=1024 * 1024)
    with zipfile.ZipFile(output_path, "r") as package:
        for info in package.infolist():
            data_offset = info.header_offset + 30 + len(info.filename.encode("utf-8")) + len(info.extra)
            if info.compress_type != zipfile.ZIP_STORED or data_offset % 64 != 0:
                raise NuRecOpenUSDPackagingError(["usdz_deterministic_layout_invalid"])
        return {
            "root_member": package.infolist()[0].filename,
            "package_member_count": len(package.infolist()),
        }


def _inspect_package(path: Path, targets: Mapping[str, str]) -> dict[str, Any]:
    from pxr import Usd, UsdGeom, UsdPhysics, UsdUtils  # type: ignore

    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise NuRecOpenUSDPackagingError(["packaged_openusd_stage_open_failed"])
    appearance = stage.GetPrimAtPath(targets["appearance"])
    collision = stage.GetPrimAtPath(targets["collision"])
    if not appearance or not appearance.IsValid() or not collision or not collision.IsValid():
        raise NuRecOpenUSDPackagingError(["packaged_required_prim_missing"])
    particlefields = _subtree_count(
        appearance, lambda item: str(item.GetTypeName()) == "ParticleField3DGaussianSplat"
    )
    collisions = _subtree_count(collision, lambda item: item.HasAPI(UsdPhysics.CollisionAPI))
    try:
        _layers, _assets, unresolved = UsdUtils.ComputeAllDependencies(str(path))
    except Exception as exc:  # noqa: BLE001
        raise NuRecOpenUSDPackagingError(["packaged_dependency_inspection_failed"]) from exc
    unresolved = [str(value) for value in unresolved]
    if unresolved:
        raise NuRecOpenUSDPackagingError(["packaged_dependency_unresolved"])
    meters = float(UsdGeom.GetStageMetersPerUnit(stage))
    up_axis = str(UsdGeom.GetStageUpAxis(stage))
    if meters != 1.0 or up_axis != "Z":
        raise NuRecOpenUSDPackagingError(["packaged_units_or_up_axis_invalid"])
    if particlefields < 1 or collisions < 1:
        raise NuRecOpenUSDPackagingError(["packaged_appearance_or_collision_missing"])
    return {
        "stage_meters_per_unit": meters,
        "up_axis": up_axis,
        "appearance_prim_present": True,
        "collision_prim_present": True,
        "particlefield_prim_count": particlefields,
        "collision_api_prim_count": collisions,
        "collision_api_configured": True,
        "missing_asset_count": 0,
    }


def _replay_if_valid(final_dir: Path, output_name: str) -> dict[str, Any] | None:
    result_path = final_dir / "nurec_openusd_packaging_result.v1.json"
    package_path = final_dir / output_name
    if not result_path.is_file() or not package_path.is_file():
        return None
    try:
        result = build_nurec_openusd_packaging_result(
            json.loads(result_path.read_text(encoding="utf-8"))
        )
    except (OSError, json.JSONDecodeError, ReconstructionGeometryContractError):
        return None
    if result.get("package_digest") != _sha256_file(package_path):
        return None
    return result


def package_nurec_openusd(
    *,
    source_artifact: Mapping[str, Any],
    artifact_root: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    """Compose and inspect the exact self-contained package requested by a frozen contract."""
    started = time.monotonic()
    try:
        request = build_nurec_openusd_packaging_request(source_artifact)
    except ReconstructionGeometryContractError as exc:
        raise NuRecOpenUSDPackagingError([f"packaging_request_invalid:{code}" for code in exc.codes]) from exc
    root = Path(artifact_root).resolve()
    if Path(artifact_root).is_symlink() or not root.is_dir():
        raise NuRecOpenUSDPackagingError(["packaging_artifact_root_invalid"])
    destination_root = Path(output_root)
    if destination_root.is_symlink():
        raise NuRecOpenUSDPackagingError(["packaging_output_root_symlink_forbidden"])
    destination_root.mkdir(parents=True, exist_ok=True)
    destination_root = destination_root.resolve()
    content_id = request["packaging_request_digest"][7:]
    final_dir = destination_root / content_id
    replay = _replay_if_valid(final_dir, request["output_name"])
    if replay is not None:
        return replay
    if final_dir.exists() or final_dir.is_symlink():
        raise NuRecOpenUSDPackagingError(["packaging_existing_output_incomplete_or_tampered"])

    appearance = _safe_source(root, request["appearance_asset"], "appearance_asset")
    collider = _safe_source(root, request["collider_asset"], "collider_asset")
    _inspect_source_stage(
        appearance,
        request["appearance_asset"]["source_prim_path"],
        appearance=True,
    )
    _inspect_source_stage(
        collider,
        request["collider_asset"]["source_prim_path"],
        appearance=False,
    )

    temporary = Path(tempfile.mkdtemp(prefix=".nurec-package-", dir=destination_root))
    try:
        from pxr import Sdf, Usd, UsdGeom, UsdUtils  # type: ignore

        root_layer = temporary / "blueprint_reconstruction.usda"
        stage = Usd.Stage.CreateNew(str(root_layer))
        if stage is None:
            raise NuRecOpenUSDPackagingError(["packaging_stage_create_failed"])
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        world = UsdGeom.Xform.Define(stage, "/World").GetPrim()
        stage.SetDefaultPrim(world)
        UsdGeom.Xform.Define(stage, "/World/BlueprintReconstruction")
        targets = request["target_prim_paths"]
        appearance_target = stage.OverridePrim(targets["appearance"])
        appearance_target.GetReferences().AddReference(
            str(appearance), Sdf.Path(request["appearance_asset"]["source_prim_path"])
        )
        collision_target = stage.OverridePrim(targets["collision"])
        collision_target.GetReferences().AddReference(
            str(collider), Sdf.Path(request["collider_asset"]["source_prim_path"])
        )
        root_prim = stage.GetPrimAtPath("/World/BlueprintReconstruction")
        root_prim.SetCustomDataByKey(
            "blueprint:sourceCaptureDigest", request["source_capture_digest"]
        )
        root_prim.SetCustomDataByKey(
            "blueprint:appearanceAssetDigest", request["appearance_asset"]["digest"]
        )
        root_prim.SetCustomDataByKey(
            "blueprint:colliderAssetDigest", request["collider_asset"]["digest"]
        )
        root_prim.SetCustomDataByKey(
            "blueprint:colliderQualificationDigest", request["collider_qualification_digest"]
        )
        stage.GetRootLayer().Save()

        composed = Usd.Stage.Open(str(root_layer))
        if composed is None:
            raise NuRecOpenUSDPackagingError(["packaging_composed_stage_open_failed"])
        raw_package = temporary / "raw.usdz"
        packaged = UsdUtils.CreateNewUsdzPackage(Sdf.AssetPath(str(root_layer)), str(raw_package))
        if packaged is not True or not raw_package.is_file():
            raise NuRecOpenUSDPackagingError(["openusd_usdz_packaging_failed"])
        final_package = temporary / request["output_name"]
        zip_layout = _normalize_usdz(raw_package, final_package)
        inspection = _inspect_package(final_package, targets)
        package_digest = _sha256_file(final_package)
        value = dict(request)
        value.pop("packaging_request_digest", None)
        value.pop("schema_version", None)
        value.update(
            {
                "producing_method": "nurec_openusd_packaging",
                "implementation_version": PACKAGER_IMPLEMENTATION_VERSION,
                "output_digests": [
                    {"artifact_id": "nurec_openusd_package", "digest": package_digest}
                ],
                "provider_runtime_identity": {
                    "provider": "local",
                    "runtime": "openusd",
                    "openusd_version": list(Usd.GetVersion()),
                },
                "cost_usd": 0.0,
                "duration_seconds": round(time.monotonic() - started, 6),
                "parent_artifact_or_event": {"digest": request["packaging_request_digest"]},
                "packaging_request_digest": request["packaging_request_digest"],
                "appearance_asset_digest": request["appearance_asset"]["digest"],
                "package_digest": package_digest,
                "package_artifact_reference": f"{content_id}/{request['output_name']}",
                "package_format": "usdz",
                "self_contained": True,
                "deterministic_archive": True,
                "shared_visual_physics_frame": True,
                **inspection,
                **zip_layout,
                "proof_effect": "packaging_compatibility_candidate_only",
                "claim_ceiling": "openusd_package",
            }
        )
        value.pop("appearance_asset", None)
        value.pop("collider_asset", None)
        value.pop("target_prim_paths", None)
        value.pop("output_format", None)
        value.pop("output_name", None)
        result = build_nurec_openusd_packaging_result(value)
        result_path = temporary / "nurec_openusd_packaging_result.v1.json"
        result_path.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, final_dir)
        return result
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


__all__ = [
    "NuRecOpenUSDPackagingError",
    "PACKAGER_IMPLEMENTATION_VERSION",
    "package_nurec_openusd",
    "validate_safe_usdz_archive",
]
