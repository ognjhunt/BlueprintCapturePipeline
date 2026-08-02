"""Package registered external appearance and collision layers for Isaac.

The package keeps the Gaussian appearance and static collision candidate as
separate OpenUSD layers under stable prim paths. Cross-export frame registration
is authored as an Xform on the appearance layer. Packaging is useful support;
it does not validate metric scale, contact, reach, or task success.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

from .decision_evidence_contracts import canonical_digest
from .gaussian_splat_decode import SplatData, read_standard_3dgs_ply
from .particlefield_usd import write_particlefield_usd


REQUEST_SCHEMA = "external_scene_isaac_package_request.v1"
RESULT_SCHEMA = "external_scene_isaac_package_result.v1"


class ExternalSceneIsaacPackageError(ValueError):
    def __init__(self, codes: Sequence[str]) -> None:
        self.codes = tuple(sorted(set(str(code) for code in codes if str(code))))
        super().__init__(";".join(self.codes))


def _digest(value: Any) -> bool:
    text = str(value or "")
    return (
        len(text) == 71
        and text.startswith("sha256:")
        and all(character in "0123456789abcdef" for character in text[7:])
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def build_external_scene_isaac_package_request(value: Mapping[str, Any]) -> dict[str, Any]:
    try:
        request = json.loads(json.dumps(dict(value), allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ExternalSceneIsaacPackageError(["external_package_request_not_json"]) from exc
    supplied = request.pop("request_digest", None)
    errors: list[str] = []
    if request.get("schema_version") != REQUEST_SCHEMA:
        errors.append("external_package_request_schema_invalid")
    for key in (
        "appearance_scene_digest",
        "analysis_splat_digest",
        "collision_candidate_digest",
        "collision_asset_digest",
        "scene_frame_binding_digest",
    ):
        if not _digest(request.get(key)):
            errors.append(f"external_package_{key}_invalid")
    matrix = request.get("source_to_collision_stage_matrix")
    if (
        not isinstance(matrix, list)
        or len(matrix) != 16
        or any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in matrix)
    ):
        errors.append("external_package_frame_matrix_invalid")
    if request.get("metric_scale_status") not in {
        "validated",
        "provider_declared_not_independently_validated",
        "unverified",
    }:
        errors.append("external_package_metric_status_invalid")
    if request.get("collision_validated") not in {True, False}:
        errors.append("external_package_collision_status_invalid")
    if request.get("source_video_available") not in {True, False}:
        errors.append("external_package_source_video_status_invalid")
    if request.get("generated_fill_allowed") is not False:
        errors.append("external_package_generated_fill_forbidden")
    maximum_nonfinite = request.get("maximum_nonfinite_splat_fraction")
    if (
        isinstance(maximum_nonfinite, bool)
        or not isinstance(maximum_nonfinite, (int, float))
        or not 0.0 <= float(maximum_nonfinite) <= 0.01
    ):
        errors.append("external_package_nonfinite_threshold_invalid")
    expected = canonical_digest(request, digest_field="request_digest")
    if supplied is not None and supplied != expected:
        errors.append("external_package_request_digest_mismatch")
    if errors:
        raise ExternalSceneIsaacPackageError(errors)
    request["request_digest"] = expected
    return request


def compile_external_scene_isaac_package(
    *,
    analysis_splat_path: str | Path,
    collision_usd_path: str | Path,
    output_path: str | Path,
    request: Mapping[str, Any],
) -> dict[str, Any]:
    admitted = build_external_scene_isaac_package_request(request)
    splat = Path(analysis_splat_path).resolve(strict=True)
    collision = Path(collision_usd_path).resolve(strict=True)
    if splat.suffix.lower() != ".ply" or _sha256(splat) != admitted["analysis_splat_digest"]:
        raise ExternalSceneIsaacPackageError(["external_package_splat_binding_invalid"])
    if (
        collision.suffix.lower() not in {".usd", ".usda", ".usdc"}
        or _sha256(collision) != admitted["collision_asset_digest"]
    ):
        raise ExternalSceneIsaacPackageError(["external_package_collision_binding_invalid"])
    destination = Path(output_path)
    if destination.suffix.lower() != ".usdz" or destination.is_symlink():
        raise ExternalSceneIsaacPackageError(["external_package_output_invalid"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdUtils
    except ImportError as exc:
        raise ExternalSceneIsaacPackageError(["external_package_openusd_unavailable"]) from exc
    temporary = Path(tempfile.mkdtemp(prefix=".external-isaac-package-", dir=destination.parent))
    try:
        source_splat = read_standard_3dgs_ply(splat)
        arrays = [
            source_splat.xyz,
            source_splat.opacity,
            source_splat.f_dc,
            source_splat.scales,
            source_splat.quats,
        ]
        if source_splat.sh_rest is not None:
            arrays.append(source_splat.sh_rest)
        finite = np.ones(source_splat.count, dtype=bool)
        for array in arrays:
            value = np.asarray(array)
            finite &= np.isfinite(value).all(axis=1) if value.ndim > 1 else np.isfinite(value)
        dropped_nonfinite = int((~finite).sum())
        dropped_fraction = dropped_nonfinite / max(1, source_splat.count)
        if dropped_fraction > float(admitted["maximum_nonfinite_splat_fraction"]):
            raise ExternalSceneIsaacPackageError(
                ["external_package_nonfinite_splat_fraction_exceeded"]
            )
        sanitized_splat = SplatData(
            count=int(finite.sum()),
            xyz=np.asarray(source_splat.xyz)[finite],
            opacity=np.asarray(source_splat.opacity)[finite],
            f_dc=np.asarray(source_splat.f_dc)[finite],
            scales=np.asarray(source_splat.scales)[finite],
            quats=np.asarray(source_splat.quats)[finite],
            properties=source_splat.properties,
            sh_rest=(
                np.asarray(source_splat.sh_rest)[finite]
                if source_splat.sh_rest is not None
                else None
            ),
        )
        appearance = temporary / "appearance.usdc"
        authored = write_particlefield_usd(
            sanitized_splat,
            appearance,
            prim_path="/World/BlueprintReconstruction/Appearance/Gaussians",
            up_axis="Z",
        )
        if authored.get("status") != "completed":
            raise ExternalSceneIsaacPackageError(
                list(authored.get("blockers") or ["external_package_appearance_authoring_failed"])
            )
        appearance_stage = Usd.Stage.Open(str(appearance))
        appearance_xform = UsdGeom.Xform.Define(
            appearance_stage, "/World/BlueprintReconstruction/Appearance"
        )
        # Registration artifacts use the common column-vector convention.
        # OpenUSD/Gf composes row vectors, so transpose exactly once at authoring.
        matrix = Gf.Matrix4d(
            *[float(item) for item in admitted["source_to_collision_stage_matrix"]]
        ).GetTranspose()
        appearance_xform.ClearXformOpOrder()
        appearance_xform.AddTransformOp().Set(matrix)
        appearance_stage.GetRootLayer().Save()
        packaged_collision = temporary / "collision.usda"
        shutil.copy2(collision, packaged_collision)
        collision_stage = Usd.Stage.Open(str(packaged_collision))
        collider = collision_stage.GetPrimAtPath(
            "/World/BlueprintReconstruction/Collision/ExternalSceneMesh"
        )
        if not collider.IsValid() or not collider.HasAPI(UsdPhysics.CollisionAPI):
            raise ExternalSceneIsaacPackageError(["external_package_collision_api_missing"])
        root = Usd.Stage.CreateNew(str(temporary / "default.usda"))
        UsdGeom.SetStageMetersPerUnit(root, 1.0)
        UsdGeom.SetStageUpAxis(root, UsdGeom.Tokens.z)
        world = UsdGeom.Xform.Define(root, "/World")
        root.SetDefaultPrim(world.GetPrim())
        root.GetRootLayer().subLayerPaths = ["appearance.usdc", "collision.usda"]
        root.GetRootLayer().Save()
        package_temp = temporary / "scene.usdz"
        if not UsdUtils.CreateNewUsdzPackage(
            Sdf.AssetPath(str(temporary / "default.usda")), str(package_temp)
        ):
            raise ExternalSceneIsaacPackageError(["external_package_usdz_creation_failed"])
        os.replace(package_temp, destination)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)
    result = {
        "schema_version": RESULT_SCHEMA,
        "status": "candidate_packaged",
        "request_digest": admitted["request_digest"],
        "appearance_scene_digest": admitted["appearance_scene_digest"],
        "analysis_splat_digest": admitted["analysis_splat_digest"],
        "collision_candidate_digest": admitted["collision_candidate_digest"],
        "scene_frame_binding_digest": admitted["scene_frame_binding_digest"],
        "package_digest": _sha256(destination),
        "package_artifact_path": str(destination.resolve()),
        "package_bytes": destination.stat().st_size,
        "source_splat_count": source_splat.count,
        "packaged_splat_count": sanitized_splat.count,
        "dropped_nonfinite_splat_count": dropped_nonfinite,
        "dropped_nonfinite_splat_fraction": round(dropped_fraction, 12),
        "maximum_nonfinite_splat_fraction": admitted["maximum_nonfinite_splat_fraction"],
        "appearance_prim_path": "/World/BlueprintReconstruction/Appearance",
        "collision_prim_path": "/World/BlueprintReconstruction/Collision",
        "stage_meters_per_unit": 1.0,
        "stage_up_axis": "Z",
        "metric_scale_status": admitted["metric_scale_status"],
        "independent_metric_scale_proven": admitted["metric_scale_status"] == "validated",
        "collision_validated": admitted["collision_validated"],
        "source_video_available": admitted["source_video_available"],
        "source_video_required_for_candidate_packaging": False,
        "generated_fill_used": False,
        "proof_effect": "external_scene_isaac_package_candidate",
        "claim_ceiling": "isaac_input_candidate",
        "unsupported_claims": [
            "contact_dynamics",
            "metric_reach",
            "simulated_task_success",
            "physical_task_success",
            "deployment_readiness",
        ],
    }
    result["package_result_digest"] = canonical_digest(result, digest_field="package_result_digest")
    return result


__all__ = [
    "REQUEST_SCHEMA",
    "RESULT_SCHEMA",
    "ExternalSceneIsaacPackageError",
    "build_external_scene_isaac_package_request",
    "compile_external_scene_isaac_package",
]
