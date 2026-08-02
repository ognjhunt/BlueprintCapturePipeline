"""Compile a static Isaac collider candidate from an authorized external scene mesh.

This is the reduced-authority lane used when Blueprint has a GLB/mesh export but
not the original capture video or metric anchor. It creates useful simulator
geometry without promoting the external mesh to captured collision truth.
Independent scale and collider qualification remain separate gates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

from .decision_evidence_contracts import canonical_digest, canonical_json


REQUEST_SCHEMA = "external_scene_collision_compilation_request.v1"
RESULT_SCHEMA = "external_scene_collision_candidate.v1"
SUPPORTED_SCALE_STATUSES = {
    "provider_declared_not_independently_validated",
    "sensor_metric_unvalidated",
    "unverified",
    "validated",
}


class ExternalSceneCollisionCandidateError(ValueError):
    def __init__(self, codes: Sequence[str]) -> None:
        self.codes = tuple(sorted(set(str(code) for code in codes if str(code))))
        super().__init__(";".join(self.codes))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def build_external_scene_collision_request(value: Mapping[str, Any]) -> dict[str, Any]:
    try:
        request = json.loads(json.dumps(dict(value), allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ExternalSceneCollisionCandidateError(["collision_request_not_json"]) from exc
    supplied = request.pop("request_digest", None)
    errors: list[str] = []
    if request.get("schema_version") != REQUEST_SCHEMA:
        errors.append("collision_request_schema_invalid")
    digest = str(request.get("source_asset_digest") or "")
    if len(digest) != 71 or not digest.startswith("sha256:"):
        errors.append("collision_request_source_digest_invalid")
    if request.get("source_format") != "glb":
        errors.append("collision_request_source_format_invalid")
    frame = request.get("source_coordinate_frame")
    if (
        not isinstance(frame, Mapping)
        or frame.get("up_axis") not in {"Y", "Z"}
        or frame.get("handedness") != "right"
    ):
        errors.append("collision_request_coordinate_frame_invalid")
    if request.get("metric_scale_status") not in SUPPORTED_SCALE_STATUSES:
        errors.append("collision_request_metric_scale_status_invalid")
    if request.get("source_video_available") not in {True, False}:
        errors.append("collision_request_source_video_status_invalid")
    if request.get("generated_fill_allowed") is not False:
        errors.append("collision_request_generated_fill_forbidden")
    if request.get("collision_validated") is not False:
        errors.append("collision_request_prevalidated_forbidden")
    expected = canonical_digest(request, digest_field="request_digest")
    if supplied is not None and supplied != expected:
        errors.append("collision_request_digest_mismatch")
    if errors:
        raise ExternalSceneCollisionCandidateError(errors)
    request["request_digest"] = expected
    return request


def _flatten_glb(path: Path):
    try:
        import trimesh  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ExternalSceneCollisionCandidateError(["trimesh_unavailable"]) from exc
    try:
        loaded = trimesh.load(path, force="scene", process=False)
        if isinstance(loaded, trimesh.Scene):
            to_geometry = getattr(loaded, "to_geometry", None)
            mesh = to_geometry() if callable(to_geometry) else loaded.dump(concatenate=True)
            component_count = len(loaded.geometry)
        else:
            mesh = loaded
            component_count = 1
    except Exception as exc:  # noqa: BLE001
        raise ExternalSceneCollisionCandidateError(["external_glb_load_failed"]) from exc
    vertices = np.asarray(getattr(mesh, "vertices", []), dtype=np.float64)
    faces = np.asarray(getattr(mesh, "faces", []), dtype=np.int64)
    if (
        vertices.ndim != 2
        or vertices.shape[1:] != (3,)
        or faces.ndim != 2
        or faces.shape[1:] != (3,)
        or len(vertices) < 3
        or len(faces) < 1
        or not np.isfinite(vertices).all()
        or np.any(faces < 0)
        or np.any(faces >= len(vertices))
    ):
        raise ExternalSceneCollisionCandidateError(["external_glb_geometry_invalid"])
    return vertices, faces, component_count


def _to_z_up(vertices: np.ndarray, *, source_up_axis: str) -> tuple[np.ndarray, str]:
    if source_up_axis == "Z":
        return np.ascontiguousarray(vertices), "identity_z_up"
    # glTF is right-handed Y-up. Rotate +90 degrees around X so +Y becomes +Z
    # while preserving handedness: (x, y, z) -> (x, -z, y).
    transformed = np.column_stack((vertices[:, 0], -vertices[:, 2], vertices[:, 1]))
    return np.ascontiguousarray(transformed), "right_handed_y_up_to_z_up_rx_plus_90"


def _write_usd(
    path: Path,
    *,
    vertices: np.ndarray,
    faces: np.ndarray,
    request: Mapping[str, Any],
) -> None:
    try:
        from pxr import Gf, Usd, UsdGeom, UsdPhysics  # type: ignore
    except ImportError as exc:
        raise ExternalSceneCollisionCandidateError(["openusd_unavailable"]) from exc
    stage = Usd.Stage.CreateNew(str(path))
    if stage is None:
        raise ExternalSceneCollisionCandidateError(["collision_usd_stage_create_failed"])
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    world = UsdGeom.Xform.Define(stage, "/World")
    collision = UsdGeom.Xform.Define(stage, "/World/BlueprintReconstruction/Collision")
    mesh = UsdGeom.Mesh.Define(stage, "/World/BlueprintReconstruction/Collision/ExternalSceneMesh")
    mesh.CreatePointsAttr([Gf.Vec3f(*[float(item) for item in row]) for row in vertices])
    mesh.CreateFaceVertexCountsAttr([3] * len(faces))
    mesh.CreateFaceVertexIndicesAttr([int(index) for face in faces for index in face])
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr().Set("none")
    collision.GetPrim().SetCustomDataByKey(
        "blueprint:sourceAssetDigest", request["source_asset_digest"]
    )
    collision.GetPrim().SetCustomDataByKey(
        "blueprint:metricScaleStatus", request["metric_scale_status"]
    )
    collision.GetPrim().SetCustomDataByKey("blueprint:collisionValidated", False)
    collision.GetPrim().SetCustomDataByKey("blueprint:generatedFillUsed", False)
    stage.SetDefaultPrim(world.GetPrim())
    stage.GetRootLayer().Save()


def compile_external_scene_collision_candidate(
    *,
    source_path: str | Path,
    request: Mapping[str, Any],
    output_path: str | Path,
) -> dict[str, Any]:
    """Author one static triangle-mesh collider candidate from a bound GLB."""

    admitted = build_external_scene_collision_request(request)
    source = Path(source_path)
    if source.is_symlink():
        raise ExternalSceneCollisionCandidateError(["external_glb_symlink_forbidden"])
    try:
        source = source.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ExternalSceneCollisionCandidateError(["external_glb_missing"]) from exc
    if source.suffix.lower() != ".glb" or not source.is_file():
        raise ExternalSceneCollisionCandidateError(["external_glb_path_invalid"])
    if _sha256(source) != admitted["source_asset_digest"]:
        raise ExternalSceneCollisionCandidateError(["external_glb_digest_mismatch"])
    destination = Path(output_path)
    if destination.is_symlink() or destination.suffix.lower() not in {".usd", ".usda", ".usdc"}:
        raise ExternalSceneCollisionCandidateError(["collision_output_path_invalid"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    vertices, faces, component_count = _flatten_glb(source)
    vertices, transform = _to_z_up(
        vertices,
        source_up_axis=str(admitted["source_coordinate_frame"]["up_axis"]),
    )
    _write_usd(destination, vertices=vertices, faces=faces, request=admitted)
    low = vertices.min(axis=0)
    high = vertices.max(axis=0)
    extents = high - low
    obvious_scale_mismatch = bool(max(extents) < 0.25 or max(extents) > 1000.0)
    result = {
        "schema_version": RESULT_SCHEMA,
        "status": "candidate_compiled",
        "request_digest": admitted["request_digest"],
        "source_asset_digest": admitted["source_asset_digest"],
        "source_video_available": admitted["source_video_available"],
        "source_video_required_for_candidate_compilation": False,
        "source_coordinate_frame": dict(admitted["source_coordinate_frame"]),
        "output_coordinate_frame": {
            "up_axis": "Z",
            "handedness": "right",
            "stage_meters_per_unit": 1.0,
            "transform": transform,
        },
        "metric_scale_status": admitted["metric_scale_status"],
        "independent_known_distance_anchor": (admitted["metric_scale_status"] == "validated"),
        "source_component_count": int(component_count),
        "vertex_count": int(len(vertices)),
        "triangle_count": int(len(faces)),
        "bounds_stage_units": {
            "min": [round(float(item), 9) for item in low],
            "max": [round(float(item), 9) for item in high],
            "extents": [round(float(item), 9) for item in extents],
        },
        "obvious_scale_mismatch_detected": obvious_scale_mismatch,
        "collider_prim_path": "/World/BlueprintReconstruction/Collision/ExternalSceneMesh",
        "collider_asset_path": str(destination.resolve()),
        "collider_asset_digest": _sha256(destination),
        "collision_api_configured": True,
        "static_triangle_mesh": True,
        "generated_fill_used": False,
        "collision_validated": False,
        "blockers": (
            []
            if admitted["metric_scale_status"] == "validated" and not obvious_scale_mismatch
            else ["independent_metric_scale_pending"]
        ),
        "qualification_gaps": [
            *(
                []
                if admitted["metric_scale_status"] == "validated"
                else ["independent_metric_scale_missing"]
            ),
            "collider_contact_qualification_pending",
        ],
        "proof_effect": "external_scene_collision_candidate_only",
        "claim_ceiling": "isaac_collision_candidate",
        "unsupported_claims": [
            *([] if admitted["metric_scale_status"] == "validated" else ["metric_scale"]),
            "robot_footprint_clearance",
            "contact_dynamics",
            "task_success",
            "physical_success",
            "deployment_readiness",
        ],
    }
    if not all(math.isfinite(float(item)) for item in [*low, *high, *extents]):
        raise ExternalSceneCollisionCandidateError(["collision_result_bounds_nonfinite"])
    result["collision_candidate_digest"] = canonical_digest(
        result, digest_field="collision_candidate_digest"
    )
    return result


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(canonical_json(dict(value)) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_external_scene_collision_candidate(
    *,
    source_path: str | Path,
    request: Mapping[str, Any],
    output_path: str | Path,
    result_path: str | Path,
    request_path: str | Path | None = None,
) -> dict[str, Any]:
    """Compile and immutably record the admitted request and result ledger."""

    admitted = build_external_scene_collision_request(request)
    result = compile_external_scene_collision_candidate(
        source_path=source_path,
        request=admitted,
        output_path=output_path,
    )
    if request_path is not None:
        _write_json_atomic(Path(request_path), admitted)
    _write_json_atomic(Path(result_path), result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compile a digest-bound external GLB into an Isaac collider candidate."
    )
    parser.add_argument("--source", required=True)
    parser.add_argument("--request", required=True)
    parser.add_argument("--output-usd", required=True)
    parser.add_argument("--result-out", required=True)
    parser.add_argument("--admitted-request-out")
    args = parser.parse_args(argv)
    request = json.loads(Path(args.request).read_text(encoding="utf-8"))
    if not isinstance(request, Mapping):
        raise ExternalSceneCollisionCandidateError(["collision_request_not_json_object"])
    result = write_external_scene_collision_candidate(
        source_path=args.source,
        request=request,
        output_path=args.output_usd,
        result_path=args.result_out,
        request_path=args.admitted_request_out,
    )
    print(canonical_json(result))
    return 0


__all__ = [
    "REQUEST_SCHEMA",
    "RESULT_SCHEMA",
    "ExternalSceneCollisionCandidateError",
    "build_external_scene_collision_request",
    "compile_external_scene_collision_candidate",
    "write_external_scene_collision_candidate",
]


if __name__ == "__main__":
    raise SystemExit(main())
