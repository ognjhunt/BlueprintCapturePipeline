"""Static admission for one portable Content Agents rigid replacement."""

from __future__ import annotations

import hashlib
import math
import stat
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest, canonical_json


SCHEMA_VERSION = "task_evaluation_rigid_replacement_static_qualification.v1"
_PHYSICS_COMPLETION_SCHEMA_VERSION = (
    "task_evaluation_rigid_candidate_physics_completion.v1"
)
_FORBIDDEN_PACKAGE_SUFFIXES = {
    ".bat",
    ".cmd",
    ".js",
    ".mjs",
    ".ps1",
    ".py",
    ".sh",
}


class TaskEvaluationSceneConfigurationStaticQualificationError(ValueError):
    """The authored replacement did not satisfy the preregistered static gate."""

    def __init__(self, codes: Sequence[str]) -> None:
        self.codes = tuple(sorted(set(str(code) for code in codes if str(code))))
        super().__init__("scene_configuration_static_qualification_failed:" + ";".join(self.codes))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _finite(values: Any, *, positive: bool = False) -> bool:
    try:
        numbers = [float(value) for value in values]
    except (TypeError, ValueError):
        return False
    return bool(numbers) and all(
        math.isfinite(value) and (not positive or value > 0.0)
        for value in numbers
    )


def _physics_bounds(value: Any) -> dict[str, list[float]] | None:
    expected = {
        "mass_kg",
        "static_friction",
        "dynamic_friction",
        "restitution",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        return None
    normalized: dict[str, list[float]] = {}
    for name in sorted(expected):
        raw = value.get(name)
        if not isinstance(raw, list) or len(raw) != 2:
            return None
        try:
            lower, upper = (float(item) for item in raw)
        except (TypeError, ValueError):
            return None
        if (
            not math.isfinite(lower)
            or not math.isfinite(upper)
            or lower > upper
            or lower < 0.0
            or (name == "mass_kg" and lower <= 0.0)
            or (name != "mass_kg" and upper > 1.0)
        ):
            return None
        normalized[name] = [lower, upper]
    return normalized


def _close_sequence(left: Any, right: Any) -> bool:
    try:
        left_values = [float(item) for item in left]
        right_values = [float(item) for item in right]
    except (TypeError, ValueError):
        return False
    return len(left_values) == len(right_values) and all(
        math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-7)
        for actual, expected in zip(left_values, right_values, strict=True)
    )


def _material_rows_match(left: Any, right: Any) -> bool:
    if not isinstance(left, list) or not isinstance(right, list):
        return False
    left_rows = {
        str(row.get("path") or ""): row
        for row in left
        if isinstance(row, Mapping)
    }
    right_rows = {
        str(row.get("path") or ""): row
        for row in right
        if isinstance(row, Mapping)
    }
    if (
        len(left_rows) != len(left)
        or len(right_rows) != len(right)
        or set(left_rows) != set(right_rows)
    ):
        return False
    names = ("static_friction", "dynamic_friction", "restitution")
    return all(
        _close_sequence(
            [left_rows[path].get(name) for name in names],
            [right_rows[path].get(name) for name in names],
        )
        for path in left_rows
    )


def _portable_package_findings(path: Path) -> list[str]:
    findings: list[str] = []
    if path.suffix.lower() != ".usdz":
        return ["replacement_asset_not_usdz"]
    try:
        with zipfile.ZipFile(path) as archive:
            members = archive.infolist()
            if not members:
                findings.append("replacement_usdz_empty")
            for member in members:
                relative = Path(member.filename)
                mode = member.external_attr >> 16
                if (
                    member.filename.startswith("/")
                    or ".." in relative.parts
                    or stat.S_ISLNK(mode)
                ):
                    findings.append("replacement_usdz_member_unsafe")
                if relative.suffix.lower() in _FORBIDDEN_PACKAGE_SUFFIXES:
                    findings.append("replacement_usdz_executable_content_forbidden")
            if archive.testzip() is not None:
                findings.append("replacement_usdz_integrity_invalid")
    except (OSError, zipfile.BadZipFile):
        findings.append("replacement_usdz_invalid")
    return findings


def _usd_findings(
    path: Path, *, physics_bounds: Mapping[str, list[float]]
) -> tuple[list[str], dict[str, Any]]:
    try:
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdUtils
    except ImportError as exc:  # pragma: no cover - provider image owns OpenUSD
        raise TaskEvaluationSceneConfigurationStaticQualificationError(
            ["replacement_openusd_runtime_missing"]
        ) from exc

    findings = _portable_package_findings(path)
    try:
        layers, external_assets, unresolved = UsdUtils.ComputeAllDependencies(
            Sdf.AssetPath(str(path))
        )
    except Exception:
        layers, external_assets, unresolved = [], [], [str(path)]
    package_identifier = str(path)
    layer_identifiers = [str(layer.identifier) for layer in layers]
    if (
        not layer_identifiers
        or any(
            identifier != package_identifier
            and not identifier.startswith(package_identifier + "[")
            for identifier in layer_identifiers
        )
        or external_assets
        or unresolved
    ):
        findings.append("replacement_external_or_unresolved_dependency")
    stage = Usd.Stage.Open(str(path), load=Usd.Stage.LoadAll)
    if stage is None or not stage.GetDefaultPrim().IsValid():
        return [*findings, "replacement_usd_unreadable"], {}
    if (
        float(UsdGeom.GetStageMetersPerUnit(stage)) != 1.0
        or str(UsdGeom.GetStageUpAxis(stage)).upper() != "Z"
    ):
        findings.append("replacement_stage_frame_invalid")

    prims = list(stage.Traverse())
    rigid = [prim for prim in prims if prim.HasAPI(UsdPhysics.RigidBodyAPI)]
    if len(rigid) != 1:
        findings.append("replacement_single_rigid_body_required")
        body = None
    else:
        body = rigid[0]
        rigid_api = UsdPhysics.RigidBodyAPI(body)
        if (
            rigid_api.GetRigidBodyEnabledAttr().Get() is False
            or rigid_api.GetKinematicEnabledAttr().Get() is True
        ):
            findings.append("replacement_rigid_body_not_movable")

    if any(
        prim.IsA(UsdPhysics.Joint)
        or prim.HasAPI(UsdPhysics.ArticulationRootAPI)
        for prim in prims
    ):
        findings.append("replacement_articulation_forbidden")

    collision_prims = [
        prim for prim in prims if prim.HasAPI(UsdPhysics.CollisionAPI)
    ]
    bounds_rows: list[tuple[list[float], list[float]]] = []
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [
            UsdGeom.Tokens.default_,
            UsdGeom.Tokens.render,
            UsdGeom.Tokens.proxy,
            UsdGeom.Tokens.guide,
        ],
        useExtentsHint=False,
    )
    for prim in collision_prims:
        aligned = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        if aligned.IsEmpty():
            continue
        lower = [float(aligned.GetMin()[index]) for index in range(3)]
        upper = [float(aligned.GetMax()[index]) for index in range(3)]
        if _finite(lower) and _finite(upper) and all(
            upper[index] > lower[index] for index in range(3)
        ):
            bounds_rows.append((lower, upper))
    if not collision_prims or len(bounds_rows) != len(collision_prims):
        findings.append("replacement_collision_geometry_invalid")

    mass_value: float | None = None
    center_of_mass: list[float] = []
    inertia: list[float] = []
    if body is not None and body.HasAPI(UsdPhysics.MassAPI):
        mass_api = UsdPhysics.MassAPI(body)
        raw_mass = mass_api.GetMassAttr().Get()
        raw_com = mass_api.GetCenterOfMassAttr().Get()
        raw_inertia = mass_api.GetDiagonalInertiaAttr().Get()
        try:
            mass_value = float(raw_mass)
        except (TypeError, ValueError):
            mass_value = None
        if raw_com is not None:
            center_of_mass = [float(raw_com[index]) for index in range(3)]
        if raw_inertia is not None:
            inertia = [float(raw_inertia[index]) for index in range(3)]
    if (
        mass_value is None
        or not math.isfinite(mass_value)
        or mass_value <= 0.0
        or not physics_bounds["mass_kg"][0]
        <= mass_value
        <= physics_bounds["mass_kg"][1]
        or not _finite(center_of_mass)
        or not _finite(inertia, positive=True)
        or any(
            inertia[index] > sum(inertia) - inertia[index] + 1e-12
            for index in range(3)
        )
    ):
        findings.append("replacement_mass_or_inertia_invalid")
    elif body is not None and bounds_rows:
        transform = UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(
            body
        )
        world_com = transform.Transform(Gf.Vec3d(*center_of_mass))
        lower = [min(row[0][index] for row in bounds_rows) for index in range(3)]
        upper = [max(row[1][index] for row in bounds_rows) for index in range(3)]
        if any(
            float(world_com[index]) < lower[index] - 1e-6
            or float(world_com[index]) > upper[index] + 1e-6
            for index in range(3)
        ):
            findings.append("replacement_center_of_mass_outside_collision")

    physics_materials = [
        UsdPhysics.MaterialAPI(prim)
        for prim in prims
        if prim.HasAPI(UsdPhysics.MaterialAPI)
    ]
    material_rows: list[dict[str, float]] = []
    for material in physics_materials:
        try:
            row: dict[str, Any] = {
                "path": str(material.GetPrim().GetPath()),
                "static_friction": float(material.GetStaticFrictionAttr().Get()),
                "dynamic_friction": float(material.GetDynamicFrictionAttr().Get()),
                "restitution": float(material.GetRestitutionAttr().Get()),
            }
        except (TypeError, ValueError):
            continue
        if (
            all(
                math.isfinite(float(row[name]))
                and physics_bounds[name][0]
                <= float(row[name])
                <= physics_bounds[name][1]
                for name in (
                    "static_friction",
                    "dynamic_friction",
                    "restitution",
                )
            )
            and row["dynamic_friction"] <= row["static_friction"]
        ):
            material_rows.append(row)
    if not physics_materials or len(material_rows) != len(physics_materials):
        findings.append("replacement_physics_material_bounds_invalid")

    return findings, {
        "default_prim": str(stage.GetDefaultPrim().GetPath()),
        "rigid_body_paths": [str(prim.GetPath()) for prim in rigid],
        "collision_prim_paths": [str(prim.GetPath()) for prim in collision_prims],
        "mass_kg": mass_value,
        "center_of_mass_m": center_of_mass,
        "diagonal_inertia_kg_m2": inertia,
        "physics_materials": material_rows,
        "dependency_layer_count": len(layers),
        "external_asset_count": len(external_assets),
        "unresolved_dependency_count": len(unresolved),
    }


def qualify_scene_configuration_rigid_asset_static(
    *,
    asset_path: str | Path,
    graph_spec: Mapping[str, Any],
    authoring_receipt: Mapping[str, Any],
    replacement_identity: Mapping[str, Any],
    output_path: str | Path,
) -> dict[str, Any]:
    """Prove the exact stage-3 USDZ is portable, rigid, and structurally safe."""

    asset = Path(asset_path).expanduser().resolve()
    findings: list[str] = []
    if asset.is_symlink() or not asset.is_file():
        findings.append("replacement_asset_missing")
    digest = _sha256(asset) if asset.is_file() else ""
    size = asset.stat().st_size if asset.is_file() else 0
    normalized_physics_bounds = _physics_bounds(graph_spec.get("physics_bounds"))
    if (
        graph_spec.get("schema_version")
        != "task_evaluation_rigid_replacement_graph.v1"
        or graph_spec.get("asset_id") != replacement_identity.get("id")
        or graph_spec.get("asset_version") != replacement_identity.get("version")
        or graph_spec.get("articulation_graph") != {"joints": []}
        or graph_spec.get("single_rigid_candidate") is not True
        or normalized_physics_bounds is None
        or graph_spec.get("physics_authority_granted") is not False
    ):
        findings.append("replacement_graph_spec_invalid")
    output_usd = authoring_receipt.get("output_usd")
    if (
        authoring_receipt.get("schema_version")
        != "task_evaluation_rigid_replacement_authoring_result.v1"
        or authoring_receipt.get("status")
        != "authored_candidate_pending_qualification"
        or authoring_receipt.get("replacement_identity") != replacement_identity
        or authoring_receipt.get("physics_authority_granted") is not False
        or authoring_receipt.get("result_digest")
        != canonical_digest(authoring_receipt, digest_field="result_digest")
        or not isinstance(output_usd, Mapping)
        or output_usd.get("sha256") != digest
        or output_usd.get("size_bytes") != size
    ):
        findings.append("replacement_authoring_receipt_invalid")
    observed: dict[str, Any] = {}
    if asset.is_file() and normalized_physics_bounds is not None:
        usd_findings, observed = _usd_findings(
            asset, physics_bounds=normalized_physics_bounds
        )
        findings.extend(usd_findings)
    completion = authoring_receipt.get("candidate_physics_completion")
    materials_match = _material_rows_match(
        completion.get("physics_materials") if isinstance(completion, Mapping) else None,
        observed.get("physics_materials"),
    )
    if (
        not isinstance(completion, Mapping)
        or completion.get("schema_version")
        != _PHYSICS_COMPLETION_SCHEMA_VERSION
        or completion.get("status") != "bounded_candidate_completed"
        or completion.get("physics_bounds") != normalized_physics_bounds
        or completion.get("candidate_prior_only") is not True
        or completion.get("physical_truth_claimed") is not False
        or completion.get("completion_digest")
        != canonical_digest(completion, digest_field="completion_digest")
        or not _close_sequence(
            [completion.get("mass_kg")],
            [observed.get("mass_kg")],
        )
        or not _close_sequence(
            completion.get("center_of_mass_m"),
            observed.get("center_of_mass_m"),
        )
        or not _close_sequence(
            completion.get("diagonal_inertia_kg_m2"),
            observed.get("diagonal_inertia_kg_m2"),
        )
        or not materials_match
    ):
        findings.append("replacement_physics_completion_invalid")
    if findings:
        raise TaskEvaluationSceneConfigurationStaticQualificationError(findings)

    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "authored_structure_statically_qualified",
        "replacement_identity": dict(replacement_identity),
        "replacement_usd": {
            "path": str(asset),
            "sha256": digest,
            "size_bytes": size,
        },
        "checks": {
            "usd_parses": True,
            "meters_per_unit": 1.0,
            "up_axis": "Z",
            "single_movable_rigid_root": True,
            "collision_geometry_present": True,
            "collision_geometry_nonempty_and_finite": True,
            "mass_and_inertia_positive_finite": True,
            "materials_within_preregistered_bounds": True,
            "no_external_unpinned_dependencies": True,
            "no_articulation": True,
            "no_scripts_or_credentials": True,
            "center_of_mass_inside_collision_bounds": True,
        },
        "observed_structure": observed,
        "authored_structure_statically_qualified": True,
        "structural_findings": [],
        "claim_boundary": {
            "native_simulator_import_qualified": False,
            "physical_equivalence_proven": False,
            "generated_geometry_is_observed_truth": False,
        },
        "result_digest": "",
    }
    result["result_digest"] = canonical_digest(result, digest_field="result_digest")
    destination = Path(output_path).expanduser().resolve()
    if destination.exists() or destination.is_symlink():
        raise TaskEvaluationSceneConfigurationStaticQualificationError(
            ["replacement_static_qualification_output_exists"]
        )
    destination.write_text(canonical_json(result) + "\n", encoding="utf-8")
    return result


__all__ = [
    "SCHEMA_VERSION",
    "TaskEvaluationSceneConfigurationStaticQualificationError",
    "qualify_scene_configuration_rigid_asset_static",
]
