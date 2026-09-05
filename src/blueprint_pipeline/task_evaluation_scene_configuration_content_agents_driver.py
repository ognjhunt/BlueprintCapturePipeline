"""Bridge a generic scene-configuration stage to released Content Agents.

This is deliberately an adapter, not another Content Agents implementation.
It translates the Website-bound stage input into the manifest/configuration
files consumed by ``adp_content_agents_provider_runner.py``, runs that released
runtime inside the already allocated parent GPU, and seals its authored USD as
a candidate for the independent static and native-import stages that follow.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess  # nosec B404 - executable is package-manifest-bound
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, UsdUtils

from .adp_content_agents_vast import (
    CONTENT_IMAGE_MODEL,
    CONTENT_LLM_MODEL,
    CONTENT_LLM_REASONING_EFFORT,
    SOURCE_COMMIT,
    SOURCE_TREE,
    SOURCE_VERSION,
    _bounded_content_agent_render_selection,
    _derive_joint_agent_plan,
    _materialize_remote_configs,
    _validate_remote_configs,
)
from .decision_evidence_contracts import canonical_digest, canonical_json
from .provider_archive import extract_provider_archive
from .production_cad_skill_sources import SOURCE_SPECS
from .task_evaluation_scene_configuration_disclosure import MATERIALIZED_STATUS
from .task_evaluation_scene_configuration_stage_tool import (
    COMPONENT_RESULT_SCHEMA_VERSION,
)
from .task_evaluation_scene_configuration_openai_gate import (
    scene_configuration_openai_stage_gate,
    scene_configuration_openai_stage_scope,
)
from .task_evaluation_scene_configuration_content_agents_failure_evidence import (
    ContentAgentsRuntimeFailureEvidenceError,
    failure_evidence_secret_values,
    read_content_agents_runtime_result,
)
from .task_evaluation_scene_configuration_render_handoff import (
    ARTIFACT_ROLE as PROVIDER_RENDER_REFERENCE_ROLE,
    TaskEvaluationSceneConfigurationRenderHandoffError,
    validate_provider_render_handoff,
)


_INPUT_ENV = "BLUEPRINT_SCENE_CONFIGURATION_STAGE_INPUT"
_DEPENDENCIES_ENV = "BLUEPRINT_SCENE_CONFIGURATION_STAGE_DEPENDENCIES"
_OUTPUT_ENV = "BLUEPRINT_SCENE_CONFIGURATION_STAGE_OUTPUT_ROOT"
_RESULT_ENV = "BLUEPRINT_SCENE_CONFIGURATION_COMPONENT_RESULT"
_PACKAGE_ENV = "BLUEPRINT_SCENE_CONFIGURATION_COMPONENT_ROOT"
_ADAPTER_ID = "content_agents_rigid_replacement"
CONTENT_AGENTS_COMPONENT_TIMEOUT_SECONDS = 7_000
CONTENT_AGENTS_RUNNER_CLOSURE_MARGIN_SECONDS = 1_000
_PARENT_NATIVE_RUNTIME_ENV = (
    "PYTHONPATH",
    "LD_LIBRARY_PATH",
    "PXR_PLUGINPATH_NAME",
    "ISAAC_PATH",
    "EXP_PATH",
    "CARB_APP_PATH",
)
PHYSICS_COMPLETION_SCHEMA_VERSION = (
    "task_evaluation_rigid_candidate_physics_completion.v1"
)
_EXPECTED_PACKAGE_FILES = {
    "content_agents_source.zip",
    "content_agents_source_receipt.json",
    "run_adp_content_agents_provider_runtime.sh",
    "adp_content_agents_provider_runner.py",
    "provider_archive.py",
    "content_agents_model_compatibility.py",
    "content_agents_model_compatibility_plan.json",
    "text_to_cad_skills_source.zip",
    "multi_agent_cad_source.zip",
    "cad_skill_source_receipt.json",
    "multi_agent_cad_skill.md",
    "production_cad_skill_sources.py",
    "material_agent.yaml",
    "texture_agent.yaml",
    "physics_agent.yaml",
}


class TaskEvaluationSceneConfigurationContentAgentsError(RuntimeError):
    """The released Content Agents runtime could not satisfy the stage."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _read(path: Path, *, code: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TaskEvaluationSceneConfigurationContentAgentsError(code) from exc
    if path.is_symlink() or not isinstance(value, Mapping):
        raise TaskEvaluationSceneConfigurationContentAgentsError(code)
    return dict(value)


def _required_path(environment: Mapping[str, str], name: str) -> Path:
    unresolved = str(environment.get(name) or "").strip()
    if not unresolved:
        raise TaskEvaluationSceneConfigurationContentAgentsError(
            f"scene_configuration_content_agents_environment_missing:{name}"
        )
    return Path(unresolved).expanduser().resolve()


def _file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "digest": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _dependency_candidate(dependencies: list[Any]) -> tuple[Mapping[str, Any], Path]:
    matches = [
        artifact
        for result in dependencies
        if isinstance(result, Mapping)
        for artifact in result.get("output_artifacts") or []
        if isinstance(artifact, Mapping) and artifact.get("role") == "source_object_candidate_mesh"
    ]
    if len(matches) != 1:
        raise TaskEvaluationSceneConfigurationContentAgentsError(
            "scene_configuration_content_agents_source_candidate_missing"
        )
    record = matches[0]
    path = Path(str(record.get("path") or "")).expanduser().resolve()
    if (
        path.is_symlink()
        or not path.is_file()
        or path.stat().st_size != record.get("size_bytes")
        or _sha256(path) != record.get("digest")
    ):
        raise TaskEvaluationSceneConfigurationContentAgentsError(
            "scene_configuration_content_agents_source_candidate_invalid"
        )
    return record, path


def _normalize_candidate(source: Path, destination: Path) -> dict[str, Any]:
    source_stage = Usd.Stage.Open(str(source))
    if source_stage is None or not source_stage.GetDefaultPrim().IsValid():
        raise TaskEvaluationSceneConfigurationContentAgentsError(
            "scene_configuration_content_agents_source_usd_invalid"
        )
    source_prim = source_stage.GetPrimAtPath("/Root/SourceObjectCandidate")
    if not source_prim.IsValid():
        raise TaskEvaluationSceneConfigurationContentAgentsError(
            "scene_configuration_content_agents_source_prim_missing"
        )
    source_layer = source_stage.Flatten()
    layer = Sdf.Layer.CreateNew(str(destination))
    if layer is None:
        raise TaskEvaluationSceneConfigurationContentAgentsError(
            "scene_configuration_content_agents_input_create_failed"
        )
    asset_spec = Sdf.CreatePrimInLayer(layer, Sdf.Path("/Asset"))
    asset_spec.specifier = Sdf.SpecifierDef
    asset_spec.typeName = "Xform"
    geometry_spec = Sdf.CreatePrimInLayer(layer, Sdf.Path("/Asset/Geometry"))
    geometry_spec.specifier = Sdf.SpecifierDef
    geometry_spec.typeName = "Xform"
    if not Sdf.CopySpec(
        source_layer,
        Sdf.Path("/Root/SourceObjectCandidate"),
        layer,
        Sdf.Path("/Asset/Geometry/Visual"),
    ):
        raise TaskEvaluationSceneConfigurationContentAgentsError(
            "scene_configuration_content_agents_input_copy_failed"
        )
    layer.defaultPrim = "Asset"
    layer.Save()
    stage = Usd.Stage.Open(str(destination))
    if stage is None:
        raise TaskEvaluationSceneConfigurationContentAgentsError(
            "scene_configuration_content_agents_input_reopen_failed"
        )
    UsdGeom.SetStageMetersPerUnit(stage, UsdGeom.GetStageMetersPerUnit(source_stage))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.GetStageUpAxis(source_stage))
    asset = stage.GetPrimAtPath("/Asset")
    geometry = stage.GetPrimAtPath("/Asset/Geometry")
    visual = stage.GetPrimAtPath("/Asset/Geometry/Visual")
    meshes = [prim for prim in Usd.PrimRange(visual) if prim.IsA(UsdGeom.Mesh)]
    if (
        not asset.IsValid()
        or not geometry.IsValid()
        or not visual.IsValid()
        or not meshes
    ):
        raise TaskEvaluationSceneConfigurationContentAgentsError(
            "scene_configuration_content_agents_candidate_scope_invalid"
        )
    stripped_physics_schemas: set[str] = set()
    physics_apis = (
        UsdPhysics.ArticulationRootAPI,
        UsdPhysics.CollisionAPI,
        UsdPhysics.MassAPI,
        UsdPhysics.MaterialAPI,
        UsdPhysics.MeshCollisionAPI,
        UsdPhysics.RigidBodyAPI,
    )
    for prim in Usd.PrimRange(visual):
        if prim.IsA(UsdPhysics.Joint):
            raise TaskEvaluationSceneConfigurationContentAgentsError(
                "scene_configuration_content_agents_candidate_scope_invalid"
            )
        for schema in physics_apis:
            if prim.HasAPI(schema):
                stripped_physics_schemas.add(schema.__name__)
                prim.RemoveAPI(schema)
        for property_name in prim.GetPropertyNames():
            if str(property_name).startswith(("physics:", "physx")):
                prim.RemoveProperty(property_name)
    if any(
        str(schema).startswith(("Physics", "Physx"))
        for prim in Usd.PrimRange(visual)
        for schema in prim.GetAppliedSchemas()
    ):
        raise TaskEvaluationSceneConfigurationContentAgentsError(
            "scene_configuration_content_agents_candidate_scope_invalid"
        )
    for prim in meshes:
        UsdGeom.Imageable(prim).GetPurposeAttr().Set(UsdGeom.Tokens.default_)
    source_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [
            UsdGeom.Tokens.default_,
            UsdGeom.Tokens.render,
            UsdGeom.Tokens.proxy,
            UsdGeom.Tokens.guide,
        ],
        useExtentsHint=False,
    )
    source_bounds = source_cache.ComputeWorldBound(visual).ComputeAlignedRange()
    if source_bounds.IsEmpty():
        raise TaskEvaluationSceneConfigurationContentAgentsError(
            "scene_configuration_content_agents_candidate_bounds_invalid"
        )
    source_lower = source_bounds.GetMin()
    source_upper = source_bounds.GetMax()
    source_center = [
        (float(source_lower[index]) + float(source_upper[index])) / 2.0
        for index in range(3)
    ]
    source_to_candidate_translation = [
        -source_center[0],
        -source_center[1],
        -float(source_lower[2]),
    ]
    UsdGeom.Xformable(geometry).AddTranslateOp(
        opSuffix="blueprintCandidateLocalFrame"
    ).Set(Gf.Vec3d(*source_to_candidate_translation))
    material = UsdShade.Material.Define(stage, "/Asset/Looks/GeneratedCandidate")
    for prim in meshes:
        if not UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0]:
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)
    stage.GetRootLayer().Save()
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(), [UsdGeom.Tokens.default_], useExtentsHint=False
    )
    bounds = cache.ComputeWorldBound(asset).ComputeAlignedRange()
    if bounds.IsEmpty():
        raise TaskEvaluationSceneConfigurationContentAgentsError(
            "scene_configuration_content_agents_candidate_bounds_invalid"
        )
    normalized_lower = [float(value) for value in bounds.GetMin()]
    normalized_upper = [float(value) for value in bounds.GetMax()]
    if (
        not math.isclose(normalized_lower[2], 0.0, rel_tol=0.0, abs_tol=1e-7)
        or not math.isclose(
            (normalized_lower[0] + normalized_upper[0]) / 2.0,
            0.0,
            rel_tol=0.0,
            abs_tol=1e-7,
        )
        or not math.isclose(
            (normalized_lower[1] + normalized_upper[1]) / 2.0,
            0.0,
            rel_tol=0.0,
            abs_tol=1e-7,
        )
    ):
        raise TaskEvaluationSceneConfigurationContentAgentsError(
            "scene_configuration_content_agents_candidate_local_frame_invalid"
        )
    return {
        "source_input_usd_sha256": _sha256(source),
        "normalized_input_usd_sha256": _sha256(destination),
        "transformations": [
            "copy_exact_sage_candidate_subtree_to_asset_working_copy",
            "strip_source_collision_and_physics_authority_from_working_copy",
            "rebase_working_copy_to_object_local_xy_center_and_bottom_z_zero",
            "normalize_mesh_purpose_to_default",
            "bind_missing_materials_to_generated_candidate_material",
        ],
        "stripped_physics_schemas": sorted(stripped_physics_schemas),
        "source_world_bounds_m": {
            "minimum": [float(value) for value in source_lower],
            "maximum": [float(value) for value in source_upper],
        },
        "source_to_candidate_translation_m": source_to_candidate_translation,
        "candidate_local_bounds_m": {
            "minimum": normalized_lower,
            "maximum": normalized_upper,
        },
        "candidate_rigid_body_root_transform_identity": True,
        "default_purpose_bbox_nonempty": True,
        "scene_configuration_sage_candidate_working_copy": True,
        "mesh_count": len(meshes),
        "mesh_prim_paths": sorted(str(prim.GetPath()) for prim in meshes),
        "default_material_path": str(material.GetPath()),
        "joint_count": 0,
        "rigid_body_count": 0,
        "articulation_root_count": 0,
    }


def _copy_package_runtime(package_root: Path, runtime: Path) -> None:
    missing = [name for name in _EXPECTED_PACKAGE_FILES if not (package_root / name).is_file()]
    if missing:
        raise TaskEvaluationSceneConfigurationContentAgentsError(
            "scene_configuration_content_agents_package_incomplete:" + ",".join(sorted(missing))
        )
    for name in _EXPECTED_PACKAGE_FILES - {
        "material_agent.yaml",
        "texture_agent.yaml",
        "physics_agent.yaml",
    }:
        source = package_root / name
        target = runtime / name
        shutil.copyfile(source, target)
        if source.stat().st_mode & 0o111:
            target.chmod(0o700)


def _validate_source_receipt(runtime: Path) -> None:
    receipt = _read(
        runtime / "content_agents_source_receipt.json",
        code="scene_configuration_content_agents_source_receipt_invalid",
    )
    if (
        receipt.get("schema_version")
        != "task_evaluation_content_agents_component_source.v1"
        or receipt.get("repository")
        != "https://github.com/NVIDIA-Omniverse/usd-content-agents"
        or receipt.get("commit") != SOURCE_COMMIT
        or receipt.get("tree") != SOURCE_TREE
        or receipt.get("version") != SOURCE_VERSION
        or receipt.get("license") != "Apache-2.0"
        or receipt.get("archive_sha256")
        != _sha256(runtime / "content_agents_source.zip")
        or receipt.get("receipt_digest")
        != canonical_digest(receipt, digest_field="receipt_digest")
    ):
        raise TaskEvaluationSceneConfigurationContentAgentsError(
            "scene_configuration_content_agents_source_receipt_invalid"
        )


def _materialize_cad_skill_runtime(runtime: Path) -> dict[str, Any]:
    receipt = _read(
        runtime / "cad_skill_source_receipt.json",
        code="scene_configuration_cad_skill_receipt_invalid",
    )
    specs = {str(spec["id"]): spec for spec in SOURCE_SPECS}
    sources = receipt.get("sources")
    if (
        receipt.get("schema_version")
        != "task_evaluation_cad_skill_component_source.v1"
        or receipt.get("status") != "pinned_sources_packaged"
        or receipt.get("scene_specific_source") is not False
        or receipt.get("skill_count") != 10
        or not isinstance(sources, list)
        or len(sources) != 2
        or receipt.get("receipt_digest")
        != canonical_digest(receipt, digest_field="receipt_digest")
    ):
        raise TaskEvaluationSceneConfigurationContentAgentsError(
            "scene_configuration_cad_skill_receipt_invalid"
        )
    by_id = {
        str(row.get("id") or ""): row
        for row in sources
        if isinstance(row, Mapping)
    }
    archives = {
        "text-to-cad": runtime / "text_to_cad_skills_source.zip",
        "multi-agent-cad": runtime / "multi_agent_cad_source.zip",
    }
    for source_id, spec in specs.items():
        row = by_id.get(source_id)
        if (
            not isinstance(row, Mapping)
            or any(
                row.get(field) != spec[field]
                for field in (
                    "repository",
                    "commit",
                    "tree",
                    "license",
                    "license_sha256",
                )
            )
            or row.get("skills") != list(spec["skills"])
            or row.get("archive_sha256") != _sha256(archives[source_id])
        ):
            raise TaskEvaluationSceneConfigurationContentAgentsError(
                "scene_configuration_cad_skill_receipt_invalid"
            )
    root = runtime / "cad_authoring"
    text_root = root / "text-to-cad"
    multi_root = root / "Multi-Agent-CAD"
    extract_provider_archive(archives["text-to-cad"], text_root)
    extract_provider_archive(archives["multi-agent-cad"], multi_root)
    multi_skill = root / "skills" / "multi-agent-cad" / "SKILL.md"
    multi_skill.parent.mkdir(parents=True)
    shutil.copyfile(runtime / "multi_agent_cad_skill.md", multi_skill)
    for skill in specs["text-to-cad"]["skills"]:
        if not (text_root / "skills" / str(skill) / "SKILL.md").is_file():
            raise TaskEvaluationSceneConfigurationContentAgentsError(
                f"scene_configuration_cad_skill_missing:{skill}"
            )
    if not all(
        (multi_root / relative).is_file()
        for relative in (
            "multi_agent_cad/WORKFLOW.md",
            "multi_agent_cad/graph.py",
            "environment.yml",
        )
    ) or not multi_skill.is_file():
        raise TaskEvaluationSceneConfigurationContentAgentsError(
            "scene_configuration_cad_skill_missing:multi-agent-cad"
        )
    return {
        "status": "materialized",
        "root": str(root),
        "receipt_digest": receipt["receipt_digest"],
        "source_commits": {
            source_id: spec["commit"] for source_id, spec in specs.items()
        },
        "skills": sorted(
            [
                *[str(item) for item in specs["text-to-cad"]["skills"]],
                "multi-agent-cad",
            ]
        ),
    }


def _reference_frames(
    stage_input: Mapping[str, Any], dependencies: list[Any]
) -> list[Path]:
    """Resolve stage 1's explicit render handoff, never stale envelope state."""

    matches = [
        artifact
        for result in dependencies
        if isinstance(result, Mapping)
        for artifact in result.get("output_artifacts") or []
        if isinstance(artifact, Mapping)
        and artifact.get("role") == PROVIDER_RENDER_REFERENCE_ROLE
    ]
    if len(matches) != 1:
        raise TaskEvaluationSceneConfigurationContentAgentsError(
            "scene_configuration_content_agents_references_missing"
        )
    record = matches[0]
    unresolved_path = Path(str(record.get("path") or "")).expanduser()
    path = unresolved_path.resolve()
    if (
        unresolved_path.is_symlink()
        or not path.is_file()
        or path.stat().st_size != record.get("size_bytes")
        or _sha256(path) != record.get("digest")
    ):
        raise TaskEvaluationSceneConfigurationContentAgentsError(
            "scene_configuration_content_agents_reference_invalid:artifact_binding"
        )
    try:
        manifest, frames = validate_provider_render_handoff(path)
    except TaskEvaluationSceneConfigurationRenderHandoffError as exc:
        raise TaskEvaluationSceneConfigurationContentAgentsError(
            "scene_configuration_content_agents_reference_invalid:manifest"
        ) from exc
    render = (
        (stage_input.get("construction_envelope") or {}).get(
            "render_inputs_result"
        )
        or {}
    )
    expected_control_plane_digest = (
        render.get("control_plane_result_digest")
        or render.get("result_digest")
    )
    if manifest.get("control_plane_render_result_digest") != expected_control_plane_digest:
        raise TaskEvaluationSceneConfigurationContentAgentsError(
            "scene_configuration_content_agents_reference_invalid:handoff_control_plane_digest"
        )
    if (render.get("status") == MATERIALIZED_STATUS
            and manifest.get("source_render_result_digest") != render.get("result_digest")):
        raise TaskEvaluationSceneConfigurationContentAgentsError(
            "scene_configuration_content_agents_reference_invalid:handoff_render_result_digest"
        )
    return list(frames)


def _physics_output(root: Path) -> Path:
    # The released runner executes material -> physics -> texture so the final
    # Texture Agent export preserves both prior authored layers and its portable
    # sibling texture tree. Prefer that completed chain output; keep the legacy
    # physics-workdir search only for older retained runtime results.
    completed_chain = root / "texture_workdir/output/textured_output.usd"
    if completed_chain.is_file() and not completed_chain.is_symlink():
        return completed_chain
    physics_root = root / "physics_workdir"
    candidates = sorted(
        path
        for path in physics_root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".usd", ".usda", ".usdc"}
    )
    preferred = [path for path in candidates if "physics" in path.name.lower()]
    if not (preferred or candidates):
        raise TaskEvaluationSceneConfigurationContentAgentsError(
            "scene_configuration_content_agents_physics_output_missing"
        )
    return (preferred or candidates)[0]


def _physics_bounds(configuration: Mapping[str, Any]) -> dict[str, list[float]]:
    required = configuration.get("required_output")
    source_keys = {
        "mass_kg": "mass_kg_bounds",
        "static_friction": "static_friction_bounds",
        "dynamic_friction": "dynamic_friction_bounds",
        "restitution": "restitution_bounds",
    }
    bounds: dict[str, list[float]] = {}
    if not isinstance(required, Mapping):
        raise TaskEvaluationSceneConfigurationContentAgentsError(
            "scene_configuration_content_agents_physics_bounds_invalid"
        )
    for destination, source in source_keys.items():
        raw = required.get(source)
        if not isinstance(raw, list) or len(raw) != 2:
            raise TaskEvaluationSceneConfigurationContentAgentsError(
                "scene_configuration_content_agents_physics_bounds_invalid"
            )
        try:
            lower, upper = (float(value) for value in raw)
        except (TypeError, ValueError) as exc:
            raise TaskEvaluationSceneConfigurationContentAgentsError(
                "scene_configuration_content_agents_physics_bounds_invalid"
            ) from exc
        if (
            not math.isfinite(lower)
            or not math.isfinite(upper)
            or lower > upper
            or lower < 0.0
            or (destination == "mass_kg" and lower <= 0.0)
            or (destination != "mass_kg" and upper > 1.0)
        ):
            raise TaskEvaluationSceneConfigurationContentAgentsError(
                "scene_configuration_content_agents_physics_bounds_invalid"
            )
        bounds[destination] = [lower, upper]
    return bounds


def _vector3(value: Any) -> list[float]:
    try:
        result = [float(value[index]) for index in range(3)]
    except (TypeError, ValueError, IndexError):
        return []
    return result if all(math.isfinite(item) for item in result) else []


def _metric_envelope_spec(configuration: Mapping[str, Any]) -> dict[str, Any]:
    raw = configuration.get("metric_envelope")
    if not isinstance(raw, Mapping):
        raise TaskEvaluationSceneConfigurationContentAgentsError(
            "scene_configuration_content_agents_metric_envelope_invalid"
        )
    minimum = _vector3(raw.get("minimum_xyz_m"))
    maximum = _vector3(raw.get("maximum_xyz_m"))
    tolerance = raw.get("maximum_dimension_relative_error")
    if (
        not minimum
        or not maximum
        or isinstance(tolerance, bool)
        or not isinstance(tolerance, (int, float))
        or not math.isfinite(float(tolerance))
        or not 0.0 <= float(tolerance) <= 1.0
    ):
        raise TaskEvaluationSceneConfigurationContentAgentsError(
            "scene_configuration_content_agents_metric_envelope_invalid"
        )
    dimensions = [maximum[index] - minimum[index] for index in range(3)]
    if not all(value > 0.0 for value in dimensions):
        raise TaskEvaluationSceneConfigurationContentAgentsError(
            "scene_configuration_content_agents_metric_envelope_invalid"
        )
    return {
        "minimum_xyz_m": minimum,
        "maximum_xyz_m": maximum,
        "expected_dimensions_m": dimensions,
        "maximum_dimension_relative_error": float(tolerance),
    }


def _validate_metric_envelope_dimensions(
    *, envelope: Mapping[str, Any], observed_dimensions: Any
) -> dict[str, Any]:
    observed = _vector3(observed_dimensions)
    expected = _vector3(envelope.get("expected_dimensions_m"))
    tolerance = float(envelope["maximum_dimension_relative_error"])
    if not observed or not expected or not all(value > 0.0 for value in observed):
        raise TaskEvaluationSceneConfigurationContentAgentsError(
            "scene_configuration_content_agents_metric_envelope_invalid"
        )
    relative_errors = [
        abs(observed[index] - expected[index]) / expected[index]
        for index in range(3)
    ]
    if any(error > tolerance for error in relative_errors):
        raise TaskEvaluationSceneConfigurationContentAgentsError(
            "scene_configuration_content_agents_metric_envelope_mismatch"
        )
    return {
        "status": "within_preregistered_metric_envelope",
        "expected_dimensions_m": expected,
        "observed_collision_dimensions_m": observed,
        "dimension_relative_errors": relative_errors,
        "maximum_dimension_relative_error": tolerance,
    }


def _complete_candidate_physics(
    source: Path, *, bounds: Mapping[str, list[float]]
) -> dict[str, Any]:
    """Complete missing candidate-only COM/inertia and enforce sealed bounds."""

    stage = Usd.Stage.Open(str(source), load=Usd.Stage.LoadAll)
    if (
        stage is None
        or not stage.GetDefaultPrim().IsValid()
        or float(UsdGeom.GetStageMetersPerUnit(stage)) != 1.0
        or str(UsdGeom.GetStageUpAxis(stage)).upper() != "Z"
    ):
        raise TaskEvaluationSceneConfigurationContentAgentsError(
            "scene_configuration_content_agents_physics_output_frame_invalid"
        )
    prims = list(stage.Traverse())
    rigid = [prim for prim in prims if prim.HasAPI(UsdPhysics.RigidBodyAPI)]
    collision = [prim for prim in prims if prim.HasAPI(UsdPhysics.CollisionAPI)]
    if (
        len(rigid) != 1
        or not collision
        or any(
            prim.IsA(UsdPhysics.Joint)
            or prim.HasAPI(UsdPhysics.ArticulationRootAPI)
            for prim in prims
        )
    ):
        raise TaskEvaluationSceneConfigurationContentAgentsError(
            "scene_configuration_content_agents_physics_structure_invalid"
        )
    body = rigid[0]
    rigid_api = UsdPhysics.RigidBodyAPI(body)
    if (
        rigid_api.GetRigidBodyEnabledAttr().Get() is False
        or rigid_api.GetKinematicEnabledAttr().Get() is True
    ):
        raise TaskEvaluationSceneConfigurationContentAgentsError(
            "scene_configuration_content_agents_physics_structure_invalid"
        )

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
    world_to_body = (
        UsdGeom.XformCache(Usd.TimeCode.Default())
        .GetLocalToWorldTransform(body)
        .GetInverse()
    )
    local_points: list[Gf.Vec3d] = []
    for prim in collision:
        aligned = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        if aligned.IsEmpty():
            continue
        lower = aligned.GetMin()
        upper = aligned.GetMax()
        for x in (lower[0], upper[0]):
            for y in (lower[1], upper[1]):
                for z in (lower[2], upper[2]):
                    local_points.append(
                        world_to_body.Transform(Gf.Vec3d(float(x), float(y), float(z)))
                    )
    if not local_points:
        raise TaskEvaluationSceneConfigurationContentAgentsError(
            "scene_configuration_content_agents_collision_bounds_invalid"
        )
    lower = [min(float(point[index]) for point in local_points) for index in range(3)]
    upper = [max(float(point[index]) for point in local_points) for index in range(3)]
    dimensions = [upper[index] - lower[index] for index in range(3)]
    if not all(math.isfinite(value) and value > 0.0 for value in dimensions):
        raise TaskEvaluationSceneConfigurationContentAgentsError(
            "scene_configuration_content_agents_collision_bounds_invalid"
        )

    modifications: list[str] = []
    mass_api = UsdPhysics.MassAPI.Apply(body)
    mass_lower, mass_upper = bounds["mass_kg"]
    try:
        mass = float(mass_api.GetMassAttr().Get())
    except (TypeError, ValueError):
        mass = math.nan
    if not math.isfinite(mass) or not mass_lower <= mass <= mass_upper:
        mass = (mass_lower + mass_upper) / 2.0
        mass_api.CreateMassAttr(mass)
        modifications.append("mass_from_preregistered_bounds_midpoint")

    center = _vector3(mass_api.GetCenterOfMassAttr().Get())
    if not center or any(
        center[index] < lower[index] or center[index] > upper[index]
        for index in range(3)
    ):
        center = [(lower[index] + upper[index]) / 2.0 for index in range(3)]
        mass_api.CreateCenterOfMassAttr(Gf.Vec3f(*center))
        modifications.append("center_of_mass_from_collision_bounds_center")

    inertia = _vector3(mass_api.GetDiagonalInertiaAttr().Get())
    inertia_valid = bool(inertia) and all(value > 0.0 for value in inertia)
    if inertia_valid:
        inertia_valid = all(
            inertia[index] <= sum(inertia) - inertia[index] + 1e-12
            for index in range(3)
        )
    if not inertia_valid:
        x, y, z = dimensions
        inertia = [
            mass * (y * y + z * z) / 12.0,
            mass * (x * x + z * z) / 12.0,
            mass * (x * x + y * y) / 12.0,
        ]
        mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(*inertia))
        modifications.append("diagonal_inertia_from_collision_aabb")

    material_apis = [
        UsdPhysics.MaterialAPI(prim)
        for prim in prims
        if prim.HasAPI(UsdPhysics.MaterialAPI)
    ]
    if not material_apis:
        material = UsdShade.Material.Define(
            stage, "/Asset/Looks/BlueprintPhysicsCandidate"
        )
        material_apis = [UsdPhysics.MaterialAPI.Apply(material.GetPrim())]
        for prim in collision:
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                material, UsdShade.Tokens.weakerThanDescendants, "physics"
            )
        modifications.append("physics_material_from_preregistered_bounds_midpoints")
    material_rows: list[dict[str, Any]] = []
    attributes = {
        "static_friction": "GetStaticFrictionAttr",
        "dynamic_friction": "GetDynamicFrictionAttr",
        "restitution": "GetRestitutionAttr",
    }
    for material_api in material_apis:
        values: dict[str, float] = {}
        changed = False
        for name, getter_name in attributes.items():
            lower_bound, upper_bound = bounds[name]
            attribute = getattr(material_api, getter_name)()
            try:
                value = float(attribute.Get())
            except (TypeError, ValueError):
                value = math.nan
            if not math.isfinite(value) or not lower_bound <= value <= upper_bound:
                value = (lower_bound + upper_bound) / 2.0
                attribute.Set(value)
                changed = True
            values[name] = value
        if values["dynamic_friction"] > values["static_friction"]:
            values["static_friction"] = sum(bounds["static_friction"]) / 2.0
            values["dynamic_friction"] = sum(bounds["dynamic_friction"]) / 2.0
            material_api.GetStaticFrictionAttr().Set(values["static_friction"])
            material_api.GetDynamicFrictionAttr().Set(values["dynamic_friction"])
            changed = True
        if changed:
            modifications.append("physics_material_values_conformed_to_bounds")
        material_rows.append(
            {"path": str(material_api.GetPrim().GetPath()), **values}
        )

    stage.GetRootLayer().Save()
    reopened = Usd.Stage.Open(str(source), load=Usd.Stage.LoadAll)
    if reopened is None or not reopened.GetDefaultPrim().IsValid():
        raise TaskEvaluationSceneConfigurationContentAgentsError(
            "scene_configuration_content_agents_physics_completion_readback_invalid"
        )
    completion: dict[str, Any] = {
        "schema_version": PHYSICS_COMPLETION_SCHEMA_VERSION,
        "status": "bounded_candidate_completed",
        "rigid_body_path": str(body.GetPath()),
        "collision_prim_paths": sorted(str(prim.GetPath()) for prim in collision),
        "physics_bounds": {key: list(value) for key, value in bounds.items()},
        "mass_kg": mass,
        "center_of_mass_m": center,
        "diagonal_inertia_kg_m2": inertia,
        "collision_bounds_body_frame_m": {
            "minimum": lower,
            "maximum": upper,
        },
        "collision_dimensions_m": dimensions,
        "physics_materials": material_rows,
        "modifications": sorted(set(modifications)),
        "candidate_prior_only": True,
        "physical_truth_claimed": False,
        "completion_digest": "",
    }
    completion["completion_digest"] = canonical_digest(
        completion, digest_field="completion_digest"
    )
    return completion


def _package_replacement_asset(source: Path, destination: Path) -> None:
    """Carry the exact authored layer and every referenced asset together."""

    stage = Usd.Stage.Open(str(source), load=Usd.Stage.LoadAll)
    if stage is None or not stage.GetDefaultPrim().IsValid():
        raise TaskEvaluationSceneConfigurationContentAgentsError(
            "scene_configuration_content_agents_physics_output_invalid"
        )
    try:
        packaged = UsdUtils.CreateNewUsdzPackage(
            Sdf.AssetPath(str(source)), str(destination)
        )
    except Exception as exc:
        raise TaskEvaluationSceneConfigurationContentAgentsError(
            "scene_configuration_content_agents_physics_output_package_failed"
        ) from exc
    reopened = Usd.Stage.Open(str(destination), load=Usd.Stage.LoadAll)
    if not packaged or reopened is None or not reopened.GetDefaultPrim().IsValid():
        raise TaskEvaluationSceneConfigurationContentAgentsError(
            "scene_configuration_content_agents_physics_output_package_failed"
        )


def execute_content_agents_component(
    *,
    environment: Mapping[str, str] | None = None,
    runner: Any = subprocess.run,
    cost_gate_factory: Any = scene_configuration_openai_stage_gate,
) -> dict[str, Any]:
    """Invoke the released runtime once and seal its candidate artifacts."""

    values = dict(os.environ if environment is None else environment)
    stage_input = _read(
        _required_path(values, _INPUT_ENV),
        code="scene_configuration_content_agents_input_invalid",
    )
    dependencies_path = _required_path(values, _DEPENDENCIES_ENV)
    try:
        dependencies = json.loads(dependencies_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TaskEvaluationSceneConfigurationContentAgentsError(
            "scene_configuration_content_agents_dependencies_invalid"
        ) from exc
    stage = stage_input.get("stage") or {}
    configuration = stage_input.get("configuration") or {}
    if (
        stage.get("adapter", {}).get("id") != _ADAPTER_ID
        or configuration.get("schema_version") != "rigid_replacement_authoring_configuration.v1"
        or not isinstance(dependencies, list)
    ):
        raise TaskEvaluationSceneConfigurationContentAgentsError(
            "scene_configuration_content_agents_input_invalid"
        )
    metric_envelope = _metric_envelope_spec(configuration)
    source_record, source_candidate = _dependency_candidate(dependencies)
    output_root = _required_path(values, _OUTPUT_ENV)
    package_root = _required_path(values, _PACKAGE_ENV)
    component_result_path = _required_path(values, _RESULT_ENV)
    runtime = output_root / "released_content_agents_runtime"
    runtime.mkdir(mode=0o700)
    (runtime / "configs").mkdir()
    (runtime / "input").mkdir()
    _copy_package_runtime(package_root, runtime)
    _validate_source_receipt(runtime)
    cad_skill_runtime = _materialize_cad_skill_runtime(runtime)
    extract_provider_archive(
        runtime / "content_agents_source.zip",
        runtime / "content_agents_source",
    )
    normalized = _normalize_candidate(source_candidate, runtime / "input/source_asset.usda")
    references = _reference_frames(stage_input, dependencies)
    reference_relpaths: list[str] = []
    for index, source in enumerate(references):
        name = "reference.png" if index == 0 else f"reference_{index + 1:04d}.png"
        shutil.copyfile(source, runtime / "input" / name)
        reference_relpaths.append(f"../input/{name}")
    selection = _bounded_content_agent_render_selection(
        usd_path=runtime / "input/source_asset.usda",
        mesh_prim_paths=normalized["mesh_prim_paths"],
    )
    config_sources = {
        name: package_root / name
        for name in (
            "material_agent.yaml",
            "texture_agent.yaml",
            "physics_agent.yaml",
        )
    }
    config_hashes = _materialize_remote_configs(
        config_sources=config_sources,
        destination=runtime / "configs",
        variant="scene_configuration_v1",
        agent_mesh_prim_paths=normalized["mesh_prim_paths"],
        agent_render_prim_paths=selection["selected_mesh_prim_paths"],
        agent_default_material_path=normalized["default_material_path"],
        reference_image_relpaths=reference_relpaths,
    )
    _validate_remote_configs(
        source=runtime / "content_agents_source",
        config_sources={name: runtime / "configs" / name for name in config_sources},
    )
    joint_plan = _derive_joint_agent_plan(
        input_variant="scene_configuration_v1", input_normalization=normalized
    )
    manifest = {
        "schema_version": "adp_content_agents_provider_bundle.v1",
        "status": "ready",
        "source_repository": "https://github.com/NVIDIA-Omniverse/usd-content-agents",
        "source_commit": SOURCE_COMMIT,
        "source_tree": SOURCE_TREE,
        "source_version": SOURCE_VERSION,
        "input_variant": "scene_configuration_v1",
        "input_usd_sha256": normalized["normalized_input_usd_sha256"],
        "input_usd_normalization": normalized,
        "agent_dataset_render_selection": selection,
        "reference_image_sha256s": [_sha256(path) for path in references],
        "remote_config_sha256": config_hashes,
        "runtime_input_binding": {
            "relative_path": "input/source_asset.usda",
            "sha256": normalized["normalized_input_usd_sha256"],
        },
        "joint_agent_plan": joint_plan,
        "model_identity": {
            "text": CONTENT_LLM_MODEL,
            "reasoning_effort": CONTENT_LLM_REASONING_EFFORT,
            "image": CONTENT_IMAGE_MODEL,
        },
        "cad_skill_runtime": cad_skill_runtime,
        "retry_cap": 0,
        "provider_zero_required_after_return": True,
        "raw_secret_values_recorded": False,
    }
    (runtime / "adp_content_agents_provider_manifest.json").write_text(
        canonical_json(manifest) + "\n", encoding="utf-8"
    )
    child_environment = {
        **values,
        "BLUEPRINT_ADP_CONTENT_AGENTS_OUTPUT_DIR": str(runtime / "runtime_output"),
        "BLUEPRINT_PRODUCTION_CAD_SKILLS_ROOT": cad_skill_runtime["root"],
    }
    # The parent provider intentionally carries both a sealed standalone
    # ``usd-core`` tree and Isaac/Kit's native runtime.  The released Content
    # Agents runtime provisions an independent Python 3.12/OpenUSD closure.
    # Do not let the parent's Python modules, plugin registry, or C++ loader
    # paths cross that process boundary and silently mix incompatible ABIs.
    for name in _PARENT_NATIVE_RUNTIME_ENV:
        child_environment.pop(name, None)
    stage_scope = scene_configuration_openai_stage_scope(
        values, stage="content_agents"
    )
    secret_path = Path(stage_scope["api_key_file"]).expanduser().resolve()
    if (
        secret_path.is_symlink()
        or not secret_path.is_file()
        or secret_path.stat().st_mode & 0o077
    ):
        raise TaskEvaluationSceneConfigurationContentAgentsError(
            "scene_configuration_content_agents_secret_file_invalid"
        )
    child_token = secret_path.read_text(encoding="utf-8").strip()
    if not child_token:
        raise TaskEvaluationSceneConfigurationContentAgentsError(
            "scene_configuration_content_agents_secret_file_invalid"
        )
    child_environment["OPENAI_API_KEY"] = child_token
    cost_gate = cost_gate_factory(
        environment=values,
        stage="content_agents",
        run_id=f"{stage_input['run_id']}-content-agents",
        request_digest=_sha256(_required_path(values, _INPUT_ENV)),
        candidate_digest=str(source_record["digest"]),
        output_root=runtime / "official_openai_cost",
    )
    cost_gate.reserve()
    try:
        completed = runner(
            [str(runtime / "run_adp_content_agents_provider_runtime.sh")],
            cwd=runtime,
            env=child_environment,
            capture_output=True,
            text=True,
            check=False,
            timeout=CONTENT_AGENTS_COMPONENT_TIMEOUT_SECONDS,
        )
    except Exception as exc:
        cost_gate.complete(
            provider_call_performed=True,
            runtime_result_digest=None,
            runtime_exception_type=type(exc).__name__,
        )
        raise
    runtime_output = runtime / "runtime_output"
    try:
        runtime_result = read_content_agents_runtime_result(
            completed=completed,
            runtime_result_path=(
                runtime_output / "adp_content_agents_vast_result.json"
            ),
            evidence_path=(
                runtime / "content_agents_runtime_failure_evidence.v1.json"
            ),
            secret_values=failure_evidence_secret_values(
                values, known_values=(child_token,)
            ),
        )
    except ContentAgentsRuntimeFailureEvidenceError as exc:
        cost_gate.complete(
            provider_call_performed=True,
            runtime_result_digest=exc.runtime_result_digest,
            runtime_exception_type=type(exc).__name__,
        )
        raise TaskEvaluationSceneConfigurationContentAgentsError(str(exc)) from exc
    except Exception as exc:
        cost_gate.complete(
            provider_call_performed=True,
            runtime_result_digest=None,
            runtime_exception_type=type(exc).__name__,
        )
        raise
    cost_gate.complete(
        provider_call_performed=True,
        runtime_result_digest=str(runtime_result.get("result_digest") or "") or None,
        runtime_exception_type=None,
    )
    authored = _physics_output(runtime_output)
    physics_bounds = _physics_bounds(configuration)
    physics_completion = _complete_candidate_physics(
        authored, bounds=physics_bounds
    )
    physics_completion["metric_envelope_validation"] = (
        _validate_metric_envelope_dimensions(
            envelope=metric_envelope,
            observed_dimensions=physics_completion["collision_dimensions_m"],
        )
    )
    physics_completion["completion_digest"] = canonical_digest(
        physics_completion, digest_field="completion_digest"
    )
    asset = output_root / "content_agents_replacement_candidate.usdz"
    _package_replacement_asset(authored, asset)
    identity = configuration["replacement_identity"]
    graph = {
        "schema_version": "task_evaluation_rigid_replacement_graph.v1",
        "asset_id": identity["id"],
        "asset_version": identity["version"],
        "articulation_graph": {"joints": []},
        "single_rigid_candidate": True,
        "physics_bounds": physics_bounds,
        "physics_authority_granted": False,
    }
    graph_path = output_root / "replacement_graph_spec.v1.json"
    graph_path.write_text(canonical_json(graph) + "\n", encoding="utf-8")
    receipt = {
        "schema_version": "task_evaluation_rigid_replacement_authoring_result.v1",
        "status": "authored_candidate_pending_qualification",
        "replacement_identity": identity,
        "source_candidate_digest": source_record["digest"],
        "source_candidate_claim": (
            "sage_candidate_geometry_not_observed_truth_or_physics_authority"
        ),
        "content_agents_runtime_result": _file_record(
            runtime_output / "adp_content_agents_vast_result.json"
        ),
        "output_usd": {
            "sha256": _sha256(asset),
            "size_bytes": asset.stat().st_size,
        },
        "candidate_physics_completion": physics_completion,
        "physics_authority_granted": False,
        "result_digest": "",
    }
    receipt["result_digest"] = canonical_digest(receipt, digest_field="result_digest")
    receipt_path = output_root / "replacement_authoring_receipt.v1.json"
    receipt_path.write_text(canonical_json(receipt) + "\n", encoding="utf-8")
    artifacts = [
        {"role": "replacement_asset", **_file_record(asset)},
        {"role": "replacement_authoring_receipt", **_file_record(receipt_path)},
        {"role": "replacement_graph_spec", **_file_record(graph_path)},
    ]
    result = {
        "schema_version": COMPONENT_RESULT_SCHEMA_VERSION,
        "status": "completed",
        "adapter_id": _ADAPTER_ID,
        "stage_id": stage["stage_id"],
        "provider_mutations_performed": 0,
        "nested_paid_execution_requested": False,
        "artifacts": artifacts,
        "result_digest": "",
    }
    result["result_digest"] = canonical_digest(result, digest_field="result_digest")
    component_result_path.write_text(canonical_json(result) + "\n", encoding="utf-8")
    return result


def main() -> int:
    execute_content_agents_component()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "TaskEvaluationSceneConfigurationContentAgentsError",
    "execute_content_agents_component",
    "main",
]
