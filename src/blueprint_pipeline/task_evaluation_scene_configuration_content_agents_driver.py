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
import os
import shutil
import subprocess  # nosec B404 - executable is package-manifest-bound
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pxr import Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

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
from .task_evaluation_scene_configuration_stage_tool import (
    COMPONENT_RESULT_SCHEMA_VERSION,
)
from .task_evaluation_scene_configuration_openai_gate import (
    scene_configuration_openai_stage_gate,
)


_INPUT_ENV = "BLUEPRINT_SCENE_CONFIGURATION_STAGE_INPUT"
_DEPENDENCIES_ENV = "BLUEPRINT_SCENE_CONFIGURATION_STAGE_DEPENDENCIES"
_OUTPUT_ENV = "BLUEPRINT_SCENE_CONFIGURATION_STAGE_OUTPUT_ROOT"
_RESULT_ENV = "BLUEPRINT_SCENE_CONFIGURATION_COMPONENT_RESULT"
_PACKAGE_ENV = "BLUEPRINT_SCENE_CONFIGURATION_COMPONENT_ROOT"
_ADAPTER_ID = "content_agents_rigid_replacement"
_EXPECTED_PACKAGE_FILES = {
    "content_agents_source.zip",
    "content_agents_source_receipt.json",
    "run_adp_content_agents_provider_runtime.sh",
    "adp_content_agents_provider_runner.py",
    "provider_archive.py",
    "content_agents_model_compatibility.py",
    "content_agents_model_compatibility_plan.json",
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
    if not Sdf.CopySpec(
        source_layer,
        Sdf.Path("/Root/SourceObjectCandidate"),
        layer,
        Sdf.Path("/Asset/Visual"),
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
    visual = stage.GetPrimAtPath("/Asset/Visual")
    meshes = [prim for prim in Usd.PrimRange(visual) if prim.IsA(UsdGeom.Mesh)]
    if (
        not asset.IsValid()
        or not visual.IsValid()
        or not meshes
        or any(
            prim.HasAPI(UsdPhysics.RigidBodyAPI)
            or prim.HasAPI(UsdPhysics.CollisionAPI)
            or prim.IsA(UsdPhysics.Joint)
            for prim in Usd.PrimRange(asset)
        )
    ):
        raise TaskEvaluationSceneConfigurationContentAgentsError(
            "scene_configuration_content_agents_candidate_scope_invalid"
        )
    for prim in meshes:
        UsdGeom.Imageable(prim).GetPurposeAttr().Set(UsdGeom.Tokens.default_)
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
    return {
        "source_input_usd_sha256": _sha256(source),
        "normalized_input_usd_sha256": _sha256(destination),
        "transformations": [
            "copy_exact_sage_candidate_subtree_to_asset_working_copy",
            "normalize_mesh_purpose_to_default",
            "bind_missing_materials_to_generated_candidate_material",
        ],
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


def _reference_frames(stage_input: Mapping[str, Any]) -> list[Path]:
    render = (stage_input.get("construction_envelope") or {}).get("render_inputs_result") or {}
    paths: list[Path] = []
    for row in render.get("derived_frames") or []:
        if not isinstance(row, Mapping):
            continue
        path = Path(str(row.get("path") or "")).expanduser().resolve()
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != row.get("size_bytes")
            or _sha256(path) != row.get("digest")
        ):
            raise TaskEvaluationSceneConfigurationContentAgentsError(
                "scene_configuration_content_agents_reference_invalid"
            )
        paths.append(path)
    if not paths:
        raise TaskEvaluationSceneConfigurationContentAgentsError(
            "scene_configuration_content_agents_references_missing"
        )
    return paths


def _physics_output(root: Path) -> Path:
    candidates = sorted(
        path
        for path in (root / "physics_workdir").rglob("*")
        if path.is_file() and path.suffix.lower() in {".usd", ".usda", ".usdc"}
    )
    preferred = [path for path in candidates if "physics" in path.name.lower()]
    if not (preferred or candidates):
        raise TaskEvaluationSceneConfigurationContentAgentsError(
            "scene_configuration_content_agents_physics_output_missing"
        )
    return (preferred or candidates)[0]


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
    extract_provider_archive(
        runtime / "content_agents_source.zip",
        runtime / "content_agents_source",
    )
    normalized = _normalize_candidate(source_candidate, runtime / "input/source_asset.usda")
    references = _reference_frames(stage_input)
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
    }
    secret_file = str(values.get("OPENAI_API_KEY_FILE") or "").strip()
    if secret_file:
        secret_path = Path(secret_file).expanduser().resolve()
        if (
            secret_path.is_symlink()
            or not secret_path.is_file()
            or secret_path.stat().st_mode & 0o077
        ):
            raise TaskEvaluationSceneConfigurationContentAgentsError(
                "scene_configuration_content_agents_secret_file_invalid"
            )
        child_environment["OPENAI_API_KEY"] = secret_path.read_text(encoding="utf-8").strip()
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
            timeout=7_000,
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
        runtime_result = _read(
            runtime_output / "adp_content_agents_vast_result.json",
            code="scene_configuration_content_agents_runtime_result_missing",
        )
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
    if completed.returncode != 0 or runtime_result.get("status") != "completed":
        raise TaskEvaluationSceneConfigurationContentAgentsError(
            "scene_configuration_content_agents_runtime_failed"
        )
    authored = _physics_output(runtime_output)
    asset = output_root / "content_agents_replacement_candidate.usda"
    shutil.copyfile(authored, asset)
    identity = configuration["replacement_identity"]
    graph = {
        "schema_version": "task_evaluation_rigid_replacement_graph.v1",
        "asset_id": identity["id"],
        "asset_version": identity["version"],
        "articulation_graph": {"joints": []},
        "single_rigid_candidate": True,
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
