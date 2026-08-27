from __future__ import annotations

import hashlib
import json
import math
import subprocess
import zipfile
from pathlib import Path

import pytest
from PIL import Image
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdUtils

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_evaluation_scene_configuration_content_agents_driver import (
    TaskEvaluationSceneConfigurationContentAgentsError,
    _metric_envelope_spec,
    _validate_metric_envelope_dimensions,
    execute_content_agents_component,
)


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _record(path: Path, *, role: str | None = None) -> dict:
    value = {
        "path": str(path),
        "digest": _sha256(path),
        "size_bytes": path.stat().st_size,
    }
    if role:
        value["role"] = role
    return value


def _candidate(path: Path) -> None:
    stage = Usd.Stage.CreateNew(str(path))
    root = UsdGeom.Xform.Define(stage, "/Root").GetPrim()
    stage.SetDefaultPrim(root)
    UsdGeom.Xform.Define(stage, "/Root/SourceObjectCandidate")
    mesh = UsdGeom.Mesh.Define(stage, "/Root/SourceObjectCandidate/body")
    mesh.CreatePointsAttr(
        [
            Gf.Vec3f(0.0, 0.0, 0.0),
            Gf.Vec3f(0.1, 0.0, 0.0),
            Gf.Vec3f(0.0, 0.1, 0.1),
        ]
    )
    mesh.CreateFaceVertexCountsAttr([3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
    UsdGeom.Xformable(mesh.GetPrim()).AddTranslateOp().Set(
        Gf.Vec3d(2.9742285, -6.7605156, 0.818319)
    )
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr(
        UsdPhysics.Tokens.convexHull
    )
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    stage.GetRootLayer().Save()


def _package(path: Path) -> None:
    path.mkdir()
    source_archive = path / "content_agents_source.zip"
    with zipfile.ZipFile(source_archive, "w") as archive:
        archive.writestr(
            "apps/material_agent/data/materials/material_libs_default/materials.yaml",
            "materials: {}\n",
        )
    source_receipt = {
        "schema_version": "task_evaluation_content_agents_component_source.v1",
        "repository": (
            "https://github.com/NVIDIA-Omniverse/usd-content-agents"
        ),
        "commit": "36dbf3f274f8e256637230a05a085853f65cc175",
        "tree": "d36ddaed4c3ea44ab81c9f8178ab40d2eb0f8fe3",
        "version": "0.5.2",
        "license": "Apache-2.0",
        "archive_sha256": _sha256(source_archive),
        "receipt_digest": "",
    }
    source_receipt["receipt_digest"] = canonical_digest(
        source_receipt, digest_field="receipt_digest"
    )
    (path / "content_agents_source_receipt.json").write_text(
        json.dumps(source_receipt), encoding="utf-8"
    )
    for name in (
        "run_adp_content_agents_provider_runtime.sh",
        "adp_content_agents_provider_runner.py",
        "provider_archive.py",
        "content_agents_model_compatibility.py",
    ):
        target = path / name
        target.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        if name.endswith(".sh"):
            target.chmod(0o755)
    (path / "content_agents_model_compatibility_plan.json").write_text("{}\n", encoding="utf-8")
    assets = ROOT / "docs/arm_decision_proof_v1/assets"
    for source_name, destination_name in (
        ("adp009a_content_agents_material.vast.yaml", "material_agent.yaml"),
        ("adp009a_content_agents_texture.vast.yaml", "texture_agent.yaml"),
        ("adp009a_content_agents_physics.vast.yaml", "physics_agent.yaml"),
    ):
        (path / destination_name).write_bytes((assets / source_name).read_bytes())


def test_reuses_released_content_agents_runner_and_seals_candidate(
    tmp_path: Path,
) -> None:
    candidate = tmp_path / "candidate.usda"
    _candidate(candidate)
    reference = tmp_path / "reference.png"
    Image.new("RGB", (16, 16), (240, 240, 240)).save(reference)
    package = tmp_path / "package"
    _package(package)
    output = tmp_path / "output"
    output.mkdir()
    stage_input = {
        "run_id": "configure-scene-839873-v1",
        "stage": {
            "stage_id": "03-author-replacement",
            "adapter": {"id": "content_agents_rigid_replacement", "version": "v1"},
        },
        "configuration": {
            "schema_version": "rigid_replacement_authoring_configuration.v1",
            "replacement_identity": {"id": "mug", "version": "v1"},
            "metric_envelope": {
                "minimum_xyz_m": [0.0, 0.0, 0.0],
                "maximum_xyz_m": [2.0, 2.0, 2.0],
                "maximum_dimension_relative_error": 0.05,
            },
            "required_output": {
                "format": "OpenUSD",
                "rigid_body": True,
                "single_movable_root": True,
                "visual_mesh_separate_from_collision": True,
                "units": "meters",
                "up_axis": "Z",
                "mass_kg_bounds": [0.2, 0.8],
                "static_friction_bounds": [0.3, 0.9],
                "dynamic_friction_bounds": [0.2, 0.8],
                "restitution_bounds": [0.0, 0.15],
            },
        },
        "construction_envelope": {"render_inputs_result": {"derived_frames": [_record(reference)]}},
    }
    stage_input_path = output / "input.json"
    stage_input_path.write_text(json.dumps(stage_input), encoding="utf-8")
    dependencies = [{"output_artifacts": [_record(candidate, role="source_object_candidate_mesh")]}]
    dependencies_path = output / "dependencies.json"
    dependencies_path.write_text(json.dumps(dependencies), encoding="utf-8")
    component_result = output / "component-result.json"
    observed: list[list[str]] = []
    cost_events: list[str] = []

    class CostGate:
        def reserve(self):
            cost_events.append("reserved")

        def complete(self, **_kwargs):
            cost_events.append("completed")

    def cost_gate_factory(**_kwargs):
        return CostGate()

    def run(command, *, env, **_kwargs):
        observed.append(command)
        for name in (
            "PYTHONPATH",
            "LD_LIBRARY_PATH",
            "PXR_PLUGINPATH_NAME",
            "ISAAC_PATH",
            "EXP_PATH",
            "CARB_APP_PATH",
        ):
            assert name not in env
        assert env["PATH"] == "/usr/bin"
        runtime_output = Path(env["BLUEPRINT_ADP_CONTENT_AGENTS_OUTPUT_DIR"])
        physics = runtime_output / "physics_workdir/physics_candidate.usda"
        physics.parent.mkdir(parents=True)
        dependency = physics.parent / "physics_dependency.usda"
        dependency_stage = Usd.Stage.CreateNew(str(dependency))
        dependency_root = UsdGeom.Cube.Define(
            dependency_stage, "/PhysicsGeometry"
        ).GetPrim()
        dependency_stage.SetDefaultPrim(dependency_root)
        UsdPhysics.CollisionAPI.Apply(dependency_root)
        dependency_stage.GetRootLayer().Save()
        physics_stage = Usd.Stage.CreateNew(str(physics))
        physics_root = UsdGeom.Xform.Define(physics_stage, "/Asset").GetPrim()
        physics_stage.SetDefaultPrim(physics_root)
        UsdGeom.SetStageMetersPerUnit(physics_stage, 1.0)
        UsdGeom.SetStageUpAxis(physics_stage, UsdGeom.Tokens.z)
        UsdPhysics.RigidBodyAPI.Apply(physics_root)
        UsdPhysics.MassAPI.Apply(physics_root).CreateMassAttr(0.4)
        physics_stage.DefinePrim("/Asset/Geometry").GetReferences().AddReference(
            str(dependency), "/PhysicsGeometry"
        )
        physics_stage.GetRootLayer().Save()
        (runtime_output / "adp_content_agents_vast_result.json").write_text(
            json.dumps(
                {
                    "schema_version": "adp_content_agents_vast_result.v1",
                    "status": "completed",
                    "material_agent_executed": True,
                    "texture_agent_executed": True,
                    "physics_agent_executed": True,
                    "validation_agent_executed": True,
                    "retry_cap": 0,
                    "blockers": [],
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    stage_key = tmp_path / "openai_api_key_content_agents"
    stage_key.write_text("test-content-agents-key", encoding="utf-8")
    stage_key.chmod(0o600)
    result = execute_content_agents_component(
        environment={
            "BLUEPRINT_SCENE_CONFIGURATION_STAGE_INPUT": str(stage_input_path),
            "BLUEPRINT_SCENE_CONFIGURATION_STAGE_DEPENDENCIES": str(dependencies_path),
            "BLUEPRINT_SCENE_CONFIGURATION_STAGE_OUTPUT_ROOT": str(output),
            "BLUEPRINT_SCENE_CONFIGURATION_COMPONENT_RESULT": str(component_result),
            "BLUEPRINT_SCENE_CONFIGURATION_COMPONENT_ROOT": str(package),
            "PATH": "/usr/bin",
            "PYTHONPATH": "/sealed/provider_python_runtime",
            "LD_LIBRARY_PATH": "/isaac-sim/kit:/sealed/usd-core/lib",
            "PXR_PLUGINPATH_NAME": "/isaac-sim/kit/plugins",
            "ISAAC_PATH": "/isaac-sim",
            "EXP_PATH": "/isaac-sim/apps",
            "CARB_APP_PATH": "/isaac-sim/kit",
            "OPENAI_CONTENT_AGENTS_API_KEY_FILE": str(stage_key),
            "OPENAI_CONTENT_AGENTS_API_KEY_ID": "key_content_agents",
            "BLUEPRINT_OPENAI_CONTENT_AGENTS_COST_SCOPE_ATTESTATION_FILE": str(
                tmp_path / "cost_scope_content_agents.json"
            ),
        },
        runner=run,
        cost_gate_factory=cost_gate_factory,
    )

    assert observed == [
        [str(output / "released_content_agents_runtime/run_adp_content_agents_provider_runtime.sh")]
    ]
    assert result["status"] == "completed"
    assert cost_events == ["reserved", "completed"]
    assert result["provider_mutations_performed"] == 0
    assert result["nested_paid_execution_requested"] is False
    assert {row["role"] for row in result["artifacts"]} == {
        "replacement_asset",
        "replacement_authoring_receipt",
        "replacement_graph_spec",
    }
    assert result["result_digest"] == canonical_digest(result, digest_field="result_digest")
    replacement = next(
        Path(row["path"])
        for row in result["artifacts"]
        if row["role"] == "replacement_asset"
    )
    layers, external_assets, unresolved = UsdUtils.ComputeAllDependencies(
        Sdf.AssetPath(str(replacement))
    )
    assert replacement.suffix == ".usdz"
    assert len(layers) == 2
    assert all(
        layer.identifier == str(replacement)
        or layer.identifier.startswith(str(replacement) + "[")
        for layer in layers
    )
    assert external_assets == []
    assert unresolved == []
    replacement_stage = Usd.Stage.Open(str(replacement), load=Usd.Stage.LoadAll)
    assert replacement_stage is not None
    rigid = [
        prim
        for prim in replacement_stage.Traverse()
        if prim.HasAPI(UsdPhysics.RigidBodyAPI)
    ]
    assert len(rigid) == 1
    mass = UsdPhysics.MassAPI(rigid[0])
    assert float(mass.GetMassAttr().Get()) == pytest.approx(0.4)
    assert all(math.isfinite(float(value)) for value in mass.GetCenterOfMassAttr().Get())
    assert all(float(value) > 0.0 for value in mass.GetDiagonalInertiaAttr().Get())
    physics_materials = [
        UsdPhysics.MaterialAPI(prim)
        for prim in replacement_stage.Traverse()
        if prim.HasAPI(UsdPhysics.MaterialAPI)
    ]
    assert len(physics_materials) == 1
    assert 0.3 <= float(physics_materials[0].GetStaticFrictionAttr().Get()) <= 0.9
    assert 0.2 <= float(physics_materials[0].GetDynamicFrictionAttr().Get()) <= 0.8
    assert 0.0 <= float(physics_materials[0].GetRestitutionAttr().Get()) <= 0.15
    receipt = json.loads((output / "replacement_authoring_receipt.v1.json").read_text())
    assert receipt["source_candidate_digest"] == _sha256(candidate)
    assert receipt["physics_authority_granted"] is False
    completion = receipt["candidate_physics_completion"]
    assert completion["schema_version"] == (
        "task_evaluation_rigid_candidate_physics_completion.v1"
    )
    assert completion["mass_kg"] == pytest.approx(0.4)
    assert completion["candidate_prior_only"] is True
    assert completion["physical_truth_claimed"] is False
    assert completion["collision_dimensions_m"] == pytest.approx([2.0, 2.0, 2.0])
    assert completion["metric_envelope_validation"]["status"] == (
        "within_preregistered_metric_envelope"
    )
    assert "center_of_mass_from_collision_bounds_center" in completion["modifications"]
    assert "diagonal_inertia_from_collision_aabb" in completion["modifications"]
    assert "physics_material_from_preregistered_bounds_midpoints" in completion["modifications"]
    manifest = json.loads(
        (
            output / "released_content_agents_runtime/adp_content_agents_provider_manifest.json"
        ).read_text()
    )
    assert manifest["input_variant"] == "scene_configuration_v1"
    assert manifest["retry_cap"] == 0
    normalization = manifest["input_usd_normalization"]
    assert normalization["stripped_physics_schemas"] == [
        "CollisionAPI",
        "MeshCollisionAPI",
    ]
    assert normalization["candidate_rigid_body_root_transform_identity"] is True
    local_bounds = normalization["candidate_local_bounds_m"]
    assert local_bounds["minimum"][2] == pytest.approx(0.0, abs=1e-7)
    assert (
        local_bounds["minimum"][0] + local_bounds["maximum"][0]
    ) / 2.0 == pytest.approx(0.0, abs=1e-7)
    assert (
        local_bounds["minimum"][1] + local_bounds["maximum"][1]
    ) / 2.0 == pytest.approx(0.0, abs=1e-7)
    normalized_stage = Usd.Stage.Open(
        str(output / "released_content_agents_runtime/input/source_asset.usda")
    )
    assert normalized_stage is not None
    normalized_asset = normalized_stage.GetDefaultPrim()
    assert str(normalized_asset.GetPath()) == "/Asset"
    assert not UsdGeom.Xformable(normalized_asset).GetOrderedXformOps()
    assert all(
        not prim.HasAPI(UsdPhysics.CollisionAPI)
        and not prim.HasAPI(UsdPhysics.MeshCollisionAPI)
        for prim in normalized_stage.Traverse()
    )


def test_metric_envelope_refuses_wrong_size_candidate() -> None:
    envelope = _metric_envelope_spec(
        {
            "metric_envelope": {
                "minimum_xyz_m": [2.9, -6.9, 0.75],
                "maximum_xyz_m": [3.0, -6.7, 0.85],
                "maximum_dimension_relative_error": 0.05,
            }
        }
    )
    with pytest.raises(
        TaskEvaluationSceneConfigurationContentAgentsError,
        match="scene_configuration_content_agents_metric_envelope_mismatch",
    ):
        _validate_metric_envelope_dimensions(
            envelope=envelope,
            observed_dimensions=[0.1, 0.25, 0.1],
        )
