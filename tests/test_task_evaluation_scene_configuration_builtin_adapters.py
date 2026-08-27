from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_evaluation_scene_configuration_builtin_adapters import (
    execute_artifixer3d_observed_object_removal,
    execute_content_agents_rigid_replacement,
    execute_native_task_scene_assembly,
    execute_sage_exact_prim_excision,
    execute_simready_static_rigid_qualification,
    execute_simready_native_import_qualification,
)


pytest.importorskip("pxr")
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, UsdUtils  # noqa: E402


def sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def artifact(role: str, path: Path) -> dict[str, object]:
    return {
        "role": role,
        "path": str(path),
        "digest": sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _portable_rigid_asset(path: Path) -> None:
    dependency_path = path.with_suffix(".body.usda")
    dependency = Usd.Stage.CreateNew(str(dependency_path))
    body = UsdGeom.Xform.Define(dependency, "/Body").GetPrim()
    dependency.SetDefaultPrim(body)
    UsdPhysics.RigidBodyAPI.Apply(body)
    mass = UsdPhysics.MassAPI.Apply(body)
    mass.CreateMassAttr(1.0)
    mass.CreateCenterOfMassAttr(Gf.Vec3f(0.0, 0.0, 0.0))
    mass.CreateDiagonalInertiaAttr(Gf.Vec3f(0.1, 0.1, 0.1))
    collider = UsdGeom.Cube.Define(dependency, "/Body/Collider")
    collider.CreateSizeAttr(0.1)
    UsdPhysics.CollisionAPI.Apply(collider.GetPrim())
    material = UsdShade.Material.Define(dependency, "/Body/PhysicsMaterial")
    physics_material = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    physics_material.CreateStaticFrictionAttr(0.5)
    physics_material.CreateDynamicFrictionAttr(0.4)
    physics_material.CreateRestitutionAttr(0.1)
    dependency.GetRootLayer().Save()

    source = path.with_suffix(".usda")
    stage = Usd.Stage.CreateNew(str(source))
    root = UsdGeom.Xform.Define(stage, "/Asset").GetPrim()
    stage.SetDefaultPrim(root)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    stage.DefinePrim("/Asset/Body", "Xform").GetReferences().AddReference(
        str(dependency_path), "/Body"
    )
    stage.GetRootLayer().Save()
    assert UsdUtils.CreateNewUsdzPackage(Sdf.AssetPath(str(source)), str(path))


def test_artifixer_handler_admits_only_qualified_generated_appearance(
    tmp_path: Path,
) -> None:
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    appearance = runtime / "configured-appearance.usdc"
    appearance.write_bytes(b"generated-appearance")
    review = {
        "schema_version": "task_evaluation_artifixer_ai_visual_review.v1",
        "status": "accepted",
        "publisher_instance_id": "104",
        "decision": "accepted",
        "semantic_object_absence_review_passed": True,
        "multiview_consistency_review_passed": True,
        "review_frame_count": 8,
        "all_review_frames_digest_bound": True,
        "ai_visual_review_completed": True,
        "human_review_completed": False,
        "generated_output_is_capture_or_physical_evidence": False,
        "reviewer": {
            "identity": "artifixer-independent-vision-reviewer-v1",
            "runtime": "openai_agents_sdk",
            "model": "gpt-5.6-terra",
        },
        "receipt_digest": "",
    }
    review["receipt_digest"] = canonical_digest(
        review, digest_field="receipt_digest"
    )
    review_path = runtime / "appearance-review.json"
    review_path.write_text(json.dumps(review), encoding="utf-8")
    receipt = {
        "schema_version": "task_evaluation_artifixer_object_removal_result.v1",
        "status": "qualified_generated_appearance_edit",
        "publisher_instance_id": "104",
        "raw_interiorgs_bytes_sent_to_external_provider": False,
        "visual_review_receipt_digest": review["receipt_digest"],
        "visual_review_receipt_sha256": sha256(review_path),
        "semantic_object_free_visual_review_passed": True,
        "multiview_consistency_review_passed": True,
        "generated_pixels_labeled": True,
        "result_digest": "",
    }
    receipt["result_digest"] = canonical_digest(
        receipt, digest_field="result_digest"
    )
    receipt_path = runtime / "appearance-receipt.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    configuration = {
        "schema_version": "observed_appearance_object_removal_configuration.v1",
        "source_object": {"publisher_instance_id": "104"},
        "production_render_required": True,
        "required_views": {"minimum": 8},
        "provider_disclosure": {"raw_interiorgs_bytes": False},
        "output_requirements": {"generated_pixels_labeled": True},
    }
    configuration_path = tmp_path / "appearance-configuration.json"
    configuration_path.write_text(json.dumps(configuration), encoding="utf-8")
    output = tmp_path / "appearance-output"
    output.mkdir()

    result = execute_artifixer3d_observed_object_removal(
        envelope={},
        stage={
            "stage_id": "stage-1",
            "capability": "observed_appearance_object_removal",
            "execution_class": "gpu_canary",
        },
        configuration=configuration,
        configuration_path=configuration_path,
        dependency_results=(),
        output_root=output,
        provider_runtime_artifacts=(
            artifact("configured_appearance_without_source_object", appearance),
            artifact("appearance_removal_receipt", receipt_path),
            artifact("appearance_visual_review_receipt", review_path),
        ),
    )

    assert {row["role"] for row in result["output_artifacts"]} == {
        "configured_appearance_without_source_object",
        "appearance_removal_receipt",
        "appearance_visual_review_receipt",
    }
    assert result["provider_mutations_performed"] == 0


def test_artifixer_handler_rejects_unbound_review_boolean_only_receipt(
    tmp_path: Path,
) -> None:
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    appearance = runtime / "configured-appearance.usdc"
    appearance.write_bytes(b"generated-appearance")
    receipt = {
        "schema_version": "task_evaluation_artifixer_object_removal_result.v1",
        "status": "qualified_generated_appearance_edit",
        "publisher_instance_id": "104",
        "raw_interiorgs_bytes_sent_to_external_provider": False,
        "semantic_object_free_visual_review_passed": True,
        "multiview_consistency_review_passed": True,
        "generated_pixels_labeled": True,
        "result_digest": "",
    }
    receipt["result_digest"] = canonical_digest(
        receipt, digest_field="result_digest"
    )
    receipt_path = runtime / "appearance-receipt.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    output = tmp_path / "appearance-output"
    output.mkdir()

    with pytest.raises(
        RuntimeError,
        match="scene_configuration_provider_runtime_artifact_missing:appearance_visual_review_receipt",
    ):
        execute_artifixer3d_observed_object_removal(
            envelope={},
            stage={
                "stage_id": "stage-1",
                "capability": "observed_appearance_object_removal",
                "execution_class": "gpu_canary",
            },
            configuration={
                "schema_version": "observed_appearance_object_removal_configuration.v1",
                "source_object": {"publisher_instance_id": "104"},
                "production_render_required": True,
                "required_views": {"minimum": 8},
                "provider_disclosure": {"raw_interiorgs_bytes": False},
                "output_requirements": {"generated_pixels_labeled": True},
            },
            configuration_path=receipt_path,
            dependency_results=(),
            output_root=output,
            provider_runtime_artifacts=(
                artifact("configured_appearance_without_source_object", appearance),
                artifact("appearance_removal_receipt", receipt_path),
            ),
        )


def test_content_agents_handler_retains_candidate_for_independent_checks(
    tmp_path: Path,
) -> None:
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    asset = runtime / "mug.usda"
    asset.write_text("#usda 1.0\n", encoding="utf-8")
    source_candidate = runtime / "source-candidate.usda"
    source_candidate.write_text("#usda 1.0\n", encoding="utf-8")
    identity = {"id": "replacement-mug", "version": "v1"}
    receipt = {
        "schema_version": "task_evaluation_rigid_replacement_authoring_result.v1",
        "status": "authored_candidate_pending_qualification",
        "replacement_identity": identity,
        "source_candidate_digest": sha256(source_candidate),
        "source_candidate_claim": (
            "sage_candidate_geometry_not_observed_truth_or_physics_authority"
        ),
        "output_usd": {
            "sha256": sha256(asset),
            "size_bytes": asset.stat().st_size,
        },
        "result_digest": "",
    }
    receipt["result_digest"] = canonical_digest(
        receipt, digest_field="result_digest"
    )
    receipt_path = runtime / "authoring.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    graph = {"asset_id": "replacement-mug", "articulation_graph": {"joints": []}}
    graph_path = runtime / "graph.json"
    graph_path.write_text(json.dumps(graph), encoding="utf-8")
    configuration = {
        "schema_version": "rigid_replacement_authoring_configuration.v1",
        "replacement_identity": identity,
        "required_output": {"rigid_body": True, "single_movable_root": True},
        "physics_authority_granted_by_authoring": False,
    }
    configuration_path = tmp_path / "authoring-configuration.json"
    configuration_path.write_text(json.dumps(configuration), encoding="utf-8")
    output = tmp_path / "authoring-output"
    output.mkdir()

    result = execute_content_agents_rigid_replacement(
        envelope={"recipe": {"subject_identity": identity}},
        stage={
            "stage_id": "stage-3",
            "capability": "rigid_replacement_authoring",
            "execution_class": "gpu_canary",
        },
        configuration=configuration,
        configuration_path=configuration_path,
        dependency_results=(
            {},
            {
                "output_artifacts": [
                    artifact("source_object_candidate_mesh", source_candidate)
                ]
            },
        ),
        output_root=output,
        provider_runtime_artifacts=(
            artifact("replacement_asset", asset),
            artifact("replacement_authoring_receipt", receipt_path),
            artifact("replacement_graph_spec", graph_path),
        ),
    )

    assert {row["role"] for row in result["output_artifacts"]} == {
        "replacement_asset",
        "replacement_authoring_receipt",
        "replacement_graph_spec",
    }


def test_sage_exact_prim_excision_removes_only_requested_prim(tmp_path: Path) -> None:
    source = tmp_path / "source.usda"
    stage = Usd.Stage.CreateNew(str(source))
    stage.DefinePrim("/Root", "Xform")
    stage.DefinePrim("/Root/Target", "Xform")
    stage.DefinePrim("/Root/Support", "Xform")
    stage.GetRootLayer().Save()
    source_digest = sha256(source)
    configuration = {
        "schema_version": "collision_object_excision_configuration.v1",
        "collision_source_digest": source_digest,
        "exact_target_prim": "/Root/Target",
        "expected_target": {"point_count": 1, "face_count": 1},
        "operation": "deactivate_exact_prim_only",
        "validation": {
            "target_absent_after_excision": True,
            "all_non_target_prim_digests_unchanged": True,
            "stage_units_and_up_axis_unchanged": True,
            "before_and_after_prim_manifests_required": True,
        },
    }
    configuration_path = tmp_path / "configuration.json"
    configuration_path.write_text(json.dumps(configuration), encoding="utf-8")
    output = tmp_path / "output"
    output.mkdir()
    envelope = {
        "recipe": {"subject_identity": {"id": "source-mug", "version": "v1"}},
        "materialized_references": [
            {
                "contract_path": "scene.geometry.collision",
                "materialized_path": str(source),
                "digest": source_digest,
                "size_bytes": source.stat().st_size,
                "full_byte_service_account_readback_passed": True,
            }
        ],
    }
    result = execute_sage_exact_prim_excision(
        envelope=envelope,
        stage={
            "stage_id": "stage-2",
            "capability": "collision_object_excision",
            "execution_class": "no_spend",
        },
        configuration=configuration,
        configuration_path=configuration_path,
        dependency_results=({"status": "completed"},),
        output_root=output,
    )
    assert result["stage_result_digest"] == canonical_digest(
        result, digest_field="stage_result_digest"
    )
    removed = Usd.Stage.Open(
        next(
            row["path"]
            for row in result["output_artifacts"]
            if row["role"] == "configured_collision_without_source_object"
        )
    )
    assert not removed.GetPrimAtPath("/Root/Target").IsValid()
    assert removed.GetPrimAtPath("/Root/Support").IsValid()
    source_candidate = next(
        Path(row["path"])
        for row in result["output_artifacts"]
        if row["role"] == "source_object_candidate_mesh"
    )
    candidate = Usd.Stage.Open(str(source_candidate))
    assert candidate.GetPrimAtPath("/Root/SourceObjectCandidate").IsValid()
    candidate_row = next(
        row
        for row in result["output_artifacts"]
        if row["role"] == "source_object_candidate_mesh"
    )
    assert candidate_row["observed_source_truth"] is False
    assert candidate_row["movable_physics_authority"] is False


def test_static_handler_requires_exact_stage3_asset_spec_and_receipt(
    tmp_path: Path,
) -> None:
    dependency = tmp_path / "dependency"
    dependency.mkdir()
    asset = dependency / "mug.usdz"
    _portable_rigid_asset(asset)
    graph_spec = {
        "schema_version": "task_evaluation_rigid_replacement_graph.v1",
        "asset_id": "replacement-mug",
        "asset_version": "v1",
        "articulation_graph": {"joints": []},
        "single_rigid_candidate": True,
        "physics_authority_granted": False,
    }
    graph_spec_path = dependency / "graph-spec.json"
    graph_spec_path.write_text(json.dumps(graph_spec), encoding="utf-8")
    authoring = {
        "schema_version": "task_evaluation_rigid_replacement_authoring_result.v1",
        "status": "authored_candidate_pending_qualification",
        "replacement_identity": {"id": "replacement-mug", "version": "v1"},
        "output_usd": {
            "path": str(asset),
            "sha256": sha256(asset),
            "size_bytes": asset.stat().st_size,
        },
        "physics_authority_granted": False,
        "result_digest": "",
    }
    authoring["result_digest"] = canonical_digest(
        authoring, digest_field="result_digest"
    )
    authoring_path = dependency / "authoring.json"
    authoring_path.write_text(json.dumps(authoring), encoding="utf-8")

    def artifact(role: str, path: Path) -> dict[str, object]:
        return {
            "role": role,
            "path": str(path),
            "digest": sha256(path),
            "size_bytes": path.stat().st_size,
        }

    dependency_results = (
        {
            "output_artifacts": [
                artifact("replacement_asset", asset),
                artifact("replacement_authoring_receipt", authoring_path),
                artifact("replacement_graph_spec", graph_spec_path),
            ]
        },
    )
    output = tmp_path / "output"
    output.mkdir()
    configuration = {
        "schema_version": "replacement_static_qualification_configuration.v1",
        "replacement_identity": {"id": "replacement-mug", "version": "v1"},
        "required_checks": {
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
        },
        "center_of_mass_must_lie_inside_collision_bounds": True,
    }
    configuration_path = tmp_path / "static.json"
    configuration_path.write_text(json.dumps(configuration), encoding="utf-8")

    result = execute_simready_static_rigid_qualification(
        envelope={
            "recipe": {
                "subject_identity": {"id": "replacement-mug", "version": "v1"}
            }
        },
        stage={
            "stage_id": "stage-4",
            "capability": "replacement_static_qualification",
            "execution_class": "no_spend",
        },
        configuration=configuration,
        configuration_path=configuration_path,
        dependency_results=dependency_results,
        output_root=output,
    )
    roles = {row["role"] for row in result["output_artifacts"]}
    assert roles == {
        "statically_qualified_replacement_asset",
        "static_qualification_receipt",
    }
    assert result["provider_mutations_performed"] == 0
    qualification = json.loads(
        (output / "static_qualification_receipt.v1.json").read_text()
    )
    assert qualification["schema_version"] == (
        "task_evaluation_rigid_replacement_static_qualification.v1"
    )
    assert qualification["checks"]["no_external_unpinned_dependencies"] is True


def test_native_import_handler_promotes_only_exact_static_asset(
    tmp_path: Path,
) -> None:
    dependency = tmp_path / "dependency-native"
    dependency.mkdir()
    asset = dependency / "replacement.usda"
    asset.write_text("#usda 1.0\n", encoding="utf-8")
    static_receipt = dependency / "static.json"
    static_receipt.write_text('{"status":"qualified"}\n', encoding="utf-8")
    identity = {"id": "replacement-mug", "version": "v1"}
    runtime = {
        "schema_version": "task_evaluation_replacement_native_import_result.v1",
        "status": "qualified",
        "replacement_identity": identity,
        "asset_digest": sha256(asset),
        "static_qualification_digest": sha256(static_receipt),
        "native_isaac_executed": True,
        "native_simulator_import_qualified": True,
        "support_contact_observed": True,
        "deterministic_reset_state_digest_repeat_count": 3,
        "blockers": [],
        "result_digest": "",
    }
    runtime["result_digest"] = canonical_digest(
        runtime, digest_field="result_digest"
    )
    runtime_path = tmp_path / "native-runtime.json"
    runtime_path.write_text(json.dumps(runtime), encoding="utf-8")
    configuration = {
        "schema_version": "replacement_native_import_qualification_configuration.v1",
        "replacement_identity": identity,
        "required_checks": {
            "stage_import": True,
            "rigid_body_enabled": True,
            "collider_enabled": True,
            "gravity_settle_seconds": 3.0,
            "maximum_settle_translation_m": 0.01,
            "maximum_settle_rotation_rad": 0.08,
            "support_contact_required": True,
            "explosion_or_tunneling_forbidden": True,
            "deterministic_reset_required": True,
            "state_digest_repeat_count": 3,
        },
    }
    configuration_path = tmp_path / "native-configuration.json"
    configuration_path.write_text(json.dumps(configuration), encoding="utf-8")
    output = tmp_path / "native-output"
    output.mkdir()

    result = execute_simready_native_import_qualification(
        envelope={"recipe": {"subject_identity": identity}},
        stage={
            "stage_id": "stage-5",
            "capability": "replacement_native_import_qualification",
            "execution_class": "gpu_canary",
        },
        configuration=configuration,
        configuration_path=configuration_path,
        dependency_results=(
            {
                "output_artifacts": [
                    artifact("statically_qualified_replacement_asset", asset),
                    artifact("static_qualification_receipt", static_receipt),
                ]
            },
        ),
        output_root=output,
        provider_runtime_artifacts=(
            artifact("native_import_runtime_result", runtime_path),
        ),
    )

    assert {row["role"] for row in result["output_artifacts"]} == {
        "native_qualified_replacement_asset",
        "native_import_qualification_receipt",
    }
    assert result["provider_mutations_performed"] == 0


def test_scene_assembly_emits_robot_neutral_candidate_for_control_plane_publication(
    tmp_path: Path,
) -> None:
    dependencies = tmp_path / "dependencies"
    dependencies.mkdir()
    appearance = dependencies / "appearance.usdc"
    appearance.write_bytes(b"appearance")
    collision = dependencies / "collision.usda"
    collision.write_bytes(b"collision")
    replacement = dependencies / "replacement.usda"
    replacement.write_bytes(b"replacement")
    native_receipt = dependencies / "native.json"
    native_receipt.write_text('{"qualified":true}\n', encoding="utf-8")

    def artifact(role: str, path: Path) -> dict[str, object]:
        return {
            "role": role,
            "path": str(path),
            "digest": sha256(path),
            "size_bytes": path.stat().st_size,
        }

    dependency_results = (
        {
            "output_artifacts": [
                artifact(
                    "configured_appearance_without_source_object", appearance
                )
            ]
        },
        {
            "output_artifacts": [
                artifact(
                    "configured_collision_without_source_object", collision
                )
            ]
        },
        {"output_artifacts": []},
        {"output_artifacts": []},
        {
            "output_artifacts": [
                artifact("native_qualified_replacement_asset", replacement),
                artifact("native_import_qualification_receipt", native_receipt),
            ]
        },
    )
    configuration = {
        "schema_version": "task_evaluation_scene_assembly_configuration.v1",
        "scene_identity": {"id": "interiorgs-839873", "version": "mug-v1"},
        "replacement": {
            "qualified_asset_from_stage": "stage-5",
            "source_and_replacement_visual_instances_must_not_coexist": True,
            "source_and_replacement_collision_instances_must_not_coexist": True,
        },
        "robot_mount_interface": {
            "publish_robot_neutral_scene_mount_frame": True,
            "robot_specific_base_transform_and_reachability_deferred_to_each_evaluation": True,
        },
        "evaluation_episode_executed_in_this_run": False,
        "scene_construction_repeated_per_evaluation": False,
    }
    configuration_path = tmp_path / "assembly.json"
    configuration_path.write_text(json.dumps(configuration), encoding="utf-8")
    output = tmp_path / "output"
    output.mkdir()

    result = execute_native_task_scene_assembly(
        envelope={
            "run_id": "scene-839873-configuration-v1",
            "team_namespace": "blueprint-adp",
            "expected_production_commit": "a" * 40,
            "recipe": {
                "scene_identity": configuration["scene_identity"],
                "task_identity": {
                    "id": "scene-839873-mug-planar-push",
                    "version": "v1",
                },
                "subject_identity": {
                    "id": "scene-839873-mug-replacement",
                    "version": "v1",
                },
            },
        },
        stage={
            "stage_id": "stage-6",
            "capability": "scene_assembly",
            "execution_class": "no_spend",
        },
        configuration=configuration,
        configuration_path=configuration_path,
        dependency_results=dependency_results,
        output_root=output,
    )

    manifest_path = next(
        Path(row["path"])
        for row in result["output_artifacts"]
        if row["role"] == "configured_scene_bundle_candidate_manifest"
    )
    manifest = json.loads(manifest_path.read_text())
    assert manifest["robot_neutral"] is True
    assert manifest["robot_specific_base_registration_included"] is False
    assert manifest["evaluation_episode_executed"] is False
    assert result["provider_mutations_performed"] == 0
