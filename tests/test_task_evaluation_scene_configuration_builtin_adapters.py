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
from blueprint_pipeline.task_evaluation_scene_configuration_adapters import (
    TaskEvaluationSceneConfigurationAdapterError,
)
from blueprint_pipeline.task_evaluation_scene_configuration_static_qualification import (
    _is_exact_package_member,
    _usd_findings,
)
from blueprint_pipeline.task_evaluation_scene_configuration_disclosure import (
    MATERIALIZED_STATUS,
)
from blueprint_pipeline.task_evaluation_scene_configuration_render_handoff import (
    materialize_provider_render_handoff,
)


pytest.importorskip("pxr")
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, UsdUtils  # noqa: E402


PHYSICS_BOUNDS = {
    "mass_kg": [0.2, 0.8],
    "static_friction": [0.3, 0.9],
    "dynamic_friction": [0.2, 0.8],
    "restitution": [0.0, 0.15],
}


def sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def artifact(role: str, path: Path) -> dict[str, object]:
    return {
        "role": role,
        "path": str(path),
        "digest": sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _portable_rigid_asset(
    path: Path,
    *,
    dynamic_triangle_mesh: bool = False,
    embedded_texture: bool = False,
    body_translation: tuple[float, float, float] | None = None,
    bind_physics_material: bool = True,
    unowned_collision: bool = False,
) -> None:
    dependency_path = path.with_suffix(".body.usda")
    dependency = Usd.Stage.CreateNew(str(dependency_path))
    body = UsdGeom.Xform.Define(dependency, "/Body").GetPrim()
    dependency.SetDefaultPrim(body)
    UsdPhysics.RigidBodyAPI.Apply(body)
    mass = UsdPhysics.MassAPI.Apply(body)
    mass.CreateMassAttr(0.5)
    mass.CreateCenterOfMassAttr(Gf.Vec3f(0.0, 0.0, 0.0))
    mass.CreateDiagonalInertiaAttr(Gf.Vec3f(0.1, 0.1, 0.1))
    if dynamic_triangle_mesh:
        collider = UsdGeom.Mesh.Define(dependency, "/Body/Collider")
        collider.CreatePointsAttr(
            [
                Gf.Vec3f(-0.05, -0.05, -0.05),
                Gf.Vec3f(0.05, -0.05, -0.05),
                Gf.Vec3f(0.0, 0.05, -0.05),
                Gf.Vec3f(0.0, 0.0, 0.05),
            ]
        )
        collider.CreateFaceVertexCountsAttr([3, 3, 3, 3])
        collider.CreateFaceVertexIndicesAttr([0, 2, 1, 0, 1, 3, 1, 2, 3, 2, 0, 3])
    else:
        collider = UsdGeom.Cube.Define(dependency, "/Body/Collider")
        collider.CreateSizeAttr(0.1)
    UsdPhysics.CollisionAPI.Apply(collider.GetPrim())
    material = UsdShade.Material.Define(dependency, "/Body/PhysicsMaterial")
    physics_material = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    physics_material.CreateStaticFrictionAttr(0.5)
    physics_material.CreateDynamicFrictionAttr(0.4)
    physics_material.CreateRestitutionAttr(0.1)
    if bind_physics_material:
        UsdShade.MaterialBindingAPI.Apply(collider.GetPrim()).Bind(
            material, UsdShade.Tokens.weakerThanDescendants, "physics"
        )
    if embedded_texture:
        texture_path = path.with_suffix(".png")
        texture_path.write_bytes(b"digest-bound-packaged-texture")
        shader = UsdShade.Shader.Define(dependency, "/Body/Looks/Texture")
        shader.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(str(texture_path))
    dependency.GetRootLayer().Save()

    source = path.with_suffix(".usda")
    stage = Usd.Stage.CreateNew(str(source))
    root = UsdGeom.Xform.Define(stage, "/Asset").GetPrim()
    stage.SetDefaultPrim(root)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    referenced_body = stage.DefinePrim("/Asset/Body", "Xform")
    referenced_body.GetReferences().AddReference(
        str(dependency_path), "/Body"
    )
    if unowned_collision:
        outside = UsdGeom.Cube.Define(stage, "/Asset/UnownedCollision").GetPrim()
        UsdPhysics.CollisionAPI.Apply(outside)
        UsdShade.MaterialBindingAPI.Apply(outside).Bind(
            UsdShade.Material(stage.GetPrimAtPath("/Asset/Body/PhysicsMaterial")),
            UsdShade.Tokens.weakerThanDescendants,
            "physics",
        )
    if body_translation is not None:
        UsdGeom.Xformable(referenced_body).AddTranslateOp().Set(Gf.Vec3d(*body_translation))
    stage.GetRootLayer().Save()
    assert UsdUtils.CreateNewUsdzPackage(Sdf.AssetPath(str(source)), str(path))


def test_static_gate_rejects_valid_coefficients_on_an_unbound_material(tmp_path: Path) -> None:
    path = tmp_path / "unbound.usdz"
    _portable_rigid_asset(path, bind_physics_material=False)
    findings, observed = _usd_findings(path, physics_bounds=PHYSICS_BOUNDS)
    assert observed["physics_materials"]
    assert "replacement_collision_physics_material_unbound" in findings


def test_static_gate_rejects_collider_outside_the_only_rigid_body(tmp_path: Path) -> None:
    path = tmp_path / "unowned.usdz"
    _portable_rigid_asset(path, unowned_collision=True)
    findings, _ = _usd_findings(path, physics_bounds=PHYSICS_BOUNDS)
    assert "replacement_collision_rigid_body_owner_invalid" in findings
    assert "replacement_collision_physics_material_unbound" not in findings


def _physics_completion() -> dict[str, object]:
    completion: dict[str, object] = {
        "schema_version": "task_evaluation_rigid_candidate_physics_completion.v1",
        "status": "bounded_candidate_completed",
        "rigid_body_path": "/Asset/Body",
        "collision_prim_paths": ["/Asset/Body/Collider"],
        "collision_bounds_body_frame_m": {
            "minimum": [-0.05, -0.05, -0.05],
            "maximum": [0.05, 0.05, 0.05],
        },
        "collision_dimensions_m": [0.1, 0.1, 0.1],
        "physics_bounds": PHYSICS_BOUNDS,
        "mass_kg": 0.5,
        "center_of_mass_m": [0.0, 0.0, 0.0],
        "diagonal_inertia_kg_m2": [0.1, 0.1, 0.1],
        "physics_materials": [
            {
                "path": "/Asset/Body/PhysicsMaterial",
                "static_friction": 0.5,
                "dynamic_friction": 0.4,
                "restitution": 0.1,
            }
        ],
        "modifications": [],
        "candidate_prior_only": True,
        "physical_truth_claimed": False,
        "completion_digest": "",
    }
    completion["completion_digest"] = canonical_digest(
        completion, digest_field="completion_digest"
    )
    return completion


def test_static_gate_rejects_dynamic_triangle_mesh_collision(tmp_path: Path) -> None:
    asset = tmp_path / "dynamic-triangle-mesh.usdz"
    _portable_rigid_asset(asset, dynamic_triangle_mesh=True)

    findings, observed = _usd_findings(asset, physics_bounds=PHYSICS_BOUNDS)

    assert "replacement_dynamic_mesh_collision_approximation_invalid" in findings
    assert observed["dynamic_mesh_collision_approximations"] == [
        {"path": "/Asset/Body/Collider", "approximation": ""}
    ]
    assert observed["collision_bounds_asset_root_m"]["minimum"] == pytest.approx(
        [-0.05, -0.05, -0.05]
    )
    assert observed["collision_bounds_asset_root_m"]["maximum"] == pytest.approx(
        [0.05, 0.05, 0.05]
    )
    assert observed["collision_bounds_body_frame_m"]["minimum"] == pytest.approx(
        [-0.05, -0.05, -0.05]
    )
    assert observed["collision_bounds_body_frame_m"]["maximum"] == pytest.approx(
        [0.05, 0.05, 0.05]
    )
    assert observed["collision_dimensions_m"] == pytest.approx([0.1, 0.1, 0.1])


def test_static_gate_accepts_asset_embedded_in_exact_usdz_package(
    tmp_path: Path,
) -> None:
    asset = tmp_path / "packaged-texture.usdz"
    _portable_rigid_asset(asset, embedded_texture=True)

    findings, observed = _usd_findings(asset, physics_bounds=PHYSICS_BOUNDS)

    assert "replacement_external_or_unresolved_dependency" not in findings
    assert observed["external_asset_count"] == 0
    assert observed["embedded_package_asset_count"] == 1


def test_static_gate_retains_asset_and_rigid_body_collision_frames(
    tmp_path: Path,
) -> None:
    asset = tmp_path / "translated-rigid-body.usdz"
    _portable_rigid_asset(asset, body_translation=(0.25, -0.1, 0.2))

    findings, observed = _usd_findings(asset, physics_bounds=PHYSICS_BOUNDS)

    assert findings == []
    assert observed["collision_bounds_asset_root_m"]["minimum"] == pytest.approx(
        [0.20, -0.15, 0.15]
    )
    assert observed["collision_bounds_asset_root_m"]["maximum"] == pytest.approx(
        [0.30, -0.05, 0.25]
    )
    assert observed["collision_bounds_body_frame_m"]["minimum"] == pytest.approx(
        [-0.05, -0.05, -0.05]
    )
    assert observed["collision_bounds_body_frame_m"]["maximum"] == pytest.approx(
        [0.05, 0.05, 0.05]
    )


def test_static_gate_keeps_external_asset_identifier_fail_closed(
    tmp_path: Path,
) -> None:
    package = tmp_path / "replacement.usdz"
    external = tmp_path / "outside.png"

    assert not _is_exact_package_member(
        external,
        package_identifier=str(package),
    )
    assert not _is_exact_package_member(
        f"{package}[../outside.png]",
        package_identifier=str(package),
    )


def test_artifixer_handler_admits_only_qualified_generated_appearance(
    tmp_path: Path,
) -> None:
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    appearance = runtime / "configured-appearance.usdc"
    appearance.write_bytes(b"generated-appearance")
    reference = runtime / "reference.png"
    reference.write_bytes(b"digest-bound-reference")
    render_inputs = {
        "status": MATERIALIZED_STATUS,
        "derived_frames": [
            {
                "camera_id": "camera-0",
                "path": str(reference),
                "digest": sha256(reference),
                "size_bytes": reference.stat().st_size,
            }
        ],
        "derived_frame_count": 1,
        "render_completed_on_provider": False,
        "result_digest": "",
    }
    render_inputs["result_digest"] = canonical_digest(
        render_inputs, digest_field="result_digest"
    )
    control_plane_result_digest = render_inputs["result_digest"]
    disclosure_decision = {
        "schema_version": "task_evaluation_scene_configuration_disclosure_decision.v1",
        "render_execution_site": "provider_gpu",
        "source_appearance_bytes_to_provider": True,
        "rights_admission_permits_upload": True,
        "stage_configuration_requests_upload": True,
        "human_authority_accepts_provider_terms": True,
        "refusals": [],
        "provider_training_authorized": False,
        "public_redistribution_authorized": False,
        "decision_digest": "",
    }
    disclosure_decision["decision_digest"] = canonical_digest(
        disclosure_decision, digest_field="decision_digest"
    )
    render_inputs.update(
        {
            "control_plane_result_digest": control_plane_result_digest,
            "disclosure_decision": disclosure_decision,
            "render_completed_on_provider": True,
        }
    )
    render_inputs["result_digest"] = canonical_digest(
        render_inputs, digest_field="result_digest"
    )
    assert render_inputs["result_digest"] != control_plane_result_digest
    render_handoff = materialize_provider_render_handoff(
        render_inputs=render_inputs,
        output_root=runtime,
    )
    thumbnail_path = runtime / "configured-task-thumbnail.png"
    thumbnail_path.write_bytes(b"exact-selected-render-frame")
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
        "task_thumbnail_is_exact_review_frame": True,
        "task_thumbnail_selection": {
            "camera_id": "camera-03",
            "frame_sha256": sha256(thumbnail_path),
            "rationale": "Upright task view.",
        },
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
        "provider_disclosure": {
            "raw_interiorgs_bytes": True,
            "provider_training": False,
            "public_redistribution": False,
        },
        "output_requirements": {"generated_pixels_labeled": True},
    }
    configuration_path = tmp_path / "appearance-configuration.json"
    configuration_path.write_text(json.dumps(configuration), encoding="utf-8")
    output = tmp_path / "appearance-output"
    output.mkdir()

    result = execute_artifixer3d_observed_object_removal(
        envelope={"render_inputs_result": render_inputs},
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
            artifact("configured_task_thumbnail", thumbnail_path),
            render_handoff,
        ),
    )

    assert {row["role"] for row in result["output_artifacts"]} == {
        "configured_appearance_without_source_object",
        "appearance_removal_receipt",
        "appearance_visual_review_receipt",
        "configured_task_thumbnail",
        "provider_render_reference_manifest",
    }
    assert result["provider_mutations_performed"] == 0

    handoff_path = Path(str(render_handoff["path"]))
    valid_handoff_bytes = handoff_path.read_bytes()
    bad_handoff_manifest = json.loads(valid_handoff_bytes)
    bad_handoff_manifest["control_plane_render_result_digest"] = (
        "sha256:" + "0" * 64
    )
    bad_handoff_manifest["manifest_digest"] = canonical_digest(
        bad_handoff_manifest, digest_field="manifest_digest"
    )
    handoff_path.write_text(json.dumps(bad_handoff_manifest), encoding="utf-8")
    bad_render_handoff = artifact(
        "provider_render_reference_manifest", handoff_path
    )
    invalid_output = tmp_path / "appearance-output-invalid-handoff"
    invalid_output.mkdir()
    with pytest.raises(
        TaskEvaluationSceneConfigurationAdapterError,
        match=(
            r"^artifixer3d_object_removal_result_invalid:"
            r"handoff_control_plane_digest$"
        ),
    ):
        execute_artifixer3d_observed_object_removal(
            envelope={"render_inputs_result": render_inputs},
            stage={
                "stage_id": "stage-1",
                "capability": "observed_appearance_object_removal",
                "execution_class": "gpu_canary",
            },
            configuration=configuration,
            configuration_path=configuration_path,
            dependency_results=(),
            output_root=invalid_output,
            provider_runtime_artifacts=(
                artifact(
                    "configured_appearance_without_source_object", appearance
                ),
                artifact("appearance_removal_receipt", receipt_path),
                artifact("appearance_visual_review_receipt", review_path),
                artifact("configured_task_thumbnail", thumbnail_path),
                bad_render_handoff,
            ),
        )
    handoff_path.write_bytes(valid_handoff_bytes)

    pause_review = {
        "schema_version": "task_evaluation_artifixer_visual_review_pause_receipt.v1",
        "status": "visual_review_paused_ungraded",
        "decision": "not_reviewed",
        "visual_review_mode": "paused_ungraded",
        "publisher_instance_id": "104",
        "review_frame_count": 8,
        "frames": [
            {"camera_id": f"camera-{index}", "frame_sha256": sha256(thumbnail_path)}
            for index in range(8)
        ],
        "all_review_frames_digest_bound": True,
        "ai_visual_review_completed": False,
        "human_review_completed": False,
        "semantic_object_absence_review_passed": False,
        "multiview_consistency_review_passed": False,
        "task_thumbnail_is_exact_review_frame": False,
        "task_thumbnail_is_exact_rendered_frame": True,
        "task_thumbnail_selection": {
            "camera_id": "camera-0",
            "frame_sha256": sha256(thumbnail_path),
            "rationale": "Deterministic ungraded thumbnail.",
        },
        "selector": {
            "kind": "system",
            "identity": "deterministic_ungraded_thumbnail_selector",
            "runtime": "blueprint_pipeline",
            "model": "none",
        },
        "review_provider_call_performed": False,
        "generated_output_is_capture_or_physical_evidence": False,
        "warning_label": "Visual review paused - appearance ungraded",
        "receipt_digest": "",
    }
    pause_review["receipt_digest"] = canonical_digest(
        pause_review, digest_field="receipt_digest"
    )
    review_path.write_text(json.dumps(pause_review), encoding="utf-8")
    pause_removal = {
        "schema_version": "task_evaluation_artifixer_object_removal_result.v1",
        "status": "completed_ungraded_generated_appearance_edit",
        "visual_review_mode": "paused_ungraded",
        "publisher_instance_id": "104",
        "raw_interiorgs_bytes_sent_to_external_provider": False,
        "visual_review_receipt_digest": pause_review["receipt_digest"],
        "visual_review_receipt_sha256": sha256(review_path),
        "semantic_object_free_visual_review_passed": False,
        "multiview_consistency_review_passed": False,
        "review_provider_call_performed": False,
        "ungraded_publication_acknowledged": True,
        "warning_label": "Visual review paused - appearance ungraded",
        "generated_pixels_labeled": True,
        "result_digest": "",
    }
    pause_removal["result_digest"] = canonical_digest(
        pause_removal, digest_field="result_digest"
    )
    receipt_path.write_text(json.dumps(pause_removal), encoding="utf-8")
    paused_output = tmp_path / "paused-appearance-output"
    paused_output.mkdir()
    paused_result = execute_artifixer3d_observed_object_removal(
        envelope={
            "render_inputs_result": render_inputs,
            "request": {
                "appearance_review_override": {
                    "mode": "paused_ungraded",
                    "scope": "artifixer_appearance_only",
                    "ungraded_publication_acknowledged": True,
                    "review_provider_call_permitted": False,
                    "warning_label": "Visual review paused - appearance ungraded",
                }
            },
        },
        stage={
            "stage_id": "stage-1",
            "capability": "observed_appearance_object_removal",
            "execution_class": "gpu_canary",
        },
        configuration=configuration,
        configuration_path=configuration_path,
        dependency_results=(),
        output_root=paused_output,
        provider_runtime_artifacts=(
            artifact("configured_appearance_without_source_object", appearance),
            artifact("appearance_removal_receipt", receipt_path),
            artifact("appearance_visual_review_receipt", review_path),
            artifact("configured_task_thumbnail", thumbnail_path),
            render_handoff,
        ),
    )
    assert paused_result["status"] == "completed"


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
    asset = runtime / "mug.usdz"
    _portable_rigid_asset(asset)
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
        "candidate_physics_completion": _physics_completion(),
        "physics_authority_granted": False,
        "result_digest": "",
    }
    receipt["result_digest"] = canonical_digest(
        receipt, digest_field="result_digest"
    )
    receipt_path = runtime / "authoring.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    graph = {
        "schema_version": "task_evaluation_rigid_replacement_graph.v1",
        "asset_id": "replacement-mug",
        "asset_version": "v1",
        "articulation_graph": {"joints": []},
        "single_rigid_candidate": True,
        "physics_bounds": PHYSICS_BOUNDS,
        "physics_authority_granted": False,
    }
    graph_path = runtime / "graph.json"
    graph_path.write_text(json.dumps(graph), encoding="utf-8")
    configuration = {
        "schema_version": "rigid_replacement_authoring_configuration.v1",
        "replacement_identity": identity,
        "required_output": {
            "format": "OpenUSD",
            "rigid_body": True,
            "single_movable_root": True,
            "units": "meters",
            "up_axis": "Z",
            "mass_kg_bounds": PHYSICS_BOUNDS["mass_kg"],
            "static_friction_bounds": PHYSICS_BOUNDS["static_friction"],
            "dynamic_friction_bounds": PHYSICS_BOUNDS["dynamic_friction"],
            "restitution_bounds": PHYSICS_BOUNDS["restitution"],
        },
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
        "physics_bounds": PHYSICS_BOUNDS,
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
        "candidate_physics_completion": _physics_completion(),
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


# --- supplemental passive destination (tray) inside the same stage chain -----

from blueprint_pipeline.task_evaluation_scene_configuration_static_qualification import (  # noqa: E402
    qualify_scene_configuration_rigid_asset_static,
)


DESTINATION_IDENTITY = {"id": "document-tray", "version": "v1"}
DESTINATION_PHYSICS_BOUNDS = {
    "mass_kg": [0.5, 1.0],
    "static_friction": [0.4, 0.8],
    "dynamic_friction": [0.3, 0.7],
    "restitution": [0.0, 0.1],
}


def _tray_asset(path: Path, *, wall_height: float = 0.04) -> None:
    outer_x, outer_y, base, wall = 0.33, 0.48, 0.005, 0.005
    source = path.with_suffix(".usda")
    stage = Usd.Stage.CreateNew(str(source))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    asset = UsdGeom.Xform.Define(stage, "/Asset").GetPrim()
    stage.SetDefaultPrim(asset)
    UsdPhysics.RigidBodyAPI.Apply(asset)
    mass = UsdPhysics.MassAPI.Apply(asset)
    mass.CreateMassAttr(0.75)
    mass.CreateCenterOfMassAttr(Gf.Vec3f(0.0, 0.0, 0.0046))
    mass.CreateDiagonalInertiaAttr(Gf.Vec3f(0.0144, 0.0068, 0.0212))
    physics_prim = stage.DefinePrim("/Asset/Materials/Physics", "Material")
    physics = UsdPhysics.MaterialAPI.Apply(physics_prim)
    physics.CreateStaticFrictionAttr(0.6)
    physics.CreateDynamicFrictionAttr(0.45)
    physics.CreateRestitutionAttr(0.05)
    boxes = {
        "/Asset/Colliders/Bottom": ((outer_x, outer_y, base), (0.0, 0.0, base / 2.0)),
        "/Asset/Colliders/Left": ((wall, outer_y, wall_height), (-(outer_x - wall) / 2.0, 0.0, base + wall_height / 2.0)),
        "/Asset/Colliders/Right": ((wall, outer_y, wall_height), ((outer_x - wall) / 2.0, 0.0, base + wall_height / 2.0)),
        "/Asset/Colliders/Front": ((outer_x - 2 * wall, wall, wall_height), (0.0, -(outer_y - wall) / 2.0, base + wall_height / 2.0)),
        "/Asset/Colliders/Back": ((outer_x - 2 * wall, wall, wall_height), (0.0, (outer_y - wall) / 2.0, base + wall_height / 2.0)),
    }
    for prim_path, (size, center) in boxes.items():
        cube = UsdGeom.Cube.Define(stage, prim_path)
        cube.CreateSizeAttr(1.0)
        xform = UsdGeom.Xformable(cube.GetPrim())
        xform.AddTranslateOp().Set(Gf.Vec3d(*center))
        xform.AddScaleOp().Set(Gf.Vec3d(*size))
        cube.CreatePurposeAttr(UsdGeom.Tokens.guide)
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(
            UsdShade.Material(physics_prim),
            UsdShade.Tokens.weakerThanDescendants,
            "physics",
        )
    stage.GetRootLayer().Save()
    assert UsdUtils.CreateNewUsdzPackage(Sdf.AssetPath(str(source)), str(path))


def _json_file(path: Path, value: dict) -> Path:
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return path


def supplemental_destination_inputs(root: Path, *, wall_height: float = 0.04) -> dict:
    """Author a tray the way the production SimReady materializer does."""

    root.mkdir(parents=True, exist_ok=True)
    asset = root / "passive_destination_simready.usdz"
    _tray_asset(asset, wall_height=wall_height)
    _findings, observed = _usd_findings(asset, physics_bounds=DESTINATION_PHYSICS_BOUNDS)
    assert _findings == []
    completion = {
        "schema_version": "task_evaluation_rigid_candidate_physics_completion.v1",
        "status": "bounded_candidate_completed",
        "physics_bounds": DESTINATION_PHYSICS_BOUNDS,
        "candidate_prior_only": True,
        "physical_truth_claimed": False,
        "mass_kg": observed["mass_kg"],
        "center_of_mass_m": observed["center_of_mass_m"],
        "diagonal_inertia_kg_m2": observed["diagonal_inertia_kg_m2"],
        "collision_bounds_body_frame_m": observed["collision_bounds_body_frame_m"],
        "collision_dimensions_m": observed["collision_dimensions_m"],
        "collision_prim_paths": observed["collision_prim_paths"],
        "physics_materials": observed["physics_materials"],
        "completion_digest": "",
    }
    completion["completion_digest"] = canonical_digest(
        completion, digest_field="completion_digest"
    )
    authoring = {
        "schema_version": "task_evaluation_rigid_replacement_authoring_result.v1",
        "status": "authored_candidate_pending_qualification",
        "replacement_identity": DESTINATION_IDENTITY,
        "physics_authority_granted": False,
        "output_usd": {"sha256": sha256(asset), "size_bytes": asset.stat().st_size},
        "candidate_physics_completion": completion,
        "result_digest": "",
    }
    authoring["result_digest"] = canonical_digest(authoring, digest_field="result_digest")
    authoring_path = _json_file(root / "passive_destination_authoring_receipt.v1.json", authoring)
    graph = {
        "schema_version": "task_evaluation_rigid_replacement_graph.v1",
        "asset_id": DESTINATION_IDENTITY["id"],
        "asset_version": DESTINATION_IDENTITY["version"],
        "articulation_graph": {"joints": []},
        "single_rigid_candidate": True,
        "physics_bounds": DESTINATION_PHYSICS_BOUNDS,
        "physics_authority_granted": False,
    }
    static_path = root / "passive_destination_static_qualification.v1.json"
    qualify_scene_configuration_rigid_asset_static(
        asset_path=asset,
        graph_spec=graph,
        authoring_receipt=authoring,
        replacement_identity=DESTINATION_IDENTITY,
        output_path=static_path,
    )
    rights = {
        "schema_version": "task_evaluation_rigid_destination_rights_admission.v1",
        "status": "admitted",
        "destination_identity": DESTINATION_IDENTITY,
        "license_identifier": "Blueprint-generated-development-asset",
        "private_provider_processing_allowed": True,
        "provider_training_allowed": False,
        "public_redistribution_allowed": False,
        "rights_admission_digest": "",
    }
    rights["rights_admission_digest"] = canonical_digest(
        rights, digest_field="rights_admission_digest"
    )
    rights_path = _json_file(root / "passive_destination_rights_admission.v1.json", rights)

    def record(path: Path) -> dict:
        return {"path": str(path), "sha256": sha256(path), "size_bytes": path.stat().st_size}

    simready = {
        "schema_version": "task_evaluation_passive_destination_simready.v1",
        "status": "static_qualified_pending_native_import_and_placement",
        "destination_identity": DESTINATION_IDENTITY,
        "asset": record(asset),
        "authoring_receipt": record(authoring_path),
        "static_qualification": record(static_path),
        "rights_admission": record(rights_path),
        "static_result_digest": json.loads(static_path.read_text())["result_digest"],
        "intended_support_prim_paths": ["/Asset"],
        "intended_support_collision_prim_paths": ["/Asset/Colliders/Bottom"],
        "interior_bounds_body_frame_m": {
            "minimum": [-0.16, -0.235, 0.005],
            "maximum": [0.16, 0.235, 0.005 + wall_height],
        },
        "native_import_qualified": False,
        "placement_qualified": False,
        "result_digest": "",
    }
    simready["result_digest"] = canonical_digest(simready, digest_field="result_digest")
    simready_path = _json_file(root / "passive_destination_simready_result.v1.json", simready)

    def reference(path: Path) -> dict:
        return {
            "uri": f"s3://blueprint-production-inputs/destination/{path.name}",
            "digest": sha256(path),
            "size_bytes": path.stat().st_size,
        }

    def materialized(contract_path: str, path: Path) -> dict:
        return {
            "contract_path": contract_path,
            **reference(path),
            "materialized_path": str(path),
            "full_byte_service_account_readback_passed": True,
        }

    return {
        "asset": asset,
        "authoring_receipt": authoring_path,
        "static_qualification": static_path,
        "rights_admission": rights_path,
        "simready_result": simready_path,
        "recipe_supplemental_destination": {
            "identity": DESTINATION_IDENTITY,
            "relation": "inside",
            "asset": reference(asset),
            "static_qualification": reference(static_path),
            "rights_admission": reference(rights_path),
            "authoring_receipt": reference(authoring_path),
            "simready_result": reference(simready_path),
        },
        "materialized_references": [
            materialized("task.destination.asset", asset),
            materialized("task.destination.static_qualification", static_path),
            materialized("task.destination.rights_admission", rights_path),
            materialized(
                "construction.recipe.supplemental_destination.authoring_receipt",
                authoring_path,
            ),
            materialized(
                "construction.recipe.supplemental_destination.simready_result",
                simready_path,
            ),
        ],
    }


def _subject_stage3_dependency(tmp_path: Path) -> tuple[tuple[dict, ...], dict, Path]:
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
        "physics_bounds": PHYSICS_BOUNDS,
        "physics_authority_granted": False,
    }
    graph_spec_path = _json_file(dependency / "graph-spec.json", graph_spec)
    authoring = {
        "schema_version": "task_evaluation_rigid_replacement_authoring_result.v1",
        "status": "authored_candidate_pending_qualification",
        "replacement_identity": {"id": "replacement-mug", "version": "v1"},
        "output_usd": {
            "path": str(asset),
            "sha256": sha256(asset),
            "size_bytes": asset.stat().st_size,
        },
        "candidate_physics_completion": _physics_completion(),
        "physics_authority_granted": False,
        "result_digest": "",
    }
    authoring["result_digest"] = canonical_digest(authoring, digest_field="result_digest")
    authoring_path = _json_file(dependency / "authoring.json", authoring)
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
    configuration_path = _json_file(tmp_path / "static.json", configuration)
    dependency_results = (
        {
            "output_artifacts": [
                artifact("replacement_asset", asset),
                artifact("replacement_authoring_receipt", authoring_path),
                artifact("replacement_graph_spec", graph_spec_path),
            ]
        },
    )
    return dependency_results, configuration, configuration_path


def test_static_handler_requalifies_the_supplemental_destination_from_exact_bytes(
    tmp_path: Path,
) -> None:
    dependency_results, configuration, configuration_path = _subject_stage3_dependency(tmp_path)
    destination = supplemental_destination_inputs(tmp_path / "destination")
    output = tmp_path / "output"
    output.mkdir()

    result = execute_simready_static_rigid_qualification(
        envelope={
            "recipe": {
                "subject_identity": {"id": "replacement-mug", "version": "v1"},
                "supplemental_destination": destination["recipe_supplemental_destination"],
            },
            "materialized_references": destination["materialized_references"],
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

    rows = {row["role"]: row for row in result["output_artifacts"]}
    assert set(rows) == {
        "statically_qualified_replacement_asset",
        "static_qualification_receipt",
        "statically_qualified_destination_asset",
        "destination_static_qualification_receipt",
        "destination_static_requalification_receipt",
    }
    assert rows["statically_qualified_destination_asset"]["digest"] == sha256(
        destination["asset"]
    )
    # The published receipt is the exact request-declared byte string ...
    assert rows["destination_static_qualification_receipt"]["digest"] == sha256(
        destination["static_qualification"]
    )
    # ... and the run independently re-derived the same qualification.
    requalified = json.loads(
        Path(rows["destination_static_requalification_receipt"]["path"]).read_text()
    )
    declared = json.loads(destination["static_qualification"].read_text())
    assert requalified["observed_structure"] == declared["observed_structure"]
    assert requalified["replacement_identity"] == DESTINATION_IDENTITY
    assert requalified["replacement_usd"]["sha256"] == declared["replacement_usd"]["sha256"]


def test_static_handler_refuses_a_destination_whose_declared_receipt_drifts(
    tmp_path: Path,
) -> None:
    dependency_results, configuration, configuration_path = _subject_stage3_dependency(tmp_path)
    destination = supplemental_destination_inputs(tmp_path / "destination")
    # Drift the declared static receipt *consistently*: every digest join in
    # the SimReady result, recipe, and materialized rows agrees with the drifted
    # bytes, so only the run's own re-derivation can notice the lie.
    declared = json.loads(destination["static_qualification"].read_text())
    declared["observed_structure"]["mass_kg"] = 0.9
    declared["result_digest"] = canonical_digest(declared, digest_field="result_digest")
    destination["static_qualification"].write_text(json.dumps(declared, sort_keys=True))
    simready = json.loads(destination["simready_result"].read_text())
    simready["static_qualification"]["sha256"] = sha256(destination["static_qualification"])
    simready["static_qualification"]["size_bytes"] = (
        destination["static_qualification"].stat().st_size
    )
    simready["static_result_digest"] = declared["result_digest"]
    simready["result_digest"] = canonical_digest(simready, digest_field="result_digest")
    destination["simready_result"].write_text(json.dumps(simready, sort_keys=True))
    rebind = {
        "task.destination.static_qualification": destination["static_qualification"],
        "construction.recipe.supplemental_destination.simready_result": destination[
            "simready_result"
        ],
    }
    for row in destination["materialized_references"]:
        drifted = rebind.get(row["contract_path"])
        if drifted is not None:
            row["digest"] = sha256(drifted)
            row["size_bytes"] = drifted.stat().st_size
    for name, drifted in (
        ("static_qualification", destination["static_qualification"]),
        ("simready_result", destination["simready_result"]),
    ):
        destination["recipe_supplemental_destination"][name] = {
            "uri": destination["recipe_supplemental_destination"][name]["uri"],
            "digest": sha256(drifted),
            "size_bytes": drifted.stat().st_size,
        }
    output = tmp_path / "output"
    output.mkdir()
    with pytest.raises(
        TaskEvaluationSceneConfigurationAdapterError,
        match="simready_static_destination_requalification_mismatch",
    ):
        execute_simready_static_rigid_qualification(
            envelope={
                "recipe": {
                    "subject_identity": {"id": "replacement-mug", "version": "v1"},
                    "supplemental_destination": destination["recipe_supplemental_destination"],
                },
                "materialized_references": destination["materialized_references"],
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


def _native_runtime_result(*, identity: dict, asset: Path, static_receipt: Path) -> dict:
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
    runtime["result_digest"] = canonical_digest(runtime, digest_field="result_digest")
    return runtime


def _native_stage_inputs(tmp_path: Path) -> dict:
    dependency = tmp_path / "dependency-native"
    dependency.mkdir()
    asset = dependency / "replacement.usda"
    asset.write_text("#usda 1.0\n", encoding="utf-8")
    static_receipt = dependency / "static.json"
    static_receipt.write_text('{"status":"qualified"}\n', encoding="utf-8")
    tray = dependency / "tray.usdz"
    tray.write_bytes(b"PK-tray")
    tray_static = dependency / "tray-static.json"
    tray_static.write_text('{"status":"qualified","replacement_identity":"tray"}\n')
    identity = {"id": "replacement-mug", "version": "v1"}
    runtime_path = _json_file(
        tmp_path / "native-runtime.json",
        _native_runtime_result(identity=identity, asset=asset, static_receipt=static_receipt),
    )
    destination_runtime_path = _json_file(
        tmp_path / "destination-native-runtime.json",
        _native_runtime_result(
            identity=DESTINATION_IDENTITY, asset=tray, static_receipt=tray_static
        ),
    )
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
    return {
        "identity": identity,
        "asset": asset,
        "static_receipt": static_receipt,
        "tray": tray,
        "tray_static": tray_static,
        "runtime_path": runtime_path,
        "destination_runtime_path": destination_runtime_path,
        "configuration": configuration,
        "configuration_path": _json_file(tmp_path / "native-configuration.json", configuration),
    }


def _run_native_stage(tmp_path: Path, inputs: dict, *, destination_runtime: Path | None):
    output = tmp_path / "native-output"
    output.mkdir()
    provider_artifacts = [artifact("native_import_runtime_result", inputs["runtime_path"])]
    if destination_runtime is not None:
        provider_artifacts.append(
            artifact("destination_native_import_runtime_result", destination_runtime)
        )
    return execute_simready_native_import_qualification(
        envelope={
            "recipe": {
                "subject_identity": inputs["identity"],
                "supplemental_destination": {
                    "identity": DESTINATION_IDENTITY,
                    "relation": "inside",
                },
            }
        },
        stage={
            "stage_id": "stage-5",
            "capability": "replacement_native_import_qualification",
            "execution_class": "gpu_canary",
        },
        configuration=inputs["configuration"],
        configuration_path=inputs["configuration_path"],
        dependency_results=(
            {
                "output_artifacts": [
                    artifact("statically_qualified_replacement_asset", inputs["asset"]),
                    artifact("static_qualification_receipt", inputs["static_receipt"]),
                    artifact("statically_qualified_destination_asset", inputs["tray"]),
                    artifact("destination_static_qualification_receipt", inputs["tray_static"]),
                ]
            },
        ),
        output_root=output,
        provider_runtime_artifacts=tuple(provider_artifacts),
    )


def test_native_import_handler_promotes_the_destination_alongside_the_subject(
    tmp_path: Path,
) -> None:
    inputs = _native_stage_inputs(tmp_path)
    result = _run_native_stage(
        tmp_path, inputs, destination_runtime=inputs["destination_runtime_path"]
    )
    rows = {row["role"]: row for row in result["output_artifacts"]}
    assert set(rows) == {
        "native_qualified_replacement_asset",
        "native_import_qualification_receipt",
        "native_qualified_destination_asset",
        "destination_native_import_qualification_receipt",
    }
    assert rows["native_qualified_destination_asset"]["digest"] == sha256(inputs["tray"])
    assert rows["destination_native_import_qualification_receipt"]["digest"] == sha256(
        inputs["destination_runtime_path"]
    )


def test_native_import_handler_refuses_a_destination_result_bound_to_another_asset(
    tmp_path: Path,
) -> None:
    inputs = _native_stage_inputs(tmp_path)
    drifted = _json_file(
        tmp_path / "drifted-destination-runtime.json",
        _native_runtime_result(
            identity=DESTINATION_IDENTITY,
            asset=inputs["asset"],  # the subject's bytes, not the tray's
            static_receipt=inputs["tray_static"],
        ),
    )
    with pytest.raises(
        TaskEvaluationSceneConfigurationAdapterError,
        match="simready_native_import_destination_result_invalid",
    ):
        _run_native_stage(tmp_path, inputs, destination_runtime=drifted)


def test_native_import_handler_refuses_a_declared_destination_without_its_result(
    tmp_path: Path,
) -> None:
    inputs = _native_stage_inputs(tmp_path)
    with pytest.raises(
        TaskEvaluationSceneConfigurationAdapterError,
        match="scene_configuration_provider_runtime_artifact_missing:destination_native_import_runtime_result",
    ):
        _run_native_stage(tmp_path, inputs, destination_runtime=None)
