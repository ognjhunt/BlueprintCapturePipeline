from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
from pxr import Usd

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_evaluation_scene_configuration_adapters import (
    SceneConfigurationAdapterRegistry,
)
from blueprint_pipeline.task_evaluation_scene_configuration_builtin_adapters import (
    builtin_scene_configuration_adapter_handlers,
)
from blueprint_pipeline.task_evaluation_scene_configuration_builtin_producers import (
    builtin_scene_configuration_stage_producer_registry,
)
from blueprint_pipeline.task_evaluation_scene_configuration_diagnostic_runtime import (
    execute_scene_configuration_diagnostic_stage_chain,
)
from blueprint_pipeline.task_evaluation_scene_configuration_render_handoff import (
    materialize_provider_render_handoff,
)
from blueprint_pipeline.task_evaluation_scene_configuration_stage_producers import (
    PRODUCTION_RESULT_SCHEMA_VERSION,
)
from blueprint_pipeline.task_evaluation_scene_construction_recipe import CAPABILITY_ORDER
from tests.test_task_evaluation_scene_configuration_builtin_adapters import (
    PHYSICS_BOUNDS,
    _physics_completion,
    _portable_rigid_asset,
    artifact,
    sha256,
)
from tests.test_task_evaluation_scene_configuration_builtin_producers import (
    _toolchain,
)


def _write_configuration(root: Path, stage_id: str, value: dict) -> tuple[dict, Path]:
    path = root / f"{stage_id}.json"
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return value, path


def _production_result(stage: dict, artifacts: list[dict]) -> dict:
    value = {
        "schema_version": PRODUCTION_RESULT_SCHEMA_VERSION,
        "status": "completed",
        "stage_id": stage["stage_id"],
        "capability": stage["capability"],
        "provider_mutations_performed": 0,
        "paid_execution_requested": False,
        "executed_inside_parent_configuration_run": True,
        "artifacts": artifacts,
        "production_result_digest": "",
    }
    value["production_result_digest"] = canonical_digest(
        value, digest_field="production_result_digest"
    )
    return value


def test_real_builtin_chain_accepts_checkpoint_hydration_and_skips_paid_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commit = "a" * 40
    source = tmp_path / "collision.usda"
    stage = Usd.Stage.CreateNew(str(source))
    stage.DefinePrim("/Root", "Xform")
    stage.DefinePrim("/Root/Target", "Xform")
    stage.DefinePrim("/Root/Support", "Xform")
    stage.GetRootLayer().Save()
    render_root = tmp_path / "render"
    render_root.mkdir()
    render_frames = []
    for index in range(8):
        frame = render_root / f"camera-{index}.png"
        frame.write_bytes(f"frame-{index}".encode())
        render_frames.append(
            {
                "camera_id": f"camera-{index}",
                "path": str(frame),
                "digest": sha256(frame),
                "size_bytes": frame.stat().st_size,
            }
        )
    render_inputs = {
        "status": "derived_method_inputs_materialized",
        "derived_frames": render_frames,
        "derived_frame_count": 8,
        "render_completed_on_provider": False,
        "result_digest": "",
    }
    render_inputs["result_digest"] = canonical_digest(
        render_inputs, digest_field="result_digest"
    )
    identities = list(builtin_scene_configuration_adapter_handlers())
    stages = []
    for index, (capability, identity) in enumerate(
        zip(CAPABILITY_ORDER, identities, strict=True), start=1
    ):
        stages.append(
            {
                "stage_id": f"stage-{index}",
                "capability": capability,
                "adapter": {"id": identity.adapter_id, "version": identity.version},
                "execution_class": identity.execution_class,
                "depends_on": [] if index == 1 else [f"stage-{index - 1}"],
            }
        )
    subject = {"id": "replacement-mug", "version": "v1"}
    scene_identity = {"id": "interiorgs-839873", "version": "mug-v1"}
    configuration_values = [
        {
            "schema_version": "observed_appearance_object_removal_configuration.v1",
            "source_object": {"publisher_instance_id": "104"},
            "production_render_required": True,
            "required_views": {"minimum": 8},
            "provider_disclosure": {"raw_interiorgs_bytes": False},
            "output_requirements": {"generated_pixels_labeled": True},
        },
        {
            "schema_version": "collision_object_excision_configuration.v1",
            "collision_source_digest": sha256(source),
            "exact_target_prim": "/Root/Target",
            "expected_target": {"point_count": 1, "face_count": 1},
            "operation": "deactivate_exact_prim_only",
            "validation": {
                "target_absent_after_excision": True,
                "all_non_target_prim_digests_unchanged": True,
                "stage_units_and_up_axis_unchanged": True,
                "before_and_after_prim_manifests_required": True,
            },
        },
        {
            "schema_version": "rigid_replacement_authoring_configuration.v1",
            "replacement_identity": subject,
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
        },
        {
            "schema_version": "replacement_static_qualification_configuration.v1",
            "replacement_identity": subject,
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
        },
        {
            "schema_version": "replacement_native_import_qualification_configuration.v1",
            "replacement_identity": subject,
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
        },
        {
            "schema_version": "task_evaluation_scene_assembly_configuration.v1",
            "scene_identity": scene_identity,
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
        },
    ]
    configurations = {
        f"stage-{index}": _write_configuration(
            tmp_path, f"stage-{index}", value
        )
        for index, value in enumerate(configuration_values, start=1)
    }
    envelope = {
        "run_id": "diagnostic-real-builtins",
        "team_namespace": "blueprint-adp",
        "expected_production_commit": commit,
        "recipe": {
            "stage_sequence": stages,
            "subject_identity": subject,
            "scene_identity": scene_identity,
            "task_identity": {"id": "mug-planar-push", "version": "v1"},
        },
        "materialized_references": [
            {
                "contract_path": "scene.geometry.collision",
                "materialized_path": str(source),
                "digest": sha256(source),
                "size_bytes": source.stat().st_size,
                "full_byte_service_account_readback_passed": True,
            }
        ],
        "render_inputs_result": render_inputs,
    }
    checkpoint = {"checkpoint_digest": "sha256:" + "c" * 64}
    monkeypatch.setattr(
        "blueprint_pipeline.task_evaluation_scene_configuration_diagnostic_runtime."
        "diagnostic_checkpoint_scientific_binding_digest",
        lambda **_kwargs: "sha256:" + "d" * 64,
    )
    monkeypatch.setattr(
        "blueprint_pipeline.task_evaluation_scene_configuration_diagnostic_runtime."
        "validate_scene_configuration_diagnostic_checkpoint",
        lambda **_kwargs: checkpoint,
    )
    monkeypatch.setattr(
        "blueprint_pipeline.task_evaluation_scene_configuration_diagnostic_runtime."
        "hydrate_scene_configuration_diagnostic_completed_stages",
        lambda **_kwargs: [],
    )
    checkpoint_root = tmp_path / "checkpoint"
    checkpoint_root.mkdir()
    environment = {
        "BLUEPRINT_SCENE_CONFIGURATION_DIAGNOSTIC_CHECKPOINT_ROOT": str(
            checkpoint_root
        )
    }
    calls: list[str] = []

    def run(command, *, env, **_kwargs):
        stage_input = json.loads(
            Path(env["BLUEPRINT_SCENE_CONFIGURATION_STAGE_INPUT"]).read_text()
        )
        stage_value = stage_input["stage"]
        adapter_id = stage_value["adapter"]["id"]
        calls.append(adapter_id)
        assert env[
            "BLUEPRINT_SCENE_CONFIGURATION_DIAGNOSTIC_CHECKPOINT_ROOT"
        ] == str(checkpoint_root)
        output = Path(env["BLUEPRINT_SCENE_CONFIGURATION_STAGE_OUTPUT_ROOT"])
        artifacts = []
        if adapter_id == "artifixer3d_observed_object_removal":
            appearance = output / "appearance.usdc"
            appearance.write_bytes(b"appearance")
            thumbnail = output / "thumbnail.png"
            thumbnail.write_bytes(b"thumbnail")
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
                    "camera_id": "camera-0",
                    "frame_sha256": sha256(thumbnail),
                    "rationale": "digest-bound diagnostic fixture",
                },
                "reviewer": {"identity": "fixture", "runtime": "fixture", "model": "fixture"},
                "receipt_digest": "",
            }
            review["receipt_digest"] = canonical_digest(review, digest_field="receipt_digest")
            review_path = output / "review.json"
            review_path.write_text(json.dumps(review))
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
            receipt["result_digest"] = canonical_digest(receipt, digest_field="result_digest")
            receipt_path = output / "removal.json"
            receipt_path.write_text(json.dumps(receipt))
            handoff = materialize_provider_render_handoff(
                render_inputs=render_inputs, output_root=output
            )
            artifacts = [
                artifact("configured_appearance_without_source_object", appearance),
                artifact("appearance_removal_receipt", receipt_path),
                artifact("appearance_visual_review_receipt", review_path),
                artifact("configured_task_thumbnail", thumbnail),
                handoff,
            ]
        elif adapter_id == "content_agents_rigid_replacement":
            dependencies = json.loads(
                Path(env["BLUEPRINT_SCENE_CONFIGURATION_STAGE_DEPENDENCIES"]).read_text()
            )
            candidate = next(
                Path(row["path"])
                for result in dependencies
                for row in result.get("output_artifacts", [])
                if row["role"] == "source_object_candidate_mesh"
            )
            asset = output / "replacement.usdz"
            _portable_rigid_asset(asset)
            authoring = {
                "schema_version": "task_evaluation_rigid_replacement_authoring_result.v1",
                "status": "authored_candidate_pending_qualification",
                "replacement_identity": subject,
                "source_candidate_digest": sha256(candidate),
                "source_candidate_claim": "sage_candidate_geometry_not_observed_truth_or_physics_authority",
                "output_usd": {"sha256": sha256(asset), "size_bytes": asset.stat().st_size},
                "candidate_physics_completion": _physics_completion(),
                "physics_authority_granted": False,
                "result_digest": "",
            }
            authoring["result_digest"] = canonical_digest(authoring, digest_field="result_digest")
            authoring_path = output / "authoring.json"
            authoring_path.write_text(json.dumps(authoring))
            graph = {
                "schema_version": "task_evaluation_rigid_replacement_graph.v1",
                "asset_id": subject["id"],
                "asset_version": subject["version"],
                "articulation_graph": {"joints": []},
                "single_rigid_candidate": True,
                "physics_bounds": PHYSICS_BOUNDS,
                "physics_authority_granted": False,
            }
            graph_path = output / "graph.json"
            graph_path.write_text(json.dumps(graph))
            artifacts = [
                artifact("replacement_asset", asset),
                artifact("replacement_authoring_receipt", authoring_path),
                artifact("replacement_graph_spec", graph_path),
            ]
        else:
            dependencies = json.loads(
                Path(env["BLUEPRINT_SCENE_CONFIGURATION_STAGE_DEPENDENCIES"]).read_text()
            )
            asset = next(
                Path(row["path"])
                for result in dependencies
                for row in result.get("output_artifacts", [])
                if row["role"] == "statically_qualified_replacement_asset"
            )
            static_receipt = next(
                Path(row["path"])
                for result in dependencies
                for row in result.get("output_artifacts", [])
                if row["role"] == "static_qualification_receipt"
            )
            runtime = {
                "schema_version": "task_evaluation_replacement_native_import_result.v1",
                "status": "qualified",
                "replacement_identity": subject,
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
            runtime_path = output / "native.json"
            runtime_path.write_text(json.dumps(runtime))
            artifacts = [artifact("native_import_runtime_result", runtime_path)]
        Path(env["BLUEPRINT_SCENE_CONFIGURATION_STAGE_RESULT"]).write_text(
            json.dumps(_production_result(stage_value, artifacts))
        )
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    producers = builtin_scene_configuration_stage_producer_registry(
        expected_source_commit=commit,
        toolchain_root=_toolchain(tmp_path / "toolchain-fixture", commit),
        runner=run,
        environment=environment,
    )
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    result = execute_scene_configuration_diagnostic_stage_chain(
        diagnostic_bootstrap_mode="checkpoint_resume",
        checkpoint_root=checkpoint_root,
        envelope=envelope,
        configurations=configurations,
        output_root=outputs,
        registry=SceneConfigurationAdapterRegistry(
            builtin_scene_configuration_adapter_handlers()
        ),
        producer_registry=producers,
        stage_one_resume_producer=lambda **kwargs: producers.execute(
            **{
                key: value
                for key, value in kwargs.items()
                if key not in {"checkpoint", "checkpoint_root"}
            }
        ),
        parent_deadline_epoch=100_000,
        clock=lambda: 1_000,
    )

    assert result["status"] == "completed_diagnostic_only_not_qualification_eligible"
    assert result["stage_count"] == 6
    assert calls == [
        "artifixer3d_observed_object_removal",
        "content_agents_rigid_replacement",
        "simready_native_import_qualification",
    ]
    assert all(row["diagnostic_only"] is True for row in result["stage_results"])
