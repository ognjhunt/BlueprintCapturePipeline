from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline import task_evaluation_scene_configuration_builtin_adapters as adapters
from blueprint_pipeline.task_evaluation_scene_configuration_artifixer_driver import (
    _diagnostic_rejection_permitted,
    _materialize_diagnostic_rejected_artifixer_artifacts,
)
from blueprint_pipeline.task_evaluation_scene_configuration_builtin_producers import (
    builtin_scene_configuration_stage_producer_registry,
)
from blueprint_pipeline.task_evaluation_scene_configuration_diagnostic_checkpoint import (
    TaskEvaluationSceneConfigurationDiagnosticCheckpointError,
    _validate_rejected_appearance_stage_result,
)
from blueprint_pipeline.task_evaluation_scene_configuration_stage_producers import (
    ADMITTED_PRODUCER_IDENTITIES,
    PRODUCTION_RESULT_SCHEMA_VERSION,
)
from blueprint_pipeline.task_evaluation_scene_configuration_stage_tool import (
    COMPONENT_RESULT_SCHEMA_VERSION,
    TaskEvaluationSceneConfigurationStageToolError,
    _validate_component_result,
)
from tests.test_task_evaluation_scene_configuration_builtin_producers import _toolchain


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact(role: str, path: Path) -> dict[str, object]:
    return {
        "role": role,
        "path": str(path),
        "digest": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _fixture(tmp_path: Path) -> dict[str, object]:
    frames: list[dict[str, object]] = []
    review_frames: list[dict[str, object]] = []
    for index in range(8):
        path = tmp_path / f"frame-{index}.png"
        path.write_bytes(f"frame-{index}".encode())
        decision = "rejected" if index == 0 else "accepted"
        frames.append(
            {
                "task_id": "remove-source-object-104",
                "camera_id": f"camera-{index}",
                "frame_sha256": _sha256(path),
                "orientation_is_upright": True,
                "source_object_absent": True,
                "repair_is_locally_plausible": decision == "accepted",
                "preserves_non_target_content": decision == "accepted",
                "decision": decision,
                "rationale": "table texture changed" if index == 0 else "accepted",
            }
        )
        review_frames.append(
            {
                "camera_id": f"camera-{index}",
                "final_frame": {"path": str(path), "sha256": _sha256(path)},
            }
        )
    execution: dict[str, object] = {
        "schema_version": "task_evaluation_artifixer_ai_visual_review_execution.v1",
        "status": "completed",
        "configuration_run_id": "diagnostic-run",
        "publisher_instance_id": "104",
        "task_id": "remove-source-object-104",
        "decision": "rejected",
        "review_frame_count": 8,
        "reviewer": {
            "kind": "ai",
            "identity": "artifixer-independent-vision-reviewer-v1",
            "runtime": "openai_agents_sdk",
            "model": "gpt-5.6-terra",
        },
        "frames": frames,
        "task_thumbnail": {
            "camera_id": "camera-1",
            "frame_sha256": frames[1]["frame_sha256"],
            "rationale": "clear accepted view",
        },
        "provider_called": True,
        "provider": "openai",
        "response_store": False,
        "tracing_disabled": True,
        "raw_secret_values_recorded": False,
        "generated_output_is_capture_or_physical_evidence": False,
        "physics_or_collision_authority_granted": False,
        "execution_digest": "",
    }
    execution["execution_digest"] = canonical_digest(
        execution, digest_field="execution_digest"
    )
    execution_path = tmp_path / "review-execution.json"
    execution_path.write_text(json.dumps(execution), encoding="utf-8")
    appearance = tmp_path / "candidate.usdz"
    appearance.write_bytes(b"diagnostic-rejected-appearance")
    render_reference = tmp_path / "render-reference.json"
    render_reference.write_text("{}\n", encoding="utf-8")
    configuration = {
        "schema_version": "observed_appearance_object_removal_configuration.v1",
        "source_object": {"publisher_instance_id": "104"},
        "production_render_required": True,
        "required_views": {"minimum": 8},
        "provider_disclosure": {
            "raw_interiorgs_bytes": False,
            "provider_training": False,
            "public_redistribution": False,
        },
        "output_requirements": {"generated_pixels_labeled": True},
    }
    configuration_path = tmp_path / "configuration.json"
    configuration_path.write_text(json.dumps(configuration), encoding="utf-8")
    review = {
        "decision": "rejected",
        "review_receipt": None,
        "execution_receipt": {
            "path": str(execution_path),
            "sha256": _sha256(execution_path),
            "size_bytes": execution_path.stat().st_size,
            "execution_digest": execution["execution_digest"],
        },
    }
    return {
        "review": review,
        "review_frames": review_frames,
        "appearance": appearance,
        "render_reference": render_reference,
        "configuration": configuration,
        "configuration_path": configuration_path,
    }


def _materialize(tmp_path: Path) -> tuple[dict[str, object], tuple[dict, ...]]:
    fixture = _fixture(tmp_path)
    output = tmp_path / "driver-output"
    output.mkdir()
    render_reference = output / "provider-render-reference.json"
    render_reference.write_bytes(Path(fixture["render_reference"]).read_bytes())
    artifacts = _materialize_diagnostic_rejected_artifixer_artifacts(
        review=fixture["review"],
        review_frames=fixture["review_frames"],
        native_appearance_source=fixture["appearance"],
        configuration=fixture["configuration"],
        output_root=output,
        render_handoff=_artifact(
            "provider_render_reference_manifest", render_reference
        ),
        source_diagnostic_checkpoint_digest="sha256:" + "1" * 64,
        post_training_binding_digest="sha256:" + "2" * 64,
    )
    return fixture, tuple(artifacts)


def test_seven_of_eight_review_emits_only_nonqualifying_diagnostic_roles(
    tmp_path: Path,
) -> None:
    _fixture_value, artifacts = _materialize(tmp_path)
    roles = {row["role"] for row in artifacts}
    receipt_path = Path(
        next(row["path"] for row in artifacts if row["role"] == "appearance_rejection_receipt")
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))

    assert roles == {
        "diagnostic_rejected_appearance_candidate",
        "appearance_rejection_receipt",
        "appearance_visual_review_execution",
        "provider_render_reference_manifest",
    }
    assert "configured_appearance_without_source_object" not in roles
    assert "configured_task_thumbnail" not in roles
    assert receipt["accepted_review_frame_count"] == 7
    assert receipt["rejected_review_frame_count"] == 1
    assert receipt["qualification_eligible"] is False
    assert receipt["offering_publication_permitted"] is False

    assert _diagnostic_rejection_permitted(
        stage_input={"execution_mode": "diagnostic_only"},
        environment={"BLUEPRINT_SCENE_CONFIGURATION_DIAGNOSTIC_ONLY": "1"},
    ) is True
    assert _diagnostic_rejection_permitted(
        stage_input={"execution_mode": "production"},
        environment={"BLUEPRINT_SCENE_CONFIGURATION_DIAGNOSTIC_ONLY": "1"},
    ) is False
    assert _diagnostic_rejection_permitted(
        stage_input={"execution_mode": "diagnostic_only"}, environment={}
    ) is False

    component = {
        "schema_version": COMPONENT_RESULT_SCHEMA_VERSION,
        "status": "completed",
        "adapter_id": "artifixer3d_observed_object_removal",
        "stage_id": "stage-1",
        "provider_mutations_performed": 0,
        "nested_paid_execution_requested": False,
        "diagnostic_only": True,
        "qualification_eligible": False,
        "configured_revision_publication_permitted": False,
        "offering_publication_permitted": False,
        "terminal_e2e_completion_permitted": False,
        "artifacts": list(artifacts),
        "result_digest": "",
    }
    component["result_digest"] = canonical_digest(
        component, digest_field="result_digest"
    )
    assert len(
        _validate_component_result(
            component,
            adapter_id="artifixer3d_observed_object_removal",
            stage_id="stage-1",
            output_root=receipt_path.parent,
            diagnostic_only=True,
        )
    ) == 4
    with pytest.raises(
        TaskEvaluationSceneConfigurationStageToolError,
        match="scene_configuration_component_artifact_roles_invalid",
    ):
        _validate_component_result(
            component,
            adapter_id="artifixer3d_observed_object_removal",
            stage_id="stage-1",
            output_root=receipt_path.parent,
            diagnostic_only=False,
        )


def test_diagnostic_adapter_continues_but_production_adapter_refuses_same_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture, artifacts = _materialize(tmp_path)
    render_result_digest = "sha256:" + "c" * 64
    monkeypatch.setattr(
        adapters,
        "validate_provider_render_handoff",
        lambda _path: (
            {
                "control_plane_render_result_digest": render_result_digest,
                "render_completed_on_provider": False,
            },
            [],
        ),
    )
    envelope = {
        "render_inputs_result": {
            "result_digest": render_result_digest,
            "disclosure_decision": {},
        }
    }
    stage = {
        "stage_id": "stage-1",
        "capability": "observed_appearance_object_removal",
        "execution_class": "gpu_canary",
    }
    output = tmp_path / "adapter-output"
    output.mkdir()
    result = adapters.execute_artifixer3d_diagnostic_object_removal(
        envelope=envelope,
        stage=stage,
        configuration=fixture["configuration"],
        configuration_path=fixture["configuration_path"],
        dependency_results=(),
        output_root=output,
        provider_runtime_artifacts=artifacts,
    )

    assert result["appearance_visual_review_rejected"] is True
    assert result["qualification_eligible"] is False
    assert result["offering_publication_permitted"] is False
    with pytest.raises(
        RuntimeError,
        match="scene_configuration_provider_runtime_artifact_missing:configured_appearance_without_source_object",
    ):
        adapters.execute_artifixer3d_observed_object_removal(
            envelope=envelope,
            stage=stage,
            configuration=fixture["configuration"],
            configuration_path=fixture["configuration_path"],
            dependency_results=(),
            output_root=tmp_path / "production-output",
            provider_runtime_artifacts=artifacts,
        )

    _validate_rejected_appearance_stage_result(result, stage_id="stage-1")
    changed = dict(result)
    changed["offering_publication_permitted"] = True
    changed["stage_result_digest"] = canonical_digest(
        changed, digest_field="stage_result_digest"
    )
    with pytest.raises(
        TaskEvaluationSceneConfigurationDiagnosticCheckpointError,
        match="scene_configuration_diagnostic_completed_stage_artifact_invalid:stage-1",
    ):
        _validate_rejected_appearance_stage_result(changed, stage_id="stage-1")


def test_stage_six_assembles_rejected_candidate_as_nonpublishable_diagnostic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture, artifacts = _materialize(tmp_path)
    render_result_digest = "sha256:" + "c" * 64
    monkeypatch.setattr(
        adapters,
        "validate_provider_render_handoff",
        lambda _path: (
            {
                "control_plane_render_result_digest": render_result_digest,
                "render_completed_on_provider": False,
            },
            [],
        ),
    )
    stage_one_output = tmp_path / "stage-one-adapter"
    stage_one_output.mkdir()
    stage_one = adapters.execute_artifixer3d_diagnostic_object_removal(
        envelope={
            "render_inputs_result": {
                "result_digest": render_result_digest,
                "disclosure_decision": {},
            }
        },
        stage={
            "stage_id": "stage-1",
            "capability": "observed_appearance_object_removal",
            "execution_class": "gpu_canary",
        },
        configuration=fixture["configuration"],
        configuration_path=fixture["configuration_path"],
        dependency_results=(),
        output_root=stage_one_output,
        provider_runtime_artifacts=artifacts,
    )
    collision = tmp_path / "collision.usda"
    collision.write_bytes(b"collision")
    replacement = tmp_path / "replacement.usdz"
    replacement.write_bytes(b"replacement")
    native_receipt = tmp_path / "native-receipt.json"
    native_receipt.write_text("{}\n", encoding="utf-8")
    later = {
        "status": "completed",
        "output_artifacts": [
            _artifact("configured_collision_without_source_object", collision),
            _artifact("native_qualified_replacement_asset", replacement),
            _artifact("native_import_qualification_receipt", native_receipt),
        ],
    }
    scene_identity = {"id": "interiorgs-839873", "version": "v1"}
    configuration = {
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
    }
    configuration_path = tmp_path / "assembly-configuration.json"
    configuration_path.write_text(json.dumps(configuration), encoding="utf-8")
    output = tmp_path / "assembly-output"
    output.mkdir()
    result = adapters.execute_native_task_scene_diagnostic_assembly(
        envelope={
            "run_id": "diagnostic-run",
            "team_namespace": "blueprint-adp",
            "expected_production_commit": "a" * 40,
            "recipe": {
                "scene_identity": scene_identity,
                "task_identity": {"id": "relocation", "version": "v1"},
                "subject_identity": {"id": "cup", "version": "v1"},
            },
        },
        stage={
            "stage_id": "stage-6",
            "capability": "native_task_scene_assembly",
            "execution_class": "no_spend",
        },
        configuration=configuration,
        configuration_path=configuration_path,
        dependency_results=(stage_one, later),
        output_root=output,
    )
    manifest_path = Path(result["output_artifacts"][0]["path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert result["appearance_visual_review_rejected"] is True
    assert result["offering_publication_permitted"] is False
    assert manifest["status"] == (
        "assembled_diagnostic_with_rejected_appearance_not_publishable"
    )
    assert manifest["configured_revision_publication_permitted"] is False
    _validate_rejected_appearance_stage_result(result, stage_id="stage-6")


def test_diagnostic_producer_accepts_rejection_roles_only_in_diagnostic_mode(
    tmp_path: Path,
) -> None:
    commit = "a" * 40
    toolchain = _toolchain(tmp_path, commit)
    identity = next(
        row
        for row in ADMITTED_PRODUCER_IDENTITIES
        if row.adapter_id == "artifixer3d_observed_object_removal"
    )

    def run(command, *, env, **_kwargs):
        output = Path(env["BLUEPRINT_SCENE_CONFIGURATION_STAGE_OUTPUT_ROOT"])
        artifacts = []
        for role in (
            "diagnostic_rejected_appearance_candidate",
            "appearance_rejection_receipt",
            "appearance_visual_review_execution",
            "provider_render_reference_manifest",
        ):
            path = output / f"{role}.bin"
            path.write_bytes(role.encode())
            artifacts.append(_artifact(role, path))
        result = {
            "schema_version": PRODUCTION_RESULT_SCHEMA_VERSION,
            "status": "completed",
            "stage_id": "stage-1",
            "capability": identity.capability,
            "provider_mutations_performed": 0,
            "paid_execution_requested": False,
            "executed_inside_parent_configuration_run": True,
            "diagnostic_only": True,
            "qualification_eligible": False,
            "configured_revision_publication_permitted": False,
            "offering_publication_permitted": False,
            "terminal_e2e_completion_permitted": False,
            "artifacts": artifacts,
            "production_result_digest": "",
        }
        result["production_result_digest"] = canonical_digest(
            result, digest_field="production_result_digest"
        )
        Path(env["BLUEPRINT_SCENE_CONFIGURATION_STAGE_RESULT"]).write_text(
            json.dumps(result), encoding="utf-8"
        )
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    stage = {
        "stage_id": "stage-1",
        "capability": identity.capability,
        "adapter": {"id": identity.adapter_id, "version": identity.version},
        "execution_class": "gpu_canary",
    }
    configuration_path = tmp_path / "producer-configuration.json"
    configuration_path.write_text("{}\n", encoding="utf-8")
    diagnostic = builtin_scene_configuration_stage_producer_registry(
        expected_source_commit=commit,
        toolchain_root=toolchain,
        runner=run,
        diagnostic_only=True,
    )
    diagnostic_output = tmp_path / "diagnostic-producer-output"
    diagnostic_output.mkdir()
    assert len(
        diagnostic.execute(
            stage=stage,
            envelope={"run_id": "diagnostic-run"},
            configuration={},
            configuration_path=configuration_path,
            dependency_results=(),
            output_root=diagnostic_output,
        )
    ) == 4

    production = builtin_scene_configuration_stage_producer_registry(
        expected_source_commit=commit,
        toolchain_root=toolchain,
        runner=run,
        diagnostic_only=False,
    )
    production_output = tmp_path / "production-producer-output"
    production_output.mkdir()
    with pytest.raises(
        RuntimeError,
        match="scene_configuration_stage_production_artifact_roles_invalid",
    ):
        production.execute(
            stage=stage,
            envelope={"run_id": "production-run"},
            configuration={},
            configuration_path=configuration_path,
            dependency_results=(),
            output_root=production_output,
        )


def test_diagnostic_provider_runner_installs_diagnostic_only_adapters() -> None:
    runner = Path("scripts/task_evaluation_scene_configuration_diagnostic_provider_runner.py")
    source = runner.read_text(encoding="utf-8")
    assert "builtin_scene_configuration_diagnostic_adapter_handlers()" in source
    assert "builtin_scene_configuration_adapter_handlers()" not in source
