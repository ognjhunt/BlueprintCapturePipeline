import json
from pathlib import Path
import shutil

import pytest

from blueprint_pipeline.common import write_json
from blueprint_pipeline.local_reconstruction_adapters import _sha256_file
from blueprint_pipeline.website_native_background import prepare_collision_stage
from blueprint_pipeline.website_scene_runtime_inputs import prepare_website_runtime_inputs
from blueprint_pipeline.website_task_preparation import compile_website_scene_preparation
from blueprint_pipeline.task_evaluation_scene_configuration_adapters import SceneConfigurationAdapterRegistry
from blueprint_pipeline.task_evaluation_scene_configuration_builtin_adapters import builtin_scene_configuration_adapter_handlers
from blueprint_pipeline.task_evaluation_scene_configuration_content_agents_driver import _reference_frames, _dependency_candidate
from blueprint_pipeline.task_evaluation_scene_configuration_stage_configuration import validate_immutable_stage_configurations
from tests.test_website_task_preparation import _arguments


def staged(tmp_path):
    control = tmp_path / "control"
    control.mkdir()
    args = _arguments(control)
    preparation = compile_website_scene_preparation(**args)
    runtime = prepare_website_runtime_inputs(preparation=preparation, base_scene=args["base_scene"],
        source_geometry=args["source_geometry"], task_masks=args["task_masks"], output_root=control / "native")
    inputs = prepare_collision_stage(control / "native/runtime_inputs.json")
    worker = tmp_path / "worker"
    worker.mkdir()
    rows = []
    for i, row in enumerate(inputs["references"]):
        path = worker / f"transport-{i}"
        shutil.copyfile(row["path"], path)
        rows.append({"contract_path": row["contract_path"], "materialized_path": str(path),
                     "digest": row["digest"], "size_bytes": row["size_bytes"],
                     "full_byte_service_account_readback_passed": True})
    envelope = {"materialized_references": rows, "recipe": {"stage_sequence": [inputs["stage"]],
        "subject_identity": runtime["object_authoring"]["configuration"]["replacement_identity"]}}
    config_path = worker / "configuration.json"
    write_json(config_path, inputs["configuration"])
    # The worker must consume only transported bytes, with no accidental access
    # to the website/control-plane machine's original paths.
    shutil.rmtree(control)
    return runtime, inputs, envelope, config_path


def execute(tmp_path, inputs, envelope, config_path):
    registry = SceneConfigurationAdapterRegistry(builtin_scene_configuration_adapter_handlers())
    validate_immutable_stage_configurations(envelope=envelope, configurations={"stage-2": inputs["configuration"]})
    return registry.execute(stage=inputs["stage"], envelope=envelope, configuration=inputs["configuration"],
        configuration_path=config_path, dependency_results=(), output_root=tmp_path / "result")


def test_native_collision_stage_preserves_background_and_transports_original_references(tmp_path):
    runtime, inputs, envelope, config_path = staged(tmp_path)
    result = execute(tmp_path, inputs, envelope, config_path)
    assert result["status"] == "completed"
    assert result["provider_mutations_performed"] == 0
    artifacts = {row["role"]: row for row in result["output_artifacts"]}
    collision = artifacts["configured_collision_without_source_object"]
    assert _sha256_file(Path(collision["path"])) == runtime["collision"]["digest"]
    config = runtime["object_authoring"]["configuration"]
    frames = _reference_frames({"configuration": config}, [result])
    assert frames and all(frame.is_relative_to(tmp_path / "result") for frame in frames)
    record, candidate = _dependency_candidate([result])
    assert record["digest"] == runtime["object_authoring"]["source_candidate"]["digest"]
    assert candidate.is_file()
    receipt = json.loads(Path(artifacts["website_background_reuse_receipt"]["path"]).read_text())
    assert receipt["source_prim_excision_performed"] is False
    assert receipt["background_bytes_unchanged"] is True
    assert receipt["physical_measurement_proven"] is False


@pytest.mark.parametrize("changed", ["collision", "frame", "subject", "runtime"])
def test_native_collision_stage_refuses_changed_transport_or_task(tmp_path, changed):
    runtime, inputs, envelope, config_path = staged(tmp_path)
    if changed == "subject":
        envelope["recipe"]["subject_identity"] = {"id": "different-task-object", "version": "v1"}
    else:
        suffix = {"collision": "scene.geometry.collision", "frame": ".frames.0", "runtime": ".runtime_inputs"}[changed]
        row = next(r for r in envelope["materialized_references"] if r["contract_path"].endswith(suffix))
        Path(row["materialized_path"]).write_bytes(b"changed")
    with pytest.raises((ValueError, RuntimeError), match="invalid|mismatch"):
        execute(tmp_path, inputs, envelope, config_path)
