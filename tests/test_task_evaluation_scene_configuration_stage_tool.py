from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_evaluation_scene_configuration_builtin_producers import (
    TOOLCHAIN_ROOT_ENV,
)
from blueprint_pipeline.task_evaluation_scene_configuration_stage_tool import (
    COMPONENT_RESULT_SCHEMA_VERSION,
    TaskEvaluationSceneConfigurationStageToolError,
    execute_stage_tool,
)
from scripts.build_task_evaluation_scene_configuration_toolchain import (
    build_published_scene_configuration_toolchain,
)
from tests.test_build_task_evaluation_scene_configuration_toolchain import (
    _component_packages,
)


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _environment(tmp_path: Path, *, adapter_id: str) -> tuple[dict[str, str], Path]:
    commit = "a" * 40
    toolchain = tmp_path / "toolchain"
    publication = build_published_scene_configuration_toolchain(
        source_commit=commit,
        output_root=toolchain,
        readback=lambda path: path.read_bytes(),
        readback_actor="service-account:test",
        component_packages=_component_packages(tmp_path),
    )
    output = tmp_path / "output"
    output.mkdir()
    envelope = {
        "run_id": "configure-scene-generic",
        "expected_production_commit": commit,
        "envelope_digest": "",
    }
    envelope["envelope_digest"] = canonical_digest(
        envelope, digest_field="envelope_digest"
    )
    stage_input = {
        "schema_version": (
            "task_evaluation_scene_configuration_stage_production_input.v1"
        ),
        "run_id": envelope["run_id"],
        "stage": {
            "stage_id": "stage-1",
            "capability": "observed_appearance_object_removal",
            "adapter": {"id": adapter_id, "version": "v1"},
            "execution_class": "gpu_canary",
        },
        "configuration": {"scene_identity": {"id": "any-scene"}},
        "configuration_sha256": "sha256:" + "b" * 64,
        "source_commit": commit,
        "toolchain_digest": publication["toolchain_digest"],
        "construction_envelope": envelope,
    }
    input_path = output / "input.json"
    input_path.write_text(json.dumps(stage_input), encoding="utf-8")
    dependencies = output / "dependencies.json"
    dependencies.write_text("[]", encoding="utf-8")
    result = output / "result.json"
    return {
        "BLUEPRINT_SCENE_CONFIGURATION_STAGE_INPUT": str(input_path),
        "BLUEPRINT_SCENE_CONFIGURATION_STAGE_DEPENDENCIES": str(dependencies),
        "BLUEPRINT_SCENE_CONFIGURATION_STAGE_OUTPUT_ROOT": str(output),
        "BLUEPRINT_SCENE_CONFIGURATION_STAGE_RESULT": str(result),
        TOOLCHAIN_ROOT_ENV: str(toolchain),
    }, result


def test_executes_only_manifest_bound_component_and_seals_artifacts(
    tmp_path: Path,
) -> None:
    adapter_id = "artifixer3d_observed_object_removal"
    environment, result_path = _environment(tmp_path, adapter_id=adapter_id)
    observed: list[list[str]] = []

    def run(command, *, env, **_kwargs):
        observed.append(command)
        output = Path(env["BLUEPRINT_SCENE_CONFIGURATION_STAGE_OUTPUT_ROOT"])
        artifacts = []
        for role in (
            "configured_appearance_without_source_object",
            "appearance_removal_receipt",
            "appearance_visual_review_receipt",
        ):
            path = output / f"{role}.bin"
            path.write_bytes(role.encode())
            artifacts.append(
                {
                    "role": role,
                    "path": str(path),
                    "digest": _sha256(path),
                    "size_bytes": path.stat().st_size,
                }
            )
        component = {
            "schema_version": COMPONENT_RESULT_SCHEMA_VERSION,
            "status": "completed",
            "adapter_id": adapter_id,
            "stage_id": "stage-1",
            "provider_mutations_performed": 0,
            "nested_paid_execution_requested": False,
            "artifacts": artifacts,
            "result_digest": "",
        }
        component["result_digest"] = canonical_digest(
            component, digest_field="result_digest"
        )
        Path(env["BLUEPRINT_SCENE_CONFIGURATION_COMPONENT_RESULT"]).write_text(
            json.dumps(component), encoding="utf-8"
        )
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    result = execute_stage_tool(
        adapter_id=adapter_id,
        environment=environment,
        runner=run,
    )

    toolchain = Path(environment[TOOLCHAIN_ROOT_ENV])
    assert observed == [
        [str(toolchain / "components" / adapter_id / "package" / "run")]
    ]
    assert result["status"] == "completed"
    assert result["provider_mutations_performed"] == 0
    assert result["executed_inside_parent_configuration_run"] is True
    assert json.loads(result_path.read_text()) == result


def test_rejects_scene_selected_adapter_or_artifact_path(tmp_path: Path) -> None:
    environment, _result_path = _environment(
        tmp_path, adapter_id="artifixer3d_observed_object_removal"
    )
    with pytest.raises(
        TaskEvaluationSceneConfigurationStageToolError,
        match="scene_configuration_stage_tool_input_invalid",
    ):
        execute_stage_tool(
            adapter_id="content_agents_rigid_replacement",
            environment=environment,
            runner=lambda *_args, **_kwargs: None,
        )
