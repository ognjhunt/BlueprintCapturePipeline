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
            "configured_task_thumbnail",
            "provider_render_reference_manifest",
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


def test_component_failure_retains_its_redacted_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A paid stage that dies must say why in the run that paid for it.

    Run ``adp-new-scene-simple-relocation-839873-032eaa09-r2-web-20260827T011512Z``
    rented an RTX 4090, reached stage 1, and returned exactly
    ``scene_configuration_component_failed:artifixer3d_observed_object_removal:1``.
    The component's traceback had been captured by ``capture_output=True`` and
    then dropped, so the only way to read it was to rent the GPU again. The
    stage producer retains this process's stderr, so the redacted streams
    written here travel out with the run's own evidence.
    """

    adapter_id = "artifixer3d_observed_object_removal"
    environment, _result_path = _environment(tmp_path, adapter_id=adapter_id)
    opaque_secret = "opaque-file-credential-value-839873"
    secret_path = tmp_path / "openai-key"
    secret_path.write_text(opaque_secret, encoding="utf-8")
    secret_path.chmod(0o600)
    environment["OPENAI_API_KEY_FILE"] = str(secret_path)

    def run(command, **_kwargs):
        return subprocess.CompletedProcess(
            command,
            1,
            stdout="loaded 1029923 gaussians\n",
            stderr=(
                "Traceback (most recent call last):\n"
                "  File \"runner.py\", line 4, in <module>\n"
                "RuntimeError: refused with Authorization: Bearer sk-not-a-real-key\n"
                f"opaque credential={opaque_secret}\n"
            ),
        )

    with pytest.raises(
        TaskEvaluationSceneConfigurationStageToolError,
        match=f"scene_configuration_component_failed:{adapter_id}:1",
    ):
        execute_stage_tool(
            adapter_id=adapter_id, environment=environment, runner=run
        )

    captured = capsys.readouterr().err
    assert f"scene_configuration_component_failed:{adapter_id}" in captured
    assert "returncode=1" in captured
    assert "component_result_written=False" in captured
    # The component's own cause survives...
    assert "RuntimeError: refused with" in captured
    assert "loaded 1029923 gaussians" in captured
    # ...and its credential material does not.
    assert "sk-not-a-real-key" not in captured
    assert opaque_secret not in captured
    assert "REDACTED_SECRET" in captured
    assert "<redacted>" in captured


def test_component_failure_output_is_bounded_and_says_what_it_dropped(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A component looping on output must not bury the log that has to survive."""

    from blueprint_pipeline.task_evaluation_scene_configuration_stage_tool import (
        _COMPONENT_FAILURE_STREAM_TAIL_BYTES,
    )

    adapter_id = "artifixer3d_observed_object_removal"
    environment, _result_path = _environment(tmp_path, adapter_id=adapter_id)
    noise = "x" * (_COMPONENT_FAILURE_STREAM_TAIL_BYTES * 3)

    def run(command, **_kwargs):
        return subprocess.CompletedProcess(
            command, 1, stdout="", stderr=noise + "\nFINAL CAUSE\n"
        )

    with pytest.raises(TaskEvaluationSceneConfigurationStageToolError):
        execute_stage_tool(
            adapter_id=adapter_id, environment=environment, runner=run
        )

    captured = capsys.readouterr().err
    assert "FINAL CAUSE" in captured
    assert "earlier bytes dropped" in captured
    assert len(captured) < _COMPONENT_FAILURE_STREAM_TAIL_BYTES * 2
