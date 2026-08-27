from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_evaluation_scene_configuration_builtin_producers import (
    builtin_scene_configuration_stage_producer_registry,
)
from blueprint_pipeline.task_evaluation_scene_configuration_stage_producers import (
    ADMITTED_PRODUCER_IDENTITIES,
    PRODUCTION_RESULT_SCHEMA_VERSION,
)
from scripts.build_task_evaluation_scene_configuration_toolchain import (
    build_published_scene_configuration_toolchain,
)
from tests.test_build_task_evaluation_scene_configuration_toolchain import (
    _component_packages,
)


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _toolchain(tmp_path: Path, commit: str) -> Path:
    root = tmp_path / "toolchain"
    build_published_scene_configuration_toolchain(
        source_commit=commit,
        output_root=root,
        readback=lambda path: path.read_bytes(),
        readback_actor="service-account:test",
        component_packages=_component_packages(tmp_path),
    )
    return root


def test_builtin_producer_executes_only_sealed_entrypoint_and_redacts_secret(
    tmp_path: Path,
) -> None:
    commit = "a" * 40
    toolchain = _toolchain(tmp_path, commit)
    secret = tmp_path / "openai-key"
    secret.write_text("super-secret-value\n", encoding="utf-8")
    secret.chmod(0o400)
    calls: list[list[str]] = []

    def run(command, *, env, **_kwargs):
        calls.append(command)
        output = Path(env["BLUEPRINT_SCENE_CONFIGURATION_STAGE_OUTPUT_ROOT"])
        stage_input = json.loads(
            Path(env["BLUEPRINT_SCENE_CONFIGURATION_STAGE_INPUT"]).read_text(
                encoding="utf-8"
            )
        )
        assert stage_input["construction_envelope"]["run_id"] == "configure-scene"
        result_path = Path(env["BLUEPRINT_SCENE_CONFIGURATION_STAGE_RESULT"])
        roles = (
            "configured_appearance_without_source_object",
            "appearance_removal_receipt",
            "appearance_visual_review_receipt",
            "configured_task_thumbnail",
            "provider_render_reference_manifest",
        )
        artifacts = []
        for role in roles:
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
        result = {
            "schema_version": PRODUCTION_RESULT_SCHEMA_VERSION,
            "status": "completed",
            "stage_id": "stage-1",
            "capability": "observed_appearance_object_removal",
            "provider_mutations_performed": 0,
            "paid_execution_requested": False,
            "executed_inside_parent_configuration_run": True,
            "artifacts": artifacts,
            "production_result_digest": "",
        }
        result["production_result_digest"] = canonical_digest(
            result, digest_field="production_result_digest"
        )
        result_path.write_text(json.dumps(result), encoding="utf-8")
        return subprocess.CompletedProcess(
            command, 0, stdout="key=super-secret-value", stderr=""
        )

    registry = builtin_scene_configuration_stage_producer_registry(
        expected_source_commit=commit,
        toolchain_root=toolchain,
        runner=run,
        environment={"OPENAI_API_KEY_FILE": str(secret)},
    )
    output = tmp_path / "output"
    output.mkdir()
    identity = ADMITTED_PRODUCER_IDENTITIES[0]
    configuration = tmp_path / "configuration.json"
    configuration.write_text("{}\n", encoding="utf-8")
    artifacts = registry.execute(
        stage={
            "stage_id": "stage-1",
            "capability": identity.capability,
            "adapter": {"id": identity.adapter_id, "version": identity.version},
            "execution_class": "gpu_canary",
        },
        envelope={"run_id": "configure-scene"},
        configuration={},
        configuration_path=configuration,
        dependency_results=(),
        output_root=output,
    )

    assert len(artifacts) == 5
    assert calls == [[str(toolchain / "stages" / identity.adapter_id)]]
    log = (output / "stage_producer.log").read_text(encoding="utf-8")
    assert "super-secret-value" not in log
    assert "REDACTED_SECRET" in log


def test_builtin_producer_rejects_raw_secret_environment(tmp_path: Path) -> None:
    commit = "b" * 40
    registry = builtin_scene_configuration_stage_producer_registry(
        expected_source_commit=commit,
        toolchain_root=_toolchain(tmp_path, commit),
        environment={"OPENAI_API_KEY": "must-not-cross-runtime-boundary"},
    )
    output = tmp_path / "output"
    output.mkdir()
    configuration = tmp_path / "configuration.json"
    configuration.write_text("{}\n", encoding="utf-8")
    identity = ADMITTED_PRODUCER_IDENTITIES[0]

    with pytest.raises(
        RuntimeError, match="scene_configuration_raw_secret_environment_forbidden"
    ):
        registry.execute(
            stage={
                "stage_id": "stage-1",
                "capability": identity.capability,
                "adapter": {"id": identity.adapter_id, "version": identity.version},
                "execution_class": "gpu_canary",
            },
            envelope={"run_id": "configure-scene"},
            configuration={},
            configuration_path=configuration,
            dependency_results=(),
            output_root=output,
        )


def test_builtin_producer_retains_redacted_partial_output_on_timeout(
    tmp_path: Path,
) -> None:
    commit = "c" * 40
    secret = tmp_path / "openai-key"
    secret.write_text("timeout-secret-value\n", encoding="utf-8")
    secret.chmod(0o400)

    def time_out(command, **kwargs):
        raise subprocess.TimeoutExpired(
            command,
            kwargs["timeout"],
            output=b"partial stdout timeout-secret-value",
            stderr=(
                b"partial stderr timeout-secret-value "
                b"https://object.invalid/out?X-Amz-Signature=signed-timeout-value"
            ),
        )

    registry = builtin_scene_configuration_stage_producer_registry(
        expected_source_commit=commit,
        toolchain_root=_toolchain(tmp_path, commit),
        runner=time_out,
        environment={"OPENAI_API_KEY_FILE": str(secret)},
    )
    output = tmp_path / "output"
    output.mkdir()
    configuration = tmp_path / "configuration.json"
    configuration.write_text("{}\n", encoding="utf-8")
    identity = ADMITTED_PRODUCER_IDENTITIES[0]

    with pytest.raises(
        RuntimeError,
        match=(
            "scene_configuration_stage_producer_timeout:"
            "artifixer3d_observed_object_removal:7800"
        ),
    ):
        registry.execute(
            stage={
                "stage_id": "stage-1",
                "capability": identity.capability,
                "adapter": {"id": identity.adapter_id, "version": identity.version},
                "execution_class": "gpu_canary",
            },
            envelope={"run_id": "configure-scene"},
            configuration={},
            configuration_path=configuration,
            dependency_results=(),
            output_root=output,
        )

    log = (output / "stage_producer.log").read_text(encoding="utf-8")
    assert "partial stdout" in log
    assert "partial stderr" in log
    assert "timeout-secret-value" not in log
    assert "signed-timeout-value" not in log
    assert "<redacted>" in log
    assert log.count("REDACTED_SECRET") == 2
