from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_evaluation_scene_configuration_builtin_producers import (
    TOOLCHAIN_SCHEMA_VERSION,
    builtin_scene_configuration_stage_producer_registry,
)
from blueprint_pipeline.task_evaluation_scene_configuration_stage_producers import (
    ADMITTED_PRODUCER_IDENTITIES,
    PRODUCTION_RESULT_SCHEMA_VERSION,
)


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _toolchain(tmp_path: Path, commit: str) -> Path:
    root = tmp_path / "toolchain"
    stages = {}
    for identity in ADMITTED_PRODUCER_IDENTITIES:
        executable = root / "stages" / identity.adapter_id
        executable.parent.mkdir(parents=True, exist_ok=True)
        executable.write_text("#!/bin/sh\nexit 99\n", encoding="utf-8")
        executable.chmod(0o555)
        stages[identity.adapter_id] = {
            "entrypoint": executable.relative_to(root).as_posix(),
            "network_policy": (
                "disabled"
                if identity.adapter_id == "simready_native_import_qualification"
                else "provider_and_openai_api"
            ),
            "secrets_via_files_only": True,
            "raw_secret_values_in_argv_or_logs": False,
        }
    files = [
        {
            "relative_path": path.relative_to(root).as_posix(),
            "sha256": _sha256(path),
            "size_bytes": path.stat().st_size,
            "executable": True,
        }
        for path in sorted((root / "stages").iterdir())
    ]
    manifest = {
        "schema_version": TOOLCHAIN_SCHEMA_VERSION,
        "status": "published_full_byte_readback_passed",
        "source_commit": commit,
        "full_byte_service_account_readback_passed": True,
        "stages": stages,
        "files": files,
        "toolchain_digest": "",
    }
    manifest["toolchain_digest"] = canonical_digest(
        manifest, digest_field="toolchain_digest"
    )
    (root / f"{TOOLCHAIN_SCHEMA_VERSION}.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    for path in sorted(root.rglob("*"), reverse=True):
        path.chmod(0o555 if path.is_dir() or path.name != f"{TOOLCHAIN_SCHEMA_VERSION}.json" else 0o444)
    root.chmod(0o555)
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

    assert len(artifacts) == 3
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
