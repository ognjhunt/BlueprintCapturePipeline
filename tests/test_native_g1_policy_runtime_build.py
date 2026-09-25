from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from blueprint_pipeline import native_g1_policy_runtime_build as builder
from blueprint_pipeline.decision_evidence_contracts import canonical_digest


def _checkout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    source = tmp_path / "official"
    server = source / "lerobot/scripts/serve_lerobot_vla_http.py"
    server.parent.mkdir(parents=True)
    server.write_text("server")
    pyproject = source / "lerobot/pyproject.toml"
    pyproject.write_text("project")
    monkeypatch.setattr(builder, "_source_revision", lambda _path: builder.PINNED_SOURCE_REVISION)
    monkeypatch.setattr(
        builder, "PINNED_POLICY_SERVER_SHA256", hashlib.sha256(server.read_bytes()).hexdigest()
    )
    monkeypatch.setattr(
        builder, "PYPROJECT_SHA256", hashlib.sha256(pyproject.read_bytes()).hexdigest()
    )
    return source


def test_dependency_lock_is_exact_pinned_bytes() -> None:
    assert builder._sha256(builder.LOCK) == "sha256:" + builder.LOCK_SHA256


def test_plan_uses_same_pinned_image_and_copied_python(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _checkout(tmp_path, monkeypatch)
    output = tmp_path / "new-build"
    plan = builder.prepare_g1_policy_runtime_build(checkout=source, output_dir=output)
    assert not output.exists()
    assert plan["status"] == "planned_not_built"
    assert plan["plan_digest"] == canonical_digest(plan, digest_field="plan_digest")
    command = plan["command"]
    assert command[:5] == ["docker", "run", "--rm", "--pull", "never"]
    assert command[command.index("--entrypoint") + 2] == builder.NATIVE_TASK_ARENA_IMAGE
    assert "-m venv --copies" in command[-1]
    assert "--require-hashes" in command[-1]
    assert "DiffusionPolicy" in command[-1] and "PI05Policy" in command[-1]
    assert command[command.index("--network") + 1] == "bridge"
    assert plan["model_bytes_downloaded"] is False


def test_in_container_plan_uses_pinned_lock_without_nested_docker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _checkout(tmp_path, monkeypatch)
    plan = builder.prepare_g1_policy_runtime_build(
        checkout=source,
        output_dir=tmp_path / "new-build",
        execution_mode="inside_isaac_container",
    )
    assert plan["image"] == builder.NATIVE_TASK_ARENA_IMAGE
    assert plan["command"][:4] == ["/bin/bash", "-euo", "pipefail", "-c"]
    assert "--require-hashes" in plan["command"][-1]
    assert "-m venv --copies" in plan["command"][-1]
    assert plan["command"][-1].index("linux-libc-dev") < plan["command"][-1].index("-m pip install")
    assert plan["command"][-1].index("python3.12-dev") < plan["command"][-1].index("-m pip install")
    assert "test -f /usr/include/linux/input-event-codes.h" in plan["command"][-1]
    assert "test -f /usr/include/python3.12/Python.h" in plan["command"][-1]
    assert not (tmp_path / "new-build").exists()


def test_in_container_execution_refuses_unverified_image_before_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _checkout(tmp_path, monkeypatch)
    output = tmp_path / "new-build"
    plan = builder.prepare_g1_policy_runtime_build(
        checkout=source, output_dir=output, execution_mode="inside_isaac_container"
    )
    monkeypatch.setattr(builder.sys, "platform", "linux")
    monkeypatch.delenv(builder.PINNED_IMAGE_ENV, raising=False)
    with pytest.raises(ValueError, match="g1_policy_runtime_pinned_container_unverified"):
        builder.execute_g1_policy_runtime_build(plan=plan)
    assert not output.exists()


def test_in_container_execution_runs_direct_shell_after_transport_assertion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _checkout(tmp_path, monkeypatch)
    output = tmp_path / "new-build"
    plan = builder.prepare_g1_policy_runtime_build(
        checkout=source, output_dir=output, execution_mode="inside_isaac_container"
    )
    monkeypatch.setattr(builder.sys, "platform", "linux")
    monkeypatch.setenv(builder.PINNED_IMAGE_ENV, builder.NATIVE_TASK_ARENA_IMAGE)
    original_is_file = Path.is_file
    monkeypatch.setattr(
        Path, "is_file",
        lambda path: True if str(path) == "/isaac-sim/python.sh" else original_is_file(path),
    )

    def run(command: list[str], **_kwargs: object) -> SimpleNamespace:
        assert command[:4] == ["/bin/bash", "-euo", "pipefail", "-c"]
        (output / "policy-runtime/bin").mkdir(parents=True)
        (output / "policy-runtime/bin/python").write_bytes(b"python")
        (output / "installed.freeze").write_text("torch==2.10.0\n")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(builder.subprocess, "run", run)
    result = builder.execute_g1_policy_runtime_build(plan=plan)
    assert result["status"] == "built_import_probe_passed_no_cuda_probe"
    assert result["execution_mode"] == "inside_isaac_container"


def test_plan_refuses_changed_upstream_pyproject(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _checkout(tmp_path, monkeypatch)
    (source / "lerobot/pyproject.toml").write_text("changed")
    with pytest.raises(ValueError, match="g1_policy_runtime_source_or_lock_identity_mismatch"):
        builder.prepare_g1_policy_runtime_build(checkout=source, output_dir=tmp_path / "new-build")


def test_execute_rechecks_source_before_creating_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _checkout(tmp_path, monkeypatch)
    output = tmp_path / "new-build"
    plan = builder.prepare_g1_policy_runtime_build(checkout=source, output_dir=output)
    (source / "lerobot/pyproject.toml").write_text("changed")
    monkeypatch.setattr(builder.sys, "platform", "linux")
    with pytest.raises(ValueError, match="g1_policy_runtime_source_or_lock_identity_mismatch"):
        builder.execute_g1_policy_runtime_build(plan=plan)
    assert not output.exists()


def test_execute_retains_success_receipt_without_claiming_cuda_or_episode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _checkout(tmp_path, monkeypatch)
    plan = builder.prepare_g1_policy_runtime_build(
        checkout=source, output_dir=tmp_path / "new-build"
    )
    monkeypatch.setattr(builder.sys, "platform", "linux")

    def run(_command: list[str], **_kwargs: object) -> SimpleNamespace:
        output = tmp_path / "new-build"
        (output / "policy-runtime/bin").mkdir(parents=True)
        (output / "policy-runtime/bin/python").write_bytes(b"python")
        (output / "installed.freeze").write_text("torch==2.10.0\n")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(builder.subprocess, "run", run)
    result = builder.execute_g1_policy_runtime_build(plan=plan)
    assert result["status"] == "built_import_probe_passed_no_cuda_probe"
    assert result["cuda_device_probed"] is False
    assert result["episode_executed"] is False
    assert result["result_digest"] == canonical_digest(result, digest_field="result_digest")
    assert json.loads((tmp_path / "new-build" / (builder.SCHEMA + ".result.json")).read_text()) == result


def test_execute_retains_failure_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _checkout(tmp_path, monkeypatch)
    plan = builder.prepare_g1_policy_runtime_build(
        checkout=source, output_dir=tmp_path / "new-build"
    )
    monkeypatch.setattr(builder.sys, "platform", "linux")
    monkeypatch.setattr(
        builder.subprocess, "run", lambda *_args, **_kwargs: SimpleNamespace(returncode=12)
    )
    result = builder.execute_g1_policy_runtime_build(plan=plan)
    assert result["status"] == "blocked"
    assert result["container_exit_code"] == 12
    assert result["installed_freeze_sha256"] is None
    assert (tmp_path / "new-build/build.log").is_file()
