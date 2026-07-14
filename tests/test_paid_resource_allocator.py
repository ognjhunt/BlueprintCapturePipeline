import json
import os
import subprocess
from argparse import Namespace
from pathlib import Path

import pytest

from blueprint_pipeline import paid_resource_allocator as allocator
from blueprint_pipeline.groot_oscar_infrastructure_admission import (
    MIN_BUILD_FREE_BYTES,
    build_live_machine_capability_evidence,
)
from blueprint_pipeline.paid_resource_admission import PaidResourceAdmissionBlocked


def _write_inputs(tmp_path: Path, *, paid: bool = True) -> Namespace:
    packet = tmp_path / "packet.json"
    builder = tmp_path / "builder.json"
    spend = tmp_path / "spend.json"
    packet.write_text(
        json.dumps(
            {
                "status": "ready",
                "source_commit": "a" * 40,
                "source_worktree_dirty": False,
                "provider_launch_performed_by_packet": False,
            }
        ),
        encoding="utf-8",
    )
    builder.write_text(
        json.dumps(
            {
                "provider": "github_actions",
                "purpose": "image_build",
                "platform": "linux/amd64",
                "docker_daemon_verified": True,
                "docker_buildx_verified": True,
                "free_disk_bytes": MIN_BUILD_FREE_BYTES,
                "registry_push_auth_file_verified": True,
                "independent_teardown_watchdog": True,
                "expected_source_commit": "a" * 40,
            }
        ),
        encoding="utf-8",
    )
    spend.write_text(
        json.dumps(
            {
                "paid_mutation_authorized": paid,
                "max_spend_usd": 1.0,
                "hard_ttl_seconds": 7200,
                "one_resource_limit": True,
                "independent_teardown_watchdog": True,
            }
        ),
        encoding="utf-8",
    )
    script = tmp_path / "build.sh"
    script.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    script.chmod(0o755)
    return Namespace(
        output_dir=str(tmp_path / "out"),
        packet_manifest=str(packet),
        builder_evidence=str(builder),
        spend=str(spend),
        mount_path=str(tmp_path),
        build_workdir=str(tmp_path),
        build_script=str(script),
    )


def _verified_live() -> dict:
    return build_live_machine_capability_evidence(
        {
            "observation_source": "live_machine_probe",
            "system": "Linux",
            "architecture": "x86_64",
            "mount_path": "/builder",
            "free_bytes": MIN_BUILD_FREE_BYTES,
            "docker_cli_present": True,
            "docker_daemon_responding": True,
            "docker_buildx_available": True,
            "builder_ready_marker": True,
        }
    )


def test_local_cpu_allocator_sets_canonical_context_after_both_admissions(
    tmp_path: Path, monkeypatch
) -> None:
    observed = {}
    monkeypatch.setattr(allocator, "observe_local_machine", lambda **_kwargs: _verified_live())

    def fake_run(command, **kwargs):
        observed["command"] = command
        observed["environment"] = kwargs["env"]
        return Namespace(returncode=0)

    monkeypatch.setattr(allocator.subprocess, "run", fake_run)
    result = allocator._run_local_cpu_build(_write_inputs(tmp_path))
    assert result["status"] == "completed"
    assert observed["environment"]["BLUEPRINT_CANONICAL_CPU_BUILD_CONTEXT"] == "true"
    assert (tmp_path / "out/cpu_build_execution_admission.json").is_file()


@pytest.mark.parametrize(
    "script_name",
    [
        "build_push_groot_oscar_foundation_image.sh",
        "build_push_groot_oscar_release_image.sh",
        "build_push_groot_oscar_closed_loop_image.sh",
    ],
)
def test_legacy_build_scripts_cannot_be_reenabled_by_environment(script_name: str) -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / script_name
    completed = subprocess.run(
        ["bash", str(script)],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "BLUEPRINT_CANONICAL_CPU_BUILD_CONTEXT": "true"},
    )
    assert completed.returncode == 2
    assert "legacy build path disabled" in completed.stderr


def test_local_cpu_allocator_rejects_before_build_process_when_not_admitted(
    tmp_path: Path, monkeypatch
) -> None:
    called = False

    def fake_run(*_args, **_kwargs):
        nonlocal called
        called = True
        return Namespace(returncode=0)

    monkeypatch.setattr(allocator.subprocess, "run", fake_run)
    with pytest.raises(PaidResourceAdmissionBlocked):
        allocator._run_local_cpu_build(_write_inputs(tmp_path, paid=False))
    assert called is False


def test_allocator_cli_never_prints_provider_result_secrets(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        allocator,
        "_run_local_cpu_build",
        lambda _args: {"status": "completed", "password": "do-not-print"},
    )
    exit_code = allocator.main(
        [
            "cpu-build-local",
            "--output-dir",
            "out",
            "--packet-manifest",
            "packet.json",
            "--builder-evidence",
            "builder.json",
            "--spend",
            "spend.json",
            "--mount-path",
            ".",
            "--build-workdir",
            ".",
            "--build-script",
            "build.sh",
        ]
    )
    assert exit_code == 0
    assert json.loads(capsys.readouterr().out) == {"success": True}


def test_cpu_build_run_blocks_before_provider_when_live_prerequisites_fail(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        allocator,
        "_run_cpu_prerequisite_gate",
        lambda _output: {"status": "blocked", "blockers": ["pin_missing"]},
    )
    provider_called = False

    def fail_if_provider_called(**_kwargs):
        nonlocal provider_called
        provider_called = True
        return {}

    monkeypatch.setattr(allocator, "run_builder", fail_if_provider_called)
    result = allocator._run_cpu(Namespace(output_dir=str(tmp_path)))
    assert result["status"] == "blocked_before_allocation"
    assert result["provider_mutation_attempted"] is False
    assert result["blockers"] == ["pin_missing"]
    assert provider_called is False


def test_cpu_build_cli_rechecks_prerequisites_before_detached_supervisor(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    monkeypatch.setattr(
        allocator,
        "_run_cpu_prerequisite_gate",
        lambda _output: {"status": "blocked", "blockers": ["pin_missing"]},
    )
    supervisor_called = False

    def fail_if_supervisor_called(**_kwargs):
        nonlocal supervisor_called
        supervisor_called = True
        return {}

    monkeypatch.setattr(
        allocator, "launch_detached_builder", fail_if_supervisor_called
    )
    exit_code = allocator.main(
        [
            "cpu-build",
            "--output-dir",
            str(tmp_path),
            "--packet-manifest",
            "packet.json",
            "--builder-evidence",
            "builder.json",
            "--spend",
            "spend.json",
            "--login-private-key",
            "login-key",
            "--host-private-key",
            "host-key",
            "--ssh-key-id",
            "1",
            "--allow-paid",
        ]
    )
    assert exit_code == 2
    assert json.loads(capsys.readouterr().out) == {"success": False}
    assert supervisor_called is False
