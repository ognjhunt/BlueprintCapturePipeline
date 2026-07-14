import json
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
