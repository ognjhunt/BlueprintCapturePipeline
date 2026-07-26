import json
import os
import signal
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


def test_detached_model_volume_supervisor_ignores_only_sigint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[int, object]] = []
    monkeypatch.setenv(allocator.DETACHED_MODEL_VOLUME_SUPERVISOR_ENV, "1")
    monkeypatch.setattr(
        allocator.signal,
        "signal",
        lambda signum, handler: calls.append((signum, handler)),
    )

    assert allocator._configure_detached_supervisor_signal_policy("model-volume-run") is True
    assert calls == [(signal.SIGINT, signal.SIG_IGN)]


def test_foreground_model_volume_keeps_default_signal_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(allocator.DETACHED_MODEL_VOLUME_SUPERVISOR_ENV, raising=False)
    monkeypatch.setattr(
        allocator.signal,
        "signal",
        lambda *_args: (_ for _ in ()).throw(AssertionError("signal policy changed")),
    )

    assert allocator._configure_detached_supervisor_signal_policy("model-volume-run") is False


def test_detached_cpu_build_supervisor_ignores_only_sigint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[int, object]] = []
    monkeypatch.setenv(allocator.DETACHED_CPU_BUILD_SUPERVISOR_ENV, "1")
    monkeypatch.setattr(
        allocator.signal,
        "signal",
        lambda signum, handler: calls.append((signum, handler)),
    )

    assert allocator._configure_detached_supervisor_signal_policy("cpu-build-run") is True
    assert calls == [(signal.SIGINT, signal.SIG_IGN)]


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
        "_run_cpu_prerequisite_gate",
        lambda _output: {"status": "ready", "blockers": []},
    )
    monkeypatch.setattr(
        allocator,
        "_run_local_cpu_build",
        lambda _args: {"status": "completed", "password": "do-not-print"},
    )
    exit_code = allocator.main(
        [
            "cpu-build",
            "--execution-plane",
            "local",
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


def test_gpu_warm_worker_uses_canonical_allocator_and_redacts_stdout(monkeypatch, capsys) -> None:
    observed = {}

    def fake_run_active_worker(**kwargs):
        observed.update(kwargs)
        return {
            "status": "completed",
            "api_key": "do-not-print",
        }

    monkeypatch.setattr(allocator, "run_active_worker", fake_run_active_worker)
    exit_code = allocator.main(
        [
            "gpu-warm-worker",
            "--output-dir",
            "out",
            "--release-evidence",
            "release.json",
            "--model-cache-evidence",
            "model.json",
            "--watchdog-handoff-evidence",
            "handoff.json",
            "--resource-name-prefix",
            "blueprint-groot-oscar-serverless-test-",
            "--expected-source-commit",
            "c" * 40,
            "--campaign-budget-ledger",
            "budget.json",
            "--campaign-initial-spent-usd",
            "14.708611",
            "--campaign-initial-used-gpu-seconds",
            "15785",
            "--campaign-io-evidence",
            "campaign_io.json",
            "--carrier-volume-admission",
            "carrier.json",
            "--gpu-type-id",
            "NVIDIA RTX 6000 Ada Generation",
            "--execute",
        ]
    )

    assert exit_code == 0
    assert observed["execute"] is True
    assert observed["initial_gpu_seconds"] == 15_785
    assert observed["expected_source_commit"] == "c" * 40
    assert observed["campaign_io_evidence"] == "campaign_io.json"
    assert observed["carrier_volume_admission"] == "carrier.json"
    assert observed["gpu_type_ids"] == ("NVIDIA RTX 6000 Ada Generation",)
    assert json.loads(capsys.readouterr().out) == {"success": True}


def test_gpu_warm_worker_routes_only_through_canonical_allocator(monkeypatch, capsys) -> None:
    observed = {}

    def fake_run(**kwargs):
        observed.update(kwargs)
        return {
            "status": "completed",
            "provider_secret": "do-not-print",
        }

    monkeypatch.setattr(allocator, "run_active_worker", fake_run)
    exit_code = allocator.main(
        [
            "gpu-warm-worker",
            "--output-dir",
            "out",
            "--release-evidence",
            "release.json",
            "--model-cache-evidence",
            "cache.json",
            "--watchdog-handoff-evidence",
            "handoff.json",
            "--resource-name-prefix",
            "blueprint-groot-oscar-serverless-test-",
            "--expected-source-commit",
            "c" * 40,
            "--campaign-budget-ledger",
            "budget.json",
            "--campaign-initial-spent-usd",
            "14.708611",
            "--campaign-initial-used-gpu-seconds",
            "15785",
            "--campaign-io-evidence",
            "campaign_io.json",
            "--execute",
        ]
    )

    assert exit_code == 0
    assert observed["execute"] is True
    assert observed["initial_gpu_seconds"] == 15_785
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

    monkeypatch.setattr(allocator, "launch_detached_builder", fail_if_supervisor_called)
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


def test_cpu_build_run_forwards_fixed_model_cache_secret_files(tmp_path: Path, monkeypatch) -> None:
    observed = {}
    monkeypatch.setattr(
        allocator, "_run_cpu_prerequisite_gate", lambda _output: {"status": "ready"}
    )
    monkeypatch.setattr(
        allocator,
        "run_builder",
        lambda **kwargs: observed.update(kwargs) or {"status": "completed"},
    )
    args = Namespace(
        output_dir=str(tmp_path / "out"),
        packet_manifest="packet.json",
        builder_evidence="builder.json",
        spend="spend.json",
        token_file="do-token",
        docker_username_file="docker-user",
        docker_password_file="docker-pat",
        hf_token_file="hf-token",
        runpod_s3_access_key_file="s3-access",
        runpod_s3_secret_key_file="s3-secret",
        login_private_key="login-key",
        host_private_key="host-key",
        ssh_key_id=7,
        region="sfo3",
        allow_paid=True,
    )
    assert allocator._run_cpu(args)["status"] == "completed"
    assert observed["hf_token_file"] == Path("hf-token")
    assert observed["runpod_s3_access_key_file"] == Path("s3-access")
    assert observed["runpod_s3_secret_key_file"] == Path("s3-secret")


def test_model_volume_run_forwards_storage_only_composite_arguments(monkeypatch, capsys) -> None:
    observed = {}

    def fake_run_model_volume(**kwargs):
        observed.update(kwargs)
        return {"status": "completed"}

    monkeypatch.setattr(allocator, "run_storage_model_volume", fake_run_model_volume)
    exit_code = allocator.main(
        [
            "model-volume-run",
            "--output-dir",
            "out",
            "--data-center-id",
            "US-WA-1",
            "--storage-hourly-rate-usd",
            "0.004861111111",
            "--builder-evidence",
            "builder.json",
            "--builder-spend",
            "spend.json",
            "--login-private-key",
            "login-key",
            "--host-private-key",
            "host-key",
            "--ssh-key-id",
            "7",
            "--runtime-source-release-image-ref",
            "docker.io/blueprint/thin@sha256:" + "1" * 64,
            "--runtime-source-release-evidence",
            "thin-release.json",
            "--carrier-image-ref",
            "pytorch/pytorch:runtime@sha256:" + "2" * 64,
            "--replacement-source-output",
            "retained-cache",
            "--allow-paid",
        ]
    )
    assert exit_code == 0
    assert observed["storage_hourly_rate_usd"] == pytest.approx(0.004861111111)
    assert observed["builder_evidence_path"] == Path("builder.json")
    assert observed["builder_spend_path"] == Path("spend.json")
    assert observed["runtime_source_release_evidence_path"] == Path("thin-release.json")
    assert observed["replacement_source_output_dir"] == Path("retained-cache")
    assert json.loads(capsys.readouterr().out) == {"success": True}


def test_model_volume_rejects_runtime_bundle_without_thin_release_evidence(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        allocator,
        "launch_detached_model_volume",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("detached paid supervisor reached")),
    )
    exit_code = allocator.main(
        [
            "model-volume",
            "--output-dir",
            "out",
            "--data-center-id",
            "US-WA-1",
            "--storage-hourly-rate-usd",
            "0.011666666667",
            "--builder-evidence",
            "builder.json",
            "--builder-spend",
            "spend.json",
            "--login-private-key",
            "login-key",
            "--host-private-key",
            "host-key",
            "--ssh-key-id",
            "7",
            "--runtime-source-release-image-ref",
            "docker.io/blueprint/sealed@sha256:" + "9" * 64,
            "--carrier-image-ref",
            "pytorch/pytorch:runtime@sha256:" + "2" * 64,
            "--allow-paid",
        ]
    )
    assert exit_code == 2
    assert json.loads(capsys.readouterr().out) == {"success": False}


def test_model_volume_retention_forwards_bounded_existing_cache_contract(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    observed: dict[str, object] = {}

    def fake_retain(**kwargs: object) -> dict[str, object]:
        observed.update(kwargs)
        return {"status": "retained"}

    monkeypatch.setattr(allocator, "retain_verified_model_cache", fake_retain)
    exit_code = allocator.main(
        [
            "model-volume",
            "--output-dir",
            "retained",
            "--storage-hourly-rate-usd",
            "0.004861111111",
            "--retain-existing-output",
            "verified-cache",
            "--retention-ttl-seconds",
            str(7 * 24 * 60 * 60),
            "--retention-max-spend-usd",
            "1.0",
            "--campaign-spent-to-date-usd",
            "13.0",
            "--campaign-total-spend-cap-usd",
            "20.0",
            "--runpod-s3-access-key-file",
            "s3-access",
            "--runpod-s3-secret-key-file",
            "s3-secret",
            "--allow-paid",
        ]
    )

    assert exit_code == 0
    assert observed == {
        "output_dir": Path("retained"),
        "source_output_dir": Path("verified-cache"),
        "retention_ttl_seconds": 7 * 24 * 60 * 60,
        "storage_hourly_rate_usd": pytest.approx(0.004861111111),
        "max_retention_spend_usd": 1.0,
        "campaign_spent_to_date_usd": 13.0,
        "campaign_total_spend_cap_usd": 20.0,
        "runpod_s3_access_key_file": Path("s3-access"),
        "runpod_s3_secret_key_file": Path("s3-secret"),
        "allow_paid": True,
    }
    assert json.loads(capsys.readouterr().out) == {"success": True}


def test_gpu_canary_forwards_strict_policy_smoke_probe_kind(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    observed: dict[str, object] = {}

    def fake_run_canary(**kwargs: object) -> dict[str, object]:
        observed.update(kwargs)
        return {"status": "dry_run_ready"}

    monkeypatch.setattr(allocator, "run_canary", fake_run_canary)
    monkeypatch.setattr(
        allocator, "_source_checkout_blockers", lambda _expected: ([], "c" * 40)
    )
    exit_code = allocator.main(
        [
            "gpu-canary",
            "--provider-launch-request",
            "request.json",
            "--release-evidence",
            "release.json",
            "--model-cache-evidence",
            "models.json",
            "--preflight-bundle",
            "preflight.json",
            "--admission-out",
            "admission.json",
            "--bound-request-out",
            "bound.json",
            "--adapter-output",
            "adapter.json",
            "--pod-name",
            "strict-smoke-pod",
            "--expected-source-commit",
            "c" * 40,
            "--provider-output-put-url-file",
            "output-put-url.txt",
            "--probe-kind",
            "strict-policy-smoke",
        ]
    )
    assert exit_code == 0
    assert observed["probe_kind"] == "strict-policy-smoke"
    assert observed["expected_source_commit"] == "c" * 40
    assert observed["provider_output_put_url_file"] == "output-put-url.txt"
    assert json.loads(capsys.readouterr().out) == {"success": True}


def test_gpu_canary_rejects_missing_source_and_output_sink_before_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def must_not_dispatch(**_kwargs: object) -> dict[str, object]:
        raise AssertionError("canary dispatched without required admission inputs")

    monkeypatch.setattr(allocator, "run_canary", must_not_dispatch)
    admission = tmp_path / "admission.json"
    exit_code = allocator.main(
        [
            "gpu-canary",
            "--provider-launch-request",
            "request.json",
            "--release-evidence",
            "release.json",
            "--model-cache-evidence",
            "models.json",
            "--preflight-bundle",
            "preflight.json",
            "--admission-out",
            str(admission),
            "--bound-request-out",
            "bound.json",
            "--adapter-output",
            "adapter.json",
            "--pod-name",
            "strict-smoke-pod",
        ]
    )
    assert exit_code == 2
    result = json.loads(admission.read_text(encoding="utf-8"))
    assert result["provider_mutations_performed"] == 0
    assert result["blockers"] == [
        "gpu_canary_required_arguments_missing:"
        "expected_source_commit,provider_output_put_url_file"
    ]
    assert json.loads(capsys.readouterr().out) == {"success": False}


def test_gpu_canary_forwards_finetune_qualification_identity_file(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    observed: dict[str, object] = {}

    def fake_run_finetune(**kwargs: object) -> dict[str, object]:
        observed.update(kwargs)
        return {"status": "dry_run_ready"}

    monkeypatch.setattr(
        allocator,
        "run_g1_microwave_finetune_job",
        fake_run_finetune,
    )
    exit_code = allocator.main(
        [
            "gpu-canary",
            "--provider",
            "vast",
            "--probe-kind",
            allocator.G1_MICROWAVE_FINETUNE_PROBE_KIND,
            "--finetune-provider-bundle",
            "finetune-bundle.json",
            "--finetune-object-store-stage-dir",
            "input-stage",
            "--finetune-checkpoint-object-store-stage-dir",
            "checkpoint-stage",
            "--finetune-checkpoint-vast-session-manifest",
            "qualification-session.json",
            "--qualification-identity-file",
            "operator-key",
            "--provider-launch-request",
            "request.json",
            "--release-evidence",
            "release.json",
            "--model-cache-evidence",
            "models.json",
            "--preflight-bundle",
            "preflight.json",
            "--admission-out",
            "admission.json",
            "--bound-request-out",
            "bound.json",
            "--adapter-output",
            "adapter.json",
            "--pod-name",
            "finetune-pod",
        ]
    )

    assert exit_code == 0
    assert observed["qualification_identity_file"] == "operator-key"
    assert json.loads(capsys.readouterr().out) == {"success": True}


def test_gpu_canary_dispatches_one_single_kitchen_episode(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    observed: dict[str, object] = {}

    def fake_run_single_episode(**kwargs: object) -> dict[str, object]:
        observed.update(kwargs)
        return {"status": "dry_run_ready"}

    monkeypatch.setattr(allocator, "run_single_episode", fake_run_single_episode)
    exit_code = allocator.main(
        [
            "gpu-canary",
            "--provider",
            "runpod",
            "--probe-kind",
            "single-kitchen-episode",
            "--provider-launch-request",
            "request.json",
            "--release-evidence",
            "release.json",
            "--model-cache-evidence",
            "models.json",
            "--preflight-bundle",
            "preflight.json",
            "--admission-out",
            "admission.json",
            "--bound-request-out",
            "bound.json",
            "--adapter-output",
            "adapter.json",
            "--pod-name",
            "single-episode-pod",
            "--episode-bundle",
            "episode.zip",
            "--provider-bundle-url-file",
            "bundle-url.txt",
            "--provider-output-put-url-file",
            "output-put-url.txt",
            "--provider-output-get-url-file",
            "output-get-url.txt",
            "--execute",
        ]
    )

    assert exit_code == 0
    assert observed == {
        "provider_name": "runpod",
        "episode_bundle": "episode.zip",
        "provider_bundle_url_file": "bundle-url.txt",
        "provider_output_put_url_file": "output-put-url.txt",
        "provider_output_get_url_file": "output-get-url.txt",
        "provider_bootstrap_url_file": None,
        "release_evidence": "release.json",
        "provider_launch_request": "request.json",
        "preflight_bundle": "preflight.json",
        "admission_out": "admission.json",
        "bound_request_out": "bound.json",
        "adapter_output": "adapter.json",
        "pod_name": "single-episode-pod",
        "execute": True,
        "qualification_checkpoint_report": None,
        "qualification_checkpoint_part_stage_dirs": (),
    }
    assert json.loads(capsys.readouterr().out) == {"success": True}


def test_gpu_canary_dispatches_single_kitchen_episode_to_vast(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    observed: dict[str, object] = {}

    def fake_run_single_episode(**kwargs: object) -> dict[str, object]:
        observed.update(kwargs)
        return {"status": "dry_run_ready"}

    monkeypatch.setattr(allocator, "run_single_episode", fake_run_single_episode)
    exit_code = allocator.main(
        [
            "gpu-canary",
            "--provider",
            "vast",
            "--probe-kind",
            "single-kitchen-episode",
            "--provider-launch-request",
            "request.json",
            "--release-evidence",
            "release.json",
            "--model-cache-evidence",
            "models.json",
            "--preflight-bundle",
            "preflight.json",
            "--admission-out",
            "admission.json",
            "--bound-request-out",
            "bound.json",
            "--adapter-output",
            "adapter.json",
            "--pod-name",
            "single-episode-vast",
            "--episode-bundle",
            "episode.zip",
            "--provider-bundle-url-file",
            "bundle-url.txt",
            "--provider-output-put-url-file",
            "output-put-url.txt",
            "--provider-output-get-url-file",
            "output-get-url.txt",
            "--provider-bootstrap-url-file",
            "bootstrap-url.txt",
        ]
    )

    assert exit_code == 0
    assert observed["provider_name"] == "vast"
    assert observed["provider_bootstrap_url_file"] == "bootstrap-url.txt"
    assert observed["execute"] is False
    assert json.loads(capsys.readouterr().out) == {"success": True}


def test_gpu_canary_defaults_bind_authorized_strict_staged_plan(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    observed: dict[str, object] = {}

    def fake_run_canary(**kwargs: object) -> dict[str, object]:
        observed.update(kwargs)
        return {"status": "submitted"}

    monkeypatch.setattr(allocator, "run_canary", fake_run_canary)
    monkeypatch.setattr(
        allocator, "_source_checkout_blockers", lambda _expected: ([], "c" * 40)
    )
    exit_code = allocator.main(
        [
            "gpu-canary",
            "--provider-launch-request",
            "request.json",
            "--release-evidence",
            "release.json",
            "--model-cache-evidence",
            "models.json",
            "--preflight-bundle",
            "preflight.json",
            "--admission-out",
            "admission.json",
            "--bound-request-out",
            "bound.json",
            "--adapter-output",
            "adapter.json",
            "--pod-name",
            "strict-smoke-pod",
            "--expected-source-commit",
            "c" * 40,
            "--provider-output-put-url-file",
            "output-put-url.txt",
            "--campaign-budget-ledger",
            "budget.json",
            "--campaign-initial-spent-usd",
            "14.557003",
            "--campaign-initial-used-gpu-seconds",
            "15624",
            "--campaign-max-hourly-rate-usd",
            "1.99",
            "--authorize-reduced-canary-timeout",
            "--execute",
        ]
    )
    assert exit_code == 0
    assert observed["probe_kind"] == "strict-policy-smoke"
    campaign_budget = observed["campaign_budget"]
    assert isinstance(campaign_budget, dict)
    assert campaign_budget["combined_gpu_wall_cap_seconds"] == 21_000
    assert campaign_budget["reservation_gpu_seconds"] == 480
    assert campaign_budget["maximum_canary_reservation_gpu_seconds"] == 480
    assert campaign_budget["future_campaign_allowance_gpu_seconds"] == 3_500
    assert campaign_budget["minimum_reconciled_spend_usd"] == 14.557003
    assert campaign_budget["minimum_reconciled_gpu_seconds"] == 15_624
    assert json.loads(capsys.readouterr().out) == {"success": True}


def test_gpu_canary_dispatches_persistent_host_bake_through_allocator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    observed: dict[str, object] = {}

    def fake_prebake(**kwargs: object) -> dict[str, object]:
        observed.update(kwargs)
        return {"status": "dry_run_ready"}

    monkeypatch.setattr(allocator, "run_digitalocean_prebake", fake_prebake)
    out = tmp_path / "prebake"
    exit_code = allocator.main(
        [
            "gpu-canary",
            "--provider",
            "digitalocean",
            "--probe-kind",
            "persistent-host-bake",
            "--provider-launch-request",
            str(out / "request.json"),
            "--release-evidence",
            str(out / "release.json"),
            "--model-cache-evidence",
            str(out / "models.json"),
            "--preflight-bundle",
            str(out / "preflight.json"),
            "--admission-out",
            str(out / "admission.json"),
            "--bound-request-out",
            str(out / "bound.json"),
            "--adapter-output",
            str(out / "result.json"),
            "--pod-name",
            "prebake-test",
            "--campaign-budget-ledger",
            str(out / "budget.json"),
            "--campaign-initial-spent-usd",
            "14.557003",
            "--campaign-initial-used-gpu-seconds",
            "15624",
            "--campaign-reservation-seconds",
            "1396",
            "--future-campaign-allowance-seconds",
            "3980",
            "--campaign-max-hourly-rate-usd",
            "3.50",
            "--login-private-key",
            "login-key",
            "--host-private-key",
            "host-key",
            "--ssh-key-id",
            "55252816",
        ]
    )
    assert exit_code == 0
    assert observed["execute"] is False
    assert observed["reservation_seconds"] == 1_396
    assert observed["future_gpu_seconds"] == 3_980
    assert observed["gpu_wall_cap_seconds"] == 21_000
    assert json.loads(capsys.readouterr().out) == {"success": True}


def test_gpu_allocator_dispatches_authorized_persistent_carrier_campaign(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    observed: dict[str, object] = {}

    def fake_campaign(**kwargs: object) -> dict[str, object]:
        observed.update(kwargs)
        return {"status": "completed"}

    monkeypatch.setattr(allocator, "run_persistent_carrier_campaign", fake_campaign)
    monkeypatch.setattr(
        allocator, "_source_checkout_blockers", lambda _expected: ([], "c" * 40)
    )
    out = tmp_path / "persistent"
    exit_code = allocator.main(
        [
            "gpu-canary",
            "--probe-kind",
            "persistent-policy-wam-loop",
            "--provider-launch-request",
            str(out / "request.json"),
            "--release-evidence",
            str(out / "release.json"),
            "--model-cache-evidence",
            str(out / "models.json"),
            "--preflight-bundle",
            str(out / "preflight.json"),
            "--carrier-volume-admission",
            str(out / "carrier.json"),
            "--policy-observation",
            str(out / "observation.json"),
            "--persistent-job-dir",
            str(out / "job"),
            "--admission-out",
            str(out / "admission.json"),
            "--bound-request-out",
            str(out / "bound.json"),
            "--adapter-output",
            str(out / "result.json"),
            "--pod-name",
            "blueprint-persistent-exact",
            "--expected-source-commit",
            "c" * 40,
            "--campaign-budget-ledger",
            str(out / "budget.json"),
            "--campaign-initial-spent-usd",
            "14.557003",
            "--campaign-initial-used-gpu-seconds",
            "15624",
            "--campaign-max-hourly-rate-usd",
            "0.74",
            "--authorize-persistent-carrier-campaign",
            "--execute",
        ]
    )

    assert exit_code == 0
    assert observed["execute"] is True
    assert observed["expected_source_commit"] == "c" * 40
    budget = observed["campaign_budget"]
    assert isinstance(budget, dict)
    assert budget["campaign_stage"] == "persistent_carrier_campaign"
    assert budget["reservation_gpu_seconds"] == 18_600
    assert budget["combined_gpu_wall_cap_seconds"] == 36_000
    assert budget["future_campaign_allowance_gpu_seconds"] == 0
    assert json.loads(capsys.readouterr().out) == {"success": True}


def test_gpu_canary_source_checkout_binding_rejects_mismatch_and_dirty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(allocator, "_current_origin_main_commit", lambda: "c" * 40)
    monkeypatch.setattr(allocator, "_current_remote_main_commit", lambda: "c" * 40)
    monkeypatch.setattr(
        allocator,
        "_current_checkout_source_state",
        lambda: ("c" * 40, True),
    )
    assert allocator._source_checkout_blockers("b" * 40) == (
        ["gpu_canary_expected_source_commit_not_current_checkout"],
        "c" * 40,
    )
    monkeypatch.setattr(
        allocator,
        "_current_checkout_source_state",
        lambda: ("c" * 40, False),
    )
    assert allocator._source_checkout_blockers("c" * 40) == (
        ["gpu_canary_checkout_not_clean"],
        "c" * 40,
    )


def test_gpu_canary_source_checkout_binding_requires_origin_main_parity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        allocator,
        "_current_checkout_source_state",
        lambda: ("c" * 40, True),
    )
    monkeypatch.setattr(allocator, "_current_remote_main_commit", lambda: "c" * 40)
    monkeypatch.setattr(allocator, "_current_origin_main_commit", lambda: "b" * 40)
    assert allocator._source_checkout_blockers("c" * 40) == (
        ["gpu_canary_checkout_not_origin_main"],
        "c" * 40,
    )
    monkeypatch.setattr(allocator, "_current_origin_main_commit", lambda: "")
    assert allocator._source_checkout_blockers("c" * 40) == (
        ["gpu_canary_origin_main_commit_unavailable"],
        "c" * 40,
    )


def test_gpu_canary_source_checkout_binding_requires_live_remote_main_parity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        allocator,
        "_current_checkout_source_state",
        lambda: ("c" * 40, True),
    )
    monkeypatch.setattr(allocator, "_current_origin_main_commit", lambda: "c" * 40)
    monkeypatch.setattr(allocator, "_current_remote_main_commit", lambda: "b" * 40)
    assert allocator._source_checkout_blockers("c" * 40) == (
        ["gpu_canary_checkout_not_remote_main"],
        "c" * 40,
    )
    monkeypatch.setattr(allocator, "_current_remote_main_commit", lambda: "")
    assert allocator._source_checkout_blockers("c" * 40) == (
        ["gpu_canary_remote_main_commit_unavailable"],
        "c" * 40,
    )


def test_gpu_qualification_control_plane_identity_records_main_drift_without_blocking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        allocator, "_current_checkout_source_state", lambda: ("c" * 40, True)
    )
    monkeypatch.setattr(allocator, "_current_origin_main_commit", lambda: "b" * 40)
    monkeypatch.setattr(allocator, "_current_remote_main_commit", lambda: "a" * 40)

    blockers, identity = allocator._control_plane_checkout_blockers()

    assert blockers == []
    assert identity["orchestrator_source_commit"] == "c" * 40
    assert identity["checkout_clean"] is True
    assert identity["orchestrator_equals_origin_main"] is False
    assert identity["orchestrator_equals_remote_main"] is False
    assert identity["main_parity_is_diagnostic_not_runtime_identity"] is True


def test_gpu_qualification_control_plane_identity_still_requires_clean_checkout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        allocator, "_current_checkout_source_state", lambda: ("c" * 40, False)
    )
    monkeypatch.setattr(allocator, "_current_origin_main_commit", lambda: "b" * 40)
    monkeypatch.setattr(allocator, "_current_remote_main_commit", lambda: "a" * 40)

    blockers, identity = allocator._control_plane_checkout_blockers()

    assert blockers == ["gpu_canary_orchestrator_checkout_not_clean"]
    assert identity["orchestrator_source_commit"] == "c" * 40


def test_gpu_canary_rejects_digitalocean_provider_for_runpod_probe(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    out = tmp_path / "provider-mismatch"
    exit_code = allocator.main(
        [
            "gpu-canary",
            "--provider",
            "digitalocean",
            "--probe-kind",
            "strict-policy-smoke",
            "--provider-launch-request",
            str(out / "request.json"),
            "--release-evidence",
            str(out / "release.json"),
            "--model-cache-evidence",
            str(out / "models.json"),
            "--preflight-bundle",
            str(out / "preflight.json"),
            "--admission-out",
            str(out / "admission.json"),
            "--bound-request-out",
            str(out / "bound.json"),
            "--adapter-output",
            str(out / "result.json"),
            "--pod-name",
            "must-not-launch-runpod",
        ]
    )
    assert exit_code == 2
    result = json.loads((out / "result.json").read_text())
    assert result["provider_mutations_performed"] == 0
    assert result["blockers"] == ["digitalocean_gpu_canary_requires_persistent_host_bake"]
    assert json.loads(capsys.readouterr().out) == {"success": False}


def test_strict_probe_runbook_arms_watchdog_within_budget_reservation() -> None:
    runbook = (Path(__file__).parents[1] / "docs/runbooks/groot-oscar-thin-release.md").read_text(
        encoding="utf-8"
    )
    assert 'deadline="$(( $(date +%s) + 480 ))"' in runbook
    assert 'deadline="$(( $(date +%s) + 900 ))"' not in runbook


def test_gpu_qualification_gpu_status_is_a_successful_control_observation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        allocator,
        "run_qualification_session",
        lambda **_kwargs: {"status": "gpu_status_collected_continuing_spend"},
    )
    out = tmp_path / "gpu-status"

    exit_code = allocator.main(
        [
            "gpu-canary",
            "--provider",
            "vast",
            "--probe-kind",
            allocator.SINGLE_KITCHEN_QUALIFICATION_PROBE_KIND,
            "--qualification-action",
            "gpu-status",
            "--qualification-session-manifest",
            str(out / "session.json"),
            "--provider-launch-request",
            str(out / "request.json"),
            "--release-evidence",
            str(out / "release.json"),
            "--model-cache-evidence",
            str(out / "models.json"),
            "--preflight-bundle",
            str(out / "preflight.json"),
            "--admission-out",
            str(out / "admission.json"),
            "--bound-request-out",
            str(out / "bound.json"),
            "--adapter-output",
            str(out / "result.json"),
            "--pod-name",
            "retained-qualification-gpu",
            "--execute",
        ]
    )

    assert exit_code == 0
    assert json.loads(capsys.readouterr().out) == {"success": True}


def test_gpu_qualification_allocate_requires_independent_source_commit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        allocator,
        "run_qualification_session",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("qualification session reached without source binding")
        ),
    )
    out = tmp_path / "missing-source-binding"

    exit_code = allocator.main(
        [
            "gpu-canary",
            "--provider", "vast",
            "--probe-kind", allocator.SINGLE_KITCHEN_QUALIFICATION_PROBE_KIND,
            "--qualification-action", "allocate",
            "--qualification-session-manifest", str(out / "session.json"),
            "--episode-bundle", str(out / "episode.zip"),
            "--provider-bundle-url-file", str(out / "bundle-url.txt"),
            "--provider-output-put-url-file", str(out / "put-url.txt"),
            "--provider-output-get-url-file", str(out / "get-url.txt"),
            "--provider-launch-request", str(out / "request.json"),
            "--release-evidence", str(out / "release.json"),
            "--model-cache-evidence", str(out / "models.json"),
            "--preflight-bundle", str(out / "preflight.json"),
            "--admission-out", str(out / "admission.json"),
            "--bound-request-out", str(out / "bound.json"),
            "--adapter-output", str(out / "result.json"),
            "--pod-name", "retained-qualification-gpu",
            "--execute",
        ]
    )

    assert exit_code == 2
    result = json.loads((out / "result.json").read_text())
    assert result["provider_mutations_performed"] == 0
    assert result["blockers"] == [
        "single_kitchen_qualification_required_arguments_missing:expected_image_source_commit"
    ]
    for name in ("request.json", "preflight.json", "admission.json", "bound.json"):
        assert json.loads((out / name).read_text()) == result
    assert json.loads(capsys.readouterr().out) == {"success": False}


def test_gpu_qualification_allocate_blocks_dirty_or_unavailable_orchestrator_before_session(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        allocator,
        "run_qualification_session",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("qualification session reached from an invalid orchestrator checkout")
        ),
    )
    monkeypatch.setattr(
        allocator,
        "_control_plane_checkout_blockers",
        lambda: (
            ["gpu_canary_orchestrator_checkout_not_clean"],
            {
                "schema_version": "blueprint.gpu_canary_control_plane_identity.v1",
                "orchestrator_source_commit": "c" * 40,
            },
        ),
    )
    out = tmp_path / "mismatched-checkout"

    exit_code = allocator.main(
        [
            "gpu-canary",
            "--provider", "vast",
            "--probe-kind", allocator.SINGLE_KITCHEN_QUALIFICATION_PROBE_KIND,
            "--qualification-action", "allocate",
            "--qualification-session-manifest", str(out / "session.json"),
            "--episode-bundle", str(out / "episode.zip"),
            "--provider-bundle-url-file", str(out / "bundle-url.txt"),
            "--provider-output-put-url-file", str(out / "put-url.txt"),
            "--provider-output-get-url-file", str(out / "get-url.txt"),
            "--provider-launch-request", str(out / "request.json"),
            "--release-evidence", str(out / "release.json"),
            "--model-cache-evidence", str(out / "models.json"),
            "--preflight-bundle", str(out / "preflight.json"),
            "--admission-out", str(out / "admission.json"),
            "--bound-request-out", str(out / "bound.json"),
            "--adapter-output", str(out / "result.json"),
            "--pod-name", "retained-qualification-gpu",
            "--expected-source-commit", "b" * 40,
            "--execute",
        ]
    )

    assert exit_code == 2
    assert json.loads((out / "result.json").read_text()) == {
        "status": "blocked",
        "blockers": ["gpu_canary_orchestrator_checkout_not_clean"],
        "control_plane_identity": {
            "schema_version": "blueprint.gpu_canary_control_plane_identity.v1",
            "orchestrator_source_commit": "c" * 40,
        },
        "provider_mutations_performed": 0,
    }
    for name in ("request.json", "preflight.json", "admission.json", "bound.json"):
        assert json.loads((out / name).read_text()) == json.loads(
            (out / "result.json").read_text()
        )
    assert json.loads(capsys.readouterr().out) == {"success": False}


def test_gpu_qualification_allows_clean_orchestrator_commit_to_differ_from_image(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    observed: dict[str, object] = {}

    def fake_run_qualification_session(**kwargs: object) -> dict[str, object]:
        observed.update(kwargs)
        return {"status": "dry_run_ready"}

    monkeypatch.setattr(allocator, "run_qualification_session", fake_run_qualification_session)
    monkeypatch.setattr(
        allocator,
        "_control_plane_checkout_blockers",
        lambda: (
            [],
            {
                "schema_version": "blueprint.gpu_canary_control_plane_identity.v1",
                "orchestrator_source_commit": "c" * 40,
                "checkout_clean": True,
                "orchestrator_equals_remote_main": False,
            },
        ),
    )
    out = tmp_path / "separate-identities"
    exit_code = allocator.main(
        [
            "gpu-canary",
            "--provider", "vast",
            "--probe-kind", allocator.SINGLE_KITCHEN_QUALIFICATION_PROBE_KIND,
            "--qualification-action", "allocate",
            "--qualification-session-manifest", str(out / "session.json"),
            "--episode-bundle", str(out / "episode.zip"),
            "--provider-bundle-url-file", str(out / "bundle-url.txt"),
            "--provider-output-put-url-file", str(out / "put-url.txt"),
            "--provider-output-get-url-file", str(out / "get-url.txt"),
            "--provider-launch-request", str(out / "request.json"),
            "--release-evidence", str(out / "release.json"),
            "--model-cache-evidence", str(out / "models.json"),
            "--preflight-bundle", str(out / "preflight.json"),
            "--admission-out", str(out / "admission.json"),
            "--bound-request-out", str(out / "bound.json"),
            "--adapter-output", str(out / "result.json"),
            "--pod-name", "retained-qualification-gpu",
            "--expected-image-source-commit", "b" * 40,
        ]
    )

    assert exit_code == 0
    assert observed["expected_source_commit"] == "b" * 40
    assert observed["orchestrator_source_commit"] == "c" * 40
    assert json.loads(capsys.readouterr().out) == {"success": True}


def test_gpu_qualification_component_stop_is_a_successful_control_action(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        allocator,
        "run_qualification_session",
        lambda **_kwargs: {"status": "component_stopped_continuing_spend"},
    )
    out = tmp_path / "component-stop"

    exit_code = allocator.main(
        [
            "gpu-canary",
            "--provider",
            "vast",
            "--probe-kind",
            allocator.SINGLE_KITCHEN_QUALIFICATION_PROBE_KIND,
            "--qualification-action",
            "stop-component",
            "--qualification-component",
            "episode",
            "--qualification-session-manifest",
            str(out / "session.json"),
            "--provider-launch-request",
            str(out / "request.json"),
            "--release-evidence",
            str(out / "release.json"),
            "--model-cache-evidence",
            str(out / "models.json"),
            "--preflight-bundle",
            str(out / "preflight.json"),
            "--admission-out",
            str(out / "admission.json"),
            "--bound-request-out",
            str(out / "bound.json"),
            "--adapter-output",
            str(out / "result.json"),
            "--pod-name",
            "retained-qualification-gpu",
            "--execute",
        ]
    )

    assert exit_code == 0
    assert json.loads(capsys.readouterr().out) == {"success": True}
