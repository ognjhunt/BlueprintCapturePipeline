"""Hermetic tests for the sealed GR00T+OSCAR DigitalOcean launcher."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import pytest

from blueprint_pipeline import gpu_render_providers as providers
from blueprint_pipeline import groot_oscar_digitalocean_closed_loop_job as J
from blueprint_pipeline import groot_oscar_closed_loop_image as gocl
from blueprint_pipeline.paid_resource_admission import (
    PAID_LANE_ADMISSION_SCHEMA_VERSION,
    build_paid_lane_admission,
    require_paid_resource_admission,
)


IMAGE_HASH = "b" * 64
DIGEST_REF = "docker.io/nijelhunt/blueprint-groot-oscar-eval@sha256:" + IMAGE_HASH
SOURCE_COMMIT = "a" * 40
DIRTY_PATCH = "f" * 64
TASK_PROMPT = (
    "Open the dishwasher door; if the dishwasher is already open, close the dishwasher door."
)
REAL_CONSISTENCY_COMMAND = "python -m owner_runtime.shared_model_inverse_scorer"


def _gpu_render_grant():
    admission = build_paid_lane_admission(resource_class="gpu_render", blockers=[])
    return require_paid_resource_admission(
        admission,
        resource_class="gpu_render",
        expected_schema_version=PAID_LANE_ADMISSION_SCHEMA_VERSION,
    )


def _completed_closed_loop_manifest(
    *,
    min_coherent_horizon_frames: int = 2,
    min_measured_coherent_horizon_frames: int = 2,
    forward_inverse_consistency_proven: bool = True,
    generated_video_success_label_passed: bool = False,
) -> dict:
    return {
        "status": "completed",
        "steps_executed": 3,
        "forward_inverse_consistency_proven": forward_inverse_consistency_proven,
        "episode_termination": {
            "reason": "task_target_reached_at_step_3",
            "steps_cap": J.DEFAULT_EPISODE_MAX_STEPS,
            "stop_on_task_completion": True,
            "min_steps": J.DEFAULT_MIN_TASK_COMPLETION_STEPS,
            "task_completed_early": True,
        },
        "generated_clip_coherence": {
            "min_coherent_horizon_frames_required": min_coherent_horizon_frames,
            "min_measured_coherent_horizon_frames": min_measured_coherent_horizon_frames,
        },
        "proof": {
            "feed_forward_verified": True,
            "policy_observes_wam_generated_next_observation": True,
            "fresh_learned_policy_requery_steps": 3,
            "external_episode_consistency_scorer_ran_steps": 3,
            "forward_inverse_consistency_proven_steps": 3
            if forward_inverse_consistency_proven
            else 2,
        },
        "success_proof": {
            "manipulation_success_proven": False,
            "simulated_manipulation_success_shown": False,
            "generated_video_success_label_passed": generated_video_success_label_passed,
            "real_world_task_success_proven": False,
            "success_proof_separate_from_structural_loop_proof": True,
        },
    }


def _active_plan() -> dict:
    return gocl.build_sealed_launch_plan(
        env={gocl.IMAGE_REF_ENV: DIGEST_REF, gocl.SEALED_CONFIRMED_ENV: "true"},
        start_frame="/workspace/initial_policy_frame.png",
        route_file="/workspace/route.json",
        steps=J.DEFAULT_EPISODE_MAX_STEPS,
        task_prompt=TASK_PROMPT,
        output_dir="/workspace/closed_loop_out",
        wam_consistency_command=REAL_CONSISTENCY_COMMAND,
    )


def _worker_image_manifest_diagnostic(tmp_path: Path) -> Path:
    """Write a completed registry manifest diagnostic bound to DIGEST_REF."""
    path = tmp_path / "isaac_worker_image_manifest_diagnostic.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "isaac_worker_image_manifest_diagnostic.v2",
                "status": "completed",
                "image_ref": "docker.io/nijelhunt/blueprint-groot-oscar-eval:20260712",
                "resolved_digest": f"sha256:{IMAGE_HASH}",
                "resolved_digest_ref": DIGEST_REF,
                "runnable_platform": "linux/amd64",
                "layer_count": 35,
                "total_compressed_size_bytes": 50_455_509_186,
                "largest_layer_size_bytes": 10_585_790_213,
                "layers_over_1gb": 4,
                "large_image_pull_risk": True,
                "split_layer_layout_suitable": False,
                "recommended_startup_no_runtime_timeout_seconds": 1800,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_cli_worker_image_manifest_diagnostic_flag_threads_to_job(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    diagnostic_path = _worker_image_manifest_diagnostic(tmp_path)
    captured: dict[str, object] = {}

    def _capture(**kwargs) -> dict:
        captured.update(kwargs)
        return {"status": "prepared"}

    monkeypatch.setattr(J, "run_groot_oscar_digitalocean_closed_loop_job", _capture)

    exit_code = J.main(
        [
            "--start-frame",
            str(tmp_path / "frame.png"),
            "--route-file",
            str(tmp_path / "route.json"),
            "--task-prompt",
            TASK_PROMPT,
            "--out-dir",
            str(tmp_path / "job"),
            "--worker-image-manifest-diagnostic",
            str(diagnostic_path),
        ]
    )

    assert exit_code == 0
    assert captured["worker_image_manifest_diagnostic"] == str(diagnostic_path)


def test_worker_bootstrap_runs_healthcheck_groot_and_task_adaptive_closed_loop() -> None:
    script = J.build_worker_bootstrap_script(_active_plan())

    assert "upload_phase container_bash_started" in script
    assert "curl \\\n  --http1.1" in script
    assert "--retry 8" in script
    assert "--retry-all-errors" in script
    assert "--retry-max-time 900" in script
    assert "--max-time 1200" in script
    assert "groot_oscar_closed_loop_image_healthcheck.py --require-cuda" in script
    assert "/opt/gr00t-venv/bin/python -m gr00t.eval.run_gr00t_server" in script
    assert "BLUEPRINT_CLOSED_LOOP_RC=\"$RC\" python /workspace/write_result.py" in script
    assert "upload_phase runner_done" in script
    assert "--require-fresh-learned-policy-requery" in script
    assert "--allow-wam-consistency-scoring" in script
    assert "--require-forward-inverse-consistency" in script
    assert REAL_CONSISTENCY_COMMAND in script
    assert "--stop-on-task-completion" in script
    assert "--min-steps 3" in script
    assert "--min-coherent-horizon-frames 2" in script
    assert "--oscar-height 480" in script
    assert "--oscar-width 640" in script
    assert "--oscar-height 240" not in script
    assert ".replace(" not in script
    assert '"Content-Type: application/zip"' in script
    assert 'terminal_phase = phase in {"runner_done", "runner_timeout"}' in script
    assert "max_attempts = 5 if terminal_phase else 1" in script
    assert '"--connect-timeout", "30", "--max-time", "300"' in script
    assert "if uploaded.returncode == 0:" in script
    assert "raise SystemExit(75)" in script
    assert (
        'if [ "$1" = "runner_done" ] || [ "$1" = "runner_timeout" ]; then\n'
        '    BLUEPRINT_BOOTSTRAP_PHASE="$1" python /workspace/upload_progress.py\n'
    ) in script
    assert (
        'BLUEPRINT_BOOTSTRAP_PHASE="$1" python /workspace/upload_progress.py || true'
        in script
    )
    assert "SECRET_ROOT=/workspace/.runtime-secrets" in script
    assert "/run/blueprint-secrets" not in script
    assert "export GROOT_PID GEAR_SONIC_PID ISAAC_TASK_PID" in script
    assert "groot_policy_server_exited_before_ready" in script
    assert "official_gear_sonic_controller_exited_before_ready" in script
    assert "persistent_isaac_task_executor_exited_before_ready" in script


@pytest.mark.parametrize(
    ("return_codes", "expected_calls", "expected_exit"),
    [
        ([7, 0], 2, None),
        ([7, 7, 7, 7, 7], 5, 75),
    ],
)
def test_terminal_output_upload_retries_and_fails_closed_when_exhausted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    return_codes: list[int],
    expected_calls: int,
    expected_exit: int | None,
) -> None:
    script = J.build_worker_bootstrap_script(_active_plan())
    source = script.split("cat > /workspace/upload_progress.py <<'PY'\n", 1)[1].split(
        "\nPY\n\ncat > /workspace/write_result.py",
        1,
    )[0]
    source = source.replace(
        'workspace = Path("/workspace")',
        f"workspace = Path({str(tmp_path)!r})",
    )
    (tmp_path / "out").mkdir()
    calls: list[list[str]] = []

    def fake_run(command, *, check):  # noqa: ANN001
        assert check is False
        calls.append(list(command))
        return subprocess.CompletedProcess(command, return_codes[len(calls) - 1])

    monkeypatch.setenv("BLUEPRINT_BOOTSTRAP_PHASE", "runner_done")
    monkeypatch.setenv(
        "BLUEPRINT_WORKER_RUNTIME_MANIFEST_SIGNED_PUT_URL",
        "https://objects.example/output?redacted-signature",
    )
    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)

    if expected_exit is None:
        exec(compile(source, "upload_progress.py", "exec"), {})
    else:
        with pytest.raises(SystemExit) as exc_info:
            exec(compile(source, "upload_progress.py", "exec"), {})
        assert exc_info.value.code == expected_exit

    assert len(calls) == expected_calls
    assert all("--connect-timeout" in command and "--max-time" in command for command in calls)


def test_digitalocean_provider_blocks_20gb_fallback_before_create(
    tmp_path: Path,
    monkeypatch,
) -> None:
    token_path = tmp_path / "do-token.txt"
    token_path.write_text("fake-token", encoding="utf-8")
    monkeypatch.setenv("DIGITALOCEAN_TOKEN_FILE", str(token_path))
    monkeypatch.setenv("BLUEPRINT_DO_GPU_SIZES", "gpu-4000adax1-20gb")

    def fail_do_call(*_args: object, **_kwargs: object) -> tuple[int, dict]:
        raise AssertionError("20 GB candidate must be rejected before droplet create")

    monkeypatch.setattr(providers, "_do_call", fail_do_call)

    provider = providers.DigitalOceanRenderProvider()
    launch = provider.launch(
        tmp_path / "job",
        {
            "name": "blueprint-groot-oscar-closed-loop",
            "size": "gpu-4000adax1-20gb",
            "region": "atl1",
            "image": providers.DO_GPU_BASE_IMAGE,
            "ssh_keys": [12345],
            "min_gpu_ram_mb": J.DEFAULT_MIN_GPU_RAM_MB,
            "max_hourly_rate_usd": J.DEFAULT_MAX_HOURLY_RATE_USD,
            "prelaunch_spend_guard": {
                "required_before_provider_launch": True,
                "can_launch": True,
            },
        },
        cold=True,
        paid_resource_admission_grant=_gpu_render_grant(),
    )

    assert launch["status"] == "blocked"
    assert "digitalocean_gpu_size_below_min_vram" in launch["blockers"]
    assert launch["gpu_ram_policy"]["min_gpu_ram_mb"] == J.DEFAULT_MIN_GPU_RAM_MB
    assert launch["gpu_ram_policy"]["allowed_size_candidates"] == []
    assert launch["gpu_ram_policy"]["rejected_size_candidates"][0]["size"] == (
        "gpu-4000adax1-20gb"
    )
    assert launch["gpu_ram_policy"]["rejected_size_candidates"][0]["reason"] == (
        "below_min_gpu_ram"
    )


def test_closed_loop_result_contract_blocks_when_strict_success_label_unproven() -> None:
    failing = J._closed_loop_result_contract(
        watch={
            "status": "completed",
            "runner_result": {
                "status": "completed",
                "closed_loop_manifest": _completed_closed_loop_manifest(
                    generated_video_success_label_passed=False
                ),
            },
            "runner_result_source": "isaac_runtime_result.json",
        },
        expected_steps_cap=J.DEFAULT_EPISODE_MAX_STEPS,
        expected_min_task_completion_steps=J.DEFAULT_MIN_TASK_COMPLETION_STEPS,
        expected_min_coherent_horizon_frames=J.DEFAULT_MIN_COHERENT_HORIZON_FRAMES,
        expected_forward_inverse_consistency_required=True,
        expected_generated_video_success_label_required=True,
    )

    assert failing["status"] == "FAIL"
    assert "closed_loop_result_generated_video_success_label_not_proven" in failing[
        "blockers"
    ]
    assert failing["generated_video_success_label"] == {
        "required": True,
        "passed": False,
    }

    passing = J._closed_loop_result_contract(
        watch={
            "status": "completed",
            "runner_result": {
                "status": "completed",
                "closed_loop_manifest": _completed_closed_loop_manifest(
                    generated_video_success_label_passed=True
                ),
            },
            "runner_result_source": "isaac_runtime_result.json",
        },
        expected_steps_cap=J.DEFAULT_EPISODE_MAX_STEPS,
        expected_min_task_completion_steps=J.DEFAULT_MIN_TASK_COMPLETION_STEPS,
        expected_min_coherent_horizon_frames=J.DEFAULT_MIN_COHERENT_HORIZON_FRAMES,
        expected_forward_inverse_consistency_required=True,
        expected_generated_video_success_label_required=True,
    )

    assert passing["status"] == "PASS"
    assert passing["generated_video_success_label"] == {
        "required": True,
        "passed": True,
    }


