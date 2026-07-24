"""Regression guards for the defects raised in review of PR #179.

Each test pins one specific failure mode that review caught, so a later refactor
that reintroduces it fails here rather than in production.
"""

from __future__ import annotations

import inspect
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from blueprint_pipeline import judge_spend_governor as gov
from blueprint_pipeline import roboworld_progress_judge as judge
from blueprint_pipeline import vast_provider_adapter as vast
from blueprint_pipeline.action_normalization import normalize_actions
from blueprint_pipeline.action_space_registry import (
    UNITREE_G1_ARM_HAND_43D,
    get_action_space,
)
from blueprint_pipeline.oscar_resident_worker import (
    ResidentWorkerError,
    start_resident_oscar_generate_from_args,
)


# -- P1: the CLI option must be accepted by the function main() forwards to ---


def test_adapter_accepts_the_gpu_selection_policy_keyword() -> None:
    """`main()` forwards `vars(args)`, so every CLI option needs a parameter.

    Without this, every console/module invocation of the Vast adapter raised
    TypeError before it could write even a dry-run result.
    """

    parameters = inspect.signature(vast.run_vast_provider_adapter).parameters
    assert "gpu_selection_policy" in parameters


def test_every_adapter_cli_option_maps_to_a_run_parameter() -> None:
    """Generalises the above so the next added flag cannot reintroduce it."""

    parser = vast._build_arg_parser() if hasattr(vast, "_build_arg_parser") else None
    if parser is None:
        pytest.skip("adapter parser is constructed inline in main()")
    parameters = set(inspect.signature(vast.run_vast_provider_adapter).parameters)
    missing = [
        action.dest
        for action in parser._actions
        if action.dest not in {"help"} and action.dest not in parameters
    ]
    assert not missing, f"CLI options with no run parameter: {missing}"


def test_adapter_dry_run_completes(tmp_path: Path) -> None:
    """End-to-end proof the forwarding path works, not just the signature."""

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "blueprint_pipeline.vast_provider_adapter",
            "--job-dir",
            str(tmp_path / "job"),
            "--mode",
            "dry-run",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )
    assert completed.returncode == 0, completed.stderr[-2000:]
    assert "dry_run_ready" in completed.stdout


# -- P1: a failed startup must not leak the worker process ------------------


_FAILING_WORKER = textwrap.dedent(
    """
    import json, sys, time
    # Announce a payload the client must reject, then stay alive holding the
    # "GPU" so a leak is observable.
    print(json.dumps({"schema_version": "oscar_resident_worker_ready.v1",
                      "status": "ready", "model_load_seconds": 1.0}))
    sys.stdout.flush()
    time.sleep(300)
    """
)


def test_worker_is_closed_when_startup_validation_fails(tmp_path: Path) -> None:
    """start() spawns before it validates, so a rejected handshake must tear down."""

    script = tmp_path / "worker.py"
    script.write_text(_FAILING_WORKER, encoding="utf-8")

    class _Args:
        oscar_repo = str(tmp_path)
        checkpoint = str(tmp_path / "ckpt")
        oscar_num_steps = 35
        oscar_guidance = 6.0
        oscar_height = 480
        oscar_width = 640
        oscar_fps = 15.0
        provider_timeout_seconds = 30.0
        oscar_resident_worker_max_restarts = 0

    import blueprint_pipeline.oscar_resident_worker as module

    real_argv = module.build_resident_worker_argv
    module.build_resident_worker_argv = lambda **_kwargs: [sys.executable, str(script)]
    try:
        with pytest.raises(ResidentWorkerError, match="gpu_residency_unproven"):
            start_resident_oscar_generate_from_args(
                _Args(),
                python=sys.executable,
                extract_next_frame=lambda video, out_dir: None,
                require_gpu_residency=True,
            )
    finally:
        module.build_resident_worker_argv = real_argv

    # The spawned process must not survive the rejected handshake: a leaked
    # worker would hold the GPU for the rest of the job.
    leaked = subprocess.run(
        ["pgrep", "-f", str(script)], capture_output=True, text=True, check=False
    )
    assert leaked.returncode != 0, f"leaked worker pids: {leaked.stdout.strip()}"


# -- P2: a stalled judge command must be bounded and still settle -----------


def _spend_policy(**overrides):
    kwargs = {
        "campaign_id": "regression",
        "usd_per_1k_input_tokens": 0.10,
        "estimated_tokens_per_frame": 100,
        "target_spend_usd": 5.0,
        "hard_cap_usd": 50.0,
    }
    kwargs.update(overrides)
    return gov.build_judge_spend_policy(**kwargs)


def _ready_request():
    return judge.build_judge_request(
        rollout_id="rollout-1",
        criterion_id="registered_task_success",
        task_instruction="place the box",
        frame_uris=[f"frame://{index}" for index in range(60)],
        view_roles={"fixed_external_left": ["task_progress"]},
        duration_seconds=25.0,
        segment_count=3,
        source_frame_count=300,
    )


def test_stalled_judge_command_is_bounded_and_settled(monkeypatch, tmp_path) -> None:
    """A hung provider must not block forever or leave the ledger unsettled."""

    monkeypatch.setenv(judge.GATE_ENV, "1")
    monkeypatch.setenv(
        judge.JUDGE_COMMAND_ENV, f"{sys.executable} -c \"import time; time.sleep(120)\""
    )
    governor = gov.JudgeSpendGovernor(policy=_spend_policy())

    result = judge.run_progress_judge_command(
        _ready_request(), output_dir=tmp_path, governor=governor, timeout_seconds=2.0
    )

    assert result["status"] == "blocked"
    assert "progress_judge_command_timed_out" in result["blockers"]
    # Settled despite the timeout: a stuck provider call is still billable.
    assert governor.request_count == 1
    assert governor.spent_usd > 0.0
    assert result["spend_ledger"]["request_count"] == 1


def test_failed_judge_command_is_also_settled(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv(judge.GATE_ENV, "1")
    monkeypatch.setenv(judge.JUDGE_COMMAND_ENV, f"{sys.executable} -c \"raise SystemExit(3)\"")
    governor = gov.JudgeSpendGovernor(policy=_spend_policy())

    result = judge.run_progress_judge_command(
        _ready_request(), output_dir=tmp_path, governor=governor, timeout_seconds=60.0
    )

    assert result["status"] == "blocked"
    assert "progress_judge_command_failed" in result["blockers"]
    assert governor.request_count == 1


def test_judge_command_is_not_shell_interpreted(monkeypatch, tmp_path) -> None:
    """Shell metacharacters must not be executable in operator config."""

    marker = tmp_path / "pwned"
    monkeypatch.setenv(judge.GATE_ENV, "1")
    monkeypatch.setenv(
        judge.JUDGE_COMMAND_ENV, f"{sys.executable} -c pass; touch {marker}"
    )
    governor = gov.JudgeSpendGovernor(policy=_spend_policy())

    judge.run_progress_judge_command(
        _ready_request(), output_dir=tmp_path, governor=governor, timeout_seconds=60.0
    )

    assert not marker.exists()


# -- P2: normalization must use the selected action space -------------------


def test_normalize_actions_honours_a_non_sc3_action_space() -> None:
    """Defaulting back to SC3 would raise on 43-D stats instead of normalizing."""

    space = get_action_space(UNITREE_G1_ARM_HAND_43D)
    rows = [[0.1 * index + 0.01 * dim for dim in range(space.dim)] for index in range(4)]
    stats = {
        "expected_dim": space.dim,
        "per_dimension": [{"mean": 0.0, "std": 1.0} for _ in range(space.dim)],
    }

    normalized = normalize_actions(rows, stats=stats, action_schema_id=UNITREE_G1_ARM_HAND_43D)

    assert len(normalized) == len(rows)
    assert all(len(row) == space.dim for row in normalized)

    # And the SC3 default still rejects those stats, so the parameter is load-bearing.
    with pytest.raises(ValueError, match="normalization_stats_dimension_contract_invalid"):
        normalize_actions(rows, stats=stats)


def test_normalize_actions_still_defaults_to_sc3() -> None:
    rows = [[0.1 * index] * 7 for index in range(4)]
    stats = {"expected_dim": 7, "per_dimension": [{"mean": 0.0, "std": 1.0} for _ in range(7)]}
    assert len(normalize_actions(rows, stats=stats)) == len(rows)
