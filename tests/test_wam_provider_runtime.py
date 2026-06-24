from __future__ import annotations

import sys
from pathlib import Path

from blueprint_pipeline.wam_provider_runtime import (
    live_provider_gate_blockers,
    normalize_provider_rollouts,
    parse_wam_provider_commands,
    run_provider_command,
)


def test_provider_rollout_normalization_handles_empty_and_invalid_numeric_payloads() -> None:
    assert (
        normalize_provider_rollouts(
            payload={"wam_rollout_results": "not-a-rollout-list"},
            substrate="cosmos3_wam",
            generated_at="2026-06-20T00:00:00+00:00",
        )
        == []
    )

    rollouts = normalize_provider_rollouts(
        payload={
            "rollouts": [
                {
                    "policy_id": "provider_policy",
                    "scenario_eval_run_id": "run-1",
                    "predicted_success": True,
                    "uncertainty_score": True,
                },
                {
                    "policy_id": "fallback_uncertainty_policy",
                    "scenario_eval_run_id": "run-2",
                    "success": False,
                    "uncertainty_score": "not-a-number",
                },
            ]
        },
        substrate="cosmos3_wam",
        generated_at="2026-06-20T00:00:00+00:00",
    )

    assert rollouts[0]["uncertainty_score"] == 0.35
    assert rollouts[1]["uncertainty_score"] == 0.35
    assert rollouts[1]["predicted_success"] is False


def test_provider_command_parser_and_live_gate_are_fail_closed(monkeypatch) -> None:
    monkeypatch.delenv("BLUEPRINT_ALLOW_LIVE_WAM_PROVIDER", raising=False)
    commands = parse_wam_provider_commands(
        [
            "cosmos3_wam=/opt/cosmos adapter",
            "oscar=/opt/oscar-adapter",
        ]
    )
    assert commands == {
        "cosmos3_wam": "/opt/cosmos adapter",
        "oscar_wam": "/opt/oscar-adapter",
    }

    for invalid in (
        "private_model=/tmp/provider",
        "fixture_wam=/tmp/fixture",
        "mujoco=/tmp/sim",
        "cosmos3_wam=",
    ):
        try:
            parse_wam_provider_commands([invalid])
        except ValueError as exc:
            assert "--wam-provider-command" in str(exc)
        else:  # pragma: no cover - keeps failure message concrete
            raise AssertionError(f"accepted invalid WAM provider command: {invalid}")

    assert live_provider_gate_blockers(allow_live_provider=False) == [
        "allow_live_wam_provider_not_enabled",
        "BLUEPRINT_ALLOW_LIVE_WAM_PROVIDER_not_enabled",
    ]
    monkeypatch.setenv("BLUEPRINT_ALLOW_LIVE_WAM_PROVIDER", "true")
    assert live_provider_gate_blockers(allow_live_provider=False) == [
        "allow_live_wam_provider_not_enabled"
    ]
    assert live_provider_gate_blockers(allow_live_provider=True) == []


def test_provider_command_reports_parse_launch_timeout_and_invalid_json_failures(
    tmp_path: Path,
) -> None:
    runtime_package = tmp_path / "runtime.json"
    runtime_package.write_text("{}", encoding="utf-8")

    parse_status, _, parse_detail = run_provider_command(
        command_text="'unterminated",
        runtime_package_path=runtime_package,
        output_path=tmp_path / "parse" / "output.json",
        substrate="cosmos3_wam",
        artifact_output_uri=None,
        timeout_seconds=1,
    )
    assert parse_status == "blocked"
    assert parse_detail["blockers"] == ["wam_provider_command_parse_failed:ValueError"]

    launch_status, _, launch_detail = run_provider_command(
        command_text=str(tmp_path / "missing-provider-binary"),
        runtime_package_path=runtime_package,
        output_path=tmp_path / "launch" / "output.json",
        substrate="cosmos3_wam",
        artifact_output_uri=None,
        timeout_seconds=1,
    )
    assert launch_status == "blocked"
    assert launch_detail["blockers"] == ["wam_provider_command_launch_failed:FileNotFoundError"]

    sleeper = tmp_path / "slow_provider.py"
    sleeper.write_text(
        "\n".join(
            [
                "import sys",
                "import time",
                "print('provider starting')",
                "print('provider stderr', file=sys.stderr)",
                "time.sleep(5)",
            ]
        ),
        encoding="utf-8",
    )
    timeout_status, _, timeout_detail = run_provider_command(
        command_text=f"{sys.executable} {sleeper}",
        runtime_package_path=runtime_package,
        output_path=tmp_path / "timeout" / "output.json",
        substrate="cosmos3_wam",
        artifact_output_uri="gs://bucket/output",
        timeout_seconds=0.1,
    )
    assert timeout_status == "blocked"
    assert timeout_detail["blockers"] == ["wam_provider_command_timeout"]

    invalid_json = tmp_path / "invalid_json_provider.py"
    invalid_json.write_text(
        "\n".join(
            [
                "import os",
                "with open(os.environ['BLUEPRINT_WAM_PROVIDER_OUTPUT'], 'w', encoding='utf-8') as fh:",
                "    fh.write('{not valid json')",
            ]
        ),
        encoding="utf-8",
    )
    json_status, _, json_detail = run_provider_command(
        command_text=f"{sys.executable} {invalid_json}",
        runtime_package_path=runtime_package,
        output_path=tmp_path / "json" / "output.json",
        substrate="cosmos3_wam",
        artifact_output_uri=None,
        timeout_seconds=1,
    )
    assert json_status == "blocked"
    assert json_detail["blockers"] == ["wam_provider_output_json_invalid:JSONDecodeError"]
