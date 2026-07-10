from __future__ import annotations

import json
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


def test_provider_rollout_normalization_excludes_text_to_video_preview() -> None:
    payload = {
        "endpoint_class": "text_to_video_preview",
        "task_evaluation_run_eligible": False,
        "rollouts": [
            {
                "rollout_id": "preview",
                "policy_id": "not-a-policy-eval",
                "scenario_eval_run_id": "preview-only",
                "generated_video_path": "preview.mp4",
            }
        ],
    }

    assert normalize_provider_rollouts(
        payload=payload,
        substrate="oscar_wam",
        generated_at="now",
    ) == []


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


def test_backbone_identity_check_uses_strategy_catalog_and_is_backend_agnostic() -> None:
    from blueprint_pipeline.wam_provider_runtime import (
        check_provider_backbone_identity,
        expected_base_model_for_substrate,
    )

    assert expected_base_model_for_substrate("cosmos3_wam") == "Cosmos3-Nano"
    assert expected_base_model_for_substrate("oscar_wam") == "Cosmos-Predict2.5-2B"
    assert expected_base_model_for_substrate("fixture_wam") == ""

    verified = check_provider_backbone_identity(
        payload={"base_model": "cosmos3-nano"},
        substrate="cosmos3_wam",
        generated_at="now",
    )
    assert verified["status"] == "verified"
    assert verified["backbone_identity_mismatch"] is False

    nested = check_provider_backbone_identity(
        payload={"model_provenance": {"base_model": "Cosmos3-Nano"}},
        substrate="cosmos3_wam",
        generated_at="now",
    )
    assert nested["backbone_identity_verified"] is True

    # cosmos3_wam artifacts can never come from a Predict2.5 command...
    mismatch = check_provider_backbone_identity(
        payload={"base_model": "Cosmos-Predict2.5-2B"},
        substrate="cosmos3_wam",
        generated_at="now",
    )
    assert mismatch["status"] == "mismatch"
    assert mismatch["backbone_identity_mismatch"] is True

    # ...and vice versa.
    reverse = check_provider_backbone_identity(
        payload={"base_model": "Cosmos3-Nano"},
        substrate="oscar_wam",
        generated_at="now",
    )
    assert reverse["backbone_identity_mismatch"] is True

    silent = check_provider_backbone_identity(
        payload={"rollouts": []},
        substrate="cosmos3_wam",
        generated_at="now",
    )
    assert silent["status"] == "not_self_reported"
    assert silent["backbone_identity_mismatch"] is False


def test_provider_command_hard_fails_on_backbone_mismatch_with_typed_artifact(
    tmp_path: Path,
) -> None:
    from blueprint_pipeline.wam_provider_runtime import run_provider_command

    runtime_package = tmp_path / "runtime.json"
    runtime_package.write_text("{}", encoding="utf-8")
    provider = tmp_path / "wrong_family_provider.py"
    provider.write_text(
        "\n".join(
            [
                "import json, os",
                "payload = {",
                "    'schema_version': 'cosmos3_wam_command_adapter.v1',",
                "    'status': 'completed',",
                "    'base_model': 'Cosmos-Predict2.5-2B',",
                "    'rollouts': [],",
                "}",
                "with open(os.environ['BLUEPRINT_WAM_PROVIDER_OUTPUT'], 'w', encoding='utf-8') as fh:",
                "    json.dump(payload, fh)",
            ]
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "wam_provider" / "wam_provider_output.json"

    status, payload, detail = run_provider_command(
        command_text=f"{sys.executable} {provider}",
        runtime_package_path=runtime_package,
        output_path=output_path,
        substrate="cosmos3_wam",
        artifact_output_uri=None,
        timeout_seconds=30,
    )

    assert status == "blocked"
    assert payload == {}
    assert any(
        blocker.startswith("wam_provider_backbone_identity_mismatch:")
        for blocker in detail["blockers"]
    )
    check = detail["backbone_identity_check"]
    assert check["backbone_identity_mismatch"] is True
    assert check["expected_base_model"] == "Cosmos3-Nano"
    assert check["self_reported_base_model"] == "Cosmos-Predict2.5-2B"
    error_path = output_path.parent / "wam_provider_backbone_identity_error.json"
    assert error_path.is_file()
    error_payload = json.loads(error_path.read_text(encoding="utf-8"))
    assert error_payload["schema_version"] == "wam_provider_backbone_identity_error.v1"
    assert error_payload["error"] == "wam_provider_backbone_identity_mismatch"


def test_provider_command_accepts_matching_backbone_identity(tmp_path: Path) -> None:
    from blueprint_pipeline.wam_provider_runtime import run_provider_command

    runtime_package = tmp_path / "runtime.json"
    runtime_package.write_text("{}", encoding="utf-8")
    provider = tmp_path / "matching_provider.py"
    provider.write_text(
        "\n".join(
            [
                "import json, os",
                "payload = {",
                "    'schema_version': 'cosmos3_wam_command_adapter.v1',",
                "    'status': 'completed',",
                "    'base_model': 'Cosmos3-Nano',",
                "    'rollouts': [],",
                "}",
                "with open(os.environ['BLUEPRINT_WAM_PROVIDER_OUTPUT'], 'w', encoding='utf-8') as fh:",
                "    json.dump(payload, fh)",
            ]
        ),
        encoding="utf-8",
    )

    status, payload, detail = run_provider_command(
        command_text=f"{sys.executable} {provider}",
        runtime_package_path=runtime_package,
        output_path=tmp_path / "match" / "output.json",
        substrate="cosmos3_wam",
        artifact_output_uri=None,
        timeout_seconds=30,
    )

    assert status == "completed"
    assert payload["base_model"] == "Cosmos3-Nano"
    assert detail["backbone_identity_check"]["status"] == "verified"


def test_normalize_provider_rollouts_requires_fixture_provenance_and_blocks_correlation() -> None:
    fixture_rollouts = normalize_provider_rollouts(
        payload={
            "rollouts": [
                {
                    "policy_id": "policy-1",
                    "scenario_eval_run_id": "run-1",
                    "predicted_success": True,
                    "uncertainty_score": 0.2,
                    "metrics": {"spearman": 0.9, "pearson": 0.8, "custom_metric": 1.0},
                }
            ]
        },
        substrate="fixture_wam",
        generated_at="2026-07-02T00:00:00+00:00",
    )
    fixture_rollout = fixture_rollouts[0]
    assert fixture_rollout["fixture_evaluator_only"] is True
    assert fixture_rollout["claim_boundary"]["fixture_evaluator_only"] is True
    assert "spearman" not in fixture_rollout["metrics"]
    assert "pearson" not in fixture_rollout["metrics"]
    assert fixture_rollout["metrics"]["custom_metric"] == 1.0

    model_rollouts = normalize_provider_rollouts(
        payload={
            "rollouts": [
                {
                    "policy_id": "policy-1",
                    "scenario_eval_run_id": "run-1",
                    "predicted_success": True,
                }
            ]
        },
        substrate="cosmos3_wam",
        generated_at="2026-07-02T00:00:00+00:00",
    )
    assert model_rollouts[0]["fixture_evaluator_only"] is False
    assert model_rollouts[0]["claim_boundary"]["fixture_evaluator_only"] is False
