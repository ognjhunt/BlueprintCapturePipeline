from __future__ import annotations

from pathlib import Path

import pytest

from blueprint_pipeline import vast_provider_adapter as adapter


def test_vast_module_cli_preserves_fail_closed_argument_mapping(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    def run(**kwargs):
        calls.append(kwargs)
        return {
            "status": "dry_run_ready",
            "vast_instance_ids": [],
            "blockers": [],
        }

    monkeypatch.setattr(adapter, "run_vast_provider_adapter", run)
    job = tmp_path / "job"
    avoidlist = tmp_path / "avoidlist.json"
    ledger = tmp_path / "ledger.json"

    exit_code = adapter.main(
        [
            "--job-dir",
            str(job),
            "--mode",
            "dry-run",
            "--allow-vast-api-call",
            "--allow-vast-instance-launch",
            "--allowed-machine-id",
            "41",
            "--allowed-machine-id",
            "42",
            "--machine-avoidlist",
            str(avoidlist),
            "--session-budget-ledger",
            str(ledger),
            "--provider-bundle-kind",
            "task_evaluation_scene_configuration",
        ]
    )

    assert exit_code == 0
    assert len(calls) == 1
    call = calls[0]
    assert call["allow_instance_launch"] is True
    assert call["allowed_machine_ids"] == ["41", "42"]
    assert call["machine_avoidlist_path"] == str(avoidlist)
    assert call["session_budget_ledger_path"] == str(ledger)
    assert call["provider_bundle_kind"] == "task_evaluation_scene_configuration"
    assert "allow_vast_instance_launch" not in call
    assert "allowed_machine_id" not in call
    assert "machine_avoidlist" not in call
    assert "session_budget_ledger" not in call
    output = capsys.readouterr().out
    assert f"result={job.resolve() / 'vast_provider_adapter_result.json'}" in output
    assert "status=dry_run_ready" in output


def test_vast_module_cli_still_blocks_direct_live_mutation(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        adapter,
        "run_vast_provider_adapter",
        lambda **_kwargs: pytest.fail("direct live CLI reached the adapter"),
    )

    assert (
        adapter.main(
            [
                "--job-dir",
                str(tmp_path / "job"),
                "--mode",
                "live-startup-probe",
                "--allow-vast-api-call",
                "--allow-vast-instance-launch",
            ]
        )
        == 2
    )
    assert "legacy_vast_provider_mutation_cli_disabled" in capsys.readouterr().err
