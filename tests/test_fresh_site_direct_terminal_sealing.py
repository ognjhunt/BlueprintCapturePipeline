from __future__ import annotations

import json
from pathlib import Path

import pytest

from blueprint_pipeline.nvidia_warehouse_native_camera_gpu_admission import (
    _seal_terminal as seal_native_camera_terminal,
)
from blueprint_pipeline.openpi_policy_ranking_runpod import (
    _seal_terminal as seal_diagnostic_terminal,
)
from blueprint_pipeline.task_evaluation_launch_dispatcher import _terminal_evidence


SEALERS = (seal_diagnostic_terminal, seal_native_camera_terminal)


def _completed_result(*, absence_confirmed: bool = True) -> dict[str, object]:
    return {
        "status": "completed",
        "blockers": [],
        "instance_id": "42",
        "monitor": {
            "status": "completed",
            "continuing_spend": False,
            "teardown": {
                "status": "provider_terminal",
                "completed_at": "2026-08-16T22:45:00+00:00",
                "provider_absence_confirmed": absence_confirmed,
                "terminations": [
                    {"instance_id": "42", "status": "deleted", "http": 200}
                ],
            },
        },
        # These older direct transports expose only the legacy spelling. The
        # live terminal contract consumes the normalized spelling.
        "continuing_spend": False,
    }


def _terminal_profile(result_path: Path) -> dict[str, object]:
    return {
        "terminal_contract": {
            "result_path": str(result_path),
            "success_statuses": ["completed"],
            "required_values": {
                "continuing_spend_from_this_run": False,
                "retry_cap": 0,
            },
            "required_path_fields": [
                "artifact_manifest_path",
                "teardown_manifest_path",
            ],
        }
    }


@pytest.mark.parametrize("sealer", SEALERS)
def test_direct_fresh_site_terminal_satisfies_the_live_dispatcher_contract(
    tmp_path: Path, sealer
) -> None:
    output = tmp_path / "allocator" / "result.json"
    output.parent.mkdir(parents=True)
    for name in (
        "openpi_policy_ranking_provider_output.zip",
        "openpi_policy_ranking_output_validation.json",
        "groot_oscar_runpod_canary_watchdog.json",
        "nvidia_warehouse_native_camera_provider_output.zip",
        "nvidia_warehouse_native_camera_output_validation.json",
        "nvidia_warehouse_native_camera_monitor.json",
    ):
        (output.parent / name).write_bytes(b"retained-direct-provider-evidence")

    # Match the real call sequence: the direct provider result is written once
    # while the monitor still owns the live instance, then again after teardown.
    submitted = sealer(
        {"status": "submitted", "instance_id": "42", "continuing_spend": True},
        output,
    )
    assert submitted["artifact_manifest_path"] is None
    assert not (output.parent / "artifact_manifest.json").exists()

    sealed = sealer(_completed_result(), output)
    output.write_text(json.dumps(sealed), encoding="utf-8")
    evidence = _terminal_evidence(
        _terminal_profile(output), execute=True, run_root=tmp_path
    )

    assert sealed["retry_cap"] == 0
    assert sealed["continuing_spend_from_this_run"] is False
    assert Path(str(sealed["artifact_manifest_path"])).is_file()
    assert Path(str(sealed["teardown_manifest_path"])).is_file()
    assert evidence["status"] == "passed"
    assert evidence["blockers"] == []

    manifest = json.loads(
        Path(str(sealed["artifact_manifest_path"])).read_text(encoding="utf-8")
    )
    assert "provider_runtime_output" in manifest["observed_roles"]
    assert "provider_runtime_validation" in manifest["observed_roles"]

    teardown = json.loads(
        Path(str(sealed["teardown_manifest_path"])).read_text(encoding="utf-8")
    )
    assert teardown["status"] == "completed"
    assert teardown["vast_instance_ids"] == [42]
    assert teardown["continuing_spend_from_this_run"] is False


@pytest.mark.parametrize("sealer", SEALERS)
def test_direct_fresh_site_terminal_cannot_invent_teardown_absence(
    tmp_path: Path, sealer
) -> None:
    output = tmp_path / "allocator" / "result.json"
    output.parent.mkdir(parents=True)

    sealed = sealer(_completed_result(absence_confirmed=False), output)
    output.write_text(json.dumps(sealed), encoding="utf-8")
    evidence = _terminal_evidence(
        _terminal_profile(output), execute=True, run_root=tmp_path
    )

    assert sealed["status"] == "completed"
    assert sealed["teardown_manifest_path"] is None
    assert evidence["status"] == "blocked"
    assert "allocator_terminal_artifact_missing:teardown_manifest_path" in evidence[
        "blockers"
    ]


@pytest.mark.parametrize("sealer", SEALERS)
def test_direct_fresh_site_terminal_preserves_a_nonzero_retry_claim(
    tmp_path: Path, sealer
) -> None:
    output = tmp_path / "allocator" / "result.json"
    output.parent.mkdir(parents=True)
    result = _completed_result()
    result["retry_cap"] = 1

    sealed = sealer(result, output)
    output.write_text(json.dumps(sealed), encoding="utf-8")
    evidence = _terminal_evidence(
        _terminal_profile(output), execute=True, run_root=tmp_path
    )

    assert sealed["retry_cap"] == 1
    assert evidence["status"] == "blocked"
    assert "allocator_terminal_value_mismatch:retry_cap" in evidence["blockers"]
