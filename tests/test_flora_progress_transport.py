"""Drive the canonical log watchdog with count-only FLORA progress evidence."""

from __future__ import annotations

from pathlib import Path

import pytest

from blueprint_pipeline import vast_provider_adapter as vpa
from blueprint_pipeline.flora_runtime_progress import progress_marker


_TRANSPORT_FINISHED = "BLUEPRINT_VAST_PROVIDER_BUNDLE_COMPLETED_OR_BLOCKED"


def _observe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    snapshots: list[str],
    *,
    no_progress_seconds: int = 25,
    max_wait_seconds: int = 999,
    remote_progress_probe=None,
):
    """Advance fake wall time only when the canonical loop reads another tail."""
    clock = {"now": 0.0, "reads": 0}

    def fetch(_url: str, **_kwargs) -> str:
        clock["now"] += 10.0
        index = min(clock["reads"], len(snapshots) - 1)
        clock["reads"] += 1
        return snapshots[index]

    monkeypatch.setattr(vpa.time, "monotonic", lambda: clock["now"])
    monkeypatch.setattr(vpa.time, "sleep", lambda *_args: None)
    monkeypatch.setattr(
        vpa, "_api_json",
        lambda **_kwargs: (200, {"result_url": "https://example.invalid/log.txt"}),
    )
    monkeypatch.setattr(vpa, "_fetch_text", fetch)
    monkeypatch.setattr(
        vpa, "_instance_liveness",
        lambda **_kwargs: {"observed": True, "status": "running", "exited": False},
    )
    return vpa._request_logs_and_fetch(
        instance_id=123,
        api_key="test-only-secret",
        output_log_path=tmp_path / "onstart.log",
        secret_values=["test-only-secret"],
        wait_seconds=0,
        retry_interval_seconds=1,
        max_wait_seconds=max_wait_seconds,
        success_markers=[_TRANSPORT_FINISHED],
        no_progress_seconds=no_progress_seconds,
        remote_progress_probe=remote_progress_probe,
    )


@pytest.mark.parametrize("next_record,expected_polls", [(False, 4), (True, 6)])
def test_only_a_new_flora_record_resets_canonical_no_progress_timer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, next_record: bool, expected_polls: int,
) -> None:
    first = progress_marker("rubric", 1, 20)
    later = progress_marker("rubric", 2 if next_record else 1, 20)
    result = _observe(tmp_path, monkeypatch, [
        first,
        first + "\nAccepted publickey for root\n",
        later,
        later + "\nDisconnected from user root\n",
    ])
    attempts = result["log_poll_attempts"]
    assert result["break_reason"] == "no_log_progress_timeout"
    assert result["no_progress_timeout_seconds"] == 25
    assert len(attempts) == expected_polls
    assert attempts[0]["flora_progress_observed"] is True
    assert attempts[1]["progress_observed"] is False
    assert attempts[2]["flora_progress_observed"] is next_record
    assert attempts[2]["no_progress_elapsed_seconds"] == (0.0 if next_record else 20.0)
    assert attempts[-1]["no_progress_elapsed_seconds"] == 30.0
    assert all(row["success_marker_found"] is False for row in attempts)


def test_flora_cursor_survives_duplicate_reordered_regressed_and_missing_tails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    newest = progress_marker("rubric", 2, 20)
    older = progress_marker("rubric", 1, 20)
    terminal = progress_marker("terminal", 20, 20)
    remote_calls = {"count": 0}

    def remote_probe(_connection):
        remote_calls["count"] += 1
        return {
            "status": "observed",
            "milestones": [
                "BLUEPRINT_POLICY_CANARY_PROGRESS:cell="
                + str(remote_calls["count"])
                + ":stage=static_preflight_passed"
            ],
        }

    result = _observe(tmp_path, monkeypatch, [
        "\n".join([terminal, newest]),
        "\n".join([newest, terminal, newest, "Accepted publickey for root"]),
        "\n".join([older, terminal, "Disconnected from user root"]),
        "BLUEPRINT_WAM_RUNTIME_PHASE:worker:unexpected:started\n",
        "BLUEPRINT_UNRELATED_PROGRESS:records=999\nsshd session opened\n",
        "BLUEPRINT_WAM_RUNTIME_PHASE:worker:unexpected:started\n"
        "BLUEPRINT_WAM_RUNTIME_PHASE:worker:unexpected:completed\n",
    ], no_progress_seconds=45, remote_progress_probe=remote_probe)
    attempts = result["log_poll_attempts"]
    assert result["break_reason"] == "no_log_progress_timeout"
    assert len(attempts) == 6
    assert attempts[0]["progress_observed"] is True
    assert all(row["progress_observed"] is False for row in attempts[1:])
    assert all(row["flora_progress_cursor"] == [4, 2] for row in attempts)
    # Even another transport's advancing evidence cannot override typed FLORA tracking.
    assert any(row["remote_progress_observed"] for row in attempts[1:])
    assert any(row["runtime_phase_progress_observed"] for row in attempts[1:])
    assert attempts[-1]["no_progress_elapsed_seconds"] == 50.0


def test_startup_preflight_and_mode_counts_do_not_claim_completed_benchmark(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    markers = [
        progress_marker("startup", 1, 5),
        progress_marker("startup", 5, 5),
        progress_marker("preflight", 0, 20),
        progress_marker("preflight", 20, 20),
        progress_marker("terminal", 0, 20),
        progress_marker("terminal", 20, 20),
        progress_marker("dense", 0, 20),
        progress_marker("dense", 20, 20),
        progress_marker("rubric", 0, 20),
        progress_marker("rubric", 20, 20),
    ]
    result = _observe(tmp_path, monkeypatch, markers, no_progress_seconds=15)
    attempts = result["log_poll_attempts"]
    assert result["break_reason"] == "no_log_progress_timeout"
    assert all(row["flora_progress_observed"] for row in attempts[:len(markers)])
    assert attempts[-1]["flora_progress_cursor"] == [4, 20]
    assert all(row["success_marker_found"] is False for row in attempts)
    assert result["observed_blueprint_marker_lines"] == markers
    assert all(line.endswith(":benchmark_complete=false") for line in markers)
    assert not any("benchmark_report" in line or _TRANSPORT_FINISHED in line for line in markers)


def test_actual_transport_completion_remains_separate_from_flora_progress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    partial = progress_marker("rubric", 3, 20)
    result = _observe(tmp_path, monkeypatch, [partial, partial, _TRANSPORT_FINISHED])
    attempts = result["log_poll_attempts"]
    assert result["break_reason"] == "success_marker_found"
    assert attempts[-1]["success_marker_found"] is True
    assert attempts[-1]["flora_progress_observed"] is False
    assert attempts[-1]["flora_progress_cursor"] == [4, 3]
    # The transport's completed-or-blocked signal is not a completed scientific result.
    assert "benchmark_complete=true" not in "\n".join(result["observed_blueprint_marker_lines"])


def test_advancing_flora_records_never_extend_existing_maximum_wait(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _observe(tmp_path, monkeypatch,
        [progress_marker("rubric", count, 20) for count in range(1, 10)],
        no_progress_seconds=25, max_wait_seconds=35)
    attempts = result["log_poll_attempts"]
    assert result["break_reason"] == "max_wait_deadline_reached"
    assert len(attempts) == 4
    assert all(row["flora_progress_observed"] for row in attempts)
    assert all(row["no_progress_elapsed_seconds"] == 0.0 for row in attempts)
    assert result["no_progress_timeout_reached"] is False
