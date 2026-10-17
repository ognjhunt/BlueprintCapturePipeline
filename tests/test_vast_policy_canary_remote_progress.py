from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import subprocess

import pytest

from blueprint_pipeline import vast_policy_canary_remote_progress as progress


@pytest.mark.parametrize("maximum_cells", [10, 200])
def test_remote_progress_keeps_only_fixed_milestones(tmp_path: Path, monkeypatch, maximum_cells) -> None:
    identity = tmp_path / "identity"
    identity.write_text("test-only", encoding="ascii")
    known_hosts = tmp_path / "known_hosts"
    known_hosts.write_text("test-only", encoding="ascii")
    monkeypatch.setattr(progress, "_identity_file", lambda: identity)
    monkeypatch.setattr(
        progress, "enroll_vast_ssh_host_key",
        lambda *_args, **_kwargs: {
            "status": "enrolled", "known_hosts_file": str(known_hosts),
        },
    )
    monkeypatch.setattr(
        progress, "_validated_vast_known_hosts_pin",
        lambda *_args, **_kwargs: (known_hosts, "a" * 64),
    )
    monkeypatch.setattr(progress, "_ssh_command", lambda **_kwargs: ["ssh"])
    monkeypatch.setattr(
        progress.subprocess, "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0,
            stdout=(
                b"REMOTE_STAGE:runtime_source_receipt\n"
                b"BLUEPRINT_POLICY_CANARY_PROGRESS:cell=0:stage=static_preflight_passed\n"
                b"raw secret value must never be retained\n"
                b"BLUEPRINT_POLICY_CANARY_PROGRESS:cell=99:stage=policy_loaded\n"
                b"BLUEPRINT_POLICY_CANARY_PROGRESS:cell=199:stage=episode_completed\n"
                b"BLUEPRINT_POLICY_CANARY_PROGRESS:cell=200:stage=episode_completed\n"
            ),
        ),
    )
    result = progress.probe_policy_canary_remote_progress(
        {"ssh_host": "ssh.vast.ai", "ssh_port": 1234}, attempt_dir=tmp_path,
        maximum_cell_count=maximum_cells,
    )
    assert result["status"] == "observed"
    assert result["milestones"] == [
        "REMOTE_STAGE:runtime_source_receipt",
        "BLUEPRINT_POLICY_CANARY_PROGRESS:cell=0:stage=static_preflight_passed",
    ] + (["BLUEPRINT_POLICY_CANARY_PROGRESS:cell=99:stage=policy_loaded",
          "BLUEPRINT_POLICY_CANARY_PROGRESS:cell=199:stage=episode_completed"] if maximum_cells == 200 else [])
    assert "secret" not in str(result)
    assert result["raw_remote_output_recorded"] is False


def test_remote_script_keeps_latest_cell_199_inside_its_output_bound(tmp_path):
    root = tmp_path / "adp_arena_provider_bundle/runtime_output/cell_runs"
    for index in range(200):
        directory = root / f"{index:02d}"
        directory.mkdir(parents=True)
        (directory / "cell_progress.log").write_text("".join(
            f"BLUEPRINT_POLICY_CANARY_PROGRESS:cell={index}:stage={stage}\n"
            for stage in sorted(progress._CELL_STAGES)))
    script = progress._REMOTE_SCRIPT.replace("/workspace /tmp/blueprint_vast_work", str(tmp_path))
    result = subprocess.run(["sh", "-c", script], capture_output=True, check=True)
    assert len(result.stdout) <= progress._MAX_OUTPUT_BYTES
    assert b"cell=199:stage=episode_completed" in result.stdout
    assert len(result.stdout.splitlines()) <= 100


def test_remote_progress_requires_pinned_ssh(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(progress, "_identity_file", lambda: None)
    result = progress.probe_policy_canary_remote_progress(
        {"ssh_host": "ssh.vast.ai", "ssh_port": 1234}, attempt_dir=tmp_path,
    )
    assert result == {
        "status": "unavailable", "milestones": [],
        "reason": "endpoint_or_identity_unavailable",
    }


def test_remote_cursor_is_ordered_by_execution_and_cannot_advance_with_replays():
    def cursor(*rows):
        return progress.remote_progress_cursor({'status': 'observed', 'milestones': list(rows)})
    first = 'BLUEPRINT_POLICY_CANARY_PROGRESS:cell=10:stage=episode_completed'
    next_cell = 'BLUEPRINT_POLICY_CANARY_PROGRESS:cell=11:stage=static_preflight_passed'
    late = 'BLUEPRINT_POLICY_CANARY_PROGRESS:cell=199:stage=episode_completed'
    assert cursor(first) < cursor(next_cell) < cursor(late)
    assert cursor(first, next_cell) == cursor(next_cell, first, first)
    assert cursor(late, first) == cursor(late)
    assert cursor('BLUEPRINT_POLICY_CANARY_PROGRESS:cell=200:stage=episode_completed') is None
    assert cursor('BLUEPRINT_POLICY_CANARY_PROGRESS:cell=199:stage=unknown') is None
    assert cursor('raw noise') is None
