from __future__ import annotations

import json
from pathlib import Path

from blueprint_pipeline.task_evaluation_remote_isaaclab_control_sweep import (
    RemoteIsaacLabControlSweepRunner,
)


def test_remote_runner_reuses_warm_instance_without_allocation(
    tmp_path: Path, monkeypatch
) -> None:
    calls = []
    expected = {
        "schema_version": "task_evaluation_control_search_sweep_result.v1",
        "status": "completed_development_only",
    }
    monkeypatch.setattr(
        "blueprint_pipeline.task_evaluation_remote_isaaclab_control_sweep."
        "_enroll_warm_host_key",
        lambda *args, **kwargs: {
            "status": "enrolled",
            "known_hosts_file": str(tmp_path / "known_hosts"),
        },
    )

    def ssh(**kwargs):
        calls.append(kwargs)
        argv = kwargs["remote_argv"]
        return {
            "status": "completed",
            "stdout": json.dumps(expected) if argv[:2] == ["cat", "--"] else "",
            "stderr": "",
            "blockers": [],
        }

    monkeypatch.setattr(
        "blueprint_pipeline.task_evaluation_remote_isaaclab_control_sweep."
        "_run_warm_ssh",
        ssh,
    )
    monkeypatch.setattr(
        "blueprint_pipeline.task_evaluation_remote_isaaclab_control_sweep."
        "validate_control_search_funnel_plan",
        lambda value: value,
    )
    monkeypatch.setattr(
        "blueprint_pipeline.task_evaluation_remote_isaaclab_control_sweep."
        "validate_isaaclab_control_sweep_schedule",
        lambda value, plan: value,
    )
    monkeypatch.setattr(
        "blueprint_pipeline.task_evaluation_remote_isaaclab_control_sweep."
        "validate_control_search_sweep_result",
        lambda value, plan: dict(value),
    )
    runner = RemoteIsaacLabControlSweepRunner(
        warm_session={
            "status": "ready",
            "continuing_spend": True,
            "remote_work_dir": "/workspace",
        },
        local_transport_root=tmp_path / "transport",
    )

    result = runner.execute(
        plan={"plan_digest": "sha256:" + "1" * 64},
        schedule={
            "schedule_digest": "sha256:" + "2" * 64,
            "candidate_inventory_digest": "sha256:" + "3" * 64,
        },
        candidate_inventory={"inventory_digest": "sha256:" + "3" * 64},
    )

    assert result == expected
    assert len(calls) == 5
    worker = calls[3]["remote_argv"]
    assert worker[:2] == ["/bin/bash", "-c"]
    assert "native_task_arena_control_sweep_worker" in worker[2]
    assert all("create" not in " ".join(call["remote_argv"]) for call in calls)
