from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.native_task_arena_control_sweep_worker import (
    run_control_sweep_worker,
)


def _write(path: Path, value: dict) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_worker_routes_sealed_inputs_and_closes_app(tmp_path, monkeypatch) -> None:
    plan = {
        "schema_version": "task_evaluation_control_search_funnel_plan.v1",
        "status": "planned",
        "claim_ceiling": "development_only_control_search",
        "qualification_effect": "none_until_full_fidelity_replay",
        "candidate_index": [{"candidate_id": "candidate-0"}],
        "vector_sweep": {"appearance_mode": "omitted", "camera_mode": "disabled"},
        "shortlist": {"learned_grader_used": False},
        "full_fidelity_replay": {
            "search_result_alone_may_not_qualify_controls": True
        },
        "plan_digest": "",
    }
    plan["plan_digest"] = canonical_digest(plan, digest_field="plan_digest")
    schedule = {
        "schema_version": "task_evaluation_isaaclab_control_sweep_schedule.v1",
        "status": "scheduled",
        "plan_digest": plan["plan_digest"],
        "candidate_inventory_digest": "sha256:" + "1" * 64,
        "vector_env_count": None,
        "wave_count": None,
        "assignment_count": None,
        "boot_once_reuse_across_waves": True,
        "reset_before_every_wave": True,
        "waves": [],
        "schedule_digest": "",
    }
    inventory = {"inventory_digest": schedule["candidate_inventory_digest"]}
    packet = tmp_path / "packet"
    packet.mkdir()
    app = SimpleNamespace(closed=False)
    app.close = lambda: setattr(app, "closed", True)
    monkeypatch.setattr(
        "blueprint_pipeline.native_task_arena_control_sweep_worker."
        "validate_control_search_funnel_plan",
        lambda value: value,
    )
    monkeypatch.setattr(
        "blueprint_pipeline.native_task_arena_control_sweep_worker."
        "validate_isaaclab_control_sweep_schedule",
        lambda value, plan: value,
    )
    monkeypatch.setattr(
        "blueprint_pipeline.native_task_arena_control_sweep_worker."
        "launch_native_task_isaaclab",
        lambda receipt, device: (app, {"status": "launched"}),
    )
    observed = {}

    def execute(**kwargs):
        observed.update(kwargs)
        return {
            "schema_version": "task_evaluation_control_search_sweep_result.v1",
            "status": "completed_development_only",
            "result_digest": "sha256:" + "2" * 64,
        }

    monkeypatch.setattr(
        "blueprint_pipeline.native_task_arena_control_sweep_worker."
        "execute_isaaclab_control_sweep",
        execute,
    )
    output = tmp_path / "result.json"

    result = run_control_sweep_worker(
        plan_path=_write(tmp_path / "plan.json", plan),
        schedule_path=_write(tmp_path / "schedule.json", schedule),
        candidate_inventory_path=_write(tmp_path / "inventory.json", inventory),
        scene_plan_path=_write(tmp_path / "scene.json", {"plan_digest": "x"}),
        packet_root=packet,
        provisioning_receipt_path=tmp_path / "provisioning.json",
        output_path=output,
    )

    assert result["status"] == "completed_development_only"
    assert json.loads(output.read_text()) == result
    assert app.closed is True
    assert observed["bundle_root"] == packet.resolve()
    assert observed["plan"] == plan
    assert observed["schedule"] == schedule
