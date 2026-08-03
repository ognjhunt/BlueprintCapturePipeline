"""SkyPilot pilot lane (C4 step b): out-of-process, grant-gated, fail-closed."""

from __future__ import annotations

import ast
from pathlib import Path


from blueprint_pipeline.paid_lane_guard import load_pending_teardowns
from blueprint_pipeline.paid_resource_admission import (
    PAID_LANE_ADMISSION_SCHEMA_VERSION,
    build_paid_lane_admission,
    require_paid_resource_admission,
)
from blueprint_pipeline.skypilot_promotion_gate import (
    SKYPILOT_PROMOTION_GATES,
    evaluate_skypilot_promotion,
)
from blueprint_pipeline.skypilot_provisioner import (
    SKYPILOT_BIN_ENV,
    SKYPILOT_RESOURCE_CLASS,
    launch_disposable_vast_smoke,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _grant():
    admission = build_paid_lane_admission(
        resource_class=SKYPILOT_RESOURCE_CLASS, blockers=[]
    )
    return require_paid_resource_admission(
        admission,
        resource_class=SKYPILOT_RESOURCE_CLASS,
        expected_schema_version=PAID_LANE_ADMISSION_SCHEMA_VERSION,
    )


def _task_yaml(tmp_path: Path, *, cloud: str = "vast", cap: str = "0.40") -> Path:
    path = tmp_path / "smoke_task.yaml"
    path.write_text(
        "resources:\n"
        f"  cloud: {cloud}\n"
        f"  max_hourly_cost: {cap}\n"
        "run: |\n"
        "  echo smoke\n",
        encoding="utf-8",
    )
    return path


def _sky_env(tmp_path: Path) -> dict[str, str]:
    sky = tmp_path / "sky"
    sky.write_text("#!/bin/sh\n", encoding="utf-8")
    return {SKYPILOT_BIN_ENV: str(sky)}


class _FakeRunner:
    def __init__(self, *, launch_rc: int = 0, down_rc: int = 0) -> None:
        self.launch_rc = launch_rc
        self.down_rc = down_rc
        self.calls: list[list[str]] = []

    def __call__(self, command: list[str]):  # noqa: ANN001
        self.calls.append(list(command))

        class _Result:
            stdout = ""
            stderr = ""

        result = _Result()
        result.returncode = self.down_rc if "down" in command else self.launch_rc
        return result


def test_module_never_imports_skypilot() -> None:
    source = (
        REPO_ROOT / "src/blueprint_pipeline/skypilot_provisioner.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".")[0])
    assert "sky" not in roots
    assert "skypilot" not in roots


def test_launch_requires_grant_fail_closed(tmp_path: Path) -> None:
    runner = _FakeRunner()
    result = launch_disposable_vast_smoke(
        run_id="pilot-1",
        task_yaml_path=_task_yaml(tmp_path),
        cluster_name="bp-skypilot-pilot-1",
        env=_sky_env(tmp_path),
        runner=runner,
        vast_inventory=lambda: {"status": "passed", "instances": []},
    )
    assert result["status"] == "blocked"
    assert "skypilot_pilot_launch_disabled" in result["blockers"]
    assert result["allocation_created"] is False
    assert runner.calls == []
    assert load_pending_teardowns() == []


def test_launch_unavailable_without_pinned_binary(tmp_path: Path) -> None:
    runner = _FakeRunner()
    result = launch_disposable_vast_smoke(
        run_id="pilot-1",
        task_yaml_path=_task_yaml(tmp_path),
        cluster_name="bp-skypilot-pilot-1",
        paid_resource_admission_grant=_grant(),
        env={},
        runner=runner,
        vast_inventory=lambda: {"status": "passed", "instances": []},
    )
    assert result["status"] == "unavailable"
    assert "skypilot_bin_missing" in result["blockers"]
    assert result["allocation_created"] is False
    assert runner.calls == []
    assert load_pending_teardowns() == []


def test_task_constraints_fail_closed(tmp_path: Path) -> None:
    runner = _FakeRunner()
    wrong_cloud = launch_disposable_vast_smoke(
        run_id="pilot-1",
        task_yaml_path=_task_yaml(tmp_path, cloud="runpod"),
        cluster_name="bp-skypilot-pilot-1",
        paid_resource_admission_grant=_grant(),
        env=_sky_env(tmp_path),
        runner=runner,
        vast_inventory=lambda: {"status": "passed", "instances": []},
    )
    assert wrong_cloud["status"] == "blocked_constraints"
    assert "skypilot_task_cloud_not_vast" in wrong_cloud["blockers"]
    assert runner.calls == []

    no_cap_yaml = tmp_path / "no_cap.yaml"
    no_cap_yaml.write_text("resources:\n  cloud: vast\nrun: echo x\n", encoding="utf-8")
    no_cap = launch_disposable_vast_smoke(
        run_id="pilot-1",
        task_yaml_path=no_cap_yaml,
        cluster_name="bp-skypilot-pilot-1",
        paid_resource_admission_grant=_grant(),
        env=_sky_env(tmp_path),
        runner=runner,
        vast_inventory=lambda: {"status": "passed", "instances": []},
    )
    assert no_cap["status"] == "blocked_constraints"
    assert "skypilot_task_max_hourly_cost_missing" in no_cap["blockers"]
    assert runner.calls == []


def test_launch_down_and_provider_zero_closes_teardown(tmp_path: Path) -> None:
    runner = _FakeRunner()
    result = launch_disposable_vast_smoke(
        run_id="pilot-1",
        task_yaml_path=_task_yaml(tmp_path),
        cluster_name="bp-skypilot-pilot-1",
        paid_resource_admission_grant=_grant(),
        env=_sky_env(tmp_path),
        runner=runner,
        vast_inventory=lambda: {"status": "passed", "instances": []},
    )
    assert result["status"] == "completed_provider_zero"
    assert result["allocation_created"] is True
    assert result["teardown_proof"]["status"] == "PASS"
    assert (
        result["teardown_proof"]["provider_terminal_status_source"] == "provider_api"
    )
    assert load_pending_teardowns() == []
    closed = load_pending_teardowns(include_closed=True)
    assert len(closed) == 1
    assert closed[0]["status"] == "closed"
    launch_call = runner.calls[0]
    assert "--down" in launch_call
    assert any("down" in call for call in runner.calls[1:])


def test_down_failure_with_live_instance_keeps_pending_open(tmp_path: Path) -> None:
    runner = _FakeRunner(down_rc=1)
    result = launch_disposable_vast_smoke(
        run_id="pilot-2",
        task_yaml_path=_task_yaml(tmp_path),
        cluster_name="bp-skypilot-pilot-2",
        paid_resource_admission_grant=_grant(),
        env=_sky_env(tmp_path),
        runner=runner,
        vast_inventory=lambda: {
            "status": "passed",
            "instances": [{"label": "bp-skypilot-pilot-2", "actual_status": "running"}],
        },
    )
    assert result["status"] == "launched_teardown_unproven"
    assert result["teardown_proof"]["status"] == "FAIL"
    assert result["teardown_proof"]["open_billing_risk"] is True
    open_records = load_pending_teardowns()
    assert len(open_records) == 1
    assert open_records[0]["status"] == "open"


def test_missing_inventory_is_teardown_unproven(tmp_path: Path) -> None:
    result = launch_disposable_vast_smoke(
        run_id="pilot-3",
        task_yaml_path=_task_yaml(tmp_path),
        cluster_name="bp-skypilot-pilot-3",
        paid_resource_admission_grant=_grant(),
        env=_sky_env(tmp_path),
        runner=_FakeRunner(),
        vast_inventory=None,
    )
    assert result["status"] == "launched_teardown_unproven"
    assert result["teardown_proof"]["status"] == "FAIL"
    assert len(load_pending_teardowns()) == 1


def test_ambiguous_launch_marks_registry(tmp_path: Path) -> None:
    result = launch_disposable_vast_smoke(
        run_id="pilot-4",
        task_yaml_path=_task_yaml(tmp_path),
        cluster_name="bp-skypilot-pilot-4",
        paid_resource_admission_grant=_grant(),
        env=_sky_env(tmp_path),
        runner=_FakeRunner(launch_rc=1),
        vast_inventory=lambda: {"status": "passed", "instances": []},
    )
    assert result["allocation_outcome_ambiguous"] is True
    records = load_pending_teardowns(include_closed=True)
    assert len(records) == 1
    assert records[0]["allocation_outcome_ambiguous"] is True


def test_promotion_gates_are_all_required() -> None:
    assert len(SKYPILOT_PROMOTION_GATES) == 11
    evidence = {
        gate: {"proven": True, "evidence_path": f"evidence/{gate}.json"}
        for gate in SKYPILOT_PROMOTION_GATES
    }
    promotable = evaluate_skypilot_promotion(evidence)
    assert promotable["status"] == "promotable"
    assert promotable["promoted_scope_allowed"] == ["vast_disposable_smoke"]

    evidence.pop("orphan_recovery_after_skypilot_state_loss")
    blocked = evaluate_skypilot_promotion(evidence)
    assert blocked["status"] == "not_promotable"
    assert (
        "skypilot_promotion_gate_unproven:orphan_recovery_after_skypilot_state_loss"
        in blocked["blockers"]
    )

    unproven = {
        gate: {"proven": gate != "hourly_price_cap_enforcement", "evidence_path": "e"}
        for gate in SKYPILOT_PROMOTION_GATES
    }
    partial = evaluate_skypilot_promotion(unproven)
    assert partial["status"] == "not_promotable"
