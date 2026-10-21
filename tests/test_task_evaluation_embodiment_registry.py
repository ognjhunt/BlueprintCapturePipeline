"""Only an integrated robot setup may run; every other listed robot says why."""

from __future__ import annotations

import dataclasses

from blueprint_pipeline import task_evaluation_embodiment_registry as registry


def test_only_the_integrated_droid_franka_runs_and_every_other_robot_names_its_gap() -> None:
    assert registry.embodiment_registry_violations() == []
    runnable = [row.embodiment_id for row in registry.EMBODIMENT_REGISTRY if row.available]
    assert runnable == ["droid_franka_robotiq_2f85"]
    for row in registry.EMBODIMENT_REGISTRY:
        if not row.available:
            assert row.reason and row.missing
    for robot in ("universal_robots_ur5e", "fanuc_crx_10ia", "abb_gofa_crb15000", "kinova_gen3"):
        assert registry.embodiment(robot).status == "not_integrated"


def test_flipping_a_robot_to_runnable_without_integration_is_caught(monkeypatch) -> None:
    ur5e = registry.embodiment("universal_robots_ur5e")
    monkeypatch.setattr(registry, "EMBODIMENT_REGISTRY", (dataclasses.replace(ur5e, status="runnable"),))
    assert registry.embodiment_registry_violations() == ["runnable_embodiment_incomplete:universal_robots_ur5e"]
