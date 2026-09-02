"""Stdlib-only smoke for the signed camera-sweep worker overlay."""

from __future__ import annotations

import ast
from pathlib import Path

from blueprint_pipeline import native_task_arena_policy_canary_worker as worker


def test_worker_overlay_contains_the_preload_camera_contract() -> None:
    source_path = Path(worker.__file__).resolve()
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    called_attributes = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert "prepolicy_camera_gate" in worker.CellRuntime.__dataclass_fields__
    assert "set_world_poses_from_view" in called_attributes
    assert "policy_canary_runtime_observation_integrity_gate.v1" in source
    assert "not_required_for_internal_diagnostic_policy_execution" in source
    assert "policy_observation_integrity_passed" in source
    assert "official_ranking_permitted" in source
    assert "scene_promotion_permitted" in source


if __name__ == "__main__":
    test_worker_overlay_contains_the_preload_camera_contract()
