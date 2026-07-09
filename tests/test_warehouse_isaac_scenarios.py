from __future__ import annotations

import json
from pathlib import Path

from blueprint_pipeline.warehouse_isaac_scenarios import (
    WAREHOUSE_ISAAC_SCENARIO_MATRIX_SCHEMA_VERSION,
    build_warehouse_isaac_scenarios,
)


def test_build_warehouse_isaac_scenarios_writes_fixture_backed_matrix(tmp_path: Path) -> None:
    fixture = tmp_path / "warehouse_task_min"
    asset = fixture / "assets" / "warehouse_min.splat"
    asset.parent.mkdir(parents=True)
    asset.write_text("fixture warehouse splat", encoding="utf-8")

    result = build_warehouse_isaac_scenarios(
        fixture_root=fixture,
        output_dir=tmp_path / "out",
        generated_at="2026-07-08T00:00:00+00:00",
    )

    assert result["status"] == "ready_for_generic_isaac_site_3dgs_eval"
    assert result["site_type"] == "warehouse"
    assert result["blockers"] == []
    assert "isaac_g1_site_3dgs_realistic_eval" in result["recommended_command"]
    assert result["claim_boundary"]["not_a_lightwheel_kitchen_claim"] is True

    matrix_path = Path(result["artifact_paths"]["scenario_eval_matrix"])
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    assert matrix["schema_version"] == WAREHOUSE_ISAAC_SCENARIO_MATRIX_SCHEMA_VERSION
    assert matrix["site_type"] == "warehouse"
    assert matrix["scenario_eval_run_count"] == 3
    assert {row["site_type"] for row in matrix["runs"]} == {"warehouse"}
    assert {row["scenario_family"] for row in matrix["runs"]} == {
        "warehouse_articulation_review",
        "warehouse_line_side_delivery",
        "warehouse_material_handling",
    }
    assert all(
        row["claim_boundary"]["requires_generic_isaac_site_3dgs_execution"]
        for row in matrix["runs"]
    )
    assert Path(result["artifact_paths"]["handoff_manifest"]).is_file()


def test_build_warehouse_isaac_scenarios_fails_closed_when_asset_missing(tmp_path: Path) -> None:
    result = build_warehouse_isaac_scenarios(
        fixture_root=tmp_path / "missing-fixture",
        output_dir=tmp_path / "out",
        generated_at="2026-07-08T00:00:00+00:00",
    )

    assert result["status"] == "blocked"
    assert "warehouse_fixture_root_missing" in result["blockers"]
    assert "warehouse_scene_asset_missing" in result["blockers"]
    assert result["claim_boundary"]["runtime_execution_requires_generic_isaac_provider_or_local_runtime"] is True
