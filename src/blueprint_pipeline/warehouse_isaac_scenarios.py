"""Fixture-backed warehouse scenario family for the generic Isaac/3DGS lane."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .common import ensure_dir, utc_now_iso, write_json


WAREHOUSE_ISAAC_SCENARIOS_SCHEMA_VERSION = "warehouse_isaac_scenarios.v1"
WAREHOUSE_ISAAC_SCENARIO_MATRIX_SCHEMA_VERSION = "warehouse_isaac_scenario_eval_matrix.v1"
DEFAULT_FIXTURE_RELATIVE = "tests/fixtures/warehouse_task_min"
DEFAULT_OUTPUT_RELATIVE = "pipeline/warehouse_isaac_scenarios"


WAREHOUSE_SCENARIO_DEFINITIONS: tuple[dict[str, Any], ...] = (
    {
        "scenario_id": "warehouse_tote_to_bin",
        "task_id": "carry_object_to_drop_zone",
        "task_text": "Carry the blue tote into the target bin zone.",
        "scenario_family": "warehouse_material_handling",
        "variation_ids": ["object_rotation", "narrow_approach_angle", "missing_label"],
        "camera_ids": ["head_pov", "wrist", "task_focus", "third_person"],
        "route_waypoints": [[0.0, 0.0, 0.8], [1.0, 1.2, 0.8], [1.02, 2.01, 0.8]],
        "success_proxy": "industrial_containment",
        "target_object_id": "target_bin_01",
    },
    {
        "scenario_id": "warehouse_packout_station_arrival",
        "task_id": "walk_to_target",
        "task_text": "Navigate from the aisle entry to the packout station.",
        "scenario_family": "warehouse_line_side_delivery",
        "variation_ids": ["blocked_path", "human_crossing", "glare"],
        "camera_ids": ["head_pov", "torso", "overhead"],
        "route_waypoints": [[0.0, 0.0, 0.8], [0.6, 0.8, 0.8], [1.4, 1.6, 0.8]],
        "success_proxy": "industrial_zone_arrival_or_transfer",
        "target_object_id": "packout_station_01",
    },
    {
        "scenario_id": "warehouse_dock_door_openable_check",
        "task_id": "desk_object_contact_check",
        "task_text": "Inspect the dock door handle and clearance zone.",
        "scenario_family": "warehouse_articulation_review",
        "variation_ids": ["occlusion", "wrong_object_nearby", "narrow_approach_angle"],
        "camera_ids": ["head_pov", "task_focus", "third_person"],
        "route_waypoints": [[0.0, 0.0, 0.8], [0.4, 0.5, 0.8]],
        "success_proxy": "industrial_placement_at_target",
        "target_object_id": "dock_door_01",
    },
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _string(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _default_fixture_root() -> Path:
    return _repo_root() / DEFAULT_FIXTURE_RELATIVE


def _default_scene_asset(fixture_root: Path) -> Path:
    return fixture_root / "assets" / "warehouse_min.splat"


def _default_output_dir(fixture_root: Path) -> Path:
    return fixture_root / DEFAULT_OUTPUT_RELATIVE


def _scenario_run_rows(
    *,
    fixture_root: Path,
    scene_asset: Path,
    generated_at: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, definition in enumerate(WAREHOUSE_SCENARIO_DEFINITIONS, start=1):
        scenario_id = _string(definition["scenario_id"])
        rows.append(
            {
                "scenario_eval_run_id": f"warehouse_isaac_{index:02d}_{scenario_id}",
                "episode_id": f"warehouse_episode_{index:02d}_{scenario_id}",
                "scenario_id": scenario_id,
                "task_id": definition["task_id"],
                "task_text": definition["task_text"],
                "site_type": "warehouse",
                "scenario_family": definition["scenario_family"],
                "variation_ids": list(definition["variation_ids"]),
                "spawn_id": "warehouse_fixture_entry",
                "camera_ids": list(definition["camera_ids"]),
                "route_waypoints": list(definition["route_waypoints"]),
                "success_proxy": definition["success_proxy"],
                "target_object_id": definition["target_object_id"],
                "source_fixture_root": str(fixture_root),
                "scene_asset": str(scene_asset),
                "generated_at": generated_at,
                "claim_boundary": {
                    "warehouse_scenario_family_built": True,
                    "requires_generic_isaac_site_3dgs_execution": True,
                    "scenario_matrix_is_not_isaac_runtime_success": True,
                    "fixture_splat_is_support_asset_not_raw_customer_capture": True,
                },
            }
        )
    return rows


def build_warehouse_isaac_scenarios(
    *,
    fixture_root: str | Path | None = None,
    scene_asset: str | Path | None = None,
    output_dir: str | Path | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Write a warehouse scenario family for ``isaac_g1_site_3dgs_realistic_eval``.

    The output is intentionally a scenario matrix and handoff manifest, not a
    claim that Isaac Sim ran. The generic Isaac/3DGS job consumes the matrix via
    ``--scenario-eval-matrix`` when provider/runtime execution is explicitly run.
    """

    generated = generated_at or utc_now_iso()
    fixture = Path(fixture_root).expanduser() if fixture_root else _default_fixture_root()
    fixture = fixture.resolve()
    asset = Path(scene_asset).expanduser() if scene_asset else _default_scene_asset(fixture)
    asset = asset.resolve()
    out_dir = Path(output_dir).expanduser() if output_dir else _default_output_dir(fixture)
    out_dir = out_dir.resolve()
    ensure_dir(out_dir)

    blockers: list[str] = []
    if not fixture.is_dir():
        blockers.append("warehouse_fixture_root_missing")
    if not asset.is_file():
        blockers.append("warehouse_scene_asset_missing")

    runs = _scenario_run_rows(fixture_root=fixture, scene_asset=asset, generated_at=generated)
    matrix = {
        "schema_version": WAREHOUSE_ISAAC_SCENARIO_MATRIX_SCHEMA_VERSION,
        "generated_at": generated,
        "site_type": "warehouse",
        "scene_family": "warehouse_fixture_min",
        "scenario_eval_run_count": len(runs),
        "runs": runs,
        "claim_boundary": {
            "warehouse_isaac_family_exists": True,
            "matrix_can_feed_generic_isaac_site_3dgs_lane": True,
            "runtime_execution_not_claimed_by_matrix": True,
        },
    }
    matrix_path = out_dir / "warehouse_isaac_scenario_eval_matrix.json"
    write_json(matrix_path, matrix)

    handoff = {
        "schema_version": WAREHOUSE_ISAAC_SCENARIOS_SCHEMA_VERSION,
        "generated_at": generated,
        "status": "ready_for_generic_isaac_site_3dgs_eval" if not blockers else "blocked",
        "blockers": blockers,
        "site_type": "warehouse",
        "fixture_root": str(fixture),
        "scene_asset": str(asset),
        "scenario_eval_matrix_path": str(matrix_path),
        "scenario_eval_run_count": len(runs),
        "recommended_command": (
            "python -m blueprint_pipeline.isaac_g1_site_3dgs_realistic_eval "
            f"--spz-asset {asset} --scenario-eval-matrix {matrix_path}"
        ),
        "claim_boundary": {
            "warehouse_scenario_family_is_committed": True,
            "runtime_execution_requires_generic_isaac_provider_or_local_runtime": True,
            "not_a_lightwheel_kitchen_claim": True,
            "not_physical_robot_execution_or_deployment_proof": True,
        },
    }
    handoff_path = out_dir / "warehouse_isaac_handoff_manifest.json"
    write_json(handoff_path, handoff)

    return {
        **handoff,
        "artifact_paths": {
            "scenario_eval_matrix": str(matrix_path),
            "handoff_manifest": str(handoff_path),
        },
        "scenario_eval_matrix": matrix,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-root")
    parser.add_argument("--scene-asset")
    parser.add_argument("--output-dir")
    args = parser.parse_args(argv)
    payload = build_warehouse_isaac_scenarios(
        fixture_root=args.fixture_root,
        scene_asset=args.scene_asset,
        output_dir=args.output_dir,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not payload.get("blockers") else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
