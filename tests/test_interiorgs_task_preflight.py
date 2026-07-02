"""Hermetic tests for the InteriorGS CPU task-placement preflight lane."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from blueprint_pipeline.interiorgs_task_preflight import (
    PREFLIGHT_SCHEMA_VERSION,
    discover_scene_assets,
    load_task_specs,
    render_task_views,
    run_preflight,
    select_task_specs,
)
from tests.test_interiorgs_scene_placement import (
    _labels_payload,
    _structure_payload,
    write_synthetic_compressed_ply,
)


def _task_targets_payload():
    return {
        "bootstrap_generated": True,
        "facility_name": "Synthetic Scene",
        "source": "interiorgs",
        "tasks": [
            {"task_id": "pick_place_manipulation", "source": "interiorgs"},
            {"task_id": "Pick up pot_88 and place it in the target zone",
             "source": "interiorgs_prompt"},
            {"task_id": "Turn on bath_heater_79 and then turn it off",
             "source": "interiorgs_prompt"},
            {"task_id": "Open and close door_61", "source": "interiorgs_prompt"},
        ],
    }


@pytest.fixture()
def scene_dir(tmp_path: Path) -> Path:
    (tmp_path / "labels.json").write_text(json.dumps(_labels_payload()))
    (tmp_path / "structure.json").write_text(json.dumps(_structure_payload()))
    (tmp_path / "task_targets.synthetic.json").write_text(json.dumps(_task_targets_payload()))
    write_synthetic_compressed_ply(
        tmp_path / "3dgs_compressed.ply",
        [
            (-0.2, -0.2, 0.0, 4.0, 4.0, 2.6),
            (2.0, 2.0, 0.01, 9.2, 6.2, 2.6),
        ],
    )
    return tmp_path


class TestAssetDiscovery:
    def test_finds_all_assets(self, scene_dir):
        assets = discover_scene_assets(scene_dir)
        assert assets["splat"].name == "3dgs_compressed.ply"
        assert assets["labels"].name == "labels.json"
        assert assets["structure"].name == "structure.json"
        assert assets["task_file"].name == "task_targets.synthetic.json"

    def test_missing_assets_are_none(self, tmp_path):
        assets = discover_scene_assets(tmp_path)
        assert all(v is None for v in assets.values())


class TestTaskSpecs:
    def test_load_marks_abstract(self, scene_dir):
        specs = load_task_specs(scene_dir / "task_targets.synthetic.json")
        by_id = {s["task_id"]: s for s in specs}
        assert by_id["pick_place_manipulation"]["abstract"] is True
        assert by_id["Open and close door_61"]["abstract"] is False

    def test_select_filters_and_limits(self, scene_dir):
        specs = load_task_specs(scene_dir / "task_targets.synthetic.json")
        assert len(select_task_specs(specs)) == 3  # abstract excluded
        assert len(select_task_specs(specs, include_abstract=True)) == 4
        only_pot = select_task_specs(specs, only=["pot_88"])
        assert [s["task_id"] for s in only_pot] == [
            "Pick up pot_88 and place it in the target zone"
        ]
        assert len(select_task_specs(specs, limit=1)) == 1


class TestRunPreflight:
    def test_end_to_end_manifest(self, scene_dir, tmp_path):
        out = tmp_path / "out"
        manifest = run_preflight(scene_dir=scene_dir, out_dir=out)
        assert manifest["schema_version"] == PREFLIGHT_SCHEMA_VERSION
        assert (out / "preflight_manifest.json").is_file()
        assert manifest["scene_gates_passed"] is True
        scene_gate_names = {g["name"] for g in manifest["scene_gates"]}
        assert {
            "splat_asset_present",
            "splat_chunk_bounds_readable",
            "labels_loaded",
            "structure_loaded",
            "labels_within_splat_bounds",
            "floor_consistency",
        } <= scene_gate_names

        reports = {t["task_id"]: t for t in manifest["tasks"]}
        assert len(reports) == 3  # abstract task excluded

        pot = reports["Pick up pot_88 and place it in the target zone"]
        assert pot["all_gates_passed"] is True, pot["gates"]
        assert pot["target"]["id"] == "88"
        gate = {g["name"]: g for g in pot["gates"]}
        assert gate["target_resolved"]["evidence"]["method"] == "instance"
        assert gate["placement_validated"]["status"] == "PASS"
        assert pot["standoff_fixture_ids"] == ["92"]
        assert set(pot["cameras"]) == {"head_pov", "third_person", "overhead", "task_focus"}
        assert len(pot["splat_render_cameras"]) == 4

        # Ceiling-mounted unit: placement may solve, but the reach gate must fail.
        heater = reports["Turn on bath_heater_79 and then turn it off"]
        heater_gates = {g["name"]: g for g in heater["gates"]}
        assert heater_gates["target_within_reach_envelope"]["status"] == "FAIL"
        assert heater["all_gates_passed"] is False

        door = reports["Open and close door_61"]
        door_gates = {g["name"]: g for g in door["gates"]}
        assert door_gates["target_resolved"]["evidence"]["target_id"] == "61"
        assert door["stance"]["openable_target"] is True

        assert manifest["summary"]["tasks_evaluated"] == 3
        assert manifest["summary"]["tasks_passed"] >= 1

    def test_missing_splat_fails_scene_gate(self, scene_dir, tmp_path):
        (scene_dir / "3dgs_compressed.ply").unlink()
        manifest = run_preflight(scene_dir=scene_dir, out_dir=tmp_path / "out")
        gates = {g["name"]: g["status"] for g in manifest["scene_gates"]}
        assert gates["splat_asset_present"] == "FAIL"
        assert manifest["scene_gates_passed"] is False
        # Sidecar-only placement still runs: labels don't need the splat bytes.
        assert manifest["summary"]["tasks_evaluated"] == 3

    def test_task_filter_and_limit(self, scene_dir, tmp_path):
        manifest = run_preflight(
            scene_dir=scene_dir, out_dir=tmp_path / "out", tasks=["door_61"]
        )
        assert [t["task_id"] for t in manifest["tasks"]] == ["Open and close door_61"]

    def test_explicit_task_without_task_file(self, scene_dir, tmp_path):
        (scene_dir / "task_targets.synthetic.json").unlink()
        manifest = run_preflight(
            scene_dir=scene_dir,
            out_dir=tmp_path / "out",
            tasks=["Pick up pot_88 and place it in the target zone"],
        )
        assert manifest["summary"]["tasks_evaluated"] == 1
        assert manifest["tasks"][0]["target"]["id"] == "88"

    def test_no_labels_blocks_tasks(self, scene_dir, tmp_path):
        (scene_dir / "labels.json").unlink()
        manifest = run_preflight(scene_dir=scene_dir, out_dir=tmp_path / "out")
        gates = {g["name"]: g["status"] for g in manifest["scene_gates"]}
        assert gates["labels_loaded"] == "FAIL"
        assert manifest["summary"]["tasks_evaluated"] == 0


class TestRenderTaskViews:
    def test_blocked_when_harness_missing(self, tmp_path):
        result = render_task_views(
            tmp_path / "scene.ply",
            {"splat_render_cameras": [{"id": "head_pov", "spec": {}}]},
            tmp_path / "renders",
            repo_root=tmp_path,
        )
        assert result["status"] == "blocked"
        assert "splat_render_harness_missing" in result["blockers"]
