from __future__ import annotations

import json
import struct
from pathlib import Path

import trimesh

from blueprint_pipeline import scene_eval_autogen as sea
from blueprint_pipeline.robot_eval_dataset import (
    SCENARIO_VARIATION_DEFINITIONS,
    TASK_ONTOLOGY_DEFINITIONS,
)

_ONTOLOGY_IDS = {item["task_id"] for item in TASK_ONTOLOGY_DEFINITIONS}
_VARIATION_IDS = {item["variation_id"] for item in SCENARIO_VARIATION_DEFINITIONS}


def _write_ascii_ply(path: Path, points: list[tuple[float, float, float]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "ply",
        "format ascii 1.0",
        f"element vertex {len(points)}",
        "property float x",
        "property float y",
        "property float z",
        "end_header",
        *(f"{x} {y} {z}" for x, y, z in points),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _room_points() -> list[tuple[float, float, float]]:
    # A 6m x 4m room: floor grid plus two box-shaped clusters and a ceiling band.
    points: list[tuple[float, float, float]] = []
    for ix in range(13):
        for iy in range(9):
            points.append((ix * 0.5, iy * 0.5, 0.0))
    for iz in range(4):
        points.append((1.0, 1.0, 0.2 + iz * 0.2))
        points.append((5.0, 3.0, 0.2 + iz * 0.2))
    for ix in range(7):
        points.append((ix * 1.0, 2.0, 2.6))
    return points


def _write_binary_vertex_ply(path: Path, points: list[tuple[float, float, float]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = "\n".join(
        [
            "ply",
            "format binary_little_endian 1.0",
            f"element vertex {len(points)}",
            "property float x",
            "property float y",
            "property float z",
            "property float opacity",
            "end_header",
            "",
        ]
    ).encode("ascii")
    body = b"".join(struct.pack("<ffff", x, y, z, 1.0) for x, y, z in points)
    path.write_bytes(header + body)
    return path


def _write_kitchen_usda(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "#usda 1.0",
                "(",
                '    upAxis = "Z"',
                "    metersPerUnit = 1",
                ")",
                'def Xform "Kitchen" {',
                '    def Mesh "sink_01" {}',
                '    def Mesh "cabinet_door" {}',
                '    def Mesh "kettle" {}',
                '    def Mesh "counter_top" {}',
                '    def Mesh "fridge" {}',
                "}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return path


def test_ascii_ply_generates_min_tasks_and_many_scenarios(tmp_path: Path) -> None:
    scene = _write_ascii_ply(tmp_path / "site.ply", _room_points())
    manifest = sea.generate_scene_eval_tasks(
        scene, tmp_path / "out", generated_at="2026-07-02T00:00:00+00:00"
    )
    assert manifest["status"] == "completed"
    assert manifest["meets_minimum_task_count"] is True
    assert manifest["task_count"] >= sea.MIN_TASK_COUNT
    # baseline + 11 variations x 3 seeds per task
    expected_per_task = 1 + len(SCENARIO_VARIATION_DEFINITIONS) * sea.DEFAULT_SEEDS_PER_VARIATION
    assert manifest["min_scenarios_per_task"] == expected_per_task
    assert manifest["scenario_count"] == manifest["task_count"] * expected_per_task
    assert manifest["geometry_grounding"] == "bounds_recovered"

    for name in (
        "scene_analysis.json",
        "auto_task_cards.json",
        "auto_scenario_cards.json",
        "auto_eval_cards.json",
        "scenario_family_library.json",
        "scene_eval_autogen_manifest.json",
    ):
        assert (tmp_path / "out" / name).is_file()

    task_cards = json.loads((tmp_path / "out" / "auto_task_cards.json").read_text())
    for card in task_cards["cards"]:
        assert card["ontology_task_id"] in _ONTOLOGY_IDS
        assert card["success_criteria"]
        assert card["threshold_profile"]["min_success_rate"] > 0
        assert card["zone_pair_status"] == "validated_zone_pair"

    scenario_cards = json.loads((tmp_path / "out" / "auto_scenario_cards.json").read_text())
    variation_ids = {card["variation_id"] for card in scenario_cards["cards"]}
    assert _VARIATION_IDS <= variation_ids
    assert "baseline" in variation_ids


def test_glb_scene_generates_review_scope_without_claiming_metric_scale(tmp_path: Path) -> None:
    scene = tmp_path / "apartment.glb"
    scene.write_bytes(trimesh.creation.box(extents=[9.0, 3.0, 7.0]).export(file_type="glb"))
    output = tmp_path / "glb-out"

    manifest = sea.generate_scene_eval_tasks(
        scene,
        output,
        site_id="private-apartment-001",
        generated_at="2026-08-02T00:00:00+00:00",
    )

    assert manifest["status"] == "completed"
    assert manifest["task_count"] >= sea.MIN_TASK_COUNT
    analysis = json.loads((output / "scene_analysis.json").read_text())
    assert analysis["scene"]["asset_type"] == "glb"
    assert analysis["scene"]["up_axis"] == "Y"
    assert analysis["scene"]["metric_scale_status"] == (
        "provider_declared_not_independently_validated"
    )
    assert analysis["scene"]["metric_scale_proven"] is False
    assert analysis["claim_boundary"]["default_robot_id"] == "franka_panda"


def test_zone_poses_stay_inside_scene_bounds(tmp_path: Path) -> None:
    scene = _write_ascii_ply(tmp_path / "site.ply", _room_points())
    manifest = sea.generate_scene_eval_tasks(scene, tmp_path / "out")
    assert manifest["status"] == "completed"
    analysis = json.loads((tmp_path / "out" / "scene_analysis.json").read_text())
    low = analysis["scene"]["bounds"]["min"]
    high = analysis["scene"]["bounds"]["max"]
    for zone in analysis["zones"]:
        pose = zone["pose_xyz"]
        assert pose is not None
        for axis in range(3):
            assert low[axis] - 1e-6 <= pose[axis] <= high[axis] + 1e-6


def test_binary_vertex_ply_bounds_recovered(tmp_path: Path) -> None:
    scene = _write_binary_vertex_ply(tmp_path / "splat.ply", _room_points())
    ingested = sea.ingest_scene_file(scene)
    assert ingested["status"] == "completed"
    assert ingested["bounds"] is not None
    assert ingested["estimate_method"] == "binary_vertex_xyz_stride_sample"
    assert ingested["bounds"]["min"][2] == 0.0
    assert ingested["bounds"]["max"][0] == 6.0

    manifest = sea.generate_scene_eval_tasks(scene, tmp_path / "out")
    assert manifest["status"] == "completed"
    assert manifest["geometry_grounding"] == "bounds_recovered"


def test_binary_splat_trims_sparse_outliers_and_infers_y_up(tmp_path: Path) -> None:
    points = [
        (
            float(index % 20) * 0.5,
            float((index // 20) % 6) * 0.5,
            float((index // 120) % 18) * 0.5,
        )
        for index in range(2_000)
    ]
    points.extend([(-250.0, -250.0, -250.0), (250.0, 250.0, 250.0)])
    scene = _write_binary_vertex_ply(tmp_path / "scaniverse_splat.ply", points)

    ingested = sea.ingest_scene_file(scene)

    assert ingested["status"] == "completed"
    assert ingested["up_axis"] == "Y"
    assert ingested["up_axis_source"] == "smallest_robust_extent_heuristic"
    assert ingested["metric_scale_status"] == "unverified"
    assert ingested["metric_scale_proven"] is False
    extents = [
        ingested["bounds"]["max"][axis] - ingested["bounds"]["min"][axis] for axis in range(3)
    ]
    assert max(extents) < 20.0
    assert extents[1] == min(extents)


def test_usda_semantic_grounding_adds_object_tasks(tmp_path: Path) -> None:
    scene = _write_kitchen_usda(tmp_path / "kitchen_scene.usda")
    manifest = sea.generate_scene_eval_tasks(scene, tmp_path / "out")
    assert manifest["status"] == "completed"
    assert manifest["environment"] == "kitchen"
    assert manifest["task_count"] >= sea.MIN_TASK_COUNT

    task_cards = json.loads((tmp_path / "out" / "auto_task_cards.json").read_text())
    by_source = {card["task_id"]: card for card in task_cards["cards"]}
    object_grounded = [
        card for card in task_cards["cards"] if card["grounding_source"] == "scene_object_hint"
    ]
    assert object_grounded, f"expected object tasks, got {sorted(by_source)}"
    task_ids = set(by_source)
    assert any(task_id.startswith("open_close_") for task_id in task_ids)
    assert any(task_id.startswith("pick_place_") for task_id in task_ids)


def test_deterministic_across_runs(tmp_path: Path) -> None:
    scene = _write_ascii_ply(tmp_path / "site.ply", _room_points())
    first = sea.generate_scene_eval_tasks(
        scene, tmp_path / "out_a", generated_at="2026-07-02T00:00:00+00:00"
    )
    second = sea.generate_scene_eval_tasks(
        scene, tmp_path / "out_b", generated_at="2026-07-02T00:00:00+00:00"
    )
    assert first["deterministic_fingerprint"] == second["deterministic_fingerprint"]
    cards_a = (tmp_path / "out_a" / "auto_scenario_cards.json").read_text()
    cards_b = (tmp_path / "out_b" / "auto_scenario_cards.json").read_text()
    assert cards_a == cards_b


def test_missing_and_unsupported_inputs_block(tmp_path: Path) -> None:
    missing = sea.generate_scene_eval_tasks(tmp_path / "nope.ply", tmp_path / "out_missing")
    assert missing["status"] == "blocked"
    assert "scene_file_missing" in missing["blockers"]
    assert (tmp_path / "out_missing" / "scene_eval_autogen_manifest.json").is_file()

    bad = tmp_path / "scene.xyz"
    bad.write_text("not a scene", encoding="utf-8")
    unsupported = sea.generate_scene_eval_tasks(bad, tmp_path / "out_bad")
    assert unsupported["status"] == "blocked"
    assert any(item.startswith("unsupported_scene_suffix") for item in unsupported["blockers"])


def test_malformed_ply_blocks_instead_of_raising(tmp_path: Path) -> None:
    corrupt = tmp_path / "corrupt.ply"
    corrupt.write_bytes(b"ply\nformat binary_little_endian 1.0\nno header end here")
    manifest = sea.generate_scene_eval_tasks(corrupt, tmp_path / "out_corrupt")
    assert manifest["status"] == "blocked"
    assert any(
        item.startswith("scene_file_unreadable_or_malformed") for item in manifest["blockers"]
    )
    assert (tmp_path / "out_corrupt" / "scene_eval_autogen_manifest.json").is_file()

    code = sea.main([str(corrupt), "--output-dir", str(tmp_path / "cli_corrupt")])
    assert code == 1


def test_scenario_family_library_feeds_variation_instantiator_shape(tmp_path: Path) -> None:
    scene = _write_ascii_ply(tmp_path / "site.ply", _room_points())
    manifest = sea.generate_scene_eval_tasks(scene, tmp_path / "out")
    assert manifest["status"] == "completed"
    library = json.loads((tmp_path / "out" / "scenario_family_library.json").read_text())
    assert library["family_count"] == manifest["task_count"]
    for family in library["families"]:
        assert family["family_id"]
        assert family["scenario_id"]
        assert family["task_id"]
        variation_ids = {item["variation_id"] for item in family["variations"]}
        assert variation_ids == _VARIATION_IDS


def test_cli_main_writes_artifacts_and_reports_status(tmp_path: Path, capsys) -> None:
    scene = _write_ascii_ply(tmp_path / "site.ply", _room_points())
    out_dir = tmp_path / "cli_out"
    code = sea.main([str(scene), "--output-dir", str(out_dir), "--seeds-per-variation", "1"])
    assert code == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["status"] == "completed"
    assert printed["task_count"] >= sea.MIN_TASK_COUNT
    manifest = json.loads((out_dir / "scene_eval_autogen_manifest.json").read_text())
    # 1 baseline + 11 variations x 1 seed
    assert manifest["min_scenarios_per_task"] == 1 + len(SCENARIO_VARIATION_DEFINITIONS)

    missing_code = sea.main([str(tmp_path / "absent.ply"), "--output-dir", str(tmp_path / "x")])
    assert missing_code == 1
