from __future__ import annotations

import json
from pathlib import Path

import pytest

from blueprint_pipeline.deformable_scene_catalog import build_deformable_scene_catalog


def _box(ins_id: str, label: str, center: tuple[float, float, float]) -> dict:
    cx, cy, cz = center
    return {
        "ins_id": ins_id,
        "label": label,
        "bounding_box": [
            {"x": x, "y": y, "z": z}
            for x in (cx - 0.1, cx + 0.1)
            for y in (cy - 0.1, cy + 0.1)
            for z in (cz - 0.1, cz + 0.1)
        ],
    }


def _scene(root: Path, directory: str, objects: list[dict], *, with_pair: bool = True) -> None:
    scene = root / directory
    scene.mkdir(parents=True)
    scene_id = directory.split("_")[-1]
    scene.joinpath("labels.json").write_text(json.dumps(objects), encoding="utf-8")
    rooms = [{"profile": [[0, 0], [2, 0], [2, 2], [0, 2]]}]
    if not with_pair:
        rooms.append({"profile": [[3, 0], [5, 0], [5, 2], [3, 2]]})
    scene.joinpath("structure.json").write_text(
        json.dumps({"rooms": rooms, "walls": [], "holes": []}), encoding="utf-8"
    )
    scene.joinpath("3dgs_compressed.ply").write_bytes(f"appearance-{scene_id}".encode())


def _collision(root: Path, scene_id: str) -> None:
    destination = root / "Collision_Mesh" / scene_id
    destination.mkdir(parents=True)
    destination.joinpath(f"{scene_id}_collision.usd").write_bytes(b"collision")


def test_catalog_is_complete_outcome_blind_and_selects_only_same_room_pair(
    tmp_path: Path,
) -> None:
    interiorgs = tmp_path / "InteriorGS"
    sage = tmp_path / "SAGE"
    interiorgs.mkdir()
    sage.mkdir()
    _scene(
        interiorgs,
        "0001_840001",
        [_box("1", "towel", (0.5, 0.5, 0.8)), _box("2", "basket", (1.2, 0.5, 0.8))],
    )
    _collision(sage, "840001")
    _scene(interiorgs, "0002_840002", [_box("3", "chair", (0.5, 0.5, 0.8))])
    _collision(sage, "840002")
    _scene(
        interiorgs,
        "0003_840003",
        [_box("4", "towel", (0.5, 0.5, 0.8)), _box("5", "basket", (3.5, 0.5, 0.8))],
        with_pair=False,
    )
    _collision(sage, "840003")

    observed = build_deformable_scene_catalog(
        interiorgs_roots=[interiorgs],
        sage_roots=[sage],
        previously_used_scene_ids=["840002"],
        expected_scene_count=3,
    )

    assert observed["known_scene_count"] == 3
    assert observed["semantic_shortlist_scene_ids"] == ["840001"]
    by_scene = {row["scene_id"]: row for row in observed["scenes"]}
    assert by_scene["840001"]["same_publisher_room_compatible_pair_count"] == 1
    assert by_scene["840002"]["rejection_reasons"] == ["previously_used_scene"]
    assert by_scene["840003"]["rejection_reasons"] == [
        "no_same_publisher_room_compatible_pair"
    ]
    assert observed["learned_policy_outcomes_inspected"] is False
    assert observed["catalog_digest"].startswith("sha256:")


def test_catalog_fails_closed_on_scene_count_or_identity_ambiguity(tmp_path: Path) -> None:
    root_a = tmp_path / "a"
    root_b = tmp_path / "b"
    sage = tmp_path / "sage"
    root_a.mkdir()
    root_b.mkdir()
    sage.mkdir()
    _scene(root_a, "0001_840001", [_box("1", "towel", (0.5, 0.5, 0.8))])

    with pytest.raises(ValueError, match="known_scene_count_mismatch"):
        build_deformable_scene_catalog(
            interiorgs_roots=[root_a], sage_roots=[sage], expected_scene_count=2
        )

    _scene(root_b, "9999_840001", [_box("2", "basket", (0.5, 0.5, 0.8))])
    with pytest.raises(ValueError, match="interiorgs_scene_id_ambiguous"):
        build_deformable_scene_catalog(
            interiorgs_roots=[root_a, root_b], sage_roots=[sage]
        )
