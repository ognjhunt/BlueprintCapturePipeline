"""Withdrawal deletion covers everything a website navigation job's handoff writes.

Beta journey step: site withdrawal after a navigation job (a scene kept as filmed, the G1 walking to
a goal) reached its scene half. The handoff writes, in the capture's own workspace,
``navigation_task_spec.json``, ``navigation_scene_assets.json`` with its appearance and collision
USDs, and the scene-inputs record's root (``website_navigation_scene_inputs``) with its packet, staged
scene and G1 USD copies and robot-free probe receipt; the team registry's derived row names that
record. Each is deleted with the rest of the site.

Hermetic: the deletion tests' synthetic site and fakes, the navigation scene-inputs tests' synthetic
room and a tiny G1-shaped robot under a fixed test identity. No capture, mesh, splat or video of a
real site, and nothing reaches a provider, a bucket, the WebApp or the host.
"""
# Covers: src/blueprint_pipeline/website_capture_withdrawal_team_scene.py
# Covers: src/blueprint_pipeline/website_capture_withdrawal_deletion.py
from __future__ import annotations

import json
from pathlib import Path

import pytest

from blueprint_pipeline import website_capture_withdrawal_deletion as deletion
from blueprint_pipeline.team_candidate_quick10_registry import write_derived_profile
from blueprint_pipeline.team_candidate_registry_reader import REGISTRY_ENV
from blueprint_pipeline.website_locomotion_task import NAVIGATION_TASK_FILE
from blueprint_pipeline.website_navigation_scene_assets import FILE as ASSETS_FILE
from blueprint_pipeline.website_navigation_scene_inputs import OUTPUT_DIRECTORY, write_navigation_scene_inputs
from blueprint_pipeline.website_preparation_revision import workspace
from tests import test_website_capture_withdrawal_deletion as base
from tests.test_website_capture_withdrawal_team_scene import _registry
from tests.test_website_navigation_scene_inputs import _scene_assets, navigation_scene_inputs_arguments

world = base.world  # the deletion tests' withdrawn site, runs, B2 and roots


def _files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file())


@pytest.fixture
def navigation(world, tmp_path, monkeypatch):
    """A navigation job's handoff outputs, laid out where the handoff writes them, and its derived row."""
    capture = world["capture"]
    root = workspace(capture)
    (tmp_path / "fixture").mkdir()
    arguments = navigation_scene_inputs_arguments(tmp_path / "fixture", monkeypatch)
    # The A1 task spec and the A2 scene files, in the capture workspace as the handoff writes them.
    task_path = base._write(root / NAVIGATION_TASK_FILE, arguments["navigation_task_spec"])
    scene_files = _scene_assets(root / "navigation_scene")
    assets_path = base._write(root / ASSETS_FILE, scene_files)
    usds = [Path(scene_files[role]["path"]) for role in ("appearance", "collision")]
    # The pinned G1 is the robot's own file, outside the site; the record stages a copy of it.
    written = write_navigation_scene_inputs(navigation_task_spec=arguments["navigation_task_spec"],
                                            scene_assets=scene_files, robot_record=arguments["robot_record"],
                                            output_root=root / OUTPUT_DIRECTORY)
    record = written["record"]
    record_root = Path(written["path"]).parent
    # The team registry's derived row, written by the registry's own writer.
    registry = Path(world["resolution"]["team_registry"])
    derived = tmp_path / "inputs/team-scene-profiles"
    _registry(registry, derived)
    monkeypatch.setenv(REGISTRY_ENV, str(registry))
    row = write_derived_profile({
        "capture_root": str(capture), "scene_id": "site-req1", "task_id": record["task_id"],
        "scenario_id": "scene.nominal", "robot_free_probe_root": str(record_root),
        "scene_revision_digest": record["scene_revision_digest"], "navigation_scene_inputs_path": written["path"]})
    assert row["status"] == "written"
    return {"root": root, "task": task_path, "assets": assets_path, "usds": usds, "record": Path(written["path"]),
            "record_root": record_root, "probe": Path(record["robot_free_probe"]["path"]),
            "packet": Path(record["packet_root"]), "row": Path(row["path"]), "derived": derived,
            "robot": Path(arguments["robot_record"]["path"])}


def test_withdrawal_deletes_every_file_a_navigation_job_wrote(world, navigation):
    job = navigation
    inputs = job["record_root"] / ".inputs"
    # The record's root holds its packet, the staged scene USDs and the staged G1 USD, and the probe receipt.
    staged = [inputs / "scene/scene_collision.usda", inputs / "scene/scene_appearance.usdz",
              inputs / "robot" / job["robot"].name]
    assert all(path.is_file() for path in staged)
    assert job["probe"].parent == job["record_root"] and job["packet"].parent == job["record_root"]
    assert all(path.is_file() for path in (job["task"], job["assets"], *job["usds"], job["record"], job["probe"]))
    written = [*_files(job["root"])]

    gcs = base.FakeGcs()
    plan = base._plan(world, cloud_storage=gcs, uncovered_roots=())
    assert plan["unresolved"] == []
    # The derived row naming the record is planned by the team section, by its capture key and its bytes.
    assert [row[0] for row in plan["team_scene"]["derived_profile_rows"]] == [str(job["row"])]
    receipt = base._apply(world, plan, cloud_storage=gcs, uncovered_roots=())

    for path in (job["task"], job["assets"], *job["usds"], job["record"], job["probe"], job["row"], *staged,
                 *written):
        assert not path.exists(), path
    assert _files(job["record_root"]) == [] and _files(job["root"]) == []
    # Nothing of the site is left but its withdrawal journal.
    assert [path for path in _files(world["site"]) if "website_withdrawal" not in path.parts] == []
    assert receipt["deleted"]["team_scene"]["derived_profile_rows"] == [str(job["row"])]
    assert receipt["deleted"]["scene_files"] == plan["scene_files"]["count"] >= len(written)
    assert receipt["pipeline_controlled_deletion_complete"] is True
    # The robot's own pinned file is not the site's.
    assert job["robot"].is_file()


def test_a_derived_row_naming_a_navigation_record_outside_the_site_fails_closed(world, navigation, tmp_path):
    job = navigation
    elsewhere = tmp_path / "elsewhere" / "website_navigation_scene_inputs.v1.json"
    base._write(elsewhere, {"schema_version": "blueprint.website_navigation_scene_inputs.v1"})
    row = json.loads(job["row"].read_text())
    job["row"].write_text(json.dumps({**row, "navigation_scene_inputs_path": str(elsewhere)}))
    before = base._snapshot(tmp_path)

    plan = base._plan(world, uncovered_roots=())
    assert plan["unresolved"] == [{"path": str(job["row"]),
                                   "reason": "website_withdrawal_deletion_navigation_scene_inputs_outside_site"}]
    with pytest.raises(deletion.WithdrawalDeletionError, match="unresolved_copies"):
        base._apply(world, plan, uncovered_roots=())
    assert base._snapshot(tmp_path) == before and elsewhere.is_file() and job["record"].is_file()

