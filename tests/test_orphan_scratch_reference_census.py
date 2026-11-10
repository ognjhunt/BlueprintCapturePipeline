"""The orphan scratch reference census: conservative, read-only, and reported per folder."""

# Covers (for impacted-test selection):
#   src/blueprint_pipeline/control_plane_lane_scratch_census.py
#   src/blueprint_pipeline/control_plane_capacity_controller.py
#   src/blueprint_pipeline/control_plane_storage_gc.py

from __future__ import annotations

import json
from pathlib import Path

import pytest

from blueprint_pipeline import control_plane_capacity_controller as cap
from blueprint_pipeline.control_plane_lane_scratch_census import (
    REFERENCE_CENSUS_FILENAME,
    build_reference_census,
    publish_reference_census,
)
from blueprint_pipeline.control_plane_storage_pins import write_storage_pin

NOW = 1_800_000_000.0
KINDS = ("queue", "process", "container_mount", "unit_file", "pin",
         "live_release", "active_run")


def _proc(root: Path, tmp_path: Path) -> Path:
    """A /proc with only ``self``: its own mount namespace maps the whole test tree."""
    me = root / "self"
    (me / "ns").mkdir(parents=True)
    (me / "ns" / "mnt").symlink_to("mnt:[1]")
    (me / "mountinfo").write_text(f"1 0 8:1 / {tmp_path} rw - ext4 /dev/x rw\n")
    return root


def _world(tmp_path: Path) -> dict:
    work, inputs = tmp_path / "work", tmp_path / "inputs"
    folders = {kind: work / f"held-by-{kind.replace('_', '-')}" for kind in KINDS}
    folders["free"] = inputs / "nothing-refers-to-me"
    for folder in folders.values():
        folder.mkdir(parents=True)
        (folder / "data.bin").write_bytes(b"x" * 64)
    queue = tmp_path / "queue"
    (queue / "pending").mkdir(parents=True)
    (queue / "pending" / "m.json").write_text(json.dumps({"input": str(folders["queue"])}))
    proc = _proc(tmp_path / "proc", tmp_path)
    worker = proc / "4242"
    (worker / "fd").mkdir(parents=True)
    (worker / "cmdline").write_bytes(b"python\0")
    (worker / "environ").write_bytes(b"")
    (worker / "cwd").symlink_to(folders["process"])
    (worker / "ns").mkdir()
    (worker / "ns" / "mnt").symlink_to("mnt:[1]")
    container = proc / "5151"
    (container / "fd").mkdir(parents=True)
    (container / "cmdline").write_bytes(b"sim\0")
    (container / "environ").write_bytes(b"")
    (container / "cwd").symlink_to("/")
    (container / "ns").mkdir()
    (container / "ns" / "mnt").symlink_to("mnt:[2]")
    relative = folders["container_mount"].relative_to(tmp_path)
    (container / "mountinfo").write_text(f"7 1 8:1 /{relative} /inputs rw - ext4 /dev/x rw\n")
    units = tmp_path / "units"
    units.mkdir()
    (units / "job.service").write_text(f"[Service]\nWorkingDirectory=/x/{folders['unit_file'].name}\n")
    pins = tmp_path / "pins"
    write_storage_pin(pins_root=pins, kind="preparation", owner_id="prep-1", paths=[folders["pin"]],
                      now=lambda: NOW)
    release = folders["live_release"] / "release"
    release.mkdir()
    link = tmp_path / "release-link"
    link.symlink_to(release)
    return {"folders": folders, "kwargs": {
        "work_root": work, "inputs_root": inputs, "process_root": proc, "pins_root": pins, "queue_roots": [queue],
        "release_link": link, "active_run_roots": [folders["active_run"] / "run-1"], "unit_directories": [units],
        "now": NOW}}


def test_each_kind_of_reference_holds_its_folder_and_only_the_free_one_is_unreferenced(tmp_path):
    world = _world(tmp_path)
    before = sorted(str(path) for path in tmp_path.rglob("*"))

    census = build_reference_census(**world["kwargs"])

    assert census["status"] == "complete", census["scan_errors"]
    assert census["mutations"] == 0
    rows = {row["path"]: row["references"] for row in census["rows"]}
    for kind in KINDS:
        assert kind in rows[str(world["folders"][kind])], kind
    assert rows[str(world["folders"]["free"])] == []
    assert sorted(str(path) for path in tmp_path.rglob("*")) == before  # read-only


def test_a_symlink_beside_a_folder_holds_it_and_leaves_the_census_incomplete(tmp_path):
    world = _world(tmp_path)
    held = tmp_path / "work" / "held-by-queue"
    (tmp_path / "work" / "current").symlink_to(held / "data.bin")

    census = build_reference_census(**world["kwargs"])

    assert census["status"] == "incomplete" and "work_child_unsafe" in census["scan_errors"]
    assert "sibling_link" in {row["path"]: row["references"] for row in census["rows"]}[str(held)]


def test_a_container_mounting_the_whole_volume_holds_only_what_it_uses(tmp_path):
    world = _world(tmp_path)
    container = tmp_path / "proc" / "5151"
    (container / "mountinfo").write_text("7 1 8:1 / /host rw - ext4 /dev/x rw\n")
    used = world["folders"]["free"]
    (container / "fd" / "3").symlink_to(f"/host/{used.relative_to(tmp_path)}/data.bin")

    census = build_reference_census(**world["kwargs"])

    rows = {row["path"]: row["references"] for row in census["rows"]}
    assert census["status"] == "complete", census["scan_errors"]
    assert rows[str(used)] == ["container_mount"]
    assert "container_mount" not in rows[str(world["folders"]["queue"])]


@pytest.mark.parametrize("breakage", ["queue_root", "unit_file", "foreign_mountinfo", "inputs_root",
                                      "no_queue_inventory", "deadline"])
def test_an_unreadable_or_partial_inventory_makes_the_census_incomplete(tmp_path, breakage, monkeypatch):
    world = _world(tmp_path)
    kwargs = dict(world["kwargs"])
    if breakage == "queue_root":
        kwargs["queue_roots"] = [tmp_path / "missing-queue"]
    elif breakage == "unit_file":
        (tmp_path / "units" / "huge.service").write_bytes(b"x" * (1024 * 1024 + 1))
    elif breakage == "foreign_mountinfo":
        (tmp_path / "proc" / "5151" / "mountinfo").unlink()
        (tmp_path / "proc" / "5151" / "mountinfo").mkdir()
    elif breakage == "inputs_root":
        kwargs["inputs_root"] = tmp_path / "not-there"
    elif breakage == "no_queue_inventory":
        kwargs["queue_roots"] = None
    else:
        ticks = iter([0.0] + [10_000.0] * 10_000)
        monkeypatch.setattr("blueprint_pipeline.control_plane_lane_scratch_census.time.monotonic",
                            lambda: next(ticks))

    census = build_reference_census(**kwargs)

    assert census["status"] == "incomplete"
    assert census["scan_errors"]
    projection = cap.orphan_scratch_census_projection(
        census, {"orphan_scratch_root_bytes": [[str(world["folders"]["free"]), 9]]}, now=NOW + 60)
    assert projection["status"] == "incomplete"
    assert projection["unreferenced_bytes"] == 0
    assert projection["referenced_bytes"] == 9


def _census(rows, *, observed=NOW, status="complete", errors=()):
    return {"schema_version": cap.ORPHAN_CENSUS_SCHEMA_VERSION, "status": status,
            "observed_at_epoch": observed, "rows": rows, "scan_errors": list(errors), "mutations": 0}


def test_projection_counts_only_listed_unreferenced_folders_of_a_fresh_census():
    survey = {"status": "complete", "observed_at_epoch": NOW - 100, "orphan_scratch_root_bytes_complete": True,
              "orphan_scratch_root_bytes": [["/w/free", 30], ["/w/held", 20], ["/w/unlisted", 10],
                                            ["<redacted>", 5]]}  # the survey publishes names sanitized
    census = _census([{"path": "/w/free", "references": []},
                      {"path": "/w/held", "references": ["process"]},
                      {"path": "/w/a?token=x", "references": []}])

    projection = cap.orphan_scratch_census_projection(census, survey, now=NOW + 600)

    assert projection["status"] == "complete"
    assert projection["age_seconds"] == 600.0
    assert projection["unreferenced_bytes"] == projection["retirement_would_free_bytes"] == 30
    assert projection["referenced_bytes"] == 35
    assert projection["retirement_authorized"] is False
    by_root = {row["root"]: row for row in projection["roots"]}
    assert by_root["/w/free"]["disposition"] == "unreferenced"
    assert by_root["/w/held"]["references"] == ["process"]
    assert by_root["/w/unlisted"]["references"] == ["not_in_census"]
    assert by_root["<redacted>"]["disposition"] == "referenced_or_unproven"  # a redacted name proves nothing


@pytest.mark.parametrize("census", [
    None,
    _census([{"path": "/w/free", "references": []}], observed=NOW - 3 * 3600),
    _census([{"path": "/w/free", "references": []}], status="incomplete", errors=["tree_unreadable"]),
    {**_census([{"path": "/w/free", "references": []}]),
     "schema_version": "control_plane_lane_scratch_census.v1"},
])
def test_a_missing_stale_incomplete_or_foreign_census_frees_nothing(census):
    projection = cap.orphan_scratch_census_projection(
        census, {"orphan_scratch_root_bytes": [["/w/free", 30]]}, now=NOW)
    assert projection["unreferenced_bytes"] == 0
    assert projection["referenced_bytes"] == 30
    assert projection["status"] in {"missing", "stale", "incomplete"}


def test_the_capacity_summary_reports_the_census(tmp_path, monkeypatch):
    monkeypatch.setattr("blueprint_pipeline.task_evaluation_scene_spend.observe_configured_scene_project_spend",
                        lambda **_kwargs: None)
    gc_root = tmp_path / "storage-gc"
    gc_root.mkdir()
    (gc_root / "orphan-scratch-census.json").write_text(json.dumps(
        _census([{"path": "/mnt/blueprint-work/sdk-inputs", "references": []}], observed=NOW - 120)))
    survey = {"schema_version": "control_plane_disk_usage_survey.v1", "status": "complete",
              "observed_at_epoch": NOW, "mounts": [], "by_class": [], "top_roots": [], "top_owners": [],
              "unclassified_roots": [], "orphan_scratch_root_bytes_complete": True,
              "orphan_scratch_root_bytes": [["/mnt/blueprint-work/sdk-inputs", 30 * 1024**3]]}

    cap.run_controller(mounts=[str(tmp_path)], report_root=tmp_path / "capacity",
                       reservation_root=tmp_path / "ledger", webhook_url="", volume=None, ack="", token="",
                       disk_usage=_disk_usage(), now=NOW, survey=lambda **_k: survey,
                       storage_gc_summary_path=gc_root / "summary.json")

    summary = json.loads((tmp_path / "capacity" / "summary.json").read_text())
    census = summary["usage"]["orphan_scratch_census"]
    assert census["status"] == "complete" and census["age_seconds"] == 120.0
    assert census["unreferenced_bytes"] == 30 * 1024**3
    assert census["roots"][0] == {"root": "/mnt/blueprint-work/sdk-inputs", "allocated_bytes": 30 * 1024**3,
                                  "disposition": "unreferenced", "references": []}


def _disk_usage():
    from collections import namedtuple
    usage = namedtuple("usage", "total used free")
    return lambda _path: usage(154 * 1024**3, 54 * 1024**3, 100 * 1024**3)


def test_gc_census_failure_withdraws_the_record_and_never_raises(tmp_path):
    record = tmp_path / REFERENCE_CENSUS_FILENAME
    record.write_text("{}")

    def broken(**_kwargs):
        raise OSError("proc gone")

    assert publish_reference_census(tmp_path, pins_root=tmp_path, queue_roots=[], active_run_roots=[],
                                           build=broken) is False
    assert not record.exists()
    assert publish_reference_census(
        tmp_path, pins_root=tmp_path, queue_roots=[], active_run_roots=[],
        build=lambda **kwargs: {"queue_roots": kwargs["queue_roots"], "rows": []}) is True
    assert json.loads(record.read_text())["queue_roots"] is None  # no queue inventory is not "no queues"
