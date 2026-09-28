"""Read-only inventory of unregistered scratch, with shared inodes counted once."""

# Covers (for impacted-test selection):
#   src/blueprint_pipeline/control_plane_lane_scratch_census.py
#   scripts/lane_scratch_census.py

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import json

from blueprint_pipeline.control_plane_storage_pins import write_storage_pin


def _bytes(path: Path) -> int:
    info = path.lstat()
    return info.st_blocks * 512 or info.st_size


def test_census_lists_families_with_unique_inode_bytes(tmp_path: Path) -> None:
    from blueprint_pipeline.control_plane_lane_scratch_census import build_census

    work, inputs = tmp_path / "work", tmp_path / "inputs"
    g1, arena = work / "g1-checkpoint-cache", inputs / "arena-controls-r1"
    g1.mkdir(parents=True)
    arena.mkdir(parents=True)
    data = g1 / "weights.bin"
    data.write_bytes(b"x" * 4096)
    (arena / "weights-link.bin").hardlink_to(data)
    link = g1 / "out-link"
    link.symlink_to(tmp_path / "outside")
    before = sorted(str(path.relative_to(tmp_path)) for path in tmp_path.rglob("*"))

    report = build_census(work_root=work, inputs_root=inputs, process_root=tmp_path / "proc",
                          pins_root=tmp_path / "pins", queue_roots=(), release_link=tmp_path / "none")

    assert report["unique_allocated_bytes"] == _bytes(g1) + _bytes(arena) + _bytes(data) + _bytes(link)
    assert {(row["family"], row["owner_guess"]) for row in report["rows"]} == {
        ("g1", "g1-lane"), ("arena", "arena-lane")}
    assert sorted(str(path.relative_to(tmp_path)) for path in tmp_path.rglob("*")) == before
    assert report["status"] == "incomplete"  # no process or pin inventory was supplied


def test_census_records_process_queue_pin_release_and_active_run_references(tmp_path: Path) -> None:
    from blueprint_pipeline.control_plane_lane_scratch_census import build_census

    work, inputs = tmp_path / "work", tmp_path / "inputs"
    work.mkdir()
    inputs.mkdir()
    g1 = work / "g1-active-run"
    g1.mkdir()
    (g1 / "data.bin").write_bytes(b"x")
    proc = tmp_path / "proc" / "123"
    (proc / "fd").mkdir(parents=True)
    (proc / "cmdline").write_bytes(b"runner\0")
    (proc / "environ").write_bytes(b"")
    (proc / "cwd").symlink_to(g1)
    (proc / "fd" / "0").symlink_to(g1 / "data.bin")
    queue = tmp_path / "queue" / "pending"
    queue.mkdir(parents=True)
    (queue / "item.json").write_text(f'{{"root":"{g1}"}}', encoding="utf-8")
    pins = tmp_path / "pins"
    write_storage_pin(pins_root=pins, kind="preparation", owner_id="prep-1", paths=[g1])
    release = tmp_path / "active-release"
    release.symlink_to(g1)

    report = build_census(work_root=work, inputs_root=inputs, process_root=tmp_path / "proc",
                          pins_root=pins, queue_roots=(queue.parent,), release_link=release,
                          active_run_roots=(g1,))

    [row] = report["rows"]
    assert set(row["references"]) == {"process", "queue", "pin", "live_release", "active_run"}
    assert report["status"] == "complete"


def test_census_reports_missing_and_symlink_roots_without_following_them(tmp_path: Path) -> None:
    from blueprint_pipeline.control_plane_lane_scratch_census import build_census

    real = tmp_path / "real"
    real.mkdir()
    (real / "private").mkdir()
    work = tmp_path / "work"
    work.symlink_to(real)
    report = build_census(work_root=work, inputs_root=tmp_path / "missing",
                          process_root=tmp_path / "proc", pins_root=tmp_path / "pins",
                          queue_roots=(), release_link=tmp_path / "none")
    assert report["status"] == "incomplete"
    assert report["rows"] == []
    assert any("root_unsafe" in error for error in report["scan_errors"])
    assert any("root_missing" in error for error in report["scan_errors"])
    assert not (real / "private" / ".lane-scratch.v1.json").exists()


def test_census_marks_symlink_candidate_incomplete(tmp_path: Path) -> None:
    from blueprint_pipeline.control_plane_lane_scratch_census import build_census

    work, inputs = tmp_path / "work", tmp_path / "inputs"
    work.mkdir()
    inputs.mkdir()
    (tmp_path / "outside").mkdir()
    (work / "g1-linked").symlink_to(tmp_path / "outside", target_is_directory=True)
    (tmp_path / "proc").mkdir()
    (tmp_path / "pins").mkdir()
    (tmp_path / "release").mkdir()
    report = build_census(work_root=work, inputs_root=inputs, process_root=tmp_path / "proc",
                          pins_root=tmp_path / "pins", queue_roots=(),
                          release_link=tmp_path / "release", active_run_roots=())
    assert report["status"] == "incomplete"
    assert "work_child_unsafe" in report["scan_errors"]
    assert report["rows"] == []


def test_census_marks_bad_pin_inventory_incomplete(tmp_path: Path) -> None:
    from blueprint_pipeline.control_plane_lane_scratch_census import build_census

    work, inputs = tmp_path / "work", tmp_path / "inputs"
    work.mkdir()
    inputs.mkdir()
    (work / "g1-unknown").mkdir()
    (tmp_path / "proc").mkdir()
    (tmp_path / "pins" / "preparation").mkdir(parents=True)
    (tmp_path / "pins" / "preparation" / "broken.json").write_text("not json")
    (tmp_path / "release").mkdir()

    report = build_census(work_root=work, inputs_root=inputs, process_root=tmp_path / "proc",
                          pins_root=tmp_path / "pins", queue_roots=(),
                          release_link=tmp_path / "release", active_run_roots=())
    assert report["status"] == "incomplete"
    assert "pin_inventory_unreadable" in report["scan_errors"]


def test_census_rejects_pin_without_expiry(tmp_path: Path) -> None:
    from blueprint_pipeline.control_plane_lane_scratch_census import build_census

    work, inputs = tmp_path / "work", tmp_path / "inputs"
    work.mkdir()
    inputs.mkdir()
    folder = work / "g1-unknown"
    folder.mkdir()
    (tmp_path / "proc").mkdir()
    pin_dir = tmp_path / "pins" / "preparation"
    pin_dir.mkdir(parents=True)
    (pin_dir / "bad.json").write_text(json.dumps({
        "schema_version": "control_plane_storage_pin.v1", "kind": "preparation",
        "owner_id": "bad", "paths": [str(folder)], "released_at_epoch": None,
    }))
    (tmp_path / "release").mkdir()
    report = build_census(work_root=work, inputs_root=inputs, process_root=tmp_path / "proc",
                          pins_root=tmp_path / "pins", queue_roots=(),
                          release_link=tmp_path / "release", active_run_roots=())
    assert report["status"] == "incomplete"
    assert "pin_inventory_unreadable" in report["scan_errors"]


def test_census_finds_waiting_queue_reference_by_folder_name(tmp_path: Path) -> None:
    from blueprint_pipeline.control_plane_lane_scratch_census import build_census

    work, inputs = tmp_path / "work", tmp_path / "inputs"
    work.mkdir()
    inputs.mkdir()
    folder = work / "g1-unknown"
    folder.mkdir()
    queue = tmp_path / "queue" / "waiting_external"
    queue.mkdir(parents=True)
    (queue / "job.json").write_text(json.dumps({"scratch_name": folder.name}))
    (tmp_path / "proc").mkdir()
    (tmp_path / "pins").mkdir()
    (tmp_path / "release").mkdir()
    report = build_census(work_root=work, inputs_root=inputs, process_root=tmp_path / "proc",
                          pins_root=tmp_path / "pins", queue_roots=(queue.parent,),
                          release_link=tmp_path / "release", active_run_roots=())
    assert report["rows"][0]["references"] == ["queue"]


def test_census_marks_unsafe_queue_state_and_file_incomplete(tmp_path: Path) -> None:
    from blueprint_pipeline.control_plane_lane_scratch_census import build_census

    work, inputs = tmp_path / "work", tmp_path / "inputs"
    work.mkdir()
    inputs.mkdir()
    (work / "g1-unknown").mkdir()
    queue_root = tmp_path / "queue"
    queue_root.mkdir()
    (queue_root / "pending").symlink_to(tmp_path / "outside", target_is_directory=True)
    processing = queue_root / "processing"
    processing.mkdir()
    (processing / "bad.json").symlink_to(tmp_path / "outside")
    (tmp_path / "proc").mkdir()
    (tmp_path / "pins").mkdir()
    (tmp_path / "release").mkdir()
    report = build_census(work_root=work, inputs_root=inputs, process_root=tmp_path / "proc",
                          pins_root=tmp_path / "pins", queue_roots=(queue_root,),
                          release_link=tmp_path / "release", active_run_roots=())
    assert report["status"] == "incomplete"
    assert "queue_inventory_unavailable" in report["scan_errors"]
    assert "queue_inventory_unreadable" in report["scan_errors"]


def test_census_does_not_treat_unreadable_queue_state_as_absent(tmp_path: Path, monkeypatch) -> None:
    from blueprint_pipeline.control_plane_lane_scratch_census import build_census

    work, inputs = tmp_path / "work", tmp_path / "inputs"
    work.mkdir()
    inputs.mkdir()
    folder = work / "g1-unknown"
    folder.mkdir()
    queue = tmp_path / "queue" / "pending"
    queue.mkdir(parents=True)
    (queue / "item.json").write_text(json.dumps({"scratch": folder.name}))
    (tmp_path / "proc").mkdir()
    (tmp_path / "pins").mkdir()
    (tmp_path / "release").mkdir()
    original_exists = Path.exists
    monkeypatch.setattr(Path, "exists", lambda path: False if path == queue else original_exists(path))
    report = build_census(work_root=work, inputs_root=inputs, process_root=tmp_path / "proc",
                          pins_root=tmp_path / "pins", queue_roots=(queue.parent,),
                          release_link=tmp_path / "release", active_run_roots=())
    assert report["rows"][0]["references"] == ["queue"]


def test_census_deadline_marks_partial_scan_incomplete(tmp_path: Path) -> None:
    from blueprint_pipeline.control_plane_lane_scratch_census import build_census

    work, inputs = tmp_path / "work", tmp_path / "inputs"
    work.mkdir()
    inputs.mkdir()
    (work / "g1-unknown").mkdir()
    report = build_census(work_root=work, inputs_root=inputs, process_root=tmp_path / "proc",
                          pins_root=tmp_path / "pins", queue_roots=(),
                          release_link=tmp_path / "release", active_run_roots=(),
                          max_seconds=1e-9)
    assert report["status"] == "incomplete"
    assert "census_deadline_reached" in report["scan_errors"]


def test_census_cli_prints_annotation_table_and_json(tmp_path: Path) -> None:
    work, inputs = tmp_path / "work", tmp_path / "inputs"
    work.mkdir()
    inputs.mkdir()
    (work / "drawer-diagnostic").mkdir()
    (tmp_path / "proc").mkdir()
    (tmp_path / "pins").mkdir()
    release = tmp_path / "release"
    release.mkdir()
    report_path = tmp_path / "census.json"
    script = Path(__file__).resolve().parents[1] / "scripts" / "lane_scratch_census.py"
    result = subprocess.run([
        sys.executable, str(script), "--work-root", str(work), "--inputs-root", str(inputs),
        "--process-root", str(tmp_path / "proc"), "--pins-root", str(tmp_path / "pins"),
        "--release-link", str(release), "--active-run-inventory-empty",
        "--queue-inventory-empty",
        "--json-out", str(report_path),
    ], check=False, text=True, capture_output=True)
    assert result.returncode == 0
    assert "drawer-diagnostic" in result.stdout and "owner decision" in result.stdout
    assert "age seconds" in result.stdout and "approved expiry" in result.stdout
    assert json.loads(report_path.read_text())["rows"][0]["family"] == "drawer"


def test_census_table_escapes_tabs_in_folder_names(tmp_path: Path) -> None:
    from scripts.lane_scratch_census import _table

    line = _table({"rows": [{"family": "drawer", "owner_guess": "drawer-lane",
                              "allocated_bytes": 1, "newest_mtime_epoch": 1000,
                              "age_seconds": 1, "references": [],
                              "path": str(tmp_path / "drawer-\tname")}],
                   "status": "complete", "candidate_count": 1,
                   "unique_allocated_bytes": 1, "scan_errors": []}).splitlines()[1]
    assert len(line.split("\t")) == 9
    assert "\\t" in line
