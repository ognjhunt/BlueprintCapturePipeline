"""The orphan scratch reference census: per-kind evidence, read-only, never optimistic."""

# Covers (for impacted-test selection):
#   src/blueprint_pipeline/control_plane_lane_scratch_census.py
#   src/blueprint_pipeline/control_plane_capacity_controller.py
#   src/blueprint_pipeline/control_plane_storage_gc.py

from __future__ import annotations

import json
from collections import namedtuple
from pathlib import Path

import pytest

from blueprint_pipeline import control_plane_capacity_controller as cap
from blueprint_pipeline.control_plane_lane_scratch_census import (
    REFERENCE_CENSUS_FILENAME,
    REFERENCE_KINDS,
    build_reference_census,
    publish_reference_census,
)
from blueprint_pipeline.control_plane_storage_pins import write_storage_pin

NOW = 1_800_000_000.0
# Kinds a synthetic folder can be held by, and the folder each one holds.
HELD = ("queue", "active_run", "capture_reconstruction", "pubsub_handoff", "door_request", "process",
        "container_mount", "docker_container", "unit_file", "env_file", "release_source", "pin",
        "live_release")


def _world(tmp_path: Path) -> dict:
    work, inputs = tmp_path / "work", tmp_path / "inputs"
    folders = {kind: work / f"held-by-{kind.replace('_', '-')}" for kind in HELD}
    folders["free"] = inputs / "nothing-refers-to-me"
    for folder in folders.values():
        folder.mkdir(parents=True)
        (folder / "data.bin").write_bytes(b"x" * 64)
    etc = tmp_path / "etc"

    def record(root: Path, relative: str, kind: str, text: str | None = None) -> Path:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text or json.dumps({"path": str(folders[kind])}))
        return root

    queue = record(tmp_path / "queue", "pending/m.json", "queue")
    # Finding 1: a run record names the folder while no process holds it.
    runs = record(tmp_path / "launch-runs", "run-7/attempt-2/run.json", "active_run")
    capture = record(tmp_path / "capture-queue", "pending/c.json", "capture_reconstruction")
    handoff = record(tmp_path / "handoffs", "bucket/scenes/s1/handoff.json", "pubsub_handoff")
    door = record(tmp_path / "door-requests", "r1.json", "door_request")
    docker = record(tmp_path / "docker", "abc123/hostconfig.json", "docker_container",
                    json.dumps({"Binds": [f"{folders['docker_container']}:/inputs:ro"]}))
    # A unit under /usr/lib/systemd/system, beside the /etc one.
    usr_units = record(tmp_path / "usr-lib-systemd", "job.service", "unit_file",
                       f"[Service]\nWorkingDirectory={folders['unit_file']}\n")
    env = record(etc / "blueprint", "pipeline.env", "env_file", f"SDK_ROOT={folders['env_file']}\n")
    record(etc / "blueprint" / "provider-secrets", "key.env", "free")  # never read
    release = folders["live_release"] / "release"
    record(release, "src/paths.py", "release_source", f"ROOT = '{folders['release_source']}'\n")
    link = tmp_path / "release-link"
    link.symlink_to(release)
    proc = tmp_path / "proc"
    me = proc / "self"
    (me / "ns").mkdir(parents=True)
    (me / "ns" / "mnt").symlink_to("mnt:[1]")
    (me / "mountinfo").write_text(f"1 0 8:1 / {tmp_path} rw - ext4 /dev/x rw\n")
    for pid, namespace, cwd in (("4242", "mnt:[1]", folders["process"]), ("5151", "mnt:[2]", "/")):
        (proc / pid / "fd").mkdir(parents=True)
        (proc / pid / "cmdline").write_bytes(b"python\0")
        (proc / pid / "environ").write_bytes(b"")
        (proc / pid / "cwd").symlink_to(cwd)
        (proc / pid / "ns").mkdir()
        (proc / pid / "ns" / "mnt").symlink_to(namespace)
    relative = folders["container_mount"].relative_to(tmp_path)
    (proc / "5151" / "mountinfo").write_text(f"7 1 8:1 /{relative} /inputs rw - ext4 /dev/x rw\n")
    pins = tmp_path / "pins"
    write_storage_pin(pins_root=pins, kind="preparation", owner_id="prep-1", paths=[folders["pin"]],
                      now=lambda: NOW)
    kwargs = {
        "work_root": work, "inputs_root": inputs, "process_root": proc, "pins_root": pins,
        "queue_roots": [queue], "release_link": link, "active_run_roots": [runs],
        "unit_directories": [tmp_path / "etc-systemd", usr_units], "env_directories": [env],
        "env_excluded": [etc / "blueprint" / "provider-secrets"], "capture_roots": [capture],
        "handoff_roots": [handoff], "door_request_roots": [door], "docker_containers": docker,
        "source_roots": [], "now": NOW,
    }
    return {"folders": folders, "kwargs": kwargs}


def _rows(census: dict) -> dict:
    return {row["path"]: row for row in census["rows"]}


def test_every_kind_is_checked_and_holds_its_folder_only_the_free_one_has_no_reference(tmp_path):
    world = _world(tmp_path)
    before = sorted(str(path) for path in tmp_path.rglob("*"))

    census = build_reference_census(**world["kwargs"])

    assert {kind: row["status"] for kind, row in census["kinds"].items()} == dict.fromkeys(REFERENCE_KINDS,
                                                                                          "checked")
    assert census["status"] == "complete"
    assert "not a statement that deletion is safe" in census["statement"]
    rows = _rows(census)
    for kind in HELD:
        assert kind in rows[str(world["folders"][kind])]["references"], kind
    assert rows[str(world["folders"]["free"])]["references"] == []  # the secrets file was not read
    assert census["mutations"] == 0
    assert sorted(str(path) for path in tmp_path.rglob("*")) == before


def test_a_run_record_holds_its_folder_between_processes(tmp_path):
    world = _world(tmp_path)
    kwargs = {**world["kwargs"], "active_run_roots": [tmp_path / "empty-runs"]}
    (tmp_path / "empty-runs").mkdir()
    held = str(world["folders"]["active_run"])

    assert build_reference_census(**kwargs)["rows"] and "active_run" not in _rows(
        build_reference_census(**kwargs))[held]["references"]
    assert "active_run" in _rows(build_reference_census(**world["kwargs"]))[held]["references"]


def test_a_running_container_mounting_the_scratch_root_is_root_reachability_not_a_hold(tmp_path):
    world = _world(tmp_path)
    (tmp_path / "proc" / "5151" / "mountinfo").write_text("7 1 8:1 /work /data rw - ext4 /dev/x rw\n")
    (tmp_path / "proc" / "5151" / "fd" / "3").symlink_to("/data/held-by-queue/data.bin")

    census = build_reference_census(**world["kwargs"])

    rows = _rows(census)
    assert {"kind": "container_mount", "root": str(tmp_path / "work"), "folders": len(HELD)} in census[
        "root_reachable_by"]
    assert "container_mount" in rows[str(world["folders"]["queue"])]["references"]  # the one it uses
    assert "container_mount" not in rows[str(world["folders"]["pin"])]["references"]
    assert all(row["not_proven"] == [] for row in rows.values())


def test_a_sandboxed_unit_seeing_the_host_volume_holds_only_what_it_uses(tmp_path):
    world = _world(tmp_path)
    proc = tmp_path / "proc" / "5151"
    (proc / "mountinfo").write_text(f"7 1 8:1 / {tmp_path} rw - ext4 /dev/x rw\n")
    used = world["folders"]["free"]
    (proc / "fd" / "3").symlink_to(used / "data.bin")

    rows = _rows(build_reference_census(**world["kwargs"]))

    assert "container_mount" in rows[str(used)]["references"]
    assert "container_mount" not in rows[str(world["folders"]["queue"])]["references"]


@pytest.mark.parametrize(("change", "kind", "status"), [
    ({"queue_roots": None}, "queue", "not_checked"),
    ({"active_run_roots": None}, "active_run", "not_checked"),
    ({"release_link": Path("/nonexistent/release")}, "release_source", "not_checked"),
    ({"pins_root": Path("/nonexistent/pins")}, "pin", "failed"),
    ({"queue_roots": [Path("/nonexistent/queue")]}, "queue", "failed"),
])
def test_a_kind_not_checked_or_failed_makes_the_census_incomplete_and_totals_unknown(
        tmp_path, change, kind, status):
    world = _world(tmp_path)

    census = build_reference_census(**{**world["kwargs"], **change})

    assert census["kinds"][kind]["status"] == status
    assert census["status"] == "incomplete"
    free = str(world["folders"]["free"])
    projection = cap.orphan_scratch_census_projection(
        census, {"orphan_scratch_root_bytes": [[free, 9]]}, now=NOW + 60)
    assert projection["status"] == "incomplete"
    assert projection["no_specific_reference_found_bytes"] is None
    assert projection["roots"][0]["disposition"] == "not_proven"


def test_an_unreadable_record_or_a_deadline_fails_the_kind(tmp_path, monkeypatch):
    world = _world(tmp_path)
    (tmp_path / "door-requests" / "big.json").write_bytes(b" " * (4 * 1024 * 1024 + 1))
    census = build_reference_census(**world["kwargs"])
    assert census["kinds"]["door_request"]["status"] == "failed"
    assert census["status"] == "incomplete"

    ticks = iter([0.0] + [10_000.0] * 100_000)
    monkeypatch.setattr("blueprint_pipeline.control_plane_lane_scratch_census.time.monotonic",
                        lambda: next(ticks))
    assert build_reference_census(**_world(tmp_path / "again")["kwargs"])["status"] == "incomplete"


@pytest.mark.parametrize("name", ["run+a", "x@y", "sp ace", "torch-2.1+cu118-wheels"])
def test_a_record_naming_a_folder_by_full_path_finds_it_whatever_its_name(tmp_path, name):
    world = _world(tmp_path)
    folder = tmp_path / "work" / name
    folder.mkdir()
    (tmp_path / "work" / "run").mkdir()
    (tmp_path / "door-requests" / "probe.json").write_text(json.dumps({"input": f"{folder}/"}))

    rows = _rows(build_reference_census(**world["kwargs"]))

    assert "door_request" in rows[str(folder)]["references"]


def test_a_full_path_match_respects_the_name_boundary(tmp_path):
    world = _world(tmp_path)
    for name in ("run", "run2"):
        (tmp_path / "work" / name).mkdir()
    (tmp_path / "door-requests" / "probe.json").write_text(json.dumps({"input": str(tmp_path / "work" / "run2")}))

    rows = _rows(build_reference_census(**world["kwargs"]))

    assert "door_request" in rows[str(tmp_path / "work" / "run2")]["references"]
    assert "door_request" not in rows[str(tmp_path / "work" / "run")]["references"]


def _spellings(path: Path) -> list[str]:
    text = str(path)
    return [json.dumps({"p": text}).replace("/", "\\/"),  # JSON \/ escaping
            str(path.parent / "x" / ".." / path.name),  # a .. segment
            text.replace("/", "%2F").replace(" ", "%20").replace("+", "%2B").replace(":", "%3A")
            .replace("@", "%40")]  # percent-encoding


@pytest.mark.parametrize("spelling", [0, 1, 2])
def test_a_plain_name_is_found_in_any_spelling_of_its_path(tmp_path, spelling):
    world = _world(tmp_path)
    folder = tmp_path / "work" / "plain-folder_1.2"
    folder.mkdir()
    (tmp_path / "door-requests" / "probe.json").write_text(_spellings(folder)[spelling])

    row = _rows(build_reference_census(**world["kwargs"]))[str(folder)]

    assert "door_request" in row["references"] and row["not_proven"] == []


@pytest.mark.parametrize("spelling", [0, 1, 2])
@pytest.mark.parametrize("name", ["a+b", "c:d", "e@f", "g h", "tilde~x"])
def test_a_special_name_is_not_proven_whatever_the_spelling(tmp_path, name, spelling):
    world = _world(tmp_path)
    folder = tmp_path / "work" / name
    folder.mkdir()
    (tmp_path / "door-requests" / "probe.json").write_text(_spellings(folder)[spelling])

    census = build_reference_census(**world["kwargs"])

    row = _rows(census)[str(folder)]
    assert {"reason": "name_not_reliably_matchable"} in row["not_proven"]
    projection = cap.orphan_scratch_census_projection(
        census, {"orphan_scratch_root_bytes": [[str(folder), 1]]}, now=NOW + 60)
    assert projection["roots"][0]["disposition"] in {"referenced", "not_proven"}
    assert projection["roots"][0]["disposition"] != "no_specific_reference_found"


def test_a_name_a_record_could_escape_is_never_no_specific_reference_found(tmp_path):
    world = _world(tmp_path)
    (tmp_path / "work" / 'quo"te').mkdir()

    row = _rows(build_reference_census(**world["kwargs"]))[str(tmp_path / "work" / 'quo"te')]

    assert row["references"] == [] and row["not_proven"] == [{"reason": "name_not_reliably_matchable"}]


@pytest.mark.parametrize("kind", ["docker_container", "unit_file", "release_source"])
def test_a_record_naming_only_the_scratch_root_changes_no_row_and_is_said_once(tmp_path, kind):
    world = _world(tmp_path)
    work = tmp_path / "work"
    if kind == "docker_container":  # stopped, set to restart, the whole scratch root bound in
        (tmp_path / "docker" / "def456").mkdir()
        (tmp_path / "docker" / "def456" / "hostconfig.json").write_text(json.dumps(
            {"Binds": [f"{work}:/scratch"], "RestartPolicy": {"Name": "always"}}))
    elif kind == "unit_file":
        (tmp_path / "usr-lib-systemd" / "sim.service").write_text(f"[Service]\nReadWritePaths={work}/\n")
    else:
        release = world["folders"]["live_release"] / "release"
        (release / "src" / "roots.py").write_text(f"WORK_ROOT = '{work}'\n")

    census = build_reference_census(**world["kwargs"])

    assert {"kind": kind, "root": str(work), "folders": len(HELD)} in census["root_reachable_by"]
    assert "cannot rule that out" in census["root_reachability_statement"]
    assert census["ancestor_references"] == []
    rows = _rows(census)
    assert all(row["not_proven"] == [] for row in rows.values())
    projection = cap.orphan_scratch_census_projection(
        census, {"orphan_scratch_root_bytes": [[path, 1] for path in rows]}, now=NOW + 60)
    by_root = {row["root"]: row["disposition"] for row in projection["roots"]}
    assert by_root[str(world["folders"]["free"])] == "no_specific_reference_found"
    assert projection["root_reachable_by"] and projection["no_specific_reference_found_bytes"] == 1


def test_an_intermediate_ancestor_makes_only_the_folders_below_it_not_proven(tmp_path):
    world = _world(tmp_path)
    lane = tmp_path / "work" / "lanes" / "g1"
    stray = lane / "unregistered-folder"
    stray.mkdir(parents=True)
    (tmp_path / "usr-lib-systemd" / "g1.service").write_text(f"[Service]\nReadWritePaths=-{lane}\n")

    census = build_reference_census(**world["kwargs"])

    rows = _rows(census)
    assert rows[str(stray)]["not_proven"] == [{"reason": "ancestor_referenced", "kind": "unit_file",
                                               "ancestor": str(lane)}]
    assert rows[str(world["folders"]["free"])]["not_proven"] == []
    assert census["root_reachable_by"] == []
    projection = cap.orphan_scratch_census_projection(
        census, {"orphan_scratch_root_bytes": [[str(stray), 5]]}, now=NOW + 60)
    assert projection["roots"][0]["disposition"] == "not_proven"


def _census(rows, *, observed=NOW, status="complete"):
    return {"schema_version": cap.ORPHAN_CENSUS_SCHEMA_VERSION, "status": status, "observed_at_epoch": observed,
            "kinds": dict.fromkeys(REFERENCE_KINDS, {"status": "checked"}), "candidate_listing_errors": [],
            "rows": rows, "mutations": 0}


def _row(path, references=(), use="unknown", run_time_only=False, mtime=NOW - 10 * 86400):
    return {"path": path, "references": list(references), "lane_family": "other", "use_guess": use,
            "run_time_only": run_time_only, "top_level_mtime_epoch": mtime, "not_proven": []}


def test_the_report_carries_family_use_age_and_the_evidence_statement():
    survey = {"status": "complete", "orphan_scratch_root_bytes_complete": True,
              "orphan_scratch_root_bytes": [["/w/sdk-inputs", 30], ["/w/held", 20], ["/w/unlisted", 10]]}
    census = _census([_row("/w/sdk-inputs", use="rebuildable_run_input_cache", run_time_only=True),
                      _row("/w/held", ["active_run"])])

    projection = cap.orphan_scratch_census_projection(census, survey, now=NOW + 600)

    assert "not a statement that deletion is safe" in projection["statement"]
    assert projection["status"] == "complete" and projection["age_seconds"] == 600.0
    assert projection["no_specific_reference_found_bytes"] == 30
    assert projection["no_specific_reference_found_run_time_only_bytes"] == 30
    assert projection["retirement_authorized"] is False
    first = projection["roots"][0]
    assert first == {"root": "/w/sdk-inputs", "allocated_bytes": 30, "disposition": "no_specific_reference_found",
                     "references": [], "not_proven": [], "lane_family": "other",
                     "use_guess": "rebuildable_run_input_cache",
                     "run_time_only": True, "top_level_unchanged_days": 10.0}
    by_root = {row["root"]: row["disposition"] for row in projection["roots"]}
    assert by_root == {"/w/sdk-inputs": "no_specific_reference_found", "/w/held": "referenced", "/w/unlisted": "not_proven"}
    assert "unreferenced" not in json.dumps(projection) and "safe_to" not in json.dumps(projection)


@pytest.mark.parametrize("census", [
    None,
    _census([_row("/w/free")], observed=NOW - 3 * 3600),
    _census([_row("/w/free")], status="incomplete"),
    {**_census([_row("/w/free")]), "kinds": {**dict.fromkeys(REFERENCE_KINDS, {"status": "checked"}),
                                             "env_file": {"status": "not_checked", "reason": "x"}}},
    {**_census([_row("/w/free")]), "schema_version": "control_plane_lane_scratch_census.v1"},
])
def test_a_missing_stale_incomplete_or_foreign_census_proves_nothing(census):
    projection = cap.orphan_scratch_census_projection(
        census, {"orphan_scratch_root_bytes": [["/w/free", 30]]}, now=NOW)
    assert projection["no_specific_reference_found_bytes"] is None
    assert projection["roots"][0]["disposition"] == "not_proven"


def test_the_capacity_summary_reports_the_census(tmp_path, monkeypatch):
    monkeypatch.setattr("blueprint_pipeline.task_evaluation_scene_spend.observe_configured_scene_project_spend",
                        lambda **_kwargs: None)
    gc_root = tmp_path / "storage-gc"
    gc_root.mkdir()
    (gc_root / REFERENCE_CENSUS_FILENAME).write_text(json.dumps(
        _census([_row("/mnt/blueprint-work/sdk-inputs", use="rebuildable_run_input_cache", run_time_only=True)],
                observed=NOW - 120)))
    survey = {"schema_version": "control_plane_disk_usage_survey.v1", "status": "complete",
              "observed_at_epoch": NOW, "mounts": [], "by_class": [], "top_roots": [], "top_owners": [],
              "unclassified_roots": [], "orphan_scratch_root_bytes_complete": True,
              "orphan_scratch_root_bytes": [["/mnt/blueprint-work/sdk-inputs", 30 * 1024**3]]}
    usage = namedtuple("usage", "total used free")

    cap.run_controller(mounts=[str(tmp_path)], report_root=tmp_path / "capacity",
                       reservation_root=tmp_path / "ledger", webhook_url="", volume=None, ack="", token="",
                       disk_usage=lambda _path: usage(154 * 1024**3, 54 * 1024**3, 100 * 1024**3),
                       now=NOW, survey=lambda **_k: survey, storage_gc_summary_path=gc_root / "summary.json")

    census = json.loads((tmp_path / "capacity" / "summary.json").read_text())["usage"]["orphan_scratch_census"]
    assert census["status"] == "complete" and census["age_seconds"] == 120.0
    assert census["no_specific_reference_found_bytes"] == 30 * 1024**3
    assert census["roots"][0]["run_time_only"] is True
    assert census["kinds"]["active_run"] == {"status": "checked"}


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
