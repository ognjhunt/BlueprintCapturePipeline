"""Storage GC reclaims the scratch inputs activation lookaheads leaked, only once the owner opts in."""

# Covers (for impacted-test selection):
#   src/blueprint_pipeline/control_plane_replay_cache_gc.py
#   src/blueprint_pipeline/control_plane_storage_gc.py
#   src/blueprint_pipeline/completed_replay_cache_retention.py
#   src/blueprint_pipeline/task_evaluation_stage_replay.py
#   deploy/systemd/blueprint-control-plane-storage-gc.service

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from blueprint_pipeline import completed_replay_cache_retention as retention
from blueprint_pipeline import control_plane_replay_cache_gc as replay_gc
from blueprint_pipeline import control_plane_storage_gc as gc_module
from blueprint_pipeline.control_plane_storage_gc import RUN_ACK, run_storage_gc
from blueprint_pipeline.control_plane_storage_roots import require_storage_class

NOW = 20_000_000.0
ACTIVATIONS = "/var/lib/blueprint/pipeline-control-plane/task-evaluation-configured-controls/scene-configuration-activations"


@pytest.fixture(autouse=True)
def no_live_readers(monkeypatch):
    # The host inventory is /proc; the retention module's own tests cover a live reader.
    monkeypatch.setattr(retention, "active_reference", lambda _root, **_kwargs: False)


def _noclass(*_args, **_kwargs) -> None:
    return None


def _activation(parent_root: Path, name: str = "scene-841007-preparation") -> tuple[Path, Path, Path]:
    """One activation whose lookahead replayed its parent and left a copied store blob behind."""

    lookahead = parent_root / name / "lookahead"
    replay_child = lookahead / "parent-scene-841007-20260927T000000Z-x1"
    store = replay_child / "prepared-references" / "content-addressed" / "sha256"
    store.mkdir(parents=True)
    payload = name.encode() * 1000
    blob = store / hashlib.sha256(payload).hexdigest()
    blob.write_bytes(payload)
    blob.chmod(0o440)
    report = replay_child / "stage_replay_report.v1.json"
    report.write_text(json.dumps({
        "schema_version": "task_evaluation_parent_replay_report.v1", "nothing_fetched": True,
        "paid_execution_requested": False, "provider_mutation_performed": False,
        "scratch_queue_root": str(replay_child / "launch-preparations"),
    }))
    lookahead_report = lookahead / ("f" * 64 + ".json")
    lookahead_report.write_text(json.dumps({"schema_version": "task_evaluation_progression_replay.v1"}))
    for path, stamp in ((blob, NOW - 7200), (report, NOW - 7100), (lookahead_report, NOW - 7000)):
        os.utime(path, (stamp, stamp))
    return blob, report, lookahead_report


def _tick(tmp_path: Path, parent_root: Path, **kwargs):
    return run_storage_gc(content_store_roots=[], derived_roots=[], queue_roots=[], pins_root=tmp_path / "pins",
                          now=lambda: NOW, replay_parent_roots=[parent_root], **{"classifier": _noclass, **kwargs})


def test_the_phase_leaves_every_file_outside_the_scratch_inputs(tmp_path) -> None:
    """Other files in a replay (a writable binary the standalone unit's rules would take)
    are not this phase's business: it opted into a replay's scratch inputs and nothing else."""

    parent_root = tmp_path / "scene-configuration-activations"
    blob, report, _lookahead_report = _activation(parent_root)
    binary = report.parent / "render.ply"
    binary.write_bytes(b"b" * 100_000)
    os.utime(binary, (NOW - 7200, NOW - 7200))
    size = blob.stat().st_size

    assert _tick(tmp_path, parent_root)["replay_caches"]["estimated_candidate_bytes"] == size
    tick = _tick(tmp_path, parent_root, apply=True, ack=RUN_ACK, replay_cache_retention_enabled=True)

    assert tick["replay_caches"]["removed_bytes"] == size
    assert not blob.exists() and binary.exists()


def test_replay_cache_phase_plans_by_default_and_applies_only_when_enabled(tmp_path) -> None:
    parent_root = tmp_path / "scene-configuration-activations"
    blob, report, lookahead_report = _activation(parent_root)
    size = blob.stat().st_size

    for apply, enabled in ((True, False), (False, True)):
        tick = _tick(tmp_path, parent_root, apply=apply, ack=RUN_ACK if apply else "",
                     replay_cache_retention_enabled=enabled)
        phase = tick["replay_caches"]
        assert (phase["status"], phase["enabled"], phase["replay_root_count"]) == ("dry_run", enabled, 1)
        assert (phase["estimated_candidate_bytes"], phase["removed_bytes"]) == (size, 0)
        assert "candidate_bytes" not in phase, "a tick that only plans verifies nothing"
        assert blob.exists() and "phase_errors" not in tick and "alerts" not in tick

    tick = _tick(tmp_path, parent_root, apply=True, ack=RUN_ACK, replay_cache_retention_enabled=True)

    phase = tick["replay_caches"]
    assert (phase["status"], phase["enabled"]) == ("applied", True)
    assert phase["candidate_bytes"] == phase["removed_bytes"] == size
    assert "estimated_candidate_bytes" not in phase
    assert not blob.exists()
    assert report.is_file() and lookahead_report.is_file(), "the reports the activation records stay"
    assert tick["report_digest"] == gc_module.canonical_digest(tick, digest_field="report_digest")
    assert _tick(tmp_path, parent_root, apply=True, ack=RUN_ACK,
                 replay_cache_retention_enabled=True)["replay_caches"]["candidate_bytes"] == 0


def test_the_phase_waits_an_hour_after_a_replay_closes(tmp_path) -> None:
    """The phase passes its own one-hour closed gate, not the retention module's one-minute
    default: a lookahead replay that finished half an hour ago is left alone."""

    parent_root = tmp_path / "scene-configuration-activations"
    blob, report, _lookahead_report = _activation(parent_root)
    size = blob.stat().st_size
    os.utime(report, (NOW - 1800, NOW - 1800))

    assert _tick(tmp_path, parent_root)["replay_caches"]["estimated_candidate_bytes"] == 0
    enabled = {"apply": True, "ack": RUN_ACK, "replay_cache_retention_enabled": True}
    assert _tick(tmp_path, parent_root, **enabled)["replay_caches"]["removed_bytes"] == 0
    assert blob.exists()

    os.utime(report, (NOW - 3601, NOW - 3601))
    assert _tick(tmp_path, parent_root, **enabled)["replay_caches"]["removed_bytes"] == size


def test_plan_only_ticks_never_hash(tmp_path, monkeypatch) -> None:
    """Until the owner opts in, an hourly tick must not read the backlog: hashing every
    candidate copy each hour would reread about 13 GiB on the root disk for nothing. It
    sweeps no process table either; a tick that applies rechecks everything."""

    parent_root = tmp_path / "scene-configuration-activations"
    blob, _report, _lookahead_report = _activation(parent_root)
    size = blob.stat().st_size
    real_file_sha = retention.file_sha

    def refuse(*args, **_kwargs):
        raise AssertionError(f"read {args[:2]} on a tick that only plans")

    for name in ("file_sha", "_held_sha", "active_reference"):
        monkeypatch.setattr(retention, name, refuse)
    for apply, enabled in ((True, False), (False, True), (False, False)):
        tick = _tick(tmp_path, parent_root, apply=apply, ack=RUN_ACK if apply else "",
                     replay_cache_retention_enabled=enabled)
        phase = tick["replay_caches"]
        assert ("phase_errors" in tick, phase["errors"]) == (False, [])
        assert phase["estimated_candidate_bytes"] == size

    hashed: list[Path] = []
    monkeypatch.undo()
    monkeypatch.setattr(retention, "active_reference", lambda _root, **_kwargs: False)
    monkeypatch.setattr(retention, "file_sha", lambda path: hashed.append(path) or real_file_sha(path))
    tick = _tick(tmp_path, parent_root, apply=True, ack=RUN_ACK, replay_cache_retention_enabled=True)
    assert tick["replay_caches"]["removed_bytes"] == size and hashed, "a tick that applies rechecks the report"


def _real_lookahead(tmp_path: Path, monkeypatch, *, released: bool) -> tuple[dict, Path, bytes]:
    """The real parent replay run into an activation's lookahead, the store on another volume.

    With ``released=False`` it leaves its scratch inputs behind, as every replay before the
    release did: that is the 2026-09-27 backlog.
    """

    from functools import partial
    from types import SimpleNamespace

    from blueprint_pipeline import task_evaluation_stage_replay as replay
    from tests.test_task_evaluation_stage_replay import (
        PREFIXES, SERVICE_ACCOUNT, _materialized_parent, _store_on_another_volume,
    )

    monkeypatch.setattr(replay, "DEFAULT_RESERVATION_ROOT", tmp_path / "disk-reservations")
    monkeypatch.setattr(replay, "reserve_control_plane_disk", partial(
        replay.reserve_control_plane_disk,
        disk_usage=lambda _path: SimpleNamespace(total=512 * 2**30, used=128 * 2**30, free=384 * 2**30)))
    if not released:
        monkeypatch.setattr(replay, "_release_scratch_inputs", lambda _path: {"files": 0, "inodes": 0, "bytes": 0})
    request, parent_queue, input_root, children = _materialized_parent(tmp_path)
    store = input_root / "content-addressed" / "sha256"
    other = b"another preparation's input" * 100
    (store / hashlib.sha256(other).hexdigest()).write_bytes(other)
    _store_on_another_volume(monkeypatch, store)
    monkeypatch.setenv(replay.driver.CHILD_QUEUE_ENV, str(children))
    activations = tmp_path / "scene-configuration-activations"
    report = replay.replay_parent(
        parent_queue_root=parent_queue, preparation_id=request["preparation_id"], child_queue_root=children,
        input_root=input_root, replay_root=activations / request["preparation_id"] / "lookahead",
        allowed_uri_prefixes=PREFIXES, service_account=SERVICE_ACCOUNT,
        advancer=lambda context: {"status": "waiting_for_child", "evidence_refs": []})
    return report, activations, other


def _enabled_tick(tmp_path: Path, activations: Path, report: dict, *, apply: bool = True) -> dict:
    return run_storage_gc(content_store_roots=[], derived_roots=[], queue_roots=[], pins_root=tmp_path / "pins",
                          now=lambda: os.path.getmtime(report["report_path"]) + 7200, classifier=_noclass,
                          replay_parent_roots=[activations], apply=apply, ack=RUN_ACK if apply else "",
                          replay_cache_retention_enabled=True)


def test_a_lookahead_now_keeps_its_report_and_queue_and_no_scratch_inputs(tmp_path, monkeypatch) -> None:
    report, activations, _other = _real_lookahead(tmp_path, monkeypatch, released=True)

    assert not (Path(report["report_path"]).parent / "prepared-references").exists()
    assert Path(report["report_path"]).is_file() and any(Path(report["scratch_queue_root"]).rglob("*.json"))
    tick = _enabled_tick(tmp_path, activations, report)
    assert (tick["replay_caches"]["removed_bytes"], "phase_errors" in tick) == (0, False)


def test_a_lookahead_the_replay_leaked_is_reclaimed_with_its_linked_materializations(tmp_path, monkeypatch) -> None:
    """The 2026-09-27 backlog as the real replay wrote it, before it released its inputs: the
    phase removes every copy together with the names the worker linked to it, counts each copy
    once, and keeps the parent report and the scratch queue."""

    report, activations, _other = _real_lookahead(tmp_path, monkeypatch, released=False)
    inputs = Path(report["report_path"]).parent / "prepared-references"
    names = [path for path in inputs.rglob("*") if path.is_file() and not path.is_symlink()]
    copies = {(path.stat().st_dev, path.stat().st_ino): path.stat().st_size for path in names}
    assert len(names) > len(copies), "the worker linked materialized names to the copies"

    tick = _enabled_tick(tmp_path, activations, report)

    assert tick["replay_caches"]["removed_bytes"] == sum(copies.values()) and "phase_errors" not in tick
    assert not [path for path in inputs.rglob("*") if path.is_file()]
    assert Path(report["report_path"]).is_file() and any(Path(report["scratch_queue_root"]).rglob("*.json"))


def test_finished_lookahead_scratch_inputs_are_reclaimed_whole(tmp_path, monkeypatch) -> None:
    """2026-09-27, 18:30 UTC: the phase estimated 4.8 GB across 105 lookaheads whose activations
    held 12.95 GiB. Nearly all of a lookahead's bytes sat in parent-*/prepared-references, much of
    it under no store name, which the store-copy rule cannot see. A finished replay made that whole
    tree as scratch, so the phase now takes all of it: the store copies, the names the worker
    linked to them, copies no store name covers, and the directories they leave empty. The parent
    report and the scratch queue stay, and each inode is counted once."""

    report, activations, _other = _real_lookahead(tmp_path, monkeypatch, released=False)
    inputs = Path(report["report_path"]).parent / "prepared-references"
    preparation = inputs / report["preparation_id"]
    blob = next((inputs / "content-addressed" / "sha256").iterdir())
    unlinked = preparation / "copied" / blob.name  # the blob's bytes, but its own inode and no store name
    unlinked.parent.mkdir()
    unlinked.write_bytes(blob.read_bytes())
    derived = preparation / "derived" / "view-0.bin"
    derived.parent.mkdir()
    derived.write_bytes(b"derived from an input" * 100)
    (preparation / "derived" / "empty").mkdir()
    stamp = os.path.getmtime(report["report_path"]) - 60
    for path in (unlinked, derived):
        os.utime(path, (stamp, stamp))
    names = [path for path in inputs.rglob("*") if path.is_file() and not path.is_symlink()]
    inodes = {(path.stat().st_dev, path.stat().st_ino): path.stat().st_size for path in names}
    assert len(names) > len(inodes), "the worker linked materialized names to the copies"

    estimate = _enabled_tick(tmp_path, activations, report, apply=False)["replay_caches"]
    tick = _enabled_tick(tmp_path, activations, report)

    phase = tick["replay_caches"]
    assert estimate["estimated_candidate_bytes"] == phase["candidate_bytes"] == phase["removed_bytes"] == sum(
        inodes.values())
    assert (phase["skipped"], phase["errors"], "phase_errors" in tick) == ([], [], False)
    assert inputs.is_dir() and not any(inputs.iterdir()), "every file and every emptied directory went"
    assert Path(report["report_path"]).is_file() and any(Path(report["scratch_queue_root"]).rglob("*.json"))


def test_phase_estimate_counts_scratch_inputs_without_hashing(tmp_path, monkeypatch) -> None:
    """A plan-only tick counts every scratch input a finished lookahead left, whatever its name,
    each inode once however many names it has, still from names, links and sizes alone: it hashes
    nothing, reads no file's bytes and sweeps no process table. A tick that applies removes what
    was estimated and hashes only the replay's report: nothing about a scratch input rests on a
    digest."""

    parent_root = tmp_path / "scene-configuration-activations"
    blob, report, _lookahead_report = _activation(parent_root)
    preparation = report.parent / "prepared-references" / "scene-841007-preparation"
    linked = preparation / blob.name
    derived = preparation / "derived" / "frame.json"
    derived.parent.mkdir(parents=True)
    os.link(blob, linked)
    derived.write_bytes(b"{}" * 500)
    os.utime(derived, (NOW - 7200, NOW - 7200))
    size = blob.stat().st_size + derived.stat().st_size

    def refuse(*args, **_kwargs):
        raise AssertionError(f"read {args[:2]} on a tick that only plans")

    for name in ("file_sha", "_held_sha", "active_reference"):
        monkeypatch.setattr(retention, name, refuse)
    phase = _tick(tmp_path, parent_root)["replay_caches"]
    assert (phase["estimated_candidate_bytes"], phase["errors"]) == (size, [])

    monkeypatch.undo()
    monkeypatch.setattr(retention, "active_reference", lambda _root, **_kwargs: False)
    monkeypatch.setattr(retention, "_held_sha", refuse)
    hashed: list[Path] = []
    real_file_sha = retention.file_sha
    monkeypatch.setattr(retention, "file_sha", lambda path: hashed.append(path) or real_file_sha(path))
    phase = _tick(tmp_path, parent_root, apply=True, ack=RUN_ACK, replay_cache_retention_enabled=True)["replay_caches"]

    assert phase["candidate_bytes"] == phase["removed_bytes"] == size
    assert set(hashed) == {report}
    assert not blob.exists() and not linked.exists() and not derived.exists() and report.is_file()


def test_replay_cache_phase_refuses_a_non_work_root(tmp_path) -> None:
    parent_root = tmp_path / "scene-configuration-activations"
    blob, _report, _lookahead_report = _activation(parent_root)

    with pytest.raises(ValueError, match="^control_plane_storage_gc_replay_root_class:cache$"):
        replay_gc.reclaim_replay_caches(
            parent_roots=["/var/lib/blueprint/task-evaluation-inputs/prepared-references"], apply=True,
            enabled=True, now=lambda: NOW, classifier=require_storage_class)

    tick = _tick(tmp_path, parent_root, apply=True, ack=RUN_ACK, replay_cache_retention_enabled=True,
                 classifier=require_storage_class)

    assert tick["replay_caches"] == {"status": "error", "error": "ValueError"}
    assert tick["phase_errors"] == ["replay_caches"]
    assert blob.exists()


def test_replay_cache_phase_never_follows_a_link(tmp_path) -> None:
    parent_root = tmp_path / "scene-configuration-activations"
    elsewhere = tmp_path / "elsewhere"
    linked_blob, _, _ = _activation(elsewhere, "linked-preparation")
    lookahead_blob, _, _ = _activation(elsewhere, "linked-lookahead")
    parent_root.mkdir()
    (parent_root / "linked-preparation").symlink_to(elsewhere / "linked-preparation", target_is_directory=True)
    (parent_root / "linked-lookahead").mkdir()
    (parent_root / "linked-lookahead" / "lookahead").symlink_to(
        elsewhere / "linked-lookahead" / "lookahead", target_is_directory=True)

    phase = _tick(tmp_path, parent_root, apply=True, ack=RUN_ACK, replay_cache_retention_enabled=True)["replay_caches"]

    assert phase["replay_root_count"] == 0 and phase["removed_bytes"] == 0
    assert phase["skipped"] == [
        {"root": str(parent_root / "linked-lookahead"), "reason": "symlink_not_followed"},
        {"root": str(parent_root / "linked-preparation"), "reason": "symlink_not_followed"},
    ]
    assert linked_blob.exists() and lookahead_blob.exists()

    linked_root = tmp_path / "linked-activations"
    linked_root.symlink_to(parent_root, target_is_directory=True)
    tick = _tick(tmp_path, linked_root, apply=True, ack=RUN_ACK, replay_cache_retention_enabled=True)
    assert tick["replay_caches"] == {"status": "error", "error": "ValueError"}


def test_one_lookahead_error_is_recorded_and_the_others_still_reclaim(tmp_path, monkeypatch) -> None:
    parent_root = tmp_path / "scene-configuration-activations"
    good, _, _ = _activation(parent_root, "a-preparation")
    kept, _, _ = _activation(parent_root, "b-preparation")
    size = good.stat().st_size
    real_plan = retention.plan_replay_cache_retention

    def plan(*, replay_root, **kwargs):
        if Path(replay_root).parent.name == "b-preparation":
            raise ValueError("replay_cache_root_unsafe")
        return real_plan(replay_root=replay_root, **kwargs)

    monkeypatch.setattr(retention, "plan_replay_cache_retention", plan)

    phase = replay_gc.reclaim_replay_caches(parent_roots=[parent_root], apply=True, enabled=True,
                                            now=lambda: NOW, classifier=_noclass)

    assert phase["errors"] == [{"root": str(parent_root / "b-preparation" / "lookahead"), "error": "ValueError"}]
    assert (phase["replay_root_count"], phase["removed_bytes"]) == (2, size)
    assert not good.exists() and kept.exists()


def test_the_phase_report_is_bounded(tmp_path, monkeypatch) -> None:
    parent_root = tmp_path / "scene-configuration-activations"
    _activation(parent_root)

    def plan(**_kwargs):
        return {"rows": [], "candidate_bytes": 0,
                "kept": [{"root": f"replay-{index}", "reason": "active_reference"} for index in range(53)]}

    monkeypatch.setattr(retention, "plan_replay_cache_retention", plan)

    phase = replay_gc.reclaim_replay_caches(parent_roots=[parent_root], apply=True, enabled=True,
                                            now=lambda: NOW, classifier=_noclass)

    assert len(phase["kept"]) == 50 and phase["omitted_kept_count"] == 3
    assert (phase["skipped"], phase["omitted_skipped_count"]) == ([], 0)
    assert (phase["errors"], phase["omitted_errors_count"]) == ([], 0)


@pytest.mark.parametrize("value", [None, "", "1", "true", " YES ", "0", "false", "No", "sometimes", "2"])
def test_replay_cache_setting_parses_like_scene_retirement(value) -> None:
    replay_env = {} if value is None else {replay_gc.REPLAY_CACHE_RETENTION_ENV: value}
    scene_env = {} if value is None else {gc_module.SCENE_WORKSPACE_RETIREMENT_ENV: value}

    enabled, alert = replay_gc.replay_cache_retention_setting(replay_env)
    scene_enabled, scene_alert = gc_module.scene_workspace_retirement_setting(scene_env)

    assert enabled == scene_enabled
    assert alert == (None if scene_alert is None else "replay_cache_retention_setting_invalid")


def test_replay_cache_retention_needs_its_own_opt_in() -> None:
    others = {gc_module.SCENE_WORKSPACE_RETIREMENT_ENV: "1", gc_module.EVIDENCE_OFFLOAD_ENV: "1"}
    assert replay_gc.replay_cache_retention_setting(others) == (False, None)
    assert replay_gc.replay_cache_retention_setting({replay_gc.REPLAY_CACHE_RETENTION_ENV: "1"}) == (True, None)


def test_an_invalid_replay_cache_setting_only_plans_and_alerts(tmp_path) -> None:
    parent_root = tmp_path / "scene-configuration-activations"
    blob, _, _ = _activation(parent_root)
    alert = "replay_cache_retention_setting_invalid"

    tick = _tick(tmp_path, parent_root, apply=True, ack=RUN_ACK, replay_cache_retention_enabled=False,
                 replay_cache_retention_alert=alert, scene_workspace_retirement_alert="scene_workspace_retirement_setting_invalid")

    assert tick["alerts"] == ["scene_workspace_retirement_setting_invalid", alert]
    # As the scene workspace phase does, the phase's own record carries its alert too.
    assert (tick["replay_caches"]["status"], tick["replay_caches"]["alerts"]) == ("dry_run", [alert])
    assert blob.exists()


def test_both_opt_ins_share_one_parser() -> None:
    opt_ins = (
        (replay_gc.replay_cache_retention_setting, replay_gc.REPLAY_CACHE_RETENTION_ENV,
         "replay_cache_retention_setting_invalid"),
        (gc_module.scene_workspace_retirement_setting, gc_module.SCENE_WORKSPACE_RETIREMENT_ENV,
         "scene_workspace_retirement_setting_invalid"),
    )
    for parse, name, invalid in opt_ins:
        for value in ("1", "no", "", "later"):
            assert parse({name: value}) == replay_gc._truthy_setting({name: value}, name, invalid)


def test_the_command_line_reads_replay_roots_and_the_opt_in(tmp_path, monkeypatch, capsys) -> None:
    seen: list[dict] = []

    def run(**kwargs):
        seen.append(kwargs)
        return {"schema_version": gc_module.RUN_SCHEMA_VERSION, "report_digest": "sha256:0"}

    monkeypatch.setattr(gc_module, "run_storage_gc", run)
    monkeypatch.setenv(replay_gc.REPLAY_PARENT_ROOTS_ENV, "/a/activations:/b/activations")
    monkeypatch.setenv(replay_gc.REPLAY_CACHE_RETENTION_ENV, "sometimes")

    assert gc_module.main(["run", "--pins-root", str(tmp_path / "pins")]) == 0
    monkeypatch.setenv(replay_gc.REPLAY_CACHE_RETENTION_ENV, "1")
    assert gc_module.main(["run", "--pins-root", str(tmp_path / "pins"), "--replay-parent-root", "/c/activations"]) == 0

    assert seen[0]["replay_parent_roots"] == ["/a/activations", "/b/activations"]
    assert (seen[0]["replay_cache_retention_enabled"], seen[0]["replay_cache_retention_alert"]) == (
        False, "replay_cache_retention_setting_invalid")
    assert "storage_gc_alert:replay_cache_retention_setting_invalid" in capsys.readouterr().err
    assert seen[1]["replay_parent_roots"] == ["/c/activations"]
    assert (seen[1]["replay_cache_retention_enabled"], seen[1]["replay_cache_retention_alert"]) == (True, None)


def test_gc_unit_can_write_the_replay_parent_roots_it_names() -> None:
    unit = (Path(__file__).resolve().parents[1] / "deploy/systemd/blueprint-control-plane-storage-gc.service").read_text(
        encoding="utf-8")
    roots: list[str] = []
    writable: set[str] = set()
    read_only: set[str] = set()
    for line in unit.splitlines():
        if line.startswith(f"Environment={replay_gc.REPLAY_PARENT_ROOTS_ENV}="):
            roots.extend(part for part in line.split("=", 2)[2].split(":") if part)
        if line.startswith("ReadWritePaths="):
            writable.update(part.lstrip("-") for part in line.split("=", 1)[1].split())
        if line.startswith("ReadOnlyPaths="):
            read_only.update(part.lstrip("-") for part in line.split("=", 1)[1].split())

    assert roots == [ACTIVATIONS]
    assert all(root in writable for root in roots)
    # A read-only entry at or below a writable root would win over it.
    assert not any(path == root or path.startswith(root + "/") for root in roots for path in read_only)
    assert f"Environment={replay_gc.REPLAY_CACHE_RETENTION_ENV}=" not in unit, "retention stays an operator opt-in"
