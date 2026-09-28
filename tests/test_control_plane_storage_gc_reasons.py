"""The storage GC names why it kept what it kept: typed retention reasons, with their bytes."""

# Covers (for impacted-test selection):
#   src/blueprint_pipeline/control_plane_storage_gc_reasons.py
#   src/blueprint_pipeline/control_plane_storage_references.py

from __future__ import annotations

import errno
import functools
import json
import os
import stat
from pathlib import Path
from types import SimpleNamespace

import pytest

from blueprint_pipeline import completed_replay_cache_retention as retention
from blueprint_pipeline import control_plane_storage_gc_reasons as reasons
from blueprint_pipeline import task_evaluation_configured_scene_object_store as store
from blueprint_pipeline.control_plane_storage_gc import RUN_ACK, run_storage_gc
from blueprint_pipeline.control_plane_storage_pins import live_pinned_paths, release_storage_pin, write_storage_pin
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from tests.test_completed_replay_cache_retention import _fails_instead_of_blocking, _refuse_reading
from tests.test_task_evaluation_configured_scene_object_store import _ContentAddressedClient

NOW = 30_000_000.0
# The real sweep, before the fixture below replaces it with an empty process table.
REAL_PROCESS_REFERENCE = retention.process_reference


@pytest.fixture(autouse=True)
def isolated_disk_ledger(tmp_path, monkeypatch):
    from blueprint_pipeline.control_plane_disk_budget import reserve_control_plane_disk

    monkeypatch.setattr("blueprint_pipeline.control_plane_evidence_offload.reserve_control_plane_disk",
                        functools.partial(reserve_control_plane_disk,
                            disk_usage=lambda _: SimpleNamespace(total=100 * 1024**3, free=80 * 1024**3)))
    monkeypatch.setattr("blueprint_pipeline.control_plane_evidence_offload.DEFAULT_RESERVATION_ROOT",
                        tmp_path / "disk-reservations")
    # No process on this host references anything unless a test says so.
    monkeypatch.setattr(retention, "process_reference", lambda _root, **_kwargs: None)
    monkeypatch.setattr(retention, "process_reference_index", lambda **_kwargs: lambda _root: False)


def _noclass(*_args, **_kwargs) -> None:
    return None


def test_lane_summary_is_typed_private_and_never_a_byte_forecast():
    private = "PRIVATE_MARKER"
    report = {"status": "applied", "observed_at_epoch": NOW, "opt_in": {"lane_scratch": True},
        "lane_scratch": {"status": "report_only", "complete": True, "enabled_requested": True,
            "apply_supported": False, "execution_authorized": False, "mutations": 0,
            "registered_count": 2, "unregistered_count": 1, "observed_registered_count": 2,
            "observed_unregistered_count": 1, "logical_bytes": 10, "allocated_bytes": 512,
            "candidate_bytes": 999, "removed_bytes": 0, "rows": [{"owner": private, "root": "/" + private}],
            "retained_by_reason": {"live_lease": {"count": 2, "bytes": 999}}, "error": private}}
    summary = reasons.build_storage_gc_summary(report)
    phase = summary["phases"]["lane_scratch"]
    assert phase["mode"] == "report_only" and phase["complete"] is True
    assert phase["apply_supported"] is phase["execution_authorized"] is False
    assert phase["candidate_bytes"] is None and phase["removed_or_offloaded_bytes"] == 0
    assert phase["registered_count"] == 2 and phase["logical_bytes"] == 10
    assert phase["retained_by_reason"] == {"live_lease": {"count": 2, "bytes": None}}
    assert summary["top_retained"] == summary["top_retained_reasons"] == []
    assert private not in json.dumps(summary) and summary["opt_in"]["lane_scratch"] is True


@pytest.mark.parametrize("value", [True, -1, "1", 1.5, None])
def test_lane_summary_rejects_noninteger_negative_and_bool_footprints(value):
    raw = {"status": "report_only", "complete": "yes", "mutations": value, "logical_bytes": value,
           "allocated_bytes": value, "registered_count": value, "observed_registered_count": value,
           "candidate_bytes": value, "removed_bytes": value}
    phase = reasons.build_storage_gc_summary({"lane_scratch": raw})["phases"]["lane_scratch"]
    assert phase["complete"] is None
    for key in ("mutations", "logical_bytes", "allocated_bytes", "registered_count", "observed_registered_count"):
        assert phase[key] is None
    assert phase["candidate_bytes"] is phase["removed_or_offloaded_bytes"] is None


def _cold_run(evidence: Path, name: str, *, size: int = 4096) -> Path:
    run = evidence / name
    run.mkdir(parents=True)
    (run / "launch_receipt.json").write_text("{}", encoding="utf-8")
    (run / "frames.bin").write_bytes(b"f" * size)
    old = NOW - 30 * 86400
    for path in (run / "launch_receipt.json", run / "frames.bin", run):
        os.utime(path, (old, old))
    return run


def _settlement(root: Path, run_name: str, reopened: str) -> Path:
    record = root / "scene-a" / "cancelled-unstarted-controls" / "controls-1.json"
    record.parent.mkdir(parents=True, exist_ok=True)
    record.write_text(json.dumps({"launch_receipt": {
        "path": f"/var/lib/blueprint/launch-runs/{run_name}/{reopened}"}}), encoding="utf-8")
    return record


def test_evidence_protection_reason_keeps_the_existing_order(tmp_path, monkeypatch) -> None:
    """Each check still runs in the order the closure had; the first that holds names the reason."""

    run = _cold_run(tmp_path / "launch-runs", "run-1")
    settlement = tmp_path / "scene-intents"
    _settlement(settlement, "run-1", "launch_profile.json")
    pins = tmp_path / "pins"
    write_storage_pin(pins_root=pins, kind="activation", owner_id="act-1", paths=[run], now=lambda: NOW)
    queue = tmp_path / "queue"
    (queue / "pending").mkdir(parents=True)
    (queue / "pending" / "row.json").write_text(json.dumps({"run": "run-1"}), encoding="utf-8")
    process = {"answer": retention.PROCESS_REFERENCED}
    monkeypatch.setattr(retention, "process_reference", lambda _root, **_kwargs: process["answer"])

    def reason(settlement_roots=(settlement,)):
        return reasons.evidence_protection_reason(
            run, settlement_roots=settlement_roots, pins_root=pins, queue_roots=[queue], now=lambda: NOW)

    assert reason(settlement_roots=(tmp_path / "missing-scene-intents",)) == "protected_unreadable_settlement"
    assert reason() == "protected_process"
    process["answer"] = retention.PROCESS_INVENTORY_UNREADABLE
    assert reason() == "protected_process_inventory_unreadable"
    process["answer"] = None
    assert reason() == "protected_pin"
    release_storage_pin(pins_root=pins, kind="activation", owner_id="act-1", now=lambda: NOW)
    assert reason() == "protected_settlement"
    _settlement(settlement, "run-1", "launch_receipt.json")  # retained by the pointer: not a reopen
    assert reason() == "protected_queue"
    (queue / "pending" / "row.json").unlink()
    assert reason() is None
    assert reasons.EVIDENCE_PROTECTION_REASONS == (
        "protected_unreadable_settlement", "protected_process", "protected_process_inventory_unreadable",
        "protected_pin", "protected_settlement", "protected_queue")


def test_unreadable_process_inventory_protects_with_a_reason(tmp_path, monkeypatch) -> None:
    """2026-09-27: six registry runs failed result-artifact offload with a bare PermissionError.

    A /proc entry the GC may not read now protects the run, and the tick says so
    instead of raising.
    """

    evidence = tmp_path / "launch-runs"
    run = _cold_run(evidence, "run-1", size=5000)
    proc = tmp_path / "proc"
    (proc / "4242" / "fd").mkdir(parents=True)
    (proc / "4242" / "cmdline").write_bytes(b"python")
    (proc / "4242" / "environ").write_bytes(b"")
    _refuse_reading(monkeypatch, proc / "4242" / "environ", PermissionError(errno.EACCES, "Permission denied"))
    # The tick sweeps the fake process table instead of this host's.
    monkeypatch.setattr(retention, "process_reference", lambda root, *, process_root=None, ignored_process_ids=():
                        REAL_PROCESS_REFERENCE(root, process_root=proc, ignored_process_ids=ignored_process_ids))
    queue = tmp_path / "queue"
    (queue / "pending").mkdir(parents=True)
    common = dict(content_store_roots=[], derived_roots=[], queue_roots=[queue], pins_root=tmp_path / "pins",
                  evidence_roots=[evidence], now=lambda: NOW, classifier=_noclass)

    assert retention.active_reference(run, process_root=proc) is True
    assert reasons.evidence_protection_reason(
        run, settlement_roots=(), pins_root=tmp_path / "pins", queue_roots=[queue], now=lambda: NOW,
    ) == "protected_process_inventory_unreadable"
    planned = run_storage_gc(**common)
    assert planned["evidence_offload"]["retained_by_reason"] == {
        "protected_process_inventory_unreadable": {"count": 1, "bytes": 5002}}
    applied = run_storage_gc(**common, apply=True, ack=RUN_ACK, offload_enabled=True, publisher=functools.partial(
        store.publish_configured_scene_artifact, client=_ContentAddressedClient(), bucket="blueprint-production-inputs"))
    assert "phase_errors" not in applied
    assert applied["evidence_offload"]["offloaded_count"] == 0
    assert applied["evidence_offload"]["retained_by_reason"] == planned["evidence_offload"]["retained_by_reason"]
    assert run.is_dir() and (run / "frames.bin").stat().st_size == 5000


def test_protected_pin_evidence_names_pin_kinds(tmp_path) -> None:
    """2026-09-27: evidence offload kept 17 runs (7.6 GB) as protected_pin and could not say
    which pins held them. Each is now counted under the kinds of the pins that hold it, with
    how many distinct pins hold that kind's runs, so the runs can be checked against the pin
    proofs. No owner or path reaches the reasons."""

    evidence = tmp_path / "launch-runs"
    own = _cold_run(evidence, "run-secret-own", size=1000)
    two = _cold_run(evidence, "run-secret-two", size=1100)
    three = _cold_run(evidence, "run-secret-three", size=1200)
    shared = _cold_run(evidence, "run-secret-shared", size=2000)
    _cold_run(evidence, "run-secret-free", size=3000)
    pins = tmp_path / "pins"
    # Young pins: this tick's pin pass keeps them all, whatever their proofs.
    write_storage_pin(pins_root=pins, kind="activation", owner_id="run-secret-own", paths=[own], now=lambda: NOW)
    # One pin holding two runs is one owner, not two.
    write_storage_pin(pins_root=pins, kind="activation", owner_id="act-secret-two", paths=[two, three],
                      now=lambda: NOW)
    write_storage_pin(pins_root=pins, kind="activation", owner_id="act-secret-a", paths=[shared / "inputs"],
                      now=lambda: NOW)
    write_storage_pin(pins_root=pins, kind="preparation", owner_id="prep-secret-b", paths=[shared], now=lambda: NOW)

    reason = reasons.evidence_protection_reason(
        shared, settlement_roots=(), pins_root=pins, queue_roots=[], now=lambda: NOW)
    report = run_storage_gc(content_store_roots=[], derived_roots=[], queue_roots=[], pins_root=pins,
                            evidence_roots=[evidence], now=lambda: NOW, classifier=_noclass)

    assert reason == reasons.PROTECTED_PIN and type(reason) is str
    assert reasons.pin_protection(shared, pins_root=pins, now=lambda: NOW) == (
        "activation+preparation", {("activation", "act-secret-a"), ("preparation", "prep-secret-b")})
    size = {run.name: sum(path.stat().st_size for path in run.rglob("*") if path.is_file())
            for run in (own, two, three, shared)}
    by_kind = {"activation": {"count": 3, "bytes": size[own.name] + size[two.name] + size[three.name],
                              "owner_count": 2},
               "activation+preparation": {"count": 1, "bytes": size[shared.name], "owner_count": 2}}
    manifest = report["evidence_offload"]
    assert manifest["retained_by_reason"] == {"protected_pin": {
        "count": 4, "bytes": sum(size.values()), "by_kind": by_kind}}
    assert [row["name"] for row in manifest["candidates"]] == ["run-secret-free"]
    summary = reasons.build_storage_gc_summary(report)
    assert summary["phases"]["evidence_offload"]["retained_by_reason"]["protected_pin"]["by_kind"] == by_kind
    assert "secret" not in json.dumps(manifest["retained_by_reason"]) + json.dumps(summary)
    # A reason given no detail still counts as it always did.
    plain: dict = {}
    reasons.count_retained(plain, "protected_pin", 5)
    assert plain == {"protected_pin": {"count": 1, "bytes": 5}}


def test_pin_kinds_name_exactly_the_live_pinned_paths(tmp_path) -> None:
    """The kind map only labels what ``live_pinned_paths`` pins; it never decides."""

    pins = tmp_path / "pins"
    first, shared, expired = (tmp_path / name for name in ("first", "shared", "expired"))
    write_storage_pin(pins_root=pins, kind="activation", owner_id="act", paths=[first, shared], now=lambda: NOW)
    write_storage_pin(pins_root=pins, kind="preparation", owner_id="prep", paths=[shared], now=lambda: NOW)
    write_storage_pin(pins_root=pins, kind="compilation", owner_id="comp", paths=[expired], now=lambda: NOW,
                      ttl_seconds=1)

    kinds = reasons.live_pin_kinds(pins, now=lambda: NOW + 10)

    assert kinds == {str(first): "activation", str(shared): "activation+preparation"}
    assert set(kinds) == live_pinned_paths(pins, now=lambda: NOW + 10)


def _strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)


def test_storage_gc_writes_a_door_readable_summary(tmp_path, monkeypatch) -> None:
    """2026-09-27: the door can read storage-gc/latest.json, but that report is large and
    scatters its reasons across phases. summary.json beside it says, per phase, what was
    planned, reclaimed and kept and why, is published exactly as latest.json is (0644 in
    a 0755 directory), and carries no run names, host paths or secrets."""

    import stat
    import sys
    import time

    from blueprint_pipeline import control_plane_storage_gc as gc_module

    now = time.time()
    evidence = tmp_path / "launch-runs"
    queued = _cold_run(evidence, "run-secret-name-queued", size=7000)
    hot = _cold_run(evidence, "run-secret-name-hot", size=5000)
    for path in (*hot.iterdir(), hot):
        os.utime(path, (now - 3600, now - 3600))
    registry_run = _cold_run(evidence, "run-secret-name-registry", size=3000)
    (registry_run / "artifacts" / "result_delivery").mkdir(parents=True)
    (registry_run / "artifacts" / "result_delivery" / "artifact_registry.json").write_text("{}", encoding="utf-8")
    _cold_run(evidence, "run-secret-name-cold", size=1000)
    derived = tmp_path / "prepared-references"
    for name, size in (("prep-secret-idle", 100), ("prep-secret-pinned", 300)):
        (derived / name).mkdir(parents=True)
        (derived / name / "blob.bin").write_bytes(b"b" * size)
        for path in (derived / name / "blob.bin", derived / name):
            os.utime(path, (NOW, NOW))
    pins = tmp_path / "pins"
    write_storage_pin(pins_root=pins, kind="activation", owner_id="act-1", paths=[derived / "prep-secret-pinned"])
    queue = tmp_path / "queue"
    (queue / "pending").mkdir(parents=True)
    (queue / "pending" / "row.json").write_text(json.dumps({"run": queued.name}), encoding="utf-8")
    absent_scratch = tmp_path / "engineering-scratch"
    environment = {
        gc_module.CONTENT_STORE_ROOTS_ENV: "", gc_module.PLAN_ONLY_DERIVED_ROOTS_ENV: "",
        gc_module.SETTLEMENT_ROOTS_ENV: "", gc_module.WORKSPACE_BUNDLE_ROOTS_ENV: "",
        gc_module.SCENE_WORKSPACE_ROOTS_ENV: "", "BLUEPRINT_CONTROL_PLANE_GC_REPLAY_PARENT_ROOTS": "",
        gc_module.EVIDENCE_OFFLOAD_ENV: "", gc_module.SCENE_WORKSPACE_RETIREMENT_ENV: "",
        "BLUEPRINT_CONTROL_PLANE_GC_REPLAY_CACHE_RETENTION": "", gc_module.EVIDENCE_ABANDONED_AFTER_ENV: "",
        gc_module.DERIVED_ROOTS_ENV: str(derived), gc_module.QUEUE_ROOTS_ENV: str(queue),
        gc_module.EVIDENCE_ROOTS_ENV: str(evidence), gc_module.SCRATCH_ROOTS_ENV: str(absent_scratch),
        gc_module.EVIDENCE_HOT_WINDOW_ENV: "172800", gc_module.DERIVED_MINIMUM_AGE_ENV: "3600",
    }
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(gc_module, "require_storage_class", _noclass)
    report_dir = tmp_path / "storage-gc"
    report_dir.mkdir(mode=0o700)  # as an older tick left it

    assert gc_module.main(["run", "--apply", "--ack", RUN_ACK, "--pins-root", str(pins),
                           "--report-out", str(report_dir / "latest.json")]) == 0

    latest, published = report_dir / "latest.json", report_dir / "summary.json"
    assert stat.S_IMODE(report_dir.stat().st_mode) == 0o755
    assert stat.S_IMODE(latest.stat().st_mode) == 0o644
    assert stat.S_IMODE(published.stat().st_mode) == 0o644
    report = json.loads(latest.read_text(encoding="utf-8"))
    summary = json.loads(published.read_text(encoding="utf-8"))
    assert published.stat().st_size < reasons.MAX_SUMMARY_BYTES
    assert summary["schema_version"] == "control_plane_storage_gc_summary.v1"
    assert (summary["status"], summary["observed_at_epoch"]) == ("applied", report["observed_at_epoch"])
    assert summary["source_report_digest"] == report["report_digest"]
    assert summary["opt_in"] == {
        "evidence_offload": False, "scene_workspace_retirement": False, "replay_cache_retention": False,
        "extended_pin_proofs": False, "lane_scratch": False, "result_residue_offload": False}
    # Both the report-only lane scratch phase and the residue offload phase reach the summary.
    assert summary["phases"]["lane_scratch"]["mode"] == "report_only"
    assert summary["phases"]["result_residue_offload"]["enabled"] is False
    assert (summary["phase_errors"], summary["skipped_roots"]) == ([], [str(absent_scratch)])
    sizes = {run.name: sum(p.stat().st_size for p in run.rglob("*") if p.is_file())
             for run in (queued, hot, registry_run)}
    walk_seconds = {phase: summary["phases"][phase].pop("walk_seconds")
                    for phase in ("derived_directories", "evidence_offload")}
    assert all(isinstance(seconds, float) and seconds >= 0 for seconds in walk_seconds.values())
    assert summary["phases"]["derived_directories"] == {
        "status": "applied", "candidate_bytes": 100, "removed_or_offloaded_bytes": 100,
        "retained_by_reason": {"pinned": {"count": 1, "bytes": 300, "by_kind": {
            "activation": {"count": 1, "bytes": 300}}}},
        "walked_file_count": 2,
    }
    assert summary["phases"]["evidence_offload"] == {
        "status": "dry_run", "candidate_bytes": 1002, "removed_or_offloaded_bytes": None,
        "retained_by_reason": {
            "protected_queue": {"count": 1, "bytes": sizes[queued.name]},
            "hot": {"count": 1, "bytes": sizes[hot.name]},
            "result_registry": {"count": 1, "bytes": sizes[registry_run.name]},
        },
        # receipt and frames in each run, and the registry in the registry run
        "walked_file_count": 9,
    }
    assert summary["phases"]["result_artifact_offload"] == {
        "run_count": 1, "candidate_bytes": None, "removed_or_offloaded_bytes": 0,
        "retained_by_reason": {"offload_failed:registry": {"count": 1, "bytes": None}},
        "failures": [{"scope": "run", "stage": "registry", "error_type": "TaskEvaluationResultDeliveryError",
                      "errno": None, "count": 1}],
    }
    assert summary["top_retained"] == [
        {"phase": "evidence_offload", "reason": "protected_queue", "count": 1, "bytes": sizes[queued.name]},
        {"phase": "evidence_offload", "reason": "hot", "count": 1, "bytes": sizes[hot.name]},
        {"phase": "evidence_offload", "reason": "result_registry", "count": 1, "bytes": sizes[registry_run.name]},
        {"phase": "derived_directories", "reason": "pinned", "count": 1, "bytes": 300},
    ]
    # Nothing but the configured roots names a host path, and no run or directory is named.
    text = published.read_text(encoding="utf-8")
    assert "secret-name" not in text and "prep-secret" not in text
    assert [value for value in _strings(summary) if "/" in value] == [str(absent_scratch)]
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "deploy" / "operator-door"))
    from operator_door.config import DoorConfig
    from operator_door.fsview import FileView
    from operator_door.secrets_guard import scan_bytes

    assert scan_bytes(published.read_bytes()) is None
    contents, _ = FileView(DoorConfig(read_roots=(str(tmp_path),), hidden_paths=())).read_range(str(published))
    assert json.loads(contents) == json.loads(published.read_text(encoding="utf-8"))
    assert "operator_door" in sys.modules


def test_the_summary_counts_terminal_pin_candidates_releases_and_the_opt_in(tmp_path) -> None:
    """The pin phase's summary named nothing but its status, with null retention. It now
    counts the candidates and the pins released, says whether the extended proofs may
    release, and counts why the rest were kept, naming no owner and no path."""

    pins = tmp_path / "pins"
    for owner, created in (("prep-secret-stale", NOW - 9 * 86400), ("prep-secret-young", NOW - 3600)):
        write_storage_pin(pins_root=pins, kind="preparation", owner_id=owner, now=lambda created=created: created,
                          paths=[f"/var/lib/blueprint/task-evaluation-inputs/prepared-references/{owner}"])
    # The stale preparation's sealed envelope binds it to a release other than the running one.
    preparations = tmp_path / "task-evaluation-launch-preparations"
    envelope = {"schema_version": "task_evaluation_launch_preparation_envelope.v1", "request_digest": "sha256:" + "1" * 64,
                "request": {"preparation_id": "prep-secret-stale", "expected_production_commit": "a" * 40},
                "envelope_digest": ""}
    envelope["envelope_digest"] = canonical_digest(envelope, digest_field="envelope_digest")
    (preparations / "materialized").mkdir(parents=True)
    (preparations / "materialized" / f"prep-secret-stale-{'1' * 64}.json").write_text(json.dumps(envelope), encoding="utf-8")
    common = dict(content_store_roots=[], derived_roots=[], queue_roots=[preparations], pins_root=pins, now=lambda: NOW,
                  classifier=_noclass, apply=True, ack=RUN_ACK, running_commit="b" * 40)

    listed = reasons.build_storage_gc_summary(run_storage_gc(**common))
    applied = reasons.build_storage_gc_summary(run_storage_gc(**common, extended_pin_proofs_enabled=True))

    phase = {"status": "applied", "candidate_bytes": None, "removed_or_offloaded_bytes": None,
             "retained_by_reason": {"pin_young": {"count": 1, "bytes": None}}, "candidate_count": 1}
    assert listed["phases"]["terminal_cache_pins"] == {**phase, "released_count": 0, "enabled": False}
    assert applied["phases"]["terminal_cache_pins"] == {**phase, "released_count": 1, "enabled": True}
    assert (listed["opt_in"]["extended_pin_proofs"], applied["opt_in"]["extended_pin_proofs"]) == (False, True)
    assert "secret" not in json.dumps(listed) + json.dumps(applied)
    # A report from before the counts does not claim there were none.
    older = reasons.build_storage_gc_summary({"status": "applied", "terminal_cache_pins": {
        "status": "applied", "candidates": [], "released": [], "kept": []}})["phases"]["terminal_cache_pins"]
    assert (older["candidate_count"], older["released_count"], older["enabled"]) == (None, None, None)


def test_the_summary_stays_small_and_names_only_typed_reasons() -> None:
    """However many reasons a phase reports, the summary keeps the largest, bounded, and a
    reason that is not a typed string (a path, say) is never copied into it."""

    by_reason = {f"reason_{index:04d}": {"count": 1, "bytes": index} for index in range(2000)}
    by_reason["/var/lib/blueprint/leaked/path"] = {"count": 1, "bytes": 10**9}
    failures = [{"status": "retained", "run_directory": f"run-{index}", "reason": "OSError",
                 "error_type": f"Error{index}", "errno": index, "stage": "evict"} for index in range(500)]
    report = {
        "schema_version": "control_plane_storage_gc_run.v1", "status": "dry_run", "observed_at_epoch": NOW,
        "skipped_roots": [], "report_digest": "sha256:" + "0" * 64,
        "derived_directories": {"status": "dry_run", "candidate_bytes": 0, "retained_by_reason": by_reason},
        "evidence_offload": {"status": "error", "error": "PermissionError"},
        "phase_errors": ["evidence_offload"],
        "result_artifact_offload": failures,
    }

    summary = reasons.build_storage_gc_summary(report)

    assert len(json.dumps(summary, indent=2, sort_keys=True).encode()) < reasons.MAX_SUMMARY_BYTES
    derived = summary["phases"]["derived_directories"]["retained_by_reason"]
    assert len(derived) == 50 and "unrecognized_reason" in derived and "/var" not in json.dumps(summary)
    assert summary["top_retained"][0] == {
        "phase": "derived_directories", "reason": "unrecognized_reason", "count": 1, "bytes": 10**9}
    assert len(summary["top_retained"]) == 10
    assert summary["phases"]["evidence_offload"]["error_type"] == "PermissionError"
    assert summary["phase_errors"] == ["evidence_offload"]
    assert len(summary["phases"]["result_artifact_offload"]["failures"]) == 20
    assert "run-1" not in json.dumps(summary)
    # A report written before the tick recorded its opt-ins does not claim they were off.
    assert summary["opt_in"] == {
        "evidence_offload": None, "scene_workspace_retirement": None, "replay_cache_retention": None,
        "extended_pin_proofs": None, "lane_scratch": None, "result_residue_offload": None}


def test_summary_ranks_global_reason_totals_before_phase_and_top_ten_caps() -> None:
    one_byte_reasons = {
        f"reason_{index:02d}": {"count": 1, "bytes": 1} for index in range(60)
    }
    one_byte_reasons["zz_shared"] = {"count": 1, "bytes": 1}
    report = {
        "status": "applied", "observed_at_epoch": NOW,
        "derived_directories": {"status": "applied", "retained_by_reason": one_byte_reasons},
        "content_store": {"status": "applied", "retained_by_reason": {
            "zz_shared": {"count": 1, "bytes": 1},
        }},
    }

    summary = reasons.build_storage_gc_summary(report)

    assert "zz_shared" not in summary["phases"]["derived_directories"]["retained_by_reason"]
    assert summary["top_retained_reasons"] == [
        {"reason": "zz_shared", "bytes": 2},
        {"reason": "reason_00", "bytes": 1},
        {"reason": "reason_01", "bytes": 1},
    ]


def _quiet_tick(monkeypatch) -> None:
    """Every configured root empty: a tick that only reconciles an empty pin ledger."""

    from blueprint_pipeline import control_plane_storage_gc as gc_module

    for name in (gc_module.CONTENT_STORE_ROOTS_ENV, gc_module.DERIVED_ROOTS_ENV, gc_module.PLAN_ONLY_DERIVED_ROOTS_ENV,
                 gc_module.QUEUE_ROOTS_ENV, gc_module.EVIDENCE_ROOTS_ENV, gc_module.SETTLEMENT_ROOTS_ENV,
                 gc_module.SCRATCH_ROOTS_ENV, gc_module.WORKSPACE_BUNDLE_ROOTS_ENV, gc_module.SCENE_WORKSPACE_ROOTS_ENV,
                 "BLUEPRINT_CONTROL_PLANE_GC_REPLAY_PARENT_ROOTS", gc_module.EVIDENCE_OFFLOAD_ENV,
                 gc_module.SCENE_WORKSPACE_RETIREMENT_ENV, "BLUEPRINT_CONTROL_PLANE_GC_REPLAY_CACHE_RETENTION"):
        monkeypatch.setenv(name, "")
    monkeypatch.setattr(gc_module, "require_storage_class", _noclass)


@pytest.mark.parametrize("failure", ["build", "write"])
def test_a_failed_summary_withdraws_the_previous_ticks(tmp_path, monkeypatch, capsys, failure) -> None:
    """A summary that cannot be built or written must not leave the previous tick's
    summary.json beside the new latest.json: the door reads the summary first, and a
    plausible stale one is fabricated state. ENOSPC between the two writes is the likely
    trigger, under the very disk pressure this GC exists for. latest.json stays, and the
    unit fails so the failure is seen."""

    from blueprint_pipeline import control_plane_storage_gc as gc_module

    _quiet_tick(monkeypatch)
    report_dir = tmp_path / "storage-gc"
    report_dir.mkdir()
    (report_dir / "summary.json").write_text(json.dumps(
        {"schema_version": reasons.SUMMARY_SCHEMA_VERSION, "status": "applied", "top_retained": []}), encoding="utf-8")
    if failure == "build":
        def broken(_report):
            raise RuntimeError("summary_projection_bug")

        monkeypatch.setattr(gc_module, "build_storage_gc_summary", broken)
    else:
        real_dump = json.dump

        def full_disk(value, stream, **kwargs):
            if isinstance(value, dict) and value.get("schema_version") == reasons.SUMMARY_SCHEMA_VERSION:
                raise OSError(errno.ENOSPC, "No space left on device")
            return real_dump(value, stream, **kwargs)

        monkeypatch.setattr(json, "dump", full_disk)

    code = gc_module.main(["run", "--pins-root", str(tmp_path / "pins"),
                           "--report-out", str(report_dir / "latest.json")])

    assert code == 1
    printed = json.loads(capsys.readouterr().out)
    latest = json.loads((report_dir / "latest.json").read_text(encoding="utf-8"))
    assert latest["schema_version"] == "control_plane_storage_gc_run.v1"
    assert latest["report_digest"] == printed["report_digest"]
    assert sorted(path.name for path in report_dir.iterdir()) == ["latest.json"]


def test_a_report_named_summary_json_is_not_overwritten_by_its_summary(tmp_path, monkeypatch, capsys) -> None:
    from blueprint_pipeline import control_plane_storage_gc as gc_module

    _quiet_tick(monkeypatch)
    report = tmp_path / "storage-gc" / "summary.json"

    assert gc_module.main(["run", "--pins-root", str(tmp_path / "pins"), "--report-out", str(report)]) == 0

    written = json.loads(report.read_text(encoding="utf-8"))
    assert written["schema_version"] == "control_plane_storage_gc_run.v1"
    assert written["report_digest"] == json.loads(capsys.readouterr().out)["report_digest"]
    assert sorted(path.name for path in report.parent.iterdir()) == ["summary.json"]


def test_the_summary_tells_kept_nothing_from_not_reported() -> None:
    """``{}`` means a phase kept nothing; ``None`` means it does not say what it kept. Applied
    content-store, stranded-row, scratch and bundle receipts, a replay cache pass and the
    terminal pin pass carry no retained counts, and the summary must not claim they kept
    nothing. An artifact another pass already evicted is gone, not kept."""

    report = {
        "status": "applied", "observed_at_epoch": NOW, "report_digest": "sha256:" + "1" * 64, "skipped_roots": [],
        "stranded_queue_rows": {"status": "applied", "stranded_bytes": 0, "stranded": [], "skipped": []},
        "terminal_cache_pins": {"status": "applied", "candidates": [], "released": [], "kept": []},
        "derived_directories": {"status": "applied", "candidate_bytes": 0, "removed_bytes": 0,
                                "retained_by_reason": {}},
        "planned_derived_directories": {"status": "dry_run", "candidate_bytes": 0,
                                        "retained_counts": {"pinned": 0, "queue_referenced": 0, "young": 0}},
        "content_store": {"status": "applied", "removed_bytes": 0, "removed": [], "skipped": []},
        # A receipt whose manifest predates the reasons.
        "evidence_offload": {"status": "applied", "offloaded_bytes": 0, "retained_by_reason": None},
        "scratch_directories": {"status": "applied", "removed_bytes": 0, "removed": [], "skipped": []},
        "workspace_bundles": {"status": "applied", "removed_bytes": 0, "removed": [], "skipped": []},
        "replay_caches": {"status": "applied", "candidate_bytes": 0, "removed_bytes": 0, "kept": []},
        "result_artifact_offload": [{"status": "applied", "candidate_bytes": 90_000, "offloaded_bytes": 0,
                                     "skipped": [{"relative_path": "evidence/a.mp4", "reason": "already_evicted"}]}],
    }

    phases = reasons.build_storage_gc_summary(report)["phases"]

    assert {name for name, phase in phases.items() if phase["retained_by_reason"] is None} == {
        "stranded_queue_rows", "terminal_cache_pins", "content_store", "evidence_offload",
        "scratch_directories", "workspace_bundles", "replay_caches"}
    assert phases["derived_directories"]["retained_by_reason"] == {}
    # Counts of zero kept nothing.
    assert phases["planned_derived_directories"]["retained_by_reason"] == {}
    assert phases["result_artifact_offload"]["retained_by_reason"] == {}
    counted = reasons.build_storage_gc_summary({**report, "content_store": {
        "status": "dry_run", "candidate_bytes": 0, "retained_counts": {"linked": 3, "young": 0}}})
    assert counted["phases"]["content_store"]["retained_by_reason"] == {"linked": {"count": 3, "bytes": None}}


def test_the_reasons_read_references_without_reaching_into_the_gc() -> None:
    """The GC imports the reasons module, so the reasons must not import the GC back. Both
    read queue and settlement references from one leaf module, and the GC keeps its old
    names for every existing caller."""

    import ast
    import inspect

    from blueprint_pipeline import control_plane_storage_gc as gc_module
    from blueprint_pipeline import control_plane_storage_references as references

    def imported(module) -> set[str]:
        return {node.module for node in ast.walk(ast.parse(inspect.getsource(module)))
                if isinstance(node, ast.ImportFrom) and node.module}

    assert "control_plane_storage_gc" not in imported(reasons)
    assert not imported(references) & {
        "control_plane_storage_gc", "control_plane_storage_gc_reasons", "control_plane_evidence_offload"}
    assert gc_module._queue_reference_text is references.queue_reference_text
    assert gc_module._settlement_reference_text is references.settlement_reference_text
    assert gc_module.settlement_reopens_beyond_retained_receipts is references.settlement_reopens_beyond_retained_receipts
    assert (gc_module._MAX_QUEUE_MESSAGE_BYTES, gc_module.QUEUE_STATES, gc_module.SETTLEMENT_RECORD_GLOBS) == (
        references.MAX_QUEUE_MESSAGE_BYTES, references.QUEUE_STATES, references.SETTLEMENT_RECORD_GLOBS)


@pytest.mark.parametrize("swap,code", [
    # Read once more after the swap, the FIFO is seen for what it is; a file swapped again is not read.
    ("fifo", "queue_row_not_regular"), ("link", "queue_row_linked"), ("file", "queue_row_changed")])
def test_a_queue_row_swapped_after_its_lstat_fails_the_strict_read(tmp_path, monkeypatch, swap, code) -> None:
    """The strict reader opens each row without following a link or waiting for a writer, and what
    it opened must be the regular file the row's lstat saw. A row swapped for a FIFO, a link or
    another file in between (and again when it is read once more) refuses the read, instead of
    hanging the tick or reading a file whose kind and size were never checked."""

    from blueprint_pipeline import control_plane_storage_references as references

    queue = tmp_path / "queue"
    (queue / "pending").mkdir(parents=True)
    row = queue / "pending" / "row.json"
    row.write_text('{"name": "named"}', encoding="utf-8")
    (tmp_path / "elsewhere.json").write_text('{"name": "elsewhere"}', encoding="utf-8")
    held = iter(range(8))
    real_lstat = Path.lstat

    def lstat_then_swap(self):
        observed = real_lstat(self)
        if self == row and stat.S_ISREG(observed.st_mode):
            # Every swapped-out inode stays allocated, so no new file can reuse its number (Linux
            # reuses a freed inode number at once), on the first read and on the read once more.
            os.link(row, tmp_path / f"held-{next(held)}.json")
            row.unlink()
            if swap == "fifo":
                os.mkfifo(row)
            elif swap == "link":
                row.symlink_to(tmp_path / "elsewhere.json")
            else:
                row.write_text('{"name": "replaced"}', encoding="utf-8")
        return observed

    monkeypatch.setattr(Path, "lstat", lstat_then_swap)
    with _fails_instead_of_blocking(5), pytest.raises(references.QueueReferenceUnreadable, match=code):
        references.queue_reference_text([queue], strict=True)


def test_a_fifo_queue_row_fails_the_strict_read_as_not_regular(tmp_path) -> None:
    """A row that is not a regular file refuses the strict read under its own code, and the read
    never waits for a FIFO's writer."""

    from blueprint_pipeline import control_plane_storage_references as references

    queue = tmp_path / "queue"
    (queue / "processing").mkdir(parents=True)
    os.mkfifo(queue / "processing" / "row.json")

    with _fails_instead_of_blocking(5), pytest.raises(references.QueueReferenceUnreadable,
                                                      match="queue_row_not_regular"):
        references.queue_reference_text([queue], strict=True)


@pytest.mark.parametrize("when", ["listed", "swapped_after_lstat"])
def test_a_fifo_queue_row_never_blocks_the_original_reader(tmp_path, monkeypatch, when) -> None:
    """Every original caller, the storage GC's protection checks among them, reads its queues
    without ``strict``. That reader opened each row by name, so one FIFO row in a queue directory
    blocked the tick until systemd stopped it. It now reads each row through the strict reader's
    descriptor and skips a FIFO, as it skips every row it cannot read, and reads the others."""

    from blueprint_pipeline import control_plane_storage_references as references

    queue = tmp_path / "queue"
    (queue / "pending").mkdir(parents=True)
    (queue / "processing").mkdir()
    fifo = queue / "pending" / "a-fifo.json"
    (queue / "pending" / "b-row.json").write_text('{"name": "named-b"}', encoding="utf-8")
    (queue / "processing" / "c-row.json").write_text('{"name": "named-c"}', encoding="utf-8")
    if when == "listed":
        os.mkfifo(fifo)
    else:
        fifo.write_text('{"name": "swapped"}', encoding="utf-8")
        real_lstat = Path.lstat

        def lstat_then_swap(self):
            observed = real_lstat(self)
            if self == fifo and stat.S_ISREG(observed.st_mode):
                fifo.unlink()
                os.mkfifo(fifo)
            return observed

        monkeypatch.setattr(Path, "lstat", lstat_then_swap)

    with _fails_instead_of_blocking(5):
        text = references.queue_reference_text([queue])

    assert text == '{"name": "named-b"}\n{"name": "named-c"}'


@pytest.mark.parametrize("replacements,refused", [(1, False), (2, True)])
def test_a_row_claimed_while_it_is_read_is_read_again_once(tmp_path, monkeypatch, replacements, refused) -> None:
    """The dispatcher claims a row by creating an empty placeholder in processing/ and replacing the
    pending row onto it (``task_evaluation_launch_dispatcher``'s ``claim``). A strict read that took
    the placeholder's lstat then opens the row: the name changed once, so the row is read again
    instead of failing the read. A name that changes again on the second read still fails it."""

    from blueprint_pipeline import control_plane_storage_references as references

    queue = tmp_path / "queue"
    for state in ("pending", "processing"):
        (queue / state).mkdir(parents=True)
    row = json.dumps({"activation_id": "claimed-run"})
    (queue / "pending" / "row.json").write_text(row, encoding="utf-8")
    placeholder = queue / "processing" / "row.json"
    os.close(os.open(placeholder, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600))
    real_lstat = Path.lstat
    replaced: list[int] = []

    def lstat_then_claim(self):
        observed = real_lstat(self)
        if self == placeholder and len(replaced) < replacements:
            replaced.append(1)
            if len(replaced) == 1:
                os.replace(queue / "pending" / "row.json", placeholder)  # the claim lands
            else:
                fresh = queue / "rewritten.json.tmp"
                fresh.write_text(row, encoding="utf-8")
                os.replace(fresh, placeholder)
        return observed

    monkeypatch.setattr(Path, "lstat", lstat_then_claim)
    if refused:
        with pytest.raises(references.QueueReferenceUnreadable, match="queue_row_changed"):
            references.queue_reference_text([queue], strict=True)
    else:
        # Read in pending before the claim, and in processing after it.
        assert references.queue_reference_text([queue], strict=True).count("claimed-run") == 2
    assert len(replaced) == replacements


def test_a_fifo_settlement_record_is_counted_unreadable_without_blocking(tmp_path) -> None:
    """Every protection check reads the settlement records, and read them by name, so one FIFO
    record blocked the tick. Each record is now read through the queue reader's descriptor: a FIFO
    is refused without blocking and counted as unreadable, which already protects every run, and
    the other records are still read."""

    from blueprint_pipeline import control_plane_storage_references as references

    root = tmp_path / "settlements"
    (root / "intent-1" / "attempts").mkdir(parents=True)
    (root / "intent-1" / "preparations").mkdir()
    os.mkfifo(root / "intent-1" / "attempts" / "a-fifo.json")
    (root / "intent-1" / "attempts" / "b.json").write_text('{"run": "named-b"}', encoding="utf-8")
    (root / "intent-1" / "preparations" / "c.json").write_text('{"run": "named-c"}', encoding="utf-8")

    with _fails_instead_of_blocking(5):
        text, unreadable = references.settlement_reference_text([root])

    assert (text, unreadable) == ('{"run": "named-b"}\n{"run": "named-c"}', 1)


def test_the_summary_copies_only_the_offloads_own_stages() -> None:
    """A failure's stage reaches the summary only if it is one of the result-artifact
    offload's stages; any other string, however typed it looks, is unrecognized."""

    from blueprint_pipeline.task_evaluation_result_artifact_store import OFFLOAD_STAGES

    rows = [
        {"status": "retained", "run_directory": "run-a", "reason": "PermissionError",
         "error_type": "PermissionError", "errno": 1, "stage": "evict"},
        {"status": "retained", "run_directory": "run-b", "reason": "OSError",
         "error_type": "OSError", "errno": 5, "stage": "teleport"},
        {"status": "retained", "run_directory": "run-c", "reason": "TaskEvaluationResultDeliveryError",
         "error_type": "TaskEvaluationResultDeliveryError", "errno": None, "stage": "plan"},
        {"status": "applied", "candidate_bytes": 10, "offloaded_bytes": 0, "skipped": [
            {"relative_path": "evidence/a.mp4", "reason": "OSError", "error_type": "OSError",
             "errno": 28, "stage": "publish"}]},
    ]

    phase = reasons.build_storage_gc_summary(
        {"status": "applied", "result_artifact_offload": rows})["phases"]["result_artifact_offload"]

    assert OFFLOAD_STAGES == ("registry", "plan", "protection", "publish", "evict")
    assert phase["retained_by_reason"] == {
        "offload_failed:evict": {"count": 1, "bytes": None},
        "offload_failed:unrecognized_stage": {"count": 1, "bytes": None},
        "offload_failed:plan": {"count": 1, "bytes": None},
        "artifact_offload_failed:publish": {"count": 1, "bytes": None},
    }
    assert sorted((row["scope"], row["stage"]) for row in phase["failures"]) == [
        ("artifact", "publish"), ("run", "evict"), ("run", "plan"), ("run", "unrecognized_stage")]


def test_replay_scan_error_leaves_candidate_bytes_unknown() -> None:
    phase = reasons.build_storage_gc_summary({"status": "applied", "replay_caches": {
        "status": "applied", "candidate_bytes": 0, "removed_bytes": 0,
        "errors": [{"root": "/private/scene", "error": "PermissionError"}],
        "omitted_errors_count": 0,
    }})["phases"]["replay_caches"]

    assert phase["candidate_bytes"] is None
    assert "/private/scene" not in str(phase)
    complete = reasons.build_storage_gc_summary({"status": "applied", "replay_caches": {
        "status": "applied", "candidate_bytes": 0, "removed_bytes": 0,
        "errors": [], "omitted_errors_count": 0,
    }})["phases"]["replay_caches"]
    assert complete["candidate_bytes"] == 0
    dry_run = reasons.build_storage_gc_summary({"status": "dry_run", "replay_caches": {
        "status": "dry_run", "candidate_bytes": 0, "estimated_candidate_bytes": 0,
        "removed_bytes": 0, "errors": [{"error": "PermissionError"}],
    }})["phases"]["replay_caches"]
    assert dry_run["estimated_candidate_bytes"] is None


def test_partial_result_artifact_scan_leaves_candidate_bytes_unknown() -> None:
    rows = [
        {"status": "applied", "candidate_bytes": 0, "offloaded_bytes": 0},
        {"status": "retained", "stage": "registry", "error_type": "OSError"},
    ]
    phase = reasons.build_storage_gc_summary({
        "status": "applied", "result_artifact_offload": rows,
    })["phases"]["result_artifact_offload"]

    assert phase["candidate_bytes"] is None
    complete = reasons.build_storage_gc_summary({"status": "applied", "result_artifact_offload": [
        {"status": "applied", "candidate_bytes": 0, "offloaded_bytes": 0},
    ]})["phases"]["result_artifact_offload"]
    assert complete["candidate_bytes"] == 0
