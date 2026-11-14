"""Stage 8: a website job's scene run stays on the host while the job is open, and a run already moved to cold
storage is one typed hold, never "in progress" for ever.

Built from capture-447ae4f4's own records (``tests/fixtures/capture_447ae4f4_stage8``, read from the host
2026-11-11 with the operator door): its team-scene-packet request, the offload pointer that stood where its
scene run had been (members trimmed to the one retained receipt, then re-sealed), and its last screening plan
and ledger campaign (``screening-f3d50faef0dfe2fba1c40454ad67fc32``: two rows completed, two deferred for the
site's money, trimmed to nine rows and re-sealed). Nothing reaches the host, a provider or the network.
"""

# Covers (for impacted-test selection):
#   src/blueprint_pipeline/open_job_source_runs.py
#   src/blueprint_pipeline/scene_run_cold_storage.py
#   src/blueprint_pipeline/control_plane_storage_gc_reasons.py
#   src/blueprint_pipeline/scene_stage_holds.py
#   src/blueprint_pipeline/rigid_robot_free_probe_bundle.py
#   src/blueprint_pipeline/team_scene_packet_worker.py

from __future__ import annotations

import base64
import copy
import json
import os
import time
from pathlib import Path

import pytest

from blueprint_pipeline import beta_policy_screening_coordinator as coordinator
from blueprint_pipeline import scene_stage_holds as holds
from blueprint_pipeline import completed_replay_cache_retention as retention
from blueprint_pipeline import open_job_source_runs as open_jobs
from blueprint_pipeline import open_job_testing_final as testing
from blueprint_pipeline import scene_run_cold_storage as cold
from blueprint_pipeline.control_plane_storage_gc import run_storage_gc
from blueprint_pipeline.control_plane_storage_gc_reasons import (
    PROTECTED_OPEN_JOB_SOURCE,
    evidence_protection_reason,
)
from blueprint_pipeline.decision_evidence_contracts import canonical_digest

FIXTURES = Path(__file__).parent / "fixtures" / "capture_447ae4f4_stage8"
HOT_WINDOW = 172_800  # the host unit's BLUEPRINT_CONTROL_PLANE_EVIDENCE_HOT_WINDOW_SECONDS


def _fixture(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def _sealed(value: dict, field: str) -> dict:
    value = {**value, field: ""}
    value[field] = canonical_digest(value, digest_field=field)
    return value


@pytest.fixture(autouse=True)
def disk_above_its_floor(monkeypatch):
    """The work disk has room unless a test says otherwise (``_under_floor``)."""
    from types import SimpleNamespace

    monkeypatch.setattr(open_jobs, "measure_disk", lambda _path: SimpleNamespace(total=100 * 1024**3,
                                                                                 free=90 * 1024**3, used=0))


@pytest.fixture(autouse=True)
def no_process_holds_anything(monkeypatch):
    monkeypatch.setattr(retention, "process_reference", lambda _root, **_kwargs: None)
    monkeypatch.setattr(retention, "process_reference_index", lambda **_kwargs: lambda _root: False)


@pytest.fixture
def world(tmp_path):
    """447's scene run, its packet request, its intent and its screening queue, under one temporary host."""

    control = tmp_path / "pipeline-control-plane"
    pointer = _fixture("launch_run.offloaded.v1.json")
    runs = control / "task-evaluation-launch-runs"
    run = runs / pointer["directory"]
    receipt = base64.b64decode(pointer["retained_receipt_bytes"]["launch_receipt.json"])
    (run / "allocator").mkdir(parents=True)
    (run / "launch_receipt.json").write_bytes(receipt)
    (run / "allocator" / "vast_provider_runtime_output.zip").write_bytes(b"z" * 8192)
    now = time.time()
    idle = now - HOT_WINDOW - 3600  # past the two-day hot window, as 447's run was on 2026-11-06
    for path in (*run.rglob("*"), run):
        os.utime(path, (idle, idle))
    request = _fixture("team_scene_packet_request.json")
    intent_root = control / "task-evaluation-scene-intents"
    intent = _sealed({"intent_id": request["intent_id"],
                      "request": {"execution": {"expires_at_epoch": now + 86_400}}}, "intent_digest")
    (intent_root / request["intent_id"]).mkdir(parents=True)
    (intent_root / request["intent_id"] / "intent.json").write_text(json.dumps(intent), encoding="utf-8")
    request = {**request, "source_launch_root": str(run), "intent_digest": intent["intent_digest"]}
    packets = control / "team-scene-packets"
    (packets / "processed").mkdir(parents=True)
    (packets / "processed" / "31e12dca983b8837295ac911c44dee68.json").write_text(json.dumps(request))
    screening = control / "beta-policy-screening"
    plan = _fixture("screening_plan.json")
    (screening / "plans").mkdir(parents=True)
    (screening / "attempts").mkdir()
    (screening / "plans" / f"{plan['campaign_id']}.json").write_text(json.dumps(plan), encoding="utf-8")
    campaign = _fixture("screening_campaign_state.json")

    def ledger(state: dict) -> None:
        value = _sealed({"schema_version": campaign["schema_version"], "campaigns": {plan["campaign_id"]: state},
                         "last_processed_campaign": plan["campaign_id"]}, "ledger_digest")
        (screening / coordinator.LEDGER_FILENAME).write_text(json.dumps(value), encoding="utf-8")

    ledger(campaign["state"])
    return {"run": run, "runs": runs, "pointer": pointer, "request": request, "packets": packets,
            "intent_root": intent_root, "intent": intent, "screening": screening, "campaign": campaign,
            "ledger": ledger, "now": now, "receipt": receipt}


def _reason(world, **overrides):
    roots = {"team_scene_packet_queue_root": world["packets"], "screening_queue_root": world["screening"],
             "scene_intent_root": world["intent_root"], **overrides}
    return evidence_protection_reason(world["run"], settlement_roots=(), pins_root=world["run"].parent / "pins",
                                      queue_roots=[], now=time.time, **roots)


def _gc(world, **roots):
    return run_storage_gc(content_store_roots=[], derived_roots=[], queue_roots=[], pins_root=world["runs"] / "pins",
                          evidence_roots=[world["runs"]], hot_window_seconds=HOT_WINDOW,
                          classifier=lambda *_args, **_kwargs: None, **roots)


# Point 1: a run an open job still needs is not moved off the host.

def test_before_the_fix_the_clean_up_picks_447s_scene_run_for_cold_storage(world):
    report = _gc(world)
    assert [row["name"] for row in report["evidence_offload"]["candidates"]] == [world["run"].name]
    assert _reason(world, team_scene_packet_queue_root=None, scene_intent_root=None) is None


def test_with_the_job_open_the_clean_up_keeps_447s_scene_run_by_name(world):
    report = _gc(world, team_scene_packet_queue_root=world["packets"], screening_queue_root=world["screening"],
                 scene_intent_root=world["intent_root"])
    offload = report["evidence_offload"]
    assert offload["candidates"] == []
    assert list(offload["retained_by_reason"]) == [PROTECTED_OPEN_JOB_SOURCE]
    assert _reason(world) == PROTECTED_OPEN_JOB_SOURCE


def test_447s_testing_is_not_final_because_a_deploy_plans_its_money_deferred_rows_again(world):
    # The real ledger: two rows deferred beta_screening_site_budget_exhausted at $26.63 of $30.
    intent_id = world["request"]["intent_id"]
    assert testing.testing_final(intent_id, screening_queue_root=world["screening"]) is False
    assert testing.publish_testing_record(world["screening"]) == {"status": "written", "intents": 1, "final": 0}
    assert open_jobs.testing_final(intent_id, screening_queue_root=world["screening"]) is False
    assert _reason(world) == PROTECTED_OPEN_JOB_SOURCE


def test_once_testing_is_final_the_run_goes_cold_on_the_two_day_rule(world):
    state = copy.deepcopy(world["campaign"]["state"])
    for row in state["candidates"].values():
        if row["status"] == "deferred":
            row.update(status="unavailable", blockers=["candidate_infeasible_measured_geometry"])
    world["ledger"](state)
    intent_id = world["request"]["intent_id"]
    assert testing.testing_final(intent_id, screening_queue_root=world["screening"]) is True
    # The GC reads only the screening pass's record: until the pass writes one, the run is kept.
    assert _reason(world) == PROTECTED_OPEN_JOB_SOURCE
    testing.publish_testing_record(world["screening"])
    assert open_jobs.testing_final(intent_id, screening_queue_root=world["screening"]) is True
    assert _reason(world) is None


def test_with_the_trial_retired_no_testing_record_is_written_so_the_run_is_kept(world):
    """The early trial's timer is retired (2026-11-14): nothing writes the testing record any more.

    A job with no record is simply not testing-final, so the storage collector keeps its source run
    (planner amendment A3): nothing is deleted sooner, even when every screening row has ended.
    """
    state = copy.deepcopy(world["campaign"]["state"])
    for row in state["candidates"].values():
        if row["status"] == "deferred":
            row.update(status="unavailable", blockers=["candidate_infeasible_measured_geometry"])
    world["ledger"](state)
    intent_id = world["request"]["intent_id"]
    assert not (world["screening"] / open_jobs.TESTING_RECORD_FILENAME).exists()
    assert open_jobs.testing_final(intent_id, screening_queue_root=world["screening"]) is False
    assert _reason(world) == PROTECTED_OPEN_JOB_SOURCE


@pytest.mark.parametrize("fault", ["ledger_changed_since", "record_tampered", "record_unreadable"])
def test_a_testing_record_the_gc_cannot_trust_keeps_the_run(world, fault):
    state = copy.deepcopy(world["campaign"]["state"])
    for row in state["candidates"].values():
        if row["status"] == "deferred":
            row.update(status="unavailable", blockers=["candidate_infeasible_measured_geometry"])
    world["ledger"](state)
    testing.publish_testing_record(world["screening"])
    record = world["screening"] / open_jobs.TESTING_RECORD_FILENAME
    if fault == "ledger_changed_since":
        world["ledger"](copy.deepcopy(world["campaign"]["state"]))  # a later write: the record is for another ledger
    elif fault == "record_tampered":
        value = json.loads(record.read_text())
        record.write_text(json.dumps({**value, "final_by_intent": {"other": True, **value["final_by_intent"]}}))
    else:
        record.write_text("{not json")
    assert open_jobs.testing_final(world["request"]["intent_id"], screening_queue_root=world["screening"]) is False
    assert _reason(world) == PROTECTED_OPEN_JOB_SOURCE


def test_once_the_jobs_window_has_ended_the_run_goes_cold(world):
    intent = _sealed({"intent_id": world["intent"]["intent_id"],
                      "request": {"execution": {"expires_at_epoch": world["now"] - 60}}}, "intent_digest")
    (world["intent_root"] / intent["intent_id"] / "intent.json").write_text(json.dumps(intent))
    request = {**world["request"], "intent_digest": intent["intent_digest"]}
    (world["packets"] / "processed" / "31e12dca983b8837295ac911c44dee68.json").write_text(json.dumps(request))
    assert _reason(world) is None


@pytest.mark.parametrize("fault", ["intent_missing", "intent_digest_changed", "intent_tampered"])
def test_a_window_that_cannot_be_read_counts_as_ended_so_nothing_is_kept_for_ever(world, fault):
    path = world["intent_root"] / world["intent"]["intent_id"] / "intent.json"
    if fault == "intent_missing":
        path.unlink()
    elif fault == "intent_digest_changed":
        request = {**world["request"], "intent_digest": "sha256:" + "0" * 64}
        (world["packets"] / "processed" / "31e12dca983b8837295ac911c44dee68.json").write_text(json.dumps(request))
    else:
        tampered = {**world["intent"], "request": {"execution": {"expires_at_epoch": world["now"] + 10**9}}}
        path.write_text(json.dumps(tampered))
    assert _reason(world) is None


def test_an_unreadable_screening_ledger_keeps_the_run_while_the_window_is_open(world):
    (world["screening"] / coordinator.LEDGER_FILENAME).write_text("{not json")
    assert _reason(world) == PROTECTED_OPEN_JOB_SOURCE


def test_a_run_no_packet_request_names_is_not_kept(world):
    (world["packets"] / "processed" / "31e12dca983b8837295ac911c44dee68.json").unlink()
    assert _reason(world) is None


# Point 2: a run already in cold storage is one typed state.

def _offloaded(world) -> Path:
    import shutil

    shutil.rmtree(world["run"])
    pointer = world["runs"] / (world["run"].name + cold.POINTER_SUFFIX)
    pointer.write_text(json.dumps(world["pointer"]), encoding="utf-8")
    return pointer


def test_before_the_fix_the_offloaded_447_run_reads_as_an_invalid_launch_receipt(world):
    from blueprint_pipeline.task_evaluation_configured_controls_source_evidence import (
        TaskEvaluationConfiguredControlsProgressionWorkerError,
        _validate_source,
    )

    _offloaded(world)
    with pytest.raises(TaskEvaluationConfiguredControlsProgressionWorkerError,
                       match="configured_controls_worker_launch_receipt_invalid"):
        _validate_source(world["run"])


def test_the_offloaded_447_run_is_named_cold_storage_and_its_hold_is_typed(world):
    _offloaded(world)
    assert cold.in_cold_storage(world["run"]) is True
    with pytest.raises(ValueError, match=f"^{cold.COLD_STORAGE}$"):
        cold.require_on_disk(world["run"])
    assert cold.cold_storage_hold() == ("team_scene_packet_blocked", "scene_run_in_cold_storage")
    assert cold.cold_storage_hold()[0] in holds.REASONS  # a reason the website accepts today


def test_a_missing_run_without_a_pointer_or_with_a_broken_pointer_is_not_called_cold_storage(world):
    import shutil

    shutil.rmtree(world["run"])
    assert cold.in_cold_storage(world["run"]) is False
    pointer = world["runs"] / (world["run"].name + cold.POINTER_SUFFIX)
    pointer.write_text(json.dumps({**world["pointer"], "status": "restored"}))
    assert cold.in_cold_storage(world["run"]) is False
    pointer.write_text(json.dumps({**world["pointer"], "directory": "another-run"}))
    assert cold.in_cold_storage(world["run"]) is False


def test_a_restored_run_is_on_disk_again_even_though_its_pointer_stays(world):
    _offloaded(world)
    world["run"].mkdir()
    (world["run"] / "launch_receipt.json").write_bytes(world["receipt"])
    assert cold.in_cold_storage(world["run"]) is False
    cold.require_on_disk(world["run"])


def test_the_robot_free_probe_on_an_offloaded_run_is_held_by_name_not_left_in_progress(world, monkeypatch):
    from blueprint_pipeline import rigid_robot_free_probe_bundle as bundle
    from blueprint_pipeline import team_candidate_quick10_registry as registry
    from blueprint_pipeline import team_robot_free_probe

    _offloaded(world)
    capture = world["run"].parent / "capture"
    profile = {"robot_free_probe_root": str(world["run"]), "capture_root": str(capture)}
    monkeypatch.setattr(registry, "profile_for_capture", lambda _root: profile)
    monkeypatch.setattr(registry, "scene_facts", lambda _profile: {"scene": {
        "scene_revision_digest": world["request"]["scene_revision_digest"], "task_spec": {"task_kind": "x"}}})
    monkeypatch.setattr(team_robot_free_probe, "probe_request", lambda **_kwargs: {"request_digest": "sha256:" + "1" * 64})
    state = bundle.current_probe_state(capture)
    assert state["status"] == cold.COLD_STORAGE
    # Before: ``probe_root_unavailable`` gave no hold, and the screening pass read it as progress (RELEASE).
    assert holds.probe_hold("team_evaluation_robot_free_probe_receipt_missing", capture) == (
        "team_scene_packet_blocked", "scene_run_in_cold_storage")


def test_the_team_half_recompile_of_an_offloaded_run_is_refused_by_the_same_code(world):
    from blueprint_pipeline import team_scene_packet_worker as packets

    _offloaded(world)
    with pytest.raises(ValueError, match=f"^{cold.COLD_STORAGE}$"):
        packets._compile(world["packets"], world["request"], status={"request_key": "k"}, capture=world["run"],
                         inputs={}, fetcher=None)


class _Restore:
    """The archive's restore and the disk ledger, faked: it lands the run with the offload's old times."""

    def __init__(self, monkeypatch, *, disk=True, transfer=True, intact=True):
        from contextlib import nullcontext

        from blueprint_pipeline import control_plane_disk_budget as budget
        from blueprint_pipeline import control_plane_evidence_offload as offload
        from blueprint_pipeline.control_plane_disk_ledger import ControlPlaneDiskBudgetError

        self.disk, self.transfer, self.intact, self.reserved, self.calls = disk, transfer, intact, [], 0

        def reserve(role, **kwargs):
            if not self.disk:
                raise ControlPlaneDiskBudgetError("control_plane_disk_budget_exceeded")
            self.reserved.append((role, kwargs["expected_bytes"]))
            return nullcontext()

        def restore(*, pointer_path, destination):
            self.calls += 1
            if not self.transfer:
                raise ConnectionError("artifact store unreachable")  # transient: tried again
            if not self.intact:
                raise offload.ControlPlaneEvidenceOffloadError("control_plane_evidence_restore_digest_mismatch")
            Path(destination).mkdir(parents=True)
            (Path(destination) / "launch_receipt.json").write_text("{}")
            old = time.time() - 30 * 86400
            for path in (Path(destination) / "launch_receipt.json", Path(destination)):
                os.utime(path, (old, old))
            return {"status": "restored"}

        monkeypatch.setattr(budget, "reserve_control_plane_disk", reserve)
        monkeypatch.setattr(offload, "restore_offloaded_evidence", restore)


def test_an_open_jobs_run_comes_back_by_itself_and_the_maintenance_pass_releases_its_hold(tmp_path, monkeypatch):
    from blueprint_pipeline import scene_stage_maintenance as maintenance
    from blueprint_pipeline import scene_run_restore
    from tests.test_beta_policy_screening_worker import prepared_source_queue

    queue = prepared_source_queue(tmp_path, monkeypatch)
    # The fixture's profile names no probe root: its robot-free probe reads as passed.
    monkeypatch.setattr(maintenance, "probe_gate", lambda _profile: {})
    run = Path(queue["requests"][0]["source_launch_root"])
    pointer = _fixture("launch_run.offloaded.v1.json")
    run.parent.mkdir(parents=True, exist_ok=True)
    (run.parent / (run.name + cold.POINTER_SUFFIX)).write_text(
        json.dumps(_sealed({**pointer, "directory": run.name}, "pointer_digest")), encoding="utf-8")
    monkeypatch.setattr(scene_run_restore, "retry_delay", lambda _failures: 0)
    fake = _Restore(monkeypatch, disk=False)
    kwargs = {"queue_root": queue["screening_queue"], "packet_queue_root": queue["packet_queue"]}
    # No room for the download and its extraction: nothing is restored or spent, the job is held by name.
    [outcome] = maintenance.scene_stage_pass(**kwargs)["sources"]
    assert outcome["blockers"] == [cold.COLD_STORAGE] and fake.calls == 0
    assert outcome["scene_run_restore"]["status"] == "refused"
    [held] = holds.pending_holds(queue_root=queue["screening_queue"])
    assert (held["reason_code"], held["detail_code"]) == ("team_scene_packet_blocked", "scene_run_in_cold_storage")
    # Room returns: the next pass restores it (no person), reads it, and releases the hold.
    fake.disk = True
    [planned] = maintenance.scene_stage_pass(**kwargs)["sources"]
    assert planned["scene_run_restore"]["status"] == "restored" and fake.calls == 1
    assert fake.reserved == [("evidence_offload", 2 * pointer["size_bytes"])]
    assert planned["status"] == "scene_stage_ready" and planned["evaluation_hold"]["status"] == "released"
    # Fresh: every restored file's time is now, so the clean-up's two-day rule does not take it again at once.
    assert min(path.stat().st_mtime for path in (run, *run.rglob("*"))) > time.time() - 60


def test_a_restore_keeps_trying_while_the_window_is_open_spaced_by_the_retry_policy(world, monkeypatch):
    """A failed try spends nothing: it is tried again while the job's window runs (``job_retry_policy``), and
    only the window closing ends it (AGENTS.md: the job's window is its deadline)."""
    from blueprint_pipeline import scene_run_restore as restore
    from blueprint_pipeline.job_retry_policy import outage_retry_delay

    _offloaded(world)
    fake = _Restore(monkeypatch, transfer=False)
    assert restore.restore_open_job_run(world["run"], window_open=False) == {"status": "window_closed"}
    assert fake.calls == 0  # 447's window has ended: not restored
    clock = [1000.0]
    for failures in range(1, 9):
        result = restore.restore_open_job_run(world["run"], window_open=True, now=lambda: clock[0])
        delay = outage_retry_delay(failures, first=900, cap=6 * 3600)
        assert result["status"] == "refused" and result["retry_not_before"] == clock[0] + delay, result
        assert restore.retry_delay(failures) == delay
        # Before its time: it waits, and tries nothing.
        assert restore.restore_open_job_run(world["run"], window_open=True,
                                            now=lambda: clock[0] + delay - 1)["status"] == "waiting_retry"
        clock[0] += delay
    assert fake.calls == 8  # never given up while the window is open
    assert [restore.retry_delay(n) for n in (1, 2, 3, 4, 5, 6)] == [900, 1800, 3600, 7200, 14400, 21600]
    # The window closes: the tries end, nothing more is fetched.
    assert restore.restore_open_job_run(world["run"], window_open=False, now=lambda: clock[0]) == {
        "status": "window_closed"}
    assert fake.calls == 8


def test_a_job_marked_final_whose_half_is_compiled_again_gets_its_run_back(world, monkeypatch):
    """Review point 5: "final" ignores a later scene-half change; the run it then needs is restored, not lost."""
    from blueprint_pipeline import scene_run_restore as restore
    from blueprint_pipeline import team_scene_packet_worker as packets

    state = copy.deepcopy(world["campaign"]["state"])
    for row in state["candidates"].values():
        if row["status"] == "deferred":
            row.update(status="unavailable", blockers=["candidate_infeasible_measured_geometry"])
    world["ledger"](state)
    testing.publish_testing_record(world["screening"])
    assert _reason(world) is None  # final: the clean-up may move it
    _offloaded(world)
    # The half is compiled again (a new station): it needs the run, and the job's window is still open.
    with pytest.raises(ValueError, match=f"^{cold.COLD_STORAGE}$"):
        packets._compile(world["packets"], world["request"], status={"request_key": "k"}, capture=world["run"],
                         inputs={}, fetcher=None)
    _Restore(monkeypatch)
    assert restore.restore_open_job_run(world["run"], window_open=True)["status"] == "restored"
    blocked = world["packets"] / "processed" / "31e12dca983b8837295ac911c44dee68.json"
    assert packets._run_restored_since_refused(blocked, {"status": "blocked", "blockers": [cold.COLD_STORAGE]})


def _under_floor(monkeypatch, deficit):
    from types import SimpleNamespace

    from blueprint_pipeline.control_plane_disk_budget import floor_bytes

    total = 100 * 1024**3
    floor = floor_bytes(total, role="evidence_offload")
    monkeypatch.setattr(open_jobs, "measure_disk",
                        lambda _path: SimpleNamespace(total=total, free=floor - deficit, used=0))


def _second_open_job(world, *, older):
    run = world["runs"] / "website-second-run"
    run.mkdir()
    (run / "frames.bin").write_bytes(b"x" * 16384)
    stamp = os.stat(world["run"]).st_mtime + (-3600 if older else 3600)
    for path in (*run.rglob("*"), run):
        os.utime(path, (stamp, stamp))
    request = {**world["request"], "source_launch_root": str(run)}
    (world["packets"] / "processed" / "second.json").write_text(json.dumps(request))
    return run


def test_at_the_floor_the_oldest_open_jobs_run_goes_first_and_only_as_far_as_needed(world, monkeypatch):
    older = _second_open_job(world, older=True)
    _under_floor(monkeypatch, deficit=1)  # one byte under: the oldest run covers it
    assert open_jobs.open_job_source_reason(older, packet_queue_root=world["packets"],
                                            screening_queue_root=world["screening"],
                                            intent_root=world["intent_root"]) is None
    assert _reason(world) == PROTECTED_OPEN_JOB_SOURCE  # 447's newer run stays
    _under_floor(monkeypatch, deficit=0)  # at or above the floor: both are kept
    assert open_jobs.open_job_source_reason(older, packet_queue_root=world["packets"],
                                            screening_queue_root=world["screening"],
                                            intent_root=world["intent_root"]) == PROTECTED_OPEN_JOB_SOURCE


def test_at_the_floor_a_run_whose_job_has_a_test_running_is_never_moved(world, monkeypatch):
    state = copy.deepcopy(world["campaign"]["state"])
    next(iter(state["candidates"].values())).update(status="running")
    world["ledger"](state)
    _under_floor(monkeypatch, deficit=10**12)
    assert open_jobs.job_test_active(world["request"]["intent_id"], screening_queue_root=world["screening"])
    assert _reason(world) == PROTECTED_OPEN_JOB_SOURCE


def test_a_team_half_refused_for_cold_storage_is_held_by_name_on_the_first_pass(tmp_path, monkeypatch):
    from blueprint_pipeline import scene_stage_maintenance as maintenance
    from blueprint_pipeline import team_scene_packet_worker as packets
    from tests.test_beta_policy_screening_worker import prepared_source_queue

    queue = prepared_source_queue(tmp_path, monkeypatch)
    key = packets.request_key(queue["requests"][0])
    (queue["packet_queue"] / "status" / f"{key}.json").write_text(json.dumps(
        {"status": "blocked", "blockers": [cold.COLD_STORAGE], "release_commit": "a" * 40}))
    monkeypatch.setattr(packets, "current_release_commit", lambda: "a" * 40)
    [outcome] = maintenance.scene_stage_pass(queue_root=queue["screening_queue"],
                                             packet_queue_root=queue["packet_queue"])["sources"]
    assert outcome["blockers"] == ["beta_screening_scene_packet_not_ready"]
    [held] = holds.pending_holds(queue_root=queue["screening_queue"])
    assert (held["reason_code"], held["detail_code"]) == ("team_scene_packet_blocked", "scene_run_in_cold_storage")


def _kept_pointer(run: Path, files: dict[str, bytes]) -> None:
    import base64
    import hashlib

    members = [{"relative_path": name, "size_bytes": len(raw), "sha256": "sha256:" + hashlib.sha256(raw).hexdigest()}
               for name, raw in files.items()]
    pointer = {"schema_version": cold.POINTER_SCHEMA, "status": "offloaded", "directory": run.name,
               "members": members, "size_bytes": 1,
               "retained_receipt_bytes": {name: base64.b64encode(raw).decode() for name, raw in files.items()}}
    (run.parent / (run.name + cold.POINTER_SUFFIX)).write_text(json.dumps(_sealed(pointer, "pointer_digest")))


def test_the_offload_keeps_what_a_replan_reads_and_it_is_read_back_without_a_restore(tmp_path, monkeypatch):
    """Review gap 4: 447's records are 1.6, 2.4, 5.4 and 35.5 KB, all under the 64 KiB receipt bound."""
    from blueprint_pipeline import control_plane_retained_receipt as kept
    from blueprint_pipeline import task_evaluation_configured_controls_source_evidence as evidence
    from blueprint_pipeline.robot_free_probe_gate import retained_probe_receipt

    run = tmp_path / "runs" / "website-run"
    (run / "allocator").mkdir(parents=True)
    result = run / "allocator" / "result.v1.json"
    receipt = {"terminal_evidence": {"result": {"path": str(result)}}}
    (run / "launch_receipt.json").write_text(json.dumps(receipt))
    members = [{"relative_path": name, "size_bytes": 100} for name in (
        "launch_receipt.json", "allocator/result.v1.json", "webapp_sync_succeeded.json",
        "post_teardown_provider_zero_receipt.json", "robot_free_readiness.v1.json", "frames.bin")]
    members[-1]["size_bytes"] = 10**9
    assert kept.replan_members(run, members) == {"allocator/result.v1.json", "webapp_sync_succeeded.json",
                                                 "post_teardown_provider_zero_receipt.json",
                                                 "robot_free_readiness.v1.json"}
    import shutil

    shutil.rmtree(run)
    files = {"launch_receipt.json": json.dumps(receipt).encode(), "allocator/result.v1.json": b'{"r": 1}',
             "webapp_sync_succeeded.json": b'{"s": 1}', "post_teardown_provider_zero_receipt.json": b'{"z": 1}',
             "robot_free_readiness.v1.json": b'{"status": "passed"}'}
    _kept_pointer(run, files)
    assert kept.retained_member(run, "webapp_sync_succeeded.json") == b'{"s": 1}'
    assert retained_probe_receipt(run) == {"status": "passed"}  # planning reads it from the pointer
    seen = {}

    def validate(root, *, load_json, sha256, is_file):
        seen.update(result=load_json(result, blocker="x"), present=is_file(result), digest=sha256(result),
                    sync=load_json(root / "webapp_sync_succeeded.json", blocker="x"))
        return seen["result"], {}, {}

    monkeypatch.setattr(evidence, "validate_source", validate)
    evidence.validate_retained_source(run)
    assert seen["result"] == {"r": 1} and seen["present"] is True and seen["sync"] == {"s": 1}
    assert seen["digest"].startswith("sha256:")
    # A record the pointer does not keep, or whose bytes changed, is not read: the re-plan restores instead.
    assert kept.retained_member(run, "frames.bin") is None


def test_a_restore_never_writes_a_withdrawn_jobs_run_back(world, monkeypatch):
    """Review 36 (7a): withdrawn after the fetch began: what was fetched is deleted, nothing lands."""
    from blueprint_pipeline import scene_run_restore as restore

    _offloaded(world)
    _Restore(monkeypatch)
    answers = iter([True, False])  # current when the restore starts, withdrawn before it places anything
    result = restore.restore_open_job_run(world["run"], window_open=True, still_current=lambda: next(answers))
    assert result == {"status": restore.WITHDRAWN}
    assert not world["run"].exists() and not (world["run"].parent / f".{world['run'].name}.restoring").exists()
    # Withdrawn before it starts: nothing is fetched at all.
    fake = _Restore(monkeypatch)
    assert restore.restore_open_job_run(world["run"], window_open=True, still_current=lambda: False) == {
        "status": restore.WITHDRAWN}
    assert fake.calls == 0


def test_a_restore_waits_while_a_deletion_holds_its_lock(world, monkeypatch):
    import fcntl

    from blueprint_pipeline import scene_run_restore as restore

    _offloaded(world)
    _Restore(monkeypatch)
    lock = world["packets"] / ".lock"
    lock.write_text("")
    with lock.open("rb") as held:
        fcntl.flock(held, fcntl.LOCK_EX)  # a withdrawal deletion running now
        busy = restore.restore_open_job_run(world["run"], window_open=True, packet_queue_root=world["packets"])
    assert busy["status"] == restore.BUSY and not world["run"].exists()
    done = restore.restore_open_job_run(world["run"], window_open=True, packet_queue_root=world["packets"])
    assert done["status"] == restore.RESTORED and world["run"].is_dir()


def test_three_failed_tries_then_a_success_inside_the_window_restore_the_run(world, monkeypatch):
    """Review 36 (7b) and the PR review: three failed tries no longer end an open job in cold storage."""
    import shutil

    from blueprint_pipeline import scene_run_restore as restore

    _offloaded(world)
    fake = _Restore(monkeypatch, transfer=False)
    clock = [0.0]
    for failures in range(1, 4):
        assert restore.restore_open_job_run(world["run"], window_open=True, now=lambda: clock[0])["status"] == "refused"
        clock[0] += restore.retry_delay(failures)
    fake.transfer = True
    assert restore.restore_open_job_run(world["run"], window_open=True, now=lambda: clock[0])["status"] == restore.RESTORED
    # Offloaded again later: a fresh episode with its own bound.
    shutil.rmtree(world["run"])
    fake.transfer = False
    assert restore.restore_open_job_run(world["run"], window_open=True, now=lambda: clock[0])["status"] == "refused"


def test_at_the_floor_a_job_with_pending_or_waitlisted_rows_keeps_its_run(world, monkeypatch):
    """Review 36 (7c)."""
    for status in ("pending", "waitlisted"):
        state = copy.deepcopy(world["campaign"]["state"])
        next(iter(state["candidates"].values())).update(status=status)
        world["ledger"](state)
        _under_floor(monkeypatch, deficit=10**12)
        assert _reason(world) == PROTECTED_OPEN_JOB_SOURCE


def test_a_crashed_restores_staging_is_cleaned_unless_a_restorer_holds_the_run(world):
    """Review 8c (7): staging left by a crash is removed on the next pass, open job or not."""
    import fcntl

    from blueprint_pipeline import scene_run_restore as restore

    _offloaded(world)
    staging = world["run"].parent / f".{world['run'].name}.restoring"
    (staging / "allocator").mkdir(parents=True)
    with restore._restorer_lock(world["run"]).open("a+b") as held:
        fcntl.flock(held, fcntl.LOCK_EX)  # a restorer at work
        assert restore.clean_stale_staging(world["run"]) is False and staging.exists()
    assert restore.clean_stale_staging(world["run"]) is True and not staging.exists()


# A restore killed mid-way leaves the site's bytes only where the run's own name ties them (PR review, privacy).


def _real_offload(world, tmp_path):
    """447's run archived for real, its pointer beside where it stood, and a store that serves the archive."""
    import hashlib
    import shutil
    import tarfile

    run = world["run"]
    archive = tmp_path / "store" / "evidence.tar"
    archive.parent.mkdir()
    files = sorted(path for path in run.rglob("*") if path.is_file())
    with tarfile.open(archive, "w") as tar:
        for path in files:
            tar.add(path, arcname=path.relative_to(run).as_posix())
    digest = lambda raw: "sha256:" + hashlib.sha256(raw).hexdigest()  # noqa: E731
    members = [{"relative_path": path.relative_to(run).as_posix(), "size_bytes": path.stat().st_size,
                "sha256": digest(path.read_bytes())} for path in files]
    pointer = _sealed({"schema_version": cold.POINTER_SCHEMA, "status": "offloaded", "directory": run.name,
                       "uri": "gs://store/evidence.tar", "digest": digest(archive.read_bytes()),
                       "size_bytes": archive.stat().st_size, "members": members}, "pointer_digest")
    shutil.rmtree(run)
    (world["runs"] / (run.name + cold.POINTER_SUFFIX)).write_text(json.dumps(pointer), encoding="utf-8")
    return archive


def _materializer(archive, *, kill_after_bytes=None):
    def materialize(*, reference, destination, maximum_size_bytes):
        raw = archive.read_bytes()
        if kill_after_bytes is not None:
            Path(destination).write_bytes(raw[:kill_after_bytes])  # part of the archive is on disk
            raise KeyboardInterrupt  # the process is killed mid-download
        Path(destination).write_bytes(raw)
        return {"status": "materialized"}
    return materialize


def _restore_with(world, materializer):
    from contextlib import nullcontext
    from functools import partial

    from blueprint_pipeline import control_plane_evidence_offload as offload
    from blueprint_pipeline import scene_run_restore as restore

    return restore.restore_open_job_run(
        world["run"], window_open=True, reserve=lambda *_a, **_k: nullcontext(),
        restorer=partial(offload.restore_offloaded_evidence, materializer=materializer))


def _outside_the_run_named_staging(world) -> list[str]:
    """Everything in the runs folder a withdrawal could not tie to the run by its name, and any run bytes."""
    run = world["run"].name
    allowed = {run + cold.POINTER_SUFFIX, f".{run}.restore.lock", f".{run}.restoring", "pins"}
    return sorted(path.name for path in world["runs"].iterdir() if path.name not in allowed)


_KILLED_RESTORE = """
import json, os, signal, sys, tarfile
from contextlib import nullcontext
from functools import partial
from pathlib import Path

from blueprint_pipeline import control_plane_evidence_offload as offload
from blueprint_pipeline import scene_run_restore as restore

run, archive, phase = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
raw = archive.read_bytes()

def materialize(*, reference, destination, maximum_size_bytes):
    if phase == "download":
        Path(destination).write_bytes(raw[: len(raw) // 2])  # part of the archive is on disk
        os.kill(os.getpid(), signal.SIGKILL)  # killed mid-download: no clean-up runs
    Path(destination).write_bytes(raw)

real = tarfile.TarFile.extractall

def killed(self, path, *args, **kwargs):
    real(self, path, members=self.getmembers()[:1], filter="data")  # one member written
    os.kill(os.getpid(), signal.SIGKILL)  # killed mid-extraction

tarfile.TarFile.extractall = killed
restore.restore_open_job_run(run, window_open=True, reserve=lambda *_a, **_k: nullcontext(),
                             restorer=partial(offload.restore_offloaded_evidence, materializer=materialize))
"""


@pytest.mark.parametrize("phase", ["download", "extraction"])
def test_a_restore_killed_mid_way_leaves_its_bytes_only_in_the_run_named_staging(world, tmp_path, phase):
    """The process is SIGKILLed during the download or the extraction (no ``finally`` runs)."""
    import signal
    import subprocess
    import sys

    from blueprint_pipeline import scene_run_restore as restore
    from blueprint_pipeline import website_capture_withdrawal_deletion as deletion

    archive = _real_offload(world, tmp_path)
    staging = restore.staging_path(world["run"])
    done = subprocess.run([sys.executable, "-c", _KILLED_RESTORE, str(world["run"]), str(archive), phase],
                          capture_output=True, text=True, timeout=120)
    assert done.returncode == -signal.SIGKILL, done.stderr
    # Nothing of the run outside the one directory named after it; the run itself has not landed.
    assert _outside_the_run_named_staging(world) == [] and not world["run"].exists()
    written = [path for path in staging.rglob("*") if path.is_file()]
    assert written, "the killed restore wrote part of the run"
    assert all(staging in path.parents for path in written)
    # A withdrawal deletion planned now ties the staging directory to the run and deletes it with it.
    unresolved: list = []
    units = deletion._Units([world["runs"]], unresolved)
    assert unresolved == [] and staging in units.units[(str(world["runs"]), world["run"].name)]["leftovers"]
    # The next pass removes it (no restorer holds the run), then restores the run whole.
    assert restore.clean_stale_staging(world["run"]) is True and not staging.exists()
    done = _restore_with(world, _materializer(archive))
    assert done["status"] == restore.RESTORED and (world["run"] / "launch_receipt.json").is_file()
    assert not staging.exists() and _outside_the_run_named_staging(world) == sorted(
        [world["run"].name, world["run"].name + restore.STATE_SUFFIX])


def test_an_archive_that_does_not_match_its_pointer_is_tried_once_until_the_pointer_changes(world, monkeypatch):
    """PR re-check: an integrity failure cannot clear by waiting, so the archive is not downloaded every six hours."""
    from blueprint_pipeline import scene_run_restore as restore

    pointer_path = _offloaded(world)
    fake = _Restore(monkeypatch, intact=False)
    clock = [0.0]
    first = restore.restore_open_job_run(world["run"], window_open=True, now=lambda: clock[0])
    assert first["status"] == restore.ARCHIVE_INVALID and "retry_not_before" not in first
    assert first["error"] == "control_plane_evidence_restore_digest_mismatch"
    for _ in range(3):
        clock[0] += 7 * 3600
        assert restore.restore_open_job_run(world["run"], window_open=True,
                                            now=lambda: clock[0])["status"] == restore.ARCHIVE_INVALID
    assert fake.calls == 1  # never fetched again while the pointer is unchanged
    # A changed pointer (the run offloaded again) is a new episode: it is tried, and lands.
    changed = _sealed({**world["pointer"], "size_bytes": world["pointer"]["size_bytes"] + 1}, "pointer_digest")
    pointer_path.write_text(json.dumps(changed), encoding="utf-8")
    fake.intact = True
    assert restore.restore_open_job_run(world["run"], window_open=True, now=lambda: clock[0])["status"] == restore.RESTORED
    assert fake.calls == 2


def test_an_archive_that_will_not_open_is_an_integrity_failure():
    import tarfile

    from blueprint_pipeline import scene_run_restore as restore

    assert restore._integrity_failure(tarfile.ReadError("not a tar"))
    assert not restore._integrity_failure(ConnectionError("store unreachable"))
    assert not restore._integrity_failure(TimeoutError())


@pytest.mark.parametrize("fault, detail", [("integrity", "scene_run_restore_archive_integrity_failed"),
                                           ("transfer", "scene_run_in_cold_storage")])
def test_the_site_is_told_restoring_only_while_a_restore_can_still_land(tmp_path, monkeypatch, fault, detail):
    """An archive that fails its own integrity is not being restored: the hold names that (the website's general
    "a problem on our side" sentence, which its ops alert reports to staff), never ``scene_run_in_cold_storage``;
    a transient failure keeps the cold-storage hold ("being restored")."""
    from blueprint_pipeline import scene_stage_maintenance as maintenance
    from blueprint_pipeline import scene_run_restore
    from tests.test_beta_policy_screening_worker import prepared_source_queue

    queue = prepared_source_queue(tmp_path, monkeypatch)
    # The fixture's profile names no probe root: its robot-free probe reads as passed.
    monkeypatch.setattr(maintenance, "probe_gate", lambda _profile: {})
    run = Path(queue["requests"][0]["source_launch_root"])
    pointer = _fixture("launch_run.offloaded.v1.json")
    run.parent.mkdir(parents=True, exist_ok=True)
    (run.parent / (run.name + cold.POINTER_SUFFIX)).write_text(
        json.dumps(_sealed({**pointer, "directory": run.name}, "pointer_digest")), encoding="utf-8")
    monkeypatch.setattr(scene_run_restore, "retry_delay", lambda _failures: 0)
    _Restore(monkeypatch, transfer=fault != "transfer", intact=fault != "integrity")
    kwargs = {"queue_root": queue["screening_queue"], "packet_queue_root": queue["packet_queue"]}
    for _ in range(2):  # the second pass reads the record the first one wrote
        [outcome] = maintenance.scene_stage_pass(**kwargs)["sources"]
    [held] = holds.pending_holds(queue_root=queue["screening_queue"])
    assert (held["reason_code"], held["detail_code"]) == ("team_scene_packet_blocked", detail)
    assert outcome["scene_run_restore"]["status"] == (
        scene_run_restore.ARCHIVE_INVALID if fault == "integrity" else "refused")
