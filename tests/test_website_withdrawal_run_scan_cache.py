"""A withdrawal plan keeps what each run's records name and reads again only what changed.

Beta journey step: site withdrawal. The first withdrawal planned on the host read about 9 GiB of run
records and was killed at its unit's 45-minute start timeout on every try (2026-11-25). Every test
runs on the offline drill's synthetic world under a temporary root; nothing reads /var/lib/blueprint.
"""
# Covers: src/blueprint_pipeline/website_withdrawal_run_scan_cache.py
# Covers: src/blueprint_pipeline/website_capture_withdrawal_deletion.py
# Covers: src/blueprint_pipeline/website_capture_withdrawal_automatic.py
from __future__ import annotations

import io
import json
import os
import random
import time
from pathlib import Path

import pytest

from blueprint_pipeline import website_capture_withdrawal as withdrawal
from blueprint_pipeline import website_capture_withdrawal_automatic as automatic
from blueprint_pipeline import website_capture_withdrawal_deletion as deletion
from blueprint_pipeline import website_capture_withdrawal_drill as drill
from blueprint_pipeline import website_withdrawal_run_scan_cache as run_scan

# A request id of the shape the website gives a job, long enough for the cache to answer for it.
REQUEST = "capture-ceba4511-bfca-4a22-8fdc-0ffae40d27c3"


def _command(request_id: str) -> dict:
    return {"schema_version": "website_capture_withdrawal.v1", "request_id": request_id,
            "scene_id": f"site-{request_id}", "capture_id": f"walkthrough-{request_id}",
            "withdrawal_id": f"withdrawal-{request_id}", "requested_at_iso": "2026-10-06T00:00:00Z"}


@pytest.fixture()
def world(tmp_path: Path) -> dict:
    world = drill.build_world(tmp_path, REQUEST)
    withdrawal.acknowledge_withdrawal(capture_root=world["capture"], command=_command(REQUEST))
    world["resolution"] = {**world["resolution"],
                           "object_store": deletion.B2ObjectVersions(world["store"], drill.BUCKET),
                           "cloud_storage": world["gcs"]}
    world["cache"] = tmp_path / "report" / run_scan.DIRNAME
    return world


def _plan(world: dict, **kwargs) -> dict:
    return deletion.plan_withdrawal_deletion(capture_root=world["capture"], **world["resolution"], **kwargs)


def _scan_both(records: list[bytes], vocabulary: set[bytes], request_id: str) -> tuple[tuple, tuple]:
    plain = deletion._Scan(vocabulary, request_id)
    for record in records:
        plain.feed(io.BytesIO(record))
    facts = run_scan.Facts()
    for record in records:
        facts.feed(io.BytesIO(record))
    kept = deletion._Scan(vocabulary, request_id)
    kept.absorb(facts)
    return ((plain.named, plain.other_sites, plain.providers), (kept.named, kept.other_sites, kept.providers))


@pytest.mark.parametrize("chunk,overlap", [(1024 * 1024, 4096), (64, 16), (37, 23), (13, 5)])
def test_the_kept_facts_answer_exactly_what_reading_answers_at_every_window_edge(monkeypatch, chunk, overlap):
    # Small windows put window edges inside tokens, site ids and provider fields; the two readings must agree.
    monkeypatch.setattr(deletion, "_CHUNK", chunk)
    monkeypatch.setattr(deletion, "_OVERLAP", overlap)
    words = [f"site-{REQUEST}", f"walkthrough-{REQUEST}", "scene-" + "a1" * 32,
             "launch-capture-ceba4511-construction-0001", "team-scored-run-drill-0001",
             "website-31b82ec72981d4694d0a947f-92acda8b-20261126t021840z-e4dd82c12a67-activation-auto-launch"]
    noise = ["0.12345678901234", "deadbeefcafe0123456789abcdef0123", "abe19c87-5997-4c7c-aedf-6d10fb6abd27",
             "frame_000123.png", "2026-10-26T03", "x", "launch", "site-other-capture-0002", "walkthrough-drill-other",
             "site-capture-ceba4511-bfca-4a22-8fdc-0ffae40d27c3.retired.v1.json"]
    generator = random.Random(chunk * 1000 + overlap)
    records = []
    for _ in range(40):
        parts = []
        for _ in range(generator.randrange(5, 60)):
            pick = generator.random()
            if pick < 0.2:
                parts.append(json.dumps({"run": generator.choice(words)}))
            elif pick < 0.3:
                parts.append(json.dumps({"scene_id": generator.choice(["site-drill-other", f"site-{REQUEST}"])}))
            elif pick < 0.4:
                parts.append(f'"path": "scenes/site-{generator.choice(["drill-other", REQUEST, "x.y"])}/captures"')
            elif pick < 0.5:
                parts.append(json.dumps({"provider": generator.choice(["world_labs", "openai", "gemini"])}))
            else:
                parts.append(generator.choice(noise + words) + generator.choice([" ", ",", "", "-", "/", "\n"]))
        records.append("".join(parts).encode())
    vocabulary = {word.encode() for word in words}
    plain, kept = _scan_both(records, vocabulary, REQUEST)
    assert kept == plain
    if chunk + overlap > 100:  # a window that can hold a whole id: the case is not empty
        assert plain[0] and plain[1] and plain[2]


def test_only_words_of_the_kept_shape_can_be_answered_from_the_cache():
    assert run_scan.cacheable(f"site-{REQUEST}".encode())
    assert run_scan.cacheable(b"scene-" + b"0f" * 32)
    assert run_scan.cacheable(b"team-eval-7b5d46d97dae6175714134ada374a7d2-policy-canary-3de56f9357e3-activation")
    assert not run_scan.cacheable(b"site-req1")  # shorter than a run name can be
    assert not run_scan.cacheable(b"abe19c87-5997-4c7c-aedf-6d10fb6abd27")  # hex digits only
    assert not run_scan.cacheable(b"launchrunwithoutahyphen")


def test_a_plan_from_kept_facts_is_the_plan_from_reading_and_reads_nothing_the_second_time(world):
    plain = _plan(world)
    first = run_scan.RunScanCache(world["cache"])
    cached = _plan(world, run_scan_cache=world["cache"])
    assert cached == plain and cached["unresolved"] == []
    assert {run["name"] for run in cached["runs"]} == {path.name for path in world["site_runs"]} | {
        f"launch-{REQUEST}-controls-0002"}
    entries = sorted(path for path in world["cache"].rglob("*.json"))
    assert len(entries) == 4  # every run directory: the site's three and the other site's one
    assert all(path.stat().st_mode & 0o077 == 0 for path in entries)
    # Second plan: nothing read, nothing written, the same plan.
    before = {path: path.stat().st_mtime_ns for path in entries}
    assert _plan(world, run_scan_cache=world["cache"], scan_deadline=time.monotonic() - 1) == plain
    assert {path: path.stat().st_mtime_ns for path in entries} == before
    assert first.identity == run_scan.scanner_identity()


def test_a_record_that_changed_is_read_again(world):
    _plan(world, run_scan_cache=world["cache"])
    other = world["other_run"] / "launch_profile.json"
    # The other site's run now also names this site's capture: reading binds it, and it names two sites.
    other.write_text(json.dumps({"capture_root": str(world["other_capture"]), "source": f"walkthrough-{REQUEST}"}))
    plain = _plan(world)
    cached = _plan(world, run_scan_cache=world["cache"])
    assert cached == plain
    assert world["other_run"].name in {run["name"] for run in cached["runs"]}
    assert {"path": str(world["other_run"]), "reason": "website_withdrawal_deletion_run_names_another_site"} \
        in cached["unresolved"]


def test_a_plan_stops_at_its_reading_time_after_one_run_and_the_next_goes_on_from_what_was_kept(world):
    plain = _plan(world)
    progress = []
    for _ in range(10):
        try:
            plan = _plan(world, run_scan_cache=world["cache"], scan_deadline=time.monotonic() - 1)
        except run_scan.ScanBudgetSpent as stopped:
            progress.append(stopped.progress["runs_read"])
            continue
        break
    # One run read per plan (it always reads at least one), then every run is kept and the plan completes.
    assert progress == [1, 1, 1] and plan == plain


def test_a_run_that_cannot_be_kept_never_keeps_the_plan_from_finishing(world, monkeypatch):
    # Codex review on #3759: a run read but not kept used to count as the pass's progress, so a pass
    # that re-read it could stop where the last one did, every hour, for good.
    plain = _plan(world)
    monkeypatch.setattr(run_scan, "MAX_TOKENS", 0)  # no run that names anything can be kept
    for _ in range(6):
        try:
            plan = _plan(world, run_scan_cache=world["cache"], scan_deadline=time.monotonic() - 1)
        except run_scan.ScanBudgetSpent:
            continue
        break
    else:
        pytest.fail("the plan never finished")
    assert plan == plain


def test_an_unreadable_record_stays_unresolved_with_a_kept_entry(world):
    if os.geteuid() == 0:
        pytest.skip("root reads every file")
    _plan(world, run_scan_cache=world["cache"])
    record = world["other_run"] / "launch_profile.json"
    mode = record.stat().st_mode
    record.chmod(0)
    try:
        assert _plan(world, run_scan_cache=world["cache"]) == _plan(world)
    finally:
        record.chmod(mode)


def test_a_vocabulary_word_the_cache_cannot_answer_reads_every_run_and_keeps_nothing(tmp_path):
    world = drill.build_world(tmp_path, "req1")
    withdrawal.acknowledge_withdrawal(capture_root=world["capture"], command=_command("req1"))
    resolution = {**world["resolution"], "object_store": deletion.B2ObjectVersions(world["store"], drill.BUCKET)}
    cache = tmp_path / "report" / run_scan.DIRNAME
    plain = deletion.plan_withdrawal_deletion(capture_root=world["capture"], **resolution)
    cached = deletion.plan_withdrawal_deletion(capture_root=world["capture"], **resolution, run_scan_cache=cache,
                                               scan_deadline=time.monotonic() - 1)
    assert cached == plain and not cache.exists()


def test_an_entry_goes_when_its_run_goes(world):
    _plan(world, run_scan_cache=world["cache"])
    cache = run_scan.RunScanCache(world["cache"])
    other = world["other_run"]
    entry = cache._path(str(other.parent), other.name)
    assert entry.is_file()
    import shutil
    shutil.rmtree(other)
    _plan(world, run_scan_cache=world["cache"])
    assert not entry.exists() and len(list(world["cache"].rglob("*.json"))) == 3


def test_an_apply_removes_the_entries_of_the_runs_it_deletes(world):
    resolution = {**world["resolution"], "run_scan_cache": world["cache"]}
    plan = deletion.plan_withdrawal_deletion(capture_root=world["capture"], **resolution)
    cache = run_scan.RunScanCache(world["cache"])
    deleted = [cache._path(run["root"], run["name"]) for run in plan["runs"] if run["directory"]]
    assert deleted and all(path.is_file() for path in deleted)
    deletion.apply_withdrawal_deletion(capture_root=world["capture"], expected_plan_digest=plan["digest"],
                                       authorize_complete_deletion=True, **resolution)
    assert not any(path.exists() for path in deleted)
    # The other site's run keeps its entry.
    assert cache._path(str(world["other_run"].parent), world["other_run"].name).is_file()


def test_an_hourly_pass_that_runs_out_of_reading_time_is_scanning_not_failing_and_the_next_pass_goes_on(world, tmp_path):
    resolution = {**world["resolution"], "run_scan_cache": world["cache"]}
    report_root = tmp_path / "report"

    def run(seconds):
        return automatic.run_pass(
            site_roots=[world["partition"]], report_root=report_root, resolution=lambda: resolution,
            quiet=list, approval_path=tmp_path / "etc/absent.json", approval_owner_uid=os.getuid(),
            now=time.time(), webhook_url="https://alerts.invalid/hook", poster=lambda url, report: None,
            plan_reading_seconds=seconds)

    def state():
        return json.loads((world["site"] / "website_withdrawal" / automatic.STATE_FILENAME).read_text())

    summary = run(0)
    assert state()["status"] == "scanning" and state()["failure_count"] == 0
    assert state()["waiting_on"] == ["run_records_being_read"] and state()["run_scan"]["runs_read"] == 1
    assert summary["withdrawals"]["scanning"] == 1 and summary["withdrawals"]["failing"] == 0
    assert summary["alerts"] == [] and summary["action"]["status"] == "scanning"
    assert automatic.capacity_projection(summary)["withdrawals"]["scanning"] == 1
    # Due again on the very next pass (no backoff), and it goes on from the runs already read.
    summary = run(0)
    assert state()["status"] == "scanning" and summary["action"]["run_scan"]["runs_reused"] == 1
    summary = run(automatic.PLAN_READING_SECONDS)
    assert state()["status"] == "planned" and state()["waiting_on"] == ["standing_approval_absent"]
    assert summary["withdrawals"]["scanning"] == 0 and summary["withdrawals"]["awaiting_owner_approval"] == 1
    assert json.loads((world["site"] / "website_withdrawal" / automatic.PLAN_FILENAME).read_text()) == _plan(world)


def test_the_production_resolution_keeps_the_cache_under_the_report_root(monkeypatch, tmp_path):
    monkeypatch.setenv(automatic.REPORT_ROOT_ENV, str(tmp_path / "report"))
    monkeypatch.setattr(deletion.B2ObjectVersions, "configured", classmethod(lambda cls: None))
    from blueprint_pipeline import website_capture_withdrawal_provider_copies as provider_copies
    monkeypatch.setattr(provider_copies, "production_deleters", dict)
    assert automatic._production_resolution()["run_scan_cache"] == tmp_path / "report" / run_scan.DIRNAME
