"""Hermetic withdrawal deletion of a site's beta screening attempts and their promoted archives.

Beta journey step: site withdrawal after the site's candidates were screened.
Temporary queue and work roots and the fake B2 client of the deletion tests
only; nothing reaches a provider, a bucket or the control-plane host.
"""
# Covers: src/blueprint_pipeline/website_capture_withdrawal_screening.py
# Covers: src/blueprint_pipeline/website_capture_withdrawal_deletion.py
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest

from blueprint_pipeline import beta_policy_screening_coordinator as coordinator
from blueprint_pipeline import website_capture_withdrawal_deletion as deletion
from blueprint_pipeline import website_capture_withdrawal_screening as screening
from blueprint_pipeline.beta_policy_screening_settlement import SETTLEMENT_BASIS, SETTLEMENT_SCHEMA
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.provider_output_member_index import SCHEMA as INDEX_SCHEMA
from blueprint_pipeline.provider_output_promotion_records import RECEIPT_SCHEMA
from tests import test_website_capture_withdrawal_deletion as base

world = base.world  # the deletion tests' site, runs, B2 and roots
SITE, OTHER_SITE = "site-req1", "site-req2"
STAGING = "allocator/attempts/attempt_001/object_store_staging"
CUSTOMER_BYTES = (b"\x89PNG synthetic site frame", b"synthetic episode video of the site")


def _reference(kind: str, digit: str, filename: str) -> dict:
    """A remote-verified CAS reference as ``publish_configured_scene_stream`` returns it."""
    return {"schema_version": "task_evaluation_scene_artifact_reference.v1", "status": "remote_verified",
            "artifact_kind": kind, "uri": base._uri(kind, digit, filename), "digest": "sha256:" + digit * 64,
            "size_bytes": 4096, "content_addressed_key": True, "remote_identity_verified": True,
            "full_byte_service_account_readback_passed": True}


def _output(digit: str) -> dict:
    return _reference("policy-canary-provider-output", digit, "vast_provider_runtime_output.zip")


def _witness(digit: str) -> dict:
    return _reference("policy-canary-paired-witness", digit, "policy_canary_paired_witness.zip")


def _promotion(output: dict, witness: dict | None = None) -> dict:
    """A sealed promotion receipt naming the promoted output (and witness) in each place the writer does."""
    staged = {"output": {"key_sha256": "1" * 64, "state": "promoted",
                         "versions": [{"size_bytes": 4096, "etag": "e1", "durable_uri": output["uri"],
                                       "durable_reference": output}]}}
    if witness is not None:
        staged["paired_witness"] = {"key_sha256": "2" * 64, "state": "promoted", "versions": [
            {"size_bytes": 4096, "etag": "e2", "durable_uri": witness["uri"], "durable_reference": witness}]}
    return base._sealed({"schema_version": RECEIPT_SCHEMA, "status": "promoted", "durable_reference": output,
                         "staged_objects": staged, "private_url_recorded": False}, "receipt_digest")


def _attempt_dir(directory: Path, *, output: dict | None = None, witness: dict | None = None) -> Path:
    run = directory / "allocator/attempts/attempt_001"
    base._write(directory / "allocator_result.json", {"status": "completed"})
    base._write(run / "vast_provider_run/frames/0001.png", CUSTOMER_BYTES[0])
    base._write(run / "vast_provider_run/cell_runs/00/episodes/media/ep/external.mp4", CUSTOMER_BYTES[1])
    if output is not None:
        base._write(directory / STAGING / "provider_output_promotion.v1.json", _promotion(output, witness))
        base._write(run / "provider_output_member_index.v1.json", base._sealed(
            {"schema_version": INDEX_SCHEMA, "archive": {"sha256": output["digest"], "durable_reference": output},
             "members": []}, "index_digest"))
    return directory


def _settled(attempt_id: str, maximum: str, cost: str) -> dict:
    receipt = {"schema_version": SETTLEMENT_SCHEMA, "settlement_basis": SETTLEMENT_BASIS, "attempt_id": attempt_id,
               "maximum_cost_usd": maximum, "settled_cost_usd": cost, "settled_at": "2026-11-05T00:00:00+00:00"}
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    return {"settled_cost_usd": cost, "settlement_basis": SETTLEMENT_BASIS, "settled_at": receipt["settled_at"],
            "settlement_receipt_digest": receipt["receipt_digest"], "settlement_receipt": receipt}


def _attempt(attempt_id: str, work: Path | None, state: str, *, allocated: bool = True, **extra) -> dict:
    return {"attempt_id": attempt_id, "generation": 0, "maximum_cost_usd": "4", "reservation_status": state,
            "provider_allocation_observed": allocated,
            **({coordinator.ATTEMPT_DIR_FIELD: str(work / "attempts" / attempt_id)} if work else {}), **extra}


def _campaign(site: str, digest: str, candidates: dict) -> dict:
    return {"plan_digest": digest, "site_id": site, "revision": 3, "created_at": "2026-11-05T00:00:00+00:00",
            "updated_at": "2026-11-05T00:00:00+00:00", "published_revision": 3, "published_digest": None,
            "candidates": candidates}


def _save(queue: Path, ledger: dict) -> None:
    ledger["ledger_digest"] = canonical_digest(ledger, digest_field="ledger_digest")
    coordinator._write(queue / coordinator.LEDGER_FILENAME, ledger)


def _ledger(queue: Path) -> dict:
    return json.loads((queue / coordinator.LEDGER_FILENAME).read_text())


@pytest.fixture
def screened(world):
    """Two of the site's campaigns (one reserved before the work root existed) and another site's."""
    queue, work = (Path(world["resolution"][key]) for key in ("screening_queue_root", "screening_work_root"))
    for directory in (queue / "plans", queue / "attempts", queue / "publications", work / "attempts"):
        directory.mkdir(parents=True, exist_ok=True)
    (queue / ".ledger.lock").touch()
    ids = {name: "screen-" + digit * 32 for name, digit in
           (("settled", "1"), ("history", "2"), ("current", "3"), ("active", "4"), ("held", "5"), ("other", "6"))}
    archives = {"output": _output("1"), "witness": _witness("2"), "active": _output("c"), "held": _output("d"),
                "other": _output("e")}
    _attempt_dir(work / "attempts" / ids["settled"], output=archives["output"], witness=archives["witness"])
    _attempt_dir(work / "attempts" / ids["history"])
    _attempt_dir(work / "attempts" / ids["active"], output=archives["active"])
    _attempt_dir(queue / "attempts" / ids["held"], output=archives["held"])
    _attempt_dir(work / "attempts" / ids["other"], output=archives["other"])
    for name in ("settled", "history", "active", "other"):
        base._write(queue / "attempts" / f"{ids[name]}.attempt_dir.json", {"attempt_id": ids[name]})
    for archive in archives.values():
        world["store"].put(archive["uri"], b"synthetic archive", versions=2, marker=True)
    campaigns = {
        "screening-site-current": _campaign(SITE, "sha256:" + "a" * 64, {
            "learned-a": {"status": "completed", "blockers": [], "attempt": _attempt(
                ids["settled"], work, "committed", **_settled(ids["settled"], "4", "0.44"))},
            "learned-b": {"status": "blocked", "blockers": ["beta_screening_gpu_capacity_unavailable"],
                          "attempt_history": [_attempt(ids["history"], work, "released", allocated=False)],
                          "attempt": _attempt(ids["current"], work, "released", allocated=False, generation=1)},
            "learned-c": {"status": "running", "blockers": [], "attempt": _attempt(ids["active"], work, "reserved")}}),
        # Reserved before the work root: its directory is the queue root's, named by no recorded path.
        "screening-site-superseded": _campaign(SITE, "sha256:" + "b" * 64, {
            "scripted": {"status": "completed", "blockers": [], "attempt": _attempt(
                ids["held"], None, "committed", settlement_held={"reason": "beta_screening_settlement_unproven",
                                                                 "held_at": "2026-11-05T00:00:00+00:00"})}}),
        "screening-other-site": _campaign(OTHER_SITE, "sha256:" + "c" * 64, {
            "learned-a": {"status": "completed", "blockers": [], "attempt": _attempt(
                ids["other"], work, "committed", **_settled(ids["other"], "4", "0.91"))}}),
    }
    for campaign_id, state in campaigns.items():
        site = state["site_id"]
        base._write(queue / "plans" / f"{campaign_id}.json", {
            "campaign_id": campaign_id, "site_id": site, "scene_id": site, "capture_id": f"walkthrough-{site[5:]}",
            "source": {"scene_intent_id": "scene-" + "0" * 64}, "plan_digest": state["plan_digest"]})
    base._write(queue / "publications/screening-site-current.json", {"campaign_id": "screening-site-current"})
    _save(queue, {"schema_version": coordinator.LEDGER_SCHEMA, "campaigns": campaigns})
    return {"queue": queue, "work": work, "ids": ids, "archives": archives,
            "dirs": {name: (queue if name == "held" else work) / "attempts" / ids[name] for name in ids}}


def _budgets(queue: Path) -> dict:
    ledger = coordinator._ledger(queue)
    return {site: coordinator._budget(ledger, site) for site in (SITE, OTHER_SITE)}


def test_withdrawal_deletes_finished_screening_attempts_and_archives_and_keeps_the_money(world, screened, tmp_path):
    queue, ids, archives, dirs = screened["queue"], screened["ids"], screened["archives"], screened["dirs"]
    ledger_bytes, budgets = (queue / coordinator.LEDGER_FILENAME).read_bytes(), _budgets(queue)
    assert budgets[SITE]["committed_upper_bound_usd"] == "8.44"  # 0.44 settled + 4 held + 4 reserved
    plan = base._plan(world)
    assert plan["unresolved"] == []
    section = plan["screening"]
    assert section["campaigns"] == ["screening-site-current", "screening-site-superseded"]
    assert {row["attempt_id"]: row["money"] for row in section["attempts"]} == {
        ids["settled"]: "settled", ids["history"]: "released", ids["current"]: "released",
        ids["held"]: "held_at_maximum"}
    assert section["deferred"] == [{"campaign_id": "screening-site-current", "candidate_id": "learned-c",
                                    "attempt_id": ids["active"], "reason": screening.ACTIVE}]
    planned = {row["uri"]: row for row in plan["objects"]}
    for name in ("output", "witness", "held"):
        assert len(planned[archives[name]["uri"]]["versions"]) == 3  # two versions and a delete marker
    assert archives["active"]["uri"] not in planned and archives["other"]["uri"] not in planned

    receipt = base._apply(world, plan)

    for name in ("settled", "history", "held"):
        assert not dirs[name].exists(), name
    assert not any(world["store"].present(archives[name]["uri"]) for name in ("output", "witness", "held"))
    # The active attempt and another site's attempt keep their directories and archives.
    for name in ("active", "other"):
        assert (dirs[name] / STAGING / "provider_output_promotion.v1.json").is_file()
        assert world["store"].present(archives[name]["uri"])
    # Money: no ledger byte changed, the coordinator reads the same budgets and has nothing to settle.
    assert (queue / coordinator.LEDGER_FILENAME).read_bytes() == ledger_bytes
    assert _budgets(queue) == budgets
    assert coordinator._settle_finished(queue) == 0
    assert (queue / coordinator.LEDGER_FILENAME).read_bytes() == ledger_bytes

    report = receipt["deleted"]["beta_screening"]
    keys = sorted(world["resolution"]["object_store"].key(archives[name]["uri"]) for name in ("output", "witness", "held"))
    assert report["archive_object_keys"] == keys
    assert (report["archive_objects_deleted"], report["archive_object_versions_deleted"]) == (3, 9)
    assert (report["attempt_directories_deleted"], report["attempt_directories_already_absent"]) == (3, 1)
    assert report["attempt_files_deleted"] == 5 + 3 + 5 and report["ledger_rows_deleted"] == 0
    assert report["deferred"] == section["deferred"]
    holders = {row["holder"]: row for row in receipt["remaining"]}
    assert holders["beta_screening_attempts_deferred"]["attempts"][0]["attempt_id"] == ids["active"]
    assert holders["beta_screening_staging_store"]["attempt_ids"] == [ids["held"]]
    assert str(queue / coordinator.LEDGER_FILENAME) in holders["beta_screening_money_and_plan_records"]["paths"]
    assert receipt["covered_copies_deleted"] is False and receipt["pipeline_controlled_deletion_complete"] is False
    sealed = (Path(plan["site_root"]) / "website_withdrawal/deletion_receipts" / f"{plan['digest'][7:]}.json").read_bytes()
    assert not any(content in sealed for content in CUSTOMER_BYTES)


def test_a_deferred_attempt_is_deleted_by_the_next_run_once_it_settles(world, screened):
    queue, ids, archives = screened["queue"], screened["ids"], screened["archives"]
    base._apply(world, base._plan(world))
    assert screened["dirs"]["active"].is_dir()
    ledger = _ledger(queue)
    row = ledger["campaigns"]["screening-site-current"]["candidates"]["learned-c"]
    row["status"] = "completed"
    row["attempt"].update(reservation_status="committed", **_settled(ids["active"], "4", "1.31"))
    _save(queue, ledger)
    budgets = _budgets(queue)
    plan = base._plan(world)
    assert plan["unresolved"] == [] and plan["screening"]["deferred"] == []
    receipt = base._apply(world, plan)
    assert not screened["dirs"]["active"].exists() and not world["store"].present(archives["active"]["uri"])
    assert _budgets(queue) == budgets and budgets[SITE]["committed_upper_bound_usd"] == "5.75"
    assert receipt["covered_copies_deleted"] is True
    assert receipt["deleted"]["beta_screening"]["attempt_directories_deleted"] == 1
    assert screened["dirs"]["other"].is_dir() and world["store"].present(archives["other"]["uri"])


def test_a_crash_mid_deletion_finishes_on_the_next_run(world, screened, monkeypatch):
    archives, dirs = screened["archives"], screened["dirs"]
    real = shutil.rmtree

    def crash(path, *args, **kwargs):
        if not Path(path).name.startswith("screen-"):
            return real(path, *args, **kwargs)
        # Part of the first attempt directory goes, its promotion receipt with it, then the process dies.
        (Path(path) / STAGING / "provider_output_promotion.v1.json").unlink(missing_ok=True)
        raise OSError("killed")

    monkeypatch.setattr(screening.shutil, "rmtree", crash)
    with pytest.raises(OSError):
        base._apply(world, base._plan(world))
    # Objects go first and are proven absent before any local record is removed.
    assert not any(world["store"].present(archives[name]["uri"]) for name in ("output", "witness", "held"))
    assert dirs["settled"].is_dir()
    monkeypatch.setattr(screening.shutil, "rmtree", real)
    plan = base._plan(world)
    assert plan["unresolved"] == []
    receipt = base._apply(world, plan)
    for name in ("settled", "history", "held"):
        assert not dirs[name].exists(), name
    assert receipt["deleted"]["beta_screening"]["attempt_directories_deleted"] == 3
    # Nothing is left to do: a third run deletes nothing and still defers the active attempt.
    plan = base._plan(world)
    assert plan["objects"] == [] and all(row["directory"] is None for row in plan["screening"]["attempts"])
    assert [row["attempt_id"] for row in plan["screening"]["deferred"]] == [screened["ids"]["active"]]


@pytest.mark.parametrize("status,update,reason", [
    ("blocked", {"reservation_status": "reserved"}, screening.RESERVATION_OPEN),
    ("completed", {"reservation_status": "committed"}, screening.SETTLEMENT_PENDING),
    ("reconciliation_required", {"reservation_status": "committed"}, screening.ACTIVE),
])
def test_an_attempt_whose_money_is_not_final_is_deferred_not_deleted(world, screened, status, update, reason):
    queue, ids = screened["queue"], screened["ids"]
    ledger = _ledger(queue)
    row = ledger["campaigns"]["screening-site-current"]["candidates"]["learned-a"]
    row["status"] = status
    for field in ("settled_cost_usd", "settlement_basis", "settled_at", "settlement_receipt_digest",
                  "settlement_receipt"):
        row["attempt"].pop(field)
    row["attempt"].update(update)
    _save(queue, ledger)
    plan = base._plan(world)
    assert {"campaign_id": "screening-site-current", "candidate_id": "learned-a", "attempt_id": ids["settled"],
            "reason": reason} in plan["screening"]["deferred"]
    base._apply(world, plan)
    assert (screened["dirs"]["settled"] / STAGING / "provider_output_promotion.v1.json").is_file()
    assert world["store"].present(screened["archives"]["output"]["uri"])
    assert not screened["dirs"]["history"].exists()


@pytest.mark.parametrize("fault,reason", [
    ("unsealed_receipt", "screening_promotion_receipt_unsealed"),
    ("invalid_reference", "screening_archive_reference_invalid"),
    ("unbound_directory", "screening_attempt_unbound"),
    ("unreadable_ledger", "screening_ledger_unreadable"),
    ("missing_ledger", "screening_ledger_missing"),
    ("shared_archive", "object_shared_with_unbound_run"),
    ("symlink", "unsafe_entry"),
])
def test_anything_unresolved_in_screening_fails_closed_before_any_deletion(world, screened, tmp_path, fault, reason):
    receipt_path = screened["dirs"]["settled"] / STAGING / "provider_output_promotion.v1.json"
    if fault == "unsealed_receipt":
        base._write(receipt_path, {**json.loads(receipt_path.read_text()), "durable_reference": screened["archives"]
                                   ["other"]})
    elif fault == "invalid_reference":
        base._write(receipt_path, base._sealed({**json.loads(receipt_path.read_text()), "durable_reference": {
            "uri": "s3://another-bucket/key"}}, "receipt_digest"))
    elif fault == "unbound_directory":
        base._write(screened["work"] / "attempts/screen-unknown/allocator_result.json", {"status": "completed"})
    elif fault == "unreadable_ledger":
        (screened["queue"] / coordinator.LEDGER_FILENAME).write_text("{not json")
    elif fault == "missing_ledger":
        (screened["queue"] / coordinator.LEDGER_FILENAME).unlink()
    elif fault == "shared_archive":
        other = screened["dirs"]["other"] / STAGING / "provider_output_promotion.v1.json"
        base._write(other, _promotion(screened["archives"]["output"]))
    else:
        os.symlink("/etc", screened["dirs"]["history"] / "escape")
    plan = base._plan(world)
    assert any(row["reason"].endswith(reason) for row in plan["unresolved"]), plan["unresolved"]
    before = base._snapshot(tmp_path)
    with pytest.raises(deletion.WithdrawalDeletionError, match="unresolved_copies"):
        base._apply(world, plan)
    assert base._snapshot(tmp_path) == before and world["store"].deleted == []


def test_retained_review_videos_in_the_capture_go_with_the_scene_files(world, screened):
    """Review videos kept beside the capture (``pipeline/beta_screening_review_videos/``) are scene files."""
    videos = world["capture"] / "pipeline/beta_screening_review_videos" / screened["ids"]["settled"]
    base._write(videos / "manifest.json", {"schema_version": "beta_screening_review_videos_manifest.v1"})
    base._write(videos / "cell-00/external.mp4", CUSTOMER_BYTES[1])
    plan = base._plan(world)
    assert plan["scene_files"]["count"] == 5
    base._apply(world, plan)
    assert [path for path in videos.parents[1].rglob("*") if path.is_file()] == []


def test_without_a_screening_queue_the_deletion_says_so(world, screened):
    plan = base._plan(world, screening_queue_root=None)
    assert plan["screening"]["attempts"] == [] and plan["unresolved"] == []
    receipt = base._apply(world, plan, screening_queue_root=None)
    assert "beta_screening_not_scanned" in {row["holder"] for row in receipt["remaining"]}
    assert screened["dirs"]["settled"].is_dir()


def test_default_roots_are_the_screening_unit_roots():
    unit = (Path(__file__).resolve().parents[1] / "deploy/systemd/blueprint-beta-policy-screening.service").read_text()
    for variable, default in ((coordinator.QUEUE_ROOT_ENV, screening.DEFAULT_QUEUE_ROOT),
                              (coordinator.WORK_ROOT_ENV, screening.DEFAULT_WORK_ROOT)):
        if not os.environ.get(variable):
            assert f"Environment={variable}={default}\n" in unit


def test_a_campaign_whose_plan_names_the_site_capture_is_bound_whatever_its_ledger_site(world, screened):
    plans = screened["queue"] / "plans"
    plan_path = plans / "screening-other-site.json"
    base._write(plan_path, {**json.loads(plan_path.read_text()), "capture_id": "walkthrough-req1"})
    plan = base._plan(world)
    assert "screening-other-site" in plan["screening"]["campaigns"]
    base._apply(world, plan)
    assert not screened["dirs"]["other"].exists() and not world["store"].present(screened["archives"]["other"]["uri"])
