# Covers (for impacted-test selection):
#   src/blueprint_pipeline/beta_screening_unspent_bundle_prune.py
#   src/blueprint_pipeline/beta_policy_screening_worker.py
"""A closed attempt that provably reached no GPU provider loses its bundle and keeps every receipt."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from blueprint_pipeline import beta_policy_screening_coordinator as coordinator
from blueprint_pipeline import beta_screening_unspent_bundle_prune as prune
from blueprint_pipeline.decision_evidence_contracts import canonical_digest

NO_ALLOCATION = {"status": "blocked", "vast_instance_ids": [], "provider_mutations_performed": 0,
                 "provider_create_attempted": False, "vast_side_effects_may_have_occurred": False,
                 "continuing_spend_from_this_run": False}
KEPT = ("bundle_receipt.json", "prepared.json", "allocator_launch_intent.json", "allocator_result.json")


def _attempt(work: Path, number: int, *, reservation="released", allocated=False, intent=None, result=None,
             bundle=True) -> dict:
    attempt_id = f"screen-{number:032x}"
    directory = work / "attempts" / attempt_id
    directory.mkdir(parents=True)
    (directory / "bundle_receipt.json").write_text('{"bundle_sha256": "sha256:' + "a" * 64 + '"}\n')
    (directory / "prepared.json").write_text("{}\n")
    if intent is not None:
        (directory / "allocator_launch_intent.json").write_text(json.dumps(intent))
    if result is not None:
        (directory / "allocator_result.json").write_text(json.dumps(result))
    if bundle:
        (directory / "bundle").mkdir()
        (directory / "bundle" / "native_g1_beta_provider_bundle.zip").write_bytes(b"x" * 2048)
        (directory / "bundle" / "native_g1_beta_provider_bundle.v1.json").write_text("{}")
    return {"attempt_id": attempt_id, "attempt_dir": str(directory), "maximum_cost_usd": "4",
            "reservation_status": reservation, "provider_allocation_observed": allocated,
            **({"settled_cost_usd": "0.62"} if reservation == "committed" else {})}


def _ledger(queue: Path, rows: dict) -> None:
    queue.mkdir(parents=True, exist_ok=True)
    (queue / "attempts").mkdir(exist_ok=True)
    ledger = {"schema_version": coordinator.LEDGER_SCHEMA, "campaigns": {
        "screening-" + "b" * 32: {"site_id": "site-capture-test", "candidates": rows}}}
    ledger["ledger_digest"] = canonical_digest(ledger, digest_field="ledger_digest")
    (queue / coordinator.LEDGER_FILENAME).write_text(json.dumps(ledger))


@pytest.fixture
def roots(tmp_path, monkeypatch):
    queue, work = tmp_path / "queue", tmp_path / "work"
    work.mkdir()
    monkeypatch.setenv(coordinator.WORK_ROOT_ENV, str(work))
    return queue, work


def test_prunes_only_proven_unspent_closed_attempts_and_keeps_every_receipt(roots):
    queue, work = roots
    refused = _attempt(work, 1, intent={"status": "refused_before_spawn", "blocker": "x"})
    no_offer = _attempt(work, 2, result=NO_ALLOCATION)
    spent = _attempt(work, 3, reservation="committed", allocated=True, result=NO_ALLOCATION)
    observed = _attempt(work, 4, allocated=True, result=NO_ALLOCATION)
    unproven = _attempt(work, 5, result={"status": "blocked", "vast_instance_ids": [123]})
    active = _attempt(work, 6, intent={"status": "refused_before_spawn"})
    (Path(refused["attempt_dir"]) / "bundle.interrupted-0f").mkdir()
    _ledger(queue, {
        "dp": {"status": "blocked", "attempt_history": [no_offer], "attempt": refused},
        "spent": {"status": "completed", "attempt": spent},
        "observed": {"status": "blocked", "attempt": observed},
        "unproven": {"status": "blocked", "attempt": unproven},
        "active": {"status": "running", "attempt": active},
    })

    outcome = prune.prune_unspent_bundles(queue_root=queue)

    assert sorted(row["attempt_id"] for row in outcome["pruned"]) == [refused["attempt_id"], no_offer["attempt_id"]]
    assert {row["proof"] for row in outcome["pruned"]} == {"refused_before_spawn", "allocator_proved_no_allocation"}
    for attempt in (refused, no_offer):
        directory = Path(attempt["attempt_dir"])
        assert not (directory / "bundle").exists() and not list(directory.glob("bundle.interrupted-*"))
        assert all((directory / name).exists() for name in ("bundle_receipt.json", "prepared.json"))
        receipt = json.loads((directory / prune.RECEIPT).read_text())
        assert receipt["schema_version"] == prune.SCHEMA and receipt["removed_bytes"] >= 2048
        assert receipt["bundle_receipt_sha256"].startswith("sha256:") and "yes prune" in receipt["owner_decision"]
    for attempt in (spent, observed, unproven, active):
        directory = Path(attempt["attempt_dir"])
        assert (directory / "bundle/native_g1_beta_provider_bundle.zip").is_file()
        assert not (directory / prune.RECEIPT).exists()
    # A second pass finds nothing left to remove.
    assert prune.prune_unspent_bundles(queue_root=queue)["pruned"] == []


def test_a_symlink_in_a_bundle_refuses_by_name_and_removes_nothing(roots, tmp_path):
    queue, work = roots
    attempt = _attempt(work, 7, intent={"status": "refused_before_spawn"})
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"keep")
    os.symlink(outside, Path(attempt["attempt_dir"]) / "bundle" / "link")
    _ledger(queue, {"dp": {"status": "blocked", "attempt": attempt}})

    outcome = prune.prune_unspent_bundles(queue_root=queue)

    assert outcome["pruned"] == []
    assert outcome["refused"] == [{"attempt_id": attempt["attempt_id"],
                                   "blocker": "beta_screening_bundle_prune_unsafe_entry"}]
    assert outside.read_bytes() == b"keep"
    assert (Path(attempt["attempt_dir"]) / "bundle/native_g1_beta_provider_bundle.zip").is_file()


def test_an_attempt_directory_outside_the_coordinator_layout_is_never_touched(roots, tmp_path):
    queue, work = roots
    attempt = _attempt(work, 8, intent={"status": "refused_before_spawn"})
    elsewhere = tmp_path / "elsewhere" / "attempts" / attempt["attempt_id"]
    elsewhere.parent.mkdir(parents=True)
    Path(attempt["attempt_dir"]).rename(elsewhere)
    attempt["attempt_dir"] = str(elsewhere)
    _ledger(queue, {"dp": {"status": "blocked", "attempt": attempt}})

    assert prune.prune_unspent_bundles(queue_root=queue)["pruned"] == []
    assert (elsewhere / "bundle/native_g1_beta_provider_bundle.zip").is_file()


def test_the_worker_prunes_before_it_launches_anything(monkeypatch, tmp_path):
    from blueprint_pipeline import beta_policy_screening_worker as worker

    order = []
    monkeypatch.setattr(prune, "prune_unspent_bundles", lambda **_: order.append("prune") or {"status": "processed"})
    monkeypatch.setattr(worker, "deliver_publications", lambda **_: order.append("publish") or {})
    monkeypatch.setattr(coordinator, "process_screening_queue", lambda **_: order.append("launch") or {})
    monkeypatch.setattr("blueprint_pipeline.beta_policy_screening_status.write_worker_status", lambda *a, **k: {})
    monkeypatch.setattr(worker, "backfill_runtime_diagnostics", lambda root: {})
    monkeypatch.setattr("blueprint_pipeline.beta_policy_screening_holds.deliver_holds", lambda **_: {})
    monkeypatch.setattr("blueprint_pipeline.website_candidate_selection.retain_selections", lambda **_: {})
    monkeypatch.setattr("blueprint_pipeline.website_candidate_selection.deliver_selections", lambda **_: {})
    monkeypatch.setattr("blueprint_pipeline.beta_screening_review_video_backfill.backfill_review_videos",
                        lambda **_: {})
    (tmp_path / "queue" / "attempts").mkdir(parents=True)

    assert worker.main(["--queue-root", str(tmp_path / "queue")]) == 0
    assert order.index("prune") < order.index("launch")


def _ledger_campaigns(queue: Path, campaigns: dict) -> None:
    queue.mkdir(parents=True, exist_ok=True)
    (queue / "attempts").mkdir(exist_ok=True)
    ledger = {"schema_version": coordinator.LEDGER_SCHEMA, "campaigns": {
        "screening-" + key * 32: {"site_id": "site-capture-test", "candidates": rows} for key, rows in campaigns.items()}}
    ledger["ledger_digest"] = canonical_digest(ledger, digest_field="ledger_digest")
    (queue / coordinator.LEDGER_FILENAME).write_text(json.dumps(ledger))


def _sealed_days_ago(attempt: dict, days: float) -> None:
    receipt = Path(attempt["attempt_dir"]) / "bundle_receipt.json"
    stamp = receipt.stat().st_mtime - days * 86400
    os.utime(receipt, (stamp, stamp))


def _spent(work: Path, number: int, *, days: float = 3.0, **fields) -> dict:
    attempt = _attempt(work, number, reservation="committed", allocated=True, result=NO_ALLOCATION)
    attempt.update(fields)
    outputs = Path(attempt["attempt_dir"]) / "allocator/attempts/attempt_001/vast_provider_run"
    outputs.mkdir(parents=True)
    (outputs / "vast_provider_runtime_output.zip").write_bytes(b"episodes")
    (outputs / "vast_teardown_manifest.json").write_text("{}")
    base = Path(attempt["attempt_dir"]) / "bundle" / "base" / "scene"
    base.mkdir(parents=True)
    (base / "packet.usda").write_bytes(b"y" * 1024)
    _sealed_days_ago(attempt, days)
    return attempt


def test_a_settled_spent_attempt_older_than_two_days_loses_only_its_bundle(roots):
    """Owner decision 2026-11-09 20:50:12Z ("go with option A for the disk cleanup"): the work volume had 5.16 GB
    available against a G1 run's 8.8 GB collection reservation, so the next G1 admission was refused at $0."""
    queue, work = roots
    old = _spent(work, 11)
    recent = _spent(work, 12, days=1.0)
    held = _spent(work, 13, settlement_held={"reason": "beta_screening_settlement_teardown_unproven"})
    held.pop("settled_cost_usd")
    unsettled = _spent(work, 14)
    unsettled.pop("settled_cost_usd")
    live_campaign = _spent(work, 15)
    running = _attempt(work, 16, reservation="reserved", allocated=True)
    _ledger_campaigns(queue, {
        "c": {"dp": {"status": "completed", "attempt": old}, "pi05": {"status": "blocked", "attempt": recent,
                                                                      "attempt_history": [held, unsettled]}},
        "d": {"dp": {"status": "blocked", "attempt_history": [live_campaign]},
              "pi05": {"status": "running", "attempt": running}},
    })

    outcome = prune.prune_unspent_bundles(queue_root=queue)

    assert outcome["pruned"] == [] and outcome["refused"] == []
    assert [row["attempt_id"] for row in outcome["spent_pruned"]] == [old["attempt_id"]]
    assert outcome["removed_bytes"] == outcome["spent_pruned"][0]["removed_bytes"] >= 2048 + 1024
    directory = Path(old["attempt_dir"])
    assert not (directory / "bundle").exists()
    # Every receipt and every output stays.
    assert all((directory / name).exists() for name in ("bundle_receipt.json", "prepared.json", "allocator_result.json"))
    assert (directory / "allocator/attempts/attempt_001/vast_provider_run/vast_provider_runtime_output.zip").read_bytes() \
        == b"episodes"
    receipt = json.loads((directory / prune.RECEIPT).read_text())
    assert receipt["proof"] == "settled_spend" and receipt["settled_cost_usd"] == "0.62"
    assert "option A" in receipt["owner_decision"] and receipt["bundle_receipt_sha256"].startswith("sha256:")
    assert {row["path"] for row in receipt["removed"]} == {"bundle"} and receipt["removed"][0]["files"] == 3
    # Sealed one day ago, held at its maximum, unsettled, or in a campaign with a running row: untouched.
    for attempt in (recent, held, unsettled, live_campaign, running):
        assert (Path(attempt["attempt_dir"]) / "bundle/native_g1_beta_provider_bundle.zip").is_file()
        assert not (Path(attempt["attempt_dir"]) / prune.RECEIPT).exists()
    assert prune.prune_unspent_bundles(queue_root=queue)["spent_pruned"] == []


def test_spent_pruning_is_bounded_per_tick(roots):
    queue, work = roots
    attempts = [_spent(work, 20 + index) for index in range(3)]
    _ledger_campaigns(queue, {"e": {"dp": {"status": "completed", "attempt": attempts[0],
                                           "attempt_history": attempts[1:]}}})
    first = prune.prune_unspent_bundles(queue_root=queue, spent_per_tick=2)
    assert len(first["spent_pruned"]) == 2
    assert len(prune.prune_unspent_bundles(queue_root=queue, spent_per_tick=2)["spent_pruned"]) == 1
    assert prune.SPENT_MIN_AGE_SECONDS == 2 * 86400 and prune.SPENT_PER_TICK >= 27
