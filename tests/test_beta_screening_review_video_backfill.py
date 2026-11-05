# Covers (for impacted-test selection):
#   src/blueprint_pipeline/beta_screening_review_video_backfill.py
#   src/blueprint_pipeline/beta_policy_screening_worker.py
#   src/blueprint_pipeline/beta_policy_screening_coordinator.py
"""Every run that used a machine keeps its review videos; one settled before that gets one retention pass."""
from __future__ import annotations

import json
import shutil

import pytest

from blueprint_pipeline import beta_policy_screening_coordinator as coordinator
from blueprint_pipeline import beta_screening_review_video_backfill as backfill
from blueprint_pipeline import beta_screening_review_videos as videos
from tests import test_beta_screening_review_videos as review_videos
from tests.test_beta_policy_screening_coordinator import Backend, native_result, publication

PLENTY, RUN = review_videos.PLENTY, review_videos.RUN
world = review_videos.world  # the review videos' fixture: a plan, its attempt's archive and a capture


class _Prepared(Backend):
    def prepare_candidate(self, **kwargs):
        self.preparations += 1
        return {"sealed": "local-inputs", "run_id": RUN}


def _ledger(root):
    return json.loads((root / coordinator.LEDGER_FILENAME).read_text())


def _row(root, world):
    state = _ledger(root)["campaigns"][world.plan["campaign_id"]]
    return state, state["candidates"][world.candidate["candidate_id"]]


def _settled_before_retention(root, world):
    """A completed run reconciled without retention, whose attempt still holds its run's archive."""
    coordinator.enqueue_screening_campaign(world.plan, queue_root=root)
    observation = {"status": "completed", "provider_allocation_performed": True, "provider_zero_verified": True,
                   "result": native_result(world.plan)}
    coordinator.process_screening_queue(require_publication_ack=False, queue_root=root,
                                        executor=_Prepared(observation))
    _, row = _row(root, world)
    assert row["status"] == "completed" and "review_videos" not in row["summary"]
    attempt_dir = coordinator._attempt_path(root, row["attempt"])
    lane = attempt_dir / "allocator/attempts/attempt_001"
    (lane / videos.LOCAL_ARCHIVE).parent.mkdir(parents=True)
    shutil.copyfile(world.lane / videos.LOCAL_ARCHIVE, lane / videos.LOCAL_ARCHIVE)
    (attempt_dir / "allocator_result.json").write_text(json.dumps({"attempt_root": str(lane)}))
    return attempt_dir


def _retain(world):
    def retain(**arguments):
        return videos.retain(**arguments, capture_root=world.capture, disk_usage=lambda _path: PLENTY)
    return retain


def test_a_run_settled_before_retention_gets_its_videos_and_a_publication_that_lists_them(tmp_path, world):
    root = tmp_path / "queue"
    attempt_dir = _settled_before_retention(root, world)
    before_state, before_row = _row(root, world)
    before = publication(root)

    result = backfill.backfill_review_videos(queue_root=root, retain=_retain(world))

    [outcome] = result["attempts"]
    assert outcome["status"] == "retained" and outcome["attempt_id"] == attempt_dir.name
    state, row = _row(root, world)
    reference = row["summary"]["review_videos"]
    assert reference["status"] == "retained" and len(reference["videos"]) == 18
    assert (world.capture / "pipeline" / videos.DIRNAME / attempt_dir.name / videos.MANIFEST_NAME).is_file()
    # Only the reference is new: the result, the attempt and its money are the same row's.
    assert {key: value for key, value in row["summary"].items() if key != "review_videos"} == before_row["summary"]
    assert row["attempt"] == before_row["attempt"] and row["status"] == before_row["status"]
    assert state["revision"] == before_state["revision"] + 1
    after = publication(root)
    assert after["revision"] == before["revision"] + 1
    assert after["candidates"] == before["candidates"] and after["budget"] == before["budget"]
    [entry] = after["review_videos"]
    assert entry["candidate_id"] == world.candidate["candidate_id"]
    assert entry["manifest_digest"] == reference["manifest_digest"]
    # Once per row: the next tick has nothing to do and changes nothing.
    ledger = _ledger(root)
    assert backfill.backfill_review_videos(queue_root=root, retain=_retain(world)) == {
        "status": "processed", "attempts": []}
    assert _ledger(root) == ledger


def test_a_run_whose_records_are_gone_is_recorded_once_and_its_publication_is_unchanged(tmp_path, world):
    root = tmp_path / "queue"
    attempt_dir = _settled_before_retention(root, world)
    shutil.rmtree(attempt_dir)
    before_state, _ = _row(root, world)
    before = publication(root)
    calls = []

    result = backfill.backfill_review_videos(queue_root=root, retain=lambda **kwargs: calls.append(kwargs))

    assert result["attempts"][0]["status"] == "not_retained" and not calls
    state, row = _row(root, world)
    assert row["summary"]["review_videos"] == videos.not_retained(backfill.ATTEMPT_UNAVAILABLE)
    assert state["revision"] == before_state["revision"]
    assert publication(root)["publication_digest"] == before["publication_digest"]
    assert backfill.backfill_review_videos(queue_root=root)["attempts"] == []


def test_a_withdrawn_capture_is_never_copied(tmp_path, world):
    root = tmp_path / "queue"
    attempt_dir = _settled_before_retention(root, world)
    review_videos._withdraw(world.capture)

    result = backfill.backfill_review_videos(queue_root=root, retain=_retain(world))

    assert result["attempts"][0]["status"] == "not_retained"
    _, row = _row(root, world)
    assert row["summary"]["review_videos"]["status"] == "not_retained"
    assert not (world.capture / "pipeline" / videos.DIRNAME / attempt_dir.name).exists()


def test_a_row_changed_while_its_videos_were_copied_is_never_overwritten(tmp_path, world):
    root = tmp_path / "queue"
    _settled_before_retention(root, world)
    keep = _retain(world)
    recorded = videos.not_retained("recorded_meanwhile")

    def retain(**arguments):
        reference = keep(**arguments)
        with coordinator._lock(root):  # the reconcile path recorded its own reference meanwhile
            ledger = coordinator._ledger(root)
            state = ledger["campaigns"][world.plan["campaign_id"]]
            row = state["candidates"][world.candidate["candidate_id"]]
            row["summary"] = {**row["summary"], "review_videos": recorded}
            coordinator._save(root, ledger)
        return reference

    result = backfill.backfill_review_videos(queue_root=root, retain=retain)

    assert result["attempts"][0]["status"] == "changed"
    assert _row(root, world)[1]["summary"]["review_videos"] == recorded


def test_rows_are_taken_a_bounded_number_per_call(tmp_path, world, monkeypatch):
    root = tmp_path / "queue"
    _settled_before_retention(root, world)
    ledger = coordinator._ledger(coordinator._root(root))
    due = backfill._due(ledger, 5)
    assert len(due) == 1
    ledger["campaigns"]["campaign-0"] = json.loads(json.dumps(ledger["campaigns"][world.plan["campaign_id"]]))
    assert [item[0] for item in backfill._due(ledger, 5)] == ["campaign-0", world.plan["campaign_id"]]
    assert len(backfill._due(ledger, 1)) == 1


@pytest.mark.parametrize("fault", ["ledger", "retain"])
def test_the_backfill_never_raises(tmp_path, world, monkeypatch, fault):
    root = tmp_path / "queue"
    _settled_before_retention(root, world)
    if fault == "ledger":
        monkeypatch.setattr(coordinator, "_ledger", lambda _root: (_ for _ in ()).throw(OSError("disk")))
        assert backfill.backfill_review_videos(queue_root=root)["status"] == "failed"
    else:
        result = backfill.backfill_review_videos(queue_root=root, retain=lambda **kwargs: {"garbage": True})
        assert result["attempts"][0]["status"] == "not_retained"
        assert _row(root, world)[1]["summary"]["review_videos"] == videos.not_retained(
            "beta_screening_review_videos_failed")


def _blocked(root, world, review_videos=None):
    coordinator.enqueue_screening_campaign(world.plan, queue_root=root)
    observation = {"status": "blocked", "provider_allocation_performed": True, "provider_zero_verified": True,
                   "no_allocation_proven": False, "blockers": ["beta_screening_native_result_missing_or_invalid"],
                   **({"review_videos": review_videos} if review_videos is not None else {})}
    coordinator.process_screening_queue(require_publication_ack=False, queue_root=root,
                                        executor=_Prepared(observation))
    return _row(root, world)


def test_a_spent_run_without_a_result_keeps_its_videos_on_its_row_and_never_publishes_them(tmp_path, world):
    reference = videos.retain(plan=world.plan, candidate=world.candidate, prepared={"run_id": RUN},
                              adapter={"attempt_root": str(world.lane)}, attempt_dir=world.attempt,
                              capture_root=world.capture, disk_usage=lambda _path: PLENTY)
    _, plain = _blocked(tmp_path / "plain", world)
    plain_publication = publication(tmp_path / "plain")
    state, row = _blocked(tmp_path / "kept", world, reference)
    assert row["status"] == "blocked" and row["attempt"]["reservation_status"] == "committed"
    assert row["review_videos"]["manifest_digest"] == reference["manifest_digest"]
    # The verdict, its blockers and its money are those of the same run without videos.
    assert (row["status"], row["blockers"]) == (plain["status"], plain["blockers"])
    assert row["attempt"]["reservation_status"] == plain["attempt"]["reservation_status"]
    published = publication(tmp_path / "kept")
    assert "review_videos" not in published and published["candidates"] == plain_publication["candidates"]


def test_a_spent_run_without_a_result_settled_earlier_gets_one_pass_without_a_new_revision(tmp_path, world):
    root = tmp_path / "queue"
    before_state, row = _blocked(root, world)
    assert "review_videos" not in row
    attempt_dir = coordinator._attempt_path(root, row["attempt"])
    lane = attempt_dir / "allocator/attempts/attempt_001"
    (lane / videos.LOCAL_ARCHIVE).parent.mkdir(parents=True)
    shutil.copyfile(world.lane / videos.LOCAL_ARCHIVE, lane / videos.LOCAL_ARCHIVE)
    (attempt_dir / "allocator_result.json").write_text(json.dumps({"attempt_root": str(lane)}))
    before = publication(root)

    [outcome] = backfill.backfill_review_videos(queue_root=root, retain=_retain(world))["attempts"]

    assert outcome["status"] == "retained" and outcome["row_status"] == "blocked"
    state, row = _row(root, world)
    assert row["review_videos"]["status"] == "retained" and state["revision"] == before_state["revision"]
    assert publication(root)["publication_digest"] == before["publication_digest"]
    assert backfill.backfill_review_videos(queue_root=root)["attempts"] == []


def test_a_run_that_never_got_a_machine_is_not_due(tmp_path, world):
    root = tmp_path / "queue"
    coordinator.enqueue_screening_campaign(world.plan, queue_root=root)
    refused = {"status": "blocked", "provider_allocation_performed": False, "provider_zero_verified": True,
               "no_allocation_proven": True, "blockers": ["vast_no_offer"]}
    coordinator.process_screening_queue(require_publication_ack=False, queue_root=root, executor=_Prepared(refused))
    assert _row(root, world)[1]["attempt"]["reservation_status"] != "committed"
    assert backfill._due(coordinator._ledger(coordinator._root(root)), 5) == []
