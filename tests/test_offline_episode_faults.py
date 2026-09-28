"""Interrupt the real episode runner at lifecycle notifications using fake Isaac."""
import hashlib
import json

import pytest

from blueprint_pipeline import native_task_arena_policy_canary_worker as worker
from blueprint_pipeline.adp009d_policy_episode import run_policy_episode
from tests.test_native_task_arena_policy_canary_lifecycle_rehearsal import (
    _stage_runtime_root, _rehearsal_runtime, _sealed_result, FakeIsaac,
    PROVIDER_RESULT_FILENAME, CANDIDATE_IDS,
)


@pytest.mark.parametrize("phase,observed,queried,delivered", [
    ("episode_started", False, False, False),
    ("first_observation", True, False, False),
    ("policy_query_started", True, False, False),
    ("policy_response_received", True, True, False),
    ("episode_running", True, True, True),
    ("episode_media_sealed", True, True, True),
])
def test_real_episode_interruption_preserves_reached_truth_and_other_candidate(
    tmp_path, phase, observed, queried, delivered,
):
    runtime_root, output = _stage_runtime_root(tmp_path)
    child = output / "cell_runs" / "03"
    child.mkdir(parents=True)
    isaac = FakeIsaac(child / PROVIDER_RESULT_FILENAME)
    base = _rehearsal_runtime(isaac)
    injected = []
    reached = []
    def runner(**kwargs):
        def interrupt(progress):
            reached.append((kwargs["candidate_id"], progress["phase"]))
            if kwargs["candidate_id"] == CANDIDATE_IDS[0] and progress["phase"] == phase:
                injected.append(phase)
                raise RuntimeError("offline_interruption")
        return run_policy_episode(**kwargs, progress_callback=interrupt)
    runtime = worker.CellRuntime(**{**base.__dict__, "run_policy_episode": runner})
    with pytest.raises(SystemExit):
        worker._run_selected_cell(3, runtime_root=runtime_root, output_root=child,
            provider_output_root=output, cell_runtime=runtime)
    result = _sealed_result(child / PROVIDER_RESULT_FILENAME)
    assert injected == [phase]
    failed, healthy = result["episodes"]
    assert failed["status"] == "blocked"
    assert failed["candidate_policy_queried"] is queried
    assert failed["actions_reached_robot"] is delivered
    assert healthy["status"] == "completed"
    assert healthy["candidate_policy_queried"] is True
    assert isaac.launches == isaac.builds == isaac.environment_closes == 1
    assert isaac.result_sealed_at_close is True
    suffix = "failure_evidence" if observed else "failure_gap"
    gaps = list((child / "episodes").glob(f"*.{suffix}.json"))
    assert len(gaps) == 1
    evidence = json.loads(gaps[0].read_text())
    assert evidence["first_observation_retained"] is observed
    attempted = queried or phase == "policy_query_started"
    assert evidence["candidate_policy_query_attempted"] is attempted
    assert evidence["policy_response_status"] == (
        "received" if queried else "unproven" if attempted else "not_attempted"
    )
    assert ("policy_query_receipt" in evidence["evidence_artifacts"]) is attempted
    assert evidence["episode"]["score"]["status"] == "not_scored"
    assert evidence["episode_failure_stage"] == ("after_first_observation" if observed else "before_first_observation")
    if not observed:
        assert not evidence["candidate_policy_action_queries"]
        assert not evidence["commanded_actions"]
    for row in evidence["evidence_artifacts"].values():
        if row:
            path = child / row["relative_path"]
            assert path.stat().st_size == row["size_bytes"]
            assert "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"]


@pytest.mark.parametrize("fault", ["missing_manifest", "changed_manifest", "missing_video", "missing_frame", "unlisted_missing_frame", "rebound_missing_terminal"])
def test_worker_does_not_reseal_incomplete_episode_media_as_completed(tmp_path, fault):
    runtime_root, output = _stage_runtime_root(tmp_path)
    child = output / "cell_runs" / "03"
    child.mkdir(parents=True)
    isaac = FakeIsaac(child / PROVIDER_RESULT_FILENAME)
    base = _rehearsal_runtime(isaac)
    removed = []
    def runner(**kwargs):
        episode = run_policy_episode(**kwargs)
        if kwargs["candidate_id"] != CANDIDATE_IDS[0]:
            return episode
        role = "video" if fault == "missing_video" else "frame_manifest" if fault not in {"missing_frame", "unlisted_missing_frame"} else "policy_input_camera_frame"
        row = next(row for row in episode["media_artifacts"] if role in row["role"])
        path = kwargs["media_output_dir"] / row["relative_path"]
        removed.append((path, row["sha256"]))
        if fault == "changed_manifest":
            path.write_bytes(b"corrupt manifest")
        elif fault == "rebound_missing_terminal":
            from blueprint_pipeline.decision_evidence_contracts import canonical_digest
            manifest = json.loads(path.read_text())
            manifest["terminal_observation"] = None
            manifest["frame_manifest_digest"] = canonical_digest(manifest, digest_field="frame_manifest_digest")
            path.write_text(json.dumps(manifest))
            row["sha256"] = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
            row["size_bytes"] = path.stat().st_size
        else:
            path.unlink()
            if fault == "unlisted_missing_frame":
                episode["media_artifacts"].remove(row)
        return episode
    runtime = worker.CellRuntime(**{**base.__dict__, "run_policy_episode": runner})
    with pytest.raises(SystemExit):
        worker._run_selected_cell(3, runtime_root=runtime_root, output_root=child,
            provider_output_root=output, cell_runtime=runtime)
    result = _sealed_result(child / PROVIDER_RESULT_FILENAME)
    failed, healthy = result["episodes"]
    assert len(removed) == 1
    evidence_path = next((child / "episodes").glob("*.failure_evidence.json"))
    evidence = json.loads(evidence_path.read_text())
    assert evidence["visual_evidence"]["media_gap"]["type"] == "after_first_observation_media_integrity_failed"
    assert evidence["episode"]["score"]["status"] == "not_scored"
    assert failed["status"] == "blocked"
    assert failed["candidate_policy_queried"] is True
    assert failed["actions_reached_robot"] is True
    assert healthy["status"] == "completed"
    assert isaac.result_sealed_at_close is True
