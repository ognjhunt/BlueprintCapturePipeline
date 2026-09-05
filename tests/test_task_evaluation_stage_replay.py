"""Replay a failed SAM child's saved job against candidate code: no GPU, no model, exact refusing predicate."""

from __future__ import annotations

import json
import os
import textwrap
from pathlib import Path

import pytest

from blueprint_pipeline import task_evaluation_sam31_preparation_execution as execution
from blueprint_pipeline import task_evaluation_sam31_preparation_stages as stages
from blueprint_pipeline import task_evaluation_stage_replay as replay
from blueprint_pipeline.decision_evidence_contracts import canonical_digest

CHILD = "sam31-" + "7" * 64
COMMIT = "c" * 40


def _queue(tmp_path: Path, *, state: str = "failed") -> tuple[Path, Path, dict]:
    root = tmp_path / "children"
    for name in ("pending", "processing", "waiting_external", "completed", "failed", "results"):
        (root / name).mkdir(parents=True)
    job = {
        "schema_version": execution.JOB_SCHEMA, "child_id": CHILD, "parent_preparation_id": "prep-1",
        "parent_request_digest": "sha256:" + "1" * 64, "plan_digest": "sha256:" + "2" * 64, "phase": "sam31_review",
        "inputs_digest": "sha256:" + "3" * 64, "expected_source_commit": COMMIT,
        "plan_ref": {"path": str(tmp_path / "plan.json"), "sha256": "sha256:" + "2" * 64, "size_bytes": 2},
        "inputs": {"task_selection": {"path": str(tmp_path / "sel.json"), "sha256": "sha256:" + "4" * 64, "size_bytes": 2}},
    }
    job["job_digest"] = canonical_digest(job, digest_field="job_digest")
    (root / state / f"{CHILD}.json").write_text(json.dumps(job), encoding="utf-8")
    result = {"schema_version": execution.RESULT_SCHEMA, "child_id": CHILD, "phase": "sam31_review", "status": "failed",
              "blocker": "sam31_preparation_review_stage_failed:Sam31TrackSelectionReviewError", "artifacts": {}}
    (root / "results" / f"{CHILD}.json").write_text(json.dumps(result), encoding="utf-8")
    return root, root / state / f"{CHILD}.json", job


def _fake_validation(monkeypatch) -> None:
    monkeypatch.setattr(
        execution, "_validated_job",
        lambda job, **kwargs: ({"expected_production_commit": COMMIT, "preparation_id": "prep-1"}, {"source_commit": COMMIT}),
    )


def test_locate_child_finds_the_saved_job_in_any_terminal_state(tmp_path: Path) -> None:
    for state in ("failed", "completed", "waiting_external"):
        root, job_path, _ = _queue(tmp_path / state, state=state)
        located = replay.locate_child(root, CHILD)
        assert located.job_path == job_path
        assert located.result_path == root / "results" / f"{CHILD}.json"
        assert located.state == state
    with pytest.raises(FileNotFoundError):
        replay.locate_child(tmp_path / "failed" / "children", "sam31-" + "0" * 64)


def test_replay_runs_the_saved_job_in_a_fresh_root_and_names_the_refusing_predicate(tmp_path: Path, monkeypatch) -> None:
    """R13 (2026-09-05) failed in sam31_review while assembling 16 frames from 9
    detections; the fix was verified by an ad hoc script scp'd to the host.  This
    is that verification as a command: the saved job, candidate code, a scratch
    output root, the refusing predicate by name, and nothing paid."""

    root, job_path, job = _queue(tmp_path)
    _fake_validation(monkeypatch)
    seen: dict = {}
    source = textwrap.dedent(
        """
        def execute_stage(job):
            seen.update(job)
            frames = job["inputs"]
            detected = 9
            if (
                len(frames) != 1
                or detected != 16
            ):
                raise ValueError("sam31_review_camera_frame_set_invalid")
            return {"status": "completed", "artifacts": {}}
        """
    )
    namespace: dict = {"seen": seen}
    fixture = tmp_path / "candidate_stage.py"
    fixture.write_text(source, encoding="utf-8")
    exec(compile(source, str(fixture), "exec"), namespace)  # noqa: S102 - test fixture stage
    monkeypatch.setattr(stages, "execute_stage", namespace["execute_stage"])
    before = job_path.read_bytes()

    report = replay.replay_child(
        queue_root=root, child_id=CHILD, parent_queue_root=tmp_path / "parent", input_root=tmp_path / "inputs",
        replay_root=tmp_path / "replays", approved_roots=(tmp_path,),
    )

    assert report["status"] == "refused"
    assert report["blocker"] == "sam31_review_camera_frame_set_invalid"
    assert report["fired_predicates"] == ["detected != 16"]
    assert report["phase"] == "sam31_review" and report["child_id"] == CHILD
    assert report["saved_result"]["status"] == "failed"
    # The stage saw the saved job with request and plan, a fresh scratch root, and no resume state.
    assert seen["request"]["preparation_id"] == "prep-1" and seen["plan"]["source_commit"] == COMMIT
    assert Path(seen["output_root"]).is_relative_to(tmp_path / "replays") and Path(seen["output_root"]).is_dir()
    assert seen["resume_only"] is False and seen["previous_progress"] is None
    assert seen["queue_root"] == str(root)
    # Nothing in the queue moved or changed, and the report was written beside the outputs.
    assert job_path.read_bytes() == before and sorted(p.name for p in (root / "failed").iterdir()) == [f"{CHILD}.json"]
    written = json.loads(Path(report["report_path"]).read_text(encoding="utf-8"))
    assert written["blocker"] == report["blocker"]


def test_replay_reports_completion_and_the_outcome(tmp_path: Path, monkeypatch) -> None:
    root, _, _ = _queue(tmp_path, state="completed")
    _fake_validation(monkeypatch)
    monkeypatch.setattr(stages, "execute_stage", lambda job: {"status": "completed", "artifacts": {"x": {"path": "/x"}}})

    report = replay.replay_child(
        queue_root=root, child_id=CHILD, parent_queue_root=tmp_path / "parent", input_root=tmp_path / "inputs",
        replay_root=tmp_path / "replays", approved_roots=(tmp_path,),
    )

    assert report["status"] == "completed"
    assert report["outcome"]["artifacts"] == {"x": {"path": "/x"}}
    assert report["fired_predicates"] == []


def test_a_job_the_current_contract_refuses_is_reported_not_raised(tmp_path: Path, monkeypatch) -> None:
    root, _, _ = _queue(tmp_path)

    def refuse(job, **kwargs):
        if (job.get("phase") == "sam31_review" or job.get("phase") == "other"):
            raise ValueError("job_contract_invalid")
        return {}, {}

    monkeypatch.setattr(execution, "_validated_job", refuse)

    report = replay.replay_child(
        queue_root=root, child_id=CHILD, parent_queue_root=tmp_path / "parent", input_root=tmp_path / "inputs",
        replay_root=tmp_path / "replays", approved_roots=(tmp_path,),
    )

    assert report["status"] == "job_refused"
    assert report["blocker"] == "job_contract_invalid"
    assert report["fired_predicates"] == ["job.get('phase') == 'sam31_review'"]


def test_isolation_command_runs_as_the_service_user_without_network() -> None:
    argv = replay.isolation_command(
        ["/venv/bin/python", "-m", "blueprint_pipeline.task_evaluation_stage_replay", "--child", CHILD],
        user="blueprint", environment_files=["/etc/blueprint/pipeline-control-plane.env"],
        environment={"BLUEPRINT_TASK_EVALUATION_SAM31_PREPARATION_PROFILE_FILE": "/etc/p.json"},
    )
    assert argv[:4] == ["systemd-run", "--wait", "--pipe", "--collect"]
    assert "-p" in argv and "PrivateNetwork=yes" in argv and "User=blueprint" in argv
    assert "EnvironmentFile=/etc/blueprint/pipeline-control-plane.env" in argv
    assert "--setenv=BLUEPRINT_TASK_EVALUATION_SAM31_PREPARATION_PROFILE_FILE=/etc/p.json" in argv
    assert argv[-4:] == ["-m", "blueprint_pipeline.task_evaluation_stage_replay", "--child", CHILD]
    assert os.environ.get("BLUEPRINT_ALLOW_VAST_INSTANCE_LAUNCH") is None
