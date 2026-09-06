"""Replay a failed SAM child's saved job against candidate code: no GPU, no model, exact refusing predicate."""

from __future__ import annotations

import hashlib
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


def test_the_input_root_comes_from_the_job_not_from_a_guess(tmp_path: Path) -> None:
    """The first host run defaulted to task-evaluation-inputs; the worker's store is
    one level down, under prepared-references, so every blob looked missing."""

    inputs = tmp_path / "prepared-references"
    (inputs / "content-addressed" / "sha256").mkdir(parents=True)
    plan = inputs / "prep-1" / ("d" * 64)
    plan.parent.mkdir()
    plan.write_text("{}", encoding="utf-8")
    job = {"plan_ref": {"path": str(plan)}}

    assert replay.discover_input_root(job) == inputs
    assert replay.discover_input_root({"plan_ref": {"path": str(tmp_path / "elsewhere" / "plan")}}) is None
    assert replay.DEFAULT_INPUT_ROOT.name == "prepared-references"


def test_replay_falls_back_to_the_job_input_root_when_the_given_one_has_no_store(tmp_path: Path, monkeypatch) -> None:
    root, _, job = _queue(tmp_path, state="completed")
    inputs = tmp_path / "prepared-references"
    (inputs / "content-addressed" / "sha256").mkdir(parents=True)
    plan = inputs / "prep-1" / ("d" * 64)
    plan.parent.mkdir()
    plan.write_text("{}", encoding="utf-8")
    job["plan_ref"]["path"] = str(plan)
    job["job_digest"] = canonical_digest(job, digest_field="job_digest")
    (root / "completed" / f"{CHILD}.json").write_text(json.dumps(job), encoding="utf-8")
    seen: dict = {}

    def validated(job, **kwargs):
        seen["input_root"] = kwargs["input_root"]
        return {"expected_production_commit": COMMIT}, {"source_commit": COMMIT}

    monkeypatch.setattr(execution, "_validated_job", validated)
    monkeypatch.setattr(stages, "execute_stage", lambda job: {"status": "completed", "artifacts": {}})

    report = replay.replay_child(
        queue_root=root, child_id=CHILD, parent_queue_root=tmp_path / "parent", input_root=tmp_path / "wrong",
        replay_root=tmp_path / "replays", approved_roots=(tmp_path,),
    )

    assert report["status"] == "completed"
    assert seen["input_root"] == inputs



# --------------------------------------------------------------------------- #
# Parent-level replay: the whole parent worker pass on a scratch queue
# --------------------------------------------------------------------------- #

from blueprint_pipeline import task_evaluation_launch_preparation_worker as worker  # noqa: E402
from blueprint_pipeline.task_evaluation_launch_preparation_queue import stage_launch_preparation_request  # noqa: E402
from tests.test_task_evaluation_launch_preparation_worker import (  # noqa: E402
    SERVICE_ACCOUNT, _rebind_recipe, fetcher, production_request_with_fetchable_bytes,
)

PREFIXES = ["s3://blueprint-production-inputs/"]


def _sam_request():
    """A production request whose stage one routes to the SAM advancement (as in the execution tests)."""
    import hashlib as _hashlib
    request, payloads = production_request_with_fetchable_bytes()
    plan_data = json.dumps({"schema_version": "test_plan.v1", "reviewer_kind": "ai"}).encode()
    plan_uri = "s3://blueprint-production-inputs/plan.json"
    plan_ref = {"uri": plan_uri, "digest": "sha256:" + _hashlib.sha256(plan_data).hexdigest(), "size_bytes": len(plan_data)}
    payloads[plan_uri] = plan_data
    request["runtime"]["mounts"].append({"source": plan_ref, "container_path": "/inputs/sam31-plan.json", "mode": "read_only"})
    recipe = json.loads(payloads[request["construction"]["recipe"]["uri"]])
    stage_ref = recipe["stage_sequence"][0]["configuration"]
    stage = {"required_views": {"mask_source": "sam31_reviewed_calibrated_object_masks"},
             "sam31_review_kind": "ai", "sam31_preparation_plan": plan_ref}
    data = json.dumps(stage).encode()
    payloads[stage_ref["uri"]] = data
    stage_ref.update(digest="sha256:" + _hashlib.sha256(data).hexdigest(), size_bytes=len(data))
    _rebind_recipe(request, payloads, recipe)
    return request, payloads


def _tree_digest(root: Path) -> dict:
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.rglob("*")) if p.is_file()}


def test_parent_replay_reruns_the_worker_on_a_scratch_queue_and_leaves_production_untouched(tmp_path: Path, monkeypatch) -> None:
    """2026-09-05 23:42Z: the parent refused its first ``ready`` advancement after every GPU
    stage had passed; the stage replay covered children only.  This replays the parent
    worker itself: envelope and progress copied to a scratch queue, child results copied,
    the content store reused read-only, nothing fetched, the render step a boundary."""

    request, payloads = _sam_request()
    parent_queue, input_root, children = tmp_path / "parent", tmp_path / "prepared-references", tmp_path / "children"
    stage_launch_preparation_request(value=request, queue_root=parent_queue, submitted_by="blueprint-webapp")
    worker.process_launch_preparation_queue(
        queue_root=parent_queue, input_root=input_root, allowed_uri_prefixes=PREFIXES,
        service_account=SERVICE_ACCOUNT, source_commit=request["expected_production_commit"], fetcher=fetcher(payloads),
        sam31_preparation_advancer=lambda context: {"status": "waiting_for_child", "evidence_refs": []},
    )
    before = _tree_digest(parent_queue), _tree_digest(input_root)
    monkeypatch.setenv(replay.driver.CHILD_QUEUE_ENV, str(children))

    report = replay.replay_parent(
        parent_queue_root=parent_queue, preparation_id=request["preparation_id"], child_queue_root=children,
        input_root=input_root, replay_root=tmp_path / "replays", allowed_uri_prefixes=PREFIXES,
        service_account=SERVICE_ACCOUNT,
        advancer=lambda context: {"status": "waiting_for_child", "evidence_refs": []},
    )

    assert report["status"] == "waiting_for_child"
    assert report["row"]["blockers"] in (None, [])
    assert report["nothing_fetched"] is True
    assert (before[0], before[1]) == (_tree_digest(parent_queue), _tree_digest(input_root))
    scratch = Path(report["scratch_queue_root"])
    assert scratch.is_relative_to(tmp_path / "replays") and any(scratch.rglob("*.json"))
    assert Path(report["report_path"]).is_file()


def test_parent_replay_reaches_the_render_boundary_after_a_ready_advancement(tmp_path: Path, monkeypatch) -> None:
    request, payloads = _sam_request()
    parent_queue, input_root, children = tmp_path / "parent", tmp_path / "prepared-references", tmp_path / "children"
    stage_launch_preparation_request(value=request, queue_root=parent_queue, submitted_by="blueprint-webapp")
    worker.process_launch_preparation_queue(
        queue_root=parent_queue, input_root=input_root, allowed_uri_prefixes=PREFIXES,
        service_account=SERVICE_ACCOUNT, source_commit=request["expected_production_commit"], fetcher=fetcher(payloads),
        sam31_preparation_advancer=lambda context: {"status": "waiting_for_child", "evidence_refs": []},
    )
    monkeypatch.setenv(replay.driver.CHILD_QUEUE_ENV, str(children))

    def ready(context):
        out = Path(context["output_root"]) / "evidence"
        refs = []
        for name in ("a", "b", "c", "d", "e"):
            path = out / f"{name}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"name": name}), encoding="utf-8")
            refs.append({"path": str(path), "sha256": "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(), "size_bytes": path.stat().st_size})
        return {"status": "ready", "evidence_refs": refs, "sam31_exact_mask_inputs": {r["path"]: r for r in refs},
                "sam31_preparation_result": {"status": "exact_mask_inputs_ready"}, "human_review_required": False, "candidate_policy_queried": False}

    report = replay.replay_parent(
        parent_queue_root=parent_queue, preparation_id=request["preparation_id"], child_queue_root=children,
        input_root=input_root, replay_root=tmp_path / "replays", allowed_uri_prefixes=PREFIXES,
        service_account=SERVICE_ACCOUNT, advancer=ready,
    )

    assert report["sam31_ready"] is True
    assert report["reached_render_inputs_boundary"] is True
    assert report["status"] == "blocked" and "ReplayBoundary" in ";".join(report["row"]["blockers"])
