"""Replay a failed SAM child's saved job against candidate code: no GPU, no model, exact refusing predicate."""

from __future__ import annotations

from functools import partial
from types import SimpleNamespace

import errno
import hashlib
import json
import os
import shutil
import textwrap
from pathlib import Path

import pytest

from blueprint_pipeline import task_evaluation_sam31_preparation_execution as execution
from blueprint_pipeline import task_evaluation_sam31_preparation_stages as stages
from blueprint_pipeline import task_evaluation_stage_replay as replay
from blueprint_pipeline.decision_evidence_contracts import canonical_digest

CHILD = "sam31-" + "7" * 64
COMMIT = "c" * 40


@pytest.fixture(autouse=True)
def isolated_disk_ledger(tmp_path, monkeypatch):
    ledger = tmp_path / "disk-reservations"
    monkeypatch.setattr(replay, "DEFAULT_RESERVATION_ROOT", ledger)
    # Inject only device observations; reservation, floor and release stay real.
    monkeypatch.setattr(replay, "reserve_control_plane_disk", partial(
        replay.reserve_control_plane_disk,
        disk_usage=lambda _path: SimpleNamespace(total=512 * 2**30, used=128 * 2**30, free=384 * 2**30),
    ))
    yield
    assert not list(ledger.glob("*.json")), "replay leaked a disk reservation"


def test_disk_refusal_precedes_any_replay_scratch_or_handler(tmp_path, monkeypatch):
    from blueprint_pipeline.control_plane_disk_budget import ControlPlaneDiskBudgetError
    def refuse(*args, **kwargs):
        assert args == ("stage_replay",)
        raise ControlPlaneDiskBudgetError("control_plane_disk_budget_exceeded")
    monkeypatch.setattr(replay, "reserve_control_plane_disk", refuse)
    for function in (replay.replay_child, replay.replay_parent):
        with pytest.raises(ControlPlaneDiskBudgetError):
            function(replay_root=tmp_path / "scratch")
    assert not (tmp_path / "scratch").exists()


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


@pytest.mark.parametrize("allow_paid,purpose", [(False, "retained_offline_replay"), (True, "new_execution")])
def test_replay_selects_retained_contract_only_when_paid_execution_is_disabled(tmp_path, monkeypatch, allow_paid, purpose):
    root, _, _ = _queue(tmp_path)
    observed = []
    def validate(job, **kwargs):
        observed.append(kwargs["validation_purpose"])
        return {"expected_production_commit": COMMIT}, {"source_commit": COMMIT}
    monkeypatch.setattr(execution, "_validated_job", validate)
    monkeypatch.setattr(stages, "execute_stage", lambda _: {"status": "completed", "artifacts": {}})
    report = replay.replay_child(queue_root=root, child_id=CHILD, parent_queue_root=tmp_path / "parent",
        input_root=tmp_path / "inputs", replay_root=tmp_path / "replays", approved_roots=(tmp_path,), allow_paid=allow_paid)
    assert report["status"] == "completed"
    assert observed == [purpose]


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


def test_calibration_replay_passes_validated_request_to_cpu_only_continuation(tmp_path, monkeypatch):
    from blueprint_pipeline import source_calibration_finalization_reuse as continuation
    root, job_path, job = _queue(tmp_path)
    monkeypatch.delenv("BLUEPRINT_TASK_EVALUATION_SCENE_PROGRESSION_CONFIG", raising=False)
    job["phase"] = "calibrated_views"
    job["job_digest"] = canonical_digest(job, digest_field="job_digest")
    job_path.write_text(json.dumps(job))
    _fake_validation(monkeypatch)
    monkeypatch.setattr(stages, "execute_stage", lambda *_: pytest.fail("replay entered GPU stage"))
    seen = []
    def cpu_only(**kwargs):
        assert kwargs["job"]["request"]["preparation_id"] == "prep-1"
        assert kwargs["job"]["plan"] == kwargs["plan"]
        assert kwargs["job_path"] == job_path
        assert kwargs["parent_queue_root"] == tmp_path / "parent"
        assert kwargs["input_root"] == tmp_path / "inputs"
        assert Path(kwargs["run_root"]).is_relative_to(tmp_path / "replays")
        seen.append(kwargs)
        return {"status": "completed", "provider_mutation_performed": False, "artifacts": {}}
    monkeypatch.setattr(continuation, "replay_retained_calibration", cpu_only)
    before = job_path.read_bytes()
    report = replay.replay_child(queue_root=root, child_id=CHILD, parent_queue_root=tmp_path / "parent",
        input_root=tmp_path / "inputs", replay_root=tmp_path / "replays", approved_roots=(tmp_path,))
    assert report["status"] == "completed" and len(seen) == 1
    assert report["retained_gpu_return_cpu_finalization_only"] is True
    assert report["paid_execution_requested"] is False and report["provider_mutation_performed"] is False
    assert job_path.read_bytes() == before


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
    assert f"WorkingDirectory={os.getcwd()}" in argv
    assert os.environ.get("BLUEPRINT_ALLOW_VAST_INSTANCE_LAUNCH") is None
    explicit = replay.isolation_command(["python", "-m", "x"], working_directory="/opt/release/src")
    assert "WorkingDirectory=/opt/release/src" in explicit


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
    # A boundary row is not read by the next consumers, so their admission is not replayed for it.
    assert report["next_consumer_admission"] == [] and report["next_consumers_admitted"] is False



def test_envelope_uri_prefixes_come_from_the_request_itself() -> None:
    """Inside an isolated shell the unit's JSON prefix list did not reach the replay; a replay
    fetches nothing, so the envelope's own reference URIs are the only prefixes it needs."""

    envelope = {"request": {
        "construction": {"recipe": {"uri": "s3://blueprint/task-evaluation/production-inputs/adp-x/recipe.json", "digest": "sha256:" + "0" * 64}},
        "scene": {"appearance": {"representation": {"uri": "https://huggingface.co/datasets/spatialverse/InteriorGS/resolve/abc/scene.ply"}}},
        "runtime": {"mounts": [{"source": {"uri": "s3://blueprint-task-evaluation-artifacts-prod/blueprint/x/y.json"}}]},
    }}

    assert replay.envelope_uri_prefixes(envelope) == [
        "https://huggingface.co/datasets/",
        "s3://blueprint-task-evaluation-artifacts-prod/blueprint/",
        "s3://blueprint/task-evaluation/",
    ]
    assert replay.envelope_uri_prefixes({"request": {}}) == []


def test_parent_fetch_boundary_is_not_ready_for_progression(tmp_path: Path):
    request, _ = _sam_request()
    queue = tmp_path / "parent"
    stage_launch_preparation_request(value=request, queue_root=queue, submitted_by="blueprint-webapp")
    report = replay.replay_parent(parent_queue_root=queue, preparation_id=request["preparation_id"],
        child_queue_root=tmp_path / "children", input_root=tmp_path / "empty-inputs",
        replay_root=tmp_path / "replays", allowed_uri_prefixes=PREFIXES, service_account=SERVICE_ACCOUNT)
    assert report["sam31_ready"] is False
    assert report["reached_render_inputs_boundary"] is False


def _materialized_parent(tmp_path: Path) -> tuple[dict, Path, Path, Path]:
    """A parent whose inputs the real worker already materialized into the production store."""

    request, payloads = _sam_request()
    parent_queue, input_root, children = tmp_path / "parent", tmp_path / "prepared-references", tmp_path / "children"
    stage_launch_preparation_request(value=request, queue_root=parent_queue, submitted_by="blueprint-webapp")
    worker.process_launch_preparation_queue(
        queue_root=parent_queue, input_root=input_root, allowed_uri_prefixes=PREFIXES,
        service_account=SERVICE_ACCOUNT, source_commit=request["expected_production_commit"], fetcher=fetcher(payloads),
        sam31_preparation_advancer=lambda context: {"status": "waiting_for_child", "evidence_refs": []},
    )
    return request, parent_queue, input_root, children


def _replay_materialized_parent(tmp_path: Path, request: dict, parent_queue: Path, input_root: Path, children: Path) -> dict:
    return replay.replay_parent(
        parent_queue_root=parent_queue, preparation_id=request["preparation_id"], child_queue_root=children,
        input_root=input_root, replay_root=tmp_path / "replays", allowed_uri_prefixes=PREFIXES,
        service_account=SERVICE_ACCOUNT,
        advancer=lambda context: {"status": "waiting_for_child", "evidence_refs": []},
    )


def _store_on_another_volume(monkeypatch, store: Path) -> None:
    """The host's lookahead: activations on the root disk, the content store on the work volume.

    Linking a store blob into the scratch fails with EXDEV, so the replay copies it; links
    made inside the scratch (the worker's own projections) stay on one device and succeed.
    """

    real_link = os.link

    def link(source, destination, *args, **kwargs):
        if Path(source).parent == store:
            raise OSError(errno.EXDEV, "Invalid cross-device link")
        return real_link(source, destination, *args, **kwargs)

    monkeypatch.setattr(replay.os, "link", link)


def test_parent_replay_releases_its_scratch_inputs_including_linked_materializations(tmp_path: Path, monkeypatch) -> None:
    """2026-09-27: every scene-configuration activation replays its parent under
    ``<activation>/lookahead``. The activations live on the root disk and the content store on
    the work volume, so each replay copied the whole store and nothing removed the copy: 107
    activations held about 13 GiB. The worker also hard-links every blob the preparation
    references into ``prepared-references/<preparation>/``, so the whole scratch input tree is
    released, and each copy's bytes count once however many names it had."""

    request, parent_queue, input_root, children = _materialized_parent(tmp_path)
    store = input_root / "content-addressed" / "sha256"
    # The production store also holds the inputs of every other live preparation.
    other = b"another preparation's input" * 100
    (store / hashlib.sha256(other).hexdigest()).write_bytes(other)
    (store / hashlib.sha256(other).hexdigest()).chmod(0o440)
    blobs = sorted(store.iterdir())
    before = _tree_digest(input_root)
    _store_on_another_volume(monkeypatch, store)
    monkeypatch.setenv(replay.driver.CHILD_QUEUE_ENV, str(children))

    report = _replay_materialized_parent(tmp_path, request, parent_queue, input_root, children)

    run_root = Path(report["report_path"]).parent
    released = report["scratch_inputs_released"]
    assert report["status"] == "waiting_for_child"
    assert report["content_blobs_linked"] == len(blobs)
    assert not (run_root / "prepared-references").exists()
    # Every store blob was copied once; the names the worker linked to a copy free nothing more.
    assert (released["inodes"], released["bytes"]) == (len(blobs), sum(path.stat().st_size for path in blobs))
    assert released["files"] > released["inodes"], "the worker's linked materializations went too"
    saved = json.loads(Path(report["report_path"]).read_text(encoding="utf-8"))
    assert saved["scratch_inputs_released"] == released
    assert _tree_digest(input_root) == before
    assert any(Path(report["scratch_queue_root"]).rglob("*.json"))


def test_parent_replay_releases_scratch_when_the_worker_refuses(tmp_path: Path, monkeypatch) -> None:
    request, parent_queue, input_root, children = _materialized_parent(tmp_path)
    store = input_root / "content-addressed" / "sha256"
    before = _tree_digest(input_root)
    _store_on_another_volume(monkeypatch, store)
    monkeypatch.setenv(replay.driver.CHILD_QUEUE_ENV, str(children))
    seen: list[list[str]] = []

    def refuse(**kwargs):
        seen.append(sorted(path.name for path in (Path(kwargs["input_root"]) / "content-addressed" / "sha256").iterdir()))
        raise RuntimeError("worker refused in fixture")

    monkeypatch.setattr(worker, "process_launch_preparation_queue", refuse)

    report = _replay_materialized_parent(tmp_path, request, parent_queue, input_root, children)

    run_root = Path(report["report_path"]).parent
    assert report["status"] == "worker_refused"
    # Recorded as on the normal path, so storage GC can tell a finished refused replay.
    assert (report["nothing_fetched"], report["fetch_attempts"]) == (True, [])
    assert seen == [sorted(path.name for path in store.iterdir())], "the worker ran against the whole scratch copy"
    assert not (run_root / "prepared-references").exists()
    assert report["scratch_inputs_released"] == {
        "files": len(seen[0]), "inodes": len(seen[0]), "bytes": sum(path.stat().st_size for path in store.iterdir()),
    }
    saved = json.loads(Path(report["report_path"]).read_text(encoding="utf-8"))
    assert (saved["status"], saved["nothing_fetched"], saved["scratch_inputs_released"]) == (
        "worker_refused", True, report["scratch_inputs_released"])
    assert _tree_digest(input_root) == before


def test_parent_replay_releases_a_partial_copy_when_the_scratch_disk_fills(tmp_path: Path, monkeypatch) -> None:
    request, parent_queue, input_root, children = _materialized_parent(tmp_path)
    store = input_root / "content-addressed" / "sha256"
    before = _tree_digest(input_root)
    _store_on_another_volume(monkeypatch, store)
    monkeypatch.setenv(replay.driver.CHILD_QUEUE_ENV, str(children))
    real_copy = shutil.copy2
    copied: list[str] = []

    def copy2(source, destination, *args, **kwargs):
        if Path(source).parent == store and copied:
            raise OSError(errno.ENOSPC, "No space left on device")
        result = real_copy(source, destination, *args, **kwargs)
        if Path(source).parent == store:
            copied.append(Path(source).name)
        return result

    monkeypatch.setattr(replay.shutil, "copy2", copy2)

    with pytest.raises(OSError) as raised:
        _replay_materialized_parent(tmp_path, request, parent_queue, input_root, children)

    assert raised.value.errno == errno.ENOSPC and len(copied) == 1
    [run_root] = (tmp_path / "replays").iterdir()
    assert not (run_root / "prepared-references").exists()
    assert _tree_digest(input_root) == before


def test_a_release_that_fails_is_recorded_and_the_report_still_written(tmp_path: Path, monkeypatch) -> None:
    request, parent_queue, input_root, children = _materialized_parent(tmp_path)
    _store_on_another_volume(monkeypatch, input_root / "content-addressed" / "sha256")
    monkeypatch.setenv(replay.driver.CHILD_QUEUE_ENV, str(children))
    real_rmtree = shutil.rmtree

    def rmtree(path, *args, **kwargs):
        if Path(path).name == "prepared-references":
            raise PermissionError(errno.EACCES, "Permission denied")
        return real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(replay.shutil, "rmtree", rmtree)

    report = _replay_materialized_parent(tmp_path, request, parent_queue, input_root, children)

    refusal = {"files": 0, "inodes": 0, "bytes": 0, "refused": "replay_scratch_inputs_release_failed:PermissionError"}
    assert (report["status"], report["scratch_inputs_released"]) == ("waiting_for_child", refusal)
    assert json.loads(Path(report["report_path"]).read_text(encoding="utf-8"))["scratch_inputs_released"] == refusal
    assert (Path(report["report_path"]).parent / "prepared-references").is_dir(), "nothing is claimed for what stayed"


def test_a_failing_release_never_masks_the_replays_own_error(tmp_path: Path, monkeypatch) -> None:
    request, parent_queue, input_root, children = _materialized_parent(tmp_path)
    store = input_root / "content-addressed" / "sha256"
    _store_on_another_volume(monkeypatch, store)
    monkeypatch.setenv(replay.driver.CHILD_QUEUE_ENV, str(children))
    real_copy, real_is_symlink = shutil.copy2, Path.is_symlink

    def copy2(source, destination, *args, **kwargs):
        if Path(source).parent == store:
            raise OSError(errno.ENOSPC, "No space left on device")
        return real_copy(source, destination, *args, **kwargs)

    def is_symlink(self):
        if self.name == "prepared-references":
            raise PermissionError(errno.EACCES, "Permission denied")
        return real_is_symlink(self)

    monkeypatch.setattr(replay.shutil, "copy2", copy2)
    monkeypatch.setattr(Path, "is_symlink", is_symlink)

    with pytest.raises(OSError) as raised:
        _replay_materialized_parent(tmp_path, request, parent_queue, input_root, children)

    assert raised.value.errno == errno.ENOSPC


def test_scratch_inputs_release_counts_each_inode_once_and_never_follows_a_link(tmp_path: Path) -> None:
    production = tmp_path / "production" / "sha256"
    production.mkdir(parents=True)
    blob = production / ("a" * 64)
    blob.write_bytes(b"production bytes")
    linked_inputs = tmp_path / "linked-run" / "prepared-references"
    linked_inputs.parent.mkdir(parents=True)
    linked_inputs.symlink_to(production.parent, target_is_directory=True)

    assert replay._release_scratch_inputs(linked_inputs) == {
        "files": 0, "inodes": 0, "bytes": 0, "refused": "replay_scratch_inputs_unsafe"}
    assert linked_inputs.is_symlink() and blob.read_bytes() == b"production bytes"

    inputs = tmp_path / "run" / "prepared-references"
    store = inputs / "content-addressed" / "sha256"
    store.mkdir(parents=True)
    copy = store / ("b" * 64)
    copy.write_bytes(b"a scratch copy")
    (inputs / "preparation").mkdir()
    os.link(copy, inputs / "preparation" / copy.name)  # a materialized name: the copy counts once
    os.link(blob, store / blob.name)  # a link shared with the production store frees nothing
    (store / ("c" * 64)).symlink_to(blob)
    (inputs / "linked-directory").symlink_to(production, target_is_directory=True)

    assert replay._release_scratch_inputs(inputs) == {"files": 3, "inodes": 1, "bytes": len(b"a scratch copy")}
    assert not inputs.exists() and blob.read_bytes() == b"production bytes" and blob.stat().st_nlink == 1
    assert replay._release_scratch_inputs(inputs) == {"files": 0, "inodes": 0, "bytes": 0}


def _queued_preparation(tmp_path: Path) -> tuple[Path, Path]:
    """A parent staged by the real producer, materialized, with a queued result — as the host holds it."""

    request, _payloads = production_request_with_fetchable_bytes()
    queue = tmp_path / "preparations"
    receipt = stage_launch_preparation_request(value=request, queue_root=queue, submitted_by="blueprint-webapp-intake")
    pending = Path(str(receipt["queue_path"]))
    (queue / "materialized").mkdir(exist_ok=True)
    pending.rename(queue / "materialized" / pending.name)
    result = {
        "schema_version": "task_evaluation_launch_preparation_result.v1",
        "status": "queued_for_production_scene_configuration",
        "preparation_id": request["preparation_id"], "run_id": request["run_id"], "run_mode": request["run_mode"],
        "team_namespace": request["team_namespace"], "source_commit": request["expected_production_commit"],
        "provider_mutation_performed": False, "paid_execution_requested": False, "references": [], "result_digest": "",
    }
    result["result_digest"] = canonical_digest(result, digest_field="result_digest")
    (queue / "results").mkdir(exist_ok=True)
    result_path = queue / "results" / pending.name
    result_path.write_text(json.dumps(result, sort_keys=True) + "\n")
    return queue, result_path


def test_next_consumer_replay_admits_a_queued_parent_and_names_a_consumer_that_would_refuse(tmp_path: Path) -> None:
    """The look-ahead runs the NEXT workers' own validators on the replayed result and envelope."""

    queue, result_path = _queued_preparation(tmp_path)

    rows = {row["consumer"]: row for row in replay.replay_next_consumers(result_path=result_path, queue_root=queue)}

    assert rows["task_evaluation_scene_configuration_activation_automation"] == {
        "consumer": "task_evaluation_scene_configuration_activation_automation", "status": "accepted",
    }
    controls = rows["task_evaluation_configured_controls_continuation_provisioning"]
    assert controls["status"] == "refused"  # a bare test request carries none of the controls templates
    assert controls["blocker"].startswith("configured_controls_provisioning_")

    # 2026-09-06: the deployed activation automation compared the envelope schema against a name no
    # producer writes; the replay names that predicate before any paid stage has run.
    envelope_path = queue / "materialized" / result_path.name
    envelope = json.loads(envelope_path.read_text())
    envelope["schema_version"] = "task_evaluation_launch_preparation_intake_envelope.v1"
    envelope["envelope_digest"] = ""
    envelope["envelope_digest"] = canonical_digest(envelope, digest_field="envelope_digest")
    envelope_path.chmod(0o644)  # the producer writes its records read-only
    envelope_path.write_text(json.dumps(envelope, sort_keys=True) + "\n")

    [activation] = [row for row in replay.replay_next_consumers(result_path=result_path, queue_root=queue)
                    if row["consumer"] == "task_evaluation_scene_configuration_activation_automation"]

    assert activation["status"] == "refused"
    assert activation["blocker"] == "scene_configuration_activation_preparation_envelope_invalid"
    assert activation["fired_predicates"] == ["envelope.get('schema_version') != PREPARATION_ENVELOPE_SCHEMA_VERSION"]


def test_unit_environment_is_read_from_systemd_and_execute_gates_are_left_behind() -> None:
    """2026-09-06: an isolated parent replay reported ``sam31_server_profile_missing`` because the
    parent unit's SAM profile binding lives in an EnvironmentFile drop-in the replay never read.
    ``--isolate`` now takes the unit's own EnvironmentFiles and Environment, minus execute gates."""

    shown = (
        "EnvironmentFiles=/etc/blueprint/pipeline-control-plane.env (ignore_errors=yes)\n"
        "EnvironmentFiles=/etc/blueprint/sam31/scene-x.env (ignore_errors=no)\n"
        "Environment=BLUEPRINT_TASK_EVALUATION_CONTROL_PLANE_REPO=/opt/blueprint/x "
        "BLUEPRINT_TASK_EVALUATION_LAUNCH_EXECUTE=true BLUEPRINT_TASK_EVALUATION_LAUNCH_EXECUTE_ID=gate-1 "
        "BLUEPRINT_VAST_SSH_IDENTITY_FILE=/etc/blueprint/provider-secrets/vast_ssh_id_ed25519\n"
    )
    calls: list = []

    def show(unit: str) -> str:
        calls.append(unit)
        return shown

    files, environment = replay.unit_environment("blueprint-task-evaluation-launch-preparation.service", show=show)

    assert calls == ["blueprint-task-evaluation-launch-preparation.service"]
    assert files == ["/etc/blueprint/pipeline-control-plane.env", "/etc/blueprint/sam31/scene-x.env"]
    assert environment == {
        "BLUEPRINT_TASK_EVALUATION_CONTROL_PLANE_REPO": "/opt/blueprint/x",
        "BLUEPRINT_VAST_SSH_IDENTITY_FILE": "/etc/blueprint/provider-secrets/vast_ssh_id_ed25519",
    }
    command = replay.isolation_command(
        ["python", "-m", "x"], user="blueprint", environment_files=files, environment=environment,
        working_directory="/tmp/code/src",
    )
    assert command.count("EnvironmentFile=/etc/blueprint/sam31/scene-x.env") == 1
    assert "--setenv=BLUEPRINT_TASK_EVALUATION_CONTROL_PLANE_REPO=/opt/blueprint/x" in command
    assert not any("LAUNCH_EXECUTE" in item for item in command)
    assert replay.default_unit_for(parent=True) == "blueprint-task-evaluation-launch-preparation.service"
    assert replay.default_unit_for(parent=False) == "blueprint-task-evaluation-sam31-preparation-execution.service"


def test_real_low_space_probe_refuses_without_scratch_or_leaked_reservation(tmp_path, monkeypatch):
    from blueprint_pipeline.control_plane_disk_budget import (
        reserve_control_plane_disk, ControlPlaneDiskBudgetError,
    )
    monkeypatch.setattr(replay, "reserve_control_plane_disk", partial(
        reserve_control_plane_disk,
        disk_usage=lambda _path: SimpleNamespace(total=512 * 2**30, used=504 * 2**30, free=8 * 2**30),
    ))
    with pytest.raises(ControlPlaneDiskBudgetError, match="control_plane_disk_budget_exceeded"):
        replay.replay_child(replay_root=tmp_path / "scratch")
    assert not (tmp_path / "scratch").exists()
    assert not list((tmp_path / "disk-reservations").glob("*.json"))


def test_cli_writes_typed_capacity_refusal_before_starting_saved_stage(tmp_path, monkeypatch):
    from blueprint_pipeline.control_plane_disk_budget import reserve_control_plane_disk
    monkeypatch.setattr(replay, "reserve_control_plane_disk", partial(
        reserve_control_plane_disk,
        disk_usage=lambda _path: SimpleNamespace(total=512 * 2**30, used=504 * 2**30, free=8 * 2**30),
    ))
    output = tmp_path / "report.json"
    assert replay.main(["--child", "sam31-missing", "--queue-root", str(tmp_path / "queue"),
                        "--replay-root", str(tmp_path / "scratch"), "--json-out", str(output)]) == 2
    report = json.loads(output.read_text())
    assert report["blocker"] == "control_plane_disk_budget_exceeded"
    assert report["phase"] == "replay_admission"
    assert report["stage_handler_started"] is False
    assert report["provider_mutation_performed"] is False
    assert not (tmp_path / "scratch").exists()


def test_isolated_cli_pins_the_invoking_code_after_all_environment_files(tmp_path, monkeypatch):
    calls = []
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PYTHONPATH", "/old-checkout/src")
    monkeypatch.setattr(replay.subprocess, "run", lambda command, **kwargs: calls.append(command) or SimpleNamespace(returncode=0))
    assert replay.main(["--child", CHILD, "--isolate", "--no-unit-environment"]) == 0
    command = calls[0]
    executable = command[command.index("--") + 1:]
    root = Path(replay.__file__).resolve().parents[2]
    assert executable[:2] == ["/usr/bin/env", f"PYTHONPATH={root / 'src'}"]
    assert f"WorkingDirectory={root}" in command
