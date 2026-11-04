# Covers (for impacted-test selection):
#   src/blueprint_pipeline/website_early_task_hold.py
#   src/blueprint_pipeline/website_atlas_pipeline.py
#   src/blueprint_pipeline/website_capture_preparation.py
#   src/blueprint_pipeline/website_scene_handoff.py
#   src/blueprint_pipeline/website_task_context_contracts.py
#   src/blueprint_pipeline/task_family_native_runtime.py
"""A website task Blueprint cannot evaluate holds right after its Atlas review; nothing is paid.

No native profile (close a door) or no installed runtime (wipe a counter) used to buy
the full Atlas scene and only then hold at the handoff. The review has already read the
whole job, so the hold now comes before any World Labs stage, the capture stage keeps it
without its clean-plate analysis, and the hold resumes once the capability ships. A job
Blueprint can evaluate (capture-447's rigid same-table move) is unchanged. Synthetic only.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from blueprint_pipeline import paid_resource_allocator as allocator
from blueprint_pipeline import website_atlas_pipeline as pipeline
from blueprint_pipeline import website_early_task_hold as early
from blueprint_pipeline.common import write_json
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_family_native_runtime import CATALOG_PATH_ENV
from blueprint_pipeline.website_task_context_contracts import holds_native_task_preparation
from tests.test_website_preparation_status import prepared_source  # noqa: F401  (fixture, by name)

TRAY = "Pick up the wooden tray from the dining table and place it at the other end of the same table."
DOOR = "Close the cabinet door under the sink."
WIPE = "Wipe the kitchen counter next to the sink."
CANNOT_EVALUATE = {
    # A complete reading that no registered profile covers: no native profile.
    "door": (DOOR, "door_closing", [("door_closing", "Close the cabinet door")], early.COVERAGE_GAP),
    # A registered family with no runtime installed on this host.
    "wipe": (WIPE, "wiping", [("wiping", "Wipe the kitchen counter")], early.FAMILY_RUNTIME_GAP),
}


def _context(family, description):
    context = {"schema_version": "website_site_task_context.v1", "request_id": "synthetic-job",
               "scene_id": "site-synthetic-job", "capture_id": "walkthrough-synthetic-job",
               "task_family": family, "description": description, "confirmed": False, "confirmed_at": None}
    context["context_digest"] = canonical_digest(context, digest_field="context_digest")
    return context


def _receipt(context, source, *, semantics=None):
    proposal = {"status": "proposed", **({"task_semantics": semantics} if semantics is not None else {})}
    binding = {"task_context_digest": context["context_digest"], "source_video_digest": source,
               "survey_digest": "sha256:" + "b" * 64, "model": "fake-review-model"}
    return {"status": "completed", "binding": binding, "binding_digest": canonical_digest(binding),
            "proposal": proposal, "proposal_digest": canonical_digest(proposal), "model": "fake-review-model"}


def _semantics(family, steps, *, covers=True, unresolved=()):
    return {"operative_job": "The robot job.", "task_family": family,
            "steps": [{"operation": operation, "description": quote, "task_quote": quote} for operation, quote in steps],
            "covers_entire_job": covers, "unresolved_aspects": list(unresolved), "preparation_notes": []}


@pytest.fixture
def job(tmp_path, monkeypatch):
    """A website job whose Atlas review is retained, at the first scene stage."""
    from blueprint_pipeline import atlas_video_inputs as inputs, website_atlas_pose as pose_api
    monkeypatch.delenv(CATALOG_PATH_ENV, raising=False)
    capture = tmp_path / "walkthrough-synthetic-job"
    root = capture / "pipeline" / "atlas"
    (root / "survey").mkdir(parents=True)
    video = capture / "raw" / "walkthrough.mov"
    video.parent.mkdir()
    video.write_bytes(b"synthetic walkthrough")
    survey = {"schema_version": "atlas_video_survey.v1", "source_video_digest": pipeline._sha256_file(video),
              "display_rotation_degrees": 0, "heldout_pixels_included": False, "frames": []}
    survey["digest"] = canonical_digest(survey, digest_field="digest")
    write_json(root / "survey/atlas_video_survey.json", survey)
    config = {"authority": {"provider_retention": "approved"}, "remaining_cost_usd": 10,
              "policy": {"maximum_review_cost_usd": 1, "pose_input_cost_usd": .01, "rgbd_view_cost_usd": .05}}
    monkeypatch.setattr(pipeline, "load_configuration", lambda *_, **__: config)
    monkeypatch.setattr(pose_api, "atlas_api_config", lambda: ("https://example.test", "non-secret"))
    monkeypatch.setattr(inputs, "prepare_video_inputs", lambda **_: {"frames": []})
    bought = []
    for name in ("run_sponsored_website_atlas_pose", "run_sponsored_website_atlas_task", "run_sponsored_website_atlas_plan"):
        monkeypatch.setattr(allocator, name, lambda **kw: bought.append(kw) or pytest.fail("no paid stage here"))
    monkeypatch.setattr(allocator, "run_sponsored_website_atlas_automation",
                        lambda **kw: bought.append("automation") or {"status": "processing", "stage": "automation"})

    def start(family, description, semantics=None):
        context = _context(family, description)
        receipt = _receipt(context, survey["source_video_digest"], semantics=semantics)
        write_json(root / "selection_binding.json", {"survey_relative_path": "survey/atlas_video_survey.json",
            "review": {"automation": {"objects": [{"disposition": "remove"}]}}, "planner_receipt": receipt})
        return context

    def run(context):
        return pipeline.run_atlas_pipeline(source_video=video, output_root=root, task_context=context)

    return SimpleNamespace(root=root, capture=capture, video=video, bought=bought, start=start, run=run,
                           source=survey["source_video_digest"])


def _held(job, kind):
    description, family, steps, code = CANNOT_EVALUATE[kind]
    context = job.start("general_task", description, _semantics(family, steps))
    result = job.run(context)
    return context, result, code


@pytest.mark.parametrize("kind", sorted(CANNOT_EVALUATE))
def test_a_task_we_cannot_evaluate_holds_after_the_review_before_any_scene_stage(job, kind):
    context, result, code = _held(job, kind)
    assert job.bought == []  # no pose, generated view, splat, mesh or background automation
    assert result["status"] == "needs_input" and result["stage"] == early.STAGE and result["blockers"] == [code]
    assert result["splat_available"] is False and result["mesh_available"] is False
    record = json.loads((job.root / early.HOLD_FILE).read_text())
    assert record["digest"] == canonical_digest(record, digest_field="digest") == result["task_preparation_hold"]["digest"]
    assert record["hold_point"] == "after_atlas_review" and record["scene_purchased"] is False
    gap = record["gap"]
    assert gap["code"] == code and gap["job_disposition"] == "held_for_preparation"
    assert gap["task_context_digest"] == context["context_digest"] and "task_semantics" in gap
    if code == early.FAMILY_RUNTIME_GAP:
        assert gap["task_family_profile"] == "wiping" and gap["required_task_families"] == ["wiping"]
        assert gap["runtime_catalog"] == {"configured": False, "catalog_digest": None, "task_families": []}
    # Nothing to clean up or archive: no Atlas asset was ever made for it.
    assert not any((job.root / name).exists() for name in ("automation", "generated", "splat", "mesh", "result.json"))


def _catalog(tmp_path, families):
    catalog = {"schema_version": "task_family_native_runtime_catalog.v1", "runtimes": [
        {"runtime_id": "fixture-runtime", "factory_id": "gymnasium_native_task_family.v1", "task_families": families}]}
    catalog["catalog_digest"] = canonical_digest(catalog, digest_field="catalog_digest")
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(catalog))
    path.chmod(0o600)
    return path


@pytest.mark.parametrize("case", ["tray_declared", "tray_read", "clarification_pending", "runtime_installed"])
def test_a_task_we_can_evaluate_or_cannot_yet_decide_buys_its_scene_as_before(job, tmp_path, monkeypatch, case):
    if case == "tray_declared":
        # capture-447's shape: the website screens it rigid; no reading is asked for.
        context = job.start("rigid_pick_and_place", TRAY)
    elif case == "tray_read":
        context = job.start("general_task", TRAY, _semantics("rigid_pick_place", [("rigid_pick_place", TRAY)]))
    elif case == "clarification_pending":
        # A registered profile with an unresolved aspect needs the clarification bought later.
        context = job.start("general_task", TRAY, _semantics("rigid_pick_place", [("rigid_pick_place", TRAY)],
                                                             covers=False, unresolved=["tray contents"]))
    else:
        monkeypatch.setenv(CATALOG_PATH_ENV, str(_catalog(tmp_path, ["wiping"])))
        context = job.start("general_task", WIPE, _semantics("wiping", [("wiping", "Wipe the kitchen counter")]))
    result = job.run(context)
    assert job.bought == ["automation"] and result == {"status": "processing", "stage": "automation"}
    assert not (job.root / early.HOLD_FILE).exists()


def test_a_job_that_already_started_its_scene_keeps_the_late_hold(job):
    (job.root / "pose" / "atlas_pose").mkdir(parents=True)
    _, result, _ = _held(job, "door")
    assert job.bought == ["automation"] and not (job.root / early.HOLD_FILE).exists()


def test_an_unreadable_runtime_catalog_holds_nothing(job, tmp_path, monkeypatch):
    path = tmp_path / "catalog.json"
    path.write_text("{not json")
    monkeypatch.setenv(CATALOG_PATH_ENV, str(path))
    context = job.start("general_task", WIPE, _semantics("wiping", [("wiping", "Wipe the kitchen counter")]))
    job.run(context)
    assert job.bought == ["automation"]


def _preview_job(job, result):
    from blueprint_pipeline.website_atlas_preview import advance_preparation_step
    value = {"schema_version": "website_atlas_preview_job.v1", "request_id": "synthetic-job",
             "scene_id": "site-synthetic-job", "capture_id": "walkthrough-synthetic-job", "state": "processing",
             "reason": None, "stage": "queued", "revision": 1, "revised_at_epoch": 0, "updated_at_epoch": 0,
             "transient_failures": 0, "enqueued_at_epoch": 0, "next_attempt_at_epoch": 0, "retry_at_epoch": None}
    stepped = advance_preparation_step(value, job_dir=job.root, source_video=job.video, task_context={},
                                       checkpoint=lambda *_: None, now=lambda: 100, run_step=lambda: result)
    write_json(job.root / "preview_job.json", stepped)
    (job.root / "preview.lock").touch()
    return stepped


@pytest.mark.parametrize("kind", sorted(CANNOT_EVALUATE))
def test_the_preview_and_capture_stages_keep_the_hold_and_a_retry_resumes_it(job, monkeypatch, kind):
    from blueprint_pipeline.website_reconstruction_resume import _final
    context, result, code = _held(job, kind)
    preview = _preview_job(job, result)
    # Terminal, so the capture's waiting handoff replays; not failed, and no archive.
    assert (preview["state"], preview["reason"], preview["stage"]) == ("needs_input", code, early.STAGE)
    assert _final(preview) and preview.get("archive") is None
    held = early.retain_task_preparation_hold(capture_root=job.capture, task_context=context, source_video=job.video)
    assert held["status"] == "needs_input" and held["blockers"] == [code]
    assert held["visual_reconstruction_ready"] is False and held["provider_mutation_performed"] is False
    assert held["task_preparation_hold"]["hold_point"] == "after_atlas_review"
    assert holds_native_task_preparation(held, context)
    stored = json.loads((job.capture / "pipeline/website_scene_preparation/handoff.json").read_text())
    assert stored == held

    def tampered(change):
        value = deepcopy(held)
        change(value)
        value["digest"] = canonical_digest(value, digest_field="digest")
        return holds_native_task_preparation(value, context)

    assert not tampered(lambda value: value.pop("task_preparation_hold"))
    assert not tampered(lambda value: value["task_preparation_hold"].update(hold_point="after_atlas_scene"))
    if code == early.FAMILY_RUNTIME_GAP:
        # Never decided once a scene exists: the late form of this code is refused.
        assert not tampered(lambda value: value.update(visual_reconstruction_ready=True))
        assert not tampered(lambda value: value["native_task_preparation_gap"].update(task_family_profile="pouring"))
        assert not tampered(lambda value: value["native_task_preparation_gap"]["runtime_catalog"].update(
            task_families=["wiping"]))
    else:
        assert not tampered(lambda value: value["native_task_preparation_gap"].update(unsupported_operations=[]))
    # A retry reopens the job: the capture stage waits for it rather than keeping a stale hold.
    from blueprint_pipeline.website_atlas_preview import retry_failed_job
    reopened = retry_failed_job(job.root / "preview_job.json", now=lambda: 200)
    assert reopened["state"] == "processing"
    assert early.retain_task_preparation_hold(capture_root=job.capture, task_context=context) is None


def test_once_the_runtime_ships_the_retried_job_buys_its_scene_without_another_review(job, tmp_path, monkeypatch):
    context, result, _ = _held(job, "wipe")
    assert job.bought == []
    monkeypatch.setenv(CATALOG_PATH_ENV, str(_catalog(tmp_path, ["wiping"])))
    assert job.run(context) == {"status": "processing", "stage": "automation"}
    # run_sponsored_website_atlas_plan was never called: the retained review was reused.
    assert job.bought == ["automation"]


def test_the_capture_stage_holds_before_its_clean_plate_analysis(job):
    from blueprint_pipeline.website_capture_preparation import run_website_capture_preparation
    context, result, code = _held(job, "door")
    _preview_job(job, result)
    descriptor = SimpleNamespace(metadata={"site_task_context": context}, requested_outputs=["preview_simulation"])
    pipeline_dir = job.capture / "pipeline"
    value = run_website_capture_preparation(
        descriptor=descriptor, capture_root=job.capture, pipeline_dir=pipeline_dir, source_video=job.video,
        derived_scene_generation_allowed=True, gates=[],
        prepare_media=lambda **_: pytest.fail("no clean-plate analysis for a held task"),
        check_privacy=lambda **_: pytest.fail("nothing is processed"), make_gate=lambda *_: None)
    assert value["status"] == "completed" and value["native_execution_complete"] is False
    assert value["website_scene_preparation"]["blockers"] == [code]
    assert holds_native_task_preparation(value["website_scene_preparation"], context)
    manifest = json.loads((pipeline_dir / "provider_run_manifest.json").read_text())
    assert manifest["status"] == "not_requested"


def test_the_handoff_keeps_a_hold_that_arrived_while_the_clean_plate_ran(job, monkeypatch):
    from blueprint_pipeline import website_scene_handoff as handoff
    context, result, code = _held(job, "door")
    _preview_job(job, result)
    removal = job.capture / "pipeline/removal.json"
    write_json(removal, {"entries": []})
    monkeypatch.setattr(handoff, "native_task_preparation_gap", lambda *_, **__: pytest.fail("no scene to read"))
    value = handoff.prepare_website_scene_handoff(
        descriptor={"capture_id": context["capture_id"], "scene_id": context["scene_id"],
                    "metadata": {"site_task_context": context}},
        clean_plate={"privacy_verified": True, "status": "objects_removed", "task_masks": {},
                     "removal_manifest_path": str(removal), "input_video_path": str(job.video)},
        capture_root=job.capture, now=1800000000)
    assert value["blockers"] == [code] and holds_native_task_preparation(value, context)


def test_a_scene_bought_before_this_change_keeps_its_late_hold_readable():
    """The late form (visual reconstruction ready, no early record) is unchanged."""
    context = _context("general_task", DOOR)
    source = "sha256:" + "a" * 64
    from blueprint_pipeline.website_task_semantics import task_semantics_from_receipt
    semantics = task_semantics_from_receipt(receipt=_receipt(context, source, semantics=_semantics(
        "door_closing", [("door_closing", "Close the cabinet door")])), task_context=context,
        source_video_digest=source)
    gap = early.early_task_preparation_gap(context, task_semantics=semantics, source_video_digest=source)
    value = {"schema_version": "website_scene_handoff.v1", "status": "needs_input", "blockers": [gap["code"]],
             "scene_id": context["scene_id"], "capture_id": context["capture_id"], "simulator_ready": False,
             "visual_reconstruction_ready": True, "native_task_preparation_gap": gap}
    value["digest"] = canonical_digest(value, digest_field="digest")
    assert holds_native_task_preparation(value, context)
    value["visual_reconstruction_ready"] = False
    value["digest"] = canonical_digest(value, digest_field="digest")
    assert not holds_native_task_preparation(value, context)


def test_installed_runtimes_list_the_catalogs_families(tmp_path, monkeypatch):
    from blueprint_pipeline.task_family_native_runtime import installed_task_family_runtimes
    monkeypatch.delenv(CATALOG_PATH_ENV, raising=False)
    assert installed_task_family_runtimes() == {"configured": False, "catalog_digest": None, "task_families": []}
    path = _catalog(tmp_path, ["wiping", "pouring"])
    monkeypatch.setenv(CATALOG_PATH_ENV, str(path))
    value = installed_task_family_runtimes()
    assert value["configured"] is True and value["task_families"] == ["pouring", "wiping"]
    monkeypatch.setenv(CATALOG_PATH_ENV, "relative/catalog.json")
    assert installed_task_family_runtimes() is None
    assert Path(path).is_file()


@pytest.mark.parametrize("schema", ["website_preparation_status.v1", "website_preparation_status.v3"])
def test_the_status_reader_reports_an_early_hold_as_awaiting_inputs(request, schema):
    from tests import test_website_preparation_status as status_tests
    from blueprint_pipeline import pubsub_handoff_listener as listener
    from blueprint_pipeline.website_preparation_status import read_preparation_status
    source = request.getfixturevalue("prepared_source")
    root, owner, context, selected, ledger, _ = source
    context["task_family"] = "unimplemented_task_family"  # a declared family with no native profile
    context["context_digest"] = canonical_digest(context, digest_field="context_digest")
    selected["task_context_digest"] = context["context_digest"]
    write_json(root / "pipeline/website_task_context.json", context)
    gap = early.early_task_preparation_gap(context, task_semantics=None, source_video_digest=None)
    assert gap["code"] == early.PROFILE_GAP
    record = {"schema_version": early.SCHEMA, "hold_point": early.HOLD_POINT, "capture_id": owner["capture_id"],
              "scene_id": owner["scene_id"], "task_context_digest": context["context_digest"],
              "source_video_digest": "sha256:" + "a" * 64, "gap": gap, "scene_purchased": False}
    record["digest"] = canonical_digest(record, digest_field="digest")
    write_json(root / "pipeline/atlas" / early.HOLD_FILE, record)
    write_json(root / "pipeline/atlas/preview_job.json", {"schema_version": "website_atlas_preview_job.v1",
               "state": "needs_input", "stage": early.STAGE, "reason": gap["code"]})
    handoff = early.retain_task_preparation_hold(capture_root=root, task_context=context)
    assert handoff is not None and handoff["visual_reconstruction_ready"] is False
    listener._write_output_commit(root, scene_id=owner["scene_id"], capture_id=owner["capture_id"],
        attempt_count=ledger["attempt_count"], result={"status": "completed", "native_execution_complete": False,
                                                     "website_scene_preparation": handoff},
        owner="synthetic-worker", token=ledger["lease_token"], ledger_update={"status": "completed"})
    result = read_preparation_status(capture_root=root, selectors=selected, schema_version=schema)
    assert result["state"] == "awaiting_inputs" and result["code"] == "preparation_inputs_pending"
    if schema.endswith("v3"):
        assert result["outcome"] is None  # a preparation hold is not a robot verdict
    assert status_tests._post(source).json()["state"] == "awaiting_inputs"
