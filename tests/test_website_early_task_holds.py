# Covers (for impacted-test selection):
#   src/blueprint_pipeline/website_early_task_hold.py
#   src/blueprint_pipeline/website_atlas_pipeline.py
#   src/blueprint_pipeline/website_capture_preparation.py
#   src/blueprint_pipeline/website_scene_handoff.py
#   src/blueprint_pipeline/website_task_context_contracts.py
#   src/blueprint_pipeline/task_family_native_runtime.py
"""No website task is held right after its Atlas review any more (owner direction 2026-11-16).

No native profile (close a door) or no installed runtime (wipe a counter) used to hold here,
before any World Labs stage. Any job is now accepted: it buys its scene, and the handoff ships
one Blueprint has no profile or runtime for as the room package. A handoff hold an earlier
release wrote still reads as held. Synthetic only.
"""
from __future__ import annotations

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
    from blueprint_pipeline import atlas_video_inputs as inputs
    from blueprint_pipeline import website_atlas_pose as pose_api
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
def test_no_task_is_held_after_its_review_any_more_it_buys_its_scene(job, kind):
    """Owner direction 2026-11-16: any job is accepted; the handoff ships what Blueprint has no profile for as the
    room package (``tests/test_website_scene_handoff_navigation``), so nothing is held before the scene."""
    context, result, _code = _held(job, kind)
    assert job.bought == ["automation"] and result == {"status": "processing", "stage": "automation"}
    assert not (job.root / "task_preparation_hold.json").exists()


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
    assert not (job.root / "task_preparation_hold.json").exists()


def test_a_job_that_already_started_its_scene_keeps_the_late_hold(job):
    (job.root / "pose" / "atlas_pose").mkdir(parents=True)
    _, result, _ = _held(job, "door")
    assert job.bought == ["automation"] and not (job.root / "task_preparation_hold.json").exists()


def test_an_unreadable_runtime_catalog_holds_nothing(job, tmp_path, monkeypatch):
    path = tmp_path / "catalog.json"
    path.write_text("{not json")
    monkeypatch.setenv(CATALOG_PATH_ENV, str(path))
    context = job.start("general_task", WIPE, _semantics("wiping", [("wiping", "Wipe the kitchen counter")]))
    job.run(context)
    assert job.bought == ["automation"]


def test_a_scene_bought_before_this_change_keeps_its_late_hold_readable():
    """The late form (visual reconstruction ready, no early record) is unchanged."""
    context = _context("general_task", DOOR)
    source = "sha256:" + "a" * 64
    from blueprint_pipeline.website_task_semantics import task_semantics_from_receipt
    semantics = task_semantics_from_receipt(receipt=_receipt(context, source, semantics=_semantics(
        "door_closing", [("door_closing", "Close the cabinet door")])), task_context=context,
        source_video_digest=source)
    from blueprint_pipeline.website_task_preparation import native_task_preparation_gap
    gap = native_task_preparation_gap(context, task_semantics=semantics, source_video_digest=source)
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
