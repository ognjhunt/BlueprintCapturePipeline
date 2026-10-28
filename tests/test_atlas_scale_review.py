"""Scale repairs reuse retained views, check endpoints, and leave prior evidence intact."""
from copy import deepcopy
import json

import pytest

from blueprint_pipeline import atlas_scale_review as repair
from blueprint_pipeline import atlas_scene_measurements as sizes
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from tests.test_atlas_scene_automation import view, plan, track


def proposal(automation):
    refs = deepcopy(automation["scale_references"])
    for row in refs:
        row["segmentation_prompt"] = "visible reference object"
    return {"status": "proposed", "scale_references": refs, "summary": "Corrected visible object edges"}


def test_review_remeasures_new_endpoints_without_overwriting_retained_scale(tmp_path):
    frames = [view(tmp_path)]
    original = plan()
    original["scale_references"][0]["endpoints_normalized"][0][0] = .05
    old_path = tmp_path / "old.json"
    old = sizes.estimate_sizes(frames=frames, automation=original, pose_digest="pose", output=old_path)
    before = old_path.read_bytes()
    assert repair.needs_scale_review(old)
    masks = [{"reference_index": 0, "frame_id": "a", "instances": track(frames)["track"]["observations"]}]
    fixed, path = repair.measure_review(frames=frames, pose_digest="pose", automation=original,
        proposal=proposal(plan()), reference_sources=[], reference_masks=masks, output_root=tmp_path / "revised")
    assert fixed["meters_per_unit_estimate"] == pytest.approx(1.)
    assert old["meters_per_unit_estimate"] < fixed["meters_per_unit_estimate"]
    assert fixed["binding"]["automation"]["objects"] == original["objects"]
    assert not repair.needs_scale_review(fixed)
    assert old_path.read_bytes() == before and path != old_path
    assert repair.measure_review(frames=frames, pose_digest="pose", automation=original,
        proposal=proposal(plan()), reference_sources=[], reference_masks=masks,
        output_root=tmp_path / "revised") == (fixed, path)


def test_repair_cannot_accept_foreground_endpoint_or_missing_object_masks(tmp_path):
    frames = [view(tmp_path)]
    p = proposal(plan())
    p["scale_references"][0]["endpoints_normalized"][0][0] = .05
    masks = [{"reference_index": 0, "frame_id": "a", "instances": track(frames)["track"]["observations"]}]
    options = dict(frames=frames, pose_digest="pose", automation=plan(), proposal=p,
                   reference_sources=[], output_root=tmp_path / "revised")
    result, _ = repair.measure_review(**options, reference_masks=masks)
    assert result["status"] == "relative_sizes_only"
    assert result["reference_gaps"][0]["reason"] == "atlas_reference_endpoint_off_object"
    with pytest.raises(ValueError, match="object_masks_required"):
        repair.measure_review(**options, reference_masks=None)


@pytest.mark.parametrize("basis, reviewed", [
    ("camera_height_prior_only", True), ("surface_height_prior_only", True), ("height_priors_only", True),
    ("single_object_with_height_priors", False), ("majority_with_height_priors", False), ("single_object", False)])
def test_a_guesstimate_on_height_priors_alone_still_asks_for_references(basis, reviewed):
    measured = {"status": "estimated", "binding": {"reference_masks_digest": "masks"},
                "reference_agreement": {"basis": basis}}
    assert repair.rests_on_height_priors(measured) is reviewed
    assert repair.needs_scale_review(measured) is reviewed
    assert not repair.rests_on_height_priors({**measured, "status": "relative_sizes_only"})


def test_the_review_consults_the_camera_height_when_its_references_fail(tmp_path):
    # The only reference fails its mask check, so the cameras' height is asked; one wall view shows no floor.
    frames = [view(tmp_path, "a")]
    p = proposal(plan())
    p["scale_references"][0]["endpoints_normalized"][0][0] = .05
    masks = [{"reference_index": 0, "frame_id": "a", "instances": track(frames)["track"]["observations"]}]
    result, _ = repair.measure_review(frames=frames, pose_digest="pose", automation=plan(), proposal=p,
                                      reference_sources=[], reference_masks=masks, output_root=tmp_path / "revised")
    assert result["height_priors"]["camera"]["status"] == "unavailable"


@pytest.mark.parametrize("change", ["unknown_frame", "unnamed", "abstained"])
def test_invalid_reference_proposals_cannot_be_measured(change):
    p = proposal(plan())
    if change == "unknown_frame":
        p["scale_references"][0]["frame_id"] = "not_seen"
    elif change == "unnamed":
        p["scale_references"][0].pop("segmentation_prompt")
    else:
        p["status"] = "abstained"
    with pytest.raises(ValueError, match="atlas_scale_review"):
        repair.revised_plan(automation=plan(), proposal=p, frame_ids={"a"}, reference_sources=[])


def test_automatic_repair_reviews_full_views_and_keeps_one_revision(tmp_path, monkeypatch):
    from blueprint_pipeline import atlas_context_fov as fov, atlas_reference_masks as masking
    frames = [view(tmp_path)]
    source = {"frame_id": "a", "image_path": frames[0]["image_path"], "image_digest": frames[0]["image_digest"]}
    monkeypatch.setattr(fov, "full_view_sources", lambda *_: [source])
    monkeypatch.setattr(sizes, "load_views", lambda *_: ({}, frames))
    checked = []
    def segment(**kw):
        assert kw["references"][0]["segmentation_prompt"] == "visible reference object"
        return [{"reference_index": 0, "frame_id": "a", "instances": track(frames)["track"]["observations"]}]
    monkeypatch.setattr(masking, "segment_scale_references", segment)
    def planner(**kw):
        survey = json.loads(kw["survey_path"].read_text())
        assert survey["frames"][0]["source_image_path"] == source["image_path"]
        assert survey["display_rotation_degrees"] == 0
        assert not kw["scaling"]["named_object_masks_checked"]
        assert kw["scaling"]["previous_measurement_digest"] == "prior"
        return {"proposal": proposal(plan()), "reference_sources": []}
    result, path = repair.repair_source_scale(pose_root=tmp_path, pose={"digest": "pose"},
        inputs={"binding": {"source_video_digest": "video"}}, automation=plan(),
        previous={"digest": "prior"}, task_context={}, configuration={}, check=checked.append, planner_runner=planner)
    assert result["status"] == "estimated" and checked == ["scale_review"]
    assert path.is_relative_to(tmp_path / "scale_review/measurements")


def _automatic_options(tmp_path, monkeypatch):
    from blueprint_pipeline import atlas_context_fov as fov, atlas_reference_masks as masking
    frames = [view(tmp_path)]
    source = {"frame_id": "a", "image_path": frames[0]["image_path"], "image_digest": frames[0]["image_digest"]}
    monkeypatch.setattr(fov, "full_view_sources", lambda *_: [source])
    monkeypatch.setattr(sizes, "load_views", lambda *_: ({}, frames))
    monkeypatch.setattr(masking, "segment_scale_references", lambda **_: [
        {"reference_index": 0, "frame_id": "a", "instances": track(frames)["track"]["observations"]}])
    return dict(pose_root=tmp_path, pose={"digest": "pose"},
        inputs={"binding": {"source_video_digest": "video"}}, automation=plan(),
        previous={"digest": "prior"}, task_context={}, configuration={}, check=lambda _: None)


def _coverage_request():
    return {"scope": "operation_coverage", "task_context_digest": "sha256:" + "a" * 64,
            "source_video_digest": "sha256:" + "b" * 64,
            "parent_semantics_digest": "sha256:" + "c" * 64}


@pytest.mark.parametrize("status", ["completed", "submitting", "invalid_output"])
def test_retained_scale_attempts_keep_the_exact_legacy_request(tmp_path, monkeypatch, status):
    options = _automatic_options(tmp_path, monkeypatch)
    planner_root = tmp_path / "scale_review/planner"
    planner_root.mkdir(parents=True)
    (planner_root / "retained.json").write_text(json.dumps({"status": status}))
    observed = []
    def cached_planner(**kw):
        # The normal planner owns validating/adopting/settling this same request.
        assert "operation_coverage_request" not in kw["scaling"]
        observed.append(kw["scaling"])
        return {"proposal": proposal(plan()), "reference_sources": []}
    repair.repair_source_scale(**options, planner_runner=cached_planner,
        operation_coverage_request=_coverage_request(), operation_coverage_needed=True)
    assert len(observed) == 1
    assert not (tmp_path / "scale_review/operation-coverage-request.json").exists()


def test_new_scale_clarification_keeps_the_same_cached_input_after_resolution(tmp_path, monkeypatch):
    options = _automatic_options(tmp_path, monkeypatch)
    cache, calls, consumed = {}, [], []
    request = _coverage_request()
    def cached_planner(**kw):
        assert kw["scaling"]["operation_coverage_request"] == request
        identity = canonical_digest({"survey": json.loads(kw["survey_path"].read_text()), "scaling": kw["scaling"]})
        if identity not in cache:
            calls.append(identity)
            cache[identity] = {"proposal": {**proposal(plan()), "operation_coverage": {"independent": "reading"}},
                               "reference_sources": []}
            kw["output_root"].mkdir(parents=True, exist_ok=True)
            (kw["output_root"] / (identity[7:] + ".json")).write_text(json.dumps(cache[identity]))
        return cache[identity]
    first = repair.repair_source_scale(**options, planner_runner=cached_planner,
        operation_coverage_request=request, operation_coverage_needed=True,
        operation_coverage_consumer=consumed.append)
    retained = tmp_path / "scale_review/operation-coverage-request.json"
    before = retained.read_bytes()
    repeated = repair.repair_source_scale(**options, planner_runner=cached_planner,
        operation_coverage_request=request, operation_coverage_needed=False,
        operation_coverage_consumer=consumed.append)
    assert first == repeated and len(calls) == 1 and len(consumed) == 2
    assert retained.read_bytes() == before
    assert first[0]["binding"]["automation"]["scale_review"]["proposal_digest"] == canonical_digest(
        next(iter(cache.values()))["proposal"])


@pytest.mark.parametrize("changed", ["body", "symlink", "oversized"])
def test_changed_scale_clarification_sidecar_stops_before_a_model_call(tmp_path, monkeypatch, changed):
    options = _automatic_options(tmp_path, monkeypatch)
    root = tmp_path / "scale_review"
    root.mkdir()
    sidecar = root / "operation-coverage-request.json"
    if changed == "symlink":
        target = tmp_path / "other.json"
        target.write_text("{}")
        sidecar.symlink_to(target)
    elif changed == "oversized":
        sidecar.write_bytes(b" " * 65_537)
    else:
        sidecar.write_text(json.dumps({**_coverage_request(), "parent_semantics_digest": "another-parent"}))
    def unavailable(**_):
        raise AssertionError("Changed sidecar must not invoke a planner")
    with pytest.raises(ValueError, match="atlas_scale_operation_coverage_request_changed"):
        repair.repair_source_scale(**options, planner_runner=unavailable,
            operation_coverage_request=_coverage_request(), operation_coverage_needed=True)


def test_scale_abstention_still_retains_independent_operation_coverage(tmp_path, monkeypatch):
    options = _automatic_options(tmp_path, monkeypatch)
    receipt = {"proposal": {"status": "abstained", "scale_references": [], "summary": "No defensible reference",
                           "operation_coverage": {"independent": "reading"}}}
    consumed = []
    with pytest.raises(ValueError, match="atlas_scale_review_abstained"):
        repair.repair_source_scale(**options, planner_runner=lambda **_: receipt,
            operation_coverage_request=_coverage_request(), operation_coverage_needed=True,
            operation_coverage_consumer=consumed.append)
    assert consumed == [receipt]


def test_unknown_scale_proposal_fields_are_still_rejected():
    p = {**proposal(plan()), "unrecognized": "not a scale field"}
    with pytest.raises(ValueError):
        repair.revised_plan(automation=plan(), proposal=p, frame_ids={"a"}, reference_sources=[])
