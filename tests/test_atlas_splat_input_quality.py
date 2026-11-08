"""A refined splat's posed views: consistent, with parallax, a few held back; and its one rebuild.

Fixtures are the retained JSON of two jobs on the same request form (cameras, source times,
survey sharpness and the draft's cross-view pairs; no pixels): capture-5b067498's library
aisle, whose refined splat came out blurry, and capture-447ae4f4's kitchen, which came out clear.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pytest

from blueprint_pipeline import atlas_splat_input_quality as quality
from blueprint_pipeline import atlas_splat_rebuild as rebuilds
from blueprint_pipeline.decision_evidence_contracts import canonical_digest

FIXTURES = Path(__file__).parent / "fixtures" / "splat_input_quality"
OPENING = {f"decoded-{n:09d}" for n in (0, 12, 24, 29, 36, 49, 55, 58, 61, 74)}


def _fixture(name):
    return json.loads((FIXTURES / name).read_text())


def _select(value, **changes):
    options = dict(frames=value["frames"], agreement={"pairs": value["pairs"]},
                   timestamps={row["frame_id"]: row["timestamp_seconds"] for row in value["frames"]},
                   sharpness={row["frame_id"]: row["sharpness"] for row in value["frames"] if row["sharpness"]},
                   scene_depth=value["scene_depth_units"], pose_refinement_accepted=value["pose_refinement_accepted"],
                   route_frame_ids=value["route_frame_ids"])
    return quality.select_splat_frames(**{**options, **changes})


def test_the_library_pattern_sends_a_consistent_dolly_and_records_why_the_rest_was_left_out():
    value = _fixture("library_aisle_capture_5b067498.json")
    # The only check there was: the pooled median, well inside the limit.
    assert value["pooled_median_relative_residual"] < quality.MAX_FRAME_RESIDUAL
    record = _select(value)
    times = {row["frame_id"]: row["timestamp_seconds"] for row in value["frames"]}
    # The opening ~3 s, each judged on its own pairs, including the two views that matched only each other.
    assert {row["frame_id"] for row in record["consistency"]["dropped"]} == OPENING
    assert {row["frame_id"]: row["reason"] for row in record["consistency"]["dropped"]}["decoded-000000029"] \
        == "agrees_only_with_views_that_disagree_elsewhere"
    # The 3.3-unit jump in 0.16 s splits the pan from the dolly, and the pan's registration is unconfirmed.
    assert record["motion"]["jumps"] == [{"after_frame_id": "decoded-000000199", "before_frame_id": "decoded-000000203",
                                          "translation_units": pytest.approx(3.33, abs=.01),
                                          "rotation_degrees": pytest.approx(5.34, abs=.01), "seconds": .16}]
    kinds = [(row["kind"], row["decision"], len(row["frame_ids"])) for row in record["motion"]["segments"]]
    assert kinds == [("no_parallax", "left_out_across_pose_jump", 12), ("moving", "primary", 19)]
    # Dolly-dominated (here dolly only) and consistent: every sent view is from after the jump and agrees.
    assert len(record["sent_frame_ids"]) == 19 and all(times[key] >= 8.1 for key in record["sent_frame_ids"])
    residuals = quality.frame_residuals(value["pairs"], record["sent_frame_ids"])
    assert max(residual for residual, _ in residuals.values()) <= quality.MAX_FRAME_RESIDUAL
    left = {row["frame_id"] for row in record["left_out"]}
    assert left | set(record["sent_frame_ids"]) == {row["frame_id"] for row in value["frames"]} and len(left) == 22
    assert record["flags"] == ["pose_refinement_not_accepted", "splat_inconsistent_frames_dropped",
                               "splat_pose_jump_segment_left_out", "task_route_frames_left_out"]
    assert record["route_frames_left_out"] == ["decoded-000000000", "decoded-000000199"]
    # 19 views are too few to hold any back.
    assert record["heldout_frame_ids"] == [] and record["heldout_unavailable_reason"] == "too_few_views_to_hold_back"
    assert record["digest"] == canonical_digest(record, digest_field="digest")


def test_the_kitchen_pattern_is_unchanged():
    value = _fixture("kitchen_capture_447ae4f4.json")
    record = _select(value)
    assert record["sent_frame_ids"] == [row["frame_id"] for row in value["frames"]]
    assert record["left_out"] == [] and record["flags"] == [] and record["heldout_frame_ids"] == []
    assert [(row["kind"], row["decision"]) for row in record["motion"]["segments"]] == [("moving", "primary")]


def test_a_pan_without_a_jump_keeps_only_its_widest_views_beside_the_dolly():
    value = _fixture("library_aisle_capture_5b067498.json")
    jump = np.subtract(*[next(row["extrinsics"]["position"] for row in value["frames"] if row["frame_id"] == key)
                         for key in ("decoded-000000203", "decoded-000000199")])
    # The same capture, registered: the pan stood just behind where the dolly starts.
    for row in value["frames"]:
        if row["timestamp_seconds"] < 8.1:
            row["extrinsics"]["position"] = list(np.add(row["extrinsics"]["position"], .95 * jump))
    record = _select(value)
    assert record["motion"]["jumps"] == []
    segments = record["motion"]["segments"]
    assert [(row["kind"], row["decision"]) for row in segments] == [("no_parallax", "capped"), ("moving", "primary")]
    assert len(segments[0]["kept_frame_ids"]) == quality.STATIONARY_FRAME_CAP
    assert "decoded-000000199" in segments[0]["kept_frame_ids"]  # the view next to the dolly seeds the spread
    assert len(record["sent_frame_ids"]) == 19 + quality.STATIONARY_FRAME_CAP
    assert "splat_no_parallax_frames_capped" in record["flags"]
    assert {row["reason"] for row in record["left_out"]} == {"depth_disagrees_with_measured_neighbours",
                                                              "agrees_only_with_views_that_disagree_elsewhere",
                                                              "no_parallax_view_beyond_cap"}


def test_measured_pairs_that_tie_a_segment_across_a_jump_keep_it():
    value = _fixture("library_aisle_capture_5b067498.json")
    value["pairs"] += [{"source": "decoded-000000199", "target": "decoded-000000203", "median_relative_residual": .01},
                       {"source": "decoded-000000187", "target": "decoded-000000215", "median_relative_residual": .02}]
    record = _select(value)
    assert [(row["kind"], row["decision"]) for row in record["motion"]["segments"]] == [
        ("no_parallax", "capped"), ("moving", "primary")]
    assert len(record["sent_frame_ids"]) == 19 + quality.STATIONARY_FRAME_CAP
    assert "splat_pose_jump_segment_left_out" not in record["flags"]


def _line(count, *, step=.25, seconds=.5):
    frames = [{"frame_id": f"v{i:03d}", "timestamp_seconds": i * seconds, "sharpness": 100 + (i % 3),
               "extrinsics": {"position": [0.0, 1.5, -step * i], "quaternion": [0, 0, 0, 1]}} for i in range(count)]
    pairs = [{"source": a["frame_id"], "target": b["frame_id"], "median_relative_residual": .004}
             for a, b in zip(frames, frames[1:])]
    return {"frames": frames, "pairs": pairs, "scene_depth_units": 3.0, "pose_refinement_accepted": True,
            "route_frame_ids": ["v000", f"v{count - 1:03d}"]}


def test_a_long_consistent_dolly_holds_back_sharp_interior_views_and_sends_the_rest():
    value = _line(40)
    record = _select(value)
    held = record["heldout_frame_ids"]
    assert len(held) == quality.HELDOUT_FRAME_COUNT and len(record["sent_frame_ids"]) == 38
    order = [row["frame_id"] for row in value["frames"]]
    assert all(2 <= order.index(key) <= 37 for key in held) and not set(held) & set(value["route_frame_ids"])
    assert record["flags"] == []


def test_too_few_consistent_views_flag_and_keep_the_best_instead_of_stopping():
    value = _line(3)
    value["pairs"] = [{"source": "v000", "target": "v001", "median_relative_residual": .4},
                      {"source": "v001", "target": "v002", "median_relative_residual": .9},
                      {"source": "v000", "target": "v002", "median_relative_residual": .2}]
    record = _select(value)
    assert len(record["sent_frame_ids"]) == quality.PROVIDER_MINIMUM_FRAMES
    assert "splat_consistent_frames_below_provider_minimum" in record["flags"]
    assert record["consistency"]["restored_frame_ids"]
    # No agreement and no depth scale: everything goes on, named.
    record = _select(_line(5), agreement=None, scene_depth=None)
    assert len(record["sent_frame_ids"]) == 5 and record["motion"]["reason"] == "scene_depth_unavailable"
    assert record["flags"] == ["splat_cross_view_agreement_unavailable"]


def test_a_capture_that_never_moved_keeps_every_view():
    record = _select(_line(12, step=0.0))
    assert len(record["sent_frame_ids"]) == 12 and record["flags"] == ["splat_no_parallax_capture"]


def _plane_splat(colours, k):
    from blueprint_pipeline.gaussian_splat_decode import SplatData
    height, width, _ = colours.shape
    v, u = np.mgrid[0:height, 0:width]
    depth = 2.0
    xyz = np.stack([(u + .5 - k["cx"]) / k["fx"] * depth, -(v + .5 - k["cy"]) / k["fy"] * depth,
                    np.full(u.shape, -depth)], axis=-1).reshape(-1, 3)
    count = len(xyz)
    return SplatData(count=count, xyz=xyz.astype(np.float32), opacity=np.full(count, 4.0, np.float32),
                     f_dc=((colours.reshape(-1, 3) - .5) / .28209479177387814).astype(np.float32),
                     scales=np.full((count, 3), -5.0, np.float32), quats=np.tile([1, 0, 0, 0], (count, 1)).astype(
                         np.float32), properties=())


def test_the_heldout_score_compares_a_centre_render_with_each_photo_and_flags_held_out_views_below_training():
    rng = np.random.default_rng(0)
    k = {"fx": 300.0, "fy": 300.0, "cx": 160.0, "cy": 90.0, "width": 320, "height": 180}
    colours = rng.random((180, 320, 3))
    splat = _plane_splat(colours, k)
    camera = {"intrinsics": k, "position": np.zeros(3), "rotation": np.eye(3)}
    match = {**camera, "frame_id": "train", "rgb": np.round(colours * 255).astype(np.uint8)}
    other = {**camera, "frame_id": "held", "rgb": np.round((1 - colours) * 255).astype(np.uint8)}
    good = quality.render_photo_score(splat, match)
    assert good["coverage"] > .99 and good["psnr_db"] > 35 and good["edge_energy_ratio"] == pytest.approx(1, abs=.05)
    assert quality.render_photo_score(splat, other)["psnr_db"] < 10
    record = quality.heldout_quality(splat=splat, frames=[match, other], selection={
        "digest": "sha256:" + "d" * 64, "posed_frame_ids": ["train", "held"], "sent_frame_ids": ["train"],
        "heldout_frame_ids": ["held"]})
    assert record["status"] == "scored" and record["flags"] == ["heldout_render_below_training"]
    assert [row["frame_id"] for row in record["training"]] == ["train"]
    assert record["image_quality_measurement"] is False and record["claim_scope"] == "development_only"


def _posed(value):
    return [{**row, "depth": np.full((4, 4), value["scene_depth_units"], np.float32),
             "confidence": np.ones((4, 4), np.float32)} for row in value["frames"]]


def test_the_pipeline_records_the_selection_once_before_the_purchase_and_reads_it_back(tmp_path):
    from blueprint_pipeline import website_atlas_pipeline as pipeline
    value = _fixture("library_aisle_capture_5b067498.json")
    source_root = tmp_path / "pose/full-view-source-v1"
    source_root.mkdir(parents=True)
    (source_root / "atlas_geometry_inputs.json").write_text(json.dumps({"frames": [
        {"frame_id": row["frame_id"], "timestamp_seconds": row["timestamp_seconds"], "sharpness": row["sharpness"]}
        for row in value["frames"]]}))
    pose = {"digest": "sha256:" + "1" * 64, "operation": {"response": {"poseRefinementAccepted": False}}}
    draft = {"digest": "sha256:" + "2" * 64, "pose_result_digest": pose["digest"],
             "cross_view_agreement": {"pairs": value["pairs"]}}
    review = {"coverage_requirements": [{"role": "task_region", "required_frame_ids": value["route_frame_ids"]},
                                        {"role": "surrounding_scene", "required_frame_ids": ["decoded-000000012"]}]}
    options = dict(frames=_posed(value), pose=pose, draft=draft, output_root=tmp_path, review=review,
                   source_root=source_root)
    # A splat bought before this record existed keeps what it was sent.
    assert pipeline._splat_input_selection(**options, bought=True) is None
    record = pipeline._splat_input_selection(**options, bought=False)
    assert len(record["sent_frame_ids"]) == 19 and record["pose_result_digest"] == pose["digest"]
    assert record["route_frame_ids"] == sorted(value["route_frame_ids"])
    assert record["scene_depth_units"] == pytest.approx(value["scene_depth_units"])
    saved = json.loads((tmp_path / quality.SELECTION_FILE).read_text())
    assert saved == record and record["digest"] == canonical_digest(record, digest_field="digest")
    # Read back, never re-derived; a different draft refuses rather than silently reselecting.
    assert pipeline._splat_input_selection(**options, bought=True) == record
    with pytest.raises(ValueError, match="atlas_splat_input_selection_changed"):
        pipeline._splat_input_selection(**{**options, "draft": {**draft, "digest": "sha256:" + "3" * 64}},
                                        bought=False)


@pytest.mark.parametrize("case", ["refused_pose", "few_views", "unaffordable", "bought"])
def test_a_rebuild_buys_its_splat_only_on_an_accepted_consistent_pose_and_never_falls_back(tmp_path, monkeypatch,
                                                                                          case):
    from blueprint_pipeline import atlas_native_splat, website_atlas_pipeline as pipeline
    frames = [{"frame_id": "a"}, {"frame_id": "b"}, {"frame_id": "c"}]
    selection = {"digest": "sha256:" + "4" * 64, "pose_refinement_accepted": case != "refused_pose",
                 "flags": ["splat_consistent_frames_below_floor"] if case == "few_views" else [],
                 "sent_frame_ids": ["a", "c"], "left_out": [{"frame_id": "b"}], "heldout_frame_ids": []}
    monkeypatch.setattr(pipeline, "_splat_input_selection", lambda **_: selection)
    monkeypatch.setattr(pipeline, "load_configuration", lambda *_: {
        "remaining_cost_usd": 3 if case == "unaffordable" else 17.62})
    bought = []
    monkeypatch.setattr(atlas_native_splat, "run_native_splat", lambda **kw: bought.append(kw) or {
        "status": "completed", "splat_root": str(tmp_path / "refined_splat-minimal"), "splat_manifest": {}})
    result = pipeline._preserved_appearance(frames=frames, pose={"digest": "p"}, draft={"digest": "d"},
        output_root=tmp_path, policy={"refined_splat_cost_usd": 5, "mesh_export_cost_usd": .5},
        task_context={"context_digest": "t"}, source_digest="s", configuration={}, check=lambda *_: None,
        experimental=False, max_polls=1, admitted_completion=False, review={}, source_root=tmp_path, rebuild=True)
    if case == "bought":
        assert result["basis"] == "atlas_refined_splat" and [f["frame_id"] for f in bought[0]["frames"]] == ["a", "c"]
        return
    reasons = {"refused_pose": "atlas_splat_rebuild_pose_refinement_not_accepted",
               "few_views": "atlas_splat_rebuild_consistent_views_insufficient",
               "unaffordable": "atlas_splat_rebuild_allowance_insufficient"}
    assert result == {"status": "needs_input", "stage": "refined_splat", "reason": reasons[case]} and not bought


def write_preserved_records(job_dir, *, accepted=False, pairs=None):
    """The retained records of a ready preserved job laid out as capture-5b067498's was; returns its result."""
    value = _fixture("library_aisle_capture_5b067498.json")
    source = job_dir / "pose/task-coverage-0/full-view-source-v1"
    source.mkdir(parents=True)
    pose = {"operation": {"response": {"poseRefinementAccepted": accepted}},
            "operation_receipt": {"input_usage": {"frames": [{"frame_id": row["frame_id"]} for row in value["frames"]]}}}
    pose["digest"] = canonical_digest(pose)
    (source / "atlas_posed_rgbd_result.json").write_text(json.dumps(pose))
    (job_dir / "splat").mkdir()
    (job_dir / "splat/atlas_depth_splat_manifest.json").write_text(json.dumps(
        {"pose_result_digest": pose["digest"], "cross_view_agreement": {"pairs": value["pairs"] if pairs is None
                                                                         else pairs}}))
    (job_dir / "pose/atlas_view_selection.json").write_text(json.dumps({"frames": value["frames"][:31]}))
    (job_dir / "configuration.json").write_text(json.dumps({"policy": {
        "pose_input_cost_usd": .01, "refined_splat_cost_usd": 5, "mesh_export_cost_usd": .5,
        "maximum_review_cost_usd": 4}}))
    result = {"schema_version": "website_atlas_pipeline_result.v1", "captured_scene_preserved": True,
              "generated_geometry_admitted": False, "appearance": {"basis": "atlas_refined_splat"}, "model": "m",
              "source_pose": {"status": "completed", "pose_relative_path": "pose/task-coverage-0/full-view-source-v1"},
              "download": {"zip_path": "parent.zip"}}
    result["digest"] = canonical_digest(result, digest_field="digest")
    (job_dir / "result.json").write_text(json.dumps(result))
    return result


def _ready_job(tmp_path, *, accepted=False, pairs=None):
    from blueprint_pipeline.website_atlas_preview_store import SCHEMA
    job_dir = tmp_path / "pipeline/atlas"
    job_dir.mkdir(parents=True)
    result = write_preserved_records(job_dir, accepted=accepted, pairs=pairs)
    job = {"schema_version": SCHEMA, "state": "ready", "reason": None, "stage": "archived", "revision": 10,
           "enqueued_at_epoch": int(time.time()) - 3600, "result_digest": result["digest"], "refined_splat": None,
           "provider_cleanup": {"state": "completed"}, "retries": []}
    (job_dir / "preview_job.json").write_text(json.dumps(job))
    (job_dir / "preview.lock").touch()
    return job_dir, job, result


def test_a_ready_library_like_job_qualifies_for_one_rebuild_with_its_reasons_and_cost(tmp_path):
    job_dir, job, result = _ready_job(tmp_path)
    eligible = rebuilds.rebuild_eligibility(job_dir, job)
    assert eligible == {"parent_result_digest": result["digest"],
                        "reasons": ["pose_refinement_not_accepted", "views_disagree_with_measured_neighbours"],
                        "parent_posed_frame_count": 41, "views_dropped_by_consistency": 10,
                        # 96 views plus its 10 task-source views, the $5 splat and the $0.50 mesh.
                        "expected_cost_usd": 6.56,
                        "maximum_cost_usd": pytest.approx((128 + 10) * .01 + 5.5 + 4)}
    # A job whose pose was accepted and agrees, or one already bought under a selection, does not qualify.
    clean, clean_job, _ = _ready_job(tmp_path / "clean", accepted=True, pairs=[])
    assert rebuilds.rebuild_eligibility(clean, clean_job) is None
    (job_dir / quality.SELECTION_FILE).write_text("{}")
    assert rebuilds.rebuild_eligibility(job_dir, job) is None
    (job_dir / quality.SELECTION_FILE).unlink()
    assert rebuilds.rebuild_eligibility(job_dir, {**job, "state": "processing"}) is None
    assert rebuilds.rebuild_eligibility(job_dir, {**job, "result_digest": "sha256:" + "0" * 64}) is None


def test_atlas_retry_reopens_an_eligible_ready_job_once_and_writes_its_request(tmp_path):
    from blueprint_pipeline.website_atlas_preview import retry_failed_job
    job_dir, job, result = _ready_job(tmp_path)
    reopened = retry_failed_job(job_dir / "preview_job.json")
    assert reopened["state"] == "processing" and reopened["retries"][-1]["reason"] == "refined_splat_rebuild_requested"
    request = json.loads((job_dir / rebuilds.REQUEST_FILE).read_text())
    root = job_dir / request["root"]
    assert root.parent == job_dir / rebuilds.ROOT and request["parent_result_digest"] == result["digest"]
    assert json.loads((root / "parent_result.json").read_text()) == result
    # Ready again on its original scene, it does not qualify a second time.
    (job_dir / "preview_job.json").write_text(json.dumps(job))
    with pytest.raises(ValueError, match="atlas_preview_retry_requires_recoverable_failure"):
        retry_failed_job(job_dir / "preview_job.json")


@pytest.mark.parametrize("outcome", ["processing", "stopped", "published"])
def test_the_rebuild_runs_in_its_own_root_and_publishes_or_hands_the_parent_back(tmp_path, monkeypatch, outcome):
    from blueprint_pipeline import atlas_video_inputs, website_atlas_pipeline as pipeline
    job_dir, job, parent = _ready_job(tmp_path)
    (job_dir / "survey").mkdir()
    (job_dir / "survey/atlas_video_survey.json").write_text("{}")
    (job_dir / "selection_binding.json").write_text(json.dumps(
        {"survey_relative_path": "survey/atlas_video_survey.json", "review": {"subject_origin": "no_manipulated_subject"}}))
    request = rebuilds.request_rebuild(job_dir, job, at_epoch=1)
    root = job_dir / request["root"]
    selected = []
    monkeypatch.setattr(atlas_video_inputs, "prepare_video_inputs", lambda **kw: selected.append(kw) or {"frames": []})

    def scene(**kw):
        assert kw["rebuild"] is True and kw["output_root"] == root and kw["pose_root"] == root / "pose"
        if outcome == "processing":
            return {"status": "processing", "stage": "source_pose"}
        if outcome == "stopped":
            return {"status": "needs_input", "stage": "refined_splat",
                    "reason": "atlas_splat_rebuild_pose_refinement_not_accepted"}
        return {"status": "needs_input", "appearance": {"basis": "atlas_refined_splat"}, "download": {"zip": "new"},
                "source_pose": {"status": "completed", "pose_relative_path": "pose/task-coverage-1/full-view-source-v1"},
                "digest": "sha256:" + "5" * 64}
    monkeypatch.setattr(pipeline, "_preserved_scene", scene)
    options = dict(job_dir=job_dir, task_context={}, policy={"refined_splat_cost_usd": 5}, held_policy={
        "refined_splat_cost_usd": 5}, configuration={}, source_digest="s", check=lambda *_: None,
        experimental=False, max_polls=1)
    value = pipeline._run_splat_rebuild(**options)
    # A new selection under the current profile: the splat target and the relative sharpness floor.
    assert selected[0]["splat_frame_target"] == 96 and selected[0]["relative_sharpness_fraction"] == .6
    assert json.loads((root / "atlas_scene_review.json").read_text()) == {"subject_origin": "no_manipulated_subject"}
    if outcome == "processing":
        assert value["status"] == "processing" and not (root / rebuilds.OUTCOME_FILE).exists()
        return
    if outcome == "stopped":
        assert value == parent and json.loads((job_dir / "result.json").read_text()) == parent
        assert json.loads((root / rebuilds.OUTCOME_FILE).read_text())["reason"] \
            == "atlas_splat_rebuild_pose_refinement_not_accepted"
    else:
        assert value["source_pose"]["pose_relative_path"] == request["root"] + "/pose/task-coverage-1/full-view-source-v1"
        assert value["splat_rebuild"] == {"parent_result_digest": parent["digest"], "request_digest": request["digest"],
                                          "root": request["root"]}
        assert value["digest"] == canonical_digest(value, digest_field="digest")
        assert json.loads((job_dir / "result.json").read_text()) == value
        assert json.loads((root / "parent_result.json").read_text()) == parent
    # A finished rebuild is read back, never run again.
    monkeypatch.setattr(pipeline, "_preserved_scene", lambda **_: pytest.fail("rebuild ran twice"))
    assert pipeline._run_splat_rebuild(**options) == value


def test_the_controller_runs_a_requested_rebuild_instead_of_the_ready_job(tmp_path, monkeypatch):
    from blueprint_pipeline import website_atlas_pipeline as pipeline, website_atlas_pose as pose_api
    video = tmp_path / "source.mov"
    video.write_bytes(b"retained-source")
    root = tmp_path / "job"
    (root / "scene_planner").mkdir(parents=True)
    saved = {"authority": {"provider_retention": "project_library_retention_approved"},
             "policy": {"maximum_review_cost_usd": 4, "pose_input_cost_usd": .01, "refined_splat_cost_usd": 5},
             "remaining_cost_usd": 17.62}
    (root / "configuration.json").write_text(json.dumps(saved))
    (root / "selection_binding.json").write_text("{}")
    monkeypatch.setattr(pipeline, "load_configuration", lambda *_, **__: json.loads(json.dumps(saved)))
    monkeypatch.setattr(pose_api, "atlas_api_config", lambda: ("url", "test-non-secret"))
    context = {"capture_id": "c", "context_digest": "ctx"}
    ran = []
    monkeypatch.setattr(pipeline, "_run_splat_rebuild", lambda **kw: ran.append(kw) or {"status": "processing"})
    (root / rebuilds.REQUEST_FILE).write_text("{}")
    assert pipeline.run_atlas_pipeline(source_video=video, output_root=root, task_context=context) == {
        "status": "processing"}
    assert ran[0]["job_dir"] == root and ran[0]["policy"]["refined_splat_cost_usd"] == 5


@pytest.mark.parametrize("appearance", ["stopped", "refined"])
def test_a_rebuild_scene_skips_completion_and_buys_the_mesh_only_on_its_new_refined_splat(tmp_path, monkeypatch,
                                                                                         appearance):
    from blueprint_pipeline import atlas_depth_splat as splats, atlas_mesh_registration as registration
    from blueprint_pipeline import atlas_result_bundle as bundles, atlas_source_pose_inputs as source_inputs
    from blueprint_pipeline import paid_resource_allocator as allocator, website_atlas_pipeline as pipeline
    source_root = tmp_path / "pose/full-view-source-v1"
    source_root.mkdir(parents=True)
    (source_root / "atlas_posed_rgbd_result.json").write_text("{}")
    (source_root / "scene_measurements.json").write_text("{}")
    (tmp_path / "atlas_scene_review.json").write_text("{}")
    (tmp_path / "splat").mkdir()
    (tmp_path / "splat/atlas_depth_splat_manifest.json").write_text(json.dumps(
        {"digest": "draft", "splat_kind": "depth_fused_draft"}))
    monkeypatch.setattr(pipeline, "_source_pose", lambda **_: {"status": "completed"})
    monkeypatch.setattr(source_inputs, "prepare_source_pose", lambda _: (source_root, {}))
    monkeypatch.setattr(splats, "load_posed_rgbd", lambda _: ({"digest": "pose"}, [{"frame_id": "a"}]))
    monkeypatch.setattr(allocator, "run_sponsored_website_atlas_automation",
                        lambda **_: pytest.fail("a rebuild buys no completion"))
    refined = {"digest": "sha256:" + "6" * 64, "splat_kind": "atlas_refined"}
    seen = []
    monkeypatch.setattr(pipeline, "_preserved_appearance", lambda **kw: seen.append(kw) or (
        {"status": "needs_input", "stage": "refined_splat", "reason": "atlas_splat_rebuild_pose_refinement_not_accepted"}
        if appearance == "stopped" else
        {"status": "completed", "basis": "atlas_refined_splat", "splat_root": str(tmp_path / "refined_splat-minimal"),
         "splat_manifest": refined}))
    meshes = []
    monkeypatch.setattr(splats, "splats2mesh_attempt", lambda root, output: ({"request": {}, "uploads": {}}, output))

    def mesh(**kw):
        meshes.append(kw)
        kw["output_root"].mkdir(parents=True)
        (kw["output_root"] / "result.json").write_text(json.dumps({"outputs": [{"role": "mesh", "path": "m.glb"}]}))
        return {"status": "completed"}
    monkeypatch.setattr(allocator, "run_sponsored_website_atlas_task", mesh)
    monkeypatch.setattr(registration, "register_atlas_mesh", lambda **_: None)
    monkeypatch.setattr(bundles, "export_pose_bundle", lambda **_: {"zip_path": "rebuilt.zip"})
    result = pipeline._preserved_scene(pose_root=tmp_path / "pose", output_root=tmp_path, inputs={},
        review={"subject_origin": "no_manipulated_subject", "automation": {"objects": [], "removal_instruction": ""},
                "uncaptured_regions": []},
        policy={"mesh_export_cost_usd": .5, "rgbd_view_cost_usd": .05, "refined_splat_cost_usd": 5,
                "rate_card_reference": "r", "reviewed_by": "r", "reviewed_on": "r"},
        task_context={"context_digest": "t"}, source_digest="s", configuration={}, check=lambda *_: None,
        experimental=False, max_polls=1, model="m", rebuild=True)
    assert seen[0]["rebuild"] is True and seen[0]["source_root"] == source_root
    if appearance == "stopped":
        assert result["reason"] == "atlas_splat_rebuild_pose_refinement_not_accepted"
        assert not meshes and not (tmp_path / "result.json").exists()
        return
    assert len(meshes) == 1 and meshes[0]["output_root"] == tmp_path / "mesh-refined" / refined["digest"][7:23]
    assert result["appearance"]["basis"] == "atlas_refined_splat" and result["generated_completion_performed"] is False
    assert "scene_completion" not in result and result["download"] == {"zip_path": "rebuilt.zip"}
