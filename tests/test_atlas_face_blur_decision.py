"""Faces are blurred unless the capture's sealed grant names a listed Privacy version (owner decision 2026-11-17)."""
import json
from copy import deepcopy
from datetime import datetime, timezone

import numpy as np
import pytest
from PIL import Image

from blueprint_pipeline import atlas_privacy_screen as privacy
from blueprint_pipeline import atlas_video_inputs as video
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.local_reconstruction_adapters import _sha256_file

LISTED = sorted(privacy.NO_FACE_BLUR_PRIVACY_VERSIONS)
BLURRED_PIXEL = (0, 0, 0)


def _pipeline(tmp_path, scene="site-job", capture="walkthrough-job"):
    path = tmp_path / "bucket/scenes" / scene / "captures" / capture / "pipeline"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _grant(pipeline, version="absent", *, seal=True, **changes):
    grant = {"schema_version": "website_scene_sponsorship.v1", "sponsor": "blueprint",
             "purpose": "scene_preparation", "request_id": "job",
             "scene_id": pipeline.parent.parent.parent.name, "capture_id": pipeline.parent.name,
             "task_context_digest": "sha256:" + "c" * 64, "max_total_spend_usd": 20}
    if version != "absent":
        grant["privacy_policy_version"] = version
    grant.update(changes)
    if seal:
        grant["authority_digest"] = canonical_digest(grant, digest_field="authority_digest")
    (pipeline / privacy.GRANT_FILE).write_text(json.dumps(grant))
    return grant


def _blur(calls):
    def blur(image):
        calls.append(image.size)
        out = image.copy()
        out.paste(BLURRED_PIXEL, (0, 0, image.width, image.height))
        return out, [[0, 0, 2, 2]]
    return blur


# The decision ---------------------------------------------------------------------------------

@pytest.mark.parametrize("version", LISTED)
def test_a_listed_version_skips_the_blur_and_names_why(tmp_path, version):
    pipeline = _pipeline(tmp_path)
    _grant(pipeline, version)
    root = pipeline / "atlas/pose"
    root.mkdir(parents=True)
    assert privacy.face_blur_required(root) == (
        False, {"method": "no_face_blur", "privacy_policy_version": version})
    # The pipeline directory itself and a deeper rebuild root decide the same.
    assert privacy.face_blur_required(pipeline)[0] is False
    assert privacy.face_blur_required(pipeline / "atlas/splat-rebuild/x/pose")[0] is False


def test_the_listed_set_is_exactly_the_two_candidate_versions():
    assert privacy.NO_FACE_BLUR_PRIVACY_VERSIONS == frozenset({"2026-11-17.1", "2026-11-18.1"})


@pytest.mark.parametrize("version", ["absent", None, "", "2026-11-13.1", "2026-11-12.2", "2026-11-25.1",
                                     "2026-11-17.2", " 2026-11-17.1", 20261117, ["2026-11-17.1"]])
def test_every_other_version_blurs(tmp_path, version):
    pipeline = _pipeline(tmp_path)
    _grant(pipeline, version)
    assert privacy.face_blur_required(pipeline / "atlas") == (True, privacy.screen_identity())


@pytest.mark.parametrize("case", ["bad_digest", "unsealed", "schema", "sponsor", "purpose", "other_capture",
                                  "other_scene", "no_request", "not_object", "not_json", "symlink", "too_large",
                                  "directory", "missing"])
def test_an_unverified_or_absent_grant_blurs(tmp_path, case):
    pipeline = _pipeline(tmp_path)
    path = pipeline / privacy.GRANT_FILE
    if case == "bad_digest":
        grant = _grant(pipeline, LISTED[0])
        grant["max_total_spend_usd"] = 25
        path.write_text(json.dumps(grant))
    elif case == "unsealed":
        _grant(pipeline, LISTED[0], seal=False)
    elif case == "schema":
        _grant(pipeline, LISTED[0], schema_version="website_scene_sponsorship.v2")
    elif case == "sponsor":
        _grant(pipeline, LISTED[0], sponsor="someone")
    elif case == "purpose":
        _grant(pipeline, LISTED[0], purpose="other")
    elif case == "other_capture":
        _grant(pipeline, LISTED[0], capture_id="walkthrough-other")
    elif case == "other_scene":
        _grant(pipeline, LISTED[0], scene_id="site-other")
    elif case == "no_request":
        _grant(pipeline, LISTED[0], request_id=" ")
    elif case == "not_object":
        path.write_text(json.dumps([LISTED[0]]))
    elif case == "not_json":
        path.write_text("{not json")
    elif case == "symlink":
        elsewhere = _pipeline(tmp_path / "elsewhere")
        _grant(elsewhere, LISTED[0], scene_id="site-job", capture_id="walkthrough-job")
        path.symlink_to(elsewhere / privacy.GRANT_FILE)
    elif case == "too_large":
        _grant(pipeline, LISTED[0], padding="x" * 70_000)
    elif case == "directory":
        path.mkdir()
    assert privacy.face_blur_required(pipeline / "atlas") == (True, privacy.screen_identity())


@pytest.mark.parametrize("layout", ["no_pipeline", "not_under_captures", "nearest_pipeline_has_none"])
def test_a_path_that_is_not_a_website_capture_blurs(tmp_path, layout):
    if layout == "no_pipeline":
        root = tmp_path / "internal/atlas/pose"
        (tmp_path / "internal").mkdir()
        (tmp_path / "internal" / privacy.GRANT_FILE).write_text("{}")
    elif layout == "not_under_captures":
        pipeline = tmp_path / "owned/job/pipeline"
        pipeline.mkdir(parents=True)
        _grant(pipeline, LISTED[0], scene_id="owned", capture_id="job")
        root = pipeline / "atlas"
    else:
        pipeline = _pipeline(tmp_path)
        _grant(pipeline, LISTED[0])
        root = pipeline / "atlas/pipeline/pose"
    assert privacy.face_blur_required(root) == (True, privacy.screen_identity())


def test_an_error_while_deciding_blurs(tmp_path, monkeypatch):
    pipeline = _pipeline(tmp_path)
    _grant(pipeline, LISTED[0])

    def broken(_root):
        raise OSError("unreadable")
    monkeypatch.setattr(privacy, "_listed_version", broken)
    assert privacy.face_blur_required(pipeline) == (True, privacy.screen_identity())


def test_retained_blurred_work_keeps_blurring_and_unblurred_work_never_lowers_a_required_job(tmp_path):
    pipeline = _pipeline(tmp_path)
    _grant(pipeline, LISTED[0])
    held_blurred = {**privacy.screen_identity(), "blurred_faces_by_frame": {"0": 0}}
    assert privacy.screen_for(pipeline, held_blurred) == (True, privacy.screen_identity())
    assert privacy.screen_for(pipeline, privacy.no_blur_identity(LISTED[0])) == (
        False, privacy.no_blur_identity(LISTED[0]))
    _grant(pipeline, "2026-11-13.1")
    assert privacy.screen_for(pipeline, privacy.no_blur_identity(LISTED[0])) == (True, privacy.screen_identity())


# Call site 1: atlas_video_inputs.prepare_video_inputs -----------------------------------------

def _survey(tmp_path, pipeline):
    source = pipeline / "atlas/survey"
    source.mkdir(parents=True)
    frames = []
    for index in range(3):
        path = source / f"{index}.png"
        Image.new("RGB", (64, 48), (index * 40 + 20, 80, 160)).save(path)
        frames.append({"frame_id": str(index), "timestamp_seconds": index,
                       "source_image_path": str(path), "source_image_digest": _sha256_file(path),
                       "sharpness": 100, "clipped_fraction": 0})
    survey = {"schema_version": video.SURVEY_SCHEMA, "heldout_pixels_included": False,
              "frames": frames, "display_rotation_degrees": 0,
              "source_video_digest": "sha256:" + "a" * 64, "sample_count": 4, "heldout_frame_count": 1}
    survey["digest"] = canonical_digest(survey, digest_field="digest")
    path = source / "survey.json"
    path.write_text(json.dumps(survey))
    review = {"schema_version": video.REVIEW_SCHEMA, "survey_digest": survey["digest"],
              "moving_people_excluded": True, "same_scene_state": True, "reviewed_by": "fixture",
              "scene_state": "unchanged", "static_intervals_seconds": [[0, 2]], "anchor_frame_id": "0",
              "coverage_requirements": [
                  {"role": "task_object", "label": "source", "required_frame_ids": ["0"]},
                  {"role": "support_surface", "label": "destination", "required_frame_ids": ["1"]},
                  {"role": "surrounding_scene", "label": "room", "required_frame_ids": ["2"]}],
              "selection_budget": {"maximum_pose_cost_usd": 1, "input_cost_usd": .01,
                                   "rate_card_reference": "fixture", "reviewed_by": "fixture",
                                   "reviewed_on": datetime.now(timezone.utc).date().isoformat()}}
    return path, review, frames


@pytest.fixture
def video_job(tmp_path, monkeypatch):
    pipeline = _pipeline(tmp_path)
    path, review, frames = _survey(tmp_path, pipeline)
    monkeypatch.setattr(video, "_features", lambda *_: None)
    monkeypatch.setattr(video, "select_coverage_views", lambda **_: ([0, 1, 2], []))
    calls = []
    monkeypatch.setattr(privacy, "blur_faces", _blur(calls))
    output = pipeline / "atlas/pose"
    return pipeline, path, review, frames, output, calls


def _views(output):
    return {p.stem: np.asarray(Image.open(p)) for p in sorted((output / "atlas_source_views").glob("*.png"))}


def test_video_inputs_are_blurred_when_required(video_job):
    pipeline, path, review, frames, output, calls = video_job
    _grant(pipeline, "2026-11-13.1")
    video.prepare_video_inputs(survey_path=path, review=review, output_root=output)
    assert len(calls) == 3
    assert all((view == 0).all() for view in _views(output).values())
    inputs = json.loads((output / video.GEOMETRY_INPUTS_FILE).read_text())
    expected = {**privacy.screen_identity(), "blurred_faces_by_frame": {"0": 1, "1": 1, "2": 1}}
    assert inputs["binding"]["privacy_screen"] == expected
    assert json.loads((output / "atlas_view_selection.json").read_text())["privacy_screen"] == expected


@pytest.mark.parametrize("version", LISTED)
def test_video_inputs_go_out_untouched_under_a_listed_version(video_job, version):
    pipeline, path, review, frames, output, calls = video_job
    _grant(pipeline, version)
    video.prepare_video_inputs(survey_path=path, review=review, output_root=output)
    assert calls == []
    for row in frames:
        assert np.array_equal(_views(output)[row["frame_id"]], np.asarray(Image.open(row["source_image_path"])))
    identity = {"method": "no_face_blur", "privacy_policy_version": version}
    inputs = json.loads((output / video.GEOMETRY_INPUTS_FILE).read_text())
    assert inputs["binding"]["privacy_screen"] == identity
    assert json.loads((output / "atlas_view_selection.json").read_text())["privacy_screen"] == identity


def test_a_missing_detector_stops_only_a_job_that_requires_blurring(video_job, monkeypatch):
    import cv2
    pipeline, path, review, _, output, _ = video_job
    monkeypatch.undo()
    monkeypatch.setattr(video, "_features", lambda *_: None)
    monkeypatch.setattr(video, "select_coverage_views", lambda **_: ([0, 1, 2], []))
    monkeypatch.delattr(cv2, "FaceDetectorYN")
    _grant(pipeline, "2026-11-12.2")
    with pytest.raises(ValueError, match="atlas_privacy_screen_unavailable"):
        video.prepare_video_inputs(survey_path=path, review=review, output_root=output)
    _grant(pipeline, LISTED[0])
    result = video.prepare_video_inputs(survey_path=path, review=review, output_root=output)
    assert result is not None


def _no_rebuild(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("completed input preparation was recomputed")
    for name in ("_features", "select_coverage_views", "add_distinct_views"):
        monkeypatch.setattr(video, name, fail)
    monkeypatch.setattr(privacy, "blur_faces", fail)


@pytest.mark.parametrize("now", ["2026-11-13.1", *LISTED, "absent"])
def test_retained_blurred_inputs_are_reused_by_any_job(video_job, monkeypatch, now):
    pipeline, path, review, _, output, _ = video_job
    first = video.prepare_video_inputs(survey_path=path, review=review, output_root=output)  # no grant: blurred
    before = {p: p.read_bytes() for p in output.rglob("*") if p.is_file()}
    _grant(pipeline, now)
    _no_rebuild(monkeypatch)
    assert video.prepare_video_inputs(survey_path=path, review=review, output_root=output) == first
    assert {p: p.read_bytes() for p in before} == before


def test_retained_unblurred_inputs_are_reused_under_the_same_listed_version(video_job, monkeypatch):
    pipeline, path, review, _, output, _ = video_job
    _grant(pipeline, LISTED[0])
    first = video.prepare_video_inputs(survey_path=path, review=review, output_root=output)
    _no_rebuild(monkeypatch)
    assert video.prepare_video_inputs(survey_path=path, review=review, output_root=output) == first


@pytest.mark.parametrize("now", ["2026-11-13.1", "absent", LISTED[1], "unsealed", "removed"])
def test_retained_unblurred_inputs_are_never_reused_by_a_job_that_requires_blurring(video_job, monkeypatch, now):
    pipeline, path, review, _, output, _ = video_job
    _grant(pipeline, LISTED[0])
    video.prepare_video_inputs(survey_path=path, review=review, output_root=output)
    if now == "removed":
        (pipeline / privacy.GRANT_FILE).unlink()
    elif now == "unsealed":
        _grant(pipeline, LISTED[0], seal=False)
    else:
        _grant(pipeline, now)
    with pytest.raises(ValueError, match="atlas_retained_video_inputs_invalid"):
        video.prepare_video_inputs(survey_path=path, review=review, output_root=output)
    # A partial handoff (no selection plan) is rebuilt blurred and refused against the unblurred record.
    (output / "atlas_view_selection.json").unlink()
    with pytest.raises(ValueError, match="atlas_geometry_inputs_identity_conflict"):
        video.prepare_video_inputs(survey_path=path, review=review, output_root=output)


def test_a_partial_blurred_handoff_keeps_blurring_under_a_listed_version(video_job):
    pipeline, path, review, _, output, calls = video_job
    first = video.prepare_video_inputs(survey_path=path, review=review, output_root=output)
    (output / "atlas_view_selection.json").unlink()
    _grant(pipeline, LISTED[0])
    assert video.prepare_video_inputs(survey_path=path, review=review, output_root=output) == first
    assert len(calls) == 6


# Call site 2: atlas_source_pose_inputs.task_source_inputs --------------------------------------

def _task_source_job(tmp_path, monkeypatch, original_privacy=None):
    from blueprint_pipeline import website_scene_geometry
    pipeline = _pipeline(tmp_path)
    pose_root = pipeline / "atlas/pose"
    pose_root.mkdir(parents=True)
    images = tmp_path / "images"
    images.mkdir()

    def row(key, time):
        path = images / f"{key}.png"
        Image.new("RGB", (4, 4), (90, 120, 150)).save(path)
        return {"frame_id": key, "timestamp_seconds": time, "source_image_path": str(path), "image_path": str(path),
                "source_image_digest": "d", "image_digest": "d", "display_rotation_degrees": 0}
    binding = {"source_video_digest": "sha256:v"}
    if original_privacy is not None:
        binding["privacy_screen"] = original_privacy
    original = {"binding": binding, "frames": [row("a", 9.0)], "digest": "sha256:o"}
    candidates = [row("task", 1.0)]
    clean = pipeline / "clean_plate"
    (clean / "source_geometry/worker_inputs").mkdir(parents=True)
    sources = {"schema_version": "website_source_frames.v1", "binding": {"source_video_digest": "sha256:v"},
               "geometry_input_digest": "sha256:g"}
    sources["digest"] = canonical_digest(sources, digest_field="digest")
    (clean / "clean_plate_stage_manifest.json").write_text(json.dumps({"source_frames": sources}))
    (clean / "source_geometry/worker_inputs/geometry_inputs.json").write_text(json.dumps(
        {"digest": "sha256:g", "binding": sources["binding"], "heldout_pixels_included": False}))
    monkeypatch.setattr(website_scene_geometry, "_bound_frames",
                        lambda value, root, geometry: candidates if value.get("digest") == "sha256:g"
                        else value["frames"])
    return pipeline, pose_root, original, candidates


def test_task_views_are_blurred_when_required(tmp_path, monkeypatch):
    from blueprint_pipeline import atlas_source_pose_inputs as source_inputs
    pipeline, pose_root, original, _ = _task_source_job(tmp_path, monkeypatch)
    _grant(pipeline, "2026-11-25.1")
    calls = []
    monkeypatch.setattr(privacy, "blur_faces", _blur(calls))
    root, result = source_inputs.task_source_inputs(pose_root, original)
    assert calls == [(4, 4)]
    assert (np.asarray(Image.open(root / "views/task.png")) == 0).all()
    assert result["task_view_privacy_screen"] == {**privacy.screen_identity(), "blurred_faces_by_frame": {"task": 1}}


@pytest.mark.parametrize("version", LISTED)
def test_task_views_go_out_untouched_under_a_listed_version(tmp_path, monkeypatch, version):
    from blueprint_pipeline import atlas_source_pose_inputs as source_inputs
    pipeline, pose_root, original, candidates = _task_source_job(tmp_path, monkeypatch)
    _grant(pipeline, version)
    monkeypatch.setattr(privacy, "blur_faces", lambda image: pytest.fail("blurred a listed version's frame"))
    root, result = source_inputs.task_source_inputs(pose_root, original)
    assert np.array_equal(np.asarray(Image.open(root / "views/task.png")),
                          np.asarray(Image.open(candidates[0]["source_image_path"])))
    assert result["task_view_privacy_screen"] == {"method": "no_face_blur", "privacy_policy_version": version}


def test_retained_unblurred_task_views_or_background_are_never_reused_by_a_job_that_requires_blurring(
        tmp_path, monkeypatch):
    from blueprint_pipeline import atlas_source_pose_inputs as source_inputs
    identity = privacy.no_blur_identity(LISTED[0])
    pipeline, pose_root, original, _ = _task_source_job(tmp_path, monkeypatch, original_privacy=identity)
    _grant(pipeline, LISTED[0])
    monkeypatch.setattr(privacy, "blur_faces", _blur([]))
    root, first = source_inputs.task_source_inputs(pose_root, original)
    assert source_inputs.task_source_inputs(pose_root, original) == (root, first)
    _grant(pipeline, "2026-11-13.1")
    with pytest.raises(ValueError, match="atlas_task_source_inputs_changed"):
        source_inputs.task_source_inputs(pose_root, original)
    # Retained unblurred task views over a blurred parent: refused on their own record.
    blurred_parent = deepcopy(original)
    blurred_parent["binding"]["privacy_screen"] = {**privacy.screen_identity(), "blurred_faces_by_frame": {"a": 0}}
    _grant(pipeline, LISTED[0])
    root, views = source_inputs.task_source_inputs(pose_root, blurred_parent)
    assert views["task_view_privacy_screen"] == identity
    (pipeline / privacy.GRANT_FILE).unlink()
    with pytest.raises(ValueError, match="atlas_task_source_inputs_changed"):
        source_inputs.task_source_inputs(pose_root, blurred_parent)


def test_retained_blurred_task_views_are_reused_by_any_job(tmp_path, monkeypatch):
    from blueprint_pipeline import atlas_source_pose_inputs as source_inputs
    pipeline, pose_root, original, _ = _task_source_job(tmp_path, monkeypatch)
    monkeypatch.setattr(privacy, "blur_faces", _blur([]))
    first = source_inputs.task_source_inputs(pose_root, original)
    _grant(pipeline, LISTED[1])
    monkeypatch.setattr(privacy, "blur_faces", lambda image: pytest.fail("rebuilt"))
    assert source_inputs.task_source_inputs(pose_root, original) == first


# Call site 3: atlas_recorded_source_recovery.recover_recorded_sources --------------------------

class _Stop(Exception):
    pass


def _recovery(tmp_path, monkeypatch):
    from blueprint_pipeline import atlas_background_automation as background
    from blueprint_pipeline import atlas_scene_automation
    from tests.test_atlas_recorded_source_recovery import fixture
    tmp_path.mkdir(parents=True, exist_ok=True)
    pipeline = _pipeline(tmp_path)
    args, _, rows = fixture(tmp_path)
    args["root"] = pipeline / "atlas/recorded_source_recovery/abc"
    monkeypatch.setattr(background, "_provider_preflight", lambda: None)

    def stop(inputs):
        raise _Stop()
    monkeypatch.setattr(atlas_scene_automation, "_unposed_source_views", stop)
    return pipeline, args, rows


def _recover(args):
    from blueprint_pipeline import atlas_recorded_source_recovery as recovery
    with pytest.raises(_Stop):
        recovery.recover_recorded_sources(**args)
    return json.loads((args["root"] / "recovery_inputs.json").read_text())


def test_recovered_views_are_blurred_when_required(tmp_path, monkeypatch):
    pipeline, args, _ = _recovery(tmp_path, monkeypatch)
    _grant(pipeline, "2026-11-13.1")
    calls = []
    monkeypatch.setattr(privacy, "blur_faces", _blur(calls))
    inputs = _recover(args)
    assert len(calls) == 1
    assert (np.asarray(Image.open(args["root"] / "later.png")) == 0).all()
    assert inputs["binding"]["privacy_screen"] == {**privacy.screen_identity(),
                                                   "blurred_faces_by_frame": {"later": 1}}


@pytest.mark.parametrize("version", LISTED)
def test_recovered_views_go_out_untouched_under_a_listed_version(tmp_path, monkeypatch, version):
    pipeline, args, rows = _recovery(tmp_path, monkeypatch)
    _grant(pipeline, version)
    monkeypatch.setattr(privacy, "blur_faces", lambda image: pytest.fail("blurred a listed version's frame"))
    inputs = _recover(args)
    assert np.array_equal(np.asarray(Image.open(args["root"] / "later.png")),
                          np.asarray(Image.open(rows[-1]["source_image_path"])))
    assert inputs["binding"]["privacy_screen"] == {"method": "no_face_blur", "privacy_policy_version": version}


def test_recovery_resume_keeps_retained_blurred_views_and_refuses_unblurred_ones(tmp_path, monkeypatch):
    pipeline, args, _ = _recovery(tmp_path, monkeypatch)
    monkeypatch.setattr(privacy, "blur_faces", _blur([]))
    first = _recover(args)  # no grant: blurred
    _grant(pipeline, LISTED[0])
    assert _recover(args) == first  # the retained blurred record still binds; nothing conflicts
    other, args2, _ = _recovery(tmp_path / "second", monkeypatch)
    _grant(other, LISTED[0])
    _recover(args2)
    _grant(other, "2026-11-13.1")
    from blueprint_pipeline import atlas_recorded_source_recovery as recovery
    # The retained unblurred view itself refuses the blurred rebuild, before any record or upload.
    with pytest.raises(ValueError, match="atlas_completion_preservation_inputs_changed"):
        recovery.recover_recorded_sources(**args2)
    assert _recover_inputs_unchanged(args2)


def _recover_inputs_unchanged(args):
    inputs = json.loads((args["root"] / "recovery_inputs.json").read_text())
    return inputs["binding"]["privacy_screen"] == privacy.no_blur_identity(LISTED[0])
