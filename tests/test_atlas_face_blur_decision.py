"""Faces are blurred unless the capture's sealed grant names a Privacy version on or after 2026-11-17.1
(owner decision 2026-11-17)."""
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

# The six versions the hand-kept list named before the one rule replaced it (C-09); each decides
# exactly as it did then: not blurred.
LISTED = ["2026-11-17.1", "2026-11-18.1", "2026-11-19.1", "2026-11-20.1", "2026-11-21.1", "2026-11-22.1"]
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


def test_one_rule_replaces_the_list():
    assert privacy.FIRST_NO_FACE_BLUR_PRIVACY_VERSION == "2026-11-17.1"
    assert not hasattr(privacy, "NO_FACE_BLUR_PRIVACY_VERSIONS")


TODAY = datetime(2026, 11, 23, tzinfo=timezone.utc).date()


@pytest.mark.parametrize("version,expected", [
    # Every version the old list named: identical (not blurred).
    *[(version, True) for version in LISTED],
    # A later Privacy version needs no Pipeline change.
    ("2026-11-17.2", True), ("2026-11-23.1", True), ("2026-11-24.1", True),
    # Earlier versions blur.
    ("2026-11-17.0", False), ("2026-11-16.9", False), ("2026-11-13.1", False), ("2025-12-31.1", False),
    # Unknown: dated after tomorrow (UTC), or not a real calendar day.
    ("2026-11-25.1", False), ("2999-01-01.1", False), ("2026-02-30.1", False), ("2026-13-01.1", False),
    # Missing or unparseable.
    (None, False), ("", False), (" 2026-11-17.1", False), ("2026-11-17", False), ("2026-11-17.12345", False),
    ("2026-11-17.x", False), (20261117, False), (["2026-11-17.1"], False),
])
def test_the_one_rule_fails_to_the_private_side(version, expected):
    assert privacy.is_no_face_blur_version(version, today=TODAY) is expected


@pytest.mark.parametrize("version", ["absent", None, "", "2026-11-13.1", "2026-11-12.2", "2999-01-01.1",
                                     "2026-11-17.0", "2026-02-30.1", " 2026-11-17.1", 20261117, ["2026-11-17.1"]])
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


@pytest.mark.parametrize("layout", ["no_pipeline", "not_under_captures", "not_under_scenes", "nearest_pipeline_has_none"])
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
    elif layout == "not_under_scenes":
        pipeline = tmp_path / "bucket/other/site-job/captures/walkthrough-job/pipeline"
        pipeline.mkdir(parents=True)
        _grant(pipeline, LISTED[0])
        _authority(pipeline, LISTED[0])
        root = pipeline / "atlas"
        assert privacy.face_blur_decision(root) == (True, privacy.screen_identity())
        assert not (pipeline / privacy.DECISION_FILE).exists()  # never sealed
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
    pipeline, path, review, _, output, calls = video_job
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


@pytest.mark.parametrize("version,stops", [("2026-11-12.2", True), ("absent", True), (LISTED[0], False)])
def test_a_missing_detector_stops_only_a_job_that_requires_blurring(tmp_path, monkeypatch, version, stops):
    import cv2
    pipeline = _pipeline(tmp_path)
    path, review, _ = _survey(tmp_path, pipeline)
    monkeypatch.setattr(video, "_features", lambda *_: None)
    monkeypatch.setattr(video, "select_coverage_views", lambda **_: ([0, 1, 2], []))
    monkeypatch.delattr(cv2, "FaceDetectorYN")
    _authority(pipeline, version)
    output = pipeline / "atlas/pose"
    if stops:
        with pytest.raises(ValueError, match="atlas_privacy_screen_unavailable"):
            video.prepare_video_inputs(survey_path=path, review=review, output_root=output)
    else:
        assert video.prepare_video_inputs(survey_path=path, review=review, output_root=output) is not None


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
    _grant(pipeline, "2026-11-13.1")
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
    # Retained unblurred task views over a parent that recorded no screen: refused on their own record.
    bare_parent = deepcopy(original)
    bare_parent["binding"].pop("privacy_screen")
    _grant(pipeline, LISTED[0])
    _, views = source_inputs.task_source_inputs(pose_root, bare_parent)
    assert views["task_view_privacy_screen"] == identity
    _grant(pipeline, "2026-11-13.1")
    with pytest.raises(ValueError, match="atlas_task_source_inputs_changed"):
        source_inputs.task_source_inputs(pose_root, bare_parent)


def test_a_blurred_parent_keeps_its_added_task_views_blurred_under_a_listed_version(tmp_path, monkeypatch):
    from blueprint_pipeline import atlas_source_pose_inputs as source_inputs
    blurred = {**privacy.screen_identity(), "blurred_faces_by_frame": {"a": 0}}
    pipeline, pose_root, original, _ = _task_source_job(tmp_path, monkeypatch, original_privacy=blurred)
    _grant(pipeline, LISTED[0])
    calls = []
    monkeypatch.setattr(privacy, "blur_faces", _blur(calls))
    root, result = source_inputs.task_source_inputs(pose_root, original)
    assert calls == [(4, 4)] and (np.asarray(Image.open(root / "views/task.png")) == 0).all()
    assert result["task_view_privacy_screen"] == {**privacy.screen_identity(), "blurred_faces_by_frame": {"task": 1}}


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
    # The sealed decision said unblurred: the job stops before any paid step, record or upload.
    args2["planner_runner"] = lambda **_: pytest.fail("a paid step ran")
    with pytest.raises(ValueError, match="atlas_privacy_screen_decision_changed"):
        recovery.recover_recorded_sources(**args2)
    assert _recover_inputs_unchanged(args2)


def _recover_inputs_unchanged(args):
    inputs = json.loads((args["root"] / "recovery_inputs.json").read_text())
    return inputs["binding"]["privacy_screen"] == privacy.no_blur_identity(LISTED[0])


# The Atlas authority: on disk before the first frame leaves ------------------------------------

def _authority(pipeline, version="absent", *, seal=True, **changes):
    """A website Atlas authority shaped as the test job's, sealed as the website seals it (RFC 8785)."""
    from blueprint_pipeline.decision_evidence_contracts import cross_runtime_canonical_digest
    sha = "sha256:" + "d" * 64
    authority = {
        "schema_version": "website_atlas_authorization.v2", "request_id": "job",
        "capture_id": pipeline.parent.name, "source_video_digest": "sha256:" + "a" * 64,
        "source_video_identity": {"kind": "website_browser_video", "object_name": "o", "generation": "1",
                                  "crc32c": "x"},
        "capture_rights_digest": sha, "policy_digest": sha, "provider_terms_reference": sha,
        "planner": {"provider": "openai", "provider_terms_reference": sha, "private_processing_authorized": True,
                    "provider_training_authorized": False, "approval_reference": "policy"},
        "accepted_by": "owner", "approval_reference": "policy", "grant_basis": "automatic_consented_preview",
        "private_processing_authorized": True, "provider_training_authorized": False,
        "provider_retention": "account_does_not_retain", "maximum_cost_usd": 40.5, "maximum_requests": 32,
        "created_at_epoch": 1794936171, "expires_at_epoch": 1795540971, "claim_scope": "development_only"}
    if version != "absent":
        authority["privacy_policy_version"] = version
    authority.update(changes)
    if seal:
        authority["authorization_digest"] = cross_runtime_canonical_digest(
            authority, digest_field="authorization_digest")
    configuration = {"schema_version": "website_atlas_configuration.v1", "source_video_digest": "sha256:" + "a" * 64,
                     "task_context_digest": "sha256:" + "c" * 64, "dispatch_authorized": False,
                     "policy": {"maximum_review_cost_usd": 1}, "authority": authority}
    path = pipeline / privacy.CONFIGURATION_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(configuration))
    return configuration


@pytest.mark.parametrize("version", LISTED)
def test_the_authority_alone_decides_before_the_grant_exists(tmp_path, version):
    pipeline = _pipeline(tmp_path)
    _authority(pipeline, version)
    assert privacy.face_blur_required(pipeline / "atlas/pose") == (False, privacy.no_blur_identity(version))


@pytest.mark.parametrize("authority,grant,skips", [
    (LISTED[0], LISTED[0], True),          # both name the same listed version
    ("absent", LISTED[0], True),           # an authority issued before the field: the grant decides, as before
    (LISTED[0], "absent", True),           # a grant without the field does not contradict
    (LISTED[0], LISTED[1], False),         # disagreement blurs
    (LISTED[0], "2026-11-13.1", False),
    ("2026-11-13.1", LISTED[0], False),
    ("2026-11-13.1", "absent", False),
    ("2999-01-01.1", "absent", False),  # dated after tomorrow: unknown, so it blurs
    ("absent", "absent", False),
])
def test_authority_and_grant_together(tmp_path, authority, grant, skips):
    pipeline = _pipeline(tmp_path)
    _authority(pipeline, authority)
    _grant(pipeline, grant)
    assert privacy.face_blur_required(pipeline / "atlas")[0] is (not skips)


@pytest.mark.parametrize("case", ["bad_digest", "unsealed", "canonical_python_digest_only", "schema", "configuration_schema",
                                  "owned_configuration", "other_capture", "other_video", "no_request",
                                  "malformed_short", "malformed_number", "malformed_empty", "malformed_spaced",
                                  "authority_not_object", "not_json", "symlink", "directory"])
def test_an_unverified_authority_blurs_even_when_the_grant_lists_a_version(tmp_path, case):
    pipeline = _pipeline(tmp_path)
    _grant(pipeline, LISTED[0])
    path = pipeline / privacy.CONFIGURATION_FILE
    if case == "bad_digest":
        configuration = _authority(pipeline, LISTED[0])
        configuration["authority"]["maximum_cost_usd"] = 99
        path.write_text(json.dumps(configuration))
    elif case == "unsealed":
        _authority(pipeline, LISTED[0], seal=False)
    elif case == "canonical_python_digest_only":
        configuration = _authority(pipeline, LISTED[0], seal=False, maximum_cost_usd=40.0)
        configuration["authority"]["authorization_digest"] = canonical_digest(
            configuration["authority"], digest_field="authorization_digest")
        path.write_text(json.dumps(configuration))
    elif case == "schema":
        _authority(pipeline, LISTED[0], schema_version="website_atlas_authorization.v3")
    elif case == "configuration_schema":
        configuration = _authority(pipeline, LISTED[0])
        configuration["schema_version"] = "owned_capture_atlas_configuration.v1"
        path.write_text(json.dumps(configuration))
    elif case == "owned_configuration":
        _authority(pipeline, LISTED[0], schema_version="owned_capture_atlas_authorization.v1")
    elif case == "other_capture":
        _authority(pipeline, LISTED[0], capture_id="walkthrough-other")
    elif case == "other_video":
        _authority(pipeline, LISTED[0], source_video_digest="sha256:" + "b" * 64)
    elif case == "no_request":
        _authority(pipeline, LISTED[0], request_id="")
    elif case.startswith("malformed"):
        value = {"malformed_short": "2026-11-17", "malformed_number": 20261117, "malformed_empty": "",
                 "malformed_spaced": "2026-11-17.1 "}[case]
        _authority(pipeline, value)
    elif case == "authority_not_object":
        configuration = _authority(pipeline, LISTED[0])
        configuration["authority"] = [configuration["authority"]]
        path.write_text(json.dumps(configuration))
    elif case == "not_json":
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{")
    elif case == "symlink":
        elsewhere = _pipeline(tmp_path / "elsewhere")
        _authority(elsewhere, LISTED[0], capture_id="walkthrough-job")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.symlink_to(elsewhere / privacy.CONFIGURATION_FILE)
    elif case == "directory":
        path.mkdir(parents=True)
    assert privacy.face_blur_required(pipeline / "atlas") == (True, privacy.screen_identity())


def test_an_authority_without_the_field_verifies_and_one_with_it_verifies(tmp_path):
    pipeline = _pipeline(tmp_path)
    without = _authority(pipeline)
    assert "privacy_policy_version" not in without["authority"]
    assert privacy.atlas_authority_privacy_version(without, capture_id="walkthrough-job") is None
    with_field = _authority(pipeline, "2026-11-13.1")
    assert privacy.atlas_authority_privacy_version(with_field, capture_id="walkthrough-job") == "2026-11-13.1"
    # The field is inside the sealed bytes: the two authorities differ only by it and by their digests.
    assert {k: v for k, v in with_field["authority"].items() if k not in {"privacy_policy_version",
            "authorization_digest"}} == {k: v for k, v in without["authority"].items() if k != "authorization_digest"}
    assert with_field["authority"]["authorization_digest"] != without["authority"]["authorization_digest"]


def _configuration_pair(tmp_path, stored_version, current_version):
    from blueprint_pipeline import website_atlas_pipeline as pipeline_module
    pipeline = _pipeline(tmp_path)
    stored = _authority(pipeline, stored_version)
    current = _authority(pipeline, current_version)
    return pipeline_module, stored, current


@pytest.mark.parametrize("version", ["absent", LISTED[0]])
def test_a_resumed_job_keeps_its_authority_with_or_without_the_field(tmp_path, version):
    module, stored, current = _configuration_pair(tmp_path, version, version)
    assert module.resumed_policy(stored, current) == {"policy": current["policy"]}
    # A raised limit is still accepted with the field present, as on main.
    raised = deepcopy(current)
    raised["authority"]["maximum_cost_usd"] = 60
    assert module.resumed_policy(stored, raised) == {"policy": raised["policy"]}


@pytest.mark.parametrize("stored,current", [("absent", LISTED[0]), (LISTED[0], "absent"), (LISTED[0], LISTED[1])])
def test_a_version_added_to_or_changed_in_an_existing_authority_is_refused_on_resume(tmp_path, stored, current):
    module, stored_configuration, current_configuration = _configuration_pair(tmp_path, stored, current)
    with pytest.raises(ValueError, match="atlas_runtime_authority_or_policy_changed"):
        module.resumed_policy(stored_configuration, current_configuration)


@pytest.mark.parametrize("version,blurred", [*[(v, False) for v in LISTED], ("2026-11-13.1", True),
                                              ("absent", True), ("2999-01-01.1", True)])
def test_the_real_order_configuration_present_grant_absent(video_job, monkeypatch, version, blurred):
    """As on the host (2026-11-17 test job): configuration.json before the first frame, the grant ~50 min later."""
    pipeline, path, review, frames, output, calls = video_job
    _authority(pipeline, version)
    assert not (pipeline / privacy.GRANT_FILE).exists()
    first = video.prepare_video_inputs(survey_path=path, review=review, output_root=output)
    views = _views(output)
    record = json.loads((output / video.GEOMETRY_INPUTS_FILE).read_text())["binding"]["privacy_screen"]
    if blurred:
        assert len(calls) == 3 and all((view == 0).all() for view in views.values())
        assert record == {**privacy.screen_identity(), "blurred_faces_by_frame": {"0": 1, "1": 1, "2": 1}}
    else:
        assert calls == []
        for row in frames:
            assert np.array_equal(views[row["frame_id"]], np.asarray(Image.open(row["source_image_path"])))
        assert record == privacy.no_blur_identity(version)
    # The grant arrives later carrying the same version: the retained inputs are reused unchanged.
    _grant(pipeline, version)
    _no_rebuild(monkeypatch)
    assert video.prepare_video_inputs(survey_path=path, review=review, output_root=output) == first


# The sealed decision ------------------------------------------------------------------------------

def _seal(pipeline):
    return json.loads((pipeline / privacy.DECISION_FILE).read_text())


@pytest.mark.parametrize("version,required", [(LISTED[0], False), ("2026-11-13.1", True), ("absent", True)])
def test_the_first_decision_is_sealed_and_kept(tmp_path, version, required):
    pipeline = _pipeline(tmp_path)
    _authority(pipeline, version)
    first = privacy.face_blur_decision(pipeline / "atlas/pose")
    assert first[0] is required
    seal = _seal(pipeline)
    assert seal["face_blur_required"] is required
    assert seal["privacy_policy_version"] == (None if required else version)
    before = (pipeline / privacy.DECISION_FILE).read_bytes()
    assert privacy.face_blur_decision(pipeline / "atlas/recorded_source_recovery/x") == first
    assert (pipeline / privacy.DECISION_FILE).read_bytes() == before


def test_a_job_sealed_blurred_stays_blurred_when_a_listed_version_appears_later(tmp_path):
    pipeline = _pipeline(tmp_path)
    assert privacy.face_blur_decision(pipeline / "atlas")[0] is True  # nothing on disk yet
    _authority(pipeline, LISTED[0])
    _grant(pipeline, LISTED[0])
    assert privacy.face_blur_required(pipeline / "atlas")[0] is False
    assert privacy.face_blur_decision(pipeline / "atlas") == (True, privacy.screen_identity())
    assert privacy.unblurred_frames_may_exist(pipeline / "atlas") is False


@pytest.mark.parametrize("case", ["flip", "other_version", "tampered", "not_json", "extra_field", "symlink"])
def test_a_flipped_or_doubtful_seal_blurs_and_counts_as_possibly_unblurred(tmp_path, case):
    pipeline = _pipeline(tmp_path)
    _authority(pipeline, LISTED[0])
    assert privacy.face_blur_decision(pipeline / "atlas")[0] is False
    path = pipeline / privacy.DECISION_FILE
    seal = _seal(pipeline)
    if case == "flip":
        _authority(pipeline, "2026-11-13.1")
    elif case == "other_version":
        _authority(pipeline, LISTED[1])
    elif case == "tampered":
        path.write_text(json.dumps({**seal, "privacy_policy_version": LISTED[1]}))
        _authority(pipeline, LISTED[1])
    elif case == "not_json":
        path.write_text("{")
    elif case == "extra_field":
        value = {**seal, "note": "x"}
        value["digest"] = canonical_digest(value, digest_field="digest")
        path.write_text(json.dumps(value))
    else:
        moved = path.with_name("elsewhere.json")
        path.rename(moved)
        path.symlink_to(moved)
    assert privacy.face_blur_decision(pipeline / "atlas") == (True, privacy.screen_identity())
    assert privacy.unblurred_frames_may_exist(pipeline / "atlas") is True


def test_a_forced_flip_stops_recovery_before_it_records_unblurred_background_views_as_blurred(
        tmp_path, monkeypatch):
    """The background was prepared unblurred under the sealed decision; then the decision flips."""
    pipeline, args, _ = _recovery(tmp_path, monkeypatch)
    _authority(pipeline, LISTED[0])
    assert privacy.face_blur_decision(pipeline / "atlas/pose") == (False, privacy.no_blur_identity(LISTED[0]))
    _authority(pipeline, "2026-11-13.1")  # forced: the reviewer could build no real flip
    monkeypatch.setattr(privacy, "blur_faces", lambda image: pytest.fail("a frame was screened"))
    args["planner_runner"] = lambda **_: pytest.fail("a paid step ran")
    from blueprint_pipeline import atlas_recorded_source_recovery as recovery
    with pytest.raises(ValueError, match="atlas_privacy_screen_decision_changed"):
        recovery.recover_recorded_sources(**args)
    assert not (args["root"] / "recovery_inputs.json").exists()


def test_a_forced_flip_never_reuses_or_relabels_unblurred_video_inputs(video_job, monkeypatch):
    pipeline, path, review, _, output, _ = video_job
    _authority(pipeline, LISTED[0])
    video.prepare_video_inputs(survey_path=path, review=review, output_root=output)
    _authority(pipeline, "2026-11-13.1")
    with pytest.raises(ValueError, match="atlas_retained_video_inputs_invalid"):
        video.prepare_video_inputs(survey_path=path, review=review, output_root=output)


# The legacy bucket copy never writes the deciding records ---------------------------------------

def test_the_legacy_staging_copy_skips_the_records_that_decide_the_blur(tmp_path, monkeypatch):
    from blueprint_pipeline import pubsub_handoff_listener as listener
    from blueprint_pipeline.pubsub_handoff_scene_operations import _stage_handoff_capture_body

    prefix = "scenes/site-job/captures/walkthrough-job"
    names = [f"{prefix}/raw/capture_upload_complete.json", f"{prefix}/raw/manifest.json",
             *(f"{prefix}/{relative}" for relative in sorted(privacy.DECISION_PATHS)),
             f"{prefix}/pipeline/other.json"]
    assert privacy.DECISION_PATHS == {"pipeline/website_scene_sponsorship.json", "pipeline/atlas/configuration.json",
                                      "pipeline/atlas/face_blur_decision.json"}

    class Blob:
        def __init__(self, name):
            self.name = name

    class Client:
        def list_blobs(self, bucket, prefix):
            return [Blob(name) for name in names]

    downloaded = []

    def download(*, downloads, manifest_rows, storage_root, capture_root):
        for blob, destination in downloads:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text("{}")
            downloaded.append(blob.name)
    monkeypatch.setattr(listener, "download_with_reservation", download)
    monkeypatch.setattr(listener, "_staging_manifest_row", lambda blob, name, relative_path: {
        "name": name, "relative_path": relative_path})
    monkeypatch.setattr(listener, "_previous_staging_rows", lambda root, handoff: {})
    monkeypatch.setattr(listener, "_preserve_local_website_derivatives", lambda *a: None)

    class Handoff:
        source_finalize = None
        bucket = "bucket"
        scene_id = "site-job"
        capture_id = "walkthrough-job"
        capture_prefix = prefix
    (tmp_path / "bucket" / prefix).mkdir(parents=True)
    (tmp_path / "bucket" / prefix / "pipeline_handoff.json").write_text("{}")
    root = _stage_handoff_capture_body(listener, Handoff(), storage_root=tmp_path, storage_client=Client())
    assert sorted(downloaded) == sorted(name for name in names
                                        if name[len(prefix) + 1:] not in privacy.DECISION_PATHS)
    for relative in privacy.DECISION_PATHS:
        assert not (root / relative).exists()
    manifest = json.loads((root / listener.STAGING_MANIFEST_FILENAME).read_text())
    assert not {row["relative_path"] for row in manifest["objects"]} & privacy.DECISION_PATHS


def test_the_screen_stops_are_privacy_walls(tmp_path, monkeypatch):
    import cv2
    from blueprint_pipeline.hard_stops import PrivacyRightsStop
    monkeypatch.delattr(cv2, "FaceDetectorYN")
    with pytest.raises(PrivacyRightsStop, match="atlas_privacy_screen_unavailable"):
        privacy.blur_faces(Image.new("RGB", (8, 8)))
    monkeypatch.undo()
    pipeline, args, _ = _recovery(tmp_path, monkeypatch)
    _authority(pipeline, LISTED[0])
    privacy.face_blur_decision(pipeline / "atlas/pose")
    _authority(pipeline, "2026-11-13.1")
    from blueprint_pipeline import atlas_recorded_source_recovery as recovery
    with pytest.raises(PrivacyRightsStop, match="atlas_privacy_screen_decision_changed"):
        recovery.recover_recorded_sources(**args)
