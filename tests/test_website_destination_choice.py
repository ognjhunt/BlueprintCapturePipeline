"""Part 4d (2026-11-12): show the site where the object goes, and let it pick another place, never a wait.

Geometry is 447's retained simulator frame (``tests.test_website_destination_area``): the tray's start box,
the table's support box and the region "the other end of the same table".
"""
# Covers (for impacted-test selection):
#   src/blueprint_pipeline/website_destination_choice.py
#   src/blueprint_pipeline/website_task_target_choice.py
#   src/blueprint_pipeline/website_surface_target_admission.py
import json
import math

import numpy as np
import pytest

from blueprint_pipeline import website_destination_choice as place
from blueprint_pipeline import website_task_target_choice as choice
from blueprint_pipeline.common import write_json
from blueprint_pipeline.decision_evidence_contracts import canonical_digest, cross_runtime_canonical_digest
from tests.test_website_destination_area import SUBJECT_MAX, SUBJECT_MIN, SUPPORT_MAX, SUPPORT_MIN, _choose


def _pipeline(tmp_path):
    pipeline = tmp_path / "scenes/site-x/captures/walkthrough-x/pipeline"
    pipeline.mkdir(parents=True)
    return pipeline


def _offered():
    sealed, record = _choose()
    rows = place.place_candidates(choice=record, subject_min=SUBJECT_MIN, subject_max=SUBJECT_MAX,
                                  support_min=SUPPORT_MIN, support_max=SUPPORT_MAX)
    return sealed, record, rows


def test_the_places_offered_start_with_the_one_in_use_and_stay_off_the_start():
    _, record, rows = _offered()
    assert 2 <= len(rows) <= place.MAX_PLACES
    assert rows[0]["kind"] == "named_region" and rows[0]["centre_xy"] == pytest.approx(record["position_world_m"][:2])
    for row in rows:
        inside = all(SUBJECT_MIN[i] <= row["centre_xy"][i] <= SUBJECT_MAX[i] for i in (0, 1))
        assert not inside and SUPPORT_MIN[0] <= row["centre_xy"][0] <= SUPPORT_MAX[0]
    assert all(math.dist(a["centre_xy"], b["centre_xy"]) >= place.DISTINCT_PLACE_M
               for i, a in enumerate(rows) for b in rows[i + 1:])
    assert len({row["candidate_id"] for row in rows}) == len(rows)


def _question(pipeline, record, rows):
    question = place.build_place_question(preparation={"binding": {}, "destination_choice": record}, candidates=rows,
        object_label="Wooden tray", scene_id="site-x", capture_id="walkthrough-x",
        source_geometry={"digest": "sha256:" + "d" * 64, "binding": {"source_video_digest": "sha256:" + "e" * 64}},
        picture={"frame_id": "f1", "reference_time_ms": 3500, "png_sha256": "sha256:" + "f" * 64})
    write_json(pipeline / choice.QUESTION_DIR / f"{place.TARGET_ID}.question.json", question)
    return question


def _answer(pipeline, question, candidate_id):
    write_json(pipeline / "website_task_context.json", {"context_digest": "sha256:" + "c" * 64})
    value = {"schema_version": choice.ANSWER_SCHEMA, "request_id": "x", "scene_id": "site-x",
             "capture_id": "walkthrough-x", "completion_marker_generation": "1",
             "producer_delivery_key": "sha256:" + "1" * 64, "source_payload_sha256": "sha256:" + "2" * 64,
             "task_context_digest": "sha256:" + "c" * 64, "question_digest": question["question_digest"],
             "target_id": place.TARGET_ID, "candidate_set_digest": question["candidate_set_digest"],
             "guessed_candidate_id": question["guessed_candidate_id"],
             "choice": ({"kind": "confirm_guess", "candidate_id": candidate_id}
                        if candidate_id == question["guessed_candidate_id"]
                        else {"kind": "other_candidate", "candidate_id": candidate_id}),
             "answered_at": "2026-11-12T10:00:00Z"}
    value["answer_digest"] = cross_runtime_canonical_digest(value, digest_field="answer_digest")
    write_json(pipeline / choice.QUESTION_DIR / f"{place.TARGET_ID}.answer.json", value)


def test_the_question_says_where_and_is_kept_out_of_the_object_corrections(tmp_path):
    pipeline = _pipeline(tmp_path)
    _, record, rows = _offered()
    question = _question(pipeline, record, rows)
    assert question["sentence"] == "We will test moving the wooden tray from here to here."
    assert place.read_place_question(pipeline) == question
    # Not an object question: the object corrections and the object choices never see it.
    assert choice.current_questions(pipeline) == [] and choice.corrections(pipeline) == []
    assert choice.correction_binding_digest(pipeline) is None  # unanswered: nothing reruns


def test_the_site_picks_another_place_and_the_preparation_reruns_once_on_it(tmp_path):
    pipeline = _pipeline(tmp_path)
    sealed, record, rows = _offered()
    question = _question(pipeline, record, rows)
    picked = rows[1]
    _answer(pipeline, question, picked["candidate_id"])
    digest = choice.correction_binding_digest(pipeline)
    assert digest is not None and place.place_correction(pipeline)["candidate_id"] == picked["candidate_id"]
    # Owed once, and no longer once a run on it completed.
    assert choice.correction_pending({"status": "completed", "attempt_history": [{}]}, pipeline) == digest
    done = {"status": "completed", "attempt_history": [{choice.CORRECTION_MARK: digest}]}
    assert choice.correction_pending(done, pipeline) is None
    # The preparation seals the site's place: no longer a guess, never moved at submission.
    output = pipeline / "website_scene_preparation" / "revisions" / digest[7:]
    destination, settled = place.settle_place(output, sealed, record, subject_min=SUBJECT_MIN,
                                              subject_max=SUBJECT_MAX, support=(SUPPORT_MIN, SUPPORT_MAX))
    assert destination["position_world_m"][:2] == pytest.approx(picked["centre_xy"])
    assert "radius_m" not in destination  # a picked spot is scored as today's 5 cm disc
    assert settled["basis"] == place.SITE_CHOICE and settled["flags"] == []
    assert settled["site_choice"]["candidate_id"] == picked["candidate_id"]
    assert [row["candidate_id"] for row in settled["candidates"]] == [row["candidate_id"] for row in question["candidates"]]
    from blueprint_pipeline.website_surface_target_admission import submission_relocates_destination
    assert submission_relocates_destination({"destination_choice": settled}, destination,
                                            subject_min=SUBJECT_MIN, subject_max=SUBJECT_MAX) is False


def test_confirming_the_place_changes_nothing(tmp_path):
    pipeline = _pipeline(tmp_path)
    sealed, record, rows = _offered()
    question = _question(pipeline, record, rows)
    _answer(pipeline, question, question["guessed_candidate_id"])
    assert choice.correction_binding_digest(pipeline) is None
    destination, settled = place.settle_place(pipeline / "website_scene_preparation", sealed, record,
        subject_min=SUBJECT_MIN, subject_max=SUBJECT_MAX, support=(SUPPORT_MIN, SUPPORT_MAX))
    assert destination == sealed and settled["basis"] == "named_region"


def _frame(tmp_path):
    """A camera 1.2 m above the table centre, looking straight down; the identity registration."""
    from PIL import Image
    image = tmp_path / "frame.png"
    Image.new("RGB", (640, 480), (90, 90, 90)).save(image)
    centre = [(SUPPORT_MIN[0] + SUPPORT_MAX[0]) / 2, (SUPPORT_MIN[1] + SUPPORT_MAX[1]) / 2, 2.0]
    camera_from_world = np.eye(4)
    camera_from_world[:3, :3] = np.diag([1.0, -1.0, -1.0])
    camera_from_world[:3, 3] = -camera_from_world[:3, :3] @ np.asarray(centre)
    return {"frame_id": "f1", "timestamp_seconds": 3.5, "image_path": str(image), "image_size": [640, 480],
            "camera_from_world": camera_from_world.tolist(),
            "intrinsics": [[300.0, 0, 320.0], [0, 300.0, 240.0], [0, 0, 1]]}


def test_the_picture_outlines_the_start_and_draws_the_area_clipped_to_the_surface(tmp_path):
    from PIL import Image
    pipeline = _pipeline(tmp_path)
    sealed, record, _ = _offered()
    record = {**record, "candidates": _offered()[2]}
    preparation = {"digest": "sha256:" + "9" * 64, "binding": {}, "destination_choice": record, "destination": sealed,
                   "coordinate_frame": {"runtime_to_simulator": np.eye(4).tolist()},
                   "registration": {"source_to_runtime": np.eye(4).tolist()},
                   "authoring_inputs": {"metric_envelope": {"minimum_xyz_m": SUBJECT_MIN, "maximum_xyz_m": SUBJECT_MAX}}}
    edge = SUPPORT_MAX[0] - 0.05  # the surface ends 5 cm short of its box here, as a real table's rounded end

    def covers(points):
        return np.asarray(points)[:, 0] <= edge
    question = place.retain_place_view(pipeline_root=pipeline, preparation=preparation, frames=[_frame(tmp_path)],
        source_geometry={"digest": "sha256:" + "d" * 64}, object_label="Wooden tray", covers=covers)
    folder = pipeline / choice.QUESTION_DIR
    picture = json.loads((folder / place.PICTURE_RECORD).read_text())
    assert picture["frame_id"] == "f1" and picture["private"] is True
    assert question["picture"] == {key: picture[key] for key in ("frame_id", "reference_time_ms", "png_sha256")}
    # Every place offered is drawn with its letter, in the question's order; A is the place in use.
    assert list(picture["letters"]) == list("ABCD")[:len(question["candidates"])] and len(question["candidates"]) >= 2
    assert list(picture["letters"].values()) == [row["candidate_id"] for row in question["candidates"]]
    assert question["guessed_candidate_id"] == picture["letters"]["A"]
    image = np.asarray(Image.open(folder / place.PICTURE_FILE))
    green = (image[:, :, 1] > 150) & (image[:, :, 0] < 80)
    orange = (image[:, :, 0] > 200) & (image[:, :, 1] > 100) & (image[:, :, 1] < 180) & (image[:, :, 2] < 60)
    blue = (image[:, :, 2] > 200) & (image[:, :, 0] < 80) & (image[:, :, 1] > 90) & (image[:, :, 1] < 150)
    assert green.any() and orange.any() and blue.any()
    outline = place.area_outline(sealed["position_world_m"], sealed["radius_m"], sealed["position_world_m"][2],
                                 covers=covers)
    assert max(point[0] for point in outline) <= edge + 1e-9
    public = place.public_place_question(pipeline.parent)
    assert public["sentence"].startswith("We will test moving") and "centre_xy" not in public["candidates"][0]
    assert canonical_digest(question, digest_field="question_digest") == question["question_digest"]


def test_a_place_on_a_guessed_surface_is_flagged_as_a_guess(tmp_path):
    sealed, record, _ = _offered()
    assert record["flags"] == []
    _, settled = place.settle_place(_pipeline(tmp_path) / "website_scene_preparation", sealed, record,
        subject_min=SUBJECT_MIN, subject_max=SUBJECT_MAX, support=(SUPPORT_MIN, SUPPORT_MAX), surface_guessed=True)
    assert settled["flags"] == ["scene_estimate:destination_guessed"] and settled["surface_guess"] == "best_supported_guess"


def test_until_the_website_lists_it_the_place_question_is_kept_on_the_host_only(tmp_path, monkeypatch):
    """Nothing new reaches a released website: no status question and no answer poll until the switch is on."""
    from blueprint_pipeline import website_task_context as context_reader
    from blueprint_pipeline.website_preparation_outcome import PLACE_QUESTION_ENV as NEW_REASONS_ENV
    pipeline = _pipeline(tmp_path)
    _, record, rows = _offered()
    question = _question(pipeline, record, rows)
    asked = []
    monkeypatch.setattr(context_reader, "website_webapp_request", lambda **kw: asked.append(kw) or {})
    monkeypatch.setenv(NEW_REASONS_ENV, "false")
    assert choice.public_question(pipeline.parent) is None and choice.asked_place_question(pipeline) is None
    assert place.read_place_question(pipeline) == question  # still saved on the host
    monkeypatch.setenv(NEW_REASONS_ENV, "true")
    assert choice.public_question(pipeline.parent)["kind"] == "task_destination_choice"
    assert choice.asked_place_question(pipeline) == question and asked == []


def test_the_picture_is_read_like_the_scene_preview_and_only_behind_its_switch(tmp_path, monkeypatch):
    from blueprint_pipeline.website_preparation_outcome import PLACE_QUESTION_ENV
    (tmp_path / "bucket").mkdir()
    test_the_picture_outlines_the_start_and_draws_the_area_clipped_to_the_surface(tmp_path / "bucket")
    pipeline = tmp_path / "bucket/scenes/site-x/captures/walkthrough-x/pipeline"
    digest = json.loads((pipeline / choice.QUESTION_DIR / place.PICTURE_RECORD).read_text())["png_sha256"]
    read = dict(request_id="x", capture_id="walkthrough-x", digest=digest, storage_root=tmp_path)
    monkeypatch.delenv(PLACE_QUESTION_ENV, raising=False)
    with pytest.raises(place.PlacePictureUnavailable):
        place.open_place_picture(**read)  # off by default: nothing is delivered
    monkeypatch.setenv(PLACE_QUESTION_ENV, "true")
    ledger = pipeline.parent / "pipeline_job_ledger.json"
    ledger.write_text(json.dumps({"status": "completed"}))
    assert place.open_place_picture(**read) == (pipeline / choice.QUESTION_DIR / place.PICTURE_FILE).read_bytes()
    with pytest.raises(place.PlacePictureUnavailable):
        place.open_place_picture(**{**read, "digest": "sha256:" + "0" * 64})  # not the current question's picture
    for ended in ({"status": "terminal_authority_ended", "terminal_code": "consent_expired"},
                  {"status": "failed_retryable",
                   "last_error": "website_control_task-context_http_409:website_run_closed_by_owner"}):
        ledger.write_text(json.dumps(ended))
        with pytest.raises(place.PlacePictureUnavailable):
            place.open_place_picture(**read)  # an ended or closed job serves no picture


def _view(tmp_path, *, narrow=False):
    pipeline = _pipeline(tmp_path)
    sealed, record, rows = _offered()
    preparation = {"digest": "sha256:" + "9" * 64, "binding": {}, "destination_choice": {**record, "candidates": rows},
                   "destination": sealed, "coordinate_frame": {"runtime_to_simulator": np.eye(4).tolist()},
                   "registration": {"source_to_runtime": np.eye(4).tolist()},
                   "authoring_inputs": {"metric_envelope": {"minimum_xyz_m": SUBJECT_MIN, "maximum_xyz_m": SUBJECT_MAX}}}
    frame = _frame(tmp_path)
    if narrow:
        # A narrow view: the start and the place in use only.
        frame["intrinsics"] = [[300.0, 0, 320.0], [0, 300.0, 240.0], [0, 0, 1]]
        frame["image_size"] = [640, 480]
    return pipeline, preparation, frame, rows


def test_a_place_out_of_view_is_not_offered_and_one_place_publishes_no_question(tmp_path, monkeypatch):
    pipeline, preparation, frame, rows = _view(tmp_path)
    full = place.retain_place_view(pipeline_root=pipeline, preparation=preparation, frames=[frame],
                                   source_geometry={"digest": "sha256:" + "d" * 64}, object_label="Tray")
    assert full is not None
    # Only the place in use can be drawn: no question, and the earlier unanswered one is no longer current.
    real = place._project
    def hide_others(points, view, sim_to_source):
        pixels, depth = real(points, view, sim_to_source)
        centre = np.asarray(points)[:, :2].mean(axis=0)
        if len(points) == 48 and math.dist(centre, rows[0]["centre_xy"]) > 1e-6:
            depth = -np.abs(depth)  # behind the camera: out of view
        return pixels, depth
    monkeypatch.setattr(place, "_project", hide_others)
    assert place.retain_place_view(pipeline_root=pipeline, preparation=preparation, frames=[frame],
                                   source_geometry={"digest": "sha256:" + "d" * 64}, object_label="Tray") is None
    assert place.read_place_question(pipeline) is None
    assert not (pipeline / choice.QUESTION_DIR / place.PICTURE_RECORD).exists()
