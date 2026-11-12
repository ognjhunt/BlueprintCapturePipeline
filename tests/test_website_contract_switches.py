"""What ships by default is something the live website accepts (review 2026-11-12).

Three switches, one per website contract (``website_preparation_outcome``). Each side is checked against the
website's current strict readers (``server/utils/taskPreparationOutcome.ts``, ``taskTargetChoiceContract.ts``
at f9b8a4478), mirrored here: an outcome has exactly its five keys and a known reason keeps its shape; a
question is the object question, kind ``task_target_choice``, candidate ids ``c`` plus 16 hex digits.
"""
# Covers (for impacted-test selection):
#   src/blueprint_pipeline/website_preparation_outcome.py
#   src/blueprint_pipeline/website_task_target_choice.py
#   src/blueprint_pipeline/website_destination_choice.py
import json
import re
from pathlib import Path

import pytest

from blueprint_pipeline import website_preparation_outcome as outcome
from blueprint_pipeline import website_task_target_choice as choice

#: The website's checkout beside this one, when it is there.
WEBAPP = Path(__file__).resolve().parents[2] / "Blueprint-WebApp" / "server" / "utils" / "taskPreparationOutcome.ts"
#: The website's v7 reasons at f9b8a4478 (read from its source when the checkout is here).
WEBSITE_V7_REASONS = {"website_front_normal_undetermined", "website_registration_ambiguous", "website_registration_poor_fit",
                      "website_registration_conflicts_provider_anchor", "website_registration_anchor_deviation",
                      "website_registration_ground_plane_inconsistent", "website_task_object_not_observed",
                      "website_capture_frames_unusable", "website_task_not_named"}
OUTCOME_KEYS = {"reason", "category", "basis", "robot_id", "evidence_digest"}


def _website_reasons() -> set[str]:
    if WEBAPP.is_file():
        text = WEBAPP.read_text()
        listed = re.search(r"PREPARATION_RECAPTURE_REASONS = \[(.*?)\] as const", text, re.S)
        if listed:
            return set(re.findall(r'"([a-z_]+)"', listed.group(1))) | {"website_task_not_named"}
    return WEBSITE_V7_REASONS


def website_accepts_outcome(value) -> bool:
    """The live website's v7 outcome reader: exact keys; a known reason in its list, or one it does not know."""
    if value is None:
        return True
    if set(value) != OUTCOME_KEYS:
        return False
    known = _website_reasons() | {"task_evidence_incomplete", "robot_carry_path_rejected", "robot_fit_none"}
    return value["reason"] in known or re.fullmatch(r"[a-z][a-z0-9_:.-]{0,127}", value["reason"]) is not None


def website_accepts_question(value) -> bool:
    if value is None:
        return True
    return (value.get("kind") == "task_target_choice"
            and set(value) == {"kind", "question_digest", "target_id", "semantic_label", "candidate_set_digest",
                               "guessed_candidate_id", "candidates"}
            and all(re.fullmatch(r"c[0-9a-f]{16}", row["candidate_id"]) for row in value["candidates"]))


@pytest.fixture(autouse=True)
def _defaults(monkeypatch):
    for name in (outcome.NEW_REASONS_ENV, outcome.ENDINGS_ENV, outcome.PLACE_QUESTION_ENV):
        monkeypatch.delenv(name, raising=False)


def test_the_defaults_are_on_off_off():
    assert outcome.website_lists_new_reasons() and not outcome.website_lists_endings()
    assert not outcome.website_lists_place_question()
    assert {"website_task_object_not_observed", "website_capture_frames_unusable"} <= _website_reasons()


def test_switch_one_sends_the_reasons_the_website_words_by_default(monkeypatch):
    assert outcome.sent_recapture_reason("website_task_object_not_observed") == "website_task_object_not_observed"
    waiting = outcome.awaiting_job_sentence({"status": "failed_retryable", "last_error": "x: website_task_not_named",
                                             "capture_id": "c", "source_payload_sha256": "sha256:" + "1" * 64},
                                            task_context_digest="sha256:" + "2" * 64)
    assert waiting["reason"] == "website_task_not_named" and website_accepts_outcome(waiting)
    monkeypatch.setenv(outcome.NEW_REASONS_ENV, "false")
    assert outcome.sent_recapture_reason("website_task_object_not_observed") == "task_evidence_incomplete"


def test_switch_two_keeps_the_endings_the_website_does_not_list_off(tmp_path, monkeypatch):
    from tests.test_preparation_endings_said_truthfully import _held, _read, _spent_ledger
    from blueprint_pipeline.website_preparation_retry import RETRIES_SPENT
    root = _held(tmp_path, "task_destination_interior_geometry_required")
    assert website_accepts_outcome(_read(root)) and _read(root)["reason"] == "task_evidence_incomplete"
    assert outcome.retries_spent(_spent_ledger(RETRIES_SPENT), task_context_digest="sha256:" + "c" * 64) is None
    monkeypatch.setenv(outcome.ENDINGS_ENV, "true")
    # On, the ending carries its cause: the current website refuses it, which is why it ships off.
    assert _read(root)["reason"] == outcome.NOT_POSSIBLE and not website_accepts_outcome(_read(root))


def test_switch_three_keeps_the_place_question_on_the_host_by_default(tmp_path, monkeypatch):
    from tests.test_website_destination_choice import _offered, _pipeline, _question
    pipeline = _pipeline(tmp_path)
    _, record, rows = _offered()
    _question(pipeline, record, rows)
    assert choice.public_question(pipeline.parent) is None and website_accepts_question(choice.public_question(pipeline.parent))
    monkeypatch.setenv(outcome.PLACE_QUESTION_ENV, "true")
    # On, its kind is one the current website refuses: it ships off until the website lists it.
    assert not website_accepts_question(choice.public_question(pipeline.parent))


# --- The website's place-question side, as built (snap-webapp-place, under review) ---------------------------

PLACE_LABELS = {"The place your job names", "The place we found from your job's words", "The other end of the same surface",
                "The spot on the same surface farthest from where it starts", "The place you picked"}


def website_accepts_place_question(value) -> bool:
    """``taskDestinationChoiceQuestionSchema``: its keys, ``p`` + 16 hex ids, 2 to 4 places, the guess among them."""
    ids = [row["candidate_id"] for row in value["candidates"]]
    return (value["kind"] == "task_destination_choice" and value["target_id"] == "destination.place"
            and set(value) <= {"kind", "question_digest", "target_id", "semantic_label", "sentence",
                               "candidate_set_digest", "guessed_candidate_id", "candidates", "picture"}
            and 1 <= len(value["semantic_label"]) <= 60 and 1 <= len(value["sentence"]) <= 200
            and 2 <= len(ids) <= 4 and len(set(ids)) == len(ids) and value["guessed_candidate_id"] in ids
            and all(re.fullmatch(r"p[0-9a-f]{16}", item) for item in ids)
            and all(set(row) == {"candidate_id", "label", "reference_time_ms"} and row["label"] in PLACE_LABELS
                    for row in value["candidates"])
            and set(value.get("picture") or {"png_sha256": 1, "frame_id": 1}) == {"png_sha256", "frame_id"})


def test_the_place_question_matches_the_websites_place_contract(tmp_path, monkeypatch):
    from tests.test_website_destination_choice import test_the_picture_outlines_the_start_and_draws_the_area_clipped_to_the_surface as draw
    monkeypatch.setenv(outcome.PLACE_QUESTION_ENV, "true")
    draw(tmp_path)
    public = choice.public_question(tmp_path / "scenes/site-x/captures/walkthrough-x")
    assert website_accepts_place_question(public) and re.fullmatch(r"sha256:[0-9a-f]{64}", public["picture"]["png_sha256"])


def test_an_answered_place_binds_the_same_digest_as_the_website(tmp_path, monkeypatch):
    """The website's test (``website-task-target-choice.test.ts``): crossRuntimeDigest({corrections: [],
    destination_place: {target_id, question_digest: sha256:3..., candidate_id: pfedcba9876543210}})."""
    from blueprint_pipeline import website_destination_choice as place
    monkeypatch.setattr(place, "place_correction", lambda _root: {"target_id": "destination.place",
        "question_digest": "sha256:" + "3" * 64, "candidate_id": "pfedcba9876543210"})
    assert choice.correction_binding_digest(tmp_path) == (
        "sha256:87177cc9417d447e6235d36d21b36ce99e612be718e60af50a10a5789152a47b")


def test_with_switches_two_and_three_off_the_record_is_what_main_publishes(tmp_path):
    """The released website refuses the whole status for either; off, nothing of them reaches it."""
    from tests.test_preparation_endings_said_truthfully import _held, _read, _spent_ledger
    from tests.test_website_destination_choice import _offered, _pipeline, _question
    from blueprint_pipeline.website_preparation_retry import RETRIES_SPENT
    root = _held(tmp_path / "a", "task_destination_interior_geometry_required")
    handoff = json.loads((root / "pipeline/website_scene_preparation/handoff.json").read_text())
    assert _read(root) == {"category": "insufficient_evidence", "reason": "task_evidence_incomplete",
                           "basis": "missing_evidence", "robot_id": None, "evidence_digest": handoff["digest"]}
    assert outcome.retries_spent(_spent_ledger(RETRIES_SPENT), task_context_digest="sha256:" + "c" * 64) is None
    pipeline = _pipeline(tmp_path / "b")
    before = choice.public_question(pipeline.parent)
    _, record, rows = _offered()
    _question(pipeline, record, rows)
    assert choice.public_question(pipeline.parent) == before is None and choice.asked_place_question(pipeline) is None


def test_the_picture_route_is_the_one_the_website_reads(tmp_path, monkeypatch):
    from fastapi import FastAPI, Request
    from fastapi.testclient import TestClient
    from blueprint_pipeline import website_atlas_preview_http as http
    from tests.test_website_destination_choice import test_the_picture_outlines_the_start_and_draws_the_area_clipped_to_the_surface as draw
    (tmp_path / "bucket").mkdir()
    monkeypatch.setenv(outcome.PLACE_QUESTION_ENV, "true")
    draw(tmp_path / "bucket")
    monkeypatch.delenv(outcome.PLACE_QUESTION_ENV)
    folder = tmp_path / "bucket/scenes/site-x/captures/walkthrough-x/pipeline/website_task_target_choices"
    (folder.parent.parent / "pipeline_job_ledger.json").write_text(json.dumps({"status": "completed"}))
    digest = json.loads((folder / "destination.place.picture.json").read_text())["png_sha256"]
    monkeypatch.setenv(http.HANDOFF_STORAGE_ROOT_ENV, str(tmp_path))
    app = FastAPI()

    @app.middleware("http")
    async def admitted(request: Request, call_next):
        request.state.intake_client_id = "blueprint-webapp"
        return await call_next(request)
    http.register_website_atlas_preview_routes(app, require_admission=lambda: None)
    path = "/api/live-pipeline/website-destination-pictures/walkthrough-x/destination.place.png"
    query = {"request_id": "x", "digest": digest}
    headers = {"x-blueprint-pipeline-signature": "sha256=test"}
    client = TestClient(app)
    assert client.get(path, params=query, headers=headers).status_code == 404  # switch (iii) off
    monkeypatch.setenv(outcome.PLACE_QUESTION_ENV, "true")
    served = client.get(path, params=query, headers=headers)
    assert served.status_code == 200 and served.headers["content-type"] == "image/png"
    assert served.content == (folder / "destination.place.png").read_bytes()
    assert client.get(path, params=query).status_code == 403  # unsigned


def test_the_shared_contract_describes_the_place_question_and_its_picture(tmp_path, monkeypatch):
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads((Path(__file__).resolve().parents[1] / "docs/schemas/website_task_target_choice.v1.schema.json").read_text())
    from tests.test_website_destination_choice import test_the_picture_outlines_the_start_and_draws_the_area_clipped_to_the_surface as draw
    monkeypatch.setenv(outcome.PLACE_QUESTION_ENV, "true")
    draw(tmp_path)
    public = choice.public_question(tmp_path / "scenes/site-x/captures/walkthrough-x")
    jsonschema.validate(public, schema)
    without_picture = {key: value for key, value in public.items() if key != "picture"}
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(without_picture, schema)  # a place question always names its picture
    assert "no image or pixel data" not in schema["description"] and "never stores it" in schema["description"]
