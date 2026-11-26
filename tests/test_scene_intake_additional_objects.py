"""A several-object job's further objects in the scene intake's request, and the task keys the intake reads.

``task.additional_objects`` names each object a job builds and composes back besides ``task.subject``, the
top-ranked one the success rule scores: its rank in the build plan and a subject of the same kind. The Pipeline's
intake (``task_evaluation_scene_intent_contracts.validate_request``) admits it exactly as the WebApp's sponsored intake
does, and refuses every shape the WebApp refuses. A one-object job leaves the field out entirely, so its request and
digests are unchanged (``tests/test_website_preparation_request.py`` pins them); ``website_preparation_request``
builds the field, and no live caller names any further object yet.

A pick-and-place or articulated task now carries only the keys a release reads; any other key is refused, never
ignored. Every producer of the request sends only those: compile (``website_preparation_request.intake_request``), its
development test, correction and renewal (markers inside ``task.subject``), and the WebApp, whose public and sponsored
task schemas are strict.
"""
# Covers (for impacted-test selection):
#   src/blueprint_pipeline/task_evaluation_scene_intent_contracts.py
#   src/blueprint_pipeline/website_preparation_request.py
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from blueprint_pipeline import task_evaluation_scene_intent_contracts as contracts, website_build_plan
from blueprint_pipeline.task_evaluation_scene_intake import stage_scene_intent
from blueprint_pipeline.task_evaluation_scene_intent_contracts import SceneIntakeError, task_fields_not_read, validate_request
from blueprint_pipeline.website_preparation_request import additional_objects, intake_request, request_subject
from tests.test_task_evaluation_scene_intake import articulated_request, request

NAVIGATION = json.loads((Path(__file__).parent / "fixtures/website-navigation-reissue.v1.json").read_text())["original"]
#: No further objects: the field left out (``None`` is a null field, which is refused).
ABSENT = object()


def further(rank: int, label: str = "plate", origin: str = "removed_before_reconstruction") -> dict:
    return {"rank": rank, "subject": {"description": label, "aabb_min_xyz": [0.0, 0.0, 0.0],
                                      "aabb_max_xyz": [0.2, 0.2, 0.02], "coordinate_frame": "Z_up_estimated_meters",
                                      "geometry_origin": origin, "complete_object_dimensions": False}}


def preparation(rows=ABSENT, *, task=None) -> dict:
    """A scene preparation's request (no policies), with ``rows`` as its further objects when given."""
    value = request()
    value["execution"].update(purpose="scene_preparation", policy_candidates=[])
    if task is not None:
        value["task"] = task
    if rows is not ABSENT:
        value["task"]["additional_objects"] = rows
    return value


def refused(value: dict, code: str) -> None:
    with pytest.raises(SceneIntakeError, match=f"^scene_intake_{code}$"):
        validate_request(value, now=100)


def test_the_pipelines_copy_of_the_webapps_rules():
    # Blueprint-WebApp server/utils/taskEvaluationSceneIntake.ts: ADDITIONAL_OBJECTS_MAX, additionalObjects (rank
    # z.number().int().min(2).max(ADDITIONAL_OBJECTS_MAX + 1), rows .min(1).max(ADDITIONAL_OBJECTS_MAX), ranks rising),
    # sponsoredSceneTask (pick-and-place only); server/utils/websiteSceneSponsorship.ts: WEBSITE_SUBJECT_ORIGINS and
    # the test_environment refusal in validateWebsiteSponsoredIntake. The sibling checkout's own text is compared in
    # tests/test_webapp_contract_mirrors.py.
    assert contracts.ADDITIONAL_OBJECTS_FIELD == "additional_objects"
    assert contracts.ADDITIONAL_OBJECTS_MAX == 7
    assert contracts.ADDITIONAL_OBJECT_RANKS == (2, 8)
    assert contracts.ADDITIONAL_OBJECT_ORIGINS == ("removed_before_reconstruction", "synthetic_design")
    assert contracts.ADDITIONAL_OBJECT_FORBIDDEN_SUBJECT_FIELDS == ("test_environment",)
    assert contracts.REQUEST_SCHEMA == "task_evaluation_scene_intake_request.v1"
    # The build plan ranks up to eight objects; the scored one is first, so a further one is ranked 2 to 8.
    assert contracts.ADDITIONAL_OBJECTS_MAX + 1 == website_build_plan.MAX_OBJECTS


@pytest.mark.parametrize("rows", [
    [further(2)],
    [further(rank) for rank in range(2, 9)],
    [further(3), further(5, "bowl"), further(8, "lid")],
    [further(2, "designed ball", "synthetic_design"), further(4)],
    [{"rank": 2, "subject": {"geometry_origin": "synthetic_design"}}],
], ids=["one", "seven", "ranks_left_out", "designed", "any_subject_of_the_kind"])
def test_further_objects_the_webapp_admits_are_admitted(tmp_path, rows):
    value = preparation(rows)
    assert validate_request(value, now=100) == value
    staged = stage_scene_intent(value=value, queue_root=tmp_path, authenticated_client="webapp",
                                trusted_clients={"webapp"}, now=100)
    intent = json.loads((tmp_path / staged["intent_id"] / "intent.json").read_text())
    assert intent["request"]["task"]["additional_objects"] == rows


def test_a_request_without_further_objects_is_the_request_as_before():
    value = preparation()
    assert "additional_objects" not in value["task"]
    assert validate_request(value, now=100) == value


@pytest.mark.parametrize("rows, code", [
    (None, "task_additional_objects_invalid"),
    (further(2), "task_additional_objects_invalid"),
    ([], "task_additional_objects_invalid"),
    ([further(rank) for rank in range(2, 10)], "task_additional_objects_invalid"),
    ([further(2)] * 8, "task_additional_objects_invalid"),
    (["plate"], "task_additional_objects_invalid"),
    ([{**further(2), "role": "job_object"}], "task_additional_objects_invalid"),
    ([{"rank": 2}], "task_additional_objects_invalid"),
    ([{"subject": further(2)["subject"]}], "task_additional_objects_invalid"),
    ([further(1)], "task_additional_objects_rank_invalid"),
    ([further(0)], "task_additional_objects_rank_invalid"),
    ([further(9)], "task_additional_objects_rank_invalid"),
    ([{**further(2), "rank": 2.0}], "task_additional_objects_rank_invalid"),
    ([{**further(2), "rank": "2"}], "task_additional_objects_rank_invalid"),
    ([{**further(2), "rank": True}], "task_additional_objects_rank_invalid"),
    ([further(3), further(2)], "task_additional_objects_rank_invalid"),
    ([further(2), further(2)], "task_additional_objects_rank_invalid"),
    ([further(2), further(4), further(3)], "task_additional_objects_rank_invalid"),
    ([{"rank": 2, "subject": {}}], "task_additional_object_subject_invalid"),
    ([{"rank": 2, "subject": "plate"}], "task_additional_object_subject_invalid"),
    ([{"rank": 2, "subject": ["plate"]}], "task_additional_object_subject_invalid"),
    ([further(2, origin="retained_in_captured_reconstruction")], "task_additional_object_subject_invalid"),
    ([further(2, origin="kept_as_filmed")], "task_additional_object_subject_invalid"),
    ([{"rank": 2, "subject": {"description": "plate"}}], "task_additional_object_subject_invalid"),
    ([further(2), {"rank": 3, "subject": {**further(3)["subject"], "test_environment": {"kind": "x"}}}],
     "task_additional_object_test_environment_forbidden"),
    ([{"rank": 2, "subject": {**further(2)["subject"], "test_environment": None}}],
     "task_additional_object_test_environment_forbidden"),
], ids=["null", "a_row_not_a_list", "empty", "eight", "eight_repeated", "row_not_a_mapping", "extra_field",
        "no_subject", "no_rank", "rank_1", "rank_0", "rank_9", "float_rank", "text_rank", "bool_rank", "out_of_order",
        "repeated", "not_rising", "empty_subject", "text_subject", "list_subject", "retained_origin", "filmed_origin",
        "no_origin", "test_environment", "null_test_environment"])
def test_every_shape_the_webapp_refuses_is_refused(rows, code):
    refused(preparation(rows), code)


def test_further_objects_on_any_other_task_are_refused():
    articulated = articulated_request()
    articulated["execution"].update(purpose="scene_preparation", policy_candidates=[])
    articulated["task"]["additional_objects"] = [further(2)]
    refused(articulated, "task_additional_objects_strategy_invalid")
    navigation = copy.deepcopy(NAVIGATION)
    assert validate_request(navigation, now=100) == navigation
    navigation["task"]["additional_objects"] = [further(2)]
    refused(navigation, "task_additional_objects_strategy_invalid")
    family = preparation(task={"task_id": "stack-cups", "strategy": "stacking",
                               "task_success_contract_digest": "sha256:" + "1" * 64,
                               "native_runtime_contract_digest": "sha256:" + "2" * 64,
                               "task_family_runtime_selection": {"runtime_id": "stacking-runtime",
                                                                 "runtime_catalog_digest": "sha256:" + "3" * 64}})
    assert validate_request(family, now=100) == family
    family["task"]["additional_objects"] = [further(2)]
    refused(family, "task_additional_objects_strategy_invalid")


@pytest.mark.parametrize("articulated", [False, True], ids=["pick_and_place", "articulated"])
def test_a_task_key_no_release_reads_is_kept_and_flagged_never_refused(articulated, caplog):
    """Only privacy, rights and money stop a job (a review of PR #3754): a key this release does not read is kept as
    sent and flagged ``scene_intake_task_fields_not_read``, and every key one reads is kept without a flag."""
    def base():
        value = articulated_request() if articulated else request()
        value["execution"].update(purpose="scene_preparation", policy_candidates=[])
        return value
    for key, item in (("note", "x"), ("additional_object", [further(2)]),
                      ("articulation" if not articulated else "destination", {"id": "tray"})):
        value = base()
        value["task"][key] = item
        caplog.clear()
        with caplog.at_level("WARNING"):
            assert validate_request(value, now=100) == value
        assert task_fields_not_read(value["task"]) == [key]
        assert [(row.getMessage(), row.fields) for row in caplog.records] == [("scene_intake_task_fields_not_read", [key])]
    caplog.clear()
    with caplog.at_level("WARNING"):
        validate_request(base(), now=100)
    assert caplog.records == [] and task_fields_not_read(base()["task"]) == []
    reuse = base()
    reuse["task"]["reuse_completed_stages"] = False
    assert validate_request(reuse, now=100) == reuse
    # An evaluation request's own keys (a scene preparation may not name them: validate_execution).
    evaluation = articulated_request() if articulated else request()
    evaluation["task"].update(robot_binding_id="robot-1", robot_binding_digest="sha256:" + "9" * 64,
                              episode_interpretation=True, evaluation_source={"kind": "x"}, reuse_completed_stages=True)
    assert validate_request(evaluation, now=100) == evaluation


def test_the_builder_names_each_further_object_by_rank_with_a_subject_built_as_the_scored_ones():
    records = [{"rank": 2, "entry": {"target_id": "t-plate", "semantic_label": "dinner plate"},
                "aabb_min": [0.1, 0.2, 0.75], "aabb_max": [0.36, 0.46, 0.77],
                "origin": "removed_before_reconstruction", "complete": False},
               {"rank": 4, "entry": {"target_id": "t-ball"}, "aabb_min": [-0.3, 0.0, 0.75],
                "aabb_max": [-0.2, 0.1, 0.85], "origin": "synthetic_design", "complete": True}]
    built = additional_objects(records)
    assert built == [
        {"rank": 2, "subject": {"description": "dinner plate", "aabb_min_xyz": [0.1, 0.2, 0.75],
                                "aabb_max_xyz": [0.36, 0.46, 0.77], "coordinate_frame": "Z_up_estimated_meters",
                                "geometry_origin": "removed_before_reconstruction",
                                "complete_object_dimensions": False}},
        {"rank": 4, "subject": {"description": "t-ball", "aabb_min_xyz": [-0.3, 0.0, 0.75],
                                "aabb_max_xyz": [-0.2, 0.1, 0.85], "coordinate_frame": "Z_up_estimated_meters",
                                "geometry_origin": "synthetic_design", "complete_object_dimensions": True}}]
    assert [row["subject"] for row in built] == [
        request_subject(item["entry"], aabb_min=item["aabb_min"], aabb_max=item["aabb_max"], origin=item["origin"],
                        complete=item["complete"]) for item in records]

    base = preparation()
    spend = {"owner": base["owner"], "consent": base["consent"], "max_total_spend_usd": 4, "max_paid_attempts": 2,
             "expires_at_epoch": 1000}
    arguments = dict(capture_id="upload-1", task_context={"context_digest": "sha256:" + "c" * 64},
                     base_scene={"splat_binding_id": "website-splat-" + "d" * 32, "splat_digest": "sha256:" + "d" * 64},
                     spend=spend, articulated=False, subject=further(1)["subject"], support_label="table",
                     support_aabb=([0.0, 0.0, 0.0], [1.0, 1.0, 0.75]), mechanism=None, destination={"id": "tray"},
                     success={"inside": True}, anthropic=False)
    alone = intake_request(**arguments)
    assert intake_request(**arguments, further_objects=[]) == alone and "additional_objects" not in alone["task"]
    several = intake_request(**arguments, further_objects=records)
    assert several["task"]["additional_objects"] == built
    assert {key: item for key, item in several["task"].items() if key != "additional_objects"} == alone["task"]
    assert {key: item for key, item in several.items() if key != "task"} == {
        key: item for key, item in alone.items() if key != "task"}
    assert validate_request(several, now=100) == several
    assert validate_request(alone, now=100) == alone
    # The intake refuses further objects on an articulated task; the builder never drops them in silence.
    opening = intake_request(**{**arguments, "articulated": True,
                                "mechanism": {"joint_type": "prismatic", "part_label": "drawer"}},
                             further_objects=records)
    refused(opening, "task_additional_objects_strategy_invalid")
