"""Fidelity-profile templates: explicit thresholds that a named person reviews before use.

Synthetic samples in this file never constitute live qualification.
"""
from __future__ import annotations

import copy
import math
from pathlib import Path

import pytest

from blueprint_pipeline import site_task_fidelity_profile_templates as templates
from blueprint_pipeline import site_task_simulation_qualification as qualification
from blueprint_pipeline.decision_evidence_contracts import cross_runtime_canonical_digest as digest

ROOT = Path(__file__).resolve().parents[1]
REVIEWED = {"status": "reviewed", "reviewed_by": "fixture-threshold-reviewer", "reviewed_at_iso": "2026-10-19T00:00:00Z"}


def _reviewed(family: str) -> dict:
    value = templates.template(family)
    value["threshold_review"] = {**value["threshold_review"], **REVIEWED}
    return value


def _document(family: str, *, articulation: str | None = "hinged") -> dict:
    """A complete synthetic source document shaped like the WebApp export."""
    readings = {row["sample_id"]: row for row in templates.template(family)["readings"]}
    value_for = {"length": 0.5, "mass": 1.2, "force": 20.0}
    checks = []
    for check in templates.template(family)["checks"]:
        samples = [{"sample_id": sample_id, "value": value_for[readings[sample_id]["quantity"]]}
                   for sample_id in check["sample_ids"]
                   if readings[sample_id]["articulation"] in (None, articulation)]
        checks.append({"id": check["id"], "dimension": check["dimension"], "unit": check["unit"], "samples": samples})
    return {"schema_version": "blueprint.site_task_measurement_samples.v1", "binding": {},
            "producer_id": "site_job:fixture", "observed_at_iso": "2026-10-17T09:00:00Z", "checks": checks}


def test_label_pattern_is_the_qualifiers() -> None:
    assert templates._LABEL.pattern == qualification._LABEL.pattern


@pytest.mark.parametrize("family", templates.FAMILIES)
def test_shipped_templates_are_valid_and_cover_every_dimension(family: str) -> None:
    value = templates.validate_template(templates.template(family))
    assert value["family"] == family
    assert {check["dimension"] for check in value["checks"]} == qualification.DIMENSIONS
    readings = {row["sample_id"] for row in value["readings"]}
    for check in value["checks"]:
        assert check["id"].startswith(templates.FAMILY_CHECK_PREFIX[family])
        for key in ("maximum_absolute_error", "maximum_rms_error"):
            assert type(check[key]) in (int, float) and math.isfinite(check[key]) and check[key] > 0
        assert check["maximum_rms_error"] <= check["maximum_absolute_error"]
        assert 1 <= check["minimum_sample_count"] <= len(check["sample_ids"])
        assert set(check["sample_ids"]) <= readings
        assert all(check[key].strip() for key in ("site_reading", "simulator_reading", "threshold_basis"))
    assert readings == {sample_id for check in value["checks"] for sample_id in check["sample_ids"]}


@pytest.mark.parametrize("family", templates.FAMILIES)
def test_shipped_thresholds_wait_for_a_named_reviewer(family: str) -> None:
    review = templates.template(family)["threshold_review"]
    assert review["status"] == "pending_review"
    assert review["reviewed_by"] is None and review["reviewed_at_iso"] is None
    with pytest.raises(templates.FidelityTemplateError, match="thresholds_not_reviewed"):
        templates.profile_from_template(templates.template(family), task_id="task-fixture")


@pytest.mark.parametrize("family", templates.FAMILIES)
def test_a_reviewed_template_yields_a_qualifier_profile(family: str) -> None:
    value = _reviewed(family)
    profile = templates.profile_from_template(value, task_id="task-fixture")
    assert profile["schema_version"] == "blueprint.site_task_fidelity_profile.v1"
    assert profile["family"] == family and profile["task_id"] == "task-fixture"
    assert profile["template_digest"] == digest(value)
    assert profile["thresholds_reviewed_by"] == "fixture-threshold-reviewer"
    assert [check["id"] for check in profile["checks"]] == [check["id"] for check in value["checks"]]
    for criterion in profile["checks"]:
        assert set(criterion) == {"id", "dimension", "unit", "minimum_sample_count",
                                  "maximum_absolute_error", "maximum_rms_error"}


def test_a_profile_is_bound_to_one_task() -> None:
    with pytest.raises(templates.FidelityTemplateError, match="task_id_invalid"):
        templates.profile_from_template(_reviewed("rigid_pick_place"), task_id="")


def _drop_dimension(value: dict) -> None:
    value["checks"] = [check for check in value["checks"] if check["dimension"] != "task_dynamics"]
    value["readings"] = [row for row in value["readings"] if row["sample_id"] != "object_mass"]


def _set_check(key, item):
    def mutate(value: dict) -> None:
        value["checks"][0][key] = item
    return mutate


@pytest.mark.parametrize("mutate, code", [
    (_drop_dimension, "coverage_incomplete"),
    (_set_check("maximum_absolute_error", 0), "check_invalid"),
    (_set_check("maximum_absolute_error", None), "check_invalid"),
    (_set_check("maximum_rms_error", float("nan")), "check_invalid"),
    (_set_check("maximum_rms_error", 1.0), "check_invalid"),
    (_set_check("minimum_sample_count", 3), "check_invalid"),
    (_set_check("sample_ids", ["not_a_reading"]), "check_invalid"),
    (_set_check("id", "open_close.scale.lengths"), "check_invalid"),
    (_set_check("unit", "kg"), "check_invalid"),
    (_set_check("extra", "x"), "check_invalid"),
    (lambda value: value["checks"].append(copy.deepcopy(value["checks"][0])), "check_invalid"),
    (lambda value: value["readings"].append({"sample_id": "orphan", "quantity": "length", "articulation": None}),
     "reading_unused"),
    (lambda value: value["readings"][0].update(articulation="hinged"), "reading_invalid"),
    (lambda value: value["threshold_review"].update(status="reviewed"), "review_invalid"),
    (lambda value: value["threshold_review"].update(reviewed_by="someone"), "review_invalid"),
    (lambda value: value.update(default_thresholds=True), "template_invalid"),
])
def test_validation_refuses_inexplicit_or_inconsistent_templates(mutate, code) -> None:
    value = templates.template("rigid_pick_place")
    mutate(value)
    with pytest.raises(templates.FidelityTemplateError, match=code):
        templates.validate_template(value)


def test_the_catalog_is_the_threshold_free_projection() -> None:
    catalog = templates.catalog()
    assert catalog["schema_version"] == "blueprint.site_task_measurement_catalog.v1"
    assert catalog["measurement_samples_schema"] == "blueprint.site_task_measurement_samples.v1"
    assert [family["family"] for family in catalog["families"]] == list(templates.FAMILIES)
    assert "maximum_absolute_error" not in templates.render_json(catalog)
    for family in catalog["families"]:
        source = templates.template(family["family"])
        assert [check["id"] for check in family["checks"]] == [check["id"] for check in source["checks"]]
        assert family["checks"][0] == {key: source["checks"][0][key] for key in
                                       ("id", "dimension", "unit", "minimum_sample_count", "sample_ids")}
        assert {row["unit"] for row in family["readings"]} <= {"m", "kg", "N"}


@pytest.mark.parametrize("relative, builder", [
    ("docs/site_task_fidelity_profiles/rigid_pick_place.template.v1.json",
     lambda: templates.template("rigid_pick_place")),
    ("docs/site_task_fidelity_profiles/articulated_open_close.template.v1.json",
     lambda: templates.template("articulated_open_close")),
    ("docs/schemas/site_task_measurement_catalog.v1.json", templates.catalog),
])
def test_tracked_documents_are_the_rendered_builder_bytes(relative: str, builder) -> None:
    # WebApp mirrors the catalog byte for byte, so parsed equality is not enough.
    assert (ROOT / relative).read_text(encoding="utf-8") == templates.render_json(builder())


@pytest.mark.parametrize("family, articulation", [("rigid_pick_place", None),
                                                  ("articulated_open_close", "hinged"),
                                                  ("articulated_open_close", "sliding")])
def test_coverage_of_a_complete_document(family: str, articulation: str | None) -> None:
    report = templates.measurement_coverage(templates.template(family), _document(family, articulation=articulation))
    assert report["complete"] is True
    assert report["missing_dimensions"] == []


def test_coverage_names_what_is_missing_or_unexpected() -> None:
    document = _document("rigid_pick_place")
    for check in document["checks"]:
        if check["id"] == "pick_place.task_dynamics.object_mass":
            check["samples"] = []
        if check["id"] == "pick_place.collision.clearances":
            check["samples"].append({"sample_id": "handle_height", "value": 0.9})
        if check["id"] == "pick_place.task_state.object_size":
            check["unit"] = "cm"
    report = templates.measurement_coverage(templates.template("rigid_pick_place"), document)
    rows = {row["id"]: row for row in report["checks"]}
    assert report["complete"] is False
    assert rows["pick_place.task_dynamics.object_mass"]["missing_required"] == 1
    assert rows["pick_place.collision.clearances"]["unexpected"] == ["handle_height"]
    assert rows["pick_place.task_state.object_size"]["unit_mismatch"] is True
    assert report["missing_dimensions"] == ["collision", "task_dynamics", "task_state"]


def test_coverage_refuses_a_document_from_both_articulations() -> None:
    document = _document("articulated_open_close", articulation="hinged")
    for check in document["checks"]:
        if check["id"] == "open_close.task_state.opening":
            check["samples"].append({"sample_id": "opening_travel", "value": 0.4})
    report = templates.measurement_coverage(templates.template("articulated_open_close"), document)
    assert report["articulation_conflict"] is True and report["complete"] is False


def test_coverage_refuses_the_other_familys_document() -> None:
    report = templates.measurement_coverage(templates.template("articulated_open_close"),
                                            _document("rigid_pick_place"))
    assert report["complete"] is False
    assert report["missing_dimensions"] == sorted(qualification.DIMENSIONS)
