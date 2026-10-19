"""Fidelity-profile templates: explicit thresholds that a named person reviews before use.

Synthetic samples in this file never constitute live qualification.
"""
from __future__ import annotations

import copy
import hashlib
import json
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
    value["threshold_review"] = {**value["threshold_review"], **REVIEWED,
                                 "reviewed_checks_digest": templates.reviewed_checks_digest(value)}
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
    assert review["reviewed_checks_digest"] is None
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


def test_a_review_covers_only_the_checks_it_saw() -> None:
    # A threshold raised after review must not inherit the reviewer's name.
    value = _reviewed("articulated_open_close")
    value["checks"][-1]["maximum_absolute_error"] = value["checks"][-1]["maximum_rms_error"] = 50.0
    with pytest.raises(templates.FidelityTemplateError, match="review_stale"):
        templates.profile_from_template(value, task_id="task-fixture")
    value = _reviewed("rigid_pick_place")
    value["checks"][0]["sample_ids"] = list(reversed(value["checks"][0]["sample_ids"]))
    with pytest.raises(templates.FidelityTemplateError, match="review_stale"):
        templates.validate_template(value)


def test_a_review_cannot_be_dated_in_the_future() -> None:
    value = _reviewed("rigid_pick_place")
    value["threshold_review"]["reviewed_at_iso"] = "2999-01-01T00:00:00Z"
    with pytest.raises(templates.FidelityTemplateError, match="review_in_future"):
        templates.profile_from_template(value, task_id="task-fixture")


def test_the_checks_digest_command_prints_what_a_review_records(capsys) -> None:
    assert templates.main(["checks-digest", "--family", "rigid_pick_place"]) == 0
    assert capsys.readouterr().out.strip() == templates.reviewed_checks_digest(templates.template("rigid_pick_place"))


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
    (lambda value: value["threshold_review"].update(reviewed_checks_digest="sha256:" + "0" * 64), "review_invalid"),
    (lambda value: value["threshold_review"].pop("reviewed_checks_digest"), "review_invalid"),
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


# --- End to end: WebApp-shaped exports through the unchanged qualifier -------------------------------------
# The two fixtures are byte copies of WebApp's pinned exports
# (server/tests/fixtures/site-task-measurements/). Their bindings are synthetic,
# so each test rebinds the document to the qualification fixture's request.

EXPORTS = ROOT / "tests" / "fixtures" / "site_task_measurements"
SIMULATOR = "blueprint-pipeline:simulator-measurements"


def _half_rms(family: str) -> dict[str, float]:
    return {check["id"]: check["maximum_rms_error"] / 2 for check in templates.template(family)["checks"]}


def _qualify_with_export(root: Path, family: str, *, deltas: dict[str, float], profile_family: str | None = None,
                         mutate_source=None):
    from tests.test_site_task_simulation_qualification import NOW, _reauthorize_review, _rewrite, fixture

    profile = fixture(root)
    authority = profile["site_task_qualification"]
    request = json.loads(Path(authority["request_path"]).read_text())
    source = json.loads((EXPORTS / (family + ".webapp_export.synthetic.json")).read_text())
    producer = "site_job:" + request["binding"]["scene_request_id"]
    source.update(binding=request["binding"], producer_id=producer)
    if mutate_source:
        mutate_source(source)
    simulated = copy.deepcopy(source)
    simulated["producer_id"] = SIMULATOR
    for check in simulated["checks"]:
        for sample in check["samples"]:
            sample["value"] = round(sample["value"] + deltas.get(check["id"], 0.0), 6)
    folder = Path(authority["request_path"]).parent
    for role, value in (("measurement", source), ("simulator_measurement", simulated)):
        raw = json.dumps(value, indent=2).encode()
        (folder / (role + ".json")).write_bytes(raw)
        for item in request["source_artifacts"]:
            if item["role"] == role:
                item.update(sha256="sha256:" + hashlib.sha256(raw).hexdigest(), size_bytes=len(raw))
    _rewrite(profile, "request", lambda value: value.update(source_artifacts=request["source_artifacts"]))
    fidelity = templates.profile_from_template(_reviewed(profile_family or family), task_id=request["binding"]["task_id"])
    _rewrite(profile, "profile", lambda value: (value.clear(), value.update(fidelity)))
    _reauthorize_review(profile)
    authority["allowed_measurement_producers"] = [producer]
    authority["allowed_simulator_producers"] = [SIMULATOR]
    return qualification.qualify_from_profile(profile, now=NOW)


@pytest.mark.parametrize("family", templates.FAMILIES)
def test_a_webapp_export_qualifies_inside_the_reviewed_thresholds(tmp_path: Path, family: str) -> None:
    receipt = _qualify_with_export(tmp_path, family, deltas=_half_rms(family))
    checks = templates.template(family)["checks"]
    registration = [check["id"] for check in checks if check["dimension"] in {"scale", "registration", "collision"}]
    fidelity = [check["id"] for check in checks if check["id"] not in registration]
    assert [row["id"] for row in receipt["registration_receipt"]["checks"]] == registration
    assert [row["id"] for row in receipt["fidelity_review_receipt"]["checks"]] == fidelity
    assert receipt["physical_performance_claimed"] is False
    assert receipt["adp_qualified_evidence_eligible"] is False


def test_a_sliding_front_qualifies_too(tmp_path: Path) -> None:
    def as_drawer(source: dict) -> None:
        for check in source["checks"]:
            if check["id"] == "open_close.task_state.opening":
                check["samples"] = [{"sample_id": "opening_travel", "value": 0.45}]
    receipt = _qualify_with_export(tmp_path, "articulated_open_close", deltas=_half_rms("articulated_open_close"),
                                   mutate_source=as_drawer)
    opening = [row for row in receipt["fidelity_review_receipt"]["checks"] if row["id"] == "open_close.task_state.opening"]
    assert opening[0]["sample_count"] == 1


def test_a_simulator_outside_a_threshold_fails_that_check(tmp_path: Path) -> None:
    deltas = _half_rms("rigid_pick_place")
    deltas["pick_place.task_state.object_size"] = 0.015  # 1.5 cm against a 1 cm maximum
    with pytest.raises(qualification.SiteTaskQualificationError, match="fidelity_failed:pick_place.task_state.object_size"):
        _qualify_with_export(tmp_path, "rigid_pick_place", deltas=deltas)


def test_a_missing_required_reading_fails_closed(tmp_path: Path) -> None:
    def drop_mass(source: dict) -> None:
        source["checks"] = [check for check in source["checks"] if check["id"] != "pick_place.task_dynamics.object_mass"]
    with pytest.raises(qualification.SiteTaskQualificationError, match="samples_binding_mismatch|samples_insufficient"):
        _qualify_with_export(tmp_path, "rigid_pick_place", deltas=_half_rms("rigid_pick_place"), mutate_source=drop_mass)


def test_one_familys_readings_cannot_satisfy_the_others_profile(tmp_path: Path) -> None:
    with pytest.raises(qualification.SiteTaskQualificationError, match="samples_binding_mismatch"):
        _qualify_with_export(tmp_path, "rigid_pick_place", deltas={}, profile_family="articulated_open_close")


@pytest.mark.parametrize("family", templates.FAMILIES)
def test_the_webapp_exports_cover_their_templates(family: str) -> None:
    document = json.loads((EXPORTS / (family + ".webapp_export.synthetic.json")).read_text())
    assert templates.measurement_coverage(templates.template(family), document)["complete"] is True
