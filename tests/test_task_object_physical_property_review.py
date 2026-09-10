"""Hermetic physical-review checks: scientific contradictions must abstain."""
import copy
import json

import pytest
from pydantic import ValidationError

from blueprint_pipeline.task_object_physical_property_review import (
    PhysicalPropertyReviewInput,
    PhysicalPropertyReviewProposal,
    build_physical_property_review_prompt,
    review_physical_properties,
)


def value(v, low=None, high=None, basis="estimated"):
    return dict(value=v, basis=basis, interval={"lower": v if low is None else low,
                "upper": v if high is None else high}, rationale="Bound from retained source",
                uncertainty="Measurement tolerance or material/fill uncertainty",
                evidence_ids=["capture" if basis == "measured" else "material"])


def fixtures():
    # The actual open-book envelope and the faulty GlassClear proposal from this task.
    dims = {name: value(v, basis="measured") for name, v in zip(
        ("x_m", "y_m", "z_m"), (0.295304002, 0.397696028, 0.0211374), strict=True)}
    props = dict(mass_kg=value(1.08, 1.0, 1.15), static_friction=value(.6, .55, .7),
                 dynamic_friction=value(.4, .3, .5), restitution=value(.05, .01, .1))
    optical = dict(name="PaperOpaque", transmission=0., opacity=1.)
    evidence = [dict(evidence_id=name, uri=f"retained://{name}", sha256="a" * 64,
                     kind=kind, excerpt="Test fixture: calibrated envelope or material bounds")
                for name, kind in [("capture", "capture_measurement"),
                                   ("material", "primary_reference")]]
    data = dict(object_id="open-book", object_description="One open book",
                material_description="Opaque paper pages and cover", appearance="opaque",
                dimensions=dims, measured={name: None for name in props}, proposed=props,
                optical_material=optical, admitted_restitution={"lower": 0., "upper": .2},
                evidence=evidence)
    proposal = dict(object_id="open-book", dimensions=copy.deepcopy(dims),
                    properties=copy.deepcopy(props), optical_material=copy.deepcopy(optical),
                    mass_model=dict(method="density_fill", density_kg_m3={"lower": 750., "upper": 850.},
                        envelope_fill_fraction={"lower": .45, "upper": .65}, sheet_count=None,
                        sheet_area_m2=None, grammage_g_m2=None, cover_mass_kg=None,
                        rationale="Paper is only part of open-book envelope, not a solid box",
                        uncertainty="Density and occupied fraction have retained intervals",
                        evidence_ids=["material"]), review_rationale="Review against exact metric extents")
    return data, proposal


def review(data, proposal):
    return review_physical_properties(PhysicalPropertyReviewInput.model_validate(data),
                                     PhysicalPropertyReviewProposal.model_validate(proposal))


def test_actual_book_mass_is_size_consistent_but_glass_is_blocked():
    data, proposal = fixtures()
    result = review(data, proposal)
    assert result.accepted is not None
    assert result.model_mass_range_kg.lower < 1.08 < result.model_mass_range_kg.upper
    proposal["optical_material"] = dict(name="GlassClear", transmission=1., opacity=1.)
    result = review(data, proposal)
    assert result.accepted is None
    assert "optical:opaque_object_has_transmission_or_transparency" in result.blockers
    assert "optical:opaque_object_has_glass_material" in result.blockers
    assert result.proposed.optical_material.transmission == 1.


def test_conservative_outward_rounded_mass_uncertainty_is_not_a_contradiction():
    data, proposal = fixtures()
    proposal['properties']['mass_kg'] = value(1.196, .5, 2.)
    result = review(data, proposal)
    assert result.accepted is not None
    assert result.accepted.properties.mass_kg.interval.lower < result.model_mass_range_kg.lower
    assert result.accepted.properties.mass_kg.interval.upper > result.model_mass_range_kg.upper


def test_measured_mass_wins_without_clamp_and_retains_original_proposal():
    data, proposal = fixtures()
    data["measured"]["mass_kg"] = value(.9, .89, .91, basis="measured")
    proposal["mass_model"] = None
    result = review(data, proposal)
    assert result.accepted.properties.mass_kg.value == .9
    assert result.accepted.properties.mass_kg.basis == "measured"
    assert result.proposed.properties.mass_kg.value == 1.08
    assert result.notes == ["mass_kg:preserved_authoritative_measurement_over_proposal"]
    assert result.claim_ceiling == "development_only"


def test_conflicting_density_cannot_overwrite_measured_mass():
    data, proposal = fixtures()
    data["measured"]["mass_kg"] = value(10., 9.9, 10.1, basis="measured")
    result = review(data, proposal)
    assert result.accepted is None
    assert "mass:inconsistent_with_dimension_material_model" in result.blockers
    assert data["measured"]["mass_kg"]["value"] == 10.


def test_implausible_mass_density_and_unexplained_exact_estimate_rejected():
    data, proposal = fixtures()
    proposal["properties"]["mass_kg"] = value(15., 14., 16.)
    assert "mass:inconsistent_with_dimension_material_model" in review(data, proposal).blockers
    proposal["properties"]["mass_kg"] = value(1.08)
    assert "mass_kg:estimate_requires_nonzero_uncertainty_range" in review(data, proposal).blockers


def test_capture_dimensions_and_identity_cannot_be_reestimated():
    data, proposal = fixtures()
    proposal["dimensions"]["x_m"] = value(.3, .29, .31)
    proposal["object_id"] = "other-book"
    result = review(data, proposal)
    assert result.accepted is None
    assert "dimensions.x_m:authoritative_dimension_changed" in result.blockers
    assert "object_identity_changed" in result.blockers


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_values_fail_structured_schema(invalid):
    data, proposal = fixtures()
    proposal["properties"]["mass_kg"]["value"] = invalid
    with pytest.raises(ValidationError):
        PhysicalPropertyReviewProposal.model_validate(proposal)
    data["dimensions"]["x_m"]["interval"]["upper"] = invalid
    with pytest.raises(ValidationError):
        PhysicalPropertyReviewInput.model_validate(data)


def test_unverified_citations_and_invented_measurement_abstain():
    data, proposal = fixtures()
    proposal["mass_model"]["evidence_ids"] = ["invented-web-result"]
    proposal["properties"]["mass_kg"]["basis"] = "measured"
    result = review(data, proposal)
    assert "mass_model:unverified_evidence_reference" in result.blockers
    assert "mass_kg:proposal_cannot_invent_measurement" in result.blockers


def test_contact_bounds_and_estimate_without_mass_model_abstain():
    data, proposal = fixtures()
    proposal["properties"]["dynamic_friction"] = value(.8, .75, .9)
    proposal["properties"]["restitution"] = value(.8, .7, .9)
    proposal["mass_model"] = None
    result = review(data, proposal)
    assert result.accepted is None
    assert "friction:static_must_cover_dynamic_across_uncertainty" in result.blockers
    assert "restitution:outside_admitted_bounds" in result.blockers
    assert "mass:estimate_requires_material_mass_model" in result.blockers


def test_hollow_tray_uses_material_fraction_not_solid_envelope():
    data, proposal = fixtures()
    dimensions = (.33, .48, .035)
    for obj in [data, proposal]:
        obj["object_id"] = "blue-tray"
        obj["dimensions"] = {name: value(v, basis="measured") for name, v in zip(
            ("x_m", "y_m", "z_m"), dimensions, strict=True)}
    data["object_description"] = "Tray with 5mm floor and walls"
    data["material_description"] = "Opaque blue plastic"
    proposal["optical_material"]["name"] = "BlueOpaquePlastic"
    proposal["properties"]["mass_kg"] = value(.95, .9, 1.0)
    # Exact shell volume = outer box minus open interior above the 5mm floor.
    outer = .33 * .48 * .035
    shell = outer - .32 * .47 * .03
    fill = shell / outer
    proposal["mass_model"]["density_kg_m3"] = {"lower": 850., "upper": 1100.}
    proposal["mass_model"]["envelope_fill_fraction"] = {"lower": fill - .01, "upper": fill + .01}
    assert review(data, proposal).accepted is not None
    proposal["properties"]["mass_kg"] = value(.75, .7, .8)
    assert "mass:inconsistent_with_dimension_material_model" in review(data, proposal).blockers


def test_page_model_is_supported_and_oversize_pages_rejected():
    data, proposal = fixtures()
    proposal["mass_model"].update(method="page_model", density_kg_m3=None,
        envelope_fill_fraction=None, sheet_count=200,
        sheet_area_m2={"lower": .055, "upper": .058},
        grammage_g_m2={"lower": 70., "upper": 90.}, cover_mass_kg={"lower": .2, "upper": .3})
    assert review(data, proposal).accepted is not None
    proposal["mass_model"]["sheet_area_m2"]["upper"] = 1.
    assert "mass_model:page_area_exceeds_object_envelope" in review(data, proposal).blockers


def test_schema_and_prompt_are_sdk_ready_and_do_not_call_providers():
    data, proposal = fixtures()
    schema = PhysicalPropertyReviewProposal.model_json_schema()
    assert schema["additionalProperties"] is False
    for definition in schema["$defs"].values():
        if definition.get("type") == "object":
            assert definition["additionalProperties"] is False
            assert set(definition["required"]) == set(definition["properties"])
    prompt = build_physical_property_review_prompt(PhysicalPropertyReviewInput.model_validate(data))
    assert "Do not invent citations" in prompt
    assert json.loads(prompt.split("\n", 1)[1])["dimensions"] == data["dimensions"]
    proposal["physics_qualified"] = True
    with pytest.raises(ValidationError):
        PhysicalPropertyReviewProposal.model_validate(proposal)


def test_new_object_needs_no_invented_prior_estimate():
    data, proposal = fixtures()
    data["proposed"] = None
    result = review(data, proposal)
    assert result.accepted is not None
    assert result.accepted.properties.mass_kg.value == 1.08
    prompt = build_physical_property_review_prompt(PhysicalPropertyReviewInput.model_validate(data))
    assert "proposed=null means no prior estimate exists" in prompt
    assert json.loads(prompt.split("\n", 1)[1])["proposed"] is None
