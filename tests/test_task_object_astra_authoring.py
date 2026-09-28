"""Protect exact dimensions, image delivery, visual abstention, and Astra spend."""
import base64
import json
from types import SimpleNamespace

import pytest

from blueprint_pipeline import task_object_astra_authoring as author
from blueprint_pipeline.openai_prompt_cache import usage_and_cost_receipt, worst_case_reservation_usd


def test_astra_standard_prices_cover_input_output_and_cached_input():
    assert worst_case_reservation_usd(model='gpt-6-astra', input_token_ceiling=80000,
                                     max_output_tokens=12000, cache_policy=None) == pytest.approx(1.4)
    receipt = usage_and_cost_receipt({'input_tokens': 1000, 'output_tokens': 100,
        'input_tokens_details': {'cached_tokens': 0}}, model='gpt-6-astra')
    assert receipt['estimated_total_cost_usd'] == pytest.approx(.015)


def test_visual_review_reservation_fits_retained_drawer_budget_without_weakening_review(tmp_path):
    image = tmp_path / 'reference.png'
    image.write_bytes(b'\x89PNG\r\n\x1a\nfixture')
    frame = author.SourceFrame(path=str(image), sha256=author.file_record(image)['sha256'],
                               role='observed_source', description='Original drawer pixels')
    request = SimpleNamespace(run_id='test', object_id='drawer', request_digest='sha256:' + 'a' * 64)
    review = author.AppearanceReview(source_object_recognizable=True,
        source_color_and_material_preserved=True, opaque_surfaces_opaque=True,
        required_parts_present=True, no_obvious_geometry_artifacts=True,
        blockers=[], repair_instructions='', unobserved_surface_limitations=[])
    calls = []

    class Invoker:
        def invoke(self, spec, input_value):
            calls.append(spec)
            return SimpleNamespace(output=review, model=author.MODEL, provider='fake',
                                   usage={}, cost_usd=0, cost_status='test')

    result = author.invoke_vision(Invoker(), request, capability='independent_visual_review_2_observable_v3',
                                  prompt='Review revised drawer', output_type=author.AppearanceReview,
                                  frames=[frame], root=tmp_path)
    assert author.appearance_passed(result)
    assert calls[0].max_output_tokens == 8192
    assert worst_case_reservation_usd(model='gpt-6-astra', input_token_ceiling=80000,
                                      max_output_tokens=calls[0].max_output_tokens,
                                      cache_policy=None) == pytest.approx(1.2096)
    assert 5.6741 + 1.2096 < 7.0


def test_dimension_readback_rejects_previous_rounded_book_and_transparency():
    request = SimpleNamespace(dimensions_m=(.295304002, .397696028, .0211374),
        maximum_export_error_m=.00001, physical_review_input=SimpleNamespace(appearance='opaque'))
    report = {'dimensions_m': list(request.dimensions_m), 'minimum_z_m': 0.,
              'center_xy_m': [0., 0.], 'materials': [{'alpha': 1., 'transmission': 0.}]}
    author.validate_geometry_readback(request, report)
    with pytest.raises(author.AssetAuthoringError, match='exact_dimension'):
        author.validate_geometry_readback(request, {**report, 'dimensions_m': [.294, .397, .021]})
    with pytest.raises(author.AssetAuthoringError, match='opaque_material'):
        author.validate_geometry_readback(request, {**report, 'materials': [{'alpha': 1., 'transmission': 1.}]})
    with pytest.raises(author.AssetAuthoringError, match='origin'):
        author.validate_geometry_readback(request, {**report, 'minimum_z_m': -.001})


@pytest.mark.parametrize('source', ['import os', 'open("secret")', 'bpy.data.images.load("secret")',
                                    'bpy.ops.wm.save_as_mainfile(filepath="anywhere")',
                                    'CAD_BASE.driver_add("location")'])
def test_candidate_program_rejects_io_and_dynamic_code_before_execution(source):
    with pytest.raises(author.AssetAuthoringError):
        author.validate_blender_program(source)


def test_real_reference_bytes_are_sent_as_images_and_digest_drift_refuses(tmp_path):
    image = tmp_path / 'reference.png'
    image.write_bytes(b'\x89PNG\r\n\x1a\nfixture')
    frame = author.SourceFrame(path=str(image), sha256=author.file_record(image)['sha256'],
                               role='observed_source', description='Source book pixels')
    request = SimpleNamespace(run_id='test', object_id='book', request_digest='sha256:' + 'a' * 64)
    output = author.VisualBrief(object_identity='book', observed_parts=['pages'],
        appearance_requirements=['opaque'], unknown_regions=['underside'],
        cad_brief_markdown='Exact supplied dimensions', proposed_material='paper',
        proposed_appearance='opaque')
    calls = []
    class Invoker:
        def invoke(self, spec, input_value):
            calls.append((spec, input_value))
            return SimpleNamespace(output=output, model=author.MODEL, provider='fake',
                usage={}, cost_usd=0, cost_status='test')
    author.invoke_vision(Invoker(), request, capability='inspect', prompt='Inspect',
                         output_type=author.VisualBrief, frames=[frame], root=tmp_path)
    content = calls[0][1][0]['content']
    assert content[-1]['type'] == 'input_image'
    assert base64.b64decode(content[-1]['image_url'].split(',', 1)[1]) == image.read_bytes()
    assert calls[0][0].model == author.MODEL
    image.write_bytes(b'changed')
    with pytest.raises(author.AssetAuthoringError, match='image_changed'):
        author.invoke_vision(Invoker(), request, capability='inspect2', prompt='Inspect',
                             output_type=author.VisualBrief, frames=[frame], root=tmp_path)
    assert len(calls) == 1


def test_one_failed_visual_predicate_prevents_acceptance():
    value = dict(source_object_recognizable=True, source_color_and_material_preserved=True,
        opaque_surfaces_opaque=True, required_parts_present=True,
        no_obvious_geometry_artifacts=True, blockers=[], repair_instructions='',
        unobserved_surface_limitations=[])
    assert author.appearance_passed(author.AppearanceReview(**value))
    value['required_parts_present'] = False
    assert not author.appearance_passed(author.AppearanceReview(**value))


def test_budget_cannot_be_raised_by_caller(tmp_path):
    with pytest.raises(author.AssetAuthoringError, match='budget_invalid'):
        author.budgeted_invoker(root=tmp_path, run_id='test', maximum_cost_usd=16)


def test_cad_handoff_states_real_evidence_scope_and_preserves_nominal_dimensions():
    """Replay the unavailable-mesh instruction from the retained 2026-09-15 brief.

    Source analysis inspected images and envelope metadata, not the referenced mesh.
    The CAD graph receives text and must distinguish provisional fitting from source
    recovery while retaining legitimate contradictory-constraint and identity failures.
    """
    request = SimpleNamespace(
        object_id="interiorgs-840938-object-219",
        owner_description="bottle-shaped ornament",
        dimensions_m=(0.08142562000000009, 0.060465830000000054, 0.14343724400000002),
        maximum_export_error_m=0.00001,
        construction_constraints="single solid; exact nominal envelope; no movable contents",
    )
    # Verbatim excerpts from source-84b's retained source_analysis CAD brief. In
    # particular, its recovery instruction conflicts with the available input scope.
    retained_geometry = (
        "Inspection here covers the supplied images and envelope metadata; the referenced "
        "mesh was not directly inspected. Render observations are not physical truth.\n"
        "| X | 81.42562000000009 mm |\n"
        "| Y | 60.465830000000054 mm |\n"
        "| Z | 143.43724400000002 mm |\n"
        "Recover section profiles and feature dimensions from the retained source geometry "
        "before finalizing a parametric loft or equivalent surface construction. "
        "Preserve the broad lower body, rounded heel, continuous shoulder-to-neck transition, "
        "and plain narrow termination. Do not force circular symmetry or assume elliptical "
        "sections solely from the unequal X/Y bounds.\n"
        "Any provisional blind recess, hidden closure, wall thickness, or base treatment "
        "must be documented as an assumption.\n"
    )
    brief = SimpleNamespace(cad_brief_markdown=retained_geometry + (
        "## Estimated physical properties\nmass: unknown\n"
    ))

    handoff = author.compact_cad_handoff(request, brief)
    geometry, raw_binding = handoff.split("\nBINDING GEOMETRY CONSTRAINTS\n")
    binding = json.loads(raw_binding)
    assert geometry == retained_geometry
    assert "Estimated physical properties" not in handoff
    assert binding["dimensions_mm"] == [81.4256200000001, 60.465830000000054, 143.43724400000002]
    assert binding["maximum_export_error_mm"] == 0.01
    assert binding["construction_constraints"] == request.construction_constraints
    assert (binding["origin"], binding["up_axis"], binding["units"]) == (
        "center_XY_bottom_Z", "Z", "millimetres")

    scope = binding["evidence_scope"]
    assert "source-derived images and envelope metadata only" in scope
    assert "source mesh bytes were not inspected" in scope
    assert "unavailable work, not completed source recovery" in scope
    assert "matching the observed description and exact envelope" in scope
    assert "hidden-region geometry, as a documented assumption" in scope
    assert "Do not present inferred profiles as measured or recovered source geometry" in scope
    assert "Fail with the specific unresolved constraint if binding constraints contradict" in scope
    assert "or object identity is unresolved" in scope
    assert "do not invent a different object or relax the exact envelope" in scope
    assert "NOT a valid" not in scope
