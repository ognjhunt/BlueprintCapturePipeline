"""Protect exact dimensions, image delivery, visual abstention, and Astra spend."""
import base64
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
