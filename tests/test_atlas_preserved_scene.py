"""Environment-preserving tasks never enter the object-removal side route."""
from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone

import pytest

from blueprint_pipeline import website_atlas_pipeline as pipeline
from blueprint_pipeline import atlas_depth_splat as splats
from blueprint_pipeline import atlas_result_bundle as bundles
from blueprint_pipeline import atlas_source_pose_inputs as source_inputs
from blueprint_pipeline import atlas_mesh_registration as mesh_registration
from blueprint_pipeline import paid_resource_allocator as allocator
from blueprint_pipeline import website_atlas_scene_source as scene_source
from tests.test_website_atlas_scene_source import job_fixture, write, digest
from tests.test_atlas_scene_planner import fixture as planner_fixture


@pytest.mark.parametrize("refined_splat", [False, True])
@pytest.mark.parametrize("subject_origin", ["synthetic_design", "no_manipulated_subject"])
def test_automatic_controller_routes_the_typed_plan_without_a_scene_override(tmp_path, monkeypatch, subject_origin,
                                                                             refined_splat):
    from blueprint_pipeline import atlas_video_inputs as inputs, website_atlas_pose as pose_api
    _, survey, proposal, policy = planner_fixture(tmp_path)
    video = tmp_path / "source.mov"
    video.write_bytes(b"fresh source")
    survey["source_video_digest"] = pipeline._sha256_file(video)
    survey['schema_version'] = 'atlas_video_survey.v1'
    survey['digest'] = pipeline.canonical_digest(survey, digest_field='digest')
    proposal.update(subject_origin=subject_origin, automation={
        "scale_references": [], "objects": [], "removal_instruction": "", "mask_margin_fraction": .01})
    proposal["coverage_requirements"][0]["role"] = "task_region"
    policy["reviewed_on"] = datetime.now(timezone.utc).date().isoformat()
    if refined_splat:
        policy["refined_splat_cost_usd"] = 5
    monkeypatch.setattr(pose_api, "atlas_api_config", lambda: ("configured", "test-non-secret"))
    monkeypatch.setattr(pipeline, "load_configuration", lambda *a: {
        "policy": policy, "remaining_cost_usd": 9 if refined_splat else 4,
        "authority": {"provider_retention": "project_library_retention_approved"}})
    monkeypatch.setattr(inputs, "survey_video", lambda **kw: write(kw["output_root"] / "atlas_video_survey.json", survey))
    monkeypatch.setattr(pipeline, "accuracy_survey", lambda **kw: kw["base_survey"])
    selections = []
    monkeypatch.setattr(inputs, "prepare_video_inputs", lambda **kw: selections.append(kw) or {"frames": []})
    monkeypatch.setattr(allocator, "run_sponsored_website_atlas_plan", lambda **kw: {"proposal": proposal, "model": "test"})
    routed = []
    monkeypatch.setattr(pipeline, "_preserved_scene", lambda **kw: routed.append(kw) or {"status": "processing"})
    for name in ("run_sponsored_website_atlas_task", "run_sponsored_website_atlas_automation",
                 "run_sponsored_website_atlas_pose"):
        monkeypatch.setattr(allocator, name, lambda **kw: pytest.fail("Legacy editing/generation route was entered"))
    context = {"capture_id": "fresh", "context_digest": "task", "description":
        "Navigate from the near end of the visible aisle to a stopping point before the end bookshelf, "
        "avoiding contact with the shelves on either side."}
    result = pipeline.run_atlas_pipeline(source_video=video, output_root=tmp_path / "job", task_context=context)
    assert result["status"] == "processing" and len(routed) == 1
    assert routed[0]["review"]["subject_origin"] == subject_origin
    assert routed[0]["task_context"] == context
    # Its refined splat trains on the posed selection, so a splat purchase fills it toward the target.
    assert [kw.get("splat_frame_target") for kw in selections] == [pose_api.SPLAT_FRAME_TARGET if refined_splat else None]
    assert "minimum_frames" not in selections[0] and "maximum_frames" not in selections[0]


@pytest.mark.parametrize("tamper", ["remove", "instruction", "transient"])
def test_cached_environment_plan_cannot_enter_paid_pose_after_an_edit_is_injected(tmp_path, monkeypatch, tamper):
    review = {"subject_origin": "no_manipulated_subject", "automation": {
        "objects": [{"disposition": "keep"}], "removal_instruction": ""}}
    if tamper == "remove":
        review["automation"]["objects"][0]["disposition"] = "remove"
    elif tamper == "instruction":
        review["automation"]["removal_instruction"] = "Erase the shelves"
    else:
        review["transient_frame_ids"] = ["person-in-view"]
    monkeypatch.setattr(pipeline, "_source_pose", lambda **kw: pytest.fail("Invalid plan reached paid pose"))
    with pytest.raises(ValueError, match="atlas_scene_requires_preserved_capture"):
        pipeline._preserved_scene(pose_root=tmp_path, output_root=tmp_path, inputs={}, review=review,
            policy={}, task_context={}, source_digest="source", configuration={}, check=lambda _: None,
            experimental=False, max_polls=1, model="test")


@pytest.mark.parametrize("source_status", ["processing", "needs_input", "completed"])
@pytest.mark.parametrize("cached_splat", [False, True])
@pytest.mark.parametrize("subject_origin", ["synthetic_design", "no_manipulated_subject"])
def test_preserved_reconstruction_exports_observations_and_never_buys_appearance(
        tmp_path, monkeypatch, source_status, cached_splat, subject_origin):
    pose_root = tmp_path / "pose"
    source_root = pose_root / "full-view-source-v1"
    source_root.mkdir(parents=True)
    pose = {"digest": "posed-source"}
    (source_root / "atlas_posed_rgbd_result.json").write_text(json.dumps(pose))
    (source_root / "scene_measurements.json").write_text("{}")
    (tmp_path / "atlas_scene_review.json").write_text("{}")
    proposal = {"subject_origin": subject_origin, "automation": {
        "objects": [{"disposition": "keep"}], "removal_instruction": ""},
        "uncaptured_regions": ["unobserved underside"]}
    calls, exports, decodes = [], [], []
    monkeypatch.setattr(pipeline, "_source_pose", lambda **kw: calls.append("source_pose") or {
        "status": source_status, "operation_id": "source", "reason": "scale_unresolved",
        "scale_status": "estimated_from_named_assumptions"})
    monkeypatch.setattr(source_inputs, "prepare_source_pose", lambda root: (source_root, {}))
    frames = ["observed_depth"]
    def decode(root):
        assert root == source_root
        decodes.append(root)
        return pose, frames
    monkeypatch.setattr(splats, "load_posed_rgbd", decode)
    def fuse(*, frames, result, output_root):
        assert result is pose and frames is retained_frames
        assert output_root == tmp_path / "splat"
        calls.append("fuse")
        return {"digest": "splat", "splat_kind": "depth_fused_draft"}
    retained_frames = frames
    monkeypatch.setattr(splats, "write_depth_splat", fuse)
    monkeypatch.setattr(splats, "build_depth_splat", lambda *a, **kw: pytest.fail("Source RGBD was decoded twice"))
    if cached_splat:
        write(tmp_path / "splat/atlas_depth_splat_manifest.json", {
            "digest": "splat", "splat_kind": "depth_fused_draft"})
    def mesh_attempt(root, output):
        output.mkdir()
        return {"request": {"splat": "bound"}, "uploads": {}}, output
    monkeypatch.setattr(splats, "splats2mesh_attempt", mesh_attempt)
    def task(**kw):
        assert kw["task"] == "splats2Mesh", "No unconditional image generation is authorized"
        calls.append("mesh")
        (kw["output_root"] / "result.json").write_text(json.dumps({
            "outputs": [{"role": "mesh", "path": "scene.glb"}]}))
        return {"status": "completed"}
    monkeypatch.setattr(allocator, "run_sponsored_website_atlas_task", task)
    monkeypatch.setattr(mesh_registration, "register_atlas_mesh", lambda **kw: calls.append("register"))
    monkeypatch.setattr(bundles, "export_pose_bundle", lambda **kw: exports.append(kw) or {"zip_path": "scene.zip"})
    stages = []
    result = pipeline._preserved_scene(pose_root=pose_root, output_root=tmp_path, inputs={}, review=proposal,
        policy={"mesh_export_cost_usd": .5, "rate_card_reference": "test", "reviewed_by": "test", "reviewed_on": "test"},
        task_context={"context_digest": "task"}, source_digest="source", configuration={}, check=stages.append,
        experimental=False, max_polls=1, model="test")
    if source_status == "processing":
        assert result["status"] == "processing" and calls == ["source_pose"] and not exports and not decodes
        return
    assert result["captured_scene_preserved"] is True
    assert result["subject_origin"] == subject_origin
    assert result["image_inpainting_performed"] is False and result["object_removal_performed"] is False
    assert result["generated_completion_performed"] is False and result["simulation_qualified"] is False
    assert result["uncaptured_regions"] == ["unobserved underside"]
    assert exports[0]["pose_root"] == source_root and "generated_roots" not in exports[0]
    assert decodes == [source_root]
    expected = ["source_pose"] + ([] if cached_splat else ["fuse"])
    if source_status == "completed":
        assert calls == expected + ["mesh", "register"] and stages == ["mesh"]
    else:
        assert calls == expected and not stages
        assert "scale_unresolved" in result["blockers"] and not result["mesh_available"]


@pytest.mark.parametrize("tamper", [None, "removal", "pose"])
@pytest.mark.parametrize("subject_origin", ["synthetic_design", "no_manipulated_subject"])
def test_production_scene_handoff_accepts_preserved_geometry_without_fabricating_removal(tmp_path, tamper, subject_origin):
    root = job_fixture(tmp_path, same_pose=tamper != "pose")
    result = json.loads((root / "result.json").read_text())
    background = result.pop("automation")["background"]
    result.update(subject_origin=subject_origin, captured_scene_preserved=True,
        object_removal_performed=tamper == "removal", image_inpainting_performed=False,
        generated_completion_performed=False, mesh_available=True, splat_manifest_digest=background["splat_manifest_digest"])
    shutil.move(root / "automation/background_splat", root / "splat")
    shutil.move(root / "automation/background_mesh", root / "mesh")
    write(root / "result.json", digest(result))
    write(root / "preview_job.json", {"state": "ready", "result_digest": result["digest"]})
    if tamper:
        with pytest.raises(ValueError, match="preserved_scene_contract_invalid|preserved_scene_pose_mismatch"):
            scene_source.atlas_scene_source(capture_root=tmp_path, output_root=tmp_path / "prep")
    else:
        value = scene_source.atlas_scene_source(capture_root=tmp_path, output_root=tmp_path / "prep")
        assert value["status"] == "ready" and value["scene_content"] == "captured_scene_preserved"
        assert value["subject_origin"] == subject_origin
        assert value["base_scene"]["anchor"]["basis"] == "same_atlas_pose"
        assert value["base_scene"]["scale_authority"] == "atlas_reference_estimate"
        assert not (root / "automation/background/prepared_background.json").exists()


@pytest.mark.parametrize('completion_state', ['completed', 'added_nothing', 'processing', 'source_recovery_required'])
def test_controller_consumes_admitted_completion_and_stops_before_mesh_while_unresolved(
        tmp_path, monkeypatch, completion_state):
    pose_root = tmp_path / 'pose'
    source_root = pose_root / 'full-view-source-v1'
    source_root.mkdir(parents=True)
    pose = {'digest': 'source-pose'}
    write(source_root / 'atlas_posed_rgbd_result.json', pose)
    write(tmp_path / 'atlas_scene_review.json', {})
    write(source_root / 'scene_measurements.json', {})
    baseline = {'digest': 'baseline', 'splat_kind': 'depth_fused_draft'}
    write(tmp_path / 'splat/atlas_depth_splat_manifest.json', baseline)
    completed_root = tmp_path / 'preserved_completion/splat-bound'
    completed = {'digest': 'completed-export', 'splat_kind': 'depth_fused_draft'}
    frames, calls, exports = ['source-depth'], [], []
    monkeypatch.setattr(pipeline, '_source_pose', lambda **_: {'status': 'completed'})
    monkeypatch.setattr(source_inputs, 'prepare_source_pose', lambda _: (source_root, {}))
    monkeypatch.setattr(splats, 'load_posed_rgbd', lambda _: (pose, frames))
    monkeypatch.setattr(pipeline, 'load_configuration', lambda *args: {'remaining_cost_usd': 3})
    def complete(**options):
        calls.append('completion')
        assert options['preserved_capture'] is True and options['frames'] is frames
        assert options['source_survey_path'] == tmp_path / 'source-survey.json'
        assert options['configuration']['remaining_cost_usd'] == 3
        if completion_state == 'added_nothing':
            return {'status': 'completed', 'generated_geometry_admitted': False,
                    'generated_geometry_rejected': 'atlas_preserved_completion_no_exportable_geometry',
                    'splat_root': str(tmp_path / 'splat'), 'generated_roots': [],
                    'completion': {'generated_roots': [str(tmp_path / 'native')],
                                   'remaining_unknowns': ['floor behind the camera']},
                    'geometry_recovery': {'digest': 'recovery', 'exportable_samples': 0, 'added_geometry_pixels': 0}}
        if completion_state != 'completed':
            return {'status': completion_state, 'stage': 'preserved_scene_completion', 'reason': 'source_view_available'}
        return {'status': 'completed', 'generated_geometry_admitted': True,
                'splat_root': str(completed_root), 'splat_manifest': completed,
                'completion': {'generated_roots': [str(tmp_path / 'native')]},
                'generated_roots': [str(tmp_path / 'native')], 'analysis_paths': [],
                'admission': {'digest': 'admission'}, 'admission_path': str(tmp_path / 'admission.json')}
    monkeypatch.setattr(allocator, 'run_sponsored_website_atlas_automation', complete)
    def attempt(root, output):
        assert root == (completed_root if completion_state == 'completed' else tmp_path / 'splat')
        return {'request': {}, 'uploads': {}}, output
    monkeypatch.setattr(splats, 'splats2mesh_attempt', attempt)
    def mesh(**options):
        calls.append('mesh')
        write(options['output_root'] / 'result.json', {'outputs': [{'role': 'mesh', 'path': 'scene.glb'}]})
        return {'status': 'completed'}
    monkeypatch.setattr(allocator, 'run_sponsored_website_atlas_task', mesh)
    monkeypatch.setattr(mesh_registration, 'register_atlas_mesh', lambda **_: None)
    monkeypatch.setattr(bundles, 'export_pose_bundle', lambda **options: exports.append(options) or {'zip_path': 'scene.zip'})
    result = pipeline._preserved_scene(pose_root=pose_root, output_root=tmp_path, inputs={},
        review={'subject_origin': 'no_manipulated_subject', 'automation': {'objects': [], 'removal_instruction': ''},
                'uncaptured_regions': ['behind books']}, policy={'mesh_export_cost_usd': .5, 'rgbd_view_cost_usd': .1,
                    'rate_card_reference': 'test', 'reviewed_by': 'test', 'reviewed_on': 'test'},
        task_context={'context_digest': 'task'}, source_digest='capture', configuration={}, check=lambda _: None,
        experimental=False, max_polls=1, model='test', source_survey_path=tmp_path / 'source-survey.json')
    assert json.loads((tmp_path / 'splat/atlas_depth_splat_manifest.json').read_text()) == baseline
    if completion_state == 'completed':
        assert calls == ['completion', 'mesh']
        assert result['splat_manifest_digest'] == completed['digest']
        assert result['generated_geometry_admitted'] and result['generated_completion_performed']
        assert exports[0]['splat_root'] == completed_root
        assert exports[0]['generated_roots'] == [tmp_path / 'native']
        assert result['scene_completion']['admission_digest'] == 'admission'
    elif completion_state == 'added_nothing':
        # Flagged and kept going: the captured export is meshed, and nothing generated is exported.
        assert calls == ['completion', 'mesh'] and result['mesh_available']
        assert result['splat_manifest_digest'] == baseline['digest']
        assert result['generated_completion_performed'] and not result['generated_geometry_admitted']
        assert 'generated_roots' not in exports[0] and exports[0]['splat_root'] == tmp_path / 'splat'
        assert 'atlas_preserved_completion_no_exportable_geometry' not in result['blockers']
        assert result['task_and_surrounding_scene_covered']
        assert result['scene_completion']['generated_geometry_rejected'] == (
            'atlas_preserved_completion_no_exportable_geometry')
        assert result['scene_completion']['geometry_recovery'] == {
            'digest': 'recovery', 'exportable_samples': 0, 'added_geometry_pixels': 0}
        assert result['scene_completion']['remaining_unknowns'] == ['floor behind the camera']
    elif completion_state == 'processing':
        assert calls == ['completion'] and not exports
        assert result['status'] == 'processing'
    else:
        assert calls == ['completion'] and not result['mesh_available']
        assert not result['task_and_surrounding_scene_covered']


def test_generated_scene_without_independent_geometry_admission_is_not_handed_off(tmp_path):
    root = job_fixture(tmp_path, same_pose=True)
    result = json.loads((root / 'result.json').read_text())
    result.update(subject_origin='no_manipulated_subject', captured_scene_preserved=True,
        object_removal_performed=False, image_inpainting_performed=False,
        generated_completion_performed=True, generated_geometry_admitted=False)
    write(root / 'result.json', digest(result))
    write(root / 'preview_job.json', {'state': 'ready', 'result_digest': result['digest']})
    outcome = scene_source.atlas_scene_source(capture_root=tmp_path, output_root=tmp_path / 'prep')
    assert outcome == {'status': 'needs_input', 'reason': 'atlas_preserved_completion_not_admitted'}


@pytest.mark.parametrize('tamper', [None, 'narrow_pixels', 'samples', 'admitted', 'reason'])
def test_generated_views_that_added_nothing_hand_off_the_scene_as_filmed(tmp_path, tamper):
    """The views were made and reviewed but added no geometry: flagged, and the captured
    scene goes on unchanged. A record claiming otherwise is still refused."""
    root = job_fixture(tmp_path, same_pose=True)
    result = json.loads((root / 'result.json').read_text())
    background = result.pop('automation')['background']
    # 'narrow_pixels': depth pixels were admitted, but too few neighbours for a surface sample.
    recovery = {'digest': 'sha256:' + 'a' * 64, 'exportable_samples': 0 if tamper != 'samples' else 3,
                'added_geometry_pixels': 12 if tamper == 'narrow_pixels' else 0}
    result.update(subject_origin='no_manipulated_subject', captured_scene_preserved=True,
        object_removal_performed=False, image_inpainting_performed=False,
        generated_completion_performed=True, generated_geometry_admitted=tamper == 'admitted',
        mesh_available=True, splat_manifest_digest=background['splat_manifest_digest'],
        scene_completion={'status': 'completed', 'reason': None, 'failure_class': None,
                          'generated_geometry_admitted': False, 'remaining_unknowns': ['floor behind the camera'],
                          'generated_geometry_rejected': ('atlas_preserved_completion_no_exportable_geometry'
                                                          if tamper != 'reason' else 'something_else'),
                          'geometry_recovery': recovery})
    shutil.move(root / 'automation/background_splat', root / 'splat')
    shutil.move(root / 'automation/background_mesh', root / 'mesh')
    write(root / 'result.json', digest(result))
    write(root / 'preview_job.json', {'state': 'ready', 'result_digest': result['digest']})
    outcome = scene_source.atlas_scene_source(capture_root=tmp_path, output_root=tmp_path / 'prep')
    if tamper in {None, 'narrow_pixels'}:
        assert outcome['status'] == 'ready' and outcome['scene_content'] == 'captured_scene_preserved'
        assert outcome['base_scene']['anchor']['basis'] == 'same_atlas_pose'
    else:
        assert outcome == {'status': 'needs_input', 'reason': 'atlas_preserved_completion_not_admitted'}


@pytest.mark.parametrize('outcome', ['refined', 'unaffordable', 'provider_failed', 'processing', 'admitted'])
def test_a_scene_kept_as_filmed_takes_atlas_refined_splat_with_the_draft_only_as_a_named_fallback(
        tmp_path, monkeypatch, outcome):
    """Owner, 2026-11-07: the appearance is Atlas's refined splat whenever its money allows."""
    from blueprint_pipeline import atlas_native_splat

    pose_root = tmp_path / 'pose'
    source_root = pose_root / 'full-view-source-v1'
    source_root.mkdir(parents=True)
    pose = {'digest': 'source-pose'}
    write(source_root / 'atlas_posed_rgbd_result.json', pose)
    write(tmp_path / 'atlas_scene_review.json', {})
    write(source_root / 'scene_measurements.json', {})
    draft = {'digest': 'draft', 'splat_kind': 'depth_fused_draft'}
    write(tmp_path / 'splat/atlas_depth_splat_manifest.json', draft)
    refined_root = tmp_path / 'refined_splat-minimal'
    refined = {'digest': 'sha256:' + 'b' * 64, 'splat_kind': 'atlas_refined'}
    frames, calls, exports, meshed = [{'frame_id': 'kept'}, {'frame_id': 'disagrees'}], [], [], []
    # The splat input selection (atlas_splat_input_quality) decides once which views are sent.
    selection = {'digest': 'sha256:' + 'c' * 64, 'flags': ['splat_inconsistent_frames_dropped'],
                 'sent_frame_ids': ['kept'], 'left_out': [{'frame_id': 'disagrees'}], 'heldout_frame_ids': []}
    monkeypatch.setattr(pipeline, '_splat_input_selection', lambda **_: selection)
    monkeypatch.setattr(pipeline, '_source_pose', lambda **_: {'status': 'completed'})
    monkeypatch.setattr(source_inputs, 'prepare_source_pose', lambda _: (source_root, {}))
    monkeypatch.setattr(splats, 'load_posed_rgbd', lambda _: (pose, frames))
    remaining = 3 if outcome == 'unaffordable' else 23
    monkeypatch.setattr(pipeline, 'load_configuration', lambda *args: {'remaining_cost_usd': remaining})

    def complete(**_options):
        calls.append('completion')
        if outcome == 'admitted':
            return {'status': 'completed', 'generated_geometry_admitted': True,
                    'splat_root': str(tmp_path / 'preserved_completion/splat-bound'),
                    'splat_manifest': {'digest': 'completed-export', 'splat_kind': 'depth_fused_draft'},
                    'completion': {'generated_roots': [str(tmp_path / 'native')]},
                    'generated_roots': [str(tmp_path / 'native')], 'analysis_paths': [],
                    'admission': {'digest': 'admission'}, 'admission_path': str(tmp_path / 'admission.json')}
        return {'status': 'completed', 'generated_geometry_admitted': False, 'generated_roots': [],
                'splat_root': str(tmp_path / 'splat'), 'completion': {'generated_roots': []}}
    monkeypatch.setattr(allocator, 'run_sponsored_website_atlas_automation', complete)

    def native(**options):
        calls.append('refined_splat')
        assert options['frames'] == frames[:1] and options['draft'] == draft
        assert options['input_selection_digest'] == selection['digest']
        assert options['output_root'] == tmp_path / 'refined_splat'
        if outcome == 'provider_failed':
            raise ValueError('atlas_task_operation_failed')
        if outcome == 'processing':
            return {'status': 'processing', 'stage': 'refined_splat', 'operation_id': 'op-1'}
        return {'status': 'completed', 'splat_root': str(refined_root), 'splat_manifest': refined}
    monkeypatch.setattr(atlas_native_splat, 'run_native_splat', native)
    monkeypatch.setattr(splats, 'splats2mesh_attempt', lambda root, output: meshed.append(root) or (
        {'request': {}, 'uploads': {}}, output))

    def mesh(**options):
        calls.append('mesh')
        write(options['output_root'] / 'result.json', {'outputs': [{'role': 'mesh', 'path': 'scene.glb'}]})
        return {'status': 'completed'}
    monkeypatch.setattr(allocator, 'run_sponsored_website_atlas_task', mesh)
    monkeypatch.setattr(mesh_registration, 'register_atlas_mesh', lambda **_: None)
    monkeypatch.setattr(bundles, 'export_pose_bundle', lambda **options: exports.append(options) or {'zip_path': 'z'})
    result = pipeline._preserved_scene(pose_root=pose_root, output_root=tmp_path, inputs={},
        review={'subject_origin': 'no_manipulated_subject', 'automation': {'objects': [], 'removal_instruction': ''},
                'uncaptured_regions': []},
        policy={'mesh_export_cost_usd': .5, 'rgbd_view_cost_usd': .1, 'refined_splat_cost_usd': 5,
                'rate_card_reference': 'test', 'reviewed_by': 'test', 'reviewed_on': 'test'},
        task_context={'context_digest': 'task'}, source_digest='capture', configuration={}, check=lambda _: None,
        experimental=False, max_polls=1, model='test', source_survey_path=tmp_path / 'source-survey.json')
    if outcome == 'processing':
        assert result == {'status': 'processing', 'stage': 'refined_splat', 'operation_id': 'op-1'}
        assert calls == ['completion', 'refined_splat'] and not meshed
        return
    if outcome == 'refined':
        # The refined splat is the appearance, the mesh and the export; the draft stays evidence.
        assert calls == ['completion', 'refined_splat', 'mesh'] and meshed == [refined_root]
        assert result['appearance'] == {'basis': 'atlas_refined_splat', 'splat_input': {
            'digest': selection['digest'], 'flags': selection['flags'], 'sent_frame_count': 1,
            'left_out_frame_count': 1, 'heldout_frame_ids': []}}
        assert result['splat_kind'] == 'atlas_refined' and result['splat_manifest_digest'] == refined['digest']
        assert exports[0]['splat_root'] == refined_root
        assert json.loads((tmp_path / 'splat/atlas_depth_splat_manifest.json').read_text()) == draft
        return
    reasons = {'unaffordable': 'atlas_refined_splat_unaffordable',
               'provider_failed': 'atlas_refined_splat_provider_failed',
               'admitted': 'atlas_refined_splat_would_omit_admitted_completion'}
    assert result['appearance'] == {'basis': 'depth_fused_draft_fallback', 'fallback_reason': reasons[outcome]}
    assert ('refined_splat' in calls) is (outcome == 'provider_failed')
    assert calls[-1] == 'mesh' and result['splat_kind'] == 'depth_fused_draft'


@pytest.mark.parametrize('tamper', [None, 'generated_view', 'whole_form', 'other_pose'])
def test_the_handoff_accepts_a_refined_splat_of_the_captured_frames_only(tmp_path, tamper):
    root = job_fixture(tmp_path, same_pose=True)
    result = json.loads((root / 'result.json').read_text())
    result.pop('automation')
    shutil.move(root / 'automation/background_splat', root / 'splat')
    shutil.move(root / 'automation/background_mesh', root / 'mesh')
    manifest_path = root / 'splat/atlas_depth_splat_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    source = json.loads((root / 'pose/atlas_posed_rgbd_result.json').read_text()) if (
        root / 'pose/atlas_posed_rgbd_result.json').is_file() else None
    assert source is not None, 'fixture layout changed'
    captured = [row['frame_id'] for row in source['operation_receipt']['input_usage']['frames']]
    frames = [{'frame_id': frame_id} for frame_id in captured]
    if tamper == 'generated_view':
        frames.append({'frame_id': 'completion-wide-0', 'whole_view_review': {'admit_whole': True}})
    manifest.update(splat_kind='atlas_refined', task='posedImages2Splats', generated_content=True,
                    request_form='documented_reviewed_whole' if tamper == 'whole_form' else 'documented_minimal',
                    frames=frames)
    if tamper == 'other_pose':
        manifest['pose_result_digest'] = 'sha256:' + '9' * 64
    manifest = digest(manifest)
    write(manifest_path, manifest)
    registration_path = root / 'mesh/registered/atlas_mesh_registration.json'
    registration = json.loads(registration_path.read_text())
    registration['splat_manifest_digest'] = manifest['digest']
    write(registration_path, digest(registration))
    result.update(subject_origin='no_manipulated_subject', captured_scene_preserved=True,
        object_removal_performed=False, image_inpainting_performed=False, generated_completion_performed=False,
        mesh_available=True, splat_manifest_digest=manifest['digest'])
    write(root / 'result.json', digest(result))
    write(root / 'preview_job.json', {'state': 'ready', 'result_digest': result['digest']})
    if tamper is None:
        value = scene_source.atlas_scene_source(capture_root=tmp_path, output_root=tmp_path / 'prep')
        assert value['status'] == 'ready' and value['scene_content'] == 'captured_scene_preserved'
    else:
        with pytest.raises(ValueError, match='atlas_preserved_completion_not_bound|preserved_scene_pose_mismatch'):
            scene_source.atlas_scene_source(capture_root=tmp_path, output_root=tmp_path / 'prep')


@pytest.mark.parametrize('filled', [False, True])
def test_completion_context_keeps_the_selection_viewpoints_when_the_splat_selection_was_filled(
        tmp_path, monkeypatch, filled):
    pose_root = tmp_path / 'pose'
    source_root = pose_root / 'full-view-source-v1'
    source_root.mkdir(parents=True)
    pose = {'digest': 'source-pose'}
    write(source_root / 'atlas_posed_rgbd_result.json', pose)
    write(tmp_path / 'splat/atlas_depth_splat_manifest.json', {'digest': 'draft', 'splat_kind': 'depth_fused_draft'})
    frames = [{'frame_id': key} for key in ('cover', 'spread-1', 'distinct', 'spread-2', 'task-source')]
    write(pose_root / 'atlas_view_selection.json', {'diagnostics': [
        {'frame_id': 'distinct', 'decision': 'distinct_view'}, {'frame_id': 'near', 'decision': 'near_duplicate'},
        *([{'frame_id': 'spread-1', 'decision': 'time_spread_view'},
           {'frame_id': 'spread-2', 'decision': 'time_spread_view'}] if filled else [])]})
    monkeypatch.setattr(pipeline, '_source_pose', lambda **_: {'status': 'completed'})
    monkeypatch.setattr(source_inputs, 'prepare_source_pose', lambda _: (source_root, {}))
    monkeypatch.setattr(splats, 'load_posed_rgbd', lambda _: (pose, frames))
    monkeypatch.setattr(pipeline, 'load_configuration', lambda *args: {'remaining_cost_usd': 23})
    seen = []
    monkeypatch.setattr(allocator, 'run_sponsored_website_atlas_automation',
                        lambda **options: seen.append(options) or {'status': 'processing', 'stage': 'completion'})
    result = pipeline._preserved_scene(pose_root=pose_root, output_root=tmp_path, inputs={},
        review={'subject_origin': 'no_manipulated_subject', 'automation': {'objects': [], 'removal_instruction': ''}},
        policy={'mesh_export_cost_usd': .5, 'rgbd_view_cost_usd': .1, 'refined_splat_cost_usd': 5},
        task_context={'context_digest': 'task'}, source_digest='capture', configuration={}, check=lambda _: None,
        experimental=False, max_polls=1, model='test', source_survey_path=tmp_path / 'source-survey.json')
    assert result['status'] == 'processing' and seen[0]['frames'] is frames
    if filled:
        assert seen[0]['reference_frame_ids'] == ['cover', 'distinct', 'task-source']
    else:
        assert 'reference_frame_ids' not in seen[0]
