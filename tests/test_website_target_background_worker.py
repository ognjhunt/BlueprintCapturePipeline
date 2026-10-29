"""Real queue/controller/publication, with fake paid providers and retained source pixels."""
from copy import deepcopy
import json

import numpy as np
from PIL import Image
import pytest

from blueprint_pipeline.common import write_json
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline import website_target_background_worker as worker
from blueprint_pipeline import website_task_target_choice as choice
from blueprint_pipeline import website_atlas_preview as preview
from blueprint_pipeline.atlas_video_inputs import GEOMETRY_INPUTS_FILE
from blueprint_pipeline.local_reconstruction_adapters import _sha256_file
from tests.test_website_target_background_correction import case
from tests.test_website_task_target_choice import _answer


@pytest.fixture
def correction_job(tmp_path, monkeypatch):
    capture = tmp_path / 'site-job/captures/walkthrough-job'
    job_dir = capture / 'pipeline/atlas'
    job_dir.mkdir(parents=True)
    video = capture / 'retained.mp4'
    video.write_bytes(b'synthetic retained source, never uploaded')
    data = case()
    context = {'request_id': 'job', 'scene_id': 'site-job', 'capture_id': 'walkthrough-job', 'task_family': 'rigid_pick_place'}
    context['context_digest'] = canonical_digest(context)
    data['task_context'] = context
    b = data['targets'][0]['source_track']
    a = deepcopy(b)
    a['track_id'] = 'box-left'
    for row in a['observations']:
        row['runs'] = [{'start': 21, 'length': 2}, {'start': 31, 'length': 2}]
    geometry = {'digest': 'sha256:' + '9' * 64, 'binding': {'source_video_digest': _sha256_file(video)}}
    question = choice.build_question(target={**data['targets'][0], 'semantic_label': 'box'}, candidates=[a, b],
        frames=[{'frame_id': 'view-one', 'timestamp_seconds': 4}], guessed_track_id=a['track_id'],
        scene_id=context['scene_id'], capture_id=context['capture_id'], source_geometry=geometry)
    selectors = {**{key: context[key] for key in ('request_id', 'scene_id', 'capture_id')},
        'task_context_digest': context['context_digest'], 'completion_marker_generation': '1',
        'producer_delivery_key': 'sha256:' + '1' * 64, 'source_payload_sha256': '2' * 64}
    answer = _answer(question, selectors, {'kind': 'other_candidate', 'candidate_id': data['corrections'][0]['candidate_id']})
    for name, value in [('question', question), ('answer', answer)]:
        write_json(choice._path(capture / 'pipeline', question['target_id'], name), value)
    write_json(capture / 'pipeline' / choice.QUESTION_DIR / choice.INDEX_FILE,
        {'schema_version': choice.INDEX_SCHEMA, 'questions': {question['target_id']: question['question_digest']}})
    from blueprint_pipeline.website_preparation_status import DELIVERY_FILE
    write_json(capture / DELIVERY_FILE, {'selectors': selectors})
    pixels = np.zeros((10, 10, 3), dtype=np.uint8)
    pixels[2:4, 1:3] = [255, 0, 0]  # A must return from original source.
    pixels[2:4, 6:9] = [0, 0, 255]  # B is what the owner selected.
    raw = job_dir / 'pose/views/source.png'
    raw.parent.mkdir(parents=True)
    Image.fromarray(pixels).save(raw)
    row = {'frame_id': 'view-one', 'image_path': str(raw), 'image_digest': _sha256_file(raw), 'timestamp_seconds': 4}
    inputs = {'schema_version': 'atlas_geometry_inputs.v1', 'binding': {'source_video_digest': _sha256_file(video)},
              'frames': [row], 'heldout_pixels_included': False}
    inputs['digest'] = canonical_digest(inputs, digest_field='digest')
    write_json(job_dir / 'pose' / GEOMETRY_INPUTS_FILE, inputs)
    parent = {**data['parent_result'], 'source_video_digest': _sha256_file(video),
        'task_context_digest': context['context_digest'], 'model': 'retained-model', 'edit_scope': 'required_and_additional',
        'automation': {'status': 'completed', 'background': {'status': 'completed'}}, 'download': {'original': True}}
    parent['digest'] = canonical_digest(parent, digest_field='digest')
    job = {'schema_version': preview.SCHEMA, **{key: context[key] for key in ('request_id', 'scene_id', 'capture_id')},
        'state': 'ready', 'reason': None, 'stage': 'archived', 'source_video': video.name, 'revision': 5,
        'result_digest': parent['digest'], 'enqueued_at_epoch': 10, 'next_attempt_at_epoch': 10,
        'transient_failures': 0, 'archive': {'digest': 'sha256:' + '8' * 64},
        'provider_cleanup': {'state': 'complete', 'attempts': 1, 'deleted': ['old-A-asset']}}
    config = {'authority': {'provider_retention': 'existing-owner-policy', 'authorization_id': 'original',
        'maximum_cost_usd': '15', 'expires_at': 'unchanged'}, 'policy': {'edit_cost_usd': '1', 'maximum_review_cost_usd': 1}}
    for name, value in [('result.json', parent), ('preview_job.json', job), ('configuration.json', config),
                         ('selection_binding.json', {'review': data['review']})]:
        write_json(job_dir / name, value)
    masks = {'source_video_digest': _sha256_file(video), 'targets': data['targets']}
    masks['digest'] = canonical_digest(masks, digest_field='digest')
    calls = []
    def transport(*, operation, payload, **_):
        calls.append((operation, payload))
        if operation == 'atlas-configuration':
            assert payload['create'] is False
            return {**config, 'schema_version': 'website_atlas_configuration.v1',
                'task_context_digest': context['context_digest'], 'source_video_digest': _sha256_file(video),
                'dispatch_authorized': False, 'grant_creation_allowed': False}
        assert operation == 'task-target-choice'
        return answer
    from blueprint_pipeline import website_task_context
    monkeypatch.setattr(website_task_context, 'website_webapp_request', transport)
    monkeypatch.setattr(website_task_context, 'load_current_website_task_context', lambda **_: deepcopy(context))
    monkeypatch.setattr(preview, '_withdrawn', lambda _: False)
    monkeypatch.setattr(preview, '_archive', lambda *_: {'digest': 'sha256:' + '7' * 64})
    return locals()


def test_a_to_b_queues_then_continues_with_raw_b_edit_and_original_allowance(correction_job, monkeypatch):
    f = correction_job
    queued = worker.queue_correction(capture_root=f['capture'], task_context=f['context'], task_masks=f['masks'], now=20)
    assert queued['status'] == 'pending'
    job = json.loads((f['job_dir'] / 'preview_job.json').read_text())
    original_request = deepcopy(job['target_correction'])
    root, request = worker._bound_request(f['job_dir'], job['target_correction'])
    assert json.loads((root / 'parent_result.json').read_text()) == f['parent']
    assert json.loads((root / 'parent_job.json').read_text()) == f['job']
    assert job['enqueued_at_epoch'] == 10 and job['provider_cleanup']['deleted'] == ['old-A-asset']
    effects = []
    def fake_provider(**options):
        from blueprint_pipeline.atlas_background_automation import segment_objects
        from blueprint_pipeline.atlas_scene_automation import _unposed_source_views
        assert options['pose_root'] is None
        assert options['configuration']['authority'] == f['config']['authority']
        assert options['inputs']['frames'][0]['image_path'] == str(f['raw'])
        options['checkpoint']('before_fake_edit')
        _, frames = _unposed_source_views(options['inputs'])
        selected = segment_objects(frames=frames, objects=options['review']['automation']['objects'],
            inputs=options['inputs'], task_context=f['context'], source_digest=f['masks']['source_video_digest'],
            root=root / 'fake_segmentation', selected_tracks=options['selected_tracks'])
        from blueprint_pipeline.website_track_mask_codec import decode_track_mask
        edited = frames[0]['rgb'].copy()
        edited[decode_track_mask(selected[0]['track']['observations'][0])] = 0
        assert list(edited[2, 1]) == [255, 0, 0]  # A preserved, unlike old A-absent background.
        assert list(edited[2, 6]) == [0, 0, 0]  # B removed.
        effects.append(options)
        if len(effects) == 1:
            return {'status': 'processing', 'stage': 'fake_provider'}
        return {'status': 'completed', 'background': {'status': 'completed', 'mesh_available': True,
            'download': {'zip_sha256': 'sha256:' + '7' * 64}}}
    from blueprint_pipeline import paid_resource_allocator
    monkeypatch.setattr(paid_resource_allocator, 'run_sponsored_website_atlas_automation', fake_provider)
    pending = preview._step(job, job_dir=f['job_dir'], capture_root=f['capture'], now=lambda: 21)
    assert pending['state'] == 'processing'
    assert json.loads((f['job_dir'] / 'result.json').read_text()) == f['parent']
    ready = preview._step(pending, job_dir=f['job_dir'], capture_root=f['capture'], now=lambda: 22)
    assert ready['state'] == 'ready' and ready['enqueued_at_epoch'] == 10
    assert ready['target_correction'] == original_request
    result = json.loads((f['job_dir'] / 'result.json').read_text())
    assert ready['result_digest'] == result['digest'] != f['parent']['digest']
    assert result['source_pose'] == f['parent']['source_pose']
    assert result['simulation_qualified'] is False and result['physical_evidence'] is False
    assert json.loads((root / 'parent_result.json').read_text()) == f['parent']
    write_json(f['job_dir'] / 'preview_job.json', ready)
    assert worker.queue_correction(capture_root=f['capture'], task_context=f['context'], task_masks=f['masks'], now=23) == {'status': 'ready'}
    assert len(effects) == 2


@pytest.mark.parametrize('change', ['expiry', 'authority', 'pixels', 'video', 'owner_rejection', 'archive'])
def test_changed_source_or_authority_never_publishes_b_as_completed(correction_job, monkeypatch, change):
    f = correction_job
    worker.queue_correction(capture_root=f['capture'], task_context=f['context'], task_masks=f['masks'], now=20)
    job = json.loads((f['job_dir'] / 'preview_job.json').read_text())
    def forbidden(**_):
        pytest.fail('invalid current source must not allocate provider work')
    from blueprint_pipeline import paid_resource_allocator
    monkeypatch.setattr(paid_resource_allocator, 'run_sponsored_website_atlas_automation', forbidden)
    now = 21
    if change == 'expiry':
        now = preview.MAXIMUM_JOB_SECONDS + 30
    elif change == 'authority':
        f['config']['authority']['authorization_id'] = 'a-new-grant-is-not-the-original'
    elif change == 'pixels':
        Image.new('RGB', (10, 10), 'white').save(f['raw'])
    elif change == 'video':
        f['video'].write_bytes(b'a different source')
    elif change == 'owner_rejection':
        f['answer'].update(_answer(f['question'], f['selectors'], {'kind': 'none_of_these'}))
    else:
        monkeypatch.setattr(paid_resource_allocator, 'run_sponsored_website_atlas_automation',
            lambda **_: {'status': 'completed', 'background': {'status': 'completed', 'download': {'fake': True}}})
        monkeypatch.setattr(preview, '_archive', lambda *_: (_ for _ in ()).throw(ValueError('test_archive_failed')))
    result = preview._step(job, job_dir=f['job_dir'], capture_root=f['capture'], now=lambda: now)
    assert result['state'] != 'ready'
    assert json.loads((f['job_dir'] / 'result.json').read_text()) == f['parent']


def test_lost_native_lease_cannot_queue_a_new_provider_effect(correction_job):
    from blueprint_pipeline.core.handoff_lease import bind_handoff_lease
    f = correction_job
    boundaries = []
    def lost(boundary):
        boundaries.append(boundary)
        if len(boundaries) == 3:
            raise RuntimeError('lost-original-worker-lease')
    with bind_handoff_lease(lost), pytest.raises(RuntimeError, match='lost-original-worker-lease'):
        worker.queue_correction(capture_root=f['capture'], task_context=f['context'], task_masks=f['masks'], now=20)
    assert json.loads((f['job_dir'] / 'preview_job.json').read_text()) == f['job']
    assert json.loads((f['job_dir'] / 'result.json').read_text()) == f['parent']


def test_negative_answer_can_be_clarified_without_losing_its_retained_fact(correction_job):
    from blueprint_pipeline.website_target_background_correction import retain_rejected_selection
    f = correction_job
    ledger = {'producer_delivery_key': f['selectors']['producer_delivery_key'],
              'source_payload_sha256': f['selectors']['source_payload_sha256']}
    original_b = deepcopy(f['answer'])
    no = _answer(f['question'], f['selectors'], {'kind': 'none_of_these'})
    f['answer'].update(no)
    choice.fetch_answer(f['capture'], ledger, f['question'], now=30)
    handoff = retain_rejected_selection(capture_root=f['capture'], task_context=f['context'], source_video=f['video'])
    old_digest = choice.correction_binding_digest(f['capture'] / 'pipeline')
    completed = {'status': 'completed', 'attempt_history': [{choice.CORRECTION_MARK: old_digest}]}
    assert choice.correction_pending(completed, f['capture'] / 'pipeline') is None
    f['answer'].update(original_b)
    choice.fetch_answer(f['capture'], ledger, f['question'], now=95)
    new_digest = choice.correction_binding_digest(f['capture'] / 'pipeline')
    assert new_digest != old_digest
    assert choice.correction_pending(completed, f['capture'] / 'pipeline') == new_digest
    retained = f['capture'] / 'pipeline' / choice.QUESTION_DIR / 'history' / (no['answer_digest'][7:] + '.json')
    assert json.loads(retained.read_text()) == no
    rejected = f['capture'] / 'pipeline/website_scene_preparation/rejections' / handoff['selection_rejection']['digest'][7:] / 'handoff.json'
    assert json.loads(rejected.read_text()) == handoff


def test_reaffirming_the_original_guess_releases_only_its_rejected_outcome(correction_job):
    from blueprint_pipeline.website_target_background_correction import retain_rejected_selection
    from blueprint_pipeline.website_preparation_revision import workspace
    f = correction_job
    ledger = {'producer_delivery_key': f['selectors']['producer_delivery_key'],
              'source_payload_sha256': f['selectors']['source_payload_sha256']}
    f['answer'].update(_answer(f['question'], f['selectors'], {'kind': 'none_of_these'}))
    choice.fetch_answer(f['capture'], ledger, f['question'], now=30)
    retain_rejected_selection(capture_root=f['capture'], task_context=f['context'], source_video=f['video'])
    rejection_digest = choice.correction_binding_digest(f['capture'] / 'pipeline')
    completed = {'status': 'completed', 'attempt_history': [{choice.CORRECTION_MARK: rejection_digest}]}
    assert choice.correction_pending(completed, f['capture'] / 'pipeline') is None
    f['answer'].update(_answer(f['question'], f['selectors'], {'kind': 'confirm_guess',
        'candidate_id': f['question']['guessed_candidate_id']}))
    choice.fetch_answer(f['capture'], ledger, f['question'], now=95)
    assert choice.correction_pending(completed, f['capture'] / 'pipeline') is not None
    assert choice.correction_binding_digest(f['capture'] / 'pipeline') is None
    assert workspace(f['capture']) == f['capture'] / 'pipeline/website_scene_preparation'
    assert worker.queue_correction(capture_root=f['capture'], task_context=f['context'], task_masks=f['masks'], now=96) == {'status': 'ready'}
    # Completion replaces only the current projection; the immutable rejection
    # remains retained. No correction revision or new allowance is created.
    write_json(f['capture'] / 'pipeline/website_scene_preparation/handoff.json', {'status': 'intake_ready'})
    assert choice.correction_pending(completed, f['capture'] / 'pipeline') is None
