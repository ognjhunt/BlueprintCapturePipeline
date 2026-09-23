"""Reuse a completed SDK candidate and its conversation without rerunning tools.

The enclosing phase-adoption contract seals every retained byte and restores
the entire inference ledger. This reader additionally checks tool history,
candidate bindings and completed review responses before accepting a checkpoint.
"""
from __future__ import annotations

import base64
import json
from contextlib import closing
from pathlib import Path
import shutil
import sqlite3

from .decision_evidence_contracts import canonical_digest, canonical_json
from .task_object_astra_authoring import (
    AppearanceReview, AssetAuthoringError, BlenderProgram, VisualBrief, appearance_passed, file_record, save_json,
    validate_geometry_readback, validate_request,
)
from .task_object_physical_property_review import (
    PhysicalPropertyReviewInput, PhysicalPropertyReviewProposal, review_physical_properties,
)


def _read(path):
    return json.loads(path.read_text())


def inspect_agent_candidate(runtime: Path, request_value: dict) -> dict:
    root, session = runtime / 'authoring', runtime / 'inference/asset_session'
    previous = _read(root / 'request.json')
    request = validate_request(previous)
    def relevant(value):
        return {k: v for k, v in value.items() if k not in {'request_digest', 'expected_production_commit'}}
    if relevant(previous) != relevant(request_value):
        raise AssetAuthoringError('agent_resume_request_changed')
    lineage_path = root / 'retained_requests.json'
    lineage = _read(lineage_path) if lineage_path.exists() else []
    from .task_object_astra_inherited_inference import inherited_balance
    inherited = inherited_balance(runtime / 'inference', request.run_id)
    known_runs = {request.run_id, *(row['run_id'] for row in inherited['journals'])}
    def scientific(value):
        return {k: v for k, v in relevant(value).items() if k != 'run_id'}
    for ancestor in lineage:
        validate_request(ancestor)
        if scientific(ancestor) != scientific(previous) or ancestor['run_id'] not in known_runs:
            raise AssetAuthoringError('agent_resume_request_changed')
    request_runs = {r['request_digest']: r['run_id'] for r in [previous, *lineage]}
    request_digests = set(request_runs)
    if _read(session / 'binding.json') != dict(request_digest=request.request_digest,
            run_id=request.run_id, object_id=request.object_id):
        raise AssetAuthoringError('agent_resume_binding_changed')
    database = session / 'conversation.sqlite'
    wal = session / 'conversation.sqlite-wal'
    if wal.exists() and wal.stat().st_size:
        raise AssetAuthoringError('agent_resume_conversation_not_checkpointed')
    # Only closed/checkpointed sessions are reusable. Immutable inspection must
    # not create WAL sidecars in the retained, digest-bound source directory.
    with closing(sqlite3.connect(database.as_uri() + '?mode=ro&immutable=1', uri=True)) as connection:
        messages = [json.loads(row[0]) for row in connection.execute(
            'SELECT message_data FROM agent_messages WHERE session_id=? ORDER BY id', (request.object_id,))]
    returned = {m.get('call_id') for m in messages if m.get('type') == 'function_call_output'}
    latest = {}
    for index, message in enumerate(messages):
        if message.get('type') != 'function_call':
            continue
        call_id, name = message['call_id'], message['name']
        token = canonical_digest({'call_id': call_id})[7:]
        started = _read(session / 'tools' / (token + '.started.json'))
        if (call_id not in returned or started != dict(tool=name, call_id=call_id,
                arguments=json.loads(message['arguments']))):
            raise AssetAuthoringError('agent_resume_tool_history_changed')
        outcome = _read(session / 'tools' / (token + '.result.json'))
        latest[name] = (index, started['arguments'], outcome)
    names = ('observe_object', 'build_cad', 'render_candidate')
    if (any(n not in latest for n in names)
            or [latest[n][0] for n in names] != sorted(latest[n][0] for n in names)
            or (latest.get('inspect_candidate', (latest['render_candidate'][0],))[0]
                < latest['render_candidate'][0])
            or latest['observe_object'][2].get('status') != 'recorded'
            or latest['build_cad'][2].get('status') != 'built'
            or latest['render_candidate'][2].get('status') != 'rendered_pending_independent_review'
            or not messages or (messages[-1].get('role') not in {'assistant', 'user'}
                and not (messages[-1].get('type') == 'function_call_output'
                    and messages[-1].get('call_id') == messages[latest['render_candidate'][0]].get('call_id')))):
        raise AssetAuthoringError('agent_resume_candidate_not_completed')
    source = _read(root / 'source_analysis.json')
    brief = VisualBrief.model_validate(source['output'])
    if (source.get('origin') != 'asset_authoring_session' or source.get('request_digest') not in request_digests
            or source.get('references') != previous['source_frames']
            or brief.model_dump(mode='json') != latest['observe_object'][1]['brief']):
        raise AssetAuthoringError('agent_resume_source_analysis_changed')
    cad = _read(root / 'cad_result.json')
    original_root = Path(cad['step']['path']).parent.parent

    def verified(record):
        relative = Path(record['path']).relative_to(original_root)
        path = root / relative
        actual = file_record(path)
        if any(actual[k] != record[k] for k in ('sha256', 'size_bytes')):
            raise AssetAuthoringError('agent_resume_artifact_changed')
        return path

    for key in ('step', 'stl'):
        verified(cad[key])
    if (cad.get('execution') != 'agent_program_through_pinned_cad_cli'
            or 'program' not in cad or verified(cad['program']).read_text() != latest['build_cad'][1]['program']):
        raise AssetAuthoringError('agent_resume_cad_program_changed')
    readback = cad['readback']
    if (not cad.get('passed') or not cad.get('source_unchanged_after') or not readback.get('passed')
            or not readback.get('valid') or readback.get('solid_count') != 1
            or readback != latest['build_cad'][2]['readback']
            or readback != _read(verified(cad['step']).parent / 'step-readback.json')
            or readback['expected_dimensions_mm'] != [v * 1000 for v in request.dimensions_m]
            or readback['absolute_tolerance_mm'] != request.maximum_export_error_m * 1000):
        raise AssetAuthoringError('agent_resume_cad_readback_changed')
    rounds = sorted(root.glob('appearance-[0-9][0-9]'))
    attempt = rounds[-1]
    number = int(attempt.name.split('-')[-1]) + 1
    program = BlenderProgram.model_validate(_read(attempt / 'blender_program.json'))
    if (program.model_dump(mode='json') != latest['render_candidate'][1]['program']
            or (attempt / 'asset_program.py').read_text() != program.program):
        raise AssetAuthoringError('agent_resume_blender_program_changed')
    measurement = _read(attempt / 'geometry_readback.json')
    validate_geometry_readback(request, measurement, brief.proposed_appearance)
    receipt = _read(attempt / 'final_visual_mesh_receipt.json')
    if (receipt.get('receipt_digest') != canonical_digest(receipt, digest_field='receipt_digest')
            or receipt.get('source_cad_stl_sha256') != cad['stl']['sha256']
            or file_record(attempt / 'candidate.stl')['sha256'] != cad['stl']['sha256']
            or receipt.get('author_program_sha256') != file_record(attempt / 'asset_program.py')['sha256']
            or receipt.get('mesh_sha256') != file_record(attempt / 'final_visual_mesh.json')['sha256']
            or receipt.get('candidate_usd_sha256') != file_record(attempt / 'candidate.usdc')['sha256']):
        raise AssetAuthoringError('agent_resume_render_receipt_changed')
    render_artifacts = latest['render_candidate'][2].get('artifacts')
    if render_artifacts is not None:
        expected = {'candidate.usdc', 'candidate.blend', 'final_visual_mesh.json',
                    'final_visual_mesh_receipt.json', 'geometry_readback.json',
                    'perspective.png', 'top.png', 'side.png'}
        if set(render_artifacts) != expected or any(
                Path(record['path']).name != name
                or Path(record['path']).parent.name != attempt.name
                or any(file_record(attempt / name)[key] != record[key]
                       for key in ('sha256', 'size_bytes'))
                for name, record in render_artifacts.items()):
            raise AssetAuthoringError('agent_resume_render_artifacts_changed')
    elif 'inspect_candidate' not in latest:
        # A direct render-to-review handoff needs digest-bound views. Earlier
        # sessions can use the inspected image bytes recorded in the tool log.
        raise AssetAuthoringError('agent_resume_render_artifacts_missing')
    else:
        for name in ('candidate.blend', 'perspective.png', 'top.png', 'side.png'):
            file_record(attempt / name)
    if 'inspect_candidate' in latest:
        inspected = latest['inspect_candidate'][2]
        images = [v.get('image_url') for v in inspected if v.get('type') == 'input_image']
        expected_images = ['data:image/png;base64,' + base64.b64encode((attempt / name).read_bytes()).decode()
                           for name in ('perspective.png', 'top.png', 'side.png')]
    else:
        images = expected_images = []  # The render now hands off directly to independent review.
    if images != expected_images or measurement != latest['render_candidate'][2]['measurement']:
        raise AssetAuthoringError('agent_resume_inspected_render_changed')
    physics_input = _read(root / 'physical_review_input.json')
    physical_phase = _read(root / f'physical_property_review_{number}.json')
    proposal = PhysicalPropertyReviewProposal.model_validate(physical_phase['output'])
    physics = review_physical_properties(PhysicalPropertyReviewInput.model_validate(physics_input), proposal)
    if (physics.accepted is None or physics.model_dump(mode='json') != _read(root / 'physical_property_review_result.json')
            or physical_phase.get('request_digest') not in request_digests
            or physical_phase.get('references') != previous['source_frames']):
        raise AssetAuthoringError('agent_resume_physics_review_changed')
    aliases = [e for e in physics_input['evidence'] if e['evidence_id'] == 'source-appearance-analysis']
    if aliases and (len(aliases) != 1 or aliases[0]['sha256'] != file_record(root / 'source_analysis.json')['sha256'].removeprefix('sha256:')):
        raise AssetAuthoringError('agent_resume_physics_source_changed')
    completions = [_read(p) for p in (runtime / 'inference/inference_reservations/completed').glob('*.json')]
    for journal in inherited['journals']:
        completions.extend(_read(p) for p in (runtime / 'inference' / journal['relative_root'] /
                           'inference_reservations/completed').glob('*.json'))
    matched = [c for c in completions if c.get('capability') == request.object_id + f'_physical_property_review_{number}'
        and c.get('structured_output_digest') == canonical_digest(proposal.model_dump(mode='json'))
        and c.get('run_id') == request_runs[physical_phase['request_digest']] and c.get('provider') == 'openai' and c.get('model') == 'gpt-6-astra'
        and c.get('inference_completion_digest') == canonical_digest(c, digest_field='inference_completion_digest')]
    if len(matched) != 1:
        raise AssetAuthoringError('agent_resume_physics_completion_missing')
    completed_result, retained_visual_review = None, None
    from .task_object_agent_tools import APPEARANCE_SCOPE
    current_review = f'independent_visual_review_{number}_{APPEARANCE_SCOPE}'
    review_path = attempt / (current_review + '.json')
    if not review_path.exists():
        review_path = attempt / f'independent_visual_review_{number}.json'
    if review_path.exists():
        phase = _read(review_path)
        review = AppearanceReview.model_validate(phase['output'])
        references = phase.get('references', [])
        if (phase.get('request_digest') not in request_digests
                or references[:len(previous['source_frames'])] != previous['source_frames']
                or [r['sha256'] for r in references[len(previous['source_frames']):]] !=
                    [file_record(attempt / n)['sha256'] for n in ('perspective.png', 'top.png', 'side.png')]
                or len([c for c in completions if c.get('capability') == request.object_id + '_' + review_path.stem
                        and c.get('run_id') == request_runs[phase['request_digest']] and c.get('provider') == 'openai' and c.get('model') == 'gpt-6-astra'
                        and c.get('structured_output_digest') == canonical_digest(review.model_dump(mode='json'))
                        and c.get('inference_completion_digest') == canonical_digest(c, digest_field='inference_completion_digest')]) != 1):
            raise AssetAuthoringError('agent_resume_final_review_changed')
        retained_visual_review = review.model_dump(mode='json')
    if messages[-1].get('role') == 'user':
        feedback = 'Independent review requires corrections. Keep working in this session:\n' + canonical_json(
            {'accepted': False, 'review': retained_visual_review})
        if retained_visual_review is None or messages[-1].get('content') != feedback:
            raise AssetAuthoringError('agent_resume_candidate_not_completed')
    if (root / 'result.json').exists():
        from .task_object_astra_retained_artifacts import completed_authoring
        if retained_visual_review is None or not appearance_passed(review, generated=request.generated_specification is not None):
            raise AssetAuthoringError('agent_resume_final_review_not_passed')
        completed_result = completed_authoring(root, previous, record_validator=verified)
        completion = _read(session / 'completion.json')
        if any(completion.get(k) != v for k, v in dict(request_digest=request.request_digest,
                run_id=request.run_id, object_id=request.object_id, status='independently_accepted_candidate',
                result_digest=completed_result['result_digest']).items()):
            raise AssetAuthoringError('agent_resume_completion_changed')
        def mapped(value):
            if isinstance(value, dict):
                if {'path', 'sha256', 'size_bytes'} <= set(value):
                    return {**value, **file_record(verified(value))}
                return {k: mapped(v) for k, v in value.items()}
            if isinstance(value, list):
                return [mapped(v) for v in value]
            return value
        completed_result = mapped(completed_result)
    author_calls = [int(c['capability'].removeprefix(request.object_id + '_author_turn_')) for c in completions
                    if str(c.get('capability', '')).startswith(request.object_id + '_author_turn_')]
    # A rejection under an obsolete review scope is historical evidence, not
    # reusable feedback for the corrected appearance-only contract. Accepted
    # historical assets retain their original acceptance and provenance.
    if (retained_visual_review is not None and review_path.stem != current_review
            and not appearance_passed(review, generated=request.generated_specification is not None)):
        retained_visual_review = None
    return {'original_root': str(original_root), 'brief': brief.model_dump(mode='json'), 'cad': cad,
        'render_attempts': number, 'cad_attempts': max(int(p.name.split('-')[-1]) for p in root.glob('cad-[0-9][0-9]')) + 1,
        'author_calls': max(author_calls), 'physics_input': physics_input, 'physics_proposal': proposal.model_dump(mode='json'),
        'completed_result': completed_result, 'retained_requests': [*lineage, previous],
        'retained_visual_review': retained_visual_review,
        'current_scope_visual_reviews': sum(
            '_independent_visual_review_' in str(c.get('capability'))
            and str(c.get('capability')).endswith('_' + APPEARANCE_SCOPE) for c in completions),
        'completed_visual_reviews': sum('_independent_visual_review_' in str(c.get('capability')) for c in completions)}


def restore_agent_candidate(asset, runtime: Path, state_root: Path, *, source_request=None) -> dict:
    current = asset.request.model_dump(mode='json')
    if source_request is not None:
        from .task_evaluation_partial_astra_successor import semantic_request
        if (_read(runtime / 'authoring/request.json') != source_request
                or semantic_request(source_request) != semantic_request(current)):
            raise AssetAuthoringError('agent_resume_successor_inputs_changed')
    state = inspect_agent_candidate(runtime, source_request or current)
    source = runtime / 'authoring'
    original = Path(state['original_root'])
    for path in source.iterdir():
        if path.name in {'request.json', 'failure.json', 'cache', 'tmp', 'xdg'}:
            continue
        target = asset.root / path.name
        if path.is_dir():
            shutil.copytree(path, target)
        else:
            shutil.copyfile(path, target)
    def relocate(value):
        if isinstance(value, dict):
            return {k: relocate(v) for k, v in value.items()}
        if isinstance(value, list):
            return [relocate(v) for v in value]
        if isinstance(value, str) and value.startswith(str(original) + '/'):
            return str(asset.root / Path(value).relative_to(original))
        return value
    asset.brief = VisualBrief.model_validate(state['brief'])
    save_json(asset.root / 'retained_requests.json', state['retained_requests'])
    asset.cad = relocate(state['cad'])
    save_json(asset.root / 'cad_result.json', asset.cad)
    asset.cad_attempts, asset.render_attempts = state['cad_attempts'], state['render_attempts']
    attempt = asset.root / f'appearance-{asset.render_attempts - 1:02d}'
    asset.candidate = {'directory': attempt, 'cad_digest': canonical_digest(asset.cad),
        'brief_digest': canonical_digest(asset.brief.model_dump(mode='json')),
        'artifacts': {name: file_record(attempt / name) for name in (
            'candidate.usdc', 'candidate.blend', 'final_visual_mesh.json', 'final_visual_mesh_receipt.json',
            'geometry_readback.json', 'perspective.png', 'top.png', 'side.png')}}
    asset.retained_physics = (state['physics_input'], state['physics_proposal'])
    asset.retained_visual_review = state['retained_visual_review']
    asset.source_evidence_identity = next((e for e in state['physics_input']['evidence']
                                         if e['evidence_id'] == 'source-appearance-analysis'), None)
    prior_session = runtime / 'inference/asset_session'
    shutil.copyfile(prior_session / 'conversation.sqlite', state_root / 'conversation.sqlite')
    shutil.copytree(prior_session / 'tools', state_root / 'tools')
    asset.validate_candidate()
    return state
