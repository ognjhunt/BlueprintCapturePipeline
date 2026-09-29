# Covers (for impacted-test selection):
#   src/blueprint_pipeline/task_evaluation_scene_retirement.py
#   src/blueprint_pipeline/task_evaluation_scene_retirement_access.py
#   src/blueprint_pipeline/task_evaluation_scene_retirement_generations.py
"""ADP-009D/day28: real enrolled scene, full preservation and restoration.

All bytes are tiny development-only fixtures. Historical planner flags remain
false: protected consent, actual births and current engine checks supply action
authority independently. No provider, host or production cleanup is exercised.
"""
from __future__ import annotations

import copy
import ast
import hashlib
import json
import os
import tempfile
import stat
import time
from pathlib import Path

import pytest


def _raw(path):
    path = Path(path)
    data = path.read_bytes()
    return {'path': str(path), 'sha256': 'sha256:' + hashlib.sha256(data).hexdigest(),
            'size_bytes': len(data)}


def _sealed_file(path, value, field, *, mode=0o600):
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest
    value = dict(value)
    value[field] = canonical_digest(value, digest_field=field)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, separators=(',', ':')))
    path.chmod(mode)
    return value


def _strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _strings(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _strings(item)


def _record_pairs(args):
    for group in ('seed_records', 'downstream_records', 'source_records', 'bridge_records'):
        for role, values in args.get(group, {}).items():
            values = ([] if values is None else [values]) if role in ('intent', 'projection') else values
            yield from values


def _rebase_complete_graph(args, anchor, changes, *, requests=None, immutable_paths=(),
                           complete_identities=False, replacement_sizes=None, attempt_runtime=None):
    """Resolve fixture aliases through the complete acyclic raw-record DAG.

    The existing stat-only fixture rebaser intentionally replaces a selector
    only once. Authentic intake plus a newly selected SAM parent needs the
    transitive replacement too. This is bounded fixture encoding, no observer
    or action implementation and no successful protection flags.
    """
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest, cross_runtime_canonical_digest
    original = copy.deepcopy(args)
    pairs = list(_record_pairs(original))
    maps, sizes = dict(changes), dict(replacement_sizes or {})
    anchor = str(anchor)

    def aliases(value):
        seen = set()
        while value in maps and value not in seen:
            seen.add(value)
            value = maps[value]
        return value

    def text(value):
        value = aliases(value)
        if value == '/retained' or value.startswith('/retained/'):
            value = anchor + value[len('/retained'):]
        for old, new in sorted(maps.items(), key=lambda item: -len(item[0])):
            new = aliases(new)
            if old.startswith('sha256:'):
                value = value.replace(old[7:], new[7:])
            elif old.startswith(('sam31-', 'scene-configuration-')):
                value = value.replace(old, new)
        return value

    def visit(value, *, allow_replacement=True):
        if isinstance(value, str):
            return text(value)
        if isinstance(value, list):
            return [visit(item) for item in value]
        if not isinstance(value, dict):
            return value
        replacement = (requests or {}).get(canonical_digest(value)) if allow_replacement else None
        if replacement is not None:
            return visit(copy.deepcopy(replacement), allow_replacement=False)
        changed = {key: visit(item) for key, item in value.items()}
        if (value.get('schema_version') == 'task_evaluation_scene_attempt_binding.v1'
                and changed.get('attempt_id') in (attempt_runtime or {})):
            changed['runtime_digest'] = attempt_runtime[changed['attempt_id']]
        if complete_identities:
            for key in ('identity','scene_identity','task_identity','output_identity','subject_identity'):
                identity=changed.get(key)
                if isinstance(identity,dict) and 'id' in identity and 'version' not in identity:
                    identity['version']='v1'
        for field in ('sha256', 'digest'):
            selected = value.get(field)
            if isinstance(selected, str) and 'size_bytes' in value:
                while selected in maps and selected not in sizes:
                    next_value = maps[selected]
                    if next_value == selected:
                        break
                    selected = next_value
                if selected in sizes:
                    changed['size_bytes'] = sizes[selected]
        if 'request' in value and value.get('request_digest') == canonical_digest(value['request']):
            changed['request_digest'] = canonical_digest(changed['request'])
            maps[value['request_digest']] = changed['request_digest']
        if value.get('schema_version') == 'task_evaluation_sam31_preparation_execution_job.v1':
            changed['inputs_digest'] = canonical_digest({name: {key: row[key] for key in ('sha256', 'size_bytes')}
                                                        for name, row in changed['inputs'].items()})
            maps[value['inputs_digest']] = changed['inputs_digest']
            changed['child_id'] = 'sam31-' + canonical_digest({key: changed[key] for key in
                ('parent_request_digest', 'plan_digest', 'phase', 'inputs_digest')})[7:]
            maps[value['child_id']] = changed['child_id']
        if value.get('schema_version') == 'task_evaluation_scene_attempt.v1' and value.get('attempt_id', '').startswith('scene-configuration-'):
            changed['attempt_id'] = 'scene-configuration-' + changed['input_digest'][7:31]
            maps[value['attempt_id']] = changed['attempt_id']
        for field, digest in value.items():
            if (field.endswith('_digest') or field == 'digest') and isinstance(digest, str):
                method = (canonical_digest if digest == canonical_digest(value, digest_field=field) else
                          cross_runtime_canonical_digest if digest == cross_runtime_canonical_digest(value, digest_field=field)
                          else None)
                if method:
                    changed[field] = method(changed, digest_field=field)
                    maps[digest] = changed[field]
        if value.get('schema_version') == 'task_evaluation_launch_preparation_result.v1' and value.get('status') == 'queued_for_production_episode_compilation':
            from blueprint_pipeline.task_evaluation_scene_compilation_owner_preparations import HANDOFF, PRE
            def pre_handoff(record):
                record = {key: item for key, item in record.items() if key not in HANDOFF | {'result_digest'}}
                return dict(record, status=PRE)
            maps[canonical_digest(pre_handoff(value), digest_field='result_digest')] = canonical_digest(
                pre_handoff(changed), digest_field='result_digest')
        maps[canonical_digest(value)] = canonical_digest(changed)
        maps[cross_runtime_canonical_digest(value)] = cross_runtime_canonical_digest(changed)
        return changed

    previous = None
    for _ in range(64):
        current = {}
        for path, raw in pairs:
            if path in immutable_paths:
                current[path, raw] = raw
                continue
            try:
                value = json.loads(raw)
            except (ValueError, UnicodeError):
                rewritten = raw
            else:
                rewritten = json.dumps(visit(value), sort_keys=True).encode()
            current[path, raw] = rewritten
            digest = 'sha256:' + hashlib.sha256(raw).hexdigest()
            maps[digest] = 'sha256:' + hashlib.sha256(rewritten).hexdigest()
            sizes[digest] = len(rewritten)
        if current == previous:
            break
        previous = current
    else:
        raise AssertionError('authentic fixture raw selector graph did not converge')
    result = copy.deepcopy(original)
    for key in ('roots', 'parent_routes', 'retained_metadata_roots'):
        result[key] = visit(original[key])
    for group in ('seed_records', 'downstream_records', 'source_records', 'bridge_records'):
        for role, rows in original.get(group, {}).items():
            def rebound(pair):
                return text(pair[0]), current[pair]
            result[group][role] = (None if rows is None else rebound(rows)) if role in ('intent', 'projection') else [rebound(pair) for pair in rows]
    return result


def _add_current_sam(args, owned_current_task):
    """Bind a real-shaped current SAM job to the scene's own preparation.

    The older connected fixture intentionally contains only an adopted original
    SAM prefix. Re-seal the genuine metadata DAG; never annotate an owner flag.
    """
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest
    from tests.scene_lifecycle_fixture_support import rebase_graph
    from tests.test_scene_source_family_sam import fixture as sam_fixture
    from tests.test_scene_source_family_website import pair, ref, seal
    current = sam_fixture(parent_id='prep-1')
    previous_task = current['source_records']['sam_host_tasks'][0]
    current_task = pair('/retained/host-inputs/current-scene-task.json', owned_current_task)
    # Current and independently owned original tasks are different raw records.
    current = _rebase_complete_graph(current, '/retained', {previous_task[0]: current_task[0],
        current['source_records']['sam_profiles'][0][0]: '/retained/metadata/current-scene-profile.json',
        current['source_records']['sam_plans'][0][0]: '/retained/metadata/current-scene-plan.json',
        's3://test/plan.json': 's3://test/current-scene-plan.json'},
        requests={canonical_digest(json.loads(previous_task[1])): owned_current_task})
    old_parent = json.loads(current['source_records']['sam_parent_envelopes'][0][1])
    previous_pair = args['seed_records']['preparation_envelopes'][0]
    previous = json.loads(previous_pair[1])
    parent = copy.deepcopy(previous)
    parent['request']['runtime']['mounts'] = old_parent['request']['runtime']['mounts']
    for mount in parent['request']['runtime']['mounts']:
        mount.update(mode='read_only',container_path='/inputs/sam31-plan.json')
    parent['request_digest'] = canonical_digest(parent['request'])
    parent = seal(parent, 'envelope_digest')
    path = args['roots']['preparation_queue_root'] + '/completed/prep-1-' + parent['request_digest'][7:] + '.json'
    parent_pair = pair(path, parent)
    args['seed_records']['preparation_envelopes'][0] = parent_pair
    # Preserve the real producer's retained copy on its own configured route.
    # Semantic ownership must distinguish identical retained copies from
    # conflicting byte versions while retaining each raw provenance path.
    parent_queue = '/retained/sam-parent-queue'
    current['parent_routes'].append({'queue_root': parent_queue,
                                    'input_root': args['roots']['preparation_input_root']})
    args['parent_routes'].append({'queue_root': parent_queue,
                                 'input_root': args['roots']['preparation_input_root']})
    current['source_records']['sam_parent_envelopes'] = [pair(
        parent_queue + '/completed/prep-1-' + parent['request_digest'][7:] + '.json', parent)]
    changes = {old_parent['request_digest']: parent['request_digest']}
    current = rebase_graph(current, '/retained', remote_digest_replacements=changes)
    for role, rows in current['source_records'].items():
        args['source_records'].setdefault(role, [])
        args['source_records'][role] += [row for row in rows if row not in args['source_records'][role]]
    return {previous['request_digest']: parent['request_digest'],
            ref(previous_pair)['sha256']: ref(parent_pair)['sha256']}


def _add_adopted_current_sam(args, base, owned_current_task):
    """Keep the a-commit prefix while giving its b-commit adoption its own parent."""
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest
    from tests.test_scene_source_family_sam import fixture as sam_fixture
    current = sam_fixture(parent_id='current-parent')
    previous_task = current['source_records']['sam_host_tasks'][0]
    previous_plan = current['source_records']['sam_plans'][0]
    plan = json.loads(previous_plan[1])
    for key in ('scene_identity', 'task_identity', 'publisher_scene_id'):
        plan[key] = copy.deepcopy(owned_current_task[key])
    plan['plan_digest'] = canonical_digest(plan, digest_field='plan_digest')
    previous_parent = current['source_records']['sam_parent_envelopes'][0]
    request = json.loads(previous_parent[1])['request']
    request['scene']['identity'] = copy.deepcopy(owned_current_task['scene_identity'])
    request['task']['identity'] = copy.deepcopy(owned_current_task['task_identity'])
    adopted_task = next(row for row in args['source_records']['sam_host_tasks']
        if json.loads(row[1])['expected_production_commit']=='b'*40)
    current = _rebase_complete_graph(current, base, {'a'*40:'b'*40,
        previous_task[0]:adopted_task[0],
        current['source_records']['sam_profiles'][0][0]:'/retained/metadata/adopted-current-profile.json',
        current['source_records']['sam_plans'][0][0]:'/retained/metadata/adopted-current-plan.json',
        's3://test/plan.json':'s3://test/adopted-current-plan.json'},
        requests={canonical_digest(json.loads(previous_task[1])):owned_current_task,
                  canonical_digest(json.loads(previous_plan[1])):plan,
                  canonical_digest(json.loads(previous_parent[1])['request']):request},
        complete_identities=True)
    for role, rows in current['source_records'].items():
        if role == 'sam_host_tasks':
            continue
        args['source_records'].setdefault(role,[])
        args['source_records'][role] += [row for row in rows if row not in args['source_records'][role]]


def _link_current_parent(args, current_birth):
    """Write the immutable owner link for the real b worker's selected parent."""
    from blueprint_pipeline.task_evaluation_controls_autoprovision import build_preparation_link
    from blueprint_pipeline.task_evaluation_scene_intake import write_exclusive

    intent = json.loads(args['seed_records']['intent'][1])
    attempt = json.loads(Path(current_birth['path']).read_bytes())
    parent_pair = next(row for row in args['source_records']['sam_parent_envelopes']
                       if json.loads(row[1])['request']['preparation_id'] == 'current-parent')
    parent = json.loads(parent_pair[1])
    request = parent['request']
    result_pair = next(row for row in args['source_records']['sam_parent_results']
                       if Path(row[0]).name == Path(parent_pair[0]).name)
    result = json.loads(result_pair[1])
    assert result['status'] == 'queued_for_production_scene_configuration'
    assert result['source_commit'] == request['expected_production_commit'] == attempt['source_commit']
    link = build_preparation_link(intent_id=intent['intent_id'], intent_digest=intent['intent_digest'],
        preparation_id=request['preparation_id'], request_digest=parent['request_digest'],
        expected_production_commit=request['expected_production_commit'],
        team_namespace=request['team_namespace'], scene_id=request['scene']['identity']['id'],
        task_id=request['task']['identity']['id'], result_filename=Path(parent_pair[0]).name)
    path = (Path(args['roots']['intent_root']) / intent['intent_id'] / 'preparations' /
            (parent['request_digest'][7:] + '.json'))
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o750)
    write_exclusive(path, link)
    args['seed_records']['preparation_links'].append((str(path), path.read_bytes()))
    args['seed_records']['preparation_envelopes'].append(parent_pair)
    args['seed_records']['preparation_results'].append(result_pair)
    args['source_records']['sam_parent_envelopes'].remove(parent_pair)
    args['source_records']['sam_parent_results'].remove(result_pair)


def _complete_current_requests(args):
    """Tiny actual producer fields; no readback, reader or ownership waiver."""
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest
    args=_rebase_complete_graph(args,'/retained',{},complete_identities=True)
    maps={}
    leaf={'uri':'s3://fixture/contracts.bin','digest':'sha256:'+hashlib.sha256(b'x').hexdigest(),'size_bytes':1}
    native_revision=args['bridge_records']['configured_revisions'][0]
    revision_ref={'uri':'s3://fixture/revision.json','digest':_raw_pair(native_revision)['sha256'],
        'size_bytes':len(native_revision[1])}
    requests={}
    def request_value(value):
        body=copy.deepcopy(value)
        if body['schema_version']=='task_evaluation_launch_activation_request.v1':
            initial=body['lane']=='task_evaluation_scene_configuration'
            body['lineage']={'kind':'initial_project' if initial else 'predecessor',**{
                k:dict(leaf) for k in (('project_spend_reconciliation','initial_provider_zero') if initial else
                    ('prior_authority','prior_result','prior_launch_receipt','prior_webapp_sync','prior_provider_zero',
                     'prior_spend_reconciliation','construction_result'))}}
            body.setdefault('authorization',{}).update(reference='fixture-owner-reviewed')
            body['requested_mutations']={'profile_publication':True,'catalog_synchronization':True,
                'standing_authorization':True,'policy_campaign_queue':False}
            return body
        scene=body['scene']
        if body['preparation_id']=='native-prep':
            scene.update(mode='reuse_configured_revision',configured_revision=dict(revision_ref))
        else:
            scene.update(mode='configure_source_scene',source_manifest=dict(leaf),
                appearance={'kind':'observed','representation':dict(leaf),'renderer_qualification':dict(leaf)},
                geometry={'kind':'observed','collision':dict(leaf),'validation':dict(leaf)},
                registration={k:dict(leaf) for k in ('metric_registration','support_plane','robot_mount_interface',
                    'workspace_clearance','camera_calibration')},
                rights={'admission':dict(leaf),'evidence':[{'role':role,'artifact':dict(leaf)}
                    for role in ('publisher_terms','human_authority_record')],'source_bytes_redistributable':False})
        body.setdefault('construction',{'mode':'reuse_configured_scene'})
        task=body['task']
        task.pop('artifact',None)
        if body['preparation_id']=='native-prep':
            task.update(binding_mode='reuse_configured_template',subject={'mode':'configured_scene_object'})
        else:
            task.update(binding_mode='define_configuration_template',subject={'mode':'configured_scene_object'},
                definition=dict(leaf),success_criteria=dict(leaf),execution=dict(leaf))
        body['sensors']={'configuration':dict(leaf)}
        body.setdefault('runtime',{}).update(health_protocol=dict(leaf))
        body['runtime'].setdefault('mounts',[])
        for mount in body['runtime']['mounts']:
            mount.update(mode='read_only',container_path='/inputs/sam31-plan.json')
        bundle=body.setdefault('execution_adapter',{}).get('runtime_source_bundle',{})
        body['execution_adapter']['runtime_source_bundle']={**leaf,**bundle}
        if body['run_mode']!='scene_configuration':
            body['robot']={k:dict(leaf) for k in ('configuration','kinematics','joint_bounds','base_registration','controller_configuration')}
            body['controller']={'kind':'zero_action','configuration':dict(leaf)}
        return body
    for group in ('seed_records','source_records','bridge_records'):
        for role,rows in args[group].items():
            for name,raw in rows if role not in ('intent','projection') else []:
                value=json.loads(raw) if raw.startswith(b'{') else {}
                if value.get('schema_version') in ('task_evaluation_launch_preparation_envelope.v1','task_evaluation_launch_activation_envelope.v1'):
                    requests[value['request_digest']]=request_value(value['request'])
    args=_rebase_complete_graph(args,'/retained',{},requests=requests)
    # The actual worker preserves every materialized request leaf, with content
    # reuse by raw digest. Keep original transitive result rows as well.
    parent={}
    for group in ('seed_records','source_records','bridge_records'):
        for role,rows in args[group].items():
            for name,raw in rows if role not in ('intent','projection') else []:
                value=json.loads(raw) if raw.startswith(b'{') else {}
                if value.get('schema_version')=='task_evaluation_launch_preparation_envelope.v1':
                    parent[value['request']['preparation_id']]=value['request']
    def leaves(value,path=''):
        if isinstance(value,dict):
            if set(value)=={'uri','digest','size_bytes'}:
                yield path,value
            else:
                for key,item in value.items():
                    yield from leaves(item,path+'.'+key if path else key)
        elif isinstance(value,list):
            for i,item in enumerate(value):
                yield from leaves(item,path+'.'+str(i))
    changed_results={}
    for group in ('seed_records','bridge_records'):
        for role,rows in args[group].items():
            for i,(name,raw) in enumerate(rows if role not in ('intent','projection') else []):
                value=json.loads(raw) if raw.startswith(b'{') else {}
                if value.get('schema_version')!='task_evaluation_launch_preparation_result.v1':
                    continue
                old=value['result_digest']
                request=parent[value['preparation_id']]
                refs={r['contract_path']:r for r in value.get('references',[])}
                if 'task.artifact' in refs:
                    row=refs.pop('task.artifact')
                    row['contract_path']='task.definition'
                    refs['task.definition']=row
                for contract,remote in leaves(request):
                    refs[contract]=dict(contract_path=contract,**remote,
                        materialized_path=args['roots']['preparation_input_root']+'/'+value['preparation_id']+'/'+remote['digest'][7:],
                        content_addressed_reuse=False,full_byte_service_account_readback_passed=True)
                value.update(references=list(refs.values()),reference_count=len(refs),
                    unique_object_count=len({(r['digest'],r['size_bytes']) for r in refs.values()}),content_addressed_reuse_count=0)
                value['result_digest']=canonical_digest(value,digest_field='result_digest')
                maps[old]=value['result_digest']
                changed_results[value['preparation_id']]=value
                rows[i]=(name,json.dumps(value,sort_keys=True).encode())
    from blueprint_pipeline.task_evaluation_scene_compilation_owner_preparations import HANDOFF,PRE
    rows=args['downstream_records']['compilation_envelopes']
    for i,(name,raw) in enumerate(rows):
        value=json.loads(raw)
        final=changed_results[value['preparation_id']]
        inverse={k:v for k,v in final.items() if k not in HANDOFF|{'result_digest'}}
        inverse['status']=PRE
        old=value['envelope_digest']
        value.update(materialized_references=final['references'],request=parent[value['preparation_id']],
            preparation_result_digest=canonical_digest(inverse,digest_field='result_digest'))
        value['envelope_digest']=canonical_digest(value,digest_field='envelope_digest')
        maps[old]=value['envelope_digest']
        rows[i]=(name,json.dumps(value,sort_keys=True).encode())
    args=_rebase_complete_graph(args,'/retained',maps)
    for i,(name,raw) in enumerate(args['seed_records']['activation_envelopes']):
        args['seed_records']['activation_envelopes'][i]=(name.replace('/pending/','/prepared/'),raw)
    local=args['roots']['preparation_input_root']+'/native-prep/'+revision_ref['digest'][7:]
    args['bridge_records']['configured_revisions']=[(local,native_revision[1])]
    if args['roots']['preparation_input_root'] not in args['retained_metadata_roots']:
        args['retained_metadata_roots'].append(args['roots']['preparation_input_root'])
    return args


def _raw_pair(pair):
    return {'sha256':'sha256:'+hashlib.sha256(pair[1]).hexdigest(),'size_bytes':len(pair[1])}


def _native_fixture_records(args):
    """Complete genuine intake fields and tiny promised materialized bytes."""
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest
    maps = {}
    args = _complete_current_requests(copy.deepcopy(args))
    for group in ('seed_records', 'downstream_records', 'source_records', 'bridge_records'):
        for role, pairs in args.get(group, {}).items():
            if role in ('intent', 'projection'):
                continue
            for index, (path, raw) in enumerate(pairs):
                try:
                    value = json.loads(raw)
                except ValueError:
                    continue
                if value.get('schema_version') not in (
                    'task_evaluation_launch_preparation_envelope.v1',
                    'task_evaluation_launch_activation_envelope.v1'):
                    continue
                value.update(submitted_by='fixture-authenticated-intake',
                    submitted_at_iso='1970-01-01T00:01:40+00:00',
                    provider_mutation_performed_inside_intake=False,
                    catalog_mutation_performed_inside_intake=False)
                if value['schema_version'] == 'task_evaluation_launch_activation_envelope.v1':
                    value.update(standing_authorization_published_inside_intake=False, paid_execution_requested=False)
                old_seal = value['envelope_digest']
                value['envelope_digest'] = canonical_digest(value, digest_field='envelope_digest')
                rewritten = json.dumps(value, sort_keys=True).encode()
                maps[old_seal] = value['envelope_digest']
                maps['sha256:' + hashlib.sha256(raw).hexdigest()] = 'sha256:' + hashlib.sha256(rewritten).hexdigest()
                pairs[index] = (path, rewritten)
    payloads = {}
    existing = {path: raw for path, raw in _record_pairs(args)}
    retained_bytes = {('sha256:' + hashlib.sha256(raw).hexdigest(), len(raw)): raw
                      for raw in existing.values()}
    def collect(value):
        if isinstance(value, dict):
            if {'materialized_path', 'digest', 'size_bytes'} <= value.keys():
                selected = retained_bytes.get((value['digest'], value['size_bytes']))
                payloads[value['materialized_path']] = (value['digest'], selected if selected is not None
                    else b'x' if value['size_bytes']==1 else b'tiny-bundle')
            for item in value.values():
                collect(item)
        elif isinstance(value, list):
            for item in value:
                collect(item)
    for _, raw in list(_record_pairs(args)):
        try:
            collect(json.loads(raw))
        except ValueError:
            pass
    # A real local derivative supplies the bytes named by the publication
    # receipt. The later readback uses a distinct object copy, never source bytes.
    for _, raw in args['source_records']['submission_publications']:
        value=json.loads(raw)
        manifest_path = next(path for path, _ in _record_pairs(args) if path.endswith('/materialized/submission/bundle_manifest.v1.json'))
        for row in value['published_objects']:
            if row['relative_path'] != 'bundle_manifest.v1.json':
                payloads[str(Path(manifest_path).parent / row['relative_path'])] = (
                    row['digest'], json.dumps({'fixture':True}, sort_keys=True).encode())
    for path, (digest, raw) in payloads.items():
        maps[digest] = 'sha256:' + hashlib.sha256(raw).hexdigest()
        if path in existing:
            assert existing[path] == raw, 'materialized fixture conflicts with retained raw bytes'
        else:
            args['source_records']['opaque_evidence'].append((path, raw))
    result = _rebase_complete_graph(args, '/retained', maps)
    # Content-addressed materialization can converge multiple contract paths
    # onto one physical object; retain one exact record, never drop variants.
    result['source_records']['opaque_evidence'] = list(dict.fromkeys(result['source_records']['opaque_evidence']))
    count_maps={}
    for group in ('seed_records','bridge_records'):
        for role,rows in result[group].items():
            for i,(path,raw) in enumerate(rows if role not in ('intent','projection') else []):
                value=json.loads(raw)
                if value.get('schema_version')!='task_evaluation_launch_preparation_result.v1':
                    continue
                old=value['result_digest']
                from blueprint_pipeline.task_evaluation_scene_compilation_owner_preparations import HANDOFF,PRE
                def preimage(record):
                    return dict({k:v for k,v in record.items() if k not in HANDOFF|{'result_digest'}},status=PRE)
                prior_inverse=canonical_digest(preimage(value),digest_field='result_digest')
                value['unique_object_count']=len({(r['digest'],r['size_bytes']) for r in value['references']})
                value['content_addressed_reuse_count']=sum(r['content_addressed_reuse'] for r in value['references'])
                value['result_digest']=canonical_digest(value,digest_field='result_digest')
                count_maps[old]=value['result_digest']
                count_maps[prior_inverse]=canonical_digest(preimage(value),digest_field='result_digest')
                rows[i]=(path,json.dumps(value,sort_keys=True).encode())
    return _rebase_complete_graph(result,'/retained',count_maps)


def _complete_installed_queue_layouts(context):
    """Install the real producer's finite states and auxiliary directories."""
    from blueprint_pipeline.task_evaluation_sam31_phase_queue import STATES as sam_states
    from blueprint_pipeline.task_evaluation_launch_activation_queue import QUEUE_STATES as activation_states
    from blueprint_pipeline.control_plane_queue_auxiliary_observation import _ROLES
    from blueprint_pipeline.task_evaluation_scene_construction_queue import QUEUE_STATES as construction_states
    Path(context['pins_root']).mkdir(exist_ok=True)
    current = context['primary_queue_contracts'][0]
    contracts = [current, {'root_path':context['roots']['sam_queue_root'], 'states':list(sam_states)},
                 {'root_path':context['roots']['activation_queue_root'], 'states':list(activation_states)},
                 {'root_path':context['roots']['compilation_queue_root'], 'states':list(construction_states)}]
    context['primary_queue_contracts'] = contracts
    for contract in contracts:
        for state in contract['states']:
            (Path(contract['root_path']) / state).mkdir(parents=True, exist_ok=True)
    for contract in context['auxiliary_queue_contracts']:
        for relative, *_ in _ROLES[contract['family']]:
            (Path(contract['root_path']) / relative).mkdir(parents=True, exist_ok=True)


def _selected_worker_preparations(args, base, policy, monkeypatch, owner, birth, current_birth):
    """Run the no-provider queue producer for selected successful parents.

    The composed historical graph includes unfinished preparatory records. Keep
    their original bytes outside the selected queue and bind the scene's active
    links to fresh, actually materialized worker results.
    """
    import os
    import pwd
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest
    from blueprint_pipeline.task_evaluation_scene_progression_transport import submit_owned_preparation
    from blueprint_pipeline.task_evaluation_launch_preparation_worker import process_launch_preparation_queue
    from blueprint_pipeline import task_evaluation_scene_retirement_cache as cache
    from tests.test_task_evaluation_launch_preparation_worker import (
        fetcher, fake_scene_render_inputs, production_request_with_fetchable_bytes)

    queue_root = Path(args['roots']['preparation_queue_root'])
    queue_root.mkdir(parents=True, exist_ok=True)
    input_root = Path(args['roots']['preparation_input_root'])
    input_root.mkdir(parents=True, exist_ok=True)
    sam_execution_root = Path(args['roots']['sam_execution_root'])
    sam_execution_root.mkdir(parents=True, exist_ok=True)
    authority_root = base / 'preparation-authority'
    authority_root.mkdir(mode=0o700)
    policy['roots'] = [{'root': str(input_root), 'storage_class': 'cache', 'device': input_root.stat().st_dev},
                       {'root': str(authority_root), 'storage_class': 'host', 'device': authority_root.stat().st_dev},
                       {'root': str(sam_execution_root), 'storage_class': 'host', 'device': sam_execution_root.stat().st_dev}]
    _sealed_file(base / 'policy.json', policy, 'policy_digest', mode=0o644)
    monkeypatch.setenv('BLUEPRINT_SCENE_RETIREMENT_POLICY_FILE', str(base / 'policy.json'))
    monkeypatch.setenv('BLUEPRINT_TASK_EVALUATION_SCENE_INTAKE_ROOT', str(base / 'intents'))
    monkeypatch.setenv('BLUEPRINT_TASK_EVALUATION_SCENE_INTAKE_CLIENT_IDS', 'webapp')
    monkeypatch.setattr(cache.time, 'time', lambda: 200)
    old_rows = {'prep-1': (args['seed_records']['preparation_envelopes'][0],
                            args['seed_records']['preparation_results'][0])}
    adopted_parent=next(row for row in args['source_records']['sam_parent_envelopes']
        if json.loads(row[1])['request']['preparation_id']=='current-parent')
    old_rows['current-parent']=(adopted_parent,None)
    old_original_parent = next(row for row in args['source_records']['sam_parent_envelopes']
                               if json.loads(row[1])['request']['preparation_id']=='parent-1')
    old_alternate = next(row for row in args['source_records']['sam_parent_envelopes']
                         if json.loads(row[1])['request']['preparation_id']=='prep-1')
    history = base / 'unselected-queue-history'
    history.mkdir(mode=0o700)
    changes = {}
    replacement_sizes = {}
    immutable = set()
    results = {}
    requests = {}
    for identifier, (old_envelope, old_result) in old_rows.items():
        for label, row in (('envelope', old_envelope), ('result', old_result)):
            if row is not None:
                (history / (identifier + '-' + label + '.json')).write_bytes(row[1])
        request, payloads = production_request_with_fetchable_bytes()
        old_request = json.loads(old_envelope[1])['request']
        request.update(preparation_id=identifier, run_id=old_request['run_id'],
                       team_namespace=old_request['team_namespace'],
                       expected_production_commit=old_request['expected_production_commit'])
        request['scene_intent_digest'] = json.loads(Path(owner['path']).read_bytes())['intent_digest']
        request['scene']['identity'] = {'id':old_request['scene']['identity']['id'],'version':'v1'}
        request['task']['identity'] = {'id':old_request['task']['identity']['id'],'version':'v1'}
        recipe_ref = request['construction']['recipe']
        recipe = json.loads(payloads[recipe_ref['uri']])
        if identifier == 'current-parent':
            recipe_ref['uri'] = 's3://blueprint-production-inputs/current-parent-recipe.json'
        recipe.update(team_namespace=request['team_namespace'],scene_identity=request['scene']['identity'],
                      task_identity=request['task']['identity'])
        old_mount = old_request['runtime']['mounts'][0]
        request['runtime']['mounts'][0] = copy.deepcopy(old_mount)
        request['runtime']['mounts'][0].update(mode='read_only',container_path='/inputs/sam31-plan.json')
        mount = request['runtime']['mounts'][0]['source']
        plan = next(row for row in args['source_records']['sam_plans']
                    if _raw_pair(row)=={'sha256':mount['digest'],'size_bytes':mount['size_bytes']})
        payloads[mount['uri']] = plan[1]
        stage_ref = recipe['stage_sequence'][0]['configuration']
        stage = json.loads(payloads[stage_ref['uri']])
        if identifier == 'current-parent':
            stage_ref['uri'] = 's3://blueprint-production-inputs/current-parent-stage-1.json'
        stage.update(sam31_review_kind='ai', sam31_preparation_plan=copy.deepcopy(mount))
        stage_bytes = json.dumps(stage, sort_keys=True).encode()
        stage_ref.update(digest='sha256:'+hashlib.sha256(stage_bytes).hexdigest(),
                         size_bytes=len(stage_bytes))
        payloads[stage_ref['uri']] = stage_bytes
        recipe['recipe_digest'] = canonical_digest(recipe,digest_field='recipe_digest')
        recipe_bytes = json.dumps(recipe,sort_keys=True).encode()
        recipe_ref.update(digest='sha256:'+hashlib.sha256(recipe_bytes).hexdigest(),size_bytes=len(recipe_bytes))
        payloads[recipe_ref['uri']] = recipe_bytes
        request_path = authority_root / (identifier + '-submission-request.json')
        request_path.write_text(json.dumps(request, sort_keys=True))
        request_path.chmod(0o600)
        intent = json.loads(Path(owner['path']).read_bytes())
        selected_birth = current_birth if identifier=='current-parent' else birth
        attempt = json.loads(Path(selected_birth['path']).read_bytes())
        factory_path = authority_root / (identifier + '-factory.json')
        _sealed_file(factory_path, dict(schema_version='website_scene_attempt_factory.v1',
            status='publication_ready', intent_digest=intent['intent_digest'],
            attempt_digest=attempt['attempt_digest'], source_commit=attempt['source_commit'],
            submission_request=_raw(request_path), provider_mutation_performed=False), 'factory_digest')
        from blueprint_pipeline.task_evaluation_launch_preparation_contract import validate_launch_preparation_request
        from blueprint_pipeline.task_evaluation_scene_configuration_submission_inputs import read
        validated = validate_launch_preparation_request(read(request_path))
        assert validated == request
        assert attempt['source_commit'] == request['expected_production_commit']
        assert request['scene_intent_digest'] == intent['intent_digest']
        assert request['task']['identity']['id'] == intent['request']['task']['task_id']
        staged = submit_owned_preparation(request_path=request_path,
            config={'preparation_queue_root': str(queue_root)}, intent_reference=owner,
            attempt_reference=selected_birth, factory_reference=_raw(factory_path))
        assert staged['status'] == 'submitted'
        requests[identifier] = request
        def forbidden_adapter(**_kwargs):
            raise AssertionError('scene preparation must not invoke a provider adapter')
        run = process_launch_preparation_queue(queue_root=queue_root,input_root=input_root,
            allowed_uri_prefixes=['s3://blueprint-production-inputs/','s3://test/'],
            service_account=pwd.getpwuid(os.geteuid()).pw_name,
            source_commit=request['expected_production_commit'],fetcher=fetcher(payloads),
            adapter_materializer=forbidden_adapter,scene_render_input_materializer=fake_scene_render_inputs,
            construction_queue_root=base/'worker-construction-queue')
        assert len(run['results'])==1 and run['results'][0]['status']=='queued_for_production_scene_configuration',run
        digest=canonical_digest(request)
        name=identifier+'-'+digest[7:]+'.json'
        new_envelope=next((queue_root/state/name for state in ('materialized','completed')
                           if (queue_root/state/name).is_file()),queue_root/'materialized'/name)
        new_result=queue_root/'results'/name
        identity=queue_root/'identities'/(identifier+'.json')
        assert all(path.is_file() for path in (new_envelope,new_result,identity))
        new_envelope_pair=(str(new_envelope),new_envelope.read_bytes())
        new_result_pair=(str(new_result),new_result.read_bytes())
        results[identifier]=(new_envelope_pair,new_result_pair,(str(identity),identity.read_bytes()))
        old_value=json.loads(old_envelope[1])
        changes.update({old_envelope[0]:str(new_envelope),
            old_value['request_digest']:digest,
            _raw_pair(old_envelope)['sha256']:_raw_pair(new_envelope_pair)['sha256']})
        replacement_sizes[_raw_pair(old_envelope)['sha256']]=len(new_envelope_pair[1])
        if old_result is not None:
            changes.update({old_result[0]:str(new_result),
                json.loads(old_result[1])['result_digest']:json.loads(new_result_pair[1])['result_digest'],
                _raw_pair(old_result)['sha256']:_raw_pair(new_result_pair)['sha256']})
            replacement_sizes[_raw_pair(old_result)['sha256']]=len(new_result_pair[1])
        immutable.update((str(new_envelope),str(new_result),str(identity)))
    args['seed_records']['preparation_envelopes'][0],args['seed_records']['preparation_results'][0],_ = results['prep-1']
    args['source_records']['sam_parent_results'].append(results['current-parent'][1])
    (history/'prep-1-alternate-envelope.json').write_bytes(old_alternate[1])
    primary = results['prep-1'][0]
    alternate_path = str(Path(old_alternate[0]).parent / Path(primary[0]).name)
    alternate = (alternate_path, primary[1])
    immutable.add(alternate_path)
    changes.update({old_alternate[0]:alternate_path,
        json.loads(old_alternate[1])['request_digest']:json.loads(primary[1])['request_digest'],
        _raw_pair(old_alternate)['sha256']:_raw_pair(alternate)['sha256']})
    replacement_sizes[_raw_pair(old_alternate)['sha256']]=len(primary[1])
    (history/'parent-1-original-envelope.json').write_bytes(old_original_parent[1])
    original_parent_root = base/'original-sam-parent-queue'
    original_parent = (str(original_parent_root/'completed'/Path(old_original_parent[0]).name),old_original_parent[1])
    changes[old_original_parent[0]]=original_parent[0]
    immutable.add(original_parent[0])
    args['parent_routes'].append({'queue_root':str(original_parent_root),
                                  'input_root':args['roots']['preparation_input_root']})
    args['source_records']['sam_parent_envelopes']=[
        original_parent if row==old_original_parent else alternate if row==old_alternate
        else results['current-parent'][0] if row==adopted_parent else row
        for row in args['source_records']['sam_parent_envelopes']]
    args['source_records']['queue_identities']=[row[2] for row in results.values()]
    from blueprint_pipeline.task_evaluation_scene_intake import reserve_scene_attempt
    request = requests['prep-1']
    request_digest = canonical_digest(request)
    old_attempt_index,old_attempt = next((index,row) for index,row in enumerate(args['seed_records']['attempts'])
        if json.loads(row[1]).get('schema_version')=='task_evaluation_scene_attempt.v1'
        and json.loads(row[1]).get('attempt_id','').startswith('scene-configuration-'))
    (history/'prep-1-scene-attempt.json').write_bytes(old_attempt[1])
    new_attempt = reserve_scene_attempt(queue_root=args['roots']['intent_root'],intent_id=args['intent_id'],
        attempt_id='scene-configuration-'+request_digest[7:31],source_commit=request['expected_production_commit'],
        runtime_digest=request['execution_adapter']['runtime_source_bundle']['digest'],
        input_digest=request_digest,provider='vast',maximum_spend_usd=1.0,now=200)
    attempt_path = str(Path(args['roots']['intent_root'])/args['intent_id']/'attempts'/(new_attempt['attempt_id']+'.json'))
    attempt_pair = (attempt_path,Path(attempt_path).read_bytes())
    args['seed_records']['attempts'][old_attempt_index]=attempt_pair
    old_attempt_value=json.loads(old_attempt[1])
    changes.update({old_attempt[0]:attempt_path,
        old_attempt_value['attempt_id']:new_attempt['attempt_id'],
        old_attempt_value['attempt_digest']:new_attempt['attempt_digest'],
        _raw_pair(old_attempt)['sha256']:_raw_pair(attempt_pair)['sha256']})
    replacement_sizes[_raw_pair(old_attempt)['sha256']]=len(attempt_pair[1])
    immutable.add(attempt_path)
    immutable.add(args['seed_records']['intent'][0])
    immutable.update(path for path,_ in args['source_records']['sam_host_tasks'])
    # The real worker's process queue owns this cache root. Bind the seed
    # inventory to its actual publication location before any plan is built.
    args['roots']['content_store_root'] = str(input_root / 'content-addressed' / 'sha256')
    return _rebase_complete_graph(args,base,changes,immutable_paths=immutable,
                                  replacement_sizes=replacement_sizes,
                                  attempt_runtime={new_attempt['attempt_id']:new_attempt['runtime_digest']})


def _source_owned_configuration(base, monkeypatch, policy):
    """Create the selected preparation through the actual no-provider producer."""
    import pwd
    from urllib.parse import urlsplit
    from tests import test_task_evaluation_completed_scene_progression as progression
    from tests.test_task_evaluation_scene_configuration_submission import SHA
    from tests.test_task_evaluation_scene_configuration_submission_publication import Store
    from blueprint_pipeline import task_evaluation_scene_progression as engine
    from blueprint_pipeline import task_evaluation_scene_configuration_submission_publication as publication
    from blueprint_pipeline import task_evaluation_launch_preparation_worker as worker
    from blueprint_pipeline.task_evaluation_launch_preparation_queue import ensure_launch_preparation_queue_root
    from blueprint_pipeline.task_evaluation_owner_source_store import PREFIX

    source = base / 'source-produced'
    source.mkdir()
    queue = ensure_launch_preparation_queue_root(source / 'preparations')
    input_root = source / 'worker-inputs'
    input_root.mkdir()
    progression_output = source / 'progression-output'
    progression_output.mkdir()
    original_owner = progression._owner
    def active_owner(store, now, **kwargs):
        owner = original_owner(store, now, **kwargs)
        owner['execution'].update(max_total_spend_usd=100,
                                  expires_at_epoch=now + 6 * 24 * 60 * 60)
        return owner
    monkeypatch.setattr(progression, '_owner', active_owner)
    try:
        config, intent_id, intake, issued_at = progression._config(
            source, monkeypatch, submission_enabled=True, source_kind='mesh', real_destination=True,
            extra={'preparation_queue_root': str(queue),
                   'publication_lock_root': str(source / 'publication-locks'),
                   'service_account': pwd.getpwuid(os.geteuid()).pw_name,
                   'submission_transport': 'local_owned_queue', 'activation_enabled': False})
    finally:
        monkeypatch.setattr(progression, '_owner', original_owner)
    policy['roots'] = [
        {'root': str(input_root), 'storage_class': 'cache', 'device': input_root.stat().st_dev},
        {'root': str(progression_output), 'storage_class': 'host',
         'device': progression_output.stat().st_dev}]
    _sealed_file(base / 'policy.json', policy, 'policy_digest', mode=0o644)
    monkeypatch.setenv('BLUEPRINT_SCENE_RETIREMENT_POLICY_FILE', str(base / 'policy.json'))
    store = Store()
    monkeypatch.setattr(publication, '_verified_checkout_head', lambda: SHA)
    def publish(**kwargs):
        return publication.publish_scene_configuration_submission(**kwargs, client=store)
    progression_run = engine.process_scene_intents(config_path=config, publisher=publish, now=issued_at)
    assert len(progression_run['results']) == 1
    assert progression_run['results'][0]['status'] == 'running', progression_run
    def fetch(uri, destination, maximum_bytes):
        if uri.startswith(PREFIX):
            return worker.default_reference_fetcher(uri, destination, maximum_bytes)
        data = store.objects[urlsplit(uri).path.lstrip('/')]
        assert len(data) == maximum_bytes
        destination.write_bytes(data)
    preparation_run = worker.process_launch_preparation_queue(
        queue_root=queue, input_root=input_root,
        allowed_uri_prefixes=['s3://blueprint/task-evaluation/'],
        service_account=pwd.getpwuid(os.geteuid()).pw_name,
        source_commit=SHA, fetcher=fetch, construction_queue_root=source / 'construction')
    assert len(preparation_run['results']) == 1
    assert preparation_run['results'][0]['status'] == 'queued_for_production_scene_configuration', preparation_run
    result_path = next((queue / 'results').glob('*.json'))
    result = json.loads(result_path.read_bytes())
    envelope_path = next((queue / state / result_path.name for state in ('materialized', 'completed')
                          if (queue / state / result_path.name).is_file()), queue / 'materialized' / result_path.name)
    assert envelope_path.is_file()
    request = json.loads(envelope_path.read_bytes())['request']
    member = input_root / request['preparation_id']
    generation = Path(policy['generation_store']) / (hashlib.sha256(str(member).encode()).hexdigest() + '.json')
    assert json.loads(generation.read_bytes())['canonical_path'] == str(member)
    return {'base': source, 'queue': queue, 'input_root': input_root, 'progression_output': progression_output,
            'intake': intake, 'intent_id': intent_id, 'owner': _raw(intake / intent_id / 'intent.json'),
            'request': request, 'envelope': (str(envelope_path), envelope_path.read_bytes()),
            'result': (str(result_path), result_path.read_bytes()), 'issued_at': issued_at}


def _authentic_connected_graph(base, monkeypatch, policy):
    from tests.test_scene_lifecycle_connected_acquisition import full_connected_finished_scene, installed
    from tests.test_task_evaluation_scene_intake import request, stage, attempt
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest, cross_runtime_canonical_digest

    args = _native_fixture_records(full_connected_finished_scene())
    # The source-family regression fixture deliberately has no current parent
    # for this progress row. A successful retirement must use a separately
    # produced, parent-bound progress chain instead of adopting those bytes.
    args['source_records']['source_progress'] = []
    old_intent = json.loads(args['seed_records']['intent'][1])
    body = request()
    body['execution']['allowed_providers'] = ['vast', 'openai']
    body['owner'] = old_intent['request']['owner']
    body['consent']['accepted_by'] = body['owner']['user_id']
    body['source'] = old_intent['request']['source']
    body['task']['task_id'] = old_intent['request']['task']['task_id']
    intake = base / 'intents'
    accepted = stage(intake, body)
    issued_attempt = attempt(intake, accepted, commit='a', cost=1)
    current_attempt = attempt(intake, accepted, attempt_id='a2', commit='b', cost=1)
    actual_path = intake / accepted['intent_id'] / 'intent.json'
    actual = json.loads(actual_path.read_bytes())
    prior_body = copy.deepcopy(body)
    prior_body['submission_id'] = 'original-sam-owner'
    prior_body['owner'] = {'user_id': 'original-owner', 'organization_id': 'original-org'}
    prior_body['consent']['accepted_by'] = 'original-owner'
    prior = stage(intake, prior_body)
    prior_attempt = attempt(intake, prior, commit='a')
    prior_path = intake / prior['intent_id'] / 'intent.json'
    prior_value = json.loads(prior_path.read_bytes())
    owner = _raw(actual_path)
    birth = _raw(intake / accepted['intent_id'] / 'attempts' / (issued_attempt['attempt_id'] + '.json'))
    current_birth = _raw(intake / accepted['intent_id'] / 'attempts' / (current_attempt['attempt_id'] + '.json'))
    prior_owner = _raw(prior_path)
    prior_birth = _raw(intake / prior['intent_id'] / 'attempts' / (prior_attempt['attempt_id'] + '.json'))
    from datetime import datetime, timezone
    def owned_task(task, intent, raw_owner, raw_attempt):
        task = copy.deepcopy(task)
        for key in ('strategy', 'subject', 'support', 'destination', 'success'):
            task[key] = intent['request']['task'][key]
        task['scene_intent_authority'] = {'intent': raw_owner, 'intent_digest': intent['intent_digest'], 'attempt': raw_attempt}
        task['human_authority'] = {'accepted_by': intent['request']['owner']['user_id'],
            'authority_reference': 'scene-intent:' + intent['intent_digest'],
            'accepted_on': datetime.fromtimestamp(intent['request']['consent']['accepted_at_epoch'], timezone.utc).isoformat()}
        return task
    task_versions = [json.loads(raw) for _, raw in args['source_records']['sam_host_tasks']]
    original_task = next(task for task in task_versions if task['expected_production_commit'] == 'a'*40)
    adopted_task = next(task for task in task_versions if task['expected_production_commit'] == 'b'*40)
    own_current = owned_task(original_task, actual, owner, birth)
    replacements = {canonical_digest(old_intent['request']): actual['request']}
    for task in task_versions:
        replacements[canonical_digest(task)] = owned_task(task, prior_value if task['expected_production_commit'] == 'a'*40 else actual,
            prior_owner if task['expected_production_commit'] == 'a'*40 else owner,
            prior_birth if task['expected_production_commit'] == 'a'*40 else current_birth)
    current_changes = _add_current_sam(args, own_current)
    args = _rebase_complete_graph(args,'/retained',current_changes)
    args = _native_fixture_records(args)
    changes={}
    changes.update({old_intent['intent_digest']: actual['intent_digest'],
                    old_intent['task_content_digest']: actual['task_content_digest'],
                    canonical_digest(old_intent['request']): canonical_digest(actual['request']),
                    cross_runtime_canonical_digest(old_intent['request']): cross_runtime_canonical_digest(actual['request'])})
    old_id = args['intent_id']
    # The shared rebaser recomputes every existing seal and raw selector. Exact
    # full-string replacements also move IDs embedded in producer filenames.
    for path, raw in list(_record_pairs(args)):
        values = [path]
        try:
            values.extend(_strings(json.loads(raw)))
        except (ValueError, UnicodeError):
            pass
        for value in values:
            if old_id in value:
                changes[value] = value.replace(old_id, accepted['intent_id']).replace('/retained', str(base))
    changes[old_id] = accepted['intent_id']
    args['intent_id'] = accepted['intent_id']
    args['seed_records']['intent'] = (str(actual_path), actual_path.read_bytes())
    args = _rebase_complete_graph(args, base, changes, requests=replacements,
                                 immutable_paths={str(actual_path), str(prior_path)})
    assert all(json.loads(raw).get('scene_intent_authority',{}).get('intent')==owner
        for _,raw in args['source_records']['sam_host_tasks']
        if json.loads(raw)['expected_production_commit']=='b'*40), [
            (path,json.loads(raw).get('scene_intent_authority',{}).get('intent'))
            for path,raw in args['source_records']['sam_host_tasks']
            if json.loads(raw)['expected_production_commit']=='b'*40]
    _add_adopted_current_sam(args, base, owned_task(adopted_task, actual, owner, current_birth))
    task_bytes_before_worker=list(args['source_records']['sam_host_tasks'])
    args = _selected_worker_preparations(args, base, policy, monkeypatch, owner, birth, current_birth)
    _link_current_parent(args, current_birth)
    assert args['source_records']['sam_host_tasks']==task_bytes_before_worker, [
        (old[0],hashlib.sha256(old[1]).hexdigest(),new[0],hashlib.sha256(new[1]).hexdigest())
        for old,new in zip(task_bytes_before_worker,args['source_records']['sam_host_tasks']) if old!=new]
    assert all(json.loads(raw).get('scene_intent_authority',{}).get('intent')==owner
        for _,raw in args['source_records']['sam_host_tasks']
        if json.loads(raw)['expected_production_commit']=='b'*40), 'worker rebase changed b owner'
    # Queue intake writes one sealed identity for each exact request before its
    # envelope. The older historical fixture omitted these companion records.
    # Preserve every observed queue root, including the original SAM parent.
    identities = {}
    for path, raw in _record_pairs(args):
        try:
            envelope = json.loads(raw)
        except (ValueError, UnicodeError):
            continue
        schema = envelope.get('schema_version')
        if schema not in {'task_evaluation_launch_preparation_envelope.v1',
                          'task_evaluation_launch_activation_envelope.v1'}:
            continue
        family = 'preparation' if 'preparation' in schema else 'activation'
        identifier = envelope['request'][family + '_id']
        identity_path = str(Path(path).parent.parent / 'identities' / (identifier + '.json'))
        identity = {'schema_version': 'task_evaluation_launch_' + family + '_identity.v1',
                    family + '_id': identifier, 'request_digest': envelope['request_digest']}
        identity['identity_digest'] = canonical_digest(identity, digest_field='identity_digest')
        encoded = json.dumps(identity, sort_keys=True).encode()
        assert identity_path not in identities or identities[identity_path] == encoded
        identities[identity_path] = encoded
    args['source_records']['queue_identities'] = sorted({
        **identities, **dict(args['source_records']['queue_identities'])}.items())
    # Intake publishes immutable owner bytes. The metadata installer must not
    # reopen that real authority for writing merely to install fixture copies.
    authentic_pair = (str(actual_path), actual_path.read_bytes())
    args['seed_records']['intent'] = None
    args, context, _, _ = installed(base, args, already_rebased=True)
    args['seed_records']['intent'] = authentic_pair
    _complete_installed_queue_layouts(context)
    _produce_current_sam_phase(args, context, base, monkeypatch)
    _produce_current_sam_progress(args, context, base)
    assert actual_path.read_bytes() == authentic_pair[1]
    monkeypatch.setenv('BLUEPRINT_TASK_EVALUATION_SCENE_INTAKE_ROOT', str(intake))
    monkeypatch.setenv('BLUEPRINT_TASK_EVALUATION_SCENE_INTAKE_CLIENT_IDS', 'webapp')
    from blueprint_pipeline.task_evaluation_scene_owner_authority import validate_task_scene_owner
    for task_path, raw in args['source_records']['sam_host_tasks']:
        task = json.loads(raw)
        verified = validate_task_scene_owner(task, now=200)
        expected = prior['intent_id'] if task['expected_production_commit'] == 'a'*40 and task['scene_intent_authority']['intent']==prior_owner else accepted['intent_id']
        assert verified['intent_id'] == expected
    task_index = {(path, 'sha256:' + hashlib.sha256(raw).hexdigest(), len(raw)): json.loads(raw)
                  for path, raw in args['source_records']['sam_host_tasks']}
    for path, raw in args['source_records']['sam_plans']:
        selector = json.loads(raw)['host_inputs']['task_request']
        assert (selector['path'], selector['sha256'], selector['size_bytes']) in task_index, (
            path, selector, [(key, value.get('scene_intent_authority',{}).get('intent'))
                             for key,value in task_index.items() if key[0]==selector['path']])
        task = task_index[selector['path'], selector['sha256'], selector['size_bytes']]
        expected_owner = owner if Path(path).name in {'current-scene-plan.json', 'adopted-current-plan.json'} else prior_owner
        assert task['scene_intent_authority']['intent'] == expected_owner, (
            path, task['expected_production_commit'], task['scene_intent_authority']['intent']['path'],
            expected_owner['path'])
    for _, raw in args['source_records']['sam_adoptions']:
        selector = json.loads(raw)['current_host_inputs']['task_request']
        assert task_index[selector['path'], selector['sha256'], selector['size_bytes']]['scene_intent_authority']['intent'] == owner
    assert actual['intent_digest'] == canonical_digest(actual, digest_field='intent_digest')
    from blueprint_pipeline.task_evaluation_scene_source_family_inventory import ROLES, _join
    from blueprint_pipeline.task_evaluation_scene_source_family_sam import SCHEMAS as SAM_SCHEMAS
    for role, (schema, field) in SAM_SCHEMAS.items():
        if field:
            for path, raw in args['source_records'].get(role, []):
                value = json.loads(raw)
                if value.get('schema_version') == schema:
                    assert value[field] == canonical_digest(value, digest_field=field), (role, path)
    source_args={key:args[key] for key in ('intent_id','seed_records','downstream_records',
        'roots','parent_routes','retained_metadata_roots')}
    source_args['source_records']={role:args['source_records'][role] for role in ROLES}
    source_args['metadata_roots']=source_args.pop('retained_metadata_roots')
    _join(**source_args)
    return args, context, (accepted['intent_id'], owner, birth), (prior['intent_id'], prior_owner, prior_birth)


def _produce_current_sam_phase(args, context, base, monkeypatch):
    """Replace only the new b fixture phase with real queue/stage output."""
    from blueprint_pipeline import task_evaluation_sam31_preparation_execution as execution
    from blueprint_pipeline import task_evaluation_sam31_preparation_cpu_stages as cpu
    from blueprint_pipeline.task_evaluation_sam31_phase_queue import enqueue_sam31_phase
    from blueprint_pipeline.task_evaluation_scene_configuration_sam31_plan import PROFILE_ENV
    from tests.test_scene_source_family_website import ref

    rows = args['source_records']
    job_pair = next(row for row in rows['sam_jobs']
                    if json.loads(row[1])['parent_preparation_id'] == 'current-parent')
    job = json.loads(job_pair[1])
    result_pair = next(row for row in rows['sam_results']
                       if json.loads(row[1])['child_id'] == job['child_id'])
    prior_result = json.loads(result_pair[1])
    assert set(prior_result['artifacts']) == {'phase_artifact'}
    output = Path(args['roots']['sam_execution_root']) / job['parent_request_digest'][7:] / job['child_id']
    prior_receipt = next(row for row in rows['sam_execution_receipts']
                         if Path(row[0]).parent == output)
    prior_artifact = next(row for row in rows['opaque_evidence']
                          if row[0] == prior_result['artifacts']['phase_artifact']['path'])
    assert prior_artifact[1] == b'tiny-evidence'
    assert {path.name for path in output.iterdir()} == {
        Path(prior_receipt[0]).name, Path(prior_artifact[0]).name}
    for pair in (job_pair, result_pair, prior_receipt, prior_artifact):
        Path(pair[0]).unlink()
    output.rmdir()
    assert list(output.parent.iterdir()) == []
    output.parent.rmdir()
    for role, pair in (('sam_jobs', job_pair), ('sam_results', result_pair),
                       ('sam_execution_receipts', prior_receipt), ('opaque_evidence', prior_artifact)):
        rows[role].remove(pair)
    intake = enqueue_sam31_phase(queue_root=args['roots']['sam_queue_root'],
        parent_preparation_id=job['parent_preparation_id'],
        parent_request_digest=job['parent_request_digest'],
        expected_source_commit=job['expected_source_commit'], plan_ref=job['plan_ref'],
        phase=job['phase'], inputs=job['inputs'])
    assert intake['status'] == 'queued' and intake['child_id'] == job['child_id']
    plan = next(json.loads(raw) for path, raw in rows['sam_plans'] if path == job['plan_ref']['path'])
    profile = next(row for row in rows['sam_profiles']
                   if ref(row)['sha256'] == plan['server_profile_sha256'])
    monkeypatch.setenv(PROFILE_ENV, profile[0])
    monkeypatch.setattr(execution, '_verified_checkout_head', lambda: job['expected_source_commit'])
    def tiny_cpu(context):
        destination = Path(context['output_root']) / 'stage_result.json'
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps({'child_id': context['child_id'], 'phase': context['phase']},
                                          sort_keys=True))
        return {'status': 'completed', 'artifacts': {'stage_result': _raw(destination)}}
    monkeypatch.setattr(cpu, 'execute_cpu_stage', tiny_cpu)
    run = execution.process_sam31_phase_queue(queue_root=args['roots']['sam_queue_root'],
        parent_queue_root=args['roots']['preparation_queue_root'],
        preparation_input_root=args['roots']['preparation_input_root'],
        execution_root=args['roots']['sam_execution_root'], approved_roots=(base,))
    assert run['results'][0]['status'] == 'completed', json.loads(Path(intake['result_path']).read_bytes()).get('blocker')
    assert run['results'] == [{'child_id': job['child_id'], 'status': 'completed',
                              'result_path': intake['result_path']}]
    rows['sam_jobs'].append((str(Path(args['roots']['sam_queue_root']) / 'completed' /
                             (job['child_id'] + '.json')),
                            Path(args['roots']['sam_queue_root'], 'completed', job['child_id'] + '.json').read_bytes()))
    rows['sam_results'].append((intake['result_path'], Path(intake['result_path']).read_bytes()))
    actual_receipt = output / 'phase_execution_receipt.v1.json'
    rows['sam_execution_receipts'].append((str(actual_receipt), actual_receipt.read_bytes()))
    stage_ref = json.loads(Path(intake['result_path']).read_bytes())['artifacts']['stage_result']
    rows['opaque_evidence'].append((stage_ref['path'], Path(stage_ref['path']).read_bytes()))
    assert Path(rows['sam_jobs'][-1][0]).read_bytes() == rows['sam_jobs'][-1][1]


def _produce_current_sam_progress(args, context, base):
    """Write the current SAM final through its exact selected parent queue."""
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest
    from blueprint_pipeline.task_evaluation_sam31_preparation_queue import advance_sam31_for_preparation
    from tests.test_scene_source_family_website import ref

    rows = args['source_records']
    adoption_pair = rows['sam_adoptions'][0]
    adoption = json.loads(adoption_pair[1])
    artifacts = {}
    for _, raw in rows['sam_results']:
        artifacts.update(json.loads(raw)['artifacts'])
    artifacts.update({name: item['successor'] for name, item in adoption['administrative_rebindings'].items()})
    evidence = {name: artifacts[name] for name in ('calibrated_mask_set', 'segment_cutout_set',
        'track_selection_review', 'selection_inputs', 'standard_splat_conversion')}
    current_job = next(json.loads(raw) for _, raw in rows['sam_jobs']
        if json.loads(raw)['parent_preparation_id'] == 'current-parent')
    parent = next(json.loads(raw) for _, raw in [*rows['sam_parent_envelopes'],
                                                  *args['seed_records']['preparation_envelopes']]
        if json.loads(raw)['request']['preparation_id'] == 'current-parent')
    assert current_job['parent_request_digest'] == parent['request_digest']
    assert current_job['plan_digest'] == ref(next(row for row in rows['sam_plans']
        if row[0] == current_job['plan_ref']['path']))['sha256']
    final = {'schema_version': 'task_evaluation_sam31_preparation_result.v1',
        'status': 'exact_mask_inputs_ready', 'source_commit': parent['request']['expected_production_commit'],
        'plan_digest': current_job['plan_digest'], 'evidence': evidence,
        'stage_result_receipts': [],
        'completed_prefix_adoption': {'receipt': ref(adoption_pair),
            'original_execution_commit': adoption['original_execution_commit'],
            'through_phase': adoption['through_phase'],
            'original_phase_result_receipts': [phase['result'] for phase in adoption['phase_records']]}}
    final['result_digest'] = canonical_digest(final, digest_field='result_digest')
    advancement = {'status': 'ready', 'sam31_preparation_result': final,
        'sam31_exact_mask_inputs': evidence,
        'evidence_refs': list(evidence.values())}
    queue = Path(args['roots']['preparation_queue_root'])
    result = advance_sam31_for_preparation(queue_root=queue,
        envelope_context={'request': parent['request'], 'request_digest': parent['request_digest'],
                          'stage_one_configuration': {}}, approved_roots=(base,),
        advancer=lambda _: advancement)
    assert result == advancement
    path = next((queue / 'source-progress' /
        ('current-parent-' + parent['request_digest'][7:])).glob('000001-*.json'))
    row = (str(path), path.read_bytes())
    rows['source_progress'] = [row]
    context['retained_metadata_files'].append({'role': 'source_progress', 'path': str(path)})


def _snapshot(roots):
    result = {}
    for root in roots:
        for path in [root, *root.rglob('*')]:
            relative = str(path.relative_to(root))
            info = path.stat()
            result[str(root), relative] = (stat.S_IMODE(info.st_mode),
                path.read_bytes() if path.is_file() else None,
                (info.st_dev, info.st_ino) if path.is_file() else None)
    return result


def _installed_cohort():
    """Declare the actual finite consumers, including required special lifetimes.

    A declaration is never a clearance Boolean: the engine must independently
    prove these installed identities and their current lifetime closure.
    """
    source = Path(__file__).resolve().parents[1] / 'src' / 'blueprint_pipeline'
    modules = ('task_evaluation_scene_intake', 'task_evaluation_scene_progression',
        'task_evaluation_launch_preparation_worker', 'task_evaluation_launch_activation_worker',
        'task_evaluation_episode_compilation_worker', 'task_evaluation_sam31_prefix_adoption',
        'task_evaluation_sam31_preparation_execution', 'task_evaluation_scene_configuration_sam31_preparation_driver',
        'task_evaluation_launch_dispatcher', 'task_evaluation_policy_canary_dispatcher',
        'task_evaluation_scene_configuration_submission_publication', 'website_scene_dispatch',
        'website_native_submission', 'artifixer_completed_training_reuse', 'control_plane_storage_pins')
    special = {'pubsub_handoff_listener': ('stage_handoff_capture', 'process_handoff_payload', 'pull_and_process'),
        'task_evaluation_terminal_scene_attempt_settlement': ('retained_hold', 'budget_retained_hold',
            'settle_retired_attempt_rows', 'sweep_retired_attempts'),
        'task_evaluation_scene_owner_authority': ('reopen_scene_intent', 'validate_task_scene_owner'),
        'live_pipeline_result_artifact_resolution': ('resolve_live_pipeline_result_artifact',),
        'live_pipeline_result_artifact_response': ('result_artifact_response', 'ResultArtifactFileResponse.__call__')}
    rows = []
    for module in (*modules, *special):
        path = source / (module + '.py')
        raw = path.read_bytes()
        tree = ast.parse(raw)
        functions = [node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and any(isinstance(item, ast.Call) and isinstance(item.func, ast.Name)
                    and item.func.id == 'scene_participant' for item in node.decorator_list)]
        functions += list(special.get(module, ()))
        for name in sorted(set(functions)):
            rows.append({'entrypoint': 'blueprint_pipeline.' + module + ':' + name,
                'installed_source_sha': 'sha256:' + hashlib.sha256(raw).hexdigest(),
                'lifetime_contract_version': 'scene_retirement_lifetime.v1'})
    assert rows and len({row['entrypoint'] for row in rows}) == len(rows)
    return rows


def _consented_inventories(members, cache_objects=()):
    """Owner issuance hashes real bytes; the metadata-only planner does not."""
    from blueprint_pipeline.task_evaluation_scene_retirement_preservation import ActionAllowance, _inventory_members, _payload
    from blueprint_pipeline.task_evaluation_scene_retirement_mutation import inventory_digest
    allowance = ActionAllowance(expires_at=999, now=lambda: 200, monotonic=time.monotonic,
        local_bytes=1024*1024, archive_bytes=2*1024*1024, remote_bytes=4*1024*1024, elapsed_seconds=60)
    preserved = _inventory_members(members, allowance, cache_aliases=[
        {key: row[key] for key in ('canonical_path', 'digest', 'size_bytes')} for row in cache_objects])
    files = preserved['files']
    for row in files:
        digest = hashlib.sha256()
        for chunk in _payload(members[row['member_index']] / row['relative_path'], row, allowance):
            digest.update(chunk)
        row['sha256'] = 'sha256:' + digest.hexdigest()
    return {str(member): inventory_digest(preserved, index) for index, member in enumerate(members)}


class MemoryArchiveTransport:
    """Actual streamed bytes, with source-presence checks during fresh readback."""

    def __init__(self, members, published_objects):
        self.members = members
        self.published_objects = published_objects
        self.objects = {}
        self.events = []
        self.retirement = True

    def put_archive(self, key, iterable_chunks):
        assert self.retirement
        assert all(path.exists() for path in self.members), 'local member removed before ALL preservation'
        chunks = []
        for chunk in iterable_chunks:
            assert isinstance(chunk, bytes) and len(chunk) <= 1024 * 1024
            chunks.append(chunk)
        data = b''.join(chunks)
        uri = 's3://fixture-private/' + key
        assert uri not in self.objects, 'archive overwritten instead of preserving history'
        self.objects[uri] = data
        self.events.append(('upload', uri))
        return {'uri': uri, 'sha256': 'sha256:' + hashlib.sha256(data).hexdigest(), 'size_bytes': len(data)}

    def read_published_object_charged(self, uri, allowance, *, expected_size_bytes):
        data = self.published_objects[uri]
        assert len(data) == expected_size_bytes
        self.events.append(('published-readback', uri))
        for start in range(0, len(data), 37):
            allowance.tick()
            chunk = data[start:start + 37]
            allowance.charge('remote_bytes', len(chunk))
            yield chunk
            allowance.tick()

    def read_archive(self, uri):
        if self.retirement:
            assert all(path.exists() for path in self.members), 'fresh readback followed local removal'
        self.events.append(('readback', uri))
        data = self.objects[uri]
        for start in range(0, len(data), 37):
            yield data[start:start + 37]


@pytest.fixture
def short_scene_directory():
    # This all-family fixture repeats every absolute path through many bounded
    # lineage projections. Use a normal short owned directory so test runner
    # temp-base depth does not consume the production 30s planner allowance.
    with tempfile.TemporaryDirectory(prefix='scene-', dir=Path(tempfile.gettempdir()).resolve()) as path:
        yield Path(path)


@pytest.mark.slow
def test_current_sam_worker_and_parent_link_use_real_selected_owner(short_scene_directory, monkeypatch):
    from tests.test_scene_retirement_real_participants import access_fixture

    base = short_scene_directory.resolve()
    _, policy, placeholder = access_fixture(base, monkeypatch)
    placeholder.rmdir()
    journals = Path(policy['journal_store'])
    journals.mkdir(mode=0o700)
    (journals / 'retired').mkdir(mode=0o700)
    monkeypatch.delenv('BLUEPRINT_SCENE_RETIREMENT_POLICY_FILE')
    args, _, main_owner, original_owner = _authentic_connected_graph(base, monkeypatch, policy)
    links = [(path, json.loads(raw)) for path, raw in args['seed_records']['preparation_links']]
    current = [(path, link) for path, link in links if link['preparation_id'] == 'current-parent']
    assert len(current) == 1
    link_path, link = current[0]
    assert Path(link_path).read_bytes() == next(raw for path, raw in args['seed_records']['preparation_links']
                                                if path == link_path)
    assert link['intent_id'] == main_owner[0] != original_owner[0]
    parent = next(json.loads(raw) for path, raw in args['seed_records']['preparation_envelopes']
                  if path.endswith('/' + link['result_filename']))
    result = next(json.loads(raw) for path, raw in args['seed_records']['preparation_results']
                  if path.endswith('/' + link['result_filename']))
    assert parent['request_digest'] == link['request_digest']
    assert result['status'] == 'queued_for_production_scene_configuration'
    job = next(json.loads(raw) for _, raw in args['source_records']['sam_jobs']
               if json.loads(raw)['parent_preparation_id'] == 'current-parent')
    result = next(json.loads(raw) for _, raw in args['source_records']['sam_results']
                  if json.loads(raw)['child_id'] == job['child_id'])
    assert job['parent_request_digest'] == link['request_digest']
    assert result['status'] == 'completed' and set(result['artifacts']) == {'stage_result'}
    stage_path = Path(result['artifacts']['stage_result']['path'])
    assert json.loads(stage_path.read_bytes()) == {'child_id': job['child_id'],
                                                    'phase': 'source_selections'}
    receipt = next(path for path, _ in args['source_records']['sam_execution_receipts']
                   if job['child_id'] in Path(path).parts)
    assert Path(receipt).is_file()


@pytest.mark.slow
def test_source_owned_configuration_is_enrolled_before_activation(short_scene_directory, monkeypatch):
    from tests.test_scene_retirement_real_participants import access_fixture

    base = short_scene_directory.resolve()
    _, policy, placeholder = access_fixture(base, monkeypatch)
    placeholder.rmdir()
    source = _source_owned_configuration(base, monkeypatch, policy)
    assert source['request']['scene_intent_digest'] == json.loads(
        Path(source['owner']['path']).read_bytes())['intent_digest']
    assert json.loads(source['result'][1])['status'] == 'queued_for_production_scene_configuration'
    assert source['request']['scene']['identity']['id'].startswith('completed-scene-')


@pytest.mark.slow
def test_terminal_scene_retires_every_folder_it_wrote(short_scene_directory, monkeypatch):
    # RED remains an actual feature failure rather than collection loss or xfail.
    # Authenticate the complete planner fixture before importing the engine.
    from blueprint_pipeline import task_evaluation_scene_retirement_access as access
    from blueprint_pipeline.task_evaluation_scene_retirement_authority import cohort_digest
    from blueprint_pipeline.task_evaluation_scene_lifecycle_plan import build_scene_lifecycle_plan, FAMILIES
    from tests.scene_lifecycle_fixture_support import stable_shared_ancestors
    from tests.test_scene_retirement_real_participants import access_fixture

    base = short_scene_directory.resolve()
    _, policy, placeholder = access_fixture(base, monkeypatch)
    placeholder.rmdir()
    journals = Path(policy['journal_store'])
    journals.mkdir(mode=0o700)
    (journals / 'retired').mkdir(mode=0o700)
    # Issue intake before enrollment: stage's real publisher remains an actual
    # participant; disabled root policy grants it no fictional cleanup authority.
    monkeypatch.delenv('BLUEPRINT_SCENE_RETIREMENT_POLICY_FILE')
    args, context, main_owner, original_owner = _authentic_connected_graph(base, monkeypatch, policy)
    current_job = next(json.loads(raw) for _, raw in args['source_records']['sam_jobs']
                       if json.loads(raw)['parent_preparation_id'] == 'current-parent')
    current_result = next(json.loads(raw) for _, raw in args['source_records']['sam_results']
                          if json.loads(raw)['child_id'] == current_job['child_id'])
    assert set(current_result['artifacts']) == {'stage_result'}
    stage_ref = current_result['artifacts']['stage_result']
    assert json.loads(Path(stage_ref['path']).read_bytes()) == {
        'child_id': current_job['child_id'], 'phase': 'source_selections'}
    phase_root = Path(args['roots']['sam_execution_root']) / current_job['parent_request_digest'][7:] / current_job['child_id']
    phase_generation_path = Path(policy['generation_store']) / (hashlib.sha256(str(phase_root).encode()).hexdigest() + '.json')
    assert json.loads(phase_generation_path.read_bytes())['canonical_path'] == str(phase_root)
    from blueprint_pipeline.control_plane_storage_gc import DEFAULT_MINIMUM_AGE_SECONDS
    cache_root = Path(context['roots']['preparation_input_root']) / 'content-addressed' / 'sha256'
    active_cache = sorted(path for path in cache_root.iterdir() if len(path.name) == 64 and path.is_file())
    assert active_cache
    for path in active_cache:
        os.utime(path, (200 - DEFAULT_MINIMUM_AGE_SECONDS - 1,
                        200 - DEFAULT_MINIMUM_AGE_SECONDS - 1))
    stable_shared_ancestors(monkeypatch, base)
    initial = build_scene_lifecycle_plan(intent_id=args['intent_id'], context=context, observed_at_epoch=200)
    assert 'historical_lineage' in initial, (initial.get('blockers'),initial.get('reason'),initial.get('status'))
    assert {row['family'] for row in initial['family_obligations'] if row['member_count']} == set(FAMILIES), initial
    selected = [row for row in initial['measured_members'] if row.get('kinds')]
    shared_rows = [row for row in selected if 'prepared_cache_object' in row.get('kinds',[])]
    shared_content = [Path(row['path']) for row in shared_rows]
    assert len(shared_content) >= 2, [(row['path'],row['status'],row['keeps']) for row in shared_rows]
    absent_shared = [row for row in shared_rows if not Path(row['path']).exists()]
    assert not absent_shared, [(row['path'], row['status'], row['keeps']) for row in shared_rows]
    missing_cache = sorted(path.name for path in set(active_cache) - set(shared_content))
    assert not missing_cache, missing_cache
    # A birth owns directories. Native compilation's exact file obligations are
    # retained under their actual owning directory. The original planner
    # remains KEEP; action must independently prove each policy-born CAS alias.
    paths = {Path(row['path']).parent if Path(row['path']).is_file() else Path(row['path'])
             for row in selected if Path(row['path']) not in shared_content}
    members = sorted((path for path in paths if not any(other != path and path.is_relative_to(other)
                     for other in paths)), key=str)
    for member in members:
        member.mkdir(parents=True, exist_ok=True)
    policy_roots = {str(path.parent): {'root': str(path.parent),
        'storage_class': 'cache' if path.parent == Path(context['roots']['preparation_input_root']) else 'host',
        'device': path.parent.stat().st_dev} for path in members}
    authority_root = base / 'preparation-authority'
    policy_roots[str(authority_root)] = {'root': str(authority_root), 'storage_class': 'host',
                                       'device': authority_root.stat().st_dev}
    policy['roots'] = list(policy_roots.values())
    policy['principals'] = [{'principal_id': 'fixture-owner', 'actions': ['retire', 'restore'],
                            'owner_intent_ids': [main_owner[0], original_owner[0]],
                            'private_archive_classes': ['host']}]
    policy['private_archive_allowed_classes'] = ['host']
    policy['consumer_cohort'] = _installed_cohort()
    policy['reference_context'] = context
    policy['limits'] = {'logical_payload_bytes': 1024 * 1024, 'archive_bytes': 2 * 1024 * 1024,
                        'remote_bytes': 4 * 1024 * 1024, 'elapsed_seconds': 60}
    policy_path = base / 'policy.json'
    _sealed_file(policy_path, policy, 'policy_digest', mode=0o644)
    monkeypatch.setattr(access, '_INSTALLED_POLICY', policy_path)
    monkeypatch.setenv('BLUEPRINT_SCENE_RETIREMENT_POLICY_FILE', str(policy_path))
    generations = {}
    original_paths = {Path(row['path']) for row in selected if 'sam_original_execution_dependency' in row['kinds']}
    # Recreate these tiny fixture directories through the actual authenticated
    # producer birth hook. No pre-existing legacy folder is silently adopted.
    for member in members:
        generation_path = Path(policy['generation_store']) / (hashlib.sha256(str(member).encode()).hexdigest() + '.json')
        if generation_path.is_file():
            state = json.loads(generation_path.read_bytes())
            assert state['schema_version'] == 'scene_member_generation.v1' and state['state'] == 'active'
            assert state['canonical_path'] == str(member) and state['owner_raw_ref'] == main_owner[1]
            generations[str(member)] = state
            continue
        parked = member.with_name(member.name + '.fixture-before-birth')
        member.rename(parked)
        selected_owner = original_owner if any(path.is_relative_to(member) for path in original_paths) else main_owner
        state = access.birth_scene_member(member, owner_intent_id=selected_owner[0],
            owner_raw_ref=selected_owner[1], birth_request_raw_ref=selected_owner[2], now=200)
        assert state and state['state'] == 'active'
        generations[str(member)] = state
        for child in list(parked.iterdir()):
            child.rename(member / child.name)
        parked.rmdir()
    original = _snapshot([*members, *shared_content])
    cache_objects = []
    for path in active_cache:
        generation_path = Path(policy['generation_store']) / (hashlib.sha256(str(path).encode()).hexdigest() + '.json')
        generation = json.loads(generation_path.read_bytes())
        assert generation['schema_version'] == 'scene_content_generation.v1'
        assert generation['state'] == 'active' and generation['canonical_path'] == str(path)
        cache_objects.append({'canonical_path': str(path), 'digest': generation['digest'],
            'size_bytes': generation['size_bytes'], 'generation_id': generation['generation_id'],
            'generation_raw_ref': _raw(generation_path),
            'source_raw_ref': generation['source_publication_raw_ref']})
    inventories = _consented_inventories(members, cache_objects)
    plan = build_scene_lifecycle_plan(intent_id=args['intent_id'], context=context, observed_at_epoch=200)
    assert 'finished_observation' in plan, plan
    assert plan['finished_observation']['status'] == 'completed'
    assert {row['family'] for row in plan['family_obligations'] if row['member_count']} == set(FAMILIES)
    assert plan['action'] == 'KEEP' and plan['cleanup_authorized'] is False
    assert all(row['action'] == 'KEEP' for row in plan['family_obligations'])
    # The observer reports deferred original-record obligations. Retirement
    # must prove their exact selected bytes and preservation at action time.
    assert all(row['complete'] is True for row in plan['reference_observation']['child_scopes']), [
        (row['child'],row['complete'],row.get('reason'))
        for row in plan['reference_observation']['child_scopes']]
    assert set(plan['reference_observation']['blockers']) <= {
        'deferred_semantic_object','deferred_downstream_document','deferred_parent_reference_proof'}, [
            (fact.get('reason'),fact.get('contract_path'),fact.get('source',{}).get('row_path'))
            for row in plan['reference_observation']['protections'] if row.get('kind')=='missing_edge_obligations'
            for fact in [row.get('observation',{})] if fact.get('reason') in {'identity_envelope_unresolved','result_selector_missing'}]
    plan_path = base / 'retained-plan.json'
    plan_path.write_text(json.dumps(plan, sort_keys=True))
    plan_path.chmod(0o600)
    consent_path = base / 'authority' / 'retire-consent.json'
    consent_path.parent.mkdir(mode=0o700)
    consent = {'schema_version': 'scene_retirement_consent.v1', 'consent_id': '1' * 32,
        'principal_id': 'fixture-owner', 'intent_id': args['intent_id'], 'intent_raw_ref': main_owner[1],
        'plan_raw_ref': _raw(plan_path), 'retired_journal_raw_ref': None,
        'policy_sha256': _raw(policy_path)['sha256'],
        'cohort_sha256': cohort_digest(policy['consumer_cohort']),
        'action': 'retire', 'created_at': 199, 'expires_at': 999,
        'members': [{'canonical_path': str(path), 'class': 'host',
            'owner_intent_id': generations[str(path)]['owner_intent_id'],
            'owner_raw_ref': generations[str(path)]['owner_raw_ref'],
            'generation_id': generations[str(path)]['generation_id'],
            'dev': path.stat().st_dev, 'ino': path.stat().st_ino, 'mode': path.stat().st_mode,
            'inventory_sha256': inventories[str(path)]}
            for path in members], 'private_archive_classes': ['host'],
        'cache_objects': cache_objects}
    _sealed_file(consent_path, consent, 'consent_digest')
    published = {}
    for _, raw in args['source_records']['submission_publications']:
        receipt = json.loads(raw)
        for row in receipt['published_objects']:
            assert not row['relative_path'].startswith('source/')
            matches = [data for path, data in _record_pairs(args) if path.endswith('/' + row['relative_path'])
                       and 'sha256:' + hashlib.sha256(data).hexdigest() == row['digest']]
            assert matches and all(data == matches[0] for data in matches), row
            published[row['uri']] = bytes(matches[0])
    transport = MemoryArchiveTransport(members, published)
    from blueprint_pipeline.task_evaluation_scene_retirement import retire_scene, restore_scene
    assert any(row.get('pre_handoff_binding_verified') for row in
        plan['historical_lineage']['preparation_handoff_observations']), (
        plan['historical_lineage']['preparation_handoff_observations'])
    retired = retire_scene(plan_path, consent_path, transport=transport, now=lambda: 200, monotonic=time.monotonic)
    assert retired['status'] == 'retired', retired
    shared_keeps = [{'canonical_path':row['path'],'action':'KEEP',
                     'observation_status':row['status'],'reasons':list(row['keeps'])}
                    for row in absent_shared]
    assert retired['unselected_shared_content_keeps'] == shared_keeps
    assert set(row['canonical_path'] for row in retired['members']) == set(map(str, members))
    assert all(not path.exists() for path in members)
    assert all(not path.exists() for path in shared_content)
    journal_ref = retired['retired_journal_raw_ref']
    assert _raw(journal_ref['path']) == journal_ref
    immutable_snapshot = Path(journal_ref['path']).read_bytes()
    assert json.loads(immutable_snapshot)['unselected_shared_content_keeps'] == shared_keeps
    intent_receipt = Path(retired['intent_receipt_path'])
    assert intent_receipt.is_file()
    published_receipt = json.loads(intent_receipt.read_bytes())
    assert published_receipt['status'] == 'retired'
    assert published_receipt['unselected_shared_content_keeps'] == shared_keeps
    assert transport.objects and all(('readback', uri) in transport.events for uri in transport.objects)
    restore_consent = dict(consent, consent_id='2' * 32, action='restore', plan_raw_ref=None,
                           retired_journal_raw_ref=journal_ref)
    restore_path = consent_path.with_name('restore-consent.json')
    _sealed_file(restore_path, restore_consent, 'consent_digest')
    transport.retirement = False
    restored = restore_scene(Path(journal_ref['path']), restore_path, transport=transport,
                             now=lambda: 201, monotonic=time.monotonic)
    assert restored['status'] == 'restored', restored
    assert restored['unselected_shared_content_keeps'] == shared_keeps
    assert json.loads(intent_receipt.read_bytes())['unselected_shared_content_keeps'] == shared_keeps
    assert all(path.exists() for path in shared_content)
    roundtrip = _snapshot([*members, *shared_content])
    assert {key: value[:2] for key, value in roundtrip.items()} == {key: value[:2] for key, value in original.items()}
    groups = {}
    for key, (_, data, inode) in original.items():
        if data is not None:
            groups.setdefault(inode, []).append(key)
    for names in groups.values():
        assert len({roundtrip[name][2] for name in names}) == 1, 'approved hardlink group was not restored'
    assert Path(journal_ref['path']).read_bytes() == immutable_snapshot
    # A later writer is never overwritten by replay of an old restore receipt.
    foreign = members[0] / 'new-capture.bin'
    foreign.write_bytes(b'new-owner-payload')
    before = _snapshot(members)
    try:
        repeated = restore_scene(Path(journal_ref['path']), restore_path, transport=transport,
                                 now=lambda: 202, monotonic=time.monotonic)
    except (ValueError, RuntimeError):
        pass
    else:
        assert repeated['status'] not in {'restored', 'retired'}, repeated
    assert _snapshot(members) == before
