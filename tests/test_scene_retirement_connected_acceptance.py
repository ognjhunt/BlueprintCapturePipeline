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
from urllib.parse import urlsplit

import pytest


# Give the real activation worker a live intake owner while allowing the
# subsequent local retirement observation to see a full day of idle age.
SCENE_SOURCE_EPOCH = int(time.time())
SCENE_RETIREMENT_EPOCH = SCENE_SOURCE_EPOCH + 2 * 24 * 60 * 60


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
        # Intake owns these original bytes. Rewriting a protected raw selector
        # through another fixture record's digest alias can detach it from the
        # very file whose identity it is meant to prove.
        if (set(value) == {'path', 'sha256', 'size_bytes'}
                and value['path'] in immutable_paths):
            return copy.deepcopy(value)
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
    previous_plan = current['source_records']['sam_plans'][0]
    plan = json.loads(previous_plan[1])
    for key in ('scene_identity', 'task_identity', 'publisher_scene_id'):
        plan[key] = copy.deepcopy(owned_current_task[key])
    plan['plan_digest'] = canonical_digest(plan, digest_field='plan_digest')
    current_task = pair('/retained/host-inputs/current-scene-task.json', owned_current_task)
    # Current and independently owned original tasks are different raw records.
    current = _rebase_complete_graph(current, '/retained', {previous_task[0]: current_task[0],
        current['source_records']['sam_profiles'][0][0]: '/retained/metadata/current-scene-profile.json',
        current['source_records']['sam_plans'][0][0]: '/retained/metadata/current-scene-plan.json',
        's3://test/plan.json': 's3://test/current-scene-plan.json'},
        requests={canonical_digest(json.loads(previous_task[1])): owned_current_task,
                  canonical_digest(json.loads(previous_plan[1])): plan})
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


def _selected_worker_preparations(args, base, policy, monkeypatch, owner, birth, current_birth, source):
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
        fetcher, production_request_with_fetchable_bytes)
    from blueprint_pipeline.task_evaluation_scene_configuration_render_inputs import (
        materialize_scene_configuration_render_inputs)

    queue_root = Path(args['roots']['preparation_queue_root'])
    queue_root.mkdir(parents=True, exist_ok=True)
    input_root = Path(args['roots']['preparation_input_root'])
    input_root.mkdir(parents=True, exist_ok=True)
    sam_execution_root = Path(args['roots']['sam_execution_root'])
    sam_execution_root.mkdir(parents=True, exist_ok=True)
    authority_root = base / 'preparation-authority'
    authority_root.mkdir(mode=0o700)
    # The source configuration producer already enrolled its factory output.
    # Its preparation storage authority still names that exact factory file.
    policy['roots'] = [{'root': str(input_root), 'storage_class': 'cache', 'device': input_root.stat().st_dev},
                       {'root': str(source['progression_output']), 'storage_class': 'host',
                        'device': source['progression_output'].stat().st_dev},
                       {'root': str(authority_root), 'storage_class': 'host', 'device': authority_root.stat().st_dev},
                       {'root': str(sam_execution_root), 'storage_class': 'host', 'device': sam_execution_root.stat().st_dev}]
    _sealed_file(base / 'policy.json', policy, 'policy_digest', mode=0o644)
    monkeypatch.setenv('BLUEPRINT_SCENE_RETIREMENT_POLICY_FILE', str(base / 'policy.json'))
    monkeypatch.setenv('BLUEPRINT_TASK_EVALUATION_SCENE_INTAKE_ROOT', str(source['intake']))
    monkeypatch.setenv('BLUEPRINT_TASK_EVALUATION_SCENE_INTAKE_CLIENT_IDS', 'blueprint-webapp,webapp')
    monkeypatch.setattr(cache.time, 'time', lambda: source['issued_at'] + 2)
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
        request['scene']['identity'] = copy.deepcopy(source['request']['scene']['identity'])
        request['task']['identity'] = copy.deepcopy(source['request']['task']['identity'])
        # These historical branches are test-owned completed meshes. Feed the
        # actual CPU materializer a real, tiny mesh and its sealed normalization.
        mesh = b'o fixture_surface\nv 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n'
        mesh_ref = request['scene']['geometry']['collision']
        mesh_ref.update(digest='sha256:' + hashlib.sha256(mesh).hexdigest(), size_bytes=len(mesh))
        payloads[mesh_ref['uri']] = mesh
        request['scene']['appearance']['kind'] = 'other_observed'
        normalization = {'schema_version': 'fixture_completed_mesh_normalization.v1',
                         'output': {'sha256': mesh_ref['digest']}, 'normalization_digest': ''}
        normalization['normalization_digest'] = canonical_digest(
            normalization, digest_field='normalization_digest')
        normalization_bytes = json.dumps(normalization, sort_keys=True).encode()
        normalization_ref = request['scene']['geometry']['validation']
        normalization_ref.update(digest='sha256:' + hashlib.sha256(normalization_bytes).hexdigest(),
                                 size_bytes=len(normalization_bytes))
        payloads[normalization_ref['uri']] = normalization_bytes
        if identifier == 'prep-1':
            definition = {'schema_version': 'task_evaluation_rigid_relocation_template.v1',
                'task_identity': request['task']['identity'],
                'object_identity': request['task']['subject']['identity'],
                'strategy': request['task']['strategy'],
                'start_center_xyz_m': [0.0, 0.0, 0.0],
                'target_center_xyz_m': [0.0, 0.1, 0.0]}
            definition_ref = request['task']['definition']
            definition_bytes = json.dumps(definition, sort_keys=True).encode()
            definition_ref.update(uri='s3://blueprint-production-inputs/prep-1-task-definition.json',
                                  digest='sha256:' + hashlib.sha256(definition_bytes).hexdigest(),
                                  size_bytes=len(definition_bytes))
            payloads[definition_ref['uri']] = definition_bytes
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
        recipe['stage_sequence'][0]['adapter']['id'] = 'provided_mesh_appearance_excision'
        recipe['stage_sequence'][0]['execution_class'] = 'no_spend'
        stage.update(schema_version='task_evaluation_provided_mesh_appearance_excision.v1',
                     source_origin='owner_provided_completed_asset',
                     source_bytes_unchanged_required=True,
                     unobserved_surfaces_recovered=False, physical_truth_claimed=False,
                     generated_appearance=False, collision_source_digest=mesh_ref['digest'],
                     exact_target_prim='/Root/fixture_surface',
                     sam31_review_kind='ai', sam31_preparation_plan=copy.deepcopy(mount))
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
            adapter_materializer=forbidden_adapter,
            scene_render_input_materializer=materialize_scene_configuration_render_inputs,
            construction_queue_root=source['base']/'construction')
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
        input_digest=request_digest,provider='vast',
        maximum_spend_usd=request['spend']['hard_cap_usd'],now=SCENE_SOURCE_EPOCH + 2)
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
    queue = ensure_launch_preparation_queue_root(source / 'queue')
    input_root = source / 'inputs'
    input_root.mkdir()
    progression_output = source / 'factories'
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
                   'factory_output_root': str(progression_output),
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


def _website_owned_configuration(base, monkeypatch, policy):
    """Produce one capture, registration, intent and preparation for the same owner."""
    import copy
    import pwd
    from urllib.parse import urlsplit

    import numpy as np
    import trimesh

    from blueprint_pipeline import public_scene_host_input_intake
    from blueprint_pipeline import task_evaluation_launch_preparation_worker as worker
    from blueprint_pipeline import task_evaluation_scene_configuration_submission_publication as publication
    from blueprint_pipeline import task_evaluation_scene_progression as engine
    from blueprint_pipeline.common import write_json
    from blueprint_pipeline.decision_evidence_contracts import cross_runtime_canonical_json
    from blueprint_pipeline.local_reconstruction_adapters import _sha256_file
    from blueprint_pipeline.task_evaluation_launch_preparation_queue import ensure_launch_preparation_queue_root
    from blueprint_pipeline.task_evaluation_public_scene_attempt_factory import RELEASE_SCHEMA, record
    from blueprint_pipeline.task_evaluation_scene_intake import stage_scene_intent
    from blueprint_pipeline.website_scene_handoff import prepare_website_scene_handoff
    from tests.test_task_evaluation_scene_configuration_submission import SHA, production_fixture
    from tests.test_task_evaluation_scene_configuration_submission_publication import Store
    from tests.test_website_native_appearance import inputs
    from tests.test_website_task_preparation import _task_context

    source = base / 'website-source-produced'
    capture = source / 'pubsub' / 'bucket' / 'scenes' / 'site-req1' / 'captures' / 'walkthrough-req1'
    pipeline = capture / 'pipeline'
    pipeline.mkdir(parents=True)
    args, _, _ = inputs(pipeline)
    args['task_context'] = _task_context(confirmed_at=SCENE_SOURCE_EPOCH - 100)
    args['spend'] = copy.deepcopy(args['spend'])
    args['spend']['expires_at_epoch'] = SCENE_SOURCE_EPOCH + 6 * 24 * 60 * 60
    args['spend']['max_paid_attempts'] = 4
    args['spend']['consent']['accepted_at_epoch'] = SCENE_SOURCE_EPOCH - 1
    collider = Path(args['base_scene']['collision_mesh_path'])
    mesh = trimesh.load(collider, force='mesh')
    mesh.apply_transform(np.diag([1.0, -1.0, -1.0, 1.0]))
    mesh.export(collider)
    assets = pipeline / 'assets.json'
    write_json(assets, {'world_id': 'world-1', 'downloads': [
        {'kind': kind, 'local_path': args['base_scene'][key + '_path'],
         'sha256': _sha256_file(Path(args['base_scene'][key + '_path']))[7:]}
        for kind, key in (('splat_ply', 'splat'), ('collider_mesh_glb', 'collision_mesh'))]})
    removal = pipeline / 'removal.json'
    write_json(removal, args['removal_manifest'])
    binding_root = source / 'bindings'
    monkeypatch.setenv('BLUEPRINT_WEBSITE_SCENE_BINDING_ROOT', str(binding_root))
    handoff = prepare_website_scene_handoff(
        descriptor={'capture_id': 'walkthrough-req1', 'scene_id': 'site-req1', 'metadata': {
            'site_task_context': args['task_context'], 'website_scene_execution_authority': args['spend']}},
        clean_plate={'privacy_verified': True, 'status': 'objects_removed',
                     'task_masks': args['task_masks'], 'source_geometry': args['source_geometry'],
                     'removal_manifest_path': str(removal)},
        provider_run={'status': 'ready', 'world_id': 'world-1', 'provider_run_id': 'op-1',
                      'worldlabs_asset_materialization': {'manifest_path': str(assets)}},
        capture_root=capture, now=SCENE_SOURCE_EPOCH)
    assert handoff['status'] == 'intake_ready' and handoff.get('source_registration'), handoff
    preparation = json.loads(Path(handoff['preparation_path']).read_bytes())
    intake = source / 'intents'
    accepted = stage_scene_intent(value=json.loads(cross_runtime_canonical_json(preparation['intake_request'])),
                                  queue_root=intake, authenticated_client='blueprint-webapp',
                                  trusted_clients={'blueprint-webapp'}, now=SCENE_SOURCE_EPOCH)
    intent_id = accepted['intent_id']
    monkeypatch.setenv('BLUEPRINT_TASK_EVALUATION_SCENE_INTAKE_ROOT', str(intake))
    monkeypatch.setenv('BLUEPRINT_TASK_EVALUATION_SCENE_INTAKE_CLIENT_IDS', 'blueprint-webapp')
    queue = ensure_launch_preparation_queue_root(source / 'queue')
    input_root = source / 'inputs'
    input_root.mkdir()
    progression_output = source / 'factories'
    progression_output.mkdir()
    policy['roots'] = [
        {'root': str(input_root), 'storage_class': 'cache', 'device': input_root.stat().st_dev},
        {'root': str(progression_output), 'storage_class': 'host',
         'device': progression_output.stat().st_dev}]
    _sealed_file(base / 'policy.json', policy, 'policy_digest', mode=0o644)
    monkeypatch.setenv('BLUEPRINT_SCENE_RETIREMENT_POLICY_FILE', str(base / 'policy.json'))
    fixture = production_fixture(source / 'release')
    machinery_path = source / 'machinery.json'
    _sealed_file(machinery_path, {
        'schema_version': 'task_evaluation_website_scene_machinery.v1',
        'maximum_preparation_spend_usd': 0, 'provider': 'vast'}, 'machinery_digest')
    (source / 'repo').mkdir()
    release_path = source / 'release.json'
    _sealed_file(release_path, {
        'schema_version': RELEASE_SCHEMA, 'source_commit': SHA,
        'runtime_digest': 'sha256:' + 'f' * 64, 'repo_root': str(source / 'repo'),
        'runtime_publication_root': str(fixture['runtime_publication_root']),
        'namespace_timestamp': '20260919T120000Z', 'release_admission_mode': 'promoted',
        **{key: record(fixture[key]) for key in
           ('deploy_receipt', 'release_provenance', 'release_environment')}}, 'release_digest')
    config = source / 'config.json'
    _sealed_file(config, {
        'schema_version': engine.CONFIG_SCHEMA, 'intent_root': str(intake),
        'public_source_binding_root': str(source / 'unused'),
        'website_source_binding_root': str(binding_root),
        'website_source_machinery_path': str(machinery_path),
        'release_binding_path': str(release_path),
        'factory_output_root': str(progression_output),
        'trusted_clients': ['blueprint-webapp'], 'submission_enabled': True,
        'submission_transport': 'local_owned_queue', 'preparation_queue_root': str(queue),
        'service_account': pwd.getpwuid(os.geteuid()).pw_name,
        'publication_lock_root': str(source / 'locks')}, 'config_digest')
    monkeypatch.setattr(public_scene_host_input_intake, '_verified_checkout_head', lambda: SHA)
    monkeypatch.setattr(publication, '_verified_checkout_head', lambda: SHA)
    store = Store()
    def publish(**kwargs):
        return publication.publish_scene_configuration_submission(**kwargs, client=store)
    progression = engine.process_scene_intents(config_path=config, publisher=publish, now=SCENE_SOURCE_EPOCH)
    assert len(progression['results']) == 1 and progression['results'][0]['status'] == 'running', progression
    def fetch(uri, destination, maximum_bytes):
        data = store.objects[urlsplit(uri).path.lstrip('/')]
        assert len(data) <= maximum_bytes
        destination.write_bytes(data)
    prepared = worker.process_launch_preparation_queue(
        queue_root=queue, input_root=input_root,
        allowed_uri_prefixes=['s3://blueprint/task-evaluation/production-inputs/'],
        service_account=pwd.getpwuid(os.geteuid()).pw_name, source_commit=SHA,
        fetcher=fetch, construction_queue_root=source / 'construction')
    assert len(prepared['results']) == 1
    assert prepared['results'][0]['status'] == 'queued_for_production_scene_configuration', prepared
    result_path = next((queue / 'results').glob('*.json'))
    envelope_path = next((queue / state / result_path.name for state in ('materialized', 'completed')
                          if (queue / state / result_path.name).is_file()), queue / 'materialized' / result_path.name)
    request = json.loads(envelope_path.read_bytes())['request']
    return {'base': source, 'queue': queue, 'input_root': input_root, 'progression_output': progression_output,
            'intake': intake, 'intent_id': intent_id, 'owner': _raw(intake / intent_id / 'intent.json'),
            'request': request, 'envelope': (str(envelope_path), envelope_path.read_bytes()),
            'result': (str(result_path), result_path.read_bytes()), 'issued_at': SCENE_SOURCE_EPOCH,
            'capture': capture, 'handoff': handoff, 'binding_root': binding_root,
            'published_objects': dict(store.objects)}


def _select_source_owned_lineage(args, source, base, original_graph):
    """Select the source producer's workspace; retain unrelated old fixture bytes."""
    records = args['seed_records']
    history = base / 'unselected-queue-history'
    history.mkdir(mode=0o700, exist_ok=True)
    historical = []
    for role in ('source_snapshots', 'factories', 'source_submissions'):
        assert len(records[role]) == len(original_graph['seed_records'][role])
        historical.extend((role, row) for row in original_graph['seed_records'][role])
        records[role] = []
    old_admin = [row for row in records['attempts']
                 if json.loads(row[1])['schema_version'] == 'task_evaluation_scene_preparation_attempt.v1']
    assert len(old_admin) == 1
    records['attempts'].remove(old_admin[0])
    original_admin = [row for row in original_graph['seed_records']['attempts']
        if json.loads(row[1])['schema_version'] == 'task_evaluation_scene_preparation_attempt.v1']
    assert len(original_admin) == 1
    historical.append(('attempt', original_admin[0]))
    if 'capture' in source:
        for role in ('website_registrations', 'website_bindings', 'website_handoffs',
                     'website_preparations', 'website_runtime_inputs', 'website_task_contexts',
                     'submission_publications'):
            historical.extend((role, row) for row in original_graph['source_records'][role])
    retained_index = []
    for index, (role, (original_path, raw)) in enumerate(historical):
        path = history / f'source-history-{index:02d}-{role}.json'
        path.write_bytes(raw)
        args['source_records']['opaque_evidence'].append((str(path), raw))
        retained_index.append({'role': role, 'original_path': original_path, 'retained_path': str(path),
                               'sha256': 'sha256:' + hashlib.sha256(raw).hexdigest(),
                               'size_bytes': len(raw), 'disposition': 'KEEP'})
    index_path = history / 'source-history-index.json'
    index_bytes = json.dumps({'schema_version': 'retained_source_history_fixture.v1',
                              'records': retained_index}, sort_keys=True).encode()
    index_path.write_bytes(index_bytes)
    args['source_records']['opaque_evidence'].append((str(index_path), index_bytes))
    selected = list((source['intake'] / source['intent_id'] / 'preparation-attempts').glob('source-*.json'))
    assert len(selected) == 1
    attempt = selected[0]
    records['attempts'].append((str(attempt), attempt.read_bytes()))
    workspace = source['progression_output'] / source['intent_id'] / attempt.stem
    for role, relative_paths in {
        'source_snapshots': ('source_binding.json', 'machinery.json', 'release_binding.json'),
        'factories': ('factory.json',),
        'source_submissions': ('materialized/submission/scene_configuration_preparation_request.v1.json',
                               'materialized/submission/bundle_manifest.v1.json'),
    }.items():
        for relative in relative_paths:
            path = workspace / relative
            assert path.is_file(), path
            records[role].append((str(path), path.read_bytes()))
    args['roots']['factory_output_root'] = str(source['progression_output'])
    if 'capture' in source:
        publication = workspace / 'publication.json'
        assert publication.is_file(), publication
        args['source_records']['submission_publications'] = [
            (str(publication), publication.read_bytes())]
        capture_base = source['capture'] / 'pipeline' / 'website_scene_preparation'
        website_paths = {
            'website_registrations': [Path(source['handoff']['source_registration']['path'])],
            'website_bindings': list((source['progression_output'] / source['intent_id'] /
                                      'website-source').glob('*.json')),
            'website_handoffs': [capture_base / 'handoff.json'],
            'website_preparations': [capture_base / 'preparation.json'],
            'website_runtime_inputs': [capture_base / 'native' / 'runtime_inputs.json'],
            'website_task_contexts': [capture_base / 'task_context.json'],
        }
        for role, paths in website_paths.items():
            assert len(paths) == 1 and paths[0].is_file(), (role, paths)
            args['source_records'][role] = [(str(path), path.read_bytes()) for path in paths]
        args['roots']['website_source_binding_root'] = str(source['binding_root'])
        args['roots']['pubsub_root'] = str(source['base'] / 'pubsub')


def _park_synthetic_native_activation(args, base):
    """Keep an old test-only native pair outside the selected real queue."""
    history = base / 'unselected-queue-history' / 'synthetic-native-activation'
    history.mkdir(parents=True)
    records = []
    for role in ('native_activation_envelopes', 'native_activation_results',
                 'native_owner_records'):
        rows = args['bridge_records'][role]
        for index, (original_path, raw) in enumerate(rows):
            target = history / f'{role}-{index}.json'
            target.write_bytes(raw)
            records.append({'role': role, 'original_path': original_path,
                            'retained_path': str(target),
                            'sha256': 'sha256:' + hashlib.sha256(raw).hexdigest(),
                            'size_bytes': len(raw), 'disposition': 'KEEP'})
        args['bridge_records'][role] = []
    assert records
    (history / 'index.json').write_text(json.dumps({
        'schema_version': 'retained_synthetic_native_fixture.v1',
        'records': records}, sort_keys=True))


def _authentic_connected_graph(base, monkeypatch, policy, *, activation=True, website=False):
    from tests.test_scene_lifecycle_connected_acquisition import full_connected_finished_scene, installed
    from tests.test_task_evaluation_scene_intake import stage, attempt
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest, cross_runtime_canonical_digest

    source = (_website_owned_configuration(base, monkeypatch, policy) if website
              else _source_owned_configuration(base, monkeypatch, policy))
    original_graph = full_connected_finished_scene()
    args = _native_fixture_records(original_graph)
    # The source-family regression fixture deliberately has no current parent
    # for this progress row. A successful retirement must use a separately
    # produced, parent-bound progress chain instead of adopting those bytes.
    args['source_records']['source_progress'] = []
    old_intent = json.loads(args['seed_records']['intent'][1])
    intake = source['intake']
    accepted = {'intent_id': source['intent_id']}
    issued_attempt = attempt(intake, accepted, commit='a', cost=1, now=source['issued_at'] + 1)
    current_attempt = attempt(intake, accepted, attempt_id='a2', commit='b', cost=1,
                              now=source['issued_at'] + 1)
    actual_path = intake / accepted['intent_id'] / 'intent.json'
    actual = json.loads(actual_path.read_bytes())
    prior_body = copy.deepcopy(actual['request'])
    prior_body['submission_id'] = 'original-sam-owner'
    prior_body['owner'] = {'user_id': 'original-owner', 'organization_id': 'original-org'}
    prior_body['consent']['accepted_by'] = 'original-owner'
    prior_body['consent']['accepted_at_epoch'] = 99
    prior_body['execution']['expires_at_epoch'] = 1000
    prior_body['execution']['max_total_spend_usd'] = 4
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
        historical_scene_version = task['scene_identity']['version']
        historical_task_version = task['task_identity']['version']
        task['scene_identity'] = copy.deepcopy(source['request']['scene']['identity'])
        if raw_owner == prior_owner:
            task['scene_identity']['version'] = historical_scene_version
        task['task_identity'] = copy.deepcopy(source['request']['task']['identity'])
        if raw_owner == prior_owner:
            task['task_identity']['version'] = historical_task_version
        for key in ('strategy', 'subject', 'support', 'destination', 'success'):
            task[key] = intent['request']['task'][key]
        task['scene_intent_authority'] = {'intent': raw_owner, 'intent_digest': intent['intent_digest'], 'attempt': raw_attempt}
        task['human_authority'] = {'accepted_by': intent['request']['owner']['user_id'],
            'authority_reference': 'scene-intent:' + intent['intent_digest'],
            'accepted_on': datetime.fromtimestamp(intent['request']['consent']['accepted_at_epoch'], timezone.utc).isoformat()}
        return task
    task_versions = [json.loads(raw) for _, raw in args['source_records']['sam_host_tasks']]
    original_task = next(task for task in task_versions if task['expected_production_commit'] == 'a'*40)
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
    old_preparation = json.loads(args['seed_records']['preparation_envelopes'][0][1])['request']
    changes.update({old_preparation['scene']['identity']['id']: source['request']['scene']['identity']['id'],
                    old_preparation['task']['identity']['id']: source['request']['task']['identity']['id'],
                    old_preparation['team_namespace']: source['request']['team_namespace'],
                    old_preparation['run_id']: source['request']['run_id']})
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
            old_scene = old_preparation['scene']['identity']['id']
            new_scene = source['request']['scene']['identity']['id']
            revised = value.replace(old_id, accepted['intent_id'])
            if old_scene in revised and new_scene not in revised:
                revised = revised.replace(old_scene, new_scene)
            if revised != value:
                changes[value] = revised.replace('/retained', str(source['base']))
    changes[old_id] = accepted['intent_id']
    args['intent_id'] = accepted['intent_id']
    args['seed_records']['intent'] = (str(actual_path), actual_path.read_bytes())
    args = _rebase_complete_graph(args, source['base'], changes, requests=replacements,
                                 immutable_paths={str(actual_path), str(prior_path)})
    assert all(json.loads(raw).get('scene_intent_authority',{}).get('intent')==owner
        for _,raw in args['source_records']['sam_host_tasks']
        if json.loads(raw)['expected_production_commit']=='b'*40), [
            (path,json.loads(raw).get('scene_intent_authority',{}).get('intent'))
            for path,raw in args['source_records']['sam_host_tasks']
            if json.loads(raw)['expected_production_commit']=='b'*40]
    selected_current_task = next(json.loads(raw) for _, raw in args['source_records']['sam_host_tasks']
        if json.loads(raw)['expected_production_commit'] == 'b'*40
        and json.loads(raw)['scene_intent_authority']['intent'] == owner)
    _add_adopted_current_sam(args, source['base'], selected_current_task)
    task_bytes_before_worker=list(args['source_records']['sam_host_tasks'])
    args = _selected_worker_preparations(args, base, policy, monkeypatch, owner, birth, current_birth, source)
    _link_current_parent(args, current_birth)
    _select_source_owned_lineage(args, source, base, original_graph)
    if website:
        _park_synthetic_native_activation(args, base)
    assert Path(args['roots']['preparation_queue_root']) == source['queue']
    assert Path(args['roots']['preparation_input_root']) == source['input_root']
    source_identity = source['queue'] / 'identities' / (source['request']['preparation_id'] + '.json')
    assert source_identity.is_file()
    args['seed_records']['preparation_envelopes'].append(source['envelope'])
    args['seed_records']['preparation_results'].append(source['result'])
    args['source_records']['queue_identities'].append((str(source_identity), source_identity.read_bytes()))
    source_links = [path for path in (source['intake'] / source['intent_id'] / 'preparations').glob('*.json')
                    if json.loads(path.read_bytes())['preparation_id'] == source['request']['preparation_id']]
    assert len(source_links) == 1
    args['seed_records']['preparation_links'].append((str(source_links[0]), source_links[0].read_bytes()))
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
    # The source progression already wrote its own numbered event chain. The
    # historical composed fixture must remain byte-exact evidence elsewhere;
    # it cannot overwrite the live owner's event names or projection.
    historical_progress = base / 'unselected-queue-history' / 'historical-progress'
    historical_progress.mkdir(parents=True)
    for path, raw in [*args['seed_records']['events'], args['seed_records']['projection']]:
        target = historical_progress / ('projection.json' if path.endswith('/progression.json')
                                        else Path(path).name)
        target.write_bytes(raw)
        args['source_records']['opaque_evidence'].append((str(target), raw))
    progress_dir = source['intake'] / source['intent_id']
    args['seed_records']['events'] = [(str(path), path.read_bytes()) for path in sorted(
        (progress_dir / 'progression-events').glob('*.json'))]
    projection_path = progress_dir / 'progression.json'
    args['seed_records']['projection'] = (str(projection_path), projection_path.read_bytes())
    authentic_pair = (str(actual_path), actual_path.read_bytes())
    args['seed_records']['intent'] = None
    args, context, _, _ = installed(base, args, already_rebased=True,
                                   shared_hardlinks=not website)
    args['seed_records']['intent'] = authentic_pair
    _complete_installed_queue_layouts(context)
    _produce_current_sam_phase(args, context, base, monkeypatch)
    _produce_current_sam_progress(args, context, base)
    assert actual_path.read_bytes() == authentic_pair[1]
    monkeypatch.setenv('BLUEPRINT_TASK_EVALUATION_SCENE_INTAKE_ROOT', str(intake))
    monkeypatch.setenv('BLUEPRINT_TASK_EVALUATION_SCENE_INTAKE_CLIENT_IDS', 'blueprint-webapp,webapp')
    from blueprint_pipeline.task_evaluation_scene_owner_authority import validate_task_scene_owner
    for task_path, raw in args['source_records']['sam_host_tasks']:
        task = json.loads(raw)
        verified = validate_task_scene_owner(task, now=200 if task['scene_intent_authority']['intent'] == prior_owner
                                             else SCENE_SOURCE_EPOCH + 2)
        expected = prior['intent_id'] if task['expected_production_commit'] == 'a'*40 and task['scene_intent_authority']['intent']==prior_owner else accepted['intent_id']
        assert verified['intent_id'] == expected
    task_index = {(path, 'sha256:' + hashlib.sha256(raw).hexdigest(), len(raw)): json.loads(raw)
                  for path, raw in args['source_records']['sam_host_tasks']}
    for path, raw in args['source_records']['sam_plans']:
        plan_value = json.loads(raw)
        selector = plan_value['host_inputs']['task_request']
        assert (selector['path'], selector['sha256'], selector['size_bytes']) in task_index, (
            path, selector, [(key, value.get('scene_intent_authority',{}).get('intent'))
                             for key,value in task_index.items() if key[0]==selector['path']])
        task = task_index[selector['path'], selector['sha256'], selector['size_bytes']]
        for identity_key in ('scene_identity', 'task_identity', 'publisher_scene_id'):
            assert task.get(identity_key) == plan_value.get(identity_key), (
                path, identity_key, task.get(identity_key), plan_value.get(identity_key))
        expected_owner = owner if Path(path).name in {'current-scene-plan.json', 'adopted-current-plan.json'} else prior_owner
        assert task['scene_intent_authority']['intent'] == expected_owner, (
            path, task['expected_production_commit'], task['scene_intent_authority']['intent']['path'],
            expected_owner['path'])
    for _, raw in args['source_records']['sam_adoptions']:
        selector = json.loads(raw)['current_host_inputs']['task_request']
        assert task_index[selector['path'], selector['sha256'], selector['size_bytes']]['scene_intent_authority']['intent'] == owner
    assert actual['intent_digest'] == cross_runtime_canonical_digest(actual, digest_field='intent_digest')
    from blueprint_pipeline.task_evaluation_scene_source_family_inventory import ROLES, _join
    from blueprint_pipeline.task_evaluation_scene_source_family_sam import SCHEMAS as SAM_SCHEMAS
    for role, (schema, field) in SAM_SCHEMAS.items():
        if field:
            for path, raw in args['source_records'].get(role, []):
                value = json.loads(raw)
                if value.get('schema_version') == schema:
                    assert value[field] == canonical_digest(value, digest_field=field), (role, path)
    for role in ('website_handoffs', 'website_task_contexts'):
        for path, raw in args['source_records'][role]:
            value = json.loads(raw)
            assert f"/scenes/{value['scene_id']}/captures/{value['capture_id']}/" in path, (role, path, value)
    source_args={key:args[key] for key in ('intent_id','seed_records','downstream_records',
        'roots','parent_routes','retained_metadata_roots')}
    source_args['source_records']={role:args['source_records'][role] for role in ROLES}
    source_args['metadata_roots']=source_args.pop('retained_metadata_roots')
    allowed_roots = [*source_args['metadata_roots'], args['roots']['host_input_root'],
                     args['roots']['sam_queue_root'], args['roots']['sam_execution_root'],
                     args['roots']['factory_output_root'],
                     *(route['input_root'] for route in args['parent_routes'])]
    for role in SAM_SCHEMAS:
        if role in {'sam_parent_envelopes', 'sam_parent_results', 'source_progress', 'source_resume_signals'}:
            continue
        for path, _ in args['source_records'].get(role, []):
            assert any(Path(path).is_relative_to(root) for root in allowed_roots), (role, path, allowed_roots)
    _join(**source_args)
    if activation:
        _produce_current_activation(args, context, base, monkeypatch, policy, source)
    context['scene_construction_queue_root'] = str(source['base'] / 'construction')
    return args, context, (accepted['intent_id'], owner, birth), (prior['intent_id'], prior_owner, prior_birth), source


def _produce_current_activation(args, context, base, monkeypatch, policy, source):
    """Run the no-provider activation producer for the owned preparation."""
    from datetime import datetime, timedelta, timezone
    import grp
    import pwd
    import socket
    import types
    import zipfile
    from blueprint_pipeline import task_evaluation_scene_configuration_activation_automation as activation
    from blueprint_pipeline.task_evaluation_launch_activation_worker import process_launch_activation_queue
    from blueprint_pipeline.project_spend_reconciliation import materialize_project_spend_reconciliation
    from scripts.prepare_paid_lane_launch import (
        _load_scene_configuration_context, validate_paid_lane_launch, prepare_paid_lane_launch)
    from tests.astra_toolchain_fixture import astra_toolchain_fixture
    from tests.test_completed_scene_consumer_rehearsal import _preparation_runner
    from tests.test_project_spend_reconciliation import _human_baseline
    from tests.test_task_evaluation_scene_configuration_activation_automation import _provider_zero, _publisher

    queue = source['queue']
    result_path = Path(source['result'][0])
    result = json.loads(result_path.read_bytes())
    assert result['status'] == 'queued_for_production_scene_configuration'
    envelope_path = queue / 'materialized' / result_path.name
    request = json.loads(envelope_path.read_bytes())['request']
    # The composed lineage originally supplied a fixture-only activation-1
    # result. Preserve those exact bytes outside this scene's selected queue so
    # the real intake and worker can own the same pending identity and result.
    old_activation = 'activation-1'
    old_history = base / 'unselected-queue-history' / old_activation
    old_history.mkdir(parents=True)
    parked = set()
    for group, role in (('seed_records', 'activation_envelopes'),
                        ('downstream_records', 'activation_results'),
                        ('seed_records', 'configuration_progressions'),
                        ('downstream_records', 'launch_progressions')):
        if role.endswith('progressions'):
            matches = list(args[group][role])
        else:
            matches = [row for row in args[group][role] if json.loads(row[1]).get(
                'activation_id', json.loads(row[1]).get('request', {}).get('activation_id')) == old_activation]
        assert len(matches) == 1
        path, raw = matches[0]
        target = old_history / (role + '.json')
        Path(path).rename(target)
        assert target.read_bytes() == raw
        parked.add(path)
        args[group][role].remove(matches[0])
        args['source_records']['opaque_evidence'].append((str(target), raw))
    old_identity = Path(args['roots']['activation_queue_root']) / 'identities' / (old_activation + '.json')
    old_identity_target = old_history / 'identity.json'
    old_identity.rename(old_identity_target)
    parked.add(str(old_identity))
    args['source_records']['queue_identities'] = [row for row in args['source_records']['queue_identities']
        if row[0] != str(old_identity)]
    args['source_records']['opaque_evidence'].append((str(old_identity_target), old_identity_target.read_bytes()))
    context['retained_metadata_files'] = [row for row in context['retained_metadata_files']
        if row['path'] not in parked]
    baseline, _ = _human_baseline(base / 'activation-baseline.json')
    spend = base / 'activation-project-spend.json'
    materialize_project_spend_reconciliation(
        baseline_authority_path=baseline, posted_reconciliation_paths=[], expected_coverage_ids=[],
        completeness_reference=str(baseline), authorized_by='fixture-owner',
        authorized_on=datetime.now(timezone.utc).isoformat(), output_path=spend)
    registry = base / 'activation-intents'
    activation.provision_scene_configuration_activation_intent(
        expected_production_commit=request['expected_production_commit'],
        team_namespace=request['team_namespace'], scene_id=request['scene']['identity']['id'],
        task_id=request['task']['identity']['id'], authorization_reference='scene-intent:' + request['scene_intent_digest'],
        authorized_by=pwd.getpwuid(os.geteuid()).pw_name, profile_revision='fixture', valid_for_seconds=3600,
        project_spend_reconciliation_path=spend, rights_scope='internal_noncommercial_research_only',
        maximum_hard_cap_usd=request['spend']['hard_cap_usd'], release_reference='development-fixture',
        intent_root=registry, materialization_root=base / 'activation-intent-inputs', release_scoped=True)
    lineage = _publisher('scene-configuration-activation-lineage')
    window = _publisher('coordinator-release-windows')
    now = datetime.now(timezone.utc)
    activation_queue = Path(args['roots']['activation_queue_root'])
    staged = activation.advance_scene_configuration_activation(
        preparation_result_path=result_path, preparation_queue_root=queue,
        activation_queue_root=activation_queue, progression_root=args['roots']['configuration_progression_root'],
        intent_root=registry, provider_zero_collector=lambda: _provider_zero(now - timedelta(seconds=1)),
        lineage_publisher_factory=lambda: lineage, release_window_publisher_factory=lambda: window,
        now=now, running_commit=request['expected_production_commit'])
    assert staged['status'] == 'scene_configuration_activation_queued', staged.get('blockers', staged)
    payloads = {**lineage.published, **window.published}
    def fetch(uri, destination, maximum_bytes):
        data = payloads[uri]
        assert len(data) == maximum_bytes
        destination.write_bytes(data)
    monkeypatch.setattr(socket.socket, 'connect', lambda *_: pytest.fail('activation attempted network access'))
    activation_root = Path(args['roots']['activation_output_root'])
    activation_id = staged['activation_id']
    owned = activation_root / activation_id
    policy['roots'].append({'root': str(activation_root), 'storage_class': 'host',
                            'device': activation_root.stat().st_dev})
    _sealed_file(base / 'policy.json', policy, 'policy_digest', mode=0o644)
    plans = []
    def prepare(*, lane, context_path, **_):
        preparation_context = _load_scene_configuration_context(context_path, expected_lane=lane)
        plans.append(validate_paid_lane_launch(lane, preparation_context))
        return prepare_paid_lane_launch(lane, preparation_context,
            runner=_preparation_runner(base, monkeypatch,
                construction_queue_root=source['base'] / 'construction'))
    controls = base / 'controls-intents'
    controls.mkdir()
    # The source clock predates the retirement observation; Python ZIP needs
    # the real wall-clock timestamp for its tiny local toolchain bytes.
    monkeypatch.setattr(zipfile, 'time', types.SimpleNamespace(
        time=lambda: time.time_ns() / 1_000_000_000, localtime=time.localtime))
    toolchain = astra_toolchain_fixture(base / 'toolchain', request['expected_production_commit'], monkeypatch)
    assert list((activation_queue / 'pending').glob('*.json')), {
        state: [path.name for path in (activation_queue / state).glob('*.json')]
        for state in ('pending', 'processing', 'prepared', 'blocked', 'results')}
    run = process_launch_activation_queue(
        queue_root=activation_queue, preparation_queue_root=queue,
        preparation_input_root=args['roots']['preparation_input_root'], activation_root=activation_root,
        allowed_uri_prefixes=['s3://blueprint/task-evaluation/production-inputs/'],
        service_account=pwd.getpwuid(os.geteuid()).pw_name,
        service_group=grp.getgrgid(os.getegid()).gr_name,
        repository_root=Path(__file__).resolve().parents[1],
        destination_prefix='s3://blueprint/task-evaluation/production-inputs/fixture',
        release_window_prefix='s3://blueprint/task-evaluation/production-inputs/coordinator-release-windows/',
        profile_dir=owned / 'profiles', webapp_catalog=owned / 'catalog.json',
        standing_authorization_dir=owned / 'standing-authorizations',
        scene_construction_queue_root=source['base'] / 'construction',
        scene_configuration_toolchain_root=toolchain,
        configured_controls_autostart_intent_root=controls,
        source_commit=request['expected_production_commit'], fetcher=fetch, preparer=prepare)
    assert plans and plans[0]['status'] == 'validated_no_commands_run', run
    assert run['results'][0]['status'] == 'profile_authority_materialized_no_execution', run
    assert run['results'][0]['provider_mutation_performed'] is False
    assert run['results'][0]['paid_execution_requested'] is False
    actual_name = next((activation_queue / 'prepared').glob(activation_id + '-*.json')).name
    activation_request = json.loads((activation_queue / 'prepared' / actual_name).read_bytes())['request']
    receipt = owned / 'launch-set' / 'profile_publication_receipt.v1.json'
    authorization = next((owned / 'standing-authorizations').glob('*.json'))
    profile = owned / 'profiles' / (run['results'][0]['profile_id'] + '.json')
    release_window = owned / 'references' / activation_request['release_window']['digest'][7:]
    assert json.loads(profile.read_bytes())['profile_digest'] == run['results'][0]['profile_digest']
    assert _raw(release_window) == {'path': str(release_window),
        'sha256': activation_request['release_window']['digest'],
        'size_bytes': activation_request['release_window']['size_bytes']}
    assert json.loads(release_window.read_bytes())['window_digest'] == run['results'][0]['release_window_digest']
    assert run['results'][0]['profile_publication_receipt_digest'] == _raw(receipt)['sha256']
    assert run['results'][0]['standing_authorization_digest'] == _raw(authorization)['sha256']
    for artifact in (profile, release_window, receipt, authorization):
        raw = artifact.read_bytes()
        args['source_records']['opaque_evidence'].append((str(artifact), raw))
        context['retained_metadata_files'].append({'role': 'opaque_evidence', 'path': str(artifact)})
    args['seed_records']['activation_envelopes'].append((str(activation_queue / 'prepared' / actual_name),
        (activation_queue / 'prepared' / actual_name).read_bytes()))
    args['downstream_records']['activation_results'].append((str(activation_queue / 'results' / actual_name),
        (activation_queue / 'results' / actual_name).read_bytes()))
    actual_identity = activation_queue / 'identities' / (activation_id + '.json')
    args['source_records']['queue_identities'].append((str(actual_identity), actual_identity.read_bytes()))
    return run['results'][0], receipt, authorization


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
    for label, candidate in [('plan', job['plan_ref']), *job['inputs'].items()]:
        actual_ref = _raw(Path(candidate['path']))
        assert actual_ref == candidate, (label, candidate, actual_ref)
    result_pair = next(row for row in rows['sam_results']
                       if json.loads(row[1])['child_id'] == job['child_id'])
    prior_result = json.loads(result_pair[1])
    assert set(prior_result['artifacts']) == {'phase_artifact'}
    output = Path(args['roots']['sam_execution_root']) / job['parent_request_digest'][7:] / job['child_id']
    prior_output = Path(prior_result['artifacts']['phase_artifact']['path']).parent
    assert prior_output.is_relative_to(args['roots']['sam_execution_root'])
    receipt_matches = [row for row in rows['sam_execution_receipts']
                       if Path(row[0]).parent == prior_output]
    assert len(receipt_matches) == 1
    prior_receipt = receipt_matches[0]
    assert json.loads(prior_receipt[1])['job_digest'] == job['job_digest']
    prior_artifact = next(row for row in rows['opaque_evidence']
                          if row[0] == prior_result['artifacts']['phase_artifact']['path'])
    assert prior_artifact[1] == b'tiny-evidence'
    assert {path.name for path in prior_output.iterdir()} == {
        Path(prior_receipt[0]).name, Path(prior_artifact[0]).name}
    for pair in (job_pair, result_pair, prior_receipt, prior_artifact):
        Path(pair[0]).unlink()
    prior_output.rmdir()
    assert list(prior_output.parent.iterdir()) == []
    prior_output.parent.rmdir()
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
        local_bytes=64*1024*1024, archive_bytes=96*1024*1024,
        remote_bytes=192*1024*1024, elapsed_seconds=180)
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
def test_current_sam_worker_and_parent_link_use_real_selected_owner(short_scene_directory, monkeypatch):
    from tests.test_scene_retirement_real_participants import access_fixture

    base = short_scene_directory.resolve()
    _, policy, placeholder = access_fixture(base, monkeypatch)
    placeholder.rmdir()
    journals = Path(policy['journal_store'])
    journals.mkdir(mode=0o700)
    (journals / 'retired').mkdir(mode=0o700)
    monkeypatch.delenv('BLUEPRINT_SCENE_RETIREMENT_POLICY_FILE')
    args, _, main_owner, original_owner, _ = _authentic_connected_graph(base, monkeypatch, policy, activation=False)
    history_index = json.loads((base / 'unselected-queue-history' / 'source-history-index.json').read_bytes())['records']
    assert history_index and all(row['disposition'] == 'KEEP' for row in history_index)
    for row in history_index:
        raw = Path(row['retained_path']).read_bytes()
        assert row['original_path'].startswith('/retained/')
        assert row['sha256'] == 'sha256:' + hashlib.sha256(raw).hexdigest()
        assert row['size_bytes'] == len(raw)
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
def test_source_produced_preparation_reaches_local_activation_receipt(short_scene_directory, monkeypatch):
    from tests.test_scene_retirement_real_participants import access_fixture

    base = short_scene_directory.resolve()
    _, policy, placeholder = access_fixture(base, monkeypatch)
    placeholder.rmdir()
    journals = Path(policy['journal_store'])
    journals.mkdir(mode=0o700)
    (journals / 'retired').mkdir(mode=0o700)
    monkeypatch.delenv('BLUEPRINT_SCENE_RETIREMENT_POLICY_FILE')
    args, _, _, _, _ = _authentic_connected_graph(base, monkeypatch, policy)
    results = [json.loads(raw) for _, raw in args['downstream_records']['activation_results']]
    assert len(results) == 1
    result = results[0]
    assert result['status'] == 'profile_authority_materialized_no_execution'
    receipt = Path(args['roots']['activation_output_root']) / result['activation_id'] / 'launch-set' / 'profile_publication_receipt.v1.json'
    assert receipt.is_file() and result['profile_publication_receipt_digest'] == _raw(receipt)['sha256']


@pytest.mark.slow
@pytest.mark.parametrize("authority_end", ["revoked", "expired"])
def test_terminal_scene_retires_every_folder_it_wrote(short_scene_directory, monkeypatch, authority_end):
    # RED remains an actual feature failure rather than collection loss or xfail.
    # Authenticate the complete planner fixture before importing the engine.
    from blueprint_pipeline import task_evaluation_scene_retirement_access as access
    from blueprint_pipeline.task_evaluation_scene_retirement_authority import cohort_digest
    from blueprint_pipeline.task_evaluation_scene_lifecycle_plan import build_scene_lifecycle_plan, FAMILIES
    from tests.scene_lifecycle_fixture_support import stable_shared_ancestors
    from tests.test_scene_retirement_real_participants import access_fixture

    base = short_scene_directory.resolve()
    # The website producer has a real running progression, without a task
    # result. Future execution ends through the real revocation writer or the
    # original execution expiry, followed by the seven-day grace period.
    retirement_epoch = SCENE_SOURCE_EPOCH + 14 * 24 * 60 * 60
    _, policy, placeholder = access_fixture(base, monkeypatch)
    placeholder.rmdir()
    journals = Path(policy['journal_store'])
    journals.mkdir(mode=0o700)
    (journals / 'processes').mkdir(mode=0o700)
    (journals / 'retired').mkdir(mode=0o700)
    Path(str(journals) + '.metadata').mkdir(mode=0o750)
    # Issue intake before enrollment: stage's real publisher remains an actual
    # participant; disabled root policy grants it no fictional cleanup authority.
    monkeypatch.delenv('BLUEPRINT_SCENE_RETIREMENT_POLICY_FILE')
    args, context, main_owner, original_owner, source = _authentic_connected_graph(
        base, monkeypatch, policy, website=True)
    native_history = json.loads((base / 'unselected-queue-history' /
                                 'synthetic-native-activation' / 'index.json').read_bytes())['records']
    assert {row['role'] for row in native_history} == {
        'native_activation_envelopes', 'native_activation_results', 'native_owner_records'}
    selected_metadata = {row['path'] for row in context['retained_metadata_files']}
    for row in native_history:
        retained = Path(row['retained_path']).read_bytes()
        assert row['sha256'] == 'sha256:' + hashlib.sha256(retained).hexdigest()
        assert row['size_bytes'] == len(retained) and row['retained_path'] not in selected_metadata
        assert not Path(row['original_path']).exists()
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
        os.utime(path, (retirement_epoch - DEFAULT_MINIMUM_AGE_SECONDS - 1,
                        retirement_epoch - DEFAULT_MINIMUM_AGE_SECONDS - 1))
    stable_shared_ancestors(monkeypatch, base)
    initial = build_scene_lifecycle_plan(intent_id=args['intent_id'], context=context,
                                         observed_at_epoch=retirement_epoch)
    assert 'historical_lineage' in initial, (initial.get('blockers'),initial.get('reason'),initial.get('status'))
    assert initial['finished_observation']['status'] == 'unknown'
    assert {row['family'] for row in initial['family_obligations'] if row['member_count']} == set(FAMILIES), initial
    selected = [row for row in initial['measured_members'] if row.get('kinds')]
    shared_rows = [row for row in selected if 'prepared_cache_object' in row.get('kinds',[])]
    shared_content = [Path(row['path']) for row in shared_rows]
    assert len(shared_content) >= 2, [(row['path'],row['status'],row['keeps']) for row in shared_rows]
    absent_shared = [row for row in shared_rows if not Path(row['path']).exists()]
    assert not absent_shared, [(row['path'], row['status'], row['keeps']) for row in shared_rows]
    missing_cache = sorted(path.name for path in set(active_cache) - set(shared_content))
    assert not missing_cache, {
        'missing_cache': missing_cache,
        'shared_references': [(row['digest'], row['path']) for row in
            initial['historical_lineage']['source_family_inventory']['downstream_inventory']['seed']['shared_cache_references']
            if row['digest'][7:] in missing_cache],
        'projected': [(row['receipt_digest'], row.get('binding_strength'), row['path']) for row in
            initial['historical_lineage']['source_family_inventory']['downstream_inventory']['seed']['members']
            if row.get('kind') == 'preparation_projected_file' and row.get('receipt_digest', '')[7:] in missing_cache],
    }
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
    # The real website producer includes its local toolchain bundle; these
    # finite fixture caps cover those bytes while remaining below native limits.
    policy['limits'] = {'logical_payload_bytes': 512 * 1024 * 1024,
                        'archive_bytes': 96 * 1024 * 1024,
                        'remote_bytes': 192 * 1024 * 1024, 'elapsed_seconds': 1800}
    policy_path = base / 'policy.json'
    policy = _sealed_file(policy_path, policy, 'policy_digest', mode=0o644)
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
            owner_raw_ref=selected_owner[1], birth_request_raw_ref=selected_owner[2],
            now=200 if selected_owner == original_owner else SCENE_SOURCE_EPOCH + 3)
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
    if authority_end == 'revoked':
        from blueprint_pipeline.task_evaluation_scene_intake import revoke_scene_intent
        selected_intent = json.loads(Path(main_owner[1]['path']).read_bytes())
        revoked = revoke_scene_intent(queue_root=Path(context['roots']['intent_root']),
                                      intent_id=args['intent_id'],
                                      intent_digest=selected_intent['intent_digest'],
                                      owner=selected_intent['request']['owner'],
                                      now=SCENE_SOURCE_EPOCH + 5 * 24 * 60 * 60)
        assert revoked['status'] == 'revoked' and revoked['provider_mutation_performed'] is False
    else:
        # An absent extension directory proves nothing. Observe the exact
        # selected, empty owner directory before relying on the original expiry.
        extension_root = Path(context['roots']['intent_root']) / args['intent_id'] / 'execution-window-extensions'
        extension_root.mkdir(mode=0o700)
        selected_intent = json.loads(Path(main_owner[1]['path']).read_bytes())
        assert retirement_epoch >= selected_intent['request']['execution']['expires_at_epoch'] + 7 * 86400
    closure_blocker = 'scene_owner_revoked' if authority_end == 'revoked' else 'scene_execution_owner_expired'
    from blueprint_pipeline.task_evaluation_scene_construction_queue import finalize_scene_construction
    construction_root = Path(context['scene_construction_queue_root'])
    pending = sorted((construction_root / 'pending').glob('*.json'))
    assert len(pending) == 3, pending
    for path in pending:
        queued = json.loads(path.read_bytes())
        final = finalize_scene_construction(queue_root=construction_root,
            envelope={**queued, 'control_plane_envelope_digest': queued['envelope_digest']},
            terminal_result={'status': 'blocked', 'run_id': queued['run_id'],
                'source_commit': queued['expected_production_commit'],
                'configuration_completed': False, 'configured_scene_published': False,
                'configured_scene_revision_digest': None, 'publication_result_digest': None,
                'full_byte_service_account_readback_passed': False,
                'continuing_spend_from_this_run': False,
                'blockers': [closure_blocker]})
        assert final['status'] == 'blocked' and not path.exists()
    plan = build_scene_lifecycle_plan(intent_id=args['intent_id'], context=context,
                                      observed_at_epoch=retirement_epoch)
    assert 'finished_observation' in plan, plan.get('blockers', plan)
    assert plan['finished_observation']['status'] == authority_end + '_grace_elapsed', plan['finished_observation']
    activation_result = next(json.loads(raw) for _, raw in args['downstream_records']['activation_results'])
    source_inventory = plan['historical_lineage']['source_family_inventory']
    raw_bound = [row for row in source_inventory['lexical_members']
                 if row.get('binding', {}).get('raw_artifacts_bound')]
    assert raw_bound, (
        [(row['kind'], row['path']) for row in source_inventory['lexical_members']
         if row['kind'] == 'activation_workspace'],
        [(row['role'], row['sha256'] == activation_result['profile_publication_receipt_digest'])
         for row in source_inventory['raw_versions'] if row['role'] == 'opaque_evidence'],
        activation_result['status'])
    assert any(proof['sha256'] == activation_result['profile_publication_receipt_digest']
               for row in plan['measured_members'] for proof in row['source_provenance']), (
        [(row['path'], [(proof['role'], proof['sha256'] ==
          activation_result['profile_publication_receipt_digest']) for proof in row['source_provenance']])
         for row in plan['measured_members'] if 'activation_workspace' in row['kinds']],
        [(row['path'], [(proof['role'], proof['sha256'] ==
          activation_result['profile_publication_receipt_digest']) for proof in row['source_provenance']])
         for row in raw_bound])
    assert any(proof.get('seal_digest') == activation_result['profile_digest']
               for row in plan['measured_members'] for proof in row['source_provenance'])
    assert any(proof.get('seal_digest') == activation_result['release_window_digest']
               for row in plan['measured_members'] for proof in row['source_provenance'])
    selected_digests = {proof['sha256'] for row in plan['measured_members']
                        for proof in row['source_provenance']}
    unselected_raw = [(row['observation']['contract_path'], row['observation']['digest'],
                       row['observation']['source']['role'], row['observation']['source']['row_path'])
                      for row in plan['reference_observation']['protections']
                      if row['kind'] == 'raw_digest_selector_obligations'
                      and row['observation']['digest'] not in selected_digests]
    assert not unselected_raw, unselected_raw
    selected_rows = {(proof['path'], proof['sha256'], proof['size_bytes'])
                     for row in plan['measured_members'] for proof in row['source_provenance']
                     if proof['role'] not in {'preparation_identity','activation_identity','queue_identities'}}
    bound_preparations = [row for row in
        plan['historical_lineage']['preparation_handoff_observations']
        if row.get('pre_handoff_binding_verified') is True]
    assert bound_preparations
    selected_rows.update((proof['path'], proof['sha256'], proof['size_bytes'])
                         for row in bound_preparations for proof in row['source_provenance']
                         if proof['role'] in {'native_preparation_envelopes','native_preparation_results'})
    missing_records = [(row['source']['family'], row['source']['role'],
                        Path(row['source']['row_path']).name)
                       for row in plan['reference_observation']['record_dispositions']
                       if row['source']['role'] != 'identity'
                       and (row['source']['row_path'], row['source']['raw_sha256'],
                            row['source']['raw_size_bytes']) not in selected_rows]
    assert not missing_records, missing_records[:5]
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
        'action': 'retire', 'created_at': retirement_epoch - 1,
        'expires_at': retirement_epoch + 10,
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
    for receipt_path, raw in args['source_records']['submission_publications']:
        receipt = json.loads(raw)
        for row in receipt['published_objects']:
            assert not row['relative_path'].startswith('source/')
            matches = [data for path, data in _record_pairs(args) if path.endswith('/' + row['relative_path'])
                       and 'sha256:' + hashlib.sha256(data).hexdigest() == row['digest']]
            if not matches:
                produced = source['published_objects'].get(urlsplit(row['uri']).path.lstrip('/'))
                if produced is not None and len(produced) == row['size_bytes'] and (
                        'sha256:' + hashlib.sha256(produced).hexdigest() == row['digest']):
                    matches = [produced]
            assert matches and all(data == matches[0] for data in matches), (
                receipt_path, row, urlsplit(row['uri']).path.lstrip('/') in source['published_objects'],
                len(source['published_objects']))
            published[row['uri']] = bytes(matches[0])
    transport = MemoryArchiveTransport(members, published)
    from blueprint_pipeline.task_evaluation_scene_retirement import retire_scene, restore_scene
    from blueprint_pipeline import task_evaluation_scene_retirement_supervisor as supervisor
    # The production reader closure requires a protected Linux installation.
    # The native refusal is covered separately; this fixture verifies archive,
    # journal and restore mechanics under an exact test-only reader boundary.
    def fixture_reader_closure(observed_policy, allowance):
        allowance.tick()
        assert observed_policy['policy_digest'] == policy['policy_digest']
        assert observed_policy['consumer_cohort'] == _installed_cohort()
    monkeypatch.setattr(supervisor, 'require_current_reader_closure', fixture_reader_closure)
    assert any(row.get('pre_handoff_binding_verified') for row in
        plan['historical_lineage']['preparation_handoff_observations']), (
        plan['historical_lineage']['preparation_handoff_observations'])
    retired = retire_scene(plan_path, consent_path, transport=transport,
                           now=lambda: retirement_epoch, monotonic=time.monotonic)
    assert retired['status'] == 'retired', retired.get('reason')
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
                             now=lambda: retirement_epoch + 1, monotonic=time.monotonic)
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
                                 now=lambda: retirement_epoch + 2, monotonic=time.monotonic)
    except (ValueError, RuntimeError):
        pass
    else:
        assert repeated['status'] not in {'restored', 'retired'}, repeated
    assert _snapshot(members) == before
