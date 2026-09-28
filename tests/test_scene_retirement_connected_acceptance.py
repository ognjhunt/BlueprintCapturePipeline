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


def _rebase_complete_graph(args, anchor, changes, *, requests=None, immutable_paths=()):
    """Resolve fixture aliases through the complete acyclic raw-record DAG.

    The existing stat-only fixture rebaser intentionally replaces a selector
    only once. Authentic intake plus a newly selected SAM parent needs the
    transitive replacement too. This is bounded fixture encoding, no observer
    or action implementation and no successful protection flags.
    """
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest, cross_runtime_canonical_digest
    original = copy.deepcopy(args)
    pairs = list(_record_pairs(original))
    maps, sizes = dict(changes), {}
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

    def visit(value):
        if isinstance(value, str):
            return text(value)
        if isinstance(value, list):
            return [visit(item) for item in value]
        if not isinstance(value, dict):
            return value
        replacement = (requests or {}).get(canonical_digest(value))
        if replacement is not None:
            return copy.deepcopy(replacement)
        changed = {key: visit(item) for key, item in value.items()}
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
        current['source_records']['sam_plans'][0][0]: '/retained/metadata/current-scene-plan.json',
        's3://test/plan.json': 's3://test/current-scene-plan.json'},
        requests={canonical_digest(json.loads(previous_task[1])): owned_current_task})
    old_parent = json.loads(current['source_records']['sam_parent_envelopes'][0][1])
    previous_pair = args['seed_records']['preparation_envelopes'][0]
    previous = json.loads(previous_pair[1])
    parent = copy.deepcopy(previous)
    parent['request']['runtime'] = old_parent['request']['runtime']
    parent['request_digest'] = canonical_digest(parent['request'])
    parent = seal(parent, 'envelope_digest')
    path = args['roots']['preparation_queue_root'] + '/completed/prep-1-' + parent['request_digest'][7:] + '.json'
    parent_pair = pair(path, parent)
    args['seed_records']['preparation_envelopes'][0] = parent_pair
    # A source-parent retained copy has its own actual configured queue route;
    # one raw file is never assigned two conflicting parser roles.
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


def _authentic_connected_graph(base, monkeypatch):
    from tests.test_scene_lifecycle_connected_acquisition import full_connected_finished_scene, installed
    from tests.scene_lifecycle_fixture_support import rebase_graph
    from tests.test_task_evaluation_scene_intake import request, stage, attempt
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest, cross_runtime_canonical_digest

    args = full_connected_finished_scene()
    old_intent = json.loads(args['seed_records']['intent'][1])
    body = request()
    body['execution']['allowed_providers'] = ['vast', 'openai']
    body['owner'] = old_intent['request']['owner']
    body['consent']['accepted_by'] = body['owner']['user_id']
    body['source'] = old_intent['request']['source']
    body['task']['task_id'] = old_intent['request']['task']['task_id']
    intake = base / 'intents'
    accepted = stage(intake, body)
    issued_attempt = attempt(intake, accepted)
    actual_path = intake / accepted['intent_id'] / 'intent.json'
    actual = json.loads(actual_path.read_bytes())
    prior_body = copy.deepcopy(body)
    prior_body['submission_id'] = 'original-sam-owner'
    prior_body['owner'] = {'user_id': 'original-owner', 'organization_id': 'original-org'}
    prior_body['consent']['accepted_by'] = 'original-owner'
    prior = stage(intake, prior_body)
    prior_attempt = attempt(intake, prior)
    prior_path = intake / prior['intent_id'] / 'intent.json'
    prior_value = json.loads(prior_path.read_bytes())
    owner = _raw(actual_path)
    birth = _raw(intake / accepted['intent_id'] / 'attempts' / (issued_attempt['attempt_id'] + '.json'))
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
    own_current = owned_task(original_task, actual, owner, birth)
    replacements = {canonical_digest(old_intent['request']): actual['request']}
    for task in task_versions:
        replacements[canonical_digest(task)] = owned_task(task, prior_value if task['expected_production_commit'] == 'a'*40 else actual,
            prior_owner if task['expected_production_commit'] == 'a'*40 else owner,
            prior_birth if task['expected_production_commit'] == 'a'*40 else birth)
    changes = _add_current_sam(args, own_current)
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
    # Intake publishes immutable owner bytes. The metadata installer must not
    # reopen that real authority for writing merely to install fixture copies.
    authentic_pair = (str(actual_path), actual_path.read_bytes())
    args['seed_records']['intent'] = None
    args, context, _, _ = installed(base, args)
    args['seed_records']['intent'] = authentic_pair
    assert actual_path.read_bytes() == authentic_pair[1]
    monkeypatch.setenv('BLUEPRINT_TASK_EVALUATION_SCENE_INTAKE_ROOT', str(intake))
    monkeypatch.setenv('BLUEPRINT_TASK_EVALUATION_SCENE_INTAKE_CLIENT_IDS', 'webapp')
    from blueprint_pipeline.task_evaluation_scene_owner_authority import validate_task_scene_owner
    for _, raw in args['source_records']['sam_host_tasks']:
        task = json.loads(raw)
        verified = validate_task_scene_owner(task, now=200)
        expected = prior['intent_id'] if task['expected_production_commit'] == 'a'*40 and task['scene_intent_authority']['intent']==prior_owner else accepted['intent_id']
        assert verified['intent_id'] == expected
    task_index = {(path, 'sha256:' + hashlib.sha256(raw).hexdigest(), len(raw)): json.loads(raw)
                  for path, raw in args['source_records']['sam_host_tasks']}
    for path, raw in args['source_records']['sam_plans']:
        selector = json.loads(raw)['host_inputs']['task_request']
        task = task_index[selector['path'], selector['sha256'], selector['size_bytes']]
        expected_owner = owner if Path(path).name == 'current-scene-plan.json' else prior_owner
        assert task['scene_intent_authority']['intent'] == expected_owner
    for _, raw in args['source_records']['sam_adoptions']:
        selector = json.loads(raw)['current_host_inputs']['task_request']
        assert task_index[selector['path'], selector['sha256'], selector['size_bytes']]['scene_intent_authority']['intent'] == owner
    assert actual['intent_digest'] == canonical_digest(actual, digest_field='intent_digest')
    return args, context, (accepted['intent_id'], owner, birth), (prior['intent_id'], prior_owner, prior_birth)


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


def _consented_inventories(members):
    """Owner issuance hashes real bytes; the metadata-only planner does not."""
    from blueprint_pipeline.task_evaluation_scene_retirement_preservation import ActionAllowance, _scan, _payload
    from blueprint_pipeline.task_evaluation_scene_retirement_mutation import inventory_digest
    allowance = ActionAllowance(expires_at=999, now=lambda: 200, monotonic=time.monotonic,
        local_bytes=1024*1024, archive_bytes=2*1024*1024, remote_bytes=4*1024*1024, elapsed_seconds=60)
    files, directories, scanned = [], [], []
    for index, member in enumerate(members):
        scanned.append(_scan(member, index, allowance, files, directories))
    by_inode = {}
    for row in files:
        by_inode.setdefault(tuple(row['physical_identity'][:2]), []).append(row)
        digest = hashlib.sha256()
        for chunk in _payload(members[row['member_index']] / row['relative_path'], row, allowance):
            digest.update(chunk)
        row['sha256'] = 'sha256:' + digest.hexdigest()
    for identity, rows in by_inode.items():
        assert all(row['snapshot'][-1] == len(rows) for row in rows), 'unselected hardlink alias in fixture'
        if len(rows) > 1:
            for row in rows:
                row['hardlink_group'] = 'inode-' + str(identity[0]) + '-' + str(identity[1])
    preserved = dict(members=scanned, files=files, directories=directories)
    return {str(member): inventory_digest(preserved, index) for index, member in enumerate(members)}


class MemoryArchiveTransport:
    """Actual streamed bytes, with source-presence checks during fresh readback."""

    def __init__(self, members):
        self.members = members
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

    def read_archive(self, uri):
        if self.retirement:
            assert all(path.exists() for path in self.members), 'fresh readback followed local removal'
        self.events.append(('readback', uri))
        data = self.objects[uri]
        for start in range(0, len(data), 37):
            yield data[start:start + 37]


@pytest.mark.slow
def test_terminal_scene_retires_every_folder_it_wrote(tmp_path, monkeypatch):
    # RED remains an actual feature failure rather than collection loss or xfail.
    # Authenticate the complete planner fixture before importing the engine.
    from blueprint_pipeline import task_evaluation_scene_retirement_access as access
    from blueprint_pipeline.task_evaluation_scene_retirement_authority import cohort_digest
    from blueprint_pipeline.task_evaluation_scene_lifecycle_plan import build_scene_lifecycle_plan, FAMILIES
    from tests.scene_lifecycle_fixture_support import stable_shared_ancestors
    from tests.test_scene_retirement_real_participants import access_fixture

    base = tmp_path.resolve()
    _, policy, placeholder = access_fixture(base, monkeypatch)
    placeholder.rmdir()
    # Issue intake before enrollment: stage's real publisher remains an actual
    # participant; disabled root policy grants it no fictional cleanup authority.
    monkeypatch.delenv('BLUEPRINT_SCENE_RETIREMENT_POLICY_FILE')
    args, context, main_owner, original_owner = _authentic_connected_graph(base, monkeypatch)
    stable_shared_ancestors(monkeypatch, base)
    initial = build_scene_lifecycle_plan(intent_id=args['intent_id'], context=context, observed_at_epoch=200)
    assert 'historical_lineage' in initial, initial
    assert {row['family'] for row in initial['family_obligations'] if row['member_count']} == set(FAMILIES), initial
    selected = [row for row in initial['measured_members'] if row.get('kinds')]
    # A birth owns directories. Native compilation's exact file obligations are
    # retained under their actual owning directory, never treated as extra roots.
    paths = {Path(row['path']).parent if Path(row['path']).is_file() else Path(row['path']) for row in selected}
    members = sorted((path for path in paths if not any(other != path and path.is_relative_to(other)
                     for other in paths)), key=str)
    for member in members:
        member.mkdir(parents=True, exist_ok=True)
    policy['roots'] = [{'root': str(path.parent), 'storage_class': 'host', 'device': path.parent.stat().st_dev}
                       for path in members]
    policy['principals'] = [{'principal_id': 'fixture-owner', 'actions': ['retire', 'restore'],
                            'owner_intent_ids': [main_owner[0], original_owner[0]],
                            'private_archive_classes': ['host']}]
    policy['private_archive_allowed_classes'] = ['host']
    policy['consumer_cohort'] = _installed_cohort()
    policy['limits'] = {'logical_payload_bytes': 1024 * 1024, 'archive_bytes': 2 * 1024 * 1024,
                        'remote_bytes': 4 * 1024 * 1024, 'elapsed_seconds': 60}
    policy_path = base / 'policy.json'
    _sealed_file(policy_path, policy, 'policy_digest', mode=0o644)
    monkeypatch.setenv('BLUEPRINT_SCENE_RETIREMENT_POLICY_FILE', str(policy_path))
    generations = {}
    original_paths = {Path(row['path']) for row in selected if 'sam_original_execution_dependency' in row['kinds']}
    # Recreate these tiny fixture directories through the actual authenticated
    # producer birth hook. No pre-existing legacy folder is silently adopted.
    for member in members:
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
    original = _snapshot(members)
    inventories = _consented_inventories(members)
    plan = build_scene_lifecycle_plan(intent_id=args['intent_id'], context=context, observed_at_epoch=200)
    assert plan['finished_observation']['status'] == 'completed'
    assert {row['family'] for row in plan['family_obligations'] if row['member_count']} == set(FAMILIES)
    assert plan['action'] == 'KEEP' and plan['cleanup_authorized'] is False
    assert all(row['action'] == 'KEEP' for row in plan['family_obligations'])
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
            for path in members], 'private_archive_classes': ['host']}
    _sealed_file(consent_path, consent, 'consent_digest')
    transport = MemoryArchiveTransport(members)
    from blueprint_pipeline.task_evaluation_scene_retirement import retire_scene, restore_scene
    retired = retire_scene(plan_path, consent_path, transport=transport, now=lambda: 200, monotonic=time.monotonic)
    assert retired['status'] == 'retired', retired
    assert set(row['canonical_path'] for row in retired['members']) == set(map(str, members))
    assert all(not path.exists() for path in members)
    journal_ref = retired['retired_journal_raw_ref']
    assert _raw(journal_ref['path']) == journal_ref
    immutable_snapshot = Path(journal_ref['path']).read_bytes()
    intent_receipt = Path(retired['intent_receipt_path'])
    assert intent_receipt.is_file()
    assert json.loads(intent_receipt.read_bytes())['status'] == 'retired'
    assert transport.objects and all(('readback', uri) in transport.events for uri in transport.objects)
    restore_consent = dict(consent, consent_id='2' * 32, action='restore', plan_raw_ref=None,
                           retired_journal_raw_ref=journal_ref)
    restore_path = consent_path.with_name('restore-consent.json')
    _sealed_file(restore_path, restore_consent, 'consent_digest')
    transport.retirement = False
    restored = restore_scene(Path(journal_ref['path']), restore_path, transport=transport,
                             now=lambda: 201, monotonic=time.monotonic)
    assert restored['status'] == 'restored', restored
    roundtrip = _snapshot(members)
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
