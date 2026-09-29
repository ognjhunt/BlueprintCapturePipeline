"""Finite metadata discovery then supplied-record selection; no payload reads."""
from __future__ import annotations

import hashlib
import json
import re
import stat
from pathlib import PurePosixPath

from . import task_evaluation_scene_compilation_native_owner_inventory as native
from . import task_evaluation_scene_preparation_lineage as retained
from . import task_evaluation_scene_source_attempt_lineage as attempts
from . import task_evaluation_scene_source_family_contracts as family_contracts
from . import task_evaluation_scene_downstream_terminal as terminal_contracts
from .task_evaluation_scene_compilation_native_owners import PHASES
from .task_evaluation_scene_lifecycle_acquisition import AcquisitionError, require
from .task_evaluation_scene_lineage_budget import _work_items, _work_parse

HEX = r'[0-9a-f]{64}'
ID = r'[A-Za-z0-9][A-Za-z0-9._-]{0,191}'
STEM = ID + '-' + HEX
STATES = ('pending', 'processing', 'awaiting_source_preparation', 'awaiting_capacity', 'materialized', 'completed', 'blocked')
ALL_ROLES = native.prior.downstream.seed_module._ROLES | native.prior.downstream.ROLES | native.prior.ROLES | native.ROLES
SELECTOR_ROLES = ALL_ROLES | {'revocations', 'extensions', 'progression_config'}


# Exact supported producer schemas. Raw-only roles deliberately have no schema
# promotion; unknown versions remain available as protected acquired identities.
SCHEMAS = {
    'intent': {'task_evaluation_scene_intent.v1'},
    'projection': {'task_evaluation_scene_progression.v1'},
    'other_owner_intents': {'task_evaluation_scene_intent.v1'},
    'other_owner_projections': {'task_evaluation_scene_progression.v1'},
    'other_owner_revocations': {'task_evaluation_scene_intent_revocation.v1'},
    'events': {'task_evaluation_scene_progression_event.v1'},
    'attempts': {attempts._ADMIN_SCHEMA, attempts._PAID_SCHEMA},
    'source_snapshots': set(attempts._BINDINGS) | set().union(*attempts._MACHINERY.values()) | {'task_evaluation_public_scene_release_binding.v1'},
    'factories': set(attempts._FACTORIES),
    'source_submissions': {'task_evaluation_launch_preparation_request.v1', 'task_evaluation_scene_configuration_submission_manifest.v1'},
    'preparation_links': {'task_evaluation_scene_preparation_link.v1'},
    'parent_envelopes': {'task_evaluation_launch_preparation_envelope.v1'},
    'parent_results': {'task_evaluation_launch_preparation_result.v1'},
    'preparation_results': {'task_evaluation_launch_preparation_result.v1'},
    'configuration_progressions': {'task_evaluation_scene_configuration_activation_progression.v1'},
    'launch_progressions': {'task_evaluation_scene_configuration_activation_progression.v1'},
    'activation_envelopes': {'task_evaluation_launch_activation_envelope.v1'},
    'activation_results': {'task_evaluation_launch_activation_result.v1'},
    'compilation_envelopes': {'task_evaluation_episode_compilation_envelope.v1'},
    'compilation_results': {'task_evaluation_episode_compilation_result.v1'},
    'launch_profiles': {'task_evaluation_launch_profile.v1'},
    'launch_requests': {'task_evaluation_launch_request.v1'},
    'launch_receipts': {'task_evaluation_launch_receipt.v1'},
    'terminal_states': {'task_evaluation_scene_terminal_index_state.v1', 'task_evaluation_scene_nonexecution_terminal_state.v1'},
    'canary_dispatches': {'task_evaluation_policy_canary_dispatch.v1', 'task_evaluation_policy_canary_preprovider_blocked.v1'},
    'source_progress': {'task_evaluation_sam31_preparation_progress.v1'},
    'source_resume_signals': {'task_evaluation_sam31_preparation_resume.v1'},
    'sam_execution_progress': {'task_evaluation_sam31_preparation_execution_progress.v1'},
    'website_handoffs': {'website_scene_handoff.v1'},
    'submission_publications': {'task_evaluation_scene_configuration_submission_publication.v1'},
    **family_contracts.SUPPORTED_SCHEMAS,
    **{role: {schema} for role, schema in terminal_contracts.SCHEMAS.items()},
    **{role: {schema} for role, (schema, _) in native.c.SCHEMAS.items()},
}


def supported(row):
    schemas = SCHEMAS.get(row['role'])
    return schemas is None or row['value'].get('schema_version') in schemas


class Pool:
    def __init__(self, reader, context, intent_id):
        self.reader, self.budget, self.context, self.intent_id = reader, reader.budget, context, intent_id
        self.raw, self.paths, self.groups, self.scopes = [], set(), set(), []
        self.total = 0

    def scope(self, role, directory, status):
        self.budget.charge('facts')
        self.scopes.append({'role': role, 'path': directory, 'status': status, 'ownership_proven': False})

    def read(self, role, path):
        self.budget.tick()
        if path in self.paths:
            return
        try:
            if role=='configured_revisions' and not path.endswith('.json'):
                raw=self.reader.read_json(path,configured_revision_root=(
                    self.context['roots']['preparation_input_root']))
            else:
                raw = self.reader.read_json(path)
        except FileNotFoundError:
            self.scope(role, path, 'unavailable')
            return
        except AcquisitionError as exc:
            if str(exc) == 'scene_lifecycle_metadata_unavailable':
                self.scope(role, path, 'unavailable')
                return
            raise
        require(self.total + len(raw) <= native.MAX_TOTAL_BYTES and len(self.raw) < native.MAX_RECORDS,
                'metadata_pool_limit')
        self.budget.charge('rows')
        self.budget.charge('facts')
        self.raw.append((role, path, raw))
        self.paths.add(path)
        self.total += len(raw)

    def names(self, role, directory, pattern):
        group = role, directory
        if group not in self.groups:
            self.budget.charge('groups')
            self.groups.add(group)
        try:
            names = self.reader.entries(directory)
        except FileNotFoundError:
            self.scope(role, directory, 'unavailable')
            return ()
        regex = re.compile(pattern + r'\Z')
        selected = []
        for name in _work_items(names, self.budget):
            if regex.fullmatch(name):
                self.budget.charge('facts')
                selected.append(name)
            else:
                self.scope(role, directory, 'unsupported_layout')
        self.scope(role, directory, 'observed_selected_layout')
        return selected

    def rows(self, role, directory, pattern):
        for name in self.names(role, directory, pattern):
            self.read(role, directory + '/' + name)

    def discovery(self):
        roots = self.context['roots']
        owner = roots['intent_root'] + '/' + self.intent_id
        self.read('intent', owner + '/intent.json')
        self.read('projection', owner + '/progression.json')
        self.rows('events', owner + '/progression-events', r'[0-9]{6}\.json')
        for role, dirname, pattern in [('attempts', 'attempts', ID), ('source_snapshots', 'source-snapshots', HEX),
                                      ('preparation_links', 'preparations', HEX + r'(?:\.activation)?')]:
            self.rows(role, owner + '/' + dirname, pattern + r'\.json')
        self.rows('attempts', owner + '/preparation-attempts', ID + r'\.json')
        self.read('revocations', owner + '/revoked.json')
        for other in self.names('other_owner_groups', roots['intent_root'], ID):
            if other == 'progression-cursor.json':
                # The progression writer keeps one shared root cursor. Read its
                # actual no-follow bytes as protected metadata; it is never an
                # owner directory or selected-scene member.
                self.read('progression_cursor', roots['intent_root'] + '/' + other)
                continue
            if other != self.intent_id:
                for role, name in [('other_owner_intents', 'intent.json'),
                                   ('other_owner_projections', 'progression.json'),
                                   ('other_owner_revocations', 'revoked.json')]:
                    self.read(role, roots['intent_root']+'/'+other+'/'+name)
        self.rows('extensions', owner + '/execution-window-extensions', HEX + r'\.json')
        factory = roots['factory_output_root'] + '/' + self.intent_id
        for attempt in self.names('factory_children', factory, ID):
            base = factory + '/' + attempt
            for role, leaf in [('source_snapshots', 'source_binding.json'), ('source_snapshots', 'machinery.json'),
                               ('source_snapshots', 'release_binding.json'), ('factories', 'factory.json'),
                               ('factories', 'materialized/factory_receipt.json'),
                               ('source_submissions', 'materialized/submission/scene_configuration_preparation_request.v1.json'),
                               ('source_submissions', 'materialized/submission/bundle_manifest.v1.json'),
                               ('submission_publications', 'publication.json'),
                               ('sam_prefix_selections', 'materialized/prefix_selection.json')]:
                self.read(role, base + '/' + leaf)
        self.rows('website_bindings', factory + '/website-source', HEX + r'\.json')
        self.rows('website_registrations', roots['website_source_binding_root'], HEX + r'\.json')
        for route in self.context['parent_routes']:
            queue = route['queue_root']
            for state in STATES:
                self.rows('parent_envelopes', queue + '/' + state, STEM + r'\.json')
            self.rows('parent_results', queue + '/results', STEM + r'\.json')
            for preparation in self.names('source_progress_groups', queue + '/source-progress', STEM):
                self.rows('source_progress', queue + '/source-progress/' + preparation, r'[0-9]{6,}-' + HEX + r'\.json')
            self.rows('source_resume_signals', queue + '/source-resume-pending', HEX + r'\.json')
            for preparation in self.names('resume_groups', queue + '/source-resume-completed', STEM):
                self.rows('source_resume_signals', queue + '/source-resume-completed/' + preparation, HEX + r'\.json')
        for state in ('pending', 'processing', 'prepared', 'blocked'):
            self.rows('activation_envelopes', roots['activation_queue_root'] + '/' + state, STEM + r'\.json')
        self.rows('activation_results', roots['activation_queue_root'] + '/results', STEM + r'\.json')
        for state in ('pending', 'processing', 'completed', 'blocked'):
            self.rows('compilation_envelopes', roots['compilation_queue_root'] + '/' + state, STEM + r'\.json')
        self.rows('compilation_results', roots['compilation_queue_root'] + '/results', STEM + r'\.json')
        for state in ('pending', 'processing', 'waiting_external', 'completed', 'failed'):
            self.rows('sam_jobs', roots['sam_queue_root'] + '/' + state, 'sam31-' + HEX + r'\.json')
        self.rows('sam_results', roots['sam_queue_root'] + '/results', 'sam31-' + HEX + r'(?:\.conflict-' + HEX + r')?\.json')
        for child in self.names('sam_progress_groups', roots['sam_queue_root'] + '/progress', 'sam31-' + HEX):
            self.rows('sam_execution_progress', roots['sam_queue_root'] + '/progress/' + child, r'[0-9]{6,}\.json')
        self.capture_metadata()
        self.fixed_families()
        for selector in self.context['retained_metadata_files']:
            self.read(selector['role'], selector['path'])
        if self.context['progression_config'] is not None:
            self.read('progression_config', self.context['progression_config'])

    def capture_metadata(self):
        root = self.context['roots']['pubsub_root']
        for bucket in self.names('capture_buckets', root, ID):
            scenes = root+'/'+bucket+'/scenes'
            for scene in self.names('capture_scenes', scenes, ID):
                captures = scenes+'/'+scene+'/captures'
                for capture in self.names('capture_groups', captures, ID):
                    base = captures+'/'+capture+'/pipeline/website_scene_preparation'
                    for role, leaf in [('website_handoffs', 'handoff.json'),
                                       ('website_preparations', 'preparation.json'),
                                       ('website_preparations', 'development_test/preparation.json'),
                                       ('website_runtime_inputs', 'native/runtime_inputs.json'),
                                       ('website_runtime_inputs', 'development_test/runtime_inputs.json'),
                                       ('website_task_contexts', 'task_context.json'),
                                       ('website_task_contexts', 'development_test/task_context.json')]:
                        self.read(role, base+'/'+leaf)

    def fixed_families(self):
        roots = self.context['roots']
        base = roots['configuration_progression_root']+'/scene-configuration-activations'
        for preparation in self.names('configuration_groups', base, '(?:'+STEM+'|'+ID+')'):
            for role, name in [('configuration_progressions', 'activation_progression.json'),
                               ('launch_progressions', 'launch_progression.json')]:
                self.read(role, base+'/'+preparation+'/'+name)
        for parent in self.names('sam_receipt_parents', roots['sam_execution_root'], HEX):
            base = roots['sam_execution_root']+'/'+parent
            for child in self.names('sam_receipt_children', base, 'sam31-'+HEX):
                self.read('sam_execution_receipts', base+'/'+child+'/phase_execution_receipt.v1.json')
        for launch in self.names('launch_groups', roots['launch_execution_root'], ID):
            self.launch(roots['launch_execution_root']+'/'+launch)
        owner = roots['terminal_result_root']+'/'+self.intent_id
        self.terminal(owner)
        for run in self.names('terminal_run_groups', owner+'/runs', HEX):
            self.terminal(owner+'/runs/'+run)
        for canary in self.names('canary_groups', roots['policy_canary_root'], ID):
            # A published pointer is an ID-shaped regular file alongside the
            # real canary directories, not an evidence-workspace directory.
            if canary.endswith('.offloaded.v1.json'):
                info = self.reader.stat(roots['policy_canary_root']+'/'+canary)
                if stat.S_ISREG(info.st_mode):
                    continue
            self.canary(roots['policy_canary_root']+'/'+canary)
        for pointer in self.names('canary_offload_pointers', roots['policy_canary_root'], ID+r'\.offloaded\.v1\.json'):
            absolute = roots['policy_canary_root']+'/'+pointer
            if not stat.S_ISDIR(self.reader.stat(absolute).st_mode):
                self.read('canary_offload_pointers', absolute)
        for activation in self.names('native_owner_groups', roots['activation_output_root'], ID):
            self.read('native_owner_records', roots['activation_output_root']+'/'+activation+'/scene_owner_attempt.json')
        for compilation in self.names('adapter_groups', roots['compilation_output_root'], '(?:'+STEM+'|'+ID+')'):
            self.read('compilation_adapter_results', roots['compilation_output_root']+'/'+compilation+
                      '/native-arena-adapter/task_evaluation_native_arena_adapter_result.v1.json')

    def launch(self, directory):
        for role, name in [('launch_profiles', 'launch_profile.json'), ('launch_requests', 'launch_request.json'),
                           ('launch_receipts', 'launch_receipt.json')]:
            self.read(role, directory+'/'+name)

    def terminal(self, directory):
        self.launch(directory)
        for name in ('terminal_index_state.json', 'nonexecution_terminal_state.json'):
            self.read('terminal_states', directory+'/'+name)
        for role, name in [('canary_dispatches', 'dispatch_receipt.json'),
                           ('canary_dispatches', 'policy_canary_nonexecution.json'),
                           ('canary_projections', 'policy_canary_result_projection.json'),
                           ('canary_syncs', 'policy_canary_webapp_sync.json'),
                           ('provider_zero_receipts', 'provider_zero_closure.json'),
                           ('terminal_publications', 'terminal_result_publication.json')]:
            self.read(role, directory+'/'+name)

    def canary(self, directory):
        for role, name in [('canary_dispatches', 'dispatch_receipt.json'),
                           ('canary_dispatches', 'preprovider_blocked.json'),
                           ('canary_dispatches', 'no_provider_allocation_blocked.json'),
                           ('canary_projections', 'artifacts/result_delivery/policy_canary_result_projection.json'),
                           ('canary_syncs', 'artifacts/result_delivery/policy_canary_webapp_sync.json'),
                           ('provider_zero_receipts', 'post_teardown_global_provider_zero.json'),
                           ('allocator_results', 'allocator_result.json')]:
            self.read(role, directory+'/'+name)

    def decode(self):
        # ALL lexical and actual decoded bounds precede proof hashes or joins.
        for _, _, raw in _work_items(self.raw, self.budget):
            self.budget.preflight(raw.decode('utf-8'))
        decoded = []
        for role, path, raw in _work_items(self.raw, self.budget):
            value = _work_parse(self.budget, json.loads, raw.decode('utf-8'), object_pairs_hook=retained._pairs,
                                parse_int=retained._numeric, parse_float=retained._numeric,
                                parse_constant=lambda _: require(False, 'metadata_json_invalid'))
            require(isinstance(value, dict), 'metadata_json_invalid')
            self.budget.charge('facts')
            decoded.append({'role': role, 'path': path, 'raw': raw, 'value': value})
        for row in _work_items(decoded, self.budget):
            self.budget.tick()
            row['sha256'] = 'sha256:' + hashlib.sha256(row['raw']).hexdigest()
            self.budget.tick()
        return decoded


def select(decoded, context, intent_id, budget):
    """Exact provided selectors only; absent edges remain downstream obligations."""
    d = native.prior.downstream
    seed = {role: [] for role in d.seed_module._ROLES}
    seed.update(intent=None, projection=None)
    downstream = {role: [] for role in d.ROLES}
    source = {role: [] for role in native.prior.ROLES}
    bridge = {role: [] for role in native.ROLES}
    raw_index, canonical, by_path, parent_modes, activation_modes = {}, {}, {}, {}, {}
    explicit = {row['path'] for row in _work_items(context.get('retained_metadata_files', []), budget)}
    sam_hints, linked_parents, primary_parents = set(), set(), set()
    primary_queue = context['roots']['preparation_queue_root']
    alternate_queues = set()
    for route in _work_items(context.get('parent_routes', []), budget):
        if route['queue_root'] != primary_queue:
            budget.charge('facts')
            alternate_queues.add(route['queue_root'])
    for selector in _work_items(context.get('retained_metadata_files', []), budget):
        if selector['role'] == 'sam_parent_envelopes':
            budget.charge('facts')
            sam_hints.add(selector['path'])
    selected, frontier, protected = set(), [], []
    owner = context['roots']['intent_root'] + '/' + intent_id
    for index, row in enumerate(_work_items(decoded, budget)):
        budget.charge('facts', 3)
        if row['role']=='configured_revisions' and not row['path'].endswith('.json'):
            budget.tick()
            revision=row['value']
            require(revision.get('schema_version')=='task_evaluation_configured_scene_revision.v1'
                and revision.get('revision_digest')==native.c.canonical_digest(
                    revision,digest_field='revision_digest'), 'configured_revision_projection_invalid')
        raw_index.setdefault((row['path'], row['sha256'], len(row['raw'])), []).append(index)
        by_path.setdefault(row['path'], []).append(index)
        for key, value in _work_items(row['value'].items() if supported(row) else (), budget):
            if key.endswith('_digest') and isinstance(value, str) and re.fullmatch('sha256:' + HEX, value):
                budget.charge('facts')
                canonical.setdefault(value, []).append(index)
        if (row['role'] == 'preparation_links' and supported(row)
                and row['path'].startswith(owner+'/preparations/')
                and isinstance(row['value'].get('request_digest'), str)):
            budget.charge('facts')
            linked_parents.add(row['value']['request_digest'])
        if row['role'] == 'parent_envelopes' and row['value'].get('schema_version') == 'task_evaluation_launch_preparation_envelope.v1':
            request = row['value'].get('request')
            if isinstance(request, dict):
                budget.charge('facts')
                parent_modes.setdefault((str(PurePosixPath(row['path']).parent.parent), PurePosixPath(row['path']).name), set()).add(request.get('run_mode'))
                if (str(PurePosixPath(row['path']).parent.parent) == primary_queue
                        and request.get('run_mode') == 'scene_configuration'):
                    budget.charge('facts')
                    primary_parents.add((PurePosixPath(row['path']).name,
                        row['sha256'], len(row['raw']), row['value'].get('request_digest')))
        if row['role'] == 'activation_envelopes' and supported(row):
            request = row['value'].get('request')
            if isinstance(request, dict):
                budget.charge('facts')
                activation_modes.setdefault((str(PurePosixPath(row['path']).parent.parent),
                    PurePosixPath(row['path']).name), set()).add(request.get('lane'))
        if row['path'].startswith(owner + '/') or row['path'] in explicit:
            budget.charge('facts')
            frontier.append(index)
    # Only explicit initial owner records or exact source selectors establish
    # reachability. All other acquisition observations stay protected raw evidence.
    while frontier:
        budget.charge('values')
        index = frontier.pop()
        if index in selected:
            continue
        budget.charge('facts')
        selected.add(index)
        row = decoded[index]
        value = row['value']
        if not supported(row):
            continue
        stack = [value]
        while stack:
            budget.charge('values')
            node = stack.pop()
            if isinstance(node, dict):
                if {'path', 'sha256', 'size_bytes'} <= node.keys():
                    require(type(node['size_bytes']) is int and node['size_bytes'] >= 0, 'reference_invalid')
                    for found in _work_items(raw_index.get((node['path'], node['sha256'], node['size_bytes']), ()), budget):
                        budget.charge('facts')
                        frontier.append(found)
                for key, child in _work_items(node.items(), budget):
                    if key in {'request_digest', 'preparation_request_digest', 'activation_request_digest',
                               'envelope_digest', 'preparation_result_digest', 'profile_digest',
                               'intent_digest', 'plan_digest', 'adoption_digest', 'launch_request_digest', 'launch_receipt_digest'}:
                        for found in _work_items(canonical.get(child, ()) if isinstance(child, str) else (), budget):
                            budget.charge('facts')
                            frontier.append(found)
                    if key == 'result_filename' and isinstance(child, str) and row['role'] == 'preparation_links':
                        # This producer's link names the configured canonical
                        # preparation queue, never an arbitrary equal basename.
                        exact = context['roots']['preparation_queue_root'] + '/results/' + child
                        for found in _work_items(by_path.get(exact, ()), budget):
                            budget.charge('facts')
                            frontier.append(found)
                    if key == 'request_digest' and row['role'] in {'parent_envelopes', 'activation_envelopes'}:
                        exact = str(PurePosixPath(row['path']).parent.parent) + '/results/' + PurePosixPath(row['path']).name
                        for found in _work_items(by_path.get(exact, ()), budget):
                            budget.charge('facts')
                            frontier.append(found)
                    budget.charge('facts')
                    stack.append(child)
            elif isinstance(node, list):
                for child in _work_items(node, budget):
                    budget.charge('facts')
                    stack.append(child)
    for index, row in enumerate(_work_items(decoded, budget)):
        role = row['role']
        value = row['value']
        if index in selected and not supported(row):
            budget.charge('facts')
            protected.append({'role': role, 'path': row['path'], 'sha256': row['sha256'],
                              'size_bytes': len(row['raw']), 'status': 'kept_unsupported_schema'})
            continue
        if index not in selected:
            budget.charge('facts')
            protected.append({'role': role, 'path': row['path'], 'sha256': row['sha256'],
                              'size_bytes': len(row['raw']), 'status': 'kept_unselected_metadata'})
            continue
        if role in {'parent_envelopes', 'parent_results'}:
            p = PurePosixPath(row['path'])
            modes = parent_modes.get((str(p.parent.parent), p.name), set())
            if (role == 'parent_envelopes' and row['path'] in sam_hints
                    and modes == {'scene_configuration'} and (
                        value.get('request_digest') not in linked_parents or (
                            str(p.parent.parent) in alternate_queues
                            and (p.name, row['sha256'], len(row['raw']), value.get('request_digest')) in primary_parents))):
                # An exact explicit historical SAM role cannot establish an
                # owner-link. Its source-family reader independently checks the
                # real route/name/request, and adoption selects exact raw bytes.
                # A current parent's identical retained copy belongs to its
                # configured SAM route; the primary queue remains the seed.
                role = 'sam_parent_envelopes'
            elif modes == {'scene_configuration'}:
                role = 'preparation_envelopes' if role == 'parent_envelopes' else 'preparation_results'
            elif len(modes) == 1 and modes <= {'episode_evaluation', 'destination_qualification'}:
                role = 'native_preparation_envelopes' if role == 'parent_envelopes' else 'native_preparation_results'
            else:
                budget.charge('facts')
                protected.append({'role': role, 'path': row['path'], 'sha256': row['sha256'],
                                  'size_bytes': len(row['raw']), 'status': 'kept_parent_mode_unproven'})
                continue
        elif role in {'activation_envelopes', 'activation_results'}:
            request = value.get('request')
            lane = request.get('lane') if isinstance(request, dict) and role == 'activation_envelopes' else value.get('lane')
            p = PurePosixPath(row['path'])
            modes = activation_modes.get((str(p.parent.parent), p.name), set())
            if lane is None and len(modes) == 1:
                lane = next(iter(modes))
            if lane == 'task_evaluation_scene_configuration':
                pass
            elif lane in PHASES:
                role = 'native_activation_envelopes' if role == 'activation_envelopes' else 'native_activation_results'
            else:
                budget.charge('facts')
                protected.append({'role': role, 'path': row['path'], 'sha256': row['sha256'],
                                  'size_bytes': len(row['raw']), 'status': 'kept_activation_lane_unproven'})
                continue
        pair = row['path'], row['raw']
        budget.charge('facts')
        if role in {'intent', 'projection'}:
            require(seed[role] is None, 'owner_record_ambiguous')
            seed[role] = pair
        elif role in seed:
            seed[role].append(pair)
        elif role in downstream:
            downstream[role].append(pair)
        elif role in source:
            source[role].append(pair)
        elif role in bridge:
            bridge[role].append(pair)
    return seed, downstream, source, bridge, protected
