"""Finite metadata discovery then supplied-record selection; no payload reads."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import PurePosixPath

from . import task_evaluation_scene_compilation_native_owner_inventory as native
from . import task_evaluation_scene_preparation_lineage as retained
from .task_evaluation_scene_lifecycle_acquisition import AcquisitionError, require
from .task_evaluation_scene_lineage_budget import _work_items, _work_parse

HEX = r'[0-9a-f]{64}'
ID = r'[A-Za-z0-9][A-Za-z0-9._-]{0,191}'
STEM = ID + '-' + HEX
STATES = ('pending', 'processing', 'awaiting_source_preparation', 'awaiting_capacity', 'materialized', 'completed', 'blocked')
ALL_ROLES = native.prior.downstream.seed_module._ROLES | native.prior.downstream.ROLES | native.prior.ROLES | native.ROLES
SELECTOR_ROLES = ALL_ROLES | {'revocations', 'extensions', 'progression_config'}


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
            elif name.endswith('.json'):
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
        for selector in self.context['retained_metadata_files']:
            self.read(selector['role'], selector['path'])
        if self.context['progression_config'] is not None:
            self.read('progression_config', self.context['progression_config'])

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
    raw_index, canonical, by_path, parent_modes = {}, {}, {}, {}
    explicit = {row['path'] for row in _work_items(context.get('retained_metadata_files', []), budget)}
    selected, frontier, protected = set(), [], []
    owner = context['roots']['intent_root'] + '/' + intent_id
    for index, row in enumerate(_work_items(decoded, budget)):
        budget.charge('facts', 3)
        raw_index.setdefault((row['path'], row['sha256'], len(row['raw'])), []).append(index)
        by_path.setdefault(row['path'], []).append(index)
        for key, value in _work_items(row['value'].items(), budget):
            if key.endswith('_digest') and isinstance(value, str) and re.fullmatch('sha256:' + HEX, value):
                budget.charge('facts')
                canonical.setdefault(value, []).append(index)
        if row['role'] == 'parent_envelopes' and row['value'].get('schema_version') == 'task_evaluation_launch_preparation_envelope.v1':
            request = row['value'].get('request')
            if isinstance(request, dict):
                budget.charge('facts')
                parent_modes.setdefault((str(PurePosixPath(row['path']).parent.parent), PurePosixPath(row['path']).name), set()).add(request.get('run_mode'))
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
                    if key in {'request_digest', 'envelope_digest', 'preparation_result_digest', 'profile_digest',
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
        if index not in selected:
            budget.charge('facts')
            protected.append({'role': role, 'path': row['path'], 'sha256': row['sha256'],
                              'size_bytes': len(row['raw']), 'status': 'kept_unselected_metadata'})
            continue
        if role in {'parent_envelopes', 'parent_results'}:
            p = PurePosixPath(row['path'])
            modes = parent_modes.get((str(p.parent.parent), p.name), set())
            if modes == {'scene_configuration'}:
                role = 'preparation_envelopes' if role == 'parent_envelopes' else 'preparation_results'
            elif len(modes) == 1 and modes <= {'episode_evaluation', 'destination_qualification'}:
                role = 'native_preparation_envelopes' if role == 'parent_envelopes' else 'native_preparation_results'
            else:
                budget.charge('facts')
                protected.append({'role': role, 'path': row['path'], 'sha256': row['sha256'],
                                  'size_bytes': len(row['raw']), 'status': 'kept_parent_mode_unproven'})
                continue
        elif role == 'activation_envelopes' and value.get('request', {}).get('lane') != 'task_evaluation_scene_configuration':
            role = 'native_activation_envelopes'
        elif role == 'activation_results' and value.get('lane') not in (None, 'task_evaluation_scene_configuration'):
            role = 'native_activation_results'
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
    require(seed['intent'] is not None, 'intent_unavailable')
    return seed, downstream, source, bridge, protected
