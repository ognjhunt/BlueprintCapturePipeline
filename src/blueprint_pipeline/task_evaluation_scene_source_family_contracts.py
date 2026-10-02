"""Wrapper-owned bounds and supplied-byte indexes for ADP-009D/day-28.

No reader, file access or authority is conferred by these lexical observations.
Existing downstream helpers/APIs are reused without changing their contracts.
"""
from __future__ import annotations

from .task_evaluation_scene_lineage_budget import _work_collect, _work_sort, _work, _work_items, _work_kwargs

from . import task_evaluation_scene_downstream_contracts as c

raw_digest = c.raw_digest
canonical_digest = c.canonical_digest
cross_runtime_canonical_digest = c.cross_runtime_canonical_digest
matches, path, child, under = c.matches, c.path, c.child, c.under
DIGEST, COMMIT, ID = c.DIGEST, c.COMMIT, c.ACTIVATION_ID
SUPPORTED_SCHEMAS = {
    'website_registrations': {'website_scene_source_registration.v1'},
    'website_bindings': {'website_scene_source_binding.v1'},
    'website_preparations': {'website_scene_preparation.v1'},
    'website_runtime_inputs': {'website_scene_runtime_inputs.v1'},
    'website_task_contexts': {'website_site_task_context.v1'},
    'sam_plans': {'task_evaluation_sam31_preparation_plan.v1'},
    'sam_profiles': {'task_evaluation_sam31_preparation_profile.v1'},
    'sam_recipes': {'task_evaluation_scene_construction_recipe.v1'},
    'sam_stage_configurations': {'observed_appearance_object_removal_configuration.v1'},
    'sam_host_tasks': {'task_evaluation_minimal_task_request.v1'},
    'sam_jobs': {'task_evaluation_sam31_preparation_execution_job.v1'},
    'sam_results': {'task_evaluation_sam31_preparation_execution_result.v1'},
    'sam_adoptions': {'task_evaluation_sam31_completed_prefix_adoption.v1'},
    'sam_prefix_selections': {'task_evaluation_sam31_prefix_selection.v1'},
    'sam_artifact_metadata': {'public_scene_sam31_task_input_packet.v1', 'semantic_sam31_source_track_run_request.v1',
                              'public_scene_interiorgs_edit_input_request.v2'},
    'sam_parent_envelopes': {'task_evaluation_launch_preparation_envelope.v1'},
    'sam_parent_results': {'task_evaluation_launch_preparation_result.v1'},
    'preparation_envelopes': {'task_evaluation_launch_preparation_envelope.v1'},
    'sam_execution_receipts': {'task_evaluation_sam31_phase_execution_receipt.v1', 'task_evaluation_sam31_phase_replay_receipt.v1'},
    'sam_host_evidence': {'standard_splat_conversion_receipt.v1', 'public_scene_host_input_installation_receipt.v1',
                          'public_scene_source_preparation.v1'},
}


class SceneSourceFamilyInventoryError(ValueError):
    """Fixed bounded refusal without retained private document contents."""


def require(condition, code, *, work_budget=None):
    if work_budget is not None:
        _work(work_budget)
    if not condition:
        raise SceneSourceFamilyInventoryError('scene_source_family_' + code)


def decode(groups, limits, *, work_budget=None):
    """Check all input bytes, then all actual JSON nodes, before any raw hashes."""
    if work_budget is not None:
        _work(work_budget)
    count, total, identities, paths = 0, 0, set(), {}
    immutable = {'website_registrations', 'website_bindings'}
    for role, rows in (_work_items(groups.items(), work_budget) if work_budget is not None else groups.items()):
        count += len(rows)
        require(count <= limits['MAX_RECORDS'], 'records_limit', **_work_kwargs(work_budget))
        for item in (_work_items(rows, work_budget) if work_budget is not None else rows):
            require(isinstance(item, (list, tuple)) and len(item) == 2 and type(item[1]) is bytes,
                    'record_invalid', **_work_kwargs(work_budget))
            name, raw = item
            path(name, **_work_kwargs(work_budget))
            require(len(raw) <= limits['MAX_RECORD_BYTES'] and (raw or role == 'opaque_evidence'), 'record_invalid', **_work_kwargs(work_budget))
            total += len(raw)
            require(total <= limits['MAX_TOTAL_BYTES'], 'bytes_limit', **_work_kwargs(work_budget))
            require((name, raw) not in identities, 'record_duplicate', **_work_kwargs(work_budget))
            require(name not in paths or (paths[name] == role and role not in immutable), 'path_role_ambiguous', **_work_kwargs(work_budget))
            identities.add((name, raw))
            paths[name] = role
    # The reviewed decoder bounds ALL JSON before its first hash; opaque records
    # have no decoded graph and were already counted in the shared byte budget.
    decoded = c.decode({r: rows for r, rows in (_work_items(groups.items(), work_budget) if work_budget is not None else groups.items()) if r != 'opaque_evidence'}, limits, **_work_kwargs(work_budget))
    decoded['opaque_evidence'] = []
    for name, raw in (_work_items(groups['opaque_evidence'], work_budget) if work_budget is not None else groups['opaque_evidence']):
        decoded['opaque_evidence'].append((None, {'role': 'opaque_evidence', 'path': name,
            'size_bytes': len(raw), 'sha256': raw_digest(raw, **_work_kwargs(work_budget)), 'seal_field': None, 'seal_digest': None}))
    (_work_sort(work_budget, decoded['opaque_evidence'], key=lambda r: (r[1]['path'], r[1]['sha256'])) if work_budget is not None else decoded['opaque_evidence'].sort(key=lambda r: (r[1]['path'], r[1]['sha256'])))
    return decoded


class Context(c.Context):
    """One invocation's exact indexes, reference and output occurrence budget."""
    def __init__(self, decoded, roots, limits, intent_id, source_roles, *, emission_budget=None, work_budget=None):
        if work_budget is None:
            work_budget = getattr(emission_budget, "work_budget", None)
        self.work_budget = work_budget
        if work_budget is not None:
            _work(work_budget)
        super().__init__(decoded, roots, limits, intent_id, emission_budget=emission_budget, **_work_kwargs(work_budget))
        self.source_roles = source_roles
        self.record_index = {(p['path'], p['sha256'], p['size_bytes']): row
                             for rows in (_work_items(decoded.values(), work_budget) if work_budget is not None else decoded.values()) for row in (_work_items(rows, work_budget) if work_budget is not None else rows) for p in (_work_items([row[1]], work_budget) if work_budget is not None else [row[1]])}
        self.uri_identities = {}
        self.selectors = {}
        self.observations = {}

    def selected(self, reference, source, roles=None, *, positive=True, work_budget=None):
        if work_budget is None:
            work_budget = getattr(self, "work_budget", None)
        if work_budget is not None:
            _work(work_budget)
        require(isinstance(reference, dict) and {'path', 'sha256', 'size_bytes'} <= (_work_collect(work_budget, set, reference) if work_budget is not None else set(reference)), 'reference_invalid', **_work_kwargs(work_budget))
        require(type(reference['size_bytes']) is int and reference['size_bytes'] >= (1 if positive else 0),
                'reference_invalid', **_work_kwargs(work_budget))
        self.raw_ref(reference, source)
        row = self.record_index.get(tuple(reference[k] for k in (_work_items(('path', 'sha256', 'size_bytes'), work_budget) if work_budget is not None else ('path', 'sha256', 'size_bytes'))))
        if row and roles is not None:
            require(row[1]['role'] in roles, 'reference_role_invalid', **_work_kwargs(work_budget))
            known = SUPPORTED_SCHEMAS.get(row[1]['role'])
            if known and row[0].get('schema_version') not in known:
                self.missing(row[1]['role'], 'unsupported_retained_schema', [row[1]])
                return None
        return row

    def known(self, role, schema, seal_field=None, *, work_budget=None):
        if work_budget is None:
            work_budget = getattr(self, "work_budget", None)
        if work_budget is not None:
            _work(work_budget)
        rows = []
        for row in (_work_items(self.decoded[role], work_budget) if work_budget is not None else self.decoded[role]):
            if row[0].get('schema_version') != schema:
                self.missing(role, 'unsupported_retained_schema', [row[1]], selector={'schema_version': None})
                continue
            if seal_field:
                c.seal(row, seal_field, **_work_kwargs(work_budget))
            rows.append(row)
        return rows

    def selector(self, row, field, *, work_budget=None):
        if work_budget is None:
            work_budget = getattr(self, "work_budget", None)
        if work_budget is not None:
            _work(work_budget)
        key = (row[1]['role'], row[0][field])
        self.selectors.setdefault(key, []).append(row)

    def references(self, *, roles=None, work_budget=None):
        if work_budget is None:
            work_budget = getattr(self, "work_budget", None)
        if work_budget is not None:
            _work(work_budget)
        super().references(roles=roles)
        # Generic remote records can repeat a URI but cannot promise two contents.
        for row in (_work_items(self.remote, work_budget) if work_budget is not None else self.remote):
            key = (row['digest'], row['size_bytes'])
            require(row['uri'] not in self.uri_identities or self.uri_identities[row['uri']] == key,
                    'remote_identity_conflict', **_work_kwargs(work_budget))
            self.uri_identities[row['uri']] = key

    def predecessor_remote_identities(self, rows, *, raw_rows=(), work_budget=None):
        """Preserve cross-layer URI and digest-size refusals in scoped scans."""
        if work_budget is None:
            work_budget = getattr(self, 'work_budget', None)
        for row in (_work_items(raw_rows, work_budget) if work_budget is not None else raw_rows):
            if work_budget is not None:
                work_budget.charge('facts')
            digest, size = row['sha256'], row['size_bytes']
            require(digest not in self.sizes or self.sizes[digest] == size,
                    'content_size_conflict', **_work_kwargs(work_budget))
            self.sizes[digest] = size
        for row in (_work_items(rows, work_budget) if work_budget is not None else rows):
            if work_budget is not None:
                work_budget.charge('facts')
            key = (row['digest'], row['size_bytes'])
            require(key[0] not in self.sizes or self.sizes[key[0]] == key[1],
                    'content_size_conflict', **_work_kwargs(work_budget))
            self.sizes[key[0]] = key[1]
            require(row['uri'] not in self.uri_identities or self.uri_identities[row['uri']] == key,
                    'remote_identity_conflict', **_work_kwargs(work_budget))
            self.uri_identities[row['uri']] = key

    def nested(self, value, enclosing, pointer, seal_field, *, work_budget=None):
        if work_budget is None:
            work_budget = getattr(self, "work_budget", None)
        if work_budget is not None:
            _work(work_budget)
        proof = dict(enclosing, json_pointer=pointer, seal_field=None, seal_digest=None)
        row = value, proof
        c.seal(row, seal_field, **_work_kwargs(work_budget))
        return row

    def charge_child(self, result, *, work_budget=None):
        # Count row occurrences recursively; nested children do not get an
        # independent allowance. Measure BEFORE constructing wrapper collections.
        if work_budget is None:
            work_budget = getattr(self, "work_budget", None)
        if work_budget is not None:
            _work(work_budget)
        stack = [result]
        while stack:
            if work_budget is not None:
                work_budget.charge("values")
            value = stack.pop()
            if isinstance(value, dict):
                stack.extend(_work_items(value.values(), work_budget) if work_budget is not None else value.values())
            elif isinstance(value, list):
                self.budget['rows'] += len(value)
                require(self.budget['rows'] <= self.limits['MAX_ROWS'], 'rows_limit', **_work_kwargs(work_budget))
                stack.extend(_work_items(value, work_budget) if work_budget is not None else value)
        self.budget['bytes'] += c.bounded_size(result, self.limits['MAX_OUTPUT_BYTES'] - self.budget['bytes'], **_work_kwargs(work_budget))
        for key in (_work_items(('raw_reference_obligations', 'remote_reference_obligations', 'structural_join_obligations'), work_budget) if work_budget is not None else ('raw_reference_obligations', 'remote_reference_obligations', 'structural_join_obligations')):
            for _ in (_work_items(result[key], work_budget) if work_budget is not None else result[key]):
                self.consume()


def relative(value, *, work_budget=None):
    if work_budget is not None:
        _work(work_budget)
    require(isinstance(value, str) and len(value) <= 4096 and not value.startswith('/'), 'relative_path_invalid', **_work_kwargs(work_budget))
    # Apply the same canonical path grammar after the BEFORE-encoding guard.
    require(path('/' + value, **_work_kwargs(work_budget)) == '/' + value, 'relative_path_invalid', **_work_kwargs(work_budget))
    return value


def observation(row, work_budget=None, **fields):
    if work_budget is not None:
        _work(work_budget)
    return c.observation(row, **fields, **_work_kwargs(work_budget))
