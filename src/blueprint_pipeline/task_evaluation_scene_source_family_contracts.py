"""Wrapper-owned bounds and supplied-byte indexes for ADP-009D/day-28.

No reader, file access or authority is conferred by these lexical observations.
Existing downstream helpers/APIs are reused without changing their contracts.
"""
from __future__ import annotations

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
    'sam_parent_envelopes': {'task_evaluation_launch_preparation_envelope.v1'},
    'preparation_envelopes': {'task_evaluation_launch_preparation_envelope.v1'},
    'sam_execution_receipts': {'task_evaluation_sam31_phase_execution_receipt.v1', 'task_evaluation_sam31_phase_replay_receipt.v1'},
    'sam_host_evidence': {'standard_splat_conversion_receipt.v1', 'public_scene_host_input_installation_receipt.v1',
                          'public_scene_source_preparation.v1'},
}


class SceneSourceFamilyInventoryError(ValueError):
    """Fixed bounded refusal without retained private document contents."""


def require(condition, code):
    if not condition:
        raise SceneSourceFamilyInventoryError('scene_source_family_' + code)


def decode(groups, limits):
    """Check all input bytes, then all actual JSON nodes, before any raw hashes."""
    count, total, identities, paths = 0, 0, set(), {}
    immutable = {'website_registrations', 'website_bindings'}
    for role, rows in groups.items():
        count += len(rows)
        require(count <= limits['MAX_RECORDS'], 'records_limit')
        for item in rows:
            require(isinstance(item, (list, tuple)) and len(item) == 2 and type(item[1]) is bytes,
                    'record_invalid')
            name, raw = item
            path(name)
            require(len(raw) <= limits['MAX_RECORD_BYTES'] and (raw or role == 'opaque_evidence'), 'record_invalid')
            total += len(raw)
            require(total <= limits['MAX_TOTAL_BYTES'], 'bytes_limit')
            require((name, raw) not in identities, 'record_duplicate')
            require(name not in paths or (paths[name] == role and role not in immutable), 'path_role_ambiguous')
            identities.add((name, raw))
            paths[name] = role
    # The reviewed decoder bounds ALL JSON before its first hash; opaque records
    # have no decoded graph and were already counted in the shared byte budget.
    decoded = c.decode({r: rows for r, rows in groups.items() if r != 'opaque_evidence'}, limits)
    decoded['opaque_evidence'] = []
    for name, raw in groups['opaque_evidence']:
        decoded['opaque_evidence'].append((None, {'role': 'opaque_evidence', 'path': name,
            'size_bytes': len(raw), 'sha256': raw_digest(raw), 'seal_field': None, 'seal_digest': None}))
    decoded['opaque_evidence'].sort(key=lambda r: (r[1]['path'], r[1]['sha256']))
    return decoded


class Context(c.Context):
    """One invocation's exact indexes, reference and output occurrence budget."""
    def __init__(self, decoded, roots, limits, intent_id, source_roles):
        super().__init__(decoded, roots, limits, intent_id)
        self.source_roles = source_roles
        self.record_index = {(p['path'], p['sha256'], p['size_bytes']): row
                             for rows in decoded.values() for row in rows for p in [row[1]]}
        self.uri_identities = {}
        self.selectors = {}
        self.observations = {}

    def selected(self, reference, source, roles=None, *, positive=True):
        require(isinstance(reference, dict) and {'path', 'sha256', 'size_bytes'} <= set(reference), 'reference_invalid')
        require(type(reference['size_bytes']) is int and reference['size_bytes'] >= (1 if positive else 0),
                'reference_invalid')
        self.raw_ref(reference, source)
        row = self.record_index.get(tuple(reference[k] for k in ('path', 'sha256', 'size_bytes')))
        if row and roles is not None:
            require(row[1]['role'] in roles, 'reference_role_invalid')
            known = SUPPORTED_SCHEMAS.get(row[1]['role'])
            if known and row[0].get('schema_version') not in known:
                self.missing(row[1]['role'], 'unsupported_retained_schema', [row[1]])
                return None
        return row

    def known(self, role, schema, seal_field=None):
        rows = []
        for row in self.decoded[role]:
            if row[0].get('schema_version') != schema:
                self.missing(role, 'unsupported_retained_schema', [row[1]], selector={'schema_version': None})
                continue
            if seal_field:
                c.seal(row, seal_field)
            rows.append(row)
        return rows

    def selector(self, row, field):
        key = (row[1]['role'], row[0][field])
        self.selectors.setdefault(key, []).append(row)

    def references(self):
        super().references()
        # Generic remote records can repeat a URI but cannot promise two contents.
        for row in self.remote:
            key = (row['digest'], row['size_bytes'])
            require(row['uri'] not in self.uri_identities or self.uri_identities[row['uri']] == key,
                    'remote_identity_conflict')
            self.uri_identities[row['uri']] = key

    def nested(self, value, enclosing, pointer, seal_field):
        proof = dict(enclosing, json_pointer=pointer, seal_field=None, seal_digest=None)
        row = value, proof
        c.seal(row, seal_field)
        return row

    def charge_child(self, result):
        # Count row occurrences recursively; nested children do not get an
        # independent allowance. Measure BEFORE constructing wrapper collections.
        stack = [result]
        while stack:
            value = stack.pop()
            if isinstance(value, dict):
                stack.extend(value.values())
            elif isinstance(value, list):
                self.budget['rows'] += len(value)
                require(self.budget['rows'] <= self.limits['MAX_ROWS'], 'rows_limit')
                stack.extend(value)
        self.budget['bytes'] += c.bounded_size(result, self.limits['MAX_OUTPUT_BYTES'] - self.budget['bytes'])
        for key in ('raw_reference_obligations', 'remote_reference_obligations', 'structural_join_obligations'):
            for _ in result[key]:
                self.consume()


def relative(value):
    require(isinstance(value, str) and len(value) <= 4096 and not value.startswith('/'), 'relative_path_invalid')
    # Apply the same canonical path grammar after the BEFORE-encoding guard.
    require(path('/' + value) == '/' + value, 'relative_path_invalid')
    return value


def observation(row, **fields):
    return c.observation(row, **fields)
