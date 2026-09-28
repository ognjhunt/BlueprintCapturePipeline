"""Pure bounded byte/provenance primitives for ADP-009D retained lineage."""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import PurePosixPath

from . import task_evaluation_scene_preparation_lineage as retained
from .decision_evidence_contracts import canonical_digest, cross_runtime_canonical_digest

DIGEST = re.compile(r'sha256:[0-9a-f]{64}\Z')
COMMIT = re.compile(r'[0-9a-f]{40}\Z')
ID = retained._ID
ACTIVATION_ID = re.compile(r'[A-Za-z0-9][A-Za-z0-9._-]{0,191}\Z')
LAUNCH_ID = ACTIVATION_ID  # Typed launch/activation schema IDs permit 192 characters.


class SceneDownstreamInventoryError(ValueError):
    """A bounded typed refusal, never supplied private text."""


def require(condition, code):
    if not condition:
        raise SceneDownstreamInventoryError('scene_downstream_' + code)


def matches(value, pattern=DIGEST):
    return isinstance(value, str) and pattern.fullmatch(value) is not None


def path(value):
    require(isinstance(value, str) and len(value) <= retained.MAX_PATH_BYTES, 'path_invalid')
    return retained._path(value)


def child(root, *parts):
    return retained._child(root, *parts)


def under(value, root):
    return PurePosixPath(path(value)).is_relative_to(PurePosixPath(root)) and value != root


def encoded(value):
    return retained._encoded(value)


def bounded_size(value, limit):
    """Exact compact UTF8 JSON length without allocating the encoded document."""
    size, stack = 0, [value]
    while stack:
        item = stack.pop()
        if isinstance(item, dict):
            size += 2 + max(0, len(item) - 1) + len(item)
            stack.extend(item.values())
            stack.extend(item.keys())
        elif isinstance(item, list):
            size += 2 + max(0, len(item) - 1)
            stack.extend(item)
        elif isinstance(item, str):
            require(len(item) <= limit - size, 'output_limit')
            size += 2
            for char in item:
                code = ord(char)
                require(not 0xD800 <= code <= 0xDFFF, 'json_invalid')
                size += (2 if char in '"\\\b\f\n\r\t' else 6 if code < 32 else
                         1 if code < 128 else 2 if code < 2048 else 3 if code < 65536 else 4)
                require(size <= limit, 'output_limit')
        elif item is None:
            size += 4
        elif type(item) is bool:
            size += 4 if item else 5
        elif type(item) in (int, float):
            require(math.isfinite(item), 'json_invalid')
            size += len(json.dumps(item, allow_nan=False))
        else:
            require(False, 'json_invalid')
        require(size <= limit, 'output_limit')
    return size


def raw_digest(raw):
    return 'sha256:' + hashlib.sha256(raw).hexdigest()


class OutputRows(list):
    """Charge actual occurrence rows/bytes before storing, before dedup/encoding."""
    def __init__(self, budget, limits, rows=(), *, emission_budget=None, reference=False):
        super().__init__()
        self.budget, self.limits = budget, limits
        self.emission_budget, self.reference = emission_budget, reference
        for row in rows:
            self.append(row)

    def append(self, row):
        require(self.budget['rows'] < self.limits['MAX_ROWS'], 'rows_limit')
        length = bounded_size(row, self.limits['MAX_OUTPUT_BYTES'] - self.budget['bytes'])
        if self.emission_budget is not None:
            self.emission_budget.reserve_row(row, reference=self.reference)
        self.budget['bytes'] += length
        self.budget['rows'] += 1
        super().append(row)

    def extend(self, values):
        for row in values:
            self.append(row)


def seal(row, field, *, cross=False):
    value, proof = row
    digest = cross_runtime_canonical_digest if cross else canonical_digest
    require(matches(value.get(field)) and value[field] == digest(value, digest_field=field), 'seal_invalid')
    proof.update(seal_field=field, seal_digest=value[field])


def lexical_preflight(raw, limits, budget):
    """Conservative string-aware token budget before recursive JSON allocation.

    Count keys as well as values. Strict syntax/duplicates/numbers are checked by
    the reviewed parser after this bounded structural pass, never by a regex.
    """
    text = raw.decode('utf-8')
    depth, quoted, escaped, primitive = 0, False, False, False
    for char in text:
        if quoted:
            if escaped:
                escaped = False
            elif char == '\\':
                escaped = True
            elif char == '"':
                quoted = False
            continue
        if char in ' \t\r\n,:]}':
            primitive = False
        if char == '"':
            quoted = True
            budget['tokens'] += 1
        elif char in '{[':
            depth += 1
            budget['tokens'] += 1
            require(depth <= limits['MAX_DEPTH'], 'depth_limit')
            primitive = False
        elif char in '}]':
            depth -= 1
            require(depth >= 0, 'json_invalid')
        elif char not in ' \t\r\n,:' and not primitive:
            budget['tokens'] += 1
            primitive = True
        require(budget['tokens'] <= limits['MAX_NODES'], 'nodes_limit')
    require(not quoted and depth == 0, 'json_invalid')


def decode(groups, limits):
    count = sum(len(rows) for rows in groups.values())
    require(count <= limits['MAX_RECORDS'], 'records_limit')
    total = 0
    for rows in groups.values():
        for pair in rows:
            require(isinstance(pair, (list, tuple)) and len(pair) == 2 and type(pair[1]) is bytes
                    and 0 < len(pair[1]) <= limits['MAX_RECORD_BYTES'], 'record_invalid')
            path(pair[0])
            total += len(pair[1])
            require(total <= limits['MAX_TOTAL_BYTES'], 'bytes_limit')
    budget = {'tokens': 0}
    for rows in groups.values():
        for _, raw in rows:
            lexical_preflight(raw, limits, budget)
    decoded, identities, nodes, raw_pairs = {}, set(), 0, []
    immutable = {'intent', 'events', 'attempts', 'source_snapshots', 'factories', 'source_submissions',
                 'preparation_links', 'preparation_envelopes', 'activation_envelopes', 'launch_requests',
                 'compilation_envelopes'}
    paths = {}
    for role, rows in sorted(groups.items()):
        decoded[role] = []
        for pair in rows:
            value = json.loads(pair[1].decode('utf-8'), object_pairs_hook=retained._pairs,
                               parse_int=retained._numeric, parse_float=retained._numeric,
                               parse_constant=lambda _: require(False, 'json_invalid'))
            require(isinstance(value, dict), 'record_invalid')
            proof = {'role': role, 'path': pair[0], 'size_bytes': len(pair[1]), 'seal_field': None, 'seal_digest': None}
            identity = (pair[0], pair[1])
            require(identity not in identities, 'record_duplicate')
            require(proof['path'] not in paths or (role not in immutable and paths[proof['path']] == role), 'immutable_ambiguous')
            identities.add(identity)
            paths[proof['path']] = role
            stack = [(value, 1)]
            while stack:
                current, depth = stack.pop()
                nodes += 1
                require(nodes <= limits['MAX_NODES'] and depth <= limits['MAX_DEPTH'], 'nodes_limit')
                children = current.values() if isinstance(current, dict) else current if isinstance(current, list) else ()
                stack.extend((item, depth + 1) for item in children)
            bounded_size(value, limits['MAX_TOTAL_BYTES'])  # Surrogate/type checks after decoded bounds, before hashes.
            decoded[role].append((value, proof))
            raw_pairs.append((pair[1], proof))
    for raw, proof in raw_pairs:  # No proof hashing until ALL actual decoded limits pass.
        proof['sha256'] = raw_digest(raw)
    for rows in decoded.values():
        rows.sort(key=lambda r: (r[1]['path'], r[1]['sha256'], r[1]['size_bytes']))
    return decoded


class Context:
    def __init__(self, decoded, roots, limits, intent_id, *, emission_budget=None):
        self.decoded, self.roots, self.limits, self.intent_id = decoded, roots, limits, intent_id
        self.budget = {'rows': 0, 'bytes': 0}
        self.emission_budget = emission_budget
        self.raw = self.rows(p for rows in decoded.values() for _, p in rows)
        self.index = {(p['path'], p['sha256'], p['size_bytes']): p for p in self.raw}
        self.by_path = {role: {} for role in decoded}
        for role, rows in decoded.items():
            for row in rows:
                self.by_path[role].setdefault(row[1]['path'], []).append(row)
        self.obligations, self.remote, self.structural = (self.rows(reference=True) for _ in range(3))
        self.members = self.rows()
        self.count, self.sizes = 0, {}
        for proof in self.raw:
            self.size(proof['sha256'], proof['size_bytes'])

    def rows(self, values=(), *, reference=False):
        return OutputRows(self.budget, self.limits, values, emission_budget=self.emission_budget, reference=reference)

    def provenance(self, values):
        return self.emission_budget.reserve_provenance(values) if self.emission_budget is not None else list(values)

    def seed_budget(self, seed):
        self.budget['rows'] += sum(len(v) for v in seed.values() if isinstance(v, list))
        require(self.budget['rows'] <= self.limits['MAX_ROWS'], 'rows_limit')
        self.budget['bytes'] += bounded_size(seed, self.limits['MAX_OUTPUT_BYTES'] - self.budget['bytes'])
        # Missing child joins and newly discovered retained-result obligations
        # share this API's occurrence budget; supplied reference fields were
        # already charged by the combined raw record walk.
        for _ in seed['preparation_join_obligations'] + seed['request_projection_obligations']:
            self.consume()
        for row in seed['obligations']:
            if row['role'] == 'preparation_result':
                self.consume()

    def consume(self):
        self.count += 1
        require(self.count <= self.limits['MAX_REFERENCES'], 'references_limit')

    def raw_ref(self, ref, proof):
        self.consume()
        require(isinstance(ref, dict) and {'path', 'sha256', 'size_bytes'} <= set(ref), 'reference_invalid')
        path(ref['path'])
        require(matches(ref['sha256']) and type(ref['size_bytes']) is int and ref['size_bytes'] >= 0, 'reference_invalid')
        self.size(ref['sha256'], ref['size_bytes'])
        key = tuple(ref[k] for k in ('path', 'sha256', 'size_bytes'))
        match = self.index.get(key)
        self.obligations.append({**{k: ref[k] for k in ('path', 'sha256', 'size_bytes')},
                                 'status': 'matched_retained_bytes' if match else 'kept_unresolved',
                                 'reason': None if match else 'reference_bytes_unavailable',
                                 'source_provenance': [proof], 'matched_provenance': match})
        return match

    def size(self, digest, size):
        require(digest not in self.sizes or self.sizes[digest] == size, 'content_size_conflict')
        self.sizes[digest] = size

    def references(self):
        for rows in self.decoded.values():
            for value, proof in rows:
                stack = [value]
                while stack:
                    current = stack.pop()
                    if isinstance(current, dict):
                        if 'path' in current and 'sha256' in current:
                            self.raw_ref(current, proof)
                        if 'uri' in current and 'digest' in current and 'size_bytes' in current:
                            self.consume()
                            uri, digest, size = (current[k] for k in ('uri', 'digest', 'size_bytes'))
                            # Terminal publication.digest names a projection
                            # document, while size_bytes describes its archive.
                            if current.get('schema_version') == 'task_evaluation_scene_terminal_result_publication.v1':
                                digest = current.get('archive_digest')
                            require(isinstance(uri, str) and uri.startswith(('gs://', 's3://', 'https://', 'b2://', 'r2://'))
                                    and not any(c.isspace() for c in uri) and type(size) is int and size >= 0
                                    and matches(digest), 'remote_invalid')
                            self.size(digest, size)
                            self.remote.append({'uri': uri, 'digest': digest, 'size_bytes': size,
                                                'status': 'kept_unresolved', 'reason': 'remote_availability_unverified',
                                                'source_provenance': [proof]})
                        stack.extend(current.values())
                    elif isinstance(current, list):
                        stack.extend(current)

    def missing(self, role, reason, sources, expected=None, selector=None):
        self.consume()
        self.structural.append({'role': role, 'reason': reason, 'status': 'kept_unresolved',
                                'expected_path': expected, 'selector': selector or {},
                                'source_provenance': sources})

    def member(self, value, kind, binding, sources):
        self.members.append({'path': path(value), 'kind': kind, 'binding': binding,
                             'source_provenance': sources, 'presence_checked': False,
                             'exclusive_ownership_proven': False, 'measured_bytes': None, 'restore_verified': False})


def observation(row, *, status='kept_unresolved', reason=None, **fields):
    return {'status': status, 'reason': reason, 'source_provenance': [row[1]], **fields}


def unique(rows, limit):
    """Exact output-row dedup only after occurrence budget; deterministic keys."""
    pairs = {}
    total = 0
    for row in rows:
        total += bounded_size(row, limit - total)
        pairs[encoded(row)] = row
    return [row for _, row in sorted(pairs.items())]
