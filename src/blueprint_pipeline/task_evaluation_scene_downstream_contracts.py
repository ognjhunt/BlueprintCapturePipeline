"""Pure bounded byte/provenance primitives for ADP-009D retained lineage."""
from __future__ import annotations

from .task_evaluation_scene_lineage_budget import _work_sort, _work_order, _work, _work_call, _work_hash, _work_items, _work_kwargs, _work_parse, _work_rows

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


def require(condition, code, *, work_budget=None):
    if work_budget is not None:
        _work(work_budget)
    if not condition:
        raise SceneDownstreamInventoryError('scene_downstream_' + code)


def matches(value, pattern=DIGEST, *, work_budget=None):
    if work_budget is not None:
        _work(work_budget)
    return isinstance(value, str) and pattern.fullmatch(value) is not None


def path(value, *, work_budget=None):
    if work_budget is not None:
        _work(work_budget)
    require(isinstance(value, str) and len(value) <= retained.MAX_PATH_BYTES, 'path_invalid', **_work_kwargs(work_budget))
    return retained._path(value, **_work_kwargs(work_budget))


def child(root, *parts, work_budget=None):
    if work_budget is not None:
        _work(work_budget)
    return retained._child(root, *parts, **_work_kwargs(work_budget))


def under(value, root, *, work_budget=None):
    if work_budget is not None:
        _work(work_budget)
    return PurePosixPath(path(value, **_work_kwargs(work_budget))).is_relative_to(PurePosixPath(root)) and value != root


def encoded(value, *, work_budget=None):
    if work_budget is not None:
        _work(work_budget)
    return retained._encoded(value, **_work_kwargs(work_budget))


def bounded_size(value, limit, *, work_budget=None):
    """Exact compact UTF8 JSON length without allocating the encoded document."""
    if work_budget is not None:
        _work(work_budget)
        work_budget.measure(value, cap=limit)
    size, stack = 0, [value]
    while stack:
        if work_budget is not None:
            work_budget.charge("values")
        item = stack.pop()
        if isinstance(item, dict):
            size += 2 + max(0, len(item) - 1) + len(item)
            (stack.extend(_work_items(item.values(), work_budget)) if work_budget is not None else stack.extend(item.values()))
            (stack.extend(_work_items(item.keys(), work_budget)) if work_budget is not None else stack.extend(item.keys()))
        elif isinstance(item, list):
            size += 2 + max(0, len(item) - 1)
            (stack.extend(_work_items(item, work_budget)) if work_budget is not None else stack.extend(item))
        elif isinstance(item, str):
            require(len(item) <= limit - size, 'output_limit', **_work_kwargs(work_budget))
            size += 2
            for char in (_work_items(item, work_budget) if work_budget is not None else item):
                code = ord(char)
                require(not 0xD800 <= code <= 0xDFFF, 'json_invalid', **_work_kwargs(work_budget))
                size += (2 if char in '"\\\b\f\n\r\t' else 6 if code < 32 else
                         1 if code < 128 else 2 if code < 2048 else 3 if code < 65536 else 4)
                require(size <= limit, 'output_limit', **_work_kwargs(work_budget))
        elif item is None:
            size += 4
        elif type(item) is bool:
            size += 4 if item else 5
        elif type(item) in (int, float):
            require(math.isfinite(item), 'json_invalid', **_work_kwargs(work_budget))
            size += len((_work_call(work_budget, json.dumps, item, allow_nan=False) if work_budget is not None else json.dumps(item, allow_nan=False)))
        else:
            require(False, 'json_invalid', **_work_kwargs(work_budget))
        require(size <= limit, 'output_limit', **_work_kwargs(work_budget))
    return size


def raw_digest(raw, *, work_budget=None):
    if work_budget is not None:
        _work(work_budget)
    return 'sha256:' + (_work_hash(work_budget, hashlib.sha256, raw) if work_budget is not None else hashlib.sha256(raw)).hexdigest()


class OutputRows(list):
    """Charge actual occurrence rows/bytes before storing, before dedup/encoding."""
    def __init__(self, budget, limits, rows=(), *, emission_budget=None, reference=False, work_budget=None):
        if work_budget is None:
            work_budget = getattr(emission_budget, "work_budget", None)
        self.work_budget = work_budget
        if work_budget is not None:
            _work(work_budget)
        super().__init__()
        self.budget, self.limits = budget, limits
        self.emission_budget, self.reference = emission_budget, reference
        for row in (_work_rows(rows, work_budget) if work_budget is not None else rows):
            self.append(row)

    def append(self, row, *, work_budget=None):
        if work_budget is None:
            work_budget = getattr(self, "work_budget", None)
        if work_budget is not None:
            _work(work_budget)
        require(self.budget['rows'] < self.limits['MAX_ROWS'], 'rows_limit', **_work_kwargs(work_budget))
        remaining = self.limits['MAX_OUTPUT_BYTES'] - self.budget['bytes']
        if self.emission_budget is not None:
            remaining = self.emission_budget.preflight_row(row, remaining, reference=self.reference)
        length = bounded_size(row, remaining, **_work_kwargs(work_budget))
        if self.emission_budget is not None:
            self.emission_budget.reserve_row(row, reference=self.reference)
        self.budget['bytes'] += length
        self.budget['rows'] += 1
        super().append(row)

    def extend(self, values, *, work_budget=None):
        if work_budget is None:
            work_budget = getattr(self, "work_budget", None)
        if work_budget is not None:
            _work(work_budget)
        for row in (_work_rows(values, work_budget) if work_budget is not None else values):
            self.append(row)


def seal(row, field, *, cross=False, work_budget=None):
    if work_budget is not None:
        _work(work_budget)
    value, proof = row
    digest = cross_runtime_canonical_digest if cross else canonical_digest
    require(matches(value.get(field), **_work_kwargs(work_budget)) and value[field] == (_work_call(work_budget, digest, value, digest_field=field) if work_budget is not None else digest(value, digest_field=field)), 'seal_invalid', **_work_kwargs(work_budget))
    proof.update(seal_field=field, seal_digest=value[field])


def lexical_preflight(raw, limits, budget, *, work_budget=None):
    """Conservative string-aware token budget before recursive JSON allocation.

    Count keys as well as values. Strict syntax/duplicates/numbers are checked by
    the reviewed parser after this bounded structural pass, never by a regex.
    """
    if work_budget is not None:
        _work(work_budget)
    text = raw.decode('utf-8')
    if work_budget is not None:
        work_budget.preflight(text)
    depth, quoted, escaped, primitive = 0, False, False, False
    for char in (_work_items(text, work_budget) if work_budget is not None else text):
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
            require(depth <= limits['MAX_DEPTH'], 'depth_limit', **_work_kwargs(work_budget))
            primitive = False
        elif char in '}]':
            depth -= 1
            require(depth >= 0, 'json_invalid', **_work_kwargs(work_budget))
        elif char not in ' \t\r\n,:' and not primitive:
            budget['tokens'] += 1
            primitive = True
        require(budget['tokens'] <= limits['MAX_NODES'], 'nodes_limit', **_work_kwargs(work_budget))
    require(not quoted and depth == 0, 'json_invalid', **_work_kwargs(work_budget))


def decode(groups, limits, *, work_budget=None):
    if work_budget is not None:
        _work(work_budget)
    count = sum(len(rows) for rows in (_work_items(groups.values(), work_budget) if work_budget is not None else groups.values()))
    require(count <= limits['MAX_RECORDS'], 'records_limit', **_work_kwargs(work_budget))
    total = 0
    for rows in (_work_items(groups.values(), work_budget) if work_budget is not None else groups.values()):
        for pair in (_work_items(rows, work_budget) if work_budget is not None else rows):
            require(isinstance(pair, (list, tuple)) and len(pair) == 2 and type(pair[1]) is bytes
                    and 0 < len(pair[1]) <= limits['MAX_RECORD_BYTES'], 'record_invalid', **_work_kwargs(work_budget))
            path(pair[0], **_work_kwargs(work_budget))
            total += len(pair[1])
            require(total <= limits['MAX_TOTAL_BYTES'], 'bytes_limit', **_work_kwargs(work_budget))
    budget = {'tokens': 0}
    for rows in (_work_items(groups.values(), work_budget) if work_budget is not None else groups.values()):
        for _, raw in (_work_items(rows, work_budget) if work_budget is not None else rows):
            lexical_preflight(raw, limits, budget, **_work_kwargs(work_budget))
    decoded, identities, nodes, raw_pairs = {}, set(), 0, []
    immutable = {'intent', 'events', 'attempts', 'source_snapshots', 'factories', 'source_submissions',
                 'preparation_links', 'preparation_envelopes', 'activation_envelopes', 'launch_requests',
                 'compilation_envelopes'}
    paths = {}
    for role, rows in (_work_items((_work_order(work_budget, sorted, groups.items()) if work_budget is not None else sorted(groups.items())), work_budget) if work_budget is not None else sorted(groups.items())):
        decoded[role] = []
        for pair in (_work_items(rows, work_budget) if work_budget is not None else rows):
            value = (_work_parse(work_budget, json.loads, pair[1].decode('utf-8'), object_pairs_hook=retained._pairs,
                               parse_int=retained._numeric, parse_float=retained._numeric,
                               parse_constant=lambda _: require(False, 'json_invalid')) if work_budget is not None else json.loads(pair[1].decode('utf-8'), object_pairs_hook=retained._pairs,
                               parse_int=retained._numeric, parse_float=retained._numeric,
                               parse_constant=lambda _: require(False, 'json_invalid')))
            require(isinstance(value, dict), 'record_invalid', **_work_kwargs(work_budget))
            proof = {'role': role, 'path': pair[0], 'size_bytes': len(pair[1]), 'seal_field': None, 'seal_digest': None}
            identity = (pair[0], pair[1])
            require(identity not in identities, 'record_duplicate', **_work_kwargs(work_budget))
            require(proof['path'] not in paths or (role not in immutable and paths[proof['path']] == role), 'immutable_ambiguous', **_work_kwargs(work_budget))
            identities.add(identity)
            paths[proof['path']] = role
            stack = [(value, 1)]
            while stack:
                if work_budget is not None:
                    work_budget.charge("values")
                current, depth = stack.pop()
                nodes += 1
                require(nodes <= limits['MAX_NODES'] and depth <= limits['MAX_DEPTH'], 'nodes_limit', **_work_kwargs(work_budget))
                children = current.values() if isinstance(current, dict) else current if isinstance(current, list) else ()
                (stack.extend(_work_items(((item, depth + 1) for item in (_work_items(children, work_budget) if work_budget is not None else children)), work_budget)) if work_budget is not None else stack.extend((item, depth + 1) for item in (_work_items(children, work_budget) if work_budget is not None else children)))
            bounded_size(value, limits['MAX_TOTAL_BYTES'], **_work_kwargs(work_budget))  # Surrogate/type checks after decoded bounds, before hashes.
            decoded[role].append((value, proof))
            raw_pairs.append((pair[1], proof))
    for raw, proof in (_work_items(raw_pairs, work_budget) if work_budget is not None else raw_pairs):  # No proof hashing until ALL actual decoded limits pass.
        proof['sha256'] = raw_digest(raw, **_work_kwargs(work_budget))
    for rows in (_work_items(decoded.values(), work_budget) if work_budget is not None else decoded.values()):
        (_work_sort(work_budget, rows, key=lambda r: (r[1]['path'], r[1]['sha256'], r[1]['size_bytes'])) if work_budget is not None else rows.sort(key=lambda r: (r[1]['path'], r[1]['sha256'], r[1]['size_bytes'])))
    return decoded


class Context:
    def __init__(self, decoded, roots, limits, intent_id, *, emission_budget=None, work_budget=None):
        if work_budget is None:
            work_budget = getattr(emission_budget, "work_budget", None)
        self.work_budget = work_budget
        if work_budget is not None:
            _work(work_budget)
        self.decoded, self.roots, self.limits, self.intent_id = decoded, roots, limits, intent_id
        self.budget = {'rows': 0, 'bytes': 0}
        self.emission_budget = emission_budget
        self.raw = self.rows(p for rows in (_work_items(decoded.values(), work_budget) if work_budget is not None else decoded.values()) for _, p in (_work_items(rows, work_budget) if work_budget is not None else rows))
        self.index = {(p['path'], p['sha256'], p['size_bytes']): p for p in (_work_items(self.raw, work_budget) if work_budget is not None else self.raw)}
        self.by_path = {role: {} for role in (_work_items(decoded, work_budget) if work_budget is not None else decoded)}
        for role, rows in (_work_items(decoded.items(), work_budget) if work_budget is not None else decoded.items()):
            for row in (_work_items(rows, work_budget) if work_budget is not None else rows):
                self.by_path[role].setdefault(row[1]['path'], []).append(row)
        self.obligations, self.remote, self.structural = (self.rows(reference=True) for _ in (_work_items(range(3), work_budget) if work_budget is not None else range(3)))
        self.members = self.rows()
        self.count, self.sizes = 0, {}
        for proof in (_work_items(self.raw, work_budget) if work_budget is not None else self.raw):
            self.size(proof['sha256'], proof['size_bytes'])

    def rows(self, values=(), *, reference=False, work_budget=None):
        if work_budget is None:
            work_budget = getattr(self, "work_budget", None)
        if work_budget is not None:
            _work(work_budget)
        return OutputRows(self.budget, self.limits, values, emission_budget=self.emission_budget, reference=reference, **_work_kwargs(work_budget))

    def provenance(self, values, *, work_budget=None):
        if work_budget is None:
            work_budget = getattr(self, "work_budget", None)
        if work_budget is not None:
            _work(work_budget)
        return self.emission_budget.reserve_provenance(values) if self.emission_budget is not None else list(values)

    def seed_budget(self, seed, *, work_budget=None):
        if work_budget is None:
            work_budget = getattr(self, "work_budget", None)
        if work_budget is not None:
            _work(work_budget)
        self.budget['rows'] += sum(len(v) for v in (_work_items(seed.values(), work_budget) if work_budget is not None else seed.values()) if isinstance(v, list))
        require(self.budget['rows'] <= self.limits['MAX_ROWS'], 'rows_limit', **_work_kwargs(work_budget))
        self.budget['bytes'] += bounded_size(seed, self.limits['MAX_OUTPUT_BYTES'] - self.budget['bytes'], **_work_kwargs(work_budget))
        # Missing child joins and newly discovered retained-result obligations
        # share this API's occurrence budget; supplied reference fields were
        # already charged by the combined raw record walk.
        for _ in (_work_items(seed['preparation_join_obligations'] + seed['request_projection_obligations'], work_budget) if work_budget is not None else seed['preparation_join_obligations'] + seed['request_projection_obligations']):
            self.consume()
        for row in (_work_items(seed['obligations'], work_budget) if work_budget is not None else seed['obligations']):
            if row['role'] == 'preparation_result':
                self.consume()

    def consume(self, *, work_budget=None):
        if work_budget is None:
            work_budget = getattr(self, "work_budget", None)
        if work_budget is not None:
            _work(work_budget)
        self.count += 1
        require(self.count <= self.limits['MAX_REFERENCES'], 'references_limit', **_work_kwargs(work_budget))

    def raw_ref(self, ref, proof, *, work_budget=None):
        if work_budget is None:
            work_budget = getattr(self, "work_budget", None)
        if work_budget is not None:
            _work(work_budget)
        self.consume()
        require(isinstance(ref, dict) and {'path', 'sha256', 'size_bytes'} <= set(ref), 'reference_invalid', **_work_kwargs(work_budget))
        path(ref['path'], **_work_kwargs(work_budget))
        require(matches(ref['sha256'], **_work_kwargs(work_budget)) and type(ref['size_bytes']) is int and ref['size_bytes'] >= 0, 'reference_invalid', **_work_kwargs(work_budget))
        self.size(ref['sha256'], ref['size_bytes'])
        key = tuple(ref[k] for k in (_work_items(('path', 'sha256', 'size_bytes'), work_budget) if work_budget is not None else ('path', 'sha256', 'size_bytes')))
        match = self.index.get(key)
        self.obligations.append({**{k: ref[k] for k in (_work_items(('path', 'sha256', 'size_bytes'), work_budget) if work_budget is not None else ('path', 'sha256', 'size_bytes'))},
                                 'status': 'matched_retained_bytes' if match else 'kept_unresolved',
                                 'reason': None if match else 'reference_bytes_unavailable',
                                 'source_provenance': [proof], 'matched_provenance': match})
        return match

    def size(self, digest, size, *, work_budget=None):
        if work_budget is None:
            work_budget = getattr(self, "work_budget", None)
        if work_budget is not None:
            _work(work_budget)
        require(digest not in self.sizes or self.sizes[digest] == size, 'content_size_conflict', **_work_kwargs(work_budget))
        self.sizes[digest] = size

    def references(self, *, work_budget=None):
        if work_budget is None:
            work_budget = getattr(self, "work_budget", None)
        if work_budget is not None:
            _work(work_budget)
        for rows in (_work_items(self.decoded.values(), work_budget) if work_budget is not None else self.decoded.values()):
            for value, proof in (_work_items(rows, work_budget) if work_budget is not None else rows):
                stack = [value]
                while stack:
                    if work_budget is not None:
                        work_budget.charge("values")
                    current = stack.pop()
                    if isinstance(current, dict):
                        if 'path' in current and 'sha256' in current:
                            self.raw_ref(current, proof)
                        if 'uri' in current and 'digest' in current and 'size_bytes' in current:
                            self.consume()
                            uri, digest, size = (current[k] for k in (_work_items(('uri', 'digest', 'size_bytes'), work_budget) if work_budget is not None else ('uri', 'digest', 'size_bytes')))
                            # Terminal publication.digest names a projection
                            # document, while size_bytes describes its archive.
                            if current.get('schema_version') == 'task_evaluation_scene_terminal_result_publication.v1':
                                digest = current.get('archive_digest')
                            require(isinstance(uri, str) and uri.startswith(('gs://', 's3://', 'https://', 'b2://', 'r2://'))
                                    and not any(c.isspace() for c in (_work_items(uri, work_budget) if work_budget is not None else uri)) and type(size) is int and size >= 0
                                    and matches(digest, **_work_kwargs(work_budget)), 'remote_invalid', **_work_kwargs(work_budget))
                            self.size(digest, size)
                            self.remote.append({'uri': uri, 'digest': digest, 'size_bytes': size,
                                                'status': 'kept_unresolved', 'reason': 'remote_availability_unverified',
                                                'source_provenance': [proof]})
                        (stack.extend(_work_items(current.values(), work_budget)) if work_budget is not None else stack.extend(current.values()))
                    elif isinstance(current, list):
                        (stack.extend(_work_items(current, work_budget)) if work_budget is not None else stack.extend(current))

    def missing(self, role, reason, sources, expected=None, selector=None, *, work_budget=None):
        if work_budget is None:
            work_budget = getattr(self, "work_budget", None)
        if work_budget is not None:
            _work(work_budget)
        self.consume()
        self.structural.append({'role': role, 'reason': reason, 'status': 'kept_unresolved',
                                'expected_path': expected, 'selector': selector or {},
                                'source_provenance': sources})

    def member(self, value, kind, binding, sources, *, work_budget=None):
        if work_budget is None:
            work_budget = getattr(self, "work_budget", None)
        if work_budget is not None:
            _work(work_budget)
        self.members.append({'path': path(value, **_work_kwargs(work_budget)), 'kind': kind, 'binding': binding,
                             'source_provenance': sources, 'presence_checked': False,
                             'exclusive_ownership_proven': False, 'measured_bytes': None, 'restore_verified': False})


def observation(row, *, status='kept_unresolved', reason=None, work_budget=None, **fields):
    if work_budget is not None:
        _work(work_budget)
    return {'status': status, 'reason': reason, 'source_provenance': [row[1]], **fields}


def unique(rows, limit, *, work_budget=None):
    """Exact output-row dedup only after occurrence budget; deterministic keys."""
    if work_budget is not None:
        _work(work_budget)
    pairs = {}
    total = 0
    for row in (_work_items(rows, work_budget) if work_budget is not None else rows):
        total += bounded_size(row, limit - total, **_work_kwargs(work_budget))
        pairs[encoded(row, **_work_kwargs(work_budget))] = row
    return [row for _, row in (_work_items(sorted(pairs.items()), work_budget) if work_budget is not None else sorted(pairs.items()))]
