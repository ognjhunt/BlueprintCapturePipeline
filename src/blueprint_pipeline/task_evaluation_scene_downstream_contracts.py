"""Pure bounded byte/provenance primitives for ADP-009D retained lineage."""
from __future__ import annotations

import re
from pathlib import PurePosixPath

from . import task_evaluation_scene_preparation_lineage as retained
from .decision_evidence_contracts import canonical_digest, cross_runtime_canonical_digest

DIGEST = re.compile(r'sha256:[0-9a-f]{64}\Z')
COMMIT = re.compile(r'[0-9a-f]{40}\Z')
ID = retained._ID
ACTIVATION_ID = re.compile(r'[A-Za-z0-9][A-Za-z0-9._-]{0,191}\Z')


class SceneDownstreamInventoryError(ValueError):
    """A bounded typed refusal, never supplied private text."""


def require(condition, code):
    if not condition:
        raise SceneDownstreamInventoryError('scene_downstream_' + code)


def matches(value, pattern=DIGEST):
    return isinstance(value, str) and pattern.fullmatch(value) is not None


def path(value):
    return retained._path(value)


def child(root, *parts):
    return retained._child(root, *parts)


def under(value, root):
    return PurePosixPath(path(value)).is_relative_to(PurePosixPath(root)) and value != root


def encoded(value):
    return retained._encoded(value)


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
    decoded, identities, nodes = {}, set(), 0
    immutable = {'intent', 'events', 'attempts', 'source_snapshots', 'factories', 'source_submissions',
                 'preparation_links', 'preparation_envelopes', 'activation_envelopes', 'launch_requests',
                 'compilation_envelopes'}
    paths = {}
    for role, rows in sorted(groups.items()):
        decoded[role] = []
        for pair in rows:
            row = retained._record(pair, role, set())
            value, proof = row
            proof.update(seal_field=None, seal_digest=None)
            identity = tuple(proof[k] for k in ('path', 'sha256', 'size_bytes'))
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
            decoded[role].append(row)
        decoded[role].sort(key=lambda r: (r[1]['path'], r[1]['sha256'], r[1]['size_bytes']))
    return decoded


class Context:
    def __init__(self, decoded, roots, limits, intent_id):
        self.decoded, self.roots, self.limits, self.intent_id = decoded, roots, limits, intent_id
        self.raw = [p for rows in decoded.values() for _, p in rows]
        self.index = {(p['path'], p['sha256'], p['size_bytes']): p for p in self.raw}
        self.obligations, self.remote, self.structural, self.members = [], [], [], []
        self.count, self.sizes = 0, {}

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


def unique(rows):
    """Exact output-row dedup only after occurrence budget; deterministic keys."""
    return [row for _, row in sorted({encoded(row): row for row in rows}.items())]
