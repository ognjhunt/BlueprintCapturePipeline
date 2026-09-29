"""Tiny actual-file fixtures; production acquisition is never replaced.

Only success fixtures pin mutable metadata of shared ancestors OUTSIDE their
anchor. Named/descriptor dev/inode/type/mode and every in-anchor stat stay real.
This fixture must never be used for substitution or drift-negative tests.
"""
import copy
import hashlib
import json
import os
import stat
from pathlib import Path

from blueprint_pipeline.decision_evidence_contracts import canonical_digest, cross_runtime_canonical_digest


def stable_shared_ancestors(monkeypatch, anchor):
    anchor = Path(anchor).resolve()
    original_open, original_stat, original_fstat = os.open, os.stat, os.fstat
    names, first = {}, {}

    class PinnedStat:
        __slots__ = ('_info', 'st_size', 'st_mtime_ns', 'st_ctime_ns', 'st_nlink')

        def __init__(self, info, fields):
            self._info = info
            self.st_size, self.st_mtime_ns, self.st_ctime_ns, self.st_nlink = fields

        def __getattr__(self, name):
            return getattr(self._info, name)

    def resolve(name, parent=None):
        value = Path(name)
        if value.is_absolute():
            return value
        base = names.get(parent)
        return (base if base is not None else Path.cwd()) / value

    def pinned(path, info):
        if stat.S_ISDIR(info.st_mode) and path != anchor and anchor.is_relative_to(path):
            fields = first.setdefault(str(path), (info.st_size, info.st_mtime_ns, info.st_ctime_ns, info.st_nlink))
            return PinnedStat(info, fields)
        return info

    def opened(name, flags, mode=0o777, *, dir_fd=None):
        fd = original_open(name, flags, mode, dir_fd=dir_fd)
        names[fd] = resolve(name, dir_fd)
        return fd

    def named(name, *args, dir_fd=None, **kwargs):
        return pinned(resolve(name, dir_fd), original_stat(name, *args, dir_fd=dir_fd, **kwargs))

    def descriptor(fd):
        info = original_fstat(fd)
        return pinned(names[fd], info) if fd in names else info

    monkeypatch.setattr(os, 'open', opened)
    monkeypatch.setattr(os, 'stat', named)
    monkeypatch.setattr(os, 'fstat', descriptor)


def rebase_graph(args, anchor, *, remote_digest_replacements=None):
    """Recompute existing fixture seals/selectors after a real root relocation.

    The finite fixed point maps exact original identities, including producer
    filenames. It adds no owner fields, proof roles, payload claims or schemas.
    """
    original = copy.deepcopy(args)
    pairs = []
    for group in ('seed_records', 'downstream_records', 'source_records', 'bridge_records'):
        for role, rows in original.get(group, {}).items():
            rows = ([] if rows is None else [rows]) if role in ('intent', 'projection') else rows
            pairs.extend(rows)
    maps, sizes = dict(remote_digest_replacements or {}), {}
    anchor = str(Path(anchor).resolve())

    def text(value):
        if value in maps:
            return maps[value]
        if value == '/retained' or value.startswith('/retained/'):
            value = anchor + value[len('/retained'):]
        for old, new in sorted(maps.items(), key=lambda pair: -len(pair[0])):
            if old.startswith('sha256:'):
                value = value.replace(old[7:], new[7:])
            elif old.startswith('sam31-'):
                value = value.replace(old, new)
        return value

    def visit(value):
        if isinstance(value, str):
            return text(value)
        if isinstance(value, list):
            return [visit(item) for item in value]
        if not isinstance(value, dict):
            return value
        changed = {key: visit(item) for key, item in value.items()}
        for field in ('sha256', 'digest'):
            if isinstance(value.get(field), str) and 'size_bytes' in value and value[field] in sizes:
                changed['size_bytes'] = sizes[value[field]]
        if 'request' in value and value.get('request_digest') == canonical_digest(value['request']):
            changed['request_digest'] = canonical_digest(changed['request'])
            maps[value['request_digest']] = changed['request_digest']
        if value.get('schema_version') == 'task_evaluation_sam31_preparation_execution_job.v1':
            changed['inputs_digest'] = canonical_digest({name: {k: item[k] for k in ('sha256', 'size_bytes')}
                                                       for name, item in changed['inputs'].items()})
            maps[value['inputs_digest']] = changed['inputs_digest']
            changed['child_id'] = 'sam31-' + canonical_digest({k: changed[k] for k in
                ('parent_request_digest', 'plan_digest', 'phase', 'inputs_digest')})[7:]
            maps[value['child_id']] = changed['child_id']
        for key, item in value.items():
            if key.endswith('_digest') or key == 'digest':
                if not isinstance(item, str):
                    continue
                method = (canonical_digest if item == canonical_digest(value, digest_field=key) else
                          cross_runtime_canonical_digest if item == cross_runtime_canonical_digest(value, digest_field=key) else None)
                if method:
                    changed[key] = method(changed, digest_field=key)
                    maps[item] = changed[key]
        if value.get('schema_version') == 'task_evaluation_launch_preparation_result.v1' and value.get('status') == 'queued_for_production_episode_compilation':
            from blueprint_pipeline.task_evaluation_scene_compilation_owner_preparations import HANDOFF, PRE
            def inverse(record):
                result = {k: v for k, v in record.items() if k not in HANDOFF | {'result_digest'}}
                result['status'] = PRE
                return result
            maps[canonical_digest(inverse(value), digest_field='result_digest')] = canonical_digest(inverse(changed), digest_field='result_digest')
        maps[canonical_digest(value)] = canonical_digest(changed)
        return changed

    previous = None
    for _ in range(64):
        current = {}
        for path, raw in pairs:
            try:
                decoded = json.loads(raw)
            except (ValueError, UnicodeDecodeError):
                replaced = raw
            else:
                replaced = json.dumps(visit(decoded), sort_keys=True).encode()
            current[path, raw] = replaced
            digest = 'sha256:' + hashlib.sha256(raw).hexdigest()
            maps[digest] = 'sha256:' + hashlib.sha256(replaced).hexdigest()
            sizes[digest] = len(replaced)
        if current == previous:
            break
        previous = current
    else:
        raise AssertionError('fixture identity rebasing did not converge')
    result = copy.deepcopy(original)
    result['roots'] = {key: text(value) for key, value in original['roots'].items()}
    result['parent_routes'] = visit(original.get('parent_routes', []))
    result['retained_metadata_roots'] = visit(original.get('retained_metadata_roots', ['/retained/metadata']))
    for group in ('seed_records', 'downstream_records', 'source_records', 'bridge_records'):
        for role, rows in original.get(group, {}).items():
            def rebound(pair):
                return text(pair[0]), current[pair]
            result[group][role] = (None if rows is None else rebound(rows)) if role in ('intent', 'projection') else [rebound(pair) for pair in rows]
    return result
