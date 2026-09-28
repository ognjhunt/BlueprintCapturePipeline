"""Exact owned stage/destination union under the retained restore operation EX.

This is limited to durable stage/directory receipts and exact stage inode aliases. Unknown destination
nodes never become ours from names, equal bytes or absent historical children.
"""
from __future__ import annotations

import hashlib
import os
import stat

from . import control_plane_lane_experiment_actions as actions
from . import control_plane_lane_experiment_recovery as recovery
from . import control_plane_lane_experiment_retirement as issuance
from . import control_plane_lane_owner_consents as owners
from .control_plane_lane_owner_target_versions import _require


def _identity(info, kind):
    return dict(dev=info.st_dev, ino=info.st_ino, type=kind)


def _named(files, path):
    try:
        parent, name = files.parent(path)
    except FileNotFoundError:
        return None, path.name, None
    files.location(parent)
    try:
        info = os.stat(name, dir_fd=parent, follow_symlinks=False)
    except FileNotFoundError:
        return parent, name, None
    return parent, name, info


def load(files, store_path, target, target_fd, operation, action, entry, original_rows, started, stage_ready):
    """Return only proved owned aliases; payload hashes do not grant ownership."""
    proof, previous = stage_ready[0]['body'], stage_ready[1]
    stage_name = '.restore-' + action['action_id']
    stage_path = target / stage_name
    files.phase('restore_stage_verify')
    stage = files.open(stage_name, os.O_RDONLY | os.O_DIRECTORY, parent=target_fd)
    files.parents[stage_path] = stage
    info = os.fstat(stage)
    _require(proof['stage_identity'] == _identity(info, 'directory')
             and type(proof['stage_metadata']) is list and len(proof['stage_metadata']) == 7
             and (info.st_mode, info.st_uid, info.st_gid) == tuple(proof['stage_metadata'][:3]),
             'experiment_restore_stage_changed')
    raw, _ = files.read(store_path / (action['action_id'] + '.stage-manifest.json'),
                        cap=1048576, protected=True, mode=0o600)
    _require(issuance._selector(raw, files.budget) == proof['stage_manifest'], 'experiment_restore_stage_changed')
    saved = actions._manifest_record(files, raw, entry)
    expected = {row[0]: row for row in saved['members']}
    original = {row[0]: row for row in original_rows}
    _require(set(expected) == set(original) and all(expected[path][1] == original[path][1]
             and expected[path][4] == original[path][4] for path in expected), 'experiment_restore_stage_changed')
    created, linked = {}, {}
    next_index = 2
    for index in range(2, 4098):
        if (index - 2) % 16 == 0:
            files.phase('restore_recovery_batch')
        value = recovery._read_event(files, operation, action, index, previous)
        if value is None:
            break
        event, selected = value
        body = event['body']
        _require(body.get('restore_started') == started and body.get('path') in expected,
                 'experiment_restore_operation_invalid')
        path, row = body['path'], expected[body['path']]
        if event['event_kind'] == 'restore_directory':
            _require(set(body) == {'restore_started', 'path', 'identity', 'stat_token'}
                     and row[1] == 'directory' and path not in created,
                     'experiment_restore_operation_invalid')
            created[path] = body
        else:
            _require(event['event_kind'] == 'restore_member' and set(body) == {
                'restore_started', 'index', 'path', 'sha256', 'size_bytes', 'identity'}
                and row[1] == 'file' and path not in linked and body['identity'] == {
                    'dev':int(row[2].split(':')[0]), 'ino':int(row[2].split(':')[1]), 'type':'file'}
                and body['sha256'] == row[4] and body['size_bytes'] == int(row[3].split(':')[4])
                and type(body['index']) is int and body['index'] == sorted(p for p in expected if expected[p][1] == 'file').index(path),
                'experiment_restore_operation_invalid')
            linked[path] = selected
        previous, next_index = selected, index + 1
    else:
        _require(False, 'experiment_restore_event_limit')
    files.payload(target, target_fd, expected_payload_bytes=saved['logical_bytes'])
    staged, modes, mapped, pending = [], {}, {}, []
    namespace = {target: {stage_name, *actions._METADATA}, stage_path: set()}
    def register(path):
        namespace.setdefault(path.parent, set()).add(path.name)
        if path in namespace:
            return
    for path, row in expected.items():
        source, destination = stage_path / path, target / path
        source_parent, source_name, source_info = _named(files, source)
        destination_parent, destination_name, destination_info = _named(files, destination)
        ids, token = row[2].split(':'), tuple(map(int, row[3].split(':')))
        expected_identity = dict(dev=int(ids[0]), ino=int(ids[1]), type=row[1])
        if row[1] == 'directory':
            _require(source_info is not None and stat.S_ISDIR(source_info.st_mode)
                     and _identity(source_info, 'directory') == expected_identity
                     and (stat.S_IMODE(source_info.st_mode), source_info.st_uid, source_info.st_gid) == token[:3],
                     'experiment_restore_stage_changed')
            modes[path] = tuple(map(int, original[path][3].split(':')))
            register(source)
            namespace.setdefault(source, set())
            if destination_info is not None:
                receipt = created.get(path)
                _require(receipt is not None and stat.S_ISDIR(destination_info.st_mode)
                         and receipt['identity'] == _identity(destination_info, 'directory')
                         and type(receipt['stat_token']) is list and len(receipt['stat_token']) == 7
                         and tuple(receipt['stat_token'][:3]) == (destination_info.st_mode, destination_info.st_uid, destination_info.st_gid),
                         'experiment_restore_destination_unproven')
                mapped[path] = owners._metadata(destination_info)
                register(destination)
                namespace.setdefault(destination, set())
            else:
                _require(path not in created, 'experiment_restore_destination_unproven')
            continue
        _require(source_info is not None or destination_info is not None, 'experiment_restore_stage_changed')
        if destination_info is not None:
            # The one syscall-before-event interruption is recoverable only
            # from the sealed original stage inode still visibly linked at
            # BOTH names. Equal destination bytes or a missing stage token do
            # not supply that proof. The per-inode checks below run first.
            _require(path in linked or source_info is not None, 'experiment_restore_destination_unproven')
            register(destination)
        else:
            _require(path not in linked, 'experiment_restore_destination_unproven')
        if source_info is not None:
            register(source)
        for current in (source_info, destination_info):
            if current is None:
                continue
            _require(stat.S_ISREG(current.st_mode) and _identity(current, 'file') == expected_identity
                     and (stat.S_IMODE(current.st_mode), current.st_uid, current.st_gid, current.st_size, current.st_mtime_ns)
                     == (token[0], token[1], token[2], token[4], token[5])
                     and current.st_nlink == (2 if source_info is not None and destination_info is not None else 1),
                     'experiment_restore_stage_changed')
        if path not in linked and destination_info is None:
            _require(owners._metadata(source_info) == (int(ids[0]), int(ids[1]), stat.S_IFREG | token[0], *token[1:]),
                     'experiment_restore_stage_changed')
        selected_parent, selected_name = (source_parent, source_name) if source_info is not None else (destination_parent, destination_name)
        fd = files.open(selected_name, os.O_RDONLY | os.O_NONBLOCK, parent=selected_parent)
        current, digest, size = os.fstat(fd), hashlib.sha256(), 0
        try:
            while True:
                files.check_long()
                block = files.payload_read(fd, 1024 * 1024, role='restore_stage_validate')
                if not block:
                    break
                digest.update(block)
                size += len(block)
            _require(size == token[4] and 'sha256:' + digest.hexdigest() == row[4]
                     and owners._metadata(os.fstat(fd)) == owners._metadata(current)
                     == owners._metadata(os.stat(selected_name, dir_fd=selected_parent, follow_symlinks=False)),
                     'experiment_restore_stage_changed')
            if destination_info is not None and path not in linked:
                # Record the independently proved alias; immutable publication
                # follows only after complete union/hash checks and a declared
                # metadata phase. At most the interrupted first event exists.
                _require(not pending, 'experiment_restore_destination_unproven')
                pending.append((path, owners._metadata(current), row[4], token[4]))
            if source_info is not None:
                staged.append((path, source_info, row[4]))
            if destination_info is not None:
                mapped[path] = owners._metadata(destination_info)
        finally:
            files.close(fd)
            files.trim_payload(keep=(stage,))
    # The current union must be closed over the entire actual namespace. In
    # particular a copied, guessed or unrecorded directory remains unowned.
    for path, names in sorted(namespace.items(), key=lambda pair:str(pair[0])):
        if path == target:
            fd = target_fd
        elif path == stage_path:
            fd = stage
        else:
            parent, name = files.parent(path)
            fd = files.open(name, os.O_RDONLY | os.O_DIRECTORY, parent=parent)
        seen = set()
        try:
            files.slot()
            with os.scandir(fd) as stream:
                for item in stream:
                    files.check_long()
                    count = files.work.get('restore_namespace', 0) + 1
                    _require(count <= 8208 and item.name in names and item.name not in seen,
                             'experiment_restore_namespace_changed')
                    files.work['restore_namespace'] = count
                    seen.add(item.name)
            _require(seen == names, 'experiment_restore_namespace_changed')
            files.location(fd)
        finally:
            if fd not in (target_fd, stage):
                files.close(fd)
                files.trim_payload(keep=(stage,))
    target_transition = [getattr(os.fstat(target_fd), key) for key in recovery._STAT]
    files.phase('restore_union_finalize')
    for path, metadata, digest, size in pending:
        source_parent, source_name = files.parent(stage_path / path)
        destination_parent, destination_name = files.parent(target / path)
        files.location(source_parent)
        files.location(destination_parent)
        _require(owners._metadata(os.stat(source_name, dir_fd=source_parent, follow_symlinks=False))
                 == metadata == owners._metadata(os.stat(destination_name, dir_fd=destination_parent, follow_symlinks=False)),
                 'experiment_restore_destination_unproven')
        fd = files.open(source_name, os.O_RDONLY | os.O_NONBLOCK, parent=source_parent)
        try:
            files.proof(fd)
            _require(owners._metadata(os.fstat(fd)) == metadata, 'experiment_restore_destination_unproven')
            selected = actions._event(files, operation, action, 'restore_member', dict(
                restore_started=started, index=sorted(p for p in expected if expected[p][1] == 'file').index(path),
                path=path, sha256=digest, size_bytes=size, identity={
                    'dev':metadata[0], 'ino':metadata[1], 'type':'file'}), next_index, previous, files.now())
            previous, next_index = selected, next_index + 1
            linked[path] = selected
        finally:
            files.close(fd)
    return dict(stage=stage, staged=staged, directory_modes=modes, mapped=mapped,
                created=created, linked=linked, previous=previous, index=next_index,
                target_transition=target_transition)
