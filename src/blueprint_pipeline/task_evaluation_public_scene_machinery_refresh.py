"""Refresh an operator catalog reference, preserving exact prior machinery bytes.

Use after an admitted controls catalog installation, before provisioning a new
intent. This command never changes an intent, retained attempt, or paid authority.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import tempfile

from .decision_evidence_contracts import canonical_digest
from .task_evaluation_controls_autoprovision import resolve_robot_catalog
from .task_evaluation_public_scene_attempt_factory import record
from .task_evaluation_scene_configuration_submission_inputs import read
from .task_evaluation_scene_progression_state import safe_path


def refresh(*, machinery_path, catalog_path, expected_machinery_digest, source_commit, apply=False):
    path, catalog = safe_path(machinery_path), safe_path(catalog_path)
    lock_path = safe_path(path.parent / (path.name + '.refresh.lock'))
    lock_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    with os.fdopen(lock_fd, 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        old_bytes = path.read_bytes()
        old = read(path, digest_field='machinery_digest')
        if old.get('schema_version') != 'task_evaluation_public_scene_machinery.v1':
            raise ValueError('public_scene_machinery_schema_invalid')
        if old['machinery_digest'] != expected_machinery_digest:
            raise ValueError('public_scene_machinery_expected_digest_mismatch')
        if old.get('robot_catalog', {}).get('path') != str(catalog):
            raise ValueError('public_scene_machinery_catalog_path_mismatch')
        catalog_bytes = catalog.read_bytes()
        resolve_robot_catalog(read(catalog, digest_field='catalog_digest'), source_commit=source_commit)
        new = {**old, 'robot_catalog': record(catalog)}
        new['machinery_digest'] = canonical_digest(new, digest_field='machinery_digest')
        changed = new != old
        archive = path.parent / (path.name + '.history') / (expected_machinery_digest.removeprefix('sha256:') + '.json')
        result = {'status': 'refresh_required' if changed else 'current', 'applied': False,
                  'old_machinery_digest': expected_machinery_digest,
                  'new_machinery_digest': new['machinery_digest'],
                  'provider_mutation_performed': False}
        if not apply or not changed:
            return result
        safe_path(archive)
        archive.parent.mkdir(mode=0o750, exist_ok=True)
        info = path.stat()
        try:
            with archive.open('xb') as saved:
                saved.write(old_bytes)
                saved.flush()
                os.fsync(saved.fileno())
                os.fchown(saved.fileno(), info.st_uid, info.st_gid)
                os.fchmod(saved.fileno(), 0o440)
        except FileExistsError:
            if archive.read_bytes() != old_bytes:
                raise ValueError('public_scene_machinery_archive_conflict')
        descriptor, temporary = tempfile.mkstemp(prefix='.machinery-refresh-', dir=path.parent)
        try:
            with os.fdopen(descriptor, 'w') as stream:
                json.dump(new, stream, sort_keys=True, separators=(',', ':'), allow_nan=False)
                stream.write('\n')
                stream.flush()
                os.fsync(stream.fileno())
                os.fchown(stream.fileno(), info.st_uid, info.st_gid)
                os.fchmod(stream.fileno(), info.st_mode & 0o777)
            if path.read_bytes() != old_bytes or catalog.read_bytes() != catalog_bytes:
                raise ValueError('public_scene_machinery_concurrent_change')
            os.replace(temporary, path)
            directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return {**result, 'status': 'refreshed', 'applied': True, 'archive': record(archive)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--machinery-path', required=True)
    parser.add_argument('--catalog-path', required=True)
    parser.add_argument('--expected-machinery-digest', required=True)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--apply', action='store_true')
    print(json.dumps(refresh(**vars(parser.parse_args(argv))), sort_keys=True))


if __name__ == '__main__':
    main()
