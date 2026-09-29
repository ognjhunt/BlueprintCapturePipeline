"""Native generated-member provenance, never cleanup or execution authority.

One actual born producer and retained native archive bind the generated digest.
The cache module still owns its regular-generation and hardlink protocol.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import sys
import zipfile
from contextlib import contextmanager
from pathlib import Path

from .decision_evidence_contracts import canonical_digest
from . import task_evaluation_scene_retirement_access as access
from .task_evaluation_scene_retirement_access import _canonical, _identity, _opened, _require
from .task_evaluation_scene_retirement_authority import load_document, selected_document
from .task_evaluation_scene_retirement_generations import _guard, _new_file, _named, _write

SCHEMA = 'scene_generated_content_publication.v1'
_ERROR = 'scene_retirement_generated_source_unproven'
_FIELDS = {'schema_version', 'intent_raw_ref', 'storage_authority_raw_ref',
    'producer_generation_raw_ref', 'producer_root', 'request_digest', 'bundle_raw_ref',
    'expected_reference', 'role', 'manifest_raw_ref', 'manifest_digest', 'entry',
    'external_source_raw_ref', 'external_generation_raw_ref', 'publication_digest'}


def _raw_bytes(raw, path):
    return dict(path=str(path), sha256='sha256:'+hashlib.sha256(raw).hexdigest(), size_bytes=len(raw))


def _key(path):
    return hashlib.sha256(str(path).encode()).hexdigest()+'.json'


def _generation(policy, path, *, directory):
    value, reference = load_document(Path(policy['generation_store'])/_key(path), maximum=65536)
    _require(value.get('state_digest') == canonical_digest(value, digest_field='state_digest')
             and value.get('canonical_path') == str(path)
             and value.get('state') in {'active', 'restored-active'}, _ERROR)
    with _opened(path, directory=directory) as (_, info):
        _require(tuple(value.get(name) for name in ('dev', 'ino', 'mode')) == _identity(info), _ERROR)
        if not directory:
            _require(value.get('schema_version') == 'scene_content_generation.v1'
                     and value.get('size_bytes') == info.st_size
                     and (value.get('uid'), value.get('gid')) == (info.st_uid, info.st_gid), _ERROR)
        else:
            _require(value.get('schema_version') == 'scene_member_generation.v1', _ERROR)
    return value, reference


def _producer(policy, hint):
    anchors = [Path(row['root']) for row in policy['roots']]
    path = _canonical(str(hint))
    for candidate in (path, *path.parents):
        if not any(candidate.is_relative_to(root) for root in anchors):
            break
        try:
            value, reference = _generation(policy, candidate, directory=True)
        except FileNotFoundError:
            continue
        if value.get('source_storage_authority_raw_ref') is None:
            return None
        return candidate, value, reference
    return None


def _snapshot(info):
    return info.st_size, info.st_mtime_ns, info.st_ctime_ns


class _ArchiveReader(io.RawIOBase):
    def __init__(self, use):
        self.use = use

    def readable(self):
        return True

    def seekable(self):
        return True

    def read(self, size=-1):
        self.use.guard()
        # Native ZIP metadata is at most 4 MiB; actual payload is streamed in
        # 1 MiB chunks. No unbounded read can allocate the whole source bundle.
        if size == -1:
            size = self.use.info.st_size-os.lseek(self.use.fd, 0, os.SEEK_CUR)
        _require(type(size) is int and 0 <= size <= 4*1024*1024, _ERROR)
        raw = os.read(self.use.fd, size)
        self.use.guard()
        if self.use.allowance is not None:
            self.use.allowance.charge('local_bytes', len(raw))
        return raw

    def seek(self, offset, whence=0):
        self.use.guard()
        value = os.lseek(self.use.fd, offset, whence)
        self.use.guard()
        return value

    def tell(self):
        return self.seek(0, os.SEEK_CUR)


class _BundleUse:
    def __init__(self, *, path, fd, info, parent, parent_info, policy, producer, request,
                 reference, role, allowance=None):
        self.path, self.fd, self.info = path, fd, info
        self.parent, self.parent_identity = parent, _identity(parent_info)
        self.identity, self.snapshot = _identity(info), _snapshot(info)
        self.policy, self.producer, self.request = policy, producer, request
        self.reference, self.role, self.allowance = reference, role, allowance
        self.guard()

    def guard(self):
        if self.allowance is not None:
            self.allowance.tick()
        _guard(self.parent, self.parent_identity)
        _guard(self.fd, self.identity)
        _require(_identity(os.stat(self.path.name, dir_fd=self.parent, follow_symlinks=False)) == self.identity
                 and _snapshot(os.fstat(self.fd)) == self.snapshot, _ERROR)
        if self.allowance is not None:
            self.allowance.tick()

    def verify_reference(self):
        self.guard()
        _require(type(self.reference.get('size_bytes')) is int
                 and self.info.st_size == self.reference['size_bytes']
                 and 0 < self.info.st_size <= 200*1024**3, _ERROR)
        os.lseek(self.fd, 0, os.SEEK_SET)
        digest = hashlib.sha256()
        remaining = self.info.st_size
        while remaining:
            self.guard()
            chunk = os.read(self.fd, min(1024*1024, remaining))
            self.guard()
            _require(chunk, _ERROR)
            if self.allowance is not None:
                self.allowance.charge('local_bytes', len(chunk))
            digest.update(chunk)
            remaining -= len(chunk)
        _require('sha256:'+digest.hexdigest() == self.reference.get('digest'), _ERROR)
        self.guard()
        os.lseek(self.fd, 0, os.SEEK_SET)

    def archive(self):
        return zipfile.ZipFile(_ArchiveReader(self))

    def publish(self, *, manifest_bytes, manifest, entry, external_source=None):
        if self.producer is None:
            return None
        from . import task_evaluation_scene_retirement_cache as cache
        self.guard()
        root, generation, generation_ref = self.producer
        current, _ = _generation(self.policy, root, directory=True)
        _require(current == generation, _ERROR)
        authority_ref = generation['source_storage_authority_raw_ref']
        authority = selected_document(authority_ref, maximum=65536)
        intent, _ = cache._validate(authority, self.request, now=cache.time.time())
        _require(generation['owner_raw_ref'] == authority['intent_raw_ref']
                 and generation['birth_request_raw_ref'] == authority['attempt_raw_ref']
                 and generation['owner_intent_id'] == intent['intent_id'], _ERROR)
        _require(type(manifest_bytes) is bytes and 0 < len(manifest_bytes) <= 4*1024*1024
                 and json.loads(manifest_bytes) == manifest
                 and entry in manifest['entries'], _ERROR)
        # Construction archives are generated only inside this born producer.
        # Runtime wrappers must be positively selected by the original request.
        if self.role == 'construction_packet':
            _require(self.path.is_relative_to(root), _ERROR)
        else:
            expected = self.request['execution_adapter']['runtime_source_bundle']
            _require(self.reference == expected, _ERROR)
        external_raw, external_generation = None, None
        if external_source is not None:
            external_raw, external_generation = _external_source(self.policy, external_source, entry, authority_ref)
        store = Path(self.policy['generation_store'])
        manifest_name = 'generated-manifest-'+hashlib.sha256(manifest_bytes).hexdigest()+'.json'
        with _opened(store, directory=True, protected=True) as (ledger, ledger_info):
            _publish_manifest(ledger, _identity(ledger_info), manifest_name, manifest_bytes, store)
            manifest_ref = _raw_bytes(manifest_bytes, store/manifest_name)
            value = dict(schema_version=SCHEMA, intent_raw_ref=authority['intent_raw_ref'],
                storage_authority_raw_ref=authority_ref, producer_generation_raw_ref=generation_ref,
                producer_root=str(root), request_digest=authority['request_digest'],
                bundle_raw_ref=dict(path=str(self.path), sha256=self.reference['digest'], size_bytes=self.info.st_size),
                expected_reference=dict(self.reference), role=self.role, manifest_raw_ref=manifest_ref,
                manifest_digest=manifest['manifest_digest'], entry=dict(entry),
                external_source_raw_ref=external_raw, external_generation_raw_ref=external_generation,
                publication_digest='')
            value['publication_digest'] = canonical_digest(value, digest_field='publication_digest')
            name = 'generated-publication-'+value['publication_digest'][7:]+'.json'
            self.guard()
            try:
                _write(ledger, name, value, parent_identity=_identity(ledger_info))
            except FileExistsError:
                existing, _ = load_document(store/name, maximum=65536, protected=True)
                _require(existing == value, _ERROR)
            self.guard()
        _, raw = load_document(store/name, maximum=65536, protected=True)
        return raw


def _publish_manifest(parent, parent_identity, name, raw, store):
    try:
        fd, identity = _new_file(parent, name, parent_identity=parent_identity)
    except FileExistsError:
        value, reference = load_document(store/name, maximum=4*1024*1024, protected=True)
        _require(value == json.loads(raw) and reference == _raw_bytes(raw, store/name), _ERROR)
        return
    complete = False
    try:
        view = memoryview(raw)
        while view:
            _named(parent, parent_identity, name, fd, identity)
            written = os.write(fd, view[:1024*1024])
            _require(0 < written <= len(view), _ERROR)
            view = view[written:]
        _named(parent, parent_identity, name, fd, identity)
        os.fsync(fd)
        _named(parent, parent_identity, name, fd, identity)
        os.fsync(parent)
        complete = True
    finally:
        incoming = sys.exc_info()[1]
        if not complete:
            try:
                _named(parent, parent_identity, name, fd, identity)
                os.unlink(name, dir_fd=parent)
            except (OSError, access.SceneRetirementAccessError) as error:
                if incoming is not None:
                    incoming.add_note(str(error))
        failure = access._close_owned(fd, identity)
        if failure and incoming is None:
            raise access.SceneRetirementAccessError(failure)
        if failure and incoming is not None:
            incoming.add_note(failure)


def _external_source(policy, path, entry, authority_ref):
    path = _canonical(str(path))
    generation, reference = _generation(policy, path, directory=False)
    source = selected_document(generation['source_publication_raw_ref'], maximum=65536)
    _require(source.get('schema_version') == 'scene_preparation_storage_authority.v1'
             and generation['source_publication_raw_ref'] == authority_ref
             and generation['digest'] == entry['sha256']
             and generation['size_bytes'] == entry['size_bytes'], _ERROR)
    from .task_evaluation_launch_preparation_worker import collect_preparation_references
    # Dynamic runtime layers are not direct request rows. Their exact sealed
    # wrapper relation is independently proved by the native manifest entry.
    request = selected_document(source['submission_request_raw_ref'], maximum=65536)
    _require(type(request) is dict and collect_preparation_references(request), _ERROR)
    with _opened(path) as (fd, info):
        digest = hashlib.sha256()
        remaining = info.st_size
        identity = _identity(info)
        snapshot = _snapshot(info)
        while remaining:
            _guard(fd, identity)
            chunk = os.read(fd, min(1024*1024, remaining))
            _guard(fd, identity)
            _require(chunk, _ERROR)
            digest.update(chunk)
            remaining -= len(chunk)
        _require(_snapshot(os.fstat(fd)) == snapshot and 'sha256:'+digest.hexdigest() == entry['sha256'], _ERROR)
    return dict(path=str(path), sha256=entry['sha256'], size_bytes=entry['size_bytes']), reference


@contextmanager
def bundle_lifetime(*, bundle_path, request, expected_reference, role, destination, content_store_root):
    policy = access._policy()
    if policy is None:
        yield None
        return
    path = _canonical(str(bundle_path))
    hint = _canonical(str(destination)).parent
    with access.scene_access(path, hint, *(() if content_store_root is None else (content_store_root,))):
        producer = _producer(policy, hint)
        with _opened(path.parent, directory=True) as (parent, parent_info), _opened(path) as (fd, info):
            use = _BundleUse(path=path, fd=fd, info=info, parent=parent, parent_info=parent_info,
                policy=policy, producer=producer, request=request, reference=expected_reference, role=role)
            use.verify_reference()
            yield use
            use.guard()


def _verified_record(record, policy, consent, allowance):
    _require(type(record) is dict and set(record) == _FIELDS
             and record['schema_version'] == SCHEMA
             and record['publication_digest'] == canonical_digest(record, digest_field='publication_digest')
             and record['intent_raw_ref'] == consent['intent_raw_ref'], _ERROR)
    from .task_evaluation_scene_retirement_cache import _storage_history
    for name in ('storage_authority_raw_ref', 'producer_generation_raw_ref', 'manifest_raw_ref'):
        allowance.charge('local_bytes', record[name]['size_bytes'])
    authority = selected_document(record['storage_authority_raw_ref'], maximum=65536)
    request = _storage_history(authority, allowance)
    original = selected_document(record['producer_generation_raw_ref'], maximum=65536)
    root = _canonical(record['producer_root'])
    current, _ = _generation(policy, root, directory=True)
    _require(original['generation_id'] == current['generation_id']
             and original['canonical_path'] == current['canonical_path'] == str(root)
             and original['source_storage_authority_raw_ref'] == record['storage_authority_raw_ref']
             and current['source_storage_authority_raw_ref'] == record['storage_authority_raw_ref']
             and current['owner_raw_ref'] == authority['intent_raw_ref'] == record['intent_raw_ref']
             and current['birth_request_raw_ref'] == authority['attempt_raw_ref']
             and authority['request_digest'] == record['request_digest'], _ERROR)
    retained = selected_document(record['manifest_raw_ref'], maximum=4*1024*1024, protected=True)
    _require(retained['manifest_digest'] == record['manifest_digest'], _ERROR)
    return request, retained


def validate_publication(record, *, policy, consent, allowance):
    request, retained = _verified_record(record, policy, consent, allowance)
    from . import task_evaluation_native_arena_preparation_adapter as adapter
    path = _canonical(record['bundle_raw_ref']['path'])
    _require(record['expected_reference']['digest'] == record['bundle_raw_ref']['sha256']
             and record['expected_reference']['size_bytes'] == record['bundle_raw_ref']['size_bytes'], _ERROR)
    if record['role'] == 'construction_packet':
        _require(path.is_relative_to(Path(record['producer_root'])), _ERROR)
    else:
        _require(record['role'] == 'runtime_source'
                 and record['expected_reference'] == request['execution_adapter']['runtime_source_bundle'], _ERROR)
    with _opened(path.parent, directory=True) as (parent, parent_info), _opened(path) as (fd, info):
        use = _BundleUse(path=path, fd=fd, info=info, parent=parent, parent_info=parent_info,
            policy=policy, producer=None, request=request, reference=record['expected_reference'],
            role=record['role'], allowance=allowance)
        use.verify_reference()
        with use.archive() as archive:
            actual = adapter._manifest_from_archive(archive, request=request, expected_role=record['role'])
        _require(actual == retained and record['entry'] in actual['entries'], _ERROR)
    return dict(digest=record['entry']['sha256'], size_bytes=record['entry']['size_bytes'],
                intent_raw_ref=record['intent_raw_ref'], request_digest=record['request_digest'])


def validate_external_publication_source(record, *, source_path, digest, size_bytes):
    _require(type(record) is dict and set(record) == _FIELDS and record['schema_version'] == SCHEMA
             and record['publication_digest'] == canonical_digest(record, digest_field='publication_digest')
             and record['external_source_raw_ref'] is not None
             and record['entry']['sha256'] == digest and record['entry']['size_bytes'] == size_bytes, _ERROR)
    policy = access._policy()
    _require(policy is not None, _ERROR)
    original = _canonical(record['external_source_raw_ref']['path'])
    with access.scene_access(original, source_path):
        raw, generation = _external_source(policy, original, record['entry'], record['storage_authority_raw_ref'])
        _require(raw == record['external_source_raw_ref'] and generation == record['external_generation_raw_ref'], _ERROR)
        with _opened(original) as (_, old), _opened(_canonical(str(source_path))) as (_, new):
            _require(_identity(old) == _identity(new) and old.st_size == new.st_size == size_bytes, _ERROR)
    return True
