"""Native generated-member provenance, never cleanup or execution authority.

One actual born producer and retained native archive bind the generated digest.
The cache module still owns its regular-generation and hardlink protocol.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import secrets
import stat
import sys
import zipfile
from contextlib import contextmanager, ExitStack
from pathlib import Path

from .decision_evidence_contracts import canonical_digest
from . import task_evaluation_scene_retirement_access as access
from .task_evaluation_scene_retirement_access import _canonical, _identity, _opened, _require
from .task_evaluation_scene_retirement_authority import load_document, selected_document
from .task_evaluation_scene_retirement_generations import _guard, _new_file, _named, _write

SCHEMA = 'scene_generated_content_publication.v1'
_ERROR = 'scene_retirement_generated_source_unproven'


class _UnregisteredExternal(Exception):
    """A native readable source supplies no generation/owner publication."""
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
        size = min(size, max(0, self.use.info.st_size-os.lseek(self.use.fd, 0, os.SEEK_CUR)))
        if self.use.allowance is not None:
            self.use.allowance.charge('local_bytes', size)
        raw = os.read(self.use.fd, size)
        self.use.guard()
        return raw

    def seek(self, offset, whence=0):
        self.use.guard()
        value = os.lseek(self.use.fd, offset, whence)
        self.use.guard()
        return value

    def tell(self):
        return self.seek(0, os.SEEK_CUR)


class _TemporaryWriter:
    """One NEW native member token, retained through its cache publication."""
    def __init__(self, use, path, maximum):
        self.use, self.path = use, _canonical(str(path))
        self.maximum, self.written, self.unlinked = maximum, 0, False
        self.stack = ExitStack()
        try:
            self.parent, info = self.stack.enter_context(_opened(self.path.parent, directory=True))
            self.parent_identity = _identity(info)
            use.guard()
            _guard(self.parent, self.parent_identity)
            fd = os.open(self.path.name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
                         0o440, dir_fd=self.parent)
            # No token ownership exists until independent named expectation
            # AND the first fstat match. A failure cannot authorize close.
            _guard(self.parent, self.parent_identity)
            expected = os.stat(self.path.name, dir_fd=self.parent, follow_symlinks=False)
            observed = os.fstat(fd)
            _require(stat.S_ISREG(expected.st_mode) and expected.st_nlink == 1
                     and expected.st_size == 0 and _identity(expected) == _identity(observed), _ERROR)
            self.fd, self.identity = fd, _identity(observed)
            self.guard()
        except BaseException:
            if hasattr(self, 'identity'):
                access._close_owned(self.fd, self.identity)
            self.stack.close()
            raise

    def guard(self):
        self.use.guard()
        _named(self.parent, self.parent_identity, self.path.name, self.fd, self.identity)
        info = os.fstat(self.fd)
        _require(info.st_size == self.written and stat.S_IMODE(info.st_mode) == 0o440, _ERROR)

    def write(self, raw):
        self.guard()
        _require(0 < len(raw) <= 1024*1024 and self.written+len(raw) <= self.maximum, _ERROR)
        count = os.write(self.fd, raw)
        _require(type(count) is int and 0 < count <= len(raw), _ERROR)
        self.written += count
        self.guard()
        return count

    def fsync(self):
        self.guard()
        _require(self.written == self.maximum, _ERROR)
        os.fsync(self.fd)
        self.guard()

    def unlink(self):
        if self.unlinked:
            return
        self.guard()
        os.unlink(self.path.name, dir_fd=self.parent)
        self.unlinked = True
        _guard(self.parent, self.parent_identity)
        os.fsync(self.parent)

    def close(self):
        incoming = sys.exc_info()[1]
        failure = access._close_owned(self.fd, self.identity)
        try:
            self.stack.close()
        finally:
            if failure and incoming is None:
                raise access.SceneRetirementAccessError(failure)
            if failure and incoming is not None:
                incoming.add_note(failure)


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
            count = min(1024*1024, remaining)
            if self.allowance is not None:
                self.allowance.charge('local_bytes', count)
            chunk = os.read(self.fd, count)
            self.guard()
            _require(chunk, _ERROR)
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
        with _opened(store, directory=True) as (ledger, ledger_info):
            _require((ledger_info.st_uid, ledger_info.st_gid) == access._service_identity()
                     and stat.S_IMODE(ledger_info.st_mode) == 0o700, _ERROR)
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
                existing, _ = load_document(store/name, maximum=65536)
                _require(existing == value, _ERROR)
            self.guard()
        _, raw = load_document(store/name, maximum=65536)
        return raw


    @contextmanager
    def temporary(self, path, maximum):
        _require(type(maximum) is int and 0 <= maximum <= 200*1024**3, _ERROR)
        writer = _TemporaryWriter(self, path, maximum)
        try:
            yield writer
        finally:
            writer.close()

    def _project_unowned_external(self, source, cached, target, entry):
        """Preserve verified native reuse, with no generation/owner grant."""
        from . import task_evaluation_scene_retirement_cache as cache
        source, cached = _canonical(str(source)), _canonical(str(cached))
        with access.scene_access(source), _opened(source.parent, directory=True) as (original_parent, original_info), \
                _opened(source) as (fd, info), _opened(cached.parent, directory=True) as (parent, parent_info):
            identity, snapshot = _identity(info), _snapshot(info)
            _require(info.st_size == entry['size_bytes'] and stat.S_IMODE(info.st_mode) & 0o222 == 0, _ERROR)
            digest, remaining = hashlib.sha256(), info.st_size
            while remaining:
                self.guard()
                _named(original_parent, _identity(original_info), source.name, fd, identity)
                chunk = os.read(fd, min(1024*1024, remaining))
                _named(original_parent, _identity(original_info), source.name, fd, identity)
                _require(chunk, _ERROR)
                digest.update(chunk)
                remaining -= len(chunk)
            _require(_snapshot(os.fstat(fd)) == snapshot and 'sha256:'+digest.hexdigest() == entry['sha256'], _ERROR)
            self.guard()
            _named(original_parent, _identity(original_info), source.name, fd, identity)
            _guard(parent, _identity(parent_info))
            os.link(source.name, cached.name, src_dir_fd=original_parent, dst_dir_fd=parent, follow_symlinks=False)
            _named(parent, _identity(parent_info), cached.name, fd, identity)
            os.fsync(parent)
            self.guard()
        cache.project_content(cached, target, authority=self.producer[1]['source_storage_authority_raw_ref'])
        return True

    def publish_external_member(self, *, source, cached, target, manifest_bytes, manifest, entry):
        if self.producer is None:
            return False
        from . import task_evaluation_scene_retirement_cache as cache
        with access.scene_access(source):
            try:
                authority = self.publish(manifest_bytes=manifest_bytes, manifest=manifest, entry=entry,
                                         external_source=source)
            except _UnregisteredExternal:
                return self._project_unowned_external(source, cached, target, entry)
        if authority is None:
            return False
        source, cached = _canonical(str(source)), _canonical(str(cached))
        temporary = '.'+cached.name+'.generated-'+secrets.token_hex(16)+'.pending'
        placed = False
        with _opened(source.parent, directory=True) as (original_parent, original_parent_info), \
                _opened(source) as (fd, info), _opened(cached.parent, directory=True) as (parent, parent_info):
            identity, original_identity, parent_identity = _identity(info), _identity(original_parent_info), _identity(parent_info)
            try:
                self.guard()
                _named(original_parent, original_identity, source.name, fd, identity)
                _guard(parent, parent_identity)
                os.link(source.name, temporary, src_dir_fd=original_parent, dst_dir_fd=parent, follow_symlinks=False)
                placed = True
                _named(parent, parent_identity, temporary, fd, identity)
                cache.publish_content_generation(cached, cached.parent/temporary,
                    digest=entry['sha256'], size_bytes=entry['size_bytes'], authority=authority)
                self.guard()
            finally:
                if placed:
                    _named(parent, parent_identity, temporary, fd, identity)
                    os.unlink(temporary, dir_fd=parent)
                    _guard(parent, parent_identity)
                    os.fsync(parent)
        cache.project_content(cached, target, authority=authority)
        return True


def _publish_manifest(parent, parent_identity, name, raw, store):
    # Full immutable metadata is exposed only after retained-FD write/fsync.
    temporary = '.generated-manifest-'+secrets.token_hex(16)+'.pending'
    fd, identity = _new_file(parent, temporary, parent_identity=parent_identity)
    pending = True
    try:
        view = memoryview(raw)
        while view:
            _named(parent, parent_identity, temporary, fd, identity)
            written = os.write(fd, view[:1024*1024])
            _require(0 < written <= len(view), _ERROR)
            view = view[written:]
        _named(parent, parent_identity, temporary, fd, identity)
        owner = os.fstat(parent)
        os.fchown(fd, owner.st_uid, owner.st_gid)
        _named(parent, parent_identity, temporary, fd, identity)
        os.fsync(fd)
        _named(parent, parent_identity, temporary, fd, identity)
        try:
            os.link(temporary, name, src_dir_fd=parent, dst_dir_fd=parent, follow_symlinks=False)
        except FileExistsError:
            value, reference = load_document(store/name, maximum=4*1024*1024)
            _require(value == json.loads(raw) and reference == _raw_bytes(raw, store/name), _ERROR)
        else:
            _named(parent, parent_identity, name, fd, identity)
        _named(parent, parent_identity, temporary, fd, identity)
        os.unlink(temporary, dir_fd=parent)
        pending = False
        _guard(parent, parent_identity)
        os.fsync(parent)
    finally:
        incoming = sys.exc_info()[1]
        if pending:
            try:
                _named(parent, parent_identity, temporary, fd, identity)
                os.unlink(temporary, dir_fd=parent)
            except (OSError, access.SceneRetirementAccessError) as error:
                if incoming is not None:
                    incoming.add_note(str(error))
                else:
                    raise access.SceneRetirementAccessError(_ERROR) from error
        failure = access._close_owned(fd, identity)
        if failure and incoming is None:
            raise access.SceneRetirementAccessError(failure)
        if failure and incoming is not None:
            incoming.add_note(failure)


def _external_source(policy, path, entry, authority_ref):
    path = _canonical(str(path))
    source = selected_document(authority_ref, maximum=65536)
    request = selected_document(source['submission_request_raw_ref'], maximum=65536)
    try:
        generation, reference = _generation(policy, path, directory=False)
    except FileNotFoundError:
        # Actual native preparation projection and its fixed default CAS:
        # select the authenticated preparation, never scan for matching bytes.
        producer = _producer(policy, path.parent)
        if producer is None:
            raise _UnregisteredExternal()
        prep, current, _ = producer
        _require(prep.name == request['preparation_id']
                 and current['source_storage_authority_raw_ref'] == authority_ref
                 and path.name == entry['sha256'][7:], _ERROR)
        native_cache = prep.parent/'content-addressed'/'sha256'/entry['sha256'][7:]
        try:
            generation, reference = _generation(policy, native_cache, directory=False)
        except FileNotFoundError as error:
            raise _UnregisteredExternal() from error
        with _opened(native_cache) as (_, cached), _opened(path) as (_, projected):
            _require(_identity(cached) == _identity(projected), _ERROR)
    selected = selected_document(generation['source_publication_raw_ref'], maximum=65536)
    if selected.get('schema_version') == SCHEMA:
        _require(set(selected) == _FIELDS
                 and selected['publication_digest'] == canonical_digest(selected, digest_field='publication_digest'), _ERROR)
        if selected['storage_authority_raw_ref'] != authority_ref:
            raise _UnregisteredExternal()
        verified = _validate_current_publication(selected, policy)
        _require(selected['storage_authority_raw_ref'] == authority_ref
                 and verified['digest'] == entry['sha256'] and verified['size_bytes'] == entry['size_bytes'], _ERROR)
    else:
        _require(selected.get('schema_version') == 'scene_preparation_storage_authority.v1'
                 and selected.get('authority_digest') == canonical_digest(selected, digest_field='authority_digest'), _ERROR)
        if generation['source_publication_raw_ref'] != authority_ref:
            raise _UnregisteredExternal()
    _require(generation['digest'] == entry['sha256'] and generation['size_bytes'] == entry['size_bytes'], _ERROR)
    with _opened(path.parent, directory=True) as (parent, parent_info), _opened(path) as (fd, info):
        digest = hashlib.sha256()
        remaining = info.st_size
        identity = _identity(info)
        snapshot = _snapshot(info)
        while remaining:
            _named(parent, _identity(parent_info), path.name, fd, identity)
            chunk = os.read(fd, min(1024*1024, remaining))
            _named(parent, _identity(parent_info), path.name, fd, identity)
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
    manifest_path = _canonical(record['manifest_raw_ref']['path'])
    _require(manifest_path.parent == Path(policy['generation_store']), _ERROR)
    with _opened(manifest_path) as (_, manifest_info):
        _require((manifest_info.st_uid, manifest_info.st_gid) == access._service_identity()
                 and stat.S_IMODE(manifest_info.st_mode) == 0o600 and manifest_info.st_nlink == 1, _ERROR)
    retained = selected_document(record['manifest_raw_ref'], maximum=4*1024*1024)
    _require(retained['manifest_digest'] == record['manifest_digest'], _ERROR)
    return request, retained


def _native_publication(record, request, retained, policy, allowance=None):
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


def validate_publication(record, *, policy, consent, allowance):
    request, retained = _verified_record(record, policy, consent, allowance)
    return _native_publication(record, request, retained, policy, allowance)


def _validate_current_publication(record, policy):
    _require(type(record) is dict and set(record) == _FIELDS and record['schema_version'] == SCHEMA
             and record['publication_digest'] == canonical_digest(record, digest_field='publication_digest'), _ERROR)
    from . import task_evaluation_scene_retirement_cache as cache
    authority = selected_document(record['storage_authority_raw_ref'], maximum=65536)
    request = selected_document(authority['submission_request_raw_ref'], maximum=65536)
    intent, _ = cache._validate(authority, request, now=cache.time.time())
    root = _canonical(record['producer_root'])
    current, _ = _generation(policy, root, directory=True)
    prior = selected_document(record['producer_generation_raw_ref'], maximum=65536)
    _require(current['generation_id'] == prior['generation_id']
             and current['owner_raw_ref'] == record['intent_raw_ref'] == authority['intent_raw_ref']
             and current['owner_intent_id'] == intent['intent_id']
             and current['birth_request_raw_ref'] == authority['attempt_raw_ref']
             and current['source_storage_authority_raw_ref'] == record['storage_authority_raw_ref']
             and authority['request_digest'] == record['request_digest'], _ERROR)
    retained = selected_document(record['manifest_raw_ref'], maximum=4*1024*1024)
    return _native_publication(record, request, retained, policy)


def publish_runtime_layer_authority(*, request, runtime_source, layer, input_root):
    """Actual native wrapper-derived fetch selector, before payload publication."""
    from . import task_evaluation_native_arena_preparation_adapter as adapter
    _require(type(layer) is dict and type(layer.get('size_bytes')) is int
             and type(layer.get('sha256')) is str, _ERROR)
    reference = request['execution_adapter']['runtime_source_bundle']
    _require(runtime_source['digest'] == reference['digest']
             and runtime_source['size_bytes'] == reference['size_bytes'], _ERROR)
    with bundle_lifetime(bundle_path=runtime_source['materialized_path'], request=request,
            expected_reference=reference, role='runtime_source',
            destination=Path(input_root)/'native-runtime-layer-publication', content_store_root=None) as use:
        if use is None or use.producer is None:
            return None
        with use.archive() as archive:
            manifest = adapter._manifest_from_archive(archive, request=request, expected_role='runtime_source')
            raw = archive.read(adapter.MANIFEST_NAME)
        rows = [entry for entry in manifest['entries']
                if entry.get('external_layer', {}).get('uri') == layer['uri']
                and entry['relative_path'] == layer['relative_path']
                and entry['sha256'] == layer['sha256'] and entry['size_bytes'] == layer['size_bytes']]
        _require(len(rows) == 1, _ERROR)
        return use.publish(manifest_bytes=raw, manifest=manifest, entry=rows[0])


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
