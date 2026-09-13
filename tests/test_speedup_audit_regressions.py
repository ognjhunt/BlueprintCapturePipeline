"""Offline regressions for the speed-up audit; protecting retained-stage reuse."""
import importlib.util
import json
import os
import sys

import pytest

from blueprint_pipeline import validation_verdict_store as store
from blueprint_pipeline import task_evaluation_release_identity as identity
from blueprint_pipeline.task_evaluation_sam31_prefix_evidence import reuse_verdict
from blueprint_pipeline.validation_file_digests import file_digest_scope
from blueprint_pipeline import semantic_teacher_candidate_discovery as discovery
from tests.test_semantic_teacher_candidate_discovery import (
    CAMERAS, _digest, _workspace, _write, _request, _render, _capsule_launch,
)


@pytest.fixture
def probe_modules(tmp_path, monkeypatch):
    import blueprint_pipeline
    package = tmp_path / 'probe_package'
    package.mkdir()
    monkeypatch.setattr(blueprint_pipeline, '__path__', [str(package), *blueprint_pipeline.__path__])
    monkeypatch.setattr(store, '_package_root', lambda: package)
    monkeypatch.setenv(store.ROOT_ENV, str(tmp_path / 'verdicts'))
    monkeypatch.setattr(identity, 'running_release_commit', lambda: 'c' * 40)

    def load(name, source):
        path = package / (name + '.py')
        path.write_text(source)
        fullname = 'blueprint_pipeline.' + name
        spec = importlib.util.spec_from_file_location(fullname, path)
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, fullname, module)
        spec.loader.exec_module(module)
        return module, path
    return load


def test_data_only_validator_dependency_must_invalidate_verdict(probe_modules):
    rules, path = probe_modules('audit_rules', 'LIMIT = 10\n')
    check, _ = probe_modules('audit_check',
        'from blueprint_pipeline.audit_rules import LIMIT\n'
        'def validate():\n    return {"accepted": 5 < LIMIT}\n')
    with file_digest_scope():
        assert reuse_verdict('audit-data', ('same',), {}, check.validate)['accepted']
    # Simulate a deploy changing an imported acceptance threshold. Both modules
    # were loaded before the tracer, as in the actual controller import path.
    path.write_text('LIMIT = 1\n')
    check.LIMIT = rules.LIMIT = 1
    assert check.validate() == {'accepted': False}
    with file_digest_scope():
        assert reuse_verdict('audit-data', ('same',), {}, check.validate) == {'accepted': False}


def test_nested_validator_dependency_must_invalidate_outer_verdict(probe_modules):
    leaf, path = probe_modules('audit_leaf', 'def validate():\n    return {"accepted": True}\n')
    outer, _ = probe_modules('audit_outer',
        'from blueprint_pipeline.task_evaluation_sam31_prefix_evidence import reuse_verdict\n'
        'from blueprint_pipeline.audit_leaf import validate as leaf\n'
        'def validate():\n    return reuse_verdict("audit-inner", ("same",), {}, leaf)\n')
    with file_digest_scope():
        assert reuse_verdict('audit-outer', ('same',), {}, outer.validate)['accepted']
    path.write_text('def validate():\n    return {"accepted": False}\n')
    outer.leaf = lambda: {'accepted': False}
    # The inner verdict knows its validator changed, but the outer verdict must
    # also know: the actual adoption validator nests render/tracking verdicts.
    with file_digest_scope():
        assert reuse_verdict('audit-outer', ('same',), {}, outer.validate) == {'accepted': False}


def _inputs():
    return ({c: _digest(f'rgb:{c}'.encode()) for c in CAMERAS},
            {c: _digest(f'mask:{c}'.encode()) for c in CAMERAS})


def test_capsules_with_same_scene_prefix_do_not_mix_extracted_histories(tmp_path):
    inputs, masks = _inputs()
    staging, launches = tmp_path / 'staging', tmp_path / 'launches'
    _workspace(staging, 'older', CAMERAS, inputs, masks, review1=dict.fromkeys(CAMERAS, True))
    _workspace(staging, 'newer', CAMERAS, inputs, masks, review1=dict.fromkeys(CAMERAS, False),
               repair=CAMERAS, review2=dict.fromkeys(CAMERAS, False))
    prefix = 'scene-0578bcfa7dd281944480-source-'
    old = _capsule_launch(launches / (prefix + 'old'), staging / 'older')
    new = _capsule_launch(launches / (prefix + 'new'), staging / 'newer')
    os.utime(old, (1000, 1000))
    os.utime(new, (2000, 2000))
    request = _write(tmp_path / 'current.json', _request(CAMERAS, inputs, masks))
    outcome = discovery.discover_retained_candidates(runtime_request_path=request, render=_render(inputs),
        workspace_root=tmp_path / 'absent', capsule_root=launches, output_root=tmp_path / 'discovery')
    assert outcome['candidates_retained'] == len(CAMERAS), json.dumps(outcome['examined'])


def test_missing_newest_candidate_falls_back_to_intact_history(tmp_path):
    inputs, masks = _inputs()
    root = tmp_path / 'workspaces'
    _workspace(root, 'old', CAMERAS, inputs, masks, review1=dict.fromkeys(CAMERAS, True), age=100)
    _workspace(root, 'new', CAMERAS, inputs, masks, review1=dict.fromkeys(CAMERAS, True), age=1)
    (root / 'new' / discovery.RUNTIME / 'semantic_teacher_output/tasks/remove/00000.png').unlink()
    request = _write(tmp_path / 'current.json', _request(CAMERAS, inputs, masks))
    outcome = discovery.discover_retained_candidates(runtime_request_path=request, render=_render(inputs),
        workspace_root=root, output_root=tmp_path / 'discovery')
    assert outcome['candidates_retained'] == len(CAMERAS)
    assert outcome['source_workspace'].endswith('/old')


def test_successful_none_verdict_is_reused_across_operations(tmp_path, monkeypatch):
    monkeypatch.setenv(store.ROOT_ENV, str(tmp_path / 'verdicts'))
    monkeypatch.setattr(identity, 'running_release_commit', lambda: 'c' * 40)
    calls = []
    def validate():
        calls.append(1)
        return None
    for _ in range(2):
        with file_digest_scope():
            reuse_verdict('audit-none', ('same',), {}, validate)
    assert len(calls) == 1


def test_dedup_must_not_replace_bytes_if_source_changes(tmp_path, monkeypatch):
    from blueprint_pipeline import control_plane_file_dedup as dedup
    import struct
    keeper, victim = tmp_path / 'keeper', tmp_path / 'victim'
    keeper.write_bytes(b'original')
    victim.write_bytes(b'original')
    inodes = (keeper.stat().st_ino, victim.stat().st_ino)
    monkeypatch.setattr(dedup.sys, 'platform', 'linux')
    def kernel_refuses_changed_bytes(fd, operation, request, mutate):
        keeper.write_bytes(b'modified')
        assert operation == dedup.FIDEDUPERANGE
        struct.pack_into('=i', request, 48, 1)  # FILE_DEDUPE_RANGE_DIFFERS
    monkeypatch.setattr(dedup.fcntl, 'ioctl', kernel_refuses_changed_bytes)
    assert dedup.deduplicate_pair(keeper, victim, apply=True)['status'] == 'skipped'
    assert victim.read_bytes() == b'original'
    assert (keeper.stat().st_ino, victim.stat().st_ino) == inodes


def test_nested_cache_hit_propagates_dependencies(probe_modules):
    _, path = probe_modules('audit_hit_leaf', 'def validate():\n    return {"accepted": True}\n')
    outer, _ = probe_modules('audit_hit_outer',
        'from blueprint_pipeline.task_evaluation_sam31_prefix_evidence import reuse_verdict\n'
        'from blueprint_pipeline.audit_hit_leaf import validate as leaf\n'
        'def validate():\n    return reuse_verdict("audit-hit-inner", ("same",), {}, leaf)\n')
    with file_digest_scope():
        outer.validate()  # Populate the inner in-memory and persistent entries first.
        reuse_verdict('audit-hit-outer', ('same',), {}, outer.validate)
    path.write_text('def validate():\n    return {"accepted": False}\n')
    outer.leaf = lambda: {'accepted': False}
    with file_digest_scope():
        assert reuse_verdict('audit-hit-outer', ('same',), {}, outer.validate) == {'accepted': False}


def test_dedup_unsupported_never_falls_back_to_hardlinks(tmp_path, monkeypatch):
    from blueprint_pipeline import control_plane_file_dedup as dedup
    left, right = tmp_path / 'left', tmp_path / 'right'
    left.write_bytes(b'same')
    right.write_bytes(b'same')
    monkeypatch.setattr(dedup.sys, 'platform', 'linux')
    def unsupported(*args):
        raise OSError(95, 'unsupported')
    monkeypatch.setattr(dedup.fcntl, 'ioctl', unsupported)
    assert dedup.deduplicate_pair(left, right, apply=True)['reason'] == 'os_error'
    assert left.stat().st_ino != right.stat().st_ino
    left.write_bytes(b'next')
    assert right.read_bytes() == b'same'


def test_dedup_kernel_response_records_shared_extents_without_claiming_freed_bytes(tmp_path, monkeypatch):
    from blueprint_pipeline import control_plane_file_dedup as dedup
    import struct
    left, right = tmp_path / 'left', tmp_path / 'right'
    left.write_bytes(b'same')
    right.write_bytes(b'same')
    monkeypatch.setattr(dedup.sys, 'platform', 'linux')
    def same(fd, operation, request, mutate):
        assert len(request) == 56
        assert struct.unpack_from('=QQHHI', request) == (0, 4, 1, 0, 0)
        struct.pack_into('=Q', request, 40, 4)
    monkeypatch.setattr(dedup.fcntl, 'ioctl', same)
    result = dedup.deduplicate_pair(left, right, apply=True)
    assert result['status'] == 'deduplicated' and result['bytes_deduplicated'] == 4
    assert left.stat().st_ino != right.stat().st_ino


def test_reexported_threshold_dependency_invalidates_verdict(probe_modules):
    _, path = probe_modules('audit_limits', 'LIMIT = 10\n')
    probe_modules('audit_reexport', 'from blueprint_pipeline.audit_limits import LIMIT\n')
    check, _ = probe_modules('audit_reexport_check',
        'from blueprint_pipeline.audit_reexport import LIMIT\n'
        'def validate():\n    return {"accepted": 5 < LIMIT}\n')
    with file_digest_scope():
        assert reuse_verdict('reexport', ('same',), {}, check.validate)['accepted']
    path.write_text('LIMIT = 1\n')
    check.LIMIT = 1
    with file_digest_scope():
        assert reuse_verdict('reexport', ('same',), {}, check.validate) == {'accepted': False}


def test_nested_in_memory_hit_preserves_large_consulted_file(tmp_path, probe_modules):
    from blueprint_pipeline.validation_file_digests import MINIMUM_BYTES
    path = tmp_path / 'consulted.bin'
    path.write_bytes(b'a' * (MINIMUM_BYTES + 1))
    probe_modules('audit_file_leaf',
        'from blueprint_pipeline.task_evaluation_scene_configuration_submission_inputs import sha\n'
        'from pathlib import Path\n'
        f'PATH = Path({str(path)!r})\n'
        'def validate():\n    return {"digest": sha(PATH)}\n')
    outer, _ = probe_modules('audit_file_outer',
        'from blueprint_pipeline.task_evaluation_sam31_prefix_evidence import reuse_verdict\n'
        'from blueprint_pipeline.audit_file_leaf import validate as leaf\n'
        'def validate():\n    return reuse_verdict("file-inner", ("same",), {}, leaf)\n')
    with file_digest_scope():
        original = outer.validate()
        assert reuse_verdict('file-outer', ('same',), {}, outer.validate) == original
    path.write_bytes(b'b' * (MINIMUM_BYTES + 1))
    with file_digest_scope():
        assert reuse_verdict('file-outer', ('same',), {}, outer.validate) != original
