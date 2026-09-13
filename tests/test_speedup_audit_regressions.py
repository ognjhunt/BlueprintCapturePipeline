"""Contracts from the 2026-09-13 speed-up audit of #1903/#1904/#1905 (hermetic, no paid calls).

* a persisted verdict depends on the modules its validator imports by statement and on
  every nested verdict, so a deploy that changes a threshold or a delegated validator
  recomputes instead of reusing an obsolete approval;
* a successful ``None`` verdict is reused rather than mistaken for a miss;
* capsules of different attempts never share an extraction tree, and a damaged newest
  candidate falls back to the next intact history;
* the hard-link tool never replaces bytes whose identity moved after hashing.
"""
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


def test_host_dedup_must_not_replace_bytes_if_keeper_changes_after_hash(tmp_path, monkeypatch):
    from pathlib import Path
    script = Path(__file__).parents[1] / 'scripts/control_plane_hardlink_dedup.py'
    spec = importlib.util.spec_from_file_location('control_plane_hardlink_dedup', script)
    dedup = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dedup)
    keeper, victim = tmp_path / 'a-keeper', tmp_path / 'b-victim'
    keeper.write_bytes(b'original')
    victim.write_bytes(b'original')
    live = [(str(p), p.stat()) for p in [keeper, victim]]
    original = dedup.digest
    def mutate_after_hash(path):
        digest = original(path)
        if path == str(victim):
            keeper.write_bytes(b'modified')  # a concurrent writer after verification
        return digest
    monkeypatch.setattr(dedup, 'digest', mutate_after_hash)
    skipped = []
    dedup.dedup_partition(live, True, skipped, minimum_age_seconds=0)
    assert victim.read_bytes() == b'original'
    assert any('changed during verification' in row for row in skipped), skipped
    assert not os.path.samefile(keeper, victim)


def test_host_dedup_skips_files_inside_the_quiescence_window(tmp_path):
    from pathlib import Path
    script = Path(__file__).parents[1] / 'scripts/control_plane_hardlink_dedup.py'
    spec = importlib.util.spec_from_file_location('control_plane_hardlink_dedup_quiet', script)
    dedup = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dedup)
    keeper, victim = tmp_path / 'a-keeper', tmp_path / 'b-victim'
    keeper.write_bytes(b'same bytes')
    victim.write_bytes(b'same bytes')
    live = [(str(p), p.stat()) for p in [keeper, victim]]
    skipped = []
    assert dedup.dedup_partition(list(live), True, skipped) == (0, 0)  # default window: just-written files wait
    assert not os.path.samefile(keeper, victim) and len(skipped) == 2
    freed, links = dedup.dedup_partition(list(live), True, [], minimum_age_seconds=0)
    assert links == 1 and freed == len(b'same bytes') and os.path.samefile(keeper, victim)
