# Covers (for impacted-test selection):
#   src/blueprint_pipeline/task_evaluation_scene_lifecycle_plan.py
#   src/blueprint_pipeline/task_evaluation_scene_lifecycle_pool.py
#   src/blueprint_pipeline/task_evaluation_scene_lifecycle_measurement.py
"""Actual acquisition -> reviewed child -> measured KEEP-only scene reports."""
import hashlib
import os
import time

import pytest
from pathlib import Path

from tests.scene_lifecycle_fixture_support import rebase_graph, stable_shared_ancestors
from tests.test_scene_lifecycle_plan import context_fixture
from tests.test_scene_compilation_owner_preparations import ROLES
from tests.test_scene_compilation_native_owner import fixture as native_fixture
from tests.test_scene_downstream_terminal import fixture as terminal_fixture
from tests.test_scene_source_family_website import publication_fixture
from tests.test_scene_source_family_sam import adopted_final_fixture


def full_connected_finished_scene():
    """One common intent, all actual producer families, retained original SAM."""
    args = connected_native_terminal()
    website = rebase_graph(publication_fixture(), '/retained', remote_digest_replacements={
        'sha256:'+letter*64: 'sha256:'+hashlib.sha256(('website-'+letter).encode()).hexdigest()
        for letter in 'de'})
    original = adopted_final_fixture()
    assert args['seed_records']['intent'] == website['seed_records']['intent'] == original['seed_records']['intent']
    for other in (website, original):
        for group in ('seed_records', 'downstream_records', 'source_records'):
            args.setdefault(group, {})
            for role, rows in other[group].items():
                if role in ('intent', 'projection'):
                    continue
                args[group].setdefault(role, [])
                args[group][role] += [row for row in rows if row not in args[group][role]]
    from tests.test_scene_inventory_history import event, project
    history = {'intent_id': args['intent_id'], 'roots': args['roots'], 'records': args['seed_records']}
    project(history, event(history, updates={'status': 'completed', 'phase': 'terminal'}))
    return args


def installed(tmp_path, args):
    from tests.test_scene_source_family_website import fixture as base_fixture
    base = base_fixture()
    args['roots'] = dict(base['roots'], **args['roots'])
    args.setdefault('parent_routes', base['parent_routes'])
    args.setdefault('retained_metadata_roots', base['retained_metadata_roots'])
    args.setdefault('source_records', base['source_records'])
    args.setdefault('bridge_records', {role: [] for role in ROLES})
    args = rebase_graph(args, tmp_path)
    context, _ = context_fixture(tmp_path)
    context.update(roots=args['roots'], parent_routes=args['parent_routes'],
                   retained_metadata_roots=[str(tmp_path.resolve())])
    context['primary_queue_contracts'][0]['root_path'] = args['roots']['preparation_queue_root']
    for row in context['auxiliary_queue_contracts']:
        row['root_path'] = args['roots']['preparation_queue_root' if row['family'] == 'preparation' else 'sam_queue_root']
    for row in context['reference_family_contracts']:
        row['queue_root'] = args['roots'][row['family'] + '_queue_root']
    metadata, payload = {}, set()
    for group in ('seed_records', 'downstream_records', 'source_records', 'bridge_records'):
        for role, rows in args.get(group, {}).items():
            rows = ([] if rows is None else [rows]) if role in ('intent', 'projection') else rows
            for path, raw in rows:
                target = Path(path)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(raw)
                if target.suffix == '.json':
                    assert path not in metadata or metadata[path][1] == raw
                    metadata[path] = role, raw
                else:
                    payload.add(path)
    context['retained_metadata_files'] = [dict(role=role, path=path)
        for path, (role, _) in metadata.items() if role not in ('intent', 'projection')]
    # Actual tiny regular payloads are statted, never opened. Two names and two
    # distinct scene workspaces deliberately share one physical inode.
    first = Path(args['roots']['preparation_input_root']) / 'prep-1' / 'tiny.payload'
    first.parent.mkdir(parents=True, exist_ok=True)
    first.write_bytes(b'x')
    shared = first.parent / 'shared.payload'
    os.link(first, shared)
    other = Path(args['roots']['activation_output_root']) / 'activation-1' / 'tiny.payload'
    other.parent.mkdir(parents=True, exist_ok=True)
    os.link(first, other)
    payload.update(map(str, (first, shared, other)))
    return args, context, metadata, payload


def actual_report(tmp_path, monkeypatch, args):
    args, context, metadata, payload = installed(tmp_path, args)
    stable_shared_ancestors(monkeypatch, tmp_path)
    opened = os.open
    metadata_reads = []
    def guarded(name, flags, *a, **kw):
        if not flags & os.O_DIRECTORY:
            assert not any(Path(path).name == str(name) for path in payload), 'payload was opened'
            metadata_reads.append(str(name))
        return opened(name, flags, *a, **kw)
    monkeypatch.setattr(os, 'open', guarded)
    from blueprint_pipeline.task_evaluation_scene_lifecycle_plan import build_scene_lifecycle_plan
    start = time.monotonic()
    report = build_scene_lifecycle_plan(intent_id=args['intent_id'], context=context, observed_at_epoch=900000)
    elapsed = time.monotonic() - start
    assert 'historical_lineage' in report, report
    assert 'metadata_changed_after_observation' not in report['blockers']
    assert report['action'] == 'KEEP' and report['mutations'] == 0
    assert len(report['family_obligations']) == 10
    assert all(row['action'] == 'KEEP' for row in report['family_obligations'])
    for flag in ('references_clear', 'scene_inventory_complete', 'retirement_eligible',
                 'cleanup_authorized', 'process_fences_held', 'restore_verified', 'owner_consent_verified'):
        assert report[flag] is False
    assert metadata_reads
    return report, args, metadata, elapsed


def connected_native_terminal():
    # Independent old test families reuse placeholder hashes with different
    # remote byte promises. Give native remote placeholders distinct identities
    # before composing; no proof/size mismatch is weakened by the public join.
    native = rebase_graph(native_fixture(), '/retained', remote_digest_replacements={
        'sha256:'+letter*64: 'sha256:'+hashlib.sha256(('native-'+letter).encode()).hexdigest()
        for letter in 'cdef'})
    terminal = terminal_fixture(pointer=True)
    native['seed_records']['attempts'] += terminal['seed_records']['attempts']
    for role, rows in terminal['seed_records'].items():
        if role not in ('intent', 'projection', 'attempts'):
            native['seed_records'][role] = rows
    for role, rows in terminal['downstream_records'].items():
        native['downstream_records'][role] += rows
    return native


@pytest.mark.slow
def test_terminal_scene_report_acquires_exact_members_once_with_unique_inode_bytes(tmp_path, monkeypatch):
    # One real-shaped completed scene traverses all retained producer families
    # through the actual planner and advancing clock. No child joins, readers,
    # members, budgets, caps or elapsed-clock semantics are replaced.
    report, args, metadata, elapsed = actual_report(tmp_path, monkeypatch, full_connected_finished_scene())
    historical = report['historical_lineage']
    source = historical['source_family_inventory']
    downstream = source['downstream_inventory']
    assert elapsed < 30
    assert report['finished_observation']['status'] == 'completed'
    assert report['finished_observation']['finished_for_cleanup_authority'] is False
    assert downstream['terminal_observations'][0]['archive_binding_verified']
    assert downstream['activation_observations'][0]['status'] == 'matched_retained_bytes'
    assert any(row.get('owner_metadata_binding_verified') for row in historical['compilation_native_owner_observations'])
    assert any(row.get('compiler_output_metadata_binding_verified') for row in historical['compilation_native_owner_observations'])
    assert historical['preparation_handoff_observations'][0]['pre_handoff_binding_verified']
    assert source['website_observations'][0]['capture_binding_verified']
    assert source['publication_observations'][0]['historical_publication_binding_verified']
    assert source['adoption_observations'][0]['prefix_binding_verified']
    final = next(row for row in source['sam_observations'] if row['role'] == 'sam_final')
    assert final['source_provenance'][0]['json_pointer'] == '/advancement/sam31_preparation_result'
    assert len(source['original_phase_observations']) == 10
    assert source['original_owner_transfer_authorized'] is False
    assert {kind for row in report['measured_members'] for kind in row['kinds']} >= {
        'capture_pipeline', 'administrative_source_workspace', 'preparation_workspace',
        'configuration_progression_workspace', 'activation_workspace', 'prepared_objects',
        'compilation_workspace', 'sam_original_child', 'launch_canary_workspace'}
    current = next(row for row in report['family_obligations'] if row['family'] == 'sam_current_child')
    assert current['status'] == 'not_observed_within_scope' and current['measured_allocated_bytes'] is None
    assert current['action'] == 'KEEP'
    # Independently sum actual unique inodes across only the measured paths.
    # Payloads remain stat-only, and hardlinks observed in multiple workspaces
    # retain explicit sharing without inflating allocated-byte attribution.
    from blueprint_pipeline.control_plane_disk_usage import allocated_bytes
    inodes = {}
    for row in report['measured_members']:
        target = Path(row['path'])
        if target.exists():
            for path in ([target, *target.rglob('*')] if target.is_dir() else [target]):
                info = path.stat()
                inodes[info.st_dev, info.st_ino] = allocated_bytes(info)
    assert report['unique_observed_allocated_bytes'] == sum(inodes.values())
    assert report['sharing'] and all(row['action'] == 'KEEP' for row in report['sharing'])
    versions = historical['raw_versions'] + source['raw_versions'] + downstream['raw_versions']
    for path, (role, raw) in metadata.items():
        if role in {'native_activation_results', 'terminal_publications', 'submission_publications', 'sam_adoptions'}:
            proof = next(proof for proof in versions if proof['path'] == path)
            assert proof['sha256'] == 'sha256:' + hashlib.sha256(raw).hexdigest()
            assert proof['size_bytes'] == len(raw)
    assert args['intent_id'] == report['intent_id']
