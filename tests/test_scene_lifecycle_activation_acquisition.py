# Covers (for impacted-test selection):
#   src/blueprint_pipeline/task_evaluation_scene_lifecycle_pool.py
#   src/blueprint_pipeline/task_evaluation_scene_lifecycle_acquisition.py
"""Actual acquired activation bytes preserve readable and producer-hashed IDs."""
import hashlib
import json
from pathlib import Path

import pytest

from blueprint_pipeline.control_plane_reference_budget import ReferenceCollectionBudget
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_evaluation_scene_lifecycle_acquisition import Acquisition
from blueprint_pipeline.task_evaluation_scene_lifecycle_pool import Pool, select
from blueprint_pipeline.task_evaluation_scene_lineage_budget import RetainedEmissionBudget
from tests.test_scene_downstream_execution import fixture
from tests.test_scene_inventory_history import pair, seal


def acquired_fixture(base, activation):
    """Rebind the producer-shaped metadata graph, without any payload bytes."""
    args = fixture(activation=activation)
    roots = {key: path.replace('/retained', str(base)) for key, path in args['roots'].items()}
    seed, downstream = args['seed_records'], args['downstream_records']
    seed['events'], seed['projection'] = [], None  # Legitimately unavailable history.
    for role in downstream:
        if role != 'activation_results':
            downstream[role] = []
    for role, rows in seed.items():
        if role in {'intent', 'projection'}:
            if rows is not None:
                seed[role] = rows[0].replace('/retained', str(base)), rows[1]
            continue
        rebound = []
        for oldpath, raw in rows:
            value = json.loads(raw)
            if role == 'preparation_links' and 'scene_configuration_attempt' in value:
                value['scene_configuration_attempt']['path'] = value['scene_configuration_attempt']['path'].replace('/retained', str(base))
                value = seal(value, 'link_digest')
            if role == 'preparation_results':
                for ref in value['references']:
                    ref['materialized_path'] = ref['materialized_path'].replace('/retained', str(base))
                value = seal(value, 'result_digest')
            rebound.append(pair(oldpath.replace('/retained', str(base)), value))
        seed[role] = rebound
    prepared = json.loads(seed['preparation_results'][0][1])
    envelope = json.loads(seed['activation_envelopes'][0][1])
    envelope['request']['preparation']['result_digest'] = prepared['result_digest']
    envelope['request_digest'] = canonical_digest(envelope['request'])
    readable = activation+'-'+envelope['request_digest'][7:]+'.json'
    name = readable if len(readable.encode()) <= 255 else (
        'activation-'+hashlib.sha256(activation.encode()).hexdigest()+'-'+envelope['request_digest'][7:]+'.json')
    seed['activation_envelopes'] = [pair(roots['activation_queue_root']+'/pending/'+name, seal(envelope, 'envelope_digest'))]
    path, raw = seed['configuration_progressions'][0]
    progress = json.loads(raw)
    progress.update(preparation_result_digest=prepared['result_digest'], activation_request_digest=envelope['request_digest'])
    seed['configuration_progressions'] = [pair(path, seal(progress, 'progression_digest'))]
    result = json.loads(downstream['activation_results'][0][1])
    result['preparation_result_digest'] = prepared['result_digest']
    downstream['activation_results'] = [pair(roots['activation_queue_root']+'/results/'+name, seal(result, 'result_digest'))]
    args['roots'] = roots
    return args, name


@pytest.mark.parametrize('activation', ['scene-short', 's'*192])
def test_real_acquisition_and_same_budget_child_keep_full_scene_activation_identity(tmp_path, activation, monkeypatch):
    from blueprint_pipeline import task_evaluation_scene_downstream_inventory as child
    args, name = acquired_fixture(tmp_path.resolve(), activation)
    records = []
    for group in ('seed_records', 'downstream_records'):
        for role, rows in args[group].items():
            rows = ([] if rows is None else [rows]) if role in {'intent', 'projection'} else rows
            for path, raw in rows:
                target = Path(path)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(raw)
                records.append((role, path))
    from tests.scene_lifecycle_fixture_support import stable_shared_ancestors
    stable_shared_ancestors(monkeypatch, tmp_path)
    budget = ReferenceCollectionBudget(monotonic=lambda: 0)
    context = {'roots': args['roots'], 'retained_metadata_files': []}
    with Acquisition(budget, [str(tmp_path.resolve())]) as reader:
        pool = Pool(reader, context, args['intent_id'])
        for role, path in records:
            pool.read('parent_envelopes' if role == 'preparation_envelopes' else
                      'parent_results' if role == 'preparation_results' else role, path)
        seed, downstream, _, _, protected = select(pool.decode(), context, args['intent_id'], budget)
        sink = RetainedEmissionBudget(max_bytes=16*1024*1024, max_rows=10000, max_references=10000, work_budget=budget)
        result = child._join(args['intent_id'], seed, downstream, args['roots'], emission_budget=sink, work_budget=budget)
        assert reader.verify()
    row = result['activation_observations'][0]
    assert row['activation_id'] == activation and row['status'] == 'matched_retained_bytes'
    assert any(p['path'].endswith('/results/'+name) for p in row['source_provenance'])
    assert not any(p['role'] == 'activation_results' for p in protected)
    assert result['cleanup_authorized'] is False
    assert name.startswith('activation-') is (len(activation) == 192)
