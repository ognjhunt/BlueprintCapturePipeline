# Covers (for impacted-test selection):
#   src/blueprint_pipeline/task_evaluation_scene_source_family_contracts.py
#   src/blueprint_pipeline/task_evaluation_scene_downstream_contracts.py
"""Private role-scoped scans retain the predecessor URI identity refusal."""
from __future__ import annotations

import json

import pytest

from blueprint_pipeline.control_plane_reference_budget import ReferenceCollectionBudget
from blueprint_pipeline.task_evaluation_scene_source_family_contracts import (
    Context, SceneSourceFamilyInventoryError,
)
from blueprint_pipeline.task_evaluation_scene_downstream_contracts import SceneDownstreamInventoryError


def test_scoped_source_scan_compares_predecessor_remote_identity():
    budget = ReferenceCollectionBudget._for_scene_lifecycle_plan(monotonic=lambda: 0)
    uri = 's3://fixture/source.bin'
    source = {'nested': {'uri': uri, 'digest': 'sha256:' + 'a' * 64, 'size_bytes': 1}}
    prior = {'nested': {'uri': uri, 'digest': 'sha256:' + 'b' * 64, 'size_bytes': 1}}

    def row(role, value):
        raw = json.dumps(value).encode()
        return value, {'role': role, 'path': f'/retained/{role}.json',
                       'sha256': 'sha256:' + ('c' if role == 'source' else 'd') * 64,
                       'size_bytes': len(raw), 'seal_field': None, 'seal_digest': None}

    decoded = {'source': [row('source', source)], 'predecessor': [row('predecessor', prior)]}
    limits = {'MAX_ROWS': 100, 'MAX_REFERENCES': 100, 'MAX_OUTPUT_BYTES': 100_000}
    context = Context(decoded, {}, limits, 'intent-1', frozenset({'source'}), work_budget=budget)
    context.references(roles=frozenset({'source'}))
    assert len(context.remote) == 1 and context.remote[0]['digest'] == source['nested']['digest']
    context.predecessor_remote_identities([{'uri': uri, 'digest': source['nested']['digest'],
                                            'size_bytes': 1}])
    with pytest.raises(SceneSourceFamilyInventoryError, match='remote_identity_conflict'):
        context.predecessor_remote_identities([{'uri': uri, 'digest': prior['nested']['digest'],
                                                'size_bytes': 1}])
    assert budget.counts['facts'] >= 2


@pytest.mark.parametrize('prior_kind', ['remote', 'raw'])
def test_scoped_source_scan_keeps_predecessor_digest_size_conflict(prior_kind):
    budget = ReferenceCollectionBudget._for_scene_lifecycle_plan(monotonic=lambda: 0)
    digest = 'sha256:' + 'a' * 64
    source = {'uri': 's3://fixture/source.bin', 'digest': digest, 'size_bytes': 1}
    raw = json.dumps(source).encode()
    decoded = {'source': [(source, {'role': 'source', 'path': '/retained/source.json',
                                   'sha256': 'sha256:' + 'c' * 64, 'size_bytes': len(raw),
                                   'seal_field': None, 'seal_digest': None})]}
    limits = {'MAX_ROWS': 100, 'MAX_REFERENCES': 100, 'MAX_OUTPUT_BYTES': 100_000}
    context = Context(decoded, {}, limits, 'intent-1', frozenset({'source'}), work_budget=budget)
    context.references(roles=frozenset({'source'}))
    remote = [{'uri': 's3://fixture/predecessor.bin', 'digest': digest, 'size_bytes': 2}]
    originals = [{'path': '/retained/predecessor.bin', 'sha256': digest, 'size_bytes': 2}]
    with pytest.raises(SceneSourceFamilyInventoryError, match='content_size_conflict'):
        context.predecessor_remote_identities(remote if prior_kind == 'remote' else [],
                                              raw_rows=originals if prior_kind == 'raw' else [])


@pytest.mark.parametrize('reference,valid', [
    ({'path': '/retained/frame.png', 'sha256': 'sha256:' + 'a' * 64,
      'role': 'observed_source', 'frame_id': 'frame-1'}, True),
    ({'path': '/retained/missing.bin', 'sha256': 'sha256:' + 'a' * 64}, False),
    ({'path': '/retained/frame.png', 'sha256': 'not-a-digest',
      'role': 'observed_source', 'frame_id': 'frame-1'}, False),
])
def test_incomplete_raw_reference_only_allows_valid_source_frame_descriptor(reference, valid):
    budget = ReferenceCollectionBudget._for_scene_lifecycle_plan(monotonic=lambda: 0)
    value = {'nested': reference}
    raw = json.dumps(value).encode()
    decoded = {'source': [(value, {'role': 'source', 'path': '/retained/source.json',
                                   'sha256': 'sha256:' + 'c' * 64, 'size_bytes': len(raw),
                                   'seal_field': None, 'seal_digest': None})]}
    limits = {'MAX_ROWS': 100, 'MAX_REFERENCES': 100, 'MAX_OUTPUT_BYTES': 100_000}
    context = Context(decoded, {}, limits, 'intent-1', frozenset({'source'}), work_budget=budget)
    if valid:
        context.references(roles=frozenset({'source'}))
        assert context.obligations == []
    else:
        with pytest.raises(SceneDownstreamInventoryError, match='reference_invalid'):
            context.references(roles=frozenset({'source'}))
