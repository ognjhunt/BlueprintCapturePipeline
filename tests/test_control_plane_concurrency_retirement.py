"""Fixture-owner cancellation cannot bypass reader or failed-chain fences."""
from pathlib import Path

import pytest


def test_fixture_namespace_keeps_production_retention_classes(tmp_path):
    from scripts.control_plane_concurrency_retirement import namespace_classifier
    owned=tmp_path/'owned';owned.mkdir()
    cache=owned/'prepared-references';cache.mkdir()
    classify=namespace_classifier(owned_root=owned,mapping={
        cache:'/var/lib/blueprint/task-evaluation-inputs/prepared-references'})
    assert classify(str(cache),expected='cache',code='wrong_class').storage_class=='cache'
    with pytest.raises(ValueError):classify(str(owned),expected='cache',code='wrong_class')
    with pytest.raises(ValueError):classify(str(tmp_path),expected='cache',code='wrong_class')
    with pytest.raises(ValueError):classify(str(cache),expected='evidence_hot',code='wrong_class')
    with pytest.raises(ValueError):classify(str(owned/'..'/'outside'),expected='cache',code='wrong_class')


def test_fixture_namespace_rejects_traversal_back_into_an_allowed_root(tmp_path):
    from scripts.control_plane_concurrency_retirement import namespace_classifier
    owned=tmp_path/'owned';owned.mkdir()
    cache=owned/'prepared-references';cache.mkdir()
    classify=namespace_classifier(owned_root=owned,mapping={
        cache:'/var/lib/blueprint/task-evaluation-inputs/prepared-references'})
    with pytest.raises(ValueError,match='retirement_path_not_owned'):
        classify(str(cache/'..'/'prepared-references'/'outside'),expected='cache',code='wrong_class')


def test_failed_chain_cannot_release_pins_or_delete_bytes(tmp_path):
    from scripts.control_plane_concurrency_retirement import retire_fixture_chains
    root=tmp_path/'control';root.mkdir()
    marker=root/'.concurrency-harness-owner';marker.write_text('owned')
    keep=root/'evidence';keep.write_bytes(b'failed original bytes')
    with pytest.raises(ValueError,match='retirement_chain_incomplete'):
        retire_fixture_chains(control_root=root,scenes=[{'stages':[]}],
            source_commit='a'*40,producer_pids=[],process_root=tmp_path/'unavailable-proc')
    assert keep.read_bytes()==b'failed original bytes'


def test_live_producer_prevents_every_retirement_effect(tmp_path):
    import os
    from scripts.control_plane_concurrency_load_test import REQUIRED_STAGES
    from scripts.control_plane_concurrency_retirement import retire_fixture_chains
    root=tmp_path/'control';root.mkdir()
    (root/'.concurrency-harness-owner').write_text('owned')
    scene={'stages':[{'stage':stage,'status':'completed'} for stage in REQUIRED_STAGES[:-1]]}
    with pytest.raises(ValueError,match='retirement_producer_still_alive'):
        retire_fixture_chains(control_root=root,scenes=[scene],source_commit='a'*40,
            producer_pids=[os.getpid()],process_root=tmp_path/'unavailable-proc')


def test_missing_producer_registry_cannot_assert_a_completed_child(tmp_path):
    from scripts.control_plane_concurrency_load_test import create_run_roots, REQUIRED_STAGES
    from scripts.control_plane_concurrency_retirement import retire_fixture_chains
    roots=create_run_roots(tmp_path/'control',tmp_path/'objects',tmp_path/'workers')
    scene={'stages':[{'stage':stage,'status':'completed'} for stage in REQUIRED_STAGES[:-1]]}
    with pytest.raises(ValueError,match='retirement_producer_registry_invalid'):
        retire_fixture_chains(control_root=roots['control_plane'],scenes=[scene],source_commit='a'*40,
            producer_pids=[],process_root=tmp_path/'unavailable-proc')
