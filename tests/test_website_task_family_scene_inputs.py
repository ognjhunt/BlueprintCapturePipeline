"""Family discovery binds runtime code to the protected operator registry."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from blueprint_pipeline.common import write_json
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_evaluation_scene_attempt_evidence import record
from blueprint_pipeline.task_evaluation_scene_intake import stage_scene_intent
from blueprint_pipeline.team_candidate_quick10_registry import (
    REGISTRY_ENV, REGISTRY_SCHEMA, _one_scene_half, profile_for_capture, scene_facts,
)
from blueprint_pipeline.website_task_family_packet import materialize_website_task_family_packet
from blueprint_pipeline.website_task_family_scene_inputs import read_family_scene_inputs, write_family_scene_inputs
from tests.test_website_task_family_packet import family_preparation


def source_fixture(tmp_path, monkeypatch, *, supplied_runtime=False, policy_selection=None, runtime_source_receipt=None,
                   runtime_selection=None):
    args, preparation, _runtime_path = family_preparation(tmp_path, policy_selection=policy_selection,
                                                        runtime_selection=runtime_selection)
    retained = tmp_path / 'pipeline' / 'website_scene_preparation'
    retained.mkdir(parents=True)
    runtime_path = retained / 'native_runtime_contract.json'
    write_json(runtime_path, preparation['task_evaluation_inputs']['native_runtime_contract'])
    context_path, preparation_path = retained / 'task_context.json', retained / 'preparation.json'
    write_json(context_path, args['task_context'])
    runtime_source = Path(runtime_source_receipt['path']) if runtime_source_receipt is not None else tmp_path / 'operator-runtime-source.json'
    if runtime_source_receipt is None:
        runtime_source.write_text('{"operator_native_runtime_source":"exact retained runtime"}\n')
    if supplied_runtime:
        preparation['task_evaluation_inputs']['runtime_source'] = record(runtime_source)
        preparation['digest'] = canonical_digest(preparation, digest_field='digest')
    write_json(preparation_path, preparation)
    registry = {'schema_version': REGISTRY_SCHEMA, 'profiles': [], 'derived_profiles': {
        'root': str(tmp_path / 'profiles'), 'defaults': {'allowed_founder_approvers': ['owner-1'],
            'max_hourly_rate_usd': '1.00', 'open_loop_horizon': 1,
            'native_runtime_source_receipt': record(runtime_source)}}}
    registry['registry_digest'] = canonical_digest(registry, digest_field='registry_digest')
    registry_path = tmp_path / 'registry.json'
    write_json(registry_path, registry)
    registry_path.chmod(0o600)
    monkeypatch.setenv(REGISTRY_ENV, str(registry_path))
    intent_root = tmp_path / 'intents'
    accepted = stage_scene_intent(value=preparation['intake_request'], queue_root=intent_root,
        authenticated_client='blueprint-webapp', trusted_clients={'blueprint-webapp'}, now=args['now'])
    intent_reference = record(intent_root / accepted['intent_id'] / 'intent.json')
    packet = materialize_website_task_family_packet(preparation=preparation, runtime_inputs_path=runtime_path,
        output_root=tmp_path / 'compiled' / 'packet', intent_reference=intent_reference,
        attempt_reference=record(runtime_path), release_reference=record(runtime_path))
    written = write_family_scene_inputs(packet=packet, preparation=preparation,
        preparation_reference=record(preparation_path), task_context_reference=record(context_path),
        runtime_inputs_reference=record(runtime_path), intent_reference=intent_reference)
    return written, registry_path, preparation_path, runtime_source


def test_family_scene_profile_uses_operator_runtime_and_preserves_compilation_boundary(tmp_path, monkeypatch):
    written, _registry, _preparation, runtime_source = source_fixture(tmp_path, monkeypatch)
    assert written['family_profile']['status'] == 'written'
    value = read_family_scene_inputs(written['family_scene_inputs']['path'])
    assert value['runtime_source'] == record(runtime_source)
    assert value['native_application_claimed'] is False and value['paid_authority_granted'] is False
    receipt = json.loads(Path(value['construction_result']['path']).read_text())
    assert receipt['native_application_claimed'] is False
    profile = profile_for_capture(tmp_path)
    facts = scene_facts(profile)
    assert facts['scene']['task_kind'] == 'button_press'
    assert facts['max_policy_queries'] is None and facts['open_loop_horizon'] == 1
    assert facts['prompt'] == 'Run the original multi-object job.'


def test_customer_cannot_replace_operator_runtime_source_receipt(tmp_path, monkeypatch):
    written, _registry, preparation_path, _runtime_source = source_fixture(tmp_path, monkeypatch)
    untrusted = tmp_path / 'customer-runtime-source.json'
    untrusted.write_text('{"customer":"alternate runtime"}\n')
    preparation = json.loads(preparation_path.read_text())
    preparation['task_evaluation_inputs']['runtime_source'] = record(untrusted)
    preparation['digest'] = canonical_digest(preparation, digest_field='digest')
    write_json(preparation_path, preparation)
    path = Path(written['family_scene_inputs']['path'])
    value = json.loads(path.read_text())
    value['references']['preparation'] = record(preparation_path)
    value['inputs_digest'] = canonical_digest(value, digest_field='inputs_digest')
    write_json(path, value)
    with pytest.raises(ValueError, match='family_runtime_source_operator_binding_mismatch'):
        read_family_scene_inputs(path)


def test_operator_runtime_source_bytes_cannot_change_after_registration(tmp_path, monkeypatch):
    written, _registry, _preparation, runtime_source = source_fixture(tmp_path, monkeypatch, supplied_runtime=True)
    runtime_source.write_text('{"operator_native_runtime_source":"changed"}\n')
    with pytest.raises(ValueError, match='input_bytes_mismatch'):
        read_family_scene_inputs(written['family_scene_inputs']['path'])


def test_family_profile_is_an_exclusive_scene_half():
    assert _one_scene_half({'family_scene_inputs_path': '/family.json'})
    assert not _one_scene_half({'family_scene_inputs_path': '/family.json', 'team_scene_inputs_path': '/team.json'})
    assert not _one_scene_half({'family_scene_inputs_path': '/family.json',
        'pair_runtime_inputs_path': '/pair.json', 'pair_activation_path': '/activation.json'})


def test_family_source_preserves_exact_optional_candidate_selection(tmp_path, monkeypatch):
    selection = {'candidate_id': 'registered_button_policy', 'catalog_digest': 'sha256:' + 'c' * 64,
                 'configuration_digest': 'sha256:' + 'd' * 64}
    written, _registry, _preparation, _runtime = source_fixture(tmp_path, monkeypatch,
        policy_selection=selection)
    value = read_family_scene_inputs(written['family_scene_inputs']['path'])
    assert value['policy_selection'] == selection
    path = Path(written['family_scene_inputs']['path'])
    retained = json.loads(path.read_text())
    retained['policy_selection']['candidate_id'] = 'different_policy'
    retained['inputs_digest'] = canonical_digest(retained, digest_field='inputs_digest')
    write_json(path, retained)
    with pytest.raises(ValueError, match='family_scene_inputs_source_binding_changed'):
        read_family_scene_inputs(path)
