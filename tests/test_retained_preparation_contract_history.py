"""Historical interpretation is explicit and independent of future admission policy."""
from copy import deepcopy

import pytest

from blueprint_pipeline import task_evaluation_launch_preparation_contract as contract
from blueprint_pipeline.decision_evidence_contracts import canonical_json
from blueprint_pipeline.task_evaluation_retained_preparation_contract import retained_contract_identity
from tests.test_task_evaluation_launch_preparation_contract import test_configuration_request as configuration_request


@pytest.mark.parametrize('ttl,teacher,review,external,cap,expected', [
    (25200, 2.4, .32, 3., 10., 'scene_preparation_25200_single_pass.v1'),
    (25200, 4.8, .64, 6., 12., 'scene_preparation_25200_repair.v1'),
    (27000, 4.8, .64, 6., 12., 'scene_preparation_27000_repair.v1'),
])
def test_known_historical_administrative_contracts_are_readable_but_not_new_admission(
    ttl, teacher, review, external, cap, expected,
):
    value = configuration_request()
    value['spend'].update(hard_ttl_seconds=ttl, hard_cap_usd=cap)
    openai = value['spend']['external_service_caps']['openai']
    openai.update(maximum_cost_usd=external, maximum_requests=31)
    openai['stage_max_cost_usd'].update(artifixer_semantic_teacher=teacher,
        artifixer_visual_review=review, content_agents=.2)
    before = canonical_json(value)
    assert contract.validate_retained_preparation_request(value) == value
    identity = retained_contract_identity(value)
    assert identity['contract_id'] == expected
    assert identity['request_source_commit'] == value['expected_production_commit']
    assert identity['schema_sha256'].startswith('sha256:')
    with pytest.raises(ValueError, match='runtime_budget_invalid|external_spend_invalid'):
        contract.validate_launch_preparation_request(value)
    assert canonical_json(value) == before


def test_current_schema_or_runtime_budget_changes_cannot_reinterpret_retained_bytes(monkeypatch):
    value = configuration_request()
    value['spend']['external_service_caps']['openai']['stage_max_cost_usd']['artifixer_visual_review'] = .64
    before = deepcopy(value)
    monkeypatch.setattr(contract, 'REQUIRED_PARENT_TTL_SECONDS', 36000)
    monkeypatch.setattr(contract, 'MIN_ARTIFIXER_SEMANTIC_TEACHER_SPEND_USD', 10.)
    monkeypatch.setattr(contract, 'MIN_ARTIFIXER_VISUAL_REVIEW_SPEND_USD', 10.)
    monkeypatch.setattr(contract, 'preparation_request_schema',
        lambda: pytest.fail('historical interpretation consulted mutable current intake schema'))
    assert contract.validate_retained_preparation_request(value) == before
    assert value == before


@pytest.mark.parametrize('fault', ['unknown_ttl', 'zero_requests', 'below_historical_floor', 'unknown_schema', 'paid_policy'])
def test_historical_contract_does_not_grandfather_arbitrary_values(fault):
    value = configuration_request()
    if fault == 'unknown_ttl':
        value['spend']['hard_ttl_seconds'] = 26000
    elif fault == 'zero_requests':
        value['spend']['external_service_caps']['openai']['maximum_requests'] = 0
    elif fault == 'below_historical_floor':
        value['spend']['external_service_caps']['openai']['stage_max_cost_usd']['artifixer_visual_review'] = .01
    elif fault == 'unknown_schema':
        value['schema_version'] = 'unrecognized.v0'
    else:
        value['policy_run_setup'] = {'arbitrary': 'new execution intent'}
    with pytest.raises(ValueError):
        contract.validate_retained_preparation_request(value)


def test_unavailable_retained_schema_refuses_without_falling_back_to_current(tmp_path, monkeypatch):
    from blueprint_pipeline import task_evaluation_retained_preparation_contract as retained
    missing = tmp_path / 'absent-schema.json'
    monkeypatch.setattr(retained, 'SCHEMA_PATH', missing)
    with pytest.raises(ValueError, match='retained_schema_unavailable'):
        contract.validate_retained_preparation_request(configuration_request())
    missing.write_text('[]')
    with pytest.raises(ValueError, match='retained_schema_invalid'):
        contract.validate_retained_preparation_request(configuration_request())
    assert contract.validate_launch_preparation_request(configuration_request())


@pytest.mark.parametrize('partitioned,surface', [(True, False), (False, True), (True, True)])
def test_retained_partition_and_surface_contract_is_frozen_and_digest_bound(monkeypatch, partitioned, surface):
    from tests.test_task_evaluation_launch_preparation_contract import ref
    from tests.test_task_evaluation_surface_target import target_fixture
    value = configuration_request()
    if partitioned:
        value['scene']['geometry']['source_derivation'] = ref(46)
    if surface:
        value['task'].pop('destination', None)
        value['task']['strategy'] = 'pick_and_place'
        value['task']['surface_target'] = target_fixture()
    before = deepcopy(value)
    monkeypatch.setattr(contract, 'preparation_request_schema',
        lambda: pytest.fail('retained replay consulted live schema'))
    assert contract.validate_retained_preparation_request(value) == before
    identity = retained_contract_identity(value)
    assert 'retained_source_' in identity['policy_source_path']
    assert identity['new_execution_authorized'] is False
    assert identity['new_spend_authorized'] is False
    if partitioned:
        value['scene']['geometry']['source_derivation']['digest'] = 'unverified'
    else:
        value['task']['surface_target']['radius_m'] = 100
    with pytest.raises(ValueError):
        contract.validate_retained_preparation_request(value)
