"""Root administrative consent retains exact owner intent without target authority."""
# Covers (for impacted-test selection):
#   src/blueprint_pipeline/control_plane_lane_owner_consents.py
#   src/blueprint_pipeline/control_plane_lane_scratch_decisions.py

import hashlib
import json

import pytest

from tests.test_owner_census_budget import payloads
from tests.test_lane_scratch_decisions import _annotations, _json


def policy():
    return dict(schema_version='control_plane_lane_owner_policy.v1', enabled=True,
                principals=[dict(principal='operator', owners=['owner'],
                                 allowed_actions=['keep', 'register', 'offload', 'delete'],
                                 max_consent_seconds=1000)])


def build(*, census=None, annotations=None, selected=None, config_policy=None, **overrides):
    from blueprint_pipeline import control_plane_lane_owner_consents as c
    from blueprint_pipeline.control_plane_reference_budget import ReferenceCollectionBudget
    old_census, old_annotations = payloads()
    census = old_census if census is None else census
    annotations = old_annotations if annotations is None else annotations
    args = dict(census_bytes=census, annotation_bytes=annotations,
                census_sha256='sha256:' + hashlib.sha256(census).hexdigest(), census_size_bytes=len(census),
                annotations_sha256='sha256:' + hashlib.sha256(annotations).hexdigest(), annotations_size_bytes=len(annotations),
                policy_bytes=_json(policy() if config_policy is None else config_policy),
                principal='operator', selected_paths=['/work/sample'] if selected is None else selected,
                expires_at_epoch=1050, now=1000, allowed_roots=('/work', '/inputs'), consent_id='a'*32,
                budget=ReferenceCollectionBudget(monotonic=lambda: 0))
    args.update(overrides)
    return c._build_consent(**args)


def refuse(code, **kwargs):
    from blueprint_pipeline.control_plane_lane_owner_consents import OwnerCensusConsentError
    with pytest.raises(OwnerCensusConsentError) as caught:
        build(**kwargs)
    assert caught.value.code == code
    assert caught.value.args == (code,)


def test_exact_consent_keeps_approved_owner_separate_from_guess_and_no_generation():
    value = build()
    assert value['principal'] == 'operator' and value['issuer_uid'] == 0
    assert value['issuer_kind'] == 'local_root_administrative_attestation'
    assert value['inventory_count'] == value['selected_count'] == 1
    assert value['decisions'][0]['decision']['owner'] == 'owner'
    assert value['decisions'][0]['census_row']['owner_guess'] == 'unowned'
    assert value['target_generation_bound'] is value['execution_authorized'] is False
    assert value['requires_fresh_reference_check'] is True and value['mutations'] == 0
    assert value['consent_digest'].startswith('sha256:')


@pytest.mark.parametrize(('field', 'value', 'code'), [
    ('census_sha256', 'sha256:'+'0'*64, 'owner_consent_input_identity_mismatch'),
    ('annotations_size_bytes', 3, 'owner_consent_input_identity_mismatch'),
    ('principal', 'unmapped', 'owner_consent_principal_unmapped'),
    ('expires_at_epoch', 1101, 'owner_consent_expiry_invalid'),
    ('expires_at_epoch', True, 'owner_consent_expiry_invalid'),
    ('expires_at_epoch', 10**400, 'owner_consent_expiry_invalid'),
])
def test_exact_identities_principal_and_expiry_refuse(field, value, code):
    refuse(code, **{field: value})


@pytest.mark.parametrize(('change', 'code'), [
    ({'enabled': False}, 'owner_consent_disabled'),
    ({'unexpected': 'private'}, 'owner_consent_policy_invalid'),
    ({'principals': []}, 'owner_consent_principal_unmapped'),
    ({'principals': [dict(principal='operator', owners=['someone'], allowed_actions=['keep'], max_consent_seconds=1000)]}, 'owner_consent_owner_unmapped'),
    ({'principals': [dict(principal='operator', owners=['owner'], allowed_actions=['delete'], max_consent_seconds=1000)]}, 'owner_consent_action_unapproved'),
])
def test_current_finite_policy_has_no_default_authority(change, code):
    refuse(code, config_policy=policy() | change)


@pytest.mark.parametrize('selected', [[], ['/work/not-retained'], ['/work/sample']*2, ['sample']])
def test_selection_is_exact_nonempty_unique_existing_rows(selected):
    refuse('owner_consent_selection_invalid', selected=selected)


@pytest.mark.parametrize('on_selected', [True, False])
def test_unknown_whole_inventory_row_keys_cannot_hide_under_subset(on_selected):
    raw, _ = payloads()
    census = json.loads(raw)
    if not on_selected:
        second = census['rows'][0] | dict(path='/inputs/unselected', allocated_bytes=0)
        census['rows'].append(second)
        census['candidate_count'] = census['entries_visited'] = 2
    census['rows'][0 if on_selected else 1]['opaque_evidence'] = 'PRIVATE'
    raw = _json(census)
    decisions=[dict(path=r['path'], action='keep', owner='owner', expires_at_epoch=1100) for r in census['rows']]
    refuse('owner_consent_inventory_invalid', census=raw, annotations=_annotations(raw, decisions))


def test_reference_positive_destructive_consent_is_refused():
    raw, _ = payloads()
    value = json.loads(raw)
    value['rows'][0]['references'] = ['pin']
    raw = _json(value)
    refuse('owner_consent_inventory_invalid', census=raw,
           annotations=_annotations(raw, [dict(path='/work/sample', action='delete', owner='owner', reason='discard')]))


def test_register_cache_keeps_literal_positive_budget_and_ttl_without_protocol():
    raw, _ = payloads()
    decision=dict(path='/work/sample', action='register', owner='owner', lane='g1', name='sample',
                  reason='test-cache', class_intent='cache', cleanup='owner_review', ttl_seconds=100,
                  run_ref='run-1', size_budget_bytes=4)
    value=build(annotations=_annotations(raw,[decision]))
    approved=value['decisions'][0]['decision']
    assert approved['size_budget_bytes'] == 4 and approved['ttl_seconds'] == 100
    assert 'consumer_lifetime_contract' not in approved
    for budget in [None, True, 10**400]:
        refuse('owner_consent_inventory_invalid', annotations=_annotations(raw,[decision | dict(size_budget_bytes=budget)]))


def test_ordinary_validator_still_accepts_retained_extra_fields():
    from blueprint_pipeline.control_plane_lane_scratch_decisions import validate_census_annotations
    raw, _ = payloads()
    value = json.loads(raw)
    value['rows'][0]['extension'] = 'ordinary'
    raw=_json(value)
    annotation=_annotations(raw,[dict(path='/work/sample',action='keep',owner='owner',expires_at_epoch=1100)])
    assert validate_census_annotations(raw,annotation,now=1000,allowed_roots=('/work',))['status']=='validated'


def test_both_retained_documents_preflight_before_any_integrity_hash(monkeypatch):
    from blueprint_pipeline import control_plane_lane_owner_consents as c
    from blueprint_pipeline import control_plane_reference_budget as b
    raw, _ = payloads()
    malformed = b'{"nested":' + b'['*8 + b'0' + b']'*8 + b'}'
    identities = dict(census_sha256='sha256:'+hashlib.sha256(raw).hexdigest(),
                      annotations_sha256='sha256:'+hashlib.sha256(malformed).hexdigest())
    # Build creates independent identities before installing the hash spy.
    monkeypatch.setattr(b, 'MAX_DEPTH', 4)
    monkeypatch.setattr(c.retained.hashlib, 'sha256', lambda *a, **k: pytest.fail('integrity hash before combined lexical proof'))
    with pytest.raises(b.ReferenceCollectionBudgetError, match='reference_depth_limit'):
        c._build_consent(census_bytes=raw, annotation_bytes=malformed,
            census_size_bytes=len(raw), annotations_size_bytes=len(malformed), **identities,
            policy_bytes=_json(policy()), principal='operator', selected_paths=['/work/sample'],
            expires_at_epoch=1050, now=1000, allowed_roots=('/work',), consent_id='a'*32,
            budget=b.ReferenceCollectionBudget(monotonic=lambda:0))
