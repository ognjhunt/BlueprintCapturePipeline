"""A daily official bucket must be queried in full before existing strict closure."""
from copy import deepcopy
from datetime import datetime
import json

import pytest

from blueprint_pipeline import operator_policy_canary_continuation as coordinator
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.operator_policy_canary_terminal_delivery import file_record
from blueprint_pipeline.vast_official_charge_period import validate_charge_period, VastOfficialBillingExtractionError


def retained(tmp_path, *, instance='instance-50518293', label='exact-label', start=1788998400, end=1788998400):
    root = tmp_path/'20260910T213043.971865Z'
    root.mkdir()
    row = {'start':start, 'end':end, 'source':instance, 'type':'instance', 'metadata':{'label':label}, 'amount':1.839}
    response = root/'response-001-vast.json'
    response.write_text(json.dumps({'results':[row]}))
    rec = file_record(response)
    source = {'schema_version':'blueprint.provider_billing_source_receipt.v1', 'status':'reconciled',
        'covered_provider_ids':['vast'], 'cohort_start_at':'2026-09-10T19:12:00+00:00',
        'cohort_end_at':'2026-09-10T21:30:43.971865+00:00',
        'sources':[{'provider':'vast','endpoint':'https://console.vast.ai/api/v0/charges/',
                    'retained_path':str(response),'response_digest':rec['sha256'],'response_size_bytes':rec['size_bytes']}]}
    source['receipt_digest']=canonical_digest(source,digest_field='receipt_digest')
    source_path=root/'provider_billing_source_receipt.json'
    source_path.write_text(json.dumps(source))
    intent={'instance_id':50518293,'resource_name':'exact-label','billing_period_start_at':source['cohort_start_at'],
        'intent_digest':'sha256:'+'a'*64,'billing_secrets_dir':'unused',
        'terminal_delivery_intent':{'run_root':str(tmp_path)}}
    return intent, source, row, source_path, response


def test_exact_v27_daily_charge_requires_verified_expanded_query_without_rewriting_history(tmp_path, monkeypatch):
    intent, source, row, source_path, response = retained(tmp_path)
    original_intent=deepcopy(intent)
    original_source=source_path.read_bytes()
    original_response=response.read_bytes()
    terminal=tmp_path/'allocator_result.json'
    terminal.write_text(json.dumps({'existing_instance_recovered':True,'vast_instance_ids':[50518293]}))
    original_terminal=terminal.read_bytes()
    with pytest.raises(VastOfficialBillingExtractionError,match='charge_period_invalid'):
        validate_charge_period(row,source)
    start, provenance=coordinator.retained_official_billing_query_period(intent,tmp_path)
    assert start=='2026-09-10T00:00:00+00:00'
    assert provenance['official_charge_repriced'] is False
    assert provenance['evidence'][0]['official_row_digest']==canonical_digest(row)
    assert provenance['evidence'][0]['source_receipt']['sha256']==file_record(source_path)['sha256']
    # A new provider response under this wider cohort can pass unchanged period
    # rules; this synthetic receipt is not claimed as a real refreshed API call.
    validate_charge_period(row,{**source,'cohort_start_at':start})
    called=[]
    from blueprint_pipeline import provider_billing_reconciler
    monkeypatch.setattr(provider_billing_reconciler,'reconcile_provider_billing',lambda **kwargs:called.append(kwargs) or {'status':'fixture'})
    coordinator.refresh_official_billing(intent={'billing_audit_root':str(tmp_path)},adapter={},continuation_intent=intent)
    assert called[0]['start_at']==start
    assert called[0]['required_providers']==('vast',)
    assert len(list((tmp_path/'existing_run_continuation/billing_query_periods').glob('*.json')))==1
    assert terminal.read_bytes()==original_terminal
    assert intent==original_intent and source_path.read_bytes()==original_source and response.read_bytes()==original_response


@pytest.mark.parametrize('kwargs',[{'instance':'instance-50518294'},{'label':'different-launch'}])
def test_other_instance_or_label_cannot_expand_financial_query(tmp_path,kwargs):
    intent,*_=retained(tmp_path,**kwargs)
    assert coordinator.retained_official_billing_query_period(intent,tmp_path)==(intent['billing_period_start_at'],None)


def test_modified_retained_response_cannot_supply_period_provenance(tmp_path):
    intent,_,_,_,response=retained(tmp_path)
    response.write_text(response.read_text()+' ')
    with pytest.raises(coordinator.ContinuationError,match='immutable_input_changed'):
        coordinator.retained_official_billing_query_period(intent,tmp_path)


@pytest.mark.parametrize('start,end',[(-1,0),(1788998401,1788998400),(1788998400,float('inf')),
                                     (1788998400,datetime.fromisoformat('2026-09-11T00:00:00+00:00').timestamp())])
def test_invalid_or_future_official_period_does_not_get_sanitized(tmp_path,start,end):
    intent,*_=retained(tmp_path,start=start,end=end)
    with pytest.raises(coordinator.ContinuationError,match='retained_billing_period_invalid'):
        coordinator.retained_official_billing_query_period(intent,tmp_path)


def test_previous_query_already_covering_midnight_needs_no_expansion(tmp_path):
    intent,*_=retained(tmp_path)
    intent['billing_period_start_at']='2026-09-09T00:00:00+00:00'
    assert coordinator.retained_official_billing_query_period(intent,tmp_path)==(intent['billing_period_start_at'],None)
