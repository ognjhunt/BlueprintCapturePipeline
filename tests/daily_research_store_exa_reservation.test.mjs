// The Exa reservation is the immutable intent's native budget on every write, so a later
// rewrite of cap_micros cannot hide reserved spend from the combined Exa and FindAll check.
import {test} from 'node:test';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {Store, ROOT} from '../tools/daily_research/firestore_bridge.mjs';
import {MemoryFirestore} from './fixtures/daily_research/firestore-memory.mjs';

async function fixture(now=Date.now()) {
  const db = new MemoryFirestore(), time = {now};
  db.values.set(ROOT, {enabled: true, schema_version: 'blueprint.research-control.v1'});
  const store = new Store(db, () => time.now, 'first');
  await store.acquire(); return {db, store, time};
}
const row = () => ({date: '2026-09-30', run_key: 'blueprint-researcher:2026-09-30', metadata: {run_key: 'day', payload_digest: 'hash'},
  state: 'creating', cleanup_required: true});

// Owner-directed paid expansion allowance (tools/daily_research/allocation.py mirror).
const canonicalJSON=value=>JSON.stringify(function sort(x){return Array.isArray(x)?x.map(sort):x && typeof x==='object'
  ?Object.fromEntries(Object.keys(x).sort().map(k=>[k,sort(x[k])])):x;}(value));
const PROJECT='proj_F2tFJuxLaovJru8RrtXRaqNj',AGENT='agent_5a01ec367d1042ef8632bb5f2e6af8b4919909d2abed48ed95';
function direction(version=1,supersedes=null,changes={}) {
  return {schema_version:'blueprint.research-paid-expansion-direction.v1',version,supersedes,per_run_limit_usd:'10.00',
    sources:['exa'],scope:{project_id:PROJECT,agent_id:AGENT,firestore_root:ROOT,run_key_prefix:'blueprint-researcher:',
      timezone:'America/Chicago'},effective_from:'2026-10-04T18:00:00+00:00',expires_at:'2027-01-02T18:00:00+00:00',
    approval_reference:'owner-decision-2026-10-04',approved_by:'owner',issued_at:'2026-10-04T18:00:00+00:00',
    reason:'Owner per-run allowance',...changes};
}
function entry(value) {
  const sha=createHash('sha256').update(canonicalJSON(value)).digest('hex');
  return {sha256:sha,version:value.version,
    uri:`gs://blueprint-8c1ca.appspot.com/operations/research/paid-expansion/${sha}/direction.json`,direction:value};
}
async function paidFixture() {
  const value=await fixture(Date.parse('2026-10-05T12:00:00+00:00'));
  Object.assign(value.db.values.get(ROOT),{project_id:PROJECT,agent_id:AGENT,source_commit:'c'.repeat(40)});
  return value;
}
const setDirection=(store,expected,current,enabled=true)=>store.dispatch({op:'paid_expansion_set',expected_sha256:expected,value:{enabled,current}});


function grantFor(current,value,changes={}) {
  return {schema_version:'blueprint.research-paid-expansion-grant.v1',state:'granted',run_key:value.run_key,
    frozen_at:'2026-10-05T12:00:00+00:00',direction_sha256:current.sha256,
    grant_id:createHash('sha256').update(JSON.stringify([current.sha256,value.run_key])).digest('hex'),
    direction_uri:current.uri,version:current.version,sources:['exa'],limit_micros:10000000,per_start_max_micros:5000000,
    source_commit:'c'.repeat(40),approval_reference:current.direction.approval_reference,
    valid_until:'2026-10-05T12:20:00+00:00',...changes};
}
const exaRow=date=>({...row(),date,run_key:`blueprint-researcher:${date}`,expansion_profile:'exa-guarded-v1',
  metadata:{...row().metadata,expansion_profile:'exa-guarded-v1'}});
function exaClaim(value,cap=2000000,dollars=cap/1000000) {
  const intent={request:{query:'US laundry towel handling',effort:'ultra',budget:{maxCostDollars:dollars}},grant:value.paid_expansion_grant};
  const json=JSON.stringify(intent);
  return {date:value.date,run_key:value.run_key,intent,intent_json:json,intent_sha256:createHash('sha256').update(json).digest('hex'),
    cap_micros:cap,state:'submission_unresolved',attempted:true,run_id:null};
}


// Parallel FindAll claims debit the same frozen grant (tools/daily_research/findall.py).
// The production combination: the Exa profile and the pinned FindAll registry on one row.
const findallRow=date=>({...row(),date,run_key:`blueprint-researcher:${date}`,findall_profile:'parallel-findall-v1',
  expansion_profile:'exa-guarded-v1',
  metadata:{...row().metadata,findall_tools_digest:'d'.repeat(64),expansion_profile:'exa-guarded-v1'}});
function findallEntry(value,operation,cost,findallId=null) {
  const operation_id=`${value.run_key}:findall:${operation}`,binding='sha256:'+createHash('sha256').update(operation).digest('hex');
  const prepared={schema_version:'parallel_findall_submission.v1',resource_class:'parallel_findall',operation_id,method:'POST',
    url:'https://api.parallel.ai/v1beta/findall/runs',body_json:{objective:'US sites',entity_type:'company',generator:'base',
      match_limit:5,match_conditions:[{name:'task',description:'Exact site task'}]},maximum_cost_usd:cost,
    allocation_binding_digest:binding,execution_authorized:false,network_called:false};
  return [createHash('sha256').update(operation_id).digest('hex'),{schema_version:'blueprint.findall-owner-submission.v1',
    operation_id,allocation_binding_digest:binding,prepared,state:'submission_unresolved',findall_id:findallId}];
}
const withClaims=(value,...entries)=>({...value,state:'running',parallel_findall_submissions:Object.fromEntries(entries)});
async function findallFixture(sources=['exa','findall']) {
  const fixtureValue=await paidFixture();
  const current=entry(direction(1,null,{sources}));await setDirection(fixtureValue.store,null,current);
  const value=findallRow('2026-10-05');
  value.paid_expansion_grant=grantFor(current,value,{sources});
  await fixtureValue.store.put(value);
  return {...fixtureValue,current,value};
}


test('a later write cannot lower an existing Exa reservation to admit more FindAll spend', async()=>{
  const {store,value}=await findallFixture();
  const exa=exaClaim(value,5000000);
  await store.put({...value,state:'running',exa_expansion:exa});
  const a=findallEntry(value,'call_a','5');
  await store.put({...value,state:'running',exa_expansion:exa,parallel_findall_submissions:Object.fromEntries([a])});
  const b=findallEntry(value,'call_b','1');
  // $5 Exa + $5 FindAll uses the $10 grant: one more claim is refused.
  await assert.rejects(store.put({...value,state:'running',exa_expansion:exa,parallel_findall_submissions:Object.fromEntries([a,b])}),
    /paid_expansion_reservation_exceeds_grant/);
  // Rewriting only cap_micros lower, with the same immutable intent, is refused too.
  await assert.rejects(store.put({...value,state:'running',exa_expansion:{...exa,cap_micros:1000000},
    parallel_findall_submissions:Object.fromEntries([a,b])}), /paid_expansion_claim_cap_mismatch/);
  await assert.rejects(store.put({...value,state:'running',exa_expansion:{...exa,cap_micros:1000000},
    parallel_findall_submissions:Object.fromEntries([a])}), /paid_expansion_claim_cap_mismatch/);
});
