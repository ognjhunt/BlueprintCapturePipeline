// Synthetic private pin/frozen-intent trust boundaries. No provider, credentials or live storage.
import {test} from 'node:test';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {Store,ROOT} from '../tools/daily_research/firestore_bridge.mjs';
import {MemoryFirestore} from './fixtures/daily_research/firestore-memory.mjs';
import {FakeBucket} from './fixtures/daily_research/fake-bucket.mjs';
const canonical=value=>Array.isArray(value)?value.map(canonical):value && typeof value==='object'?Object.fromEntries(Object.keys(value).sort().map(k=>[k,canonical(value[k])])):value;
const bytes=value=>Buffer.from(JSON.stringify(canonical(value)));
const sha=raw=>createHash('sha256').update(raw).digest('hex');
async function fixture() {
  const db=new MemoryFirestore(),bucket=new FakeBucket(),store=new Store(db,()=>Date.parse('2026-10-05T18:00:00Z'),undefined,null,null,null,null,false,bucket);
  db.values.set(ROOT,{schema_version:'blueprint.research-control.v1',enabled:true,config:{synthetic:'preserved'}});
  await store.acquire('research_release');
  const raw=bytes({schema_version:'blueprint.team-evidence-export.v1',manifest:{distribution:'internal_only',
    ranked_sha256:'a'.repeat(64),audit_sha256:'b'.repeat(64),scope_sha256:'c'.repeat(64),assessed_on:'2026-10-05'},teams:[]});
  const stored=await store.teamUniverseObjectPut(sha(raw),raw.toString('base64'));
  const pin={schema_version:'blueprint.team-evidence-pin.v1',enabled:true,version:1,uri:stored.uri,generation:stored.generation,
    sha256:sha(raw),bytes:raw.length,ranked_sha256:'a'.repeat(64),audit_sha256:'b'.repeat(64),scope_sha256:'c'.repeat(64),
    assessed_on:'2026-10-05',approval_reference:'synthetic-reviewed-owner'};
  return {db,bucket,store,pin,raw};
}
const row=()=>({date:'2026-10-05',run_key:'blueprint-researcher:2026-10-05',metadata:{run_key:'synthetic'},state:'creating',cleanup_required:true});
function bound(pin,raw) {
  const original=JSON.parse(raw.toString('utf8'));
  const frozen=bytes({schema_version:'blueprint.team-evidence-input.v1',run_date:'2026-10-05',as_of:'2026-10-05',export_sha256:pin.sha256,pin_version:pin.version,
    manifest:original.manifest,teams:original.teams,held_team_keys:[]});
  return {...row(),team_universe:{state:'attached',pin,sha256:sha(frozen),bytes:frozen.length,path:'/workspace/inputs/blueprint-team-evidence.json'},
    metadata:{...row().metadata,team_universe_input_digest:sha(frozen)},create_payload:{environment:{files:[
      {type:'inline',path:'/workspace/inputs/blueprint-team-evidence.json',data:frozen.toString('base64')} ]}}};
}

test('old run cannot acquire or replace team binding after pin changes',async()=>{
  const {store,pin}=await fixture();
  await store.put(row());
  assert.ok(!Object.hasOwn((await store.db.doc(`${ROOT}/runs/2026-10-05`).get()).data(),'team_universe_digest'));
  await assert.rejects(store.put({...row(),team_universe:{state:'unavailable',code:'synthetic'}}),/intent_already_bound/);
  const prior=await store.get('2026-10-05');
  await store.put({...prior,state:'completed',cleanup_required:false});
  await store.teamUniverseSet(null,pin);
  assert.equal((await store.get('2026-10-05')).team_universe,undefined);
  await store.put({...prior,state:'completed',cleanup_required:false});
});
test('new frozen attachment must bind exact current pin and exact bytes',async()=>{
  const {store,pin,raw}=await fixture();await store.teamUniverseSet(null,pin);
  await assert.rejects(store.put(bound({...pin,version:2},raw)),/current_pin_changed/);
  const bad=bound(pin,raw);bad.create_payload.environment.files[0].data=Buffer.from('{}').toString('base64');
  await assert.rejects(store.put(bad),/frozen_binding_invalid/);
  const value=bound(pin,raw);await store.put(value);
  await assert.rejects(store.put({...value,team_universe:{...value.team_universe,bytes:value.team_universe.bytes+1}}),/intent_already_bound/);
  const altered=structuredClone(value);altered.create_payload.environment.files[0].data=Buffer.from('{}').toString('base64');
  await assert.rejects(store.put(altered),/frozen_binding_invalid/);
  await store.put({...value,state:'completed',cleanup_required:false});
  await store.teamUniverseSet(pin.sha256,{...pin,version:2});
  await store.put({...value,state:'completed',cleanup_required:false});
});
for(const kind of ['unfinished','qa','repair_publication'])
test(`under-lease pin refuses ${kind}`,async()=>{
  const {store,pin}=await fixture();
  if(kind==='unfinished') store.summary=async()=>({unfinished:1});
  else if(kind==='qa') store.activeQA=async()=>({state:'active'});
  else store.workItem=async()=>({stage:'publication_pending'});
  await assert.rejects(store.teamUniverseSet(null,pin),/active_research_qa_repair_or_publication/);
});
test('generation/digest tamper and lost lease cannot install a pin',async()=>{
  const {store,bucket,pin}=await fixture();
  await assert.rejects(store.teamUniverseSet(null,{...pin,generation:'99999'}),/object_unavailable/);
  const key=[...bucket.objects.keys()][0];bucket.objects.get(key).raw[0]^=1;
  await assert.rejects(store.teamUniverseSet(null,pin),/object_digest_mismatch/);
  await store.release();await assert.rejects(store.teamUniverseSet(null,pin),/lease_lost/);
});

for(const mutation of [value=>delete value.teams,value=>value.teams.push({team_key:'tampered'}),value=>value.manifest.audit_sha256='0'.repeat(64),
  value=>value.held_team_keys.push('invented'),value=>value.as_of='2020-01-01'])
test('altered or incomplete source cannot be self-bound into agent environment',async()=>{
  const {store,pin,raw}=await fixture();await store.teamUniverseSet(null,pin);
  const row=bound(pin,raw),value=JSON.parse(Buffer.from(row.create_payload.environment.files[0].data,'base64'));
  mutation(value);const changed=bytes(value);
  row.create_payload.environment.files[0].data=changed.toString('base64');
  row.metadata.team_universe_input_digest=row.team_universe.sha256=sha(changed);row.team_universe.bytes=changed.length;
  await assert.rejects(store.put(row),/frozen_binding_invalid/);
});
