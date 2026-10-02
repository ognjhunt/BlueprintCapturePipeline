import {test} from 'node:test';
import assert from 'node:assert/strict';
import {randomBytes,createHash} from 'node:crypto';
import {Store, ROOT, LeaseChannel, ADAPTIVE_TEST} from '../tools/daily_research/firestore_bridge.mjs';
import {MemoryFirestore} from './fixtures/daily_research/firestore-memory.mjs';

async function fixture() {
  const db = new MemoryFirestore(), time = {now: Date.now()};
  db.values.set(ROOT, {enabled: true, schema_version: 'blueprint.research-control.v1'});
  const store = new Store(db, () => time.now, 'first');
  await store.acquire(); return {db, store, time};
}
const row = () => ({date: '2026-09-30', run_key: 'blueprint-researcher:2026-09-30', metadata: {run_key: 'day', payload_digest: 'hash'},
  state: 'creating', cleanup_required: true});

test('learning uses the existing fenced pipe and absent control performs no handler work', async () => {
  const {db,store,time}=await fixture();
  let invoked=0; store.learning=async request=>{invoked++;return {op:request.op};};
  assert.equal(await store.dispatch({op:'learning_context',day:row().date}),null);
  assert.equal(invoked,0);
  db.values.get(ROOT).learning={enabled:true};
  assert.deepEqual(await store.dispatch({op:'learning_context',day:row().date}),{op:'learning_context'});
  time.now+=180001;
  await assert.rejects(store.dispatch({op:'learning_daily',day:row().date}),/lease_lost/);
  assert.equal(invoked,1);
});

test('learning scope drift, disable or expiry cannot claim provider creation', async () => {
  const {db,store,time}=await fixture(), expiry=new Date(time.now+60000).toISOString();
  const learning={binding:{expiresAt:expiry},businessScope:{expiresAt:expiry},enabled:true,learningGrant:{expiresAt:expiry}};
  const bindingHash=createHash('sha256').update(JSON.stringify(learning)).digest('hex');
  db.values.get(ROOT).learning=learning;
  const value={...row(),metadata:{...row().metadata,learning_binding_digest:bindingHash}};
  await store.put(value);
  learning.enabled=false;
  await assert.rejects(store.createCheck(value.date,value.metadata),/scope_changed_or_expired/);
  learning.enabled=true; time.now+=60001; await store.renew();
  await assert.rejects(store.createCheck(value.date,value.metadata),/scope_changed_or_expired/);
  assert.equal(db.values.get(`${ROOT}/runs/${value.date}`).create_attempt_claimed,false);
});

test('Perplexity row overflow is refused before a durable run or publication claim', async () => {
  const {db, store} = await fixture();
  await assert.rejects(store.put({...row(), search_provider:'perplexity-fast-v1', payload:'x'.repeat(7000000)}), /record_resource_ceiling/);
  assert.equal(db.values.has(`${ROOT}/runs/${row().date}`), false);
});

test('fresh recurring budget gates both model create and QA claims', async () => {
  const {db, store} = await fixture();
  const config = {search_provider:'perplexity-fast-v1', soft_target_usd:2, recurring_budget_authority_reference:'owner-test-budget'};
  Object.assign(db.values.get(ROOT), {config, workflow:{enabled:true, qa_authority_reference:'owner-qa', publication_authority_reference:'owner-publication'}});
  const value = {...row(), ...config}; await store.put(value);
  db.values.get(ROOT).config = {...config, soft_target_usd:3};
  await assert.rejects(store.createCheck(value.date,value.metadata), /budget_authority_changed/);
  assert.equal(db.values.get(`${ROOT}/runs/${value.date}`).create_attempt_claimed,false);
  db.values.get(ROOT).config = config;
  await store.createCheck(value.date,value.metadata);
  const deadline = Date.now()+60000, digest = 'a'.repeat(64);
  await store.put({...value,state:'awaiting_review',qa:{state:'qa_input_unresolved',request_digest:digest,deadline_ms:deadline}});
  db.values.get(ROOT).config = {...config,recurring_budget_authority_reference:'different-authority'};
  await assert.rejects(store.qaCheck(value.date,digest,deadline), /budget_authority_changed/);
  assert.equal(db.values.get(`${ROOT}/runs/${value.date}`).qa_request_claimed,false);
  db.values.get(ROOT).config = config;
  await store.qaCheck(value.date,digest,deadline);
});

test('overlap refuses and late release cannot clear the successor lease', async () => {
  const {db, store, time} = await fixture();
  const second = new Store(db, () => time.now, 'second');
  await assert.rejects(second.acquire(), /runner_overlap/);
  await store.put(row()); time.now += 180001;
  await second.acquire(); await store.release();
  assert.equal(db.values.get(ROOT).lease.owner, 'second');
  await second.assertLease(); await assert.rejects(store.createCheck(row().date, row().metadata), /lease_lost/);
});

test('create claim is durable, one per date, and outside any provider call', async () => {
  const {db, store} = await fixture(); db.replay = true;
  await store.put(row()); await store.createCheck(row().date, row().metadata);
  await assert.rejects(store.createCheck(row().date, row().metadata), /not_admitted/);
  await assert.rejects(store.put({...row(), metadata: {run_key: 'different'}}), /intent_conflict/);
  assert.equal(db.values.get(`${ROOT}/runs/${row().date}`).create_attempt_claimed, true);
});

test('failed durable commit leaves no dated intent or create authority', async () => {
  const {db, store} = await fixture(); db.failCommit = true;
  await assert.rejects(store.put(row()), /storage failure/);
  db.failCommit = false;
  await assert.rejects(store.createCheck(row().date, row().metadata), /not_admitted/);
});

test('large immutable chunked artifacts verify exact bytes and detect corruption', async () => {
  const {db, store} = await fixture(); const raw = randomBytes(1900000), name = '2026-09-30-artifact.json';
  await store.filePut(name, raw.toString('base64'));
  assert.deepEqual(Buffer.from(await store.fileGet(name), 'base64'), raw);
  await assert.rejects(store.filePut(name, Buffer.from('replacement').toString('base64')), /identity_conflict/);
  for (const [path, data] of db.values) if (path.includes('/chunks/')) assert.ok(data.bytes.length <= 256 * 1024);
  const hash = db.values.get(`${ROOT}/files/${name}`).blob;
  db.values.get(`${ROOT}/blobs/${hash}/chunks/0`).bytes = Buffer.from('corrupt');
  await assert.rejects(store.fileGet(name));
  await assert.rejects(store.fileGet('2026-10-01-artifact.json'), /file_missing/);
});

test('heartbeat release drains renewal and reacquisition starts a clean generation', async () => {
  const {db, store} = await fixture(); await store.release();
  let beat, unblock; const events = [];
  const renew = store.renew.bind(store);
  store.renew = async () => {events.push('renew-start'); await new Promise(resolve => {unblock = resolve;}); await renew(); events.push('renew-end');};
  const release = store.release.bind(store);
  store.release = async () => {events.push('release'); return release();};
  const channel = new LeaseChannel(store, {setIntervalImpl: fn => {beat = fn; return 1;}, clearIntervalImpl: () => events.push('timer-cleared')});
  await channel.call({op: 'acquire'}); beat(); await new Promise(resolve => setImmediate(resolve));
  const releasing = channel.call({op: 'release'}); unblock(); await releasing;
  assert.ok(events.indexOf('renew-end') < events.indexOf('release'));
  await channel.call({op: 'acquire'}); await channel.call({op: 'assert_lease'});
  assert.equal(db.values.get(ROOT).lease.generation, 3); await channel.close();
});

test('summary is bounded and QA/publication queue is Blueprint-owned', async () => {
  const {db, store} = await fixture();
  await store.put(row()); assert.deepEqual(await store.summary(), {latest_date: row().date, unfinished: true, cleanup_required: true});
  await store.put({...row(), state: 'awaiting_review', packet: {findings: []}, packet_digest: 'digest'});
  const task = db.values.get(`${ROOT}/workItems/${row().date}`);
  assert.equal(task.stage, 'agent_qa_pending'); assert.equal(task.observer_receipt_required, false);
  assert.equal((await store.summary()).unfinished, false);
});

test('legacy imports require disabled control and preserve an attempted date', async () => {
  const {db, store} = await fixture();
  await assert.rejects(store.importRun({...row(), state: 'creation_unresolved'}), /requires_disabled/);
  db.values.get(ROOT).enabled = false;
  const legacy = {...row(), state: 'creation_unresolved'};
  await store.importRun(legacy); await store.importRun(legacy);
  await assert.rejects(store.importRun({...legacy, state: 'running'}), /date_conflict/);
  db.values.get(ROOT).enabled = true;
  await assert.rejects(store.createCheck(row().date, row().metadata), /not_admitted/);
});

test('imported legacy review packets reach the agent queue without rerunning research', async () => {
  const {db, store} = await fixture(); db.values.get(ROOT).enabled = false;
  const legacy = {...row(), state: 'awaiting_review', packet: {findings: []}, packet_digest: 'digest'};
  await store.importRun(legacy);
  assert.equal(db.values.get(`${ROOT}/workItems/${row().date}`).stage, 'agent_qa_pending');
  assert.equal(db.values.get(`${ROOT}/runs/${row().date}`).create_attempt_claimed, true);
});

async function adaptiveFixture() {
  const f = await fixture(), origin = {...row(), date:'2026-10-01',run_key:'blueprint-researcher:2026-10-01'};
  await f.store.put(origin);
  await f.store.put({...origin,state:'failed',raw_output_digest:'d'.repeat(64),delivery:{}});
  const daily = await f.store.adaptiveOrigin();
  const intent = {test_id:ADAPTIVE_TEST,intent_digest:'a'.repeat(64),request_digest:'b'.repeat(64),session_id:'sess_1',
    environment_id:'env_1',admission_blockers:[],provider_calls:0,profile:{enabled:false,publication_enabled:false,
      new_session_create_allowed:false,daily_state_reset_allowed:false,permanent_deletion_allowed:false}};
  const testRow = {date:'2026-10-01',test_id:ADAPTIVE_TEST,test_intent:intent,daily_blob:daily.blob,
    session_id:'sess_1',environment_id:'env_1',state:'research_input_unresolved',research_deadline_ms:f.time.now+1200000};
  await f.store.adaptivePut(testRow);
  return {...f, testRow, daily};
}

test('adaptive claim stays separate, survives lease replacement, and cannot repeat uncertain input', async () => {
  const {store,db,time,testRow,daily} = await adaptiveFixture();
  db.replay=true;
  await store.adaptiveClaim('research','b'.repeat(64),testRow.research_deadline_ms);
  const second=new Store(db,()=>time.now,'replacement');
  await assert.rejects(second.acquire(),/runner_overlap/);
  await store.release();await second.acquire();
  assert.equal((await second.adaptiveGet()).test_intent.intent_digest,'a'.repeat(64));
  await assert.rejects(second.adaptiveClaim('research','b'.repeat(64),testRow.research_deadline_ms),/not_admitted/);
  assert.deepEqual(await second.adaptiveOrigin(),daily);
  assert.equal(db.values.has(`${ROOT}/workItems/2026-10-01`),false);
});

test('adaptive admission, exact original, deadlines and immutable intent bind both claims', async () => {
  const {store,db,time,testRow,daily} = await adaptiveFixture();
  await assert.rejects(store.adaptivePut({...testRow,test_intent:{...testRow.test_intent,request_digest:'c'.repeat(64)}}),/intent_conflict/);
  await assert.rejects(store.adaptiveClaim('research','c'.repeat(64),testRow.research_deadline_ms),/not_admitted/);
  const qa={request_digest:'c'.repeat(64),deadline_ms:time.now+1800000};
  await store.adaptivePut({...testRow,state:'awaiting_review',qa});
  await store.adaptiveClaim('qa',qa.request_digest,qa.deadline_ms);
  await assert.rejects(store.adaptiveClaim('qa',qa.request_digest,qa.deadline_ms),/not_admitted/);
  db.values.get(`${ROOT}/runs/2026-10-01`).blob='e'.repeat(64);
  await assert.rejects(store.adaptivePut({...testRow,state:'running'}),/original_intent_changed/);
  db.values.get(`${ROOT}/runs/2026-10-01`).blob=daily.blob;
  time.now+=1800001;
  await assert.rejects(store.adaptiveClaim('qa',qa.request_digest,qa.deadline_ms),/lease_lost/);
});

test('adaptive artifact files are immutable in their own namespace and preserve daily bytes', async () => {
  const {store}=await adaptiveFixture(), name='2026-10-01-artifact.json';
  await store.filePut(name,Buffer.from('original').toString('base64'));
  await store.adaptiveFilePut(name,Buffer.from('test result').toString('base64'));
  assert.equal(Buffer.from(await store.fileGet(name),'base64').toString(),'original');
  assert.equal(Buffer.from(await store.adaptiveFileGet(name),'base64').toString(),'test result');
  await assert.rejects(store.adaptiveFilePut(name,Buffer.from('replacement').toString('base64')),/identity_conflict/);
});
