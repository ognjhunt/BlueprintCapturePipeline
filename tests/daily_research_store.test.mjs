import {test} from 'node:test';
import assert from 'node:assert/strict';
import {randomBytes,createHash} from 'node:crypto';
import {readFileSync,mkdtempSync,writeFileSync,rmSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {pathToFileURL} from 'node:url';
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

for (const profile of ['owner-readonly-mcp-v1','owner-delegated-research-mcp-v1'])
test(`owner MCP creation requires frozen binding and exact current ${profile}`,async()=>{
  const {db,store}=await fixture(),binding=[{server_label:'synthetic-owner-connection'}];
  const hash=createHash('sha256').update(JSON.stringify(binding)).digest('hex');
  const value={...row(),mcp_profile:profile,mcp_binding:binding,
    metadata:{...row().metadata,mcp_binding_digest:hash}};
  await store.put(value);
  await assert.rejects(store.put({...value,mcp_binding:[]}),/research_mcp_binding_changed/);
  if(profile==='owner-delegated-research-mcp-v1') {
    assert.equal(db.values.get(`${ROOT}/runs/${value.date}`).mcp_profile,profile);
    await assert.rejects(store.put({...value,mcp_profile:'owner-readonly-mcp-v1'}),/research_mcp_profile_changed/);
  } else assert.equal(db.values.get(`${ROOT}/runs/${value.date}`).mcp_profile,undefined);
  await assert.rejects(store.createCheck(value.date,value.metadata),/research_mcp_profile_changed/);
  db.values.get(ROOT).config={mcp_profile:profile==='owner-readonly-mcp-v1'?'owner-delegated-research-mcp-v1':'owner-readonly-mcp-v1'};
  await assert.rejects(store.createCheck(value.date,value.metadata),/research_mcp_profile_changed/);
  db.values.get(ROOT).config={mcp_profile:profile};
  await store.createCheck(value.date,value.metadata);
  await assert.rejects(store.createCheck(value.date,value.metadata),/not_admitted/);
  assert.deepEqual((await store.get(value.date)).mcp_binding,binding);
});

test('new owner MCP vault metadata is immutable while old charged intents stay unchanged',async()=>{
  const {store}=await fixture(),binding=[{server_label:'synthetic-owner-connection'}];
  const vaults=[{credential_id:'credential_synthetic',vault_id:'vault_synthetic'}];
  const hash=value=>createHash('sha256').update(JSON.stringify(value)).digest('hex');
  const value={...row(),mcp_profile:'owner-readonly-mcp-v1',mcp_binding:binding,mcp_vault_binding:vaults,
    metadata:{...row().metadata,mcp_binding_digest:hash(binding),mcp_vault_binding_digest:hash(vaults)}};
  await store.put(value);
  await assert.rejects(store.put({...value,mcp_vault_binding:[]}),/research_mcp_vault_binding_changed/);
  const legacy={...value,metadata:{...value.metadata}};delete legacy.metadata.mcp_vault_binding_digest;
  await assert.rejects(store.put(legacy),/firestore_intent_conflict/);
  assert.deepEqual((await store.get(value.date)).mcp_vault_binding,vaults);
});

test('supported source commits await the existing learning writer, and observation failure cannot undo research',async()=>{
  const {db,store}=await fixture(),calls=[],sourceHashes=[];let committed;
  store.learning=async request=>{
    assert.equal((await store.get(request.day)).state,'reviewed');calls.push(request);
    committed=structuredClone(db.values.get(`${ROOT}/runs/${request.day}`));
    sourceHashes.push(createHash('sha256').update(JSON.stringify(committed)).digest('hex'));
    return {event:{eventId:'source-bound-event'},append:'existing'};
  };
  await store.put(row());assert.equal(calls.length,0);
  await store.put({...row(),state:'reviewed'});
  assert.deepEqual(calls,[{op:'learning_after_run',day:'2026-09-30'}]);
  assert.deepEqual(db.values.get(`${ROOT}/runs/2026-09-30`),committed);
  const hash=committed.blob;
  assert.equal(db.values.get(`${ROOT}/learningObservations/${hash}`).status,'observed');
  await store.learning({op:'learning_after_run',day:'2026-09-30'});
  assert.equal(sourceHashes[0],sourceHashes[1]);
  store.learning=async()=>{throw new Error('private provider message contains spaces');};
  assert.equal(await store.put({...row(),state:'reviewed'}),true);
  assert.equal((await store.get('2026-09-30')).state,'reviewed');
  assert.deepEqual(db.values.get(`${ROOT}/runs/2026-09-30`),committed);
  assert.equal(db.values.get(`${ROOT}/learningObservations/${hash}`).code,'research_learning_observation_unavailable');
});

test('terminal collection alone may publish validated evidence while control stays stopped', async () => {
  const {db,store}=await fixture(), proof={session_id:'synthetic-session',qa_artifact_sha256:'a'.repeat(64)}, writes=[];
  const workflow={enabled:true,qa_authority_reference:'owner-qa',publication_authority_reference:'owner-publication'};
  Object.assign(db.values.get(ROOT),{workflow});
  store.publisher={prepare:async()=>({request_digest:'b'.repeat(64)}),reconcile:async()=>null,write:async()=>writes.push('one')};
  const value={...row(),state:'reviewed',session_id:proof.session_id,
    qa:{state:'validated',artifact_digest:proof.qa_artifact_sha256,
      terminal_collection_recovery:{native_receipt:proof,workflow_authority:workflow}},
    delivery:{notion:{state:'acknowledged'},sheets:{state:'pending'}}};
  await store.put({...value,state:'creating'});
  db.values.get(ROOT).enabled=false;
  store.terminalCollectionReceipt=proof;
  await store.put(value);
  for(const op of ['create_check','qa_check','qa_retry_check','repair_check','configure','learning_context'])
    await assert.rejects(store.dispatch({op}),/terminal_qa_operation_forbidden/);
  await store.publish(value.date);
  assert.deepEqual(writes,['one']); assert.equal(db.values.get(ROOT).enabled,false);
  db.values.get(ROOT).workflow.publication_authority_reference='changed';
  await assert.rejects(store.publish(value.date),/publication_authority_changed/);
  db.values.get(ROOT).workflow={...workflow,publication_authority_reference:'owner-publication'};
  store.terminalCollectionReceipt={...proof,qa_artifact_sha256:'c'.repeat(64)};
  await assert.rejects(store.publish(value.date),/validated_receipt_required/);
  assert.deepEqual(writes,['one']);
});

test('only the exact terminal collector command enables the stopped publication pipe', async () => {
  const temporary=mkdtempSync(join(tmpdir(),'research-terminal-command-'));
  const baseline={baseline_id:'baseline-20261002',attempt_number:1,
    root:ROOT+'/baselines/baseline-20261002',authority_reference:'Sentinel_c2046c5f146c81918921eba1ed7f6caa',soft_total_usd:25};
  const context={test_id:'baseline-20261002-attempt-0001',day:'2026-10-01',baseline,
    terminal_collection_receipt:{schema_version:'blueprint.qa-terminal-reconciliation.v1',
      source_blob_sha256:'59327fce14de04a18679932162a4342ddd3513b6e123d43dbc85d2693e957b9b'}};
  const source=readFileSync(new URL('../tools/daily_research/operators/research-perplexity-canary.mjs',import.meta.url),'utf8')
    .replace('__RESEARCH_PACKAGE_URL__',new URL('../',import.meta.url).href)
    .replace('__CANARY_CONTEXT__',JSON.stringify(context));
  const file=join(temporary,'terminal.mjs');writeFileSync(file,source);
  try {
    const {CanaryChannel}=await import(pathToFileURL(file).href);
    const channel=new CanaryChannel(new MemoryFirestore());
    for(const op of ['stage','configure','create_check','qa_check','qa_retry_check','repair_check','learning_context'])
      await assert.rejects(channel.call({op,day:context.day}),/terminal_qa_operation_forbidden/);
    assert.equal(await channel.call({op:'control'}),null);
  } finally {rmSync(temporary,{recursive:true,force:true});}
});

test('learning uses the existing fenced pipe and absent control performs no handler work', async () => {
  const {db,store,time}=await fixture();
  let invoked=0; store.learning=async request=>{invoked++;return {op:request.op};};
  assert.equal(await store.dispatch({op:'learning_context',day:row().date,allow_create:true}),null);
  assert.equal(invoked,0);
  db.values.get(ROOT).learning={enabled:true};
  assert.deepEqual(await store.dispatch({op:'learning_context',day:row().date,allow_create:true}),{op:'learning_context'});
  time.now+=180001;
  await assert.rejects(store.dispatch({op:'learning_context',day:row().date,allow_create:true}),/lease_lost/);
  assert.equal(invoked,1);
});

test('agent history admits the original company binding without a relevance grant or frozen preload',async()=>{
  const {db,store,time}=await fixture(),expiry=new Date(time.now+60000).toISOString();
  const binding={enabled:true,binding:{companyId:'authorized-company',expiresAt:expiry},
    businessScope:{subjectKeys:['company-history'],expiresAt:expiry}};
  Object.assign(db.values.get(ROOT),{learning:binding,config:{history_profile:'agent-history-v1'}});
  const value={...row(),history_profile:'agent-history-v1',history_binding:binding};
  await store.put(value);
  const calls=[];store.learning=async(request,scope)=>{calls.push({request,scope});return {ok:true,rows:[],next_cursor:null,coverage:{complete:true},semantic:{status:'unavailable'}};};
  await store.createCheck(value.date,value.metadata);
  const request={op:'history_search',day:value.date,query:'agent chosen unfamiliar task',filters:{city:'Seattle'},page_size:3};
  assert.equal((await store.dispatch(request)).ok,true);
  assert.deepEqual(calls,[{request,scope:binding}]);
  await assert.rejects(store.put({...value,history_binding:{...binding,binding:{companyId:'other'}}}),/company_history_intent_conflict/);
  store.learning=async()=>{throw new Error('company_history_cursor_invalid');};
  assert.equal((await store.dispatch({...request,cursor:'wrong'})).error.code,'company_history_cursor_invalid');
  store.learning=async()=>{throw new Error('PRIVATE_UPSTREAM_SECRET');};
  assert.equal((await store.dispatch(request)).error.code,'company_history_unavailable');
  time.now+=60001;
  await assert.rejects(store.dispatch(request),/company_history_scope_expired/);
  store.terminalCollectionReceipt={};
  await assert.rejects(store.dispatch(request),/terminal_qa_operation_forbidden/);
});

test('legacy saved sessions cannot silently acquire the company history profile',async()=>{
  const {store,db,time}=await fixture();await store.put(row());
  const expiry=new Date(time.now+60000).toISOString();
  const binding={enabled:true,binding:{expiresAt:expiry},businessScope:{expiresAt:expiry}};
  Object.assign(db.values.get(ROOT),{learning:binding,config:{history_profile:'agent-history-v1'}});
  await assert.rejects(store.put({...row(),history_profile:'agent-history-v1',history_binding:binding}),/company_history_intent_conflict/);
  let called=false;store.learning=async()=>{called=true;};
  await assert.rejects(store.dispatch({op:'history_fetch',day:row().date,record_id:'chosen'}),/company_history_authority_changed/);
  assert.equal(called,false);
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

test('two retry claims serialize, survive replacement, and cannot renew the phase or original claim', async () => {
  const {db,store,time}=await fixture();
  db.values.get(ROOT).workflow={enabled:true,qa_authority_reference:'synthetic-qa',publication_authority_reference:'synthetic-pub'};
  await store.put(row());
  const request='a'.repeat(64), deadline=time.now+120000;
  const value={...row(),state:'awaiting_review',qa:{state:'qa_input_unresolved',request_digest:request,deadline_ms:deadline}};
  await store.put(value);await store.qaCheck(value.date,request,deadline);
  const phase={started_at:new Date(time.now).toISOString(),previous_qa:structuredClone(value.qa)};
  value.qa_retry_continuation=phase;
  const error={stage:'provider_submission',http_status:503,code:'service_unavailable_error'};
  value.qa.input_error_receipt=error;
  value.qa.input_retries=[{number:1,idempotency_key:value.run_key+':qa',not_before:new Date(time.now+5000).toISOString()}];
  await store.put(value);
  const next=time.now+600000;
  await assert.rejects(store.qaRetryCheck(value.date,request,next,1),/not_admitted/);
  time.now+=5001;
  const results=await Promise.allSettled([store.qaRetryCheck(value.date,request,next,1),store.qaRetryCheck(value.date,request,next,1)]);
  assert.equal(results.filter(r=>r.status==='fulfilled').length,1);
  await store.release();const replacement=new Store(db,()=>time.now,'replacement');await replacement.acquire();
  await replacement.put(value);
  await assert.rejects(replacement.qaRetryCheck(value.date,request,next,1),/not_admitted/);
  value.qa.input_retries[0].error_receipt=error;
  value.qa.input_retries.push({number:2,idempotency_key:value.run_key+':qa',not_before:new Date(time.now+15000).toISOString()});
  await replacement.put(value);
  await assert.rejects(replacement.qaRetryCheck(value.date,request,next,2),/not_admitted/);
  time.now+=15001;await replacement.qaRetryCheck(value.date,request,next,2);await replacement.put(value);
  const persisted=db.values.get(`${ROOT}/runs/${value.date}`);
  assert.equal(persisted.qa_request_claimed,true);assert.deepEqual(persisted.qa_retry_claims,{'1':request,'2':request});
  await assert.rejects(replacement.qaRetryCheck(value.date,request,next,2),/not_admitted/);
  await assert.rejects(replacement.put({...value,qa_retry_continuation:{...phase,started_at:new Date(time.now).toISOString()}}),/phase_already_bound/);
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

test('immutable blob receipts bind verified bytes and native creation time without writes', async () => {
  const {db, store} = await fixture(), raw = Buffer.from('{"qa":"completed"}');
  await store.filePut('2026-09-30-qa.json', raw.toString('base64'));
  const hash = db.values.get(`${ROOT}/files/2026-09-30-qa.json`).blob;
  const before = JSON.stringify([...db.values]);
  const originalDoc = db.doc.bind(db);
  let time = {seconds:1790933968, nanoseconds:549726000};
  db.doc = path => {
    const ref = originalDoc(path), get = ref.get.bind(ref);
    ref.get = async () => {const snap = await get(); if (path === `${ROOT}/blobs/${hash}`) snap.createTime = time; return snap;};
    return ref;
  };
  assert.deepEqual(await store.dispatch({op:'blob_receipt',hash}), {sha256:hash,bytes:raw.toString('base64'),created_at:time});
  assert.equal(JSON.stringify([...db.values]), before);
  for (const invalid of [undefined, {seconds:1790933968,nanoseconds:1000000000}]) {
    time = invalid;
    await assert.rejects(store.blobReceipt(hash), /creation_time_unavailable/);
  }
  db.values.get(`${ROOT}/blobs/${hash}/chunks/0`).bytes = Buffer.from('corrupt');
  await assert.rejects(store.blobReceipt(hash));
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

test('replacement observer discovers saved publication cancellation states while control is disabled without admissions',async()=>{
  const {db,store}=await fixture();db.values.get(ROOT).enabled=false;
  const replacement=new Store(db,Date.now,'replacement');
  for(const state of ['running','input_unresolved','cancel_pending']) {
    db.values.set(`${ROOT}/runs/2026-09-30`,{date:'2026-09-30',state:'reviewed',publication_state:state});
    const before=JSON.stringify([...db.values]);
    assert.equal(await replacement.dispatch({op:'active_qa'}),'2026-09-30');
    assert.equal(JSON.stringify([...db.values]),before);
  }
  db.values.get(`${ROOT}/runs/2026-09-30`).publication_state='cancelled';
  assert.equal(await replacement.activeQA(),null);
});
