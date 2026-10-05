import {test} from 'node:test';
import assert from 'node:assert/strict';
import {randomBytes,createHash} from 'node:crypto';
import {readFileSync,mkdtempSync,writeFileSync,rmSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {pathToFileURL} from 'node:url';
import {Store, ROOT, LeaseChannel, ADAPTIVE_TEST} from '../tools/daily_research/firestore_bridge.mjs';
import {MemoryFirestore} from './fixtures/daily_research/firestore-memory.mjs';

async function fixture(now=Date.now()) {
  const db = new MemoryFirestore(), time = {now};
  db.values.set(ROOT, {enabled: true, schema_version: 'blueprint.research-control.v1'});
  const store = new Store(db, () => time.now, 'first');
  await store.acquire(); return {db, store, time};
}
const row = () => ({date: '2026-09-30', run_key: 'blueprint-researcher:2026-09-30', metadata: {run_key: 'day', payload_digest: 'hash'},
  state: 'creating', cleanup_required: true});

test('unstarted repair snapshot retains raw evidence without inventing an input',async()=>{
  const {db,store}=await fixture(),raw=Buffer.from('{"candidates":null}');
  const code='findall_tool_registry_binding_changed';
  const revision={number:1,state:'no_progress',error:code,feedback:[],input_attempted:false,deadline_ms:1,
    input_error_receipt:{stage:'preconditions',class:'Refusal',code,http_status:null,request_id:null}};
  const value={...row(),state:'failed',artifact_downloaded:true,
    raw_output_digest:createHash('sha256').update(raw).digest('hex'),validation_repairs:[revision]};
  await store.filePut(`${value.date}-artifact.json`,raw.toString('base64'));
  await store.put(value);
  const snapshot=await store.snapshot(value.date);
  assert.deepEqual(Buffer.from(snapshot.files.artifact,'base64'),raw);
  assert.ok(!snapshot.missing_files.includes('repair-1-input'));
  assert.equal(snapshot.files['repair-1-input'],undefined);
  assert.deepEqual(snapshot.row.validation_repairs,[revision]);
  // Unexpected input bytes remain visible for the portable export validator.
  await store.filePut(`${value.date}-repair-1-input.json`,Buffer.from('{}').toString('base64'));
  assert.equal(Buffer.from((await store.snapshot(value.date)).files['repair-1-input'],'base64').toString(),'{}');
  // Native submission claims cannot be concealed by a no-input row receipt.
  db.values.get(`${ROOT}/runs/${value.date}`).repair_claims={'1':'claimed-digest'};
  await assert.rejects(store.snapshot(value.date),/validation_repair_export_binding_mismatch/);
});

for(const attempted of [true,undefined,null,0])
test(`submitted or legacy repair snapshot still requires input (${attempted})`,async()=>{
  const {store}=await fixture(),revision={number:1,state:'no_progress'};
  if(attempted!==undefined) revision.input_attempted=attempted;
  const value={...row(),validation_repairs:[revision]};
  await store.put(value);
  assert.ok((await store.snapshot(value.date)).missing_files.includes('repair-1-input'));
});

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
    delivery:{notion:{state:'acknowledged'},sheets:{state:'pending',payload:{candidates:[]}}}};
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

test('inventory pages are complete against the independently pinned artifact, not self-consistent replacement manifests',async()=>{
  const {verificationDigest}=await import('../tools/daily_research/verification-digest.mjs');
  const {store,db}=await fixture(),r=row(),outputDigest='a'.repeat(64);
  const records=Array.from({length:3},(_,i)=>({operator:'Invented operator',site:`Invented site ${i}`,location:null,
    task_hypothesis:null,source_urls:[`https://fixture.example/${i}`],evidence_gap:'Actual work unknown',disposition:'unresolved'}));
  const artifact=Buffer.from(JSON.stringify({discovery_inventory:records})),artifactSha=createHash('sha256').update(artifact).digest('hex');
  const filename=`${r.date}-inventory-${outputDigest}-0.json`;
  const page={version:'blueprint.discovery-inventory.v1',run_key:r.run_key,source_output_digest:outputDigest,start:0,end:3,records};
  const raw=Buffer.from(JSON.stringify(page)+'\n'),pageSha=createHash('sha256').update(raw).digest('hex');
  const manifest={version:page.version,source_output_digest:outputDigest,record_count:3,page_count:1,complete_retention:true,
    records_digest:verificationDigest(records),source_artifact_file:`${r.date}-artifact.json`,source_artifact_sha256:artifactSha,
    pages:[{index:0,file:filename,sha256:pageSha,bytes:raw.length,start:0,end:3}]};
  Object.assign(r,{artifact_downloaded:true,raw_output_digest:artifactSha,packet:{candidates:[],discovery_inventory_manifest:manifest}});
  await store.put(r);await store.filePut(`${r.date}-artifact.json`,artifact.toString('base64'));await store.filePut(filename,raw.toString('base64'));
  assert.ok((await store.snapshot(r.date)).files[`inventory-${outputDigest}-0`]);
  const replaced={...page,end:2,records:records.slice(0,2)},replacedRaw=Buffer.from(JSON.stringify(replaced)+'\n');
  await assert.rejects(store.filePut(filename,replacedRaw.toString('base64')),/artifact_identity_conflict/);
  const blob=await store.blobPut(replacedRaw.toString('base64'));db.values.set(`${ROOT}/files/${filename}`,{blob});
  for(const rebindRecords of [true,false]) {
    const forged={...manifest,record_count:2,records_digest:rebindRecords?verificationDigest(replaced.records):manifest.records_digest,
      pages:[{...manifest.pages[0],end:2,bytes:replacedRaw.length,sha256:createHash('sha256').update(replacedRaw).digest('hex')}]};
    await store.put({...r,packet:{...r.packet,discovery_inventory_manifest:forged}});
    await assert.rejects(store.snapshot(r.date),/discovery_inventory_(source_binding|manifest)_invalid/);
  }
});

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

test('owner direction swap is a fenced compare-and-swap with a create-only audit chain',async()=>{
  const {db,store,time}=await paidFixture();db.replay=true;
  const first=entry(direction());
  assert.deepEqual(await setDirection(store,null,first),{enabled:true,sha256:first.sha256,version:1,audit:'created'});
  assert.deepEqual(db.values.get(ROOT).paid_expansion,{enabled:true,current:first});
  const audit=db.values.get(`${ROOT}/paidExpansionDirections/${first.sha256}`);
  assert.deepEqual({...audit,recorded_at:undefined},{...first,recorded_at:undefined});
  assert.equal(audit.recorded_at,new Date(time.now).toISOString());
  await assert.rejects(setDirection(store,null,first),/paid_expansion_direction_conflict/);
  const second=entry(direction(2,first.sha256,{per_run_limit_usd:'20.00'}));
  const competing=entry(direction(2,first.sha256,{per_run_limit_usd:'30.00'}));
  await setDirection(store,first.sha256,second);
  await assert.rejects(setDirection(store,first.sha256,competing),/paid_expansion_direction_conflict/);
  assert.equal(db.values.has(`${ROOT}/paidExpansionDirections/${competing.sha256}`),false);
  for(const [value,code] of [
    [entry(direction(4,second.sha256)),'direction_chain_invalid'],[entry(direction(3,first.sha256)),'direction_chain_invalid'],
    [entry(direction(3,second.sha256,{per_run_limit_usd:'200.00'})),'limit_invalid'],
    [entry(direction(3,second.sha256,{per_run_limit_usd:'10.0'})),'limit_invalid'],
    [entry(direction(3,second.sha256,{scope:{...direction().scope,agent_id:'agent_other'}})),'scope_mismatch'],
    [entry(direction(3,second.sha256,{approval_reference:'PENDING-owner'})),'direction_invalid'],
    [entry(direction(3,second.sha256,{reason:'café'})),'direction_invalid'],
    [entry(direction(3,second.sha256,{expires_at:'2027-10-06T18:00:00+00:00'})),'direction_invalid'],
    [{...entry(direction(3,second.sha256)),sha256:'0'.repeat(64)},'direction_digest_mismatch'],
    [{...entry(direction(3,second.sha256)),uri:'gs://other/direction.json'},'direction_digest_mismatch']])
    await assert.rejects(setDirection(store,second.sha256,value),new RegExp(`^Error: paid_expansion_${code}$`));
  await assert.rejects(store.dispatch({op:'paid_expansion_set',value:{enabled:true,current:second}}),/paid_expansion_request_invalid/);
  assert.deepEqual((await store.dispatch({op:'paid_expansion_audit'})).map(record=>record.version),[1,2]);
  time.now+=180001;
  await assert.rejects(setDirection(store,second.sha256,entry(direction(3,second.sha256))),/firestore_lease_lost/);
  assert.deepEqual(db.values.get(ROOT).paid_expansion.current,second);
});

test('the emergency brake needs no new direction while re-enabling does',async()=>{
  const {db,store}=await paidFixture();
  const first=entry(direction());await setDirection(store,null,first);
  // A brake keeps the current record exactly, even one this package can no longer parse.
  db.values.get(ROOT).paid_expansion.current.direction.future_field=true;
  const stored=structuredClone(db.values.get(ROOT).paid_expansion.current);
  assert.equal((await setDirection(store,first.sha256,stored,false)).enabled,false);
  assert.deepEqual(db.values.get(ROOT).paid_expansion,{enabled:false,current:stored});
  assert.equal((await setDirection(store,first.sha256,stored,false)).enabled,false);
  await assert.rejects(setDirection(store,first.sha256,stored,true),/reenable_requires_new_direction/);
  await assert.rejects(setDirection(store,first.sha256,{...stored,version:7},false),/paid_expansion_direction_conflict/);
  const next=entry(direction(2,first.sha256));
  await assert.rejects(setDirection(store,first.sha256,next,false),/chain_invalid/);
  await setDirection(store,first.sha256,next);
  assert.deepEqual(db.values.get(ROOT).paid_expansion,{enabled:true,current:next});
});

test('configure keeps the owner direction and neither configure nor init can write one',async()=>{
  const {db,store}=await paidFixture();
  const first=entry(direction());await setDirection(store,null,first);
  const replacement={schema_version:'blueprint.research-control.v1',enabled:true,project_id:PROJECT,agent_id:AGENT,config:{x:1}};
  await store.dispatch({op:'configure',value:replacement});
  assert.deepEqual(db.values.get(ROOT).paid_expansion,{enabled:true,current:first});
  assert.deepEqual(db.values.get(ROOT).config,{x:1});
  await store.dispatch({op:'configure',value:{...replacement,paid_expansion:{enabled:true,current:first}}});
  for(const paid of [{enabled:false,current:first},null,{enabled:true,current:entry(direction(1,null,{per_run_limit_usd:'90.00'}))}])
    await assert.rejects(store.dispatch({op:'configure',value:{...replacement,paid_expansion:paid}}),/paid_expansion_requires_direction_operation/);
  assert.deepEqual(db.values.get(ROOT).paid_expansion,{enabled:true,current:first});
  const fresh=new MemoryFirestore(),other=new Store(fresh,()=>Date.now(),'other');
  await assert.rejects(other.dispatch({op:'init',value:{schema_version:'blueprint.research-control.v1',enabled:false,
    paid_expansion:{enabled:true,current:first}}}),/paid_expansion_requires_direction_operation/);
  assert.equal(fresh.values.has(ROOT),false);
});

class FakeBucket {
  constructor() {this.name='blueprint-8c1ca.appspot.com';this.objects=new Map();this.generation=1000;}
  file(name) {
    const bucket=this;
    return {name,async save(raw,options) {
      if(options?.preconditionOpts?.ifGenerationMatch!==0) throw new Error('create-only precondition required');
      if(bucket.objects.has(name)) {const error=new Error('exists');error.code=412;throw error;}
      bucket.objects.set(name,{raw:Buffer.from(raw),generation:++bucket.generation});
    },async getMetadata() {
      const object=bucket.objects.get(name);if(!object) throw new Error('missing');
      return [{generation:object.generation,size:String(object.raw.length)}];
    },async download() {
      const object=bucket.objects.get(name);if(!object) throw new Error('missing');return [Buffer.from(object.raw)];
    }};
  }
}

test('direction objects are content-addressed, validated and create-only without a lease',async()=>{
  const {db}=await paidFixture(),bucket=new FakeBucket();
  const store=new Store(db,()=>Date.now(),'operator',null,null,null,null,false,bucket);
  const first=entry(direction()),raw=Buffer.from(canonicalJSON(first.direction));
  const put=hash=>store.dispatch({op:'paid_expansion_object_put',sha256:hash,bytes:raw.toString('base64')});
  const stored=await put(first.sha256);
  assert.deepEqual({...stored,generation:undefined},{uri:first.uri,sha256:first.sha256,bytes:raw.toString('base64'),generation:undefined});
  assert.deepEqual(await put(first.sha256),stored);
  assert.equal(bucket.objects.size,1);
  assert.deepEqual(await store.dispatch({op:'paid_expansion_object_get',sha256:first.sha256}),stored);
  await assert.rejects(put('0'.repeat(64)),/digest_mismatch/);
  const spaced=Buffer.from(JSON.stringify(first.direction,null,1)),spacedHash=createHash('sha256').update(spaced).digest('hex');
  await assert.rejects(store.dispatch({op:'paid_expansion_object_put',sha256:spacedHash,bytes:spaced.toString('base64')}),/digest_mismatch/);
  const typo=entry(direction(1,null,{per_run_limit_usd:'1000.00'})),typoRaw=Buffer.from(canonicalJSON(typo.direction));
  await assert.rejects(store.dispatch({op:'paid_expansion_object_put',sha256:typo.sha256,bytes:typoRaw.toString('base64')}),/limit_invalid/);
  await assert.rejects(store.dispatch({op:'paid_expansion_object_get',sha256:typo.sha256}),/paid_expansion_object_missing/);
  await assert.rejects(store.dispatch({op:'paid_expansion_object_get',sha256:'../x'}),/paid_expansion_request_invalid/);
  bucket.objects.get(`operations/research/paid-expansion/${first.sha256}/direction.json`).raw=Buffer.from('{}');
  await assert.rejects(store.dispatch({op:'paid_expansion_object_get',sha256:first.sha256}),/paid_expansion_object_conflict/);
  await assert.rejects(new Store(db).dispatch({op:'paid_expansion_object_get',sha256:first.sha256}),/object_transport_unavailable/);
  assert.equal(db.values.get(ROOT).paid_expansion,undefined);
});

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

test('new reservations honor a lower live amount atomically, while existing claims stay recoverable',async()=>{
  for(const amount of ['6.00','1.00']) {
    const {db,store}=await paidFixture();
    const original=entry(direction(1,null,{per_run_limit_usd:'30.00'}));await setDirection(store,null,original);
    const value=exaRow('2026-10-05');
    value.paid_expansion_grant=grantFor(original,value,{limit_micros:30000000,per_start_max_micros:15000000});
    await store.put(value);
    const lower=entry(direction(2,original.sha256,{per_run_limit_usd:amount}));
    await setDirection(store,original.sha256,lower);
    await assert.rejects(store.put({...value,state:'running',exa_expansion:exaClaim(value,5000000)}),
      /paid_expansion_reservation_exceeds_grant/);
    assert.equal(db.values.get(`${ROOT}/runs/${value.date}`).exa_expansion_intent_digest,null);
    const admitted={...value,state:'running',exa_expansion:exaClaim(value,1000000)};
    await store.put(admitted);
    await setDirection(store,lower.sha256,lower,false);
    await store.put({...admitted,exa_expansion:{...admitted.exa_expansion,run_id:'synthetic_existing'}});
    assert.equal((await store.get(value.date)).exa_expansion.run_id,'synthetic_existing');
  }
});

test('a live successor interval and audit bind every new reservation',async()=>{
  for(const change of [{expires_at:'2026-10-05T12:00:10+00:00'},
    {effective_from:'2026-10-05T12:01:00+00:00'}]) {
    const {db,store,time}=await paidFixture(),original=entry(direction());await setDirection(store,null,original);
    const value=exaRow('2026-10-05');value.paid_expansion_grant=grantFor(original,value);await store.put(value);
    const successor=entry(direction(2,original.sha256,change));await setDirection(store,original.sha256,successor);
    time.now+=11000;
    await assert.rejects(store.put({...value,state:'running',exa_expansion:exaClaim(value)}),/paid_expansion_grant_not_admitted/);
    assert.equal(db.values.get(`${ROOT}/runs/${value.date}`).exa_expansion_intent_digest,null);
  }
  const {db,store}=await paidFixture(),original=entry(direction());await setDirection(store,null,original);
  const value=exaRow('2026-10-05');value.paid_expansion_grant=grantFor(original,value);await store.put(value);
  const successor=entry(direction(2,original.sha256));await setDirection(store,original.sha256,successor);
  db.values.delete(`${ROOT}/paidExpansionDirections/${successor.sha256}`);
  await assert.rejects(store.put({...value,state:'running',exa_expansion:exaClaim(value)}),/paid_expansion_grant_not_admitted/);
});

test('direction times must be real calendar instants in both languages',async()=>{
  const {db,store}=await paidFixture();
  for(const expires_at of ['2026-11-31T18:00:00+00:00','2027-02-29T18:00:00+00:00','2026-12-01T24:00:00+00:00'])
    await assert.rejects(setDirection(store,null,entry(direction(1,null,{expires_at}))),/^Error: paid_expansion_direction_invalid$/);
  assert.equal(db.values.get(ROOT).paid_expansion,undefined);
});

test('a control copy dropped by an older release keeps one chain from the audit head',async()=>{
  const {db,store}=await paidFixture();
  const first=entry(direction());await setDirection(store,null,first);
  delete db.values.get(ROOT).paid_expansion; // An older configure replaced control wholesale.
  await assert.rejects(setDirection(store,null,entry(direction())),/paid_expansion_direction_chain_invalid/);
  const second=entry(direction(2,first.sha256,{per_run_limit_usd:'20.00'}));
  await setDirection(store,null,second);
  assert.deepEqual(db.values.get(ROOT).paid_expansion,{enabled:true,current:second});
  assert.deepEqual((await store.dispatch({op:'paid_expansion_audit'})).map(r=>[r.version,r.direction.supersedes]),
    [[1,null],[2,first.sha256]]);
});

test('claims bind the frozen grant, its source and the exact native budget',async()=>{
  const {db,store}=await paidFixture();
  const current=entry(direction());await setDirection(store,null,current);
  const value=exaRow('2026-10-05');
  value.paid_expansion_grant=grantFor(current,value);
  await store.put(value);
  assert.equal(db.values.get(`${ROOT}/runs/${value.date}`).paid_expansion_grant_unbound,false);
  await assert.rejects(store.put({...value,state:'running',exa_expansion:exaClaim(value,2000000,5)}),/paid_expansion_claim_cap_mismatch/);
  await store.put({...value,state:'running',exa_expansion:exaClaim(value,5000000)});
  const sourceless=exaRow('2026-10-06');
  sourceless.paid_expansion_grant=grantFor(current,sourceless,{sources:[]});
  await store.put(sourceless);
  await assert.rejects(store.put({...sourceless,state:'running',exa_expansion:exaClaim(sourceless)}),/paid_expansion_grant_not_admitted/);
});

test('a run manifest written by an older bridge admits no new paid claim',async()=>{
  const {db,store}=await paidFixture();
  const current=entry(direction());await setDirection(store,null,current);
  const value=exaRow('2026-10-05');
  await store.put(value);
  delete db.values.get(`${ROOT}/runs/${value.date}`).paid_expansion_grant_digest; // Older bridge rewrite.
  const regranted={...value,state:'running',paid_expansion_grant:grantFor(current,value)};
  await store.put(regranted);
  assert.equal(db.values.get(`${ROOT}/runs/${value.date}`).paid_expansion_grant_unbound,true);
  await assert.rejects(store.put({...regranted,exa_expansion:exaClaim(regranted)}),/paid_expansion_grant_required/);
  await store.put(regranted);
  await assert.rejects(store.put({...regranted,exa_expansion:exaClaim(regranted)}),/paid_expansion_grant_required/);
});

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

test('FindAll and Exa claims debit one frozen grant within its per-start maximum',async()=>{
  const {db,store,value}=await findallFixture();
  const first=findallEntry(value,'call_a','4'),second=findallEntry(value,'call_b','2.5');
  await store.put(withClaims(value,first));
  await store.put(withClaims(value,first,second));
  assert.deepEqual(Object.keys(db.values.get(`${ROOT}/runs/${value.date}`).findall_claims).sort(),[first[0],second[0]].sort());
  assert.equal(db.values.get(`${ROOT}/runs/${value.date}`).findall_unbound,false);
  // $4 + $2.50 + a $4 Exa cap is $10.50 > $10: refused in the same transaction as the claim.
  await assert.rejects(store.put({...withClaims(value,first,second),exa_expansion:exaClaim(value,4000000)}),
    /paid_expansion_reservation_exceeds_grant/);
  await assert.rejects(store.put(withClaims(value,first,second,findallEntry(value,'call_c','3.51'))),
    /paid_expansion_reservation_exceeds_grant/);
  await assert.rejects(store.put(withClaims(value,first,second,findallEntry(value,'call_d','5.01'))),
    /paid_expansion_reservation_exceeds_grant/);
  await store.put({...withClaims(value,first,second),exa_expansion:exaClaim(value,3500000)});
  assert.equal(db.values.get(`${ROOT}/runs/${value.date}`).exa_expansion_intent_digest!==null,true);
});

test('a FindAll claim is append-only and keeps its provider ID',async()=>{
  const {store,value}=await findallFixture();
  const first=findallEntry(value,'call_a','1');
  await store.put(withClaims(value,first));
  await assert.rejects(store.put({...value,state:'running'}),/findall_claim_already_consumed/);
  const lowered=findallEntry(value,'call_a','0.5');
  await assert.rejects(store.put(withClaims(value,[first[0],lowered[1]])),/findall_claim_already_consumed/);
  const known=[first[0],{...first[1],findall_id:'findall_synthetic',state:'receipt_retained',receipt_file:'x'}];
  await store.put(withClaims(value,known));
  await assert.rejects(store.put(withClaims(value,[first[0],{...known[1],findall_id:'findall_other'}])),
    /findall_original_id_changed/);
  await assert.rejects(store.put(withClaims(value,[first[0],{...known[1],findall_id:null}])),/findall_original_id_changed/);
});

for(const [label,grantChange,controlChange,code] of [
  ['a grant that omits FindAll',{sources:['exa']},null,'paid_expansion_grant_not_admitted'],
  ['the owner brake',null,fixture=>setDirection(fixture.store,fixture.current.sha256,fixture.current,false),
    'paid_expansion_grant_not_admitted'],
  ['a changed release',null,fixture=>{fixture.db.values.get(ROOT).source_commit='e'.repeat(40);},
    'paid_expansion_grant_not_admitted'],
  ['a refused grant',{state:'refused'},null,'paid_expansion_grant_required'],
]) test(`${label} refuses a new FindAll claim`,async()=>{
  const fixture=await findallFixture();
  const value=findallRow('2026-10-06');
  // Grant changes bind at the durable intent; control changes apply to a later claim.
  value.paid_expansion_grant=grantChange?.state==='refused'?{schema_version:'blueprint.research-paid-expansion-grant.v1',
    state:'refused',code:'paid_expansion_disabled',run_key:value.run_key,frozen_at:'2026-10-05T12:00:00+00:00',
    direction_sha256:null}:grantFor(fixture.current,value,{sources:['exa','findall'],...grantChange});
  await fixture.store.put(value);
  if(controlChange) await controlChange(fixture);
  await assert.rejects(fixture.store.put(withClaims(value,findallEntry(value,'call_a','1'))),new RegExp(code));
});

test('a direction without FindAll freezes a grant that cannot admit one',async()=>{
  const {store,value}=await findallFixture(['exa']);
  await assert.rejects(store.put(withClaims(value,findallEntry(value,'call_a','1'))),/paid_expansion_grant_not_admitted/);
  await store.put({...value,state:'running',exa_expansion:exaClaim(value,2000000)});
});

test('FindAll claims need the pinned profile and an exact owner-journal shape',async()=>{
  const {store,value}=await findallFixture();
  const [key,claim]=findallEntry(value,'call_a','1');
  await assert.rejects(store.put(withClaims({...value,findall_profile:undefined},[key,claim])),/findall_profile_invalid/);
  for(const bad of [[key,{...claim,operation_id:'blueprint-researcher:2026-10-04:findall:call_a'}],
      ['0'.repeat(64),claim],[key,{...claim,prepared:{...claim.prepared,maximum_cost_usd:'1e1'}}],
      [key,{...claim,prepared:{...claim.prepared,resource_class:'gpu_canary'}}],
      [key,{...claim,allocation_binding_digest:'sha256:'+'0'.repeat(64)}]])
    await assert.rejects(store.put(withClaims(value,bad)),/findall_claim_binding_invalid/);
});

test('a manifest an older bridge rewrote keeps its FindAll claims and admits no new one',async()=>{
  const {db,store,value}=await findallFixture();
  const first=findallEntry(value,'call_a','1');
  await store.put(withClaims(value,first));
  delete db.values.get(`${ROOT}/runs/${value.date}`).findall_claims; // Older bridge rewrite.
  await store.put(withClaims(value,first));
  assert.equal(db.values.get(`${ROOT}/runs/${value.date}`).findall_unbound,true);
  await assert.rejects(store.put(withClaims(value,first,findallEntry(value,'call_b','1'))),/paid_expansion_grant_required/);
  await assert.rejects(store.put({...value,state:'running'}),/findall_claim_already_consumed/);
});

test('portable snapshots carry every FindAll receipt with its digest',async()=>{
  const {store,value}=await findallFixture();
  const raw=Buffer.from('{"findall_id":"findall_synthetic"}\n'),hash=createHash('sha256').update(raw).digest('hex');
  const file=`${value.date}-tool-findall-${'f'.repeat(64)}.json`;
  await store.filePut(file,raw.toString('base64'));
  const [key,claim]=findallEntry(value,'call_a','1');
  const retained=withClaims(value,[key,{...claim,findall_id:'findall_synthetic',state:'receipt_retained',
    receipt_file:file,receipt_sha256:hash}]);
  await store.put(retained);
  const snapshot=await store.snapshot(value.date);
  assert.equal(Buffer.from(snapshot.files[file.slice(value.date.length+1,-5)],'base64').toString(),raw.toString());
  await store.put({...retained,parallel_findall_reads:{call_read:{file,sha256:'0'.repeat(64),bytes:raw.length}}});
  await assert.rejects(store.snapshot(value.date),/findall_receipt_digest_mismatch/);
});


test('FindAll snapshots export as one immutable file with a bound page layout; the parts format fails closed',async()=>{
  const {store,value}=await findallFixture();
  const hash=raw=>createHash('sha256').update(raw).digest('hex');
  const encode=value=>Buffer.from(JSON.stringify(value)+'\n');
  const snapshot={run:{findall_id:'findall_synthetic'},candidates:[{status:'matched',future:'🚀'.repeat(25000)}],unknown:true};
  const raw=encode(snapshot),page_count=Math.ceil([...raw.toString()].length/24000);
  const file=`${value.date}-tool-findall-read-${'a'.repeat(64)}.json`;
  await store.filePut(file,raw.toString('base64'));
  const receipt={file,sha256:hash(raw),bytes:raw.length,schema_version:'blueprint.findall-snapshot.v2',page_chars:24000,page_count};
  const settle=`${value.date}-tool-findall-settle-result-${hash(raw)}.json`;
  await store.filePut(settle,raw.toString('base64'));
  const settled={operation_id:`${value.run_key}:findall:call_a`,state:'terminal',receipts:[{...receipt,file:settle}],
    result_receipt:{...receipt,file:settle}};
  const retained={...withClaims(value,findallEntry(value,'call_a','1')),
    parallel_findall_reads:{call_read:{operation:'result',findall_id:'findall_synthetic',...receipt}},
    parallel_findall_settlements:{[findallEntry(value,'call_a','1')[0]]:settled}};
  await store.put(retained);
  const exported=await store.snapshot(value.date);
  assert.equal(exported.files[file.slice(value.date.length+1,-5)],raw.toString('base64'));
  assert.equal(exported.files[settle.slice(value.date.length+1,-5)],raw.toString('base64'));
  assert.ok(!Object.keys(exported.files).some(name=>name.includes('-part-')));
  for(const change of [{page_count:page_count+1},{page_count:0},{page_chars:1000},{schema_version:'blueprint.findall-snapshot-parts.v1'},
      {parts:[]},{bytes:raw.length-1}]) {
    await store.put({...retained,parallel_findall_reads:{call_read:{...receipt,...change}}});
    await assert.rejects(store.snapshot(value.date),/findall_(snapshot_binding_invalid|receipt_digest_mismatch)/);
  }
  const {page_chars,...partial}=receipt;
  await store.put({...retained,parallel_findall_reads:{call_read:partial}});
  await assert.rejects(store.snapshot(value.date),/findall_snapshot_binding_invalid/);
  const truncated=raw.subarray(0,raw.length-2),cut=file.replace('a'.repeat(64),'c'.repeat(64));
  await store.filePut(cut,truncated.toString('base64'));
  await store.put({...retained,parallel_findall_reads:{call_read:{...receipt,file:cut,sha256:hash(truncated),bytes:truncated.length}}});
  await assert.rejects(store.snapshot(value.date),/findall_snapshot_binding_invalid/);
});


test('owner runtime adjustment is fenced, next-run-only and preserves all other control fields',async()=>{
  const {db,store}=await fixture();
  const source='c'.repeat(40),original=db.values.get(ROOT);
  const config={enabled:true,discovery_profile:'adaptive-sites-v1',max_runtime_seconds:3600,qa_reserved_seconds:900,
    soft_target_usd:5,recurring_budget_authority_reference:'synthetic-owner'};
  db.values.set(ROOT,{...original,source_commit:source,config,paid_expansion:{enabled:false,current:null},
    workflow:{enabled:true},learning:{enabled:true}});
  const day='2026-10-03',path=`${ROOT}/runs/${day}`;
  const historical={state:'completed',research_runtime_seconds:1200,total_runtime_seconds:1800,blob:'historical'};
  db.values.set(path,historical);
  for(const minutes of[60,120,180,240]) {
    const before=structuredClone(db.values.get(ROOT));
    const result=await store.dispatch({op:'runtime_set',total_seconds:minutes*60,qa_seconds:900,
      expected_config:before.config,expected_source_commit:source});
    assert.equal(result.active_rows,0);
    const after=db.values.get(ROOT);
    assert.deepEqual(after,{...before,config:{...before.config,max_runtime_seconds:minutes*60,qa_reserved_seconds:900}});
    assert.deepEqual(db.values.get(path),historical);
  }
  const before=structuredClone(db.values.get(ROOT));
  for(const request of[
    {expected_config:{...before.config,max_runtime_seconds:3600}},
    {expected_source_commit:'d'.repeat(40)},
    {total_seconds:14401},{total_seconds:true},{qa_seconds:14400},
  ]) {
    await assert.rejects(store.dispatch({op:'runtime_set',total_seconds:7200,qa_seconds:900,
      expected_config:before.config,expected_source_commit:source,...request}),/runtime_(control_changed|phase_envelope_invalid)/);
    assert.deepEqual(db.values.get(ROOT),before);
  }
});

for(const active of[{state:'running'},{qa_state:'qa_running'},{repair_state:'input_unresolved'},
  {publication_state:'cancel_pending'}])
test(`runtime adjustment atomically refuses active ${Object.keys(active)[0]}`,async()=>{
  const {db,store}=await fixture();
  const source='c'.repeat(40),config={discovery_profile:'adaptive-sites-v1',max_runtime_seconds:3600,qa_reserved_seconds:900};
  db.values.set(ROOT,{...db.values.get(ROOT),source_commit:source,config});
  db.values.set(`${ROOT}/runs/2026-10-03`,{state:'completed',...active});
  const before=structuredClone(db.values.get(ROOT));
  await assert.rejects(store.dispatch({op:'runtime_set',total_seconds:7200,qa_seconds:900,
    expected_config:config,expected_source_commit:source}),/runtime_active_research_qa_repair_or_publication/);
  assert.deepEqual(db.values.get(ROOT),before);
});
