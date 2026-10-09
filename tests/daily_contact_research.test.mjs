// Offline proof of the private daily harness. No credentials/provider calls.
import test from 'node:test';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {spawnSync} from 'node:child_process';
import {readFileSync} from 'node:fs';
import {contactResearchContext,claimContactResearch,finishContactResearch,finishContactResearchSafely,verifyContactResearchDiscovery,publicContactResearchTask} from '../tools/daily_research/contact_research.mjs';
const ROOT='blueprintDailyResearch/sites-first',CONTACT='blueprintCommunications/default/contactResearchRequests';
const sha=x=>createHash('sha256').update(x).digest('hex');
const canonical=x=>Array.isArray(x)?x.map(canonical):x&&typeof x==='object'?Object.fromEntries(Object.keys(x).sort().map(k=>[k,canonical(x[k])])):x;
const json=x=>JSON.stringify(canonical(x)),digest=x=>sha(json(x));
const pyJson=x=>json(x).replace(/[\u007f-\uffff]/g,c=>'\\u'+c.charCodeAt(0).toString(16).padStart(4,'0'));
const pyDigest=x=>sha(pyJson(x));
const NOW=Date.parse('2026-10-04T12:01:00Z'),DAY='2026-10-04';
const READ=/^READ = "([^"]+)"/m.exec(readFileSync(new URL('../tools/daily_research/search.py',import.meta.url),'utf8'))[1];
function fakeStore() {
  const records=new Map(),blobs=new Map(),files=new Map();
  const snapshot=ref=>({ref,exists:records.has(ref.path),data:()=>structuredClone(records.get(ref.path))});
  const doc=path=>({path,get:async()=>snapshot(doc(path))});
  const collection=path=>({path,where(field,_op,value){return {...this,field,value};},limit(n){return {...this,n};},get(){return tx.get(this);}});
  const tx={get:async ref=>ref.value===undefined?snapshot(ref):{docs:[...records.entries()]
    .filter(([key,row])=>key.startsWith(ref.path+'/')&&row[ref.field]===ref.value).slice(0,ref.n).map(([key])=>snapshot(doc(key)))},
    set:(ref,value,options)=>records.set(ref.path,options?.merge?{...records.get(ref.path),...structuredClone(value)}:structuredClone(value)),
    create:(ref,value)=>{assert(!records.has(ref.path));records.set(ref.path,structuredClone(value));}};
  const store={db:{doc,collection},control:doc(ROOT),clock:()=>NOW,transaction:async fn=>fn(tx),
    fence:()=>{},workflowGate:()=>{},budgetGate:()=>{},
    fileGet:async name=>{assert(files.has(name),'saved file missing');return files.get(name).toString('base64');},
    blobGet:async hash=>{assert(blobs.has(hash),'saved blob missing');assert.equal(sha(blobs.get(hash)),hash);return blobs.get(hash).toString('base64');}};
  return {store,records,blobs,files,tx};
}
function fixture({unicode=false}={}) {
  const f=fakeStore(),h='a'.repeat(64);
  const publication={date:'2026-10-03',runKey:'reviewed-report:original',candidateKey:'site-original',packetDigest:h,
    rawArtifactDigest:h,sourceDigest:h,qaArtifactDigest:h,researchQaReference:'authenticated:fixture',sheetsId:'fixture-sheet',sheetsProspectId:'BP-000015',prospectId:'fixture-prospect'};
  const task={version:'blueprint.contact-research-request.v1',requestId:digest({publication,sourceDigest:h}),sourceDigest:h,publication,
    organization:unicode?'Société 🚚':'Synthetic Operator',organizationUrl:'https://facility.example',site:unicode?'Montréal':'Packing center',location:'Synthetic address',task:'packing',
    sourceUrls:['https://facility.example/facility'],preference:['relevant_professional_person','appropriate_team_inbox','general_business_inbox'],
    maxAgentAttempts:2,scope:'existing_daily_research_budget_and_tools_no_new_session_or_send'};
  const context={version:'blueprint.contact-research-input.v1',date:DAY,tasks:[task],separateSessionsAuthorized:false,sendsAuthorized:false,budget:'existing_daily_research_allocation'};
  const inputDigest=pyDigest(context),metadata={run_key:`blueprint-researcher:${DAY}`,contact_research_digest:inputDigest};
  const artifact=Buffer.from('{"synthetic":true}'),qaResult={schema_version:'blueprint.research-qa.v1',packet_digest:h,crm_digest:h,source_support_verified:true,summary:'Synthetic QA'};
  const qaRaw=Buffer.from(json(qaResult));
  const review={summary:qaResult.summary,qa_artifact_digest:sha(qaRaw),reviewer_reference:'agent-turn:session-original:turn-qa'};
  const request={call_id:'call_original',turn_id:'turn-original',name:READ,arguments:{url:'https://facility.example/operations'}};
  const output={url:request.arguments.url,requested_url:request.arguments.url,checked_at:'2026-10-04T12:00:20Z',raw_sha256:h,
    text:'Synthetic Operator operations contact.',truncated:false,evidence_scope:'complete_static_extracted_text_not_javascript_rendered'};
  const event={type:'agent.session.input.tool_result',call_id:request.call_id,turn_id:request.turn_id,success:true,output:pyJson(output)};
  const raw=Buffer.from(pyJson(event)+'\n'),filename=`${DAY}-tool-${request.call_id}.json`;
  const row={date:DAY,run_key:metadata.run_key,metadata,state:'completed',started_at:'2026-10-04T12:00:00Z',research_runtime_seconds:180,
    remote_completed_at:NOW/1000,session_id:'session-original',turn_id:request.turn_id,turn_status:'completed',packet_digest:h,
    raw_output_digest:sha(artifact),create_payload:{metadata,environment:{files:[{type:'inline',path:'/workspace/inputs/blueprint-contact-research.json',data:Buffer.from(pyJson(context)).toString('base64')}]}},
    qa:{state:'validated',turn_status:'completed',turn_id:'turn-qa',artifact_digest:sha(qaRaw),crm_digest:h,decision:review},review,
    application_tool_calls:{[request.call_id]:{request,request_digest:pyDigest(request),phase:'research',success:true,result_acknowledged:true,
      result_file:filename,result_sha256:sha(raw),result_bytes:raw.length,result_digest:pyDigest(event)}}};
  f.files.set(`${DAY}-artifact.json`,artifact);f.files.set(`${DAY}-qa.json`,qaRaw);f.files.set(filename,raw);
  f.records.set(ROOT,{config:{search_provider:'perplexity-fast-v1'}});
  f.records.set(`${ROOT}/contactResearchRuns/${DAY}`,{context,inputDigest});
  f.records.set(`${CONTACT}/${task.requestId}`,{task,state:'running',attempts:1,nativeRunKey:row.run_key,inputDigest});
  const saveRow=()=>{const bytes=Buffer.from(JSON.stringify(row)),blob=sha(bytes);f.blobs.set(blob,bytes);f.records.set(`${ROOT}/runs/${DAY}`,{state:row.state,metadata,blob});return blob;};
  return {...f,task,context,metadata,row,request,event,output,filename,saveRow};
}
async function completed(f) {
  const blob=f.saveRow();await finishContactResearch(f.store,f.row,blob);
  return f.records.get(`${CONTACT}/${f.task.requestId}`).discovery;
}
test('isolated canary runs never access or claim the production contact queue',async()=>{
  const f=fixture();
  f.store.control={path:`${ROOT}/canaries/isolated-proof`};
  f.store.db={doc(){throw Error('unexpected production doc');},collection(){throw Error('unexpected production queue');}};
  f.store.transaction=async()=>{throw Error('unexpected production transaction');};
  assert.equal(await contactResearchContext(f.store,DAY),null);
  await claimContactResearch(f.store,f.tx,DAY,{});
  await assert.rejects(()=>claimContactResearch(f.store,f.tx,DAY,f.metadata),/contact_research_scope_changed/);
  assert.equal((await finishContactResearchSafely(f.store,f.row,'a'.repeat(64))).state,'observed');
  await assert.rejects(()=>verifyContactResearchDiscovery(f.store,f.task,{}),/contact_research_scope_changed/);
  assert.equal(f.records.get(`${CONTACT}/${f.task.requestId}`).state,'running');
  assert.equal(f.records.get(`${CONTACT}/${f.task.requestId}`).attempts,1);
});
test('Unicode context hashes match the actual Python canonical encoder; claim consumes once',async()=>{
  const f=fixture({unicode:true});
  const oracle=spawnSync('python3',['-c','import sys,json,hashlib;print(hashlib.sha256(json.dumps(json.load(sys.stdin),sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest())'],{input:JSON.stringify(f.context),encoding:'utf8'});
  assert.equal(oracle.status,0);assert.equal(oracle.stdout.trim(),f.metadata.contact_research_digest);
  f.records.set(`${CONTACT}/${f.task.requestId}`,{task:f.task,state:'pending',attempts:0});
  await claimContactResearch(f.store,f.tx,DAY,f.metadata);
  assert.equal(f.records.get(`${CONTACT}/${f.task.requestId}`).attempts,1);
  await assert.rejects(()=>claimContactResearch(f.store,f.tx,DAY,f.metadata),/contact_research_request_changed/);
});
test('allowlist strips private queue fields and creates no separate-session authority',async()=>{
  const f=fixture();assert.equal(publicContactResearchTask({...f.task,secret:'private'}).secret,undefined);
  f.records.delete(`${ROOT}/contactResearchRuns/${DAY}`);
  f.records.set(`${CONTACT}/${f.task.requestId}`,{task:{...f.task,secret:'private'},state:'pending',attempts:0});
  const context=await contactResearchContext(f.store,DAY);
  assert.equal(context.separateSessionsAuthorized,false);assert.equal(context.tasks[0].secret,undefined);
  assert.deepEqual(await contactResearchContext(f.store,DAY),context);
});
test('restart finishes exact saved terminal bytes without a provider create, and settled result is idempotent',async()=>{
  const f=fixture(),blob=f.saveRow();
  assert.equal(f.records.get(`${CONTACT}/${f.task.requestId}`).state,'running');
  await finishContactResearch(f.store,f.row,blob);
  const first=structuredClone(f.records.get(`${CONTACT}/${f.task.requestId}`));
  await finishContactResearch(f.store,f.row,blob);
  assert.deepEqual(f.records.get(`${CONTACT}/${f.task.requestId}`),first);
  assert.equal(first.state,'sources_ready');assert.equal(first.discovery.sources[0].callId,'call_original');
  assert.deepEqual(await verifyContactResearchDiscovery(f.store,f.task,first.discovery),first.discovery);
});
test('legitimate cleanup can advance the current blob; altered original research cannot',async()=>{
  const f=fixture(),proof=await completed(f);
  f.row.cleanup_required=false;f.row.cleanup_receipt={session_id:'session-original'};f.row.billing_stop_verified=false;f.saveRow();
  assert.deepEqual(await verifyContactResearchDiscovery(f.store,f.task,proof),proof);
  f.row.session_id='other-session';f.saveRow();
  await assert.rejects(()=>verifyContactResearchDiscovery(f.store,f.task,proof),/contact_research_run_changed/);
});
test('second unsuccessful daily attempt remains pending without inventing a contact or result',async()=>{
  const f=fixture();f.row.application_tool_calls={};f.records.get(`${CONTACT}/${f.task.requestId}`).attempts=2;
  await finishContactResearch(f.store,f.row,f.saveRow());
  const record=f.records.get(`${CONTACT}/${f.task.requestId}`);
  assert.equal(record.state,'pending');assert.equal(record.discovery,undefined);assert.equal(record.sent,false);
});
for(const field of ['phase','turn','call','acknowledged','requestDigest','resultDigest','resultBytes','resultFile','bytes','timestamp','future','scope','truncated','rawArtifact','qaArtifact','frozenInput']) {
  test(`rejects ${field} corruption of the original native proof`,async()=>{
    const f=fixture(),proof=await completed(f),call=f.row.application_tool_calls.call_original;
    if(field==='phase') call.phase='qa';if(field==='turn') call.request.turn_id='other';if(field==='call') call.request.call_id='other';
    if(field==='acknowledged') call.result_acknowledged=false;if(field==='requestDigest') call.request_digest='b'.repeat(64);
    if(field==='resultDigest') call.result_digest='b'.repeat(64);if(field==='resultBytes') call.result_bytes+=1;
    if(field==='resultFile') call.result_file='2026-10-03-tool-call_original.json';
    if(field==='bytes') f.files.set(f.filename,Buffer.from('{}'));
    if(['timestamp','future','scope','truncated'].includes(field)) {
      if(field==='timestamp') f.output.checked_at='2026-10-04T11:00:00Z';if(field==='future') f.output.checked_at='2026-10-04T13:00:00Z';
      if(field==='scope') f.output.evidence_scope='excerpt';if(field==='truncated') f.output.truncated=true;
      f.event.output=pyJson(f.output);const raw=Buffer.from(pyJson(f.event)+'\n');f.files.set(f.filename,raw);
      call.result_sha256=sha(raw);call.result_bytes=raw.length;call.result_digest=pyDigest(f.event);
    }
    if(field==='rawArtifact') f.files.set(`${DAY}-artifact.json`,Buffer.from('{}'));
    if(field==='qaArtifact') f.files.set(`${DAY}-qa.json`,Buffer.from('{}'));
    if(field==='frozenInput') f.row.create_payload.environment.files[0].data=Buffer.from('{}').toString('base64');
    // Rebind only the current blob to force checking full frozen evidence.
    proof.run.rowBlob=f.saveRow();
    await assert.rejects(()=>verifyContactResearchDiscovery(f.store,f.task,proof));
  });
}
for(const wrapper of ['bom','fence','both']) {
  test(`accepts the native QA's already-reviewed ${wrapper} wrapper with exact normalization evidence`,async()=>{
    const f=fixture(),normal=f.files.get(`${DAY}-qa.json`),text=normal.toString('utf8');
    const raw=Buffer.from((wrapper==='bom'||wrapper==='both'?'\ufeff':'')+(wrapper==='fence'||wrapper==='both'?'```json\n'+text+'\n```':text));
    const transformations=[...(wrapper==='bom'||wrapper==='both'?['utf8_bom']:[]),...(wrapper==='fence'||wrapper==='both'?['single_json_fence']:[])];
    f.files.set(`${DAY}-qa.json`,raw);f.row.qa.artifact_digest=sha(raw);f.row.review.qa_artifact_digest=sha(raw);
    f.row.qa.artifact_format_normalization={schema_version:'blueprint.artifact-format-normalization.v1',raw_sha256:sha(raw),raw_bytes:raw.length,
      normalized_sha256:sha(normal),normalized_bytes:normal.length,transformations};
    const proof=await completed(f);assert.equal(proof.run.qaArtifactDigest,sha(raw));
    await verifyContactResearchDiscovery(f.store,f.task,proof);
  });
}
test('refuses QA prose and multiple documents without guessing or another provider request',async()=>{
  for(const text of ['Here is the review: {}','{}\n{}','```json\n{}\n```\n```json\n{}\n```']) {
    const f=fixture(),raw=Buffer.from(text);f.files.set(`${DAY}-qa.json`,raw);
    f.row.qa.artifact_digest=sha(raw);f.row.review.qa_artifact_digest=sha(raw);
    await assert.rejects(()=>finishContactResearch(f.store,f.row,f.saveRow()));
  }
});
test('auxiliary integrity failure quarantines one running claim without undoing source commit or resetting attempts',async()=>{
  const f=fixture();f.files.set(`${DAY}-artifact.json`,Buffer.from('{}'));const blob=f.saveRow();
  const result=await finishContactResearchSafely(f.store,f.row,blob);
  assert.equal(result.state,'blocked');assert.equal(f.records.get(`${ROOT}/runs/${DAY}`).state,'completed');
  assert.equal(f.records.get(`${CONTACT}/${f.task.requestId}`).attempts,1);
  assert.equal(f.records.get(`${CONTACT}/${f.task.requestId}`).state,'blocked');
});
test('settled old claims skip corrupt old receipts so current native observation remains available',async()=>{
  const f=fixture(),proof=await completed(f);f.files.set(`${DAY}-artifact.json`,Buffer.from('{}'));
  assert.equal((await finishContactResearchSafely(f.store,f.row,proof.run.rowBlob)).state,'observed');
  assert.equal(f.records.get(`${CONTACT}/${f.task.requestId}`).state,'sources_ready');
});
