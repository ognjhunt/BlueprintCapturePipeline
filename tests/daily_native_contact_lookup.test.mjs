// Real private host operations against synthetic retained bytes; no credentials/network.
import test from 'node:test';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {nativeContactOperation,prepareNativeContacts} from '../tools/daily_research/native_contact_lookup.mjs';
const ROOT='blueprintDailyResearch/sites-first',DAY='2026-10-12',NOW=Date.parse(DAY+'T12:01:00Z');
const canonical=x=>Array.isArray(x)?x.map(canonical):x&&typeof x==='object'?Object.fromEntries(Object.keys(x).sort().map(k=>[k,canonical(x[k])])):x;
const json=x=>JSON.stringify(canonical(x)).replace(/[^\x00-\x7f]/g,c=>'\\u'+c.charCodeAt(0).toString(16).padStart(4,'0'));
const sha=x=>createHash('sha256').update(x).digest('hex'),hash=x=>sha(json(x));
function fixture() {
 const records=new Map(),files=new Map(),h='a'.repeat(64),snap=ref=>({ref,exists:records.has(ref.path),data:()=>structuredClone(records.get(ref.path))});
 const doc=path=>({path,get:async()=>snap(doc(path))}),collection=path=>({path,limit(n){return {...this,n};}});
 const tx={get:async ref=>ref.n?{docs:[...records.keys()].filter(k=>k.startsWith(ref.path+'/')).map(k=>snap(doc(k))).slice(0,ref.n)}:snap(ref),
   set:(ref,v,o)=>records.set(ref.path,o?.merge?{...records.get(ref.path),...structuredClone(v)}:structuredClone(v)),
   create:(ref,v)=>{assert(!records.has(ref.path));records.set(ref.path,structuredClone(v));}};
 const candidate={candidate_key:'fixture-native',organization:'Synthetic Operator',organization_url:'https://facility.example',site:'Packing center',location:'Fixture City, TX',task:'packing'};
 const assessment={candidate_digest:hash(candidate),assessed_at:DAY+'T12:00:30Z',valid_until:'2026-10-17T12:00:30Z',
  claims:{operator:{status:'verified_fact'},physical_site:{status:'verified_fact'},site_task:{status:'inference'},plausible_fit:{status:'unresolved'}},
  counterevidence:{status:'unresolved'}};
 const artifact=Buffer.from('{"synthetic":true}'),qaValue={schema_version:'blueprint.research-qa.v1',packet_digest:h,crm_digest:h,source_support_verified:true,summary:'Synthetic QA'};
 qaValue.checks=[{candidate_key:candidate.candidate_key,duplicate:false,lead_verification:assessment}];
 const qaRaw=Buffer.from(json(qaValue)),review={summary:qaValue.summary,qa_artifact_digest:sha(qaRaw),reviewer_reference:'agent-turn:session-fixture:qa-fixture',
  lead_verification:{results:[{candidate_key:candidate.candidate_key,assessment}]}};
 const request={call_id:'call_fixture',turn_id:'research-fixture',name:'blueprint_read_source',arguments:{url:'https://facility.example/facility'}};
 const output={url:request.arguments.url,requested_url:request.arguments.url,checked_at:DAY+'T12:00:20Z',raw_sha256:h,text:'Synthetic Operator packing center.',truncated:false,evidence_scope:'complete_static_extracted_text_not_javascript_rendered'};
 const event={type:'agent.session.input.tool_result',call_id:request.call_id,turn_id:request.turn_id,success:true,output:json(output)},raw=Buffer.from(json(event)+'\n');
 const row={date:DAY,run_key:'blueprint-researcher:'+DAY,state:'completed',started_at:DAY+'T12:00:00Z',research_runtime_seconds:180,remote_completed_at:NOW/1000,
  session_id:'session-fixture',turn_id:request.turn_id,turn_status:'completed',packet_digest:h,raw_output_digest:sha(artifact),packet:{candidates:[candidate]},review,
  qa:{state:'validated',turn_status:'completed',turn_id:'qa-fixture',artifact_digest:sha(qaRaw),crm_digest:h,decision:review},
  application_tool_calls:{[request.call_id]:{request,request_digest:hash(request),phase:'research',success:true,result_acknowledged:true,
    result_file:DAY+'-tool-'+request.call_id+'.json',result_sha256:sha(raw),result_bytes:raw.length,result_digest:hash(event)}}};
 files.set(DAY+'-artifact.json',artifact);files.set(DAY+'-qa.json',qaRaw);files.set(DAY+'-tool-'+request.call_id+'.json',raw);
 records.set(ROOT,{enabled:true,config:{enabled:true},workflow:{enabled:true},lease:{generation:1},});
 records.set(ROOT+'/runs/'+DAY,{state:'completed',blob:h});
 let leased=true;
 const store={db:{doc,collection},control:doc(ROOT),clock:()=>NOW,transaction:async fn=>fn(tx),
  assertLease:async()=>{if(!leased)throw Error('firestore_lease_lost');},fence:()=>{if(!leased)throw Error('firestore_lease_lost');},workflowGate:()=>{},
  publicationFence:async()=>{if(!leased)throw Error('firestore_lease_lost');},get:async day=>day===DAY?structuredClone(row):null,
  fileGet:async name=>{assert(files.has(name));return files.get(name).toString('base64');}};
 return {store,row,records,files,lease:v=>{leased=v;}};
}
async function ready() {
 const f=fixture(),[input]=await prepareNativeContacts(f.store,DAY);
 const scope={schema_version:'blueprint.native-contact-lookup-control.v1',enabled:true,approval_reference:'synthetic-founder-only',provider:'fullenrich',resource_class:'fullenrich_work_email',
  input_ids:[input.input_id],operations:['person_search','work_email'],max_credit_quarters:9,max_calls:2,effective_from:DAY+'T12:00:00Z',expires_at:DAY+'T13:00:00Z'};
 scope.scope_id=hash(scope);
 await nativeContactOperation(f.store,{op:'native_contact_scope_set',expected_scope_id:null,value:scope});
 const context={input_id:input.input_id,input_digest:hash(input)};
 const body={current_company_domains:[{value:'facility.example',exact_match:true}],current_position_titles:[{value:'operations director',exact_match:false}],person_locations:[{value:'United States',exact_match:true}],limit:5,offset:0};
 const request={kind:'search',site_key:'c'.repeat(64),identity:['facility.example',['operations director'],'United States'],body};
 const key=hash(['fullenrich','search',request.identity]),event={schema_version:'blueprint.contact-lookup.journal.v1',event:'intent',call:key,attempt:'d'.repeat(16),kind:'search',site_key:request.site_key,most_credits:'1.25'};
 return {...f,input,scope,context,body,request,key,event};
}
test('unpaid native input preparation retains source bytes, original QA, unknown fit and no fake Parallel run',async()=>{
 const f=fixture(),[input]=await prepareNativeContacts(f.store,DAY);
 assert.equal(input.sources[0].text_sha256,sha(input.sources[0].text));assert.equal(input.assessment.claims.plausible_fit.status,'unresolved');
 assert.equal(input.provenance.row_blob,'a'.repeat(64));assert.equal(JSON.stringify(input).includes('parallel'),false);
 const saved=f.records.get(ROOT+'/nativeContactInputs/'+input.input_id);assert.equal(saved.inputDigest,hash(input));
 assert.equal(await nativeContactOperation(f.store,{op:'native_contact_scope',context:{input_id:input.input_id,input_digest:hash(input)}}),null);
});
for(const duplicateEvidence of ['cohort','qa']) test(`native preparation skips only an unresolved ${duplicateEvidence} duplicate and retains its valid neighbor`,async()=>{
 const f=fixture(),original=structuredClone(f.row.packet.candidates[0]);
 const duplicate={...original,candidate_key:'fixture-duplicate'},assessment=structuredClone(f.row.review.lead_verification.results[0].assessment);
 assessment.candidate_digest=hash(duplicate);
 f.row.packet.candidates.unshift(duplicate);
 f.row.review.lead_verification.results.unshift({candidate_key:duplicate.candidate_key,assessment,
   ...(duplicateEvidence==='cohort'?{duplicate_check:{duplicate:true}}:{})});
 const qa=JSON.parse(f.files.get(DAY+'-qa.json'));
 qa.checks.unshift({candidate_key:duplicate.candidate_key,duplicate:duplicateEvidence==='qa',lead_verification:assessment});
 const raw=Buffer.from(json(qa));f.files.set(DAY+'-qa.json',raw);
 f.row.qa.artifact_digest=f.row.review.qa_artifact_digest=sha(raw);
 const before=structuredClone(f.row),values=await prepareNativeContacts(f.store,DAY);
 assert.deepEqual(values.map(v=>v.candidate.candidate_key),[original.candidate_key]);
 assert.deepEqual(f.row,before);
 assert.equal([...f.records.keys()].filter(k=>k.startsWith(ROOT+'/nativeContactInputs/')).length,1);
 assert.equal([...f.records.keys()].some(k=>k.startsWith(ROOT+'/nativeContactCalls/')),false);
});
for(const flag of ['qa','cohort','both']) test(`a ${flag} duplicate flag never hides changed candidate assessment evidence or admits a neighbor`,async()=>{
 const f=fixture(),qa=JSON.parse(f.files.get(DAY+'-qa.json'));
 const neighbor={...f.row.packet.candidates[0],candidate_key:'fixture-valid-neighbor'},assessment=structuredClone(f.row.review.lead_verification.results[0].assessment);
 assessment.candidate_digest=hash(neighbor);f.row.packet.candidates.push(neighbor);
 f.row.review.lead_verification.results.push({candidate_key:neighbor.candidate_key,assessment});
 qa.checks.push({candidate_key:neighbor.candidate_key,duplicate:false,lead_verification:assessment});
 if(flag!=='qa') f.row.review.lead_verification.results[0].duplicate_check={duplicate:true};
 qa.checks[0].duplicate=flag!=='cohort';qa.checks[0].lead_verification.claims.operator.status='unresolved';
 const raw=Buffer.from(json(qa));f.files.set(DAY+'-qa.json',raw);
 f.row.qa.artifact_digest=f.row.review.qa_artifact_digest=sha(raw);
 await assert.rejects(()=>prepareNativeContacts(f.store,DAY),/native_contact_assessment_changed/);
 assert.equal([...f.records.keys()].some(k=>k.startsWith(ROOT+'/nativeContactInputs/')),false);
});
test('a durable intent is claimed once under the current lease and exact FullEnrich scope',async()=>{
 const f=await ready();await nativeContactOperation(f.store,{op:'native_contact_call_append',context:f.context,event:f.event,request:f.request});
 const dispatch={op:'native_contact_dispatch_admission',context:f.context,path:'/api/v2/people/search',body:f.body,body_digest:hash(f.body)};
 const admitted=await nativeContactOperation(f.store,dispatch);assert.equal(admitted.status,'admitted');assert.equal(admitted.resource_class,'fullenrich_work_email');
 assert.match(admitted.allocation_binding_digest,/^[a-f0-9]{64}$/);
 await assert.rejects(()=>nativeContactOperation(f.store,dispatch),/native_contact_dispatch_not_admitted/);
 const row=await nativeContactOperation(f.store,{op:'native_contact_call_get',context:f.context,key:f.key});
 assert.equal(row.events[0].event,'intent');assert(row.dispatchClaim);
 // A subsequent native date cannot clear the global company/person call identity.
 await assert.rejects(()=>nativeContactOperation(f.store,{op:'native_contact_call_append',context:f.context,event:{...f.event,attempt:'e'.repeat(16)},request:f.request}),/already_claimed/);
});
test('a changed work request, lost lease, active research, expired scope and missing grant refuse before dispatch',async()=>{
 for(const problem of ['body','lease','active','expired','disabled']) {
  const f=await ready();await nativeContactOperation(f.store,{op:'native_contact_call_append',context:f.context,event:f.event,request:f.request});
  if(problem==='lease')f.lease(false);
  if(problem==='active')f.records.set(ROOT+'/runs/2026-10-13',{state:'creation_unresolved'});
  if(problem==='expired')f.store.clock=()=>NOW+3600000;
  if(problem==='disabled')f.records.get(ROOT).native_contact_lookup={enabled:false};
  const body=problem==='body'?{...f.body,limit:100}:f.body;
  await assert.rejects(()=>nativeContactOperation(f.store,{op:'native_contact_dispatch_admission',context:f.context,path:'/api/v2/people/search',body,body_digest:hash(body)}),/native_contact_|firestore_lease/);
  assert.equal(f.records.get(ROOT+'/nativeContactCalls/'+f.key).dispatchClaim,null);
 }
});
test('scope identity cannot be reused to raise credit/call limits and exact sources cannot be replaced',async()=>{
 const f=await ready();const scope={...f.scope,max_calls:100};
 await assert.rejects(()=>nativeContactOperation(f.store,{op:'native_contact_scope_set',expected_scope_id:f.scope.scope_id,value:scope}),/scope_not_admitted/);
 const filename=DAY+'-tool-call_fixture.json';f.files.set(filename,Buffer.from('{"tampered":true}'));
 await assert.rejects(()=>nativeContactOperation(f.store,{op:'native_contact_scope',context:f.context}),/contact_research_receipt_changed/);
});
test('isolated canary controls cannot read or mutate native production records',async()=>{
 const f=fixture();f.store.control={path:ROOT+'/canaries/fixture-only'};
 f.store.db={doc(){throw Error('unexpected production read');}};
 await assert.rejects(()=>nativeContactOperation(f.store,{op:'native_contact_call_get',context:{}}),/native_contact_scope_changed/);
 await assert.rejects(()=>nativeContactOperation(f.store,{op:'native_contact_scope_set',expected_scope_id:null,value:{enabled:false}}),/native_contact_scope_changed/);
});
for(const redirected of [false,true]) test('native selection retains the task final URL ahead of contact pages; redirect='+redirected,async()=>{
 const f=fixture(),original=f.row.application_tool_calls.call_fixture,filename=DAY+'-tool-call_fixture.json';
 const assessment=f.row.review.lead_verification.results[0].assessment;
 assessment.sources=[{id:'S1',url:'https://facility.example/mechanical-assembly'}];
 assessment.claims.site_task.source_refs=['S1'];
 const make=(id,url,finalUrl=url)=>{
  const request={...original.request,call_id:id,arguments:{url}},
   output={...JSON.parse(JSON.parse(f.files.get(filename)).output),url:finalUrl,requested_url:url},
   event={type:'agent.session.input.tool_result',call_id:id,turn_id:request.turn_id,success:true,output:json(output)},raw=Buffer.from(json(event)+'\n');
  const name=DAY+'-tool-'+id+'.json';f.files.set(name,raw);
  return {...original,request,request_digest:hash(request),result_file:name,result_sha256:sha(raw),result_bytes:raw.length,result_digest:hash(event)};
 };
 f.row.application_tool_calls={};
 for(let n=0;n<6;n++)f.row.application_tool_calls['contact_'+n]=make('contact_'+n,'https://facility.example/contact-'+n);
 f.row.application_tool_calls.task_source=make('task_source',redirected?'https://facility.example/legacy-service':'https://facility.example/mechanical-assembly','https://facility.example/mechanical-assembly');
 const qa=JSON.parse(f.files.get(DAY+'-qa.json'));qa.checks[0].lead_verification=assessment;
 const raw=Buffer.from(json(qa));f.files.set(DAY+'-qa.json',raw);
 f.row.qa.artifact_digest=sha(raw);f.row.review.qa_artifact_digest=sha(raw);
 const [input]=await prepareNativeContacts(f.store,DAY);
 assert.equal(input.sources[0].call_id,'task_source');assert.equal(input.sources.length,6);
});
test('null, partial and non-boolean scopes are rejected without any write; disabling replaces the entire map',async()=>{
 for(const value of [null,{},[],{enabled:null},{enabled:'true'},{enabled:1},{enabled:true},{enabled:false,scope_id:'a'.repeat(64)}]) {
  const f=await ready(),before=structuredClone([...f.records]);
  await assert.rejects(()=>nativeContactOperation(f.store,{op:'native_contact_scope_set',expected_scope_id:f.scope.scope_id,value}),/native_contact_scope_/);
  assert.deepEqual([...f.records],before);
 }
 const f=await ready();await nativeContactOperation(f.store,{op:'native_contact_scope_set',expected_scope_id:f.scope.scope_id,value:{enabled:false}});
 assert.deepEqual(f.records.get(ROOT).native_contact_lookup,{enabled:false});
 assert.equal(await nativeContactOperation(f.store,{op:'native_contact_scope',context:f.context}),null);
});
test('the actual private bridge preserves stable admission errors rather than replacing them with an availability error',async()=>{
 const {Store}=await import('../tools/daily_research/firestore_bridge.mjs');
 const f=fixture();
 await assert.rejects(()=>Store.prototype.dispatch.call(f.store,{op:'native_contact_scope_set',expected_scope_id:null,value:null}),error=>
  error.constructor.name==='Refusal'&&error.message==='native_contact_scope_invalid');
});
