// Native contact recovery uses the existing daily root, tools and budget only.
import {createHash} from 'node:crypto';
const ROOT='blueprintDailyResearch/sites-first', CONTACT='blueprintCommunications/default/contactResearchRequests';
const sha=x=>createHash('sha256').update(x).digest('hex');
const canonical=x=>Array.isArray(x)?x.map(canonical):x&&typeof x==='object'
  ?Object.fromEntries(Object.keys(x).sort().map(k=>[k,canonical(x[k])])):x;
const digest=x=>sha(JSON.stringify(canonical(x)));
// Match runner.canonical exactly for frozen inputs and retained tool receipts.
const pythonDigest=x=>sha(JSON.stringify(canonical(x)).replace(/[\u007f-\uffff]/g,
  c=>'\\u'+c.charCodeAt(0).toString(16).padStart(4,'0')));
const hashOK=x=>typeof x==='string'&&/^[a-f0-9]{64}$/.test(x);
const dateOK=x=>typeof x==='string'&&/^\d{4}-\d{2}-\d{2}$/.test(x);
const fail=code=>{throw new Error(code);};
const safeUrl=value=>{
  try {const u=new URL(value);return u.protocol==='https:'&&!u.username&&!u.password&&(!u.port||u.port==='443')
    &&/^[a-z0-9.-]+\.[a-z]{2,63}$/i.test(u.hostname)&&value.length<=1000
    &&!/(?:token|api.?key|authorization|signature|credential)/i.test(u.search);}catch{return false;}
};
const operatorUrl=(value,organizationUrl)=>{
  if(!safeUrl(value)||!safeUrl(organizationUrl)) return false;
  const host=x=>new URL(x).hostname.toLowerCase().replace(/^www\./,'');
  return host(value)===host(organizationUrl)||host(value).endsWith('.'+host(organizationUrl));
};
/** Allowlisted public evidence only; arbitrary queue fields never reach a model. */
export function publicContactResearchTask(t) {
  const p=t?.publication;
  if(t?.version!=='blueprint.contact-research-request.v1'||!hashOK(t.requestId)||!hashOK(t.sourceDigest)
    ||!p||!dateOK(p.date)||!['packetDigest','rawArtifactDigest','sourceDigest','qaArtifactDigest'].every(k=>hashOK(p[k]))
    ||p.sourceDigest!==t.sourceDigest||!['runKey','candidateKey','researchQaReference','sheetsId','sheetsProspectId','prospectId'].every(k=>typeof p[k]==='string'&&p[k].length>0&&p[k].length<=1200)
    ||!['organization','site','location','task'].every(k=>typeof t[k]==='string'&&t[k].trim()&&t[k].length<=1200)
    ||!safeUrl(t.organizationUrl)||!Array.isArray(t.sourceUrls)||t.sourceUrls.length>16
    ||(t.maxAgentAttempts!==null && (!Number.isInteger(t.maxAgentAttempts)||t.maxAgentAttempts<1))||t.scope!=='existing_daily_research_budget_and_tools_no_new_session_or_send') return null;
  const task={version:t.version,requestId:t.requestId,sourceDigest:t.sourceDigest,
    publication:Object.fromEntries(['date','runKey','candidateKey','packetDigest','rawArtifactDigest','sourceDigest','qaArtifactDigest','researchQaReference','sheetsId','sheetsProspectId','prospectId'].map(k=>[k,p[k]])),
    organization:t.organization,organizationUrl:t.organizationUrl,site:t.site,location:t.location,task:t.task,
    sourceUrls:t.sourceUrls.filter(safeUrl),preference:['relevant_professional_person','appropriate_team_inbox','general_business_inbox'],
    maxAgentAttempts:t.maxAgentAttempts,scope:t.scope};
  return digest({publication:task.publication,sourceDigest:task.sourceDigest})===task.requestId?task:null;
}

export async function contactResearchContext(store,day) {
  // Isolated canary adapters must never read the production communications queue.
  if(store.control?.path!==ROOT) return null;
  if(!dateOK(day)) fail('contact_research_date_invalid');
  const ref=store.db.doc(`${ROOT}/contactResearchRuns/${day}`);
  return store.transaction(async tx=>{
    const control=(await tx.get(store.control)).data();store.fence(control);
    if(control.config?.search_provider!=='perplexity-fast-v1') return null;
    store.workflowGate(control);
    store.budgetGate(control,{...control.config,search_provider:'perplexity-fast-v1'});
    const saved=await tx.get(ref);
    if(saved.exists) return saved.data().context;
    const rows=await tx.get(store.db.collection(CONTACT).where('state','==','pending').limit(4));
    const tasks=rows.docs.map(s=>({task:publicContactResearchTask(s.data()?.task),attempts:s.data()?.attempts}))
      .filter(r=>r.task&&Number.isInteger(r.attempts)&&r.attempts>=0).slice(0,3).map(r=>r.task);
    if(!tasks.length) return null;
    const context={version:'blueprint.contact-research-input.v1',date:day,tasks,separateSessionsAuthorized:false,
      sendsAuthorized:false,budget:'existing_daily_research_allocation'};
    tx.create(ref,{context,inputDigest:pythonDigest(context)});return context;
  });
}

/** Called inside the existing one-use daily create transaction, before POST. */
export async function claimContactResearch(store,tx,day,metadata) {
  if(!metadata.contact_research_digest) return;
  if(store.control?.path!==ROOT) fail('contact_research_scope_changed');
  const manifest=await tx.get(store.db.doc(`${ROOT}/contactResearchRuns/${day}`)),context=manifest.data()?.context;
  if(!context||manifest.data().inputDigest!==metadata.contact_research_digest
    ||pythonDigest(context)!==metadata.contact_research_digest||context.date!==day) fail('contact_research_input_changed');
  const requests=await Promise.all(context.tasks.map(t=>tx.get(store.db.doc(`${CONTACT}/${t.requestId}`))));
  if(requests.some((s,i)=>s.data()?.state!=='pending'||digest(publicContactResearchTask(s.data()?.task))!==digest(context.tasks[i])
    ||!Number.isInteger(s.data()?.attempts)||s.data().attempts<0)) fail('contact_research_request_changed');
  for(const s of requests) tx.set(s.ref,{state:'running',attempts:s.data().attempts+1,
    nativeRunKey:`blueprint-researcher:${day}`,inputDigest:metadata.contact_research_digest},{merge:true});
}

const stableRow = row => Object.fromEntries(Object.entries(row).filter(([key])=>
  !['cleanup','cleanup_required','cleanup_receipt','billing_stop_verified'].includes(key)));

function approvedQaJson(raw,normalization) {
  let text=new TextDecoder('utf-8',{fatal:true,ignoreBOM:true}).decode(raw),transformations=[];
  if(text.startsWith('\ufeff')) {text=text.slice(1);transformations.push('utf8_bom');}
  let value;
  try {value=JSON.parse(text);} catch {
    const fence=/^```(?:json)?[ \t]*\r?\n([\s\S]*)\r?\n```$/i.exec(text.trim());
    if(!fence) fail('contact_research_qa_format_changed');
    text=fence[1];value=JSON.parse(text);transformations.push('single_json_fence');
  }
  const normalized=Buffer.from(text,'utf8'),expected=transformations.length?{
    schema_version:'blueprint.artifact-format-normalization.v1',raw_sha256:sha(raw),raw_bytes:raw.length,
    normalized_sha256:sha(normalized),normalized_bytes:normalized.length,transformations}:null;
  if(digest(normalization||null)!==digest(expected)) fail('contact_research_qa_format_changed');
  return value;
}

async function boundContext(store,row) {
  const manifest=(await store.db.doc(`${ROOT}/contactResearchRuns/${row.date}`).get()).data(),context=manifest?.context;
  const expected=row.metadata?.contact_research_digest;
  if(!context||context.date!==row.date||context.version!=='blueprint.contact-research-input.v1'
    ||context.separateSessionsAuthorized!==false||context.sendsAuthorized!==false
    ||!Array.isArray(context.tasks)||context.tasks.length<1||context.tasks.length>3
    ||context.tasks.some(t=>!publicContactResearchTask(t)||digest(publicContactResearchTask(t))!==digest(t))
    ||manifest.inputDigest!==expected||pythonDigest(context)!==expected) fail('contact_research_input_changed');
  const files=row.create_payload?.environment?.files?.filter(f=>f.path==='/workspace/inputs/blueprint-contact-research.json');
  if(files?.length!==1||files[0].type!=='inline'||sha(Buffer.from(files[0].data,'base64'))!==expected
    ||digest(JSON.parse(Buffer.from(files[0].data,'base64').toString('utf8')))!==digest(context)
    ||row.create_payload?.metadata?.contact_research_digest!==expected) fail('contact_research_input_changed');
  return context;
}

async function nativeSources(store,row,task) {
  const qa=row.qa;
  if(row.state!=='completed'||row.turn_status!=='completed'||!row.turn_id||!row.session_id
    ||qa?.state!=='validated'||qa.turn_status!=='completed'||!qa.turn_id||!hashOK(row.raw_output_digest)
    ||!hashOK(qa.artifact_digest)||row.review?.qa_artifact_digest!==qa.artifact_digest
    ||row.review?.reviewer_reference!==`agent-turn:${row.session_id}:${qa.turn_id}`
    ||digest(row.review)!==digest(qa.decision)) fail('contact_research_run_unverified');
  const artifact=Buffer.from(await store.fileGet(`${row.date}-artifact.json`),'base64');
  const qaFile=qa.artifact_file||`${row.date}-qa.json`;
  if(!new RegExp(`^${row.date}-qa(?:-correction-[12]-artifact)?\\.json$`).test(qaFile)) fail('contact_research_run_unverified');
  const qaRaw=Buffer.from(await store.fileGet(qaFile),'base64'),qaResult=approvedQaJson(qaRaw,qa.artifact_format_normalization);
  if(sha(artifact)!==row.raw_output_digest||sha(qaRaw)!==qa.artifact_digest
    ||qaResult.schema_version!=='blueprint.research-qa.v1'||qaResult.packet_digest!==row.packet_digest
    ||qaResult.crm_digest!==qa.crm_digest||qaResult.source_support_verified!==true
    ||qaResult.summary!==row.review.summary) fail('contact_research_run_unverified');
  const start=Date.parse(row.started_at),deadline=start+row.research_runtime_seconds*1000;
  if(!Number.isFinite(start)||!Number.isSafeInteger(row.research_runtime_seconds)||row.research_runtime_seconds<=0
    ||!Number.isFinite(row.remote_completed_at)) fail('contact_research_run_unverified');
  const calls=Object.entries(row.application_tool_calls||{}).filter(([,c])=>c.success===true
    &&c.phase==='research'&&c.request?.name==='blueprint_read_source'
    &&operatorUrl(c.request.arguments?.url,task.organizationUrl)).sort((a,b)=>{
      const rank=c=>/contact|inquir|partnership|leadership|team|people|operations/i.test(c.request.arguments.url)?0:1;
      return rank(a[1])-rank(b[1]);
    }).slice(0,6),sources=[];
  for(const [callId,call] of calls) {
    if(!/^[A-Za-z0-9_-]{1,200}$/.test(callId)||call.request.call_id!==callId||call.request.turn_id!==row.turn_id
      ||call.result_acknowledged!==true||call.result_file!==`${row.date}-tool-${callId}.json`
      ||pythonDigest(call.request)!==call.request_digest||!hashOK(call.result_sha256)) fail('contact_research_receipt_changed');
    const raw=Buffer.from(await store.fileGet(call.result_file),'base64'),event=JSON.parse(raw);
    if(raw.length!==call.result_bytes||sha(raw)!==call.result_sha256||pythonDigest(event)!==call.result_digest
      ||event.type!=='agent.session.input.tool_result'||event.call_id!==callId||event.turn_id!==row.turn_id
      ||event.success!==true||typeof event.output!=='string') fail('contact_research_receipt_changed');
    const output=JSON.parse(event.output),checked=Date.parse(output.checked_at);
    if(output.truncated!==false||output.evidence_scope!=='complete_static_extracted_text_not_javascript_rendered'
      ||!hashOK(output.raw_sha256)||output.requested_url!==call.request.arguments.url
      ||!operatorUrl(output.url,task.organizationUrl)||!Number.isFinite(checked)||checked<start
      ||checked>Math.min(deadline,row.remote_completed_at*1000,store.clock())) fail('contact_research_receipt_changed');
    if(!sources.some(s=>s.url===output.url)) sources.push({url:output.url,callId,resultSha256:call.result_sha256,
      rawSourceSha256:output.raw_sha256,checkedAt:output.checked_at});
    if(sources.length===3) break;
  }
  return sources;
}

/** Read-only verification of saved native bytes, not mutable summary assertions.
 * Only cleanup fields may change after the original completed evidence blob. */
export async function verifyContactResearchDiscovery(store,task,proof) {
  if(store.control?.path!==ROOT) fail('contact_research_scope_changed');
  if(!publicContactResearchTask(task)||proof?.requestId!==task.requestId||proof.sourceDigest!==task.sourceDigest
    ||proof.version!=='blueprint.contact-research-discovery.v1'||!dateOK(proof.run?.date)||!hashOK(proof.run?.rowBlob)) fail('contact_research_source_changed');
  const ref=store.db.doc(`${ROOT}/runs/${proof.run.date}`),current=(await ref.get()).data();
  if(current?.state!=='completed') fail('contact_research_run_changed');
  const row=JSON.parse(Buffer.from(await store.blobGet(proof.run.rowBlob),'base64'));
  const latest=JSON.parse(Buffer.from(await store.blobGet(current.blob),'base64'));
  if(row.date!==proof.run.date||row.run_key!==`blueprint-researcher:${row.date}`||row.run_key!==proof.run.runKey
    ||row.raw_output_digest!==proof.run.rawArtifactDigest||row.qa?.artifact_digest!==proof.run.qaArtifactDigest
    ||digest(stableRow(row))!==digest(stableRow(latest))||digest(current.metadata)!==digest(row.metadata)) fail('contact_research_run_changed');
  const context=await boundContext(store,row),record=(await store.db.doc(`${CONTACT}/${task.requestId}`).get()).data();
  if(!context.tasks.some(t=>digest(t)===digest(task))||record?.state!=='sources_ready'
    ||record.nativeRunKey!==row.run_key||record.inputDigest!==row.metadata.contact_research_digest
    ||!Number.isInteger(record.attempts)||record.attempts<1||digest(record.task)!==digest(task)||digest(record.discovery)!==digest(proof)) fail('contact_research_source_changed');
  const sources=await nativeSources(store,row,task);
  if(!sources.length||digest(proof.sources)!==digest(sources)||(await ref.get()).data()?.blob!==current.blob) fail('contact_research_receipt_changed');
  return proof;
}

export async function finishContactResearch(store,row,rowBlob) {
  if(store.control?.path!==ROOT) return;
  if(!row.metadata?.contact_research_digest||!['completed','failed','cancelled'].includes(row.state)) return;
  const claims=await store.db.collection(CONTACT).where('nativeRunKey','==',row.run_key).limit(4).get();
  const running=claims.docs.filter(s=>s.data()?.state==='running'&&s.data()?.inputDigest===row.metadata.contact_research_digest);
  if(!running.length) return; // Settled history never delays active observation.
  const context=await boundContext(store,row);
  const ready=row.state==='completed'&&row.qa?.state==='validated'&&hashOK(row.raw_output_digest)&&hashOK(row.qa.artifact_digest);
  for(const task of context.tasks) {
    if(!running.some(s=>s.ref.path===`${CONTACT}/${task.requestId}`)) continue;
    const sources=ready?await nativeSources(store,row,task):[];
    await store.transaction(async tx=>{
      const control=(await tx.get(store.control)).data();store.fence(control);store.workflowGate(control);
      const current=await tx.get(store.db.doc(`${ROOT}/runs/${row.date}`)),ref=store.db.doc(`${CONTACT}/${task.requestId}`),request=await tx.get(ref),old=request.data();
      if(current.data()?.blob!==rowBlob||old?.nativeRunKey!==row.run_key||old.inputDigest!==row.metadata.contact_research_digest
        ||old.state!=='running'||digest(publicContactResearchTask(old.task))!==digest(task)) return;
      const discovery=sources.length?{version:'blueprint.contact-research-discovery.v1',requestId:task.requestId,sourceDigest:task.sourceDigest,
        run:{date:row.date,runKey:row.run_key,rowBlob,rawArtifactDigest:row.raw_output_digest,qaArtifactDigest:row.qa.artifact_digest},sources}:null;
      tx.set(ref,{state:discovery?'sources_ready':'pending',
        reason:discovery?'agent_researched_operator_sources_ready':ready?'no_additional_verified_operator_source':'native_contact_research_attempt_incomplete',
        ...(discovery?{discovery}:{}),lastRunRef:current.ref.path,lastRowBlob:rowBlob,completedAt:store.clock(),sent:false},{merge:true});
    });
  }
}

/** Auxiliary recovery cannot undo an already committed research publication or
 * prevent cancellation/observation of an active session. Integrity failures
 * quarantine the original running claim without resetting its attempt count. */
export async function finishContactResearchSafely(store,row,rowBlob) {
  try {await finishContactResearch(store,row,rowBlob);return {state:'observed'};} catch(error) {
    const reason=/^contact_research_[a-z_]+$/.test(error?.message||'')?error.message:'contact_research_saved_proof_unavailable';
    try {
      await store.transaction(async tx=>{
        const control=(await tx.get(store.control)).data();store.fence(control);store.workflowGate(control);
        const current=await tx.get(store.db.doc(`${ROOT}/runs/${row.date}`));
        const claims=await tx.get(store.db.collection(CONTACT).where('nativeRunKey','==',row.run_key).limit(4));
        if(current.data()?.blob!==rowBlob) return;
        for(const request of claims.docs) if(request.data()?.state==='running'
          &&request.data()?.inputDigest===row.metadata?.contact_research_digest)
          tx.set(request.ref,{state:'blocked',reason,lastRowBlob:rowBlob,completedAt:store.clock(),sent:false},{merge:true});
        tx.set(store.db.doc(`${ROOT}/contactResearchRecoveryObservations/${row.date}`),{
          state:'blocked',reason,rowBlob,runKey:row.run_key,observedAt:store.clock(),providerStarted:false});
      });
    } catch { /* Disabled/stale authority does not grant an auxiliary write. */ }
    return {state:'blocked',reason,providerStarted:false};
  }
}
