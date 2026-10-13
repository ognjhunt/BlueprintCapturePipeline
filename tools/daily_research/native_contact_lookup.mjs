// Private, host-owned FullEnrich adapter. Absent scope means no paid dispatch.
// Uses the existing fenced worker lease and immutable native research receipts.
import {createHash} from 'node:crypto';
import {nativeSources} from './contact_research.mjs';
const ROOT='blueprintDailyResearch/sites-first';
const INPUT='blueprint.native-contact-lookup-input.v1', CONTROL='blueprint.native-contact-lookup-control.v1';
const canonical=x=>Array.isArray(x)?x.map(canonical):x&&typeof x==='object'
  ?Object.fromEntries(Object.keys(x).sort().map(k=>[k,canonical(x[k])])):x;
const json=x=>JSON.stringify(canonical(x)).replace(/[^\x00-\x7f]/g,c=>'\\u'+c.charCodeAt(0).toString(16).padStart(4,'0'));
const sha=x=>createHash('sha256').update(x).digest('hex'), hash=x=>sha(json(x));
const hex=x=>typeof x==='string'&&/^[a-f0-9]{64}$/.test(x);
const fail=code=>{throw Error(code);};
const names=x=>String(x||'').normalize('NFKC').toLowerCase().replace(/[^\p{L}\p{N}]+/gu,' ').trim();
const domain=x=>{try{const labels=new URL(x).hostname.toLowerCase().split('.');
  const country=labels.length>=3&&labels.at(-1).length===2&&['co','com','net','org','gov','ac','edu'].includes(labels.at(-2));
  return labels.slice(country?-3:-2).join('.');}catch{return null;}};
const live=s=>['creation_unresolved','running','cancel_pending','collecting','awaiting_review','reviewed'].includes(s);
const inputRef=(store,id)=>store.db.doc(`${ROOT}/nativeContactInputs/${id}`);
const callRef=(store,id)=>store.db.doc(`${ROOT}/nativeContactCalls/${id}`);
const scopeFields=['schema_version','enabled','scope_id','approval_reference','provider','resource_class','input_ids',
  'operations','max_credit_quarters','max_calls','effective_from','expires_at'];

function gate(store,control,input=null) {
  if(store.control?.path!==ROOT) fail('native_contact_scope_changed');
  store.fence(control);
  const scope=control.native_contact_lookup;
  if(!scope||scope.enabled!==true) return null;
  if(Object.keys(scope).sort().join(',')!==[...scopeFields].sort().join(',')
    ||scope.schema_version!==CONTROL||control.enabled!==true||control.config?.enabled!==true
    ||control.workflow?.enabled!==true||!hex(scope.scope_id)||typeof scope.approval_reference!=='string'
    ||!scope.approval_reference.trim()||/^PENDING/i.test(scope.approval_reference)
    ||scope.provider!=='fullenrich'||scope.resource_class!=='fullenrich_work_email'
    ||!Array.isArray(scope.input_ids)||!scope.input_ids.length||scope.input_ids.length>3||scope.input_ids.some(x=>!hex(x))
    ||new Set(scope.input_ids).size!==scope.input_ids.length
    ||!Array.isArray(scope.operations)||!scope.operations.includes('work_email')
    ||scope.operations.some(x=>!['person_search','work_email'].includes(x))||new Set(scope.operations).size!==scope.operations.length
    ||!Number.isSafeInteger(scope.max_credit_quarters)||scope.max_credit_quarters<1
    ||!Number.isSafeInteger(scope.max_calls)||scope.max_calls<1
    ||hash(Object.fromEntries(Object.entries(scope).filter(([key])=>key!=='scope_id')))!==scope.scope_id
    ||!(Date.parse(scope.effective_from)<=store.clock()&&store.clock()<Date.parse(scope.expires_at)))
    fail('native_contact_scope_not_admitted');
  if(input&&!scope.input_ids.includes(input.input_id)) fail('native_contact_input_not_authorized');
  store.workflowGate(control);
  return scope;
}
async function bound(store,context) {
  if(!hex(context?.input_id)||!hex(context?.input_digest)) fail('native_contact_input_invalid');
  const saved=(await inputRef(store,context.input_id).get()).data(),value=saved?.input;
  if(!value||saved.inputDigest!==context.input_digest||hash(value)!==context.input_digest
    ||value.input_id!==context.input_id||value.schema_version!==INPUT) fail('native_contact_input_changed');
  if(hash(Object.fromEntries(Object.entries(value).filter(([key])=>key!=='input_id')))!==value.input_id)
    fail('native_contact_input_changed');
  const current=(await store.db.doc(`${ROOT}/runs/${value.provenance.date}`).get()).data();
  const row=await store.get(value.provenance.date);
  if(current?.blob!==value.provenance.row_blob||row?.state!=='completed'
    ||row.raw_output_digest!==value.provenance.raw_artifact_digest||row.qa?.artifact_digest!==value.provenance.qa_artifact_digest)
    fail('native_contact_run_changed');
  // Check the original tool receipts, not mutable source labels.
  const refs=await nativeSources(store,row,{organizationUrl:value.candidate.organization_url},{candidateKey:value.candidate.candidate_key});
  const candidate=row.packet?.candidates?.find(c=>c.candidate_key===value.candidate.candidate_key),
    result=row.review?.lead_verification?.results?.find(r=>r.candidate_key===value.candidate.candidate_key);
  if(hash(candidate||null)!==hash(value.candidate)||hash(result?.assessment||null)!==hash(value.assessment))
    fail('native_contact_assessment_changed');
  for(const source of value.sources) {
    const ref=refs.find(r=>r.callId===source.call_id&&r.resultSha256===source.result_sha256&&r.url===source.url);
    if(!ref) fail('native_contact_source_changed');
    const event=JSON.parse(Buffer.from(await store.fileGet(`${row.date}-tool-${ref.callId}.json`),'base64'));
    const output=JSON.parse(event.output);
    if(output.text!==source.text||sha(source.text)!==source.text_sha256) fail('native_contact_source_changed');
  }
  const a=value.assessment;
  if(Date.parse(a.assessed_at)>store.clock()||(a.valid_until!==null&&!(Date.parse(a.valid_until)>store.clock())))
    fail('native_contact_assessment_stale');
  return value;
}

export async function prepareNativeContacts(store,day) {
  if(store.control?.path!==ROOT) fail('native_contact_scope_changed');
  if(!/^\d{4}-\d{2}-\d{2}$/.test(day||'')) fail('native_contact_date_invalid');
  await store.assertLease();
  // Preparing evidence is unpaid and can be reviewed before any provider scope is issued.
  const row=await store.get(day);if(row?.state!=='completed') return [];
  const meta=(await store.db.doc(`${ROOT}/runs/${day}`).get()).data(),results=row.review?.lead_verification?.results||[];
  const values=[];
  for(const candidate of row.packet?.candidates||[]) {
    const result=results.find(r=>r.candidate_key===candidate.candidate_key),assessment=result?.assessment;
    if(!assessment||assessment.claims?.operator?.status!=='verified_fact'
      ||assessment.claims?.physical_site?.status!=='verified_fact'
      ||!['verified_fact','inference'].includes(assessment.claims?.site_task?.status)
      ||assessment.counterevidence?.status==='contradicted') continue;
    let refs;
    try {
      refs=await nativeSources(store,row,{organizationUrl:candidate.organization_url},{candidateKey:candidate.candidate_key});
    } catch(error) {
      // Only authenticated candidate-local duplicate evidence is ineligible.
      // Changed QA, source receipts or run bindings still refuse the operation.
      if(error?.message==='native_contact_candidate_duplicate') continue;
      throw error;
    }
    const sources=[];
    for(const ref of refs) {
      const event=JSON.parse(Buffer.from(await store.fileGet(`${day}-tool-${ref.callId}.json`),'base64'));
      const output=JSON.parse(event.output);
      if(typeof output.text!=='string') fail('native_contact_source_changed');
      sources.push({url:ref.url,text:output.text,text_sha256:sha(output.text),checked_at:ref.checkedAt,
        call_id:ref.callId,result_sha256:ref.resultSha256,raw_source_sha256:ref.rawSourceSha256});
    }
    if(!sources.length) continue;
    const provenance={date:day,run_key:row.run_key,row_blob:meta.blob,candidate_key:candidate.candidate_key,
      raw_artifact_digest:row.raw_output_digest,qa_artifact_digest:row.qa.artifact_digest};
    const input={schema_version:INPUT,candidate,assessment,sources,provenance};
    input.input_id=hash(input);
    // Firestore documents must stay under 1MiB, never silently truncate evidence.
    if(Buffer.byteLength(json(input))>700000) fail('native_contact_input_resource_limit');
    values.push(input);
  }
  // Reserve more than half of Firestore's 10MiB transaction allowance for
  // reads, index entries and protocol overhead. Count complete write envelopes
  // plus bounded per-document overhead before any transaction or input write.
  const cohortBytes=values.reduce((bytes,input)=>bytes+Buffer.byteLength(json({input,
    inputDigest:hash(input),preparedAt:store.clock(),sent:false}))+8192,0);
  if(cohortBytes>4*1024*1024) fail('native_contact_cohort_resource_limit');
  // Validate the whole eligible cohort before creating any approvable input.
  // One transaction also prevents a later conflicting input or changed run
  // from leaving newly created siblings behind a refused preparation.
  if(values.length) await store.transaction(async tx=>{
    store.fence((await tx.get(store.control)).data());
    const run=await tx.get(store.db.doc(`${ROOT}/runs/${day}`));
    if(run.data()?.blob!==meta.blob) fail('native_contact_run_changed');
    const refs=values.map(input=>inputRef(store,input.input_id)),priors=await Promise.all(refs.map(ref=>tx.get(ref)));
    for(const [index,input] of values.entries())
      if(priors[index].exists&&priors[index].data().inputDigest!==hash(input)) fail('native_contact_input_changed');
    for(const [index,input] of values.entries())
      if(!priors[index].exists) tx.create(refs[index],{input,inputDigest:hash(input),preparedAt:store.clock(),sent:false});
  });
  return values;
}

function requestValid(input,event,request) {
  const d=domain(input.candidate.organization_url),body=request?.body;
  if(!request||event.kind!==request.kind||event.site_key!==request.site_key
    ||event.call!==hash(['fullenrich',request.kind,request.identity])||!body) return false;
  if(request.kind==='search') return event.most_credits==='1.25'
    &&request.identity[0]===d&&request.identity[2]==='United States'
    &&body.limit===5&&body.offset===0&&Object.keys(body).sort().join(',')==='current_company_domains,current_position_titles,limit,offset,person_locations'
    &&json(body.current_company_domains)===json([{value:d,exact_match:true}])
    &&json(body.person_locations)===json([{value:'United States',exact_match:true}])
    &&Array.isArray(request.identity[1])&&request.identity[1].length<=30
    &&json(body.current_position_titles)===json(request.identity[1].map(value=>({value,exact_match:false})));
  const person=body.data?.[0];
  return request.kind==='enrich'&&event.most_credits==='1'&&Array.isArray(body.data)&&body.data.length===1
    &&Object.keys(body).sort().join(',')==='data,name'&&person.domain===d
    &&person.company_name===input.candidate.organization
    &&json(person.enrich_fields)===json(['contact.work_emails'])
    &&Object.keys(person).sort().join(',')==='company_name,custom,domain,enrich_fields,first_name,last_name'
    &&typeof person.first_name==='string'&&typeof person.last_name==='string'
    &&request.identity[0]===names(person.first_name+' '+person.last_name)&&request.identity[1]===d
    &&json(person.custom)===json({call:event.call})&&body.name==='blueprint-contact-lookup-'+event.call.slice(0,12);
}

export async function nativeContactOperation(store,request) {
  if(store.control?.path!==ROOT) fail('native_contact_scope_changed');
  if(request.op==='native_contact_prepare') return prepareNativeContacts(store,request.day);
  if(request.op==='native_contact_scope_set') {
    const value=request.value;
    if(!(request.expected_scope_id===null||hex(request.expected_scope_id))) fail('native_contact_scope_invalid');
    return store.transaction(async tx=>{
      const control=(await tx.get(store.control)).data();
      await store.publicationFence(tx,control);
      if((control.native_contact_lookup?.scope_id||null)!==request.expected_scope_id) fail('native_contact_scope_conflict');
      if(value?.enabled===false) {
        if(json(value)!==json({enabled:false})) fail('native_contact_scope_invalid');
      } else if(!value||typeof value!=='object'||Array.isArray(value)||value.enabled!==true
        ||!gate(store,{...control,native_contact_lookup:value})) fail('native_contact_scope_invalid');
      // Firestore merge recursively retains nested fields. Replace this map
      // completely while preserving the fenced, transaction-read root.
      tx.set(store.control,{...control,native_contact_lookup:value});return true;
    });
  }
  if(request.op==='native_contact_inputs') {
    const control=(await store.control.get()).data(),scope=gate(store,control);
    if(!scope) return [];
    const values=[];
    for(const id of scope.input_ids) {
      const saved=(await inputRef(store,id).get()).data();
      values.push(await bound(store,{input_id:id,input_digest:saved?.inputDigest}));
    }
    return values;
  }
  const input=await bound(store,request.context||{input_id:request.input_id,input_digest:request.input_digest});
  if(request.op==='native_contact_scope') {
    const control=(await store.control.get()).data(),scope=gate(store,control,input);
    return scope?{dispatch_enabled:true,approval_reference:scope.approval_reference,person_search:scope.operations.includes('person_search')}:null;
  }
  if(request.op==='native_contact_call_get') {
    if(!hex(request.key)) fail('native_contact_call_invalid');
    store.fence((await store.control.get()).data());
    return (await callRef(store,request.key).get()).data()||null;
  }
  if(request.op==='native_contact_result_put') {
    if(request.result?.input_id!==input.input_id||request.result.sends_authorized!==false) fail('native_contact_result_invalid');
    return store.transaction(async tx=>{
      store.fence((await tx.get(store.control)).data());
      tx.set(inputRef(store,input.input_id),{result:request.result,resultDigest:hash(request.result),observedAt:store.clock()},{merge:true});return true;
    });
  }
  if(request.op==='native_contact_call_append') {
    const e=request.event;
    if(e?.schema_version!=='blueprint.contact-lookup.journal.v1'||!hex(e.call)||!/^[a-f0-9]{16}$/.test(e.attempt||'')
      ||!['intent','created','answered','refused','uncertain'].includes(e.event)) fail('native_contact_journal_invalid');
    return store.transaction(async tx=>{
      const control=(await tx.get(store.control)).data();store.fence(control);
      const ref=callRef(store,e.call),saved=(await tx.get(ref)).data(),events=saved?.events||[],last=events.at(-1);
      if(events.length>=100) fail('native_contact_journal_resource_limit');
      if(e.event==='intent') {
        const scope=gate(store,control,input);
        if(!scope||!requestValid(input,e,request.request)) fail('native_contact_request_not_admitted');
        if(e.kind==='search'&&!scope.operations.includes('person_search')) fail('native_contact_search_not_authorized');
        if(last&&last.event!=='refused') fail('native_contact_call_already_claimed');
        const meter=store.db.doc(`${ROOT}/nativeContactScopes/${scope.scope_id}`),prior=(await tx.get(meter)).data()||{};
        if(prior.scopeDigest&&prior.scopeDigest!==hash(scope)) fail('native_contact_scope_changed');
        const quarters=(prior.reservedCreditQuarters||0)+(e.kind==='search'?5:4),calls=(prior.reservedCalls||0)+1;
        if(quarters>scope.max_credit_quarters||calls>scope.max_calls) fail('native_contact_scope_exhausted');
        tx.set(meter,{reservedCreditQuarters:quarters,reservedCalls:calls,scopeDigest:hash(scope)},{merge:true});
        tx.set(ref,{events:[...events,e],request:request.request,context:request.context,scopeDigest:hash(scope),dispatchClaim:null});
      } else {
        if(!last||last.attempt!==e.attempt||!['intent','created'].includes(last.event)
          ||e.event==='created'&&last.event!=='intent') fail('native_contact_journal_invalid');
        tx.set(ref,{events:[...events,e]},{merge:true});
      }
      return true;
    });
  }
  if(request.op==='native_contact_dispatch_admission') {
    if(hash(request.body)!==request.body_digest) fail('native_contact_request_changed');
    const kind=request.path==='/api/v2/people/search'?'search':request.path==='/api/v2/contact/enrich/bulk'?'enrich':null;
    if(!kind) fail('native_contact_request_changed');
    const key=kind==='enrich'?request.body.data?.[0]?.custom?.call:
      hash(['fullenrich','search',[request.body.current_company_domains?.[0]?.value,
        request.body.current_position_titles?.map(t=>t.value),'United States']]);
    if(!hex(key)) fail('native_contact_call_invalid');
    return store.transaction(async tx=>{
      const control=(await tx.get(store.control)).data(),scope=gate(store,control,input);
      const ref=callRef(store,key),saved=(await tx.get(ref)).data(),event=saved?.events?.at(-1);
      if(!scope||saved.scopeDigest!==hash(scope)||saved.dispatchClaim||event?.event!=='intent'
        ||hash(saved.context)!==hash(request.context)||hash(saved.request?.body)!==request.body_digest
        ||!requestValid(input,event,saved.request)) fail('native_contact_dispatch_not_admitted');
      const runs=await tx.get(store.db.collection(`${ROOT}/runs`).limit(10001));
      if(runs.docs.length>10000||runs.docs.some(s=>live(s.data().state))) fail('native_contact_research_active');
      if(runs.docs.find(s=>s.ref.path===`${ROOT}/runs/${input.provenance.date}`)?.data()?.blob!==input.provenance.row_blob)
        fail('native_contact_run_changed');
      const binding=hash({scope:hash(scope),input:request.context.input_digest,call:key,attempt:event.attempt,body:request.body_digest});
      tx.set(ref,{dispatchClaim:{binding,claimedAt:store.clock(),leaseGeneration:control.lease.generation}},{merge:true});
      return {schema_version:'paid_lane_admission.v1',status:'admitted',resource_class:'fullenrich_work_email',
        blockers:[],allocation_binding_digest:binding};
    });
  }
  fail('native_contact_operation_invalid');
}
