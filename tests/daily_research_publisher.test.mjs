import {test} from 'node:test';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {Publisher,planSheets,planNotion,publicationVerification,openChecks,firstQuestion,QUESTION_TEMPLATES} from '../tools/daily_research/publisher.mjs';
import {Store,ROOT} from '../tools/daily_research/firestore_bridge.mjs';
import {MemoryFirestore} from './fixtures/daily_research/firestore-memory.mjs';
import {verificationDigest} from '../tools/daily_research/verification-digest.mjs';

const SHEET='1n95Ih0Swc-q-kZyUaDHoZh6SVzxvf_zt-CRR7i39bWY', NOTION='3eb80154161d8116858ed5f376b4b7a9';
const sha=x=>createHash('sha256').update(x).digest('hex');
const headers=['Prospect ID','Organization','Prospect type','Site / team','Contact name','Contact details','Verification',
  'Contact source URL','Robot-team fit','Task evidence URL','Stage','Owner','Next action','Next action date','Task / job',
  'Robot capability evidence URL','Evidence maturity','Geography','Evidence checked date'];
const candidate=()=>({organization:'Example Plant',site:'North',task:'Depositing',location:'Chicago',
  qualification_status:'unqualified',unknowns:['Interest unknown'],potential_robot_match:'Hypothesis',
  proposed_next_action:'Review exact operator evidence',evidence:['task','capability'].map(role=>({role,
    url:'https://plant.example/tasks',classification:'operator',claim_kind:'fact',claim:'Task described',checked_date:'2026-09-30'}))});
// Invented evidence for hermetic boundary tests, never a real verified lead.
function verification(candidates) {
  const results=candidates.map(c=>{
    const assessment={version:'blueprint.lead-verification.v1',candidate_digest:verificationDigest(c),
      assessed_at:'2026-09-30T00:00:00Z',valid_until:'2030-01-01T00:00:00Z',
      claims:Object.fromEntries(['operator','physical_site','site_task','human_workflow','plausible_fit'].map(name=>[name,{
        status:name==='plausible_fit'?'inference':'verified_fact',reason:'Synthetic exact site/task assessment for boundary tests.',source_refs:['synthetic']} ])),
      sources:[{id:'synthetic',url:'https://plant.example/tasks',publisher:'Invented Example Plant',source_date:'2026-09-30',
        event_date:'2026-09-30',checked_at:'2026-09-30T00:00:00Z',retrieval:'rendered',classification:'operator',
        quote:'Synthetic operator, North physical site and manual depositing workflow.',freshness:'current',freshness_reason:'Invented current test source.'}],
      counterevidence:{status:'checked',reason:'Invented source check; no synthetic conflict.',source_refs:['synthetic'],searches:['Synthetic automation check']}};
    return {version:'blueprint.lead-verification-result.v1',candidate_key:c.candidate_key,candidate_digest:verificationDigest(c),
      assessment,assessment_digest:verificationDigest(assessment),evaluated_at:'2026-09-30T00:00:00Z',status:'verified',
      reasons:[],eligible_for_qualified_promotion:true};
  });
  return {version:'blueprint.lead-verification.v1',results};
}
function row(candidates=[candidate()],summary='Supported brief with explicit gaps') {
  candidates=candidates.map((c,index)=>({...c,candidate_key:c.candidate_key || 'synthetic-'+index}));
  const r={date:'2026-09-30',run_key:'blueprint-researcher:2026-09-30',metadata:{run_key:'date'},
    state:'reviewed',cleanup_required:true,packet:{candidates},packet_digest:'a'.repeat(64),qa:{state:'validated'},
    review:{packet_digest:'a'.repeat(64),source_support_verified:true,crm_rechecked:true,lead_verification:verification(candidates)},delivery:{}};
  for (const [name,payload] of Object.entries({sheets:{sheet_id:SHEET,tab:'Prospects',candidates},
    notion:{parent_id:NOTION,summary,candidates}})) {
    const raw=JSON.stringify(payload);
    r.delivery[name]={key:r.run_key+':'+name,payload,payload_json:raw,payload_digest:sha(raw),state:'pending'};
  }
  return r;
}
function expiresAt(r,now) {
  for(const result of r.review.lead_verification.results) {
    result.assessment.valid_until=new Date(now).toISOString();
    result.assessment_digest=verificationDigest(result.assessment);
  }
  return r;
}

async function fixture(suppliedRow=row()) {
  const db=new MemoryFirestore(),values=[['CRM'],[],[],[],headers], pages=[],writes=[];
  const time={now:Date.now()}, faults={lost:false,changed:false,lostAppend:false,unobservedAppend:false};
  const snapshot=()=>({sheet_id:SHEET,complete:true,values:structuredClone(values)});
  const google=async(method,path,body)=>{
    if(method==='GET') return {sheets:[]};
    assert.equal(method,'PUT');
    assert.equal(decodeURIComponent(path),`/values/Prospects!A${values.length+1}:S${values.length+body.values.length}?valueInputOption=RAW`);
    assert.ok(path.includes('valueInputOption=RAW'));assert.equal(body.values[0].length,19);
    assert.ok(db.values.get(`${ROOT}/runs/2026-09-30`).publication_claimed.sheets);
    assert.ok((await store.get('2026-09-30')).delivery.sheets.plan);
    writes.push({destination:'sheets',body});values.push(...body.values);
    if(faults.lost) {faults.lost=false;throw new Error('private upstream error');}
    return {};
  };
  const notion=async(method,path,body)=>{
    if(method==='PATCH') {
      assert.equal(path,'/blocks/page-new/children');
      const claims=db.values.get(`${ROOT}/runs/2026-09-30`).publication_batches.notion;
      assert.ok(Object.values(claims).some(c=>c.request_digest===sha(JSON.stringify(body)) && c.page_id==='page-new'));
      assert.ok(body.children.length<=100 && Buffer.byteLength(JSON.stringify(body))<=500000);
      writes.push({destination:'notion-append',body});
      if(faults.unobservedAppend) throw new Error('private timeout before observable acceptance');
      pages[0].body.children.push(...body.children);
      if(faults.lostAppend) {faults.lostAppend=false;throw new Error('private lost append reply');}
      return {};
    }
    if(method==='POST') {
      assert.equal(path,'/pages');assert.equal(body.parent.page_id,NOTION);
      assert.ok(db.values.get(`${ROOT}/runs/2026-09-30`).publication_claimed.notion);
      assert.ok((await store.get('2026-09-30')).delivery.notion.plan);
      assert.ok(body.children.length<=100 && Buffer.byteLength(JSON.stringify(body))<=500000);
      pages.push({id:'page-new',body});writes.push({destination:'notion',body:structuredClone(body)});
      if(faults.lost) {faults.lost=false;throw new Error('private upstream error');}
      return {id:'page-new'};
    }
    if(path===`/pages/${NOTION}`) return {id:NOTION,object:'page'};
    if(path.startsWith(`/blocks/${NOTION}/children`)) return {has_more:false,results:pages.map(p=>({id:p.id,type:'child_page',
      child_page:{title:p.body.properties.title.title[0].text.content}}))};
    if(path==='/pages/page-new') return {parent:{page_id:NOTION}};
    if(path.startsWith('/blocks/page-new/children')) {
      const cursor=Number(new URL('https://fixture.invalid'+path).searchParams.get('start_cursor')||0);
      const results=pages[0].body.children.slice(cursor,cursor+100).map((b,index)=>({id:'block-'+(cursor+index),...b})),next=cursor+results.length;
      return {has_more:next<pages[0].body.children.length,next_cursor:String(next),results};
    }
    throw new Error('unexpected request '+path);
  };
  const publisher=new Publisher({crmReader:async()=>snapshot(),google,notion,clock:()=>time.now});
  db.values.set(ROOT,{schema_version:'blueprint.research-control.v1',enabled:true,workflow:{enabled:true,
    qa_authority_reference:'approved-QA',publication_authority_reference:'approved-fixed-targets'}});
  const store=new Store(db,()=>time.now,'one',async()=>snapshot(),publisher);
  await store.acquire();
  const r=suppliedRow; await store.put({...r,state:'creating'});await store.put(r);
  return {db,store,publisher,time,faults,values,pages,writes,r,snapshot};
}

async function finishNotion(f) {
  for(let attempt=0;attempt<100;attempt++) {
    const receipt=await f.store.publish(f.r.date);
    if(receipt) return receipt;
  }
  assert.fail('publication did not complete');
}

for(const issue of ['missing','unresolved','rejected','candidate_drift','assessment_drift','duplicate','expired'])
  test(`fresh protected publication refuses ${issue} evidence before any claim or write`,async()=>{
    const r=row(),result=r.review.lead_verification.results[0];
    if(issue==='missing') delete r.review.lead_verification;
    if(['unresolved','rejected'].includes(issue)) result.status=issue;
    if(issue==='candidate_drift') result.candidate_digest='b'.repeat(64);
    if(issue==='assessment_drift') result.assessment.claims.human_workflow.reason='Changed original evidence';
    if(issue==='duplicate') result.duplicate_of='another-candidate';
    if(issue==='expired') expiresAt(r,Date.now()-1000);
    const f=await fixture(r);
    await assert.rejects(f.store.publish(r.date),/publication_lead_verification_required/);
    assert.equal(f.writes.length,0);assert.deepEqual(f.db.values.get(`${ROOT}/runs/${r.date}`).publication_claimed,{});
    assert.equal((await f.store.get(r.date)).delivery.notion.plan,undefined);
  });

test('each fresh Notion continuation requires current verification before its claim',async()=>{
  const r=expiresAt(row([candidate()],'Invented complete report. '.repeat(12000)),Date.now()+10000),f=await fixture(r);
  assert.equal(await f.store.publish(r.date),null);assert.equal(f.writes.length,1);
  f.time.now+=10001;
  await assert.rejects(f.store.publish(r.date),/publication_lead_verification_required/);
  assert.equal(f.writes.length,1);
  assert.deepEqual(Object.keys(f.db.values.get(`${ROOT}/runs/${r.date}`).publication_batches.notion),['0']);
});

test('expiry after a durable Notion batch claim never executes or retries that write',async()=>{
  const r=expiresAt(row([candidate()],'Invented complete report. '.repeat(12000)),Date.now()+10000),f=await fixture(r);
  f.publisher.beforeNotionStep=async()=>{f.time.now+=10001;};
  await assert.rejects(f.store.publish(r.date),/publication_lead_verification_required/);
  assert.equal(f.writes.length,0);assert.ok(f.db.values.get(`${ROOT}/runs/${r.date}`).publication_batches.notion[0]);
  assert.equal(await f.store.publish(r.date),null);assert.equal(f.writes.length,0);
});

test('final Sheets target reads cannot use evidence that expires during the read',async()=>{
  const r=expiresAt(row(),Date.now()+10000),f=await fixture(r),plan=planSheets(r,f.snapshot(),f.time.now);
  f.publisher.sheetsTargetEmpty=async()=>{f.time.now+=10001;};
  await assert.rejects(f.publisher.write(r,'sheets',plan),/publication_lead_verification_required/);
  assert.equal(f.writes.length,0);
});

for(const legacy of [false,true]) test(`${legacy?'legacy missing':'expired'} assessment preserves exact claimed GET readback`,async()=>{
  const r=expiresAt(row(),Date.now()+10000),f=await fixture(r);f.faults.lost=true;
  await assert.rejects(f.store.publish(r.date),/publication_attempt_unresolved/);
  assert.equal(f.writes.length,1);
  if(legacy) {const saved=await f.store.get(r.date);delete saved.review.lead_verification;await f.store.put(saved);}
  else f.time.now+=10001;
  assert.equal((await f.store.publish(r.date)).readback_verified,true);assert.equal(f.writes.length,1);
  await assert.rejects(f.store.publish(r.date,'sheets'),/publication_lead_verification_required/);
  assert.equal(f.writes.length,1);
});

test('CRM physical identity keeps distinct named sites in one city and tasks or operators separate',async()=>{
  const r=row(),f=await fixture(r),base=Array(19).fill('');
  Object.assign(base,{0:'BP-000009',1:r.packet.candidates[0].organization,3:'South',9:'https://plant.example/tasks',
    14:r.packet.candidates[0].task,17:r.packet.candidates[0].location});f.values.push(base);
  assert.equal(planSheets(r,f.snapshot()).sheet_rows.length,1);
  base[3]='NORTH';assert.throws(()=>planSheets(r,f.snapshot()),/publication_crm_duplicate_changed/);
  base[1]='Distinct co-located operator';assert.equal(planSheets(r,f.snapshot()).sheet_rows.length,1);
  base[1]=r.packet.candidates[0].organization;base[14]='Other task';assert.equal(planSheets(r,f.snapshot()).sheet_rows.length,1);
});

test('234 KB QA report publishes every paragraph through bounded requests and paginated exact readback',async()=>{
  const summary='Full retained report; actual interest unknown. '.repeat(5200);
  const f=await fixture(row([candidate()],summary)),plan=planNotion(f.r);
  assert.ok(Buffer.byteLength(summary)>234000 && plan.paragraphs.length>100);
  assert.equal((await finishNotion(f)).readback_verified,true);
  const actual=f.pages[0].body.children.map(b=>b.paragraph.rich_text[0].text.content);
  assert.deepEqual(actual,plan.paragraphs);
  assert.ok(actual.slice(1).join('').includes(summary));
  assert.equal(f.writes.filter(w=>w.destination==='notion').length,1);
  assert.ok(f.writes.some(w=>w.destination==='notion-append'));
  assert.equal((await f.store.publish(f.r.date)).readback_verified,true);
  assert.equal(f.writes.filter(w=>w.destination==='notion').length,1);
});

test('multibyte and escaped text preserves complete content inside count and byte request limits',async()=>{
  const summary=('漢字😀𐀀\u0001'.repeat(40000));
  const f=await fixture(row([],summary)),plan=planNotion(f.r);
  assert.equal((await finishNotion(f)).readback_verified,true);
  assert.ok(plan.paragraphs.slice(1).join('').includes(summary));
  for(const write of f.writes) {
    assert.ok(write.body.children.length<=100);
    assert.ok(Buffer.byteLength(JSON.stringify(write.body))<=500000);
    for(const block of write.body.children) {
      const text=block.paragraph.rich_text[0].text.content;
      assert.ok(text.length<=1800 && !/[\uD800-\uDBFF]$/.test(text) && !/^[\uDC00-\uDFFF]/.test(text));
    }
  }
});

test('lost accepted append reply resumes by full ordered prefix without duplicate page or blocks',async()=>{
  const f=await fixture(row([], 'Retained exact source '.repeat(20000)));
  assert.equal(await f.store.publish(f.r.date),null);
  f.faults.lostAppend=true;
  await assert.rejects(f.store.publish(f.r.date),/publication_attempt_unresolved/);
  await f.store.release();const restarted=new Store(f.db,()=>f.time.now,'two',null,f.publisher);await restarted.acquire();
  f.store=restarted;
  assert.equal((await finishNotion(f)).readback_verified,true);
  assert.deepEqual(f.pages[0].body.children.map(b=>b.paragraph.rich_text[0].text.content),planNotion(f.r).paragraphs);
  assert.equal(f.writes.filter(w=>w.destination==='notion').length,1);
});

test('unknown unobserved append stays observation-only and partial or duplicate content is explicit',async()=>{
  const f=await fixture(row([], 'Retained exact source '.repeat(20000)));
  await f.store.publish(f.r.date);f.faults.unobservedAppend=true;
  await assert.rejects(f.store.publish(f.r.date),/publication_attempt_unresolved/);
  const writes=f.writes.length;
  assert.equal(await f.store.publish(f.r.date),null);assert.equal(f.writes.length,writes);
  f.pages[0].body.children.push(f.writes.at(-1).body.children[0]);
  await assert.rejects(f.store.publish(f.r.date),/publication_notion_partial_batch_unresolved/);
  f.pages[0].body.children.push(f.pages[0].body.children[0]);
  await assert.rejects(f.store.publish(f.r.date),/publication_readback_conflict/);
  assert.equal(f.writes.length,writes);
});

test('request-byte limit independently paginates fewer than ninety text blocks without truncation',async()=>{
  const f=await fixture(row([], '\u0001'.repeat(100000))),plan=planNotion(f.r);
  assert.ok(plan.paragraphs.length<90 && plan.batches.length>1);
  assert.equal((await finishNotion(f)).readback_verified,true);
  for(const write of f.writes) assert.ok(Buffer.byteLength(JSON.stringify(write.body))<=500000);
});

test('old small persisted plan and receipt remain byte-identical',async()=>{
  const r=row(),plan=planNotion(r,{legacy:true});
  assert.deepEqual(planNotion(r),plan);
  assert.equal(plan.protocol,undefined);
  const f=await fixture(r);await f.store.put({...r,delivery:{...r.delivery,notion:{...r.delivery.notion,plan}}});
  const receipt=await f.store.publish(r.date);
  assert.deepEqual(receipt,{destination:'notion',key:plan.key,payload_digest:plan.payload_digest,
    readback_verified:true,reference:'notion:page-new'});
  assert.equal(JSON.stringify(f.writes[0].body),plan.body_json);
});

test('old valid request above new conservative byte target replays exactly; unsupported old body is never posted',async()=>{
  const r=row([], '\u0001'.repeat(77000)),old=planNotion(r,{legacy:true});
  assert.ok(Buffer.byteLength(old.body_json)>450000 && Buffer.byteLength(old.body_json)<500000);
  const f=await fixture(r);await f.store.put({...r,delivery:{...r.delivery,notion:{...r.delivery.notion,plan:old}}});
  assert.equal((await f.store.publish(r.date)).readback_verified,true);
  assert.equal(JSON.stringify(f.writes[0].body),old.body_json);
  const invalid=row([], '\u0001'.repeat(100000)),legacy=planNotion(invalid,{legacy:true});
  await assert.rejects(f.publisher.write(invalid,'notion',legacy),/publication_notion_request_too_large/);
  await assert.rejects(f.publisher.write(invalid,'notion',planNotion(invalid)),/publication_paginated_batch_required/);
  assert.equal(f.writes.length,1);
});

for(const change of ['disabled','authority','lease']) test(`append final action fence refuses ${change} after durable claim`,async()=>{
  const f=await fixture(row([], 'Exact retained report '.repeat(20000)));
  await f.store.publish(f.r.date);const original=f.store.transaction.bind(f.store);
  f.store.transaction=async(fn)=>{
    const result=await original(fn);
    if(f.db.values.get(`${ROOT}/runs/${f.r.date}`).publication_batches?.notion?.[1]) {
      const control=f.db.values.get(ROOT);
      if(change==='disabled') control.enabled=false;
      if(change==='authority') control.workflow.publication_authority_reference='changed-authority';
      if(change==='lease') f.time.now=control.lease.expires_at_ms;
    }
    return result;
  };
  await assert.rejects(f.store.publish(f.r.date),/workflow_authority_missing|publication_authority_or_plan_changed|firestore_lease_lost/);
  assert.equal(f.writes.filter(write=>write.destination==='notion-append').length,0);
  assert.ok(f.db.values.get(`${ROOT}/runs/${f.r.date}`).publication_batches.notion[1]);
});

for(const change of ['disabled','authority','lease']) test(`final combined control snapshot checks ${change} without a later lease-only read`,async()=>{
  const f=await fixture(row([], 'Exact retained report '.repeat(20000)));await f.store.publish(f.r.date);
  const original=f.store.control.get.bind(f.store.control);let finalReads=0;
  f.store.control.get=async()=>{
    if(f.db.values.get(`${ROOT}/runs/${f.r.date}`).publication_batches?.notion?.[1]) {
      finalReads++;
      const control=f.db.values.get(ROOT);
      if(change==='disabled') control.enabled=false;
      if(change==='authority') control.workflow.publication_authority_reference='changed-at-final-read';
      if(change==='lease') control.lease.expires_at_ms=f.time.now;
    }
    return original();
  };
  await assert.rejects(f.store.publish(f.r.date),/workflow_authority_missing|publication_authority_or_plan_changed|firestore_lease_lost/);
  assert.equal(finalReads,1);
  assert.equal(f.writes.filter(write=>write.destination==='notion-append').length,0);
});

test('large report refuses duplicate pages and looping child cursors without additional writes',async()=>{
  const f=await fixture(row([], 'Exact retained report '.repeat(20000)));
  await f.store.publish(f.r.date);
  f.pages.push({id:'duplicate-page',body:structuredClone(f.pages[0].body)});
  await assert.rejects(f.store.publish(f.r.date),/publication_notion_duplicate_pages/);
  f.pages.pop();const original=f.publisher.notion;
  f.publisher.notion=async(method,path,...args)=>path.startsWith('/blocks/page-new/children')
    ? {has_more:true,next_cursor:'loop',results:[]} : original(method,path,...args);
  await assert.rejects(f.store.publish(f.r.date),/publication_notion_pagination_invalid/);
  assert.equal(f.writes.length,1);
});

test('transaction callback replay does not repeat paginated create or appends',async()=>{
  const f=await fixture(row([], 'Exact retained report '.repeat(20000)));
  f.db.replay=true;
  assert.equal((await finishNotion(f)).readback_verified,true);
  const plan=planNotion(f.r);
  assert.equal(f.writes.length,plan.batches.length);
  assert.deepEqual(f.pages[0].body.children.map(b=>b.paragraph.rich_text[0].text.content),plan.paragraphs);
});

test('lost large-report page-create reply resumes without creating a second page',async()=>{
  const f=await fixture(row([], 'Exact retained report '.repeat(20000)));
  f.faults.lost=true;
  await assert.rejects(f.store.publish(f.r.date),/publication_attempt_unresolved/);
  assert.equal((await finishNotion(f)).readback_verified,true);
  assert.equal(f.pages.length,1);
  assert.equal(f.writes.filter(write=>write.destination==='notion').length,1);
});

test('duplicate block IDs across content pages cannot falsely prove a repeated-text report',async()=>{
  const f=await fixture(row([], 'x'.repeat(240000)));await finishNotion(f);
  const original=f.publisher.notion;
  f.publisher.notion=async(method,path,...args)=>{
    const result=await original(method,path,...args);
    if(path.startsWith('/blocks/page-new/children')) result.results.forEach((block,index)=>{block.id='duplicated-'+index;});
    return result;
  };
  await assert.rejects(f.store.publish(f.r.date),/publication_notion_pagination_invalid/);
});

test('identical import preserves consumed publication claims and new runnable paginated restore refuses',async()=>{
  const f=await fixture(row([], 'Exact retained report '.repeat(20000)));await f.store.publish(f.r.date);
  f.faults.unobservedAppend=true;
  await assert.rejects(f.store.publish(f.r.date),/publication_attempt_unresolved/);
  const saved=await f.store.get(f.r.date),ref=`${ROOT}/runs/${f.r.date}`,before=structuredClone(f.db.values.get(ref));
  f.db.values.get(ROOT).enabled=false;
  await f.store.importRun(saved);
  assert.deepEqual(f.db.values.get(ref),before);
  const proof=(await f.store.snapshot(f.r.date)).publication_manifest;
  assert.equal(sha(proof.manifest_json),proof.manifest_digest);
  const evidence=JSON.parse(proof.manifest_json);
  assert.deepEqual(evidence.publication_batches,before.publication_batches);
  assert.equal(evidence.plans.notion.plan_digest,before.publication_batches.notion[1].plan_digest);
  f.db.values.get(ROOT).enabled=true;
  const writes=f.writes.length;
  assert.equal(await f.store.publish(f.r.date),null);
  assert.equal(f.writes.length,writes);
  f.db.values.get(ROOT).enabled=false;
  f.db.values.delete(ref);
  await assert.rejects(f.store.importRun(saved),/firestore_publication_restore_manifest_required/);
  assert.equal(f.db.values.has(ref),false);
  const archival=structuredClone(saved);archival.state='completed';
  for(const delivery of Object.values(archival.delivery)) {
    delivery.state='acknowledged';delivery.receipt={readback_verified:true};
  }
  await f.store.importRun(archival);
  assert.equal((await f.store.get(f.r.date)).state,'completed');
  assert.equal(f.writes.length,2);
});

test('snapshot retains date/run metadata admission while capturing exact immutable source row bytes',async()=>{
  const f=await fixture(row([], 'Exact retained report '.repeat(20000)));await f.store.publish(f.r.date);
  const snapshot=await f.store.snapshot(f.r.date),proof=JSON.parse(snapshot.publication_manifest.manifest_json);
  assert.equal(sha(proof.source_row_json),proof.source_row_blob);
  assert.deepEqual(JSON.parse(proof.source_row_json),snapshot.row);
  await assert.rejects(f.store.snapshot('not-a-date'),/firestore_date_invalid/);
  f.db.values.get(`${ROOT}/runs/${f.r.date}`).metadata={run_key:'wrong-lineage'};
  await assert.rejects(f.store.snapshot(f.r.date),/firestore_row_binding_invalid/);
  assert.equal(f.writes.length,1);
});

test('canonical row property-order roundtrip preserves batch identity and all original request bytes',async()=>{
  const f=await fixture(row([], 'Exact retained report '.repeat(20000)));await f.store.publish(f.r.date);
  const before=await f.store.get(f.r.date),body=before.delivery.notion.plan.body_json;
  const sorted=value=>Array.isArray(value)?value.map(sorted):value && typeof value==='object'
    ? Object.fromEntries(Object.keys(value).sort().map(key=>[key,sorted(value[key])])) : value;
  await f.store.put(sorted(before));
  assert.equal((await finishNotion(f)).readback_verified,true);
  assert.equal((await f.store.get(f.r.date)).delivery.notion.plan.body_json,body);
  const proof=JSON.parse((await f.store.snapshot(f.r.date)).publication_manifest.manifest_json);
  assert.equal(proof.plans.notion.plan_digest,proof.publication_batches.notion[0].plan_digest);
});

test('automatic publication persists each plan/claim before one write and reads back exact content',async()=>{
  const f=await fixture();
  const receipt=await f.store.publish(f.r.date);
  assert.equal(receipt.destination,'notion');assert.equal(receipt.readback_verified,true);
  let saved=await f.store.get(f.r.date);saved.delivery.notion.state='acknowledged';await f.store.put(saved);
  const sheetReceipt=await f.store.publish(f.r.date);
  assert.equal(sheetReceipt.readback_verified,true);
  assert.equal(f.values[5][0],'BP-000001');assert.deepEqual(f.values[5].slice(4,6),['','']);
  assert.equal(f.values[5][10],'Research');assert.equal(f.values[5][16],'Unverified');
  assert.deepEqual(f.writes.map(x=>x.destination),['notion','sheets']);
});

test('Sheets writes the exact A107:S107 range even when append would infer an O-column table',async()=>{
  const f=await fixture();f.values.push(...Array.from({length:101},()=>[]));
  const before=structuredClone(f.values),plan=planSheets(f.r,f.snapshot()),google=f.publisher.google;
  f.publisher.google=async(method,path,body)=>{
    if(method==='GET') return google(method,path,body);
    f.writes.push({destination:'sheets',method,path,body});
    if(method==='POST') f.values.push(...body.values.map(cells=>[...Array(14).fill(''),...cells]));
    else {
      assert.equal(method,'PUT');assert.equal(decodeURIComponent(path),'/values/Prospects!A107:S107?valueInputOption=RAW');
      f.values.splice(106,body.values.length,...structuredClone(body.values));
    }
    return {};
  };
  const result=await f.store.publish(f.r.date,'sheets');
  assert.equal(result.readback_verified,true);assert.deepEqual(f.values[106],plan.sheet_rows[0]);
  assert.deepEqual(f.values.slice(0,106),before);
  assert.deepEqual((await f.store.get(f.r.date)).delivery.sheets.plan,plan);
  assert.equal(JSON.stringify(f.writes[0].body),plan.body_json);
  assert.equal(f.db.values.get(`${ROOT}/runs/${f.r.date}`).publication_claimed.sheets,plan.request_digest);
  assert.equal((await f.store.publish(f.r.date,'sheets')).readback_verified,true);assert.equal(f.writes.length,1);
});

for(const cell of [{userEnteredValue:{formulaValue:'=""'}},{dataValidation:{condition:{type:'TEXT_EQ',values:[{userEnteredValue:'allowed'}]}}}])
  test('Sheets refuses target structure changed after planning while displayed CRM stays unchanged: '+Object.keys(cell)[0],async()=>{
    const f=await fixture(),google=f.publisher.google,before=structuredClone(f.values);let reads=0;
    f.publisher.google=async(method,path,body)=>{
      if(method==='GET' && ++reads===2) return {sheets:[{data:[{rowData:[{values:[cell]}]}]}]};
      return google(method,path,body);
    };
    await assert.rejects(f.store.publish(f.r.date,'sheets'),/publication_target_cells_not_plain_empty/);
    assert.deepEqual(f.values,before);assert.equal(f.writes.length,0);
    assert.ok(f.db.values.get(`${ROOT}/runs/${f.r.date}`).publication_claimed.sheets);
  });

test('ten agent-approved discoveries publish under the same durable claims with no three-row truncation',async()=>{
  const f=await fixture();
  const candidates=Array.from({length:10},(_,i)=>({...candidate(),organization:`Synthetic operator ${i}`,site:`Site ${i}`}));
  await f.store.put(row(candidates));
  const notion=await f.store.publish(f.r.date);
  assert.equal(notion.readback_verified,true);
  const saved=await f.store.get(f.r.date);saved.delivery.notion.state='acknowledged';await f.store.put(saved);
  const sheets=await f.store.publish(f.r.date);
  assert.equal(sheets.readback_verified,true);
  assert.equal(f.values.length,15);
  assert.deepEqual(f.values.slice(5).map(r=>r[0]),Array.from({length:10},(_,i)=>`BP-${String(i+1).padStart(6,'0')}`));
  assert.equal(f.writes.filter(x=>x.destination==='sheets').length,1);
  assert.ok(f.pages[0].body.children.some(b=>b.paragraph.rich_text[0].text.content.includes('Synthetic operator 9')));
  assert.ok(f.values.slice(5).every(r=>r[10]==='Research'&&r[16]==='Unverified'));
  const oversized=Array.from({length:800},(_,i)=>({...candidate(),site:`Distinct site ${i}`,proposed_next_action:'x'.repeat(2000)}));
  assert.throws(()=>planSheets(row(oversized),f.snapshot()),/publication_candidates_invalid/);
});

test('long QA brief is published losslessly in bounded Notion blocks and exact readback rejects truncation',async()=>{
  const f=await fixture(),summary='Supported evidence https://plant.example/tasks; interest remains unknown. '.repeat(120).slice(0,7301);
  const r=row([candidate()],summary);await f.store.put(r);
  const plan=planNotion(r);
  assert.ok(summary.length===7301 && plan.paragraphs.length<90);
  assert.ok(plan.paragraphs.every(text=>text.length<=1800));
  assert.ok(plan.paragraphs.slice(1).join('').includes(summary));
  assert.equal((await f.store.publish(r.date)).readback_verified,true);
  const written=f.pages[0].body.children.map(b=>b.paragraph.rich_text[0].text.content);
  assert.deepEqual(written,plan.paragraphs);
  assert.ok(written.slice(1).join('').includes(summary));
  f.pages[0].body.children[2].paragraph.rich_text[0].text.content+='truncated';
  await assert.rejects(f.publisher.reconcile(r,'notion',plan),/publication_readback_conflict/);
  assert.equal(f.writes.filter(x=>x.destination==='notion').length,1);
});

test('terminal collection receipt is immutable and publication authority is checked again before claim',async()=>{
  const f=await fixture(),r=await f.store.get(f.r.date);
  r.qa.terminal_collection_recovery={native_receipt:{synthetic:true},previous_qa:{cancel_attempted:true},
    workflow_authority:structuredClone(f.db.values.get(ROOT).workflow)};
  await f.store.put(r);
  f.publisher.reconcile=async()=>{
    f.db.values.get(ROOT).workflow.publication_authority_reference='different-approved-scope';
    return null;
  };
  await assert.rejects(f.store.publish(r.date),/publication_authority_changed/);
  assert.equal(f.writes.length,0);
  assert.equal(f.db.values.get(`${ROOT}/runs/${r.date}`).publication_claimed?.notion,undefined);
  const changed=structuredClone(r);changed.qa.terminal_collection_recovery.previous_qa.cancel_attempted=false;
  await assert.rejects(f.store.put(changed),/qa_terminal_collection_already_bound/);
});

for(const destination of ['notion','sheets']) test(`lost ${destination} reply restarts with GET-only reconciliation`,async()=>{
  const f=await fixture();
  if(destination==='sheets') {
    await f.store.publish(f.r.date);const r=await f.store.get(f.r.date);r.delivery.notion.state='acknowledged';await f.store.put(r);
  }
  f.faults.lost=true;
  await assert.rejects(f.store.publish(f.r.date),/publication_attempt_unresolved/);
  await f.store.release();
  const restarted=new Store(f.db,()=>f.time.now,'two',null,f.publisher);await restarted.acquire();
  const receipt=await restarted.publish(f.r.date);
  assert.equal(receipt.destination,destination);assert.equal(receipt.readback_verified,true);
  assert.equal(f.writes.filter(x=>x.destination===destination).length,1);
});

test('uncertain unobserved write is never repeated; lease replay performs no service mutations',async()=>{
  const f=await fixture();f.db.replay=true;
  f.publisher.write=async()=>{throw new Error('private timeout');};
  await assert.rejects(f.store.publish(f.r.date),/publication_attempt_unresolved/);
  f.publisher.write=async()=>assert.fail('must not repeat');
  assert.equal(await f.store.publish(f.r.date),null);assert.equal(f.writes.length,0);
});

test('disabled workflow and stale owner cannot publish or admit QA',async()=>{
  const f=await fixture();f.db.values.get(ROOT).workflow.enabled=false;
  await assert.rejects(f.store.publish(f.r.date),/workflow_authority_missing/);assert.equal(f.writes.length,0);
  f.time.now+=180001;
  const successor=new Store(f.db,()=>f.time.now,'two');await successor.acquire();
  await assert.rejects(f.store.publish(f.r.date),/lease_lost/);
});

test('QA claim is one use, durable, deadline-fenced and found while disabled',async()=>{
  const f=await fixture(),deadline=f.time.now+1000;
  const r={...f.r,state:'awaiting_review',qa:{state:'qa_input_unresolved',request_digest:'b'.repeat(64),deadline_ms:deadline}};
  await f.store.put(r);
  await assert.rejects(f.store.qaCheck(r.date,r.qa.request_digest,deadline+1),/not_admitted/);
  await f.store.qaCheck(r.date,r.qa.request_digest,deadline);
  await assert.rejects(f.store.qaCheck(r.date,r.qa.request_digest,deadline),/not_admitted/);
  f.db.values.get(ROOT).enabled=false;assert.equal(await f.store.activeQA(),r.date);
  assert.equal(f.db.values.get(`${ROOT}/runs/${r.date}`).qa_request_claimed,true);
});

test('QA deadline expiry refuses before claiming or creating an event',async()=>{
  const f=await fixture(),deadline=f.time.now;
  const r={...f.r,state:'awaiting_review',qa:{state:'qa_input_unresolved',request_digest:'b'.repeat(64),deadline_ms:deadline}};
  await f.store.put(r);await assert.rejects(f.store.qaCheck(r.date,r.qa.request_digest,deadline),/not_admitted/);
  assert.equal(f.db.values.get(`${ROOT}/runs/${r.date}`).qa_request_claimed,false);
});

test('payload, plan, QA and destination drift refuse before external writes',async()=>{
  const f=await fixture(),plan=planNotion(f.r);
  const altered=structuredClone(plan);altered.body_json+=' ';assert.throws(()=>f.publisher.validate(f.r,'notion',altered),/plan_binding/);
  const r=structuredClone(f.r);r.delivery.notion.payload.summary='altered';
  assert.throws(()=>planNotion(r),/qa_binding/);
  r.delivery.notion.payload=r.delivery.notion.payload_json=undefined;assert.throws(()=>planNotion(r));
  const wrong=structuredClone(f.r);wrong.delivery.notion.payload.parent_id='other';
  wrong.delivery.notion.payload_json=JSON.stringify(wrong.delivery.notion.payload);
  wrong.delivery.notion.payload_digest=sha(wrong.delivery.notion.payload_json);
  assert.throws(()=>planNotion(wrong),/destination_invalid/);
  wrong.qa.state='qa_running';assert.throws(()=>planNotion(wrong),/qa_binding/);
  assert.equal(f.writes.length,0);
});

test('CRM duplicates, occupied/formula targets and changed snapshots block append',async()=>{
  const f=await fixture();
  const existing=['BP-000012','Example Plant','Facility / site','North','','','','','','https://plant.example/tasks','','','','','Depositing'];
  f.values.push(existing);assert.throws(()=>planSheets(f.r,f.snapshot()),/duplicate_changed/);f.values.pop();
  f.publisher.google=async()=>({sheets:[{data:[{rowData:[{values:[{dataValidation:{condition:{type:'ONE_OF_LIST'}}}]}]}]}]});
  await assert.rejects(f.publisher.prepare(f.r,'sheets'),/not_plain_empty/);
  const plan=planSheets(f.r,f.snapshot());f.values.push(['BP-000020','Other']);
  await assert.rejects(f.publisher.write(f.r,'sheets',plan),/crm_changed/);assert.equal(f.writes.length,0);
});

test('readback conflict and incomplete Notion scans never record a receipt',async()=>{
  const f=await fixture();await f.store.publish(f.r.date);
  f.pages[0].body.children[0].paragraph.rich_text[0].text.content='corrupt';
  await assert.rejects(f.store.publish(f.r.date),/readback_conflict/);
  let cursor=0;
  f.publisher.notion=async()=>({has_more:true,next_cursor:'cursor-'+(++cursor),results:[]});
  await assert.rejects(f.publisher.reconcile(f.r,'notion',planNotion(f.r)),/scan_incomplete/);
  assert.equal(cursor,100);
});

for (const column of [1,3,9,14]) for (const missing of ['', '  ', undefined, 7]) {
  test(`incomplete CRM field ${column} (${String(missing)}) blocks a post-QA publication plan`,async()=>{
    const f=await fixture();
    const existing=['BP-000012','Example Plant','Facility / site','North','','','','','','https://plant.example/tasks','','','','','Depositing'];
    existing[column]=missing;f.values.push(existing);
    assert.throws(()=>planSheets(f.r,f.snapshot()),/crm_identity_incomplete/);
    await assert.rejects(f.publisher.prepare(f.r,'sheets'),/crm_identity_incomplete/);
    assert.equal(f.writes.length,0);
    assert.equal((await f.store.get(f.r.date)).delivery.sheets.plan,undefined);
  });
}

test('a truncated CRM identity also blocks verified publication readback',async()=>{
  const f=await fixture(),plan=planSheets(f.r,f.snapshot());
  await f.store.publish(f.r.date);
  const r=await f.store.get(f.r.date);r.delivery.notion.state='acknowledged';await f.store.put(r);
  await f.store.publish(f.r.date);
  f.values.push(['BP-000012','Other']);
  await assert.rejects(f.publisher.reconcile(f.r,'sheets',plan),/crm_identity_incomplete/);
  assert.equal(f.writes.filter(w=>w.destination==='sheets').length,1);
});

function pagedParent(f,count) {
  const original=f.publisher.notion,parentReads=[];
  f.publisher.notion=async(method,path,...args)=>{
    if(method==='GET'&&path.startsWith(`/blocks/${NOTION}/children`)) {
      const url=new URL(path,'https://api.notion.com');
      const offset=Number(url.searchParams.get('start_cursor')||0);
      const blocks=[...Array.from({length:count},(_,i)=>({id:'old-'+i,type:'paragraph'})),
        ...f.pages.map(p=>({id:p.id,type:'child_page',child_page:{title:p.body.properties.title.title[0].text.content}}))];
      parentReads.push(offset);
      return {results:blocks.slice(offset,offset+100),has_more:offset+100<blocks.length,next_cursor:String(offset+100)};
    }
    return original(method,path,...args);
  };
  return parentReads;
}

test('publication scans beyond 200 Notion children and verifies a report on the third page',async()=>{
  const f=await fixture(),reads=pagedParent(f,250);
  const receipt=await f.store.publish(f.r.date);
  assert.equal(receipt.readback_verified,true);assert.equal(receipt.reference,'notion:page-new');
  assert.deepEqual(reads,[0,100,200,0,100,200]);
  assert.equal(f.writes.length,1);
});

test('a lost Notion write reply reconciles a report beyond 200 children without another POST',async()=>{
  const f=await fixture();pagedParent(f,250);f.faults.lost=true;
  await assert.rejects(f.store.publish(f.r.date),/publication_attempt_unresolved/);
  const receipt=await f.store.publish(f.r.date);
  assert.equal(receipt.readback_verified,true);assert.equal(f.writes.length,1);
});

test('Notion duplicate reports on later pages refuse a readback receipt',async()=>{
  const f=await fixture();await f.store.publish(f.r.date);
  f.pages.push({...f.pages[0],id:'duplicate'});pagedParent(f,250);
  await assert.rejects(f.publisher.reconcile(f.r,'notion',planNotion(f.r)),/duplicate_pages/);
});

test('Notion cursor loops fail immediately without treating an incomplete scan as absence',async()=>{
  const f=await fixture();let reads=0;
  f.publisher.notion=async()=>{reads++;return {has_more:true,next_cursor:'same',results:[]};};
  await assert.rejects(f.publisher.reconcile(f.r,'notion',planNotion(f.r)),/pagination_invalid/);
  assert.equal(reads,2);assert.equal(f.writes.length,0);
});

test('Notion read budget covers both pagination and exact report verification',async()=>{
  const f=await fixture();let now=0;f.publisher.clock=()=>now;
  const original=f.publisher.notion;let reads=0;
  f.publisher.notion=async(method,path,body,options)=>{
    assert.ok(options.timeoutMs<=12000);reads++;
    if(reads===1){now=24000;return {has_more:true,next_cursor:'second',results:[]};}
    assert.equal(options.timeoutMs,1000);now=25000;
    return original(method,path,body);
  };
  await assert.rejects(f.publisher.reconcile(f.r,'notion',planNotion(f.r)),/scan_incomplete/);
  assert.equal(reads,2);assert.equal(f.writes.length,0);
});

test('no accepted candidates yields no Sheets write and a verified empty receipt',async()=>{
  const f=await fixture(),r=row([]);const plan=planSheets(r,f.snapshot());
  await f.publisher.write(r,'sheets',plan);
  assert.equal((await f.publisher.reconcile(r,'sheets',plan)).readback_verified,true);
  assert.equal(f.writes.length,0);
});

const canon=x=>Array.isArray(x)?x.map(canon):x&&typeof x==='object'?Object.fromEntries(Object.keys(x).sort().map(k=>[k,canon(x[k])])):x;
const valueDigest=x=>sha(JSON.stringify(canon(x)));
async function agentFixture(summary='Full canonical supported research; interest unknown.') {
  const f=await fixture(row([candidate()],summary));
  const r=await f.store.get(f.r.date),control=f.db.values.get(ROOT);
  r.publication_profile='agent-owned-v1';r.session_id='saved-research-session';
  const input=JSON.stringify({type:'agent.session.input.message',input:[]});
  await f.store.filePut(r.date+'-publication-input.json',Buffer.from(input+'\n').toString('base64'));
  r.publication={profile:'agent-owned-v1',session_id:r.session_id,state:'input_unresolved',request_digest:sha(input),
    input_file:r.date+'-publication-input.json',idempotency_key:r.run_key+':publication',deadline_ms:f.time.now+60000,
    baseline_turn_ids:['research-turn','qa-turn'],authority_reference:control.workflow.publication_authority_reference,
    workflow_authority:structuredClone(control.workflow)};
  await f.store.put(r);await f.store.publicationInputCheck(r.date,r.publication.request_digest,r.publication.deadline_ms);
  r.publication.state='running';r.publication.turn_id='publication-turn';await f.store.put(r);f.r=r;
  let next=0;
  f.action=async(name,args={},mutate=null)=>{
    const current=await f.store.get(r.date),action={turn_id:'publication-turn',call_id:'publication-call-'+(++next),name,arguments:args};
    current.application_tool_calls??={};current.application_tool_calls[action.call_id]={request:action,
      request_json:JSON.stringify(canon(action)),request_digest:valueDigest(action),phase:'publication',attempted:true};
    if(mutate) mutate(current,action);
    await f.store.put(current);
    return f.store.publicationAgentTool(r.date,action,r.publication.request_digest);
  };
  return f;
}

async function stoppedTerminalSheetsFixture() {
  const f=await agentFixture(),choice={destination:'sheets',strategy:'concise',summary:'Original display request'};
  const outcome=await f.action('blueprint_publish_research',choice),r=await f.store.get(f.r.date);
  const call=Object.values(r.application_tool_calls).at(-1),id=call.request.call_id;
  const event={type:'agent.session.input.tool_result',turn_id:r.publication.turn_id,call_id:id,success:false,output:JSON.stringify(outcome)};
  const raw=JSON.stringify(event)+'\n',filename=r.date+'-tool-'+id+'.json';
  await f.store.filePut(filename,Buffer.from(raw).toString('base64'));
  Object.assign(call,{result_file:filename,result_sha256:sha(raw),result_digest:valueDigest(event),success:false,result_acknowledged:true});
  r.turn_id='research-turn';r.packet.candidates[0].candidate_key='candidate-one';r.packet_digest=valueDigest(r.packet);
  r.review.lead_verification=verification(r.packet.candidates);
  const qa={packet_digest:r.packet_digest,source_support_verified:true,accepted_keys:['candidate-one'],
    checks:r.review.lead_verification.results.map(result=>({candidate_key:result.candidate_key,lead_verification:result.assessment}))},qaRaw=JSON.stringify(qa)+'\n';
  r.qa={state:'validated',turn_id:'qa-turn',artifact_file:r.date+'-qa.json',artifact_digest:sha(qaRaw)};
  await f.store.filePut(r.qa.artifact_file,Buffer.from(qaRaw).toString('base64'));
  Object.assign(r.review,{packet_digest:r.packet_digest,accepted_keys:['candidate-one'],qa_artifact_digest:r.qa.artifact_digest,
    reviewer_reference:`agent-turn:${r.session_id}:qa-turn`});r.qa.decision=structuredClone(r.review);
  r.delivery.sheets.payload.candidates=structuredClone(r.packet.candidates);
  r.delivery.sheets.payload_json=JSON.stringify(r.delivery.sheets.payload);r.delivery.sheets.payload_digest=sha(r.delivery.sheets.payload_json);
  r.delivery.notion.state='acknowledged';r.delivery.notion.receipt={destination:'notion',key:r.run_key+':notion',
    payload_digest:r.delivery.notion.payload_digest,readback_verified:true,reference:'notion:verified-fixture-page'};
  Object.assign(r.publication,{turn_status:'completed',state:'agent_finished_without_complete_receipts',
    completed_at:Math.floor(f.time.now/1000),evidence_file:r.date+'-publication-evidence.json',evidence_digest:valueDigest([])});
  await f.store.filePut(r.publication.evidence_file,Buffer.from('[]\n').toString('base64'));
  await f.store.put(r);
  const control=f.db.values.get(ROOT);control.enabled=false;control.config={enabled:false};f.store.schedulerStopped=true;
  f.time.now+=120000;
  f.request={op:'recover_terminal_sheets',day:r.date,source_row_blob:f.db.values.get(`${ROOT}/runs/${r.date}`).blob,
    payload_digest:r.delivery.sheets.payload_digest,rejected_call_id:id,publication_authority_reference:r.publication.authority_reference};
  f.original=structuredClone(r);return f;
}

test('stopped terminal Sheets recovery appends once and retains exact agent terminal and rejection history',async()=>{
  const f=await stoppedTerminalSheetsFixture();
  const result=await f.store.dispatch(f.request);assert.equal(result.state,'acknowledged');assert.equal(f.writes.length,1);
  assert.equal(f.pages.length,0);assert.deepEqual((await f.store.get(f.r.date)).publication,f.original.publication);
  assert.deepEqual((await f.store.get(f.r.date)).application_tool_calls,f.original.application_tool_calls);
  assert.equal((await f.store.get(f.r.date)).state,'reviewed');
  assert.equal(f.db.values.get(ROOT).enabled,false);assert.equal(f.db.values.get(ROOT).config.enabled,false);
  const recovery=f.db.values.get(result.recovery_ref);assert.equal(recovery.schema_version,'blueprint.terminal-sheets-recovery.v1');
  assert.ok(Date.parse(recovery.readback_at)>f.original.publication.deadline_ms);
  assert.equal((await f.store.dispatch(f.request)).state,'acknowledged');assert.equal(f.writes.length,1);
});

for(const change of ['root','config','scheduler','authority','source','payload','late','running','canary','qa','rejection','lease','destination'])
  test('stopped terminal recovery refuses '+change+' without any append',async()=>{
    const f=await stoppedTerminalSheetsFixture();
    if(change==='root') f.db.values.get(ROOT).enabled=true;
    if(change==='config') f.db.values.get(ROOT).config.enabled=true;
    if(change==='scheduler') f.store.schedulerStopped=false;
    if(change==='authority') f.db.values.get(ROOT).workflow.publication_authority_reference='different';
    if(change==='payload') f.request.payload_digest='a'.repeat(64);
    if(change==='lease') f.time.now+=180000;
    if(change==='destination') f.request.destination='notion';
    if(['source','late','running','canary','qa','rejection'].includes(change)) {
      const r=await f.store.get(f.r.date);
      if(change==='source') r.review.summary='changed';
      if(change==='late') r.publication.completed_at=Math.ceil(r.publication.deadline_ms/1000)+1;
      if(change==='running') r.publication.turn_status='in_progress';
      if(change==='canary') r.canary={baseline:{}};
      if(change==='qa') r.review.accepted_keys=['not-accepted'];
      if(change==='rejection') r.application_tool_calls[f.request.rejected_call_id].request.arguments.strategy='full';
      await f.store.put(r);
      if(change!=='source') f.request.source_row_blob=f.db.values.get(`${ROOT}/runs/${r.date}`).blob;
    }
    await assert.rejects(f.store.dispatch(f.request),/terminal_sheets_recovery_|firestore_lease_lost/);
    assert.equal(f.writes.length,0);
  });

test('an unknown terminal Sheets append is reconciled only; absent readback never licenses another append',async()=>{
  const f=await stoppedTerminalSheetsFixture();f.faults.lost=true;
  await assert.rejects(f.store.dispatch(f.request),/publication_attempt_unresolved/);
  assert.equal(f.writes.length,1);
  assert.equal((await f.store.dispatch(f.request)).state,'acknowledged');assert.equal(f.writes.length,1);
  const g=await stoppedTerminalSheetsFixture();
  g.publisher.write=async()=>{g.writes.push({destination:'sheets'});throw new Error('unknown acceptance');};
  await assert.rejects(g.store.dispatch(g.request),/publication_attempt_unresolved/);
  assert.equal((await g.store.dispatch(g.request)).state,'readback_pending');assert.equal(g.writes.length,1);
});

for(const change of ['root','lease','source']) test('terminal recovery rechecks '+change+' after claim and before append',async()=>{
  const f=await stoppedTerminalSheetsFixture(),original=f.publisher.reconcile.bind(f.publisher);
  f.publisher.reconcile=async(...args)=>{
    const result=await original(...args);
    if(change==='root') f.db.values.get(ROOT).enabled=true;
    if(change==='lease') f.time.now+=180000;
    if(change==='source') f.db.values.get(`${ROOT}/runs/${f.r.date}`).blob='f'.repeat(64);
    return result;
  };
  await assert.rejects(f.store.dispatch(f.request),/terminal_sheets_recovery_|firestore_lease_lost|publication_(?:attempt_not_admitted|authority_or_plan_changed)/);
  assert.equal(f.writes.length,0);
});

test('terminal recovery rechecks the stopped flag after the final CRM read before append',async()=>{
  const f=await stoppedTerminalSheetsFixture(),read=f.publisher.crmReader;let reads=0;
  f.publisher.crmReader=async()=>{const result=await read();if(++reads===3) f.db.values.get(ROOT).enabled=true;return result;};
  await assert.rejects(f.store.dispatch(f.request),/terminal_sheets_recovery_not_admitted/);
  assert.equal(f.writes.length,0);assert.ok(f.db.values.get(`${ROOT}/runs/${f.r.date}`).publication_claimed.sheets);
});

test('terminal recovery refuses a source change during plan preparation without overwriting it',async()=>{
  const f=await stoppedTerminalSheetsFixture(),prepare=f.publisher.prepare.bind(f.publisher);
  f.publisher.prepare=async(...args)=>{
    const plan=await prepare(...args),changed=await f.store.get(f.r.date);
    changed.review.summary='Concurrent source owner change';await f.store.put(changed);return plan;
  };
  await assert.rejects(f.store.dispatch(f.request),/terminal_sheets_recovery_source_changed/);
  assert.equal(f.writes.length,0);assert.equal((await f.store.get(f.r.date)).review.summary,'Concurrent source owner change');
  assert.equal((await f.store.get(f.r.date)).delivery.sheets.plan,undefined);
});

test('terminal recovery refuses a source change before receipt commit without overwriting or appending twice',async()=>{
  const f=await stoppedTerminalSheetsFixture(),put=f.store.blobPut.bind(f.store);let changed=false;
  f.store.blobPut=async encoded=>{
    const hash=await put(encoded),value=JSON.parse(Buffer.from(encoded,'base64').toString('utf8'));
    if(!changed && value.delivery?.sheets?.state==='acknowledged') {
      changed=true;const latest=await f.store.get(f.r.date);latest.review.summary='Concurrent receipt source change';await f.store.put(latest);
    }
    return hash;
  };
  await assert.rejects(f.store.dispatch(f.request),/terminal_sheets_recovery_source_changed/);
  assert.equal(f.writes.length,1);assert.equal((await f.store.get(f.r.date)).review.summary,'Concurrent receipt source change');
  assert.equal((await f.store.get(f.r.date)).delivery.sheets.state,'pending');
});

test('agent-owned publication requires explicit choice; inspect returns full approved source without provider writes',async()=>{
  const f=await agentFixture('Retained complete supported finding '.repeat(5000));
  await assert.rejects(f.store.publish(f.r.date),/publication_agent_choice_required/);
  const out=await f.action('blueprint_inspect_publication');assert.equal(out.success,true);
  assert.equal(out.output.destinations.notion.payload.summary,f.r.delivery.notion.payload.summary);
  assert.equal(out.output.transport_limits.notion.blocks_per_request,90);
  assert.deepEqual(out.output.presentation_rules.sheets.strategies,['full']);
  assert.equal(out.output.presentation_rules.sheets.summary,'must_be_absent');
  assert.deepEqual(f.writes,[]);assert.equal(out.output.destinations.sheets.claimed,false);
});

test('agent inspects unresolved raw discovery and receives actionable evidence repair without a publication claim',async()=>{
  const f=await agentFixture(),r=await f.store.get(f.r.date);delete r.review.lead_verification;await f.store.put(r);
  const inspected=await f.action('blueprint_inspect_publication');
  assert.equal(inspected.success,true);assert.deepEqual(inspected.output.packet,r.packet);
  assert.equal(inspected.output.verification_eligibility.sheets.eligible,false);
  assert.equal(inspected.output.lead_verification,null);
  const result=await f.action('blueprint_publish_research',{destination:'sheets',strategy:'full'});
  assert.equal(result.success,false);assert.equal(result.error.code,'publication_lead_verification_required');
  assert.equal(result.error.status,'recoverable_issue');assert.ok(result.error.allowed_repair);
  assert.ok(result.error.evidence_issues.length);assert.equal(f.writes.length,0);
  assert.deepEqual(f.db.values.get(`${ROOT}/runs/${r.date}`).publication_claimed,{});
});

test('stopped terminal Sheets recovery refuses a missing current assessment before fresh admission',async()=>{
  const f=await stoppedTerminalSheetsFixture(),r=await f.store.get(f.r.date);
  delete r.review.lead_verification;delete r.qa.decision.lead_verification;
  await f.store.put(r);f.request.source_row_blob=f.db.values.get(`${ROOT}/runs/${r.date}`).blob;
  await assert.rejects(f.store.dispatch(f.request),/publication_lead_verification_required/);
  assert.equal(f.writes.length,0);assert.deepEqual(f.db.values.get(`${ROOT}/runs/${r.date}`).publication_claimed,{});
});

test('invalid Sheets presentation returns actionable fields without claims or writes, then full succeeds',async()=>{
  const f=await agentFixture(),original=structuredClone(f.r.delivery.sheets);
  for(const choice of [{destination:'sheets',strategy:'concise',summary:'Retain this original request'},
    {destination:'sheets',strategy:'concise'},{destination:'sheets',strategy:'full',summary:null}]) {
    const out=await f.action('blueprint_publish_research',choice);
    assert.equal(out.success,false);assert.equal(out.error.code,'publication_agent_tool_arguments_invalid');
    assert.equal(out.error.status,'recoverable_issue');assert.ok(out.error.allowed_repair);
    const issue=out.error.issues.find(i=>i.path===(choice.strategy==='concise'?'/strategy':'/summary'));
    assert.ok(issue);assert.ok(issue.allowed_repair);
    const saved=await f.store.get(f.r.date);
    assert.deepEqual(saved.delivery.sheets,original);assert.equal(f.writes.length,0);
    assert.deepEqual(Object.values(saved.application_tool_calls).at(-1).request.arguments,choice);
    assert.equal(f.db.values.get(`${ROOT}/runs/${f.r.date}`).publication_claimed?.sheets,undefined);
  }
  const out=await f.action('blueprint_publish_research',{destination:'sheets',strategy:'full'});
  assert.equal(out.receipt.readback_verified,true);assert.equal(f.writes.length,1);
});

test('Notion concise feedback identifies missing blank and oversized summary without changing source',async()=>{
  const f=await agentFixture(),original=structuredClone(f.r.delivery.notion);
  for(const summary of [undefined,'  ','😀'.repeat(500001)]) {
    const out=await f.action('blueprint_publish_research',{destination:'notion',strategy:'concise',...(summary===undefined?{}:{summary})});
    assert.equal(out.error.code,'publication_agent_tool_arguments_invalid');
    assert.equal(out.error.issues[0].path,'/summary');
    assert.equal(out.error.issues[0].expectations.max_utf8_bytes,2000000);
    assert.deepEqual((await f.store.get(f.r.date)).delivery.notion,original);
    assert.equal(f.writes.length,0);
  }
});

test('agent chooses Sheets first and no outer helper creates a Notion page',async()=>{
  const f=await agentFixture();
  const out=await f.action('blueprint_publish_research',{destination:'sheets',strategy:'full'});
  assert.equal(out.success,true);assert.equal(out.receipt.destination,'sheets');assert.equal(out.receipt.readback_verified,true);
  assert.equal(f.writes.length,1);assert.equal(f.writes[0].destination,'sheets');assert.equal(f.pages.length,0);
});

test('agent concise presentation is immutable derived evidence and never replaces full QA or canonical payload',async()=>{
  const f=await agentFixture('Retained complete original result '.repeat(10000)),original=structuredClone(f.r.delivery.notion);
  const choice={destination:'notion',strategy:'concise',summary:'Supported finding; buying interest remains unknown.'};
  const result=await f.action('blueprint_publish_research',choice);
  assert.equal(result.success,true);assert.equal(result.receipt.payload_digest,original.payload_digest);
  const current=await f.store.get(f.r.date),d=current.delivery.notion;
  assert.deepEqual(d.payload,original.payload);assert.equal(d.payload_json,original.payload_json);
  assert.equal(d.presentation.source_payload_digest,original.payload_digest);
  assert.equal(d.plan.presentation_digest,d.presentation.decision_digest);
  assert.ok(f.pages[0].body.children.map(b=>b.paragraph.rich_text[0].text.content).join('').includes(choice.summary));
  assert.ok(!f.pages[0].body.children.map(b=>b.paragraph.rich_text[0].text.content).join('').includes(original.payload.summary));
  const writes=f.writes.length;
  assert.equal((await f.action('blueprint_publish_research',choice)).receipt.readback_verified,true);
  const changed=await f.action('blueprint_publish_research',{...choice,summary:'Changed presentation'});
  assert.equal(changed.success,false);assert.equal(changed.error.code,'publication_presentation_already_bound');assert.equal(f.writes.length,writes);
  current.delivery.notion.presentation.summary='Tampered';await assert.rejects(f.store.put(current),/publication_delivery_already_bound/);
});

test('agent sees unknown acceptance and reconciles exact full choice without duplicate page',async()=>{
  const f=await agentFixture();f.faults.lost=true;
  const choice={destination:'notion',strategy:'full'},failed=await f.action('blueprint_publish_research',choice);
  assert.equal(failed.success,false);assert.equal(failed.error.code,'publication_attempt_unresolved');
  assert.match(failed.error.guidance,/GET-only/);assert.equal(f.pages.length,1);
  assert.equal((await f.action('blueprint_publish_research',choice)).receipt.readback_verified,true);
  assert.equal(f.writes.length,1);
});

test('agent full choice uses bounded transport batches and unobserved append is never resent',async()=>{
  const f=await agentFixture('Full retained evidence '.repeat(20000)),choice={destination:'notion',strategy:'full'};
  assert.equal((await f.action('blueprint_publish_research',choice)).output.status,'pending');
  f.faults.unobservedAppend=true;assert.equal((await f.action('blueprint_publish_research',choice)).success,false);
  const count=f.writes.length;
  const pending=await f.action('blueprint_publish_research',choice);
  assert.equal(pending.success,true);assert.equal(pending.output.status,'pending');assert.equal(f.writes.length,count);
  const repair=await f.action('blueprint_publish_research',{destination:'notion',strategy:'concise',summary:'A shorter display'});
  assert.equal(repair.error.code,'publication_presentation_already_bound');assert.equal(f.writes.length,count);
});

test('agent tool refuses stale or unbound turn/action and keeps input admission one-use',async()=>{
  const f=await agentFixture();
  await assert.rejects(f.store.publicationInputCheck(f.r.date,f.r.publication.request_digest,f.r.publication.deadline_ms),/publication_agent_input_not_admitted/);
  const stale=await f.action('blueprint_publish_research',{destination:'notion',strategy:'full'},(r,a)=>{a.turn_id='wrong-turn';r.application_tool_calls[a.call_id].request_json=JSON.stringify(canon(a));r.application_tool_calls[a.call_id].request_digest=valueDigest(a);});
  assert.equal(stale.error.code,'publication_agent_tool_not_admitted');assert.equal(f.writes.length,0);
  const missing=await f.store.publicationAgentTool(f.r.date,{turn_id:'publication-turn',call_id:'unpersisted',name:'blueprint_inspect_publication',arguments:{}},f.r.publication.request_digest);
  assert.equal(missing.error.code,'publication_agent_tool_not_admitted');
  f.time.now=f.r.publication.deadline_ms;
  assert.equal((await f.action('blueprint_publish_research',{destination:'notion',strategy:'full'})).error.code,'publication_agent_input_not_admitted');
  assert.equal(f.writes.length,0);
});

for(const mutation of ['disable','authority','lease','deadline']) test(`agent legacy transport rechecks ${mutation} after slow final hook and durable claim`,async()=>{
  const f=await agentFixture();f.publisher.beforeNotionStep=async()=>{
    const c=f.db.values.get(ROOT);
    if(mutation==='disable') c.workflow.enabled=false;
    if(mutation==='authority') c.workflow.publication_authority_reference='changed';
    if(mutation==='lease') c.lease.owner='successor';
    if(mutation==='deadline') f.time.now=f.r.publication.deadline_ms;
  };
  const result=await f.action('blueprint_publish_research',{destination:'notion',strategy:'full'});
  assert.equal(result.success,false);assert.equal(f.writes.length,0);
  assert.ok(f.db.values.get(`${ROOT}/runs/${f.r.date}`).publication_claimed.notion);
});

test('agent receives precise bounded provider feedback and private immutable raw error evidence',async()=>{
  const f=await agentFixture(),normal=f.publisher.notion;
  const raw=JSON.stringify({object:'error',status:400,code:'validation_error',message:'body.children must contain at most 100 items'});
  f.publisher.notion=async(method,...args)=>{
    if(method!=='POST') return normal(method,...args);
    const error=new Error('publication_notion_unavailable');error.provider_response=raw;
    error.provider_feedback={provider:'notion',http_status:400,code:'validation_error',message:JSON.parse(raw).message,response_digest:sha(raw)};
    throw error;
  };
  const result=await f.action('blueprint_publish_research',{destination:'notion',strategy:'full'});
  assert.equal(result.success,false);assert.equal(result.error.provider_feedback.http_status,400);
  assert.match(result.error.provider_feedback.message,/100 items/);
  assert.equal(result.error.provider_response_json,raw);
  assert.equal(Buffer.from(await f.store.fileGet(result.error.evidence_file),'base64').toString(),raw);
  await assert.rejects(f.store.filePut(result.error.evidence_file,Buffer.from('{}').toString('base64')),/artifact_identity_conflict/);
});

function rejectInitialNotion(f,{status=400,rawStatus=status,code='validation_error'}={}) {
  const normal=f.publisher.notion;let first=true;
  f.publisher.notion=async(method,path,body,...rest)=>{
    if(method!=='POST' || !first) return normal(method,path,body,...rest);
    first=false;
    const raw=JSON.stringify({object:'error',status:rawStatus,code,message:'body.children validation rejected before creation'});
    const error=new Error('publication_notion_unavailable');error.provider_response=raw;
    error.provider_feedback={provider:'notion',http_status:status,code,message:JSON.parse(raw).message,
      request_digest:sha(JSON.stringify(body)),request_bytes:Buffer.byteLength(JSON.stringify(body)),response_digest:sha(raw)};
    throw error;
  };
}

test('confirmed initial Notion400 lets the agent choose a concise new attempt while preserving old immutable claims',async()=>{
  const f=await agentFixture('Retained complete original '.repeat(1000));rejectInitialNotion(f);
  const rejected=await f.action('blueprint_publish_research',{destination:'notion',strategy:'full'});
  assert.equal(rejected.success,false);assert.equal(rejected.error.provider_feedback.http_status,400);assert.equal(f.pages.length,0);
  assert.equal(rejected.error.recovery_policy,'new_agent_presentation_after_verified_absence');
  const priorRow=await f.store.get(f.r.date),priorManifest=structuredClone(f.db.values.get(`${ROOT}/runs/${f.r.date}`));
  const repaired=await f.action('blueprint_publish_research',{destination:'notion',strategy:'concise',summary:'Supported shorter display; interest unknown.'});
  assert.equal(repaired.success,true);assert.equal(repaired.receipt.readback_verified,true);assert.equal(f.pages.length,1);
  const current=await f.store.get(f.r.date),manifest=f.db.values.get(`${ROOT}/runs/${f.r.date}`);
  assert.deepEqual(current.delivery.notion.payload,priorRow.delivery.notion.payload);
  assert.equal(current.delivery.notion.attempt_number,1);
  assert.deepEqual(manifest.publication_claimed,priorManifest.publication_claimed);
  assert.equal(manifest.publication_attempts.notion[0].claimed,priorRow.delivery.notion.plan.request_digest);
  const historical=JSON.parse(Buffer.from(await f.store.blobGet(manifest.publication_attempts.notion[0].source_row_blob),'base64').toString());
  assert.deepEqual(historical.delivery.notion.plan,priorRow.delivery.notion.plan);
  assert.equal(manifest.publication_attempts.notion[1].claimed,current.delivery.notion.plan.request_digest);
  assert.notEqual(manifest.publication_attempts.notion[1].claimed,manifest.publication_claimed.notion);
  const proof=JSON.parse((await f.store.snapshot(f.r.date)).publication_manifest.manifest_json);
  assert.equal(proof.publication_claimed.notion,current.delivery.notion.plan.request_digest);
  assert.equal(sha(proof.attempt_history.notion[0].source_row_json),proof.attempt_history.notion[0].source_row_blob);
  assert.equal(sha(proof.attempt_history.notion[0].response_json),proof.attempt_history.notion[0].rejection.response_digest);
});

for(const invalid of [{status:500},{status:400,rawStatus:403},{status:400,code:'rate_limited'}])
  test(`nondefinitive rejection ${JSON.stringify(invalid)} never admits a replacement claim`,async()=>{
    const f=await agentFixture();rejectInitialNotion(f,invalid);
    assert.equal((await f.action('blueprint_publish_research',{destination:'notion',strategy:'full'})).success,false);
    const result=await f.action('blueprint_publish_research',{destination:'notion',strategy:'concise',summary:'New display'});
    assert.equal(result.error.code,'publication_presentation_already_bound');assert.equal(f.pages.length,0);assert.equal(f.writes.length,0);
  });

test('definitive rejection still requires a complete absence scan before new attempt admission',async()=>{
  const f=await agentFixture();rejectInitialNotion(f);
  await f.action('blueprint_publish_research',{destination:'notion',strategy:'full'});
  const normal=f.publisher.notion;f.publisher.notion=async(method,path,...args)=>path.startsWith(`/blocks/${NOTION}/children`)
    ?{results:[],has_more:true,next_cursor:'loop'}:normal(method,path,...args);
  const result=await f.action('blueprint_publish_research',{destination:'notion',strategy:'concise',summary:'New display'});
  assert.equal(result.error.code,'publication_notion_pagination_invalid');assert.equal(f.writes.length,0);
  assert.equal(f.db.values.get(`${ROOT}/runs/${f.r.date}`).publication_attempts.notion,undefined);
});

test('a rejected append cannot rewind previously accepted report batches',async()=>{
  const f=await agentFixture('Full retained original evidence '.repeat(15000)),choice={destination:'notion',strategy:'full'};
  await f.action('blueprint_publish_research',choice);
  const normal=f.publisher.notion;f.publisher.notion=async(method,path,body,...rest)=>{
    if(method!=='PATCH') return normal(method,path,body,...rest);
    const raw=JSON.stringify({object:'error',status:400,code:'validation_error',message:'append rejected'});
    const error=new Error('publication_notion_unavailable');error.provider_response=raw;
    error.provider_feedback={provider:'notion',http_status:400,code:'validation_error',request_digest:sha(JSON.stringify(body)),response_digest:sha(raw)};
    throw error;
  };
  assert.equal((await f.action('blueprint_publish_research',choice)).success,false);
  const prefix=structuredClone(f.pages[0].body.children),writes=f.writes.length;
  const revised=await f.action('blueprint_publish_research',{destination:'notion',strategy:'concise',summary:'Revised display'});
  assert.equal(revised.error.code,'publication_presentation_already_bound');assert.deepEqual(f.pages[0].body.children,prefix);assert.equal(f.writes.length,writes);
});

test('agent action binding accepts exact Python escaped Unicode bytes without weakening semantic equality',async()=>{
  const f=await agentFixture(),choice={destination:'notion',strategy:'concise',summary:'机器人 supported café; 兴趣未知 😀'};
  const result=await f.action('blueprint_publish_research',choice,(r,a)=>{
    const intent=r.application_tool_calls[a.call_id];
    intent.request_json=JSON.stringify(canon(a)).replace(/[\u007f-\uffff]/g,c=>'\\u'+c.charCodeAt(0).toString(16).padStart(4,'0'));
    intent.request_digest=sha(intent.request_json);
  });
  assert.equal(result.success,true);assert.equal(result.receipt.readback_verified,true);
});

test('interruption after definitive rejection admission recovers the same immutable new attempt without consuming another',async()=>{
  const f=await agentFixture();rejectInitialNotion(f);await f.action('blueprint_publish_research',{destination:'notion',strategy:'full'});
  const normalPut=f.store.put.bind(f.store);let fail=true;
  f.store.put=async r=>{if(r.delivery.notion.attempt_number===1 && !r.delivery.notion.plan && fail){fail=false;throw new Error('interrupted before row projection');}return normalPut(r);};
  const choice={destination:'notion',strategy:'concise',summary:'Supported shorter display'};
  assert.equal((await f.action('blueprint_publish_research',choice)).success,false);
  const result=await f.action('blueprint_publish_research',choice);
  assert.equal(result.success,true);assert.equal(result.receipt.readback_verified,true);
  assert.deepEqual(Object.keys(f.db.values.get(`${ROOT}/runs/${f.r.date}`).publication_attempts.notion),['0','1']);
  assert.equal(f.pages.length,1);
});

test('provider JSON-string tool arguments execute losslessly while exact original intent bytes stay bound',async()=>{
  const f=await agentFixture();
  const out=await f.action('blueprint_publish_research',JSON.stringify({destination:'notion',strategy:'concise',summary:'完整 source retained; concise display'}));
  assert.equal(out.success,true);assert.equal(out.receipt.readback_verified,true);
  const r=await f.store.get(f.r.date);
  assert.equal(typeof r.application_tool_calls['publication-call-1'].request.arguments,'string');
  assert.equal(r.delivery.notion.presentation.summary,'完整 source retained; concise display');
});

test('publishes verified raw site/task with unknown robot match and a blank capability URL',()=>{
  const c=candidate();c.evidence=c.evidence.filter(e=>e.role!=='capability');c.potential_robot_match='unknown';
  const r=row([c]);const snapshot={sheet_id:SHEET,complete:true,values:[['CRM'],[],[],[],headers]};
  const plan=planSheets(r,snapshot);
  assert.equal(plan.sheet_rows[0][15],'');assert.equal(plan.sheet_rows[0][8],'unknown');
  const unsupported=row([{...c,potential_robot_match:'Invented unsupported robot'}]);
  assert.throws(()=>planSheets(unsupported,snapshot),/publication_candidate_scope_invalid/);
});

test('publication accepts a lossless schema_version alias but refuses null/conflicting versions and unknown expiry',()=>{
  const r=row(),result=r.review.lead_verification.results[0],a=result.assessment;
  result.version='blueprint.lead-verification-result.v2'; // alias acceptance is a v2 rule (independent review N2)
  a.schema_version=a.version;delete a.version;result.assessment_digest=verificationDigest(a);
  assert.equal(publicationVerification(r,'sheets').eligible,true);
  for(const value of [null,'other.version']) {
    a.version=value;result.assessment_digest=verificationDigest(a);
    assert.equal(publicationVerification(r,'sheets').eligible,false);
  }
  delete a.version;a.valid_until=null;result.assessment_digest=verificationDigest(a);
  assert.equal(publicationVerification(r,'sheets').eligible,false);
});

test('a v1 result keeps its exact original version rule; the alias rule applies to v2 only',()=>{
  const r=row(),result=r.review.lead_verification.results[0],a=result.assessment;
  result.version='blueprint.lead-verification-result.v1';
  a.schema_version='inert.conflicting.marker';result.assessment_digest=verificationDigest(a);
  assert.equal(publicationVerification(r,'sheets').eligible,true);
  delete a.schema_version;a.schema_version=a.version;delete a.version;result.assessment_digest=verificationDigest(a);
  assert.equal(publicationVerification(r,'sheets').eligible,false);
});

test('retains more than100 source-supported candidates through Sheets planning without a count quota',()=>{
  const candidates=Array.from({length:125},(_,i)=>({...candidate(),site:`Invented site ${i}`}));
  const r=row(candidates),snapshot={sheet_id:SHEET,complete:true,values:[['CRM'],[],[],[],headers]};
  const plan=planSheets(r,snapshot);assert.equal(plan.sheet_rows.length,125);
  assert.equal(plan.sheet_rows[124][3],'Invented site 124');
});

// Outreach-ready hypotheses (synthetic): one verified row plus one hypothesis bound by a v3 review.
const RESULT_V3='blueprint.lead-verification-result.v3', RULE='blueprint.outreach-ready-rule.v1.1';
// Design v1.1 (Blueprint-WebApp #855): exactly one question, chosen S, then M, then A, with task and site verbatim.
const TEMPLATES={S:'Is Depositing done at your South site, or somewhere else in the company?',
  M:'Which parts of Depositing at South still need people, and what has kept them from being automated?',
  A:'What has kept the remaining Depositing work at South from being automated so far?'};
const CHECKS=['manual_workflow','existing_automation','fit','interest'];
function rebind(r) {
  for(const d of Object.values(r.delivery)) {d.payload_json=JSON.stringify(d.payload);d.payload_digest=sha(d.payload_json);}
  return r;
}
function hypothesisRow({validUntil='2030-01-01T00:00:00Z'}={}) {
  const r=row([candidate(),{...candidate(),site:'South',candidate_key:'synthetic-h'}]);
  const [verified,hypothesis]=r.packet.candidates,[first,second]=r.review.lead_verification.results;
  r.packet.lead_verification_result_version=RESULT_V3;
  for(const result of [first,second]) Object.assign(result,{version:RESULT_V3,tier:'verified',eligible_for_outreach_ready:false,
    outreach_ready:{rule_version:RULE,proving_sources:[],open_checks:[],open_questions:[],blockers:[]}});
  second.assessment.claims.human_workflow.status='unresolved';second.assessment.valid_until=validUntil;
  second.assessment_digest=verificationDigest(second.assessment);
  Object.assign(second,{status:'unresolved',eligible_for_qualified_promotion:false,tier:'outreach_ready',eligible_for_outreach_ready:true,
    outreach_ready:{...second.outreach_ready,open_checks:CHECKS,open_questions:[TEMPLATES.M]}});
  r.review.outreach_ready_keys=['synthetic-h'];
  r.outreach_ready={schema_version:'blueprint.outreach-ready-admission.v1',state:'enabled',paths:['daily_qa'],label:'hypothesis',
    max_rows_per_batch:50,sends_authorized:false};
  const entry={candidate:hypothesis,open_checks:CHECKS,open_questions:[TEMPLATES.M]};
  for(const name of ['sheets','notion']) r.delivery[name].payload={...r.delivery[name].payload,candidates:[verified],hypotheses:[entry]};
  return rebind(r);
}
const crmSnapshot=()=>({sheet_id:SHEET,complete:true,values:[['CRM'],[],[],[],headers]});

test('the question and open checks are exactly what Blueprint-WebApp #855 derives from the assessment',()=>{
  const c={task:'Depositing',site:'South'},claims=(task,workflow)=>({claims:{site_task:{status:task},human_workflow:{status:workflow}}});
  for(const [assessment,checks,template] of [
    [{...claims('inference','unresolved'),valid_until:null},['site_link','manual_workflow','freshness','existing_automation','fit','interest'],'S'],
    [{...claims('verified_fact','unresolved'),valid_until:'2030-01-01T00:00:00Z'},CHECKS,'M'],
    [{...claims('verified_fact','verified_fact'),valid_until:null},['freshness','existing_automation','fit','interest'],'A']]) {
    assert.deepEqual(openChecks(assessment),checks);assert.equal(firstQuestion(checks,c),TEMPLATES[template]);
  }
  assert.deepEqual(Object.keys(QUESTION_TEMPLATES),['S','M','A']);
});

test('hypotheses add labelled rows in the existing 19 columns and verified cells stay byte-identical',()=>{
  const r=hypothesisRow(),plain=row([candidate()]),snapshot=crmSnapshot();
  for(const name of ['sheets','notion']) {
    const verification=publicationVerification(r,name);
    assert.equal(verification.eligible,true);assert.deepEqual(verification.hypotheses,[{candidate_key:'synthetic-h',eligible:true,reasons:[]}]);
  }
  assert.equal(Object.hasOwn(publicationVerification(plain,'sheets'),'hypotheses'),false);
  const only=hypothesisRow();for(const d of Object.values(only.delivery)) d.payload.candidates=[];rebind(only);
  assert.equal(publicationVerification(only,'sheets').eligible,true);  // A day may publish hypotheses only.
  assert.deepEqual(planSheets(only,snapshot).sheet_rows.map(cells=>cells[6]),['Hypothesis']);
  const plan=planSheets(r,snapshot),base=planSheets(plain,snapshot);
  assert.equal(plan.sheet_rows.length,2);assert.ok(plan.sheet_rows.every(cells=>cells.length===19));
  assert.deepEqual(plan.hypothesis_keys,['synthetic-h']);assert.equal(Object.hasOwn(base,'hypothesis_keys'),false);
  assert.equal(Object.hasOwn(planNotion(plain),'hypothesis_keys'),false);  // Plans without hypotheses are unchanged.
  // Every verified cell is unchanged; only the marker, which binds this payload, differs in M.
  assert.deepEqual(plan.sheet_rows[0],base.sheet_rows[0].map(cell=>cell.replace(base.marker,plan.marker)));
  const hypothesis=plan.sheet_rows[1];
  assert.equal(hypothesis[0],'BP-000002');assert.equal(hypothesis[6],'Hypothesis');
  assert.equal(hypothesis[16],'Outreach-ready: operator, site, task proven');
  assert.equal(hypothesis[12],'First email asks: '+TEMPLATES.M+'\n'+plan.marker);
  // Apart from its own ID and site, a hypothesis row differs from a verified row only in G, M and Q.
  const expected=[...base.sheet_rows[0]];expected[0]='BP-000002';expected[3]='South';
  for(const column of [6,12,16]) expected[column]=hypothesis[column];
  assert.deepEqual(hypothesis,expected);
  const notion=planNotion(r),text=notion.paragraphs.slice(1).join(''),plainText=planNotion(plain).paragraphs.slice(1).join('');
  assert.deepEqual(notion.hypothesis_keys,['synthetic-h']);
  const verifiedEntry=plainText.slice(plainText.indexOf('Example Plant — North'));
  assert.ok(text.includes(verifiedEntry+'\n\nHypothesis, not verified: Example Plant — South\nTask: Depositing\n'));
  assert.ok(text.includes('Open checks: manual_workflow; existing_automation; fit; interest\n'));
  assert.ok(text.includes('First email asks: '+TEMPLATES.M+'\nDraft only; no send is authorized.\n'));
});

for(const change of ['tier','eligible','promotion','version','rule','checks','questions','template','two_questions','order',
  'direction','refused','support','sends','path','limit','overlap','expired','empty','extra','unbound','assessment',
  'duplicate','nonobject','in_crm'])
  test(`a malformed or expired hypothesis (${change}) is withheld on its own list and never blocks the verified row`,async()=>{
    const r=hypothesisRow(),result=r.review.lead_verification.results[1],entry=r.delivery.sheets.payload.hypotheses[0];
    const both=fn=>{for(const name of ['sheets','notion']) fn(r.delivery[name].payload.hypotheses);};
    let verifiedRows=['Needs recheck'];
    if(change==='tier') result.tier='none';
    if(change==='eligible') result.eligible_for_outreach_ready=false;
    if(change==='promotion') result.eligible_for_qualified_promotion=true;
    if(change==='version') result.version='blueprint.lead-verification-result.v2';
    if(change==='rule') result.outreach_ready.rule_version='blueprint.outreach-ready-rule.v1';
    if(change==='checks') both(h=>{h[0]={...entry,open_checks:['interest']};});
    if(change==='questions') both(h=>{h[0]={...entry,open_questions:['Is Depositing at South still done mostly by hand?']};});
    // The review's own block agrees, but the question is not the one #855 derives (S while no site link is open).
    if(change==='template') {result.outreach_ready.open_questions=[TEMPLATES.S];both(h=>{h[0]={...entry,open_questions:[TEMPLATES.S]};});}
    if(change==='two_questions') {const two=[TEMPLATES.M,TEMPLATES.A];result.outreach_ready.open_questions=two;
      both(h=>{h[0]={...entry,open_questions:two};});}
    if(change==='order') r.review.outreach_ready_keys=['synthetic-0'];
    if(change==='direction') delete r.outreach_ready;
    if(change==='refused') r.outreach_ready.state='refused';
    // A hypothesis-only day, so only the hypothesis check can refuse it.
    if(change==='support') {r.review.source_support_verified=false;for(const d of Object.values(r.delivery)) d.payload.candidates=[];
      verifiedRows=[];}
    if(change==='sends') r.outreach_ready.sends_authorized=true;
    if(change==='path') r.outreach_ready.paths=['site_screen'];
    if(change==='limit') r.outreach_ready.max_rows_per_batch=0;
    if(change==='overlap') both(h=>{h[0]={...entry,candidate:r.packet.candidates[0]};});
    if(change==='expired') {result.assessment.valid_until=new Date(Date.now()-1000).toISOString();result.assessment_digest=verificationDigest(result.assessment);}
    if(change==='empty') {both(h=>{h.length=0;});r.review.outreach_ready_keys=[];}
    if(change==='extra') both(h=>{h[0]={...entry,verified:true};});
    if(change==='unbound') both(h=>{h[0]={...entry,candidate:{...entry.candidate,task:'Changed task'}};});
    if(change==='assessment') result.assessment.claims.operator.reason='Changed original evidence';
    if(change==='duplicate') result.duplicate_of='synthetic-0';
    if(change==='nonobject') both(h=>{h[0]=null;});
    rebind(r);
    for(const name of ['sheets','notion']) {
      const verification=publicationVerification(r,name);
      assert.equal(verification.eligible,true);assert.deepEqual(verification.reasons,[]);
      assert.ok(verification.hypotheses.every(h=>h.eligible===false && h.reasons.length>0) || change==='empty' || change==='in_crm');
    }
    const f=await fixture(r);
    if(change==='in_crm') f.values.push(['BP-000041','Example Plant','Facility / site','South','','','Needs recheck','','',
      'https://plant.example/tasks','Research','','','','Depositing']);
    await finishNotion(f);
    const saved=await f.store.get(r.date);saved.delivery.notion.state='acknowledged';await f.store.put(saved);
    const receipt=await f.store.publish(r.date);
    assert.equal(receipt.destination,'sheets');assert.equal(receipt.readback_verified,true);
    assert.deepEqual(f.values.slice(change==='in_crm'?6:5).map(cells=>cells[6]),verifiedRows);
    const plans=(await f.store.get(r.date)).delivery;
    if(change!=='empty') assert.deepEqual(plans.sheets.plan.hypothesis_keys,[]);
    // Only Sheets checks the CRM: a hypothesis the CRM already holds still appears, labelled, in Notion.
    if(change!=='empty') assert.deepEqual(plans.notion.plan.hypothesis_keys,change==='in_crm'?['synthetic-h']:[]);
    assert.equal(JSON.stringify(f.pages[0].body).includes('Hypothesis, not verified'),change==='in_crm');
  });

test('store publication writes the hypothesis row once beside the verified row and reads both back',async()=>{
  const f=await fixture(hypothesisRow());
  assert.equal((await f.store.publish(f.r.date)).destination,'notion');
  const saved=await f.store.get(f.r.date);saved.delivery.notion.state='acknowledged';await f.store.put(saved);
  const receipt=await f.store.publish(f.r.date);
  assert.equal(receipt.reference,`sheets:${SHEET}:Prospects:BP-000001,BP-000002`);assert.equal(receipt.readback_verified,true);
  assert.deepEqual(f.writes.map(w=>w.destination),['notion','sheets']);
  assert.deepEqual(f.values.slice(5).map(r=>r[6]),['Needs recheck','Hypothesis']);
  assert.ok(JSON.stringify(f.pages[0].body).includes('Hypothesis, not verified: Example Plant'));
  assert.deepEqual((await f.store.get(f.r.date)).delivery.sheets.plan.hypothesis_keys,['synthetic-h']);
});

test('a hypothesis that expires after its plan was made is written as planned; readback replays the plan',()=>{
  const at=Date.parse('2026-10-01T00:00:00Z'),r=hypothesisRow({validUntil:new Date(at+60000).toISOString()});
  const publisher=new Publisher({crmReader:async()=>crmSnapshot(),google:async()=>({}),notion:async()=>({}),clock:()=>at});
  const plan=planSheets(r,crmSnapshot(),at),notion=planNotion(r,{now:at});
  assert.deepEqual(plan.hypothesis_keys,['synthetic-h']);assert.equal(plan.sheet_rows.length,2);
  publisher.clock=()=>at+120000;  // Expired since the plan; the verified row stays eligible.
  assert.equal(publicationVerification(r,'sheets',at+120000).eligible,true);
  assert.equal(publicationVerification(r,'sheets',at+120000).hypotheses[0].eligible,false);
  publisher.validate(r,'sheets',plan);publisher.validate(r,'sheets',plan,{reconcile:true});
  publisher.validate(r,'notion',notion);publisher.validate(r,'notion',notion,{reconcile:true});
  assert.deepEqual(planSheets(r,crmSnapshot(),at+120000).hypothesis_keys,[]);  // A fresh plan now leaves it out.
  for(const keys of [['synthetic-x'],['synthetic-h','synthetic-h'],'synthetic-h'])
    assert.throws(()=>publisher.validate(r,'sheets',{...plan,hypothesis_keys:keys}),/publication_plan_binding_invalid/);
});

test('rollback probe: a frozen record whose digest an older bridge dropped never strands the row and publishes no hypothesis',async()=>{
  // Synthetic replay of the reviewer's rollback_probe.mjs. An older bridge rewrites the whole manifest without
  // the fields it does not know; deleting outreach_ready_digest reproduces exactly that write.
  const r=hypothesisRow(),manifest=()=>f.db.values.get(`${ROOT}/runs/${r.date}`),f=await fixture(r);
  const bound=manifest().outreach_ready_digest;
  assert.match(bound,/^[a-f0-9]{64}$/);assert.equal(Object.hasOwn(manifest(),'outreach_ready_unbound'),false);
  await assert.rejects(f.store.put({...r,outreach_ready:{...r.outreach_ready,max_rows_per_batch:49}}),/outreach_ready_already_bound/);
  delete manifest().outreach_ready_digest;  // The rollback.
  await f.store.put(r);  // Roll forward: never stranded.
  assert.equal(manifest().outreach_ready_unbound,true);assert.equal(manifest().outreach_ready_digest,bound);
  const {outreach_ready:_dropped,...without}=r;
  for(const changed of [without,{...r,outreach_ready:{...r.outreach_ready,state:'refused'}}])
    await assert.rejects(f.store.put(changed),/outreach_ready_already_bound/);
  await finishNotion(f);
  const saved=await f.store.get(r.date);saved.delivery.notion.state='acknowledged';await f.store.put(saved);
  assert.equal((await f.store.publish(r.date)).readback_verified,true);
  assert.deepEqual(f.values.slice(5).map(cells=>cells[6]),['Needs recheck']);
  const plans=(await f.store.get(r.date)).delivery;
  assert.deepEqual(plans.sheets.plan.hypothesis_keys,[]);assert.deepEqual(plans.notion.plan.hypothesis_keys,[]);
  assert.equal(manifest().outreach_ready_unbound,true);  // Sticky across later writes.
  assert.deepEqual(publicationVerification(r,'sheets',Date.now(),{withheld:'outreach_ready_unbound'}).hypotheses[0].eligible,false);
  // A shadow-mode row keeps a manifest with neither field, exactly as before the feature.
  const g=await fixture(row([candidate()]));
  assert.equal(['outreach_ready_digest','outreach_ready_unbound'].some(key=>Object.hasOwn(g.db.values.get(`${ROOT}/runs/${r.date}`),key)),false);
});

test('terminal Sheets recovery recovers the verified rows on a day with hypotheses and withholds every hypothesis',async()=>{
  const f=await stoppedTerminalSheetsFixture(),r=await f.store.get(f.r.date);
  const hypothesis={...structuredClone(r.packet.candidates[0]),candidate_key:'candidate-h',site:'South'};
  r.delivery.sheets.payload.hypotheses=[{candidate:hypothesis,open_checks:CHECKS,open_questions:[TEMPLATES.M]}];
  r.delivery.sheets.payload_json=JSON.stringify(r.delivery.sheets.payload);r.delivery.sheets.payload_digest=sha(r.delivery.sheets.payload_json);
  r.review.outreach_ready_keys=['candidate-h'];r.qa.decision=structuredClone(r.review);
  await f.store.put(r);
  f.request.source_row_blob=f.db.values.get(`${ROOT}/runs/${r.date}`).blob;f.request.payload_digest=r.delivery.sheets.payload_digest;
  const result=await f.store.dispatch(f.request);
  assert.equal(result.state,'acknowledged');assert.equal(f.writes.length,1);
  assert.deepEqual(f.values.slice(5).map(cells=>cells[6]),['Needs recheck']);
  assert.deepEqual((await f.store.get(r.date)).delivery.sheets.plan.hypothesis_keys,[]);
  assert.equal((await f.store.dispatch(f.request)).state,'acknowledged');assert.equal(f.writes.length,1);
});
