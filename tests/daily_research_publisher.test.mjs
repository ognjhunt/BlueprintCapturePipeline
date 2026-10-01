import {test} from 'node:test';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {Publisher,planSheets,planNotion} from '../tools/daily_research/publisher.mjs';
import {Store,ROOT} from '../tools/daily_research/firestore_bridge.mjs';
import {MemoryFirestore} from './fixtures/daily_research/firestore-memory.mjs';

const SHEET='1n95Ih0Swc-q-kZyUaDHoZh6SVzxvf_zt-CRR7i39bWY', NOTION='3eb80154161d8116858ed5f376b4b7a9';
const sha=x=>createHash('sha256').update(x).digest('hex');
const headers=['Prospect ID','Organization','Prospect type','Site / team','Contact name','Contact details','Verification',
  'Contact source URL','Robot-team fit','Task evidence URL','Stage','Owner','Next action','Next action date','Task / job',
  'Robot capability evidence URL','Evidence maturity','Geography','Evidence checked date'];
const candidate=()=>({organization:'Example Plant',site:'North',task:'Depositing',location:'Chicago',
  qualification_status:'unqualified',unknowns:['Interest unknown'],potential_robot_match:'Hypothesis',
  proposed_next_action:'Review exact operator evidence',evidence:['task','capability'].map(role=>({role,
    url:'https://plant.example/tasks',classification:'operator',claim_kind:'fact',claim:'Task described',checked_date:'2026-09-30'}))});
function row(candidates=[candidate()]) {
  const r={date:'2026-09-30',run_key:'blueprint-researcher:2026-09-30',metadata:{run_key:'date'},
    state:'reviewed',cleanup_required:true,packet:{candidates},packet_digest:'a'.repeat(64),qa:{state:'validated'},
    review:{packet_digest:'a'.repeat(64),source_support_verified:true,crm_rechecked:true},delivery:{}};
  for (const [name,payload] of Object.entries({sheets:{sheet_id:SHEET,tab:'Prospects',candidates},
    notion:{parent_id:NOTION,summary:'Supported brief with explicit gaps',candidates}})) {
    const raw=JSON.stringify(payload);
    r.delivery[name]={key:r.run_key+':'+name,payload,payload_json:raw,payload_digest:sha(raw),state:'pending'};
  }
  return r;
}

async function fixture() {
  const db=new MemoryFirestore(),values=[['CRM'],[],[],[],headers], pages=[],writes=[];
  const time={now:Date.now()}, faults={lost:false,changed:false};
  const snapshot=()=>({sheet_id:SHEET,complete:true,values:structuredClone(values)});
  const google=async(method,path,body)=>{
    if(method==='GET') return {sheets:[]};
    assert.equal(method,'POST');assert.ok(path.startsWith('/values/Prospects!A%3AS:append?'));
    assert.ok(path.includes('valueInputOption=RAW'));assert.equal(body.values[0].length,19);
    assert.ok(db.values.get(`${ROOT}/runs/2026-09-30`).publication_claimed.sheets);
    assert.ok((await store.get('2026-09-30')).delivery.sheets.plan);
    writes.push({destination:'sheets',body});values.push(...body.values);
    if(faults.lost) {faults.lost=false;throw new Error('private upstream error');}
    return {};
  };
  const notion=async(method,path,body)=>{
    if(method==='POST') {
      assert.equal(path,'/pages');assert.equal(body.parent.page_id,NOTION);
      assert.ok(db.values.get(`${ROOT}/runs/2026-09-30`).publication_claimed.notion);
      assert.ok((await store.get('2026-09-30')).delivery.notion.plan);
      pages.push({id:'page-new',body});writes.push({destination:'notion',body});
      if(faults.lost) {faults.lost=false;throw new Error('private upstream error');}
      return {id:'page-new'};
    }
    if(path===`/pages/${NOTION}`) return {id:NOTION,object:'page'};
    if(path.startsWith(`/blocks/${NOTION}/children`)) return {has_more:false,results:pages.map(p=>({id:p.id,type:'child_page',
      child_page:{title:p.body.properties.title.title[0].text.content}}))};
    if(path==='/pages/page-new') return {parent:{page_id:NOTION}};
    if(path.startsWith('/blocks/page-new/children')) return {has_more:false,results:pages[0].body.children};
    throw new Error('unexpected request '+path);
  };
  const publisher=new Publisher({crmReader:async()=>snapshot(),google,notion});
  db.values.set(ROOT,{schema_version:'blueprint.research-control.v1',enabled:true,workflow:{enabled:true,
    qa_authority_reference:'approved-QA',publication_authority_reference:'approved-fixed-targets'}});
  const store=new Store(db,()=>time.now,'one',async()=>snapshot(),publisher);
  await store.acquire();
  const r=row(); await store.put({...r,state:'creating'});await store.put(r);
  return {db,store,publisher,time,faults,values,pages,writes,r,snapshot};
}

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
  const existing=['BP-000012','Example Plant','Facility / site','North','','','','','','','','','','','Depositing'];
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
  f.publisher.notion=async()=>({has_more:true,next_cursor:'cursor',results:[]});
  await assert.rejects(f.publisher.reconcile(f.r,'notion',planNotion(f.r)),/scan_incomplete/);
});

test('no accepted candidates yields no Sheets write and a verified empty receipt',async()=>{
  const f=await fixture(),r=row([]);const plan=planSheets(r,f.snapshot());
  await f.publisher.write(r,'sheets',plan);
  assert.equal((await f.publisher.reconcile(r,'sheets',plan)).readback_verified,true);
  assert.equal(f.writes.length,0);
});
