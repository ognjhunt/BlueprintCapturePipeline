// Hermetic site-screen admission rows (publisher planScreenSheets). Synthetic operators and *.example hosts only.
import {test} from 'node:test';
import assert from 'node:assert/strict';
import {Publisher, planScreenSheets, pythonDigest, screenMarker, screenPayloadProblem, SHEET, HYPOTHESIS_LABEL,
  HYPOTHESIS_MATURITY} from '../tools/daily_research/publisher.mjs';

const headers=['Prospect ID','Organization','Prospect type','Site / team','Contact name','Contact details','Verification',
  'Contact source URL','Robot-team fit','Task evidence URL','Stage','Owner','Next action','Next action date','Task / job',
  'Robot capability evidence URL','Evidence maturity','Geography','Evidence checked date'];
const ID='a'.repeat(64);
const entry=(number,changes={})=>({site_key:String(number).repeat(64).slice(0,64),result_digest:'b'.repeat(63)+number,
  identity:'c'.repeat(63)+number,organization:`Synthetic Operator ${number}`,site:`Synthetic Works ${number}`,
  location:`${number} Example Road, Fixture City, TX`,task:'CNC machine tending',task_url:`https://operator-${number}.example/careers`,
  checked_on:'2026-10-05',question:`What has kept the remaining CNC machine tending work at Fixture City from being automated so far?`,
  contact:{name:'',details:`plant.team@operator-${number}.example (published team inbox)`,source_url:`https://operator-${number}.example/contact`},
  ...changes});
const payload=(entries,duplicates=[])=>({schema_version:'blueprint.site-screen-sheets-payload.v1',admission_id:ID,sheet_id:SHEET,
  tab:'Prospects',entries,duplicates});
const crm=(...rows)=>({sheet_id:SHEET,complete:true,values:[['Synthetic CRM'],[],[],[],headers,...rows]});
const held=(id,organization,site,location,task)=>[id,organization,'Facility / site',site,'','','Needs recheck','','unknown',
  'https://held.example/jobs','Research','','Earlier row','',task,'','Unverified',location,'2026-10-01'];

test('screen rows are hypotheses in the existing 19 columns, each with its own marker and recipient cells',()=>{
  const plan=planScreenSheets(payload([entry(1),entry(2)]),crm(held('BP-000041','Other Operator','North','Elsewhere','Packing')));
  assert.deepEqual(plan.sheet_rows.map(row=>row[0]),['BP-000042','BP-000043']);
  const [row]=plan.sheet_rows;
  assert.equal(row.length,19);
  assert.deepEqual([row[6],row[16],row[8],row[10]],[HYPOTHESIS_LABEL,HYPOTHESIS_MATURITY,'unknown','Research']);
  assert.deepEqual([row[4],row[5],row[7]],['','plant.team@operator-1.example (published team inbox)','https://operator-1.example/contact']);
  assert.equal(row[12],`First email asks: ${entry(1).question}\n${screenMarker(ID,entry(1).result_digest)}`);
  assert.equal(plan.marker,`[screen:${ID};`);
  assert.equal(plan.body_json,JSON.stringify({majorDimension:'ROWS',values:plan.sheet_rows}));
  assert.equal(plan.payload_digest,pythonDigest(payload([entry(1),entry(2)])));
});

test('the publisher structural duplicate rule and an existing marker refuse the plan',()=>{
  const same=held('BP-000001','Synthetic Operator 1','Synthetic Works 1','1 Example Road, Fixture City, TX','CNC machine tending');
  assert.throws(()=>planScreenSheets(payload([entry(1)]),crm(same)),/screen_admission_crm_duplicate_changed/);
  const noLocation=held('BP-000001','Synthetic Operator 1','Synthetic Works 1','','CNC machine tending');
  assert.throws(()=>planScreenSheets(payload([entry(1)]),crm(noLocation)),/screen_admission_crm_duplicate_changed/);
  const marked=[...held('BP-000001','Other Operator','North','Elsewhere','Packing')];
  marked[12]=`First email asks: x?\n[screen:${ID};${'d'.repeat(64)}]`;
  assert.throws(()=>planScreenSheets(payload([entry(1)]),crm(marked)),/screen_admission_readback_conflict/);
});

test('a malformed payload is refused before any plan',()=>{
  assert.equal(screenPayloadProblem(payload([entry(1)]),ID),null);
  for(const bad of [payload([entry(1),entry(1)]),payload([entry(1,{question:'Two? Questions?'})]),
    payload([entry(1,{extra:true})]),payload([]),{...payload([entry(1)]),sheet_id:'other'},
    payload([entry(1)],[{site_key:entry(1).site_key,code:'screen_admission_crm_duplicate'}])])
    assert.equal(screenPayloadProblem(bad,ID),'screen_admission_payload_invalid');
  assert.equal(screenPayloadProblem(payload([entry(1)]),'b'.repeat(64)),'screen_admission_payload_invalid');
});

test('the CRM digest is Python canonical JSON, non-ASCII escaped',()=>{
  // tools.daily_research.runner.digest of the same value.
  assert.equal(pythonDigest([['Café São Paulo','line\nbreak',' ','😀'],{b:1,a:[true,null]}]),
    'b8f5205336ef684f2c5c141a59150f745175f243e4801adee9e9863e920ded7b');
});

test('prepare needs the CRM the worker deduplicated against; write is once and reconcile only reads',async()=>{
  const sheet=crm(),puts=[];
  const publisher=new Publisher({crmReader:async()=>structuredClone(sheet),notion:async()=>({}),google:async(method,path,body)=>{
    if(method==='GET') return {sheets:[]};
    puts.push(path);sheet.values.push(...body.values);return {};
  }});
  const p=payload([entry(1)]);
  await assert.rejects(publisher.prepareScreen(p,'e'.repeat(64)),/screen_admission_crm_changed/);
  const plan=await publisher.prepareScreen(p,pythonDigest(sheet.values));
  assert.equal(await publisher.reconcileScreen(p,plan),null);
  await publisher.writeScreen(p,plan);
  const receipt=await publisher.reconcileScreen(p,plan);
  assert.deepEqual(receipt,{destination:'sheets',kind:'screen',admission_id:ID,payload_digest:plan.payload_digest,
    readback_verified:true,reference:`sheets:${SHEET}:Prospects:BP-000001`});
  assert.deepEqual(puts,[`/values/${encodeURIComponent('Prospects!A6:S6')}?valueInputOption=RAW`]);
  await assert.rejects(publisher.writeScreen(p,plan),/publication_crm_changed_before_write/);
  await assert.rejects(publisher.reconcileScreen(p,{...plan,sheet_rows:[]}),/screen_admission_plan_binding_invalid/);
});
