// Fixed canonical destinations only. No credentials enter plans, prompts or logs.
import {createHash} from 'node:crypto';
import {isDeepStrictEqual} from 'node:util';

export const SHEET = '1n95Ih0Swc-q-kZyUaDHoZh6SVzxvf_zt-CRR7i39bWY';
export const NOTION = '3eb80154161d8116858ed5f376b4b7a9';
const HEADERS = ['Prospect ID','Organization','Prospect type','Site / team','Contact name','Contact details','Verification',
  'Contact source URL','Robot-team fit','Task evidence URL','Stage','Owner','Next action','Next action date','Task / job',
  'Robot capability evidence URL','Evidence maturity','Geography','Evidence checked date'];
const sha = raw => createHash('sha256').update(raw).digest('hex');
const fail = code => {throw new Error(code);};
const rich = text => [{type:'text', text:{content:text}}];
const normalized = x => String(x).toLowerCase().replace(/[^\p{L}\p{N}_]+/gu,' ').trim();
const identity = x => [x.organization,x.site,x.task].map(normalized).join('\n');

function bind(row, destination) {
  const delivery = row.delivery?.[destination];
  if (!delivery || row.qa?.state !== 'validated' || row.review?.source_support_verified !== true
      || row.review?.crm_rechecked !== true || row.review.packet_digest !== row.packet_digest
      || delivery.key !== `${row.run_key}:${destination}` || !/^[a-f0-9]{64}$/.test(delivery.payload_digest)
      || typeof delivery.payload_json!=='string' || sha(delivery.payload_json)!==delivery.payload_digest
      || !isDeepStrictEqual(JSON.parse(delivery.payload_json),delivery.payload))
    fail('publication_agent_qa_binding_invalid');
  if (destination === 'sheets' && (delivery.payload.sheet_id !== SHEET || delivery.payload.tab !== 'Prospects'))
    fail('publication_destination_invalid');
  if (destination === 'notion' && delivery.payload.parent_id !== NOTION) fail('publication_destination_invalid');
  if (!Array.isArray(delivery.payload.candidates) || delivery.payload.candidates.length > 3)
    fail('publication_candidates_invalid');
  return delivery;
}

export function planSheets(row, snapshot) {
  const d = bind(row, 'sheets'), values = snapshot.values;
  if (snapshot.sheet_id !== SHEET || snapshot.complete !== true || !Array.isArray(values)
      || !isDeepStrictEqual(values[4]?.slice(0,19), HEADERS)) fail('publication_crm_schema_mismatch');
  const used = values.slice(5).filter(r => r.some(x => String(x).trim()));
  const ids = used.map(r => r[0]);
  if (ids.some(id => !/^BP-\d{6}$/.test(id)) || new Set(ids).size !== ids.length)
    fail('publication_crm_ids_invalid');
  const existing = new Set(used.map(r => identity({organization:r[1],site:r[3],task:r[14]})));
  if (d.payload.candidates.some(c => existing.has(identity(c)))) fail('publication_crm_duplicate_changed');
  let sequence = Math.max(0, ...ids.map(id => Number(id.slice(3))));
  const marker = `[${d.key};${d.payload_digest}]`;
  const rows = d.payload.candidates.map(c => {
    if (++sequence > 999999) fail('publication_crm_id_capacity');
    const task = c.evidence.find(e => e.role === 'task'), capability = c.evidence.find(e => e.role === 'capability');
    if (!task || !capability || !['unqualified','needs_review'].includes(c.qualification_status))
      fail('publication_candidate_scope_invalid');
    return [`BP-${String(sequence).padStart(6,'0')}`,c.organization,'Facility / site',c.site,'','',
      'Needs recheck','',c.potential_robot_match,task.url,'Research','',c.proposed_next_action+'\n'+marker,'',c.task,
      capability.url,'Unverified',c.location,row.date];
  });
  const body = JSON.stringify({majorDimension:'ROWS',values:rows});
  return {destination:'sheets',key:d.key,payload_digest:d.payload_digest,body_json:body,request_digest:sha(body),
    marker,crm_values:values,sheet_rows:rows};
}

export function planNotion(row) {
  const d = bind(row,'notion'), marker = `${d.key};${d.payload_digest}`;
  const text = `Blueprint research ${row.date}\n${d.payload.summary}\n\n`+
    d.payload.candidates.map(c => `${c.organization} — ${c.site}\nTask: ${c.task}\nStatus: ${c.qualification_status}\n`+
      `Unknowns: ${c.unknowns.join('; ')}\nNext proposed action: ${c.proposed_next_action}\n`+
      c.evidence.map(e => `${e.claim_kind}/${e.classification}: ${e.claim}\n${e.url}\nChecked: ${e.checked_date}`).join('\n')).join('\n\n');
  const paragraphs = [marker,...Array.from({length:Math.ceil(text.length/1800)},(_,i)=>text.slice(i*1800,(i+1)*1800))];
  if (paragraphs.length > 90) fail('publication_report_too_large');
  const title = `Blueprint research ${row.date} ${d.payload_digest.slice(0,12)}`;
  const body = JSON.stringify({parent:{type:'page_id',page_id:NOTION},properties:{title:{type:'title',title:rich(title)}},
    children:paragraphs.map(content=>({object:'block',type:'paragraph',paragraph:{rich_text:rich(content)}}))});
  return {destination:'notion',key:d.key,payload_digest:d.payload_digest,title,paragraphs,
    body_json:body,request_digest:sha(body)};
}

export class Publisher {
  constructor({crmReader,google,notion}) {Object.assign(this,{crmReader,google,notion});}
  async prepare(row,destination) {
    if (destination === 'notion') {
      const parent = await this.notion('GET',`/pages/${NOTION}`);
      if (parent.id?.replaceAll('-','') !== NOTION || parent.object !== 'page') fail('publication_notion_parent_mismatch');
      return planNotion(row);
    }
    const snapshot = await this.crmReader(), plan = planSheets(row,snapshot);
    const first = snapshot.values.length+1, last = first+Math.max(0,plan.sheet_rows.length-1);
    if (plan.sheet_rows.length) {
      // Refuse formulas/validation-backed destination cells rather than rewriting native structure.
      const meta = await this.google('GET',`?ranges=${encodeURIComponent(`Prospects!A${first}:S${last}`)}`+
        '&includeGridData=true&fields=sheets(data(rowData(values(userEnteredValue,dataValidation))))');
      for (const sheet of meta.sheets || []) for (const grid of sheet.data || []) for (const row of grid.rowData || [])
        for (const cell of row.values || []) if (cell.userEnteredValue || cell.dataValidation)
          fail('publication_target_cells_not_plain_empty');
    }
    return plan;
  }
  validate(row,destination,plan) {
    const d = bind(row,destination);
    if (plan.destination !== destination || plan.key !== d.key || plan.payload_digest !== d.payload_digest
        || plan.request_digest !== sha(plan.body_json)) fail('publication_plan_binding_invalid');
    const expected = destination === 'notion' ? planNotion(row) : planSheets(row,{sheet_id:SHEET,complete:true,values:plan.crm_values});
    if (!isDeepStrictEqual(expected,plan)) fail('publication_plan_binding_invalid');
  }
  async write(row,destination,plan) {
    this.validate(row,destination,plan);
    if (destination === 'notion') {
      await this.notion('POST','/pages',JSON.parse(plan.body_json));
    } else if (plan.sheet_rows.length) {
      const current = await this.crmReader();
      if (!isDeepStrictEqual(current.values,plan.crm_values)) fail('publication_crm_changed_before_write');
      await this.google('POST',`/values/${encodeURIComponent('Prospects!A:S')}:append?valueInputOption=RAW&insertDataOption=OVERWRITE`,JSON.parse(plan.body_json));
    }
  }
  async reconcile(row,destination,plan) {
    this.validate(row,destination,plan);
    if (destination === 'sheets') {
      const snapshot = await this.crmReader();
      if (!isDeepStrictEqual(snapshot.values[4]?.slice(0,19),HEADERS)) fail('publication_crm_schema_mismatch');
      const marked = snapshot.values.slice(5).filter(r=>typeof r[12]==='string' && r[12].includes(plan.marker));
      if (!marked.length && plan.sheet_rows.length) return null;
      if (!isDeepStrictEqual(marked.map(r=>r.slice(0,19)),plan.sheet_rows)) fail('publication_readback_conflict');
      const ids = snapshot.values.slice(5).filter(r=>r.some(x=>String(x).trim())).map(r=>r[0]);
      if (new Set(ids).size!==ids.length) fail('publication_crm_id_collision');
      return {destination,key:plan.key,payload_digest:plan.payload_digest,readback_verified:true,
        reference:`sheets:${SHEET}:Prospects:${plan.sheet_rows.map(r=>r[0]).join(',') || 'no_candidates'}`};
    }
    const matches = []; let cursor = null;
    for (let page=0;page<2;page++) {
      const children = await this.notion('GET',`/blocks/${NOTION}/children?page_size=100${cursor?'&start_cursor='+encodeURIComponent(cursor):''}`);
      for (const block of children.results || []) if (block.type==='child_page' && block.child_page?.title===plan.title) matches.push(block.id);
      if (!children.has_more) {cursor=null;break;}
      cursor=children.next_cursor;
      if (!cursor) fail('publication_notion_pagination_invalid');
    }
    if (cursor) fail('publication_notion_parent_scan_incomplete');
    if (matches.length>1) fail('publication_notion_duplicate_pages');
    if (!matches.length) return null;
    const page = await this.notion('GET',`/pages/${matches[0]}`);
    if (page.parent?.page_id?.replaceAll('-','')!==NOTION) fail('publication_notion_parent_mismatch');
    const content = await this.notion('GET',`/blocks/${matches[0]}/children?page_size=100`);
    if (content.has_more) fail('publication_notion_report_scan_incomplete');
    const paragraphs=(content.results || []).map(b=>b.type==='paragraph' ? b.paragraph.rich_text.map(x=>x.plain_text??x.text?.content??'').join('') : null);
    if (!isDeepStrictEqual(paragraphs,plan.paragraphs)) fail('publication_readback_conflict');
    return {destination,key:plan.key,payload_digest:plan.payload_digest,readback_verified:true,reference:`notion:${matches[0]}`};
  }
}

export async function livePublisher(account,crmReader,token) {
  const {JWT} = await import('google-auth-library');
  const auth = new JWT({email:account.client_email,key:account.private_key,scopes:['https://www.googleapis.com/auth/spreadsheets']});
  const request = auth.transporter.request.bind(auth.transporter);
  auth.transporter.request = options=>request({...options,timeout:12000,retry:false,retryConfig:{retry:0},maxRedirects:0});
  const google = async(method,suffix,data)=>{
    const result=await auth.request({url:`https://sheets.googleapis.com/v4/spreadsheets/${SHEET}${suffix}`,
      method,data,timeout:12000,retry:false,retryConfig:{retry:0},maxRedirects:0});
    return result.data;
  };
  const notion = async(method,path,body)=>{
    if (!token) fail('publication_notion_binding_absent');
    const response=await fetch('https://api.notion.com/v1'+path,{method,redirect:'error',signal:AbortSignal.timeout(12000),
      headers:{Authorization:'Bearer '+token,'Notion-Version':'2022-06-28','Content-Type':'application/json'},
      ...(body?{body:JSON.stringify(body)}:{})});
    if (!response.ok) fail(response.status===401||response.status===403?'publication_notion_permission_denied':'publication_notion_unavailable');
    const raw=await response.text(); if (Buffer.byteLength(raw)>2000000) fail('publication_notion_response_too_large');
    return JSON.parse(raw);
  };
  return new Publisher({crmReader,google,notion});
}
