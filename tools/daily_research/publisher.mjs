// Fixed canonical destinations only. No credentials enter plans, prompts or logs.
import {createHash} from 'node:crypto';
import {isDeepStrictEqual} from 'node:util';
import {verificationDigest} from './verification-digest.mjs';

export const SHEET = '1n95Ih0Swc-q-kZyUaDHoZh6SVzxvf_zt-CRR7i39bWY';
export const NOTION = '3eb80154161d8116858ed5f376b4b7a9';
const HEADERS = ['Prospect ID','Organization','Prospect type','Site / team','Contact name','Contact details','Verification',
  'Contact source URL','Robot-team fit','Task evidence URL','Stage','Owner','Next action','Next action date','Task / job',
  'Robot capability evidence URL','Evidence maturity','Geography','Evidence checked date'];
const sha = raw => createHash('sha256').update(raw).digest('hex');
const fail = code => {throw new Error(code);};
const rich = text => [{type:'text', text:{content:text}}];
const normalized = x => String(x || '').normalize('NFKC').toLowerCase().replace(/[^\p{L}\p{N}]+/gu,' ').trim();
const identity = x => [x.organization,x.site||x.location,x.location||x.site,x.task].map(normalized).join('\n');
const NOTION_PARENT_PAGE_LIMIT = 100, NOTION_READ_BUDGET_MS = 25000;
// Outreach-ready hypotheses (verification.py result v3, outreach_ready.py). Labelled rows, never verified ones.
const OUTREACH_RESULT='blueprint.lead-verification-result.v3', OUTREACH_RULE='blueprint.outreach-ready-rule.v1.1';
// The WebApp recomputes these cells (Blueprint-WebApp #855): Sheets column G and Q; each Notion entry heading.
export const HYPOTHESIS_LABEL='Hypothesis', HYPOTHESIS_MATURITY='Outreach-ready: operator, site, task proven';
export const HYPOTHESIS_HEADING='Hypothesis, not verified';
// Design v1.1, exactly as Blueprint-WebApp #855 derives them from the assessment: open checks in rule
// order, then exactly one question, by precedence S (site link open), then M (manual workflow open), then A.
export const QUESTION_TEMPLATES={
  S:(task,site)=>`Is ${task} done at your ${site} site, or somewhere else in the company?`,
  M:(task,site)=>`Which parts of ${task} at ${site} still need people, and what has kept them from being automated?`,
  A:(task,site)=>`What has kept the remaining ${task} work at ${site} from being automated so far?`};
export const openChecks=assessment=>[...(assessment?.claims?.site_task?.status!=='verified_fact'?['site_link']:[]),
  ...(assessment?.claims?.human_workflow?.status!=='verified_fact'?['manual_workflow']:[]),
  ...(assessment?.valid_until===null?['freshness']:[]),'existing_automation','fit','interest'];
export const firstQuestion=(checks,candidate)=>QUESTION_TEMPLATES[checks.includes('site_link')?'S'
  :checks.includes('manual_workflow')?'M':'A'](candidate.task,candidate.site);
// The form #855 records for valid_until (its ISO_TIMESTAMP); outreach_ready.VALID_UNTIL mirrors it.
const VALID_UNTIL=/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,9})?(?:Z|[+-]\d{2}:\d{2})$/;
const NOTION_BATCH_BLOCKS = 90, NOTION_REQUEST_BYTES = 450000;
const paragraph = content=>({object:'block',type:'paragraph',paragraph:{rich_text:rich(content)}});
const createBody = (title,children)=>({parent:{type:'page_id',page_id:NOTION},
  properties:{title:{type:'title',title:rich(title)}},children});

function textParagraphs(text) {
  const result=[];
  for(let start=0;start<text.length;) {
    let end=Math.min(text.length,start+1800);
    if(end<text.length && /[\uD800-\uDBFF]/.test(text[end-1]) && /[\uDC00-\uDFFF]/.test(text[end])) end--;
    result.push(text.slice(start,end));start=end;
  }
  return result;
}

function crmRows(snapshot) {
  const values = snapshot.values;
  if (snapshot.sheet_id !== SHEET || snapshot.complete !== true || !Array.isArray(values)
      || !isDeepStrictEqual(values[4]?.slice(0,19), HEADERS)) fail('publication_crm_schema_mismatch');
  const used = values.slice(5).filter(r => r.some(x => String(x).trim()));
  if (used.some(r => r.length < 15 || [1,3,9,14].some(i => typeof r[i] !== 'string' || !r[i].trim())))
    fail('publication_crm_identity_incomplete');
  const ids = used.map(r => r[0]);
  if (ids.some(id => !/^BP-\d{6}$/.test(id)) || new Set(ids).size !== ids.length)
    fail('publication_crm_ids_invalid');
  return used;
}

// The v2/v3 rule: a version or schema_version alias is present and every present one is v1.
const aliasInvalid=assessment=>(!Object.hasOwn(assessment,'version') && !Object.hasOwn(assessment,'schema_version'))
  || ['version','schema_version'].some(key=>Object.hasOwn(assessment,key) && assessment[key]!=='blueprint.lead-verification.v1');

// A hypothesis payload entry is exactly what Runner.review bound: an enabled daily_qa direction
// frozen on the row, the review's outreach_ready_keys in order, and a current v3 outreach_ready
// result whose open checks and one question it repeats and #855 derives. It never overlaps a
// verified candidate. Each entry gets its own list: a hypothesis that is malformed, withheld or
// expired since review is left out of the publication and never blocks the day's verified rows.
function hypothesisEligibility(row,candidates,hypotheses,now,withheld=null) {
  const keys=row.review?.outreach_ready_keys,frozen=row.outreach_ready,entries=Array.isArray(hypotheses)?hypotheses:[];
  const shared=[];
  if(withheld) shared.push(`Hypotheses on this row are withheld from publication (${withheld}); the verified rows publish without them.`);
  if(!Array.isArray(keys) || row.review?.source_support_verified!==true
      || row.packet?.lead_verification_result_version!==OUTREACH_RESULT || frozen?.state!=='enabled'
      || frozen.sends_authorized!==false || !Array.isArray(frozen.paths) || !frozen.paths.includes('daily_qa')
      || !Number.isSafeInteger(frozen.max_rows_per_batch) || entries.length>frozen.max_rows_per_batch
      || !isDeepStrictEqual(entries.map(h=>h?.candidate?.candidate_key),keys))
    shared.push('Hypotheses publish only from an enabled daily_qa outreach-ready direction and the protected review that listed them.');
  const verified=new Set(candidates.map(c=>c?.candidate_key)),seen=new Set();
  return entries.map(h=>{
    const reasons=[...shared],c=h?.candidate,key=typeof c?.candidate_key==='string'?c.candidate_key:null;
    const results=row.review?.lead_verification?.results?.filter(r=>r.candidate_key===key) || [];
    const result=results.length===1 ? results[0] : null,outreach=result?.outreach_ready,assessment=result?.assessment;
    let bound=false,checks=null,question=null;
    try {bound=!!assessment && result.candidate_digest===verificationDigest(c)
      && result.assessment_digest===verificationDigest(assessment)
      && assessment.candidate_digest===result.candidate_digest && !aliasInvalid(assessment);
      checks=openChecks(assessment);question=firstQuestion(checks,c);}catch {}
    if(!h || typeof h!=='object' || Array.isArray(h) || !key || seen.has(key) || verified.has(key) || !bound
        || Object.keys(h).sort().join(',')!=='candidate,open_checks,open_questions'
        || !row.packet.candidates?.some(p=>isDeepStrictEqual(p,c))
        || result.version!==OUTREACH_RESULT || result.tier!=='outreach_ready' || result.eligible_for_outreach_ready!==true
        || result.eligible_for_qualified_promotion!==false || result.duplicate_of || outreach?.rule_version!==OUTREACH_RULE
        || !isDeepStrictEqual(h.open_checks,outreach.open_checks) || !isDeepStrictEqual(h.open_questions,outreach.open_questions)
        || !isDeepStrictEqual(h.open_checks,checks) || !isDeepStrictEqual(h.open_questions,[question])
        || typeof question!=='string' || question.indexOf('?')!==question.length-1)
      reasons.push(`${key || 'hypothesis'}: retain the exact outreach-ready entry the protected review bound, with its open checks and one question; a hypothesis is never verified and never sent.`);
    else if(!(Date.parse(assessment.assessed_at)<=now && (assessment.valid_until===null
        || typeof assessment.valid_until==='string' && VALID_UNTIL.test(assessment.valid_until) && now<Date.parse(assessment.valid_until))))
      reasons.push(`${key}: its assessment is no longer current, so this hypothesis is left out; the verified rows still publish.`);
    if(key) seen.add(key);
    return {candidate_key:key,eligible:!reasons.length,reasons};
  });
}

// The hypotheses a fresh plan publishes: the eligible ones at ``now``, in payload order. A replayed
// plan uses exactly the keys it recorded, which must be an ordered subset of the bound payload.
function plannedHypotheses(row,d,now,{hypothesisKeys,withheld}={}) {
  const bound=d.payload.hypotheses || [];
  if(hypothesisKeys===undefined) {
    if(now===null) return [];
    const eligible=new Set(hypothesisEligibility(row,d.payload.candidates,bound,now,withheld).filter(e=>e.eligible).map(e=>e.candidate_key));
    return bound.filter(h=>eligible.has(h?.candidate?.candidate_key));
  }
  const order=bound.map(h=>h?.candidate?.candidate_key);
  if(!bound.length || !Array.isArray(hypothesisKeys) || new Set(hypothesisKeys).size!==hypothesisKeys.length
      || hypothesisKeys.some((key,i)=>!order.includes(key) || i && order.indexOf(hypothesisKeys[i-1])>order.indexOf(key)))
    fail('publication_plan_binding_invalid');
  return bound.filter(h=>hypothesisKeys.includes(h?.candidate?.candidate_key));
}

// Protected Runner.review writes these derived receipts. Rebind exact raw
// candidates and assessments at every fresh sink claim and mutation. This is
// evidence eligibility only; the separate commercial/rights/readiness gates stay open.
// Hypotheses never change eligibility: each has its own list under ``hypotheses``.
export function publicationVerification(row,destination,now=Date.now(),{withheld=null}={}) {
  const candidates=row.delivery?.[destination]?.payload?.candidates;
  if(!Array.isArray(candidates)) return {eligible:false,reasons:['Retain an exact canonical publication candidate payload.']};
  const reasons=[];
  const hypotheses=row.delivery[destination].payload.hypotheses;
  for(const candidate of candidates) {
    const results=row.review?.lead_verification?.results?.filter(r=>r.candidate_key===candidate.candidate_key) || [];
    const result=results.length===1 ? results[0] : null;
    let bound=false;
    try {bound=!!result?.assessment && result.candidate_digest===verificationDigest(candidate)
      && result.assessment_digest===verificationDigest(result.assessment)
      && result.assessment.candidate_digest===result.candidate_digest;}catch {}
    if(row.review?.source_support_verified!==true || result?.status!=='verified'
        || result.eligible_for_qualified_promotion!==true || result.duplicate_of || !bound
        || (['blueprint.lead-verification-result.v2',OUTREACH_RESULT].includes(result.version)
          ? (!Object.hasOwn(result.assessment,'version') && !Object.hasOwn(result.assessment,'schema_version'))
            || ['version','schema_version'].some(key=>Object.hasOwn(result.assessment,key) && result.assessment[key]!=='blueprint.lead-verification.v1')
          : result.assessment.version!=='blueprint.lead-verification.v1')
        || !(Date.parse(result.assessment.assessed_at)<=now && now<Date.parse(result.assessment.valid_until))) {
      reasons.push(`${candidate.candidate_key || candidate.organization || 'candidate'}: retain a current, exact evidence assessment and protected verified review before promotion; missing, expired, duplicate or contradicted evidence remains ineligible.`);
      if(Array.isArray(result?.reasons)) reasons.push(...result.reasons);
    }
  }
  return {eligible:reasons.length===0,reasons,separate_gates:['buying_intent','consent_rights','commercial_qualification','robot_compatibility','deployment_readiness'],
    ...(hypotheses!==undefined?{hypotheses:hypothesisEligibility(row,candidates,hypotheses,now,withheld)}:{})};
}

export function requirePublicationVerification(row,destination,now=Date.now()) {
  if(!publicationVerification(row,destination,now).eligible) fail('publication_lead_verification_required');
}

function bind(row, destination, now=Date.now()) {
  const delivery = row.delivery?.[destination];
  if (!delivery || row.qa?.state !== 'validated' || typeof row.review?.source_support_verified !== 'boolean'
      || row.review?.crm_rechecked !== true || row.review.packet_digest !== row.packet_digest
      || delivery.key !== `${row.run_key}:${destination}` || !/^[a-f0-9]{64}$/.test(delivery.payload_digest)
      || typeof delivery.payload_json!=='string' || sha(delivery.payload_json)!==delivery.payload_digest
      || !isDeepStrictEqual(JSON.parse(delivery.payload_json),delivery.payload))
    fail('publication_agent_qa_binding_invalid');
  if (destination === 'sheets' && (delivery.payload.sheet_id !== SHEET || delivery.payload.tab !== 'Prospects'))
    fail('publication_destination_invalid');
  if (destination === 'notion' && delivery.payload.parent_id !== NOTION) fail('publication_destination_invalid');
  if (!Array.isArray(delivery.payload.candidates) || Buffer.byteLength(delivery.payload_json)>2000000
      || delivery.payload.hypotheses!==undefined && !Array.isArray(delivery.payload.hypotheses))
    fail('publication_candidates_invalid');
  // GET-only reconciliation proves an existing exact effect. Legacy or expired
  // assessments cannot authorize a new claim, but cannot erase its readback.
  if(now!==null) requirePublicationVerification(row,destination,now);
  if(delivery.presentation) {
    const p=delivery.presentation;
    if(destination!=='notion' || p.schema_version!=='blueprint.research-presentation.v1' || p.strategy!=='concise'
        || p.source_payload_digest!==delivery.payload_digest || typeof p.summary!=='string' || !p.summary.trim()
        || Buffer.byteLength(p.summary)>2000000 || typeof p.decision_json!=='string'
        || sha(p.decision_json)!==p.decision_digest
        || !isDeepStrictEqual(JSON.parse(p.decision_json),{destination,strategy:p.strategy,summary:p.summary,
          source_payload_digest:p.source_payload_digest})) fail('publication_presentation_binding_invalid');
    return {...delivery,payload:{...delivery.payload,summary:p.summary}};
  }
  return delivery;
}

export function planSheets(row, snapshot, now=Date.now(), options={}) {
  const d = bind(row, 'sheets',now), values = snapshot.values;
  const used = crmRows(snapshot);
  const ids = used.map(r => r[0]);
  const existing = new Set(used.map(r => identity({organization:r[1],site:r[3],location:r[17],task:r[14]})));
  const duplicate = c => existing.has(identity(c)) || used.some(r=>!normalized(r[17])
      && normalized(r[1])===normalized(c.organization) && normalized(r[3])===normalized(c.site)
      && normalized(r[14])===normalized(c.task));
  if (d.payload.candidates.some(duplicate)) fail('publication_crm_duplicate_changed');
  // A hypothesis the CRM already holds is left out of a fresh plan, never failing the verified rows;
  // a replayed plan was made from the same CRM values and so never names one.
  const planned = plannedHypotheses(row,d,now,options);
  if (options.hypothesisKeys!==undefined && planned.some(h=>duplicate(h.candidate))) fail('publication_plan_binding_invalid');
  const hypotheses = planned.filter(h=>!duplicate(h.candidate));
  let sequence = Math.max(0, ...ids.map(id => Number(id.slice(3))));
  const marker = `[${d.key};${d.payload_digest}]`;
  // The existing 19 columns: verified rows are unchanged; a hypothesis changes only G, M and Q, after them.
  const sheetRow = (c, verification, nextAction, maturity) => {
    if (++sequence > 999999) fail('publication_crm_id_capacity');
    const task = c.evidence.find(e => e.role === 'task'), capability = c.evidence.find(e => e.role === 'capability');
    if (!task || !capability && c.potential_robot_match!=='unknown' || !['unqualified','needs_review'].includes(c.qualification_status))
      fail('publication_candidate_scope_invalid');
    return [`BP-${String(sequence).padStart(6,'0')}`,c.organization,'Facility / site',c.site,'','',
      verification,'',c.potential_robot_match,task.url,'Research','',nextAction+'\n'+marker,'',c.task,
      capability?.url || '',maturity,c.location,row.date];
  };
  const rows = [...d.payload.candidates.map(c => sheetRow(c,'Needs recheck',c.proposed_next_action,'Unverified')),
    ...hypotheses.map(h => sheetRow(h.candidate,HYPOTHESIS_LABEL,'First email asks: '+h.open_questions[0],HYPOTHESIS_MATURITY))];
  const body = JSON.stringify({majorDimension:'ROWS',values:rows});
  return {destination:'sheets',key:d.key,payload_digest:d.payload_digest,body_json:body,request_digest:sha(body),
    marker,crm_values:values,sheet_rows:rows,
    ...(d.payload.hypotheses?.length?{hypothesis_keys:hypotheses.map(h=>h.candidate.candidate_key)}:{})};
}

export function planNotion(row,{legacy=false,now=Date.now(),hypothesisKeys,withheld}={}) {
  const d = bind(row,'notion',now), marker = `${d.key};${d.payload_digest}`;
  const hypotheses = plannedHypotheses(row,d,now,{hypothesisKeys,withheld});
  const recorded = d.payload.hypotheses?.length?{hypothesis_keys:hypotheses.map(h=>h.candidate.candidate_key)}:{};
  const evidence = c => c.evidence.map(e => `${e.claim_kind}/${e.classification}: ${e.claim}\n${e.url}\nChecked: ${e.checked_date}`).join('\n');
  const text = `Blueprint research ${row.date}\n${d.payload.summary}\n\n`+
    [...d.payload.candidates.map(c => `${c.organization} — ${c.site}\nTask: ${c.task}\nStatus: ${c.qualification_status}\n`+
      `Unknowns: ${c.unknowns.join('; ')}\nNext proposed action: ${c.proposed_next_action}\n`+evidence(c)),
    ...hypotheses.map(({candidate:c,open_checks:checks,open_questions:questions}) =>
      `${HYPOTHESIS_HEADING}: ${c.organization} — ${c.site}\nTask: ${c.task}\nOpen checks: ${checks.join('; ')}\n`+
      `First email asks: ${questions[0]}\nDraft only; no send is authorized.\n`+
      `Unknowns: ${c.unknowns.join('; ')}\n`+evidence(c))].join('\n\n');
  const title = `Blueprint research ${row.date} ${d.payload_digest.slice(0,12)}`;
  const oldParagraphs=[marker,...Array.from({length:Math.ceil(text.length/1800)},(_,i)=>text.slice(i*1800,(i+1)*1800))];
  const oldBody=JSON.stringify(createBody(title,oldParagraphs.map(paragraph)));
  const paragraphs=[marker,...textParagraphs(text)];
  // Replay old persisted plans byte-for-byte. New plans paginate only when needed.
  if(legacy || oldParagraphs.length<=NOTION_BATCH_BLOCKS && Buffer.byteLength(oldBody)<=NOTION_REQUEST_BYTES
      && isDeepStrictEqual(oldParagraphs,paragraphs)) {
    if(oldParagraphs.length>NOTION_BATCH_BLOCKS) fail('publication_report_too_large');
    return {destination:'notion',key:d.key,payload_digest:d.payload_digest,...(d.presentation?{presentation_digest:d.presentation.decision_digest}:{}),title,paragraphs:oldParagraphs,
      body_json:oldBody,request_digest:sha(oldBody),...recorded};
  }
  const batches=[];
  for(let start=0;start<paragraphs.length;) {
    const children=[];let body;
    while(start+children.length<paragraphs.length && children.length<NOTION_BATCH_BLOCKS) {
      const next=[...children,paragraph(paragraphs[start+children.length])];
      const candidate=JSON.stringify(start===0 ? createBody(title,next) : {children:next});
      if(Buffer.byteLength(candidate)>NOTION_REQUEST_BYTES) break;
      children.push(next.at(-1));body=candidate;
    }
    if(!children.length) fail('publication_notion_request_too_large');
    batches.push({number:batches.length,start,end:start+children.length,body_json:body,request_digest:sha(body)});
    start+=children.length;
  }
  return {destination:'notion',key:d.key,payload_digest:d.payload_digest,...(d.presentation?{presentation_digest:d.presentation.decision_digest}:{}),title,paragraphs,
    protocol:'notion-paginated-v1',batches,body_json:batches[0].body_json,request_digest:batches[0].request_digest,...recorded};
}

// Host-owned site-screen admission (tools/daily_research/screen_admission.py, design section 3). One row per admitted
// site in the existing 19 columns, labelled Hypothesis exactly like a daily hypothesis row (G, M and Q); E, F and H
// carry the chosen recipient with its source label. M ends with the row's own marker [screen:<admission_id>;<digest>].
export const SCREEN_PAYLOAD='blueprint.site-screen-sheets-payload.v1';
const SCREEN_ENTRY_FIELDS=['checked_on','contact','identity','location','organization','question','result_digest','site',
  'site_key','task','task_url'];
const hex64=x=>typeof x==='string' && /^[a-f0-9]{64}$/.test(x);
const exactKeys=(value,keys)=>!!value && typeof value==='object' && !Array.isArray(value)
  && JSON.stringify(Object.keys(value).sort())===JSON.stringify([...keys].sort());
export const screenMarker=(admissionId,resultDigest)=>`[screen:${admissionId};${resultDigest}]`;
// Python canonical JSON (sorted keys, compact, non-ASCII escaped): runner.digest of the same value.
export const pythonDigest=value=>{
  const canonical=item=>Array.isArray(item)?item.map(canonical):item && typeof item==='object'
    ?Object.fromEntries(Object.keys(item).sort().map(key=>[key,canonical(item[key])])):item;
  return sha(JSON.stringify(canonical(value)).replace(/[^\x00-\x7F]/g,c=>'\\u'+c.charCodeAt(0).toString(16).padStart(4,'0')));
};
export function screenPayloadProblem(payload,admissionId) {
  const text=x=>typeof x==='string' && x.trim()!=='' && x.length<=2000;
  if(!exactKeys(payload,['admission_id','duplicates','entries','schema_version','sheet_id','tab'])
      || payload.schema_version!==SCREEN_PAYLOAD || payload.admission_id!==admissionId || !hex64(admissionId)
      || payload.sheet_id!==SHEET || payload.tab!=='Prospects' || !Array.isArray(payload.entries) || payload.entries.length>50
      || !Array.isArray(payload.duplicates) || payload.entries.length+payload.duplicates.length>50
      || payload.entries.length+payload.duplicates.length<1) return 'screen_admission_payload_invalid';
  const sites=new Set();
  for(const entry of payload.entries) {
    if(!exactKeys(entry,SCREEN_ENTRY_FIELDS) || !hex64(entry.site_key) || !hex64(entry.result_digest) || !hex64(entry.identity)
        || !['organization','site','location','task','task_url','question'].every(key=>text(entry[key]))
        || !/^\d{4}-\d{2}-\d{2}$/.test(entry.checked_on) || entry.question.indexOf('?')!==entry.question.length-1
        || !exactKeys(entry.contact,['details','name','source_url'])
        || !Object.values(entry.contact).every(value=>typeof value==='string' && value.length<=2000) || sites.has(entry.site_key))
      return 'screen_admission_payload_invalid';
    sites.add(entry.site_key);
  }
  for(const duplicate of payload.duplicates) {
    if(!exactKeys(duplicate,['code','site_key']) || !hex64(duplicate.site_key) || sites.has(duplicate.site_key)
        || typeof duplicate.code!=='string' || !/^screen_admission_[a-z_]{1,80}$/.test(duplicate.code)) return 'screen_admission_payload_invalid';
    sites.add(duplicate.site_key);
  }
  return null;
}

/** The Sheets plan of one screen admission against a fresh CRM snapshot. The publisher's structural duplicate rule
 * still holds: a row the CRM gained since the worker's dedupe refuses the plan, which is made again on a later pass. */
export function planScreenSheets(payload,snapshot) {
  const problem=screenPayloadProblem(payload,payload?.admission_id);
  if(problem) fail(problem);
  const values=snapshot.values,used=crmRows(snapshot),ids=used.map(r=>r[0]);
  const existing=new Set(used.map(r=>identity({organization:r[1],site:r[3],location:r[17],task:r[14]})));
  const duplicate=c=>existing.has(identity(c)) || used.some(r=>!normalized(r[17])
      && normalized(r[1])===normalized(c.organization) && normalized(r[3])===normalized(c.site)
      && normalized(r[14])===normalized(c.task));
  if(payload.entries.some(duplicate)) fail('screen_admission_crm_duplicate_changed');
  const prefix=`[screen:${payload.admission_id};`;
  if(used.some(r=>typeof r[12]==='string' && r[12].includes(prefix))) fail('screen_admission_readback_conflict');
  let sequence=Math.max(0,...ids.map(id=>Number(id.slice(3))));
  const rows=payload.entries.map(e=>{
    if(++sequence>999999) fail('publication_crm_id_capacity');
    return [`BP-${String(sequence).padStart(6,'0')}`,e.organization,'Facility / site',e.site,e.contact.name,e.contact.details,
      HYPOTHESIS_LABEL,e.contact.source_url,'unknown',e.task_url,'Research','',
      `First email asks: ${e.question}\n${screenMarker(payload.admission_id,e.result_digest)}`,'',e.task,'',HYPOTHESIS_MATURITY,
      e.location,e.checked_on];
  });
  const body=JSON.stringify({majorDimension:'ROWS',values:rows});
  return {destination:'sheets',kind:'screen',admission_id:payload.admission_id,payload_digest:pythonDigest(payload),
    marker:prefix,body_json:body,request_digest:sha(body),crm_values:values,sheet_rows:rows};
}

export class Publisher {
  constructor({crmReader,google,notion,clock=Date.now}) {Object.assign(this,{crmReader,google,notion,clock});}
  async sheetsTargetEmpty(range) {
    // Empty display values can conceal formulas or validation-backed cells.
    const meta = await this.google('GET',`?ranges=${encodeURIComponent(range)}`+
      '&includeGridData=true&fields=sheets(data(rowData(values(userEnteredValue,dataValidation))))');
    for (const sheet of meta.sheets || []) for (const grid of sheet.data || []) for (const row of grid.rowData || [])
      for (const cell of row.values || []) if (cell.userEnteredValue || cell.dataValidation)
        fail('publication_target_cells_not_plain_empty');
  }
  // ``withheld`` names why every hypothesis on this row stays out of the plan (an unbound outreach
  // record, or a verified-rows-only recovery); the verified rows are planned as always.
  async prepare(row,destination,{withheld=null}={}) {
    if (destination === 'notion') {
      const parent = await this.notion('GET',`/pages/${NOTION}`);
      if (parent.id?.replaceAll('-','') !== NOTION || parent.object !== 'page') fail('publication_notion_parent_mismatch');
      return planNotion(row,{now:this.clock(),withheld});
    }
    const snapshot = await this.crmReader(), plan = planSheets(row,snapshot,this.clock(),{withheld});
    const first = snapshot.values.length+1, last = first+Math.max(0,plan.sheet_rows.length-1);
    if (plan.sheet_rows.length) await this.sheetsTargetEmpty(`Prospects!A${first}:S${last}`);
    return plan;
  }
  validate(row,destination,plan,{reconcile=false}={}) {
    const now=reconcile ? null : this.clock(),d = bind(row,destination,now);
    if (plan.destination !== destination || plan.key !== d.key || plan.payload_digest !== d.payload_digest
        || plan.request_digest !== sha(plan.body_json)) fail('publication_plan_binding_invalid');
    const hypothesisKeys = plan.hypothesis_keys;
    const expected = destination === 'notion' ? planNotion(row,{legacy:!plan.protocol,now,hypothesisKeys})
      : planSheets(row,{sheet_id:SHEET,complete:true,values:plan.crm_values},now,{hypothesisKeys});
    if (!isDeepStrictEqual(expected,plan)) fail('publication_plan_binding_invalid');
  }
  async write(row,destination,plan,{beforeWrite}={}) {
    this.validate(row,destination,plan);
    if (destination === 'notion') {
      if(plan.protocol) fail('publication_paginated_batch_required');
      const body=JSON.parse(plan.body_json);
      if(body.children.length>100 || Buffer.byteLength(plan.body_json)>500000) fail('publication_notion_request_too_large');
      if(beforeWrite) await beforeWrite();
      requirePublicationVerification(row,destination,this.clock());
      await this.notion('POST','/pages',body);
    } else if (plan.sheet_rows.length) {
      const current = await this.crmReader();
      if (!isDeepStrictEqual(current.values,plan.crm_values)) fail('publication_crm_changed_before_write');
      const first=plan.crm_values.length+1, last=first+plan.sheet_rows.length-1, range=`Prospects!A${first}:S${last}`;
      await this.sheetsTargetEmpty(range);
      if(beforeWrite) await beforeWrite();
      requirePublicationVerification(row,destination,this.clock());
      await this.google('PUT',`/values/${encodeURIComponent(range)}?valueInputOption=RAW`,JSON.parse(plan.body_json));
    }
  }
  async reconcile(row,destination,plan) {
    this.validate(row,destination,plan,{reconcile:true});
    if (destination === 'sheets') {
      const snapshot = await this.crmReader();
      const used = crmRows(snapshot);
      const marked = snapshot.values.slice(5).filter(r=>typeof r[12]==='string' && r[12].includes(plan.marker));
      if (!marked.length && plan.sheet_rows.length) return null;
      if (!isDeepStrictEqual(marked.map(r=>r.slice(0,19)),plan.sheet_rows)) fail('publication_readback_conflict');
      const ids = used.map(r=>r[0]);
      if (new Set(ids).size!==ids.length) fail('publication_crm_id_collision');
      return {destination,key:plan.key,payload_digest:plan.payload_digest,readback_verified:true,
        reference:`sheets:${SHEET}:Prospects:${plan.sheet_rows.map(r=>r[0]).join(',') || 'no_candidates'}`};
    }
    const progress=await this.notionProgress(row,plan);
    return progress.receipt;
  }
  async notionProgress(row,plan) {
    this.validate(row,'notion',plan,{reconcile:true});
    const receipt=pageId=>({destination:'notion',key:plan.key,payload_digest:plan.payload_digest,
      readback_verified:true,reference:`notion:${pageId}`});
    const step=(number,pageId=null)=>({...plan.batches[number],page_id:pageId});
    const matches = [], seen = new Set(), deadline = this.clock()+NOTION_READ_BUDGET_MS;
    const read = async path => {
      const remaining = deadline-this.clock();
      if (remaining <= 0) fail('publication_notion_parent_scan_incomplete');
      const result = await this.notion('GET',path,undefined,{timeoutMs:Math.min(12000,remaining)});
      if (this.clock() >= deadline) fail('publication_notion_parent_scan_incomplete');
      return result;
    };
    let cursor = null;
    for (let page=0;page<NOTION_PARENT_PAGE_LIMIT;page++) {
      const children = await read(`/blocks/${NOTION}/children?page_size=100${cursor?'&start_cursor='+encodeURIComponent(cursor):''}`);
      if (typeof children.has_more !== 'boolean' || !Array.isArray(children.results) || children.results.length>100)
        fail('publication_notion_pagination_invalid');
      for (const block of children.results) if (block.type==='child_page' && block.child_page?.title===plan.title) matches.push(block.id);
      if (!children.has_more) {cursor=null;break;}
      cursor=children.next_cursor;
      if (typeof cursor !== 'string' || !cursor || cursor.length>2048 || seen.has(cursor))
        fail('publication_notion_pagination_invalid');
      seen.add(cursor);
    }
    if (cursor) fail('publication_notion_parent_scan_incomplete');
    if (matches.length>1) fail('publication_notion_duplicate_pages');
    if (!matches.length) return {receipt:null,step:plan.protocol ? step(0) : null,absence_verified:true};
    if(typeof matches[0]!=='string' || !/^[A-Za-z0-9-]{1,200}$/.test(matches[0])) fail('publication_notion_pagination_invalid');
    const page = await read(`/pages/${matches[0]}`);
    if (page.parent?.page_id?.replaceAll('-','')!==NOTION) fail('publication_notion_parent_mismatch');
    const paragraphs=[],blockIds=new Set();cursor=null;seen.clear();
    const pages=plan.protocol ? Math.ceil(plan.paragraphs.length/100)+1 : 1;
    for(let index=0;index<pages;index++) {
      const content=await read(`/blocks/${matches[0]}/children?page_size=100${cursor?'&start_cursor='+encodeURIComponent(cursor):''}`);
      if(typeof content.has_more!=='boolean' || !Array.isArray(content.results) || content.results.length>100)
        fail('publication_notion_pagination_invalid');
      if(plan.protocol) for(const block of content.results) {
        if(typeof block.id!=='string' || !block.id || blockIds.has(block.id)) fail('publication_notion_pagination_invalid');
        blockIds.add(block.id);
      }
      paragraphs.push(...content.results.map(b=>b.type==='paragraph' && Array.isArray(b.paragraph?.rich_text)
        ? b.paragraph.rich_text.map(x=>x.plain_text??x.text?.content??'').join('') : null));
      if(paragraphs.length>plan.paragraphs.length || !isDeepStrictEqual(paragraphs,plan.paragraphs.slice(0,paragraphs.length)))
        fail('publication_readback_conflict');
      if(!content.has_more) {cursor=null;break;}
      cursor=content.next_cursor;
      if(typeof cursor!=='string' || !cursor || cursor.length>2048 || seen.has(cursor)) fail('publication_notion_pagination_invalid');
      seen.add(cursor);
    }
    if(cursor) fail('publication_notion_report_scan_incomplete');
    if(paragraphs.length===plan.paragraphs.length) return {receipt:receipt(matches[0]),step:null};
    if(!plan.protocol) fail('publication_readback_conflict');
    const next=plan.batches.find(batch=>batch.start===paragraphs.length);
    if(!next || next.number===0) fail('publication_notion_partial_batch_unresolved');
    return {receipt:null,step:step(next.number,matches[0])};
  }
  async writeNotionStep(row,plan,step) {
    this.validate(row,'notion',plan);
    const expected=plan.batches?.[step.number];
    if(!expected || !isDeepStrictEqual({...expected,page_id:step.page_id},step)
        || step.number===0 && step.page_id!==null
        || step.number>0 && (typeof step.page_id!=='string' || !/^[A-Za-z0-9-]{1,200}$/.test(step.page_id)))
      fail('publication_plan_binding_invalid');
    requirePublicationVerification(row,'notion',this.clock());
    await this.notion(step.number===0?'POST':'PATCH',step.number===0?'/pages':`/blocks/${step.page_id}/children`,JSON.parse(step.body_json));
  }
  // Site-screen admission rows (planScreenSheets). Prepare from a fresh CRM read that must equal the snapshot the
  // worker deduplicated against; write once after the claim; reconcile by reading only.
  async prepareScreen(payload,crmValuesDigest) {
    const snapshot=await this.crmReader();
    if(!hex64(crmValuesDigest) || pythonDigest(snapshot.values)!==crmValuesDigest) fail('screen_admission_crm_changed');
    const plan=planScreenSheets(payload,snapshot);
    if(plan.sheet_rows.length) await this.sheetsTargetEmpty(`Prospects!A${snapshot.values.length+1}:S${snapshot.values.length+plan.sheet_rows.length}`);
    return plan;
  }
  validateScreen(payload,plan) {
    const expected=planScreenSheets(payload,{sheet_id:SHEET,complete:true,values:plan?.crm_values});
    if(!isDeepStrictEqual(expected,plan)) fail('screen_admission_plan_binding_invalid');
  }
  async writeScreen(payload,plan,{beforeWrite}={}) {
    this.validateScreen(payload,plan);
    if(!plan.sheet_rows.length) return;
    const current=await this.crmReader();
    if(!isDeepStrictEqual(current.values,plan.crm_values)) fail('publication_crm_changed_before_write');
    const first=plan.crm_values.length+1,last=first+plan.sheet_rows.length-1,range=`Prospects!A${first}:S${last}`;
    await this.sheetsTargetEmpty(range);
    if(beforeWrite) await beforeWrite();
    await this.google('PUT',`/values/${encodeURIComponent(range)}?valueInputOption=RAW`,JSON.parse(plan.body_json));
  }
  async reconcileScreen(payload,plan) {
    this.validateScreen(payload,plan);
    const snapshot=await this.crmReader(),used=crmRows(snapshot);
    const marked=snapshot.values.slice(5).filter(r=>typeof r[12]==='string' && r[12].includes(plan.marker));
    if(!marked.length && plan.sheet_rows.length) return null;
    if(!isDeepStrictEqual(marked.map(r=>r.slice(0,19)),plan.sheet_rows)) fail('publication_readback_conflict');
    const ids=used.map(r=>r[0]);
    if(new Set(ids).size!==ids.length) fail('publication_crm_id_collision');
    return {destination:'sheets',kind:'screen',admission_id:plan.admission_id,payload_digest:plan.payload_digest,
      readback_verified:true,reference:`sheets:${SHEET}:Prospects:${plan.sheet_rows.map(r=>r[0]).join(',') || 'no_candidates'}`};
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
  const notion = async(method,path,body,{timeoutMs=12000}={})=>{
    if (!token) fail('publication_notion_binding_absent');
    const response=await fetch('https://api.notion.com/v1'+path,{method,redirect:'error',signal:AbortSignal.timeout(timeoutMs),
      headers:{Authorization:'Bearer '+token,'Notion-Version':'2022-06-28','Content-Type':'application/json'},
      ...(body?{body:JSON.stringify(body)}:{})});
    if (!response.ok) {
      const raw=await response.text();
      const error=new Error(response.status===401||response.status===403?'publication_notion_permission_denied':'publication_notion_unavailable');
      let detail;try {detail=JSON.parse(raw);}catch {}
      error.provider_feedback={provider:'notion',http_status:response.status,
        code:typeof detail?.code==='string' && /^[a-z_]{1,80}$/.test(detail.code)?detail.code:'unknown',
        message:typeof detail?.message==='string'?detail.message.slice(0,4000):null,
        request_digest:body?sha(JSON.stringify(body)):null,request_bytes:body?Buffer.byteLength(JSON.stringify(body)):0,
        response_digest:sha(raw),response_bytes:Buffer.byteLength(raw)};
      // Preserve bounded raw provider evidence privately; it never enters plans or logs.
      if(Buffer.byteLength(raw)<=2000000) error.provider_response=raw;
      throw error;
    }
    const raw=await response.text(); if (Buffer.byteLength(raw)>2000000) fail('publication_notion_response_too_large');
    return JSON.parse(raw);
  };
  return new Publisher({crmReader,google,notion});
}
