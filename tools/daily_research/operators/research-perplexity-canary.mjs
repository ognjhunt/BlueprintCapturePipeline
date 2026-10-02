// One-time namespace adapter; all durable claims/files/leases use the reviewed Store.
import {createInterface} from 'node:readline';
import {isDeepStrictEqual} from 'node:util';
import {createRequire} from 'node:module';
import {pathToFileURL} from 'node:url';

const PACKAGE = '__RESEARCH_PACKAGE_URL__';
const {Store, LeaseChannel, ROOT, readCanonicalCRM} = await import(PACKAGE+'tools/daily_research/firestore_bridge.mjs');
const {livePublisher} = await import(PACKAGE+'tools/daily_research/publisher.mjs');
const CONTEXT = __CANARY_CONTEXT__;
export const TEST = CONTEXT.test_id;
const DAY = CONTEXT.day;
const BASELINE = CONTEXT.baseline;
const BASELINE_SCOPE = BASELINE ? {baseline_id:'baseline-20261002',
  root:`${ROOT}/baselines/baseline-20261002`,authority_reference:'Sentinel_c2046c5f146c81918921eba1ed7f6caa',soft_total_usd:25}:null;
if (BASELINE && (!Number.isInteger(BASELINE.attempt_number) || BASELINE.attempt_number<1
  || TEST!==`baseline-20261002-attempt-${String(BASELINE.attempt_number).padStart(4,'0')}`
  || !isDeepStrictEqual({...BASELINE,attempt_number:undefined},{...BASELINE_SCOPE,attempt_number:undefined})))
  throw new Error('canary_baseline_binding_invalid');
export const CANARY = `${ROOT}/canaries/${TEST}`;
const SOURCE = '35f5c9ad43f84aa053aa7616a63a9aa4f6e32a61';
const fail = code => {throw new Error(code);};
const normalized = control => Object.fromEntries(Object.entries(control || {}).filter(([key])=>key!=='lease'));

export function scopedDatabase(db,root=CANARY) {
  const path = p => {
    if (p===root || p.startsWith(root+'/')) return p;
    if (p===ROOT || p.startsWith(ROOT+'/')) return root+p.slice(ROOT.length);
    fail('canary_path_outside_namespace');
  };
  return {doc:p=>db.doc(path(p)), collection:p=>db.collection(path(p)),
    runTransaction:(fn,options)=>db.runTransaction(fn,options)};
}

export class CanaryChannel {
  constructor(db, crmReader=null, publisher=null, clock=()=>Date.now()) {
    this.db=db; this.clock=clock;
    this.normal=new Store(db,clock);
    const publishRow=row=>({...row,run_key:`blueprint-research-canary:${TEST}`});
    const privatePublisher=BASELINE && publisher?Object.fromEntries(['prepare','write','reconcile'].map(method=>
      [method,(row,...args)=>publisher[method](publishRow(row),...args)])):publisher;
    this.store=new Store(scopedDatabase(db),clock,undefined,crmReader,privatePublisher);
    this.channel=new LeaseChannel(this.store);
  }
  async origin() {
    const control=(await this.normal.control.get()).data();
    const summary=await this.normal.summary(), qa=await this.normal.activeQA();
    const manifest=await this.db.doc(`${ROOT}/runs/2026-10-01`).get();
    const row=await this.normal.get('2026-10-01');
    return {control:normalized(control),summary,active_qa:qa,oct1_row:row,oct1_row_blob:manifest.data()?.blob};
  }
  async guard(expected,rowBlob) {
    const origin=await this.origin();
    if (!origin.control || origin.control.enabled!==false || origin.control.config?.enabled!==false
      || origin.control.source_commit!==SOURCE
      || origin.summary.unfinished || origin.summary.cleanup_required || origin.active_qa
      || !origin.oct1_row || origin.oct1_row.state!=='failed' || origin.oct1_row.cleanup_required!==false
      || !origin.oct1_row.cleanup_receipt
      || typeof rowBlob!=='string' || origin.oct1_row_blob!==rowBlob
      || !isDeepStrictEqual(origin.control,expected)) fail('canary_daily_guard_unreconciled_or_changed');
    return true;
  }
  async baselineState() {
    if (!BASELINE) fail('canary_baseline_not_selected');
    const ref=this.db.doc(BASELINE.root), snap=await ref.get(), budget=snap.exists?snap.data():null;
    if (budget && (!isDeepStrictEqual(budget.baseline,BASELINE_SCOPE)
      || !Array.isArray(budget.attempts) || budget.attempts.length>1000)) fail('canary_baseline_binding_invalid');
    if (budget?.attempts.some((a,i)=>a.number!==i+1 || !/^\d{4}-\d{2}-\d{2}$/.test(a.date)
      || a.test_id!==`baseline-20261002-attempt-${String(i+1).padStart(4,'0')}`
      || a.root!==`${ROOT}/canaries/${a.test_id}`
      || (a.state!==undefined && a.state!=='abandoned_unstarted'))) fail('canary_baseline_binding_invalid');
    return {ref,snap,budget};
  }
  async baselineCheck() {
    const current=await this.baselineState(), attempts=current.budget?.attempts||[];
    const own=attempts.find(a=>a.number===BASELINE.attempt_number);
    if (own) {
      if (own.state==='abandoned_unstarted') fail('canary_baseline_attempt_abandoned');
      if (own.test_id!==TEST || own.date!==DAY || attempts.at(-1)?.number!==own.number)
        fail('canary_baseline_attempt_identity_changed');
      if (!(await this.store.control.get()).exists) fail('canary_baseline_attempt_control_missing');
      return {...current,predecessor:null};
    }
    if (BASELINE.attempt_number!==attempts.length+1) fail('canary_baseline_attempt_order_invalid');
    const previous=attempts.at(-1)||{root:`${ROOT}/canaries/perplexity-fast-20261001`,date:'2026-10-01'};
    const store=new Store(scopedDatabase(this.db,previous.root),this.clock);
    const manifest=await this.db.doc(`${previous.root}/runs/${previous.date}`).get();
    const row=await store.get(previous.date);
    if (previous.state==='abandoned_unstarted') {
      const control=await store.control.get(), data=control.data();
      if (manifest.exists || !control.exists || data.enabled!==false || data.config?.enabled!==false
        || data.workflow?.enabled!==false || data.lease?.expires_at_ms>this.clock()
        || data.baseline_attempt_state!=='abandoned_unstarted') fail('canary_baseline_abandonment_changed');
      return {...current,predecessor:{ref:manifest.ref,exists:false,blob:null,
        control_ref:store.control,control_data:data,root:previous.root,date:previous.date}};
    }
    if (row?.state==='completed' && row.packet?.coverage?.completion_state==='coverage_complete'
      && row.qa?.state==='validated' && row.turn_status==='completed' && row.qa.turn_status==='completed'
      && ['sheets','notion'].every(name=>row.delivery?.[name]?.receipt?.readback_verified===true))
      fail('canary_baseline_already_completed');
    if (!row || !['failed','cancelled','awaiting_review','reviewed','completed'].includes(row.state)
      || row.cleanup_required!==false || !row.cleanup_receipt?.action_time_approval_reference
      || (row.qa && !['validated','qa_blocked'].includes(row.qa.state))
      || Object.values(row.delivery||{}).some(d=>d.state!=='acknowledged'))
      fail('canary_baseline_predecessor_unreconciled');
    if (row.canary?.test_id!==(attempts.at(-1)?.test_id||'perplexity-fast-20261001'))
      fail('canary_baseline_predecessor_changed');
    return {...current,predecessor:{ref:manifest.ref||this.db.doc(`${previous.root}/runs/${previous.date}`),
      exists:true,blob:manifest.data()?.blob,root:previous.root,date:previous.date}};
  }
  async abandonUnstarted() {
    const before=await this.baselineState();
    return this.db.runTransaction(async tx=>{
      const control=await tx.get(this.store.control), row=await tx.get(this.db.doc(`${CANARY}/runs/${DAY}`)),
        saved=await tx.get(before.ref), budget=saved.data(), c=control.data(), latest=budget?.attempts?.at(-1);
      if (!isDeepStrictEqual(budget?.baseline,BASELINE_SCOPE) || latest?.test_id!==TEST || latest?.date!==DAY
        || !control.exists || !isDeepStrictEqual(c.canary?.baseline,BASELINE) || c.canary?.run_date!==DAY
        || row.exists) fail('canary_baseline_abandonment_not_proven');
      if (c.lease?.expires_at_ms>this.clock()) fail('runner_overlap');
      if (latest.state==='abandoned_unstarted') {
        if (c.enabled!==false || c.baseline_attempt_state!=='abandoned_unstarted') fail('canary_baseline_abandonment_changed');
        return {state:'abandoned_unstarted',test_id:TEST,provider_mutations:0,existing:true};
      }
      // Absence of the durable intent proves no create claim/POST exists.
      // Disable under the SAME transaction so an expired writer cannot stage
      // a first intent afterward. No provider row or cleanup history is made.
      tx.set(this.store.control,{...c,enabled:false,config:{...c.config,enabled:false},
        workflow:{...c.workflow,enabled:false},baseline_attempt_state:'abandoned_unstarted'});
      tx.set(before.ref,{...budget,attempts:budget.attempts.map(a=>a.test_id===TEST?
        {...a,state:'abandoned_unstarted',no_provider_intent:true,abandoned_at_ms:this.clock()}:a)});
      return {state:'abandoned_unstarted',test_id:TEST,provider_mutations:0,existing:false};
    });
  }
  async baselineStatus() {
    const {budget}=await this.baselineState(), attempts=[];
    let reported=0, known=true, any=false;
    let searchReported=0, searchAny=false;
    for (const a of budget?.attempts||[]) {
      const row=await new Store(scopedDatabase(this.db,a.root),this.clock).get(a.date), usage=row?.canary_model_estimate;
      if (a.state==='abandoned_unstarted') {
        if (row) fail('canary_baseline_abandonment_changed');
        attempts.push({test_id:a.test_id,date:a.date,state:a.state,provider_mutations:0,
          usage_state:'not_started_no_provider_intent',reported_estimate_usd:'0'});
        any=true;continue;
      }
      const value=usage?.reported_estimate_usd??usage?.estimate_usd;
      const usable=value!==null && value!==undefined && Number.isFinite(Number(value)) && Number(value)>=0;
      if (usable) {reported+=Number(value);any=true;}
      if (!usage?.known || !usable) known=false;
      const tools=row?.application_tool_usage, searchValue=tools?.conservative_search_cost_estimate_usd;
      if (searchValue!==null && searchValue!==undefined && Number.isFinite(Number(searchValue)) && Number(searchValue)>=0) {
        searchReported+=Number(searchValue);searchAny=true;
      }
      attempts.push({test_id:a.test_id,date:a.date,state:row?.state||'intent_not_created',
        cleanup_required:row?.cleanup_required??null,usage_state:usage?.usage_state||'pending',
        reported_estimate_usd:usable?value:null,application_tool_usage:tools||null});
    }
    return {baseline:BASELINE_SCOPE,attempts,reported_model_estimate_usd:any?String(reported):null,
      complete_model_estimate_usd:known&&attempts.length?String(reported):null,
      reported_search_estimate_usd:searchAny?String(searchReported):null,
      total_billed_usd:null,all_attempts_share_one_allowance:true,
      tool_and_environment_costs:'unknown_until_billing_reconciled',hard_total_cap:false,
      prior_scope_included:false,source_refresh_performed:false};
  }
  async call(request) {
    if ((request.day!==undefined && request.day!==DAY)
      || (request.op==='put' && request.row?.date!==DAY)) fail('canary_date_scope_invalid');
    if (request.op==='baseline_check') {await this.baselineCheck();return true;}
    if (request.op==='baseline_status') return this.baselineStatus();
    if (request.op==='abandon_unstarted') return this.abandonUnstarted();
    if (request.op==='origin') return this.origin();
    if (request.op==='origin_file') {
      if (!['2026-10-01-artifact.json','knowledge.json','refresh-policy.json'].includes(request.name))
        fail('canary_origin_file_scope_invalid');
      return this.normal.fileGet(request.name);
    }
    if (request.op==='stage') {
      const candidate=request.value, binding=candidate?.canary;
      if (candidate?.enabled!==false || candidate?.config?.enabled!==false || candidate?.workflow?.enabled!==false
        || candidate.source_commit!==SOURCE || binding?.test_id!==TEST || binding.root!==CANARY
        || binding.admission?.schema_version!=='blueprint.perplexity-canary-admission.v1'
        || binding.admission?.test_id!==TEST || binding.admission?.ceiling_usd!==25
        || typeof binding.admission.authority_reference!=='string' || !binding.admission.authority_reference.trim()
        || binding.admission.authority_reference.startsWith('PENDING')) fail('canary_stage_binding_invalid');
      if (BASELINE && (!isDeepStrictEqual(binding.baseline,BASELINE) || binding.run_date!==DAY
        || binding.admission.authority_reference!==BASELINE_SCOPE.authority_reference
        || binding.admission.scope!=='baseline-research-agent-qa-canonical-publication-with-retries-no-outreach'))
        fail('canary_baseline_binding_invalid');
      await this.guard(binding.origin_control,binding.origin_row_blob);
      const baseline=BASELINE?await this.baselineCheck():null;
      // The only new document: private disabled test control, never normal control.
      return this.db.runTransaction(async tx=>{
        const normal=await tx.get(this.normal.control), row=await tx.get(this.db.doc(`${ROOT}/runs/2026-10-01`)),
          prior=await tx.get(this.store.control);
        const budget=baseline?await tx.get(baseline.ref):null;
        const predecessor=baseline?.predecessor?await tx.get(baseline.predecessor.ref):null;
        const previousControl=baseline?.predecessor?.control_ref?await tx.get(baseline.predecessor.control_ref):null;
        if (!isDeepStrictEqual(normalized(normal.data()),binding.origin_control)) fail('canary_origin_changed');
        if (row.data()?.blob!==binding.origin_row_blob) fail('canary_origin_changed');
        if (baseline && (!isDeepStrictEqual(budget.data()||null,baseline.budget)
          || (predecessor && (predecessor.exists!==baseline.predecessor.exists
            || (predecessor.exists && predecessor.data()?.blob!==baseline.predecessor.blob)))
          || (previousControl && !isDeepStrictEqual(previousControl.data(),baseline.predecessor.control_data))))
          fail('canary_baseline_predecessor_changed');
        if (prior.exists) {
          if (!isDeepStrictEqual(normalized(prior.data()).canary,binding)) fail('canary_admission_already_bound');
          return false;
        }
        if (baseline) tx.set(baseline.ref,{baseline:BASELINE_SCOPE,attempts:[...(baseline.budget?.attempts||[]),
          {number:BASELINE.attempt_number,test_id:TEST,root:CANARY,date:DAY}],
          prior_scope:{root:`${ROOT}/canaries/perplexity-fast-20261001`,included_in_new_allowance:false}});
        tx.set(this.store.control,candidate);return true;
      });
    }
    if (request.op==='guard') {
      const control=(await this.store.control.get()).data();
      await this.guard(control?.canary?.origin_control,control?.canary?.origin_row_blob);
      if (BASELINE) {
        const {budget}=await this.baselineState();
        if (budget?.attempts.at(-1)?.test_id!==TEST || budget.attempts.at(-1).state==='abandoned_unstarted')
          fail('canary_baseline_attempt_superseded');
      }
      return true;
    }
    // Read/recovery/lease operations stay available if normal control changes.
    // New paid work and publication require the same fresh disabled-root guard.
    if (['create_check','qa_check','publish','refresh_crm','configure'].includes(request.op)) {
      const control=(await this.store.control.get()).data();
      await this.guard(control?.canary?.origin_control,control?.canary?.origin_row_blob);
      if (BASELINE) {
        const latest=(await this.baselineState()).budget?.attempts.at(-1);
        if (latest?.test_id!==TEST || latest.state==='abandoned_unstarted') fail('canary_baseline_attempt_superseded');
      }
      if (request.op==='configure' && !isDeepStrictEqual(request.value?.canary,control?.canary))
        fail('canary_admission_already_bound');
    }
    if (['adaptive_origin','adaptive_get','adaptive_put','adaptive_claim','adaptive_file_put','adaptive_file_get',
      'init','import_run'].includes(request.op)) fail('canary_operation_not_allowed');
    return this.channel.call(request);
  }
  close() {return this.channel.close();}
}

async function main() {
  const require=createRequire(PACKAGE+'tools/daily_research/firestore_bridge.mjs');
  const {initializeApp,cert}=require('firebase-admin/app');
  const {getFirestore}=require('firebase-admin/firestore');
  const account=JSON.parse(process.env.FIREBASE_SERVICE_ACCOUNT_JSON || '{}');
  if (account.project_id!=='blueprint-8c1ca') fail('firestore_project_binding_mismatch');
  const crmReader=()=>readCanonicalCRM(account);
  const publisher=await livePublisher(account,crmReader,process.env.NOTION_API_TOKEN||process.env.NOTION_API_KEY);
  const channel=new CanaryChannel(getFirestore(initializeApp({credential:cert(account)})),crmReader,publisher);
  try {
    for await (const line of createInterface({input:process.stdin})) {
      try {
        if(line.length>16*1024*1024) fail('canary_request_resource_ceiling');
        const value=await channel.call(JSON.parse(line));
        process.stdout.write(JSON.stringify({ok:true,value})+'\n');
      } catch(error) {
        const code=/^canary_[a-z_]+$/.test(error.message)||/^firestore_[a-z_]+$/.test(error.message)
          || /^(?:research_|agent_|workflow_|publication_)[a-z_]+$/.test(error.message)
          || error.message==='runner_overlap' ? error.message : 'canary_bridge_unavailable';
        process.stdout.write(JSON.stringify({ok:false,error:code})+'\n');
      }
    }
  } finally {await channel.close();}
}
if(process.argv[1] && import.meta.url===pathToFileURL(process.argv[1]).href) {
  main().catch(()=>{process.stdout.write('{"ok":false,"error":"canary_bridge_unavailable"}\n');process.exitCode=1;});
}
