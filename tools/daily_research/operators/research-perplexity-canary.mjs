// One-time namespace adapter; all durable claims/files/leases use the reviewed Store.
import {createInterface} from 'node:readline';
import {isDeepStrictEqual} from 'node:util';
import {createRequire} from 'node:module';
import {pathToFileURL} from 'node:url';

const PACKAGE = '__RESEARCH_PACKAGE_URL__';
const {Store, LeaseChannel, ROOT, readCanonicalCRM} = await import(PACKAGE+'tools/daily_research/firestore_bridge.mjs');
const {livePublisher} = await import(PACKAGE+'tools/daily_research/publisher.mjs');
export const TEST = 'perplexity-fast-20261001';
export const CANARY = `${ROOT}/canaries/${TEST}`;
const SOURCE = '35f5c9ad43f84aa053aa7616a63a9aa4f6e32a61';
const fail = code => {throw new Error(code);};
const normalized = control => Object.fromEntries(Object.entries(control || {}).filter(([key])=>key!=='lease'));

export function scopedDatabase(db) {
  const path = p => {
    if (p===CANARY || p.startsWith(CANARY+'/')) return p;
    if (p===ROOT || p.startsWith(ROOT+'/')) return CANARY+p.slice(ROOT.length);
    fail('canary_path_outside_namespace');
  };
  return {doc:p=>db.doc(path(p)), collection:p=>db.collection(path(p)),
    runTransaction:(fn,options)=>db.runTransaction(fn,options)};
}

export class CanaryChannel {
  constructor(db, crmReader=null, publisher=null, clock=()=>Date.now()) {
    this.db=db; this.clock=clock;
    this.normal=new Store(db,clock);
    this.store=new Store(scopedDatabase(db),clock,undefined,crmReader,publisher);
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
  async call(request) {
    if ((request.day!==undefined && request.day!=='2026-10-01')
      || (request.op==='put' && request.row?.date!=='2026-10-01')) fail('canary_date_scope_invalid');
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
      await this.guard(binding.origin_control,binding.origin_row_blob);
      // The only new document: private disabled test control, never normal control.
      return this.db.runTransaction(async tx=>{
        const normal=await tx.get(this.normal.control), row=await tx.get(this.db.doc(`${ROOT}/runs/2026-10-01`)),
          prior=await tx.get(this.store.control);
        if (!isDeepStrictEqual(normalized(normal.data()),binding.origin_control)) fail('canary_origin_changed');
        if (row.data()?.blob!==binding.origin_row_blob) fail('canary_origin_changed');
        if (prior.exists) {
          if (!isDeepStrictEqual(normalized(prior.data()).canary,binding)) fail('canary_admission_already_bound');
          return false;
        }
        tx.set(this.store.control,candidate);return true;
      });
    }
    if (request.op==='guard') {
      const control=(await this.store.control.get()).data();
      await this.guard(control?.canary?.origin_control,control?.canary?.origin_row_blob);return true;
    }
    // Read/recovery/lease operations stay available if normal control changes.
    // New paid work and publication require the same fresh disabled-root guard.
    if (['create_check','qa_check','publish','refresh_crm','configure'].includes(request.op)) {
      const control=(await this.store.control.get()).data();
      await this.guard(control?.canary?.origin_control,control?.canary?.origin_row_blob);
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
