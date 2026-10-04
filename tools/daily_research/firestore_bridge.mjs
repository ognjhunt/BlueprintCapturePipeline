// Private JSON-line pipe for the Python runner; stdout is not a worker log.
import {createHash, randomUUID} from 'node:crypto';
import {gzipSync, gunzipSync} from 'node:zlib';
import {createInterface} from 'node:readline';
import {pathToFileURL} from 'node:url';
import {livePublisher,publicationVerification,requirePublicationVerification} from './publisher.mjs';
import {verificationDigest} from './verification-digest.mjs';
import {contactResearchContext,claimContactResearch,finishContactResearchSafely} from './contact_research.mjs';

export const ROOT = 'blueprintDailyResearch/sites-first';
export const ADAPTIVE_TEST = 'adaptive-discovery-20261001';
const MAX_BYTES = 8 * 1024 * 1024, CHUNK = 256 * 1024, LEASE_MS = 180000;
const TERMINAL = ['awaiting_review', 'reviewed', 'completed', 'failed', 'cancelled'];
const CLEANUP_BUCKET = 'blueprint-8c1ca.appspot.com';
const sha = bytes => createHash('sha256').update(bytes).digest('hex');
const canonicalValue = value => Array.isArray(value) ? value.map(canonicalValue) : value && typeof value === 'object'
  ? Object.fromEntries(Object.keys(value).sort().map(key => [key, canonicalValue(value[key])])) : value;
const valueHash = value => sha(JSON.stringify(canonicalValue(value)));
const pythonHash = value => sha(JSON.stringify(canonicalValue(value)).replace(/[^\x00-\x7F]/g,
  c=>'\\u'+c.charCodeAt(0).toString(16).padStart(4,'0')));
const same = (a, b) => JSON.stringify(Object.entries(a || {}).sort()) === JSON.stringify(Object.entries(b || {}).sort());
class Refusal extends Error {}
const refuse = code => {throw new Refusal(code);};
const dateOK = x => typeof x === 'string' && /^\d{4}-\d{2}-\d{2}$/.test(x);
const sourceBinding=d=>valueHash({key:d.key,payload:d.payload,payload_json:d.payload_json,payload_digest:d.payload_digest});
const deliveryBinding=d=>valueHash({key:d.key,payload:d.payload,payload_json:d.payload_json,payload_digest:d.payload_digest,
  plan:d.plan||null,presentation:d.presentation||null});
const attemptState=(run,row,name)=>row.delivery[name].attempt_number
  ?run.publication_attempts?.[name]?.[row.delivery[name].attempt_number] || {}
  :{claimed:run.publication_claimed?.[name],batches:run.publication_batches?.[name] || {},
    delivery_binding:run.publication_delivery_bindings?.[name],rejection:run.publication_rejections?.[name]};
const claimUpdate=(run,row,name,digest,batches=null)=>{
  const d=row.delivery[name],binding=deliveryBinding(d);
  if(d.attempt_number) return {publication_attempts:{...run.publication_attempts,[name]:{
    ...run.publication_attempts?.[name],[d.attempt_number]:{...attemptState(run,row,name),claimed:digest,
      delivery_binding:binding,...(batches?{batches}:{})}}}};
  return {publication_claimed:{...run.publication_claimed,[name]:digest},
    publication_delivery_bindings:{...run.publication_delivery_bindings,[name]:binding},
    publication_source_bindings:{...run.publication_source_bindings,[name]:sourceBinding(d)},
    ...(batches?{publication_batches:{...run.publication_batches,[name]:batches}}:{})};
};
const fileOK = x => typeof x === 'string' && /^(?:\d{4}-\d{2}-\d{2}-(?:artifact|evidence|output|review|recovery|qa|qa-evidence|qa-input|publication-(?:input|evidence)|qa-correction-[12]-(?:input|artifact|evidence)|repair-[1-9]\d*-(?:input|artifact)|inventory-[a-f0-9]{64}-\d+|exa-(?:http-[a-f0-9]{64}|[a-f0-9]{64}-(?:start(?:-http)?|read-[a-f0-9]{64}))|tool-[A-Za-z0-9_-]{1,200})|(?:crm|knowledge|refresh-policy))\.json$/.test(x);
// Owner-directed paid expansion allowance; mirrors tools/daily_research/allocation.py.
const PAID_DIRECTION='blueprint.research-paid-expansion-direction.v1', PAID_GRANT='blueprint.research-paid-expansion-grant.v1';
const PAID_PREFIX='operations/research/paid-expansion/', PAID_SOURCES=['exa']; // TODO(FindAll): mirror allocation.SOURCES.
const PAID_FIELDS=['approval_reference','approved_by','effective_from','expires_at','issued_at','per_run_limit_usd',
  'reason','schema_version','scope','sources','supersedes','version'];
const PAID_GRANT_FIELDS=['approval_reference','direction_sha256','direction_uri','frozen_at','grant_id','limit_micros',
  'per_start_max_micros','run_key','schema_version','source_commit','sources','state','valid_until','version'];
const hexOK=(x,n=64)=>typeof x==='string' && x.length===n && /^[a-f0-9]+$/.test(x);
const keysAre=(value,keys)=>!!value && typeof value==='object' && !Array.isArray(value)
  && JSON.stringify(Object.keys(value).sort())===JSON.stringify([...keys].sort());
const paidMicros=x=>{
  if(typeof x!=='string' || !/^[1-9]\d{0,2}(?:\.\d{2})?$/.test(x)) return null;
  const [whole,cents='0']=x.split('.'),value=Number(whole)*1000000+Number(cents)*10000;
  return value>=1000000 && value<=100000000?value:null;
};
const paidPerStart=limit=>Math.min(Math.max(Math.floor(limit/2),1000000),50000000);
const paidUri=hash=>`gs://${CLEANUP_BUCKET}/${PAID_PREFIX}${hash}/direction.json`;
// Whole UTC seconds on a real calendar day; Date.parse alone rolls 11-31 into 12-01.
const paidStamp=x=>{
  if(typeof x!=='string' || !/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\+00:00$/.test(x)) return NaN;
  const ms=Date.parse(x);
  return Number.isFinite(ms) && new Date(ms).toISOString().slice(0,19)+'+00:00'===x?ms:NaN;
};
const paidText=x=>typeof x==='string' && /^[\x20-\x7e]{1,500}$/.test(x) && x.trim()!=='';
function paidDirectionProblem(d,control) {
  if(!keysAre(d,PAID_FIELDS) || d.schema_version!==PAID_DIRECTION) return 'paid_expansion_direction_invalid';
  if(paidMicros(d.per_run_limit_usd)===null) return 'paid_expansion_limit_invalid';
  const scope=d.scope,[issued,start,end]=['issued_at','effective_from','expires_at'].map(k=>paidStamp(d[k]));
  if(!keysAre(scope,['agent_id','firestore_root','project_id','run_key_prefix','timezone'])
      || scope.project_id!==control?.project_id || scope.agent_id!==control?.agent_id || scope.firestore_root!==ROOT
      || scope.run_key_prefix!=='blueprint-researcher:' || scope.timezone!=='America/Chicago') return 'paid_expansion_scope_mismatch';
  if(!Number.isSafeInteger(d.version) || d.version<1 || d.version>1000000 || (d.version===1)!==(d.supersedes===null)
      || d.supersedes!==null && !hexOK(d.supersedes) || !Array.isArray(d.sources) || !d.sources.length
      || !d.sources.every((s,i)=>PAID_SOURCES.includes(s) && (i===0 || d.sources[i-1]<s))
      || !['approval_reference','approved_by','reason'].every(k=>paidText(d[k])) || /^PENDING/i.test(d.approval_reference.trim())
      || ![issued,start,end].every(Number.isFinite) || !(issued<=start && start<end) || end-issued>366*86400000)
    return 'paid_expansion_direction_invalid';
  return null;
}

export class Store {
  constructor(db, clock = () => Date.now(), owner = randomUUID(), crmReader = null, publisher = null, learning = null,
    terminalCollectionReceipt = null, schedulerStopped = false, archiveBucket = null) {
    this.db = db; this.clock = clock; this.owner = owner; this.generation = null;
    this.control = db.doc(ROOT);
    this.crmReader = crmReader;
    this.publisher = publisher;
    this.learning = learning;
    this.terminalCollectionReceipt = terminalCollectionReceipt;
    this.schedulerStopped = schedulerStopped;
    this.archiveBucket = archiveBucket;
  }
  async transaction(fn) {
    return this.db.runTransaction(fn, {maxAttempts: 3});
  }
  fence(control) {
    const lease = control?.lease;
    if (!lease || lease.owner !== this.owner || lease.generation !== this.generation || lease.expires_at_ms <= this.clock())
      refuse('firestore_lease_lost');
  }
  async acquire() {
    this.generation = await this.transaction(async tx => {
      const snap = await tx.get(this.control);
      if (!snap.exists) refuse('firestore_control_missing');
      const control = snap.data(), lease = control.lease;
      if (lease && lease.expires_at_ms > this.clock()) refuse('runner_overlap');
      const generation = (lease?.generation || 0) + 1;
      tx.set(this.control, {lease: {owner: this.owner, generation, expires_at_ms: this.clock() + LEASE_MS}}, {merge: true});
      return generation;
    });
    return true;
  }
  async renew() {
    await this.transaction(async tx => {
      const control = (await tx.get(this.control)).data(); this.fence(control);
      tx.set(this.control, {lease: {...control.lease, expires_at_ms: this.clock() + LEASE_MS}}, {merge: true});
    });
    return true;
  }
  async release() {
    if (this.generation === null) return true;
    await this.transaction(async tx => {
      const control = (await tx.get(this.control)).data(), lease = control?.lease;
      // A late release may only clear this generation, never a successor lease.
      if (lease?.owner === this.owner && lease?.generation === this.generation)
        tx.set(this.control, {lease: {...lease, expires_at_ms: 0}}, {merge: true});
    });
    this.generation = null; return true;
  }
  async assertLease() {
    this.fence((await this.control.get()).data()); return true;
  }
  async blobPut(encoded) {
    const raw = Buffer.from(encoded, 'base64');
    if (raw.length > MAX_BYTES) refuse('firestore_blob_too_large');
    const hash = sha(raw), compressed = gzipSync(raw), count = Math.ceil(compressed.length / CHUNK);
    const ref = this.db.doc(`${ROOT}/blobs/${hash}`);
    await this.transaction(async tx => {
      const control = (await tx.get(this.control)).data(); this.fence(control);
      const prior = await tx.get(ref);
      if (prior.exists) {
        if (prior.data().bytes !== raw.length || prior.data().sha256 !== hash) refuse('firestore_blob_conflict');
        return;
      }
      for (let i = 0; i < count; i++) tx.set(this.db.doc(`${ref.path}/chunks/${i}`), {bytes: compressed.subarray(i * CHUNK, (i + 1) * CHUNK)});
      tx.set(ref, {sha256: hash, bytes: raw.length, chunks: count, codec: 'gzip', compressed_bytes: compressed.length});
    });
    return hash;
  }
  async blobGet(hash) {
    if (!/^[a-f0-9]{64}$/.test(hash)) refuse('firestore_blob_reference_invalid');
    const ref = this.db.doc(`${ROOT}/blobs/${hash}`), snap = await ref.get();
    if (!snap.exists) refuse('firestore_blob_missing');
    const meta = snap.data();
    if (meta.sha256 !== hash || meta.codec !== 'gzip' || !Number.isInteger(meta.bytes) || meta.bytes < 0 || meta.bytes > MAX_BYTES
        || !Number.isInteger(meta.chunks) || meta.chunks < 1 || meta.chunks > 33) refuse('firestore_blob_manifest_invalid');
    const chunks = await Promise.all(Array.from({length: meta.chunks}, (_, i) => this.db.doc(`${ref.path}/chunks/${i}`).get()));
    const compressed = Buffer.concat(chunks.map(c => {
      const b = c.data()?.bytes;
      if (!c.exists || !Buffer.isBuffer(b) || b.length > CHUNK) refuse('firestore_blob_chunk_invalid');
      return b;
    }));
    if (compressed.length !== meta.compressed_bytes) refuse('firestore_blob_length_mismatch');
    const raw = gunzipSync(compressed, {maxOutputLength: MAX_BYTES});
    if (raw.length !== meta.bytes || sha(raw) !== hash) refuse('firestore_blob_digest_mismatch');
    return raw.toString('base64');
  }
  async get(day) {
    if (!dateOK(day)) refuse('firestore_date_invalid');
    const snap = await this.db.doc(`${ROOT}/runs/${day}`).get();
    if (!snap.exists) return null;
    const row = JSON.parse(Buffer.from(await this.blobGet(snap.data().blob), 'base64').toString('utf8'));
    if (row.date !== day || row.run_key !== `blueprint-researcher:${day}` || !same(row.metadata, snap.data().metadata))
      refuse('firestore_row_binding_invalid');
    return row;
  }
  async blobReceipt(hash) {
    const bytes=await this.blobGet(hash),snap=await this.db.doc(`${ROOT}/blobs/${hash}`).get(),time=snap.createTime;
    if (!Number.isSafeInteger(time?.seconds) || !Number.isSafeInteger(time?.nanoseconds)
        || time.nanoseconds<0 || time.nanoseconds>=1000000000) refuse('firestore_blob_creation_time_unavailable');
    return {sha256:hash,bytes,created_at:{seconds:time.seconds,nanoseconds:time.nanoseconds}};
  }
  async rows() {
    const snaps = await this.db.collection(`${ROOT}/runs`).limit(10001).get();
    if (snaps.docs.length > 10000) refuse('firestore_history_limit');
    const result = [];
    for (const snap of snaps.docs) result.push(await this.get(snap.id));
    return result.sort((a, b) => a.date.localeCompare(b.date));
  }
  projectWorkItem(tx, row, hash) {
    const repair = row.state==='failed' && row.turn_status==='completed' && row.artifact_downloaded===true
      && !row.qa && !Object.keys(row.delivery || {}).length;
    if (row.packet || repair) tx.set(this.db.doc(`${ROOT}/workItems/${row.date}`), {
      date: row.date, run_key: row.run_key, row_blob: hash, packet_digest: row.packet_digest || null,
      owner: 'blueprint-research-qa-publication-agent',
      stage: repair ? (row.validation_repairs?.at(-1)?.state==='no_progress' ? 'validation_repair_blocked' : 'validation_repair_pending')
        : row.state === 'awaiting_review' ? 'agent_qa_pending' : row.state === 'reviewed' ? 'publication_pending' : row.state,
      qa_state: row.qa?.state || null, qa_error: row.qa?.error || null,
      observer_receipt_required: false, scope: 'research_only_no_outreach'
    });
  }
  async put(row,terminalWrite=null) {
    if (!dateOK(row?.date) || row.run_key !== `blueprint-researcher:${row.date}`) refuse('firestore_row_binding_invalid');
    if (row.search_provider === 'perplexity-fast-v1' && Buffer.byteLength(JSON.stringify(row)) > 7000000)
      refuse('research_tool_record_resource_ceiling');
    const hash = await this.blobPut(Buffer.from(JSON.stringify(row)).toString('base64'));
    const ref = this.db.doc(`${ROOT}/runs/${row.date}`);
    await this.transaction(async tx => {
      const control = (await tx.get(this.control)).data(); this.fence(control);
      const prior = await tx.get(ref);
      if(terminalWrite) {
        if(prior.data()?.blob!==terminalWrite.expected_blob) refuse('terminal_sheets_recovery_source_changed');
        this.terminalSheetsGate(control,row,terminalWrite.context);
      }
      if (!prior.exists && (row.state !== 'creating' || control.enabled !== true)) refuse('firestore_create_not_admitted');
      if (prior.exists && !same(prior.data().metadata, row.metadata)) refuse('firestore_intent_conflict');
      if ((row.expansion_profile || null)!==(row.metadata?.expansion_profile || null)
          || row.expansion_profile && row.expansion_profile!=='exa-guarded-v1') refuse('research_expansion_profile_invalid');
      const exa=row.exa_expansion;
      if(exa && (row.expansion_profile!=='exa-guarded-v1' || exa.run_key!==row.run_key || exa.date!==row.date
          || typeof exa.intent_json!=='string' || sha(Buffer.from(exa.intent_json))!==exa.intent_sha256
          || valueHash(JSON.parse(exa.intent_json))!==valueHash(exa.intent))) refuse('expansion_intent_binding_changed');
      if(prior.data()?.exa_expansion_intent_digest && prior.data().exa_expansion_intent_digest!==exa?.intent_sha256)
        refuse('expansion_start_claim_already_consumed');
      if(prior.data()?.exa_expansion_run_id && prior.data().exa_expansion_run_id!==exa?.run_id)
        refuse('expansion_original_id_changed');
      if(prior.data()?.exa_expansion_terminal_receipt && !same(prior.data().exa_expansion_terminal_receipt,exa?.terminal_receipt))
        refuse('expansion_terminal_receipt_changed');
      const grant=row.paid_expansion_grant,grantDigest=grant===undefined || grant===null?null:valueHash(grant);
      // One frozen grant per row: bound at the durable intent, never replaced or added later.
      // A manifest an older bridge wrote cannot prove which grant the intent froze, so that
      // row stays unbound and admits no new paid claim.
      const priorGrant=prior.data()?.paid_expansion_grant_digest;
      const unbound=prior.exists && (priorGrant===undefined || prior.data().paid_expansion_grant_unbound===true);
      if(prior.exists && priorGrant!==undefined && priorGrant!==grantDigest) refuse('paid_expansion_grant_already_bound');
      if(!prior.exists && grant?.state==='granted') await this.paidGrantGate(tx,control,row,grant,true);
      if(exa && !prior.data()?.exa_expansion_intent_digest) {
        // A new paid claim debits the frozen grant; an unknown cost holds its whole cap.
        if(unbound || grant?.state!=='granted' || valueHash(exa.intent?.grant ?? null)!==grantDigest)
          refuse('paid_expansion_grant_required');
        const liveLimit=await this.paidGrantGate(tx,control,row,grant,false,'exa');
        const dollars=exa.intent?.request?.budget?.maxCostDollars;
        if(typeof dollars!=='number' || Math.round(dollars*1000000)!==exa.cap_micros) refuse('paid_expansion_claim_cap_mismatch');
        const reserved=[exa.cap_micros]; // TODO(FindAll): add this row's FindAll reservations.
        if(!reserved.every(v=>Number.isSafeInteger(v) && v>0) || exa.cap_micros>paidPerStart(liveLimit)
            || reserved.reduce((a,b)=>a+b,0)>liveLimit) refuse('paid_expansion_reservation_exceeds_grant');
      }
      if(prior.data()?.mcp_profile && prior.data().mcp_profile!==row.mcp_profile)
        refuse('research_mcp_profile_changed');
      if(row.mcp_profile && (!['owner-readonly-mcp-v1','owner-delegated-research-mcp-v1'].includes(row.mcp_profile)
          || valueHash(row.mcp_binding)!==row.metadata?.mcp_binding_digest)) refuse('research_mcp_binding_changed');
      if(row.metadata?.mcp_vault_binding_digest && (!row.mcp_profile
          || valueHash(row.mcp_vault_binding)!==row.metadata.mcp_vault_binding_digest)) refuse('research_mcp_vault_binding_changed');
      const historyBinding=row.history_profile==='agent-history-v1'?valueHash(row.history_binding):null;
      if(row.history_profile && row.history_profile!=='agent-history-v1') refuse('company_history_profile_invalid');
      if(prior.exists && ((prior.data().history_profile || null)!==(row.history_profile || null)
          || (prior.data().history_binding_digest || null)!==historyBinding)) refuse('company_history_intent_conflict');
      if(!prior.exists && historyBinding && (row.history_binding?.enabled!==true
          || !control.learning || valueHash(control.learning)!==historyBinding)) refuse('company_history_authority_changed');
      const retryPhase = row.qa_retry_continuation ? sha(Buffer.from(JSON.stringify(row.qa_retry_continuation))) : null;
      const submissionBinding = row.qa?.submission_binding ? sha(Buffer.from(JSON.stringify(row.qa.submission_binding))) : null;
      const terminalCollection = row.qa?.terminal_collection_recovery ? sha(Buffer.from(JSON.stringify(row.qa.terminal_collection_recovery))) : null;
      if (prior.data()?.qa_retry_phase_digest && prior.data().qa_retry_phase_digest !== retryPhase)
        refuse('qa_retry_phase_already_bound');
      if (prior.data()?.qa_submission_binding_digest && prior.data().qa_submission_binding_digest !== submissionBinding)
        refuse('qa_submission_already_bound');
      if (prior.data()?.qa_terminal_collection_digest && prior.data().qa_terminal_collection_digest !== terminalCollection)
        refuse('qa_terminal_collection_already_bound');
      const retry = row.qa?.input_retries?.at(-1);
      const correction = row.qa?.corrections?.at(-1);
      const correctionBindings=Object.fromEntries((row.qa?.corrections || []).map(c=>[c.number, {
        request_digest:c.request_digest,deadline_ms:c.deadline_ms,idempotency_key:c.idempotency_key,
        authority_reference:c.authority_reference,previous_review:c.previous_review,baseline_turn_ids:c.baseline_turn_ids}]));
      if (Object.entries(prior.data()?.qa_correction_bindings || {}).some(([n,b])=>!same(b,correctionBindings[n])))
        refuse('qa_correction_already_bound');
      const publication=row.publication;
      const publicationBinding=publication?valueHash({profile:publication.profile,request_digest:publication.request_digest,
        session_id:publication.session_id,
        input_file:publication.input_file,idempotency_key:publication.idempotency_key,deadline_ms:publication.deadline_ms,
        authority_reference:publication.authority_reference,workflow_authority:publication.workflow_authority,
        baseline_turn_ids:publication.baseline_turn_ids}):null;
      if(prior.data()?.publication_input_binding && prior.data().publication_input_binding!==publicationBinding)
        refuse('publication_input_already_bound');
      if(prior.data()?.publication_turn_id && prior.data().publication_turn_id!==publication?.turn_id)
        refuse('publication_turn_already_bound');
      const cleanupBinding = row.cleanup ? pythonHash(row.cleanup.binding) : null;
      if (prior.data()?.cleanup_binding_digest && prior.data().cleanup_binding_digest !== cleanupBinding)
        refuse('cleanup_binding_already_bound');
      if (row.cleanup && (!prior.data()?.cleanup_archive || valueHash(prior.data().cleanup_archive)!==valueHash(row.cleanup.archive)))
        refuse('cleanup_archive_not_verified');
      if (prior.data()?.cleanup_delete_claimed && row.cleanup?.delete_claimed !== true)
        refuse('cleanup_claim_already_consumed');
      if (prior.data()?.cleanup_delete_confirmation
          && valueHash(prior.data().cleanup_delete_confirmation)!==valueHash(row.cleanup?.delete_confirmation))
        refuse('cleanup_confirmation_already_bound');
      for(const [name,binding] of Object.entries(prior.data()?.publication_delivery_bindings || {})) {
        // Receipts/state may advance; immutable request/plan/presentation evidence may not.
        const d=row.delivery?.[name];
        const active= d?.attempt_number?prior.data().publication_attempts?.[name]?.[d.attempt_number]:null;
        if(d?.attempt_number && (!active || sourceBinding(d)!==prior.data().publication_source_bindings?.[name]
            || (d.presentation?.decision_digest || null)!==active.presentation_digest))
          refuse('publication_attempt_not_admitted');
        const expected=active?active.delivery_binding:binding;
        if(expected && expected!==deliveryBinding(d)) refuse('publication_delivery_already_bound');
      }
      tx.set(ref, {date: row.date, blob: hash, metadata: row.metadata, state: row.state, cleanup_required: row.cleanup_required,
        expansion_profile:row.expansion_profile || null,
        exa_expansion_intent_digest:exa?.intent_sha256 || null,
        exa_expansion_run_id:exa?.run_id || null,
        exa_expansion_terminal_receipt:exa?.terminal_receipt || null,
        paid_expansion_grant_digest:grantDigest,paid_expansion_grant_unbound:unbound,
        ...(row.mcp_profile==='owner-delegated-research-mcp-v1'?{mcp_profile:row.mcp_profile}:{}),
        cleanup_binding_digest: cleanupBinding,
        cleanup_archive: prior.data()?.cleanup_archive || null,
        cleanup_delete_claimed: prior.data()?.cleanup_delete_claimed === true,
        cleanup_delete_confirmation:row.cleanup?.delete_confirmation || null,
        create_attempt_claimed: prior.exists && prior.data().create_attempt_claimed === true,
        session_id: row.session_id || null, turn_id: row.turn_id || null, environment_id: row.environment_id || null,
        qa_request_digest: row.qa?.request_digest || null, qa_state: row.qa?.state || null,
        qa_deadline_ms: row.qa?.deadline_ms || null,
        search_provider: row.search_provider || null, soft_target_usd: row.soft_target_usd ?? null,
        history_profile:row.history_profile || null,history_binding_digest:historyBinding,
        recurring_budget_authority_reference: row.recurring_budget_authority_reference || null,
        qa_request_claimed: prior.exists && prior.data().qa_request_claimed === true,
        qa_retry_phase_digest: retryPhase,
        qa_submission_binding_digest: submissionBinding,
        qa_terminal_collection_digest: terminalCollection,
        qa_retry_deadline_ms: row.qa_retry_continuation ? Date.parse(row.qa_retry_continuation.started_at) + 600000 : row.qa?.submission_binding?.deadline_ms || null,
        qa_retry_workflow_authority: row.qa_retry_continuation ? null : row.qa?.submission_binding?.authority_reference || null,
        qa_retry_number: retry?.number || null, qa_retry_key: retry?.idempotency_key || null,
        qa_retry_not_before_ms: retry ? Date.parse(retry.not_before) : null,
        qa_retry_prior_503: retry ? (row.qa.input_retries.length === 1 ? row.qa.input_error_receipt : row.qa.input_retries.at(-2)?.error_receipt) : null,
        qa_retry_claims: prior.exists ? prior.data().qa_retry_claims || {} : {},
        qa_correction_bindings:correctionBindings,
        qa_correction_number:correction?.number || null, qa_correction_state:correction?.state || null,
        qa_correction_claims:prior.exists ? prior.data().qa_correction_claims || {} : {},
        repair_request_digest: row.validation_repairs?.at(-1)?.request_digest || null,
        repair_deadline_ms: row.validation_repairs?.at(-1)?.deadline_ms || null,
        repair_number: row.validation_repairs?.at(-1)?.number || null,
        repair_state: row.validation_repairs?.at(-1)?.state || null,
        repair_claims: prior.exists ? prior.data().repair_claims || {} : {},
        publication_claimed: prior.exists ? prior.data().publication_claimed || {} : {},
        publication_batches: prior.exists ? prior.data().publication_batches || {} : {},
        publication_delivery_bindings:prior.exists ? prior.data().publication_delivery_bindings || {} : {},
        publication_source_bindings:prior.data()?.publication_source_bindings || {},
        publication_attempts:prior.data()?.publication_attempts || {},publication_rejections:prior.data()?.publication_rejections || {},
        publication_input_binding:publicationBinding,publication_input_claimed:prior.data()?.publication_input_claimed || false,
        publication_request_digest:publication?.request_digest || null,publication_deadline_ms:publication?.deadline_ms || null,
        publication_authority_reference:publication?.authority_reference || null,
        publication_state:publication?.state || null,publication_turn_id:publication?.turn_id || null});
      this.projectWorkItem(tx, row, hash);
    });
    await finishContactResearchSafely(this,row,hash);
    if(this.learning && ['awaiting_review','reviewed','completed','failed','cancelled'].includes(row.state)) {
      let observation;
      try {
        const control=(await this.control.get()).data();
        const result=await this.learning({op:'learning_after_run',day:row.date},control.learning);
        observation={row_blob:hash,status:['created','existing'].includes(result?.append)?'observed':'unavailable',
          append:result?.append || null,event_id:result?.event?.eventId || null,observed_at:new Date(this.clock()).toISOString()};
      } catch(error) {
        observation={row_blob:hash,status:'unavailable',code:/^[a-z_]{1,100}$/.test(error?.message || '')
          ?error.message:'research_learning_observation_unavailable',observed_at:new Date(this.clock()).toISOString()};
      }
      // Observation cannot undo or relabel the source commit. A later source
      // owns its own observation; do not overwrite it with this older result.
      try {await this.transaction(async tx=>{
        const control=(await tx.get(this.control)).data();this.fence(control);
        const current=await tx.get(ref);
        if(current.data()?.blob===hash) tx.set(this.db.doc(`${ROOT}/learningObservations/${hash}`),
          {...observation,run_ref:ref.path,source_row_blob:hash},{merge:true});
      });}catch { /* The committed research row remains authoritative. */ }
    }
    return true;
  }
  async filePut(name, encoded) {
    if (!fileOK(name)) refuse('firestore_file_invalid');
    const hash = await this.blobPut(encoded), ref = this.db.doc(`${ROOT}/files/${name}`);
    await this.transaction(async tx => {
      const control = (await tx.get(this.control)).data(); this.fence(control);
      const prior = await tx.get(ref);
      if ((name.endsWith('-artifact.json') || name.endsWith('-qa.json') || name.endsWith('-qa-input.json') || name.endsWith('-publication-input.json') || name.endsWith('-publication-evidence.json') || name.endsWith('-recovery.json') || /-inventory-|-exa-|-tool-|-repair-|-qa-correction-[12]-input/.test(name)) && prior.exists && prior.data().blob !== hash) refuse('artifact_identity_conflict');
      tx.set(ref, {blob: hash});
    });
    return true;
  }
  async fileGet(name) {
    if (!fileOK(name)) refuse('firestore_file_invalid');
    const snap = await this.db.doc(`${ROOT}/files/${name}`).get();
    if (!snap.exists) refuse('firestore_file_missing');
    return this.blobGet(snap.data().blob);
  }
  async snapshot(day) {
    if(!dateOK(day)) refuse('firestore_date_invalid');
    const manifest=(await this.db.doc(`${ROOT}/runs/${day}`).get());
    if(!manifest.exists) refuse('run_missing');
    const metadata=manifest.data(),source_row_json=Buffer.from(await this.blobGet(metadata.blob),'base64').toString('utf8');
    const row=JSON.parse(source_row_json);
    if(row.date!==day || row.run_key!==`blueprint-researcher:${day}` || !same(row.metadata,metadata.metadata))
      refuse('firestore_row_binding_invalid');
    const files = {}, missing = [];
    for (const kind of ['artifact', 'evidence', 'output', 'review', ...(row.output_recovery ? ['recovery'] : []), ...(row.qa ? ['qa','qa-evidence'] : []),
      ...(row.qa?.input_file ? ['qa-input'] : []),
      ...(row.publication?.input_file ? ['publication-input'] : []),
      ...(row.publication?.evidence_file ? ['publication-evidence'] : []),
      ...(row.qa?.corrections?.length ? ['qa-original','qa-original-evidence'] : []),
      ...(row.qa?.corrections || []).flatMap(c=>[`qa-correction-${c.number}-input`,
        ...(c.artifact_file ? [`qa-correction-${c.number}-artifact`] : []),
        ...(c.evidence_file ? [`qa-correction-${c.number}-evidence`] : [])]),
      ...(row.validation_repairs || []).flatMap(r=>[`repair-${r.number}-input`, ...(r.artifact_file ? [`repair-${r.number}-artifact`] : [])]),
      ...Object.values(row.application_tool_calls || {}).filter(call=>call.result_file).map(call=>`tool-${call.request.call_id}`)]) {
      const names={'qa':row.qa?.artifact_file,'qa-evidence':row.qa?.evidence_file,
        'qa-original':row.qa?.corrections?.[0]?.previous_review?.artifact_file,
        'qa-original-evidence':row.qa?.corrections?.[0]?.previous_review?.evidence_file};
      try {files[kind] = await this.fileGet(names[kind] || `${day}-${kind}.json`);}
      catch (error) {
        if (!(error instanceof Refusal) || error.message !== 'firestore_file_missing') throw error;
        if (kind === 'artifact' && row.artifact_downloaded) refuse('artifact_not_downloaded_or_digest_mismatch');
        missing.push(kind);
      }
    }
    if (files.artifact && row.raw_output_digest !== sha(Buffer.from(files.artifact, 'base64')))
      refuse('artifact_not_downloaded_or_digest_mismatch');
    const inventory=row.packet?.discovery_inventory_manifest;
    if(inventory) {
      if(inventory.version!=='blueprint.discovery-inventory.v1' || inventory.complete_retention!==true
        || !Array.isArray(inventory.pages) || inventory.page_count!==inventory.pages.length
        || !/^[a-f0-9]{64}$/.test(inventory.source_output_digest)) refuse('discovery_inventory_manifest_invalid');
      const sources=[{artifact_file:`${day}-artifact.json`,artifact_digest:row.raw_output_digest},...(row.validation_repairs || [])];
      if(typeof inventory.source_artifact_file!=='string' || typeof inventory.source_artifact_sha256!=='string'
        || !sources.some(source=>source.artifact_file===inventory.source_artifact_file && source.artifact_digest===inventory.source_artifact_sha256))
        refuse('discovery_inventory_source_binding_invalid');
      const sourceRaw=Buffer.from(await this.fileGet(inventory.source_artifact_file),'base64');
      if(sha(sourceRaw)!==inventory.source_artifact_sha256) refuse('discovery_inventory_source_binding_invalid');
      let sourceText=sourceRaw.toString('utf8').replace(/^\uFEFF/,'');
      const fence=/^```(?:json)?[ \t]*\r?\n([\s\S]*)\r?\n```$/i.exec(sourceText.trim());
      if(fence) sourceText=fence[1];
      let sourceInventory;try {sourceInventory=JSON.parse(sourceText).discovery_inventory;}catch {refuse('discovery_inventory_source_binding_invalid');}
      if(!Array.isArray(sourceInventory) || verificationDigest(sourceInventory)!==inventory.records_digest)
        refuse('discovery_inventory_source_binding_invalid');
      let count=0;const retained=[];
      for(const [index,page] of inventory.pages.entries()) {
        if(page.index!==index || page.start!==count || !Number.isInteger(page.end) || page.end<=count
          || page.file!==`${day}-inventory-${inventory.source_output_digest}-${index}.json`) refuse('discovery_inventory_manifest_invalid');
        const encoded=await this.fileGet(page.file),raw=Buffer.from(encoded,'base64');
        // Verify size and digest before parsing untrusted page bytes.
        if(raw.length>100000 || raw.length!==page.bytes || sha(raw)!==page.sha256) refuse('discovery_inventory_page_binding_invalid');
        let value;try {value=JSON.parse(raw);}catch {refuse('discovery_inventory_page_binding_invalid');}
        if(value.version!==inventory.version
          || value.run_key!==row.run_key || value.source_output_digest!==inventory.source_output_digest
          || value.start!==page.start || value.end!==page.end || !Array.isArray(value.records)
          || value.records.length!==page.end-page.start) refuse('discovery_inventory_page_binding_invalid');
        files[page.file.slice(day.length+1,-5)]=encoded;retained.push(...value.records);count=page.end;
      }
      if(count!==inventory.record_count || verificationDigest(retained)!==inventory.records_digest) refuse('discovery_inventory_manifest_invalid');
    }
    const exaRefs=[...(row.exa_transport_receipts || []),
      ...['start_receipt','last_receipt','terminal_receipt'].map(key=>row.exa_expansion?.[key]).filter(Boolean)];
    for(const receipt of exaRefs) {
      if(typeof receipt.file!=='string' || !receipt.file.startsWith(`${day}-exa-`) || !receipt.file.endsWith('.json'))
        refuse('expansion_export_binding_invalid');
      const encoded=await this.fileGet(receipt.file),raw=Buffer.from(encoded,'base64');
      if(sha(raw)!==receipt.sha256 || raw.length!==receipt.bytes) refuse('expansion_receipt_digest_mismatch');
      files[receipt.file.slice(day.length+1,-5)]=encoded;
    }
    if (row.qa?.input_file) {
      const input = files['qa-input'] && Buffer.from(files['qa-input'], 'base64');
      if (row.qa.input_file !== `${day}-qa-input.json` || !input || input.at(-1) !== 10
        || sha(input.subarray(0, -1)) !== row.qa.request_digest)
        refuse('agent_qa_export_digest_mismatch');
    }
    for (const call of Object.values(row.application_tool_calls || {})) {
      if (call.result_file && (call.result_file !== `${day}-tool-${call.request.call_id}.json`
        || !files[`tool-${call.request.call_id}`] || call.result_sha256 !== sha(Buffer.from(files[`tool-${call.request.call_id}`], 'base64'))))
        refuse('research_tool_result_digest_mismatch');
    }
    const plans=Object.fromEntries(Object.entries(row.delivery || {}).filter(([,d])=>d.plan).map(([name,d])=>{
      const plan_json=JSON.stringify(canonicalValue(d.plan));return [name,{plan_json,plan_digest:sha(plan_json)}];
    }));
    const attempt_history={};
    for(const [name,attempts] of Object.entries(metadata.publication_attempts || {})) {
      attempt_history[name]={};
      for(const [number,attempt] of Object.entries(attempts)) {
        const history={...attempt};
        if(attempt.source_row_blob) history.source_row_json=Buffer.from(await this.blobGet(attempt.source_row_blob),'base64').toString('utf8');
        if(attempt.rejection?.response_blob) history.response_json=Buffer.from(await this.blobGet(attempt.rejection.response_blob),'base64').toString('utf8');
        attempt_history[name][number]=history;
      }
    }
    const activeClaims=Object.fromEntries(Object.keys(row.delivery || {}).filter(name=>attemptState(metadata,row,name).claimed)
      .map(name=>[name,attemptState(metadata,row,name).claimed]));
    const activeBatches=Object.fromEntries(Object.keys(row.delivery || {}).filter(name=>Object.keys(attemptState(metadata,row,name).batches || {}).length)
      .map(name=>[name,attemptState(metadata,row,name).batches]));
    const publicationProof={
        schema_version:'blueprint.research-publication-manifest.v1',date:row.date,run_key:row.run_key,
        source_row_blob:metadata.blob,source_row_json,plans,publication_claimed:activeClaims,
        publication_batches:activeBatches,...(Object.keys(attempt_history).length?{attempt_history}:{} )};
    const manifest_json=JSON.stringify(publicationProof);
    const publication=Object.keys(plans).length || Object.keys(metadata.publication_claimed || {}).length
      || Object.keys(metadata.publication_batches || {}).length ? {publication_manifest:{
        manifest_json,manifest_digest:sha(manifest_json)}} : {};
    const cleanup=metadata.cleanup_binding_digest ? {cleanup_manifest:{binding_digest:metadata.cleanup_binding_digest,
      archive:metadata.cleanup_archive,delete_claimed:metadata.cleanup_delete_claimed===true,
      delete_confirmation:metadata.cleanup_delete_confirmation || null}} : {};
    return {schema_version: 'blueprint.research-snapshot.v1', row, files, missing_files: missing,...publication,...cleanup};
  }
  async importRun(row) {
    if (!dateOK(row?.date) || row.run_key !== `blueprint-researcher:${row.date}` || !row.metadata)
      refuse('firestore_row_binding_invalid');
    const hash = await this.blobPut(Buffer.from(JSON.stringify(row)).toString('base64'));
    const ref = this.db.doc(`${ROOT}/runs/${row.date}`);
    return this.transaction(async tx => {
      const control = (await tx.get(this.control)).data(); this.fence(control);
      const prior = await tx.get(ref);
      if (control.enabled !== false) refuse('firestore_import_requires_disabled');
      if (prior.exists && prior.data().blob !== hash) refuse('firestore_import_date_conflict');
      // An identical import is archival reconciliation, never a reset of consumed claims.
      if(prior.exists) {this.projectWorkItem(tx,row,hash);return true;}
      const archived=row.state==='completed' && ['notion','sheets'].every(name=>
        row.delivery?.[name]?.state==='acknowledged' && row.delivery[name].receipt?.readback_verified===true);
      if((row.delivery?.notion?.plan?.protocol==='notion-paginated-v1' || row.publication
          || row.delivery?.notion?.attempt_number) && !archived)
        refuse('firestore_publication_restore_manifest_required');
      tx.set(ref, {date: row.date, blob: hash, metadata: row.metadata, state: row.state, cleanup_required: row.cleanup_required,
        create_attempt_claimed: true, session_id: row.session_id || null, turn_id: row.turn_id || null,
        expansion_profile:row.expansion_profile || null,
        exa_expansion_intent_digest:row.exa_expansion?.intent_sha256 || null,
        exa_expansion_run_id:row.exa_expansion?.run_id || null,
        exa_expansion_terminal_receipt:row.exa_expansion?.terminal_receipt || null,
        paid_expansion_grant_digest:row.paid_expansion_grant?valueHash(row.paid_expansion_grant):null,
        environment_id: row.environment_id || null});
      this.projectWorkItem(tx, row, hash);
      return true;
    });
  }
  async createCheck(day, metadata) {
    if (!dateOK(day)) refuse('firestore_date_invalid');
    return this.transaction(async tx => {
      const control = (await tx.get(this.control)).data(); this.fence(control);
      const snap = await tx.get(this.db.doc(`${ROOT}/runs/${day}`));
      if (control.enabled !== true || !snap.exists || snap.data().state !== 'creating' || snap.data().create_attempt_claimed
          || !same(snap.data().metadata, metadata)) refuse('firestore_create_not_admitted');
      this.budgetGate(control, snap.data());
      if(metadata.expansion_profile && (metadata.expansion_profile!=='exa-guarded-v1'
          || control.config?.expansion_profile!==metadata.expansion_profile
          || control.config?.mcp_profile==='owner-delegated-research-mcp-v1'))
        refuse('research_expansion_profile_changed');
      if(metadata.mcp_binding_digest && control.config?.mcp_profile!==(snap.data().mcp_profile || 'owner-readonly-mcp-v1'))
        refuse('research_mcp_profile_changed');
      if(snap.data().history_profile==='agent-history-v1') {
        if(control.config?.history_profile!=='agent-history-v1' || control.learning?.enabled!==true
            || snap.data().history_binding_digest!==valueHash(control.learning)
            || ['binding','businessScope'].some(key=>{
              const expiry=Date.parse(control.learning[key]?.expiresAt);
              return !Number.isFinite(expiry) || expiry<=this.clock();
            })) refuse('company_history_authority_changed_or_expired');
      } else if (control.learning?.enabled === true || metadata.learning_binding_digest) {
        if (control.learning?.enabled !== true || metadata.learning_binding_digest !== valueHash(control.learning)
            || ['binding','businessScope','learningGrant'].some(key => {
              const expiry = Date.parse(control.learning[key]?.expiresAt);
              return !Number.isFinite(expiry) || expiry <= this.clock();
            })) refuse('research_learning_create_scope_changed_or_expired');
      }
      await claimContactResearch(this,tx,day,metadata);
      tx.set(this.db.doc(`${ROOT}/runs/${day}`), {create_attempt_claimed: true}, {merge: true});
      return true;
    });
  }
  async summary() {
    const runs = this.db.collection(`${ROOT}/runs`);
    const [latest, unfinished, uncleaned] = await Promise.all([
      runs.orderBy('date', 'desc').limit(1).get(),
      runs.where('state', 'not-in', TERMINAL).limit(1).get(),
      runs.where('cleanup_required', '==', true).limit(1).get()
    ]);
    return {latest_date: latest.docs[0]?.id || null, unfinished: unfinished.docs.length > 0,
      cleanup_required: uncleaned.docs.length > 0};
  }
  cleanupGate(control,row,policy) {
    this.fence(control);
    if (this.schedulerStopped || control?.enabled!==true || control.config?.enabled!==true
        || policy?.enabled!==true || valueHash(control.cleanup_policy)!==valueHash(policy)
        || typeof policy.approval_reference!=='string' || !policy.approval_reference.trim()
        || policy.approval_reference.startsWith('PENDING') || !dateOK(policy.first_date)
        || policy.first_date<'2026-10-03' || row.date<policy.first_date
        || !Number.isFinite(Date.parse(policy.expires_at)) || Date.parse(policy.expires_at)<=this.clock()
        || policy.bucket!==CLEANUP_BUCKET || policy.project_id!==control.project_id
        || policy.project_id!==row.preflight?.project_id
        || policy.agent_id!==control.agent_id || policy.template_id!==control.template_id
        || row.create_payload?.agent_id!==policy.agent_id
        || row.create_payload?.environment?.environment_template_id!==policy.template_id
        || row.state!=='completed' || row.cleanup_required!==true || row.qa?.state!=='validated'
        || !['completed','agent_finished_without_complete_receipts'].includes(row.publication?.state)
        || !['completed','failed','cancelled'].includes(row.publication?.turn_status)
        || !['notion','sheets'].every(n=>row.delivery?.[n]?.state==='acknowledged'
          && row.delivery[n].receipt?.readback_verified===true
          && row.delivery[n].receipt.key===row.delivery[n].key
          && row.delivery[n].receipt.payload_digest===row.delivery[n].payload_digest))
      refuse('cleanup_standing_authority_not_admitted');
  }
  async cleanupAdmit(day,policy) {
    const control=(await this.control.get()).data(),row=await this.get(day);
    if (!row) refuse('run_missing');
    this.cleanupGate(control,row,policy);
    return true;
  }
  async cleanupStatus(day) {
    await this.assertLease();
    if (!dateOK(day)) refuse('firestore_date_invalid');
    const snap=await this.db.doc(`${ROOT}/runs/${day}`).get();
    return {delete_claimed:snap.data()?.cleanup_delete_claimed===true};
  }
  async archiveReadback(receipt) {
    if (!this.archiveBucket || receipt?.bucket!==CLEANUP_BUCKET || this.archiveBucket.name!==CLEANUP_BUCKET
        || !receipt.objects?.length) refuse('cleanup_archive_transport_unavailable');
    if (!receipt.objects.some(o=>o.name===receipt.prefix+'/source-row.json' && o.sha256===receipt.source_row_blob))
      refuse('cleanup_archive_readback_mismatch');
    for(let start=0;start<receipt.objects.length;start+=4) await Promise.all(receipt.objects.slice(start,start+4).map(async object=>{
      if (!object.name.startsWith(receipt.prefix+'/') || !/^[1-9][0-9]*$/.test(String(object.generation)))
        refuse('cleanup_archive_readback_mismatch');
      const file=this.archiveBucket.file(object.name,{generation:String(object.generation)});
      const [meta]=await file.getMetadata(),[raw]=await file.download();
      if(String(meta.generation)!==String(object.generation) || Number(meta.size)!==object.bytes
          || raw.length!==object.bytes || sha(raw)!==object.sha256) refuse('cleanup_archive_readback_mismatch');
    }));
    return true;
  }
  async cleanupArchive(request) {
    await this.cleanupAdmit(request.day,request.policy);
    const ref=this.db.doc(`${ROOT}/runs/${request.day}`),before=(await ref.get()).data();
    if(before.cleanup_archive) {
      await this.archiveReadback(before.cleanup_archive);
      return before.cleanup_archive;
    }
    if (!this.archiveBucket || this.archiveBucket.name!==CLEANUP_BUCKET) refuse('cleanup_archive_transport_unavailable');
    const row=await this.get(request.day),files=request.files;
    const required=['status.json',...(Object.values(row.delivery || {}).some(d=>d.plan)?['publication-manifest.json']:[]),
      'provider-session.json','provider-environment.json',
      'provider-turns.json','provider-items.json','provider-artifacts.json',`${request.day}-artifact.json`,
      `${request.day}-publication-evidence.json`];
    if (!files || !required.every(n=>typeof files[n]==='string')
        || valueHash(JSON.parse(Buffer.from(files['status.json'],'base64')))!==valueHash(row))
      refuse('cleanup_archive_incomplete');
    const names=[...Object.keys(files),'source-row.json'].sort(),objects=[];
    if(Object.hasOwn(files,'source-row.json')) refuse('cleanup_archive_source_changed');
    if(names.some(n=>!/^[-A-Za-z0-9_.]{1,200}$/.test(n))) refuse('cleanup_archive_name_invalid');
    const rawFiles=Object.fromEntries(Object.keys(files).map(n=>[n,Buffer.from(files[n],'base64')]));
    rawFiles['source-row.json']=Buffer.from(await this.blobGet(before.blob),'base64');
    const artifacts=JSON.parse(rawFiles['provider-artifacts.json']),session=JSON.parse(rawFiles['provider-session.json']);
    if(session.id!==row.session_id || session.environment?.id!==row.environment_id
        || valueHash(session.metadata)!==valueHash(row.metadata) || session.status!=='idle'
        || session.required_actions?.length || !Array.isArray(artifacts) || !artifacts.length
        || !artifacts.every(a=>typeof a.id==='string' && files[`provider-artifact-${sha(a.id)}.bin`]))
      refuse('cleanup_archive_incomplete');
    const secrets=value=>value && typeof value==='object' && Object.entries(value).some(([k,v])=>
      /^(?:access_token|refresh_token|api_key|private_key|client_secret|bearer_token)$/.test(k) && v
        || typeof v==='string' && /^Bearer\s/i.test(v) || secrets(v));
    if(secrets(session) || secrets(JSON.parse(rawFiles['provider-environment.json']))) refuse('cleanup_archive_credential_material');
    const manifest=Object.fromEntries(names.map(n=>[n,{sha256:sha(rawFiles[n]),bytes:rawFiles[n].length}]));
    const prefix=`operations/research/cleanup/${request.day}/${valueHash(manifest)}`;
    for(let start=0;start<names.length;start+=4) await Promise.all(names.slice(start,start+4).map(async n=>{
      const file=this.archiveBucket.file(`${prefix}/${n}`),raw=rawFiles[n];
      try {await file.save(raw,{resumable:false,preconditionOpts:{ifGenerationMatch:0},
        metadata:{contentType:'application/octet-stream',metadata:{sha256:sha(raw)}}});}
      catch(error) {if(Number(error.code)!==412) throw error;}
      const [meta]=await file.getMetadata();
      objects.push({name:file.name,sha256:sha(raw),bytes:raw.length,generation:String(meta.generation)});
    }));
    objects.sort((a,b)=>a.name.localeCompare(b.name));
    const receipt={bucket:CLEANUP_BUCKET,prefix,source_row_blob:before.blob,objects};
    await this.archiveReadback(receipt);
    await this.transaction(async tx=>{
      const control=(await tx.get(this.control)).data(),current=await tx.get(ref);
      this.cleanupGate(control,row,request.policy);
      if(current.data()?.blob!==before.blob || current.data()?.cleanup_archive) refuse('cleanup_archive_source_changed');
      tx.set(ref,{cleanup_archive:receipt},{merge:true});
    });
    return receipt;
  }
  async cleanupArchiveVerify(day,binding) {
    await this.assertLease();
    const run=(await this.db.doc(`${ROOT}/runs/${day}`).get()).data(),row=await this.get(day);
    if (!run || run.cleanup_binding_digest!==binding || pythonHash(row?.cleanup?.binding)!==binding
        || valueHash(row.cleanup.archive)!==valueHash(run.cleanup_archive)) refuse('cleanup_binding_changed');
    await this.archiveReadback(run.cleanup_archive);
    await this.assertLease();
    return true;
  }
  async cleanupClaim(day,binding,checkOnly=false) {
    const row=await this.get(day),ref=this.db.doc(`${ROOT}/runs/${day}`);
    return this.transaction(async tx=>{
      const control=(await tx.get(this.control)).data(),run=(await tx.get(ref)).data();
      this.cleanupGate(control,row,row?.cleanup?.binding?.policy);
      if(run?.cleanup_binding_digest!==binding || pythonHash(row?.cleanup?.binding)!==binding
          || valueHash(row.cleanup.archive)!==valueHash(run.cleanup_archive)
          || row.cleanup.binding.session_id!==run.session_id || row.cleanup.binding.environment_id!==run.environment_id)
        refuse('cleanup_binding_changed');
      if(checkOnly) {
        if(run.cleanup_delete_claimed!==true) refuse('cleanup_delete_not_claimed');
        return true;
      }
      if(run.cleanup_delete_claimed) return {submit:false};
      tx.set(ref,{cleanup_delete_claimed:true},{merge:true});
      tx.set(this.control,{cleanup_observation_required:true},{merge:true});
      return {submit:true};
    });
  }
  workflowGate(control, allowStopped = false) {
    const workflow=control?.workflow;
    if (!(control?.enabled === true || allowStopped && control?.enabled === false) || workflow?.enabled !== true
        || ['qa_authority_reference','publication_authority_reference'].some(k=>typeof workflow[k]!=='string'
          || !workflow[k].trim() || workflow[k].startsWith('PENDING'))) refuse('workflow_authority_missing');
  }
  budgetGate(control, row) {
    if (row.search_provider !== 'perplexity-fast-v1') return;
    const authority = row.recurring_budget_authority_reference, target = row.soft_target_usd;
    if (typeof authority !== 'string' || !authority.trim() || authority.trim().startsWith('PENDING')
        || typeof target !== 'number' || !Number.isFinite(target) || target <= 0)
      refuse('research_tool_budget_authority_not_pinned');
    if (control.config?.search_provider !== row.search_provider || control.config?.soft_target_usd !== target
        || control.config?.recurring_budget_authority_reference !== authority)
      refuse('research_tool_budget_authority_changed');
  }
  async paidGrantGate(tx,control,row,grant,atIntent,source=null) {
    // A granted record binds its audited owner direction, the live brake and the reviewed
    // release; at the durable intent it must also be frozen from the current direction.
    // Validate the live successor in this same transaction: it can tighten the source,
    // allowance or interval before a new reservation is durable. The worker repeats
    // the complete check immediately before POST, including the frozen run deadline.
    const limit=grant?.limit_micros;
    if(!keysAre(grant,PAID_GRANT_FIELDS) || grant.schema_version!==PAID_GRANT || grant.state!=='granted'
        || grant.run_key!==row.run_key || !hexOK(grant.direction_sha256) || source && !grant.sources?.includes(source)
        || grant.grant_id!==sha(JSON.stringify([grant.direction_sha256,row.run_key]))) refuse('paid_expansion_grant_not_admitted');
    const audit=(await tx.get(this.db.doc(`${ROOT}/paidExpansionDirections/${grant.direction_sha256}`))).data();
    if(!audit || audit.sha256!==grant.direction_sha256 || paidMicros(audit.direction?.per_run_limit_usd)!==limit
        || grant.per_start_max_micros!==paidPerStart(limit) || audit.version!==grant.version || audit.uri!==grant.direction_uri
        || !Array.isArray(grant.sources) || !grant.sources.every(s=>audit.direction.sources?.includes(s))
        || control?.paid_expansion?.enabled!==true || control.source_commit!==grant.source_commit
        || atIntent && control.paid_expansion.current?.sha256!==grant.direction_sha256) refuse('paid_expansion_grant_not_admitted');
    const current=control.paid_expansion.current;
    if(!keysAre(control.paid_expansion,['enabled','current']) || !keysAre(current,['sha256','version','uri','direction'])
        || paidDirectionProblem(current.direction,control) || valueHash(current.direction)!==current.sha256
        || current.version!==current.direction.version || current.uri!==paidUri(current.sha256)
        || source && !current.direction.sources.includes(source)
        || this.clock()<paidStamp(current.direction.effective_from) || this.clock()>=paidStamp(current.direction.expires_at))
      refuse('paid_expansion_grant_not_admitted');
    const liveAudit=(await tx.get(this.db.doc(`${ROOT}/paidExpansionDirections/${current.sha256}`))).data();
    if(!liveAudit || valueHash(liveAudit.direction)!==current.sha256 || liveAudit.sha256!==current.sha256
        || liveAudit.version!==current.version || liveAudit.uri!==current.uri) refuse('paid_expansion_grant_not_admitted');
    return Math.min(limit,paidMicros(current.direction.per_run_limit_usd));
  }
  async paidExpansionSet(expected,value) {
    // The only writer of control.paid_expansion: an owner direction compare-and-swap under
    // the lease. A new direction appends a create-only audit record; the same direction may
    // only be braked (enabled=false), and only a new owner direction re-enables expansion.
    if(!(expected===null || hexOK(expected)) || !keysAre(value,['current','enabled']) || typeof value.enabled!=='boolean'
        || !keysAre(value.current,['direction','sha256','uri','version'])) refuse('paid_expansion_request_invalid');
    const entry=value.current;
    return this.transaction(async tx=>{
      const control=(await tx.get(this.control)).data(); this.fence(control);
      const paid=control.paid_expansion,prior=paid?.current || null;
      if((prior?.sha256 ?? null)!==expected) refuse('paid_expansion_direction_conflict');
      if(expected!==null && entry.sha256===expected) {
        if(valueHash(entry)!==valueHash(prior)) refuse('paid_expansion_direction_conflict');
        if(value.enabled && paid.enabled!==true) refuse('paid_expansion_reenable_requires_new_direction');
        if(paid.enabled!==value.enabled) tx.set(this.control,{...control,paid_expansion:{enabled:value.enabled,current:prior}});
        return {enabled:value.enabled,sha256:entry.sha256,version:prior.version,audit:'existing'};
      }
      const problem=paidDirectionProblem(entry.direction,control);
      if(problem) refuse(problem);
      if(!hexOK(entry.sha256) || pythonHash(entry.direction)!==entry.sha256 || entry.version!==entry.direction.version
          || entry.uri!==paidUri(entry.sha256)) refuse('paid_expansion_direction_digest_mismatch');
      // The chain continues from control's current record or, when a rollback dropped it,
      // from the audit head, so one linear version chain survives.
      let parent=prior;
      if(!parent) {
        const audits=(await tx.get(this.db.collection(`${ROOT}/paidExpansionDirections`).limit(1001))).docs.map(s=>s.data());
        if(audits.length>1000) refuse('paid_expansion_audit_limit');
        parent=audits.reduce((head,record)=>!head || (record.version ?? 0)>(head.version ?? 0)?record:head,null);
      }
      if(!value.enabled || entry.direction.supersedes!==(parent?.sha256 ?? null)
          || entry.version!==(parent?(parent.version+1):1)) refuse('paid_expansion_direction_chain_invalid');
      const ref=this.db.doc(`${ROOT}/paidExpansionDirections/${entry.sha256}`);
      if((await tx.get(ref)).exists) refuse('paid_expansion_audit_conflict');
      tx.set(this.control,{...control,paid_expansion:{enabled:true,current:entry}});
      tx.set(ref,{sha256:entry.sha256,version:entry.version,uri:entry.uri,direction:entry.direction,
        recorded_at:new Date(this.clock()).toISOString()});
      return {enabled:true,sha256:entry.sha256,version:entry.version,audit:'created'};
    });
  }
  async paidExpansionAudit() {
    const snaps=await this.db.collection(`${ROOT}/paidExpansionDirections`).limit(1001).get();
    if(snaps.docs.length>1000) refuse('paid_expansion_audit_limit');
    return snaps.docs.map(snap=>snap.data()).sort((a,b)=>(a.version ?? 0)-(b.version ?? 0));
  }
  paidObject(hash) {
    if(!hexOK(hash)) refuse('paid_expansion_request_invalid');
    if(!this.archiveBucket || this.archiveBucket.name!==CLEANUP_BUCKET) refuse('paid_expansion_object_transport_unavailable');
    return this.archiveBucket.file(`${PAID_PREFIX}${hash}/direction.json`);
  }
  async paidExpansionObjectPut(hash,encoded) {
    // Content-addressed and create-only. The record grants nothing until paid_expansion_set,
    // so it needs no lease; the worker never reads it.
    const file=this.paidObject(hash),raw=Buffer.from(typeof encoded==='string'?encoded:'','base64');
    let direction;try {direction=JSON.parse(raw.toString('utf8'));} catch {refuse('paid_expansion_direction_invalid');}
    const problem=paidDirectionProblem(direction,(await this.control.get()).data());
    if(problem) refuse(problem);
    if(sha(raw)!==hash || pythonHash(direction)!==hash) refuse('paid_expansion_direction_digest_mismatch');
    try {await file.save(raw,{resumable:false,preconditionOpts:{ifGenerationMatch:0},
      metadata:{contentType:'application/json',metadata:{sha256:hash}}});}
    catch(error) {if(Number(error?.code)!==412) refuse('paid_expansion_object_write_unavailable');}
    return this.paidExpansionObjectGet(hash);
  }
  async paidExpansionObjectGet(hash) {
    const file=this.paidObject(hash);let meta,raw;
    try {[meta]=await file.getMetadata();[raw]=await file.download();}
    catch {refuse('paid_expansion_object_missing');}
    if(!Buffer.isBuffer(raw) || sha(raw)!==hash || Number(meta?.size)!==raw.length) refuse('paid_expansion_object_conflict');
    return {uri:paidUri(hash),sha256:hash,bytes:raw.toString('base64'),generation:String(meta.generation)};
  }
  async workItem() {
    const queue=this.db.collection(`${ROOT}/workItems`);
    const groups=await Promise.all(['validation_repair_pending','agent_qa_pending','publication_pending']
      .map(stage=>queue.where('stage','==',stage).limit(21).get()));
    if (groups.some(group=>group.docs.length>20)) refuse('workflow_queue_limit');
    const items=groups.flatMap(group=>group.docs.map(s=>s.data())).sort((a,b)=>a.date.localeCompare(b.date));
    return items[0] || null;
  }
  async activeQA() {
    const runs=this.db.collection(`${ROOT}/runs`);
    const groups=await Promise.all([
      ...['qa_running','qa_input_unresolved','qa_correction_input_unresolved','qa_cancel_pending'].map(state=>['qa_state',state]),
      ...['running','input_unresolved','cancel_pending'].map(state=>['repair_state',state]),
      ...['running','input_unresolved','cancel_pending'].map(state=>['publication_state',state])]
      .map(([field,state])=>runs.where(field,'==',state).limit(1).get()));
    const days=groups.flatMap(s=>s.docs.map(d=>d.id)).sort();
    return days[0] || null;
  }
  async qaCheck(day,requestDigest,deadlineMS) {
    if (!dateOK(day) || !/^[a-f0-9]{64}$/.test(requestDigest)) refuse('agent_qa_request_invalid');
    return this.transaction(async tx=>{
      const control=(await tx.get(this.control)).data();this.fence(control);this.workflowGate(control);
      const ref=this.db.doc(`${ROOT}/runs/${day}`),snap=await tx.get(ref),run=snap.data();
      if (!snap.exists || run.state!=='awaiting_review' || run.qa_state!=='qa_input_unresolved'
          || run.qa_request_digest!==requestDigest || run.qa_request_claimed
          || !Number.isSafeInteger(deadlineMS) || run.qa_deadline_ms!==deadlineMS
          || this.clock()>=deadlineMS) refuse('agent_qa_input_not_admitted');
      this.budgetGate(control, run);
      tx.set(ref,{qa_request_claimed:true},{merge:true});return true;
    });
  }
  async qaRetryCheck(day, requestDigest, deadlineMS, number) {
    if (!dateOK(day) || !/^[a-f0-9]{64}$/.test(requestDigest) || ![1,2].includes(number))
      refuse('qa_retry_request_invalid');
    return this.transaction(async tx=>{
      const control=(await tx.get(this.control)).data();this.fence(control);this.workflowGate(control);
      const ref=this.db.doc(`${ROOT}/runs/${day}`),snap=await tx.get(ref),run=snap.data();
      const error=run?.qa_retry_prior_503;
      if (!snap.exists || run.state!=='awaiting_review' || run.qa_state!=='qa_input_unresolved'
          || run.qa_request_claimed!==true || run.qa_request_digest!==requestDigest
          || (!run.qa_retry_phase_digest && !run.qa_submission_binding_digest) || run.qa_retry_number!==number
          || (run.qa_retry_workflow_authority && run.qa_retry_workflow_authority!==control.workflow.qa_authority_reference)
          || run.qa_retry_key!==`blueprint-researcher:${day}:qa`
          || !Number.isSafeInteger(deadlineMS) || run.qa_retry_deadline_ms!==deadlineMS
          || !Number.isSafeInteger(run.qa_retry_not_before_ms) || this.clock()<run.qa_retry_not_before_ms
          || this.clock()>=deadlineMS || run.qa_retry_claims?.[number]
          || (number===2 && run.qa_retry_claims?.[1]!==requestDigest)
          || error?.stage!=='provider_submission' || error.http_status!==503
          || !['service_unavailable_error','server_is_overloaded'].includes(error.code))
        refuse('qa_retry_input_not_admitted');
      this.budgetGate(control,run);
      tx.set(ref,{qa_retry_claims:{...run.qa_retry_claims,[number]:requestDigest}},{merge:true});return true;
    });
  }
  async qaCorrectionCheck(day, requestDigest, deadlineMS, number) {
    if (!dateOK(day) || !/^[a-f0-9]{64}$/.test(requestDigest) || ![1,2].includes(number))
      refuse('qa_correction_request_invalid');
    return this.transaction(async tx=>{
      const control=(await tx.get(this.control)).data();this.fence(control);this.workflowGate(control);
      const ref=this.db.doc(`${ROOT}/runs/${day}`),snap=await tx.get(ref),run=snap.data();
      const binding=run?.qa_correction_bindings?.[number];
      if (!snap.exists || run.state!=='awaiting_review' || run.qa_state!=='qa_correction_input_unresolved'
          || run.qa_request_claimed!==true || run.qa_correction_number!==number
          || run.qa_correction_state!=='input_unresolved' || binding?.request_digest!==requestDigest
          || binding.authority_reference!==control.workflow.qa_authority_reference
          || (run.qa_retry_workflow_authority && binding.authority_reference!==run.qa_retry_workflow_authority)
          || binding.idempotency_key!==`blueprint-researcher:${day}:qa:correction:${number}`
          || !/^[a-f0-9]{64}$/.test(binding.previous_review?.artifact_digest || '')
          || binding.previous_review?.turn_status!=='completed'
          || !Number.isSafeInteger(deadlineMS) || binding.deadline_ms!==deadlineMS
          || (run.qa_retry_deadline_ms || run.qa_deadline_ms)!==deadlineMS || this.clock()>=deadlineMS
          || run.qa_correction_claims?.[number] || (number===2 && !run.qa_correction_claims?.[1]))
        refuse('qa_correction_input_not_admitted');
      this.budgetGate(control,run);
      tx.set(ref,{qa_correction_claims:{...run.qa_correction_claims,[number]:requestDigest}},{merge:true});return true;
    });
  }
  publicationAgentGate(control,row,requestDigest,action=null) {
    this.fence(control);this.workflowGate(control);this.budgetGate(control,row || {});
    const p=row?.publication;
    if(row?.publication_profile!=='agent-owned-v1' || p?.profile!=='agent-owned-v1'
        || row.state!=='reviewed' || row.qa?.state!=='validated' || !/^[a-f0-9]{64}$/.test(requestDigest || '')
        || p.request_digest!==requestDigest || p.input_file!==`${row.date}-publication-input.json`
        || p.idempotency_key!==`${row.run_key}:publication` || !Number.isSafeInteger(p.deadline_ms)
        || p.deadline_ms<=this.clock() || p.authority_reference!==control.workflow.publication_authority_reference
        || !p.workflow_authority || !same(p.workflow_authority,control.workflow)
        || typeof row.session_id!=='string' || !row.session_id || p.session_id!==row.session_id)
      refuse('publication_agent_input_not_admitted');
    if(action) {
      const intent=row.application_tool_calls?.[action.call_id];
      let exactRequest;try {exactRequest=JSON.parse(intent?.request_json);}catch {}
      if(p.state!=='running' || typeof action.turn_id!=='string' || action.turn_id!==p.turn_id
          || typeof action.call_id!=='string' || !action.call_id || intent?.phase!=='publication'
          || intent.attempted!==true || typeof intent.request_json!=='string' || intent.request_digest!==sha(intent.request_json)
          || valueHash(exactRequest || null)!==valueHash(action)
          || valueHash(intent.request)!==valueHash(action)) refuse('publication_agent_tool_not_admitted');
    }
  }
  async publicationInputCheck(day,requestDigest,deadlineMS) {
    const row=await this.get(day),ref=this.db.doc(`${ROOT}/runs/${day}`),before=await ref.get();
    if(!row?.publication?.input_file) refuse('publication_input_binding_invalid');
    const input=Buffer.from(await this.fileGet(row.publication.input_file),'base64');
    let event;try {event=JSON.parse(input.toString('utf8'));}catch {refuse('publication_input_binding_invalid');}
    if(input.at(-1)!==10 || sha(input.subarray(0,-1))!==requestDigest || event?.type!=='agent.session.input.message')
      refuse('publication_input_binding_invalid');
    return this.transaction(async tx=>{
      const control=(await tx.get(this.control)).data();this.publicationAgentGate(control,row,requestDigest);
      const run=(await tx.get(ref)).data();
      if(row.publication.state!=='input_unresolved' || row.publication.deadline_ms!==deadlineMS
          || run.blob!==before.data().blob || run.publication_input_claimed
          || run.publication_request_digest!==requestDigest || run.publication_deadline_ms!==deadlineMS)
        refuse('publication_agent_input_not_admitted');
      tx.set(ref,{publication_input_claimed:true},{merge:true});return true;
    });
  }
  async admitRejectedPresentation(row,presentation,context) {
    const name='notion',d=row.delivery[name],ref=this.db.doc(`${ROOT}/runs/${row.date}`),before=(await ref.get()).data();
    const state=attemptState(before,row,name),rejection=state.rejection;
    if(!state.claimed || !rejection || rejection.request_digest!==state.claimed
        || rejection.plan_digest!==valueHash(d.plan) || Object.keys(state.batches || {}).some(n=>n!=='0'))
      refuse('publication_presentation_already_bound');
    const raw=Buffer.from(await this.blobGet(rejection.response_blob),'base64').toString('utf8');
    let body;try {body=JSON.parse(raw);}catch {refuse('publication_rejection_receipt_invalid');}
    if(sha(raw)!==rejection.response_digest || body.object!=='error' || body.status!==400
        || body.code!=='validation_error' || rejection.http_status!==400)
      refuse('publication_rejection_receipt_invalid');
    const observation=await this.publisher.notionProgress(row,d.plan);
    if(!observation.absence_verified || observation.receipt) refuse('publication_rejection_absence_not_verified');
    const number=(d.attempt_number || 0)+1;
    const pending=before.publication_attempts?.[name]?.[number];
    if(pending && (pending.prior_attempt!==(d.attempt_number || 0) || pending.rejection_digest!==rejection.response_digest
        || pending.presentation_digest!==(presentation?.decision_digest || null) || pending.claimed || pending.archived))
      refuse('publication_attempt_not_admitted');
    await this.transaction(async tx=>{
      const control=(await tx.get(this.control)).data();this.publicationAgentGate(control,row,context.request_digest,context.action);
      requirePublicationVerification(row,name,this.clock());
      const run=(await tx.get(ref)).data();
      if(run.blob!==before.blob || valueHash(attemptState(run,row,name))!==valueHash(state)
          || valueHash(run.publication_attempts?.[name]?.[number] || null)!==valueHash(pending || null))
        refuse('publication_attempt_not_admitted');
      if(pending) return;
      const priorNumber=d.attempt_number || 0;
      tx.set(ref,{publication_attempts:{...run.publication_attempts,[name]:{...run.publication_attempts?.[name],
        [priorNumber]:{...state,source_row_blob:run.blob,archived:true},
        [number]:{prior_attempt:priorNumber,rejection_digest:rejection.response_digest,
          presentation_digest:presentation?.decision_digest || null,absence_verified:true}}}},{merge:true});
    });
    d.attempt_number=number;delete d.plan;d.presentation=presentation;
    await this.put(row);
  }
  async publicationAgentTool(day,action,requestDigest) {
    let destination, row, argumentIssues;
    const invalidArguments=issues=>{argumentIssues=issues;refuse('publication_agent_tool_arguments_invalid');};
    try {
      row=await this.get(day);
      const control=(await this.control.get()).data();this.publicationAgentGate(control,row,requestDigest,action);
      let args=action?.arguments;
      if(typeof args==='string') {try {args=JSON.parse(args);}catch {invalidArguments([{path:'/',code:'invalid_json',
        expectations:{type:'object'},allowed_repair:'Provide one JSON object matching the declared publication tool fields.'}]);}}
      if(action?.name==='blueprint_inspect_publication' && args==null) args={};
      if(!action || !['blueprint_inspect_publication','blueprint_publish_research'].includes(action.name)
          || !args || typeof args!=='object' || Array.isArray(args))
        invalidArguments([{path:'/',code:'invalid_type',expectations:{type:'object'},
          allowed_repair:'Use a declared publication tool with one arguments object.'}]);
      const ref=this.db.doc(`${ROOT}/runs/${day}`),run=(await ref.get()).data();
      if(!run.publication_input_claimed) refuse('publication_agent_input_not_claimed');
      if(action.name==='blueprint_inspect_publication') {
        if(Object.keys(args).length) invalidArguments([{path:'/',code:'unexpected_properties',expectations:{allowed_properties:[]},
          allowed_repair:'Call blueprint_inspect_publication with an empty object.'}]);
        return {success:true,output:{validated_research:row.qa.decision || null,packet:row.packet,
          lead_verification:row.review?.lead_verification || null,
          verification_eligibility:Object.fromEntries(Object.keys(row.delivery || {}).map(name=>[name,publicationVerification(row,name,this.clock())])),
          destinations:Object.fromEntries(Object.entries(row.delivery || {}).map(([name,d])=>[name,{
            key:d.key,payload:d.payload,payload_digest:d.payload_digest,presentation:d.presentation || null,
            state:d.state,receipt:d.receipt || null,plan:d.plan || null,
            claimed:!!attemptState(run,row,name).claimed,batches:attemptState(run,row,name).batches || {},
            rejection:attemptState(run,row,name).rejection || null,attempts:run.publication_attempts?.[name] || {}}])),
          transport_limits:{notion:{blocks_per_request:90,utf8_json_bytes_per_request:450000,text_code_units_per_block:1800}},
          presentation_rules:{sheets:{strategies:['full'],summary:'must_be_absent',example:{destination:'sheets',strategy:'full'}},
            notion:{full:{summary:'must_be_absent'},concise:{summary:'required_nonblank',max_utf8_bytes:2000000}}},
          guidance:'Sheets requires full with summary omitted. Notion permits full with summary omitted or concise with a nonblank summary up to 2,000,000 UTF-8 bytes. Choose before a write claim. A current protected evidence assessment is required before every new candidate publication claim or batch. Missing or expired evidence stays unresolved; inspect its reasons and retain raw research. Buying intent, rights, commercial qualification, robot compatibility and deployment readiness require separate gates. Unknown writes are observation-only; never replace a consumed plan.'}};
      }
      const a=args;destination=a.destination;
      const issues=[];
      if(Object.keys(a).some(key=>!['destination','strategy','summary'].includes(key)))
        issues.push({path:'/',code:'unexpected_properties',expectations:{allowed_properties:['destination','strategy','summary']},
          allowed_repair:'Remove undeclared fields; keep the original request retained.'});
      if(!['notion','sheets'].includes(destination)) issues.push({path:'/destination',code:'invalid_enum',
        expectations:{allowed_values:['notion','sheets']},allowed_repair:'Select one already approved destination: notion or sheets.'});
      if(!['full','concise'].includes(a.strategy) || destination==='sheets' && a.strategy!=='full')
        issues.push({path:'/strategy',code:'invalid_enum',expectations:{allowed_values:destination==='sheets'?['full']:['full','concise']},
          allowed_repair:destination==='sheets'?'Use {"destination":"sheets","strategy":"full"} with summary omitted.':'Choose full or concise; concise is available only for Notion.'});
      if((destination==='sheets' || a.strategy==='full') && a.summary!==undefined)
        issues.push({path:'/summary',code:'field_must_be_absent',expectations:{present:false},
          allowed_repair:'Omit summary for Sheets and every full presentation; original canonical research remains retained.'});
      if(destination==='notion' && a.strategy==='concise' && (typeof a.summary!=='string'
          || !a.summary.trim() || Buffer.byteLength(a.summary)>2000000))
        issues.push({path:'/summary',code:'invalid_summary',expectations:{type:'string',nonblank:true,max_utf8_bytes:2000000},
          allowed_repair:'Supply a supported nonblank Notion summary within 2,000,000 UTF-8 bytes, or choose full and omit summary.'});
      if(issues.length) invalidArguments(issues);
      const d=row.delivery?.[destination];if(!d) refuse('publication_destination_invalid');
      let presentation=null;
      if(a.strategy==='concise') {
        const decision_json=JSON.stringify({destination,strategy:a.strategy,summary:a.summary,source_payload_digest:d.payload_digest});
        presentation={schema_version:'blueprint.research-presentation.v1',strategy:a.strategy,summary:a.summary,
          source_payload_digest:d.payload_digest,decision_json,decision_digest:sha(decision_json)};
      }
      if((d.plan || attemptState(run,row,destination).claimed || d.state==='acknowledged')
          && valueHash(d.presentation || null)!==valueHash(presentation)) {
        if(destination!=='notion' || d.state==='acknowledged') refuse('publication_presentation_already_bound');
        await this.admitRejectedPresentation(row,presentation,{request_digest:requestDigest,action});
      }
      if(valueHash(d.presentation || null)!==valueHash(presentation)) {d.presentation=presentation;await this.put(row);}
      const receipt=d.state==='acknowledged'?d.receipt:await this.publish(day,destination,{request_digest:requestDigest,action});
      const latest=(await ref.get()).data();
      return {success:true,output:{destination,status:receipt?'acknowledged':'pending',
        mutation_policy:attemptState(latest,row,destination).claimed?'reconcile_exact_claims_only':'not_claimed',
        canonical_payload_digest:d.payload_digest,presentation_digest:presentation?.decision_digest || null,
        guidance:receipt?'Exact destination readback verified.':'Inspect and retry the same choice to reconcile or advance verified ordered batches; never resend an unobserved claimed request.'},...(receipt?{receipt}:{})};
    } catch(error) {
      const code=/^(?:publication|workflow|firestore|research_tool)_[a-z_]{1,100}$/.test(error.message)
        ?error.message:'publication_attempt_unresolved';
      let evidence_file,definitive_rejection=false;
      if(typeof error.provider_response==='string') {
        evidence_file=`${day}-tool-publication-error-${sha(error.provider_response)}.json`;
        try {await this.filePut(evidence_file,Buffer.from(error.provider_response).toString('base64'));}catch {evidence_file=null;}
        let parsed;try {parsed=JSON.parse(error.provider_response);}catch {}
        if(destination==='notion' && error.provider_feedback?.http_status===400 && parsed?.object==='error'
            && parsed.status===400 && parsed.code==='validation_error'
            && sha(error.provider_response)===error.provider_feedback.response_digest) {
          try {
            const current=await this.get(day),ref=this.db.doc(`${ROOT}/runs/${day}`),before=(await ref.get()).data();
            const state=attemptState(before,current,destination),plan=current.delivery.notion.plan;
            const response_blob=await this.blobPut(Buffer.from(error.provider_response).toString('base64'));
            if(state.claimed===error.provider_feedback.request_digest && plan?.request_digest===state.claimed
                && !Object.keys(state.batches || {}).some(n=>n!=='0')) {
              const rejection={http_status:400,provider_code:'validation_error',response_blob,
                response_digest:sha(error.provider_response),request_digest:state.claimed,plan_digest:valueHash(plan)};
              await this.transaction(async tx=>{
                const control=(await tx.get(this.control)).data();this.publicationAgentGate(control,current,requestDigest,action);
                const run=(await tx.get(ref)).data();
                if(run.blob!==before.blob || valueHash(attemptState(run,current,destination))!==valueHash(state))
                  refuse('publication_attempt_not_admitted');
                if(current.delivery.notion.attempt_number) tx.set(ref,{publication_attempts:{...run.publication_attempts,
                  notion:{...run.publication_attempts?.notion,[current.delivery.notion.attempt_number]:{...state,rejection}}}},{merge:true});
                else tx.set(ref,{publication_rejections:{...run.publication_rejections,notion:rejection}},{merge:true});
              });
              definitive_rejection=true;
            }
          } catch { /* Failed rejection proof never grants another write. */ }
        }
      }
      return {success:false,error:{code,destination:destination || null,
        ...(argumentIssues?{status:'recoverable_issue',issues:argumentIssues,
          allowed_repair:'Correct the indicated arguments in this same turn within current authority. Inspect claims before changing a presentation; existing claims and receipts remain binding.'}:{}),
        ...(code==='publication_lead_verification_required'?{status:'recoverable_issue',
          evidence_issues:publicationVerification(row,destination,this.clock()).reasons,
          allowed_repair:'Inspect the retained evidence and reasons. Supply a supported current assessment through the existing protected research review within original authority; preserve raw findings and consumed claims. Do not invent qualification or repeat uncertain writes.'}:{}),
        recovery_policy:argumentIssues?'repair_arguments_preserve_claims':definitive_rejection?'new_agent_presentation_after_verified_absence':'preserve_claims_and_inspect',
        guidance:code==='publication_lead_verification_required'?'Current evidence verification is required for a new qualified candidate claim or batch. Raw source and existing readback remain available; missing evidence is unresolved, with all commercial and rights gates still separate.':argumentIssues?'Sheets requires full with summary omitted; Notion permits full without summary or concise with a supported nonblank summary. Original requests remain retained; argument repair grants no new authority.':definitive_rejection
          ?'The provider definitively rejected the initial create request. Choose a revised presentation; a new attempt is admitted only after complete readback proves the report absent. Original plan, claim and error remain retained.'
          :'Inspect retained source and current claims. An uncertain write is GET-only until exact readback; choose a revised presentation before a claim or after a proven initial validation rejection.',
        ...(error.provider_feedback?{provider_feedback:error.provider_feedback}:{}),
        ...(typeof error.provider_response==='string'?{provider_response_json:error.provider_response}:{}),
        ...(evidence_file?{evidence_file}:{})}};
    }
  }
  terminalSheetsGate(control,row,context) {
    this.fence(control);
    const source=context.source,original=source.publication,projection=structuredClone(row);
    for(const key of ['plan','receipt','state']) delete projection.delivery?.sheets?.[key];
    const expected=structuredClone(source);
    for(const key of ['plan','receipt','state']) delete expected.delivery.sheets[key];
    const authority=control.workflow,stored=original.workflow_authority;
    if(this.schedulerStopped!==true || control.enabled!==false || control.config?.enabled!==false
        || !authority || typeof authority.enabled!=='boolean'
        || Object.keys(authority).sort().join(',')!=='enabled,publication_authority_reference,qa_authority_reference'
        || authority.publication_authority_reference!==context.publication_authority_reference
        || authority.publication_authority_reference!==stored.publication_authority_reference
        || authority.qa_authority_reference!==stored.qa_authority_reference
        || valueHash(projection)!==valueHash(expected)) refuse('terminal_sheets_recovery_not_admitted');
  }
  async recoverTerminalSheets(request) {
    const keys=['op','day','source_row_blob','payload_digest','rejected_call_id','publication_authority_reference'];
    if(Object.keys(request).sort().join(',')!==keys.sort().join(',') || !dateOK(request.day)
        || !['source_row_blob','payload_digest'].every(k=>/^[a-f0-9]{64}$/.test(request[k] || ''))
        || !/^[A-Za-z0-9_-]{1,200}$/.test(request.rejected_call_id || '')
        || typeof request.publication_authority_reference!=='string') refuse('terminal_sheets_recovery_request_invalid');
    const source=JSON.parse(Buffer.from(await this.blobGet(request.source_row_blob),'base64').toString('utf8'));
    const p=source.publication,d=source.delivery?.sheets,review=source.review,qa=source.qa;
    if(source.date!==request.day || source.run_key!==`blueprint-researcher:${request.day}` || source.canary
        || source.state!=='reviewed' || source.publication_profile!=='agent-owned-v1'
        || !source.session_id || !source.turn_id || qa?.state!=='validated' || !qa.turn_id
        || p?.profile!=='agent-owned-v1' || p.session_id!==source.session_id || !p.turn_id
        || p.turn_status!=='completed' || p.state!=='agent_finished_without_complete_receipts'
        || !Number.isSafeInteger(p.completed_at) || !Number.isSafeInteger(p.deadline_ms)
        || p.completed_at*1000>p.deadline_ms || p.cancel_attempted || p.observation_only_reason
        || !Array.isArray(p.baseline_turn_ids) || ![source.turn_id,qa.turn_id].every(id=>p.baseline_turn_ids.includes(id))
        || p.idempotency_key!==source.run_key+':publication' || !p.workflow_authority
        || p.authority_reference!==request.publication_authority_reference
        || p.workflow_authority.publication_authority_reference!==p.authority_reference
        || review?.source_support_verified!==true || review.crm_rechecked!==true
        || review.packet_digest!==source.packet_digest || pythonHash(source.packet)!==source.packet_digest
        || review.qa_artifact_digest!==qa.artifact_digest || verificationDigest(qa.decision)!==verificationDigest(review)
        || ![`${source.date}-qa.json`,`${source.date}-qa-correction-1-artifact.json`,`${source.date}-qa-correction-2-artifact.json`].includes(qa.artifact_file)
        || review.reviewer_reference!==`agent-turn:${source.session_id}:${qa.turn_id}`
        || !Array.isArray(review.accepted_keys) || !review.accepted_keys.length
        || new Set(review.accepted_keys).size!==review.accepted_keys.length
        || d?.state!=='pending' || d.presentation || d.plan || d.receipt || d.attempt_number
        || source.delivery.notion?.state!=='acknowledged' || source.delivery.notion.receipt?.readback_verified!==true
        || source.delivery.notion.receipt.key!==source.run_key+':notion'
        || source.delivery.notion.receipt.payload_digest!==source.delivery.notion.payload_digest
        || d.key!==source.run_key+':sheets' || d.payload_digest!==request.payload_digest)
      refuse('terminal_sheets_recovery_source_invalid');
    const input=Buffer.from(await this.fileGet(p.input_file),'base64');
    if(p.input_file!==`${source.date}-publication-input.json` || input.at(-1)!==10
        || sha(input.subarray(0,-1))!==p.request_digest) refuse('terminal_sheets_recovery_evidence_invalid');
    const selected=source.packet.candidates.filter(c=>review.accepted_keys.includes(c.candidate_key));
    if(selected.length!==review.accepted_keys.length || valueHash(selected)!==valueHash(d.payload?.candidates)
        || d.payload.sheet_id!=='1n95Ih0Swc-q-kZyUaDHoZh6SVzxvf_zt-CRR7i39bWY' || d.payload.tab!=='Prospects'
        || typeof d.payload_json!=='string' || sha(d.payload_json)!==d.payload_digest
        || valueHash(JSON.parse(d.payload_json))!==valueHash(d.payload)) refuse('terminal_sheets_recovery_payload_invalid');
    const rawQA=Buffer.from(await this.fileGet(qa.artifact_file),'base64');
    let qaResult;try {qaResult=JSON.parse(rawQA.toString('utf8'));}catch {refuse('terminal_sheets_recovery_qa_invalid');}
    if(sha(rawQA)!==qa.artifact_digest || qaResult.packet_digest!==source.packet_digest
        || qaResult.source_support_verified!==true || !Array.isArray(qaResult.accepted_keys)
        || !review.accepted_keys.every(key=>qaResult.accepted_keys.includes(key))) refuse('terminal_sheets_recovery_qa_invalid');
    if(review.lead_verification) for(const result of review.lead_verification.results || []) {
      const check=qaResult.checks?.find(check=>check.candidate_key===result.candidate_key);
      if(!check || verificationDigest(check.lead_verification ?? null)!==verificationDigest(result.assessment ?? null)
          || result.assessment_digest!==verificationDigest(result.assessment ?? null))
        refuse('terminal_sheets_recovery_qa_invalid');
    }
    const evidence=JSON.parse(Buffer.from(await this.fileGet(p.evidence_file),'base64').toString('utf8'));
    if(p.evidence_file!==`${source.date}-publication-evidence.json` || !Array.isArray(evidence)
        || evidence.some(item=>item.turn_id!==p.turn_id) || pythonHash(evidence)!==p.evidence_digest)
      refuse('terminal_sheets_recovery_evidence_invalid');
    const call=source.application_tool_calls?.[request.rejected_call_id],action=call?.request;
    let args;try {args=typeof action?.arguments==='string'?JSON.parse(action.arguments):action?.arguments;}catch {}
    if(call?.phase!=='publication' || call.attempted!==true || call.success!==false || call.result_acknowledged!==true
        || action?.call_id!==request.rejected_call_id || action.turn_id!==p.turn_id
        || action.name!=='blueprint_publish_research' || args?.destination!=='sheets' || args.strategy!=='concise'
        || typeof call.request_json!=='string' || sha(call.request_json)!==call.request_digest
        || valueHash(JSON.parse(call.request_json))!==valueHash(action)
        || call.result_file!==`${source.date}-tool-${action.call_id}.json`) refuse('terminal_sheets_recovery_rejection_invalid');
    const raw=Buffer.from(await this.fileGet(call.result_file),'base64'),event=JSON.parse(raw.toString('utf8'));
    const outcome=JSON.parse(event.output);
    if(sha(raw)!==call.result_sha256 || pythonHash(event)!==call.result_digest || event.success!==false
        || event.call_id!==action.call_id || event.turn_id!==p.turn_id || outcome.success!==false
        || outcome.error?.code!=='publication_agent_tool_arguments_invalid' || outcome.error.destination!=='sheets')
      refuse('terminal_sheets_recovery_rejection_invalid');
    const context={...request,source},row=await this.get(request.day),control=(await this.control.get()).data();
    this.terminalSheetsGate(control,row,context);
    const recoveryRef=this.db.doc(`${ROOT}/terminalSheetsRecoveries/${request.source_row_blob}`);
    const binding={source_row_blob:request.source_row_blob,payload_digest:d.payload_digest,rejected_call_id:action.call_id,
      rejection_result_sha256:call.result_sha256,qa_artifact_digest:qa.artifact_digest,
      publication_evidence_digest:p.evidence_digest,publication_authority_reference:p.authority_reference};
    context.recoveryRef=recoveryRef;context.binding=binding;
    const receipt=await this.publish(request.day,'sheets',null,context);
    if(!receipt) return {state:'readback_pending',mutation_policy:'existing_claim_readback_only',recovery_ref:recoveryRef.path};
    if(receipt.destination!=='sheets' || receipt.payload_digest!==d.payload_digest || receipt.key!==d.key
        || receipt.readback_verified!==true || !receipt.reference) refuse('terminal_sheets_recovery_receipt_invalid');
    const current=await this.get(request.day);this.terminalSheetsGate((await this.control.get()).data(),current,context);
    if(current.delivery.sheets.receipt && valueHash(current.delivery.sheets.receipt)!==valueHash(receipt))
      refuse('delivery_receipt_already_bound');
    const expected_blob=sha(JSON.stringify(current));
    current.delivery.sheets.receipt=receipt;current.delivery.sheets.state='acknowledged';
    await this.put(current,{expected_blob,context});
    await this.transaction(async tx=>{
      const ctrl=(await tx.get(this.control)).data();this.terminalSheetsGate(ctrl,current,context);
      const saved=(await tx.get(recoveryRef)).data();
      if(!saved || valueHash(saved.binding)!==valueHash(binding)) refuse('terminal_sheets_recovery_binding_changed');
      if(saved.receipt && valueHash(saved.receipt)!==valueHash(receipt)) refuse('delivery_receipt_already_bound');
      if(!saved.receipt) tx.set(recoveryRef,{...saved,receipt,state:'acknowledged',readback_at:new Date(this.clock()).toISOString()});
    });
    return {state:'acknowledged',receipt,recovery_ref:recoveryRef.path,original_publication_state:p.state};
  }
  async publish(day,selectedDestination=null,agentContext=null,terminalRecovery=null) {
    await this.assertLease(); const initialControl=(await this.control.get()).data();
    if(terminalRecovery) this.terminalSheetsGate(initialControl,await this.get(day),terminalRecovery);
    else this.workflowGate(initialControl,!!this.terminalCollectionReceipt);
    if (!this.publisher) refuse('publication_binding_unavailable');
    const row=await this.get(day);
    if(row?.publication_profile==='agent-owned-v1' && !terminalRecovery) {
      if(!agentContext || !selectedDestination) refuse('publication_agent_choice_required');
      this.publicationAgentGate(initialControl,row,agentContext.request_digest,agentContext.action);
    }
    const proof=this.terminalCollectionReceipt;
    if (proof && (!['reviewed','completed'].includes(row?.state) || row?.qa?.state!=='validated'
        || valueHash(row.qa.terminal_collection_recovery?.native_receipt||null)!==valueHash(proof)
        || row.session_id!==proof.session_id || row.qa.artifact_digest!==proof.qa_artifact_sha256))
      refuse('terminal_qa_collection_validated_receipt_required');
    const collectionAuthority=row?.qa?.terminal_collection_recovery?.workflow_authority;
    if (collectionAuthority && !same(collectionAuthority,initialControl.workflow)) refuse('publication_authority_changed');
    const destination=selectedDestination || ['notion','sheets'].find(name=>row?.delivery?.[name]?.state!=='acknowledged');
    if(selectedDestination && !['notion','sheets'].includes(selectedDestination)) refuse('publication_destination_invalid');
    if (!destination) return null;
    if(terminalRecovery && destination!=='sheets') refuse('terminal_sheets_recovery_request_invalid');
    const d=row.delivery[destination];
    try {
      if(terminalRecovery && attemptState((await this.db.doc(`${ROOT}/runs/${day}`).get()).data(),row,destination).claimed && !d.plan)
        refuse('terminal_sheets_recovery_claim_plan_missing');
      if (!d.plan) {
        requirePublicationVerification(row,destination,this.clock());
        const expected_blob=terminalRecovery?sha(JSON.stringify(row)):null;
        d.plan=await this.publisher.prepare(row,destination);
        await this.put(row,terminalRecovery?{expected_blob,context:terminalRecovery}:null);
      }
      if(terminalRecovery) await this.transaction(async tx=>{
        const control=(await tx.get(this.control)).data();this.terminalSheetsGate(control,row,terminalRecovery);
        const saved=(await tx.get(terminalRecovery.recoveryRef)).data(),plan_digest=valueHash(d.plan);
        if(saved && (valueHash(saved.binding)!==valueHash(terminalRecovery.binding) || saved.plan_digest!==plan_digest))
          refuse('terminal_sheets_recovery_binding_changed');
        if(!saved) tx.set(terminalRecovery.recoveryRef,{schema_version:'blueprint.terminal-sheets-recovery.v1',
          binding:terminalRecovery.binding,plan_digest,state:'prepared',prepared_at:new Date(this.clock()).toISOString()});
      });
      if(destination==='notion' && d.plan.protocol==='notion-paginated-v1')
        return await this.publishNotionBatch(row,d.plan,initialControl.workflow,collectionAuthority,proof,agentContext);
      const terminalBlob=terminalRecovery?sha(JSON.stringify(row)):null;
      const receipt=await this.publisher.reconcile(row,destination,d.plan);
      if (receipt) return receipt;
      const ref=this.db.doc(`${ROOT}/runs/${day}`),before=await ref.get();
      if(terminalRecovery && before.data().blob!==terminalBlob) refuse('publication_authority_or_plan_changed');
      if (attemptState(before.data(),row,destination).claimed) return null; // uncertain: GET reconciliation only
      await this.transaction(async tx=>{
        const control=(await tx.get(this.control)).data();this.fence(control);
        if(terminalRecovery) this.terminalSheetsGate(control,row,terminalRecovery);else this.workflowGate(control,!!proof);
        if (collectionAuthority && !same(collectionAuthority,control.workflow)) refuse('publication_authority_changed');
        const snap=await tx.get(ref),run=snap.data();
        if (run.blob!==before.data().blob || attemptState(run,row,destination).claimed) refuse('publication_attempt_not_admitted');
        if(agentContext) this.publicationAgentGate(control,row,agentContext.request_digest,agentContext.action);
        requirePublicationVerification(row,destination,this.clock());
        tx.set(ref,claimUpdate(run,row,destination,d.plan.request_digest),{merge:true});
      });
      if(agentContext) {
        if(this.publisher.beforeNotionStep) await this.publisher.beforeNotionStep(row,d.plan,null);
        const latest=(await ref.get()).data(),control=(await this.control.get()).data();
        this.publicationAgentGate(control,row,agentContext.request_digest,agentContext.action);
        if(latest.blob!==before.data().blob || attemptState(latest,row,destination).claimed!==d.plan.request_digest)
          refuse('publication_authority_or_plan_changed');
      }
      if(terminalRecovery) {
        const latest=(await ref.get()).data(),control=(await this.control.get()).data();
        this.terminalSheetsGate(control,row,terminalRecovery);
        if(latest.blob!==before.data().blob || attemptState(latest,row,destination).claimed!==d.plan.request_digest)
          refuse('publication_authority_or_plan_changed');
      }
      requirePublicationVerification(row,destination,this.clock());
      await this.publisher.write(row,destination,d.plan,{beforeWrite:async()=>{
        requirePublicationVerification(row,destination,this.clock());
        if(!terminalRecovery) return;
        const latest=(await ref.get()).data(),control=(await this.control.get()).data();
        this.terminalSheetsGate(control,row,terminalRecovery);
        if(latest.blob!==before.data().blob || attemptState(latest,row,destination).claimed!==d.plan.request_digest)
          refuse('publication_authority_or_plan_changed');
      }});
      return await this.publisher.reconcile(row,destination,d.plan);
    } catch(error) {
      if(error instanceof Refusal && /^(?:publication|workflow|firestore|research_tool|terminal_sheets_recovery)_[a-z_]+$/.test(error.message)) throw error;
      const wrapped=new Refusal(typeof error.message==='string' && /^publication_[a-z_]+$/.test(error.message) ? error.message : 'publication_attempt_unresolved');
      if(error.provider_feedback) wrapped.provider_feedback=error.provider_feedback;
      if(error.provider_response) wrapped.provider_response=error.provider_response;
      throw wrapped;
    }
  }
  async publishNotionBatch(row,plan,authority,collectionAuthority,proof,agentContext=null) {
    const ref=this.db.doc(`${ROOT}/runs/${row.date}`),before=await ref.get();
    const claims=attemptState(before.data(),row,'notion').batches || {},planDigest=valueHash(plan);
    if(Object.values(claims).some(claim=>claim.plan_digest!==planDigest || !same(claim.workflow_authority,authority)))
      refuse('publication_authority_or_plan_changed');
    const progress=await this.publisher.notionProgress(row,plan);
    if(progress.receipt) return progress.receipt;
    const step=progress.step;
    if(claims[step.number]) return null; // Unknown acknowledgment: observe this exact batch, never replay it.
    if(step.number>0 && (!claims[step.number-1] || claims[step.number-1].end!==step.start
        || step.number>1 && claims[step.number-1].page_id!==step.page_id))
      refuse('publication_notion_unclaimed_prefix');
    const claim={...step,plan_digest:planDigest,workflow_authority:authority};
    // The immutable plan already holds full request bytes. Store only exact bindings in the manifest.
    delete claim.body_json;
    await this.transaction(async tx=>{
      const control=(await tx.get(this.control)).data();this.fence(control);this.workflowGate(control,!!proof);
      if(!same(authority,control.workflow) || collectionAuthority && !same(collectionAuthority,control.workflow))
        refuse('publication_authority_changed');
      if(agentContext) this.publicationAgentGate(control,row,agentContext.request_digest,agentContext.action);
      requirePublicationVerification(row,'notion',this.clock());
      const snap=await tx.get(ref),run=snap.data();
      if(run.blob!==before.data().blob || attemptState(run,row,'notion').batches?.[step.number])
        refuse('publication_attempt_not_admitted');
      tx.set(ref,claimUpdate(run,row,'notion',plan.request_digest,{...claims,[step.number]:claim}),{merge:true});
    });
    // Reads and the claim can be slow. Check current authority and lease immediately before mutation.
    if(this.publisher.beforeNotionStep) await this.publisher.beforeNotionStep(row,plan,step);
    const latest=(await ref.get()).data(),control=(await this.control.get()).data();
    this.fence(control);
    this.workflowGate(control,!!proof);
    if(agentContext) this.publicationAgentGate(control,row,agentContext.request_digest,agentContext.action);
    if(!same(authority,control.workflow) || latest.blob!==before.data().blob
        || !same(attemptState(latest,row,'notion').batches?.[step.number],claim)) refuse('publication_authority_or_plan_changed');
    requirePublicationVerification(row,'notion',this.clock());
    await this.publisher.writeNotionStep(row,plan,step);
    return await this.publisher.reconcile(row,'notion',plan);
  }
  async repairCheck(day, requestDigest, deadlineMS) {
    if (!dateOK(day) || !/^[a-f0-9]{64}$/.test(requestDigest)) refuse('validation_repair_request_invalid');
    const ref = this.db.doc(`${ROOT}/runs/${day}`);
    return this.transaction(async tx => {
      const control = (await tx.get(this.control)).data(); this.fence(control); this.workflowGate(control);
      const snap = await tx.get(ref), run = snap.data();
      if (!snap.exists || run.repair_request_digest!==requestDigest
          || !Number.isSafeInteger(run.repair_number) || run.repair_number<1
          || !Number.isSafeInteger(deadlineMS) || run.repair_deadline_ms!==deadlineMS
          || this.clock()>=deadlineMS || run.repair_claims?.[run.repair_number])
        refuse('validation_repair_input_not_admitted');
      this.budgetGate(control,run);
      tx.set(ref,{repair_claims:{...run.repair_claims,[run.repair_number]:requestDigest}},{merge:true});
      return true;
    });
  }
  // One authorized test reuses the Oct 1 session. It never enters daily runs or
  // workItems, and shares the existing lease and immutable blob implementation.
  adaptiveRef() {return this.db.doc(`${ROOT}/adaptiveTests/${ADAPTIVE_TEST}`);}
  async adaptiveOrigin() {
    const snap = await this.db.doc(`${ROOT}/runs/2026-10-01`).get();
    return {blob: snap.exists ? snap.data().blob : null, row: await this.get('2026-10-01')};
  }
  async adaptiveGet() {
    const snap = await this.adaptiveRef().get();
    return snap.exists ? JSON.parse(Buffer.from(await this.blobGet(snap.data().blob), 'base64').toString('utf8')) : null;
  }
  async adaptivePut(row) {
    const intent = row?.test_intent;
    if (row?.test_id !== ADAPTIVE_TEST || row.date !== '2026-10-01' || !intent
        || intent.test_id !== ADAPTIVE_TEST || intent.session_id !== row.session_id
        || intent.environment_id !== row.environment_id
        || !/^[a-f0-9]{64}$/.test(intent.intent_digest || '') || !/^[a-f0-9]{64}$/.test(row.daily_blob || ''))
      refuse('adaptive_test_binding_invalid');
    const hash = await this.blobPut(Buffer.from(JSON.stringify(row)).toString('base64'));
    const binding = sha(JSON.stringify([intent,row.session_id,row.environment_id,row.metadata,row.run_key,
      row.started_at,row.research_runtime_seconds,row.total_runtime_seconds,row.research_deadline_ms,row.daily_blob]));
    return this.transaction(async tx => {
      const control = (await tx.get(this.control)).data(); this.fence(control);
      const origin = await tx.get(this.db.doc(`${ROOT}/runs/2026-10-01`));
      const prior = await tx.get(this.adaptiveRef());
      if (!origin.exists || origin.data().blob !== row.daily_blob) refuse('adaptive_original_intent_changed');
      if (!prior.exists && row.state !== 'research_input_unresolved') refuse('adaptive_intent_not_staged');
      if (prior.exists && prior.data().binding !== binding) refuse('adaptive_intent_conflict');
      tx.set(this.adaptiveRef(), {blob: hash, intent_digest: intent.intent_digest, daily_blob: row.daily_blob,
        binding, state: row.state, claims: prior.exists ? prior.data().claims || {} : {},
        research_request_digest: intent.request_digest, research_deadline_ms: row.research_deadline_ms,
        qa_request_digest: row.qa?.request_digest || null, qa_deadline_ms: row.qa?.deadline_ms || null});
      return true;
    });
  }
  async adaptiveClaim(phase, requestDigest, deadlineMS) {
    if (!['research','qa'].includes(phase)) refuse('adaptive_phase_invalid');
    // Verify immutable intent bytes before claiming; the transaction pins the
    // same row blob, original daily blob, lease, deadline and one-use bit.
    const row = await this.adaptiveGet(), intent = row?.test_intent;
    if (!intent || intent.admission_blockers?.length !== 0 || intent.provider_calls !== 0
        || intent.profile?.enabled !== false || intent.profile?.publication_enabled !== false
        || intent.profile?.new_session_create_allowed !== false || intent.profile?.daily_state_reset_allowed !== false
        || intent.profile?.permanent_deletion_allowed !== false) refuse('adaptive_admission_incomplete');
    return this.transaction(async tx => {
      const control = (await tx.get(this.control)).data(); this.fence(control);
      const origin = await tx.get(this.db.doc(`${ROOT}/runs/2026-10-01`));
      const snap = await tx.get(this.adaptiveRef()), run = snap.data();
      if (control?.enabled !== true || !snap.exists || !origin.exists || origin.data().blob !== row.daily_blob
          || run.blob !== sha(Buffer.from(JSON.stringify(row))) || run.intent_digest !== intent.intent_digest
          || run[`${phase}_request_digest`] !== requestDigest
          || run[`${phase}_deadline_ms`] !== deadlineMS || run.claims?.[phase]
          || run.state !== (phase === 'research' ? 'research_input_unresolved' : 'awaiting_review')
          || !Number.isSafeInteger(deadlineMS) || this.clock() >= deadlineMS)
        refuse('adaptive_input_not_admitted');
      tx.set(this.adaptiveRef(), {claims: {...run.claims,[phase]: requestDigest}}, {merge: true});
      return true;
    });
  }
  async adaptiveFilePut(name, encoded) {
    if (!fileOK(name) || !name.startsWith('2026-10-01-') && name !== 'crm.json') refuse('adaptive_file_invalid');
    const hash = await this.blobPut(encoded), ref = this.db.doc(`${this.adaptiveRef().path}/files/${name}`);
    return this.transaction(async tx => {
      const control = (await tx.get(this.control)).data(); this.fence(control);
      const prior = await tx.get(ref);
      if ((name.endsWith('-artifact.json') || name.endsWith('-qa.json')) && prior.exists && prior.data().blob !== hash)
        refuse('artifact_identity_conflict');
      tx.set(ref,{blob: hash}); return true;
    });
  }
  async adaptiveFileGet(name) {
    if (!fileOK(name)) refuse('adaptive_file_invalid');
    const snap = await this.db.doc(`${this.adaptiveRef().path}/files/${name}`).get();
    if (!snap.exists) refuse('firestore_file_missing');
    return this.blobGet(snap.data().blob);
  }
  async dispatch(request) {
    if (this.terminalCollectionReceipt && !['acquire','renew','release','assert_lease','control','read_crm',
      'get','blob_receipt','rows','summary','work_item','active_qa','publish','refresh_crm','put',
      'file_put','file_get','snapshot'].includes(request.op)) refuse('terminal_qa_operation_forbidden');
    switch (request.op) {
      case 'init': {
        const value = request.value;
        if (value?.enabled !== false || value?.schema_version !== 'blueprint.research-control.v1') refuse('firestore_init_not_disabled');
        if (Object.hasOwn(value, 'paid_expansion')) refuse('paid_expansion_requires_direction_operation');
        return this.transaction(async tx => {
          const snap = await tx.get(this.control);
          if (snap.exists) refuse('firestore_control_already_exists');
          tx.set(this.control, value); return true;
        });
      }
      case 'configure': {
        const value = request.value;
        if (value?.schema_version !== 'blueprint.research-control.v1' || typeof value.enabled !== 'boolean')
          refuse('firestore_control_binding_invalid');
        return this.transaction(async tx => {
          const control = (await tx.get(this.control)).data(); this.fence(control);
          // Only paid_expansion_set writes the owner's direction; a full replace keeps it.
          if (Object.hasOwn(value, 'paid_expansion') && valueHash(value.paid_expansion ?? null) !== valueHash(control.paid_expansion ?? null))
            refuse('paid_expansion_requires_direction_operation');
          const replacement = {...value}; delete replacement.paid_expansion;
          tx.set(this.control, {...replacement, cleanup_observation_required:control.cleanup_observation_required===true,
            lease: control.lease, ...(control.paid_expansion ? {paid_expansion: control.paid_expansion} : {})}); return true;
        });
      }
      case 'acquire': return this.acquire();
      case 'renew': return this.renew();
      case 'release': return this.release();
      case 'assert_lease': return this.assertLease();
      case 'control': return (await this.control.get()).data() || null;
      case 'contact_research_context': return contactResearchContext(this,request.day);
      case 'contact_research_reconcile': {
        await this.assertLease();
        const row=await this.get(request.day);
        const manifest=(await this.db.doc(`${ROOT}/runs/${request.day}`).get()).data();
        return row ? finishContactResearchSafely(this,row,manifest.blob) : {state:'missing'};
      }
      case 'learning_context':
      {
        const control = (await this.control.get()).data();
        if (control?.learning?.enabled !== true) return null;
        await this.assertLease();
        if (typeof request.allow_create !== 'boolean') refuse('research_learning_create_direction_required');
        if (!this.learning) refuse('research_learning_binding_unavailable');
        try {return await this.learning(request, control.learning);}
        catch (error) {
          const code=error.message;
          refuse(/^(?:research_learning|business_daily|business_run|learning_consumer)_[a-z_]{1,75}$/.test(code)
            ? code : 'research_learning_unavailable');
        }
      }
      case 'history_search':
      case 'history_fetch': {
        await this.assertLease();
        const control=(await this.control.get()).data(),row=await this.get(request.day);
        if(control?.enabled!==true || control.learning?.enabled!==true || !this.learning
            || row?.history_profile!=='agent-history-v1'
            || control.config?.history_profile!=='agent-history-v1'
            || valueHash(row.history_binding)!==valueHash(control.learning))
          refuse('company_history_authority_changed');
        if(['binding','businessScope'].some(key=>{
          const expiry=Date.parse(control.learning[key]?.expiresAt);
          return !Number.isFinite(expiry) || expiry<=this.clock();
        })) refuse('company_history_scope_expired');
        try {return await this.learning(request,control.learning);}
        catch(error) {
          const code=error.message;
          return {ok:false,error:{code:/^(?:company_history|research_history)_[a-z_]{1,100}$/.test(code)
            ?code:'company_history_unavailable',guidance:'Correct the request or report the current company access/coverage gap.'}};
        }
      }
      case 'read_crm': {
        if (!this.crmReader) refuse('canonical_crm_read_unavailable');
        return this.crmReader();
      }
      case 'adaptive_origin': return this.adaptiveOrigin();
      case 'adaptive_get': return this.adaptiveGet();
      case 'adaptive_put': return this.adaptivePut(request.row);
      case 'adaptive_claim': return this.adaptiveClaim(request.phase,request.request_digest,request.deadline_ms);
      case 'adaptive_file_put': return this.adaptiveFilePut(request.name,request.bytes);
      case 'adaptive_file_get': return this.adaptiveFileGet(request.name);
      case 'get': return this.get(request.day);
      case 'blob_receipt': return this.blobReceipt(request.hash);
      case 'rows': return this.rows();
      case 'summary': return this.summary();
      case 'work_item': return this.workItem();
      case 'active_qa': return this.activeQA();
      case 'qa_check': return this.qaCheck(request.day,request.request_digest,request.deadline_ms);
      case 'qa_retry_check': return this.qaRetryCheck(request.day,request.request_digest,request.deadline_ms,request.number);
      case 'qa_correction_check': return this.qaCorrectionCheck(request.day,request.request_digest,request.deadline_ms,request.number);
      case 'repair_check': return this.repairCheck(request.day,request.request_digest,request.deadline_ms);
      case 'publish': return this.publish(request.day);
      case 'recover_terminal_sheets': return this.recoverTerminalSheets(request);
      case 'publication_input_check': return this.publicationInputCheck(request.day,request.request_digest,request.deadline_ms);
      case 'publication_agent_tool': return this.publicationAgentTool(request.day,request.action,request.request_digest);
      case 'refresh_crm': {
        await this.assertLease();
        if (!this.crmReader) refuse('canonical_crm_read_unavailable');
        const snapshot = await this.crmReader();
        await this.filePut('crm.json', Buffer.from(JSON.stringify(snapshot)).toString('base64'));
        return true;
      }
      case 'put': return this.put(request.row);
      case 'import_run': return this.importRun(request.row);
      case 'file_put': return this.filePut(request.name, request.bytes);
      case 'file_get': return this.fileGet(request.name);
      case 'snapshot': return this.snapshot(request.day);
      case 'cleanup_admit': return this.cleanupAdmit(request.day,request.policy);
      case 'cleanup_status': return this.cleanupStatus(request.day);
      case 'cleanup_archive': return this.cleanupArchive(request);
      case 'cleanup_archive_verify': return this.cleanupArchiveVerify(request.day,request.binding_digest);
      case 'cleanup_claim': return this.cleanupClaim(request.day,request.binding_digest);
      case 'cleanup_delete_check': return this.cleanupClaim(request.day,request.binding_digest,true);
      case 'create_check': return this.createCheck(request.day, request.metadata);
      case 'paid_expansion_set': {
        if (!Object.hasOwn(request, 'expected_sha256')) refuse('paid_expansion_request_invalid');
        return this.paidExpansionSet(request.expected_sha256, request.value);
      }
      case 'paid_expansion_audit': return this.paidExpansionAudit();
      case 'paid_expansion_object_put': return this.paidExpansionObjectPut(request.sha256, request.bytes);
      case 'paid_expansion_object_get': return this.paidExpansionObjectGet(request.sha256);
      default: refuse('firestore_operation_invalid');
    }
  }
}

export async function readCanonicalCRM(account) {
  // The existing Firebase identity needs read access to this exact Sheet.
  // No Sheets writes, grant changes, or alternate credential are performed.
  const {JWT} = await import('google-auth-library');
  const auth = new JWT({email: account.client_email, key: account.private_key,
    scopes: ['https://www.googleapis.com/auth/spreadsheets.readonly']});
  const sheet = '1n95Ih0Swc-q-kZyUaDHoZh6SVzxvf_zt-CRR7i39bWY';
  try {
    const result = await auth.request({url: `https://sheets.googleapis.com/v4/spreadsheets/${sheet}/values/Prospects`,
      method: 'GET', timeout: 20000, retry: false});
    const snapshot = {sheet_id: sheet, complete: true, captured_at: new Date().toISOString(), values: result.data.values};
    if (!Array.isArray(snapshot.values) || Buffer.byteLength(JSON.stringify(snapshot)) > 2000000)
      refuse('canonical_crm_read_unavailable');
    return snapshot;
  } catch {refuse('canonical_crm_read_unavailable');}
}

// Heartbeats and pipe operations share a queue. Release disables the timer
// before it is queued, and drains any already queued renewal before returning.
export class LeaseChannel {
  constructor(store, {setIntervalImpl = setInterval, clearIntervalImpl = clearInterval} = {}) {
    this.store = store; this.tail = Promise.resolve(); this.heartbeat = null; this.lost = false;
    this.setInterval = setIntervalImpl; this.clearInterval = clearIntervalImpl;
  }
  enqueue(fn) {
    const result = this.tail.then(fn);
    this.tail = result.catch(() => {});
    return result;
  }
  call(request) {
    if (request.op === 'release') {this.clearInterval(this.heartbeat); this.heartbeat = null;}
    return this.enqueue(async () => {
      if (this.lost && !['release', 'acquire'].includes(request.op)) refuse('firestore_lease_lost');
      const value = await this.store.dispatch(request);
      if (request.op === 'acquire') {
        this.lost = false;
        this.heartbeat = this.setInterval(() => {
          void this.enqueue(() => this.store.renew()).catch(() => {this.lost = true;});
        }, 20000);
      }
      return value;
    });
  }
  async close() {
    this.clearInterval(this.heartbeat); this.heartbeat = null;
    await this.tail;
    await this.store.release();
  }
}

async function main() {
  const {initializeApp, cert} = await import('firebase-admin/app');
  const {getFirestore} = await import('firebase-admin/firestore');
  const account = JSON.parse(process.env.FIREBASE_SERVICE_ACCOUNT_JSON || '{}');
  if (account.project_id !== 'blueprint-8c1ca') refuse('firestore_project_binding_mismatch');
  const crmReader=()=>readCanonicalCRM(account);
  const publisher=await livePublisher(account,crmReader,process.env.NOTION_API_TOKEN || process.env.NOTION_API_KEY);
  const app = initializeApp({credential: cert(account)});
  const db = getFirestore(app);
  const {getStorage}=await import('firebase-admin/storage');
  // The trusted worker supplies a local compiled module, never a model URL.
  const learningPath = process.env.BLUEPRINT_DAILY_RESEARCH_LEARNING_MODULE;
  const learning = learningPath ? (await import(pathToFileURL(learningPath).href)).researchLearningHost(db) : null;
  const store = new Store(db, undefined, undefined,crmReader,publisher,learning,null,
    process.env.BLUEPRINT_DAILY_RESEARCH_WORKER_ENABLED==='false',getStorage(app).bucket(CLEANUP_BUCKET));
  const channel = new LeaseChannel(store);
  for await (const line of createInterface({input: process.stdin})) {
    try {
      if (line.length > 16 * 1024 * 1024) refuse('firestore_request_too_large');
      const request = JSON.parse(line);
      const value = await channel.call(request);
      process.stdout.write(JSON.stringify({ok: true, value}) + '\n');
    } catch (error) {
      process.stdout.write(JSON.stringify({ok: false, error: error instanceof Refusal ? error.message : 'firestore_request_unavailable'}) + '\n');
    }
  }
  await channel.close();
}
if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main().catch(() => {process.stdout.write('{"ok":false,"error":"firestore_binding_unavailable"}\n'); process.exitCode = 1;});
}
