// Private JSON-line pipe for the Python runner; stdout is not a worker log.
import {createHash, randomUUID} from 'node:crypto';
import {gzipSync, gunzipSync} from 'node:zlib';
import {createInterface} from 'node:readline';
import {pathToFileURL} from 'node:url';
import {livePublisher} from './publisher.mjs';

export const ROOT = 'blueprintDailyResearch/sites-first';
export const ADAPTIVE_TEST = 'adaptive-discovery-20261001';
const MAX_BYTES = 8 * 1024 * 1024, CHUNK = 256 * 1024, LEASE_MS = 180000;
const TERMINAL = ['awaiting_review', 'reviewed', 'completed', 'failed', 'cancelled'];
const sha = bytes => createHash('sha256').update(bytes).digest('hex');
const canonicalValue = value => Array.isArray(value) ? value.map(canonicalValue) : value && typeof value === 'object'
  ? Object.fromEntries(Object.keys(value).sort().map(key => [key, canonicalValue(value[key])])) : value;
const valueHash = value => sha(JSON.stringify(canonicalValue(value)));
const same = (a, b) => JSON.stringify(Object.entries(a || {}).sort()) === JSON.stringify(Object.entries(b || {}).sort());
class Refusal extends Error {}
const refuse = code => {throw new Refusal(code);};
const dateOK = x => typeof x === 'string' && /^\d{4}-\d{2}-\d{2}$/.test(x);
const fileOK = x => typeof x === 'string' && /^(?:\d{4}-\d{2}-\d{2}-(?:artifact|evidence|output|review|recovery|qa|qa-evidence|qa-input|qa-correction-[12]-(?:input|artifact|evidence)|repair-[1-9]\d*-(?:input|artifact)|tool-[A-Za-z0-9_-]{1,200})|(?:crm|knowledge|refresh-policy))\.json$/.test(x);

export class Store {
  constructor(db, clock = () => Date.now(), owner = randomUUID(), crmReader = null, publisher = null, learning = null,
    terminalCollectionReceipt = null) {
    this.db = db; this.clock = clock; this.owner = owner; this.generation = null;
    this.control = db.doc(ROOT);
    this.crmReader = crmReader;
    this.publisher = publisher;
    this.learning = learning;
    this.terminalCollectionReceipt = terminalCollectionReceipt;
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
  async put(row) {
    if (!dateOK(row?.date) || row.run_key !== `blueprint-researcher:${row.date}`) refuse('firestore_row_binding_invalid');
    if (row.search_provider === 'perplexity-fast-v1' && Buffer.byteLength(JSON.stringify(row)) > 7000000)
      refuse('research_tool_record_resource_ceiling');
    const hash = await this.blobPut(Buffer.from(JSON.stringify(row)).toString('base64'));
    const ref = this.db.doc(`${ROOT}/runs/${row.date}`);
    await this.transaction(async tx => {
      const control = (await tx.get(this.control)).data(); this.fence(control);
      const prior = await tx.get(ref);
      if (!prior.exists && (row.state !== 'creating' || control.enabled !== true)) refuse('firestore_create_not_admitted');
      if (prior.exists && !same(prior.data().metadata, row.metadata)) refuse('firestore_intent_conflict');
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
      tx.set(ref, {date: row.date, blob: hash, metadata: row.metadata, state: row.state, cleanup_required: row.cleanup_required,
        create_attempt_claimed: prior.exists && prior.data().create_attempt_claimed === true,
        session_id: row.session_id || null, turn_id: row.turn_id || null, environment_id: row.environment_id || null,
        qa_request_digest: row.qa?.request_digest || null, qa_state: row.qa?.state || null,
        qa_deadline_ms: row.qa?.deadline_ms || null,
        search_provider: row.search_provider || null, soft_target_usd: row.soft_target_usd ?? null,
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
        publication_claimed: prior.exists ? prior.data().publication_claimed || {} : {}});
      this.projectWorkItem(tx, row, hash);
    });
    return true;
  }
  async filePut(name, encoded) {
    if (!fileOK(name)) refuse('firestore_file_invalid');
    const hash = await this.blobPut(encoded), ref = this.db.doc(`${ROOT}/files/${name}`);
    await this.transaction(async tx => {
      const control = (await tx.get(this.control)).data(); this.fence(control);
      const prior = await tx.get(ref);
      if ((name.endsWith('-artifact.json') || name.endsWith('-qa.json') || name.endsWith('-qa-input.json') || name.endsWith('-recovery.json') || /-tool-|-repair-|-qa-correction-[12]-input/.test(name)) && prior.exists && prior.data().blob !== hash) refuse('artifact_identity_conflict');
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
    const row = await this.get(day);
    if (!row) refuse('run_missing');
    const files = {}, missing = [];
    for (const kind of ['artifact', 'evidence', 'output', 'review', ...(row.output_recovery ? ['recovery'] : []), ...(row.qa ? ['qa','qa-evidence'] : []),
      ...(row.qa?.input_file ? ['qa-input'] : []),
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
    return {schema_version: 'blueprint.research-snapshot.v1', row, files, missing_files: missing};
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
      tx.set(ref, {date: row.date, blob: hash, metadata: row.metadata, state: row.state, cleanup_required: row.cleanup_required,
        create_attempt_claimed: true, session_id: row.session_id || null, turn_id: row.turn_id || null,
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
      if (control.learning?.enabled === true || metadata.learning_binding_digest) {
        if (control.learning?.enabled !== true || metadata.learning_binding_digest !== valueHash(control.learning)
            || ['binding','businessScope','learningGrant'].some(key => {
              const expiry = Date.parse(control.learning[key]?.expiresAt);
              return !Number.isFinite(expiry) || expiry <= this.clock();
            })) refuse('research_learning_create_scope_changed_or_expired');
      }
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
      ...['running','input_unresolved','cancel_pending'].map(state=>['repair_state',state])]
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
  async publish(day) {
    await this.assertLease(); const initialControl=(await this.control.get()).data();
    this.workflowGate(initialControl,!!this.terminalCollectionReceipt);
    if (!this.publisher) refuse('publication_binding_unavailable');
    const row=await this.get(day);
    const proof=this.terminalCollectionReceipt;
    if (proof && (!['reviewed','completed'].includes(row?.state) || row?.qa?.state!=='validated'
        || valueHash(row.qa.terminal_collection_recovery?.native_receipt||null)!==valueHash(proof)
        || row.session_id!==proof.session_id || row.qa.artifact_digest!==proof.qa_artifact_sha256))
      refuse('terminal_qa_collection_validated_receipt_required');
    const collectionAuthority=row?.qa?.terminal_collection_recovery?.workflow_authority;
    if (collectionAuthority && !same(collectionAuthority,initialControl.workflow)) refuse('publication_authority_changed');
    const destination=['notion','sheets'].find(name=>row?.delivery?.[name]?.state!=='acknowledged');
    if (!destination) return null;
    const d=row.delivery[destination];
    try {
      if (!d.plan) {d.plan=await this.publisher.prepare(row,destination);await this.put(row);}
      const receipt=await this.publisher.reconcile(row,destination,d.plan);
      if (receipt) return receipt;
      const ref=this.db.doc(`${ROOT}/runs/${day}`),before=await ref.get();
      if (before.data().publication_claimed?.[destination]) return null; // uncertain: GET reconciliation only
      await this.transaction(async tx=>{
        const control=(await tx.get(this.control)).data();this.fence(control);this.workflowGate(control,!!proof);
        if (collectionAuthority && !same(collectionAuthority,control.workflow)) refuse('publication_authority_changed');
        const snap=await tx.get(ref),run=snap.data();
        if (run.blob!==before.data().blob || run.publication_claimed?.[destination]) refuse('publication_attempt_not_admitted');
        tx.set(ref,{publication_claimed:{...run.publication_claimed,[destination]:d.plan.request_digest}},{merge:true});
      });
      await this.publisher.write(row,destination,d.plan);
      return await this.publisher.reconcile(row,destination,d.plan);
    } catch(error) {
      refuse(typeof error.message==='string' && /^publication_[a-z_]+$/.test(error.message) ? error.message : 'publication_attempt_unresolved');
    }
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
          tx.set(this.control, {...value, lease: control.lease}); return true;
        });
      }
      case 'acquire': return this.acquire();
      case 'renew': return this.renew();
      case 'release': return this.release();
      case 'assert_lease': return this.assertLease();
      case 'control': return (await this.control.get()).data() || null;
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
      case 'create_check': return this.createCheck(request.day, request.metadata);
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
  const db = getFirestore(initializeApp({credential: cert(account)}));
  // The trusted worker supplies a local compiled module, never a model URL.
  const learningPath = process.env.BLUEPRINT_DAILY_RESEARCH_LEARNING_MODULE;
  const learning = learningPath ? (await import(pathToFileURL(learningPath).href)).researchLearningHost(db) : null;
  const store = new Store(db, undefined, undefined,crmReader,publisher,learning);
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
