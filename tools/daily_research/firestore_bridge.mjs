// Private JSON-line pipe for the Python runner; stdout is not a worker log.
import {createHash, randomUUID} from 'node:crypto';
import {gzipSync, gunzipSync} from 'node:zlib';
import {createInterface} from 'node:readline';
import {pathToFileURL} from 'node:url';

export const ROOT = 'blueprintDailyResearch/sites-first';
const MAX_BYTES = 8 * 1024 * 1024, CHUNK = 256 * 1024, LEASE_MS = 180000;
const sha = bytes => createHash('sha256').update(bytes).digest('hex');
const same = (a, b) => JSON.stringify(Object.entries(a || {}).sort()) === JSON.stringify(Object.entries(b || {}).sort());
class Refusal extends Error {}
const refuse = code => {throw new Refusal(code);};
const dateOK = x => typeof x === 'string' && /^\d{4}-\d{2}-\d{2}$/.test(x);
const fileOK = x => typeof x === 'string' && /^(?:\d{4}-\d{2}-\d{2}-(?:artifact|evidence|output|review)|(?:crm|knowledge|refresh-policy))\.json$/.test(x);

export class Store {
  constructor(db, clock = () => Date.now(), owner = randomUUID()) {
    this.db = db; this.clock = clock; this.owner = owner; this.generation = null;
    this.control = db.doc(ROOT);
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
      const control = (await tx.get(this.control)).data(); this.fence(control);
      tx.set(this.control, {lease: {...control.lease, expires_at_ms: 0}}, {merge: true});
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
  async rows() {
    const snaps = await this.db.collection(`${ROOT}/runs`).limit(10001).get();
    if (snaps.docs.length > 10000) refuse('firestore_history_limit');
    const result = [];
    for (const snap of snaps.docs) result.push(await this.get(snap.id));
    return result.sort((a, b) => a.date.localeCompare(b.date));
  }
  async put(row) {
    if (!dateOK(row?.date) || row.run_key !== `blueprint-researcher:${row.date}`) refuse('firestore_row_binding_invalid');
    const hash = await this.blobPut(Buffer.from(JSON.stringify(row)).toString('base64'));
    const ref = this.db.doc(`${ROOT}/runs/${row.date}`);
    await this.transaction(async tx => {
      const control = (await tx.get(this.control)).data(); this.fence(control);
      const prior = await tx.get(ref);
      if (!prior.exists && (row.state !== 'creating' || control.enabled !== true)) refuse('firestore_create_not_admitted');
      if (prior.exists && !same(prior.data().metadata, row.metadata)) refuse('firestore_intent_conflict');
      tx.set(ref, {blob: hash, metadata: row.metadata, state: row.state, cleanup_required: row.cleanup_required,
        create_attempt_claimed: prior.exists && prior.data().create_attempt_claimed === true,
        session_id: row.session_id || null, turn_id: row.turn_id || null, environment_id: row.environment_id || null});
    });
    return true;
  }
  async filePut(name, encoded) {
    if (!fileOK(name)) refuse('firestore_file_invalid');
    const hash = await this.blobPut(encoded), ref = this.db.doc(`${ROOT}/files/${name}`);
    await this.transaction(async tx => {
      const control = (await tx.get(this.control)).data(); this.fence(control);
      const prior = await tx.get(ref);
      if (name.endsWith('-artifact.json') && prior.exists && prior.data().blob !== hash) refuse('artifact_identity_conflict');
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
  async createCheck(day, metadata) {
    return this.transaction(async tx => {
      const control = (await tx.get(this.control)).data(); this.fence(control);
      const snap = await tx.get(this.db.doc(`${ROOT}/runs/${day}`));
      if (control.enabled !== true || !snap.exists || snap.data().state !== 'creating' || snap.data().create_attempt_claimed
          || !same(snap.data().metadata, metadata)) refuse('firestore_create_not_admitted');
      tx.set(this.db.doc(`${ROOT}/runs/${day}`), {create_attempt_claimed: true}, {merge: true});
      return true;
    });
  }
  async dispatch(request) {
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
      case 'acquire': return this.acquire();
      case 'renew': return this.renew();
      case 'release': return this.release();
      case 'assert_lease': return this.assertLease();
      case 'control': return (await this.control.get()).data() || null;
      case 'get': return this.get(request.day);
      case 'rows': return this.rows();
      case 'put': return this.put(request.row);
      case 'file_put': return this.filePut(request.name, request.bytes);
      case 'file_get': return this.fileGet(request.name);
      case 'create_check': return this.createCheck(request.day, request.metadata);
      default: refuse('firestore_operation_invalid');
    }
  }
}

async function main() {
  const {initializeApp, cert} = await import('firebase-admin/app');
  const {getFirestore} = await import('firebase-admin/firestore');
  const account = JSON.parse(process.env.FIREBASE_SERVICE_ACCOUNT_JSON || '{}');
  if (account.project_id !== 'blueprint-8c1ca') refuse('firestore_project_binding_mismatch');
  const store = new Store(getFirestore(initializeApp({credential: cert(account)})));
  let heartbeat = null, lost = false;
  for await (const line of createInterface({input: process.stdin})) {
    try {
      if (line.length > 16 * 1024 * 1024) refuse('firestore_request_too_large');
      const request = JSON.parse(line);
      if (lost) refuse('firestore_lease_lost');
      const value = await store.dispatch(request);
      if (request.op === 'acquire') heartbeat = setInterval(() => {void store.renew().catch(() => {lost = true;});}, 20000);
      if (request.op === 'release') {clearInterval(heartbeat); heartbeat = null;}
      process.stdout.write(JSON.stringify({ok: true, value}) + '\n');
    } catch (error) {
      process.stdout.write(JSON.stringify({ok: false, error: error instanceof Refusal ? error.message : 'firestore_request_unavailable'}) + '\n');
    }
  }
  clearInterval(heartbeat);
}
if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main().catch(() => {process.stdout.write('{"ok":false,"error":"firestore_binding_unavailable"}\n'); process.exitCode = 1;});
}
