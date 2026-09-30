import {test} from 'node:test';
import assert from 'node:assert/strict';
import {randomBytes} from 'node:crypto';
import {Store, ROOT, LeaseChannel} from '../tools/daily_research/firestore_bridge.mjs';
import {MemoryFirestore} from './fixtures/daily_research/firestore-memory.mjs';

async function fixture() {
  const db = new MemoryFirestore(), time = {now: Date.now()};
  db.values.set(ROOT, {enabled: true, schema_version: 'blueprint.research-control.v1'});
  const store = new Store(db, () => time.now, 'first');
  await store.acquire(); return {db, store, time};
}
const row = () => ({date: '2026-09-30', run_key: 'blueprint-researcher:2026-09-30', metadata: {run_key: 'day', payload_digest: 'hash'},
  state: 'creating', cleanup_required: true});

test('overlap refuses and late release cannot clear the successor lease', async () => {
  const {db, store, time} = await fixture();
  const second = new Store(db, () => time.now, 'second');
  await assert.rejects(second.acquire(), /runner_overlap/);
  await store.put(row()); time.now += 180001;
  await second.acquire(); await store.release();
  assert.equal(db.values.get(ROOT).lease.owner, 'second');
  await second.assertLease(); await assert.rejects(store.createCheck(row().date, row().metadata), /lease_lost/);
});

test('create claim is durable, one per date, and outside any provider call', async () => {
  const {db, store} = await fixture(); db.replay = true;
  await store.put(row()); await store.createCheck(row().date, row().metadata);
  await assert.rejects(store.createCheck(row().date, row().metadata), /not_admitted/);
  await assert.rejects(store.put({...row(), metadata: {run_key: 'different'}}), /intent_conflict/);
  assert.equal(db.values.get(`${ROOT}/runs/${row().date}`).create_attempt_claimed, true);
});

test('failed durable commit leaves no dated intent or create authority', async () => {
  const {db, store} = await fixture(); db.failCommit = true;
  await assert.rejects(store.put(row()), /storage failure/);
  db.failCommit = false;
  await assert.rejects(store.createCheck(row().date, row().metadata), /not_admitted/);
});

test('large immutable chunked artifacts verify exact bytes and detect corruption', async () => {
  const {db, store} = await fixture(); const raw = randomBytes(1900000), name = '2026-09-30-artifact.json';
  await store.filePut(name, raw.toString('base64'));
  assert.deepEqual(Buffer.from(await store.fileGet(name), 'base64'), raw);
  await assert.rejects(store.filePut(name, Buffer.from('replacement').toString('base64')), /identity_conflict/);
  for (const [path, data] of db.values) if (path.includes('/chunks/')) assert.ok(data.bytes.length <= 256 * 1024);
  const hash = db.values.get(`${ROOT}/files/${name}`).blob;
  db.values.get(`${ROOT}/blobs/${hash}/chunks/0`).bytes = Buffer.from('corrupt');
  await assert.rejects(store.fileGet(name));
  await assert.rejects(store.fileGet('2026-10-01-artifact.json'), /file_missing/);
});

test('heartbeat release drains renewal and reacquisition starts a clean generation', async () => {
  const {db, store} = await fixture(); await store.release();
  let beat, unblock; const events = [];
  const renew = store.renew.bind(store);
  store.renew = async () => {events.push('renew-start'); await new Promise(resolve => {unblock = resolve;}); await renew(); events.push('renew-end');};
  const release = store.release.bind(store);
  store.release = async () => {events.push('release'); return release();};
  const channel = new LeaseChannel(store, {setIntervalImpl: fn => {beat = fn; return 1;}, clearIntervalImpl: () => events.push('timer-cleared')});
  await channel.call({op: 'acquire'}); beat(); await new Promise(resolve => setImmediate(resolve));
  const releasing = channel.call({op: 'release'}); unblock(); await releasing;
  assert.ok(events.indexOf('renew-end') < events.indexOf('release'));
  await channel.call({op: 'acquire'}); await channel.call({op: 'assert_lease'});
  assert.equal(db.values.get(ROOT).lease.generation, 3); await channel.close();
});

test('summary is bounded and QA/publication queue is Blueprint-owned', async () => {
  const {db, store} = await fixture();
  await store.put(row()); assert.deepEqual(await store.summary(), {latest_date: row().date, unfinished: true, cleanup_required: true});
  await store.put({...row(), state: 'awaiting_review', packet: {findings: []}, packet_digest: 'digest'});
  const task = db.values.get(`${ROOT}/workItems/${row().date}`);
  assert.equal(task.stage, 'agent_qa_pending'); assert.equal(task.observer_receipt_required, false);
  assert.equal((await store.summary()).unfinished, false);
});

test('legacy imports require disabled control and preserve an attempted date', async () => {
  const {db, store} = await fixture();
  await assert.rejects(store.importRun({...row(), state: 'creation_unresolved'}), /requires_disabled/);
  db.values.get(ROOT).enabled = false;
  const legacy = {...row(), state: 'creation_unresolved'};
  await store.importRun(legacy); await store.importRun(legacy);
  await assert.rejects(store.importRun({...legacy, state: 'running'}), /date_conflict/);
  db.values.get(ROOT).enabled = true;
  await assert.rejects(store.createCheck(row().date, row().metadata), /not_admitted/);
});
