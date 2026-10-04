import {test} from 'node:test';
import assert from 'node:assert/strict';
import {EventEmitter} from 'node:events';
import {startDailyResearchWorker} from '../tools/daily_research/render_worker.mjs';

function fixture(options = {}) {
  const children = [], logs = [], signals = [];
  const spawnImpl = () => {
    const child = new EventEmitter(); child.stdout = new EventEmitter(); child.pid = children.length + 100;
    children.push(child); return child;
  };
  const handle = startDailyResearchWorker({bundleRoot: '/isolated', python: '/venv/bin/python', enabled: true,
    spawnImpl, log: line => logs.push(JSON.parse(line)), killGroup: (pid, signal) => signals.push({pid, signal}),
    retryMs: 5, shutdownMs: 10, ...options});
  return {handle, children, logs, signals};
}
test('disabled startup never spawns', async () => {
  let spawned = false;
  await startDailyResearchWorker({enabled: false, spawnImpl: () => {spawned = true;}}).stop();
  assert.equal(spawned, false);
});
test('learning host travels only in the existing child environment and not status output', async () => {
  let captured; const child=new EventEmitter();child.stdout=new EventEmitter();child.pid=100;
  const handle=startDailyResearchWorker({bundleRoot:'/isolated',python:'/venv/python',enabled:true,
    learningHostModule:'/reviewed/research-worker-host.js',killGroup:()=>{},spawnImpl:(_python,_args,options)=>{captured=options;return child;}});
  assert.equal(captured.env.BLUEPRINT_DAILY_RESEARCH_LEARNING_MODULE,'/reviewed/research-worker-host.js');
  const stop=handle.stop();child.emit('exit');await stop;
});
test('canonical native hook alone receives durable terminal observations', async () => {
  const recorded=[], logs=[], child=new EventEmitter(); child.stdout=new EventEmitter(); child.pid=100;
  const handle=startDailyResearchWorker({bundleRoot:'/isolated',python:'/venv/python',enabled:true,
    learningHooks:{afterRun:async day=>recorded.push(day)},log:line=>logs.push(line),killGroup:()=>{},
    spawnImpl:()=>child});
  child.stdout.emit('data',JSON.stringify({date:'2026-10-02',state:'running'})+'\n');
  child.stdout.emit('data',JSON.stringify({date:'2026-10-02',state:'completed',secret:'never forwarded'})+'\n');
  await new Promise(resolve=>setImmediate(resolve));
  assert.deepEqual(recorded,['2026-10-02']);
  assert.equal(logs.some(line=>line.includes('never forwarded')),false);
  const stop=handle.stop(); child.emit('exit'); await stop;
});
test('shutdown drains the canonical terminal hook after the child has stopped', async () => {
  let finish, drained = false;
  const terminal = new Promise(resolve => {finish = resolve;});
  const {handle, children} = fixture({learningHooks:{afterRun:() => terminal}});
  children[0].stdout.emit('data','{"date":"2026-10-02","state":"completed"}\n');
  const stop = handle.stop().then(() => {drained = true;});
  children[0].emit('exit');
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(drained, false);
  finish(); await stop;
  assert.equal(drained, true);
});
for (const key of ['PERPLEXITY_API_KEY', 'EXA_API_KEY']) {
test(`${key} passes only to the application worker, never logs`, async t => {
  const previous = process.env[key];
  process.env[key] = 'offline-placeholder';
  t.after(() => {
    if (previous === undefined) delete process.env[key];
    else process.env[key] = previous;
  });
  let captured;
  const logs = [], child = new EventEmitter(); child.stdout = new EventEmitter(); child.pid = 100;
  const handle = startDailyResearchWorker({bundleRoot: '/isolated', python: '/venv/python', enabled: true,
    log: line => logs.push(line), killGroup: () => {}, spawnImpl: (python, args, options) => {
    captured = options;
    return child;
  }});
  assert.equal(captured.env[key], 'offline-placeholder');
  assert.deepEqual(logs, []);
  child.stdout.emit('data', JSON.stringify({state: 'running', private_key: 'offline-placeholder'}) + '\n');
  assert.equal(logs.some(line => line.includes('offline-placeholder')), false);
  const stop = handle.stop(); child.emit('exit'); await stop;
});
}
test('split and coalesced JSON status frames survive without forwarding raw output', async () => {
  const {handle, children, logs} = fixture(); const out = children[0].stdout;
  out.emit('data', '{"state":"await');
  out.emit('data', 'ing_review","date":"2026-09-30"}\nprivate upstream body\n{"state":"running"}\n');
  assert.deepEqual(logs.map(x => x.state), ['awaiting_review', 'running']);
  assert.equal(logs[0].date, '2026-09-30');
  const stopping = handle.stop(); children[0].emit('exit'); await stopping;
});
test('shutdown is idempotent and kills the whole private process group at deadline', async () => {
  const {handle, children, signals} = fixture();
  const first = handle.stop(); assert.equal(handle.stop(), first); await first;
  assert.deepEqual(signals, [{pid: children[0].pid, signal: 'SIGTERM'}, {pid: children[0].pid, signal: 'SIGKILL'}]);
});
test('error plus exit schedules one restart and stop clears pending restart', async () => {
  const {handle, children} = fixture(); children[0].emit('error', new Error('unprinted'));
  children[0].emit('exit'); await new Promise(resolve => setTimeout(resolve, 12));
  assert.equal(children.length, 2);
  const stopping = handle.stop(); children[1].emit('exit'); await stopping;
  await new Promise(resolve => setTimeout(resolve, 12)); assert.equal(children.length, 2);
});
