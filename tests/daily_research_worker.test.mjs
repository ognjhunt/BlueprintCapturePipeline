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
    learningModule:'/reviewed/research-worker-host.js',killGroup:()=>{},spawnImpl:(_python,_args,options)=>{captured=options;return child;}});
  assert.equal(captured.env.BLUEPRINT_DAILY_RESEARCH_LEARNING_MODULE,'/reviewed/research-worker-host.js');
  const stop=handle.stop();child.emit('exit');await stop;
});
test('Perplexity secret passes only to the application worker, never logs', async () => {
  const previous = process.env.PERPLEXITY_API_KEY;
  process.env.PERPLEXITY_API_KEY = 'offline-placeholder';
  let captured;
  const logs = [], child = new EventEmitter(); child.stdout = new EventEmitter(); child.pid = 100;
  const handle = startDailyResearchWorker({bundleRoot: '/isolated', python: '/venv/python', enabled: true,
    log: line => logs.push(line), killGroup: () => {}, spawnImpl: (python, args, options) => {
    captured = options;
    return child;
  }});
  assert.equal(captured.env.PERPLEXITY_API_KEY, 'offline-placeholder');
  assert.deepEqual(logs, []);
  const stop = handle.stop(); child.emit('exit'); await stop;
  if (previous === undefined) delete process.env.PERPLEXITY_API_KEY;
  else process.env.PERPLEXITY_API_KEY = previous;
});
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
