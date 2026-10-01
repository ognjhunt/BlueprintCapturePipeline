// Small WebApp startWorker() hook; independent of ops/outbound flags and modules.
import {spawn} from 'node:child_process';

export function startDailyResearchWorker({bundleRoot, python,
  enabled = process.env.BLUEPRINT_DAILY_RESEARCH_WORKER_ENABLED === 'true',
  spawnImpl = spawn, killGroup = (pid, signal) => process.kill(-pid, signal),
  log = console.log, shutdownMs = 25000, retryMs = 30000} = {}) {
  if (!enabled) return {stop: async () => {}};
  if (!bundleRoot || !python) throw new Error('research_runtime_paths_required');
  let child = null, retry = null, stopping = false, stopPromise = null;
  function signal(target, name) {
    try { if (target.pid) killGroup(target.pid, name); else target.kill(name); } catch { /* process already gone */ }
  }
  function launch() {
    if (stopping) return;
    const target = spawnImpl(python, ['-m', 'tools.daily_research.render', 'scheduler'], {
      cwd: bundleRoot, detached: true, env: {
        PATH: process.env.PATH, HOME: process.env.HOME, PYTHONDONTWRITEBYTECODE: '1',
        OPENAI_API_KEY: process.env.OPENAI_API_KEY,
        PERPLEXITY_API_KEY: process.env.PERPLEXITY_API_KEY,
        FIREBASE_SERVICE_ACCOUNT_JSON: process.env.FIREBASE_SERVICE_ACCOUNT_JSON,
        NOTION_API_TOKEN: process.env.NOTION_API_TOKEN, NOTION_API_KEY: process.env.NOTION_API_KEY
      }, stdio: ['ignore', 'pipe', 'ignore']
    });
    child = target;
    let buffer = '';
    target.stdout.on('data', chunk => {
      buffer += String(chunk);
      if (buffer.length > 8192) {buffer = ''; return;}
      let boundary;
      while ((boundary = buffer.indexOf('\n')) >= 0) {
        const line = buffer.slice(0, boundary); buffer = buffer.slice(boundary + 1);
        try {
          const result = JSON.parse(line);
          if (!/^[a-z_]{1,80}$/.test(result.state || '')) continue;
          log(JSON.stringify({research: true, state: result.state,
            date: /^\d{4}-\d{2}-\d{2}$/.test(result.date || '') ? result.date : null,
            error: /^[a-z_]{1,100}$/.test(result.error || '') ? result.error : null}));
        } catch { /* arbitrary output never enters worker logs */ }
      }
    });
    let exited = false;
    const restart = () => {
      if (exited) return;
      exited = true;
      signal(target, 'SIGKILL'); // includes a leftover private bridge, never another worker
      if (child === target) child = null;
      if (!stopping && retry === null) retry = setTimeout(() => {retry = null; launch();}, retryMs);
    };
    target.once('error', restart);
    target.once('exit', restart);
  }
  launch();
  return {stop: () => {
    if (stopPromise) return stopPromise;
    stopping = true; clearTimeout(retry); retry = null;
    const target = child;
    stopPromise = !target ? Promise.resolve() : new Promise(resolve => {
      const deadline = setTimeout(() => {signal(target, 'SIGKILL'); resolve();}, shutdownMs);
      target.once('exit', () => {clearTimeout(deadline); resolve();});
      signal(target, 'SIGTERM');
    });
    return stopPromise;
  }};
}
