// Small WebApp startWorker() hook; independent of ops/outbound flags and modules.
import {spawn} from 'node:child_process';

export function startDailyResearchWorker({bundleRoot, python, enabled = process.env.BLUEPRINT_DAILY_RESEARCH_WORKER_ENABLED === 'true', spawnImpl = spawn, log = console.log} = {}) {
  if (!enabled) return {stop: async () => {}};
  if (!bundleRoot || !python) throw new Error('research_runtime_paths_required');
  let child = null, retry = null, stopping = false;
  function launch() {
    if (stopping) return;
    child = spawnImpl(python, ['-m', 'tools.daily_research.render', 'scheduler'], {
      cwd: bundleRoot, env: {
        PATH: process.env.PATH, HOME: process.env.HOME, PYTHONDONTWRITEBYTECODE: '1',
        OPENAI_API_KEY: process.env.OPENAI_API_KEY,
        FIREBASE_SERVICE_ACCOUNT_JSON: process.env.FIREBASE_SERVICE_ACCOUNT_JSON
      }, stdio: ['ignore', 'pipe', 'ignore']
    });
    child.stdout.on('data', chunk => {
      // Do not forward arbitrary child stdout or upstream provider bodies.
      for (const line of String(chunk).split('\n')) {
        try {
          const result = JSON.parse(line);
          log(JSON.stringify({research: true, state: result.state || 'status', date: result.date || null,
            error: /^[a-z_]+$/.test(result.error || '') ? result.error : null}));
        } catch { /* partial protocol line or non-status output is discarded */ }
      }
    });
    const restart = () => {
      if (!stopping && retry === null) retry = setTimeout(() => {retry = null; launch();}, 30000);
    };
    child.once('error', restart);
    child.once('exit', restart);
  }
  launch();
  return {stop: async () => {
    stopping = true; clearTimeout(retry);
    if (!child || child.exitCode !== null) return;
    const target = child;
    await new Promise(resolve => {
      const deadline = setTimeout(() => {target.kill('SIGKILL'); resolve();}, 20000);
      target.once('exit', () => {clearTimeout(deadline); resolve();});
      target.kill('SIGTERM');
    });
  }};
}
