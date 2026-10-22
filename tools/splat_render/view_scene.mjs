// Private inspection of retained artifacts using the existing Spark/Three stack.
// Usage: node view_scene.mjs --splat FILE.spz --mesh FILE.glb --cameras CAMERAS.json [--port 8788]
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.dirname(fileURLToPath(import.meta.url));
const arg = (name, fallback) => {
  const i = process.argv.indexOf(`--${name}`);
  return i < 0 ? fallback : process.argv[i + 1];
};
const assets = new Map();
for (const name of ['splat', 'mesh', 'cameras']) {
  const value = arg(name, null);
  if (!value || !fs.statSync(value).isFile()) throw new Error(`missing --${name}`);
  assets.set(`/${name}`, path.resolve(value));
}
const port = Number(arg('port', '8788'));
const server = http.createServer((req, res) => {
  if (req.method !== 'GET' && req.method !== 'HEAD') { res.writeHead(405).end(); return; }
  let url;
  try { url = decodeURIComponent(new URL(req.url, 'http://localhost').pathname); }
  catch { res.writeHead(400).end(); return; }
  let file = assets.get(url);
  if (!file && ['/node_modules/', '/viewer.html', '/src/viewer_entry.mjs'].some(p => url.startsWith(p))) {
    const candidate = path.resolve(root, `.${url}`);
    if (candidate.startsWith(root + path.sep)) file = candidate;
  }
  if (url === '/') file = path.join(root, 'viewer.html');
  if (!file || !fs.existsSync(file) || !fs.statSync(file).isFile()) { res.writeHead(404).end(); return; }
  const types = {'.html':'text/html', '.mjs':'text/javascript', '.js':'text/javascript', '.json':'application/json',
                 '.wasm':'application/wasm', '.glb':'model/gltf-binary'};
  res.writeHead(200, {'Content-Type': types[path.extname(file)] || 'application/octet-stream',
    'Content-Length': fs.statSync(file).size, 'Cache-Control':'no-store', 'X-Content-Type-Options':'nosniff'});
  if (req.method === 'HEAD') { res.end(); return; }
  const stream = fs.createReadStream(file);
  stream.on('error', () => res.destroy());
  stream.pipe(res);
});
server.listen(port, '127.0.0.1', () => console.log(JSON.stringify({url:`http://127.0.0.1:${port}`,
  scope:'loopback only', claim_ceiling:'development_only'})));
for (const signal of ['SIGINT', 'SIGTERM']) process.on(signal, () => server.close(() => process.exit(0)));
