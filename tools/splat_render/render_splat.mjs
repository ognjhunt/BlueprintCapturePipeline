// Headless Spark.js splat renderer driver.
// Loads a Gaussian-splat (.ply/.compressed.ply/.spz) in headless Chromium via Spark.js,
// frames the eval's 6 camera IDs to the splat's real bounding box, and writes per-camera PNGs.
//
// Usage:
//   node render_splat.mjs --splat <file> --out <dir> [--cameras <json>] [--width N --height N] [--bg 0xRRGGBB]
//
// --cameras json (optional): [{ "id": "...", "spec": { "pos":[x,y,z], "target":[x,y,z], "fov":N, "up":[x,y,z] } }, ...]
// If omitted, six framed views are derived from the splat bounds (overhead/third_person/head_pov/torso/wrist/task_focus).
//
// Emits a JSON manifest to stdout: { status, bounds, cameras:[{id,path,bytes,nonblank_score}], blockers:[] }
import { chromium } from "playwright";
import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import zlib from "node:zlib";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

function arg(name, def = null) {
  const i = process.argv.indexOf("--" + name);
  return i !== -1 && i + 1 < process.argv.length ? process.argv[i + 1] : def;
}

const splatPath = path.resolve(arg("splat"));
const outDir = path.resolve(arg("out"));
const camerasArg = arg("cameras", null);
const width = parseInt(arg("width", "1280"), 10);
const height = parseInt(arg("height", "960"), 10);
const bg = parseInt(arg("bg", "0x0b0b10"));
const loadTimeoutMs = parseInt(arg("load-timeout-ms", "180000"), 10);
const warmupMs = parseInt(arg("warmup-ms", "2000"), 10);
const settleFrames = parseInt(arg("settle-frames", "6"), 10);
const settleMs = parseInt(arg("settle-ms", "100"), 10);

if (!splatPath || !outDir) {
  console.error("usage: node render_splat.mjs --splat <file> --out <dir> [--cameras <json>] [--width N --height N]");
  process.exit(2);
}
fs.mkdirSync(outDir, { recursive: true });

// Vector helpers (scene up = +Y; Spark places splats in three.js Y-up world space).
const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const scl = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
function dir(azimuthDeg, elevationDeg) {
  const az = (azimuthDeg * Math.PI) / 180;
  const el = (elevationDeg * Math.PI) / 180;
  // azimuth around +Y, elevation above XZ plane
  return [Math.cos(el) * Math.sin(az), Math.sin(el), Math.cos(el) * Math.cos(az)];
}

function deriveDefaultCameras(b) {
  const c = b.center;
  const r = Math.max(b.radius, 1e-3);
  const [sx, sy, sz] = b.size;
  const cams = [];
  // overhead: top-down (look down -Y); use -Z as image-up to avoid degenerate up vector
  cams.push({ id: "overhead", spec: { pos: [c[0], c[1] + r * 1.9, c[2] + r * 1e-3], target: c, fov: 55, up: [0, 0, -1] } });
  // third_person: orbiting chase, whole scene in frame
  cams.push({ id: "third_person", spec: { pos: add(c, scl(dir(-42, 22), r * 2.1)), target: c, fov: 50 } });
  // head_pov: first-person eye height, looking across the room along +X
  cams.push({ id: "head_pov", spec: { pos: [c[0] - sx * 0.34, c[1] + sy * 0.08, c[2]], target: [c[0] + sx * 0.55, c[1] + sy * 0.02, c[2]], fov: 62 } });
  // torso: lower, pulled back, full-room context
  cams.push({ id: "torso", spec: { pos: add(c, scl(dir(28, 10), r * 1.7)), target: c, fov: 56 } });
  // wrist: close-up near floor/task zone
  cams.push({ id: "wrist", spec: { pos: [c[0] + sx * 0.15, c[1] - sy * 0.18, c[2] + sz * 0.18], target: [c[0] + sx * 0.34, c[1] - sy * 0.32, c[2] + sz * 0.34], fov: 58 } });
  // task_focus: zoomed orbit on one quadrant
  cams.push({ id: "task_focus", spec: { pos: add(c, scl(dir(64, 16), r * 1.2)), target: add(c, [sx * 0.2, -sy * 0.05, sz * 0.2]), fov: 48 } });
  return cams;
}

// crude non-blank score from a PNG buffer: count distinct-ish bytes in the compressed stream
function nonblankScore(pngBuf) {
  // a fully uniform image compresses to a tiny PNG; richer images are larger per pixel
  return pngBuf.length;
}

const contentTypes = {
  ".html": "text/html",
  ".js": "text/javascript",
  ".mjs": "text/javascript",
  ".css": "text/css",
  ".ply": "application/octet-stream",
  ".spz": "application/octet-stream",
};

async function main() {
  const blockers = [];
  // static file server: tool dir + the splat at /splat
  const server = http.createServer((req, res) => {
    const urlPath = decodeURIComponent((req.url || "/").split("?")[0]);
    const file = urlPath === "/splat" ? splatPath : path.join(__dirname, urlPath);
    fs.readFile(file, (err, data) => {
      if (err) {
        res.writeHead(404);
        res.end("not found");
        return;
      }
      const ext = path.extname(file).toLowerCase();
      res.writeHead(200, {
        "content-type": contentTypes[ext] || "application/octet-stream",
        "access-control-allow-origin": "*",
        "content-length": data.length,
      });
      res.end(data);
    });
  });
  await new Promise((r) => server.listen(0, "127.0.0.1", r));
  const port = server.address().port;
  const base = `http://127.0.0.1:${port}`;

  const browser = await chromium.launch({
    headless: true,
    args: [
      "--use-gl=angle",
      "--use-angle=swiftshader",
      "--ignore-gpu-blocklist",
      "--enable-unsafe-swiftshader",
      "--disable-dev-shm-usage",
    ],
  });
  const page = await browser.newPage({ viewport: { width, height } });
  const pageErrors = [];
  page.on("pageerror", (e) => pageErrors.push(String(e.message || e)));
  page.on("console", (m) => {
    if (m.type() === "error") pageErrors.push("console.error: " + m.text());
  });

  let result = { status: "blocked", bounds: null, cameras: [], blockers, page_errors: pageErrors };
  try {
    await page.goto(`${base}/harness.html`, { waitUntil: "load", timeout: 60000 });
    await page.waitForFunction("window.__sparkReady===true", { timeout: 60000 });

    const bounds = await page.evaluate(
      async ({ u, w, h, bgc }) => await window.BlueprintSplat.load(u, w, h, bgc),
      { u: `${base}/splat`, w: width, h: height, bgc: bg },
    );
    result.bounds = bounds;
    if (!bounds || !isFinite(bounds.radius) || bounds.radius <= 0) {
      blockers.push("splat_bounds_invalid_after_load");
    }

    // settle the async splat upload before capturing any view
    await page.evaluate((ms) => window.BlueprintSplat.warmup(ms), warmupMs);

    const cameras = camerasArg ? JSON.parse(fs.readFileSync(camerasArg, "utf8")) : deriveDefaultCameras(bounds);

    for (const cam of cameras) {
      const dataUrl = await page.evaluate(
        ({ spec, sf, sm }) => window.BlueprintSplat.renderView(spec, sf, sm),
        { spec: cam.spec, sf: settleFrames, sm: settleMs },
      );
      const b64 = String(dataUrl).split(",")[1] || "";
      const buf = Buffer.from(b64, "base64");
      const outPath = path.join(outDir, `${cam.id}.png`);
      fs.writeFileSync(outPath, buf);
      result.cameras.push({ id: cam.id, path: outPath, bytes: buf.length, nonblank_score: nonblankScore(buf), spec: cam.spec });
    }
    result.status = blockers.length ? "blocked" : "completed";
  } catch (e) {
    blockers.push("render_harness_exception");
    result.error = String(e && e.stack ? e.stack : e);
  } finally {
    await browser.close().catch(() => {});
    server.close();
  }
  process.stdout.write(JSON.stringify(result, null, 2) + "\n");
  process.exit(result.status === "completed" ? 0 : 1);
}

main().catch((e) => {
  console.error(e);
  process.exit(3);
});
