// Headless Spark.js splat renderer driver.
// Loads a Gaussian-splat (.ply/.compressed.ply/.spz) in headless Chromium via Spark.js,
// frames the eval's 6 camera IDs to the splat's real bounding box, and writes per-camera PNGs.
//
// Usage:
//   node render_splat.mjs --splat <file> --out <dir> [--cameras <json>] [--width N --height N] [--bg 0xRRGGBB] [--graphics-backend swiftshader|metal|egl]
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

const splatArg = arg("splat", null);
const compositionArg = arg("composition", null);
const splatPath = splatArg ? path.resolve(splatArg) : null;
const outDir = path.resolve(arg("out"));
const camerasArg = arg("cameras", null);
const overlayArg = arg("overlay", null);
const width = parseInt(arg("width", "1280"), 10);
const height = parseInt(arg("height", "960"), 10);
const bg = parseInt(arg("bg", "0x0b0b10"));
const loadTimeoutMs = parseInt(arg("load-timeout-ms", "180000"), 10);
const warmupMs = parseInt(arg("warmup-ms", "2000"), 10);
const settleFrames = parseInt(arg("settle-frames", "6"), 10);
const settleMs = parseInt(arg("settle-ms", "100"), 10);
const graphicsBackend = String(arg("graphics-backend", "swiftshader")).toLowerCase();

if (!["swiftshader", "metal", "egl"].includes(graphicsBackend)) {
  console.error(`unsupported --graphics-backend: ${graphicsBackend}`);
  process.exit(2);
}

function graphicsArguments(backend) {
  if (backend === "metal") {
    return ["--use-gl=angle", "--use-angle=metal", "--ignore-gpu-blocklist"];
  }
  if (backend === "egl") {
    // Chromium's hardware EGL route is an ANGLE backend.  ``--use-gl=egl``
    // itself can leave WebGL disabled in headless Chromium, including a
    // container that otherwise exposes an NVIDIA GPU through nvidia-smi.
    return [
      "--use-gl=angle",
      "--use-angle=gl-egl",
      "--ignore-gpu-blocklist",
      "--disable-software-rasterizer",
      "--enable-webgl",
    ];
  }
  return [
    "--use-gl=angle",
    "--use-angle=swiftshader",
    "--ignore-gpu-blocklist",
    "--enable-unsafe-swiftshader",
  ];
}

if (process.argv.includes("--print-graphics-args")) {
  process.stdout.write(`${JSON.stringify(graphicsArguments(graphicsBackend))}\n`);
  process.exit(0);
}

if ((!splatPath && !compositionArg) || (splatPath && compositionArg) || !outDir) {
  console.error("usage: node render_splat.mjs (--splat <file> | --composition <json>) --out <dir> [--cameras <json>] [--width N --height N]");
  process.exit(2);
}
fs.mkdirSync(outDir, { recursive: true });

let composition = null;
let compositeBackgroundPath = null;
const compositeObjectPaths = new Map();
if (compositionArg) {
  composition = JSON.parse(fs.readFileSync(path.resolve(compositionArg), "utf8"));
  if (composition.schema_version !== "dynamic_splat_render_request.v1") {
    console.error("dynamic_splat_render_request_schema_invalid");
    process.exit(2);
  }
  compositeBackgroundPath = path.resolve(String(composition.background?.path || ""));
  if (!fs.existsSync(compositeBackgroundPath)) {
    console.error("dynamic_splat_background_missing");
    process.exit(2);
  }
  if (!Array.isArray(composition.objects) || composition.objects.length === 0) {
    console.error("dynamic_splat_objects_missing");
    process.exit(2);
  }
  for (const item of composition.objects) {
    const objectId = String(item.object_id || "");
    const objectPath = path.resolve(String(item.path || ""));
    if (!objectId || compositeObjectPaths.has(objectId) || !fs.existsSync(objectPath)) {
      console.error("dynamic_splat_object_invalid_or_duplicate");
      process.exit(2);
    }
    compositeObjectPaths.set(objectId, objectPath);
  }
}

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

function withTimeout(promise, timeoutMs, label) {
  if (!Number.isFinite(timeoutMs) || timeoutMs <= 0) {
    return promise;
  }
  return Promise.race([
    promise,
    new Promise((_, reject) => {
      setTimeout(() => reject(new Error(`${label}_timeout_${timeoutMs}ms`)), timeoutMs);
    }),
  ]);
}

async function main() {
  const blockers = [];
  // Static file server: tool dir plus an explicit allowlist of splat assets.
  const server = http.createServer((req, res) => {
    const urlPath = decodeURIComponent((req.url || "/").split("?")[0]);
    let file;
    if (urlPath === "/splat" && splatPath) {
      file = splatPath;
    } else if (urlPath === "/composite/background" && compositeBackgroundPath) {
      file = compositeBackgroundPath;
    } else if (urlPath.startsWith("/composite/object/")) {
      const objectId = urlPath.slice("/composite/object/".length);
      file = compositeObjectPaths.get(objectId);
    } else {
      file = path.join(__dirname, urlPath);
    }
    if (!file) {
      res.writeHead(404);
      res.end("not found");
      return;
    }
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

  const graphicsArgs = graphicsArguments(graphicsBackend);
  const browser = await chromium.launch({
    headless: true,
    args: [
      ...graphicsArgs,
      "--disable-dev-shm-usage",
    ],
  });
  const page = await browser.newPage({ viewport: { width, height } });
  const pageErrors = [];
  page.on("pageerror", (e) => pageErrors.push(String(e.message || e)));
  page.on("console", (m) => {
    if (m.type() === "error") pageErrors.push("console.error: " + m.text());
  });

  let result = {
    status: "blocked",
    graphics_backend: graphicsBackend,
    bounds: null,
    cameras: [],
    overlay: null,
    blockers,
    page_errors: pageErrors,
    composition: null,
  };
  try {
    await page.goto(`${base}/harness.html`, { waitUntil: "load", timeout: 60000 });
    await page.waitForFunction("window.__sparkReady===true", { timeout: 60000 });
    let bounds;
    if (composition) {
      const objects = composition.objects.map((item) => ({
        object_id: String(item.object_id),
        url: `${base}/composite/object/${encodeURIComponent(String(item.object_id))}`,
        T_world_object: item.T_world_object,
      }));
      bounds = await withTimeout(
        page.evaluate(
          async ({ backgroundUrl, objects: objectRows, w, h, bgc }) =>
            await window.BlueprintSplat.loadComposite(backgroundUrl, objectRows, w, h, bgc),
          {
            backgroundUrl: `${base}/composite/background`,
            objects,
            w: width,
            h: height,
            bgc: bg,
          },
        ),
        loadTimeoutMs,
        "composite_splat_load",
      );
      result.composition = bounds?.composition || null;
      const expected = [...(composition.expected_object_ids || [])].map(String).sort();
      const observed = [...(result.composition?.object_ids || [])].map(String).sort();
      if (
        composition.require_exactly_one_visual_instance_per_object !== true
        || result.composition?.exactly_one_visual_instance_per_object !== true
        || JSON.stringify(expected) !== JSON.stringify(observed)
      ) {
        blockers.push("composite_exactly_once_invariant_failed");
      }
    } else {
      bounds = await withTimeout(
        page.evaluate(
          async ({ u, w, h, bgc }) => await window.BlueprintSplat.load(u, w, h, bgc),
          { u: `${base}/splat`, w: width, h: height, bgc: bg },
        ),
        loadTimeoutMs,
        "splat_load",
      );
    }
    result.bounds = bounds;
    if (!bounds || !isFinite(bounds.radius) || bounds.radius <= 0) {
      blockers.push("splat_bounds_invalid_after_load");
    }
    result.graphics_diagnostics = await page.evaluate(() => {
      const canvas = document.getElementById("c");
      const gl = canvas && (canvas.getContext("webgl2") || canvas.getContext("webgl"));
      if (!gl) return { webgl_available: false };
      const debug = gl.getExtension("WEBGL_debug_renderer_info");
      return {
        webgl_available: true,
        vendor: debug ? gl.getParameter(debug.UNMASKED_VENDOR_WEBGL) : null,
        renderer: debug ? gl.getParameter(debug.UNMASKED_RENDERER_WEBGL) : null,
      };
    });
    if (
      graphicsBackend === "egl" &&
      (!result.graphics_diagnostics.webgl_available ||
        /swiftshader|software|llvmpipe/i.test(String(result.graphics_diagnostics.renderer || "")))
    ) {
      blockers.push("egl_gpu_renderer_unavailable");
    }

    if (overlayArg) {
      const overlay = JSON.parse(fs.readFileSync(overlayArg, "utf8"));
      result.overlay = await page.evaluate(
        (items) => window.BlueprintSplat.setOverlay(items),
        Array.isArray(overlay) ? overlay : overlay.items,
      );
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
