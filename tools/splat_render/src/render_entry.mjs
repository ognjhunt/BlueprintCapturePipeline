// Headless Spark.js render entry — bundled to dist/spark_bundle.js (IIFE) and loaded by harness.html.
// Exposes window.BlueprintSplat.{load, renderView} so a Playwright driver can frame the splat
// to its real bounding box and capture per-camera images of the actual captured scene.
import * as THREE from "three";
import { SplatMesh, SparkRenderer } from "@sparkjsdev/spark";

const state = {
  renderer: null,
  scene: null,
  camera: null,
  mesh: null,
  bounds: null,
  canvas: null,
  overlayGroup: null,
};

async function load(url, width, height, clearColor) {
  const canvas = document.getElementById("c");
  canvas.width = width;
  canvas.height = height;
  const renderer = new THREE.WebGLRenderer({
    canvas,
    antialias: true,
    preserveDrawingBuffer: true,
    alpha: false,
  });
  renderer.setPixelRatio(1);
  renderer.setSize(width, height, false);
  renderer.setClearColor(clearColor == null ? 0x0b0b10 : clearColor, 1);

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(50, width / height, 0.01, 100000);

  const spark = new SparkRenderer({ renderer });
  scene.add(spark);

  const mesh = new SplatMesh({ url });
  await mesh.initialized;
  scene.add(mesh);
  mesh.updateMatrixWorld(true);

  // Spark stores splats in packed GPU buffers (not a standard three geometry), so
  // Box3.setFromObject() returns empty. Use Spark's own splat-centers bounding box.
  let box = null;
  try {
    box = mesh.getBoundingBox(true);
  } catch (e) {
    box = null;
  }
  if (!box || box.isEmpty() || !isFinite(box.min.x)) {
    box = new THREE.Box3();
    mesh.forEachSplat((_i, center) => box.expandByPoint(center));
  }
  const center = box.getCenter(new THREE.Vector3());
  const size = box.getSize(new THREE.Vector3());
  const sphere = box.getBoundingSphere(new THREE.Sphere());

  // --- diagnostics: manual bbox + sample centers + world transform ---
  const manual = new THREE.Box3();
  const samples = [];
  let seen = 0;
  mesh.forEachSplat((i, c) => {
    manual.expandByPoint(c);
    if (seen < 5) samples.push([+c.x.toFixed(3), +c.y.toFixed(3), +c.z.toFixed(3)]);
    seen++;
  });
  state._diag = {
    getBoundingBox_min: box.min.toArray().map((v) => +v.toFixed(3)),
    getBoundingBox_max: box.max.toArray().map((v) => +v.toFixed(3)),
    forEachSplat_min: manual.min.toArray().map((v) => +v.toFixed(3)),
    forEachSplat_max: manual.max.toArray().map((v) => +v.toFixed(3)),
    mesh_position: mesh.position.toArray(),
    mesh_quaternion: mesh.quaternion.toArray(),
    mesh_scale: mesh.scale.toArray(),
    matrixWorld: mesh.matrixWorld.elements.map((v) => +v.toFixed(4)),
    sample_centers: samples,
  };

  state.renderer = renderer;
  state.scene = scene;
  state.camera = camera;
  state.mesh = mesh;
  state.canvas = canvas;
  state.bounds = {
    center: center.toArray(),
    size: size.toArray(),
    radius: sphere.radius,
    min: box.min.toArray(),
    max: box.max.toArray(),
    splatCount: mesh.numSplats != null ? mesh.numSplats : null,
  };
  return { ...state.bounds, _diag: state._diag };
}

function setOverlay(items = []) {
  const { scene } = state;
  if (!scene) throw new Error("splat_not_loaded");
  if (state.overlayGroup) scene.remove(state.overlayGroup);
  const group = new THREE.Group();
  group.name = "blueprint_hybrid_preview_overlay";
  for (const item of items) {
    const shape = item.shape || "box";
    let geometry;
    if (shape === "cylinder") {
      geometry = new THREE.CylinderGeometry(0.5, 0.5, 1, 24);
    } else {
      geometry = new THREE.BoxGeometry(1, 1, 1);
    }
    const material = new THREE.MeshBasicMaterial({
      color: item.color == null ? 0x22ccff : Number(item.color),
      opacity: item.opacity == null ? 0.92 : Number(item.opacity),
      transparent: Number(item.opacity == null ? 0.92 : item.opacity) < 1,
      depthTest: true,
      depthWrite: true,
    });
    const mesh = new THREE.Mesh(geometry, material);
    mesh.name = String(item.id || shape);
    mesh.position.set(...(item.position || [0, 0, 0]));
    mesh.scale.set(...(item.scale || [1, 1, 1]));
    if (item.rotation_euler_rad) mesh.rotation.set(...item.rotation_euler_rad);
    group.add(mesh);
  }
  scene.add(group);
  state.overlayGroup = group;
  return { item_count: group.children.length, ids: group.children.map((item) => item.name) };
}

const delay = (ms) => new Promise((r) => setTimeout(r, ms));

// Spark decodes splats in a worker and uploads/sorts them to the GPU across several
// frames AFTER mesh.initialized resolves; a one-shot render captures an empty buffer.
// Pump render frames (here and per-view) so the upload + view-dependent sort settle.
async function warmup(ms = 2500) {
  const { renderer, scene, camera } = state;
  const t0 = performance.now();
  while (performance.now() - t0 < ms) {
    renderer.render(scene, camera);
    await delay(80);
  }
}

// spec: { pos:[x,y,z], target:[x,y,z], fov:number, up?:[x,y,z] }
async function renderView(spec, settleFrames = 10, settleMs = 110) {
  const { camera, renderer, scene, canvas, bounds } = state;
  camera.fov = spec.fov || 50;
  camera.up.set(...(spec.up || [0, 1, 0]));
  camera.position.set(...spec.pos);
  camera.near = Math.max(1e-4, bounds.radius / 5000);
  camera.far = bounds.radius * 50 + 1000;
  camera.lookAt(new THREE.Vector3(...spec.target));
  camera.updateProjectionMatrix();
  camera.updateMatrixWorld(true);
  for (let k = 0; k < settleFrames; k++) {
    renderer.render(scene, camera);
    await delay(settleMs);
  }
  return canvas.toDataURL("image/png");
}

window.BlueprintSplat = { load, setOverlay, warmup, renderView };
window.__sparkReady = true;
