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
  meshes: [],
  backgroundMesh: null,
  objectMeshes: new Map(),
  bounds: null,
  canvas: null,
  overlayGroup: null,
  composition: null,
};

function setupRenderer(width, height, clearColor) {
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
  return { canvas, renderer, scene, camera };
}

function transformedMeshBounds(mesh) {
  mesh.updateMatrixWorld(true);
  let local = null;
  try {
    local = mesh.getBoundingBox(true);
  } catch (e) {
    local = null;
  }
  if (!local || local.isEmpty() || !isFinite(local.min.x)) {
    local = new THREE.Box3();
    mesh.forEachSplat((_i, center) => local.expandByPoint(center));
  }
  const world = new THREE.Box3();
  for (const x of [local.min.x, local.max.x]) {
    for (const y of [local.min.y, local.max.y]) {
      for (const z of [local.min.z, local.max.z]) {
        world.expandByPoint(new THREE.Vector3(x, y, z).applyMatrix4(mesh.matrixWorld));
      }
    }
  }
  return { local, world };
}

function finalizeSceneState({ canvas, renderer, scene, camera, meshes, backgroundMesh, objectMeshes }) {
  const box = new THREE.Box3();
  const meshDiagnostics = [];
  for (const mesh of meshes) {
    const bounds = transformedMeshBounds(mesh);
    box.union(bounds.world);
    meshDiagnostics.push({
      id: mesh.name,
      splat_count: mesh.numSplats != null ? mesh.numSplats : null,
      local_min: bounds.local.min.toArray(),
      local_max: bounds.local.max.toArray(),
      world_min: bounds.world.min.toArray(),
      world_max: bounds.world.max.toArray(),
      matrixWorld: mesh.matrixWorld.elements.map((value) => +value.toFixed(6)),
    });
  }
  if (box.isEmpty() || !isFinite(box.min.x)) throw new Error("composite_splat_bounds_invalid");
  const center = box.getCenter(new THREE.Vector3());
  const size = box.getSize(new THREE.Vector3());
  const sphere = box.getBoundingSphere(new THREE.Sphere());
  state.renderer = renderer;
  state.scene = scene;
  state.camera = camera;
  state.mesh = backgroundMesh || meshes[0] || null;
  state.meshes = meshes;
  state.backgroundMesh = backgroundMesh || null;
  state.objectMeshes = objectMeshes || new Map();
  state.canvas = canvas;
  state.bounds = {
    center: center.toArray(),
    size: size.toArray(),
    radius: sphere.radius,
    min: box.min.toArray(),
    max: box.max.toArray(),
    splatCount: meshes.reduce((sum, mesh) => sum + Number(mesh.numSplats || 0), 0),
  };
  state._diag = { meshes: meshDiagnostics };
  return { ...state.bounds, _diag: state._diag };
}

async function load(url, width, height, clearColor) {
  const initialized = setupRenderer(width, height, clearColor);
  const { scene } = initialized;

  const mesh = new SplatMesh({ url });
  await mesh.initialized;
  mesh.name = "background";
  scene.add(mesh);
  mesh.updateMatrixWorld(true);
  state.composition = null;
  return finalizeSceneState({
    ...initialized,
    meshes: [mesh],
    backgroundMesh: mesh,
    objectMeshes: new Map(),
  });
}

async function loadComposite(backgroundUrl, objects, width, height, clearColor) {
  if (!Array.isArray(objects) || objects.length === 0) throw new Error("composite_objects_missing");
  const objectIds = objects.map((item) => String(item.object_id || ""));
  if (objectIds.some((id) => !id) || new Set(objectIds).size !== objectIds.length) {
    throw new Error("composite_object_ids_missing_or_duplicate");
  }
  const initialized = setupRenderer(width, height, clearColor);
  const { scene } = initialized;
  const backgroundMesh = new SplatMesh({ url: backgroundUrl });
  await backgroundMesh.initialized;
  backgroundMesh.name = "background";
  scene.add(backgroundMesh);
  const objectMeshes = new Map();
  for (const item of objects) {
    const mesh = new SplatMesh({ url: item.url });
    await mesh.initialized;
    mesh.name = String(item.object_id);
    const matrix = item.T_world_object;
    if (!Array.isArray(matrix) || matrix.length !== 4 || matrix.some((row) => !Array.isArray(row) || row.length !== 4)) {
      throw new Error(`composite_object_transform_invalid:${mesh.name}`);
    }
    const world = new THREE.Matrix4().set(...matrix.flat().map(Number));
    world.decompose(mesh.position, mesh.quaternion, mesh.scale);
    mesh.updateMatrixWorld(true);
    scene.add(mesh);
    objectMeshes.set(mesh.name, mesh);
  }
  const meshes = [backgroundMesh, ...objectMeshes.values()];
  const bounds = finalizeSceneState({
    ...initialized,
    meshes,
    backgroundMesh,
    objectMeshes,
  });
  state.composition = {
    background_instance_count: 1,
    object_ids: [...objectMeshes.keys()].sort(),
    visual_instance_counts: Object.fromEntries([...objectMeshes.keys()].sort().map((id) => [id, 1])),
    exactly_one_visual_instance_per_object: [...objectMeshes.values()].every((mesh) => mesh.visible),
  };
  return { ...bounds, composition: state.composition };
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

// spec (framed form):  { pos:[x,y,z], target:[x,y,z], fov:number, up?:[x,y,z] }
// spec (exact form):   { pose: { T_world_camera_opencv: number[4][4] },
//                        intrinsics: { fx, fy, cx, cy, width, height, near?, far? } }
// The exact form reproduces an OpenCV-convention pinhole camera (x right, y down,
// z forward; principal point in COLMAP pixel coordinates) via an off-axis
// projection matrix, for sealed held-out view evaluation.
async function renderView(spec, settleFrames = 10, settleMs = 110) {
  const { camera, renderer, scene, canvas, bounds } = state;
  if (spec.pose && spec.intrinsics) {
    const K = spec.intrinsics;
    if (canvas.width !== K.width || canvas.height !== K.height) {
      throw new Error(`canvas_size_${canvas.width}x${canvas.height}_mismatch_intrinsics_${K.width}x${K.height}`);
    }
    const near = K.near != null ? K.near : Math.max(1e-4, bounds.radius / 5000);
    const far = K.far != null ? K.far : bounds.radius * 50 + 1000;
    const m = spec.pose.T_world_camera_opencv;
    // OpenCV camera-to-world -> three.js (OpenGL) camera-to-world: flip the
    // Y and Z basis columns.
    const converted = new THREE.Matrix4().set(
      m[0][0], -m[0][1], -m[0][2], m[0][3],
      m[1][0], -m[1][1], -m[1][2], m[1][3],
      m[2][0], -m[2][1], -m[2][2], m[2][3],
      0, 0, 0, 1,
    );
    converted.decompose(camera.position, camera.quaternion, camera.scale);
    camera.updateMatrixWorld(true);
    const left = (-K.cx * near) / K.fx;
    const right = ((K.width - K.cx) * near) / K.fx;
    const top = (K.cy * near) / K.fy;
    const bottom = (-(K.height - K.cy) * near) / K.fy;
    camera.projectionMatrix.makePerspective(left, right, top, bottom, near, far);
    camera.projectionMatrixInverse.copy(camera.projectionMatrix).invert();
  } else {
    camera.fov = spec.fov || 50;
    camera.up.set(...(spec.up || [0, 1, 0]));
    camera.position.set(...spec.pos);
    camera.near = Math.max(1e-4, bounds.radius / 5000);
    camera.far = bounds.radius * 50 + 1000;
    camera.lookAt(new THREE.Vector3(...spec.target));
    camera.updateProjectionMatrix();
  }
  camera.updateMatrixWorld(true);
  for (let k = 0; k < settleFrames; k++) {
    renderer.render(scene, camera);
    await delay(settleMs);
  }
  return canvas.toDataURL("image/png");
}

window.BlueprintSplat = { load, loadComposite, setOverlay, warmup, renderView };
window.__sparkReady = true;
