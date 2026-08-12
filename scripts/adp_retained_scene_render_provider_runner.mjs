// Execute the exact retained-scene render packet. This script never fetches a
// dependency or opens a provider API; it renders only the derived PLY inputs
// copied into its immutable bundle.
import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { spawnSync } from "node:child_process";

function argument(name) {
  const index = process.argv.indexOf(`--${name}`);
  return index >= 0 ? process.argv[index + 1] : null;
}

const runtime = path.resolve(argument("runtime") || "");
const output = path.resolve(argument("output") || "");
const rehearsal = process.argv.includes("--rehearsal");

function sha256(file) {
  return `sha256:${crypto.createHash("sha256").update(fs.readFileSync(file)).digest("hex")}`;
}

function stableJson(value) {
  if (Array.isArray(value)) return value.map(stableJson);
  if (value && typeof value === "object") {
    return Object.fromEntries(Object.keys(value).sort().map((key) => [key, stableJson(value[key])]));
  }
  return value;
}

function canonicalDigest(value, digestField) {
  const normalized = { ...value };
  delete normalized[digestField];
  return `sha256:${crypto.createHash("sha256").update(JSON.stringify(stableJson(normalized))).digest("hex")}`;
}

function write(value) {
  fs.mkdirSync(output, { recursive: true });
  fs.writeFileSync(
    path.join(output, "adp009d_retained_scene_gpu_render_result.v1.json"),
    `${JSON.stringify(value, null, 2)}\n`,
  );
}

function fail(code, rendererDiagnostic = null) {
  const result = {
    schema_version: "adp009d_retained_scene_gpu_render_result.v1",
    status: "blocked",
    released_renderer_executed: false,
    gpu_runtime_started: false,
    paid_inference_performed: false,
    provider_mutations_performed: 0,
    blockers: [code],
  };
  if (rendererDiagnostic) result.renderer_diagnostic = rendererDiagnostic;
  write(result);
  process.exitCode = 2;
  return result;
}

function inside(root, relative) {
  if (typeof relative !== "string" || !relative || path.isAbsolute(relative)) return null;
  const resolved = path.resolve(root, relative);
  return resolved.startsWith(`${root}${path.sep}`) ? resolved : null;
}

function checkedFile(root, record) {
  const target = inside(root, record?.relative_path);
  if (
    !target ||
    !fs.existsSync(target) ||
    fs.lstatSync(target).isSymbolicLink() ||
    fs.statSync(target).size !== record?.size_bytes ||
    sha256(target) !== record?.sha256
  ) {
    throw new Error("retained_scene_render_runtime_file_binding_invalid");
  }
  return target;
}

function standardPlyCount(file) {
  const header = fs.readFileSync(file, { encoding: "latin1" }).slice(0, 1024 * 1024);
  const match = /^element vertex ([0-9]+)\r?$/m.exec(header);
  return match ? Number(match[1]) : NaN;
}

function cameraSpecs(cameraRows) {
  if (!Array.isArray(cameraRows) || !cameraRows.length) {
    throw new Error("retained_scene_render_runtime_camera_contract_invalid");
  }
  const ids = new Set();
  const specs = cameraRows.map((row) => {
    const id = String(row?.camera_id || "");
    const pose = row?.T_world_camera_provider_frame;
    const intrinsics = row?.intrinsics;
    if (
      !id ||
      ids.has(id) ||
      !Array.isArray(pose) ||
      pose.length !== 4 ||
      !pose.every((line) => Array.isArray(line) && line.length === 4) ||
      !intrinsics
    ) {
      throw new Error("retained_scene_render_runtime_camera_contract_invalid");
    }
    ids.add(id);
    const normalized = {};
    for (const key of ["fx", "fy", "cx", "cy", "width", "height"]) {
      const value = Number(intrinsics[key]);
      if (!Number.isFinite(value) || value <= 0) {
        throw new Error("retained_scene_render_runtime_camera_contract_invalid");
      }
      normalized[key] = key === "width" || key === "height" ? Math.trunc(value) : value;
    }
    return { id, spec: { pose: { T_world_camera_opencv: pose }, intrinsics: normalized } };
  });
  return specs;
}

function gpuIdentity() {
  const probe = spawnSync(
    "nvidia-smi",
    ["--query-gpu=name,driver_version", "--format=csv,noheader"],
    { encoding: "utf8", timeout: 10_000 },
  );
  const rows = probe.status === 0 ? String(probe.stdout || "").trim().split("\n").filter(Boolean) : [];
  if (!rows.length) throw new Error("retained_scene_render_gpu_unavailable");
  return { nvidia_smi_detected: true, gpu_rows: rows };
}

function renderOne({ request, lane, variant, deleted, retained, gpu }) {
  const layer = variant.layer;
  const deletedSourceLayer = layer === "shared_deleted_source_layer";
  const splat = deletedSourceLayer ? deleted : retained;
  const splatRecord = deletedSourceLayer
    ? request.shared_deleted_source_layer
    : request.shared_retained_scene;
  const cameraPath = checkedFile(runtime, lane.camera_contract);
  const cameras = cameraSpecs(JSON.parse(fs.readFileSync(cameraPath, "utf8")));
  const dimensions = lane.dimensions || {};
  if (cameras.some(({ spec }) => spec.intrinsics.width !== dimensions.width || spec.intrinsics.height !== dimensions.height)) {
    throw new Error("retained_scene_render_runtime_camera_dimensions_invalid");
  }
  const label = `${lane.task_id}_${layer}_${variant.background_rgb === "#ffffff" ? "white" : "black"}`;
  const root = path.join(output, "renders", lane.task_id, label);
  const cameraSpecsPath = path.join(root, "exact_cameras.json");
  const frames = path.join(root, "frames");
  fs.mkdirSync(root, { recursive: true });
  fs.writeFileSync(cameraSpecsPath, `${JSON.stringify(cameras)}\n`);
  const background = variant.background_rgb === "#ffffff" ? "0xffffff" : "0x000000";
  const command = [
    path.join(runtime, "renderer", "render_splat.mjs"),
    "--splat", splat,
    "--out", frames,
    "--cameras", cameraSpecsPath,
    "--width", String(dimensions.width),
    "--height", String(dimensions.height),
    "--warmup-ms", "2500",
    "--settle-frames", "6",
    "--settle-ms", "100",
    "--load-timeout-ms", "600000",
    "--graphics-backend", "egl",
    "--bg", background,
  ];
  const rendered = spawnSync(process.execPath, command, { encoding: "utf8", timeout: 900_000 });
  if (rendered.error || rendered.status !== 0) {
    const error = new Error("retained_scene_render_runtime_renderer_failed");
    error.rendererDiagnostic = {
      command: "render_splat.mjs",
      exit_status: rendered.status ?? null,
      signal: rendered.signal ?? null,
      error_code: rendered.error?.code ?? null,
      error_name: rendered.error?.name ?? null,
      stdout_tail: String(rendered.stdout || "").slice(-4000),
      stderr_tail: String(rendered.stderr || "").slice(-4000),
    };
    throw error;
  }
  let harness;
  try {
    harness = JSON.parse(rendered.stdout);
  } catch {
    throw new Error("retained_scene_render_runtime_renderer_output_invalid");
  }
  if (
    harness.status !== "completed" ||
    !harness.graphics_diagnostics?.webgl_available ||
    /swiftshader|software|llvmpipe/i.test(String(harness.graphics_diagnostics?.renderer || ""))
  ) {
    throw new Error("retained_scene_render_runtime_software_renderer_rejected");
  }
  const rows = cameras.map(({ id }) => {
    const frame = path.join(frames, `${id}.png`);
    if (!fs.existsSync(frame) || fs.statSync(frame).size <= 0) {
      throw new Error("retained_scene_render_runtime_frame_missing");
    }
    return {
      camera_id: id,
      relative_path: path.relative(root, frame),
      size_bytes: fs.statSync(frame).size,
      digest: sha256(frame),
      width: dimensions.width,
      height: dimensions.height,
    };
  });
  const manifest = {
    schema_version: "sealed_camera_render_manifest.v1",
    status: "rendered_exact_cameras",
    rendered_by: "reference_spark_renderer_exact_camera",
    rendered_by_gpu: true,
    gpu_identity: gpu,
    camera_set_label: `${lane.task_id}:shared_retained_scene_gpu:${label}`,
    purpose: "ADP-009D exact retained-scene residual inpainting input",
    authorization_class: "method_input",
    candidate_set_digest: request.candidate_set_digest,
    splat_digest: splatRecord.sha256,
    source_splat: {
      digest: splatRecord.sha256,
      retained_gaussian_count: splatRecord.gaussian_count,
      retained_count_source: "verified_standard_ply_header",
    },
    source_layer_role: deletedSourceLayer
      ? "shared_deleted_source_union"
      : "shared_retained_scene",
    calibrated_camera_file: {
      digest: lane.camera_contract.sha256,
      binding: "caller_file_exact_match",
      camera_count: cameras.length,
    },
    calibrated_cameras: cameras,
    render_settings: {
      dimensions,
      supersampling: 1,
      color_space: "srgb",
      alpha_mode: "opaque_rgb",
      background_rgb: variant.background_rgb,
      exposure: { mode: "renderer_default_unmodified", ev: null },
    },
    renderer_identity: {
      ...request.renderer_identity,
      graphics_backend: "egl",
      graphics_diagnostics: harness.graphics_diagnostics,
    },
    renders: rows,
    render_count: rows.length,
    rendered_by_isaac_rtx: false,
    hidden_pixels_read_by_renderer: false,
    proof_effect: "reference_render_for_exact_mask_contained_inpainting_input_only",
    claim_ceiling: "appearance_reconstruction_candidate",
  };
  manifest.sealed_camera_render_manifest_digest = canonicalDigest(
    manifest,
    "sealed_camera_render_manifest_digest",
  );
  const manifestPath = path.join(root, "sealed_camera_render_manifest.v1.json");
  fs.writeFileSync(manifestPath, `${JSON.stringify(manifest, null, 2)}\n`);
  return { task_id: lane.task_id, layer, background_rgb: variant.background_rgb, manifest_path: manifestPath, manifest_digest: manifest.sealed_camera_render_manifest_digest };
}

function main() {
  if (!runtime || !output || !fs.existsSync(runtime)) return fail("retained_scene_render_runtime_path_invalid");
  let request;
  try {
    request = JSON.parse(fs.readFileSync(path.join(runtime, "render_request.json"), "utf8"));
    const deleted = checkedFile(runtime, request.shared_deleted_source_layer);
    const retained = checkedFile(runtime, request.shared_retained_scene);
    checkedFile(runtime, request.candidate_set);
    checkedFile(runtime, request.execution_authority);
    if (standardPlyCount(deleted) !== request.shared_deleted_source_layer.gaussian_count || standardPlyCount(retained) !== request.shared_retained_gaussian_count) {
      throw new Error("retained_scene_render_runtime_ply_count_invalid");
    }
    for (const lane of request.lanes || []) {
      checkedFile(runtime, lane.camera_contract);
      checkedFile(runtime, lane.task_freeze);
    }
    if (rehearsal) {
      const result = {
        schema_version: "provider_bundle_rehearsal.v1",
        status: "passed",
        released_renderer_executed: false,
        gpu_runtime_started: false,
        paid_inference_performed: false,
        provider_mutations_performed: 0,
        verified_task_lanes: request.lanes.length,
        blockers: [],
      };
      fs.mkdirSync(output, { recursive: true });
      fs.writeFileSync(path.join(output, "provider_bundle_rehearsal.json"), `${JSON.stringify(result, null, 2)}\n`);
      return;
    }
    const gpu = gpuIdentity();
    const variants = [];
    for (const lane of request.lanes || []) {
      for (const variant of lane.render_variants || []) {
        variants.push(renderOne({ request, lane, variant, deleted, retained, gpu }));
      }
    }
    const result = {
      schema_version: "adp009d_retained_scene_gpu_render_result.v1",
      status: "completed",
      released_renderer_executed: true,
      gpu_runtime_started: true,
      paid_inference_performed: false,
      provider_mutations_performed: 0,
      candidate_set_digest: request.candidate_set_digest,
      request_digest: request.request_digest,
      shared_deleted_source_layer_digest: request.shared_deleted_source_layer.sha256,
      shared_retained_scene_digest: request.shared_retained_scene.sha256,
      render_manifests: variants,
      blockers: [],
    };
    write(result);
  } catch (error) {
    fail(
      error instanceof Error ? error.message : "retained_scene_render_runtime_unknown_failure",
      error instanceof Error ? error.rendererDiagnostic || null : null,
    );
  }
}

main();
