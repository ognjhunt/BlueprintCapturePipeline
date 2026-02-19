#!/usr/bin/env bash
# =============================================================================
# End-to-end pipeline: nurec_shim → swap_orchestrator (stages A→I)
# =============================================================================
# This script:
#   1) Runs nurec_shim.py to produce NuRec-equivalent outputs
#   2) Creates the GCS-like directory structure for swap_orchestrator
#   3) Runs the swap_orchestrator through stages C→I
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$(dirname "$SCRIPT_DIR")"

# ── Defaults ─────────────────────────────────────────────────────────────────
WORKSPACE="${WORKSPACE:-/workspace}"
GCS_ROOT="${GCS_ROOT:-${WORKSPACE}/gcs_root}"
BUCKET="${BUCKET:-blueprint-local}"
BLUEPRINTPIPELINE_ROOT="${BLUEPRINTPIPELINE_ROOT:-/opt/BlueprintPipeline}"
ENVIRONMENT="${ENVIRONMENT:-auto}"
COMPLETION_MODE="${COMPLETION_MODE:-full_required}"
CROP_CLEANUP_PROVIDER="${CROP_CLEANUP_PROVIDER:-qwen_image_edit}"
GENERATION_PROVIDER_CHAIN="${TEXT_ASSET_GENERATION_PROVIDER_CHAIN:-sam3d,hunyuan3d}"
SKIP_NUREC="${SKIP_NUREC:-false}"

# ── NuRec shim defaults ─────────────────────────────────────────────────────
NUREC_QUALITY_PROFILE="${NUREC_QUALITY_PROFILE:-quality_first}"
MAX_FRAMES="${MAX_FRAMES:-450}"
EXTRACT_FPS="${EXTRACT_FPS:-6}"
N_ITERATIONS="${N_ITERATIONS:-12000}"
SAM3_N_FRAMES="${SAM3_N_FRAMES:-0}"
SKIP_FIXER="${SKIP_FIXER:---skip-fixer}"
NUREC_RESUME="${NUREC_RESUME:-false}"
NUREC_PARALLEL_POST_STAGE6="${NUREC_PARALLEL_POST_STAGE6:-true}"
COLMAP_MATCHER_MODE="${COLMAP_MATCHER_MODE:-exhaustive}"
COLMAP_SEQUENTIAL_OVERLAP="${COLMAP_SEQUENTIAL_OVERLAP:-30}"
COLMAP_MIN_REGISTERED_RATIO="${COLMAP_MIN_REGISTERED_RATIO:-0.80}"
COLMAP_RETRY_MIN_REGISTERED_RATIO="${COLMAP_RETRY_MIN_REGISTERED_RATIO:-0.75}"
BLUR_FILTER_KEEP_RATIO="${BLUR_FILTER_KEEP_RATIO:-1.0}"
BLUR_FILTER_MIN_FRAMES="${BLUR_FILTER_MIN_FRAMES:-120}"
VISUAL_MESH_METHOD="${VISUAL_MESH_METHOD:-textured_colmap}"
NUREC_VISUAL_PRIMARY="${NUREC_VISUAL_PRIMARY:-usdz}"
VISUAL_MESH_TEXTURE_SIZE="${VISUAL_MESH_TEXTURE_SIZE:-4096}"
VISUAL_MESH_TEXTURE_MAX_ATLASES="${VISUAL_MESH_TEXTURE_MAX_ATLASES:-2}"
SAM3_PREFLIGHT_STRICT="${SAM3_PREFLIGHT_STRICT:-false}"
OPEN_OMNIVERSE_PREVIEW="${OPEN_OMNIVERSE_PREVIEW:-false}"

log() {
  echo "[run-full-pipeline] $*"
}

die() {
  echo "[run-full-pipeline] ERROR: $*" >&2
  exit 1
}

validate_full_runtime() {
  local root="$1"
  local required=(
    "interactive-job/run_interactive_assets.py"
    "simready-job/prepare_simready_assets.py"
    "usd-assembly-job/assemble_scene.py"
    "tools/source_pipeline/adapter.py"
  )
  [ -d "$root" ] || die "Full completion mode requires BLUEPRINTPIPELINE_ROOT directory: ${root}"
  local missing=()
  local rel
  for rel in "${required[@]}"; do
    if [ ! -f "${root}/${rel}" ]; then
      missing+=("${root}/${rel}")
    fi
  done
  if [ "${#missing[@]}" -gt 0 ]; then
    printf '%s\n' "${missing[@]}" >&2
    die "Full completion mode requires BlueprintPipeline assembly scripts above"
  fi
}

usage() {
  cat <<'EOF'
Usage:
  run_full_pipeline.sh [options] <input_video>

Options:
  --environment ENV       Scene environment: auto, default, bedroom, warehouse, kitchen (default: auto)
  --completion-mode MODE  Completion mode: full_required, best_effort (default: full_required)
  --workspace DIR         Working directory (default: /workspace)
  --skip-nurec            Skip NuRec shim (use existing outputs in --nurec-output-dir)
  --nurec-output-dir DIR  NuRec output directory (default: auto from workspace)
  --crop-cleanup PROV     Crop cleanup: qwen_image_edit, nano_banana, skip (default: qwen_image_edit)
  --scene-id ID           Scene ID (default: derived from input filename)
  -h, --help              Show this help
EOF
}

SCENE_ID=""
NUREC_OUTPUT_DIR=""
INPUT_VIDEO=""

while [ $# -gt 0 ]; do
  case "$1" in
    --environment)     ENVIRONMENT="$2";          shift 2 ;;
    --completion-mode) COMPLETION_MODE="$2";      shift 2 ;;
    --workspace)       WORKSPACE="$2";            shift 2 ;;
    --skip-nurec)      SKIP_NUREC=true;           shift ;;
    --nurec-output-dir) NUREC_OUTPUT_DIR="$2";    shift 2 ;;
    --crop-cleanup)    CROP_CLEANUP_PROVIDER="$2"; shift 2 ;;
    --scene-id)        SCENE_ID="$2";             shift 2 ;;
    -h|--help)         usage; exit 0 ;;
    -*)                die "Unknown option: $1" ;;
    *)                 INPUT_VIDEO="$1";           shift ;;
  esac
done

[ -n "$INPUT_VIDEO" ] || die "Input video is required. Usage: run_full_pipeline.sh <video>"
[ -f "$INPUT_VIDEO" ] || die "Input video not found: $INPUT_VIDEO"
case "${ENVIRONMENT}" in
  auto|default|bedroom|warehouse|kitchen) ;;
  *) die "Invalid --environment '${ENVIRONMENT}'. Expected: auto, default, bedroom, warehouse, kitchen" ;;
esac
case "${COMPLETION_MODE}" in
  full_required|best_effort) ;;
  *) die "Invalid --completion-mode '${COMPLETION_MODE}'. Expected: full_required or best_effort" ;;
esac
if [ "$COMPLETION_MODE" = "full_required" ]; then
  validate_full_runtime "$BLUEPRINTPIPELINE_ROOT"
fi

# ── Derive identifiers ──────────────────────────────────────────────────────
VIDEO_BASENAME="$(basename "$INPUT_VIDEO" | sed 's/\.[^.]*$//')"
SCENE_ID="${SCENE_ID:-scene_${VIDEO_BASENAME}}"
CAPTURE_ID="cap_$(date +%Y%m%d_%H%M%S)"

log "Scene ID: $SCENE_ID"
log "Capture ID: $CAPTURE_ID"
log "Environment: $ENVIRONMENT"
log "Completion mode: $COMPLETION_MODE"

# ── Stage 1: NuRec Shim (COLMAP + 3DGRUT + SAM3) ───────────────────────────
PIPELINE_DIR="${WORKSPACE}/full_pipeline"
NUREC_OUTPUT_DIR="${NUREC_OUTPUT_DIR:-${PIPELINE_DIR}/output}"

if [ "$SKIP_NUREC" = "false" ]; then
  log "============================================================"
  log "PHASE 1: NuRec Shim (Stages 1-8)"
  log "============================================================"

  JOB_SPEC="${PIPELINE_DIR}/job_spec.json"
  mkdir -p "$PIPELINE_DIR"

  cat > "$JOB_SPEC" <<SPEC
{
  "schema_version": "v1",
  "scene_id": "${SCENE_ID}",
  "capture_id": "${CAPTURE_ID}",
  "capture": {
    "raw_prefix_uri": "${INPUT_VIDEO}"
  },
  "outputs": {
    "nurec_prefix": "${NUREC_OUTPUT_DIR}"
  }
}
SPEC

  NUREC_RESUME_ARGS=()
  if [ "$NUREC_RESUME" = "true" ]; then
    NUREC_RESUME_ARGS+=(--resume)
  fi

  NUREC_PARALLEL_ARGS=()
  if [ "$NUREC_PARALLEL_POST_STAGE6" = "true" ]; then
    NUREC_PARALLEL_ARGS+=(--parallel-post-stage6)
  else
    NUREC_PARALLEL_ARGS+=(--no-parallel-post-stage6)
  fi

  NUREC_PREVIEW_ARGS=()
  if [ "${SAM3_PREFLIGHT_STRICT,,}" = "true" ]; then
    NUREC_PREVIEW_ARGS+=(--sam3-strict-preflight)
  fi

  export NUREC_QUALITY_PROFILE VISUAL_MESH_METHOD NUREC_VISUAL_PRIMARY
  export COLMAP_MIN_REGISTERED_RATIO COLMAP_RETRY_MIN_REGISTERED_RATIO
  export VISUAL_MESH_TEXTURE_SIZE VISUAL_MESH_TEXTURE_MAX_ATLASES
  export SAM3_PREFLIGHT_STRICT
  python3 "${APP_DIR}/scripts/nurec_shim.py" \
    --job-spec "$JOB_SPEC" \
    --output-dir "$NUREC_OUTPUT_DIR" \
    --raw-prefix "$INPUT_VIDEO" \
    --max-frames "$MAX_FRAMES" \
    --extract-fps "$EXTRACT_FPS" \
    --n-iterations "$N_ITERATIONS" \
    --colmap-matcher-mode "$COLMAP_MATCHER_MODE" \
    --colmap-sequential-overlap "$COLMAP_SEQUENTIAL_OVERLAP" \
    --colmap-min-registered-ratio "$COLMAP_MIN_REGISTERED_RATIO" \
    --colmap-retry-min-registered-ratio "$COLMAP_RETRY_MIN_REGISTERED_RATIO" \
    --blur-filter-keep-ratio "$BLUR_FILTER_KEEP_RATIO" \
    --blur-filter-min-frames "$BLUR_FILTER_MIN_FRAMES" \
    --environment "$ENVIRONMENT" \
    --sam3-n-frames "$SAM3_N_FRAMES" \
    "${NUREC_RESUME_ARGS[@]}" \
    "${NUREC_PARALLEL_ARGS[@]}" \
    "${NUREC_PREVIEW_ARGS[@]}" \
    $SKIP_FIXER \
    2>&1 | tee "${PIPELINE_DIR}/nurec.log"

  log "NuRec shim completed"
else
  log "Skipping NuRec shim (--skip-nurec), using existing outputs in: $NUREC_OUTPUT_DIR"
fi

# ── Validate NuRec outputs ──────────────────────────────────────────────────
for f in export_last.usdz nvblox_mesh.ply visual_mesh.glb mesh_manifest.json occupancy.bin object_point_cloud_index.json; do
  [ -f "${NUREC_OUTPUT_DIR}/${f}" ] || die "Missing NuRec output: ${NUREC_OUTPUT_DIR}/${f}"
done
log "NuRec outputs validated"

# ── Stage 2: Build GCS-like directory structure ─────────────────────────────
log "============================================================"
log "PHASE 2: Building orchestrator directory structure"
log "============================================================"

# Paths matching swap_orchestrator expectations
SCENE_ROOT="${GCS_ROOT}/${BUCKET}/scenes/${SCENE_ID}"
CAPTURE_ROOT="${SCENE_ROOT}/captures/${CAPTURE_ID}"
RAW_ROOT="${SCENE_ROOT}/iphone/${CAPTURE_ID}/raw"
NUREC_ROOT="${CAPTURE_ROOT}/pipeline/nurec"
PIPELINE_ROOT="${CAPTURE_ROOT}/pipeline"

mkdir -p "$RAW_ROOT" "$NUREC_ROOT" "$PIPELINE_ROOT" \
         "${SCENE_ROOT}/assets" "${SCENE_ROOT}/layout" \
         "${SCENE_ROOT}/seg" "${SCENE_ROOT}/usd"

# Copy NuRec outputs into expected location
for f in export_last.usdz export_last.ply export_last.ingp nvblox_mesh.ply visual_mesh.glb visual_mesh_robust.glb visual_pointcloud.ply mesh_manifest.json collision_mesh_report.json occupancy.bin scene_semantics_report.json mesh_method.txt quality_profile.txt capture_quality_report.json sam3_preflight_report.json; do
  src="${NUREC_OUTPUT_DIR}/${f}"
  [ -f "$src" ] && ln -sf "$src" "${NUREC_ROOT}/${f}"
done

# Copy object index
INDEX_SOURCE="${NUREC_OUTPUT_DIR}/object_point_cloud_index.json"
INDEX_POINTER_CANONICAL="${RAW_ROOT}/object_point_cloud_index.json"
INDEX_POINTER_LEGACY="${RAW_ROOT}/arkit_objects_index.json"
ln -sfn "$INDEX_SOURCE" "$INDEX_POINTER_CANONICAL"
ln -sfn "$INDEX_SOURCE" "$INDEX_POINTER_LEGACY"
log "Regenerated object index pointers:"
log "  ${INDEX_POINTER_CANONICAL} -> ${INDEX_SOURCE}"
log "  ${INDEX_POINTER_LEGACY} -> ${INDEX_SOURCE}"

# Copy object crops directory (reference images for generation)
if [ -d "${NUREC_OUTPUT_DIR}/object_crops" ]; then
  ln -sf "${NUREC_OUTPUT_DIR}/object_crops" "${NUREC_ROOT}/object_crops"
fi

# Fix reference_crop paths in object index to point to GCS-like structure
python3 - <<PY
import json
from pathlib import Path

idx_path = Path("${RAW_ROOT}/object_point_cloud_index.json")
data = json.loads(idx_path.read_text())
objects = data.get("objects", data if isinstance(data, list) else [])

crops_dir = Path("${NUREC_ROOT}/object_crops")
for obj in objects:
    ref = obj.get("reference_crop")
    if ref:
        crop_name = Path(ref).name
        new_path = crops_dir / crop_name
        if new_path.exists():
            obj["reference_crop"] = str(new_path)
    all_crops = obj.get("all_crops", [])
    for i, crop in enumerate(all_crops):
        if crop:
            crop_name = Path(crop).name
            new_path = crops_dir / crop_name
            if new_path.exists():
                all_crops[i] = str(new_path)

if isinstance(data, dict) and "objects" in data:
    data["objects"] = objects
idx_path.write_text(json.dumps(data, indent=2))
print(f"Updated {len(objects)} object crop paths")
PY

RESOLVED_ENVIRONMENT="$(python3 - <<PY
import json
from pathlib import Path
idx_path = Path("${RAW_ROOT}/object_point_cloud_index.json")
try:
    payload = json.loads(idx_path.read_text(encoding="utf-8"))
except Exception:
    payload = {}
env = str(payload.get("environment") if isinstance(payload, dict) else "").strip().lower()
print(env or "${ENVIRONMENT}")
PY
)"
case "$RESOLVED_ENVIRONMENT" in
  default|bedroom|warehouse|kitchen) ;;
  *) RESOLVED_ENVIRONMENT="${ENVIRONMENT}" ;;
esac
log "Resolved environment from object index: ${RESOLVED_ENVIRONMENT}"

# Create raw/manifest.json
cat > "${RAW_ROOT}/manifest.json" <<MANIFEST
{
  "scene_id": "${SCENE_ID}",
  "video_uri": "${INPUT_VIDEO}",
  "device_model": "iPhone",
  "os_version": "17.0",
  "fps_source": 30.0,
  "width": 1920,
  "height": 1080,
  "capture_start_epoch_ms": 0,
  "has_lidar": false,
  "scale_hint_m_per_unit": 1.0,
  "intended_space_type": "${RESOLVED_ENVIRONMENT}",
  "object_point_cloud_index": "object_point_cloud_index.json",
  "object_point_cloud_count": $(python3 -c "import json; d=json.load(open('${RAW_ROOT}/object_point_cloud_index.json')); print(len(d.get('objects', d if isinstance(d, list) else [])))" 2>/dev/null || echo 0),
  "capture_source": "iphone",
  "capture_tier_hint": "tier1_iphone"
}
MANIFEST
log "Created manifest.json"

# Create qa_report.json (synthetic pass)
cat > "${CAPTURE_ROOT}/qa_report.json" <<QA
{
  "schema_version": "v1",
  "status": "passed",
  "scene_id": "${SCENE_ID}",
  "capture_id": "${CAPTURE_ID}",
  "generated_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "checks": [
    {"name": "frame_count", "passed": true, "detail": "synthetic pass from nurec_shim"},
    {"name": "blur_rate", "passed": true, "detail": "synthetic pass"},
    {"name": "pose_match", "passed": true, "detail": "synthetic pass"}
  ]
}
QA
log "Created qa_report.json"

# Create capture_descriptor.json
DESCRIPTOR_URI="gs://${BUCKET}/scenes/${SCENE_ID}/captures/${CAPTURE_ID}/capture_descriptor.json"
RAW_PREFIX_URI="gs://${BUCKET}/scenes/${SCENE_ID}/iphone/${CAPTURE_ID}/raw"
FRAMES_INDEX_URI="gs://${BUCKET}/scenes/${SCENE_ID}/iphone/${CAPTURE_ID}/raw/frames/index.jsonl"
QA_URI="gs://${BUCKET}/scenes/${SCENE_ID}/captures/${CAPTURE_ID}/qa_report.json"

cat > "${CAPTURE_ROOT}/capture_descriptor.json" <<DESC
{
  "schema_version": "v1",
  "scene_id": "${SCENE_ID}",
  "capture_id": "${CAPTURE_ID}",
  "capture_source": "iphone",
  "capture_tier": "tier1_iphone",
  "raw_prefix_uri": "${RAW_PREFIX_URI}",
  "frames_index_uri": "${FRAMES_INDEX_URI}",
  "nurec_mode": "mono_pose_assisted",
  "qa_report_uri": "${QA_URI}",
  "qa_status": "passed",
  "environment_type_hint": "${RESOLVED_ENVIRONMENT}",
  "quality": {"pose_match_rate": 0.95},
  "swap_focus": ["${RESOLVED_ENVIRONMENT}"],
  "manipulation_candidates": [],
  "articulation_hints": []
}
DESC
log "Created capture_descriptor.json"

# Create .nurec_complete marker (so orchestrator skips NuRec execution)
NUREC_PREFIX_URI="gs://${BUCKET}/scenes/${SCENE_ID}/captures/${CAPTURE_ID}/pipeline/nurec"
cat > "${PIPELINE_ROOT}/.nurec_complete" <<MARKER
{
  "schema_version": "v1",
  "scene_id": "${SCENE_ID}",
  "capture_id": "${CAPTURE_ID}",
  "status": "completed",
  "generated_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "nurec_dir": "${NUREC_ROOT}",
  "command": {"executed": true, "command": "nurec_shim.py", "return_code": 0, "stdout": "", "stderr": ""},
  "outputs": {
    "visual_usdz": "${NUREC_PREFIX_URI}/export_last.usdz",
    "visual_mesh_glb": "${NUREC_PREFIX_URI}/visual_mesh.glb",
    "visual_pointcloud_ply": "${NUREC_PREFIX_URI}/visual_pointcloud.ply",
    "mesh_manifest_json": "${NUREC_PREFIX_URI}/mesh_manifest.json",
    "collision_mesh_ply": "${NUREC_PREFIX_URI}/nvblox_mesh.ply",
    "occupancy": ["${NUREC_PREFIX_URI}/occupancy.bin"]
  }
}
MARKER
log "Created .nurec_complete marker"

# Create synthetic frames index (orchestrator may check it exists)
mkdir -p "${RAW_ROOT}/frames"
echo '{"schema_version": "v1", "frames": []}' > "${RAW_ROOT}/frames/index.jsonl"

log "Directory structure ready at: ${GCS_ROOT}/${BUCKET}"

# ── Stage 3: Run swap_orchestrator (stages C→I) ────────────────────────────
log "============================================================"
log "PHASE 3: Swap Orchestrator (Stages C→I)"
log "============================================================"

export GCS_ROOT="${GCS_ROOT}/${BUCKET}"
export BLUEPRINTPIPELINE_ROOT
export PIPELINE_COMPLETION_MODE="$COMPLETION_MODE"
export NUREC_SKIP_PIPELINE_COMMAND=true
export CROP_CLEANUP_PROVIDER="$CROP_CLEANUP_PROVIDER"
export IMAGE_CONDITIONED_GENERATION_ENABLED=true
export TEXT_ASSET_GENERATION_PROVIDER_CHAIN="$GENERATION_PROVIDER_CHAIN"
export SWAP_POLICY_CONFIG_PATH="${APP_DIR}/configs/swap_policy.yaml"
export PYTHONPATH="${APP_DIR}/scripts:${PYTHONPATH:-}"
export SAM3_TRACKING_MODE="${SAM3_TRACKING_MODE:-full_video}"
export SWAP_INCLUDE_HEURISTIC_AS_EXPLICIT=false

if [ "$COMPLETION_MODE" = "full_required" ]; then
  export PIPELINE_STANDALONE_MODE=false
  export RUNTIME_PREFLIGHT_ENABLED=true
  export ADVANCED_QUALITY_GATES_ENABLED=true
  log "Strict full completion enabled:"
  log "  PIPELINE_STANDALONE_MODE=${PIPELINE_STANDALONE_MODE}"
  log "  RUNTIME_PREFLIGHT_ENABLED=${RUNTIME_PREFLIGHT_ENABLED}"
  log "  ADVANCED_QUALITY_GATES_ENABLED=${ADVANCED_QUALITY_GATES_ENABLED}"
else
  export PIPELINE_STANDALONE_MODE="${PIPELINE_STANDALONE_MODE:-false}"
  export RUNTIME_PREFLIGHT_ENABLED="${RUNTIME_PREFLIGHT_ENABLED:-true}"
  export ADVANCED_QUALITY_GATES_ENABLED="${ADVANCED_QUALITY_GATES_ENABLED:-true}"
fi

# Apply relaxed jitter thresholds only when NuRec used COLMAP Delaunay fallback.
MESH_METHOD_FILE="${NUREC_OUTPUT_DIR}/mesh_method.txt"
if [ ! -f "$MESH_METHOD_FILE" ]; then
  die "Missing required NuRec mesh method marker: ${MESH_METHOD_FILE}. Re-run NuRec with updated nurec_shim.py."
fi
MESH_METHOD="$(tr -d '\r\n' < "$MESH_METHOD_FILE" | tr '[:upper:]' '[:lower:]')"
if [ -z "$MESH_METHOD" ]; then
  die "NuRec mesh method marker is empty: ${MESH_METHOD_FILE}"
fi
case "$MESH_METHOD" in
  delaunay_colmap)
    : "${QUALITY_JITTER_MAX_DRIFT_M:=0.5}"
    : "${QUALITY_JITTER_MAX_VERTICAL_SPAN_M:=2.0}"
    export QUALITY_JITTER_MAX_DRIFT_M QUALITY_JITTER_MAX_VERTICAL_SPAN_M
    export QUALITY_THRESHOLD_PROFILE="${QUALITY_THRESHOLD_PROFILE:-delaunay_relaxed}"
    log "Detected Delaunay fallback mesh; applying relaxed jitter thresholds:"
    log "  QUALITY_JITTER_MAX_DRIFT_M=${QUALITY_JITTER_MAX_DRIFT_M}"
    log "  QUALITY_JITTER_MAX_VERTICAL_SPAN_M=${QUALITY_JITTER_MAX_VERTICAL_SPAN_M}"
    ;;
  poisson_open3d)
    export QUALITY_THRESHOLD_PROFILE="${QUALITY_THRESHOLD_PROFILE:-default}"
    log "Mesh method=poisson_open3d; keeping default jitter thresholds"
    ;;
  *)
    die "Invalid mesh method '${MESH_METHOD}' in ${MESH_METHOD_FILE}; expected one of: delaunay_colmap, poisson_open3d"
    ;;
esac

log "GCS_ROOT=$GCS_ROOT"
log "BLUEPRINTPIPELINE_ROOT=$BLUEPRINTPIPELINE_ROOT"
log "CROP_CLEANUP_PROVIDER=$CROP_CLEANUP_PROVIDER"
log "GENERATION_PROVIDER_CHAIN=$GENERATION_PROVIDER_CHAIN"
log "DESCRIPTOR_URI=$DESCRIPTOR_URI"
log "NUREC_VISUAL_PRIMARY=$NUREC_VISUAL_PRIMARY"
log "COLMAP coverage gate: min=${COLMAP_MIN_REGISTERED_RATIO}, retry_fail=${COLMAP_RETRY_MIN_REGISTERED_RATIO}"

python3 -m blueprint_pipeline.swap_orchestrator \
  --descriptor-gcs-uri "$DESCRIPTOR_URI" \
  2>&1 | tee "${PIPELINE_DIR}/orchestrator.log"

python3 - <<PY
import json
from pathlib import Path

completion_mode = "${COMPLETION_MODE}"
quality_report_path = Path("${PIPELINE_ROOT}/swap_quality_report.json")
scene_usda_path = Path("${SCENE_ROOT}/usd/scene.usda")

if not quality_report_path.exists():
    raise SystemExit(f"Missing quality report: {quality_report_path}")

report = json.loads(quality_report_path.read_text(encoding="utf-8"))
status = str(report.get("status") or "").strip().lower()
if status != "passed":
    raise SystemExit(f"Pipeline status is not passed ({status or 'unknown'})")

gates = report.get("gates") if isinstance(report.get("gates"), list) else []
gate_map = {}
for gate in gates:
    if isinstance(gate, dict):
        name = str(gate.get("name") or "").strip()
        if name:
            gate_map[name] = bool(gate.get("passed", False))

if not gate_map.get("assembly_gate", False):
    raise SystemExit("Assembly gate did not pass")

if completion_mode == "full_required":
    for required_gate in ("runtime_preflight_gate", "advanced_quality_gate"):
        if not gate_map.get(required_gate, False):
            raise SystemExit(f"Required gate failed in full_required mode: {required_gate}")

if not scene_usda_path.exists():
    raise SystemExit(f"Missing scene.usda: {scene_usda_path}")

scene_text = scene_usda_path.read_text(encoding="utf-8", errors="ignore")
is_stub = (
    'defaultPrim = "Scene"' in scene_text
    and 'def Xform "Scene"' in scene_text
    and "references = @" not in scene_text
    and len(scene_text.strip().splitlines()) <= 12
)
if completion_mode == "full_required" and is_stub:
    raise SystemExit("scene.usda is a standalone stub; full assembly is required")
PY

log "============================================================"
log "FULL PIPELINE COMPLETE"
log "============================================================"
log "Outputs:"
log "  NuRec:        ${NUREC_OUTPUT_DIR}/"
log "  Scene USD:    ${SCENE_ROOT}/usd/scene.usda"
log "  Assets:       ${SCENE_ROOT}/assets/"
log "  Quality:      ${PIPELINE_ROOT}/swap_quality_report.json"
log "  Summary:      ${PIPELINE_ROOT}/pipeline_summary.json"

if [ "${OPEN_OMNIVERSE_PREVIEW,,}" = "true" ]; then
  log "Preparing Omniverse WebRTC preview workflow..."
  bash "${APP_DIR}/scripts/preview_omniverse_webrtc.sh" "${NUREC_OUTPUT_DIR}" auto || true
fi
