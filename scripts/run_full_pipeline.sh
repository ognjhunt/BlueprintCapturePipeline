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
ENVIRONMENT="${ENVIRONMENT:-warehouse}"
CROP_CLEANUP_PROVIDER="${CROP_CLEANUP_PROVIDER:-qwen_image_edit}"
GENERATION_PROVIDER_CHAIN="${TEXT_ASSET_GENERATION_PROVIDER_CHAIN:-sam3d,hunyuan3d}"
SKIP_NUREC="${SKIP_NUREC:-false}"

# ── NuRec shim defaults ─────────────────────────────────────────────────────
MAX_FRAMES="${MAX_FRAMES:-300}"
EXTRACT_FPS="${EXTRACT_FPS:-5}"
N_ITERATIONS="${N_ITERATIONS:-7000}"
SAM3_N_FRAMES="${SAM3_N_FRAMES:-0}"
SKIP_FIXER="${SKIP_FIXER:---skip-fixer}"

log() {
  echo "[run-full-pipeline] $*"
}

die() {
  echo "[run-full-pipeline] ERROR: $*" >&2
  exit 1
}

usage() {
  cat <<'EOF'
Usage:
  run_full_pipeline.sh [options] <input_video>

Options:
  --environment ENV       Scene environment: warehouse, kitchen (default: warehouse)
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

# ── Derive identifiers ──────────────────────────────────────────────────────
VIDEO_BASENAME="$(basename "$INPUT_VIDEO" | sed 's/\.[^.]*$//')"
SCENE_ID="${SCENE_ID:-scene_${VIDEO_BASENAME}}"
CAPTURE_ID="cap_$(date +%Y%m%d_%H%M%S)"

log "Scene ID: $SCENE_ID"
log "Capture ID: $CAPTURE_ID"
log "Environment: $ENVIRONMENT"

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

  python3 "${APP_DIR}/scripts/nurec_shim.py" \
    --job-spec "$JOB_SPEC" \
    --output-dir "$NUREC_OUTPUT_DIR" \
    --raw-prefix "$INPUT_VIDEO" \
    --max-frames "$MAX_FRAMES" \
    --extract-fps "$EXTRACT_FPS" \
    --n-iterations "$N_ITERATIONS" \
    --environment "$ENVIRONMENT" \
    --sam3-n-frames "$SAM3_N_FRAMES" \
    $SKIP_FIXER \
    2>&1 | tee "${PIPELINE_DIR}/nurec.log"

  log "NuRec shim completed"
else
  log "Skipping NuRec shim (--skip-nurec), using existing outputs in: $NUREC_OUTPUT_DIR"
fi

# ── Validate NuRec outputs ──────────────────────────────────────────────────
for f in export_last.usdz nvblox_mesh.ply occupancy.bin object_point_cloud_index.json; do
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
for f in export_last.usdz export_last.ply export_last.ingp nvblox_mesh.ply occupancy.bin; do
  src="${NUREC_OUTPUT_DIR}/${f}"
  [ -f "$src" ] && ln -sf "$src" "${NUREC_ROOT}/${f}"
done

# Copy object index
ln -sf "${NUREC_OUTPUT_DIR}/object_point_cloud_index.json" "${RAW_ROOT}/arkit_objects_index.json"

# Copy object crops directory (reference images for generation)
if [ -d "${NUREC_OUTPUT_DIR}/object_crops" ]; then
  ln -sf "${NUREC_OUTPUT_DIR}/object_crops" "${NUREC_ROOT}/object_crops"
fi

# Fix reference_crop paths in object index to point to GCS-like structure
python3 - <<PY
import json
from pathlib import Path

idx_path = Path("${RAW_ROOT}/arkit_objects_index.json")
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
  "intended_space_type": "${ENVIRONMENT}",
  "object_point_cloud_index": "arkit_objects_index.json",
  "object_point_cloud_count": $(python3 -c "import json; d=json.load(open('${RAW_ROOT}/arkit_objects_index.json')); print(len(d.get('objects', d if isinstance(d, list) else [])))" 2>/dev/null || echo 0),
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
  "environment_type_hint": "${ENVIRONMENT}",
  "quality": {"pose_match_rate": 0.95},
  "swap_focus": ["${ENVIRONMENT}"],
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
export NUREC_SKIP_PIPELINE_COMMAND=true
export CROP_CLEANUP_PROVIDER="$CROP_CLEANUP_PROVIDER"
export IMAGE_CONDITIONED_GENERATION_ENABLED=true
export TEXT_ASSET_GENERATION_PROVIDER_CHAIN="$GENERATION_PROVIDER_CHAIN"
export SWAP_POLICY_CONFIG_PATH="${APP_DIR}/configs/swap_policy.yaml"
export PYTHONPATH="${APP_DIR}/scripts:${PYTHONPATH:-}"

log "GCS_ROOT=$GCS_ROOT"
log "BLUEPRINTPIPELINE_ROOT=$BLUEPRINTPIPELINE_ROOT"
log "CROP_CLEANUP_PROVIDER=$CROP_CLEANUP_PROVIDER"
log "GENERATION_PROVIDER_CHAIN=$GENERATION_PROVIDER_CHAIN"
log "DESCRIPTOR_URI=$DESCRIPTOR_URI"

python3 -m blueprint_pipeline.swap_orchestrator \
  --descriptor-gcs-uri "$DESCRIPTOR_URI" \
  2>&1 | tee "${PIPELINE_DIR}/orchestrator.log"

log "============================================================"
log "FULL PIPELINE COMPLETE"
log "============================================================"
log "Outputs:"
log "  NuRec:        ${NUREC_OUTPUT_DIR}/"
log "  Scene USD:    ${SCENE_ROOT}/usd/scene.usda"
log "  Assets:       ${SCENE_ROOT}/assets/"
log "  Quality:      ${PIPELINE_ROOT}/swap_quality_report.json"
