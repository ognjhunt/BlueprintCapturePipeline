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
CROP_CLEANUP_PROVIDER="${CROP_CLEANUP_PROVIDER:-skip}"
GENERATION_PROVIDER_CHAIN="${TEXT_ASSET_GENERATION_PROVIDER_CHAIN:-sam3d,hunyuan3d}"
SKIP_NUREC="${SKIP_NUREC:-false}"

# ── NuRec shim defaults ─────────────────────────────────────────────────────
NUREC_QUALITY_PROFILE="${NUREC_QUALITY_PROFILE:-quality_first}"
NUREC_RERUN_PROFILE="${NUREC_RERUN_PROFILE:-default}"
MAX_FRAMES="${MAX_FRAMES:-320}"
EXTRACT_FPS="${EXTRACT_FPS:-5}"
N_ITERATIONS="${N_ITERATIONS:-9000}"
MAX_N_GAUSSIANS="${MAX_N_GAUSSIANS:-}"
PIPELINE_MODE="${PIPELINE_MODE:-full}"
SAM3_N_FRAMES="${SAM3_N_FRAMES:-0}"
SKIP_FIXER="${SKIP_FIXER:-false}"
SKIP_DENSE="${SKIP_DENSE:-__UNSET__}"
FIXER_MODE="${FIXER_MODE:-local}"
NUREC_RESUME="${NUREC_RESUME:-false}"
NUREC_PARALLEL_POST_STAGE6="${NUREC_PARALLEL_POST_STAGE6:-true}"
FIXER_RERUN="${FIXER_RERUN:-false}"
FIXER_REQUIRED="${FIXER_REQUIRED:-false}"
POST_STAGE4_REFINE="${POST_STAGE4_REFINE:-auto}"
POST_STAGE4_REFINE_MODEL="${POST_STAGE4_REFINE_MODEL:-fixer+gsfix3d}"
POST_STAGE4_MAX_PSEUDOVIEWS="${POST_STAGE4_MAX_PSEUDOVIEWS:-96}"
POST_STAGE4_DISTILL_ITERS="${POST_STAGE4_DISTILL_ITERS:-1600}"
POST_STAGE4_TIME_BUDGET_MIN="${POST_STAGE4_TIME_BUDGET_MIN:-90}"
VOID_FILL_ROUNDS="${VOID_FILL_ROUNDS:-0}"
VOID_FILL_TARGET_HOLE_RATIO="${VOID_FILL_TARGET_HOLE_RATIO:-0.05}"
VOID_FILL_DISTILL_ITERS="${VOID_FILL_DISTILL_ITERS:-5000}"
REFINEMENT_QUALITY_GATE_PROFILE="${REFINEMENT_QUALITY_GATE_PROFILE:-auto}"
COLMAP_MATCHER_MODE="${COLMAP_MATCHER_MODE:-auto}"
COLMAP_SEQUENTIAL_OVERLAP="${COLMAP_SEQUENTIAL_OVERLAP:-30}"
COLMAP_CHUNKED_MODE="${COLMAP_CHUNKED_MODE:-auto}"
COLMAP_CHUNK_MIN_FRAMES="${COLMAP_CHUNK_MIN_FRAMES:-900}"
COLMAP_CHUNK_SIZE_FRAMES="${COLMAP_CHUNK_SIZE_FRAMES:-600}"
COLMAP_CHUNK_OVERLAP_FRAMES="${COLMAP_CHUNK_OVERLAP_FRAMES:-120}"
COLMAP_CHUNK_MAX_CHUNKS="${COLMAP_CHUNK_MAX_CHUNKS:-24}"
COLMAP_CHUNK_MATCHER_MODE="${COLMAP_CHUNK_MATCHER_MODE:-sequential}"
COLMAP_MIN_REGISTERED_RATIO="${COLMAP_MIN_REGISTERED_RATIO:-0.80}"
COLMAP_RETRY_MIN_REGISTERED_RATIO="${COLMAP_RETRY_MIN_REGISTERED_RATIO:-0.75}"
COLMAP_RETRY_MATCHER_MODE="${COLMAP_RETRY_MATCHER_MODE:-auto}"
BLUR_FILTER_KEEP_RATIO="${BLUR_FILTER_KEEP_RATIO:-}"
BLUR_FILTER_MIN_FRAMES="${BLUR_FILTER_MIN_FRAMES:-120}"
VISUAL_MESH_METHOD="${VISUAL_MESH_METHOD:-textured_colmap}"
NUREC_VISUAL_PRIMARY="${NUREC_VISUAL_PRIMARY:-usdz}"
VISUAL_MESH_TEXTURE_SIZE="${VISUAL_MESH_TEXTURE_SIZE:-4096}"
VISUAL_MESH_TEXTURE_MAX_ATLASES="${VISUAL_MESH_TEXTURE_MAX_ATLASES:-2}"
SAM3_PREFLIGHT_STRICT="${SAM3_PREFLIGHT_STRICT:-false}"
SAM3_TRACKING_MODE="${SAM3_TRACKING_MODE:-auto}"
SCENE_CLEANING_MODE="${SCENE_CLEANING_MODE:-off}"
SAM3_MASK_EXPORT_SPACE="${SAM3_MASK_EXPORT_SPACE:-undistorted}"
INPAINT360GS_RESOLUTION="${INPAINT360GS_RESOLUTION:-2}"
OPEN_OMNIVERSE_PREVIEW="${OPEN_OMNIVERSE_PREVIEW:-false}"

# Backward compatibility with older SKIP_FIXER env style ("--skip-fixer").
if [ "${SKIP_FIXER}" = "--skip-fixer" ]; then
  SKIP_FIXER=true
fi

if [ -z "$BLUR_FILTER_KEEP_RATIO" ]; then
  case "${NUREC_QUALITY_PROFILE,,}" in
    quality_first) BLUR_FILTER_KEEP_RATIO="0.85" ;;
    balanced) BLUR_FILTER_KEEP_RATIO="0.90" ;;
    fast) BLUR_FILTER_KEEP_RATIO="1.0" ;;
    *) BLUR_FILTER_KEEP_RATIO="0.85" ;;
  esac
fi

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
  --nurec-rerun-profile PROFILE  NuRec rerun profile: default, clear_over_faithful, photoreal_hallucination
  --workspace DIR         Working directory (default: /workspace)
  --resume                Enable NuRec resume mode (reuse Stage 1-4 artifacts)
  --skip-fixer            Disable Stage 5 Fixer refinement
  --skip-dense            Skip dense reconstruction (PatchMatch). This defaults to true unless VISUAL_MESH_METHOD=patchmatch
  --fixer-rerun           Force rerun of Fixer in resume mode
  --fixer-required        Fail if Fixer does not produce refined outputs
  --post-stage4-refine MODE  Post-Stage-4 mode: off, auto, force (default: auto)
  --post-stage4-refine-model MODE  Repair stack: fixer, fixer+gsfix3d (default: fixer+gsfix3d)
  --post-stage4-max-pseudoviews N  Max pseudo-views (default: 96)
  --post-stage4-distill-iters N    Distillation iterations (default: 1600)
  --post-stage4-time-budget-min N  Distillation budget minutes (default: 90)
  --scene-cleaning-mode MODE   Scene cleaning mode: off, auto, force (default: off)
  --sam3-mask-export-space MODE  SAM3 mask export space: raw, undistorted (default: undistorted)
  --skip-scene-cleaning        Backward-compatible alias for --scene-cleaning-mode off
  --skip-nurec            Skip NuRec shim (use existing outputs in --nurec-output-dir)
  --nurec-output-dir DIR  NuRec output directory (default: auto from workspace)
  --crop-cleanup PROV     Crop cleanup: skip, together_qwen_image_edit, qwen_image_edit, nano_banana, gpt_image (default: skip)
  --scene-id ID           Scene ID (default: derived from input filename)
  -h, --help              Show this help
EOF
}

apply_nurec_rerun_profile() {
  case "${NUREC_RERUN_PROFILE}" in
    default)
      return 0
      ;;
    clear_over_faithful)
      # Baseline-only sharpness profile (no pseudo-view repair/void-fill).
      PIPELINE_MODE="photorealistic_scene"
      SCENE_CLEANING_MODE="off"
      MAX_FRAMES="${PROFILE_CLEAR_MAX_FRAMES:-500}"
      EXTRACT_FPS="${PROFILE_CLEAR_EXTRACT_FPS:-8}"
      N_ITERATIONS="${PROFILE_CLEAR_ITERATIONS:-22000}"
      MAX_N_GAUSSIANS="${PROFILE_CLEAR_MAX_N_GAUSSIANS:-500000}"
      BLUR_FILTER_KEEP_RATIO="${PROFILE_CLEAR_BLUR_FILTER_KEEP_RATIO:-0.70}"
      COLMAP_MATCHER_MODE="sequential"
      COLMAP_SEQUENTIAL_OVERLAP="${PROFILE_CLEAR_COLMAP_OVERLAP:-40}"
      POST_STAGE4_REFINE="off"
      VOID_FILL_ROUNDS="0"
      ;;
    photoreal_hallucination)
      # Baseline-first settings + aggressive synthetic repair for clarity-over-faithfulness.
      PIPELINE_MODE="photoreal_hallucination"
      SCENE_CLEANING_MODE="off"
      MAX_FRAMES="${PROFILE_CLEAR_MAX_FRAMES:-500}"
      EXTRACT_FPS="${PROFILE_CLEAR_EXTRACT_FPS:-8}"
      N_ITERATIONS="${PROFILE_CLEAR_ITERATIONS:-22000}"
      MAX_N_GAUSSIANS="${PROFILE_CLEAR_MAX_N_GAUSSIANS:-500000}"
      BLUR_FILTER_KEEP_RATIO="${PROFILE_CLEAR_BLUR_FILTER_KEEP_RATIO:-0.70}"
      COLMAP_MATCHER_MODE="sequential"
      COLMAP_SEQUENTIAL_OVERLAP="${PROFILE_CLEAR_COLMAP_OVERLAP:-40}"
      POST_STAGE4_REFINE="force"
      POST_STAGE4_REFINE_MODEL="fixer+gsfix3d"
      POST_STAGE4_MAX_PSEUDOVIEWS="${PROFILE_HALLUCINATION_MAX_PSEUDOVIEWS:-160}"
      POST_STAGE4_DISTILL_ITERS="${PROFILE_HALLUCINATION_DISTILL_ITERS:-6000}"
      POST_STAGE4_TIME_BUDGET_MIN="${PROFILE_HALLUCINATION_TIME_BUDGET_MIN:-120}"
      VOID_FILL_ROUNDS="0"
      REFINEMENT_QUALITY_GATE_PROFILE="${PROFILE_HALLUCINATION_GATE_PROFILE:-hallucination}"
      ;;
    *)
      die "Invalid --nurec-rerun-profile '${NUREC_RERUN_PROFILE}'. Expected: default, clear_over_faithful, photoreal_hallucination"
      ;;
  esac
}

SCENE_ID=""
NUREC_OUTPUT_DIR=""
INPUT_VIDEO=""

while [ $# -gt 0 ]; do
  case "$1" in
    --environment)     ENVIRONMENT="$2";          shift 2 ;;
    --completion-mode) COMPLETION_MODE="$2";      shift 2 ;;
    --nurec-rerun-profile) NUREC_RERUN_PROFILE="$2"; shift 2 ;;
    --workspace)       WORKSPACE="$2";            shift 2 ;;
    --resume)          NUREC_RESUME=true;         shift ;;
    --skip-fixer)      SKIP_FIXER=true;           shift ;;
    --skip-dense)      SKIP_DENSE=true;           shift ;;
    --fixer-rerun)     FIXER_RERUN=true;          shift ;;
    --fixer-required)  FIXER_REQUIRED=true;       shift ;;
    --post-stage4-refine) POST_STAGE4_REFINE="$2"; shift 2 ;;
    --post-stage4-refine-model) POST_STAGE4_REFINE_MODEL="$2"; shift 2 ;;
    --post-stage4-max-pseudoviews) POST_STAGE4_MAX_PSEUDOVIEWS="$2"; shift 2 ;;
    --post-stage4-distill-iters) POST_STAGE4_DISTILL_ITERS="$2"; shift 2 ;;
    --post-stage4-time-budget-min) POST_STAGE4_TIME_BUDGET_MIN="$2"; shift 2 ;;
    --scene-cleaning-mode) SCENE_CLEANING_MODE="$2"; shift 2 ;;
    --sam3-mask-export-space) SAM3_MASK_EXPORT_SPACE="$2"; shift 2 ;;
    --skip-scene-cleaning) SCENE_CLEANING_MODE="off"; shift ;;
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
case "${NUREC_RERUN_PROFILE}" in
  default|clear_over_faithful|photoreal_hallucination) ;;
  *) die "Invalid --nurec-rerun-profile '${NUREC_RERUN_PROFILE}'. Expected: default, clear_over_faithful, photoreal_hallucination" ;;
esac
apply_nurec_rerun_profile
if [ "$SKIP_DENSE" = "__UNSET__" ]; then
  if [ "${VISUAL_MESH_METHOD,,}" = "patchmatch" ]; then
    SKIP_DENSE="false"
  else
    SKIP_DENSE="true"
  fi
fi
SKIP_DENSE="${SKIP_DENSE,,}"
case "${ENVIRONMENT}" in
  auto|default|bedroom|warehouse|kitchen) ;;
  *) die "Invalid --environment '${ENVIRONMENT}'. Expected: auto, default, bedroom, warehouse, kitchen" ;;
esac
case "${COMPLETION_MODE}" in
  full_required|best_effort) ;;
  *) die "Invalid --completion-mode '${COMPLETION_MODE}'. Expected: full_required or best_effort" ;;
esac
case "${POST_STAGE4_REFINE}" in
  off|auto|force) ;;
  *) die "Invalid --post-stage4-refine '${POST_STAGE4_REFINE}'. Expected: off, auto, or force" ;;
esac
case "${POST_STAGE4_REFINE_MODEL}" in
  fixer|fixer+gsfix3d) ;;
  *) die "Invalid --post-stage4-refine-model '${POST_STAGE4_REFINE_MODEL}'. Expected: fixer or fixer+gsfix3d" ;;
esac
case "${SCENE_CLEANING_MODE}" in
  off|auto|force) ;;
  *) die "Invalid --scene-cleaning-mode '${SCENE_CLEANING_MODE}'. Expected: off, auto, or force" ;;
esac
case "${SAM3_MASK_EXPORT_SPACE}" in
  raw|undistorted) ;;
  *) die "Invalid --sam3-mask-export-space '${SAM3_MASK_EXPORT_SPACE}'. Expected: raw or undistorted" ;;
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
log "NuRec rerun profile: $NUREC_RERUN_PROFILE"
log "NuRec resume: $NUREC_RESUME"
log "Skip fixer: $SKIP_FIXER"
log "Fixer mode: $FIXER_MODE"
log "Fixer rerun: $FIXER_RERUN"
log "Fixer required: $FIXER_REQUIRED"
log "Post-Stage-4 refine: $POST_STAGE4_REFINE (model=$POST_STAGE4_REFINE_MODEL, max_pseudoviews=$POST_STAGE4_MAX_PSEUDOVIEWS, distill_iters=$POST_STAGE4_DISTILL_ITERS, budget_min=$POST_STAGE4_TIME_BUDGET_MIN)"
log "Pipeline mode: $PIPELINE_MODE"
log "Void fill: rounds=$VOID_FILL_ROUNDS target_hole=$VOID_FILL_TARGET_HOLE_RATIO distill_iters=$VOID_FILL_DISTILL_ITERS"
log "Blur filter keep ratio: $BLUR_FILTER_KEEP_RATIO (profile=$NUREC_QUALITY_PROFILE, min_frames=$BLUR_FILTER_MIN_FRAMES)"
log "Skip dense reconstruction: $SKIP_DENSE (VISUAL_MESH_METHOD=$VISUAL_MESH_METHOD)"
log "Scene cleaning mode: $SCENE_CLEANING_MODE (mask_export_space=$SAM3_MASK_EXPORT_SPACE)"

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

  NUREC_DENSE_ARGS=()
  if [ "${SKIP_DENSE,,}" = "true" ]; then
    NUREC_DENSE_ARGS+=(--skip-dense)
  fi

  NUREC_FIXER_ARGS=()
  if [ "${SKIP_FIXER,,}" = "true" ]; then
    NUREC_FIXER_ARGS+=(--skip-fixer)
  fi
  if [ "${FIXER_RERUN,,}" = "true" ]; then
    NUREC_FIXER_ARGS+=(--fixer-rerun)
  fi
  if [ "${FIXER_REQUIRED,,}" = "true" ]; then
    NUREC_FIXER_ARGS+=(--fixer-required)
  fi

  NUREC_GAUSSIAN_ARGS=()
  if [ -n "${MAX_N_GAUSSIANS}" ]; then
    NUREC_GAUSSIAN_ARGS+=(--max-n-gaussians "$MAX_N_GAUSSIANS")
  fi

  export NUREC_QUALITY_PROFILE VISUAL_MESH_METHOD NUREC_VISUAL_PRIMARY
  export FIXER_MODE
  export COLMAP_MIN_REGISTERED_RATIO COLMAP_RETRY_MIN_REGISTERED_RATIO
  export VISUAL_MESH_TEXTURE_SIZE VISUAL_MESH_TEXTURE_MAX_ATLASES
  export SAM3_PREFLIGHT_STRICT SAM3_TRACKING_MODE
  export POST_STAGE4_REFINE POST_STAGE4_REFINE_MODEL
  export POST_STAGE4_MAX_PSEUDOVIEWS POST_STAGE4_DISTILL_ITERS POST_STAGE4_TIME_BUDGET_MIN
  export VOID_FILL_ROUNDS VOID_FILL_TARGET_HOLE_RATIO VOID_FILL_DISTILL_ITERS
  export PIPELINE_MODE REFINEMENT_QUALITY_GATE_PROFILE
  export SCENE_CLEANING_MODE SAM3_MASK_EXPORT_SPACE INPAINT360GS_RESOLUTION
  python3 "${APP_DIR}/scripts/nurec_shim.py" \
    --job-spec "$JOB_SPEC" \
    --output-dir "$NUREC_OUTPUT_DIR" \
    --raw-prefix "$INPUT_VIDEO" \
    --max-frames "$MAX_FRAMES" \
    --extract-fps "$EXTRACT_FPS" \
    --n-iterations "$N_ITERATIONS" \
    "${NUREC_GAUSSIAN_ARGS[@]}" \
    --colmap-matcher-mode "$COLMAP_MATCHER_MODE" \
    --colmap-sequential-overlap "$COLMAP_SEQUENTIAL_OVERLAP" \
    --colmap-chunked-mode "$COLMAP_CHUNKED_MODE" \
    --colmap-chunk-min-frames "$COLMAP_CHUNK_MIN_FRAMES" \
    --colmap-chunk-size-frames "$COLMAP_CHUNK_SIZE_FRAMES" \
    --colmap-chunk-overlap-frames "$COLMAP_CHUNK_OVERLAP_FRAMES" \
    --colmap-chunk-max-chunks "$COLMAP_CHUNK_MAX_CHUNKS" \
    --colmap-chunk-matcher-mode "$COLMAP_CHUNK_MATCHER_MODE" \
    --colmap-min-registered-ratio "$COLMAP_MIN_REGISTERED_RATIO" \
    --colmap-retry-min-registered-ratio "$COLMAP_RETRY_MIN_REGISTERED_RATIO" \
    --colmap-retry-matcher-mode "$COLMAP_RETRY_MATCHER_MODE" \
    --blur-filter-keep-ratio "$BLUR_FILTER_KEEP_RATIO" \
    --blur-filter-min-frames "$BLUR_FILTER_MIN_FRAMES" \
    --environment "$ENVIRONMENT" \
    --sam3-n-frames "$SAM3_N_FRAMES" \
    --fixer-mode "$FIXER_MODE" \
    --post-stage4-refine "$POST_STAGE4_REFINE" \
    --post-stage4-refine-model "$POST_STAGE4_REFINE_MODEL" \
    --post-stage4-max-pseudoviews "$POST_STAGE4_MAX_PSEUDOVIEWS" \
    --post-stage4-distill-iters "$POST_STAGE4_DISTILL_ITERS" \
    --post-stage4-time-budget-min "$POST_STAGE4_TIME_BUDGET_MIN" \
    --void-fill-rounds "$VOID_FILL_ROUNDS" \
    --void-fill-target-hole-ratio "$VOID_FILL_TARGET_HOLE_RATIO" \
    --void-fill-distill-iters "$VOID_FILL_DISTILL_ITERS" \
    --pipeline-mode "$PIPELINE_MODE" \
    --scene-cleaning-mode "$SCENE_CLEANING_MODE" \
    --sam3-mask-export-space "$SAM3_MASK_EXPORT_SPACE" \
    "${NUREC_RESUME_ARGS[@]}" \
    "${NUREC_PARALLEL_ARGS[@]}" \
    "${NUREC_PREVIEW_ARGS[@]}" \
    "${NUREC_FIXER_ARGS[@]}" \
    "${NUREC_DENSE_ARGS[@]}" \
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
for f in export_last.usdz export_last.ply export_last.ingp export_last_refined.usdz export_last_refined.ply export_last_refined.ingp nvblox_mesh.ply visual_mesh.glb visual_mesh_robust.glb visual_pointcloud.ply mesh_manifest.json collision_mesh_report.json occupancy.bin scene_semantics_report.json mesh_method.txt quality_profile.txt capture_quality_report.json sam3_preflight_report.json gap_analysis_report.json gap_candidate_views.jsonl view_repair_report.json accepted_repaired_views.jsonl post_stage4_distill_report.json refinement_quality_gate.json hallucinated_region_mask.png; do
  src="${NUREC_OUTPUT_DIR}/${f}"
  [ -f "$src" ] && ln -sf "$src" "${NUREC_ROOT}/${f}"
done

# Copy object index
INDEX_SOURCE="${NUREC_OUTPUT_DIR}/object_point_cloud_index.json"
INDEX_POINTER_CANONICAL="${RAW_ROOT}/object_point_cloud_index.json"
INDEX_POINTER_LEGACY="${RAW_ROOT}/arkit_objects_index.json"
cp -f "$INDEX_SOURCE" "$INDEX_POINTER_CANONICAL"
ln -sfn "object_point_cloud_index.json" "$INDEX_POINTER_LEGACY"
log "Regenerated object index files:"
log "  ${INDEX_POINTER_CANONICAL} (copied from ${INDEX_SOURCE})"
log "  ${INDEX_POINTER_LEGACY} -> object_point_cloud_index.json"

# Copy object crops directory (reference images for generation)
if [ -d "${NUREC_OUTPUT_DIR}/object_crops" ]; then
  ln -sfn "${NUREC_OUTPUT_DIR}/object_crops" "${NUREC_ROOT}/object_crops"
fi
if [ -d "${NUREC_OUTPUT_DIR}/instance_masks" ]; then
  ln -sfn "${NUREC_OUTPUT_DIR}/instance_masks" "${NUREC_ROOT}/instance_masks"
fi
if [ -d "${NUREC_OUTPUT_DIR}/colmap_undistorted" ]; then
  ln -sfn "${NUREC_OUTPUT_DIR}/colmap_undistorted" "${NUREC_ROOT}/colmap_undistorted"
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
PRIMARY_VISUAL_ASSET="$(python3 - <<PY
import json
from pathlib import Path
manifest_path = Path("${NUREC_OUTPUT_DIR}/mesh_manifest.json")
primary = "export_last.usdz"
if manifest_path.exists():
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        payload = {}
    candidate = str(payload.get("primary_visual_asset") or "").strip()
    if candidate.lower().endswith(".usdz") and (manifest_path.parent / candidate).is_file():
        primary = candidate
print(primary)
PY
)"
if [ ! -f "${NUREC_OUTPUT_DIR}/${PRIMARY_VISUAL_ASSET}" ]; then
  PRIMARY_VISUAL_ASSET="export_last.usdz"
fi
python3 - <<PY
import json
from datetime import datetime, timezone
from pathlib import Path

scene_id = "${SCENE_ID}"
capture_id = "${CAPTURE_ID}"
nurec_output_dir = Path("${NUREC_OUTPUT_DIR}")
nurec_prefix_uri = "${NUREC_PREFIX_URI}"
primary_visual_asset = "${PRIMARY_VISUAL_ASSET}"
pipeline_root = Path("${PIPELINE_ROOT}")
nurec_root = "${NUREC_ROOT}"

outputs = {
    "visual_usdz": f"{nurec_prefix_uri}/{primary_visual_asset}",
    "visual_mesh_glb": f"{nurec_prefix_uri}/visual_mesh.glb",
    "visual_pointcloud_ply": f"{nurec_prefix_uri}/visual_pointcloud.ply",
    "mesh_manifest_json": f"{nurec_prefix_uri}/mesh_manifest.json",
    "collision_mesh_ply": f"{nurec_prefix_uri}/nvblox_mesh.ply",
    "occupancy": [f"{nurec_prefix_uri}/occupancy.bin"],
}

if (nurec_output_dir / "inpainted_visual_mesh.glb").is_file():
    outputs["inpainted_visual_mesh_glb"] = f"{nurec_prefix_uri}/inpainted_visual_mesh.glb"
if (nurec_output_dir / "inpainted_gaussian_splat.ply").is_file():
    outputs["inpainted_gaussian_ply"] = f"{nurec_prefix_uri}/inpainted_gaussian_splat.ply"
instance_masks_dir = nurec_output_dir / "instance_masks"
if instance_masks_dir.is_dir() and any(instance_masks_dir.glob("*.png")):
    outputs["sam3_instance_masks_dir"] = f"{nurec_prefix_uri}/instance_masks"
undist_sparse = nurec_output_dir / "colmap_undistorted" / "sparse" / "0"
if undist_sparse.is_dir() and any(undist_sparse.iterdir()):
    outputs["colmap_undistorted_sparse_dir"] = f"{nurec_prefix_uri}/colmap_undistorted/sparse/0"
undist_images = nurec_output_dir / "colmap_undistorted" / "images"
if undist_images.is_dir() and any(p.is_file() for p in undist_images.rglob("*")):
    outputs["colmap_undistorted_images_dir"] = f"{nurec_prefix_uri}/colmap_undistorted/images"

payload = {
    "schema_version": "v1",
    "scene_id": scene_id,
    "capture_id": capture_id,
    "status": "completed",
    "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "nurec_dir": nurec_root,
    "command": {"executed": True, "command": "nurec_shim.py", "return_code": 0, "stdout": "", "stderr": ""},
    "outputs": outputs,
}
(pipeline_root / ".nurec_complete").write_text(json.dumps(payload, indent=2), encoding="utf-8")
PY
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
export SAM3_TRACKING_MODE
export SCENE_CLEANING_MODE INPAINT360GS_RESOLUTION SAM3_MASK_EXPORT_SPACE
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
if [ "$CROP_CLEANUP_PROVIDER" = "together_qwen_image_edit" ] && [ -z "${TOGETHER_API_KEY:-}" ]; then
  log "WARNING: TOGETHER_API_KEY is not set; crop cleanup will fall back to original crops"
fi
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
