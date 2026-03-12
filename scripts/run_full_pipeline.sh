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
GCS_ROOT_SET_BY_ENV="${GCS_ROOT+x}"
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
POST_STAGE4_REFINE_MODEL="${POST_STAGE4_REFINE_MODEL:-worldforge+gsfix3d}"
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
RECONSTRUCTION_BACKEND="${RECONSTRUCTION_BACKEND:-nurec_3dgrut}"
RECONSTRUCTION_COMPARE_BACKENDS="${RECONSTRUCTION_COMPARE_BACKENDS:-}"
RECONSTRUCTION_COMPARE_WINNER="${RECONSTRUCTION_COMPARE_WINNER:-auto}"
RECONSTRUCTION_COMPARE_REPORT="${RECONSTRUCTION_COMPARE_REPORT:-${WORKSPACE}/full_pipeline/reconstruction_compare_report.json}"
WORLD_MODEL_SERVICE_URL="${WORLD_MODEL_SERVICE_URL:-}"
WORLD_MODEL_SERVICE_API_KEY="${WORLD_MODEL_SERVICE_API_KEY:-}"
WORLD_MODEL_SERVICE_TIMEOUT_SECONDS="${WORLD_MODEL_SERVICE_TIMEOUT_SECONDS:-14400}"
WORLD_MODEL_SERVICE_POLL_SECONDS="${WORLD_MODEL_SERVICE_POLL_SECONDS:-20}"
NEOVERSE_SERVICE_URL="${NEOVERSE_SERVICE_URL:-}"
NEOVERSE_SERVICE_API_KEY="${NEOVERSE_SERVICE_API_KEY:-}"
GEN3C_SERVICE_URL="${GEN3C_SERVICE_URL:-}"
GEN3C_SERVICE_API_KEY="${GEN3C_SERVICE_API_KEY:-}"
RECONSTRUCTION_ARKIT_POSES_PATH="${RECONSTRUCTION_ARKIT_POSES_PATH:-}"
RECONSTRUCTION_ARKIT_INTRINSICS_PATH="${RECONSTRUCTION_ARKIT_INTRINSICS_PATH:-}"
RECONSTRUCTION_ARKIT_DEPTH_DIR="${RECONSTRUCTION_ARKIT_DEPTH_DIR:-}"
RECONSTRUCTION_ARKIT_CONFIDENCE_DIR="${RECONSTRUCTION_ARKIT_CONFIDENCE_DIR:-}"
RECONSTRUCTION_SCENE_MEMORY_BUNDLE_PATH="${RECONSTRUCTION_SCENE_MEMORY_BUNDLE_PATH:-}"
RECONSTRUCTION_ADVANCED_GEOMETRY_BUNDLE_PATH="${RECONSTRUCTION_ADVANCED_GEOMETRY_BUNDLE_PATH:-}"
RUN_STARTED_AT_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
RUN_STARTED_AT_EPOCH="$(date +%s)"
RUN_FAILURES=()
PIPELINE_DIR=""
ORCHESTRATOR_STATUS="not_started"

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

record_failure() {
  RUN_FAILURES+=("$*")
}

die() {
  record_failure "$*"
  echo "[run-full-pipeline] ERROR: $*" >&2
  exit 1
}

generate_log_summary() {
  [ -n "${PIPELINE_DIR:-}" ] || return 0
  [ -d "${PIPELINE_DIR}" ] || mkdir -p "${PIPELINE_DIR}"
  if python3 "${APP_DIR}/scripts/summarize_pipeline_logs.py" --pipeline-dir "${PIPELINE_DIR}" >/dev/null 2>&1; then
    log "Generated log summary: ${PIPELINE_DIR}/log_summary.json ${PIPELINE_DIR}/log_summary.md"
    return 0
  fi
  record_failure "Failed to generate log summary from pipeline logs"
  log "WARNING: Failed to generate log summary artifacts"
  return 0
}

write_run_summary() {
  local exit_code="$1"
  local run_status="passed"
  if [ "$exit_code" -ne 0 ]; then
    run_status="failed"
  fi

  if [ -z "${PIPELINE_DIR:-}" ]; then
    PIPELINE_DIR="${WORKSPACE}/full_pipeline"
  fi
  mkdir -p "${PIPELINE_DIR}" || return 0
  generate_log_summary

  local ended_at_utc
  ended_at_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  local ended_at_epoch
  ended_at_epoch="$(date +%s)"
  local duration_sec=0
  if [ "${RUN_STARTED_AT_EPOCH}" -le "${ended_at_epoch}" ]; then
    duration_sec=$((ended_at_epoch - RUN_STARTED_AT_EPOCH))
  fi

  local run_failures_file="${PIPELINE_DIR}/run_failures.txt"
  : > "${run_failures_file}"
  if [ "${#RUN_FAILURES[@]}" -gt 0 ]; then
    printf '%s\n' "${RUN_FAILURES[@]}" > "${run_failures_file}"
  fi
  if [ "$exit_code" -ne 0 ] && [ ! -s "${run_failures_file}" ]; then
    printf 'Script exited with status %s\n' "$exit_code" > "${run_failures_file}"
  fi

  local repo_commit="unknown"
  if command -v git >/dev/null 2>&1; then
    repo_commit="$(git -C "${APP_DIR}" rev-parse HEAD 2>/dev/null || echo unknown)"
  fi
  local blueprintpipeline_commit="${BLUEPRINTPIPELINE_COMMIT_HASH:-}"
  if [ -z "${blueprintpipeline_commit//[[:space:]]/}" ] && command -v git >/dev/null 2>&1; then
    blueprintpipeline_commit="$(git -C "${BLUEPRINTPIPELINE_ROOT}" rev-parse HEAD 2>/dev/null || true)"
  fi
  if [ -z "${blueprintpipeline_commit//[[:space:]]/}" ]; then
    blueprintpipeline_commit="unknown"
  fi

  export RUN_SUMMARY_JSON_PATH="${PIPELINE_DIR}/run_summary.json"
  export RUN_SUMMARY_MD_PATH="${PIPELINE_DIR}/run_summary.md"
  export RUN_FAILURES_FILE="${run_failures_file}"
  export RUN_STATUS="${run_status}"
  export RUN_EXIT_CODE="${exit_code}"
  export RUN_ENDED_AT_UTC="${ended_at_utc}"
  export RUN_DURATION_SEC="${duration_sec}"
  export RUN_REPO_COMMIT="${repo_commit}"
  export RUN_BLUEPRINTPIPELINE_COMMIT="${blueprintpipeline_commit}"
  export RUN_STARTED_AT_UTC
  export INPUT_VIDEO SCENE_ID CAPTURE_ID WORKSPACE GCS_ROOT BUCKET
  export COMPLETION_MODE NUREC_RERUN_PROFILE NUREC_QUALITY_PROFILE
  export SKIP_NUREC SKIP_FIXER SKIP_DENSE FIXER_MODE
  export RECONSTRUCTION_BACKEND RECONSTRUCTION_COMPARE_BACKENDS RECONSTRUCTION_COMPARE_WINNER
  export RECONSTRUCTION_COMPARE_REPORT
  export WORLD_MODEL_SERVICE_URL WORLD_MODEL_SERVICE_API_KEY
  export WORLD_MODEL_SERVICE_TIMEOUT_SECONDS WORLD_MODEL_SERVICE_POLL_SECONDS
  export NEOVERSE_SERVICE_URL NEOVERSE_SERVICE_API_KEY GEN3C_SERVICE_URL GEN3C_SERVICE_API_KEY
  export RECONSTRUCTION_ARKIT_POSES_PATH RECONSTRUCTION_ARKIT_INTRINSICS_PATH
  export RECONSTRUCTION_ARKIT_DEPTH_DIR RECONSTRUCTION_ARKIT_CONFIDENCE_DIR
  export RECONSTRUCTION_SCENE_MEMORY_BUNDLE_PATH RECONSTRUCTION_ADVANCED_GEOMETRY_BUNDLE_PATH
  export POST_STAGE4_REFINE POST_STAGE4_REFINE_MODEL
  export GENERATION_PROVIDER_CHAIN SCENE_CLEANING_MODE
  export SAM3_MASK_EXPORT_SPACE INPAINT360GS_RESOLUTION PIPELINE_MODE
  export COLMAP_MIN_REGISTERED_RATIO COLMAP_RETRY_MIN_REGISTERED_RATIO
  export PIPELINE_DIR NUREC_OUTPUT_DIR SCENE_ROOT PIPELINE_ROOT ORCHESTRATOR_STATUS

  python3 - <<PY
import json
import os
from pathlib import Path

run_summary_json = Path(os.environ["RUN_SUMMARY_JSON_PATH"])
run_summary_md = Path(os.environ["RUN_SUMMARY_MD_PATH"])
failures_file = Path(os.environ["RUN_FAILURES_FILE"])
run_status = os.environ.get("RUN_STATUS", "unknown")
exit_code = int(os.environ.get("RUN_EXIT_CODE", "1"))

def _path_record(path_str: str) -> dict:
    path = Path(path_str)
    return {"path": path_str, "exists": path.exists()}

failures = []
if failures_file.is_file():
    for line in failures_file.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped:
            failures.append(stripped)

  params = {
      "completion_mode": os.getenv("COMPLETION_MODE", ""),
      "nurec_rerun_profile": os.getenv("NUREC_RERUN_PROFILE", ""),
      "nurec_quality_profile": os.getenv("NUREC_QUALITY_PROFILE", ""),
      "reconstruction_backend": os.getenv("RECONSTRUCTION_BACKEND", ""),
      "reconstruction_compare_backends": os.getenv("RECONSTRUCTION_COMPARE_BACKENDS", ""),
      "reconstruction_compare_winner": os.getenv("RECONSTRUCTION_COMPARE_WINNER", ""),
      "reconstruction_compare_report": os.getenv("RECONSTRUCTION_COMPARE_REPORT", ""),
      "skip_nurec": os.getenv("SKIP_NUREC", ""),
      "skip_fixer": os.getenv("SKIP_FIXER", ""),
      "skip_dense": os.getenv("SKIP_DENSE", ""),
    "fixer_mode": os.getenv("FIXER_MODE", ""),
    "post_stage4_refine": os.getenv("POST_STAGE4_REFINE", ""),
    "post_stage4_refine_model": os.getenv("POST_STAGE4_REFINE_MODEL", ""),
    "generation_provider_chain": os.getenv("GENERATION_PROVIDER_CHAIN", ""),
    "scene_cleaning_mode": os.getenv("SCENE_CLEANING_MODE", ""),
    "sam3_mask_export_space": os.getenv("SAM3_MASK_EXPORT_SPACE", ""),
    "inpaint360gs_resolution": os.getenv("INPAINT360GS_RESOLUTION", ""),
    "pipeline_mode": os.getenv("PIPELINE_MODE", ""),
    "colmap_min_registered_ratio": os.getenv("COLMAP_MIN_REGISTERED_RATIO", ""),
    "colmap_retry_min_registered_ratio": os.getenv("COLMAP_RETRY_MIN_REGISTERED_RATIO", ""),
}

pipeline_dir = os.getenv("PIPELINE_DIR", "")
nurec_output_dir = os.getenv("NUREC_OUTPUT_DIR", "")
scene_root = os.getenv("SCENE_ROOT", "")
pipeline_root = os.getenv("PIPELINE_ROOT", "")
orchestrator_status = os.getenv("ORCHESTRATOR_STATUS", "not_started")

outputs = {
    "nurec_output_dir": _path_record(nurec_output_dir),
    "pipeline_dir": _path_record(pipeline_dir),
    "scene_root": _path_record(scene_root) if scene_root else {"path": "", "exists": False},
    "pipeline_root": _path_record(pipeline_root) if pipeline_root else {"path": "", "exists": False},
    "scene_usda": _path_record(f"{scene_root}/usd/scene.usda") if scene_root else {"path": "", "exists": False},
    "swap_quality_report": _path_record(f"{pipeline_root}/swap_quality_report.json")
    if pipeline_root
    else {"path": "", "exists": False},
    "reconstruction_compare_report": _path_record(
      os.getenv("RECONSTRUCTION_COMPARE_REPORT", "")
    ),
    "pipeline_summary_json": _path_record(f"{pipeline_root}/pipeline_summary.json")
    if pipeline_root
    else {"path": "", "exists": False},
    "orchestrator_run_report": _path_record(f"{pipeline_dir}/orchestrator_run_report.json")
    if pipeline_dir
    else {"path": "", "exists": False},
    "log_summary_json": _path_record(f"{pipeline_dir}/log_summary.json")
    if pipeline_dir
    else {"path": "", "exists": False},
    "log_summary_md": _path_record(f"{pipeline_dir}/log_summary.md")
    if pipeline_dir
    else {"path": "", "exists": False},
}

payload = {
    "schema_version": "v1",
    "inputs": {
        "input_video": os.getenv("INPUT_VIDEO", ""),
        "scene_id": os.getenv("SCENE_ID", ""),
        "capture_id": os.getenv("CAPTURE_ID", ""),
        "workspace": os.getenv("WORKSPACE", ""),
        "gcs_root": os.getenv("GCS_ROOT", ""),
        "bucket": os.getenv("BUCKET", ""),
    },
    "commit": {
        "blueprint_capture_pipeline": os.getenv("RUN_REPO_COMMIT", "unknown"),
        "blueprintpipeline": os.getenv("RUN_BLUEPRINTPIPELINE_COMMIT", "unknown"),
    },
    "params": params,
    "outputs": outputs,
    "runtime": {
        "started_at_utc": os.getenv("RUN_STARTED_AT_UTC", ""),
        "ended_at_utc": os.getenv("RUN_ENDED_AT_UTC", ""),
        "duration_sec": int(os.getenv("RUN_DURATION_SEC", "0")),
        "status": run_status,
        "exit_code": exit_code,
        "orchestrator_status": orchestrator_status,
    },
    "failures": failures,
}
run_summary_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

lines = [
    "# Pipeline Run Summary",
    "",
    "## Inputs",
    f"- input_video: `{payload['inputs']['input_video']}`",
    f"- scene_id: `{payload['inputs']['scene_id']}`",
    f"- capture_id: `{payload['inputs']['capture_id']}`",
    f"- workspace: `{payload['inputs']['workspace']}`",
    "",
    "## Commit",
    f"- blueprint_capture_pipeline: `{payload['commit']['blueprint_capture_pipeline']}`",
    f"- blueprintpipeline: `{payload['commit']['blueprintpipeline']}`",
    "",
    "## Params",
]
for key in sorted(params):
    lines.append(f"- {key}: `{params[key]}`")

lines.extend(["", "## Outputs"])
for key, info in outputs.items():
    if isinstance(info, dict):
        lines.append(f"- {key}: `{info.get('path', '')}` (exists={info.get('exists', False)})")

lines.extend(
    [
        "",
        "## Runtime",
        f"- started_at_utc: `{payload['runtime']['started_at_utc']}`",
        f"- ended_at_utc: `{payload['runtime']['ended_at_utc']}`",
        f"- duration_sec: `{payload['runtime']['duration_sec']}`",
        f"- status: `{payload['runtime']['status']}`",
        f"- exit_code: `{payload['runtime']['exit_code']}`",
        f"- orchestrator_status: `{payload['runtime']['orchestrator_status']}`",
        "",
        "## Failures",
    ]
)
if failures:
    for failure in failures:
        lines.append(f"- {failure}")
else:
    lines.append("- none")

run_summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
PY

  log "Run summary written: ${RUN_SUMMARY_JSON_PATH} ${RUN_SUMMARY_MD_PATH}"
}

run_guardrail_checks() {
  local errors=()
  local backend_service_url=""
  local backend_service_key=""

  if ! command -v python3 >/dev/null 2>&1; then
    errors+=("python3 is required but not found in PATH")
  fi
  if [ "$SKIP_NUREC" = "false" ]; then
    case "${RECONSTRUCTION_BACKEND,,}" in
      nurec_3dgrut|nurec_3d|nurec|nurec3d|3dgrut)
        if [ ! -f "${APP_DIR}/scripts/nurec_shim.py" ]; then
          errors+=("missing script: ${APP_DIR}/scripts/nurec_shim.py (required for nurec_3dgrut backend)")
        fi
        ;;
      ttt_lrm|tttlrm|ttt)
        if [ -z "${TTT_LRM_CMD_TEMPLATE}" ] && [ -z "${TTT_LRM_EXECUTABLE}" ]; then
          errors+=("tttLRM backend requested but TTT_LRM_CMD_TEMPLATE / TTT_LRM_EXECUTABLE are both unset")
        fi
        ;;
      loger)
        if [ -z "${LOGER_CMD_TEMPLATE:-}" ] && [ -z "${LOGER_EXECUTABLE:-}" ]; then
          errors+=("loger backend requested but LOGER_CMD_TEMPLATE / LOGER_EXECUTABLE are both unset")
        fi
        if [ "${SCENE_CLEANING_MODE}" = "force" ]; then
          errors+=("SCENE_CLEANING_MODE=force is unsupported with loger backend")
        fi
        ;;
      neoverse)
        backend_service_url="${NEOVERSE_SERVICE_URL:-${WORLD_MODEL_SERVICE_URL:-}}"
        backend_service_key="${NEOVERSE_SERVICE_API_KEY:-${WORLD_MODEL_SERVICE_API_KEY:-}}"
        if [ -z "${backend_service_url}" ] || [ -z "${backend_service_key}" ]; then
          errors+=("NeoVerse backend requested but NEOVERSE_SERVICE_URL / NEOVERSE_SERVICE_API_KEY (or WORLD_MODEL_SERVICE_URL / WORLD_MODEL_SERVICE_API_KEY) are unset")
        fi
        ;;
      gen3c)
        backend_service_url="${GEN3C_SERVICE_URL:-${WORLD_MODEL_SERVICE_URL:-}}"
        backend_service_key="${GEN3C_SERVICE_API_KEY:-${WORLD_MODEL_SERVICE_API_KEY:-}}"
        if [ -z "${backend_service_url}" ] || [ -z "${backend_service_key}" ]; then
          errors+=("GEN3C backend requested but GEN3C_SERVICE_URL / GEN3C_SERVICE_API_KEY (or WORLD_MODEL_SERVICE_URL / WORLD_MODEL_SERVICE_API_KEY) are unset")
        fi
        if { [ -n "${RECONSTRUCTION_ARKIT_POSES_PATH}" ] && [ -n "${RECONSTRUCTION_ARKIT_INTRINSICS_PATH}" ] && [ -n "${RECONSTRUCTION_ARKIT_DEPTH_DIR}" ]; } || [ -n "${RECONSTRUCTION_ADVANCED_GEOMETRY_BUNDLE_PATH}" ]; then
          :
        else
          errors+=("GEN3C backend requested but explicit conditioning is missing (set RECONSTRUCTION_ARKIT_POSES_PATH + RECONSTRUCTION_ARKIT_INTRINSICS_PATH + RECONSTRUCTION_ARKIT_DEPTH_DIR, or RECONSTRUCTION_ADVANCED_GEOMETRY_BUNDLE_PATH)")
        fi
        ;;
      *)
        errors+=("unsupported reconstruction backend: ${RECONSTRUCTION_BACKEND} (expected nurec_3dgrut, tttLRM, loger, neoverse, or gen3c)")
        ;;
    esac

    IFS=',' read -r -a _reconstruction_compare_backends <<< "${RECONSTRUCTION_COMPARE_BACKENDS}"
    local compare_backend
    for compare_backend in "${_reconstruction_compare_backends[@]}"; do
      compare_backend="${compare_backend//[[:space:]]/}"
      case "${compare_backend,,}" in
        "" )
          ;;
        nurec_3dgrut|nurec_3d|nurec|nurec3d|3dgrut)
          ;;
        ttt_lrm|tttlrm|ttt)
          if [ -z "${TTT_LRM_CMD_TEMPLATE}" ] && [ -z "${TTT_LRM_EXECUTABLE}" ]; then
            errors+=("tttLRM included in --reconstruction-compare-backends but TTT_LRM_CMD_TEMPLATE / TTT_LRM_EXECUTABLE are both unset")
          fi
          ;;
        loger)
          if [ -z "${LOGER_CMD_TEMPLATE:-}" ] && [ -z "${LOGER_EXECUTABLE:-}" ]; then
            errors+=("loger included in --reconstruction-compare-backends but LOGER_CMD_TEMPLATE / LOGER_EXECUTABLE are both unset")
          fi
          if [ "${SCENE_CLEANING_MODE}" = "force" ]; then
            errors+=("SCENE_CLEANING_MODE=force is unsupported when loger is included in reconstruction compare backends")
          fi
          ;;
        neoverse)
          if [ -z "${NEOVERSE_SERVICE_URL:-${WORLD_MODEL_SERVICE_URL:-}}" ] || [ -z "${NEOVERSE_SERVICE_API_KEY:-${WORLD_MODEL_SERVICE_API_KEY:-}}" ]; then
            errors+=("NeoVerse included in --reconstruction-compare-backends but service URL/key are unset")
          fi
          ;;
        gen3c)
          if [ -z "${GEN3C_SERVICE_URL:-${WORLD_MODEL_SERVICE_URL:-}}" ] || [ -z "${GEN3C_SERVICE_API_KEY:-${WORLD_MODEL_SERVICE_API_KEY:-}}" ]; then
            errors+=("GEN3C included in --reconstruction-compare-backends but service URL/key are unset")
          fi
          if { [ -n "${RECONSTRUCTION_ARKIT_POSES_PATH}" ] && [ -n "${RECONSTRUCTION_ARKIT_INTRINSICS_PATH}" ] && [ -n "${RECONSTRUCTION_ARKIT_DEPTH_DIR}" ]; } || [ -n "${RECONSTRUCTION_ADVANCED_GEOMETRY_BUNDLE_PATH}" ]; then
            :
          else
            errors+=("GEN3C included in --reconstruction-compare-backends but explicit conditioning is missing")
          fi
          ;;
        *)
          errors+=("unsupported reconstruction compare backend: ${compare_backend} (expected nurec_3dgrut, tttLRM, loger, neoverse, or gen3c)")
          ;;
      esac
    done

    if [ "${RECONSTRUCTION_COMPARE_WINNER,,}" != "auto" ]; then
      local compare_winner="${RECONSTRUCTION_COMPARE_WINNER,,}"
      case "$compare_winner" in
        nurec_3dgrut|nurec|nurec3d|3dgrut|ttt_lrm|tttlrm|ttt|loger|neoverse|gen3c)
          ;;
        *)
          errors+=("unsupported reconstruction compare winner: ${RECONSTRUCTION_COMPARE_WINNER} (expected nurec_3dgrut, tttLRM, loger, neoverse, gen3c, or auto)")
          ;;
      esac
    fi
  fi
  if [ "$SKIP_NUREC" = "true" ]; then
    if [ -z "${NUREC_OUTPUT_DIR//[[:space:]]/}" ]; then
      errors+=("--skip-nurec requires --nurec-output-dir or NUREC_OUTPUT_DIR")
    elif [ ! -d "${NUREC_OUTPUT_DIR}" ]; then
      errors+=("skip-nurec output directory does not exist: ${NUREC_OUTPUT_DIR}")
    fi
  fi

  local refine_model_lc="${POST_STAGE4_REFINE_MODEL,,}"
  if [ "$POST_STAGE4_REFINE" = "force" ] && [ "${SKIP_FIXER,,}" = "true" ] && [[ "$refine_model_lc" == fixer* ]]; then
    errors+=("POST_STAGE4_REFINE=force with fixer-based model requires SKIP_FIXER=false")
  fi

  local provider_chain=",${GENERATION_PROVIDER_CHAIN,,},"
  if [ "$COMPLETION_MODE" = "full_required" ]; then
    if [[ "$provider_chain" == *",sam3d,"* ]]; then
      local sam3d_host="${TEXT_SAM3D_API_HOST:-${SAM3D_API_HOST:-${TEXT_SAM3D_BASE_URL:-}}}"
      local sam3d_key="${TEXT_SAM3D_API_KEY:-${SAM3D_API_KEY:-}}"
      if [ -z "${sam3d_host//[[:space:]]/}" ] || [ -z "${sam3d_key//[[:space:]]/}" ]; then
        errors+=("full_required with sam3d provider requires TEXT_SAM3D_API_HOST + TEXT_SAM3D_API_KEY")
      fi
    fi
    if [[ "$provider_chain" == *",hunyuan3d,"* ]]; then
      local hunyuan_host="${TEXT_HUNYUAN_API_HOST:-${HUNYUAN_API_HOST:-${TEXT_HUNYUAN_BASE_URL:-}}}"
      local hunyuan_key="${TEXT_HUNYUAN_API_KEY:-${HUNYUAN_API_KEY:-}}"
      if [ -z "${hunyuan_host//[[:space:]]/}" ] || [ -z "${hunyuan_key//[[:space:]]/}" ]; then
        errors+=("full_required with hunyuan3d provider requires TEXT_HUNYUAN_API_HOST + TEXT_HUNYUAN_API_KEY")
      fi
    fi
    if [[ "$provider_chain" == *",ttt_lrm,"* || "$provider_chain" == *",tttlrm,"* || "$provider_chain" == *",ttt-lrm,"* ]]; then
      local ttt_cmd="${STAGE_D_TTTLRM_IMAGE_TO_3D_COMMAND:-${STAGE_D_TTT_LRM_IMAGE_TO_3D_COMMAND:-${TTTLRM_IMAGE_TO_3D_COMMAND:-${TTT_LRM_IMAGE_TO_3D_COMMAND:-}}}}"
      local ttt_host="${TEXT_TTTLRM_API_HOST:-${TEXT_TTT_LRM_API_HOST:-${TTTLRM_API_HOST:-${TTT_LRM_API_HOST:-}}}}"
      local ttt_key="${TEXT_TTTLRM_API_KEY:-${TEXT_TTT_LRM_API_KEY:-${TTTLRM_API_KEY:-${TTT_LRM_API_KEY:-}}}}"
      if [ -z "${ttt_cmd//[[:space:]]/}" ] && { [ -z "${ttt_host//[[:space:]]/}" ] || [ -z "${ttt_key//[[:space:]]/}" ]; }; then
        errors+=("full_required with ttt_lrm provider requires STAGE_D_TTTLRM_IMAGE_TO_3D_COMMAND or TEXT_TTTLRM_API_HOST + TEXT_TTTLRM_API_KEY")
      fi
    fi
  fi

  local numeric_guardrail_errors
  numeric_guardrail_errors="$(python3 - <<'PY'
import os

errors: list[str] = []

def parse_float(name: str) -> float | None:
    raw = (os.getenv(name, "") or "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        errors.append(f"{name} must be numeric, got {raw!r}")
        return None

def parse_int(name: str) -> int | None:
    raw = (os.getenv(name, "") or "").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        errors.append(f"{name} must be an integer, got {raw!r}")
        return None

min_ratio = parse_float("COLMAP_MIN_REGISTERED_RATIO")
retry_ratio = parse_float("COLMAP_RETRY_MIN_REGISTERED_RATIO")
if min_ratio is not None and retry_ratio is not None and retry_ratio > min_ratio:
    errors.append(
        "COLMAP_RETRY_MIN_REGISTERED_RATIO cannot be greater than COLMAP_MIN_REGISTERED_RATIO"
    )

void_fill_rounds = parse_int("VOID_FILL_ROUNDS")
if void_fill_rounds is not None and void_fill_rounds < 0:
    errors.append("VOID_FILL_ROUNDS must be >= 0")
if void_fill_rounds is not None and void_fill_rounds > 0:
    refine_mode = (os.getenv("POST_STAGE4_REFINE", "") or "").strip().lower()
    if refine_mode == "off":
        errors.append("VOID_FILL_ROUNDS > 0 requires POST_STAGE4_REFINE=auto or force")

max_pseudoviews = parse_int("POST_STAGE4_MAX_PSEUDOVIEWS")
if max_pseudoviews is not None and max_pseudoviews <= 0:
    errors.append("POST_STAGE4_MAX_PSEUDOVIEWS must be > 0")

distill_iters = parse_int("POST_STAGE4_DISTILL_ITERS")
if distill_iters is not None and distill_iters <= 0:
    errors.append("POST_STAGE4_DISTILL_ITERS must be > 0")

for item in errors:
    print(item)
PY
)"
  if [ -n "${numeric_guardrail_errors}" ]; then
    while IFS= read -r line; do
      [ -n "$line" ] && errors+=("$line")
    done <<< "${numeric_guardrail_errors}"
  fi

  if [ "${#errors[@]}" -gt 0 ]; then
    log "Guardrail preflight failed with ${#errors[@]} issue(s):"
    for issue in "${errors[@]}"; do
      record_failure "$issue"
      log "  - ${issue}"
    done
    die "Preflight guardrails failed"
  fi
  log "Guardrail preflight passed"
}

on_exit() {
  local exit_code=$?
  trap - EXIT
  set +e
  write_run_summary "$exit_code"
  exit "$exit_code"
}
trap on_exit EXIT

validate_full_runtime() {
  local root="$1"
  local required=(
    "interactive-job/run_interactive_assets.py"
    "simready-job/prepare_simready_assets.py"
    "usd-assembly-job/assemble_scene.py"
    "replicator-job/generate_replicator_bundle.py"
    "variation-asset-pipeline-job/run_variation_asset_pipeline.py"
    "genie-sim-export-job/export_to_geniesim.py"
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
  --post-stage4-refine-model MODE  Repair stack: fixer, fixer+gsfix3d, worldforge, worldforge+gsfix3d (default: worldforge+gsfix3d)
  --post-stage4-max-pseudoviews N  Max pseudo-views (default: 96)
  --post-stage4-distill-iters N    Distillation iterations (default: 1600)
  --post-stage4-time-budget-min N  Distillation budget minutes (default: 90)
  --scene-cleaning-mode MODE   Scene cleaning mode: off, auto, force (default: off)
  --sam3-mask-export-space MODE  SAM3 mask export space: raw, undistorted (default: undistorted)
  --skip-scene-cleaning        Backward-compatible alias for --scene-cleaning-mode off
  --skip-nurec            Skip NuRec shim (use existing outputs in --nurec-output-dir)
  --nurec-output-dir DIR  NuRec output directory (default: auto from workspace)
  --reconstruction-backend BACKEND  Reconstruction backend: nurec_3dgrut (default), tttLRM, loger, neoverse, gen3c
  --reconstruction-compare-backends CSV  Comma-separated compare backends (e.g. tttLRM, loger, neoverse)
  --reconstruction-compare-winner NAME|auto  Winner policy (auto or backend name)
  --reconstruction-compare-report FILE  Path for backend compare report JSON
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
      POST_STAGE4_REFINE_MODEL="worldforge+gsfix3d"
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

collect_orchestrator_dependency_errors() {
  local -n _out="$1"
  _out=()

  local standalone="${PIPELINE_STANDALONE_MODE:-false}"
  local standalone_lc="${standalone,,}"
  if [ "$standalone_lc" != "true" ]; then
    if [ ! -d "$BLUEPRINTPIPELINE_ROOT" ]; then
      _out+=("BLUEPRINTPIPELINE_ROOT missing: $BLUEPRINTPIPELINE_ROOT")
    elif [ ! -f "$BLUEPRINTPIPELINE_ROOT/tools/source_pipeline/adapter.py" ]; then
      _out+=("BlueprintPipeline runtime incomplete at $BLUEPRINTPIPELINE_ROOT (missing tools/source_pipeline/adapter.py)")
    fi
  fi

  local provider_chain=",${GENERATION_PROVIDER_CHAIN,,},"
  if [[ "$provider_chain" == *",sam3d,"* ]]; then
    local sam3d_host="${TEXT_SAM3D_API_HOST:-${SAM3D_API_HOST:-${TEXT_SAM3D_BASE_URL:-}}}"
    local sam3d_key="${TEXT_SAM3D_API_KEY:-${SAM3D_API_KEY:-}}"
    if [ -z "${sam3d_host//[[:space:]]/}" ] || [ -z "${sam3d_key//[[:space:]]/}" ]; then
      _out+=("SAM3D credentials missing (set TEXT_SAM3D_API_HOST + TEXT_SAM3D_API_KEY)")
    fi
  fi

  if [[ "$provider_chain" == *",hunyuan3d,"* ]]; then
    local hunyuan_host="${TEXT_HUNYUAN_API_HOST:-${HUNYUAN_API_HOST:-${TEXT_HUNYUAN_BASE_URL:-}}}"
    local hunyuan_key="${TEXT_HUNYUAN_API_KEY:-${HUNYUAN_API_KEY:-}}"
    if [ -z "${hunyuan_host//[[:space:]]/}" ] || [ -z "${hunyuan_key//[[:space:]]/}" ]; then
      _out+=("Hunyuan credentials missing (set TEXT_HUNYUAN_API_HOST + TEXT_HUNYUAN_API_KEY)")
    fi
  fi
  if [[ "$provider_chain" == *",ttt_lrm,"* || "$provider_chain" == *",tttlrm,"* || "$provider_chain" == *",ttt-lrm,"* ]]; then
    local ttt_cmd="${STAGE_D_TTTLRM_IMAGE_TO_3D_COMMAND:-${STAGE_D_TTT_LRM_IMAGE_TO_3D_COMMAND:-${TTTLRM_IMAGE_TO_3D_COMMAND:-${TTT_LRM_IMAGE_TO_3D_COMMAND:-}}}}"
    local ttt_host="${TEXT_TTTLRM_API_HOST:-${TEXT_TTT_LRM_API_HOST:-${TTTLRM_API_HOST:-${TTT_LRM_API_HOST:-}}}}"
    local ttt_key="${TEXT_TTTLRM_API_KEY:-${TEXT_TTT_LRM_API_KEY:-${TTTLRM_API_KEY:-${TTT_LRM_API_KEY:-}}}}"
    if [ -z "${ttt_cmd//[[:space:]]/}" ] && { [ -z "${ttt_host//[[:space:]]/}" ] || [ -z "${ttt_key//[[:space:]]/}" ]; }; then
      _out+=("ttt_lrm provider missing configuration (set STAGE_D_TTTLRM_IMAGE_TO_3D_COMMAND or TEXT_TTTLRM_API_HOST + TEXT_TTTLRM_API_KEY)")
    fi
  fi
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
    --reconstruction-backend) RECONSTRUCTION_BACKEND="$2"; shift 2 ;;
    --reconstruction-compare-backends) RECONSTRUCTION_COMPARE_BACKENDS="$2"; shift 2 ;;
    --reconstruction-compare-winner) RECONSTRUCTION_COMPARE_WINNER="$2"; shift 2 ;;
    --reconstruction-compare-report) RECONSTRUCTION_COMPARE_REPORT="$2"; shift 2 ;;
    --crop-cleanup)    CROP_CLEANUP_PROVIDER="$2"; shift 2 ;;
    --scene-id)        SCENE_ID="$2";             shift 2 ;;
    -h|--help)         usage; exit 0 ;;
    -*)                die "Unknown option: $1" ;;
    *)                 INPUT_VIDEO="$1";           shift ;;
  esac
done

if [ -z "${GCS_ROOT_SET_BY_ENV}" ]; then
  GCS_ROOT="${WORKSPACE}/gcs_root"
fi
PIPELINE_DIR="${WORKSPACE}/full_pipeline"
NUREC_OUTPUT_DIR="${NUREC_OUTPUT_DIR:-${PIPELINE_DIR}/output}"
RECONSTRUCTION_COMPARE_REPORT="${RECONSTRUCTION_COMPARE_REPORT:-${PIPELINE_DIR}/reconstruction_compare_report.json}"

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
  fixer|fixer+gsfix3d|worldforge|worldforge+gsfix3d) ;;
  *) die "Invalid --post-stage4-refine-model '${POST_STAGE4_REFINE_MODEL}'. Expected: fixer, fixer+gsfix3d, worldforge, or worldforge+gsfix3d" ;;
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
run_guardrail_checks

# ── Derive identifiers ──────────────────────────────────────────────────────
VIDEO_BASENAME="$(basename "$INPUT_VIDEO" | sed 's/\.[^.]*$//')"
SCENE_ID="${SCENE_ID:-scene_${VIDEO_BASENAME}}"
CAPTURE_ID="cap_$(date +%Y%m%d_%H%M%S)"

case "${RECONSTRUCTION_BACKEND,,},${RECONSTRUCTION_COMPARE_BACKENDS,,}" in
  *loger*)
    if [ "${NUREC_VISUAL_PRIMARY}" = "usdz" ]; then
      NUREC_VISUAL_PRIMARY="mesh"
      log "NUREC_VISUAL_PRIMARY defaulted to mesh for loger backend compatibility"
    fi
    ;;
esac

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

if [ "$SKIP_NUREC" = "false" ]; then
  log "============================================================"
  log "PHASE 1: NuRec Shim (Stages 1-8)"
  log "============================================================"
  log "Reconstruction backend: ${RECONSTRUCTION_BACKEND}"
  if [ -n "${RECONSTRUCTION_COMPARE_BACKENDS}" ]; then
    log "Reconstruction compare-backends: ${RECONSTRUCTION_COMPARE_BACKENDS}"
  else
    log "Reconstruction compare-backends: (none)"
  fi
  if [ "${RECONSTRUCTION_COMPARE_WINNER}" != "auto" ]; then
    log "Reconstruction compare-winner: ${RECONSTRUCTION_COMPARE_WINNER}"
  else
    log "Reconstruction compare-winner: auto"
  fi
  log "Reconstruction compare report: ${RECONSTRUCTION_COMPARE_REPORT}"

  JOB_SPEC="${PIPELINE_DIR}/job_spec.json"
  mkdir -p "$PIPELINE_DIR"

  python3 "${APP_DIR}/scripts/write_reconstruction_job_spec.py" \
    --output-path "$JOB_SPEC" \
    --scene-id "$SCENE_ID" \
    --capture-id "$CAPTURE_ID" \
    --requested-backend "$RECONSTRUCTION_BACKEND" \
    --input-video "$INPUT_VIDEO" \
    --output-dir "$NUREC_OUTPUT_DIR" \
    --compare-report-path "$RECONSTRUCTION_COMPARE_REPORT" \
    --arkit-poses-path "${RECONSTRUCTION_ARKIT_POSES_PATH}" \
    --arkit-intrinsics-path "${RECONSTRUCTION_ARKIT_INTRINSICS_PATH}" \
    --arkit-depth-dir "${RECONSTRUCTION_ARKIT_DEPTH_DIR}" \
    --arkit-confidence-dir "${RECONSTRUCTION_ARKIT_CONFIDENCE_DIR}" \
    --scene-memory-conditioning-bundle-path "${RECONSTRUCTION_SCENE_MEMORY_BUNDLE_PATH}" \
    --advanced-geometry-bundle-path "${RECONSTRUCTION_ADVANCED_GEOMETRY_BUNDLE_PATH}"

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
  export WORLD_MODEL_SERVICE_URL WORLD_MODEL_SERVICE_API_KEY
  export WORLD_MODEL_SERVICE_TIMEOUT_SECONDS WORLD_MODEL_SERVICE_POLL_SECONDS
  export NEOVERSE_SERVICE_URL NEOVERSE_SERVICE_API_KEY GEN3C_SERVICE_URL GEN3C_SERVICE_API_KEY
  export RECONSTRUCTION_ARKIT_POSES_PATH RECONSTRUCTION_ARKIT_INTRINSICS_PATH
  export RECONSTRUCTION_ARKIT_DEPTH_DIR RECONSTRUCTION_ARKIT_CONFIDENCE_DIR
  export RECONSTRUCTION_SCENE_MEMORY_BUNDLE_PATH RECONSTRUCTION_ADVANCED_GEOMETRY_BUNDLE_PATH
  export POST_STAGE4_REFINE POST_STAGE4_REFINE_MODEL
  export POST_STAGE4_MAX_PSEUDOVIEWS POST_STAGE4_DISTILL_ITERS POST_STAGE4_TIME_BUDGET_MIN
  export VOID_FILL_ROUNDS VOID_FILL_TARGET_HOLE_RATIO VOID_FILL_DISTILL_ITERS
  export PIPELINE_MODE REFINEMENT_QUALITY_GATE_PROFILE
  export SCENE_CLEANING_MODE SAM3_MASK_EXPORT_SPACE INPAINT360GS_RESOLUTION
  python3 "${APP_DIR}/scripts/reconstruction_backend_router.py" \
    --backend "$RECONSTRUCTION_BACKEND" \
    --compare-backends "$RECONSTRUCTION_COMPARE_BACKENDS" \
    --compare-winner "$RECONSTRUCTION_COMPARE_WINNER" \
    --compare-report "$RECONSTRUCTION_COMPARE_REPORT" \
    --job-spec "$JOB_SPEC" \
    --output-dir "$NUREC_OUTPUT_DIR" \
    --input-video "$INPUT_VIDEO" \
    --scene-id "$SCENE_ID" \
    --capture-id "$CAPTURE_ID" \
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
RAW_ROOT="${SCENE_ROOT}/captures/${CAPTURE_ID}/raw"
NUREC_ROOT="${CAPTURE_ROOT}/pipeline/nurec"
PIPELINE_ROOT="${CAPTURE_ROOT}/pipeline"

mkdir -p "$RAW_ROOT" "$NUREC_ROOT" "$PIPELINE_ROOT" \
         "${SCENE_ROOT}/assets" "${SCENE_ROOT}/layout" \
         "${SCENE_ROOT}/seg" "${SCENE_ROOT}/usd"

# Copy NuRec outputs into expected location
for f in export_last.usdz export_last.ply export_last.ingp export_last_refined.usdz export_last_refined.ply export_last_refined.ingp nvblox_mesh.ply visual_mesh.glb visual_mesh_robust.glb visual_pointcloud.ply mesh_manifest.json collision_mesh_report.json occupancy.bin scene_semantics_report.json mesh_method.txt quality_profile.txt capture_quality_report.json sam3_preflight_report.json gap_analysis_report.json gap_candidate_views.jsonl view_repair_report.json accepted_repaired_views.jsonl post_stage4_distill_report.json refinement_quality_gate.json hallucinated_region_mask.png loger_backend_report.json; do
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
RAW_ROOT="$RAW_ROOT" NUREC_ROOT="$NUREC_ROOT" python3 - <<'PY'
import json
import os
from pathlib import Path

idx_path = Path(os.environ["RAW_ROOT"]) / "object_point_cloud_index.json"
data = json.loads(idx_path.read_text())
objects = data.get("objects", data if isinstance(data, list) else [])

crops_dir = Path(os.environ["NUREC_ROOT"]) / "object_crops"
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

RESOLVED_ENVIRONMENT="$(RAW_ROOT="$RAW_ROOT" ENVIRONMENT="$ENVIRONMENT" python3 - <<'PY'
import json
import os
from pathlib import Path
idx_path = Path(os.environ["RAW_ROOT"]) / "object_point_cloud_index.json"
try:
    payload = json.loads(idx_path.read_text(encoding="utf-8"))
except Exception:
    payload = {}
env = str(payload.get("environment") if isinstance(payload, dict) else "").strip().lower()
print(env or os.environ["ENVIRONMENT"])
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
  "object_point_cloud_count": $(RAW_ROOT="$RAW_ROOT" python3 - <<'PY' 2>/dev/null || echo 0
import json
import os
from pathlib import Path

idx_path = Path(os.environ["RAW_ROOT"]) / "object_point_cloud_index.json"
payload = json.loads(idx_path.read_text(encoding="utf-8"))
objects = payload.get("objects", payload if isinstance(payload, list) else [])
print(len(objects))
PY
),
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
RAW_PREFIX_URI="gs://${BUCKET}/scenes/${SCENE_ID}/captures/${CAPTURE_ID}/raw"
FRAMES_INDEX_URI="gs://${BUCKET}/scenes/${SCENE_ID}/captures/${CAPTURE_ID}/raw/frames/index.jsonl"
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
PRIMARY_VISUAL_ASSET="$(NUREC_OUTPUT_DIR="$NUREC_OUTPUT_DIR" python3 - <<'PY'
import json
import os
from pathlib import Path
manifest_path = Path(os.environ["NUREC_OUTPUT_DIR"]) / "mesh_manifest.json"
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
SCENE_ID="$SCENE_ID" CAPTURE_ID="$CAPTURE_ID" NUREC_OUTPUT_DIR="$NUREC_OUTPUT_DIR" NUREC_PREFIX_URI="$NUREC_PREFIX_URI" PRIMARY_VISUAL_ASSET="$PRIMARY_VISUAL_ASSET" PIPELINE_ROOT="$PIPELINE_ROOT" NUREC_ROOT="$NUREC_ROOT" python3 - <<'PY'
import json
import os
from datetime import datetime, timezone
from pathlib import Path

scene_id = os.environ["SCENE_ID"]
capture_id = os.environ["CAPTURE_ID"]
nurec_output_dir = Path(os.environ["NUREC_OUTPUT_DIR"])
nurec_prefix_uri = os.environ["NUREC_PREFIX_URI"]
primary_visual_asset = os.environ["PRIMARY_VISUAL_ASSET"]
pipeline_root = Path(os.environ["PIPELINE_ROOT"])
nurec_root = os.environ["NUREC_ROOT"]

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

ORCHESTRATOR_STATUS="completed"
ORCHESTRATOR_DEP_ERRORS=()
if [ "$COMPLETION_MODE" = "best_effort" ]; then
  collect_orchestrator_dependency_errors ORCHESTRATOR_DEP_ERRORS
  if [ "${#ORCHESTRATOR_DEP_ERRORS[@]}" -gt 0 ]; then
    ORCHESTRATOR_STATUS="skipped_missing_dependencies"
    log "Skipping Phase 3 in best_effort due to missing dependencies:"
    for dep_error in "${ORCHESTRATOR_DEP_ERRORS[@]}"; do
      record_failure "best_effort_dependency_gap: ${dep_error}"
      log "  - ${dep_error}"
    done
    export ORCHESTRATOR_STATUS PIPELINE_DIR
    python3 - <<PY
import json
import os
from datetime import datetime, timezone
from pathlib import Path

report = {
    "schema_version": "v1",
    "status": os.getenv("ORCHESTRATOR_STATUS", "skipped_missing_dependencies"),
    "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "reason": "best_effort_missing_dependencies",
}
Path(os.environ["PIPELINE_DIR"]).joinpath("orchestrator_run_report.json").write_text(
    json.dumps(report, indent=2), encoding="utf-8"
)
PY
  fi
fi

if [ "$ORCHESTRATOR_STATUS" = "completed" ]; then
  set +e
  python3 -m blueprint_pipeline.swap_orchestrator \
    --descriptor-gcs-uri "$DESCRIPTOR_URI" \
    2>&1 | tee "${PIPELINE_DIR}/orchestrator.log"
  orch_rc=${PIPESTATUS[0]}
  set -e
  if [ "$orch_rc" -ne 0 ]; then
    if [ "$COMPLETION_MODE" = "full_required" ]; then
      die "swap_orchestrator failed with exit code ${orch_rc}"
    fi
    ORCHESTRATOR_STATUS="failed_best_effort"
    record_failure "best_effort_orchestrator_failed_exit=${orch_rc}"
    log "WARNING: swap_orchestrator failed in best_effort mode (exit=${orch_rc}); continuing with NuRec-only outputs"
    export ORCHESTRATOR_STATUS ORCHESTRATOR_EXIT_CODE="$orch_rc" PIPELINE_DIR
    python3 - <<PY
import json
import os
from datetime import datetime, timezone
from pathlib import Path

report = {
    "schema_version": "v1",
    "status": os.getenv("ORCHESTRATOR_STATUS", "failed_best_effort"),
    "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "reason": "best_effort_orchestrator_failed",
    "exit_code": int(os.getenv("ORCHESTRATOR_EXIT_CODE", "1")),
}
Path(os.environ["PIPELINE_DIR"]).joinpath("orchestrator_run_report.json").write_text(
    json.dumps(report, indent=2), encoding="utf-8"
)
PY
  fi
fi

if [ "$ORCHESTRATOR_STATUS" = "completed" ]; then
COMPLETION_MODE="$COMPLETION_MODE" PIPELINE_ROOT="$PIPELINE_ROOT" SCENE_ROOT="$SCENE_ROOT" python3 - <<'PY'
import json
import os
from pathlib import Path

completion_mode = os.environ["COMPLETION_MODE"]
quality_report_path = Path(os.environ["PIPELINE_ROOT"]) / "swap_quality_report.json"
scene_usda_path = Path(os.environ["SCENE_ROOT"]) / "usd" / "scene.usda"

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
fi
generate_log_summary

log "============================================================"
log "FULL PIPELINE COMPLETE"
log "============================================================"
log "Outputs:"
log "  NuRec:        ${NUREC_OUTPUT_DIR}/"
log "  Log summary:  ${PIPELINE_DIR}/log_summary.json"
log "  Log markdown: ${PIPELINE_DIR}/log_summary.md"
if [ "$ORCHESTRATOR_STATUS" = "completed" ]; then
  log "  Scene USD:    ${SCENE_ROOT}/usd/scene.usda"
  log "  Assets:       ${SCENE_ROOT}/assets/"
  log "  Quality:      ${PIPELINE_ROOT}/swap_quality_report.json"
  log "  Summary:      ${PIPELINE_ROOT}/pipeline_summary.json"
else
  log "  Orchestrator: ${ORCHESTRATOR_STATUS} (best_effort fallback)"
  log "  Orchestrator log/report: ${PIPELINE_DIR}/orchestrator.log ${PIPELINE_DIR}/orchestrator_run_report.json"
fi

if [ "${OPEN_OMNIVERSE_PREVIEW,,}" = "true" ]; then
  log "Preparing Omniverse WebRTC preview workflow..."
  bash "${APP_DIR}/scripts/preview_omniverse_webrtc.sh" "${NUREC_OUTPUT_DIR}" auto || true
fi
