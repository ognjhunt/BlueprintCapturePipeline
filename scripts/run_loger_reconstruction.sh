#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 5 ]; then
  echo "Usage: run_loger_reconstruction.sh <input_video> <output_dir> <scene_id> <capture_id> <job_spec_path>" >&2
  exit 2
fi

INPUT_VIDEO="$1"
OUTPUT_DIR="$2"
SCENE_ID="$3"
CAPTURE_ID="$4"
JOB_SPEC_PATH="$5"

if [ ! -f "$INPUT_VIDEO" ]; then
  echo "[loger_recon] missing input video: $INPUT_VIDEO" >&2
  exit 1
fi

if [ ! -f "$JOB_SPEC_PATH" ]; then
  echo "[loger_recon] missing job spec: $JOB_SPEC_PATH" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$(dirname "$SCRIPT_DIR")"
mkdir -p "$OUTPUT_DIR"

NATIVE_OUTPUT_DIR="${OUTPUT_DIR}/loger_native"
mkdir -p "$NATIVE_OUTPUT_DIR"

NATIVE_START_EPOCH="$(python3 - <<'PY'
import time
print(f"{time.time():.6f}")
PY
)"

LOCAL_BIN="${LOGER_RECON_BIN:-}"
if [ -n "$LOCAL_BIN" ] && [ -x "$LOCAL_BIN" ]; then
  "$LOCAL_BIN" \
    --input-video "$INPUT_VIDEO" \
    --output-dir "$NATIVE_OUTPUT_DIR" \
    --scene-id "$SCENE_ID" \
    --capture-id "$CAPTURE_ID" \
    --job-spec "$JOB_SPEC_PATH"
elif [ -n "${LOGER_RECON_CONTAINER_IMAGE:-}" ]; then
  VIDEO_DIR="$(cd "$(dirname "$INPUT_VIDEO")" && pwd)"
  VIDEO_BASE="$(basename "$INPUT_VIDEO")"
  OUTPUT_HOST_DIR="$(cd "$OUTPUT_DIR" && pwd)"
  JOB_SPEC_DIR="$(cd "$(dirname "$JOB_SPEC_PATH")" && pwd)"
  JOB_SPEC_BASE="$(basename "$JOB_SPEC_PATH")"
  CONTAINER_BIN="${LOGER_RECON_CONTAINER_BIN:-/opt/LoGeR/bin/loger_reconstruct}"
  docker run --rm --gpus all \
    -v "$VIDEO_DIR:/loger_video:ro" \
    -v "$OUTPUT_HOST_DIR:/loger_output" \
    -v "$JOB_SPEC_DIR:/loger_job_spec:ro" \
    "${LOGER_RECON_CONTAINER_IMAGE}" \
    "$CONTAINER_BIN" \
    --input-video "/loger_video/${VIDEO_BASE}" \
    --output-dir "/loger_output/loger_native" \
    --scene-id "$SCENE_ID" \
    --capture-id "$CAPTURE_ID" \
    --job-spec "/loger_job_spec/${JOB_SPEC_BASE}"
else
  echo "[loger_recon] no native LoGeR runtime configured." >&2
  echo "[loger_recon] set LOGER_RECON_BIN or LOGER_RECON_CONTAINER_IMAGE." >&2
  exit 1
fi

NATIVE_END_EPOCH="$(python3 - <<'PY'
import time
print(f"{time.time():.6f}")
PY
)"
NATIVE_RUNTIME_SEC="$(python3 - <<PY
start = float("${NATIVE_START_EPOCH}")
end = float("${NATIVE_END_EPOCH}")
print(f"{max(0.0, end - start):.6f}")
PY
)"

python3 "${APP_DIR}/scripts/loger_contract_adapter.py" \
  --native-output-dir "$NATIVE_OUTPUT_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --input-video "$INPUT_VIDEO" \
  --job-spec "$JOB_SPEC_PATH" \
  --scene-id "$SCENE_ID" \
  --capture-id "$CAPTURE_ID" \
  --native-runtime-sec "$NATIVE_RUNTIME_SEC"
