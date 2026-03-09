#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 5 ]; then
  echo "Usage: run_ttt_lrm_reconstruction.sh <input_video> <output_dir> <scene_id> <capture_id> <job_spec_path>" >&2
  exit 2
fi

INPUT_VIDEO="$1"
OUTPUT_DIR="$2"
SCENE_ID="$3"
CAPTURE_ID="$4"
JOB_SPEC_PATH="$5"

if [ ! -f "$INPUT_VIDEO" ]; then
  echo "[ttt_lrm_recon] missing input video: $INPUT_VIDEO" >&2
  exit 1
fi

if [ ! -f "$JOB_SPEC_PATH" ]; then
  echo "[ttt_lrm_recon] missing job spec: $JOB_SPEC_PATH" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

LOCAL_BIN="${TTT_LRM_RECON_BIN:-/opt/tttLRM/bin/tttlrm_reconstruct}"
if [ -x "$LOCAL_BIN" ]; then
  exec "$LOCAL_BIN" \
    --input-video "$INPUT_VIDEO" \
    --output-dir "$OUTPUT_DIR" \
    --scene-id "$SCENE_ID" \
    --capture-id "$CAPTURE_ID" \
    --job-spec "$JOB_SPEC_PATH"
fi

CONTAINER_IMAGE="${TTT_LRM_RECON_CONTAINER_IMAGE:-}"
CONTAINER_BIN="${TTT_LRM_RECON_CONTAINER_BIN:-/opt/tttLRM/bin/tttlrm_reconstruct}"
if [ -n "$CONTAINER_IMAGE" ]; then
  VIDEO_DIR="$(cd "$(dirname "$INPUT_VIDEO")" && pwd)"
  VIDEO_BASE="$(basename "$INPUT_VIDEO")"
  OUTPUT_HOST_DIR="$(cd "$OUTPUT_DIR" && pwd)"
  JOB_SPEC_DIR="$(cd "$(dirname "$JOB_SPEC_PATH")" && pwd)"
  JOB_SPEC_BASE="$(basename "$JOB_SPEC_PATH")"

  docker run --rm --gpus all \
    -v "$VIDEO_DIR:/ttt_video:ro" \
    -v "$OUTPUT_HOST_DIR:/ttt_output" \
    -v "$JOB_SPEC_DIR:/ttt_job_spec:ro" \
    "$CONTAINER_IMAGE" \
    "$CONTAINER_BIN" \
    --input-video "/ttt_video/${VIDEO_BASE}" \
    --output-dir "/ttt_output" \
    --scene-id "$SCENE_ID" \
    --capture-id "$CAPTURE_ID" \
    --job-spec "/ttt_job_spec/${JOB_SPEC_BASE}"
  exit 0
fi

echo "[ttt_lrm_recon] no runtime configured." >&2
echo "[ttt_lrm_recon] set TTT_LRM_RECON_BIN or TTT_LRM_RECON_CONTAINER_IMAGE." >&2
exit 1
