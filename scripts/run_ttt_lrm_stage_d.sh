#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 6 ]; then
  echo "Usage: run_ttt_lrm_stage_d.sh <reference_image> <output_glb> <output_dir> <scene_id> <object_id> <room_type>" >&2
  exit 2
fi

REFERENCE_IMAGE="$1"
OUTPUT_GLB="$2"
OUTPUT_DIR="$3"
SCENE_ID="$4"
OBJECT_ID="$5"
ROOM_TYPE="$6"

if [ ! -f "$REFERENCE_IMAGE" ]; then
  echo "[ttt_lrm_stage_d] missing reference image: $REFERENCE_IMAGE" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"
mkdir -p "$(dirname "$OUTPUT_GLB")"

LOCAL_BIN="${TTT_LRM_STAGE_D_BIN:-/opt/tttLRM/bin/tttlrm_stage_d}"
if [ -x "$LOCAL_BIN" ]; then
  exec "$LOCAL_BIN" \
    --input-image "$REFERENCE_IMAGE" \
    --output-glb "$OUTPUT_GLB" \
    --output-dir "$OUTPUT_DIR" \
    --scene-id "$SCENE_ID" \
    --object-id "$OBJECT_ID" \
    --room-type "$ROOM_TYPE"
fi

CONTAINER_IMAGE="${TTT_LRM_STAGE_D_CONTAINER_IMAGE:-}"
CONTAINER_BIN="${TTT_LRM_STAGE_D_CONTAINER_BIN:-/opt/tttLRM/bin/tttlrm_stage_d}"
if [ -n "$CONTAINER_IMAGE" ]; then
  REF_DIR="$(cd "$(dirname "$REFERENCE_IMAGE")" && pwd)"
  REF_BASE="$(basename "$REFERENCE_IMAGE")"
  OUT_DIR="$(cd "$(dirname "$OUTPUT_GLB")" && pwd)"
  OUT_BASE="$(basename "$OUTPUT_GLB")"
  OUT_WORK_DIR="$(cd "$OUTPUT_DIR" && pwd)"

  docker run --rm --gpus all \
    -v "$REF_DIR:/ttt_input:ro" \
    -v "$OUT_DIR:/ttt_output_glb" \
    -v "$OUT_WORK_DIR:/ttt_output_dir" \
    "$CONTAINER_IMAGE" \
    "$CONTAINER_BIN" \
    --input-image "/ttt_input/${REF_BASE}" \
    --output-glb "/ttt_output_glb/${OUT_BASE}" \
    --output-dir "/ttt_output_dir" \
    --scene-id "$SCENE_ID" \
    --object-id "$OBJECT_ID" \
    --room-type "$ROOM_TYPE"

  if [ ! -s "$OUTPUT_GLB" ]; then
    echo "[ttt_lrm_stage_d] command succeeded but output missing: $OUTPUT_GLB" >&2
    exit 1
  fi
  exit 0
fi

echo "[ttt_lrm_stage_d] no runtime configured." >&2
echo "[ttt_lrm_stage_d] set TTT_LRM_STAGE_D_BIN or TTT_LRM_STAGE_D_CONTAINER_IMAGE." >&2
exit 1
