#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  preview_omniverse_webrtc.sh <pipeline_output_dir> [auto|usdz|mesh]

Optional env vars:
  ISAAC_WEBRTC_ENDPOINT       URL of the Isaac Sim WebRTC streaming endpoint (for example, https://host:3000)
  ISAAC_WEBRTC_REMOTE_TARGET  Optional "user@host" SSH target for direct asset upload
  ISAAC_WEBRTC_REMOTE_PATH    Remote upload destination (default: /tmp/omniverse-preview)
  ISAAC_WEBRTC_REMOTE_PORT    SSH port for upload (default: 22)
  ISAAC_WEBRTC_LOCAL_STAGE    Local staging dir for HTTP preview hosting (default: /tmp/omniverse_webrtc_preview)

What this script does:
  - Resolves primary visual artifact (USDZ preferred) from pipeline output
  - Prints exact commands to copy to Isaac Sim server and connect WebRTC client
  - Optionally uploads when ISAAC_WEBRTC_REMOTE_TARGET is set and SSH is available

Examples:
  ISAAC_WEBRTC_ENDPOINT=https://10.0.0.10:3000 bash scripts/preview_omniverse_webrtc.sh /Users/.../pipeline_output
  ISAAC_WEBRTC_REMOTE_TARGET=ubuntu@10.0.0.10 ISAAC_WEBRTC_REMOTE_PATH=/tmp/omniverse_preview bash scripts/preview_omniverse_webrtc.sh ... usdz
EOF
}

if [ $# -lt 1 ] || [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
  usage
  exit 0
fi

OUTPUT_DIR="${1%/}"
MODE="${2:-auto}"

if [ ! -d "$OUTPUT_DIR" ]; then
  echo "Missing pipeline output directory: $OUTPUT_DIR" >&2
  exit 1
fi

if [ "$MODE" != "auto" ] && [ "$MODE" != "usdz" ] && [ "$MODE" != "mesh" ]; then
  echo "Invalid mode '${MODE}'. Use auto, usdz, or mesh." >&2
  exit 1
fi

USDCANDIDATE="${OUTPUT_DIR}/export_last.usdz"
MESHCANDIDATE="${OUTPUT_DIR}/visual_mesh.glb"
MANIFEST="${OUTPUT_DIR}/mesh_manifest.json"
PRIMARY="$USDCANDIDATE"

if [ "$MODE" = "mesh" ]; then
  PRIMARY="$MESHCANDIDATE"
elif [ "$MODE" = "auto" ]; then
  if [ -f "$MANIFEST" ]; then
    manifest_primary="$(python3 - "$MANIFEST" <<'PY'
import json, sys

path = sys.argv[1]
try:
    payload = json.loads(open(path, "r", encoding="utf-8").read())
    value = (payload.get("primary_visual_asset") or "").strip()
    if value:
        print(value)
except Exception:
    pass
PY
)"  # noqa
    if [ -n "$manifest_primary" ] && [ -f "${OUTPUT_DIR}/${manifest_primary}" ]; then
      PRIMARY="${OUTPUT_DIR}/${manifest_primary}"
    fi
  fi
fi

if [ ! -f "$PRIMARY" ]; then
  if [ "$MODE" != "mesh" ] && [ -f "$USDCANDIDATE" ]; then
    PRIMARY="$USDCANDIDATE"
  elif [ -f "$MESHCANDIDATE" ]; then
    PRIMARY="$MESHCANDIDATE"
  else
    echo "No preview artifact available in $OUTPUT_DIR." >&2
    exit 1
  fi
fi

ENDPOINT="${ISAAC_WEBRTC_ENDPOINT:-${ISAAC_STREAM_ENDPOINT:-}}"
REMOTE_TARGET="${ISAAC_WEBRTC_REMOTE_TARGET:-}"
REMOTE_PATH="${ISAAC_WEBRTC_REMOTE_PATH:-/tmp/omniverse-preview}"
REMOTE_PORT="${ISAAC_WEBRTC_REMOTE_PORT:-22}"
LOCAL_STAGE="${ISAAC_WEBRTC_LOCAL_STAGE:-/tmp/omniverse_webrtc_preview}"

MODE_LABEL="neural_usdz"
[ "$PRIMARY" = "$MESHCANDIDATE" ] && MODE_LABEL="glb_sidecar"
if [ -f "$MANIFEST" ] && grep -q '"primary_visual_asset"' "$MANIFEST" >/dev/null 2>&1; then
  manifest_primary="$(python3 - "$MANIFEST" <<'PY'
import json, sys

path = sys.argv[1]
try:
    payload = json.loads(open(path, "r", encoding="utf-8").read())
    print(payload.get("primary_visual_asset") or "")
except Exception:
    print("")
PY
)"
  if [ "$manifest_primary" = "visual_mesh.glb" ]; then
    MODE_LABEL="glb_sidecar"
  elif [ "$manifest_primary" = "export_last.usdz" ]; then
    MODE_LABEL="neural_usdz"
  fi
fi

BASENAME="$(basename "$PRIMARY")"
LOCAL_COPY="${LOCAL_STAGE}/${BASENAME}"

echo "=== Omniverse WebRTC Preview Plan ==="
echo "Pipeline output: $OUTPUT_DIR"
echo "Primary visual artifact: $PRIMARY"
echo "Primary mode: $MODE_LABEL"

if [ "$MODE_LABEL" = "neural_usdz" ]; then
  echo "Note: this is the photoreal neural volume export and should be preferred in Isaac Sim."
else
  echo "Note: this is the generic mesh sidecar. Keep export_last.usdz for max visual fidelity."
fi

echo
echo "Step 1/3: Stage local asset"
mkdir -p "$LOCAL_STAGE"
cp -f "$PRIMARY" "$LOCAL_COPY"
echo "  Local staged path: $LOCAL_COPY"

if [ -n "$ENDPOINT" ]; then
  echo "  Streaming endpoint: $ENDPOINT"
else
  echo "  Streaming endpoint: <set ISAAC_WEBRTC_ENDPOINT to auto-print full launch line>"
fi

echo
echo "Step 2/3: Option A (same machine preview if your HTTP server path is exposed to Isaac)"
echo "  cd \"$LOCAL_STAGE\" && python3 -m http.server ${ISAAC_WEBRTC_HTTP_PORT:-8000}"
echo "  In Isaac: load the hosted stage URL using your preferred method."
if [ -n "$ENDPOINT" ]; then
  echo "    Primary URL example: ${ENDPOINT%/}/${BASENAME}"
fi

if [ -n "$REMOTE_TARGET" ]; then
  echo
  echo "Step 2/3: Option B (upload directly to server)"
  echo "  scp -P ${REMOTE_PORT} \"$LOCAL_COPY\" \"${REMOTE_TARGET}:${REMOTE_PATH}/\""
  echo "  Then on Isaac, open from that path:"
  echo "    ${REMOTE_PATH}/${BASENAME}"
fi

if [ -n "$ENDPOINT" ]; then
  echo
  echo "Step 3/3: Start Isaac Sim WebRTC client"
  echo "  open -a \"isaacsim-webrtc-streaming-client\" --args \"$ENDPOINT\""
  echo "  If the app does not auto-open with args, open manually and paste endpoint: $ENDPOINT"
  echo "  Then load the staged/uploaded asset path in Isaac Sim UI."
else
  echo
  echo "Set ISAAC_WEBRTC_ENDPOINT before launching the client, then re-run this script."
fi

echo
echo "Artifacts:"
echo "  USDZ: ${USDCANDIDATE}"
echo "  Mesh GLB: ${MESHCANDIDATE}"
