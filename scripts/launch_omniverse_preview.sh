#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  launch_omniverse_preview.sh <pipeline_output_dir> [auto|usdz|mesh] [--launch]

Arguments:
  pipeline_output_dir  Directory containing export_last.usdz / visual_mesh.glb
  auto/usdz/mesh       Artifact to prefer (auto defaults to mesh_manifest primary)
  --launch             Attempt to open with a local viewer after printing commands
EOF
}

if [ $# -lt 1 ] || [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
  usage
  exit 0
fi

OUTPUT_DIR="${1%/}"
MODE="${2:-auto}"
LAUNCH="${3:-}"

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
 )"
    if [ -n "$manifest_primary" ] && [ -f "${OUTPUT_DIR}/${manifest_primary}" ]; then
      PRIMARY="${OUTPUT_DIR}/${manifest_primary}"
    fi
  fi
fi

if [ ! -f "$PRIMARY" ]; then
  if [ -f "$USDCANDIDATE" ]; then
    PRIMARY="$USDCANDIDATE"
  elif [ -f "$MESHCANDIDATE" ]; then
    PRIMARY="$MESHCANDIDATE"
  else
    echo "No preview artifacts found in $OUTPUT_DIR. Run the pipeline first." >&2
    exit 1
  fi
fi

echo "Primary visual artifact: $PRIMARY"
if [ "$PRIMARY" = "$MESHCANDIDATE" ]; then
  echo "Primary mode: generic mesh"
else
  echo "Primary mode: neural USD (recommended for photoreal quality)"
fi

if [ -f "$MANIFEST" ]; then
  echo "Manifest: $MANIFEST"
fi
echo "Viewer compatibility is also documented in mesh_manifest.json: viewer_compatibility + primary_visual_asset"

if [ "$LAUNCH" != "--launch" ]; then
  echo "Run with --launch to attempt local open command."
  exit 0
fi

if command -v usdview >/dev/null 2>&1; then
  echo "Launching usdview..."
  usdview "$PRIMARY" >/tmp/launch_omniverse_preview.log 2>&1 &
  exit 0
fi

if [ "$(uname)" = "Darwin" ] && command -v open >/dev/null 2>&1; then
  candidate_apps=(
    "Isaac Sim"
    "NVIDIA Omniverse"
    "NVIDIA Omniverse Launcher"
    "Omniverse Launcher"
    "USD Composer"
  )
  for app in "${candidate_apps[@]}"; do
    if open -a "$app" "$PRIMARY" >/tmp/launch_omniverse_preview.log 2>&1; then
      exit 0
    fi
  done
  open "$PRIMARY" >/tmp/launch_omniverse_preview.log 2>&1
  exit 0
fi

echo "No known local USD/Omniverse launchers found."
echo "Manual step: drag this file into your Omniverse viewer:"
echo "  $PRIMARY"
exit 1
