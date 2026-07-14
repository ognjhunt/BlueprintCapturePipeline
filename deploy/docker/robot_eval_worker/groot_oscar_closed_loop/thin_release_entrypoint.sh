#!/usr/bin/env bash
set -euo pipefail

# A thin release is intentionally not runnable without its immutable provider
# model volume.  Verification is offline and happens before worker code starts.
/opt/robot-venv/bin/python -m blueprint_pipeline.groot_oscar_model_cache activate \
  --root "${BLUEPRINT_GROOT_OSCAR_MODEL_CACHE:-/models/blueprint-groot-oscar-v1}" \
  --out /tmp/blueprint_model_cache_verification.json >/dev/null
exec "$@"
