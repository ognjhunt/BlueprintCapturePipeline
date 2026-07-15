#!/usr/bin/env bash
set -euo pipefail

# A thin release is intentionally not runnable without its immutable provider
# model volume.  Verification is offline and happens before worker code starts.
: "${BLUEPRINT_GROOT_OSCAR_EXPECTED_MODEL_MANIFEST_DIGEST:?expected model manifest digest is required}"
model_cache="${BLUEPRINT_GROOT_OSCAR_MODEL_CACHE:-/models/blueprint-groot-oscar-v1}"
export BLUEPRINT_GROOT_OSCAR_MODEL_CACHE="$model_cache"
export BLUEPRINT_GROOT_OSCAR_OSCAR_CHECKPOINT="$model_cache/oscar"
export BLUEPRINT_GROOT_OSCAR_SONIC_CHECKPOINT="$model_cache/sonic"
/opt/oscar-venv/bin/python -m blueprint_pipeline.groot_oscar_model_cache activate \
  --root "$model_cache" \
  --expected-manifest-digest "${BLUEPRINT_GROOT_OSCAR_EXPECTED_MODEL_MANIFEST_DIGEST}" \
  --out /tmp/blueprint_model_cache_verification.json >/dev/null
# The RunPod adapter removes the historical worker executable when the image
# supplies an ENTRYPOINT. Restore it when only worker flags reach this wrapper.
if [[ $# -eq 0 || "${1}" == -* ]]; then
  set -- /opt/oscar-venv/bin/blueprint-run-robot-eval-worker "$@"
fi
exec "$@"
