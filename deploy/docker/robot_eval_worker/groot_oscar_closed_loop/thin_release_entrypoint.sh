#!/usr/bin/env bash
set -euo pipefail

# A thin release is intentionally not runnable without its immutable provider
# model volume.  Verification is offline and happens before worker code starts.
: "${BLUEPRINT_GROOT_OSCAR_EXPECTED_MODEL_MANIFEST_DIGEST:?expected model manifest digest is required}"
/opt/oscar-venv/bin/python -m blueprint_pipeline.groot_oscar_model_cache activate \
  --root "${BLUEPRINT_GROOT_OSCAR_MODEL_CACHE:-/models/blueprint-groot-oscar-v1}" \
  --expected-manifest-digest "${BLUEPRINT_GROOT_OSCAR_EXPECTED_MODEL_MANIFEST_DIGEST}" \
  --out /tmp/blueprint_model_cache_verification.json >/dev/null
exec "$@"
