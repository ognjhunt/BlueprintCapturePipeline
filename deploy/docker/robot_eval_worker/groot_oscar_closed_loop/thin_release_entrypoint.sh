#!/usr/bin/env bash
set -euo pipefail

model_asset_mode="${BLUEPRINT_GROOT_OSCAR_FOUNDATION_MODEL_ASSETS:-external}"
if [[ "$model_asset_mode" == "external" ]]; then
  # A thin release is intentionally not runnable without its immutable provider
  # model volume. Verification is offline and happens before worker code starts.
  : "${BLUEPRINT_GROOT_OSCAR_EXPECTED_MODEL_MANIFEST_DIGEST:?expected model manifest digest is required}"
  model_cache="${BLUEPRINT_GROOT_OSCAR_MODEL_CACHE:-/models/blueprint-groot-oscar-v1}"
  runtime_cache_root="$model_cache"
  if [[ "${BLUEPRINT_RUNPOD_SERVERLESS_NETWORK_VOLUME_RUNTIME:-}" == "true" ]]; then
    volume_cache_parent="$(dirname "$model_cache")"
    runtime_cache_link="/workspace/.blueprint-model-cache"
    if [[ -L "$runtime_cache_link" ]]; then
      [[ "$(readlink -f "$runtime_cache_link")" == "$(readlink -f "$volume_cache_parent")" ]] || {
        echo "serverless model-cache compatibility link mismatch" >&2
        exit 2
      }
    elif [[ -e "$runtime_cache_link" ]]; then
      echo "serverless model-cache compatibility path is not a symlink" >&2
      exit 2
    else
      ln -s "$volume_cache_parent" "$runtime_cache_link"
    fi
    runtime_cache_root="$runtime_cache_link/$(basename "$model_cache")"
  fi
  export BLUEPRINT_GROOT_OSCAR_MODEL_CACHE="$model_cache"
  export BLUEPRINT_GROOT_OSCAR_OSCAR_CHECKPOINT="$model_cache/oscar"
  export BLUEPRINT_GROOT_OSCAR_SONIC_CHECKPOINT="$model_cache/sonic"
  /opt/oscar-venv/bin/python -m blueprint_pipeline.groot_oscar_model_cache activate \
    --root "$model_cache" \
    --runtime-cache-root "$runtime_cache_root" \
    --expected-manifest-digest "${BLUEPRINT_GROOT_OSCAR_EXPECTED_MODEL_MANIFEST_DIGEST}" \
    --out /tmp/blueprint_model_cache_verification.json >/dev/null
elif [[ "$model_asset_mode" == "embedded" ]]; then
  export BLUEPRINT_GROOT_OSCAR_OSCAR_CHECKPOINT=/opt/blueprint/ckpts/oscar
  export BLUEPRINT_GROOT_OSCAR_SONIC_CHECKPOINT=/opt/blueprint/ckpts/sonic
  /opt/oscar-venv/bin/python /opt/blueprint/groot_oscar_closed_loop_image_healthcheck.py --build-time >/dev/null
else
  echo "invalid BLUEPRINT_GROOT_OSCAR_FOUNDATION_MODEL_ASSETS=$model_asset_mode" >&2
  exit 2
fi
# The RunPod adapter removes the historical worker executable when the image
# supplies an ENTRYPOINT. Restore it when only worker flags reach this wrapper.
if [[ $# -eq 0 || "${1}" == -* ]]; then
  set -- /opt/oscar-venv/bin/blueprint-run-robot-eval-worker "$@"
fi
exec "$@"
