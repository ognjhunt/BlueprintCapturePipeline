#!/usr/bin/env bash
# Prepare a provider volume once, then verify every model byte offline.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cache_root="${BLUEPRINT_GROOT_OSCAR_MODEL_CACHE:-/models/blueprint-groot-oscar-v1}"
token_file="${BLUEPRINT_GROOT_OSCAR_MODEL_CACHE_HF_TOKEN_FILE:-${HF_TOKEN_FILE:-$HOME/.blueprint-secrets/hf_token}}"

PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}" \
  python3 -m blueprint_pipeline.groot_oscar_model_cache prepare \
  --root "$cache_root" --token-file "$token_file"
PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}" \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python3 -m blueprint_pipeline.groot_oscar_model_cache verify --root "$cache_root"
