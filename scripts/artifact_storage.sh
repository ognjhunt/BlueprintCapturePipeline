#!/usr/bin/env bash

# Shared shell defaults for disposable generated artifacts. Keep this file
# dependency-free so image-build and provider scripts can source it before
# Python is available.

blueprint_artifact_cache_root() {
  if [[ -n "${BLUEPRINT_ARTIFACT_CACHE_ROOT:-}" ]]; then
    printf '%s\n' "$BLUEPRINT_ARTIFACT_CACHE_ROOT"
    return
  fi
  if [[ "$(uname -s)" == "Darwin" ]]; then
    printf '%s\n' "${HOME}/Library/Caches/BlueprintCapturePipeline"
    return
  fi
  printf '%s\n' "${XDG_CACHE_HOME:-${HOME}/.cache}/BlueprintCapturePipeline"
}

blueprint_evidence_root() {
  if [[ -n "${BLUEPRINT_EVIDENCE_ROOT:-}" ]]; then
    printf '%s\n' "$BLUEPRINT_EVIDENCE_ROOT"
    return
  fi
  if [[ "$(uname -s)" == "Darwin" ]]; then
    printf '%s\n' "${HOME}/Library/Application Support/BlueprintCapturePipeline/evidence"
    return
  fi
  printf '%s\n' "${XDG_DATA_HOME:-${HOME}/.local/share}/BlueprintCapturePipeline/evidence"
}

blueprint_legacy_output_root() {
  if [[ "${BLUEPRINT_ALLOW_REPO_OUTPUT:-}" == "1" || "${BLUEPRINT_ALLOW_REPO_OUTPUT:-}" == "true" ]]; then
    printf '%s\n' "$1/output"
  else
    blueprint_artifact_cache_root
  fi
}
