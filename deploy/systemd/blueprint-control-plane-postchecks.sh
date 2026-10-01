#!/usr/bin/env bash
set -u

repo="${BLUEPRINT_PIPELINE_REPO:-/opt/blueprint/BlueprintCapturePipeline}"
manifest="${BLUEPRINT_CONTROL_PLANE_OUTPUT_PATH:-/var/lib/blueprint/pipeline-control-plane/live_pipeline_control_plane_manifest.json}"

cd -P "${repo}" || exit 1

# A release tree has no .venv; the unit's shared interpreter carries the
# proof audit's dependencies, which a bare python3 lacks.
if [ -n "${BLUEPRINT_PIPELINE_PYTHON:-}" ] && [ -x "${BLUEPRINT_PIPELINE_PYTHON}" ]; then
  py=(env "PYTHONPATH=${PWD}/src" "${BLUEPRINT_PIPELINE_PYTHON}")
elif [ -x .venv/bin/python ]; then
  py=(env "PYTHONPATH=${PWD}/src" .venv/bin/python)
else
  py=(env "PYTHONPATH=${PWD}/src" python3)
fi

"${py[@]}" -m blueprint_pipeline.live_pipeline_proof_audit --manifest-path "${manifest}"
audit_rc=$?

"${py[@]}" -m blueprint_pipeline.live_pipeline_manifest_alert --manifest-path "${manifest}"
alert_rc=$?

if [ "${audit_rc}" -ne 0 ]; then
  exit "${audit_rc}"
fi
exit "${alert_rc}"
