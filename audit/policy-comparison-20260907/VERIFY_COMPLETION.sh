#!/usr/bin/env bash
set -euo pipefail
root_dir="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$root_dir"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$root_dir/audit/policy-comparison-20260907/offline_guard:$root_dir/src:$root_dir"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
case "${1:-core}" in
  core)
    files=(tests/test_policy_scientific_reset.py tests/test_policy_request_evidence.py tests/test_policy_comparison_completion.py
      tests/test_policy_comparison_integrity_audit.py tests/test_policy_comparison_media_audit.py tests/test_policy_comparison_rescore_identity_audit.py
      tests/test_native_task_arena_policy_canary_session.py tests/test_native_task_arena_policy_canary_bundle.py
      tests/test_exact_workcell_variation_matrix.py tests/test_task_evaluation_policy_canary_result_delivery.py
      tests/test_openpi_droid_policy_runtime.py tests/test_groot_n17_droid_policy_runtime.py
      tests/test_groot_n17_wire_client.py tests/test_adp009d_droid_observation.py)
    ;;
  episodes)
    files=(tests/test_adp009d_policy_episode.py tests/test_adp009d_control_episode.py
      tests/test_adp009d_isaac_episode_adapter.py tests/test_native_task_episode_environment.py
      tests/test_native_task_arena_policy_canary_lifecycle_rehearsal.py tests/test_provider_runtime_import_closure.py
      tests/test_adp009d_episode_evidence_index.py)
    ;;
  *) echo 'Usage: VERIFY_COMPLETION.sh core|episodes' >&2; exit 2 ;;
esac
scratch_dir="$(mktemp -d /private/tmp/policy-completion-pytest.XXXXXX)"
trap 'rm -rf "$scratch_dir"' EXIT
sandbox-exec -p '(version 1)(allow default)(deny network*)' \
  /Users/nijelhunt_1/workspace/BlueprintCapturePipeline/.venv/bin/python -m pytest \
  -o addopts='' --basetemp "$scratch_dir/pytest" "${files[@]}" \
  -k 'not real_loopback_round_trip and not loopback_server_exercises_exact_endpoint_envelopes' -q
