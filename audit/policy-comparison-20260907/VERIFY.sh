#!/usr/bin/env bash
set -euo pipefail
root_dir="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$root_dir"
audit_dir="$root_dir/audit/policy-comparison-20260907"
interpreter=/Users/nijelhunt_1/workspace/BlueprintCapturePipeline/.venv/bin/python
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$audit_dir/offline_guard:$root_dir/src:$root_dir"
group="${1:-regressions}"
extra=(-k '')
case "$group" in
  regressions)
    files=(tests/test_policy_comparison_integrity_audit.py tests/test_policy_comparison_media_audit.py tests/test_policy_comparison_rescore_identity_audit.py)
    ;;
  lifecycle)
    files=(tests/test_policy_comparison_integrity_audit.py tests/test_native_task_arena_policy_canary_session.py tests/test_native_task_arena_policy_canary_lifecycle_rehearsal.py tests/test_provider_runtime_import_closure.py)
    ;;
  clients)
    files=(tests/test_policy_comparison_integrity_audit.py tests/test_policy_comparison_media_audit.py tests/test_policy_comparison_rescore_identity_audit.py tests/test_openpi_droid_policy_runtime.py tests/test_groot_n17_droid_policy_runtime.py)
    extra=(-k 'not real_loopback_round_trip')
    ;;
  focused)
    files=(
      tests/test_native_task_arena_policy_canary_session.py
      tests/test_native_task_arena_policy_canary_bundle.py
      tests/test_exact_workcell_variation_matrix.py
      tests/test_adp009d_franka_evaluation_harness.py
      tests/test_adp009d_policy_episode.py
      tests/test_adp009d_task_scoring.py
      tests/test_adp_task_scoring.py
      tests/test_adp_rigid_retreat_scoring.py
      tests/test_policy_episode_trace_evidence.py
      tests/test_policy_episode_lifecycle.py
      tests/test_episode_visual_evidence.py
      tests/test_adp009d_episode_evidence_index.py
      tests/test_native_task_episode_environment.py
      tests/test_adp009d_droid_observation.py
      tests/test_task_evaluation_policy_canary_rescore.py
      tests/test_task_evaluation_scene_policy_binding.py
      tests/test_native_policy_canary_control_gate.py
      tests/test_policy_canary_episode_interpretation_backfill.py
      tests/test_adp_prospective_design.py
      tests/test_policy_comparison_integrity_audit.py
      tests/test_policy_comparison_media_audit.py
      tests/test_policy_comparison_rescore_identity_audit.py
    )
    ;;
  *) echo 'Usage: VERIFY.sh regressions|lifecycle|clients|focused' >&2; exit 2 ;;
esac
scratch_dir="$(mktemp -d /private/tmp/policy-integrity-pytest.XXXXXX)"
trap 'rm -rf "$scratch_dir"' EXIT
# OS denial also applies to child processes and C transports; the Python
# tripwire provides an explicit error for an unexpected Python connection.
sandbox-exec -p '(version 1)(allow default)(deny network*)' "$interpreter" -m pytest \
  -o addopts='' --basetemp "$scratch_dir/pytest" "${files[@]}" "${extra[@]}" -q
