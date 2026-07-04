#!/usr/bin/env bash
# PIPE-05: fast CI lane.
#
# The full `pytest` suite spawns real subprocesses (Isaac / provider / render /
# module-entrypoint tests) and does not finish in a reasonable CI window. The
# `slow`/`gpu`/`integration` markers are registered in pyproject.toml with
# --strict-markers; the deeper follow-up is to tag every heavy test @pytest.mark.slow
# and default-deselect it. Until that sweep lands, this script runs the critical-path
# subset (the launch-blocking contracts: privacy, rights/proof, webapp sync,
# qualification, evaluation-prep, alpha readiness, e2e fail-closed) which completes in
# well under a minute and is the right pre-push gate.
#
# Usage:
#   scripts/pytest_fast.sh                # critical-path subset
#   scripts/pytest_fast.sh -m "not slow"  # once heavy tests are marked slow
set -euo pipefail

PY="${PYTHON:-.venv/bin/python}"
[ -x "$PY" ] || PY="python3"

exec "$PY" -m pytest -q -p no:cacheprovider \
  tests/test_proof_contracts.py \
  tests/test_pipe_beta_privacy_rights_gates.py \
  tests/test_privacy_processing.py \
  tests/test_qualification_alpha.py \
  tests/test_qualification_coverage_edges.py \
  tests/test_evaluation_prep_stage_coverage_edges.py \
  tests/test_site_world_packaging.py \
  tests/test_alpha_readiness.py \
  tests/test_small_launch_and_proof_coverage.py \
  tests/test_pubsub_handoff_listener.py \
  tests/test_storage_trigger.py \
  "$@"
