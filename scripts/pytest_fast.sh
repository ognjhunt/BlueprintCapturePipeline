#!/usr/bin/env bash
# PIPE-05: fast CI lane.
#
# Heavy tests (real subprocess spawns, Isaac/render pipelines, module-entrypoint
# round-trips) are tagged @pytest.mark.slow / @pytest.mark.gpu and deselected by
# default via addopts in pyproject.toml, so this lane is simply the marker
# expression — no hardcoded file list. It executes the launch-blocking contracts
# (privacy, rights/proof, success-claim ledger, qualification, alpha readiness)
# hermetically and finishes in well under 90 seconds.
#
# Usage:
#   scripts/pytest_fast.sh            # fast lane (same as bare `pytest`)
#   scripts/pytest_full.sh            # full suite including slow/gpu tests
set -euo pipefail

PY="${PYTHON:-.venv/bin/python}"
[ -x "$PY" ] || PY="python3"

exec "$PY" -m pytest -q -p no:cacheprovider -m "not slow and not gpu" "$@"
