#!/usr/bin/env bash
# PIPE-05: fast CI lane.
#
# Heavy tests (real subprocess spawns, Isaac/render pipelines, module-entrypoint
# round-trips) are tagged @pytest.mark.slow / @pytest.mark.gpu and deselected by
# default via addopts in pyproject.toml, so this lane is currently the marker
# expression rather than a hardcoded file list. Treat it as a repository-wide
# integration diagnostic, not as the default build-loop or ordinary-PR command:
# the non-slow collection can still grow beyond the intended wall-time budget.
#
# Usage:
#   scripts/pytest_fast.sh            # repository integration diagnostic
#   scripts/pytest_full.sh            # full suite including slow/gpu tests
set -euo pipefail

PY="${PYTHON:-.venv/bin/python}"
[ -x "$PY" ] || PY="python3"

exec "$PY" -m pytest -q -p no:cacheprovider -m "not slow and not gpu" "$@"
