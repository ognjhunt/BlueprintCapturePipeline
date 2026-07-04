#!/usr/bin/env bash
# PIPE-05: full test lane, including tests tagged slow/gpu (real subprocess
# spawns, Isaac/render pipelines, module-entrypoint round-trips). The empty -m
# expression overrides the fast-lane deselection baked into pyproject addopts.
#
# Usage:
#   scripts/pytest_full.sh
#   BLUEPRINT_TEST_LOCAL_ARTIFACTS=1 scripts/pytest_full.sh  # + local output/ sweep
set -euo pipefail

PY="${PYTHON:-.venv/bin/python}"
[ -x "$PY" ] || PY="python3"

exec "$PY" -m pytest -q -p no:cacheprovider -m "" "$@"
