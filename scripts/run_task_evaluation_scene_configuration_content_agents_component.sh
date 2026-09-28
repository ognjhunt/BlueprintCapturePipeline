#!/usr/bin/env bash
set -eu

COMPONENT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN=/isaac-sim/python.sh
if [ ! -x "$PYTHON_BIN" ]; then
  PYTHON_BIN="$(command -v python3)"
fi

export BLUEPRINT_SCENE_CONFIGURATION_COMPONENT_ROOT="$COMPONENT_ROOT"
exec "$PYTHON_BIN" -m \
  blueprint_pipeline.task_evaluation_scene_configuration_content_agents_driver
