"""Neutral stage entrypoint. Legacy engines are imported only for issued routes."""
from __future__ import annotations

import os
from pathlib import Path
from .task_object_authoring_contracts import read_json, require


def main():
    stage_input = read_json(Path(os.environ["BLUEPRINT_SCENE_CONFIGURATION_STAGE_INPUT"]))
    backend = stage_input["configuration"].get("authoring_backend", "content_agents")
    if backend == "artisan_v1":
        from .task_evaluation_scene_configuration_artisan_driver import execute_artisan_component
        execute_artisan_component()
    else:
        require(backend in {"content_agents", "astra_cad_blender_v1"}, "authoring_backend_unsupported")
        from .task_evaluation_scene_configuration_content_agents_driver import execute_content_agents_component
        execute_content_agents_component()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
