"""Replay completed Astra authoring into fresh scratch with no model or GPU call."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .task_evaluation_scene_configuration_astra_driver import (
    _DEPENDENCIES_ENV, _INPUT_ENV, _OUTPUT_ENV, _PACKAGE_ENV, _RESULT_ENV, TOOLCHAIN_ROOT_ENV,
    execute_astra_component,
)


def replay_completed_astra_authoring(*, retained_stage_root: Path, replay_root: Path,
                                    toolchain_root: Path, component_root: Path, executor=execute_astra_component):
    """A missing completed boundary refuses; this command cannot finish model work."""
    retained = [retained_stage_root / "astra_cad_blender_runtime"] + sorted(
        (retained_stage_root / "astra_resume_attempts").glob("attempt-????"))
    prior = next((path for path in reversed(retained) if path.exists()), None)
    if prior is None or not replay_root.is_absolute() or replay_root.exists():
        raise ValueError("astra_replay_requires_retained_runtime_and_new_scratch")
    replay_root.mkdir(parents=True, mode=0o700)
    environment = {_INPUT_ENV: str(retained_stage_root / "stage_production_input.v1.json"),
        _DEPENDENCIES_ENV: str(retained_stage_root / "dependency_results.v1.json"),
        _OUTPUT_ENV: str(replay_root), _RESULT_ENV: str(replay_root / "component_result.v1.json"),
        _PACKAGE_ENV: str(component_root), TOOLCHAIN_ROOT_ENV: str(toolchain_root)}
    return executor(environment=environment, no_cost_replay=True, retained_runtime=prior)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retained-stage-root", required=True, type=Path)
    parser.add_argument("--replay-root", required=True, type=Path)
    parser.add_argument("--toolchain-root", required=True, type=Path)
    parser.add_argument("--component-root", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        result = replay_completed_astra_authoring(**vars(args))
    except Exception as exc:
        result = {"status": "blocked", "blocker": str(exc), "failure_type": type(exc).__name__,
                  "new_provider_calls": 0, "gpu_execution_performed": False}
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
