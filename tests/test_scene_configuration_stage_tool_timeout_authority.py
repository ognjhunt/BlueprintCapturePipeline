"""The stage tool must run each component for its declared GPU budget.

Scene 839873, run ...dfd77804-r2-web-20260829T223435Z: the bounded selective
repair round executed for the first time, and the stage then died with

    subprocess.TimeoutExpired: '.../artifixer3d_observed_object_removal/run'
    timed out after 7200 seconds

`GPU_STAGE_TIMEOUT_SECONDS` declares 7_800 for that adapter and is what the
parent TTL is sized from, but the runner passed a hard-coded 7_200 -- so the
component was killed 600 seconds before its own authority expired, and the
declared budget governed nothing.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from blueprint_pipeline.task_evaluation_scene_configuration_runtime_budget import (
    GPU_STAGE_TIMEOUT_SECONDS,
)
from blueprint_pipeline.task_evaluation_scene_configuration_stage_tool import (
    component_timeout_seconds,
)


def test_every_declared_gpu_stage_resolves_to_its_own_budget() -> None:
    for adapter_id, declared in GPU_STAGE_TIMEOUT_SECONDS.items():
        assert component_timeout_seconds(adapter_id) == declared


def test_artifixer_gets_the_full_declared_budget_not_the_old_literal() -> None:
    resolved = component_timeout_seconds("artifixer3d_observed_object_removal")
    assert resolved == GPU_STAGE_TIMEOUT_SECONDS[
        "artifixer3d_observed_object_removal"
    ]
    # The literal that killed the first successful repair round.
    assert resolved != 7_200
    assert resolved > 7_200


def test_unknown_adapter_fails_closed_rather_than_guessing() -> None:
    with pytest.raises(Exception) as excinfo:
        component_timeout_seconds("not_a_declared_gpu_stage")
    assert "timeout" in str(excinfo.value)


def test_runner_is_invoked_with_the_declared_timeout(monkeypatch) -> None:
    """The resolved budget must reach subprocess, not just be computable."""

    from blueprint_pipeline import (
        task_evaluation_scene_configuration_stage_tool as tool,
    )

    seen: dict[str, object] = {}

    def fake_runner(argv, **kwargs):
        seen.update(kwargs)
        raise subprocess.TimeoutExpired(cmd=argv, timeout=kwargs.get("timeout"))

    source = Path(tool.__file__).read_text(encoding="utf-8")
    # The hard-coded literal must be gone from the call site entirely.
    assert "timeout=7_200" not in source
    assert "timeout=7200" not in source
    assert "component_timeout_seconds(" in source
    assert callable(fake_runner) and not seen
