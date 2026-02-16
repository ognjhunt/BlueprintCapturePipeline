"""Tests for advanced quality gate orchestration."""

from __future__ import annotations

import pytest

from blueprint_pipeline.quality_gates import AdvancedQualityGateConfig, run_advanced_quality_gates


def test_quality_gates_can_be_disabled(tmp_path) -> None:
    report = run_advanced_quality_gates(
        storage_root=tmp_path,
        assets_prefix="scenes/scene_1/assets",
        nurec_outputs={},
        config=AdvancedQualityGateConfig(enabled=False),
    )
    assert report["status"] == "skipped"


def test_quality_gates_require_scene_shell_mesh(tmp_path) -> None:
    with pytest.raises(Exception):
        run_advanced_quality_gates(
            storage_root=tmp_path,
            assets_prefix="scenes/scene_1/assets",
            nurec_outputs={},
            config=AdvancedQualityGateConfig(enabled=True),
        )
