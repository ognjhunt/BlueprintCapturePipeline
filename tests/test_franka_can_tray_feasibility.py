from pathlib import Path

import pytest

from blueprint_pipeline.franka_can_tray_feasibility import _scene_xml, _stage_model


def test_scene_contract_is_scripted_control_with_can_and_tray() -> None:
    xml = _scene_xml()
    assert 'name="spraycan"' in xml
    assert 'name="tray_base"' in xml
    assert 'include file="mjx_scene.xml"' in xml


def test_stage_model_fails_closed_when_pinned_asset_is_incomplete(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="menagerie_missing"):
        _stage_model(tmp_path / "missing", tmp_path / "out")


def test_stage_model_preserves_source_with_symlinks(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "mjx_scene.xml").write_text("<mujoco/>", encoding="utf-8")
    (source / "mjx_panda.xml").write_text("<mujoco/>", encoding="utf-8")
    (source / "assets").mkdir()
    scene = _stage_model(source, tmp_path / "out")
    assert scene.is_file()
    assert (scene.parent / "mjx_scene.xml").is_symlink()
    assert (scene.parent / "assets").is_symlink()
