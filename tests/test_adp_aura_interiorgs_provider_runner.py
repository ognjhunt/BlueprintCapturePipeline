from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
from PIL import Image


def _runner_module():
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(scripts))
    try:
        spec = importlib.util.spec_from_file_location(
            "adp_aura_interiorgs_provider_runner_under_test",
            scripts / "adp_aura_interiorgs_provider_runner.py",
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        # Left first on sys.path, scripts/ shadows every later bare import of a
        # same-named package in this process (scripts/operator_door.py hid the door).
        sys.path.remove(str(scripts))
    return module


def test_loading_the_runner_leaves_sys_path_as_it_was() -> None:
    """2026-10-12 promotion lane 38272358269, shard 10: this helper left scripts/ first
    on sys.path, so scripts/operator_door.py shadowed the deploy/operator-door package
    and 14 scene-retirement bridge tests in the same shard failed to import it."""
    before = list(sys.path)

    _runner_module()

    assert sys.path == before


def test_aura_runner_retains_complete_intermediate_frame_set(tmp_path: Path) -> None:
    runner = _runner_module()
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    for index in range(2):
        Image.new("RGB", (8, 8), (index, 2, 3)).save(source / f"{index:05d}.png")

    records = runner._retain_intermediate_png_set(
        source=source,
        output=output,
        role="sdedit_images",
        expected_count=2,
    )

    assert [row["relative_path"] for row in records] == [
        "artifacts/intermediate_frames/sdedit_images/00000.png",
        "artifacts/intermediate_frames/sdedit_images/00001.png",
    ]
    assert all(str(row["sha256"]).startswith("sha256:") for row in records)


def test_aura_runner_fails_closed_when_intermediate_frame_is_missing(
    tmp_path: Path,
) -> None:
    runner = _runner_module()
    source = tmp_path / "source"
    source.mkdir()
    Image.new("RGB", (8, 8), (1, 2, 3)).save(source / "00000.png")

    with pytest.raises(
        ValueError,
        match="aurafusion360_interiorgs_inpaint_init_renders_frame_set_incomplete",
    ):
        runner._retain_intermediate_png_set(
            source=source,
            output=tmp_path / "output",
            role="inpaint_init_renders",
            expected_count=2,
        )
