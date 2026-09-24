from pathlib import Path

import pytest

from blueprint_pipeline.native_g1_usd_dependency_closure import (
    robot_usd_dependency_sources,
)


def test_relative_dependency_closure_is_sorted_and_source_local(tmp_path: Path) -> None:
    source = tmp_path / "g1.usda"
    (tmp_path / "b.usda").write_text('#usda 1.0\ndef Xform "B" {}\n')
    (tmp_path / "a.usda").write_text('#usda 1.0\ndef Xform "A" {}\n')
    (tmp_path / "texture.png").write_bytes(b"texture")
    source.write_text('''#usda 1.0
def Xform "G1" {
    def Xform "B" (references = @b.usda@</B>) {}
    def Xform "A" (references = @a.usda@</A>) {}
}
def Shader "Material" {
    asset inputs:file = @texture.png@
}
''')
    assert [relative.as_posix() for _, relative in robot_usd_dependency_sources(source)] == [
        "a.usda", "b.usda", "texture.png"
    ]


@pytest.mark.parametrize("reference", ["../outside.usda", "missing.usda"])
def test_escaping_or_unresolved_dependency_fails_closed(
    tmp_path: Path, reference: str
) -> None:
    source = tmp_path / "robot" / "g1.usda"
    source.parent.mkdir()
    (tmp_path / "outside.usda").write_text('#usda 1.0\ndef Xform "O" {}\n')
    source.write_text(
        '#usda 1.0\ndef Xform "G1" (references = @'
        + reference
        + '@</O>) {}\n'
    )
    with pytest.raises(ValueError, match="native_g1_usd_dependency_"):
        robot_usd_dependency_sources(source)
