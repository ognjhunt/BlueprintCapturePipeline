from __future__ import annotations

from pathlib import Path

from blueprint_pipeline.adp_scene_literal_lint import scan_scene_literal_violations


def test_reusable_source_has_no_historical_scene_literals() -> None:
    source = Path(__file__).resolve().parents[1] / "src/blueprint_pipeline"
    assert scan_scene_literal_violations(source) == []


def test_lint_catches_new_generic_scene_assumption(tmp_path: Path) -> None:
    (tmp_path / "generic_runtime.py").write_text(
        'SCENE_ID = "840313"\n', encoding="utf-8"
    )
    assert scan_scene_literal_violations(tmp_path) == [
        {
            "relative_path": "generic_runtime.py",
            "line_number": 1,
            "literals": ["840313"],
        }
    ]
