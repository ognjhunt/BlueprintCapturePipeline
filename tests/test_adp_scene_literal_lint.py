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


def test_a_docstring_may_explain_the_history_it_warns_against(tmp_path: Path) -> None:
    """Naming the first scene while explaining why anchoring on it was a
    mistake documents the defect; carrying the same digits in code commits it.
    Rewording the prose to pass the lint would trade the accurate explanation --
    the part that stops the next author repeating it -- for a green check."""

    (tmp_path / "screening.py").write_text(
        '"""Walk the frozen order.\n'
        "\n"
        "840313 is the only scene that ever reached a sealed source bundle, and\n"
        "later work anchored on it rather than on the frozen rank.\n"
        '"""\n'
        "\n"
        "def select(order):\n"
        '    """Return the first eligible scene, never 840313 by name."""\n'
        "    return order[0]\n",
        encoding="utf-8",
    )

    assert scan_scene_literal_violations(tmp_path) == []


def test_a_string_that_is_not_a_docstring_is_still_a_violation(tmp_path: Path) -> None:
    (tmp_path / "sneaky.py").write_text(
        "def select():\n"
        "    scene = None\n"
        '    "840313"\n'
        "    return scene\n",
        encoding="utf-8",
    )

    assert scan_scene_literal_violations(tmp_path) == [
        {"relative_path": "sneaky.py", "line_number": 3, "literals": ["840313"]}
    ]


def test_an_unparseable_module_gets_the_strict_line_scan(tmp_path: Path) -> None:
    """A lint must not relax because it could not read the file."""

    (tmp_path / "broken.py").write_text(
        'def select(\n    scene = "840313"\n', encoding="utf-8"
    )

    assert scan_scene_literal_violations(tmp_path) == [
        {"relative_path": "broken.py", "line_number": 2, "literals": ["840313"]}
    ]
