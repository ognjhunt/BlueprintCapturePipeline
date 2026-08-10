from __future__ import annotations

from pathlib import Path

import pytest

from blueprint_pipeline.flat_module_closure import (
    FlatModuleClosureError,
    resolve_flat_module_closure,
    stage_flat_module_closure,
)


def _pkg(tmp_path: Path) -> Path:
    root = tmp_path / "pkg"
    root.mkdir()
    (root / "leaf.py").write_text("VALUE = 1\n", encoding="utf-8")
    (root / "middle.py").write_text(
        "try:\n    from leaf import VALUE\n"
        "except ModuleNotFoundError:\n    from .leaf import VALUE\n",
        encoding="utf-8",
    )
    (root / "entry.py").write_text(
        "try:\n    from middle import VALUE\n"
        "except ModuleNotFoundError:\n    from .middle import VALUE\n",
        encoding="utf-8",
    )
    return root


def test_the_closure_follows_imports_all_the_way_down(tmp_path: Path) -> None:
    """Staging only the entry module leaves the runtime importing thin air."""

    closure = resolve_flat_module_closure(
        package_root=_pkg(tmp_path), entry_modules=["entry"]
    )

    assert closure == ["entry", "leaf", "middle"]


def test_staging_proves_the_flat_layout_imports_before_it_ships(
    tmp_path: Path,
) -> None:
    """A missing module is free to find here and costs a launch to find there.

    The provider bundle is flat, so the relative-import fallbacks every module
    carries stop applying. Whether that actually resolves is checkable on a
    laptop and is otherwise discovered on a GPU that is already billing.
    """

    receipt = stage_flat_module_closure(
        package_root=_pkg(tmp_path),
        entry_modules=["entry"],
        destination=tmp_path / "staged",
    )

    assert receipt["import_verified"] is True
    assert sorted(p.name for p in (tmp_path / "staged").glob("*.py")) == [
        "entry.py",
        "leaf.py",
        "middle.py",
    ]


def test_a_module_that_cannot_import_flat_fails_closed(tmp_path: Path) -> None:
    """Relative-only imports work in the repo and break in the bundle."""

    root = _pkg(tmp_path)
    (root / "entry.py").write_text("from .middle import VALUE\n", encoding="utf-8")

    with pytest.raises(FlatModuleClosureError) as excinfo:
        stage_flat_module_closure(
            package_root=root, entry_modules=["entry"], destination=tmp_path / "s2"
        )

    assert any("flat_import_failed:entry" in e for e in excinfo.value.errors)


def test_an_unknown_entry_module_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(FlatModuleClosureError) as excinfo:
        resolve_flat_module_closure(
            package_root=_pkg(tmp_path), entry_modules=["absent"]
        )

    assert any("entry_module_missing:absent" in e for e in excinfo.value.errors)


def test_the_closure_is_deterministic(tmp_path: Path) -> None:
    root = _pkg(tmp_path)
    assert resolve_flat_module_closure(
        package_root=root, entry_modules=["entry"]
    ) == resolve_flat_module_closure(package_root=root, entry_modules=["entry"])
