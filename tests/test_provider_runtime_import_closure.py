"""A bundle's shipped package must import on a provider before any GPU is paid for: the closure validator."""

from __future__ import annotations

from pathlib import Path
import textwrap

import pytest

import blueprint_pipeline.provider_runtime_import_closure as closure
from blueprint_pipeline.provider_runtime_import_closure import (
    CONTROL_PLANE_ONLY_LAZY_IMPORTS,
    assert_provider_runtime_import_closure,
    provider_runtime_import_closure_blockers,
)


REPO_PACKAGE = Path(closure.__file__).resolve().parent


def test_every_exemption_is_a_lazy_import_that_still_exists() -> None:
    for module_name, target in sorted(CONTROL_PLANE_ONLY_LAZY_IMPORTS):
        source = (REPO_PACKAGE / module_name).read_text(encoding="utf-8")
        assert target in source, (module_name, target)
        blockers = provider_runtime_import_closure_blockers(
            package_source_dir=REPO_PACKAGE,
            shipped_module_names=[module_name],
            exemptions=frozenset(),
        )
        assert any(
            blocker.endswith(f"{module_name}->{target}:lazy") for blocker in blockers
        ), (module_name, target, blockers)


def test_validator_flags_unshipped_module_level_lazy_and_subpackage_imports(
    tmp_path: Path,
) -> None:
    package = tmp_path / "blueprint_pipeline"
    (package / "core").mkdir(parents=True)
    (package / "core" / "__init__.py").write_text("", encoding="utf-8")
    (package / "shipped.py").write_text(
        textwrap.dedent(
            """
            from .present import value
            from .absent_module import other

            def lazy():
                from .absent_lazy import thing
                from blueprint_pipeline.absent_absolute import more
                return thing, more

            def uses_subpackage():
                from .core import common
                return common
            """
        ),
        encoding="utf-8",
    )
    (package / "present.py").write_text("value = 1\n", encoding="utf-8")
    (package / "exempt_module_level.py").write_text(
        "from .exempted import x\n", encoding="utf-8"
    )

    blockers = provider_runtime_import_closure_blockers(
        package_source_dir=package,
        shipped_module_names=["shipped.py", "present.py", "exempt_module_level.py", "gone.py"],
        exemptions=frozenset({("exempt_module_level.py", "exempted")}),
    )

    assert blockers == [
        "provider_runtime_import_exemption_not_lazy:exempt_module_level.py->exempted",
        "provider_runtime_import_of_unshipped_subpackage:shipped.py->core",
        "provider_runtime_import_unshipped:shipped.py->absent_absolute:lazy",
        "provider_runtime_import_unshipped:shipped.py->absent_lazy:lazy",
        "provider_runtime_import_unshipped:shipped.py->absent_module",
        "provider_runtime_module_missing:gone.py",
    ]
    with pytest.raises(ValueError, match="^fixture_code:provider_runtime_import"):
        assert_provider_runtime_import_closure(
            package_source_dir=package,
            shipped_module_names=["shipped.py"],
            code="fixture_code",
        )
