"""Every receipt-driven live profile builder declares a TTL band the allocator accepts.

The builders are rediscovered from `scripts/` rather than listed, so a builder
added tomorrow is held to the same band without anyone remembering to add it.
"""

from __future__ import annotations

import ast
import importlib.util
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"


def _builders() -> list[Path]:
    return sorted(SCRIPTS.glob("build_*live_profile.py"))


def _load(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    # Registered before execution: `@dataclass` resolves its field annotations
    # through `sys.modules[cls.__module__]`, so a builder that declares one
    # raises here unless the module it is being defined in can be found. That
    # made this contract's reach depend on whether some earlier test happened
    # to have imported the same builder first.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_there_are_builders_to_check() -> None:
    """A glob that matches nothing would make every test below vacuous."""

    assert len(_builders()) >= 3


def _cli_arguments(path: Path) -> set[str]:
    """Every flag this builder's command line offers, however it is declared.

    Four shapes are in use, and reading only some of them is not merely a wrong
    answer here -- it is a silent exemption. `_is_receipt_driven` is decided
    from this set, so a builder whose flags this misses drops out of the
    whole-skeleton and TTL-band contracts below without failing anything.

    * `parser.add_argument("--flag", ...)`, the common style;
    * a row naming its own flag in a `"flag"` field, as
      `prepare_artifixer3d_inputs` does: `{"revision": {"flag": "--revision"}}`;
    * a table keyed by the flag itself: `FLAGS = {"--revision": Flag(...)}`;
    * a row handing its flag first to a constructor:
      `PARAMS = {"revision": Param("--revision", ...)}`.

    The last three are one idea spelled three ways -- a single table that
    builds both the parser and the call, so a parameter cannot quietly lose its
    flag and be fixed at a default. Three lanes arrived at it independently and
    each reader saw only its own spelling, which is why all four are read here
    rather than any one of them.

    All are read from the syntax rather than the source text, so a docstring
    that happens to name a flag cannot satisfy a contract.
    """

    tree = ast.parse(path.read_text(encoding="utf-8"))
    declared: set[Any] = {
        node.args[0].value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and getattr(node.func, "attr", None) == "add_argument"
        and node.args
        and isinstance(node.args[0], ast.Constant)
    }
    declared.update(
        value.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Dict)
        for key, value in zip(node.keys, node.values)
        if isinstance(key, ast.Constant)
        and key.value == "flag"
        and isinstance(value, ast.Constant)
    )
    declared.update(
        key.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Dict)
        for key in node.keys
        if isinstance(key, ast.Constant)
        and isinstance(key.value, str)
        and key.value.startswith("--")
    )
    declared.update(
        node.args[0].value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and isinstance(node.args[0].value, str)
        and node.args[0].value.startswith("--")
    )
    return {item for item in declared if isinstance(item, str)}


def test_the_flag_reader_sees_every_declaration_style_in_use(tmp_path: Path) -> None:
    """The reader decides which contract every builder below is held to.

    Each shape below is a real builder's command line, and three lanes added
    three of them in the same week -- each lane's reader seeing only its own.
    Narrowing back to any one of them silently exempts the other lanes from the
    contracts keyed on `--bundle-receipt` and `--revision` rather than failing
    them, which is the failure this whole function exists to prevent. Widening
    to plain source text would let a docstring satisfy a contract, so that is
    pinned here too.
    """

    module = tmp_path / "build_probe_live_profile.py"
    module.write_text(
        textwrap.dedent(
            '''
            """A docstring naming --not-a-flag must not count."""
            PARAMETERS = {"revision": {"flag": "--revision"}}
            FLAGS = {"--bundle-receipt": Flag("bundle_receipt")}
            PARAMS = {"release": Param("--release-evidence", required=True)}
            parser.add_argument("--output", required=True)
            '''
        ),
        encoding="utf-8",
    )

    assert _cli_arguments(module) == {
        "--revision",
        "--bundle-receipt",
        "--release-evidence",
        "--output",
    }


def _is_receipt_driven(path: Path) -> bool:
    """Does this lane's profile come from a bundle receipt?

    Most do: a receipt names the archive, and resolving it against *this* host
    is what keeps an authoring path out of allocator argv. One lane
    (`adp009d_840313`) is driven by a preflight request and release evidence
    instead, so the receipt-residency half of the skeleton has nothing to act
    on there. It still shares the control surface, which is the half that
    decides whether a run is provable.

    Read from the CLI rather than by matching source text, so a comment that
    happens to mention receipts does not change which contract a lane is held
    to.
    """

    return "--bundle-receipt" in _cli_arguments(path)


@pytest.mark.parametrize(
    "path", [p for p in _builders() if _is_receipt_driven(p)], ids=lambda p: p.stem
)
def test_every_lane_spec_declares_a_ttl_band_the_allocator_agrees_with(
    path: Path,
) -> None:
    """A TTL outside the band is refused by the allocator, not by the builder.

    Every lane's allocator branch has its own band, so a spec that leaves it
    open would push the refusal past the paid boundary.
    """

    module = _load(path)
    specs = [
        value
        for value in vars(module).values()
        if type(value).__name__ == "LaneLiveProfileSpec"
    ]
    if not specs:
        # Lanes whose spec is built per candidate expose a factory instead.
        factories = [
            value
            for name, value in vars(module).items()
            if name == "_spec" and callable(value)
        ]
        assert factories, f"{path.name} exposes no LaneLiveProfileSpec"
        specs = [factories[0]("contract-probe")]
    for spec in specs:
        assert 0 < spec.min_ttl_seconds < spec.max_ttl_seconds
        assert spec.probe_kind, f"{path.name} declares no probe kind"
        assert spec.extra_path_names is not None
