"""Every live profile builder goes through the one skeleton, or it is a copy.

Three lanes could be launched from the website and eleven could not, and the
obvious next step was eleven more builders. The two that existed were 319 and
281 lines sharing 184 identical lines, so eleven more would have been eleven
more chances to omit the residency check, the spend binding, or the terminal
contract -- and each of those is only discovered after a provider is rented.

This rediscovers the builders from `scripts/` rather than listing them, so a
builder added tomorrow is held to the same thing without anyone remembering to
add it here.
"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
SHARED_MODULE = "task_evaluation_live_profile"


def _builders() -> list[Path]:
    return sorted(SCRIPTS.glob("build_*live_profile.py"))


def _load(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_there_are_builders_to_check() -> None:
    """A glob that matches nothing would make every test below vacuous."""

    assert len(_builders()) >= 3


def _cli_arguments(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {
        node.args[0].value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and getattr(node.func, "attr", None) == "add_argument"
        and node.args
        and isinstance(node.args[0], ast.Constant)
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


@pytest.mark.parametrize("path", _builders(), ids=lambda p: p.stem)
def test_every_builder_takes_its_control_surface_from_one_definition(
    path: Path,
) -> None:
    """`required_controls` and `terminal_contract` decide if a run is provable.

    A lane that quietly dropped `provider_zero_required` or stopped asking for
    a teardown manifest would keep passing until the day it mattered. There is
    one definition of both, and every builder uses it.
    """

    source = path.read_text(encoding="utf-8")
    assert SHARED_MODULE in source, f"{path.name} does not import the shared module"
    assert "shared_control_surface" in source or "build_lane_live_profile" in source
    for owned in ('"required_controls"', '"terminal_contract"', '"webapp_sync"'):
        assert owned not in source, (
            f"{path.name} sets {owned} itself; `shared_control_surface` already "
            "does, and two copies drift"
        )


@pytest.mark.parametrize(
    "path", [p for p in _builders() if _is_receipt_driven(p)], ids=lambda p: p.stem
)
def test_every_receipt_driven_builder_uses_the_whole_skeleton(path: Path) -> None:
    """Residency, spend binding, and validation are not per-lane decisions."""

    source = path.read_text(encoding="utf-8")
    if "build_lane_live_profile" not in source:
        pytest.fail(
            f"{path.name} assembles a launch profile by hand. Declare a "
            "LaneLiveProfileSpec and call build_lane_live_profile instead -- the "
            "residency, spend, and terminal-contract checks live there."
        )
    # Resolving the receipt is the skeleton's job; doing it again here means
    # this builder decided to resolve it differently.
    assert "resolve_host_resident_bundle_receipt" not in source


@pytest.mark.parametrize("path", _builders(), ids=lambda p: p.stem)
def test_every_builder_offers_a_revision_argument(path: Path) -> None:
    """Published profiles are immutable, so a rebuild needs its own id.

    Reusing one id for two different input sets at a single commit surfaces as
    `launch_profile_immutable_input_digest_mismatch` on the next launch, not at
    publish time.
    """

    assert "--revision" in _cli_arguments(path), (
        f"{path.name} cannot distinguish a rebuild"
    )


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
