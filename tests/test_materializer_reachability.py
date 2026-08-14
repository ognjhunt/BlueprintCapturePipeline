"""A function nothing can call is not a production path.

`materialize_*` functions produce the receipts every paid lane consumes. When
one is reachable from no script and from no module carrying a `main()`, the
work it does exists only for whoever opens a Python session -- and the lane
downstream of it dead-ends with a message about a missing input rather than a
message about a missing entry point.

That is how the appearance chain stalled. The ArtiFixer3D bundle CLI asks for a
candidate-inputs receipt; the four functions that produce one were reachable
from nothing; so the head of the chain could not be built at all, and the
symptom looked like an absent file.

The same defect has now been fixed in four scopes -- lanes (#512), bundle
modules (#520), authority materializers (#523), and input materializers here --
which is why this counts the whole population from source instead of listing
the ones anybody happened to notice: 73 of 161 are still unreachable.

`UNREACHABLE_MATERIALIZERS` is a ledger of debt, not a suppression list. The
bound below only ratchets down.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "src" / "blueprint_pipeline"
SCRIPTS = REPO_ROOT / "scripts"

#: 78 before the ArtiFixer3D input chain got an entry point, 74 after, and
#: 73 once the segment-mask-repair preflight -- the chain's root -- got one.
#: It may fall and never rise.
#:
#: It went to 76 once, when the semantic-teacher image-edit lane landed with all
#: three of its terminal materializers -- result, provider zero, and the
#: no-allocation closeout -- reachable from nothing. That is the shape this
#: ratchet exists to catch: the lane could be started from a production path and
#: only closed from a Python session, so a paid attempt could end with no
#: terminal artifact and a provider bill nobody reconciled. Restored to 73 by
#: `scripts/retain_semantic_teacher_image_edit_receipts.py`.
#:
#: Naming all 73 individually would be a list nobody maintains; the population
#: is rediscovered every run and only its size is pinned, so a new unreachable
#: materializer fails here even though no name was ever written down.
UNREACHABLE_MATERIALIZER_BUDGET = 73


def _module_sources() -> dict[Path, str]:
    return {path: path.read_text(encoding="utf-8") for path in sorted(SOURCE_ROOT.glob("*.py"))}


def _script_sources() -> list[str]:
    return [path.read_text(encoding="utf-8") for path in sorted(SCRIPTS.glob("*.py"))]


def _unreachable() -> list[tuple[str, str]]:
    """Every public materializer no command line can reach, by shape."""

    modules = _module_sources()
    scripts = _script_sources()
    found: list[tuple[str, str]] = []
    for path, text in modules.items():
        tree = ast.parse(text)
        names = [
            node.name
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name.startswith("materialize_")
        ]
        if not names:
            continue
        has_main = any(
            isinstance(node, ast.FunctionDef) and node.name == "main" for node in tree.body
        )
        if has_main:
            continue
        for name in names:
            if any(name in script for script in scripts):
                continue
            # A sibling module with its own entry point can also expose it.
            if any(
                name in other and "def main(" in other
                for other_path, other in modules.items()
                if other_path != path
            ):
                continue
            found.append((path.stem, name))
    return found


def test_the_population_is_discoverable() -> None:
    """A scan that matched nothing would make the budget vacuous."""

    modules = _module_sources()
    total = sum(
        1
        for text in modules.values()
        for node in ast.parse(text).body
        if isinstance(node, ast.FunctionDef) and node.name.startswith("materialize_")
    )
    assert total >= 100, f"only {total} materializers found; the scan is broken"


def test_unreachable_materializers_only_ever_decrease() -> None:
    """The ratchet. Lower the budget when you fix some; never raise it."""

    unreachable = _unreachable()

    assert len(unreachable) <= UNREACHABLE_MATERIALIZER_BUDGET, (
        f"{len(unreachable)} public materializers are reachable from no command "
        f"line, above the budget of {UNREACHABLE_MATERIALIZER_BUDGET}. Give the "
        "new one an entry point rather than raising this number: a receipt "
        "producer nothing can call is a lane that dead-ends on a missing input.\n"
        + "\n".join(f"  {module}.{name}" for module, name in sorted(unreachable)[:12])
    )


def test_the_artifixer3d_input_chain_is_reachable() -> None:
    """The head of the appearance path, and the reason this contract exists.

    Pinned by name rather than by count: these four are what the ArtiFixer3D
    bundle's `--candidate-inputs-receipt` depends on, so losing their entry
    point again would re-block the chain at its first step.
    """

    unreachable = {name for _, name in _unreachable()}

    for required in (
        "materialize_object_absent_reference_candidate_receipt",
        "materialize_artifixer3d_candidate_inputs",
        "materialize_whole_frame_semantic_teacher_receipt",
        "materialize_dual_target_artifixer3d_inputs",
    ):
        assert required not in unreachable, f"{required} lost its command line"


@pytest.mark.parametrize(
    "step",
    ["object-absent-reference", "candidate-inputs", "semantic-teacher", "dual-target"],
)
def test_every_input_step_can_supply_its_materializer(step: str) -> None:
    """Same rule as the authority issuer: the flag table is the call."""

    import importlib.util
    import inspect
    import sys

    name = "prepare_artifixer3d_inputs"
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)

    entry = module.STEPS[step]
    upstream = {
        parameter
        for parameter, value in inspect.signature(entry.materialize).parameters.items()
        if value.kind is inspect.Parameter.KEYWORD_ONLY
    }

    assert not upstream - set(entry.params), (
        f"{step} cannot supply {sorted(upstream - set(entry.params))} from a "
        "command line"
    )
