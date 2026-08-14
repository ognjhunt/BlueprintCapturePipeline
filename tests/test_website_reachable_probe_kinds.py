"""What the pipeline can execute, versus what the website can trigger.

The allocator dispatches on probe kind. A probe kind it handles but no live
profile builder emits is work that exists, runs, and cannot be reached from the
product path -- which is the exact shape of every reachability defect this
program has hit (#512, #517, #519, #520, #523). Counting lane modules missed
it, because a `*_vast.py` is transport and several probe kinds have no lane
module of their own.

So the denominator is taken from the allocator itself. Every probe kind it
dispatches on must either be emitted by a builder or be named below with a
reason. Adding a branch to the allocator without doing one or the other fails
here, rather than quietly becoming the next unreachable lane.

This is a ledger of decisions, not a suppression list. `awaiting_builder` is an
admission that the work is unreachable, not an exemption from fixing it.
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
ALLOCATOR = REPO_ROOT / "src" / "blueprint_pipeline" / "paid_resource_allocator.py"

#: Probe kinds with no live profile builder, and why. Reasons are load-bearing:
#: they say whether the gap is a decision or a debt.
NOT_WEBSITE_REACHABLE: dict[str, str] = {
    # Superseded. The appearance path is now the whole-frame GPT teacher paired
    # with original outside-mask anchors, then ArtiFixer3D, then the native
    # import gate. These are retained for their receipts, which still anchor
    # the campaign's spend, and are not to be launched again.
    "adp-aurafusion360-author-smoke": "retired_appearance_approach",
    "adp-aurafusion360-exact-residual": "retired_appearance_approach",
    "adp-aurafusion360-interiorgs": "retired_appearance_approach",
    "adp-inpaint360-interiorgs": "retired_appearance_approach",
    "adp-simpler-public-reference": "retired_appearance_approach",
    "adp009d-aura-native-live-camera": "retired_appearance_approach",
    "adp009d-aura-ovrtx-live-camera": "retired_appearance_approach",
    # Frozen by the active program contract in CLAUDE.md: Arm Decision Proof v1
    # is the sole active program, and policy-ranking, world-model, and
    # post-training work is explicitly frozen.
    "openpi-policy-ranking": "frozen_program",
    "persistent-policy-wam-loop": "frozen_program",
    "policy-ranking-cosmos-reasoner": "frozen_program",
    "policy-ranking-successor-cosmos": "frozen_program",
    "single-kitchen-episode": "frozen_program",
    "single-kitchen-finetune": "frozen_program",
    "single-kitchen-qualification": "frozen_program",
    # Not a lane. The control plane runs this against a profile before it will
    # allocate anything, so it is reached by the allocator rather than by a
    # profile of its own.
    "task-evaluation-profile-preflight": "not_a_website_lane",
    # Real debt -- executable, not retired, not frozen, and unreachable -- is
    # empty as of this lane's builder. Every probe kind the allocator can run
    # is now either reachable from the website or a recorded decision. A new
    # `awaiting_builder` row is therefore a regression, not routine bookkeeping.
}

VALID_REASONS = {
    "retired_appearance_approach",
    "frozen_program",
    "not_a_website_lane",
    "awaiting_builder",
}


def _dispatched_probe_kinds() -> dict[str, str]:
    """Every probe kind the allocator branches on, read from its own source."""

    tree = ast.parse(ALLOCATOR.read_text(encoding="utf-8"))
    constants: set[str] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Compare)
            and isinstance(node.left, ast.Attribute)
            and node.left.attr == "probe_kind"
        ):
            for comparator in node.comparators:
                elements = (
                    comparator.elts
                    if isinstance(comparator, (ast.Set, ast.Tuple, ast.List))
                    else [comparator]
                )
                constants.update(e.id for e in elements if isinstance(e, ast.Name))

    module = importlib.import_module("blueprint_pipeline.paid_resource_allocator")
    resolved: dict[str, str] = {}
    for name in sorted(constants):
        value = getattr(module, name, None)
        if isinstance(value, str):
            resolved[value] = name
    return resolved


def _builder_probe_kinds() -> dict[str, str]:
    """Every probe kind a live profile builder actually emits."""

    emitted: dict[str, str] = {}
    for path in sorted(SCRIPTS.glob("build_*_live_profile.py")):
        spec = importlib.util.spec_from_file_location(path.stem, path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        sys.modules[path.stem] = module
        spec.loader.exec_module(module)
        found: set[str] = set()
        for attribute in vars(module).values():
            # A single-lane builder declares one SPEC. A chain builder declares
            # several links, and reading only SPEC would report its whole
            # family as unreachable -- which it did, silently, until the
            # Arena builder landed and this gate stayed green.
            candidates = (
                attribute.values()
                if isinstance(attribute, dict)
                else attribute
                if isinstance(attribute, (list, tuple, set))
                else [attribute]
            )
            for candidate in candidates:
                kind = getattr(candidate, "probe_kind", None)
                if isinstance(kind, str):
                    found.add(kind)
        if isinstance(getattr(module, "PROBE_KIND", None), str):
            found.add(module.PROBE_KIND)
        if not found:
            # The two builders that predate the shared skeleton pass the kind
            # as an argv literal rather than declaring it.
            source = path.read_text(encoding="utf-8")
            found = {k for k in _dispatched_probe_kinds() if f'"{k}"' in source}
        for kind in found:
            emitted[kind] = path.stem
    return emitted


DISPATCHED = _dispatched_probe_kinds()
EMITTED = _builder_probe_kinds()


def test_the_allocator_dispatch_table_is_discoverable() -> None:
    assert len(DISPATCHED) >= 25, "the extraction stopped seeing the dispatch table"


def test_at_least_one_builder_was_loaded() -> None:
    assert EMITTED, "no builder emitted a probe kind; the extraction is broken"


@pytest.mark.parametrize("kind", sorted(DISPATCHED), ids=str)
def test_every_executable_probe_kind_is_reachable_or_named(kind: str) -> None:
    """The gate that makes an unreachable lane impossible to add silently."""

    if kind in EMITTED:
        assert kind not in NOT_WEBSITE_REACHABLE, (
            f"{kind} is emitted by {EMITTED[kind]}; remove it from "
            "NOT_WEBSITE_REACHABLE rather than leaving a stale reason."
        )
        return

    reason = NOT_WEBSITE_REACHABLE.get(kind)
    assert reason, (
        f"The allocator can execute {kind!r} and no live profile builder emits "
        "it, so it cannot be triggered from the website. Add a builder, or add "
        f"it to NOT_WEBSITE_REACHABLE with one of {sorted(VALID_REASONS)}."
    )
    assert reason in VALID_REASONS, f"{kind} carries an unrecognized reason {reason!r}"


def test_no_named_gap_outlives_the_branch_it_describes() -> None:
    """A reason for a probe kind the allocator no longer runs is stale."""

    stale = sorted(set(NOT_WEBSITE_REACHABLE) - set(DISPATCHED))

    assert not stale, f"NOT_WEBSITE_REACHABLE names probe kinds nothing dispatches: {stale}"


def test_no_builder_emits_a_probe_kind_the_allocator_will_refuse() -> None:
    """A profile the allocator cannot dispatch fails after it is published."""

    orphaned = sorted(set(EMITTED) - set(DISPATCHED))

    assert not orphaned, f"builders emit undispatchable probe kinds: {orphaned}"


def test_the_retired_appearance_lanes_are_not_quietly_relaunched() -> None:
    """Retiring these was a direction, and a builder would undo it silently."""

    retired = {
        kind
        for kind, reason in NOT_WEBSITE_REACHABLE.items()
        if reason == "retired_appearance_approach"
    }

    assert retired, "the retirement decision has been lost from this ledger"
    assert not (retired & set(EMITTED))


def test_the_reachability_debt_is_stated_rather_than_implied() -> None:
    """`awaiting_builder` is an admission; this keeps its size visible."""

    debt = sorted(
        kind
        for kind, reason in NOT_WEBSITE_REACHABLE.items()
        if reason == "awaiting_builder"
    )

    assert not debt, (
        f"unreachable executable probe kinds grew to {len(debt)}: {debt}. "
        "The debt is empty: every probe kind the allocator can execute is "
        "either reachable from the website or a recorded decision. Add the "
        "builder rather than a row here; do not reopen the ledger to pass."
    )
