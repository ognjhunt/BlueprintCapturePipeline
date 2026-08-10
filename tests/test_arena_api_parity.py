"""Pin the articulated worker's Arena API against the rigid lane's call site.

rt14 died on ``IsaacLabArenaEnvironment.__init__() got an unexpected keyword
argument 'embodiments'``. The worker had been written with ``embodiments=[...]``
and ``env_config_modifier=``, neither of which is Arena's API - and the stubbed
Arena in the composition tests accepted both, because I wrote the stub from the
same assumption as the worker. A fake built from a guess validates the guess.

So this test does not consult the fake at all. It parses the two call sites out
of the source and compares them. ``adp009d_isaac_runtime.py`` is the rigid lane
that has run on hardware many times, which makes it the authority on what these
constructors accept; if the articulated worker calls them with keywords the
rigid lane never uses, that is the bug this catches - on a laptop, not six
minutes into a paid run.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
RIGID_RUNTIME = REPO_ROOT / "src/blueprint_pipeline/adp009d_isaac_runtime.py"
ARTICULATED_WORKER = REPO_ROOT / "scripts/run_adp009d_articulated_scene_worker.py"
# Constructors whose signatures come from Arena, not from us.
ARENA_CONSTRUCTORS = ("IsaacLabArenaEnvironment", "ArenaEnvBuilder")


def _calls(path: Path, callee: str) -> tuple[int, set[str]]:
    """How many times `callee` is constructed, and with which keywords.

    Count and keywords are separate because a constructor called purely
    positionally has no keywords, and an empty set must not read as "never
    called" - that would make the parity anchor silently vacuous.
    """

    tree = ast.parse(path.read_text(encoding="utf-8"))
    count = 0
    found: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = getattr(func, "id", None) or getattr(func, "attr", None)
        if name != callee:
            continue
        count += 1
        found |= {kw.arg for kw in node.keywords if kw.arg}
    return count, found


@pytest.mark.parametrize("callee", ARENA_CONSTRUCTORS)
def test_worker_uses_only_keywords_the_rigid_lane_uses(callee: str):
    rigid_count, rigid = _calls(RIGID_RUNTIME, callee)
    articulated_count, articulated = _calls(ARTICULATED_WORKER, callee)

    assert rigid_count, f"{callee} not constructed in the rigid runtime; unanchored"
    assert articulated_count, f"{callee} not constructed in the articulated worker"

    invented = articulated - rigid
    assert not invented, (
        f"{callee} called with keywords the hardware-proven lane never uses: "
        f"{sorted(invented)}; rigid lane uses {sorted(rigid)}"
    )


def test_the_environment_is_built_through_the_builder_not_a_get_env_helper():
    """``get_env()`` was invented too; the real path goes through the builder."""

    source = ARTICULATED_WORKER.read_text(encoding="utf-8")

    assert "make_registered_and_return_cfg" in source
    assert ".get_env()" not in source
