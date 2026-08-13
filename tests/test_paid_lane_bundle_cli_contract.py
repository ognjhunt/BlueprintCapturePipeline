"""A lane that cannot be rebuilt from a command line cannot be launched twice.

The allocator refuses a bundle whose `blueprint_commit` is not the commit the
control plane is running, and every deploy moves that commit. So a lane whose
bundle exists only because someone once called a Python function is launchable
exactly until the next deploy, and then never again.

`adp_gaussian_excision_vast` was in that state: a complete
`build_gaussian_excision_vast_bundle`, no `main()`, and a bundle pinned to a
commit from days earlier. It read as "ready" everywhere and could not be run.

This rediscovers the set from source rather than listing it, so the next lane
added cannot be omitted from a list instead of given an entry point.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src" / "blueprint_pipeline"
SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"

#: Lanes whose bundle is built by a dedicated script rather than by the lane
#: module itself. Named, because "there is a script somewhere" is exactly the
#: assumption that let a lane sit unbuildable.
BUNDLE_BUILT_BY_SCRIPT = {
    "adp_retained_scene_render_vast.py": "build_retained_scene_render_bundle.py",
    "public_scene_simready_isaac_vast.py": "public_scene_simready_isaac_bundle.py",
}


def _paid_lane_modules() -> list[Path]:
    return [
        path
        for path in sorted(SOURCE_ROOT.glob("*_vast.py"))
        if "paid_resource_admission_grant" in path.read_text(encoding="utf-8")
    ]


def _builds_a_bundle(source: str) -> bool:
    """Does this module define the function that seals a provider bundle?"""

    tree = ast.parse(source)
    return any(
        isinstance(node, ast.FunctionDef)
        and node.name.startswith("build_")
        and node.name.endswith(("_bundle", "_vast_bundle"))
        for node in ast.walk(tree)
    )


def _has_entrypoint(source: str) -> bool:
    tree = ast.parse(source)
    has_main = any(
        isinstance(node, ast.FunctionDef) and node.name == "main"
        for node in ast.walk(tree)
    )
    return has_main and "__main__" in source


def test_there_are_paid_lanes_to_check() -> None:
    """A discovery that matched nothing would make the check below vacuous."""

    assert len(_paid_lane_modules()) >= 10


@pytest.mark.parametrize("path", _paid_lane_modules(), ids=lambda p: p.stem)
def test_a_lane_that_seals_a_bundle_can_be_run_from_a_command_line(
    path: Path,
) -> None:
    source = path.read_text(encoding="utf-8")
    if not _builds_a_bundle(source):
        pytest.skip("this lane's bundle is sealed elsewhere")
    if path.name in BUNDLE_BUILT_BY_SCRIPT:
        script = SCRIPTS / BUNDLE_BUILT_BY_SCRIPT[path.name]
        alt = SOURCE_ROOT / BUNDLE_BUILT_BY_SCRIPT[path.name]
        assert script.is_file() or alt.is_file(), (
            f"{path.name} is recorded as built by "
            f"{BUNDLE_BUILT_BY_SCRIPT[path.name]}, which does not exist"
        )
        return
    assert _has_entrypoint(source), (
        f"{path.name} seals a provider bundle with no `main()` entry point. The "
        "allocator refuses a bundle built at another commit and every deploy "
        "moves the commit, so this lane is launchable exactly once and then "
        "never again. Give it a CLI, or record it in BUNDLE_BUILT_BY_SCRIPT."
    )


def test_the_gaussian_excision_lane_exposes_both_of_its_steps() -> None:
    """The bundle needs a sealed wheelhouse, and sealing one is its own step."""

    from blueprint_pipeline import adp_gaussian_excision_vast as lane

    assert callable(lane.main)
    assert callable(lane.materialize_gaussian_excision_dependency_wheelhouse)
    assert callable(lane.build_gaussian_excision_vast_bundle)
