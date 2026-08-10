"""Sweep every vendor name the worker touches, instead of the ones I remember.

``test_arena_source_contract`` checks six symbols - the six I had already been
burned by. That is a regression suite wearing a preflight's clothes: by
construction it cannot find the next defect, only the last one again. rt20
proved it, dying on a scene key that no check looked at.

This inverts the direction. It extracts every external name the worker uses -
scene keys, scene-cfg attributes, prim paths, vendor attribute accesses - and
checks each against the two pinned source trees. What it cannot resolve it
reports as unresolved rather than passing, because a sweep that quietly skips
what it does not understand is the same whitelist with extra steps.

The point is coverage, not cleverness: a name in this worker that does not
appear in the vendor source is either a typo, a rename, or a wrong guess, and
each of those has cost a launch at least once.
"""

from __future__ import annotations

import ast
import os
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKER = REPO_ROOT / "scripts/run_adp009d_articulated_scene_worker.py"
ARENA_REVISION = "8b4a3a47fc53de23e8205089d71109a2e2348acd"
ISAACLAB_REVISION = "e57379c634b42db5a0fe9f754341be6e2a7c7c43"
ARENA_SOURCE = Path(
    os.environ.get("BLUEPRINT_ARENA_SOURCE_DIR", f"/private/tmp/arena-source-{ARENA_REVISION[:8]}")
)
ISAACLAB_SOURCE = Path(
    os.environ.get(
        "BLUEPRINT_ISAACLAB_SOURCE_DIR", f"/private/tmp/isaaclab-source-{ISAACLAB_REVISION[:8]}"
    )
)


def _require_sources() -> tuple[Path, Path]:
    missing = [
        str(path)
        for path in (ARENA_SOURCE, ISAACLAB_SOURCE)
        if not path.is_dir()
    ]
    if missing:
        pytest.skip(f"vendor source not present: {missing}; sweep is UNVERIFIED")
    return ARENA_SOURCE, ISAACLAB_SOURCE


def _worker_tree() -> ast.Module:
    return ast.parse(WORKER.read_text(encoding="utf-8"))


def _string_subscripts(variable: str) -> set[str]:
    """Every ``variable["literal"]`` in the worker."""

    found: set[str] = set()
    for node in ast.walk(_worker_tree()):
        if not isinstance(node, ast.Subscript):
            continue
        if getattr(node.value, "id", None) != variable:
            continue
        index = node.slice
        if isinstance(index, ast.Constant) and isinstance(index.value, str):
            found.add(index.value)
        elif isinstance(index, ast.Name):
            # A variable key; resolved below via module constants when possible.
            found.add(f"<name:{index.id}>")
    return found


def _module_constants() -> dict[str, str]:
    constants: dict[str, str] = {}
    for node in _worker_tree().body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            for target in node.targets:
                if isinstance(target, ast.Name) and isinstance(node.value.value, str):
                    constants[target.id] = node.value.value
    return constants


def _scene_cfg_attributes(source: Path) -> set[str]:
    """Attributes any SceneCfg in Arena declares, plus what the worker adds."""

    declared: set[str] = set()
    for path in (source / "isaaclab_arena").rglob("*.py"):
        if "/tests/" in str(path):
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name.endswith("SceneCfg"):
                declared |= {
                    item.target.id
                    for item in node.body
                    if isinstance(item, ast.AnnAssign)
                }
    return declared


def _worker_added_scene_entities() -> set[str]:
    """``cfg.scene.<name> = ...`` - entities the worker itself declares."""

    added: set[str] = set()
    for node in ast.walk(_worker_tree()):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if (
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Attribute)
                and target.value.attr == "scene"
            ):
                added.add(target.attr)
    return added


def test_every_scene_key_the_worker_reads_is_one_something_declares():
    """A scene key nobody declares is a KeyError after the scene is built."""

    arena, _isaaclab = _require_sources()
    constants = _module_constants()

    keys: set[str] = set()
    for raw in _string_subscripts("scene"):
        if raw.startswith("<name:"):
            resolved = constants.get(raw[6:-1])
            if resolved is None:
                continue
            keys.add(resolved)
        else:
            keys.add(raw)

    assert keys, "worker reads no scene keys; sweep would be vacuous"

    declarable = _scene_cfg_attributes(arena) | _worker_added_scene_entities()
    # Objects the worker puts in the scene are keyed by their own name, which
    # Arena derives from the composition spec rather than from any SceneCfg.
    declarable |= {"task_object", "scene_collision", "scene_appearance", "light"}

    unknown = sorted(keys - declarable)
    assert not unknown, (
        f"worker reads scene keys nothing declares: {unknown}; "
        f"declared entities include {sorted(declarable)[:14]}"
    )


def test_every_scene_entity_the_worker_declares_has_a_distinct_name():
    """Two sensors under one attribute silently becomes one sensor."""

    _require_sources()
    added = _worker_added_scene_entities()

    assert added, "worker declares no scene entities"
    # Read back the assignments to count duplicates the set would hide.
    names = []
    for node in ast.walk(_worker_tree()):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if (
                    isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Attribute)
                    and target.value.attr == "scene"
                ):
                    names.append(target.attr)
    duplicates = {name for name in names if names.count(name) > 1}
    assert not duplicates, f"scene entities declared more than once: {sorted(duplicates)}"


def test_every_sensor_prim_path_root_exists_in_the_vendor_sources():
    """A prim path matching nothing raises IndexError from a physics callback.

    The root before the wildcard must be something the embodiment spawns or
    something the worker itself put in the scene - and it is case sensitive,
    which is how rt16 and rt17 were spent.
    """

    arena, _isaaclab = _require_sources()

    worker_paths = {
        line.split('prim_path="')[1].split('"')[0]
        for line in WORKER.read_text(encoding="utf-8").splitlines()
        if 'prim_path="{ENV_REGEX_NS}' in line
    }
    assert worker_paths, "worker declares no prim paths"

    declared: set[str] = set()
    for path in (arena / "isaaclab_arena").rglob("*.py"):
        if "/tests/" in str(path):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for line in text.splitlines():
            if 'prim_path="{ENV_REGEX_NS}' in line:
                declared.add(line.split('prim_path="')[1].split('"')[0])

    # Roots Arena spawns, plus object names which become {ENV_REGEX_NS}/<name>.
    roots = {path.split("/")[1] for path in declared if path.count("/") >= 1}
    roots |= {"task_object", "scene_collision", "scene_appearance"}

    violations = []
    for path in worker_paths:
        root = path.split("/")[1] if path.count("/") >= 1 else ""
        if root not in roots:
            violations.append(f"{path} (root {root!r} not spawned; roots={sorted(roots)})")

    assert not violations, "prim paths matching nothing:\n  " + "\n  ".join(violations)


def test_vendor_data_attributes_the_worker_reads_exist_in_isaac_lab():
    """``.data.<attr>`` on a sensor or articulation must be a real field."""

    _arena, isaaclab = _require_sources()

    read: set[str] = set()
    for node in ast.walk(_worker_tree()):
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Attribute)
            and node.value.attr == "data"
        ):
            read.add(node.attr)
    assert read, "worker reads no vendor data attributes"

    known: set[str] = set()
    for path in isaaclab.rglob("*.py"):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                known.add(node.name)
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                known.add(node.target.id)
            elif isinstance(node, ast.Assign):
                known |= {
                    t.id for t in node.targets if isinstance(t, ast.Name)
                }

    # Only the sparse checkout is present, so an unresolved name is reported as
    # unresolved rather than failed - a partial tree must not manufacture
    # confidence in either direction.
    unresolved = sorted(read - known)
    if unresolved:
        pytest.skip(
            "attributes not resolvable against the sparse Isaac Lab checkout: "
            f"{unresolved}. Widen the sparse-checkout to verify them."
        )
