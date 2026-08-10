"""Check every Arena call the worker makes against Arena's real source.

Five launches, five defects, each invisible until the one before it was fixed:
the runtime/native split, the sibling assets directory, ``embodiments=`` and
``get_env()``, then ``enable_cameras``. Every one was a property of an API this
repository can read, and every one cost a GPU boot and an Arena provision to
discover.

The stub in ``test_scene_worker_composition`` cannot catch these, and rt14
proved why: I wrote the stub from the same assumption as the worker, so it
accepted keywords Arena has never had. A fake is only ever as right as its
author.

So this reads the real thing. IsaacLab-Arena is a public repository pinned at
``ARENA_REVISION``; cloned locally it answers every signature question without
a GPU, a container, or a dollar. The test walks every Arena symbol the worker
constructs, resolves it in that source, and reports **all** mismatches at once
rather than the first - which is the difference between one more launch and a
list.

Skipped when the source is not present, with the command to fetch it. A skip
here means unverified, not verified.
"""

from __future__ import annotations

import ast
import os
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKER = REPO_ROOT / "scripts/run_adp009d_articulated_scene_worker.py"
ARENA_REVISION = "8b4a3a47fc53de23e8205089d71109a2e2348acd"
ARENA_SOURCE_ENV = "BLUEPRINT_ARENA_SOURCE_DIR"
DEFAULT_ARENA_SOURCE = Path(f"/private/tmp/arena-source-{ARENA_REVISION[:8]}")
CLONE_HINT = (
    f"git clone --filter=blob:none --no-checkout "
    f"https://github.com/isaac-sim/IsaacLab-Arena.git {DEFAULT_ARENA_SOURCE} "
    f"&& git -C {DEFAULT_ARENA_SOURCE} checkout {ARENA_REVISION}"
)

# Arena symbol -> module path inside the Arena source tree.
ARENA_SYMBOLS = {
    "IsaacLabArenaEnvironment": "isaaclab_arena/environments/isaaclab_arena_environment.py",
    "ArenaEnvBuilder": "isaaclab_arena/environments/arena_env_builder.py",
    "Object": "isaaclab_arena/assets/object.py",
    "Pose": "isaaclab_arena/utils/pose.py",
    "Scene": "isaaclab_arena/scene/scene.py",
    "DroidAbsoluteJointPositionEmbodiment": "isaaclab_arena/embodiments/droid/droid.py",
}
OBJECT_TYPE_SOURCE = "isaaclab_arena/assets/object_base.py"


def _arena_source() -> Path:
    configured = os.environ.get(ARENA_SOURCE_ENV)
    root = Path(configured) if configured else DEFAULT_ARENA_SOURCE
    if not (root / "isaaclab_arena").is_dir():
        pytest.skip(
            f"Arena source not present at {root}; this check is UNVERIFIED. "
            f"Fetch it with: {CLONE_HINT}"
        )
    return root


def _class_node(path: Path, name: str) -> ast.ClassDef:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise AssertionError(f"{name} not found in {path}")


def _accepted_parameters(node: ast.ClassDef) -> tuple[set[str], set[str], bool]:
    """(accepted names, required names, accepts arbitrary keywords).

    A dataclass has no ``__init__`` in the source - its parameters are the
    annotated fields, and treating "no __init__" as "accepts nothing" would
    flag every correct call to Pose.
    """

    for item in node.body:
        if isinstance(item, ast.FunctionDef) and item.name == "__init__":
            positional = [a.arg for a in item.args.args if a.arg != "self"]
            keyword_only = [a.arg for a in item.args.kwonlyargs]
            default_count = len(item.args.defaults)
            required = set(positional[: len(positional) - default_count])
            required |= {
                a.arg
                for a, d in zip(item.args.kwonlyargs, item.args.kw_defaults)
                if d is None
            }
            return (
                set(positional) | set(keyword_only),
                required,
                item.args.kwarg is not None,
            )
    fields = {n.target.id for n in node.body if isinstance(n, ast.AnnAssign)}
    required = {
        n.target.id for n in node.body if isinstance(n, ast.AnnAssign) and n.value is None
    }
    return fields, required, False


def _worker_calls(callee: str) -> list[set[str]]:
    tree = ast.parse(WORKER.read_text(encoding="utf-8"))
    calls: list[set[str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
        if name != callee:
            continue
        # A **spread means the call cannot be checked statically; recording it
        # as an empty set would read as "checked and clean".
        if any(keyword.arg is None for keyword in node.keywords):
            continue
        calls.append({keyword.arg for keyword in node.keywords if keyword.arg})
    return calls


def test_every_arena_keyword_the_worker_passes_exists_in_arena():
    """One list of every mismatch, not the first one a GPU happens to hit."""

    root = _arena_source()
    violations: list[str] = []

    for symbol, relative in ARENA_SYMBOLS.items():
        path = root / relative
        if not path.is_file():
            violations.append(f"{symbol}: source file absent at {relative}")
            continue
        accepted, _required, arbitrary = _accepted_parameters(_class_node(path, symbol))
        if arbitrary:
            continue
        for passed in _worker_calls(symbol):
            unknown = passed - accepted
            if unknown:
                violations.append(
                    f"{symbol}: worker passes {sorted(unknown)}; "
                    f"Arena accepts {sorted(accepted)}"
                )

    assert not violations, "Arena API mismatches:\n  " + "\n  ".join(violations)


def test_every_required_arena_parameter_is_supplied():
    root = _arena_source()
    violations: list[str] = []

    for symbol, relative in ARENA_SYMBOLS.items():
        path = root / relative
        if not path.is_file():
            continue
        accepted, required, _arbitrary = _accepted_parameters(_class_node(path, symbol))
        for passed in _worker_calls(symbol):
            # Positional arguments are invisible to this check, so only flag a
            # requirement the worker supplies nowhere at all.
            missing = required - passed
            if missing and passed:
                violations.append(f"{symbol}: never supplies {sorted(missing)}")

    assert not violations, "Missing required Arena parameters:\n  " + "\n  ".join(
        violations
    )


def test_object_types_the_worker_names_exist_in_arena():
    """A renamed enum member is an AttributeError four minutes into a run."""

    root = _arena_source()
    node = _class_node(root / OBJECT_TYPE_SOURCE, "ObjectType")
    members = {
        target.targets[0].id
        for target in node.body
        if isinstance(target, ast.Assign) and isinstance(target.targets[0], ast.Name)
    }
    source = WORKER.read_text(encoding="utf-8")

    named = {
        line.split("ObjectType.")[1].split(")")[0].split(",")[0].split(" ")[0].strip()
        for line in source.splitlines()
        if "ObjectType." in line
    }

    unknown = {name for name in named if name and name not in members}
    assert not unknown, f"worker names {sorted(unknown)}; Arena has {sorted(members)}"


def test_contact_sensor_prim_paths_match_the_embodiment_source():
    """A sensor pattern that matches nothing raises IndexError, not a message.

    Isaac Lab's ContactSensor does ``self._parent_prims[0]`` with no emptiness
    check, so a wrong prim path surfaces as a bare "list index out of range"
    from inside a physics callback - two launches, rt16 and rt17.

    The trap is that DroidSceneCfg declares the scene *key* as ``robot`` and
    the *prim* as ``{ENV_REGEX_NS}/Robot``. They differ only in case.
    """

    root = _arena_source()
    droid = (root / "isaaclab_arena/embodiments/droid/droid.py").read_text(
        encoding="utf-8"
    )
    worker = WORKER.read_text(encoding="utf-8")

    declared = {
        line.split('prim_path="')[1].split('"')[0]
        for line in droid.splitlines()
        if 'prim_path="{ENV_REGEX_NS}' in line
    }
    robot_roots = {path for path in declared if path.count("/") == 1}
    assert robot_roots, "no robot prim root declared by the embodiment"

    used = {
        line.split('prim_path="')[1].split('"')[0]
        for line in worker.splitlines()
        if 'prim_path="{ENV_REGEX_NS}' in line
    }
    robot_sensor = {path for path in used if path.lower().startswith("{env_regex_ns}/robot")}
    assert robot_sensor, "worker declares no robot contact sensor"

    for path in robot_sensor:
        stem = path.rsplit("/", 1)[0]
        assert stem in robot_roots, (
            f"worker uses {path!r}; embodiment declares roots {sorted(robot_roots)} "
            "(these differ by case, which matches nothing and raises IndexError)"
        )


def test_spawn_cfg_addon_never_duplicates_what_arena_already_passes():
    """spawn_cfg_addon is splatted into the same UsdFileCfg call.

    It is not additive. Arena's _generate_articulation_cfg and
    _generate_rigid_cfg already pass activate_contact_sensors; repeating it
    raises "got multiple values for keyword argument".

    _generate_base_cfg omits it, and I first read that as an oversight to
    correct. It is not: a BASE asset is static geometry with no rigid bodies,
    and Isaac refuses with "no rigid bodies are present under this prim". Arena
    is right and I was wrong, so BASE must not receive it either.

    The safe set is per object type and read from Arena's source rather than
    remembered.
    """

    root = _arena_source()
    src = (root / "isaaclab_arena/assets/object.py").read_text(encoding="utf-8")
    tree = ast.parse(src)

    generators = {
        "_generate_articulation_cfg": "ARTICULATION",
        "_generate_rigid_cfg": "RIGID",
        "_generate_base_cfg": "BASE",
    }
    already: dict[str, set[str]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or node.name not in generators:
            continue
        keywords: set[str] = set()
        for inner in ast.walk(node):
            if isinstance(inner, ast.Call) and getattr(inner.func, "id", "") == "UsdFileCfg":
                keywords |= {k.arg for k in inner.keywords if k.arg}
        already[generators[node.name]] = keywords

    assert already.get("ARTICULATION"), "articulation generator not found"

    import importlib.util

    spec = importlib.util.spec_from_file_location("scene_worker_addon", WORKER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    violations = []
    for object_type, arena_keywords in already.items():
        addon = set(module._spawn_cfg_addon(object_type, {"visible": True}))
        clash = addon & arena_keywords
        if clash:
            violations.append(f"{object_type}: duplicates {sorted(clash)}")

    assert not violations, "spawn_cfg_addon collisions:\n  " + "\n  ".join(violations)
    # BASE is static: no rigid bodies, so contact sensors cannot attach at all.
    assert "activate_contact_sensors" not in module._spawn_cfg_addon("BASE", {})
