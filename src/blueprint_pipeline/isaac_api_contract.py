"""Check our calls into Isaac Lab and Arena against the sources production runs.

Production does not take Isaac Lab or Isaac Lab Arena from the Isaac Sim
image. It installs them from exact git objects (``ISAACLAB_COMMIT`` and
``ARENA_COMMIT`` in ``native_task_runtime_source_packet``) into
``NATIVE_TASK_ARENA_IMAGE``. Neither imports without Kit, so CI cannot load
them, and tests have used hand-written fakes instead. On 2026-11-03 the first
GPU run of the rigid robot-free probe crashed at launch on
``set_use_tiled_camera``: the fake rig defined it, Arena's ``ArenaCameraCfg``
does not.

``index`` reads the pinned sources once, with ``ast``, into a small API index
committed as a test fixture. ``check`` then reads our code, again without
importing anything, and flags:

* an Isaac Lab or Arena module or name our code imports, or names to
  ``import_module``, that the pinned source does not have;
* an attribute our code reads or writes on a module, class or object whose
  pinned class it can infer, when that class does not have it (a write to a
  missing config field is flagged too: a configclass keeps it and ignores it);
* a keyword our code passes to a pinned constructor, function or method that
  does not take it;
* a member a test fake declares that the pinned class it stands for lacks.

Types come from imports, constructor calls, annotations and field defaults in
the pinned source, ``type(...)``, ``configclass(...)``, ``deepcopy`` and
``replace``/``copy``. An object our own factories return is typed by
``RECEIVER_BINDINGS``. A type known only from an annotation also accepts the
members of the pinned subclasses, since the runtime value may be one. An
object it cannot type is not checked; the report counts what it did check.

    python -m blueprint_pipeline.isaac_api_contract check
    python -m blueprint_pipeline.isaac_api_contract index --fetch-into DIR
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .native_task_runtime_source_packet import (
    ARENA_COMMIT,
    ARENA_REPOSITORY,
    ARENA_TREE,
    ISAACLAB_COMMIT,
    ISAACLAB_PACKAGE_NAMES,
    ISAACLAB_REPOSITORY,
    ISAACLAB_TREE,
)


ROOT = Path(__file__).resolve().parents[2]
INDEX_SCHEMA_VERSION = "isaac_api_contract_index.v1"
FINDINGS_SCHEMA_VERSION = "isaac_api_contract_known_findings.v1"
INDEX_PATH = "tests/fixtures/isaac_api_contract/pinned_api.v1.json"
KNOWN_FINDINGS_PATH = "tests/fixtures/isaac_api_contract/known_findings.v1.json"
# The Arena packages the runtime source packet ships beside Isaac Lab's.
ARENA_PACKAGE_NAMES = ("isaaclab_arena", "isaaclab_arena_g1")
PINNED_PACKAGES = (*ISAACLAB_PACKAGE_NAMES, *ARENA_PACKAGE_NAMES)
SOURCE_ROOT = "src/blueprint_pipeline"
TEST_ROOT = "tests"
CONFIGCLASS_MEMBERS = ("to_dict", "from_dict", "replace", "copy", "validate")
ENUM_BASES = {"enum.Enum", "enum.IntEnum", "enum.StrEnum", "enum.Flag", "enum.IntFlag"}
# Bases that add no member our code could be looking for.
EMPTY_BASES = {"builtins.object", "abc.ABC", *ENUM_BASES}
PASSTHROUGH_CALLS = {"deepcopy", "copy.deepcopy", "copy.copy"}
CONFIG_DECORATORS = {"configclass", "dataclass", "dataclasses.dataclass"}

# Objects our code receives from its own factories, so no constructor call
# in the function names their class: (path, function, local name, class).
RECEIVER_BINDINGS: tuple[tuple[str, str, str, str], ...] = (
    (
        "src/blueprint_pipeline/native_task_robot_registry.py",
        "add_droid_exterior_right_slot",
        "embodiment",
        "isaaclab_arena.embodiments.droid.droid.DroidAbsoluteJointPositionEmbodiment",
    ),
    (
        "src/blueprint_pipeline/native_task_arena_runtime.py",
        "build_native_task_arena_environment",
        "embodiment",
        "isaaclab_arena.embodiments.droid.droid.DroidAbsoluteJointPositionEmbodiment",
    ),
)

# Type references in the index are strings: "!R" an object annotated as class
# R (it may be a subclass), "()R" what calling R returns, "=R" the value of R.


# --------------------------------------------------------------------------
# Index: the pinned sources, read with ast.
# --------------------------------------------------------------------------


def _signature(node: ast.FunctionDef | ast.AsyncFunctionDef, *, bound: bool) -> list[Any]:
    """[keyword-passable parameter names, accepts **kwargs]."""
    params = [arg.arg for arg in (*node.args.args, *node.args.kwonlyargs)]
    if bound and node.args.args:
        params = params[1:]
    return [params, node.args.kwarg is not None]


def _dotted(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        inner = _dotted(node.value)
        return f"{inner}.{node.attr}" if inner else node.attr
    return ""


def _decorator_names(node: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    names = set()
    for decorator in node.decorator_list:
        target = decorator.func if isinstance(decorator, ast.Call) else decorator
        if isinstance(target, ast.Name | ast.Attribute):
            names.update({_dotted(target), _dotted(target).rsplit(".", 1)[-1]})
    return names


def _top_level(body: Iterable[ast.stmt]) -> Iterable[ast.stmt]:
    """Statements at this level, including those under if/try/with."""
    for statement in body:
        if isinstance(statement, ast.If):
            yield from _top_level(statement.body)
            yield from _top_level(statement.orelse)
        elif isinstance(statement, ast.Try | ast.TryStar):
            yield from _top_level(statement.body)
            for handler in statement.handlers:
                yield from _top_level(handler.body)
            yield from _top_level(statement.orelse)
            yield from _top_level(statement.finalbody)
        elif isinstance(statement, ast.With):
            yield from _top_level(statement.body)
        else:
            yield statement


def _parse_string(node: ast.expr | None) -> ast.expr | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        try:
            return ast.parse(node.value, mode="eval").body
        except SyntaxError:
            return None
    return node


class _Namespace:
    """Turns names in one pinned module into dotted references."""

    def __init__(self, module: str, is_package: bool, names: Mapping[str, list[Any]]):
        self.module = module
        self.package = module if is_package else module.rpartition(".")[0]
        self.names = names

    def absolute(self, origin: str | None, level: int) -> str:
        if not level:
            return origin or ""
        parts = self.package.split(".")
        base = parts[: len(parts) - (level - 1)]
        return ".".join([*base, origin] if origin else base)

    def ref(self, node: ast.expr | None, local: Mapping[str, str] | None = None) -> str | None:
        node = _parse_string(node)
        if isinstance(node, ast.Name):
            if local and node.id in local:
                return local[node.id]
            entry = self.names.get(node.id)
            if entry is None:
                return f"builtins.{node.id}"
            if entry[0] in {"alias", "module"}:
                return entry[1]
            return f"{self.module}.{node.id}"
        if isinstance(node, ast.Attribute):
            base = self.ref(node.value, local)
            return f"{base}.{node.attr}" if base else None
        if isinstance(node, ast.Subscript):
            return self.ref(node.value, local)
        return None

    def annotation(self, node: ast.expr | None, local: Mapping[str, str] | None = None) -> str | None:
        """``!R`` for the one class an annotation names, ignoring ``| None`` and ``Optional``."""
        node = _parse_string(node)
        if node is None:
            return None
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
            sides = [side for side in (node.left, node.right)
                     if not (isinstance(side, ast.Constant) and side.value is None)]
            return self.annotation(sides[0], local) if len(sides) == 1 else None
        if isinstance(node, ast.Subscript):
            outer = self.ref(node.value, local) or ""
            return self.annotation(node.slice, local) if outer.endswith(("Optional", "ClassVar")) else None
        ref = self.ref(node, local)
        if ref is None or ref.startswith(("builtins.", "typing.")):
            return None
        return f"!{ref}"

    def value(self, node: ast.expr | None, local: Mapping[str, str] | None = None) -> str | None:
        """``()R`` for a call of R, ``=R`` for a name or attribute R."""
        if isinstance(node, ast.Call):
            ref = self.ref(node.func, local)
            return f"(){ref}" if ref and not ref.startswith("builtins.") else None
        if isinstance(node, ast.Name | ast.Attribute):
            ref = self.ref(node, local)
            return f"={ref}" if ref and not ref.startswith("builtins.") else None
        return None


def _module_names(tree: ast.Module, helper: _Namespace, names: dict[str, list[Any]], stars: list[str]) -> None:
    for statement in _top_level(tree.body):
        if isinstance(statement, ast.ImportFrom):
            origin = helper.absolute(statement.module, statement.level)
            for alias in statement.names:
                if alias.name == "*":
                    stars.append(origin)
                else:
                    names.setdefault(alias.asname or alias.name, ["alias", f"{origin}.{alias.name}"])
        elif isinstance(statement, ast.Import):
            for alias in statement.names:
                bound = alias.asname or alias.name.split(".")[0]
                names.setdefault(bound, ["module", alias.name if alias.asname else bound])
        elif isinstance(statement, ast.ClassDef):
            names[statement.name] = ["class"]
        elif isinstance(statement, ast.FunctionDef | ast.AsyncFunctionDef):
            names[statement.name] = ["function"]
        elif isinstance(statement, ast.Assign | ast.AnnAssign):
            for target in getattr(statement, "targets", None) or [statement.target]:
                for name in ast.walk(target):
                    if isinstance(name, ast.Name):
                        names.setdefault(name.id, ["var"])


def _string_sets(tree: ast.Module) -> dict[str, set[str]]:
    """Module-level names bound to literal string collections (or unions of them)."""
    sets: dict[str, set[str]] = {}

    def strings(node: ast.expr) -> set[str] | None:
        if isinstance(node, ast.Call) and _dotted(node.func) in {"frozenset", "set", "tuple"} and len(node.args) == 1:
            return strings(node.args[0])
        if isinstance(node, ast.Set | ast.Tuple | ast.List):
            values = {element.value for element in node.elts if isinstance(element, ast.Constant)}
            return values if len(values) == len(node.elts) and all(isinstance(v, str) for v in values) else None
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
            left, right = strings(node.left), strings(node.right)
            return left | right if left is not None and right is not None else None
        if isinstance(node, ast.Name):
            return sets.get(node.id)
        return None

    for statement in tree.body:
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1 and isinstance(statement.targets[0], ast.Name):
            found = strings(statement.value)
            if found is not None:
                sets[statement.targets[0].id] = found
    return sets


def _module_getattr(tree: ast.Module, helper: _Namespace, names: dict[str, list[Any]]) -> bool:
    """Read a ``__getattr__(name)`` forwarding shim; return whether other names still fail."""
    function = next((statement for statement in tree.body
                     if isinstance(statement, ast.FunctionDef) and statement.name == "__getattr__"), None)
    if function is None:
        return True
    if not function.args.args:
        return False
    parameter = function.args.args[0].arg
    sets = _string_sets(tree)
    for statement in function.body[:-1]:
        test = statement.test if isinstance(statement, ast.If) else None
        if not (isinstance(test, ast.Compare) and isinstance(test.left, ast.Name) and test.left.id == parameter
                and isinstance(test.ops[0], ast.In) and isinstance(test.comparators[0], ast.Name)
                and test.comparators[0].id in sets):
            return False
        target = None
        for inner in ast.walk(statement):
            if isinstance(inner, ast.ImportFrom):
                for alias in inner.names:
                    target = f"{helper.absolute(inner.module, inner.level)}.{alias.name}"
        for forwarded in sets[test.comparators[0].id]:
            names.setdefault(forwarded, ["alias", f"{target}.{forwarded}"] if target else ["var"])
    last = function.body[-1]
    return isinstance(last, ast.Raise) or (
        isinstance(last, ast.Return) and isinstance(last.value, ast.Call)
        and _dotted(last.value.func) == "_stub_getattr"
    )


def _method_imports(helper: _Namespace, function: ast.AST) -> dict[str, str]:
    local: dict[str, str] = {}
    for node in ast.walk(function):
        if isinstance(node, ast.ImportFrom):
            origin = helper.absolute(node.module, node.level)
            for alias in node.names:
                if alias.name != "*":
                    local[alias.asname or alias.name] = f"{origin}.{alias.name}"
        elif isinstance(node, ast.Import):
            for alias in node.names:
                head = alias.name.split(".")[0]
                local[alias.asname or head] = alias.name if alias.asname else head
    return local


def _self_targets(node: ast.AST, own: str) -> Iterable[tuple[str, ast.expr | None, ast.expr | None]]:
    """(attribute, annotation, value) for every ``self.X`` a method binds."""
    for inner in ast.walk(node):
        targets: list[tuple[ast.expr, ast.expr | None, ast.expr | None]] = []
        if isinstance(inner, ast.Assign):
            for target in inner.targets:
                if isinstance(target, ast.Tuple | ast.List):
                    targets.extend((element, None, None) for element in target.elts)
                else:
                    targets.append((target, None, inner.value if len(inner.targets) == 1 else None))
        elif isinstance(inner, ast.AnnAssign):
            targets.append((inner.target, inner.annotation, inner.value))
        elif isinstance(inner, ast.AugAssign | ast.For):
            targets.extend((target, None, None) for target in ast.walk(inner.target))
        elif isinstance(inner, ast.withitem) and inner.optional_vars is not None:
            targets.extend((target, None, None) for target in ast.walk(inner.optional_vars))
        for target, annotation, value in targets:
            if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == own:
                yield target.attr, annotation, value


def _class_record(
    node: ast.ClassDef, qualname: str, helper: _Namespace, classes: dict[str, dict[str, Any]],
    outer: Mapping[str, str],
) -> None:
    local = dict(outer)
    for statement in node.body:
        if isinstance(statement, ast.ClassDef):
            local[statement.name] = f"{qualname}.{statement.name}"
    record: dict[str, Any] = {
        "bases": [ref for ref in (helper.ref(base, outer) for base in node.bases) if ref],
        "members": {},
    }
    if _decorator_names(node) & CONFIG_DECORATORS:
        record["config"] = True
    members: dict[str, list[Any]] = record["members"]
    for statement in _top_level(node.body):
        if isinstance(statement, ast.ClassDef):
            members[statement.name] = ["class", f"{qualname}.{statement.name}"]
            _class_record(statement, f"{qualname}.{statement.name}", helper, classes, local)
        elif isinstance(statement, ast.FunctionDef | ast.AsyncFunctionDef):
            names = _decorator_names(statement)
            if names & {"property", "cached_property"}:
                members[statement.name] = ["property", helper.annotation(statement.returns, local)]
            elif names & {"setter", "deleter"}:
                if statement.name in members and members[statement.name][0] == "property":
                    members[statement.name] = ["settable", members[statement.name][1]]
            else:
                bound = "staticmethod" not in names
                members[statement.name] = [
                    "method", _signature(statement, bound=bound), helper.annotation(statement.returns, local),
                ]
            if statement.name in {"__getattr__", "__getattribute__"}:
                record["dynamic"] = True
        elif isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name):
            declared = helper.ref(statement.annotation, local) or ""
            kind = "attribute" if declared.endswith("ClassVar") else "field"
            members[statement.target.id] = [
                kind, helper.annotation(statement.annotation, local) or helper.value(statement.value, local),
            ]
        elif isinstance(statement, ast.Assign):
            for target in statement.targets:
                if isinstance(target, ast.Name):
                    # Isaac Lab's configclass makes an unannotated class value a field too.
                    kind = "field" if record.get("config") and not isinstance(statement.value, ast.Lambda) else "attribute"
                    members.setdefault(target.id, [kind, helper.value(statement.value, local)])
    for statement in node.body:
        if not isinstance(statement, ast.FunctionDef | ast.AsyncFunctionDef) or not statement.args.args:
            continue
        method_local = {**local, **_method_imports(helper, statement)}
        for name, annotation, value in _self_targets(statement, statement.args.args[0].arg):
            if name in members and members[name][0] != "instance":
                continue
            typed = helper.annotation(annotation, method_local) if annotation is not None else None
            if typed is None and not (isinstance(value, ast.Constant) and value.value is None):
                typed = helper.value(value, method_local)
            previous = members.get(name)
            if previous is None or (previous[1] is None and not previous[2:]):
                members[name] = ["instance", typed]
            elif typed is not None and previous[1] != typed:
                members[name] = ["instance", None, "mixed"]
    classes[qualname] = record


def _module_files(package_dir: Path, package: str) -> Iterable[tuple[str, bool, list[Path]]]:
    """(dotted module, is package, source files) under one installed package directory."""
    grouped: dict[str, tuple[bool, list[Path]]] = {}
    for path in sorted(package_dir.rglob("*.py*")):
        relative = path.relative_to(package_dir)
        if path.suffix not in {".py", ".pyi"} or any(part in {"test", "tests"} for part in relative.parts[:-1]):
            continue
        parts = [package, *relative.with_suffix("").parts]
        is_package = parts[-1] == "__init__"
        if is_package:
            parts = parts[:-1]
        elif path.suffix == ".pyi":
            continue
        if all(part.isidentifier() for part in parts):
            grouped.setdefault(".".join(parts), (is_package, []))[1].append(path)
    for module, (is_package, files) in sorted(grouped.items()):
        yield module, is_package, files


def build_index(package_dirs: Mapping[str, Path], *, sources: Mapping[str, Any]) -> dict[str, Any]:
    """The API index of every module under the given package directories."""
    modules: dict[str, dict[str, Any]] = {}
    classes: dict[str, dict[str, Any]] = {}
    for package, package_dir in sorted(package_dirs.items()):
        for module, is_package, files in _module_files(package_dir, package):
            names: dict[str, list[Any]] = {}
            stars: list[str] = []
            helper = _Namespace(module, is_package, names)
            trees = [ast.parse(path.read_text(encoding="utf-8"), filename=str(path)) for path in files]
            for tree in trees:
                _module_names(tree, helper, names, stars)
            closed = all([_module_getattr(tree, helper, names) for tree in trees])
            for tree in trees:
                for statement in _top_level(tree.body):
                    if isinstance(statement, ast.ClassDef):
                        _class_record(statement, f"{module}.{statement.name}", helper, classes, {})
                    elif isinstance(statement, ast.FunctionDef | ast.AsyncFunctionDef):
                        names[statement.name] = [
                            "function", _signature(statement, bound=False), helper.annotation(statement.returns),
                        ]
                    elif (isinstance(statement, ast.Assign) and len(statement.targets) == 1
                          and isinstance(statement.targets[0], ast.Name) and names.get(statement.targets[0].id) == ["var"]):
                        names[statement.targets[0].id] = ["var", helper.value(statement.value)]
                    elif (isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name)
                          and names.get(statement.target.id) == ["var"]):
                        names[statement.target.id] = [
                            "var", helper.annotation(statement.annotation) or helper.value(statement.value),
                        ]
            record: dict[str, Any] = {"names": dict(sorted(names.items()))}
            if is_package:
                record["package"] = True
            if stars:
                record["star"] = sorted(set(stars))
            if not closed:
                record["dynamic"] = True
            modules[module] = record
    return {
        "schema_version": INDEX_SCHEMA_VERSION,
        "sources": dict(sources),
        "packages": sorted(package_dirs),
        "modules": modules,
        "classes": dict(sorted(classes.items())),
    }


def fetch_pinned_sources(workdir: Path) -> dict[str, Path]:
    """Fetch only the pinned Python sources (about 25 MB) and verify their trees."""
    roots: dict[str, Path] = {}
    for name, repository, commit, tree, patterns in (
        ("isaaclab", ISAACLAB_REPOSITORY, ISAACLAB_COMMIT, ISAACLAB_TREE,
         [f"/source/{package}/{package}/**/*.py{suffix}" for package in ISAACLAB_PACKAGE_NAMES for suffix in ("", "i")]),
        ("arena", ARENA_REPOSITORY, ARENA_COMMIT, ARENA_TREE,
         [f"/{package}/**/*.py{suffix}" for package in ARENA_PACKAGE_NAMES for suffix in ("", "i")]),
    ):
        repo = workdir / name
        git = ["git", "-C", str(repo)]
        if not (repo / ".git").is_dir():
            repo.mkdir(parents=True, exist_ok=True)
            subprocess.run([*git, "init", "-q"], check=True)
            subprocess.run([*git, "remote", "add", "origin", repository], check=True)
        subprocess.run([*git, "fetch", "-q", "--depth", "1", "--filter=blob:none", "origin", commit], check=True)
        actual = subprocess.run([*git, "rev-parse", f"{commit}^{{tree}}"], check=True,
                                capture_output=True, text=True).stdout.strip()
        if actual != tree:
            raise SystemExit(f"{name} {commit} has tree {actual}, not the pinned {tree}")
        subprocess.run([*git, "sparse-checkout", "set", "--no-cone", *patterns], check=True)
        subprocess.run([*git, "checkout", "-q", "--detach", commit], check=True)
        roots[name] = repo
    return roots


def pinned_package_dirs(isaaclab_root: Path, arena_root: Path) -> dict[str, Path]:
    dirs = {package: isaaclab_root / "source" / package / package for package in ISAACLAB_PACKAGE_NAMES}
    dirs.update({package: arena_root / package for package in ARENA_PACKAGE_NAMES})
    return {package: path for package, path in dirs.items() if path.is_dir()}


def pinned_sources() -> dict[str, Any]:
    return {
        "isaaclab": {"repository": ISAACLAB_REPOSITORY, "commit": ISAACLAB_COMMIT, "tree": ISAACLAB_TREE},
        "arena": {"repository": ARENA_REPOSITORY, "commit": ARENA_COMMIT, "tree": ARENA_TREE},
    }


def prune_index(index: Mapping[str, Any], touched: Iterable[str], unresolvable: Iterable[str] = ()) -> dict[str, Any]:
    """Keep what the check touched (plus enclosing packages), folding in subclass member names."""
    api = PinnedApi(index)
    touched = set(touched)
    keep_modules = {name for name in touched if name in index["modules"]}
    for name in list(keep_modules):
        parts = name.split(".")
        keep_modules.update(".".join(parts[:end]) for end in range(1, len(parts)))
    classes = {}
    for name in sorted(touched & set(index["classes"])):
        record = dict(index["classes"][name])
        own = {member for owner in api.mro(name)[0] for member in api.classes[owner]["members"]}
        extra = sorted(api.descendant_members(name) - own)
        if extra:
            record["descendant_members"] = extra
        classes[name] = record
    return {
        **{key: value for key, value in index.items() if key not in {"modules", "classes"}},
        "module_names": sorted(index["modules"]),
        "unresolvable": sorted(set(unresolvable)),
        "modules": {name: index["modules"][name] for name in sorted(keep_modules & set(index["modules"]))},
        "classes": classes,
    }


# --------------------------------------------------------------------------
# Resolution against the index.
# --------------------------------------------------------------------------

MISSING = ("missing",)


def _frozen(value: Any) -> Any:
    if isinstance(value, list | tuple):
        return tuple(_frozen(item) for item in value)
    return value


class PinnedApi:
    """Abstract values are tuples: ("module", name), ("class", qualname),
    ("object", qualname, exact), ("function", signature, returns),
    ("method", owner, name, signature, returns, exact), ("text", value).
    ``None`` is unknown; ``MISSING`` is absent from the pinned source."""

    def __init__(self, index: Mapping[str, Any]):
        self.modules: Mapping[str, Any] = index["modules"]
        self.classes: dict[str, Any] = dict(index["classes"])
        self.packages = set(index.get("packages") or PINNED_PACKAGES)
        # A pruned index names every pinned module, so a lookup outside it is
        # reported as unindexed (rebuild the index) rather than as missing.
        self.all_modules = set(index.get("module_names") or self.modules)
        self.unresolvable = set(index.get("unresolvable") or ())
        self.pruned = "module_names" in index
        self.touched: set[str] = set()
        self.unindexed: set[str] = set()
        self._depth = 0
        self._descendants: dict[str, set[str]] | None = None

    def pinned(self, dotted: str) -> bool:
        return dotted.split(".")[0] in self.packages

    def module(self, name: str) -> Any:
        """An importable module by its full name (what ``import`` and ``import_module`` take)."""
        if name not in self.all_modules:
            return MISSING
        if name not in self.modules:
            self.unindexed.add(name)
            return None
        self.touched.add(name)
        return ("module", name)

    def resolve(self, ref: str | None) -> Any:
        """The value a dotted name in the pinned packages stands for."""
        if not ref or self._depth > 40:
            return None
        self._depth += 1
        try:
            parts = ref.split(".")
            for end in range(len(parts), 0, -1):
                module = ".".join(parts[:end])
                if module not in self.modules and module in self.all_modules:
                    self.unindexed.add(module)
                    return None
                if module in self.modules:
                    self.touched.add(module)
                    value: Any = ("module", module)
                    for part in parts[end:]:
                        value = self.attribute(value, part)
                        if value is None or value is MISSING:
                            return value
                    return value
            if ref in self.classes:
                return ("class", ref)
            return MISSING if self.pinned(ref) else None
        finally:
            self._depth -= 1

    def typed(self, ref: str | None) -> Any:
        """The value an index type reference stands for."""
        if not ref:
            return None
        if ref.startswith("!"):
            value = self.resolve(ref[1:])
            return ("object", value[1], False) if value and value is not MISSING and value[0] == "class" else None
        if ref.startswith("()"):
            return self.call_result(self.resolve(ref[2:]))
        if ref.startswith("="):
            value = self.resolve(ref[1:])
            return value if value is not MISSING else None
        return None

    def _module_name(self, module: str, name: str, seen: frozenset[str] = frozenset()) -> Any:
        record = self.modules[module]
        entry = record["names"].get(name)
        if entry is None or entry[1:] == [f"{module}.{name}"]:
            submodule = f"{module}.{name}"
            if submodule in self.modules:
                self.touched.add(submodule)
                return ("module", submodule)
            if submodule in self.all_modules:
                self.unindexed.add(submodule)
                return None
            for star in record.get("star", ()):
                if star in self.modules and star not in seen:
                    self.touched.add(star)
                    found = self._module_name(star, name, seen | {module})
                    if found is not MISSING:
                        return found
                elif star in self.all_modules:
                    self.unindexed.add(star)
                    return None
                elif not self.pinned(star):
                    return None
            return None if record.get("dynamic") or entry is not None else MISSING
        kind = entry[0]
        if kind == "class":
            self.touched.add(f"{module}.{name}")
            return ("class", f"{module}.{name}")
        if kind == "function":
            return ("function", _frozen(entry[1]), entry[2]) if len(entry) > 1 else ("function", None, None)
        if kind in {"alias", "module"}:
            return self.resolve(entry[1]) if self.pinned(entry[1]) else None
        return self.typed(entry[1]) if len(entry) > 1 else None

    def mro(self, qualname: str) -> tuple[list[str], bool]:
        """The pinned classes in lookup order, and whether every base is known."""
        order: list[str] = []
        closed = True
        pending = [qualname]
        while pending:
            current = pending.pop(0)
            if current in order:
                continue
            record = self.classes.get(current)
            if record is None:
                if self.pinned(current) and current not in self.unresolvable:
                    self.unindexed.add(current)
                closed = False
                continue
            self.touched.add(current)
            order.append(current)
            closed = closed and not record.get("dynamic")
            for base in record["bases"]:
                if base in EMPTY_BASES or base.startswith("typing."):
                    continue
                value = self.resolve(base)
                if value and value is not MISSING and value[0] == "class":
                    pending.append(value[1])
                else:
                    closed = False
        return order, closed

    def is_config(self, qualname: str) -> bool:
        return any(self.classes[name].get("config") for name in self.mro(qualname)[0])

    def is_enum(self, qualname: str) -> bool:
        return any(set(self.classes[name]["bases"]) & ENUM_BASES for name in self.mro(qualname)[0])

    def descendant_members(self, qualname: str) -> set[str]:
        """Members only pinned subclasses declare; a pruned index carries them precomputed."""
        if self.pruned:
            return set(self.classes.get(qualname, {}).get("descendant_members", ()))
        if self._descendants is None:
            touched, self._descendants = set(self.touched), {}
            for name, record in list(self.classes.items()):
                for ancestor in self.mro(name)[0][1:]:
                    self._descendants.setdefault(ancestor, set()).update(record["members"])
            self.touched = touched  # what a pruned index keeps, it keeps as descendant_members
        return self._descendants.get(qualname, set())

    def member(self, qualname: str, name: str, *, exact: bool = True) -> Any:
        """(owner, entry) for a class member, None if unknown, MISSING if absent."""
        order, closed = self.mro(qualname)
        for owner in order:
            entry = self.classes[owner]["members"].get(name)
            if entry is None:
                continue
            if entry[0] in {"field", "instance", "attribute"} and entry[1] is None and not entry[2:]:
                # A base may know the type the most derived class leaves open.
                for other in order[order.index(owner) + 1:]:
                    typed = self.classes[other]["members"].get(name)
                    if typed is not None and typed[0] == entry[0] and typed[1]:
                        return owner, [entry[0], typed[1]]
            return owner, entry
        if self.is_config(qualname) and name in CONFIGCLASS_MEMBERS:
            return qualname, ["method", None, None]
        if self.is_enum(qualname) and name in {"name", "value"}:
            return qualname, ["attribute", None]
        if not closed or (not exact and name in self.descendant_members(qualname)):
            return None
        return MISSING

    def attribute(self, value: Any, name: str) -> Any:
        """The value of ``value.name``: None unknown, MISSING absent."""
        if not value or value is MISSING or name.startswith("__"):
            return None
        if value[0] == "module":
            return self._module_name(value[1], name) if value[1] in self.modules else None
        if value[0] not in {"class", "object"}:
            return None
        exact = value[0] == "class" or value[2]
        found = self.member(value[1], name, exact=exact)
        if found is None or found is MISSING:
            return found
        owner, entry = found
        kind = entry[0]
        if kind == "class":
            self.touched.add(entry[1])
            return ("class", entry[1])
        if kind == "method":
            return ("method", value[1], name, _frozen(entry[1]), entry[2], bool(exact))
        if kind in {"property", "settable"}:
            return self.typed(entry[1]) if value[0] == "object" else None
        if kind == "attribute" and self.is_enum(owner):
            return ("object", owner, True)
        return self.typed(entry[1])

    def call_result(self, value: Any) -> Any:
        if not value or value is MISSING:
            return None
        if value[0] == "class":
            return ("object", value[1], True)
        if value[0] == "function":
            return self.typed(value[2])
        if value[0] == "method":
            if value[2] in {"replace", "copy"} and value[3] is None:
                return ("object", value[1], value[5])
            return self.typed(value[4])
        return None

    def accepted_keywords(self, value: Any) -> set[str] | None:
        """Keywords a call accepts, or None when it takes any or cannot be known."""
        if not value or value is MISSING:
            return None
        signature = None
        if value[0] == "class":
            order, closed = self.mro(value[1])
            if not closed:
                return None
            for owner in order:
                entry = self.classes[owner]["members"].get("__init__")
                if entry and entry[0] == "method":
                    signature = entry[1]
                    break
                if self.classes[owner].get("config"):
                    return self.fields(value[1])
        elif value[0] == "method" and value[2] == "replace" and value[3] is None:
            fields = self.fields(value[1])
            # An annotated config may be a subclass at runtime, with that subclass's fields.
            return fields if fields is None or value[5] else fields | self.descendant_members(value[1])
        elif value[0] == "function":
            signature = value[1]
        elif value[0] == "method":
            signature = value[3]
        if not signature:
            return None
        params, varkw = signature
        return None if varkw else set(params)

    def fields(self, qualname: str) -> set[str] | None:
        order, closed = self.mro(qualname)
        if not closed:
            return None
        return {name for owner in order if self.classes[owner].get("config")
                for name, entry in self.classes[owner]["members"].items() if entry[0] == "field"}


# --------------------------------------------------------------------------
# Our code.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Finding:
    path: str
    line: int
    scope: str
    kind: str
    detail: str

    @property
    def key(self) -> str:
        return f"{self.path}::{self.scope}::{self.kind}::{self.detail}"


@dataclass
class _Scope:
    qualname: str
    parent: _Scope | None
    owner: str | None = None  # the local class whose method this is
    values: dict[str, Any] = field(default_factory=dict)
    pending: dict[str, set[Any]] = field(default_factory=dict)
    fixed: dict[str, Any] = field(default_factory=dict)
    local: set[str] = field(default_factory=set)
    bound: set[str] = field(default_factory=set)  # typed by RECEIVER_BINDINGS, whatever assigns them
    frozen: bool = False

    def lookup(self, name: str) -> Any:
        scope: _Scope | None = self
        while scope is not None:
            if name in scope.local:
                return scope.values.get(name)
            scope = scope.parent
        return None


def _local_names(body: Sequence[ast.stmt]) -> set[str]:
    """Names a body binds, without descending into nested functions, classes or comprehensions."""
    names: set[str] = set()
    pending: list[ast.AST] = list(body)
    while pending:
        node = pending.pop()
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            names.add(node.name)
            pending.extend(node.decorator_list)
            continue
        if isinstance(node, ast.Lambda | ast.ListComp | ast.SetComp | ast.GeneratorExp | ast.DictComp):
            continue
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            names.add(node.id)
        elif isinstance(node, ast.Import | ast.ImportFrom):
            names.update(alias.asname or alias.name.split(".")[0] for alias in node.names if alias.name != "*")
        elif isinstance(node, ast.ExceptHandler) and node.name:
            names.add(node.name)
        pending.extend(ast.iter_child_nodes(node))
    return names


class _FileChecker:
    def __init__(self, api: PinnedApi, path: str, tree: ast.Module, bindings: Sequence[tuple[str, str, str, str]]):
        self.api = api
        self.path = path
        self.tree = tree
        self.bindings = {(function, name): pinned for where, function, name, pinned in bindings if where == path}
        self.used_bindings: set[tuple[str, str]] = set()
        self.findings: list[Finding] = []
        self.checked = 0
        self.emit = False

    def flag(self, node: ast.AST, scope: _Scope, kind: str, detail: str) -> None:
        if self.emit:
            self.findings.append(Finding(self.path, getattr(node, "lineno", 0), scope.qualname, kind, detail))

    def describe(self, value: Any) -> str:
        if value[0] == "module":
            return value[1]
        record = self.api.classes.get(value[1]) or {}
        return record.get("label") or value[1].rsplit(".", 1)[-1]

    def count(self) -> None:
        if self.emit:
            self.checked += 1

    # Expressions --------------------------------------------------------
    def attribute(self, node: ast.AST, scope: _Scope, value: Any, name: str, *, store: bool = False) -> Any:
        if not value or value is MISSING or value[0] not in {"module", "class", "object"} or name.startswith("__"):
            return None
        self.count()
        result = self.api.attribute(value, name)
        if result is MISSING:
            # Python lets any plain object take a new attribute; only a config ignores it.
            if not (store and value[0] == "object" and not self.api.is_config(value[1])):
                self.flag(node, scope, "missing_field" if store else "missing_attribute", f"{self.describe(value)}.{name}")
            return None
        if store and value[0] == "object":
            found = self.api.member(value[1], name)
            if found and found is not MISSING and found[1][0] == "property":
                self.flag(node, scope, "read_only_property", f"{self.describe(value)}.{name}")
        return result

    def check_keywords(self, node: ast.Call, scope: _Scope, callee: Any) -> None:
        names = [keyword.arg for keyword in node.keywords if keyword.arg]
        accepted = self.api.accepted_keywords(callee) if names else None
        if accepted is None:
            return
        if callee[0] == "method":
            label = f"{self.describe(('class', callee[1]))}.{callee[2]}"
        elif callee[0] == "class":
            label = self.describe(callee)
        else:
            label = _dotted(node.func)
        for name in names:
            self.count()
            if name not in accepted:
                self.flag(node, scope, "unknown_keyword", f"{label}({name}=)")

    def text(self, node: ast.expr | None, scope: _Scope) -> str | None:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        value = self.eval(node, scope) if node is not None else None
        return value[1] if value and value[0] == "text" else None

    def check_module_text(self, node: ast.expr, scope: _Scope) -> Any:
        name = self.text(node, scope)
        if not name or not self.api.pinned(name):
            return None
        self.count()
        value = self.api.module(name)
        if value is MISSING:
            self.flag(node, scope, "missing_module", name)
            return None
        return value

    def synthetic_class(self, node: ast.Call, scope: _Scope) -> Any:
        """``type(name, (Base,), namespace)``: a subclass with the namespace's members."""
        resolved = [self.eval(element, scope) for element in getattr(node.args[1], "elts", [])]
        if not resolved or not all(value and value[0] == "class" for value in resolved):
            return None
        members: dict[str, list[Any]] = {}
        closed = isinstance(node.args[2], ast.Dict)
        for key, value in zip(getattr(node.args[2], "keys", []), getattr(node.args[2], "values", [])):
            name = self.text(key, scope)
            if name == "__annotations__" and isinstance(value, ast.Dict):
                for inner in value.keys:
                    inner_name = self.text(inner, scope)
                    closed = closed and inner_name is not None
                    if inner_name:
                        members.setdefault(inner_name, ["field", None])
            elif name is None:
                closed = False
            elif not name.startswith("__"):
                members[name] = ["field", None]
        qualname = f"local:{self.path}:{scope.qualname}.type@{node.lineno}"
        self.api.classes[qualname] = {
            "bases": [value[1] for value in resolved] + ([] if closed else ["?open"]),
            "members": members,
            "label": "type(..., (" + ", ".join(self.describe(value) for value in resolved) + ",))",
        }
        return ("class", qualname)

    def eval(self, node: ast.expr | None, scope: _Scope) -> Any:
        if node is None:
            return None
        if isinstance(node, ast.Name):
            return scope.lookup(node.id)
        if isinstance(node, ast.Constant):
            return ("text", node.value) if isinstance(node.value, str) else None
        if isinstance(node, ast.Attribute):
            return self.attribute(node, scope, self.eval(node.value, scope), node.attr)
        if isinstance(node, ast.Call):
            return self.call(node, scope)
        if isinstance(node, ast.Lambda | ast.ListComp | ast.SetComp | ast.GeneratorExp | ast.DictComp):
            inner = _Scope(scope.qualname, scope)
            if isinstance(node, ast.Lambda):
                inner.local.update(argument.arg for argument in (*node.args.args, *node.args.kwonlyargs))
                for default in (*node.args.defaults, *node.args.kw_defaults):
                    self.eval(default, scope)
            for generator in getattr(node, "generators", []):
                inner.local.update(name.id for name in ast.walk(generator.target) if isinstance(name, ast.Name))
            for child in ast.iter_child_nodes(node):
                if isinstance(child, ast.expr):
                    self.eval(child, inner)
                elif isinstance(child, ast.comprehension):
                    for part in (child.iter, *child.ifs):
                        self.eval(part, inner)
            return None
        if isinstance(node, ast.NamedExpr):
            return self.eval(node.value, scope)
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.expr):
                self.eval(child, scope)
            elif isinstance(child, ast.keyword | ast.comprehension):
                for part in ast.iter_child_nodes(child):
                    if isinstance(part, ast.expr):
                        self.eval(part, scope)
        return None

    def call(self, node: ast.Call, scope: _Scope) -> Any:
        label = _dotted(node.func)
        arguments = [self.eval(argument, scope) for argument in node.args]
        for keyword in node.keywords:
            self.eval(keyword.value, scope)
            if keyword.arg in {"arena_module", "factory_module"}:
                self.check_module_text(keyword.value, scope)
        if label == "super":
            return ("super", scope.owner) if scope.owner else None
        if isinstance(node.func, ast.Attribute):
            receiver = self.eval(node.func.value, scope)
            if receiver and receiver[0] == "super":
                parents = self.api.mro(receiver[1])[0][1:]
                if parents and node.func.attr == "__init__":
                    self.check_keywords(node, scope, ("class", parents[0]))
                return None
            callee = self.attribute(node.func, scope, receiver, node.func.attr)
        else:
            callee = self.eval(node.func, scope)
        if label == "type" and len(node.args) == 1:
            value = arguments[0]
            return ("class", value[1]) if value and value[0] == "object" else None
        if label == "type" and len(node.args) == 3:
            return self.synthetic_class(node, scope)
        if label in PASSTHROUGH_CALLS and len(node.args) == 1:
            return arguments[0]
        if label.rsplit(".", 1)[-1] in {"configclass", "dataclass"} and len(node.args) == 1:
            value = arguments[0]
            if value and value[0] == "class" and value[1].startswith("local:"):
                self.api.classes[value[1]]["config"] = True
            return value
        if label.endswith("import_module") and node.args:
            return self.check_module_text(node.args[0], scope)
        if label in {"getattr", "setattr", "hasattr"} and len(node.args) >= 2:
            name = self.text(node.args[1], scope)
            if label == "hasattr" or (label == "getattr" and len(node.args) == 3):
                return None
            if name is not None:
                return self.attribute(node, scope, arguments[0], name, store=label == "setattr")
            target = arguments[0]
            return self.uniform_field(target[1]) if label == "getattr" and target and target[0] == "object" else None
        if callee and callee is not MISSING:
            self.check_keywords(node, scope, callee)
            return self.api.call_result(callee)
        return None

    def uniform_field(self, qualname: str) -> Any:
        """``getattr(rig, name)`` on a config whose fields are all one class: that class."""
        names = self.api.fields(qualname)
        values = {self.api.attribute(("object", qualname, False), name) for name in names or ()}
        return values.pop() if len(values) == 1 else None

    # Statements ---------------------------------------------------------
    def bind_import(self, statement: ast.Import | ast.ImportFrom, scope: _Scope) -> None:
        """Check a pinned import and bind its names; any other import binds unknowns."""
        def bind(name: str, value: Any) -> None:
            self.assign(ast.Name(id=name, ctx=ast.Store()), value, scope)

        if isinstance(statement, ast.Import):
            for alias in statement.names:
                bound = alias.name if alias.asname else alias.name.split(".")[0]
                value = None
                if self.api.pinned(alias.name):
                    self.count()
                    if self.api.module(alias.name) is MISSING:
                        self.flag(statement, scope, "missing_module", alias.name)
                    else:
                        value = self.api.module(bound)
                bind(alias.asname or bound, value)
            return
        pinned = not statement.level and statement.module and self.api.pinned(statement.module)
        module = self.api.module(statement.module) if pinned else None
        if pinned:
            self.count()
            if module is MISSING:
                self.flag(statement, scope, "missing_module", statement.module)
        for alias in statement.names:
            if alias.name == "*":
                continue
            value = None
            if pinned and module is not MISSING:
                self.count()
                value = self.api.attribute(module, alias.name)
                if value is MISSING:
                    self.flag(statement, scope, "missing_name", f"{statement.module}.{alias.name}")
                    value = None
            bind(alias.asname or alias.name, value)

    def assign(self, target: ast.expr, value: Any, scope: _Scope, *, none_literal: bool = False) -> None:
        if isinstance(target, ast.Name):
            if scope.frozen or none_literal or target.id in scope.bound:
                return
            if self.emit:
                scope.values[target.id] = value
                return
            seen = scope.pending.setdefault(target.id, set())
            seen.add(value)
            scope.values[target.id] = value if len(seen) == 1 else None
        elif isinstance(target, ast.Attribute):
            self.attribute(target, scope, self.eval(target.value, scope), target.attr, store=True)
        elif isinstance(target, ast.Subscript):
            self.eval(target, scope)
        elif isinstance(target, ast.Starred):
            self.assign(target.value, None, scope)
        elif isinstance(target, ast.Tuple | ast.List):
            for element in target.elts:
                self.assign(element, None, scope)

    def run_body(self, body: Sequence[ast.stmt], scope: _Scope) -> None:
        for statement in body:
            self.statement(statement, scope)

    def statement(self, statement: ast.stmt, scope: _Scope) -> None:
        if isinstance(statement, ast.Import | ast.ImportFrom):
            self.bind_import(statement, scope)
        elif isinstance(statement, ast.FunctionDef | ast.AsyncFunctionDef):
            for decorator in statement.decorator_list:
                self.eval(decorator, scope)
            if self.emit:
                self.function(statement, scope)
        elif isinstance(statement, ast.ClassDef):
            self.class_def(statement, scope)
        elif isinstance(statement, ast.Assign | ast.AnnAssign):
            value = self.eval(statement.value, scope) if statement.value else None
            none_literal = statement.value is None or (
                isinstance(statement.value, ast.Constant) and statement.value.value is None)
            for target in getattr(statement, "targets", None) or [statement.target]:
                self.assign(target, value, scope, none_literal=none_literal)
        elif isinstance(statement, ast.AugAssign):
            self.eval(statement.value, scope)
            if not isinstance(statement.target, ast.Name):
                self.eval(statement.target, scope)
            self.assign(statement.target, None, scope)
        elif isinstance(statement, ast.For | ast.AsyncFor):
            self.eval(statement.iter, scope)
            self.assign(statement.target, None, scope)
            self.run_body(statement.body, scope)
            self.run_body(statement.orelse, scope)
        elif isinstance(statement, ast.With | ast.AsyncWith):
            for item in statement.items:
                self.eval(item.context_expr, scope)
                if item.optional_vars is not None:
                    self.assign(item.optional_vars, None, scope)
            self.run_body(statement.body, scope)
        elif isinstance(statement, ast.Try | ast.TryStar):
            self.run_body(statement.body, scope)
            for handler in statement.handlers:
                self.eval(handler.type, scope)
                if handler.name:
                    self.assign(ast.Name(id=handler.name, ctx=ast.Store()), None, scope)
                self.run_body(handler.body, scope)
            self.run_body(statement.orelse, scope)
            self.run_body(statement.finalbody, scope)
        elif isinstance(statement, ast.If | ast.While):
            self.eval(statement.test, scope)
            self.run_body(statement.body, scope)
            self.run_body(statement.orelse, scope)
        elif isinstance(statement, ast.Match):
            self.eval(statement.subject, scope)
            for case in statement.cases:
                self.run_body(case.body, scope)
        else:
            for child in ast.iter_child_nodes(statement):
                if isinstance(child, ast.expr):
                    self.eval(child, scope)

    def class_def(self, node: ast.ClassDef, scope: _Scope) -> None:
        """A class of ours that subclasses a pinned one is checked like a pinned one."""
        bases = [self.eval(base, scope) for base in node.bases]
        qualname = f"local:{self.path}:{scope.qualname}.{node.name}".replace(":<module>.", ":")
        pinned = [value[1] for value in bases if value and value[0] == "class"]
        if pinned and qualname not in self.api.classes:
            members: dict[str, list[Any]] = {}
            for statement in node.body:
                if isinstance(statement, ast.FunctionDef | ast.AsyncFunctionDef):
                    members[statement.name] = ["method", _signature(statement, bound=True), None]
                elif isinstance(statement, ast.AnnAssign | ast.Assign):
                    for target in getattr(statement, "targets", None) or [statement.target]:
                        if isinstance(target, ast.Name):
                            members[target.id] = ["field", None]
            for statement in node.body:
                if isinstance(statement, ast.FunctionDef | ast.AsyncFunctionDef) and statement.args.args:
                    for name, _annotation, _value in _self_targets(statement, statement.args.args[0].arg):
                        members.setdefault(name, ["instance", None])
            record: dict[str, Any] = {
                "bases": pinned + (["?open"] if len(pinned) != len(node.bases) else []),
                "members": members,
            }
            if _decorator_names(node) & CONFIG_DECORATORS:
                record["config"] = True
            self.api.classes[qualname] = record
        value = ("class", qualname) if qualname in self.api.classes else None
        self.assign(ast.Name(id=node.name, ctx=ast.Store()), value, scope)
        if not self.emit:
            return
        body_scope = _Scope(f"{scope.qualname}.{node.name}".removeprefix("<module>."), scope)
        body_scope.local = _local_names(node.body)
        for statement in node.body:
            if isinstance(statement, ast.FunctionDef | ast.AsyncFunctionDef):
                # Methods see the enclosing function's names, not the class body's.
                self.function(statement, scope, owner=qualname if value else None,
                              qualname=f"{body_scope.qualname}.{statement.name}")
            else:
                self.statement(statement, body_scope)

    def function(self, node: ast.FunctionDef | ast.AsyncFunctionDef, parent: _Scope, *,
                 owner: str | None = None, qualname: str | None = None) -> None:
        name = qualname or f"{parent.qualname}.{node.name}".removeprefix("<module>.")
        for default in (*node.args.defaults, *node.args.kw_defaults):
            self.eval(default, parent)
        scope = _Scope(name, parent, owner=owner)
        scope.local = _local_names(node.body)
        arguments = [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
        arguments += [extra for extra in (node.args.vararg, node.args.kwarg) if extra is not None]
        for index, argument in enumerate(arguments):
            scope.local.add(argument.arg)
            first = owner and index == 0 and argument in node.args.args
            scope.fixed[argument.arg] = ("object", owner, False) if first else None
        self.analyze(node.body, scope)

    def analyze(self, body: Sequence[ast.stmt], scope: _Scope) -> None:
        """Settle each local name's value, then walk the body once more to emit checks.

        A name keeps a value only when every assignment in the body gives it the
        same one (``None`` literals aside), so no branch can mistype it."""
        for name in scope.local:
            if (scope.qualname, name) in self.bindings:
                self.used_bindings.add((scope.qualname, name))
                scope.bound.add(name)
                value = self.api.resolve(self.bindings[(scope.qualname, name)])
                scope.fixed[name] = ("object", value[1], True) if value and value[0] == "class" else None
        emit, self.emit = self.emit, False
        for _round in range(3):
            scope.values = dict(scope.fixed)
            scope.pending = {name: {value} for name, value in scope.fixed.items()}
            self.run_body(body, scope)
        self.emit = emit
        if not emit:
            return
        scope.values = {name: (next(iter(seen)) if len(seen) == 1 else None) for name, seen in scope.pending.items()}
        scope.frozen = True
        self.run_body(body, scope)

    def run(self) -> None:
        module = _Scope("<module>", None)
        module.local = _local_names(self.tree.body)
        self.emit = True
        self.analyze(self.tree.body, module)


# --------------------------------------------------------------------------
# Test fakes.
# --------------------------------------------------------------------------


def _declared_members(node: ast.ClassDef) -> dict[str, int]:
    declared: dict[str, int] = {}
    for statement in node.body:
        if isinstance(statement, ast.FunctionDef | ast.AsyncFunctionDef):
            declared.setdefault(statement.name, statement.lineno)
        elif isinstance(statement, ast.AnnAssign | ast.Assign):
            for target in getattr(statement, "targets", None) or [statement.target]:
                if isinstance(target, ast.Name):
                    declared.setdefault(target.id, statement.lineno)
    for statement in node.body:
        if isinstance(statement, ast.FunctionDef | ast.AsyncFunctionDef) and statement.args.args:
            for name, _annotation, value in _self_targets(statement, statement.args.args[0].arg):
                declared.setdefault(name, getattr(value, "lineno", statement.lineno))
    return {name: line for name, line in declared.items() if not name.startswith("__")}


def check_fakes(api: PinnedApi, root: Path) -> tuple[list[Finding], int]:
    """A test fake may declare only what the pinned class it stands in for has.

    A fake stands in for a pinned class when a test installs it in a fake pinned
    module (``module = types.ModuleType("isaaclab..."); module.Name = Fake``),
    or when it is a config class (``...Cfg``) named after exactly one pinned class."""
    by_short: dict[str, list[str]] = {}
    for qualname in api.classes:
        if not qualname.startswith("local:"):
            by_short.setdefault(qualname.rsplit(".", 1)[-1], []).append(qualname)
    findings: list[Finding] = []
    checked = 0
    for path in sorted((root / TEST_ROOT).glob("test_*.py")):
        text = path.read_text(encoding="utf-8", errors="replace")
        if "isaaclab" not in text:
            continue
        relative = path.relative_to(root).as_posix()
        tree = ast.parse(text)
        fakes = {node.name: node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)}
        modules: dict[str, str] = {}
        stands_for: dict[str, str] = {}
        for node in ast.walk(tree):
            if (isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                    and isinstance(node.value, ast.Call) and _dotted(node.value.func).endswith("ModuleType")
                    and node.value.args and isinstance(node.value.args[0], ast.Constant)
                    and isinstance(node.value.args[0].value, str) and api.pinned(node.value.args[0].value)):
                modules[node.targets[0].id] = node.value.args[0].value
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Attribute)
                    and isinstance(node.targets[0].value, ast.Name) and node.targets[0].value.id in modules):
                continue
            real = f"{modules[node.targets[0].value.id]}.{node.targets[0].attr}"
            checked += 1
            value = api.resolve(real)
            if value is MISSING:
                findings.append(Finding(relative, node.lineno, "<fake module>", "fake_name_not_pinned", real))
            elif value and value[0] == "class" and isinstance(node.value, ast.Name) and node.value.id in fakes:
                stands_for.setdefault(node.value.id, value[1])
        for name in fakes:
            if name not in stands_for and name.endswith("Cfg") and len(by_short.get(name, ())) == 1:
                stands_for[name] = by_short[name][0]
        for name, real in sorted(stands_for.items()):
            for member, line in sorted(_declared_members(fakes[name]).items()):
                checked += 1
                if api.member(real, member, exact=False) is MISSING:
                    findings.append(Finding(relative, line, name, "fake_member_not_pinned", f"{name}.{member}"))
    return findings, checked


# --------------------------------------------------------------------------
# Repository check.
# --------------------------------------------------------------------------


def target_files(root: Path) -> list[str]:
    """Our source files that reach into Isaac Lab or Arena."""
    pattern = re.compile(r"^\s*(?:from|import)\s+(?:%s)\b" % "|".join(PINNED_PACKAGES), re.MULTILINE)
    found = []
    for path in sorted((root / SOURCE_ROOT).rglob("*.py")):
        text = path.read_text(encoding="utf-8", errors="replace")
        if pattern.search(text) or "arena_module=" in text:
            found.append(path.relative_to(root).as_posix())
    return found


@dataclass
class Report:
    findings: list[Finding]
    checked: int
    files: list[str]
    unused_bindings: list[tuple[str, str, str, str]]
    touched: set[str]
    unindexed: set[str]


def check_repository(root: Path = ROOT, index: Mapping[str, Any] | None = None,
                     bindings: Sequence[tuple[str, str, str, str]] = RECEIVER_BINDINGS) -> Report:
    api = PinnedApi(index if index is not None else load_index(root))
    findings: list[Finding] = []
    checked = 0
    used: set[tuple[str, str, str]] = set()
    files = target_files(root)
    for path in files:
        tree = ast.parse((root / path).read_text(encoding="utf-8"), filename=path)
        checker = _FileChecker(api, path, tree, bindings)
        checker.run()
        findings.extend(checker.findings)
        checked += checker.checked
        used.update((path, function, name) for function, name in checker.used_bindings)
    fake_findings, fake_checked = check_fakes(api, root)
    unique = {finding.key: finding for finding in [*findings, *fake_findings]}
    return Report(
        findings=sorted(unique.values(), key=lambda row: (row.path, row.line, row.key)),
        checked=checked + fake_checked,
        files=files,
        unused_bindings=[binding for binding in bindings if binding[:3] not in used],
        touched=set(api.touched),
        unindexed=set(api.unindexed),
    )


def load_index(root: Path = ROOT) -> dict[str, Any]:
    return json.loads((root / INDEX_PATH).read_text(encoding="utf-8"))


def load_known_findings(root: Path = ROOT) -> dict[str, str]:
    payload = json.loads((root / KNOWN_FINDINGS_PATH).read_text(encoding="utf-8"))
    return {row["key"]: row["note"] for row in payload["findings"]}


def dump_index(index: Mapping[str, Any]) -> str:
    """One module or class per line, so a rebuild's diff names what changed."""
    lines = ["{"]
    for key in sorted(set(index) - {"modules", "classes"}):
        lines.append(f" {json.dumps(key)}: {json.dumps(index[key], sort_keys=True)},")
    for section, last in (("classes", False), ("modules", True)):
        rows = sorted(index[section].items())
        lines.append(f" {json.dumps(section)}: {{")
        lines.extend(
            f"  {json.dumps(name)}: {json.dumps(record, sort_keys=True, separators=(',', ':'))}"
            + ("," if position < len(rows) - 1 else "")
            for position, (name, record) in enumerate(rows)
        )
        lines.append(" }" + ("" if last else ","))
    return "\n".join([*lines, "}"]) + "\n"


def rebuild_index(isaaclab_root: Path, arena_root: Path, root: Path = ROOT) -> tuple[dict[str, Any], dict[str, Any]]:
    """(full index of the pinned sources, the part of it our code reaches)."""
    full = build_index(pinned_package_dirs(isaaclab_root, arena_root), sources=pinned_sources())
    report = check_repository(root, full)
    return full, prune_index(full, report.touched, report.unindexed)


def index_drift(committed: Mapping[str, Any], full: Mapping[str, Any]) -> list[str]:
    """Where a committed index says something the pinned sources do not.

    Every record it keeps must be the pinned one, so it can lag our code (that
    shows as unindexed) but can never be edited to make a missing name exist."""
    expected = prune_index(full, [*committed["modules"], *committed["classes"]], committed.get("unresolvable", ()))
    drift = [key for key in ("schema_version", "sources", "packages", "module_names")
             if committed.get(key) != expected.get(key)]
    for section in ("modules", "classes"):
        drift.extend(f"{section}:{name}" for name in sorted(committed[section])
                     if committed[section][name] != expected[section].get(name))
    return drift


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    index_parser = commands.add_parser("index", help="rebuild the committed index from the pinned sources")
    index_parser.add_argument("--fetch-into", type=Path, help="fetch the pinned sources into this directory")
    index_parser.add_argument("--isaaclab-root", type=Path)
    index_parser.add_argument("--arena-root", type=Path)
    index_parser.add_argument("--output", type=Path, default=ROOT / INDEX_PATH)
    index_parser.add_argument("--verify", action="store_true",
                              help="fail if a committed record differs from the pinned sources")
    commands.add_parser("check", help="check our code against the committed index")
    args = parser.parse_args(argv)
    if args.command == "index":
        if args.fetch_into:
            roots = fetch_pinned_sources(args.fetch_into)
            isaaclab_root, arena_root = roots["isaaclab"], roots["arena"]
        elif args.isaaclab_root and args.arena_root:
            isaaclab_root, arena_root = args.isaaclab_root, args.arena_root
        else:
            parser.error("index needs --fetch-into, or both --isaaclab-root and --arena-root")
        full, pruned = rebuild_index(isaaclab_root, arena_root)
        if args.verify:
            drift = index_drift(json.loads(args.output.read_text(encoding="utf-8")), full)
            print("\n".join(drift) or f"{args.output} says only what the pinned sources say")
            return 1 if drift else 0
        text = dump_index(pruned)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(f"{text.count(chr(10))} lines -> {args.output}")
        return 0
    report = check_repository()
    known = load_known_findings()
    for finding in report.findings:
        marker = "known" if finding.key in known else "NEW"
        print(f"{marker}\t{finding.path}:{finding.line}\t{finding.kind}\t{finding.detail}\t({finding.scope})")
    for name in sorted(report.unindexed):
        print(f"UNINDEXED\t{name}\t(rebuild the index: index --fetch-into DIR)")
    print(f"{report.checked} references checked in {len(report.files)} files; {len(report.findings)} flagged")
    return 1 if report.unindexed or any(finding.key not in known for finding in report.findings) else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
