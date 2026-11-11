"""Tests that assert a complete set, selected whenever what they enumerate changes.

A test that pins every timer, every paging code or every status version breaks
when a new member is added somewhere the impacted lane's file mapping does not
look: a new unit file no test names, or a module that contributes to a set the
test reads from another module. Such tests used to fail only in the full suite.

Three ways a test pins a set, all read from the test's own source:

* It iterates a repository directory (``SYSTEMD_DIR.iterdir()``,
  ``(ROOT / "deploy" / "systemd").glob("*.timer")``). A change to a file there
  selects it.
* It compares a module's upper-case constant with a literal collection
  (``stall_alarm.PAGE_CODES == {...}``). A change to the module that defines the
  constant, or to a module the definition takes names from, selects it.
* It declares what it pins: ``@pytest.mark.pins("blueprint_pipeline.x",
  "deploy/systemd/*.timer")``. An entry with a ``/`` is a repository path glob
  (a trailing ``/`` means everything under it); any other entry is a dotted
  module name. Use it for a rule a test asserts that another module owns.

Selection is by test node, so a pinned test costs its own few seconds, not its
file's. Standard library only: the CI plan runs under the runner's bare python3.
"""

from __future__ import annotations

import ast
import fnmatch
import re
from functools import lru_cache
from pathlib import Path, PurePosixPath
from typing import Iterable, Iterator, Mapping, Sequence

MARK = "pins"
ITERATION_METHODS = frozenset({"iterdir", "glob", "rglob"})
CONSTANT_NAME = re.compile(r"[A-Z][A-Z0-9_]*[A-Z0-9]")
# Cheap text filters: only test files that pass them are parsed.
DECLARED = "mark.pins("
ITERATES = (".iterdir()", ".glob(", ".rglob(", "listdir(")
COMPARED_LEFT = re.compile(r"([a-z_]\w*)\.([A-Z][A-Z0-9_]*[A-Z0-9])\)?\s*$")
COLLECTION_START = re.compile(r"\s*(?:[{\[(]|(?:set|frozenset|sorted|tuple|list|dict)\()")
COMPARED_RIGHT = re.compile(r"\s*([a-z_]\w*)\.([A-Z][A-Z0-9_]*[A-Z0-9])\b")
COLLECTION_CALLS = frozenset({"set", "frozenset", "sorted", "tuple", "list", "dict"})


def _module_file(root: Path, module: str, import_roots: Sequence[tuple[str, str]]) -> str:
    """The repository path of an importable module, or "" when there is none."""
    for prefix, strip in import_roots:
        package = prefix.removeprefix(strip).rstrip("/").replace("/", ".")
        if module != package and not module.startswith(package + "."):
            continue
        base = strip + module.replace(".", "/")
        for candidate in (base + ".py", base + "/__init__.py"):
            if (root / candidate).is_file():
                return candidate
    return ""


def _is_pins_mark(node: ast.expr) -> bool:
    return (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == MARK and isinstance(node.func.value, ast.Attribute)
            and node.func.value.attr == "mark")


def _declared(decorators: Iterable[ast.expr]) -> list[str]:
    targets: list[str] = []
    for decorator in decorators:
        if _is_pins_mark(decorator):
            targets.extend(arg.value for arg in decorator.args  # type: ignore[attr-defined]
                           if isinstance(arg, ast.Constant) and isinstance(arg.value, str))
    return targets


def _tests(tree: ast.Module) -> Iterator[tuple[str, ast.FunctionDef | ast.AsyncFunctionDef, list[str]]]:
    """(node id suffix, function, targets declared on it or on its class)."""
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test"):
            yield node.name, node, _declared(node.decorator_list)
        elif isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
            inherited = _declared(node.decorator_list)
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name.startswith("test"):
                    yield f"{node.name}::{item.name}", item, inherited + _declared(item.decorator_list)


class _Paths:
    """Repository-relative paths that a test module's expressions name."""

    def __init__(self, test_path: str, tree: ast.Module) -> None:
        self.test_dir = PurePosixPath(test_path).parent
        self.names: dict[str, PurePosixPath | None] = {}
        for node in tree.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                self.names[node.targets[0].id] = self.resolve(node.value)

    def resolve(self, node: ast.expr) -> PurePosixPath | None:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return PurePosixPath(node.value)
        if isinstance(node, ast.Name):
            return self.names.get(node.id)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            left, right = self.resolve(node.left), self.resolve(node.right)
            return None if left is None or right is None else left / right
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "Path" and node.args:
            if isinstance(node.args[0], ast.Name) and node.args[0].id == "__file__":
                return self.test_dir / "__file__"
            return self.resolve(node.args[0])
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in {
                "resolve", "absolute"}:
            return self.resolve(node.func.value)
        if isinstance(node, ast.Attribute) and node.attr == "parent":
            inner = self.resolve(node.value)
            return None if inner is None else inner.parent
        if (isinstance(node, ast.Subscript) and isinstance(node.value, ast.Attribute)
                and node.value.attr == "parents" and isinstance(node.slice, ast.Constant)
                and isinstance(node.slice.value, int)):
            inner = self.resolve(node.value.value)
            if inner is None or node.slice.value >= len(inner.parents):
                return None
            return inner.parents[node.slice.value]
        return None


def _iterated(function: ast.AST, paths: _Paths, root: Path) -> set[str]:
    """Path globs for every repository directory the function lists."""
    globs: set[str] = set()
    for node in ast.walk(function):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        method, target = node.func.attr, node.func.value
        if method == "listdir" and node.args:
            method, target = "iterdir", node.args[0]
        elif method not in ITERATION_METHODS:
            continue
        directory = paths.resolve(target)
        if directory is None or directory.is_absolute() or ".." in directory.parts:
            continue
        relative = directory.as_posix()
        if relative in {"", "."} or not (root / relative).is_dir():
            continue
        pattern = "*"
        if method != "iterdir" and node.args and isinstance(node.args[0], ast.Constant) \
                and isinstance(node.args[0].value, str):
            pattern = node.args[0].value
        globs.add(f"{relative}/**/{pattern}" if method == "rglob" else f"{relative}/{pattern}")
    return globs


IMPORT_FROM = re.compile(r"^[ \t]*from[ \t]+([\w.]+)[ \t]+import[ \t]+(\([^)]*\)|[^\n#]*)", re.MULTILINE)
IMPORT_AS = re.compile(r"^[ \t]*import[ \t]+([\w.]+)[ \t]+as[ \t]+(\w+)", re.MULTILINE)


def _module_aliases(source: str, root: Path, import_roots: Sequence[tuple[str, str]]) -> dict[str, str]:
    """Local name -> importable module, for every module the test file imports anywhere."""
    aliases: dict[str, str] = {}
    for module, alias in IMPORT_AS.findall(source):
        if _module_file(root, module, import_roots):
            aliases[alias] = module
    for origin, listed in IMPORT_FROM.findall(source):
        if origin.startswith("."):
            continue
        for part in re.sub(r"#[^\n]*", "", listed).strip("() \t\n").split(","):
            name, _, alias = part.strip().partition(" as ")
            module = f"{origin}.{name.strip()}"
            if name.strip() and _module_file(root, module, import_roots):
                aliases[alias.strip() or name.strip()] = module
    return aliases


def _compared_pairs(source: str) -> set[tuple[str, str]]:
    """(alias, NAME) beside every ``==`` in the text; a superset of what the AST confirms."""
    pairs: set[tuple[str, str]] = set()
    index = source.find("==")
    while index != -1:
        if source[index + 2:index + 3] != "=" and source[index - 1:index] not in {"=", "!", "<", ">"}:
            left = COMPARED_LEFT.search(source, max(0, index - 120), index)
            if (left and source[left.start() - 1:left.start()] not in {".", "_"}
                    and COLLECTION_START.match(source, index + 2)):
                pairs.add((left[1], left[2]))
            right = COMPARED_RIGHT.match(source, index + 2)
            if right and source[:index].rstrip()[-1:] in {"}", "]", ")"}:
                pairs.add((right[1], right[2]))
        index = source.find("==", index + 2)
    return pairs


def _unwrap(node: ast.expr) -> ast.expr:
    while (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
           and node.func.id in COLLECTION_CALLS and len(node.args) == 1):
        node = node.args[0]
    return node


def _is_literal_collection(node: ast.expr) -> bool:
    node = _unwrap(node)
    return isinstance(node, (ast.Set, ast.List, ast.Tuple, ast.Dict, ast.SetComp))


def _compared_constants(function: ast.AST, aliases: Mapping[str, str]) -> set[tuple[str, str]]:
    """(module, NAME) for every ``module.NAME == <literal collection>`` in the function."""
    pinned: set[tuple[str, str]] = set()
    for node in ast.walk(function):
        if not isinstance(node, ast.Compare) or len(node.ops) != 1 or not isinstance(node.ops[0], ast.Eq):
            continue
        for side, other in ((node.left, node.comparators[0]), (node.comparators[0], node.left)):
            side = _unwrap(side)
            if (isinstance(side, ast.Attribute) and isinstance(side.value, ast.Name)
                    and side.value.id in aliases and CONSTANT_NAME.fullmatch(side.attr)
                    and _is_literal_collection(other)):
                pinned.add((aliases[side.value.id], side.attr))
    return pinned


@lru_cache(maxsize=None)
def _parsed(path: Path) -> ast.Module | None:
    try:
        return ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, SyntaxError, ValueError):
        return None


def _resolve_relative(module: str, *, is_package: bool, level: int, origin: str | None) -> str:
    package = module.split(".") if is_package else module.split(".")[:-1]
    prefix = package[: len(package) - (level - 1)] if level else []
    return ".".join([*prefix, origin] if origin else prefix) if level else (origin or "")


def definition_files(root: Path, module: str, name: str, import_roots: Sequence[tuple[str, str]]) -> set[str]:
    """The module defining ``name``, and each module its definition takes a name from.

    ``PAGE_CODES = frozenset({..., *ACCOUNT_PAGE_CODES})`` with ``ACCOUNT_PAGE_CODES``
    imported from another module makes that module a definer too. Names defined in
    the same module are followed within it; other modules are followed one step.
    """
    path = _module_file(root, module, import_roots)
    if not path:
        return set()
    tree = _parsed(root / path)
    files = {path}
    if tree is None:
        return files
    is_package = path.endswith("/__init__.py")
    imported: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            origin = _resolve_relative(module, is_package=is_package, level=node.level, origin=node.module)
            for alias in node.names:
                submodule = f"{origin}.{alias.name}"
                imported[alias.asname or alias.name] = (
                    submodule if _module_file(root, submodule, import_roots) else origin)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                imported[alias.asname or alias.name.split(".")[0]] = alias.name
    assignments: dict[str, list[ast.expr]] = {}
    for node in tree.body:
        targets: list[ast.expr] = []
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets, value = [node.target], node.value
        for target in targets:
            if isinstance(target, ast.Name):
                assignments.setdefault(target.id, []).append(value)
    pending, seen = [name], set()
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        for value in assignments.get(current, ()):
            for node in ast.walk(value):
                if isinstance(node, ast.Name):
                    if node.id in imported:
                        source = _module_file(root, imported[node.id], import_roots)
                        if source:
                            files.add(source)
                    elif node.id in assignments:
                        pending.append(node.id)
    return files


def _matches(target: str, path: str) -> bool:
    if target.endswith("/"):
        return path.startswith(target)
    if "**/" in target:
        directory, _, pattern = target.partition("/**/")
        return path.startswith(directory + "/") and fnmatch.fnmatchcase(PurePosixPath(path).name, pattern)
    return fnmatch.fnmatchcase(path, target) and path.count("/") == target.count("/")


@lru_cache(maxsize=None)
def _text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _names_any(path: Path, stems: set[str]) -> bool:
    text = _text(path)
    return any(stem in text for stem in stems)


def _directory_token(changed: Sequence[str]) -> re.Pattern[str] | None:
    """Matches a test that quotes a directory holding a changed file."""
    names = {part for path in changed for part in PurePosixPath(path).parent.parts}
    if not names:
        return None
    alternatives = "|".join(sorted(map(re.escape, names)))
    return re.compile(rf"[\"'/](?:{alternatives})[\"'/]")


def pinned_tests(
    root: Path,
    *,
    changed: Sequence[str],
    test_sources: Mapping[str, str],
    import_roots: Sequence[tuple[str, str]],
) -> dict[str, set[str]]:
    """Test node id -> the changed files that select it, for every pinned test a change reaches."""
    if not changed:
        return {}
    selected: dict[str, set[str]] = {}
    changed_set = set(changed)
    directory_token = _directory_token(changed)
    any_module = any(path.endswith(".py") for path in changed)
    definitions: dict[tuple[str, str], set[str]] = {}
    changed_stems = {PurePosixPath(path).stem for path in changed if path.endswith(".py")} - {"__init__"}
    for test_path, source in test_sources.items():
        declared = DECLARED in source
        iterates = (directory_token is not None and any(token in source for token in ITERATES)
                    and directory_token.search(source) is not None)
        aliases: dict[str, str] = {}
        pairs = _compared_pairs(source) if any_module else set()
        if pairs:
            imported = _module_aliases(source, root, import_roots)
            for alias, name in pairs:
                if alias not in imported:
                    continue
                key = (imported[alias], name)
                if key not in definitions:
                    # Parse a module only when it is changed or names a changed module.
                    module_path = _module_file(root, key[0], import_roots)
                    definitions[key] = (
                        definition_files(root, *key, import_roots)
                        if module_path in changed_set or _names_any(root / module_path, changed_stems)
                        else set())
                if definitions[key] & changed_set:
                    aliases[alias] = imported[alias]
        if not (declared or iterates or aliases):
            continue
        tree = _parsed(root / test_path)
        if tree is None:
            continue
        paths = _Paths(test_path, tree) if iterates else None
        for suffix, function, targets in _tests(tree):
            globs = {target for target in targets if "/" in target}
            files = {_module_file(root, target, import_roots) for target in targets if "/" not in target}
            if paths is not None:
                globs |= _iterated(function, paths, root)
            for key in _compared_constants(function, aliases) if aliases else ():
                files |= definitions.get(key) or definition_files(root, *key, import_roots)
            reached = {path for path in changed if any(_matches(glob, path) for glob in globs)}
            reached |= files & changed_set
            if reached:
                selected[f"{test_path}::{suffix}"] = reached
    return selected


def declared_targets(source: str) -> list[tuple[str, list[str]]]:
    """(test node suffix, declared pins) for a test module's source; for validating declarations."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    return [(suffix, targets) for suffix, _function, targets in _tests(tree) if targets]


__all__ = ["declared_targets", "definition_files", "pinned_tests"]
