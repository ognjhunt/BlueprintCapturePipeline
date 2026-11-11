"""Which automatic entry points can reach an API-priced call site, and through what (E4 a, WP-23).

``verify_paid_resource_allocator.py`` reads this to check two claims in
``scripts/paid_api_call_sites.json`` structurally instead of trusting a label:

``not_automatic``
    No automatic entry point reaches the call site: no systemd unit, the Cloud
    Run job (``Dockerfile`` and ``deploy/terraform`` ``-m`` arguments), the
    Cloud Functions (``functions/``, ``main.py``).
``admitted`` with ``guards`` or ``reachable_only_through``
    Every path from an automatic entry point to the call site passes a function
    that calls one of ``guards`` (an admission, reservation or account-limit
    check), or a function of one of the ``reachable_only_through`` modules (the
    stage that runs only inside an admitted rental; the entry names that
    admission and its bound).

The graph is static and over-approximate, which is the safe direction: a
function is a caller of another when it names it (a call, a reference, a
``functools.partial``), through same-module names, nested scopes, ``from``
imports, ``import ... as`` attributes and ``self.``/``cls.`` methods. A class's
methods are called by whatever names the class. A module's import-time code
runs for every importer. A string naming a module (``"blueprint_pipeline.x"``,
``"-m blueprint_pipeline.x"``, ``".../blueprint_pipeline/x.py"``, ``scripts/x.py``)
counts as running that module's entry, which covers subprocess launches and the
file lists shipped to a rented machine. Not modeled: a module path built at
run time from pieces, and ``getattr`` with a computed name; a claim that rests
on either says so in its own row.
"""

from __future__ import annotations

import ast
import functools
import re
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_ENTRY = ("<module>", "<entry>", "main", "<import>")
_DASH_M = re.compile(r"-m\s+(blueprint_pipeline(?:\.[A-Za-z0-9_]+)+)")
_MODULE_TEXT = re.compile(r"blueprint_pipeline(?:\.[A-Za-z0-9_]+)+")
_PACKAGE_FILE = re.compile(r"(?:^|/)(blueprint_pipeline/[A-Za-z0-9_/]+)\.py")
#: The verifier and this module name every call site's file; they run none of them.
_SELF = frozenset({"verify_paid_resource_allocator.py", "paid_call_site_reachability.py"})
_SCRIPT_FILE = re.compile(r"(?:^|/)(scripts|functions)/([A-Za-z0-9_]+)\.py")


def _module_names(root: Path) -> dict[str, Path]:
    names: dict[str, Path] = {}
    for path in (root / "src").glob("blueprint_pipeline/**/*.py"):
        parts = path.relative_to(root / "src").with_suffix("").parts
        names[".".join(parts[:-1] if parts[-1] == "__init__" else parts)] = path
    for folder in ("scripts", "functions"):
        for path in (root / folder).glob("*.py"):
            if path.name not in _SELF:
                names[f"{folder}.{path.stem}"] = path
    if (root / "main.py").is_file():
        names["main"] = root / "main.py"
    return names


def _string_targets(value: str, known: dict[str, Path]) -> set[str]:
    text = value.strip()
    found = set(_DASH_M.findall(text))
    if _MODULE_TEXT.fullmatch(text):
        found.add(text)
    found.update(match.replace("/", ".") for match in _PACKAGE_FILE.findall(text))
    found.update(f"{folder}.{stem}" for folder, stem in _SCRIPT_FILE.findall(text))
    return {name for name in found if name in known}


class _Module:
    def __init__(self, name: str, path: Path, known: dict[str, Path]):
        self.name, self.known = name, known
        self.is_package = path.name == "__init__.py"
        self.tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        self.functions: dict[str, ast.AST] = {}
        self.top: dict[str, str] = {}
        self.imports: dict[str, tuple[str, str | None]] = {}
        self.refs: dict[str, set[tuple[str, object]]] = {}
        self.calls: dict[str, set[str]] = {}
        self._index(self.tree, (), "<module>")

    def _base(self, level: int, module: str | None) -> str:
        if not level:
            return module or ""
        parts = self.name.split(".")
        package = parts if self.is_package else parts[:-1]
        package = package[: len(package) - (level - 1)]
        return ".".join([*package, *([module] if module else [])])

    def _index(self, node: ast.AST, scope: tuple[str, ...], owner: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                qualname = ".".join((*scope, child.name))
                self.functions[qualname] = child
                if not scope:
                    self.top[child.name] = qualname
                for extra in [*child.decorator_list, *getattr(child, "bases", [])]:
                    self._record(owner, extra)
                self._index(child, (*scope, child.name), qualname)
                continue
            if isinstance(child, ast.ImportFrom):
                base = self._base(child.level, child.module)
                for alias in child.names:
                    local = alias.asname or alias.name
                    whole = f"{base}.{alias.name}"
                    self.imports[local] = (whole, None) if whole in self.known else (base, alias.name)
            elif isinstance(child, ast.Import):
                for alias in child.names:
                    self.imports[alias.asname or alias.name] = (alias.name, None)
            self._record(owner, child, shallow=True)
            self._index(child, scope, owner)

    def _record(self, owner: str, node: ast.AST, *, shallow: bool = False) -> None:
        refs = self.refs.setdefault(owner, set())
        calls = self.calls.setdefault(owner, set())
        for item in [node] if shallow else ast.walk(node):
            if isinstance(item, ast.Name):
                refs.add(("name", item.id))
            elif isinstance(item, ast.Attribute):
                refs.add(("attr", (ast.unparse(item.value), item.attr)))
            elif isinstance(item, ast.Constant) and isinstance(item.value, str):
                refs.update(("module", target) for target in _string_targets(item.value, self.known))
            if isinstance(item, ast.Call):
                target = item.func
                calls.add(target.id if isinstance(target, ast.Name) else getattr(target, "attr", ""))


def _parents(qualname: str) -> list[str]:
    parts = qualname.split(".")
    return [".".join(parts[:index]) for index in range(len(parts) - 1, 0, -1)]


def automatic_roots(root: Path, known: dict[str, Path]) -> set[str]:
    """Modules an automatic process starts: systemd units, the Cloud Run job, the Cloud Functions."""

    texts = [path.read_text(encoding="utf-8") for path in (root / "deploy" / "systemd").glob("*.service")]
    texts += [path.read_text(encoding="utf-8") for path in root.glob("Dockerfile")]
    texts += [path.read_text(encoding="utf-8") for path in (root / "deploy" / "terraform").glob("*.tf")]
    found = {"functions.main", "functions.storage_trigger", "main"}
    for text in texts:
        found.update(_MODULE_TEXT.findall(text))
        found.update(f"{folder}.{stem}" for folder, stem in _SCRIPT_FILE.findall(text))
    return {name for name in found if name in known}


class CallGraph:
    def __init__(self, root: Path = ROOT):
        self.known = _module_names(root)
        self.modules: dict[str, _Module] = {}
        for name, path in self.known.items():
            try:
                self.modules[name] = _Module(name, path, self.known)
            except (OSError, SyntaxError, UnicodeDecodeError):
                continue
        self.roots = automatic_roots(root, self.known)
        self.callers: dict[tuple[str, str], set[tuple[str, str]]] = {}
        for module in self.modules.values():
            self._link(module)

    def _add(self, target: tuple[str, str], source: tuple[str, str]) -> None:
        self.callers.setdefault(target, set()).add(source)

    def _link(self, module: _Module) -> None:
        for owner, refs in module.refs.items():
            source = (module.name, owner)
            for kind, value in refs:
                if kind == "module":
                    self._add((str(value), "<entry>"), source)
                elif kind == "name":
                    self._link_name(module, owner, str(value), source)
                else:
                    base, attr = value  # type: ignore[misc]
                    if base in {"self", "cls"} and owner != "<module>":
                        cls = owner.split(".")[0]
                        if f"{cls}.{attr}" in module.functions:
                            self._add((module.name, f"{cls}.{attr}"), source)
                        continue
                    head = str(base).split(".")[0]
                    if base in module.imports or head in module.imports:
                        target, name = module.imports.get(base) or module.imports[head]
                        if name is None:
                            dotted = target + str(base)[len(head):] if base not in module.imports else target
                            self._add((dotted, attr), source)
                            self._add((dotted, "<import>"), source)
                        else:
                            self._add((target, name), source)
        for qualname in module.functions:
            parents = _parents(qualname)
            self._add((module.name, qualname), (module.name, parents[0] if parents else "<module>"))

    def _link_name(self, module: _Module, owner: str, name: str, source: tuple[str, str]) -> None:
        if owner != "<module>":
            for outer in [owner, *_parents(owner)]:
                if f"{outer}.{name}" in module.functions:
                    self._add((module.name, f"{outer}.{name}"), source)
                    return
        if name in module.top:
            self._add((module.name, module.top[name]), source)
        elif name in module.imports:
            target, attr = module.imports[name]
            self._add((target, attr or "<import>"), source)

    def _next(self, node: tuple[str, str]) -> set[tuple[str, str]]:
        module_name, qualname = node
        found = set(self.callers.get(node, ()))
        module = self.modules.get(module_name)
        if qualname in {"<module>", "<import>", "<entry>", "main"}:
            for other in ("<module>", "<import>", "<entry>", "main"):
                found |= self.callers.get((module_name, other), set())
        if qualname == "<module>":
            # Import-time code runs for each importer of the module.
            found |= self.callers.get((module_name, "<import>"), set())
        head = qualname.split(".")[0]
        if module is not None and "." in qualname and isinstance(module.functions.get(head), ast.ClassDef):
            found |= self.callers.get((module_name, head), set())
        return found

    def _calls_guard(self, node: tuple[str, str], guards: frozenset[str]) -> bool:
        module = self.modules.get(node[0])
        return bool(module is not None and module.calls.get(node[1], set()) & guards)

    def start_nodes(self, module_name: str, scope: str, signal_text: str | None = None) -> set[tuple[str, str]]:
        """The call site's own function; for a module-level URL, the functions that use its constant."""

        if scope != "<module>":
            return {(module_name, scope)}
        module = self.modules[module_name]
        names: set[str] = set()
        for node in module.tree.body:
            if isinstance(node, (ast.Assign, ast.AnnAssign)) and any(
                    isinstance(item, ast.Constant) and isinstance(item.value, str) and "https://" in item.value
                    for item in ast.walk(node.value or ast.Constant(""))):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                names.update(item.id for target in targets for item in ast.walk(target) if isinstance(item, ast.Name))
        for _ in range(3):  # module-level names built from those names
            for node in module.tree.body:
                if isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value is not None and any(
                        isinstance(item, ast.Name) and item.id in names for item in ast.walk(node.value)):
                    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                    names.update(item.id for target in targets for item in ast.walk(target)
                                 if isinstance(item, ast.Name))
        starts = {(module_name, owner) for owner, refs in module.refs.items()
                  if owner != "<module>" and any(kind == "name" and value in names for kind, value in refs)}
        for other in self.modules.values():
            for local, (target, attr) in other.imports.items():
                if target == module_name and attr in names:
                    starts |= {(other.name, owner) for owner, refs in other.refs.items()
                               if ("name", local) in refs}
        return starts or {(module_name, "<module>")}

    def automatic_path(self, starts: set[tuple[str, str]], *, guards: frozenset[str] = frozenset(),
                       through: frozenset[str] = frozenset()) -> list[tuple[str, str]] | None:
        """A caller path from an automatic entry to a start node that passes no guard; ``None`` when none."""

        parent: dict[tuple[str, str], tuple[str, str]] = {}
        queue = deque(node for node in starts if not self._calls_guard(node, guards) and node[0] not in through)
        seen = set(queue)
        while queue:
            node = queue.popleft()
            if node[0] in self.roots and node[1] in _ENTRY:
                path = [node]
                while path[-1] in parent:
                    path.append(parent[path[-1]])
                return path
            for caller in self._next(node):
                if caller in seen or caller[0] in through or self._calls_guard(caller, guards):
                    continue
                seen.add(caller)
                parent[caller] = node
                queue.append(caller)
        return None

    def escapes(self, starts: set[tuple[str, str]], *, own: str, through: frozenset[str]) -> list[tuple[str, str]]:
        """Callers outside the site's own module and the ``through`` modules: every way in must pass those."""

        seen, queue, outside = set(starts), deque(starts), []
        while queue:
            node = queue.popleft()
            for caller in self._next(node):
                if caller in seen or caller[0] in through:
                    continue
                seen.add(caller)
                if caller[0] != own:
                    outside.append(caller)
                    continue
                queue.append(caller)
        return sorted(outside)


@functools.lru_cache(maxsize=None)
def graph(root: Path = ROOT) -> CallGraph:
    return CallGraph(root)


#: The shared chokepoint's checks. Every other guard must reach one of them (``unproven_guards``).
BASE_GUARDS = frozenset({"require_paid_resource_budget", "require_paid_resource_admission"})


def unproven_guards(names: frozenset[str], root: Path = ROOT) -> list[str]:
    """Guard names no function in the tree backs with a call that reaches a ``BASE_GUARDS`` check."""

    call_graph = graph(root)
    proven = set(BASE_GUARDS)
    definitions: dict[str, list[set[str]]] = {}
    for module in call_graph.modules.values():
        for qualname, node in module.functions.items():
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                body = set().union(*(module.calls.get(name, set()) for name in module.calls
                                     if name == qualname or name.startswith(qualname + ".")))
                definitions.setdefault(qualname.rsplit(".", 1)[-1], []).append(body)
    for _ in range(6):
        grown = {name for name, bodies in definitions.items() if any(body & proven for body in bodies)}
        if grown <= proven:
            break
        proven |= grown
    return sorted(name for name in names if name not in proven)


def site_module(path: str) -> str:
    relative = path[len("src/"):] if path.startswith("src/") else path
    return relative[: -len(".py")].replace("/", ".").removesuffix(".__init__")


def claim_blockers(key: str, row: dict, root: Path = ROOT) -> list[str]:
    """Blockers for a ``not_automatic`` or a structurally ``admitted`` classification of ``key``."""

    path, scope, _signal = key.split("::")
    module_name = site_module(path)
    call_graph = graph(root)
    if module_name not in call_graph.modules:
        return [f"paid_api_call_site_module_unreadable:{key}"]
    starts = call_graph.start_nodes(module_name, scope)
    # A module-level URL is also read by validators that send nothing: a row may name the functions
    # that send the request instead. Each must exist; they are then where the search starts.
    named = [str(item).partition("::") for item in row.get("request_functions") or []]
    if named:
        starts = {(site_module(path), qualname) for path, _, qualname in named}
        missing = [qualname for module, qualname in starts
                   if module not in call_graph.modules or qualname not in call_graph.modules[module].functions]
        if missing:
            return [f"paid_api_call_site_request_function_missing:{key}:{','.join(sorted(missing))}"]
    guards =frozenset(str(item) for item in row.get("guards") or [])
    through = frozenset(site_module(str(item)) for item in row.get("reachable_only_through") or [])
    kind = "not_automatic" if row.get("class") == "not_automatic" else "admitted"
    blockers = []
    found = call_graph.automatic_path(starts, guards=guards, through=through)
    if found is not None:
        chain = " <- ".join(f"{module.rsplit('.', 1)[-1]}:{qualname}" for module, qualname in found[:8])
        blockers.append(f"paid_api_call_site_{kind}_claim_refuted:{key}: an automatic entry reaches it "
                        f"unguarded: {chain}")
    if through:
        outside = call_graph.escapes(starts, own=module_name, through=through)
        if outside:
            named = ", ".join(f"{module.rsplit('.', 1)[-1]}:{qualname}" for module, qualname in outside[:5])
            blockers.append(f"paid_api_call_site_reachable_only_through_refuted:{key}: also reached from {named}")
    return blockers


__all__ = ["CallGraph", "automatic_roots", "claim_blockers", "graph", "site_module"]
