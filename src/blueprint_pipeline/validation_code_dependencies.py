"""Import-data dependencies of executed validators, including nested cache hits."""
from __future__ import annotations

import ast
from contextvars import ContextVar
import importlib.util
import inspect
from pathlib import Path
import sys

_IMPORTS = {}
_TREES = {}
_CLOSURES = {}
_DATA_CACHE = ContextVar("validation_data_closure_cache", default=None)


def _tree(path):
    path = Path(path)
    state = path.stat()
    key = (str(path), state.st_mtime_ns, state.st_ctime_ns, state.st_size)
    if key not in _TREES:
        _TREES[key] = ast.parse(path.read_bytes())
    return _TREES[key]


def imports(path, module):
    """Resolve imports without importing or executing additional application code."""
    path = Path(path)
    stat = path.stat()
    key = (str(path), stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size)
    if key in _IMPORTS:
        return _IMPORTS[key]
    package = module if path.name == '__init__.py' else module.rpartition('.')[0]
    rows = []
    tree = _tree(path)
    local = {id(child) for function in ast.walk(tree)
             if isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)) for child in ast.walk(function)}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            target = importlib.util.resolve_name('.' * node.level + (node.module or ''), package) if node.level else node.module
            for alias in node.names:
                rows.append((alias.asname or alias.name, target, alias.name, id(node) in local))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                rows.append((alias.asname or alias.name.split('.')[0], alias.name, None, id(node) in local))
    _IMPORTS[key] = rows
    return rows


def _loaded_names(node):
    return {part.id for part in ast.walk(node) if isinstance(part, ast.Name) and isinstance(part.ctx, ast.Load)} | {
        part.attr for part in ast.walk(node) if isinstance(part, ast.Attribute)}


def data_closure(names):
    """Bind import-time data construction, including helpers called to build it.

    Local imports in unrelated function bodies do not construct module data.
    Follow them only when that helper participates in initialization. Functions
    subsequently called by a validator are independently captured by the tracer.
    """
    key = tuple(sorted(names))
    cache = _DATA_CACHE.get()
    if cache is not None and key in cache:
        return cache[key]
    prior = _CLOSURES.get(key)
    if prior is not None:
        modules, witnesses = prior
        valid = True
        for name, expected in witnesses.items():
            spec = importlib.util.find_spec(name)
            path = Path(spec.origin) if spec and spec.origin else None
            try:
                state = path.stat() if path else None
            except OSError:
                state = None
            if state is None or (str(path), state.st_mtime_ns, state.st_ctime_ns, state.st_size) != expected:
                valid = False
                break
        if valid:
            if cache is not None:
                cache[key] = modules
            return modules
    seen, pending = {}, [(name, frozenset()) for name in names]
    modules = set()
    while pending:
        name, requested = pending.pop()
        if not name or not name.startswith('blueprint_pipeline'):
            continue
        modules.add(name)
        spec = importlib.util.find_spec(name)
        if spec is None or not spec.origin or not spec.origin.endswith('.py'):
            raise ValueError('validator_dependency_source_unavailable')
        tree = _tree(spec.origin)
        function_names = {n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
        requested = set(requested) & function_names
        if name in seen and requested <= seen[name]:
            continue
        requested |= seen.get(name, set())
        seen[name] = requested
        used = set(requested)
        functions = {}
        class Initialization(ast.NodeVisitor):
            def visit_FunctionDef(self, node):
                functions[node.name] = node
                used.update(_loaded_names(node.args))
                for decorator in node.decorator_list:
                    used.update(_loaded_names(decorator))

            visit_AsyncFunctionDef = visit_FunctionDef

            def visit_Import(self, node):
                pass

            visit_ImportFrom = visit_Import

            def visit_Name(self, node):
                if isinstance(node.ctx, ast.Load):
                    used.add(node.id)

            def visit_Attribute(self, node):
                used.add(node.attr)
                self.generic_visit(node)

        Initialization().visit(tree)
        expanded = set()
        while (used & functions.keys()) - expanded:
            for helper in (used & functions.keys()) - expanded:
                expanded.add(helper)
                used.update(_loaded_names(functions[helper]))
        for alias, target, member, _ in imports(spec.origin, name):
            if target and target.startswith('blueprint_pipeline') and (alias in used or alias == '*'):
                pending.append((target, frozenset([member]) if member else frozenset(used)))
                child = getattr(sys.modules.get(target), member, None) if member else None
                if inspect.ismodule(child):
                    pending.append((child.__name__, frozenset(used)))
    witnesses = {}
    for name in modules:
        spec = importlib.util.find_spec(name)
        path = Path(spec.origin)
        state = path.stat()
        witnesses[name] = (str(path), state.st_mtime_ns, state.st_ctime_ns, state.st_size)
    _CLOSURES[key] = (frozenset(modules), witnesses)
    if cache is not None:
        cache[key] = modules
    return modules


def frame_dependencies(frame, module):
    used = set(frame.f_code.co_names) | set(frame.f_code.co_varnames)
    names = set()
    for alias, target, member, local in imports(frame.f_code.co_filename, module):
        if local and alias not in used and alias != "*":
            continue
        if not target or not target.startswith('blueprint_pipeline'):
            continue
        value = getattr(sys.modules.get(target), member, None) if member else None
        if inspect.isfunction(value):
            if alias in used or alias == "*":
                names.add(target)
        else:
            # Constants can be derived at import time, before tracing begins.
            # Include all data imports, including modules imported via `from .`.
            names.update(data_closure([target]))
            if inspect.ismodule(value):
                names.update(data_closure([value.__name__]))
    return names
