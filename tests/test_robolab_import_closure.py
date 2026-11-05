"""Every third-party module on RoboLab's run path has a named provider on the worker.

On 2026-11-04 a beta screening run (pi05_droid) reached Isaac and then failed
the RoboLab preflight inside ``robolab/eval/__init__.py`` with a
ModuleNotFoundError whose module name was not retained. This test reads the
vendored, digest-pinned RoboLab archive, follows the imports that run when the
worker resolves ``robolab.eval.InferenceClient`` and ``robolab.eval.run_episode``,
and requires a provider for every third-party top-level module they reach.
"""
from __future__ import annotations

import ast
import json
import sys
import zipfile
from pathlib import Path, PurePosixPath

from blueprint_pipeline import beta_robolab_runtime as runtime
from blueprint_pipeline.native_task_arena_dependency_contract import NATIVE_TASK_ARENA_DEPENDENCY_IMPORTS
from blueprint_pipeline.native_task_runtime_source_provision import TOP_LEVEL_PACKAGES

PACKAGE = Path(runtime.__file__).resolve().parent
VENDOR = PACKAGE / "_vendor"
# ``preflight_robolab_runtime`` and ``run_robolab_episode`` resolve exactly these.
RUN_PATH_ROOTS = ("robolab.eval.base_client", "robolab.eval.episode")
# Loaded by Kit or the retained Isaac Lab sources once Isaac starts.
ISAAC_NATIVE = frozenset({"isaacsim", "omni", "pxr", "carb"})
# A module neither contracted nor shipped needs evidence that this same Isaac
# process already imports it. Keep this short; each row names the evidence.
EVIDENCED_IMAGE_MODULES = {
    "scipy": ("native_task_direct_camera_aim.py", "Every camera-enabled policy session installs the wrist "
              "camera through this module inside the Isaac process, and it imports SciPy at module level."),
}
_OPTIONAL_IMPORT_HANDLERS = frozenset({"ImportError", "ModuleNotFoundError"})


def _modules(archive: zipfile.ZipFile) -> tuple[dict[str, tuple[str, bool]], set[str]]:
    """Source modules, and directories that import as namespace packages (no code runs)."""
    modules, directories = {}, set()
    for name in archive.namelist():
        path = PurePosixPath(name)
        directories.update(".".join(parent.parts) for parent in path.parents if parent.parts)
        if path.suffix == ".py":
            parts = path.with_suffix("").parts
            package = parts[-1] == "__init__"
            modules[".".join(parts[:-1] if package else parts)] = (name, package)
    return modules, directories - set(modules)


def _optional(parents: list[ast.AST]) -> bool:
    """TYPE_CHECKING blocks never run; an ImportError fallback tolerates absence."""
    for node in parents:
        if isinstance(node, ast.If) and "TYPE_CHECKING" in ast.unparse(node.test):
            return True
        if isinstance(node, ast.Try):
            for handler in node.handlers:
                caught = set() if handler.type is None else {
                    item.id if isinstance(item, ast.Name) else getattr(item, "attr", "")
                    for item in (handler.type.elts if isinstance(handler.type, ast.Tuple) else [handler.type])}
                if caught & _OPTIONAL_IMPORT_HANDLERS:
                    return True
    return False


def _imports(source: bytes, *, module: str, package: bool, modules) -> list[tuple[str, bool]]:
    """Every unguarded import as ``(absolute name, inside a function)``."""
    found: list[tuple[str, bool]] = []

    def visit(node: ast.AST, parents: list[ast.AST], in_function: bool) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.Import, ast.ImportFrom)) and not _optional(parents):
                if isinstance(child, ast.Import):
                    names = [alias.name for alias in child.names]
                else:
                    base = module.split(".") if package else module.split(".")[:-1]
                    base = base[:len(base) - child.level + 1] if child.level > 1 else base
                    target = ".".join([*base, *([child.module] if child.module else [])]) if child.level else child.module
                    names = [f"{target}.{alias.name}" if f"{target}.{alias.name}" in modules else target
                             for alias in child.names]
                found.extend((name, in_function) for name in names)
            visit(child, [*parents, child],
                  in_function or isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)))

    visit(ast.parse(source), [], False)
    return found


def robolab_run_path_closure() -> tuple[set[str], dict[str, set[str]]]:
    """RoboLab modules executed on import, and third-party modules they import.

    RoboLab modules are followed through executed (module-level) imports and
    their parent packages. Third-party imports count wherever they appear in a
    followed module, functions included: ``run_episode`` imports Kit inside.
    """
    manifest = json.loads((VENDOR / "robolab-source.v1.json").read_text())
    with zipfile.ZipFile(VENDOR / f"robolab-{manifest['commit']}.zip") as archive:
        modules, namespaces = _modules(archive)
        sources = {module: archive.read(name) for module, (name, _package) in modules.items()}
    closure: set[str] = set()
    third_party: dict[str, set[str]] = {}
    pending = list(RUN_PATH_ROOTS)
    while pending:
        module = pending.pop()
        if module in closure or module in namespaces:
            continue
        assert module in modules, f"robolab_run_path_module_missing_from_archive:{module}"
        closure.add(module)
        parts = module.split(".")
        pending.extend(".".join(parts[:index]) for index in range(1, len(parts)))
        for name, in_function in _imports(sources[module], module=module, package=modules[module][1],
                                          modules=modules):
            top = name.split(".")[0]
            if top == "robolab":
                if not in_function:
                    parts = name.split(".")
                    while ".".join(parts) not in modules:
                        parts.pop()
                    pending.append(".".join(parts))
            elif top != "__future__" and top not in sys.stdlib_module_names:
                third_party.setdefault(top, set()).add(module)
    return closure, third_party


def _wheel_top_level_modules(wheel: Path) -> set[str]:
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
    provided = set()
    for name in names:
        parts = PurePosixPath(name).parts
        if len(parts) == 2 and parts[1] == "__init__.py":
            provided.add(parts[0])
        elif len(parts) == 1 and parts[0].endswith((".py", ".so")):
            provided.add(parts[0].split(".")[0])
    return provided


def test_run_path_closure_is_read_from_the_pinned_archive():
    closure, third_party = robolab_run_path_closure()
    assert {"robolab", "robolab.eval", "robolab.eval.episode", "robolab.eval.base_client",
            "robolab.core.utils.video_utils", "robolab.core.logging.results"} <= closure
    # Lazily resolved, TYPE_CHECKING-only or ImportError-guarded modules stay out.
    assert not {"robolab.eval.runner", "robolab.eval.summarize", "robolab.eval.gt_state"} & closure
    assert {"cv2", "imageio", "numpy", "torch", "tqdm", "h5py", "yaml", "scipy", "omni"} <= set(third_party)
    assert "isaaclab_physx" not in third_party  # only under try/except ImportError


def test_every_run_path_third_party_module_has_a_provider(tmp_path):
    _closure, third_party = robolab_run_path_closure()
    contract = {name.split(".")[0] for name in (*NATIVE_TASK_ARENA_DEPENDENCY_IMPORTS, *TOP_LEVEL_PACKAGES)}
    receipt = runtime.stage_robolab_source(tmp_path)
    shipped = set()
    for row in receipt["python_dependencies"]:
        shipped |= _wheel_top_level_modules(tmp_path / row["wheel"])
    for module, (source, _why) in EVIDENCED_IMAGE_MODULES.items():
        tree = ast.parse((PACKAGE / source).read_text())
        assert any(isinstance(node, (ast.Import, ast.ImportFrom))
                   and module in {getattr(node, "module", None) and node.module.split(".")[0],
                                  *(alias.name.split(".")[0] for alias in node.names)}
                   for node in tree.body), f"evidence_no_longer_imports:{source}->{module}"
    unprovided = {
        module: sorted(importers) for module, importers in third_party.items()
        if module not in contract and module not in shipped and module not in ISAAC_NATIVE
        and not module.startswith("isaaclab") and module not in EVIDENCED_IMAGE_MODULES
    }
    assert unprovided == {}, f"robolab_run_path_modules_without_provider:{unprovided}"
