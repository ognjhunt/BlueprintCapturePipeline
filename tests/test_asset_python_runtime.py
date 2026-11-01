"""The sandbox gets back what Kit's Python needs: its prefix and its libpython path."""
from __future__ import annotations

import sys
from pathlib import Path

from blueprint_pipeline import asset_python_runtime as runtime
from blueprint_pipeline import task_object_artisan_hooks as hooks


def test_interpreter_prefix_and_its_libpython_directory_are_admitted(tmp_path, monkeypatch):
    base = tmp_path / "isaac-sim/kit/python"
    (base / "lib").mkdir(parents=True)
    version = f"{sys.version_info.major}.{sys.version_info.minor}"
    (base / "lib" / f"libpython{version}.so.1.0").write_text("fixture")
    elsewhere = tmp_path / "isaac-sim/kit/plugins"
    elsewhere.mkdir(parents=True)
    monkeypatch.setattr(sys, "base_prefix", str(base))
    monkeypatch.setenv("LD_LIBRARY_PATH", f"{base / 'lib'}:{elsewhere}:relative/path")
    value = runtime.sandboxed_interpreter()
    assert value["roots"] == [base.resolve(), (base / "lib").resolve()]
    # Only the interpreter's own libraries; a wider launcher path is not admitted.
    assert value["library_environment"] == {"LD_LIBRARY_PATH": str((base / "lib").resolve())}
    assert value["executable"] == Path(sys.executable).resolve()


def test_a_statically_linked_interpreter_needs_no_loader_path(tmp_path, monkeypatch):
    base = tmp_path / "python"
    (base / "lib").mkdir(parents=True)
    monkeypatch.setattr(sys, "base_prefix", str(base))
    monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)
    value = runtime.sandboxed_interpreter()
    assert value["roots"] == [base.resolve()] and value["library_environment"] == {}


def test_artisan_sandboxes_receive_the_interpreter_loader_path(tmp_path):
    lib = tmp_path / "lib"
    lib.mkdir()
    write_root = tmp_path / "write"
    write_root.mkdir()
    runner = hooks.sandbox({"read_roots": [str(lib)], "write_root": str(write_root), "write_paths": None,
                            "library_environment": {"LD_LIBRARY_PATH": str(lib)},
                            "library_executables": [sys.executable]})
    assert runner.library_environment == {"LD_LIBRARY_PATH": str(lib)}
    assert runner.library_executables == (Path(sys.executable).resolve(),)
    # Contexts written before this change still build a sandbox.
    legacy = hooks.sandbox({"read_roots": [str(lib)], "write_root": str(write_root)})
    assert legacy.library_environment == {} and legacy.library_executables == ()
