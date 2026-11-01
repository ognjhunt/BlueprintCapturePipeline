"""The sandbox gets back what Kit's Python needs: its prefix and its libpython path."""
from __future__ import annotations

import sys
from pathlib import Path

from blueprint_pipeline import asset_python_runtime as runtime
from blueprint_pipeline import task_object_artisan_hooks as hooks


def test_interpreter_prefix_and_the_launcher_library_path_are_admitted(tmp_path, monkeypatch):
    """capture-447ae4f4: libpython was on the launcher path but not under the prefix.

    Keeping only entries under ``sys.base_prefix`` left the sandbox with no
    loader path, and the probe failed with ``libpython3.12.so.1.0: cannot open
    shared object file``. The whole launcher path goes back, as for Astra.
    """
    base = tmp_path / "isaac-sim/kit/python"
    base.mkdir(parents=True)
    version = f"{sys.version_info.major}.{sys.version_info.minor}"
    libs = tmp_path / "isaac-sim/kit"
    (libs / f"libpython{version}.so.1.0").write_text("fixture")
    plugins = tmp_path / "isaac-sim/kit/plugins"
    plugins.mkdir(parents=True)
    monkeypatch.setattr(sys, "base_prefix", str(base))
    monkeypatch.setattr(runtime, "_loaded_libpython_directories", lambda *_args, **_kwargs: [])
    monkeypatch.setenv("LD_LIBRARY_PATH", f"{plugins}:relative/path:{tmp_path / 'missing'}")
    value = runtime.sandboxed_interpreter()
    # base.parent holds libpython here; the launcher's plugin directory comes back too.
    assert value["roots"] == [base.resolve(), plugins.resolve(), libs.resolve()]
    assert value["library_environment"] == {"LD_LIBRARY_PATH": f"{plugins.resolve()}:{libs.resolve()}"}
    assert value["executable"] == Path(sys.executable).resolve()


def test_the_libpython_this_process_loaded_is_found_from_its_memory_map(tmp_path, monkeypatch):
    version = f"{sys.version_info.major}.{sys.version_info.minor}"
    libs = tmp_path / "opt/kit/libs"
    libs.mkdir(parents=True)
    libpython = libs / f"libpython{version}.so.1.0"
    libpython.write_text("fixture")
    maps = tmp_path / "maps"
    maps.write_text(
        "7f00-7f01 r--p 00000000 08:01 1 /usr/lib/x86_64-linux-gnu/libc.so.6\n"
        f"7f02-7f03 r-xp 00001000 08:01 2 {libpython}\n"
        "7f04-7f05 rw-p 00000000 00:00 0 \n"
        "7f06-7f07 r--p 00000000 08:01 3 [vdso]\n")
    assert runtime._loaded_libpython_directories(version, maps) == [libs.resolve()]
    assert runtime._loaded_libpython_directories(version, tmp_path / "absent") == []


def test_a_statically_linked_interpreter_needs_no_loader_path(tmp_path, monkeypatch):
    base = tmp_path / "python"
    (base / "lib").mkdir(parents=True)
    monkeypatch.setattr(sys, "base_prefix", str(base))
    # Whether the test host's own Python maps a libpython must not decide this.
    monkeypatch.setattr(runtime, "_loaded_libpython_directories", lambda *_args, **_kwargs: [])
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
