"""The pinned Cosmos/OpenWAM client wins over a host importer that ignores sys.path.

Isaac's Kit app resolves its bundled packages with a fast importer that does not
consult ``sys.path`` (on by default), and the Kit worker carries an older
websockets whose sync client has no keepalive (pi05_droid on capture-447,
2026-11-05; modelled here as 12.0). Both Cosmos runs on
capture-447 that reached policy load stopped on
``cosmos_policy_client_dependency_precedence_invalid`` inside that worker,
while the same install passed in the bare provisioning interpreter.
"""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import sys
import zipfile

import pytest


PINNED = ("websockets", "msgpack", "openpi_client", "typing_extensions")


class HostFastImporter:
    """Serves host packages from a fixed directory, never reading sys.path."""

    def __init__(self, root: Path):
        self.root = root

    def find_spec(self, fullname, path=None, target=None):
        if fullname.partition(".")[0] not in PINNED:
            return None
        parts = fullname.split(".")
        base = self.root.joinpath(*parts)
        if (base / "__init__.py").is_file():
            return importlib.util.spec_from_file_location(
                fullname, base / "__init__.py", submodule_search_locations=[str(base)])
        if base.with_suffix(".py").is_file():
            return importlib.util.spec_from_file_location(fullname, base.with_suffix(".py"))
        return None


def _drop_pinned_modules() -> None:
    for name in [name for name in sys.modules if name.partition(".")[0] in PINNED]:
        del sys.modules[name]


@pytest.fixture
def isolated_imports(monkeypatch):
    """Each test starts with none of the pinned names imported, as a fresh worker does.

    Without the first drop, a copy another test in the same process had already
    imported (``msgpack`` from the test venv, say) answers ``import`` before any
    finder runs, so the outcome depended on test order.
    """
    saved = {name: module for name, module in sys.modules.items() if name.partition(".")[0] in PINNED}
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(sys, "meta_path", list(sys.meta_path))
    monkeypatch.setattr(sys, "dont_write_bytecode", sys.dont_write_bytecode)
    _drop_pinned_modules()
    yield
    _drop_pinned_modules()
    sys.modules.update(saved)


def _files(version: str) -> dict[str, str]:
    return {
        "websockets/__init__.py": "from .version import version as __version__\n",
        "websockets/version.py": f"version = {version!r}\n",
        "websockets/sync/__init__.py": "",
        "websockets/sync/client.py": f"VERSION = {version!r}\n",
        "msgpack/__init__.py": "version = (1, 1, 2)\n",
        "openpi_client/__init__.py": "",
        "typing_extensions.py": "PINNED = True\n",
    }


def _install(tmp_path: Path, *, wheels: str, target: str, version: str = "16.0",
             only_websockets: bool = False) -> dict:
    files = {name: code for name, code in _files(version).items()
             if not only_websockets or name.startswith("websockets/")}
    wheel_root = tmp_path / wheels
    wheel_root.mkdir()
    wheel = wheel_root / "fixture-client.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        for name, code in files.items():
            archive.writestr(name, code)
    for name, code in files.items():
        installed = tmp_path / target / name
        installed.parent.mkdir(parents=True, exist_ok=True)
        installed.write_text(code)
    return {"filename": wheel.name, "size_bytes": wheel.stat().st_size,
            "sha256": hashlib.sha256(wheel.read_bytes()).hexdigest()}


def _host_websockets_12(tmp_path: Path) -> Path:
    """Isaac's bundled copy, with a module the pinned wheel does not ship."""
    root = tmp_path / "isaac-sim/extscache/omni.services.pip_archive/pip_prebundle"
    for name, code in {**_files("12.0"), "websockets/speedups.py": "HOST = True\n"}.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(code)
    return root

