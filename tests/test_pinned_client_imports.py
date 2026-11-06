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

from blueprint_pipeline import cosmos_droid_client_dependencies as cosmos_client
from blueprint_pipeline import cosmos_droid_provisioning
from blueprint_pipeline import openwam_piper_client_dependencies as openwam_client

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


@pytest.fixture
def isolated_imports(monkeypatch):
    saved = {name: module for name, module in sys.modules.items() if name.partition(".")[0] in PINNED}
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(sys, "meta_path", list(sys.meta_path))
    monkeypatch.setattr(sys, "dont_write_bytecode", sys.dont_write_bytecode)
    yield
    for name in [name for name in sys.modules if name.partition(".")[0] in PINNED]:
        del sys.modules[name]
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


def _activate_cosmos(tmp_path, monkeypatch, row):
    monkeypatch.setattr(cosmos_client, "verified_dependency_manifest", lambda path: {})
    monkeypatch.setattr(cosmos_droid_provisioning, "client_wheels", lambda manifest, minor: [row])
    cosmos_client.activate_client_dependencies(tmp_path)


def test_cosmos_client_wins_over_a_host_importer_that_ignores_sys_path(tmp_path, monkeypatch, isolated_imports):
    row = _install(tmp_path, wheels="cosmos_policy_client_wheels", target="cosmos_policy_client")
    host = _host_websockets_12(tmp_path)
    sys.meta_path.insert(0, HostFastImporter(host))
    import websockets as preloaded  # what the Kit worker held before policy load
    assert preloaded.__version__ == "12.0"

    _activate_cosmos(tmp_path, monkeypatch, row)

    target = (tmp_path / "cosmos_policy_client").resolve()
    import msgpack
    import openpi_client
    import typing_extensions
    import websockets
    import websockets.sync.client
    assert websockets.__version__ == "16.0" and websockets.sync.client.VERSION == "16.0"
    for module in (websockets, websockets.sync.client, msgpack, openpi_client, typing_extensions):
        assert Path(module.__file__).resolve().is_relative_to(target)
    # A name the pinned wheel lacks fails instead of mixing in the host copy.
    with pytest.raises(ModuleNotFoundError):
        import websockets.speedups  # noqa: F401


def test_openwam_client_wins_over_a_host_importer_that_ignores_sys_path(tmp_path, monkeypatch, isolated_imports):
    row = _install(tmp_path, wheels="openwam_client_wheels", target="openwam_client", only_websockets=True)
    sys.meta_path.insert(0, HostFastImporter(_host_websockets_12(tmp_path)))
    monkeypatch.setattr(openwam_client, "verified_dependency_manifest", lambda path: {"thin_client_wheels": [row]})

    openwam_client.activate_client_dependencies(tmp_path)

    import websockets.sync.client
    assert websockets.__version__ == "16.0"
    assert Path(websockets.__file__).resolve().parent == (tmp_path / "openwam_client/websockets").resolve()
    # OpenWAM pins only websockets; the host keeps its own copies of the rest.
    import msgpack
    assert Path(msgpack.__file__).resolve().is_relative_to(tmp_path / "isaac-sim")


def test_cosmos_precedence_is_still_enforced_on_the_pinned_install(tmp_path, monkeypatch, isolated_imports):
    # Bytes match their wheel, but the wheel is not the pinned 16.0 client.
    row = _install(tmp_path, wheels="cosmos_policy_client_wheels", target="cosmos_policy_client", version="15.0")
    with pytest.raises(ValueError, match="cosmos_policy_client_dependency_precedence_invalid"):
        _activate_cosmos(tmp_path, monkeypatch, row)


def test_repeated_activation_keeps_one_pinned_finder_first(tmp_path, monkeypatch, isolated_imports):
    row = _install(tmp_path, wheels="cosmos_policy_client_wheels", target="cosmos_policy_client")
    for _ in range(3):  # provisioning builds a fresh client on every readiness retry
        _activate_cosmos(tmp_path, monkeypatch, row)
    pinned = [finder for finder in sys.meta_path if type(finder).__name__ == "PinnedClientFinder"]
    assert len(pinned) == 1 and sys.meta_path[0] is pinned[0]
    assert sys.path.count(str(tmp_path / "cosmos_policy_client")) == 1
