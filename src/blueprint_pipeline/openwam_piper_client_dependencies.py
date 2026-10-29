"""Select the exact isolated WebSocket client before any PiPER connection."""
from __future__ import annotations

import hashlib
import importlib
from pathlib import Path
import sys
import zipfile

from .openwam_piper_dependencies import verified_dependency_manifest


def activate_client_dependencies(runtime: Path | None = None) -> None:
    """This single-candidate process cannot reuse a different preloaded codec.

    Compare installed package bytes with the retained, exact wheel before
    replacing module-cache entries. Other candidate processes are untouched.
    """
    runtime = runtime or Path(__file__).resolve().parent.parent
    manifest = verified_dependency_manifest(runtime / "openwam-piper-dependencies.v1.json")
    target = runtime / "openwam_client"
    expected_members = set()
    for row in manifest["thin_client_wheels"]:
        wheel = runtime / "openwam_client_wheels" / row["filename"]
        if (wheel.is_symlink() or not wheel.is_file() or wheel.stat().st_size != row["size_bytes"]
                or hashlib.sha256(wheel.read_bytes()).hexdigest() != row["sha256"].removeprefix("sha256:")):
            raise ValueError("openwam_client_wheel_changed")
        with zipfile.ZipFile(wheel) as archive:
            for member in archive.infolist():
                if member.is_dir() or not member.filename.startswith("websockets/"):
                    continue
                expected_members.add(member.filename)
                name = Path(member.filename)
                if name.is_absolute() or ".." in name.parts:
                    raise ValueError("openwam_client_wheel_path_invalid")
                installed = target / name
                if (installed.is_symlink() or not installed.is_file()
                        or any(parent.is_symlink() for parent in installed.parents)
                        or installed.read_bytes() != archive.read(member)):
                    raise ValueError("openwam_client_installation_changed")
    actual = {path.relative_to(target).as_posix() for path in (target / "websockets").rglob("*") if not path.is_dir()}
    if actual != expected_members:
        raise ValueError("openwam_client_installation_members_changed")
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(target))
    for name in list(sys.modules):
        if name == "websockets" or name.startswith("websockets."):
            del sys.modules[name]
    importlib.invalidate_caches()
    websockets = importlib.import_module("websockets")
    if (websockets.__version__ != "16.0"
            or Path(websockets.__file__).resolve().parent != (target / "websockets").resolve()):
        raise ValueError("openwam_client_dependency_precedence_invalid")
