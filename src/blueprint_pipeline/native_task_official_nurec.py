"""Byte-bound NVIDIA render helper without its unrelated ML metrics extension."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).parent / "config/isaacsim_nurec_601"
MANIFEST_SHA256 = "6dfd4243db17d1c333fe760100eeb18fbc69def70e0dd6f9c92e1c99ac9e6c4e"


def verified_helper(root: Path = ROOT):
    raw = (root / "manifest.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != MANIFEST_SHA256:
        raise ValueError("official_nurec_helper_manifest_changed")
    manifest = json.loads(raw)
    for row in manifest["files"]:
        path = root / row["path"]
        data = path.read_bytes()
        if path.is_symlink() or len(data) != row["size_bytes"] or "sha256:" + hashlib.sha256(data).hexdigest() != row["sha256"]:
            raise ValueError("official_nurec_helper_bytes_changed:" + row["path"])
    spec = importlib.util.spec_from_file_location("blueprint_official_nurec_601", root / "rendering_setup.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, {**manifest, "manifest_sha256": "sha256:" + MANIFEST_SHA256,
                    "activation_method": "verified_official_python_helper",
                    "metrics_imported": False, "nurec_utils_extension_required": False}


def setup_for_rendering(stage):
    module, _provenance = verified_helper()
    return module.setup_for_rendering(stage)


def runtime_data_files():
    """Exact provider-archive members; SDK helper data is part of its closure."""
    verified_helper()
    from .native_task_pink_compatibility import MANIFEST, _manifest
    _manifest()
    paths = [ROOT / "manifest.json", *(ROOT / row["path"] for row in json.loads((ROOT / "manifest.json").read_text())["files"]), MANIFEST]
    package = Path(__file__).parent
    return {"blueprint_pipeline/" + path.relative_to(package).as_posix(): path for path in paths}
