"""Verify a sealed, isolated upstream runtime; no network install on workers."""
from __future__ import annotations

from pathlib import Path
from .decision_evidence_contracts import canonical_digest
from .task_object_authoring_backend import ARTISAN_BACKEND, ARTISAN_REVISION, ARTISAN_TREE
from .task_object_authoring_contracts import require, file_record, read_json

MANIFEST = "artisan_runtime_manifest.v1.json"
SCHEMA = "task_evaluation_scene_configuration_artisan_runtime.v1"


def validate_artisan_runtime(package: Path) -> dict:
    value = read_json(package / MANIFEST)
    require(value.get("schema_version") == SCHEMA and value.get("backend") == ARTISAN_BACKEND
            and value.get("upstream_revision") == ARTISAN_REVISION and value.get("upstream_tree") == ARTISAN_TREE
            and value.get("manifest_digest") == canonical_digest(value, digest_field="manifest_digest")
            and value.get("network_install_required") is False, "artisan_runtime_manifest_invalid")
    for row in value["files"]:
        relative = Path(row["relative_path"])
        require(not relative.is_absolute() and ".." not in relative.parts, "artisan_runtime_path_invalid")
        path = package / relative
        require(path.resolve().is_relative_to(package.resolve()), "artisan_runtime_path_escape")
        current = file_record(path)
        require(current["sha256"] == row["sha256"] and current["size_bytes"] == row["size_bytes"],
                "artisan_runtime_bytes_changed:" + relative.as_posix())
    require(file_record(package / "artisan_source/uv.lock")["sha256"] == value["lock_sha256"],
            "artisan_dependency_lock_changed")
    return value


def require_toolchain_artisan_runtime(toolchain: Path) -> dict:
    return validate_artisan_runtime(toolchain / "components/content_agents_rigid_replacement/package")
