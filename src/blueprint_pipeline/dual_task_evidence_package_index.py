"""Materialize a digest-bound top-level evidence package index.

Task-level episode indexes are already digest-bound.  This module closes the
package-level gap: the Finder-friendly top-level HTML is generated from verified
task indexes and shared manifests instead of being hand-maintained.
"""

from __future__ import annotations

import hashlib
import html
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

try:  # flat provider-bundle layout
    from decision_evidence_contracts import canonical_digest
except ModuleNotFoundError:  # repository package
    from .decision_evidence_contracts import canonical_digest


PACKAGE_INDEX_SCHEMA_VERSION = "adp_dual_task_evidence_package_index.v1"
PACKAGE_INDEX_FILENAME = "evidence_package_index.v1.json"
HTML_FILENAME = "index.html"
TASK_INDEX_SCHEMA_VERSION = "adp_manipulation_episode_evidence_index.v1"

SHARED_DIGEST_FIELDS_BY_SCHEMA = {
    "scene_replacement_cad_agent_matrix.v1": "matrix_digest",
    "simready_cad_agent_reference_manifest.v1": "manifest_digest",
    "cad_agent_reference_binding_audit.v1": "audit_digest",
    "third_scene_cad_agent_visual_comparison_binding.v1": "binding_digest",
    "third_scene_agent_cad_content_agents_bundle_matrix.v1": "receipt_digest",
    "adp_content_agents_execution_readiness.v1": "receipt_digest",
}


class DualTaskEvidencePackageIndexError(ValueError):
    """Fail-closed package-index validation error."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _load_json(path: Path, *, role: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DualTaskEvidencePackageIndexError(
            f"evidence_package_json_unreadable:{role}"
        ) from exc
    if not isinstance(value, dict):
        raise DualTaskEvidencePackageIndexError(
            f"evidence_package_json_not_mapping:{role}"
        )
    return value


def _resolve_inside(root: Path, relative_path: str, *, role: str) -> Path:
    if not relative_path:
        raise DualTaskEvidencePackageIndexError(
            f"evidence_package_path_missing:{role}"
        )
    unresolved = root / relative_path
    if unresolved.is_symlink():
        raise DualTaskEvidencePackageIndexError(
            f"evidence_package_symlink_forbidden:{role}"
        )
    resolved = unresolved.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise DualTaskEvidencePackageIndexError(
            f"evidence_package_path_outside_root:{role}"
        ) from exc
    if resolved.is_symlink() or not resolved.is_file():
        raise DualTaskEvidencePackageIndexError(
            f"evidence_package_file_missing:{role}"
        )
    return resolved


def _relative_link(from_dir: Path, target: Path) -> str:
    return os.path.relpath(target, from_dir)


def _verified_task_index(
    *,
    package_root: Path,
    task: Mapping[str, Any],
) -> dict[str, Any]:
    label = str(task.get("label") or "")
    relative_path = str(task.get("relative_path") or "")
    task_id = str(task.get("task_id") or "")
    if not label or not task_id:
        raise DualTaskEvidencePackageIndexError("evidence_package_task_invalid")
    path = _resolve_inside(package_root, relative_path, role=f"task:{task_id}")
    payload = _load_json(path, role=f"task:{task_id}")
    if payload.get("schema_version") != TASK_INDEX_SCHEMA_VERSION:
        raise DualTaskEvidencePackageIndexError(
            f"evidence_package_task_index_schema_invalid:{task_id}"
        )
    if payload.get("index_digest") != canonical_digest(
        payload, digest_field="index_digest"
    ):
        raise DualTaskEvidencePackageIndexError(
            f"evidence_package_task_index_digest_invalid:{task_id}"
        )
    identity = payload.get("run_identity")
    if not isinstance(identity, Mapping) or identity.get("task_id") != task_id:
        raise DualTaskEvidencePackageIndexError(
            f"evidence_package_task_identity_mismatch:{task_id}"
        )
    html_relative = relative_path.replace("episode_evidence_index.v1.json", "OPEN_ME_episode_evidence_index.html")
    html_path = _resolve_inside(package_root, html_relative, role=f"task_html:{task_id}")
    return {
        "label": label,
        "task_id": task_id,
        "relative_path": relative_path,
        "html_relative_path": html_relative,
        "index_digest": payload["index_digest"],
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
        "html_sha256": _sha256(html_path),
        "html_size_bytes": html_path.stat().st_size,
        "episode_count": payload.get("episode_count"),
        "typed_abstention_status": (
            (payload.get("typed_abstention") or {}).get("status")
            if isinstance(payload.get("typed_abstention"), Mapping)
            else None
        ),
    }


def _verified_shared_manifest(
    *,
    workspace_root: Path,
    package_root: Path,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    label = str(manifest.get("label") or "")
    relative_path = str(manifest.get("relative_path") or "")
    if not label:
        raise DualTaskEvidencePackageIndexError("evidence_package_manifest_invalid")
    path = _resolve_inside(workspace_root, relative_path, role=f"manifest:{label}")
    payload = _load_json(path, role=f"manifest:{label}")
    schema_version = str(payload.get("schema_version") or "")
    digest_field = SHARED_DIGEST_FIELDS_BY_SCHEMA.get(schema_version)
    if digest_field is None:
        raise DualTaskEvidencePackageIndexError(
            f"evidence_package_manifest_schema_not_admitted:{schema_version or 'missing'}"
        )
    digest = str(payload.get(digest_field) or "")
    if digest != canonical_digest(payload, digest_field=digest_field):
        raise DualTaskEvidencePackageIndexError(
            f"evidence_package_manifest_digest_invalid:{label}"
        )
    return {
        "label": label,
        "relative_path": relative_path,
        "html_link": _relative_link(package_root, path),
        "schema_version": schema_version,
        "digest_field": digest_field,
        "receipt_digest": digest,
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _render_html(index: Mapping[str, Any]) -> str:
    def esc(value: Any) -> str:
        return html.escape(str(value), quote=True)

    lines = [
        "<!doctype html>",
        "<html lang=\"en\">",
        "<head>",
        "  <meta charset=\"utf-8\">",
        f"  <title>{esc(index['title'])}</title>",
        "  <style>body{font-family:-apple-system,sans-serif;max-width:68rem;margin:2rem auto;padding:0 1rem}"
        "li{margin:.6rem 0}code{background:#f3f3f3;padding:.1rem .25rem}</style>",
        "</head>",
        "<body>",
        f"  <h1>{esc(index['title'])}</h1>",
        f"  <p><strong>Status:</strong> {esc(index['status_summary'])}</p>",
        f"  <p>{esc(index['blocker_summary'])}</p>",
        f"  <p>{esc(index['cad_summary'])}</p>",
        "  <ul>",
        '    <li><a href="shared_scene/README.md">Shared scene, rights, topology, registration, cost, and provider isolation</a></li>',
    ]
    for task in index["tasks"]:
        lines.append(
            "    <li>"
            f"<a href=\"{esc(task['html_relative_path'])}\">{esc(task['label'])}</a>"
            f" — <code>{esc(task['index_digest'])}</code>"
            "</li>"
        )
    for manifest in index["shared_manifests"]:
        lines.append(
            "    <li>"
            f"<a href=\"{esc(manifest['html_link'])}\">{esc(manifest['label'])}</a>"
            f" — <code>{esc(manifest['receipt_digest'])}</code>"
            "</li>"
        )
    lines.extend(
        [
            "  </ul>",
            "  <p>Overview media, if later admitted, is review-only and can never enter policy input or deterministic scoring. Simulator artifacts cannot establish physical truth.</p>",
            "</body>",
            "</html>",
        ]
    )
    return "\n".join(lines) + "\n"


def materialize_dual_task_evidence_package_index(
    *,
    workspace_root: str | Path,
    package_root: str | Path,
    title: str,
    status_summary: str,
    blocker_summary: str,
    cad_summary: str,
    task_indexes: Sequence[Mapping[str, Any]],
    shared_manifests: Sequence[Mapping[str, Any]],
    replace_existing: bool = False,
) -> dict[str, Any]:
    """Verify task/shared evidence and emit package-level JSON + HTML index."""

    workspace = Path(workspace_root).expanduser().resolve()
    package = Path(package_root).expanduser().resolve()
    if not workspace.is_dir() or not package.is_dir():
        raise DualTaskEvidencePackageIndexError("evidence_package_root_missing")
    try:
        package.relative_to(workspace)
    except ValueError as exc:
        raise DualTaskEvidencePackageIndexError(
            "evidence_package_root_outside_workspace"
        ) from exc
    if not title or not status_summary or not blocker_summary or not cad_summary:
        raise DualTaskEvidencePackageIndexError("evidence_package_summary_missing")
    if not task_indexes:
        raise DualTaskEvidencePackageIndexError("evidence_package_tasks_missing")
    if not shared_manifests:
        raise DualTaskEvidencePackageIndexError("evidence_package_manifests_missing")

    tasks = [
        _verified_task_index(package_root=package, task=task)
        for task in task_indexes
    ]
    if len({task["task_id"] for task in tasks}) != len(tasks):
        raise DualTaskEvidencePackageIndexError("evidence_package_duplicate_task")
    manifests = [
        _verified_shared_manifest(
            workspace_root=workspace, package_root=package, manifest=manifest
        )
        for manifest in shared_manifests
    ]
    if len({manifest["relative_path"] for manifest in manifests}) != len(manifests):
        raise DualTaskEvidencePackageIndexError("evidence_package_duplicate_manifest")

    payload: dict[str, Any] = {
        "schema_version": PACKAGE_INDEX_SCHEMA_VERSION,
        "title": title,
        "status_summary": status_summary,
        "blocker_summary": blocker_summary,
        "cad_summary": cad_summary,
        "tasks": tasks,
        "task_count": len(tasks),
        "shared_manifests": manifests,
        "shared_manifest_count": len(manifests),
        "overview_review_only": True,
        "simulator_artifacts_are_not_physical_truth": True,
        "package_index_digest": "",
    }
    payload["package_index_digest"] = canonical_digest(
        payload, digest_field="package_index_digest"
    )
    json_path = package / PACKAGE_INDEX_FILENAME
    html_path = package / HTML_FILENAME
    if (json_path.is_symlink() or html_path.is_symlink()):
        raise DualTaskEvidencePackageIndexError("evidence_package_index_symlink")
    if (json_path.exists() or html_path.exists()) and not replace_existing:
        raise DualTaskEvidencePackageIndexError("evidence_package_index_exists")
    _atomic_write(json_path, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    _atomic_write(html_path, _render_html(payload))
    return payload


__all__ = [
    "DualTaskEvidencePackageIndexError",
    "HTML_FILENAME",
    "PACKAGE_INDEX_FILENAME",
    "PACKAGE_INDEX_SCHEMA_VERSION",
    "materialize_dual_task_evidence_package_index",
]

