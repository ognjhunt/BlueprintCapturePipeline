#!/usr/bin/env python3
"""Require the real LeRobot loader to round-trip one release export."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from blueprint_pipeline.lerobot_export_validation import validate_lerobot_export


SOURCE_NAME = "native_lerobot_export_source_manifest.json"
SOURCE_SCHEMA = "blueprint.native_lerobot_export_source_manifest.v1"
MAX_SOURCE_SIZE = 32 * 1024 * 1024


def _repository_sha(root: Path) -> str | None:
    configured = str(os.environ.get("GITHUB_SHA") or "").strip().lower()
    if configured:
        return configured
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.stdout.strip().lower() if completed.returncode == 0 else None


def _export_tree_evidence(
    export_dir: Path,
) -> tuple[str | None, int, int, list[str], list[dict[str, object]]]:
    blockers: list[str] = []
    if export_dir.is_symlink():
        return None, 0, 0, ["native_lerobot_export_root_symlink"], []
    if not export_dir.is_dir():
        return None, 0, 0, ["native_lerobot_export_root_missing"], []
    rows: list[str] = []
    files: list[dict[str, object]] = []
    total_bytes = 0
    file_count = 0
    for path in sorted(export_dir.rglob("*")):
        if path.is_symlink():
            blockers.append("native_lerobot_export_contains_symlink")
            continue
        if not path.is_file():
            continue
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        size = path.stat().st_size
        relative_path = path.relative_to(export_dir).as_posix()
        digest_hex = digest.hexdigest()
        rows.append(f"{relative_path}\t{size}\t{digest_hex}")
        files.append(
            {
                "path": relative_path,
                "size": size,
                "sha256": f"sha256:{digest_hex}",
            }
        )
        total_bytes += size
        file_count += 1
    if file_count == 0:
        blockers.append("native_lerobot_export_contains_no_files")
        return None, total_bytes, file_count, blockers, files
    tree_digest = hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest()
    return f"sha256:{tree_digest}", total_bytes, file_count, blockers, files


def run_probe(
    *, export_dir: Path, root: Path, source_manifest_path: Path | None = None
) -> dict[str, Any]:
    blockers: list[str] = []
    if importlib.util.find_spec("lerobot") is None:
        blockers.append("native_lerobot_package_missing")
    report = dict(validate_lerobot_export(export_dir))
    export_label = export_dir.name or "export-root"
    # The durable lane artifact must not disclose a prepared runner's absolute
    # filesystem layout. The native loader still receives the resolved path;
    # only the serialized report is reduced to a non-sensitive basename.
    report["export_dir"] = export_label
    if report.get("status") != "passed":
        blockers.extend(f"export_validation:{item}" for item in report.get("blockers", []))
    if report.get("loader") != "lerobot_native+hermetic":
        blockers.append(f"native_lerobot_loader_not_used:{report.get('loader') or 'missing'}")
    if report.get("checks", {}).get("lerobot_native_load") != "passed":
        blockers.append("native_lerobot_load_not_passed")
    export_digest, export_bytes, export_file_count, digest_blockers, files = _export_tree_evidence(
        export_dir
    )
    blockers.extend(digest_blockers)
    repository_sha = _repository_sha(root)
    generated_at = datetime.now(timezone.utc).isoformat()
    source_digest: str | None = None
    source_size: int | None = None
    if source_manifest_path is None:
        blockers.append("native_lerobot_source_manifest_path_required")
    else:
        source_manifest = {
            "schema_version": SOURCE_SCHEMA,
            "generated_at": generated_at,
            "repository_sha": repository_sha,
            "export_dir": export_label,
            "export_file_count": export_file_count,
            "export_total_bytes": export_bytes,
            "export_tree_sha256": export_digest,
            "files": files,
            "validation_report": report,
            "claim_boundary": {
                "relative_file_manifest_only": True,
                "source_manifest_is_not_dataset_quality_proof": True,
            },
        }
        encoded = (json.dumps(source_manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
        if len(encoded) > MAX_SOURCE_SIZE:
            blockers.append("native_lerobot_source_manifest_oversize")
        else:
            try:
                source_manifest_path.parent.mkdir(parents=True, exist_ok=True)
                source_manifest_path.write_bytes(encoded)
            except OSError as exc:
                blockers.append(f"native_lerobot_source_manifest_write_failed:{type(exc).__name__}")
            else:
                source_digest = f"sha256:{hashlib.sha256(encoded).hexdigest()}"
                source_size = len(encoded)
    blockers = sorted(set(blockers))
    return {
        "schema_version": "blueprint.critical_capability_lane_evidence.v1",
        "lane_id": "native_lerobot_export",
        "evidence_schema_version": "blueprint.native_lerobot_export.v1",
        "generated_at": generated_at,
        "repository_sha": repository_sha,
        "status": "passed" if not blockers else "blocked",
        "executed": True,
        "skipped_count": 0,
        "export_dir": export_label,
        "export_file_count": export_file_count,
        "export_total_bytes": export_bytes,
        "export_tree_sha256": export_digest,
        "validation_report": report,
        "artifact_digests": {
            SOURCE_NAME: source_digest,
        }
        if source_digest
        else {},
        "artifact_sizes": {
            SOURCE_NAME: source_size,
        }
        if source_size is not None
        else {},
        "blockers": blockers,
        "claim_boundary": {
            "native_loadability_is_not_dataset_quality": True,
            "native_loadability_is_not_task_success": True,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    output = args.output.expanduser().absolute()
    result = run_probe(
        export_dir=args.export_dir.expanduser().absolute(),
        root=args.root.resolve(),
        source_manifest_path=output.parent / SOURCE_NAME,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[native-lerobot] status={result['status']} output={output}")
    for blocker in result["blockers"]:
        print(f"[native-lerobot] blocker={blocker}", file=sys.stderr)
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
