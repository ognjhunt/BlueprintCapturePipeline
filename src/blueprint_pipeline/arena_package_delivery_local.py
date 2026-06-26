"""Local Arena package delivery command hook.

This command is a gated delivery hook for environments that do not yet have a
cloud upload/signed URL provider. It copies the Arena ``delivery_bundle`` into a
configured local delivery root and writes ``delivery_upload.command.json`` for
``blueprint-ingest-arena-results`` to consume.

It does not create signed URLs, perform cloud upload, verify customer
entitlement, or upgrade proof claims.
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Sequence

from .arena_result_ingest import CLAIM_BOUNDARY, DELIVERY_COMMAND_MANIFEST_SCHEMA_VERSION
from .common import ensure_dir, utc_now_iso, write_json


OUTPUT_FILENAME = "delivery_upload.command.json"
GATE_ENV = "BLUEPRINT_ALLOW_PACKAGE_DELIVERY_UPLOAD"
DELIVERY_ROOT_ENV = "BLUEPRINT_LOCAL_DELIVERY_ROOT"


def _string(value: Any) -> str:
    return str(value or "").strip()


def _truthy(value: Any) -> bool:
    return _string(value).lower() in {"1", "true", "yes", "on"}


def _relative_or_absolute(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def build_local_delivery_command_manifest(
    *,
    output_dir: str | Path = ".",
    delivery_root: str | Path | None = None,
) -> Dict[str, Any]:
    resolved_output = Path(output_dir).resolve()
    generated_at = utc_now_iso()
    root_text = _string(delivery_root or os.getenv(DELIVERY_ROOT_ENV))
    blockers: List[str] = []
    if not _truthy(os.getenv(GATE_ENV)):
        blockers.append(f"missing_env_{GATE_ENV}")
    if not root_text:
        blockers.append(f"missing_env_{DELIVERY_ROOT_ENV}")
    bundle_dir = resolved_output / "delivery_bundle"
    if not bundle_dir.is_dir():
        blockers.append("missing_delivery_bundle")

    local_access_paths: List[Dict[str, Any]] = []
    delivery_root_path = Path(root_text).expanduser().resolve() if root_text else None
    target_dir = None
    if not blockers and delivery_root_path:
        target_dir = delivery_root_path / resolved_output.name
        ensure_dir(target_dir)
        for source in sorted(path for path in bundle_dir.rglob("*") if path.is_file()):
            relative = source.relative_to(bundle_dir)
            target = target_dir / relative
            ensure_dir(target.parent)
            shutil.copy2(source, target)
            local_access_paths.append(
                {
                    "relative_path": str(relative),
                    "source_path": _relative_or_absolute(resolved_output, source),
                    "delivered_path": str(target),
                    "size_bytes": target.stat().st_size,
                }
            )
        if not local_access_paths:
            blockers.append("delivery_bundle_empty")

    manifest = {
        "schema_version": DELIVERY_COMMAND_MANIFEST_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": "local_delivery_ready_review_required" if not blockers else "blocked",
        "provider": "local_filesystem",
        "delivery_root": str(delivery_root_path) if delivery_root_path else None,
        "target_dir": str(target_dir) if target_dir else None,
        "blockers": blockers,
        "signed_urls": [],
        "local_access_paths": local_access_paths,
        "storage_upload_performed": False,
        "signed_access_performed": False,
        "entitlement_verified": False,
        "public_claim_upgrade_allowed": False,
        "proof_effect": "none",
        "claim_boundary": {
            **dict(CLAIM_BOUNDARY),
            "local_filesystem_delivery_performed": not blockers,
            "signed_delivery_access_proven": False,
            "delivery_access_is_deployment_approval": False,
            "package_delivery_is_deployment_approval": False,
            "deployment_approval_proven": False,
            "physical_robot_readiness_proven": False,
            "safety_validation_proven": False,
            "public_claim_upgrade_allowed": False,
        },
    }
    write_json(resolved_output / OUTPUT_FILENAME, manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Copy an Arena package delivery bundle to a local delivery root"
    )
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--delivery-root", default=None)
    args = parser.parse_args(argv)
    result = build_local_delivery_command_manifest(
        output_dir=args.output_dir,
        delivery_root=args.delivery_root,
    )
    print(f"[arena-local-delivery] manifest={Path(args.output_dir).resolve() / OUTPUT_FILENAME}")
    print(f"[arena-local-delivery] status={result['status']}")
    if result["blockers"]:
        print(f"[arena-local-delivery] blockers={len(result['blockers'])}")
    return 0 if result["status"] == "local_delivery_ready_review_required" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
