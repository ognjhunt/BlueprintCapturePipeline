"""Recover publication after a completed scene configuration provider run.

This path performs no provider work.  It preserves the original blocked result,
publishes the already-sealed six-stage output, and emits a separately digested
terminal receipt that the Website may use for a one-time blocked-to-completed
publication upgrade.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .common import utc_now_iso
from .decision_evidence_contracts import (
    canonical_digest,
    cross_runtime_canonical_digest,
)
from .task_evaluation_launch_terminal_evidence import (
    _scene_configuration_terminal_projection,
)
from .task_evaluation_launch_webapp_sync import sync_launch_receipt_to_webapp
from .task_evaluation_scene_configuration_bundle import (
    load_scene_configuration_provider_bundle_receipt,
)
from .task_evaluation_scene_configuration_publication import (
    RESULT_SCHEMA_VERSION as PUBLICATION_RESULT_SCHEMA_VERSION,
)
from .task_evaluation_scene_configuration_vast import (
    RESULT_SCHEMA_VERSION,
    _portable_construction_envelope,
    _publish_completed_configuration,
)
from .task_evaluation_scene_construction_queue import (
    recover_scene_construction_publication,
)


RECOVERY_SCHEMA_VERSION = (
    "task_evaluation_scene_configuration_publication_recovery.v1"
)
RECOVERED_RESULT_FILENAME = (
    "task_evaluation_scene_configuration_vast_result.publication_recovered.v1.json"
)
RECOVERED_LAUNCH_RECEIPT_FILENAME = "launch_receipt.publication_recovered.v1.json"


def _read(path: str | Path, *, code: str) -> tuple[Path, dict[str, Any]]:
    candidate = Path(path).expanduser()
    absolute = Path(os.path.abspath(candidate))
    try:
        source = candidate.resolve(strict=True)
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(code) from exc
    if (
        candidate.is_symlink()
        or source != absolute
        or not source.is_file()
        or not isinstance(value, dict)
    ):
        raise ValueError(code)
    return source, value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _artifact(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    return {
        "path": str(resolved),
        "exists": True,
        "digest": _sha256(resolved),
    }


def _write_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o750)
    payload = (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")
    descriptor = -1
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o440,
        )
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short publication recovery write")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def recover_completed_configuration_publication(
    *,
    bundle_receipt_path: str | Path,
    provider_result_path: str | Path,
    original_result_path: str | Path,
    original_launch_receipt_path: str | Path,
    queue_root: str | Path,
    output_root: str | Path,
    recovery_source_commit: str,
) -> dict[str, Any]:
    """Publish exact retained output and seal a no-provider recovery receipt."""

    if re.fullmatch(r"[0-9a-f]{40}", recovery_source_commit) is None:
        raise ValueError("scene_configuration_publication_recovery_source_invalid")
    receipt = load_scene_configuration_provider_bundle_receipt(bundle_receipt_path)
    provider_path, provider_result = _read(
        provider_result_path,
        code="scene_configuration_publication_recovery_provider_result_invalid",
    )
    original_path, original_result = _read(
        original_result_path,
        code="scene_configuration_publication_recovery_original_result_invalid",
    )
    launch_path, original_launch = _read(
        original_launch_receipt_path,
        code="scene_configuration_publication_recovery_launch_receipt_invalid",
    )
    prior_finalization = original_result.get("scene_construction_queue_finalization")
    provider_digest = str(provider_result.get("result_digest") or "")
    original_digest = str(original_result.get("result_digest") or "")
    original_launch_digest = str(original_launch.get("receipt_digest") or "")
    if (
        provider_result.get("schema_version")
        != "task_evaluation_scene_configuration_provider_result.v1"
        or provider_result.get("status") != "completed"
        or provider_result.get("blockers")
        or provider_digest
        != canonical_digest(provider_result, digest_field="result_digest")
        or original_result.get("schema_version") != RESULT_SCHEMA_VERSION
        or original_result.get("status") != "blocked"
        or original_result.get("configuration_completed") is not True
        or original_result.get("configured_scene_published") is not False
        or original_result.get("continuing_spend_from_this_run") is not False
        or original_result.get("execution_result_path") != str(provider_path)
        or original_result.get("stage_chain_result_digest")
        != (provider_result.get("stage_chain") or {}).get("result_digest")
        or original_digest
        != canonical_digest(original_result, digest_field="result_digest")
        or not isinstance(prior_finalization, Mapping)
        or original_launch.get("schema_version") != "task_evaluation_launch_receipt.v1"
        or original_launch.get("status") != "blocked"
        or original_launch_digest
        != cross_runtime_canonical_digest(
            original_launch, digest_field="receipt_digest"
        )
        or original_launch.get("source_commit") != original_result.get("source_commit")
        or receipt.get("source_commit") != original_result.get("source_commit")
        or receipt.get("run_id") != original_result.get("run_id")
        or receipt.get("bundle_sha256") != original_result.get("bundle_sha256")
        or provider_result.get("source_commit") != original_result.get("source_commit")
        or provider_result.get("run_id") != original_result.get("run_id")
        or (original_result.get("independent_watchdog") or {}).get(
            "provider_absence_confirmed"
        )
        is not True
        or (original_result.get("object_store_cleanup") or {}).get(
            "all_objects_absent"
        )
        is not True
    ):
        raise ValueError("scene_configuration_publication_recovery_binding_invalid")

    root = Path(output_root).expanduser().resolve()
    if root.exists() or root.is_symlink():
        raise ValueError("scene_configuration_publication_recovery_output_exists")
    root.mkdir(parents=True, mode=0o750)
    publication_root = root / "configured_scene_publication"
    publication = _publish_completed_configuration(
        receipt=receipt,
        execution=provider_result,
        extraction_root=provider_path.parent,
        output_root=publication_root,
    )
    if (
        publication.get("status") != "configured_scene_published"
        or publication.get("full_byte_service_account_readback_passed") is not True
    ):
        raise ValueError("scene_configuration_publication_recovery_publish_failed")

    recovered_result = dict(original_result)
    recovered_result.update(
        {
            "generated_at": utc_now_iso(),
            "status": "completed",
            "configured_scene_published": True,
            "configured_scene_revision_path": (
                publication.get("configured_scene_revision") or {}
            ).get("path"),
            "configured_scene_revision_reference": publication.get(
                "configured_scene_revision_reference"
            ),
            "configured_scene_revision_digest": publication.get(
                "configured_scene_revision_digest"
            ),
            "configured_scene_bundle_reference": publication.get(
                "configured_scene_bundle_reference"
            ),
            "task_thumbnail_reference": publication.get("task_thumbnail_reference"),
            "task_thumbnail_selection": publication.get("task_thumbnail_selection"),
            "task_thumbnail_selection_receipt_reference": publication.get(
                "task_thumbnail_selection_receipt_reference"
            ),
            "configured_scene_offering": publication.get("configured_scene_offering"),
            "publication_result_path": str(
                publication_root / f"{PUBLICATION_RESULT_SCHEMA_VERSION}.json"
            ),
            "publication_result_digest": publication.get("result_digest"),
            "full_byte_service_account_readback_passed": True,
            "blockers": [],
        }
    )
    recovered_result["publication_recovery"] = {
        "schema_version": RECOVERY_SCHEMA_VERSION,
        "status": "completed",
        "recovery_source_commit": recovery_source_commit,
        "provider_execution_repeated": False,
        "paid_execution_requested": False,
        "provider_mutation_performed": False,
        "original_configuration_result_digest": original_digest,
        "provider_result_digest": provider_digest,
        "original_terminal_receipt_digest": original_launch_digest,
    }
    finalization = recover_scene_construction_publication(
        queue_root=queue_root,
        envelope=_portable_construction_envelope(receipt),
        terminal_result=recovered_result,
        prior_finalization=prior_finalization,
    )
    recovered_result["scene_construction_queue_finalization"] = finalization
    recovered_result["result_digest"] = ""
    recovered_result["result_digest"] = canonical_digest(
        recovered_result, digest_field="result_digest"
    )
    recovered_result_path = root / RECOVERED_RESULT_FILENAME
    _write_exclusive(recovered_result_path, recovered_result)

    scene_configuration, blockers = _scene_configuration_terminal_projection(
        recovered_result
    )
    if scene_configuration is None or blockers:
        raise ValueError("scene_configuration_publication_recovery_terminal_invalid")
    recovery = {
        **recovered_result["publication_recovery"],
        "recovered_configuration_result_digest": recovered_result["result_digest"],
        "queue_finalization_digest": finalization["result_digest"],
        "recovery_digest": "",
    }
    recovery["recovery_digest"] = canonical_digest(
        recovery, digest_field="recovery_digest"
    )
    recovered_launch = dict(original_launch)
    recovered_launch.update(
        {
            "status": "completed",
            "blockers": [],
            "terminal_evidence": {
                "status": "passed",
                "result": _artifact(recovered_result_path),
                "artifacts": {
                    "configured_scene_revision_path": _artifact(
                        Path(recovered_result["configured_scene_revision_path"])
                    ),
                    "publication_result_path": _artifact(
                        Path(recovered_result["publication_result_path"])
                    ),
                    "execution_result_path": _artifact(provider_path),
                    "teardown_manifest_path": _artifact(
                        Path(recovered_result["teardown_manifest_path"])
                    ),
                },
                "scene_configuration": scene_configuration,
                "publication_recovery": recovery,
                "blockers": [],
            },
            "publication_recovery": recovery,
            "receipt_digest": "",
        }
    )
    recovered_launch["receipt_digest"] = cross_runtime_canonical_digest(
        recovered_launch, digest_field="receipt_digest"
    )
    recovered_launch_path = root / RECOVERED_LAUNCH_RECEIPT_FILENAME
    _write_exclusive(recovered_launch_path, recovered_launch)
    return {
        "schema_version": RECOVERY_SCHEMA_VERSION,
        "status": "completed",
        "provider_execution_repeated": False,
        "paid_execution_requested": False,
        "provider_mutation_performed": False,
        "original_result": _artifact(original_path),
        "original_launch_receipt": _artifact(launch_path),
        "provider_result": _artifact(provider_path),
        "recovered_result": _artifact(recovered_result_path),
        "recovered_launch_receipt": _artifact(recovered_launch_path),
        "configured_scene_revision_digest": recovered_result[
            "configured_scene_revision_digest"
        ],
        "configured_scene_offering_digest": recovered_result[
            "configured_scene_offering"
        ]["offering_digest"],
        "recovery_digest": recovery["recovery_digest"],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-receipt", required=True)
    parser.add_argument("--provider-result", required=True)
    parser.add_argument("--original-result", required=True)
    parser.add_argument("--original-launch-receipt", required=True)
    parser.add_argument("--queue-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--recovery-source-commit", required=True)
    parser.add_argument("--sync-webapp", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = recover_completed_configuration_publication(
        bundle_receipt_path=args.bundle_receipt,
        provider_result_path=args.provider_result,
        original_result_path=args.original_result,
        original_launch_receipt_path=args.original_launch_receipt,
        queue_root=args.queue_root,
        output_root=args.output_root,
        recovery_source_commit=args.recovery_source_commit,
    )
    if args.sync_webapp:
        _, receipt = _read(
            result["recovered_launch_receipt"]["path"],
            code="scene_configuration_publication_recovery_launch_receipt_invalid",
        )
        sync = sync_launch_receipt_to_webapp(receipt=receipt)
        if sync.get("status") != "succeeded":
            raise ValueError(
                "scene_configuration_publication_recovery_webapp_sync_failed:"
                + str(sync.get("reason") or sync.get("status") or "unknown")
            )
        result["webapp_sync"] = sync
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "RECOVERED_LAUNCH_RECEIPT_FILENAME",
    "RECOVERED_RESULT_FILENAME",
    "RECOVERY_SCHEMA_VERSION",
    "recover_completed_configuration_publication",
]
