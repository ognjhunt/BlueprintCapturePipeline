"""Adopt a complete pinned-SSH recovery of a policy canary without re-entering the provider.

Moved unchanged out of ``task_evaluation_policy_canary_dispatcher``. The
dispatcher keeps ``_recovered_complete_policy_canary_result`` as a wrapper that
supplies its own readers, writers, error class and -- looked up at call time,
so tests that patch the dispatcher's name still apply -- the isolated-cell
aggregator.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import zipfile
from pathlib import Path
from typing import Any, Callable, Mapping

from .decision_evidence_contracts import canonical_digest
from .native_task_arena_policy_canary_session import (
    LEARNED_ROLLOUT_COUNT,
    PROVIDER_RESULT_FILENAME,
)


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def adopt_recovered_complete_result(
    *,
    root: Path,
    native_path: Path,
    adapter: Mapping[str, Any],
    authority: Mapping[str, Any],
    runtime_inputs: Mapping[str, Any],
    aggregate: Callable[..., dict[str, Any]],
    read_record: Callable[..., dict[str, Any]],
    sha256: Callable[[Path], str],
    record: Callable[[str | Path], dict[str, Any]],
    write_record: Callable[[Path, Mapping[str, Any]], Any],
    error_factory: Callable[[str], Exception],
) -> tuple[dict[str, Any], Path] | None:
    """Adopt a complete pinned-SSH recovery without re-entering the provider."""

    attempt_root_value = str(adapter.get("attempt_root") or "").strip()
    if not attempt_root_value:
        return None
    attempt_root = Path(attempt_root_value).expanduser().resolve()
    evidence_root = attempt_root / "immutable_execution"
    if native_path != evidence_root / PROVIDER_RESULT_FILENAME:
        return None
    command_path = attempt_root / "vast_provider_run" / "vast_provider_command_result.json"
    if not command_path.is_file() or not evidence_root.is_dir() or evidence_root.is_symlink():
        return None
    command = read_record(command_path, code="policy_canary_recovered_provider_command_invalid")
    download = _mapping(command.get("provider_output_download_manifest"))
    recovery = _mapping(download.get("ssh_recovery"))
    inspection = _mapping(command.get("provider_runtime_output_zip_inspection"))
    archive = Path(str(command.get("provider_runtime_output_zip_path") or "")).expanduser().resolve()
    if (
        command.get("provider_bundle_kind")
        != "native_task_arena_policy_canary_session"
        or command.get("provider_runtime_output_zip_received") is not True
        or recovery.get("status") != "completed"
        or recovery.get("strict_host_key_checking") is not True
        or recovery.get("streamed_to_disk") is not True
        or inspection.get("zip_present") is not True
        or inspection.get("mp4_count") != 120
        or archive.is_symlink()
        or not archive.is_file()
        or archive.stat().st_size != recovery.get("recovered_size_bytes")
        or sha256(archive) != recovery.get("recovered_sha256")
    ):
        return None
    child_paths = [
        evidence_root / "cell_runs" / f"{index:02d}" / PROVIDER_RESULT_FILENAME
        for index in range(10)
    ]
    if any(not path.is_file() or path.is_symlink() for path in child_paths):
        return None
    expected_members = {
        f"cell_runs/{index:02d}/{PROVIDER_RESULT_FILENAME}": child_paths[index]
        for index in range(10)
    }
    try:
        with zipfile.ZipFile(archive) as recovered_archive:
            names = recovered_archive.namelist()
            if (
                any(name.startswith("/") or ".." in Path(name).parts for name in names)
                or sum(name.lower().endswith(".mp4") for name in names) != 120
                or not set(expected_members).issubset(names)
            ):
                return None
            for member, extracted in expected_members.items():
                if hashlib.sha256(recovered_archive.read(member)).digest() != hashlib.sha256(
                    extracted.read_bytes()
                ).digest():
                    return None
    except (OSError, zipfile.BadZipFile, KeyError):
        return None
    children = [
        read_record(path, code="policy_canary_recovered_child_result_invalid")
        for path in child_paths
    ]
    lineage_modes = {
        str(child.get("construction_lineage_mode") or "") for child in children
    }
    if len(lineage_modes) != 1 or "" in lineage_modes:
        return None
    adoption_root = root / "recovered_provider_output_adoption"
    aggregate_path = adoption_root / PROVIDER_RESULT_FILENAME
    if aggregate_path.is_file():
        existing = read_record(
            aggregate_path, code="policy_canary_recovered_result_invalid"
        )
        if (
            existing.get("status")
            != "runtime_completed_unqualified_pending_closeout"
            or not isinstance(existing.get("episodes"), list)
            or len(existing["episodes"]) != LEARNED_ROLLOUT_COUNT
            or existing.get("result_digest")
            != canonical_digest(existing, digest_field="result_digest")
        ):
            raise error_factory(
                "policy_canary_recovered_result_invalid"
            )
        return existing, aggregate_path
    if adoption_root.exists():
        if adoption_root.is_symlink() or not adoption_root.is_dir():
            raise error_factory(
                "policy_canary_recovered_output_adoption_partial"
            )
        for index, original in enumerate(child_paths):
            adopted = (
                adoption_root
                / "cell_runs"
                / f"{index:02d}"
                / PROVIDER_RESULT_FILENAME
            )
            if (
                adopted.is_symlink()
                or not adopted.is_file()
                or sha256(adopted) != sha256(original)
            ):
                raise error_factory(
                    "policy_canary_recovered_output_adoption_partial"
                )
    else:
        try:
            shutil.copytree(evidence_root, adoption_root, copy_function=os.link)
        except OSError as exc:
            raise error_factory(
                "policy_canary_recovered_output_adoption_copy_failed"
            ) from exc
    adopted_children = [
        read_record(
            adoption_root / "cell_runs" / f"{index:02d}" / PROVIDER_RESULT_FILENAME,
            code="policy_canary_recovered_child_result_invalid",
        )
        for index in range(10)
    ]
    result = aggregate(
        authority=authority,
        inputs=runtime_inputs,
        child_results=adopted_children,
        output_root=adoption_root,
        construction_lineage_mode=next(iter(lineage_modes)),
    )
    write_record(aggregate_path, result)
    adoption_receipt = {
        "schema_version": "task_evaluation_policy_canary_recovered_output_adoption.v1",
        "status": "adopted_complete_provider_output",
        "run_id": authority["run_id"],
        "archive": record(archive),
        "archive_recovery": {
            "status": recovery["status"],
            "recovered_size_bytes": recovery["recovered_size_bytes"],
            "recovered_sha256": recovery["recovered_sha256"],
            "known_hosts_sha256": recovery.get("known_hosts_sha256"),
            "strict_host_key_checking": True,
            "streamed_to_disk": True,
        },
        "child_result_digests": [child["result_digest"] for child in children],
        "episode_count": len(result["episodes"]),
        "mp4_count": 120,
        "provider_mutation_performed": False,
        "automatic_retry_performed": False,
        "adoption_digest": "",
    }
    adoption_receipt["adoption_digest"] = canonical_digest(
        adoption_receipt, digest_field="adoption_digest"
    )
    write_record(root / "recovered_provider_output_adoption.json", adoption_receipt)
    return result, aggregate_path


__all__ = ["adopt_recovered_complete_result"]
