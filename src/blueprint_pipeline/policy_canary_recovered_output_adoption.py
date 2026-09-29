"""Adopt a complete pinned-SSH recovery of a policy canary without re-entering the provider.

Moved out of ``task_evaluation_policy_canary_dispatcher``. The dispatcher keeps
``_recovered_complete_policy_canary_result`` as a wrapper that supplies its own
readers, writers, error class and -- looked up at call time, so tests that patch
the dispatcher's name still apply -- the isolated-cell aggregator.

Two sources. With the recovered ZIP on disk (download mode) the archive must
be the recovery record's bytes, hold exactly 120 MP4s, and carry the ten child
results byte for byte. In stream mode the ZIP was published to B2 and removed
behind its pointer, and only the contract's JSON is on the host: the evidence
root's member view must index an archive whose sha256 and size are the
recovery record's, with 120 MP4 members. The contract leaves the ten child
results in the archive (review I7), so each is read through the view -- one
range request, checked against its index CRC-32 and SHA-256 (PR B review M7);
a child that is on disk (an older layout) must hash to its index member
instead. Bytes that are not the indexed member's make the recovery
unadoptable, as a mismatch does in download mode; a read that fails for any
other reason is typed (``policy_canary_recovered_output_member_read_failed``)
rather than read as "not a complete recovery". No other member byte is read.

Adoption copies (review I8). The evidence tree is copied into
``recovered_provider_output_adoption/``: every file the aggregator could
rewrite -- anything that is not bulk -- as a fresh, writable copy, never a hard
link, which would write through to the evidence and fails on ingested ``0440``
members. Bulk files -- media and binaries, and the policy requests the GC
treats as bulk -- are hard links, which the aggregator never rewrites; a link
that fails is the adoption's copy failure, never a silent full copy (a
streamed tree has no bulk file on disk). By reference, the aggregator inventories the archive's remaining members
from the index, so the aggregate is download mode's, and a sibling view
descriptor lets later readers of the adoption root reach those members.
"""

from __future__ import annotations

import hashlib
import json
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
from .provider_output_member_index import BULK_EXTENSIONS

ADOPTION_DIRNAME = "recovered_provider_output_adoption"
EXPECTED_MP4_COUNT = 120
# The Quick-10 packer keeps any member but the aggregate only up to 100 MB.
MAXIMUM_CHILD_RESULT_BYTES = 128 * 1024**2


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _adoption_copier(evidence_root: Path) -> Callable[[str, str], str]:
    """Hard-link bulk evidence; copy what the aggregator could rewrite, writable."""

    def copy(source: str, destination: str) -> str:
        relative = Path(source).relative_to(evidence_root)
        if relative.suffix.lower() in BULK_EXTENSIONS or "policy-requests" in relative.parts:
            os.link(source, destination)  # a failure fails the adoption, as it always did
            return destination
        shutil.copyfile(source, destination)
        return destination
    return copy


def _local_members_match(archive: Path, expected_members: Mapping[str, Path]) -> bool:
    try:
        with zipfile.ZipFile(archive) as recovered_archive:
            names = recovered_archive.namelist()
            if (
                any(name.startswith("/") or ".." in Path(name).parts for name in names)
                or sum(name.lower().endswith(".mp4") for name in names) != EXPECTED_MP4_COUNT
                or not set(expected_members).issubset(names)
            ):
                return False
            for member, extracted in expected_members.items():
                if hashlib.sha256(recovered_archive.read(member)).digest() != hashlib.sha256(
                    extracted.read_bytes()
                ).digest():
                    return False
    except (OSError, zipfile.BadZipFile, KeyError):
        return False
    return True


def _indexed_children(view, recovery: Mapping[str, Any], expected_members: Mapping[str, Path],
                      error_factory: Callable[[str], Exception], adoption_root: Path) -> list[bytes] | None:
    """The ten children's bytes when the indexed archive is the recovery's, else None.

    A child comes from the evidence root when an older layout kept it there, else
    from an earlier pass's adopted copy that still hashes to its index member,
    else through the view; a replay therefore reads nothing from B2 again.
    """
    from .provider_output_member_view import CONTENT_MISMATCH_CODES, ProviderOutputMemberViewError

    archive = view.index["archive"]
    files = {row["path"]: row for row in view.index["members"] if row["kind"] == "file"}
    if (archive["sha256"] != recovery.get("recovered_sha256")
            or archive["size"] != recovery.get("recovered_size_bytes")
            or sum(path.lower().endswith(".mp4") for path in files) != EXPECTED_MP4_COUNT):
        return None
    children = []
    for member, extracted in expected_members.items():
        row = files.get(member)
        if row is None:
            return None
        adopted = adoption_root / member
        if extracted.is_file() and not extracted.is_symlink():
            data = extracted.read_bytes()
        elif (adopted.is_file() and not adopted.is_symlink() and adopted.stat().st_size == row["size"]
              and "sha256:" + hashlib.sha256(adopted.read_bytes()).hexdigest() == row["sha256"]):
            data = adopted.read_bytes()
        else:
            try:
                data = view.read_member(member, maximum_bytes=MAXIMUM_CHILD_RESULT_BYTES)
            except ProviderOutputMemberViewError as exc:
                if str(exc) in CONTENT_MISMATCH_CODES:
                    return None
                raise error_factory("policy_canary_recovered_output_member_read_failed") from None
        if ("sha256:" + hashlib.sha256(data).hexdigest(), len(data)) != (row["sha256"], row["size"]):
            return None
        children.append(data)
    return children


def _child_record(data: bytes, error_factory: Callable[[str], Exception]) -> dict[str, Any]:
    try:
        value = json.loads(data)
    except (UnicodeError, ValueError):
        value = None
    if not isinstance(value, dict):
        raise error_factory("policy_canary_recovered_child_result_invalid")
    return value


def _adoption_view(view, adoption_root: Path, evidence_root: Path, error_factory) -> dict:
    """Write (once) the adoption root's sibling descriptor over the evidence root's index."""
    from .provider_output_member_view import ProviderOutputMemberViewError, write_member_view_descriptor

    try:
        return write_member_view_descriptor(
            evidence_root=adoption_root,
            index_path=evidence_root.parent / view.descriptor["member_index"]["path"],
            ingestion_receipt_path=evidence_root.parent / view.descriptor["ingestion_receipt_path"])
    except ProviderOutputMemberViewError:
        raise error_factory("policy_canary_recovered_output_adoption_view_unbound") from None


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
        or inspection.get("mp4_count") != EXPECTED_MP4_COUNT
    ):
        return None
    view = None
    if archive.is_symlink() or archive.exists():
        # Download mode: the recovered ZIP itself is the proof.
        if (
            archive.is_symlink()
            or not archive.is_file()
            or archive.stat().st_size != recovery.get("recovered_size_bytes")
            or sha256(archive) != recovery.get("recovered_sha256")
        ):
            return None
    else:
        # Stream mode: the ZIP was promoted and removed; its member index is the proof.
        from .provider_output_member_view import ProviderOutputMemberViewError, open_member_view

        try:
            view = open_member_view(evidence_root)
        except ProviderOutputMemberViewError:
            raise error_factory("policy_canary_recovered_output_member_view_invalid") from None
        if view is None:
            return None
    child_paths = [
        evidence_root / "cell_runs" / f"{index:02d}" / PROVIDER_RESULT_FILENAME
        for index in range(10)
    ]
    expected_members = {
        f"cell_runs/{index:02d}/{PROVIDER_RESULT_FILENAME}": child_paths[index]
        for index in range(10)
    }
    child_data: list[bytes] | None = None
    if view is None:
        if any(not path.is_file() or path.is_symlink() for path in child_paths):
            return None
        if not _local_members_match(archive, expected_members):
            return None
        children = [
            read_record(path, code="policy_canary_recovered_child_result_invalid")
            for path in child_paths
        ]
    else:
        child_data = _indexed_children(view, recovery, expected_members, error_factory, root / ADOPTION_DIRNAME)
        if child_data is None:
            return None
        children = [_child_record(data, error_factory) for data in child_data]
    lineage_modes = {
        str(child.get("construction_lineage_mode") or "") for child in children
    }
    if len(lineage_modes) != 1 or "" in lineage_modes:
        return None
    adoption_root = root / ADOPTION_DIRNAME
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
        if view is not None:
            _adoption_view(view, adoption_root, evidence_root, error_factory)
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
            expected = (sha256(original) if child_data is None
                        else "sha256:" + hashlib.sha256(child_data[index]).hexdigest())
            if (
                adopted.is_symlink()
                or not adopted.is_file()
                or sha256(adopted) != expected
            ):
                raise error_factory(
                    "policy_canary_recovered_output_adoption_partial"
                )
    else:
        try:
            shutil.copytree(evidence_root, adoption_root, copy_function=_adoption_copier(evidence_root))
            for index, data in enumerate(child_data or []):
                # Children read through the view: fresh, writable copies (review I8).
                adopted = adoption_root / "cell_runs" / f"{index:02d}" / PROVIDER_RESULT_FILENAME
                if not adopted.is_file():
                    adopted.parent.mkdir(parents=True, exist_ok=True)
                    adopted.write_bytes(data)
        except OSError as exc:
            raise error_factory(
                "policy_canary_recovered_output_adoption_copy_failed"
            ) from exc
    descriptor = None
    archive_members: dict[str, dict[str, Any]] = {}
    if view is not None:
        descriptor = _adoption_view(view, adoption_root, evidence_root, error_factory)
        archive_members = {row["path"]: {"size_bytes": row["size"], "sha256": row["sha256"]}
                           for row in view.index["members"] if row["kind"] == "file"}
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
        **({"archive_members": archive_members} if view is not None else {}),
    )
    write_record(aggregate_path, result)
    adoption_receipt = {
        "schema_version": "task_evaluation_policy_canary_recovered_output_adoption.v1",
        "status": "adopted_complete_provider_output",
        "run_id": authority["run_id"],
        "archive": record(archive) if view is None else {
            "location": "durable_archive",
            "sha256": view.index["archive"]["sha256"],
            "size_bytes": view.index["archive"]["size"],
            "durable_reference": view.index["archive"]["durable_reference"],
            "member_index_digest": view.index["index_digest"],
        },
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
        "mp4_count": EXPECTED_MP4_COUNT,
        **({"member_view": {"path": f"{ADOPTION_DIRNAME}.member_view.v1.json",
                            "view_digest": descriptor["view_digest"]}} if descriptor is not None else {}),
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
