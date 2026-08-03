"""Deterministic, candidate-only transport for canonical 3DGS campaigns.

The trusted preparation host and the Windows/Linux trainer workers do not
share a filesystem.  This module packages the exact execution plan and the
byte-bound candidate-only COLMAP dataset into one immutable ZIP, then verifies
and extracts it on a worker.  The package carries no credentials, hidden
pixels, provider authority, or quality claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import tempfile
from typing import Any, Mapping, Sequence
import zipfile

from .canonical_3dgs_pipeline import (
    PLAN_SCHEMA,
    Canonical3DGSPipelineError,
    verify_canonical_3dgs_plan_inputs,
)
from .decision_evidence_contracts import canonical_digest, canonical_json


BUNDLE_SCHEMA = "canonical_3dgs_transport_bundle.v1"
EXTRACTION_SCHEMA = "canonical_3dgs_transport_extraction.v1"
PLAN_MEMBER = "campaign/canonical_3dgs_execution_plan.json"
MANIFEST_MEMBER = "campaign/canonical_3dgs_transport_manifest.json"
DATASET_PREFIX = "campaign/dataset/"
MAX_MEMBER_COUNT = 50_000
MAX_MEMBER_BYTES = 16 * 1024**3
MAX_TOTAL_BYTES = 200 * 1024**3
_FORBIDDEN_PARTS = {"evaluator_hidden", "held_out", "heldout", "hidden_heldout"}
_SECRET_SUFFIXES = {".env", ".key", ".pem", ".p12", ".pfx"}
_SECRET_MARKERS = {"credential", "credentials", "password", "secret", "token"}


class Canonical3DGSTransportError(ValueError):
    def __init__(self, codes: Sequence[str]) -> None:
        self.codes = tuple(sorted(set(str(code) for code in codes if str(code))))
        super().__init__(";".join(self.codes))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _digest(value: Any) -> bool:
    text = str(value or "")
    return (
        len(text) == 71
        and text.startswith("sha256:")
        and all(character in "0123456789abcdef" for character in text[7:])
    )


def _portable(value: Any) -> PurePosixPath | None:
    text = str(value or "").replace("\\", "/")
    path = PurePosixPath(text)
    if (
        not text
        or path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
        or ":" in path.parts[0]
    ):
        return None
    return path


def _forbidden(path: PurePosixPath) -> bool:
    normalized = [part.casefold().replace("-", "_") for part in path.parts]
    return (
        any(part in _FORBIDDEN_PARTS for part in normalized)
        or path.suffix.casefold() in _SECRET_SUFFIXES
        or any(marker in part for part in normalized for marker in _SECRET_MARKERS)
    )


def _write_zip_member(archive: zipfile.ZipFile, name: str, source: Path | bytes) -> None:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    info.create_system = 3
    info.external_attr = (stat.S_IFREG | 0o644) << 16
    if isinstance(source, bytes):
        archive.writestr(info, source)
        return
    with source.open("rb") as input_stream, archive.open(
        info, "w", force_zip64=True
    ) as output_stream:
        shutil.copyfileobj(input_stream, output_stream, length=1024 * 1024)


def _immutable_file(path: Path, source: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not path.is_file() or _sha256(path) != _sha256(source):
            raise Canonical3DGSTransportError(["transport_output_immutable_conflict"])
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        shutil.copyfile(source, temporary)
        try:
            os.link(temporary, path)
        except FileExistsError:
            if not path.is_file() or _sha256(path) != _sha256(temporary):
                raise Canonical3DGSTransportError(
                    ["transport_output_immutable_conflict"]
                )
    finally:
        temporary.unlink(missing_ok=True)


def _immutable_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = (canonical_json(dict(value)) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise Canonical3DGSTransportError(["transport_output_immutable_conflict"])
        return
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != payload:
                raise Canonical3DGSTransportError(
                    ["transport_output_immutable_conflict"]
                )
    finally:
        temporary.unlink(missing_ok=True)


def validate_canonical_3dgs_transport_receipt(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    receipt = json.loads(canonical_json(dict(value)))
    errors: list[str] = []
    if receipt.get("schema_version") != BUNDLE_SCHEMA or receipt.get("status") != "compiled":
        errors.append("transport_receipt_schema_or_status_invalid")
    if receipt.get("receipt_digest") != canonical_digest(
        receipt, digest_field="receipt_digest"
    ):
        errors.append("transport_receipt_digest_mismatch")
    for key in (
        "transport_bundle_digest",
        "transport_manifest_digest",
        "canonical_3dgs_execution_plan_digest",
        "worker_python_package_digest",
        "colmap_training_dataset_digest",
        "source_capture_digest",
        "frozen_split_digest",
    ):
        if not _digest(receipt.get(key)):
            errors.append(f"transport_receipt_{key}_invalid")
    members = receipt.get("dataset_members")
    if (
        not isinstance(members, list)
        or not members
        or receipt.get("dataset_member_count") != len(members)
        or len(members) > MAX_MEMBER_COUNT
    ):
        errors.append("transport_receipt_members_invalid")
    for key, expected in (
        ("hidden_heldout_pixels_included", False),
        ("raw_secret_values_included", False),
        ("provider_allocation_performed", False),
        ("paid_execution_authorized_by_bundle", False),
    ):
        if receipt.get(key) is not expected:
            errors.append(f"transport_receipt_boundary_invalid:{key}")
    if receipt.get("proof_effect") != "none":
        errors.append("transport_receipt_proof_effect_invalid")
    if errors:
        raise Canonical3DGSTransportError(errors)
    return receipt


def compile_canonical_3dgs_transport_bundle(
    *,
    plan: Mapping[str, Any],
    dataset_root: str | Path,
    bundle_path: str | Path,
    receipt_path: str | Path,
) -> dict[str, Any]:
    """Compile one byte-reproducible worker transport without uploading it."""

    try:
        root = verify_canonical_3dgs_plan_inputs(plan=plan, dataset_root=dataset_root)
    except Canonical3DGSPipelineError as exc:
        raise Canonical3DGSTransportError(
            [f"transport_plan_or_dataset_invalid:{code}" for code in exc.codes]
        ) from exc
    sources: list[tuple[Path, dict[str, Any]]] = []
    for raw in plan.get("input_artifacts") or []:
        if not isinstance(raw, Mapping):
            raise Canonical3DGSTransportError(["transport_input_member_invalid"])
        relative = _portable(raw.get("relative_path"))
        if relative is None or _forbidden(relative):
            raise Canonical3DGSTransportError(
                ["transport_input_member_path_forbidden"]
            )
        source = root.joinpath(*relative.parts)
        size = source.stat().st_size
        if size > MAX_MEMBER_BYTES:
            raise Canonical3DGSTransportError(["transport_input_member_oversized"])
        sources.append(
            (
                source,
                {
                    "relative_path": relative.as_posix(),
                    "archive_path": DATASET_PREFIX + relative.as_posix(),
                    "digest": raw["digest"],
                    "bytes": size,
                },
            )
        )
    if not sources or len(sources) > MAX_MEMBER_COUNT:
        raise Canonical3DGSTransportError(["transport_input_member_count_invalid"])
    archive_paths = [row["archive_path"] for _, row in sources]
    if len(archive_paths) != len(set(archive_paths)):
        raise Canonical3DGSTransportError(["transport_input_member_duplicate"])
    total_bytes = sum(row["bytes"] for _, row in sources)
    if total_bytes > MAX_TOTAL_BYTES:
        raise Canonical3DGSTransportError(["transport_input_total_oversized"])
    plan_digest = str(plan["canonical_3dgs_execution_plan_digest"])
    manifest = {
        "schema_version": BUNDLE_SCHEMA,
        "canonical_3dgs_execution_plan_digest": plan_digest,
        "worker_python_package_digest": plan["worker_python_package_digest"],
        "colmap_training_dataset_digest": plan["colmap_training_dataset_digest"],
        "source_capture_digest": plan["source_capture_digest"],
        "source_commit_sha": plan["source_commit_sha"],
        "frozen_split_digest": plan["frozen_split_digest"],
        "primary_method_id": plan["primary_method_id"],
        "comparison_method_ids": list(plan["comparison_method_ids"]),
        "plan_archive_path": PLAN_MEMBER,
        "dataset_root_archive_path": DATASET_PREFIX.rstrip("/"),
        "dataset_members": [row for _, row in sorted(sources, key=lambda item: item[1]["archive_path"])],
        "dataset_member_count": len(sources),
        "dataset_total_bytes": total_bytes,
        "hidden_heldout_pixels_included": False,
        "raw_secret_values_included": False,
        "provider_allocation_performed": False,
        "paid_execution_authorized_by_bundle": False,
        "proof_effect": "none",
        "claim_ceiling": "candidate_trainer_transport_only",
    }
    manifest["transport_manifest_digest"] = canonical_digest(
        manifest, digest_field="transport_manifest_digest"
    )
    destination = Path(bundle_path).expanduser().resolve()
    receipt_destination = Path(receipt_path).expanduser().resolve()
    if destination == receipt_destination or destination.is_symlink() or receipt_destination.is_symlink():
        raise Canonical3DGSTransportError(["transport_output_path_invalid"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(tempfile.mkdtemp(prefix=".canonical-3dgs-transport-", dir=destination.parent))
    try:
        temporary_bundle = temporary_root / "canonical_3dgs_transport.zip"
        plan_bytes = (canonical_json(dict(plan)) + "\n").encode("utf-8")
        manifest_bytes = (canonical_json(manifest) + "\n").encode("utf-8")
        with zipfile.ZipFile(temporary_bundle, "w", allowZip64=True) as archive:
            _write_zip_member(archive, MANIFEST_MEMBER, manifest_bytes)
            _write_zip_member(archive, PLAN_MEMBER, plan_bytes)
            for source, row in sorted(sources, key=lambda item: item[1]["archive_path"]):
                _write_zip_member(archive, row["archive_path"], source)
        receipt = {
            **manifest,
            "status": "compiled",
            "transport_bundle_digest": _sha256(temporary_bundle),
            "transport_bundle_bytes": temporary_bundle.stat().st_size,
        }
        receipt["receipt_digest"] = canonical_digest(
            receipt, digest_field="receipt_digest"
        )
        validate_canonical_3dgs_transport_receipt(receipt)
        _immutable_file(destination, temporary_bundle)
        _immutable_json(receipt_destination, receipt)
        return receipt
    finally:
        shutil.rmtree(temporary_root, ignore_errors=True)


def _validated_archive_members(
    archive: zipfile.ZipFile, receipt: Mapping[str, Any]
) -> dict[str, zipfile.ZipInfo]:
    expected = {MANIFEST_MEMBER, PLAN_MEMBER} | {
        str(row["archive_path"]) for row in receipt["dataset_members"]
    }
    infos = archive.infolist()
    names = [info.filename for info in infos]
    errors: list[str] = []
    if len(names) != len(set(names)) or set(names) != expected:
        errors.append("transport_archive_member_set_invalid")
    if len(infos) > MAX_MEMBER_COUNT + 2:
        errors.append("transport_archive_member_count_invalid")
    total = 0
    for info in infos:
        portable = _portable(info.filename)
        file_type = (info.external_attr >> 16) & 0o170000
        if portable is None or info.is_dir() or file_type == stat.S_IFLNK:
            errors.append("transport_archive_member_type_or_path_invalid")
        if info.compress_type != zipfile.ZIP_STORED:
            errors.append("transport_archive_compression_invalid")
        if info.file_size > MAX_MEMBER_BYTES:
            errors.append("transport_archive_member_oversized")
        total += info.file_size
    if total > MAX_TOTAL_BYTES:
        errors.append("transport_archive_total_oversized")
    if errors:
        raise Canonical3DGSTransportError(errors)
    return {info.filename: info for info in infos}


def extract_canonical_3dgs_transport_bundle(
    *,
    bundle_path: str | Path,
    receipt: Mapping[str, Any],
    output_root: str | Path,
) -> dict[str, Any]:
    """Verify and atomically materialize one transport on an admitted worker."""

    accepted = validate_canonical_3dgs_transport_receipt(receipt)
    bundle = Path(bundle_path).expanduser().resolve()
    if bundle.is_symlink() or not bundle.is_file() or _sha256(bundle) != accepted[
        "transport_bundle_digest"
    ]:
        raise Canonical3DGSTransportError(["transport_bundle_digest_mismatch"])
    root = Path(output_root).expanduser().resolve()
    if root.is_symlink():
        raise Canonical3DGSTransportError(["transport_extraction_root_symlink_forbidden"])
    root.mkdir(parents=True, exist_ok=True)
    final = root / accepted["transport_bundle_digest"].removeprefix("sha256:")
    temporary = Path(tempfile.mkdtemp(prefix=".canonical-3dgs-extract-", dir=root))
    try:
        with zipfile.ZipFile(bundle, "r") as archive:
            infos = _validated_archive_members(archive, accepted)
            for name in sorted(infos):
                target = temporary.joinpath(*PurePosixPath(name).parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(infos[name], "r") as source, target.open("xb") as destination:
                    shutil.copyfileobj(source, destination, length=1024 * 1024)
        manifest_path = temporary.joinpath(*PurePosixPath(MANIFEST_MEMBER).parts)
        plan_path = temporary.joinpath(*PurePosixPath(PLAN_MEMBER).parts)
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise Canonical3DGSTransportError(
                ["transport_extracted_control_json_invalid"]
            ) from exc
        if (
            manifest.get("transport_manifest_digest")
            != canonical_digest(manifest, digest_field="transport_manifest_digest")
            or manifest.get("transport_manifest_digest")
            != accepted["transport_manifest_digest"]
            or plan.get("schema_version") != PLAN_SCHEMA
            or plan.get("canonical_3dgs_execution_plan_digest")
            != canonical_digest(plan, digest_field="canonical_3dgs_execution_plan_digest")
            or plan.get("canonical_3dgs_execution_plan_digest")
            != accepted["canonical_3dgs_execution_plan_digest"]
        ):
            raise Canonical3DGSTransportError(
                ["transport_extracted_control_binding_invalid"]
            )
        dataset = temporary / "campaign/dataset"
        try:
            verify_canonical_3dgs_plan_inputs(plan=plan, dataset_root=dataset)
        except Canonical3DGSPipelineError as exc:
            raise Canonical3DGSTransportError(
                [f"transport_extracted_dataset_invalid:{code}" for code in exc.codes]
            ) from exc
        extraction = {
            "schema_version": EXTRACTION_SCHEMA,
            "status": "verified_and_extracted",
            "transport_bundle_digest": accepted["transport_bundle_digest"],
            "transport_manifest_digest": accepted["transport_manifest_digest"],
            "canonical_3dgs_execution_plan_digest": accepted[
                "canonical_3dgs_execution_plan_digest"
            ],
            "colmap_training_dataset_digest": accepted[
                "colmap_training_dataset_digest"
            ],
            "plan_relative_path": PLAN_MEMBER,
            "dataset_root_relative_path": DATASET_PREFIX.rstrip("/"),
            "hidden_heldout_pixels_included": False,
            "provider_allocation_performed": False,
            "paid_execution_authorized_by_bundle": False,
            "proof_effect": "none",
        }
        extraction["extraction_digest"] = canonical_digest(
            extraction, digest_field="extraction_digest"
        )
        _immutable_json(temporary / "canonical_3dgs_transport_extraction.json", extraction)
        if final.exists():
            existing = final / "canonical_3dgs_transport_extraction.json"
            if not existing.is_file() or json.loads(existing.read_text(encoding="utf-8")) != extraction:
                raise Canonical3DGSTransportError(
                    ["transport_extraction_immutable_conflict"]
                )
            return extraction
        os.replace(temporary, final)
        return extraction
    finally:
        if temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    package = commands.add_parser("package")
    package.add_argument("--plan", required=True)
    package.add_argument("--dataset-root", required=True)
    package.add_argument("--bundle", required=True)
    package.add_argument("--receipt", required=True)
    extract = commands.add_parser("extract")
    extract.add_argument("--bundle", required=True)
    extract.add_argument("--receipt", required=True)
    extract.add_argument("--output-root", required=True)
    arguments = parser.parse_args(argv)
    try:
        receipt = json.loads(Path(arguments.receipt).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        if arguments.command == "package":
            try:
                plan = json.loads(Path(arguments.plan).read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as plan_exc:
                raise Canonical3DGSTransportError(["transport_plan_json_invalid"]) from plan_exc
            result = compile_canonical_3dgs_transport_bundle(
                plan=plan,
                dataset_root=arguments.dataset_root,
                bundle_path=arguments.bundle,
                receipt_path=arguments.receipt,
            )
        else:
            raise Canonical3DGSTransportError(["transport_receipt_json_invalid"]) from exc
    else:
        if arguments.command == "package":
            try:
                plan = json.loads(Path(arguments.plan).read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as plan_exc:
                raise Canonical3DGSTransportError(["transport_plan_json_invalid"]) from plan_exc
            result = compile_canonical_3dgs_transport_bundle(
                plan=plan,
                dataset_root=arguments.dataset_root,
                bundle_path=arguments.bundle,
                receipt_path=arguments.receipt,
            )
        else:
            result = extract_canonical_3dgs_transport_bundle(
                bundle_path=arguments.bundle,
                receipt=receipt,
                output_root=arguments.output_root,
            )
    print(canonical_json(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BUNDLE_SCHEMA",
    "EXTRACTION_SCHEMA",
    "Canonical3DGSTransportError",
    "compile_canonical_3dgs_transport_bundle",
    "extract_canonical_3dgs_transport_bundle",
    "validate_canonical_3dgs_transport_receipt",
]
