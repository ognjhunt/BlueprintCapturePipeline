#!/usr/bin/env python3
"""Replay a rights-usable public indoor dataset through the supported local lane.

This command is intentionally a proxy test.  It binds a derived PLY to the
exact public source archive, exercises the same immutable Web-upload intake and
authorized reconstruction control plane used by the service, and preserves a
strictly appearance-only claim ceiling.  It does not turn processed dataset
artifacts into raw capture, production malware-scan, metric, physics, physical,
deployment, safety, or policy-ranking evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import stat
import subprocess
import sys
import tarfile
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, ContextManager, Mapping, Sequence
from urllib.parse import urlsplit

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from blueprint_pipeline.capture_upload_intake import (  # noqa: E402
    process_capture_upload_submission,
)
from blueprint_pipeline.local_reconstruction_adapters import (  # noqa: E402
    LOCAL_EXTERNAL_RECONSTRUCTION_IMPORT_ADAPTER,
)
from blueprint_pipeline.reconstruction_control_plane import (  # noqa: E402
    authorize_reconstruction_plan,
    execute_reconstruction_plan,
    load_reconstruction_compilation_inputs,
    prepare_reconstruction_plan,
)


SCHEMA_VERSION = "public_indoor_proxy_replay.v1"
_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9-]{0,62}$")
_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
_MAX_ARCHIVE_MEMBERS = 1_000_000
_MAX_ARCHIVE_DECLARED_BYTES = 200 * 1024 * 1024 * 1024
_MAX_SOURCE_BUNDLE_BYTES = 50 * 1024 * 1024 * 1024
_LOCAL_TRANSFER_HOST = "local.public-dataset.invalid"


class PublicIndoorProxyError(RuntimeError):
    """A stable fail-closed public-dataset replay error."""


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _write_immutable(path: Path, value: Mapping[str, Any]) -> None:
    encoded = _canonical_json(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as stream:
            stream.write(encoded)
    except FileExistsError:
        if path.is_symlink() or path.read_bytes() != encoded:
            raise PublicIndoorProxyError("public_proxy_summary_conflict")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _regular_file(value: str | Path, *, field: str) -> Path:
    path = Path(value).expanduser()
    if path.is_symlink():
        raise PublicIndoorProxyError(f"{field}_symlink_forbidden")
    resolved = path.resolve()
    try:
        mode = resolved.stat().st_mode
    except OSError as exc:
        raise PublicIndoorProxyError(f"{field}_missing") from exc
    if not stat.S_ISREG(mode) or resolved.stat().st_size <= 0:
        raise PublicIndoorProxyError(f"{field}_not_regular_file")
    return resolved


def _opaque_text(value: str, *, field: str, maximum: int = 192) -> str:
    text = value.strip()
    if (
        not text
        or len(text) > maximum
        or any(ord(character) < 32 or ord(character) == 127 for character in text)
    ):
        raise PublicIndoorProxyError(f"{field}_invalid")
    return text


def _public_source_uri(value: str) -> str:
    text = _opaque_text(value, field="dataset_source_uri", maximum=512)
    parsed = urlsplit(text)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise PublicIndoorProxyError("dataset_source_uri_invalid")
    return text


def _verified_digest(path: Path, expected: str, *, field: str) -> str:
    if not _SHA256.fullmatch(expected):
        raise PublicIndoorProxyError(f"{field}_expected_digest_invalid")
    observed = _sha256_file(path)
    if observed != expected:
        raise PublicIndoorProxyError(f"{field}_digest_mismatch")
    return observed


def _safe_archive_name(value: str) -> bool:
    normalized = value.replace("\\", "/")
    path = PurePosixPath(normalized)
    return bool(normalized) and not path.is_absolute() and ".." not in path.parts


def _inspect_tar(path: Path) -> dict[str, Any]:
    member_count = 0
    total_bytes = 0
    try:
        with tarfile.open(path, mode="r:*") as archive:
            for member in archive:
                member_count += 1
                if member_count > _MAX_ARCHIVE_MEMBERS:
                    raise PublicIndoorProxyError("source_bundle_archive_too_many_members")
                if (
                    not _safe_archive_name(member.name)
                    or member.issym()
                    or member.islnk()
                    or member.isdev()
                    or member.isfifo()
                ):
                    raise PublicIndoorProxyError("source_bundle_archive_unsafe_member")
                total_bytes += max(0, int(member.size))
                if total_bytes > _MAX_ARCHIVE_DECLARED_BYTES:
                    raise PublicIndoorProxyError("source_bundle_archive_declared_size_oversized")
    except PublicIndoorProxyError:
        raise
    except (OSError, tarfile.TarError) as exc:
        raise PublicIndoorProxyError("source_bundle_archive_invalid") from exc
    return {
        "format": "tar",
        "member_count": member_count,
        "declared_uncompressed_size_bytes": total_bytes,
        "traversal_or_link_members": 0,
    }


def _inspect_zip(path: Path) -> dict[str, Any]:
    member_count = 0
    total_bytes = 0
    try:
        with zipfile.ZipFile(path) as archive:
            for member in archive.infolist():
                member_count += 1
                unix_mode = member.external_attr >> 16
                if member_count > _MAX_ARCHIVE_MEMBERS:
                    raise PublicIndoorProxyError("source_bundle_archive_too_many_members")
                if not _safe_archive_name(member.filename) or stat.S_ISLNK(unix_mode):
                    raise PublicIndoorProxyError("source_bundle_archive_unsafe_member")
                total_bytes += max(0, int(member.file_size))
                if total_bytes > _MAX_ARCHIVE_DECLARED_BYTES:
                    raise PublicIndoorProxyError("source_bundle_archive_declared_size_oversized")
    except PublicIndoorProxyError:
        raise
    except (OSError, zipfile.BadZipFile) as exc:
        raise PublicIndoorProxyError("source_bundle_archive_invalid") from exc
    return {
        "format": "zip",
        "member_count": member_count,
        "declared_uncompressed_size_bytes": total_bytes,
        "traversal_or_link_members": 0,
    }


def inspect_source_bundle(path: Path) -> dict[str, Any]:
    name = path.name.lower()
    if name.endswith((".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tar.xz")):
        return _inspect_tar(path)
    if name.endswith(".zip"):
        return _inspect_zip(path)
    return {
        "format": "opaque_file",
        "member_count": None,
        "declared_uncompressed_size_bytes": None,
        "traversal_or_link_members": None,
    }


def _source_commit() -> str:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise PublicIndoorProxyError("source_commit_unavailable") from exc
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise PublicIndoorProxyError("source_commit_invalid")
    if dirty:
        raise PublicIndoorProxyError("source_checkout_not_clean")
    return commit


def _local_transfer_opener(source_artifact: Path):
    def open_transfer(**_kwargs: Any) -> ContextManager[BinaryIO]:
        return source_artifact.open("rb")

    return open_transfer


def _test_double_scanner(expected_digest: str):
    def scan(path: Path) -> dict[str, Any]:
        observed = _sha256_file(path)
        if observed != expected_digest:
            raise PublicIndoorProxyError("quarantined_artifact_digest_mismatch")
        return {
            "status": "passed",
            "scanner": "explicit-public-dataset-test-double-not-production-malware-gate",
            "artifact_digest": observed,
        }

    return scan


def _assert_claim_boundaries(result: Mapping[str, Any]) -> dict[str, Any]:
    ceiling = dict(result.get("claim_ceiling") or {})
    forbidden_true = (
        "raw_capture_authority",
        "captured_observation",
        "task_discovery",
        "calibrated_camera_poses",
        "metric_geometry",
        "metric_scale",
        "collision_geometry",
        "physics",
        "physical_task_success",
        "deployment_readiness",
        "safety_certification",
    )
    if any(ceiling.get(field) is not False for field in forbidden_true):
        raise PublicIndoorProxyError("public_proxy_claim_ceiling_upgraded")
    if ceiling.get("comparative_policy_ranking_verdict") != "thesis_not_supported":
        raise PublicIndoorProxyError("public_proxy_policy_verdict_changed")
    if ceiling.get("appearance_review") is not True:
        raise PublicIndoorProxyError("public_proxy_appearance_claim_missing")
    return ceiling


def run_public_indoor_proxy(
    *,
    dataset_id: str,
    dataset_source_uri: str,
    license_id: str,
    source_bundle: str | Path,
    source_bundle_sha256: str,
    source_artifact: str | Path,
    source_artifact_sha256: str,
    output_root: str | Path,
    provider_identity: str,
    consent_status: str,
    operator_identity: str,
    source_commit: str,
    expected_ply_vertices: int | None = None,
    acknowledge_test_double_malware_scan: bool = False,
) -> dict[str, Any]:
    if not _IDENTIFIER.fullmatch(dataset_id):
        raise PublicIndoorProxyError("dataset_id_invalid")
    dataset_source_uri = _public_source_uri(dataset_source_uri)
    license_id = _opaque_text(license_id, field="license_id", maximum=128)
    provider_identity = _opaque_text(
        provider_identity, field="provider_identity", maximum=128
    )
    if consent_status not in {"accepted", "not_required"}:
        raise PublicIndoorProxyError("consent_status_invalid")
    operator_identity = _opaque_text(
        operator_identity, field="operator_identity", maximum=192
    )
    if not re.fullmatch(r"[0-9a-f]{40}", source_commit):
        raise PublicIndoorProxyError("source_commit_invalid")
    if expected_ply_vertices is not None and expected_ply_vertices <= 0:
        raise PublicIndoorProxyError("expected_ply_vertices_invalid")
    if not acknowledge_test_double_malware_scan:
        raise PublicIndoorProxyError("test_double_malware_scan_acknowledgment_required")
    bundle = _regular_file(source_bundle, field="source_bundle")
    artifact = _regular_file(source_artifact, field="source_artifact")
    if bundle.stat().st_size > _MAX_SOURCE_BUNDLE_BYTES:
        raise PublicIndoorProxyError("source_bundle_oversized")
    bundle_digest = _verified_digest(
        bundle, source_bundle_sha256, field="source_bundle"
    )
    artifact_digest = _verified_digest(
        artifact, source_artifact_sha256, field="source_artifact"
    )
    archive_inspection = inspect_source_bundle(bundle)
    root = Path(output_root).expanduser()
    if root.is_symlink():
        raise PublicIndoorProxyError("output_root_symlink_forbidden")
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    digest_token = artifact_digest[7:19]
    capture_session_id = f"public-{dataset_id}-{digest_token}"
    intake_id = f"public-{dataset_id}-external-{digest_token}"
    submission = {
        "schema_version": "capture_upload_transfer_submission.v1",
        "capture_session_id": capture_session_id,
        "customer_id": "public-dataset-research",
        "organization_id": "blueprint-local-research",
        "request": {
            "schema_version": "capture_upload_session_request.v1",
            "intake_id": intake_id,
            "idempotency_key": f"{dataset_id}-{bundle_digest[7:23]}-{digest_token}",
            "capture_authority_profile": "precomputed_external_reconstruction",
            "source_type": "precomputed_external_reconstruction",
            "scene_id": f"public-{dataset_id}",
            "original_file": {
                "original_filename": artifact.name,
                "size_bytes": artifact.stat().st_size,
                "media_type": "application/ply",
            },
            "capture_device": {
                "manufacturer": "public-dataset-provider",
                "model": provider_identity,
            },
            "timing_declaration": {"status": "not_included_in_import"},
            "coordinate_frame_declaration": {
                "status": "provider_declared_unverified"
            },
            "available_sensor_streams": [
                {"stream_type": "external_reconstruction", "status": "available"}
            ],
            "source_capture_binding": {
                "source_capture_digest": bundle_digest,
                "provider_identity": provider_identity,
            },
            "governance": {
                "rights": "accepted",
                "consent": consent_status,
                "privacy": "restricted_local_only",
                "retention": {"max_days": 30},
                "revocation": {
                    "supported": True,
                    "historical_tombstone_retained": True,
                },
                "provider_constraints": {"external_processing_allowed": False},
                "allowed_uses": ["evaluation"],
            },
            "requested_task_evaluation_run_audience": "internal_research_proxy",
            "known_task_specification": None,
            "calibration_board_dimensions": None,
            "operator_notes": [
                f"dataset_source_uri={dataset_source_uri}",
                f"license_id={license_id}",
                "public_dataset_proxy_not_customer_capture",
            ],
            "permitted_reconstruction_providers": ["local_only"],
            "permitted_evidence_uses": ["appearance_review"],
        },
        "transfer": {
            "provider": "backblaze",
            "url": f"https://{_LOCAL_TRANSFER_HOST}/{artifact.name}",
            "authorization": "local-public-dataset-test-double",
            "expires_at_iso": (
                datetime.now(timezone.utc) + timedelta(minutes=10)
            ).isoformat(),
        },
    }
    capture_store = root / "capture-store"
    state_root = root / "reconstruction-state"
    receipt = process_capture_upload_submission(
        submission,
        store_root=capture_store,
        allowed_hosts=[_LOCAL_TRANSFER_HOST],
        transfer_opener=_local_transfer_opener(artifact),
        malware_scanner=_test_double_scanner(artifact_digest),
    )
    if receipt.get("admission_status") != "accepted":
        raise PublicIndoorProxyError("public_proxy_capture_not_accepted")
    plan = prepare_reconstruction_plan(
        state_root=state_root,
        capture_store_root=capture_store,
        capture_session_id=capture_session_id,
        intake_id=intake_id,
        requested_claim_types=["appearance_review"],
        idempotency_key=f"plan-{dataset_id}-{digest_token}",
    )
    candidates = plan.get("authorization_candidates") or []
    if [row.get("adapter_reference") for row in candidates] != [
        LOCAL_EXTERNAL_RECONSTRUCTION_IMPORT_ADAPTER
    ]:
        raise PublicIndoorProxyError("public_proxy_adapter_plan_invalid")
    authorization = authorize_reconstruction_plan(
        state_root=state_root,
        plan_id=plan["plan_id"],
        reconstruction_plan_digest=plan["reconstruction_plan"][
            "reconstruction_plan_digest"
        ],
        authorized_adapter_references=[LOCAL_EXTERNAL_RECONSTRUCTION_IMPORT_ADAPTER],
        actor={"role": "operator", "identity": operator_identity},
        idempotency_key=f"authorize-{dataset_id}-{digest_token}",
    )
    execution = execute_reconstruction_plan(
        state_root=state_root,
        capture_store_root=capture_store,
        plan_id=plan["plan_id"],
    )
    results = [dict(row) for row in execution.get("results") or []]
    if execution.get("state") != "partial" or len(results) != 1:
        raise PublicIndoorProxyError("public_proxy_execution_state_invalid")
    result = results[0]
    ceiling = _assert_claim_boundaries(result)
    ply_header = dict((result.get("validation_metrics") or {}).get("ply_header") or {})
    vertex_count = int((ply_header.get("elements") or {}).get("vertex") or 0)
    if expected_ply_vertices is not None and vertex_count != expected_ply_vertices:
        raise PublicIndoorProxyError("public_proxy_ply_vertex_count_mismatch")
    compilation = load_reconstruction_compilation_inputs(
        state_root=state_root,
        capture_store_root=capture_store,
        plan_id=plan["plan_id"],
        execution_result_digest=execution["execution_result_digest"],
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "dataset": {
            "dataset_id": dataset_id,
            "source_uri": dataset_source_uri,
            "license_id": license_id,
            "provider_identity": provider_identity,
            "privacy": "restricted_local_only",
            "public_dataset_proxy_not_customer_capture": True,
        },
        "source_bundle": {
            "filename": bundle.name,
            "digest": bundle_digest,
            "size_bytes": bundle.stat().st_size,
            "archive_inspection": archive_inspection,
        },
        "source_artifact": {
            "filename": artifact.name,
            "digest": artifact_digest,
            "size_bytes": artifact.stat().st_size,
            "ply_header": ply_header,
            "vertex_count": vertex_count,
        },
        "source_commit": source_commit,
        "capture_session_id": capture_session_id,
        "intake_id": intake_id,
        "capture_digest": receipt["capture_digest"],
        "envelope_digest": receipt["envelope_digest"],
        "qa_report_digest": receipt["capture_qa_report"]["qa_report_digest"],
        "admission_status": receipt["admission_status"],
        "qa_status": receipt["capture_qa_report"]["status"],
        "plan_id": plan["plan_id"],
        "reconstruction_plan_digest": plan["reconstruction_plan"][
            "reconstruction_plan_digest"
        ],
        "authorization_digest": authorization["authorization_digest"],
        "authorized_adapters": [LOCAL_EXTERNAL_RECONSTRUCTION_IMPORT_ADAPTER],
        "execution_state": execution["state"],
        "execution_result_digest": execution["execution_result_digest"],
        "result_id": result["result_id"],
        "result_outputs": result["outputs"],
        "compilation_result_digests": [
            row["reconstruction_result_digest"]
            for row in compilation["reconstruction_results"]
        ],
        "missing_representations": execution["missing_representations"],
        "next_cheapest_experiments": execution["next_cheapest_experiments"],
        "execution_cost_usd": execution["cost_usd"],
        "claim_ceiling": ceiling,
        "proof_boundary": {
            "transfer_was_local_test_double": True,
            "malware_scan_was_test_double": True,
            "production_security_gate_passed": False,
            "raw_capture_gate_passed": False,
            "customer_upload_gate_passed": False,
            "physical_task_success_established": False,
            "deployment_or_safety_approved": False,
            "comparative_policy_ranking_verdict": "thesis_not_supported",
        },
    }
    _write_immutable(root / "public_indoor_proxy_replay.json", summary)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--dataset-source-uri", required=True)
    parser.add_argument("--license-id", required=True)
    parser.add_argument("--provider-identity", required=True)
    parser.add_argument("--source-bundle", type=Path, required=True)
    parser.add_argument("--source-bundle-sha256", required=True)
    parser.add_argument("--source-artifact", type=Path, required=True)
    parser.add_argument("--source-artifact-sha256", required=True)
    parser.add_argument("--expected-ply-vertices", type=int)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--consent-status", choices=("accepted", "not_required"), required=True
    )
    parser.add_argument("--operator-identity", required=True)
    parser.add_argument(
        "--acknowledge-test-double-malware-scan",
        action="store_true",
        help="Required acknowledgment that this local replay is not a production malware gate.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        summary = run_public_indoor_proxy(
            dataset_id=args.dataset_id,
            dataset_source_uri=args.dataset_source_uri,
            license_id=args.license_id,
            source_bundle=args.source_bundle,
            source_bundle_sha256=args.source_bundle_sha256,
            source_artifact=args.source_artifact,
            source_artifact_sha256=args.source_artifact_sha256,
            output_root=args.output_root,
            provider_identity=args.provider_identity,
            consent_status=args.consent_status,
            operator_identity=args.operator_identity,
            source_commit=_source_commit(),
            expected_ply_vertices=args.expected_ply_vertices,
            acknowledge_test_double_malware_scan=(
                args.acknowledge_test_double_malware_scan
            ),
        )
    except PublicIndoorProxyError as exc:
        print(f"[public-indoor-proxy] blocked: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
