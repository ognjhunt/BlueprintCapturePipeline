"""Build the non-spending, digest-bound input bundle for Isaac verification.

This compiler never allocates a provider.  Its output is the immutable input to
Blueprint's canonical paid-resource allocator after separate budget/TTL/retry
admission succeeds.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import tempfile
from typing import Any, Mapping
import zipfile

from .common import write_json
from .decision_evidence_contracts import canonical_digest
from .isaac_reconstruction_verification import (
    IsaacReconstructionVerificationError,
    build_isaac_asset_verification_request,
)


ISAAC_WORKER_BUNDLE_SCHEMA = "isaac_verification_worker_bundle.v1"
MAX_BUNDLE_MEMBER_BYTES = 4_000_000_000
MAX_BUNDLE_TOTAL_BYTES = 5_000_000_000


class IsaacWorkerBundleError(ValueError):
    def __init__(self, codes: list[str] | tuple[str, ...]) -> None:
        self.codes = tuple(sorted(set(str(code) for code in codes if str(code))))
        super().__init__("; ".join(self.codes))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _safe_source(root: Path, reference: Any, digest: str, suffix: str, code: str) -> Path:
    text = str(reference or "").replace("\\", "/")
    relative = PurePosixPath(text)
    if (
        not text
        or relative.is_absolute()
        or any(part in {"", ".", ".."} for part in relative.parts)
        or ":" in relative.parts[0]
        or relative.suffix.lower() != suffix
    ):
        raise IsaacWorkerBundleError([f"{code}_reference_unsafe"])
    candidate = root.joinpath(*relative.parts)
    if candidate.is_symlink():
        raise IsaacWorkerBundleError([f"{code}_symlink_forbidden"])
    try:
        resolved = candidate.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise IsaacWorkerBundleError([f"{code}_missing"]) from exc
    if root != resolved and root not in resolved.parents:
        raise IsaacWorkerBundleError([f"{code}_path_escape"])
    if not resolved.is_file() or _sha256(resolved) != digest:
        raise IsaacWorkerBundleError([f"{code}_digest_mismatch"])
    if resolved.stat().st_size > MAX_BUNDLE_MEMBER_BYTES:
        raise IsaacWorkerBundleError([f"{code}_oversized"])
    return resolved


def _explicit_file(path_value: str | Path, *, digest: str, suffix: str, code: str) -> Path:
    path = Path(path_value)
    if path.is_symlink():
        raise IsaacWorkerBundleError([f"{code}_symlink_forbidden"])
    try:
        path = path.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise IsaacWorkerBundleError([f"{code}_missing"]) from exc
    if not path.is_file() or path.suffix.lower() != suffix or _sha256(path) != digest:
        raise IsaacWorkerBundleError([f"{code}_digest_or_format_mismatch"])
    if path.stat().st_size > MAX_BUNDLE_MEMBER_BYTES:
        raise IsaacWorkerBundleError([f"{code}_oversized"])
    return path


def _write_zip_member(archive: zipfile.ZipFile, name: str, source: Path) -> None:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    info.create_system = 3
    info.external_attr = (stat.S_IFREG | 0o644) << 16
    with source.open("rb") as input_stream, archive.open(info, "w", force_zip64=True) as output:
        shutil.copyfileobj(input_stream, output, length=1024 * 1024)


def compile_isaac_verification_worker_bundle(
    *,
    verification_request: Mapping[str, Any],
    package_artifact_root: str | Path,
    fixed_camera_spec_path: str | Path,
    runner_path: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    """Compile a deterministic exact-package Isaac input bundle without spending."""

    try:
        request = build_isaac_asset_verification_request(verification_request)
    except IsaacReconstructionVerificationError as exc:
        raise IsaacWorkerBundleError(
            [f"isaac_verification_request_invalid:{code}" for code in exc.codes]
        ) from exc
    package_root = Path(package_artifact_root)
    if package_root.is_symlink() or not package_root.is_dir():
        raise IsaacWorkerBundleError(["isaac_package_root_invalid"])
    package_root = package_root.resolve()
    package = _safe_source(
        package_root,
        request["package_artifact_reference"],
        request["package_digest"],
        ".usdz",
        "isaac_package",
    )
    cameras = _explicit_file(
        fixed_camera_spec_path,
        digest=request["fixed_camera_spec_digest"],
        suffix=".json",
        code="isaac_camera_spec",
    )
    try:
        camera_rows = json.loads(cameras.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise IsaacWorkerBundleError(["isaac_camera_spec_json_invalid"]) from exc
    if (
        not isinstance(camera_rows, list)
        or [row.get("id") for row in camera_rows if isinstance(row, Mapping)]
        != request["fixed_camera_ids"]
    ):
        raise IsaacWorkerBundleError(["isaac_camera_spec_ids_mismatch"])
    runner = _explicit_file(
        runner_path,
        digest=request["runtime_implementation_digest"],
        suffix=".py",
        code="isaac_runner",
    )
    total = sum(path.stat().st_size for path in (package, cameras, runner))
    if total > MAX_BUNDLE_TOTAL_BYTES:
        raise IsaacWorkerBundleError(["isaac_worker_bundle_oversized"])

    destination = Path(output_root)
    if destination.is_symlink():
        raise IsaacWorkerBundleError(["isaac_bundle_output_root_symlink_forbidden"])
    destination.mkdir(parents=True, exist_ok=True)
    destination = destination.resolve()
    content_id = request["isaac_verification_request_digest"][7:]
    final = destination / content_id
    receipt_path = final / "isaac_verification_worker_bundle.v1.json"
    bundle_path = final / "isaac_verification_worker_bundle.zip"
    if final.exists():
        try:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise IsaacWorkerBundleError(["isaac_bundle_existing_output_tampered"]) from exc
        try:
            observed_bundle_digest = _sha256(bundle_path)
        except OSError as exc:
            raise IsaacWorkerBundleError(["isaac_bundle_existing_output_tampered"]) from exc
        if receipt.get("bundle_digest") != observed_bundle_digest:
            raise IsaacWorkerBundleError(["isaac_bundle_replay_digest_mismatch"])
        return receipt

    temporary = Path(tempfile.mkdtemp(prefix=".isaac-bundle-", dir=destination))
    try:
        request_path = temporary / "isaac_asset_verification_request.v1.json"
        write_json(request_path, request)
        command = [
            "/isaac-sim/python.sh",
            "/workspace/bundle/run_isaac_splat_nurec_render.py",
            "--usdz",
            "/workspace/bundle/reconstruction.usdz",
            "--cameras",
            "/workspace/bundle/fixed_cameras.json",
            "--out-dir",
            "/workspace/out",
            "--qualification-mode",
            "--package-digest",
            request["package_digest"],
            "--verification-request-digest",
            request["isaac_verification_request_digest"],
            "--camera-spec-digest",
            request["fixed_camera_spec_digest"],
            "--runtime-container-image-digest",
            request["runtime_container_image_digest"],
            "--runtime-implementation-digest",
            request["runtime_implementation_digest"],
            "--physics-probe-steps",
            str(request["physics_probe_request"]["steps"]),
        ]
        probe = request["physics_probe_request"]
        if probe.get("ground_collider_prim"):
            command.extend(["--ground-collider-prim", str(probe["ground_collider_prim"])])
        if probe.get("ground_height_m") is not None:
            command.extend(["--ground-height", str(probe["ground_height_m"])])
        if probe.get("probe_xy_m") is not None:
            command.extend(["--probe-xy", *[str(value) for value in probe["probe_xy_m"]]])
        manifest = {
            "schema_version": ISAAC_WORKER_BUNDLE_SCHEMA,
            "isaac_verification_request_digest": request[
                "isaac_verification_request_digest"
            ],
            "package_digest": request["package_digest"],
            "fixed_camera_spec_digest": request["fixed_camera_spec_digest"],
            "runtime_implementation_digest": request["runtime_implementation_digest"],
            "runtime_container_image_digest": request["runtime_container_image_digest"],
            "fixed_camera_ids": request["fixed_camera_ids"],
            "command": command,
            "expected_runtime_schema": "isaac_splat_nurec_render_result.v3",
            "raw_secret_values_included": False,
            "provider_allocation_performed": False,
            "paid_execution_authorized_by_bundle": False,
            "canonical_allocator_command": (
                "python -m blueprint_pipeline.paid_resource_allocator gpu-canary"
            ),
            "proof_effect": "none",
            "claim_ceiling": "request_only",
        }
        manifest["bundle_manifest_digest"] = canonical_digest(
            manifest, digest_field="bundle_manifest_digest"
        )
        manifest_path = temporary / "bundle_manifest.json"
        write_json(manifest_path, manifest)
        archive_path = temporary / "isaac_verification_worker_bundle.zip"
        with zipfile.ZipFile(archive_path, "w", allowZip64=True) as archive:
            for name, source in (
                ("bundle_manifest.json", manifest_path),
                ("fixed_cameras.json", cameras),
                ("isaac_asset_verification_request.v1.json", request_path),
                ("reconstruction.usdz", package),
                ("run_isaac_splat_nurec_render.py", runner),
            ):
                _write_zip_member(archive, name, source)
        bundle_digest = _sha256(archive_path)
        receipt = {
            **manifest,
            "bundle_digest": bundle_digest,
            "bundle_artifact_reference": f"{content_id}/isaac_verification_worker_bundle.zip",
            "bundle_member_count": 5,
            "bundle_bytes": archive_path.stat().st_size,
            "cost_usd": 0.0,
        }
        write_json(temporary / "isaac_verification_worker_bundle.v1.json", receipt)
        os.replace(temporary, final)
        return receipt
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


__all__ = [
    "ISAAC_WORKER_BUNDLE_SCHEMA",
    "IsaacWorkerBundleError",
    "compile_isaac_verification_worker_bundle",
]
