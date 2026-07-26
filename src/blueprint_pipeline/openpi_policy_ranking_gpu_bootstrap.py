"""Private signed-bundle transport for the one-shot OpenPI GPU campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import urllib.request
import zipfile
from collections.abc import Sequence
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlparse

from .common import write_json
from .openpi_policy_ranking_gpu_job import run_openpi_policy_ranking_gpu_campaign
from .policy_ranking_thesis import canonical_sha256


SCHEMA_VERSION = "openpi_policy_ranking_gpu_bootstrap.v1"
INPUT_SCHEMA_VERSION = "openpi_policy_ranking_gpu_input_bundle.v1"
BACKGROUND_NAME = "captured_site_background.png"
MANIFEST_NAME = "bundle_manifest.json"
MAX_INPUT_BYTES = 5 * 1024 * 1024
POLICY_IDS = (
    "pi05_droid_jointpos_polaris",
    "pi0_fast_droid_jointpos_polaris",
    "pi0_droid_jointpos_polaris",
    "pi0_droid_jointpos_100k_polaris",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_private_input_bundle(
    *,
    background_path: str | Path,
    output_zip: str | Path,
    source_scene_id: str,
    source_revision: str,
    source_asset_sha256: str,
) -> dict[str, Any]:
    background = Path(background_path).expanduser().resolve()
    destination = Path(output_zip).expanduser().resolve()
    if not background.is_file() or background.is_symlink():
        raise FileNotFoundError("captured_site_background_missing_or_unsafe")
    if background.stat().st_size > MAX_INPUT_BYTES:
        raise ValueError("captured_site_background_too_large")
    manifest: dict[str, Any] = {
        "schema_version": INPUT_SCHEMA_VERSION,
        "source_scene_id": str(source_scene_id),
        "source_revision": str(source_revision),
        "source_asset_sha256": str(source_asset_sha256),
        "background_filename": BACKGROUND_NAME,
        "background_sha256": _sha256(background),
        "background_size_bytes": background.stat().st_size,
        "raw_3dgs_included": False,
        "redistribution_authorized": False,
        "purpose": "private_internal_noncommercial_research_gpu_execution",
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(MANIFEST_NAME, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        archive.write(background, BACKGROUND_NAME)
    return {
        "schema_version": "openpi_policy_ranking_gpu_input_bundle_receipt.v1",
        "status": "completed",
        "bundle_path": str(destination),
        "bundle_sha256": _sha256(destination),
        "bundle_size_bytes": destination.stat().st_size,
        "manifest": manifest,
    }


def extract_private_input_bundle(
    *, bundle_path: str | Path, expected_bundle_sha256: str, output_dir: str | Path
) -> dict[str, Any]:
    bundle = Path(bundle_path).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    if not bundle.is_file() or bundle.is_symlink() or bundle.stat().st_size > MAX_INPUT_BYTES:
        raise ValueError("gpu_input_bundle_missing_unsafe_or_too_large")
    if _sha256(bundle) != expected_bundle_sha256:
        raise ValueError("gpu_input_bundle_sha256_mismatch")
    with zipfile.ZipFile(bundle) as archive:
        names = archive.namelist()
        if set(names) != {MANIFEST_NAME, BACKGROUND_NAME} or len(names) != 2:
            raise ValueError("gpu_input_bundle_file_allowlist_mismatch")
        for info in archive.infolist():
            path = PurePosixPath(info.filename)
            if path.is_absolute() or ".." in path.parts or info.file_size > MAX_INPUT_BYTES:
                raise ValueError("gpu_input_bundle_member_unsafe")
        manifest = json.loads(archive.read(MANIFEST_NAME).decode("utf-8"))
        if manifest.get("schema_version") != INPUT_SCHEMA_VERSION:
            raise ValueError("gpu_input_bundle_manifest_schema_invalid")
        declared_manifest_sha = manifest.get("manifest_sha256")
        digest_payload = dict(manifest)
        digest_payload.pop("manifest_sha256", None)
        if declared_manifest_sha != canonical_sha256(digest_payload):
            raise ValueError("gpu_input_bundle_manifest_sha256_mismatch")
        background_bytes = archive.read(BACKGROUND_NAME)
    if hashlib.sha256(background_bytes).hexdigest() != manifest.get("background_sha256"):
        raise ValueError("gpu_input_background_sha256_mismatch")
    if len(background_bytes) != manifest.get("background_size_bytes"):
        raise ValueError("gpu_input_background_size_mismatch")
    output.mkdir(parents=True, exist_ok=True)
    background = output / BACKGROUND_NAME
    background.write_bytes(background_bytes)
    return {"manifest": manifest, "background_path": str(background)}


def _download_signed_input(url: str, destination: Path) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("gpu_input_url_not_safe_https")
    request = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(  # nosec B310 - exact validated HTTPS URL and redirect check
        request, timeout=180
    ) as response, destination.open("wb") as handle:
        if response.geturl() != url:
            raise ValueError("gpu_input_url_redirect_forbidden")
        total = 0
        while chunk := response.read(1024 * 1024):
            total += len(chunk)
            if total > MAX_INPUT_BYTES:
                raise ValueError("gpu_input_download_exceeds_size_cap")
            handle.write(chunk)


def _upload_output(url: str, archive_path: Path) -> int:
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("gpu_output_url_not_safe_https")
    data = archive_path.read_bytes()
    request = urllib.request.Request(
        url,
        data=data,
        method="PUT",
        headers={"Content-Type": "application/zip", "Content-Length": str(len(data))},
    )
    with urllib.request.urlopen(  # nosec B310 - exact validated HTTPS URL and redirect check
        request, timeout=300
    ) as response:
        if response.geturl() != url:
            raise ValueError("gpu_output_url_redirect_forbidden")
        response.read()
        return int(getattr(response, "status", 200))


def run_signed_gpu_bootstrap(*, workspace: str | Path = "/workspace") -> dict[str, Any]:
    root = Path(workspace).expanduser().resolve()
    input_url = os.getenv(
        "BLUEPRINT_OPENPI_POLICY_RANKING_INPUT_SECRET_URL", ""
    ).strip()
    input_sha = os.getenv("BLUEPRINT_OPENPI_POLICY_RANKING_INPUT_SHA256", "").strip()
    output_url = os.getenv(
        "BLUEPRINT_OPENPI_POLICY_RANKING_OUTPUT_SECRET_PUT_URL", ""
    ).strip()
    if not input_url or not output_url or len(input_sha) != 64:
        raise ValueError("signed_gpu_bootstrap_environment_missing")
    bundle = root / "policy-ranking-input.zip"
    extracted = root / "policy-ranking-input"
    campaign_output = root / "policy-ranking-output"
    _download_signed_input(input_url, bundle)
    extracted_input = extract_private_input_bundle(
        bundle_path=bundle,
        expected_bundle_sha256=input_sha,
        output_dir=extracted,
    )
    campaign = run_openpi_policy_ranking_gpu_campaign(
        cohort_path="/opt/blueprint/frozen/warehouse_policy_cohort_v2_joint_position.json",
        checkpoint_inventory_path="/opt/blueprint/frozen/openpi_polaris_checkpoint_inventory.json",
        captured_site_background_path=extracted_input["background_path"],
        menagerie_root="/opt/mujoco-menagerie/franka_emika_panda",
        output_dir=campaign_output,
        policy_ids=POLICY_IDS,
    )
    output_archive_base = root / "openpi-policy-ranking-output"
    archive_path = Path(
        shutil.make_archive(str(output_archive_base), "zip", root_dir=campaign_output)
    )
    upload_status = _upload_output(output_url, archive_path)
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed" if campaign["status"] == "completed" else "blocked",
        "input_bundle_sha256": input_sha,
        "input_manifest": extracted_input["manifest"],
        "campaign_manifest_sha256": campaign["manifest_sha256"],
        "output_archive_sha256": _sha256(archive_path),
        "output_archive_size_bytes": archive_path.stat().st_size,
        "output_upload_status": upload_status,
        "raw_secret_values_recorded": False,
        "physical_robot_operated": False,
    }
    result["manifest_sha256"] = canonical_sha256(result)
    write_json(root / "openpi_policy_ranking_gpu_bootstrap.json", result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build-input")
    build.add_argument("--background", required=True)
    build.add_argument("--output", required=True)
    build.add_argument("--scene-id", required=True)
    build.add_argument("--source-revision", required=True)
    build.add_argument("--source-asset-sha256", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--workspace", default="/workspace")
    args = parser.parse_args(argv)
    if args.command == "build-input":
        receipt = build_private_input_bundle(
            background_path=args.background,
            output_zip=args.output,
            source_scene_id=args.scene_id,
            source_revision=args.source_revision,
            source_asset_sha256=args.source_asset_sha256,
        )
        write_json(Path(args.output).with_suffix(".receipt.json"), receipt)
        return 0
    result = run_signed_gpu_bootstrap(workspace=args.workspace)
    return 0 if result["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_private_input_bundle", "extract_private_input_bundle"]
