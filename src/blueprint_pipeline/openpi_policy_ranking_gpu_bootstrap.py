"""Private signed-bundle transport for the one-shot OpenPI GPU campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import urllib.request
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlparse

from .common import write_json
from .five_policy_identity_smoke import (
    MAX_INPUT_BYTES as FIVE_POLICY_MAX_INPUT_BYTES,
    extract_input_bundle as extract_five_policy_input_bundle,
    run_identity_smoke as run_five_policy_identity_smoke,
)
from .new_site_diagnostic_canary_gpu import (
    INPUT_RECEIPT_SCHEMA_VERSION as CANARY_INPUT_RECEIPT_SCHEMA_VERSION,
    MAX_INPUT_BYTES as CANARY_MAX_INPUT_BYTES,
    build_canary_input_bundle,
    extract_canary_input_bundle,
    materialize_canary_background,
    run_skeleton_only_canary,
)
from .openpi_policy_ranking_gpu_job import run_openpi_policy_ranking_gpu_campaign
from .policy_ranking_thesis import canonical_sha256


SCHEMA_VERSION = "openpi_policy_ranking_gpu_bootstrap.v1"
INPUT_SCHEMA_VERSION = "openpi_policy_ranking_gpu_input_bundle.v2"
BACKGROUND_DIR = "scene_backgrounds"
MANIFEST_NAME = "bundle_manifest.json"
MAX_INPUT_BYTES = 5 * 1024 * 1024
SCENE_KINDS = {"captured_3dgs", "controlled_nvidia_usd"}
POLICY_IDS = (
    "pi05_droid_jointpos_polaris",
    "pi0_fast_droid_jointpos_polaris",
    "pi0_droid_jointpos_polaris",
    "pi0_droid_jointpos_100k_polaris",
)
EXECUTION_MODE_FULL_CAMPAIGN = "full_campaign"
EXECUTION_MODE_NEW_SITE_CANARY = "new_site_diagnostic_canary"
EXECUTION_MODE_IDENTITY_SMOKE = "five_policy_identity_smoke"
EXECUTION_MODES = {
    EXECUTION_MODE_FULL_CAMPAIGN,
    EXECUTION_MODE_NEW_SITE_CANARY,
    EXECUTION_MODE_IDENTITY_SMOKE,
}


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
    source_scene_kind: str = "captured_3dgs",
) -> dict[str, Any]:
    return build_multi_scene_private_input_bundle(
        scenes=[
            {
                "background_path": str(background_path),
                "source_scene_id": source_scene_id,
                "source_scene_kind": source_scene_kind,
                "source_revision": source_revision,
                "source_asset_sha256": source_asset_sha256,
            }
        ],
        output_zip=output_zip,
    )


def build_multi_scene_private_input_bundle(
    *,
    scenes: Sequence[Mapping[str, Any]],
    output_zip: str | Path,
) -> dict[str, Any]:
    """Bind one or more private scene backgrounds into a single GPU campaign."""

    destination = Path(output_zip).expanduser().resolve()
    normalized: list[dict[str, Any]] = []
    seen_scene_ids: set[str] = set()
    total_background_bytes = 0
    for index, row in enumerate(scenes):
        background = Path(str(row.get("background_path") or "")).expanduser().resolve()
        scene_id = str(row.get("source_scene_id") or "").strip()
        scene_kind = str(row.get("source_scene_kind") or "").strip()
        if not background.is_file() or background.is_symlink():
            raise FileNotFoundError("scene_background_missing_or_unsafe")
        if not scene_id or scene_id in seen_scene_ids:
            raise ValueError("scene_background_id_missing_or_duplicate")
        if scene_kind not in SCENE_KINDS:
            raise ValueError("scene_background_kind_invalid")
        seen_scene_ids.add(scene_id)
        total_background_bytes += background.stat().st_size
        filename = f"{BACKGROUND_DIR}/scene_{index:03d}.png"
        normalized.append(
            {
                "source_scene_id": scene_id,
                "source_scene_kind": scene_kind,
                "source_revision": str(row.get("source_revision") or ""),
                "source_asset_sha256": str(row.get("source_asset_sha256") or ""),
                "background_filename": filename,
                "background_sha256": _sha256(background),
                "background_size_bytes": background.stat().st_size,
                "_local_background_path": background,
            }
        )
    if not normalized:
        raise ValueError("scene_backgrounds_empty")
    if total_background_bytes > MAX_INPUT_BYTES:
        raise ValueError("scene_backgrounds_too_large")
    primary = normalized[0]
    public_scenes = [
        {key: value for key, value in row.items() if key != "_local_background_path"}
        for row in normalized
    ]
    manifest: dict[str, Any] = {
        "schema_version": INPUT_SCHEMA_VERSION,
        "source_scene_id": primary["source_scene_id"],
        "source_scene_kind": primary["source_scene_kind"],
        "source_revision": primary["source_revision"],
        "source_asset_sha256": primary["source_asset_sha256"],
        "background_filename": primary["background_filename"],
        "background_sha256": primary["background_sha256"],
        "background_size_bytes": primary["background_size_bytes"],
        "scenes": public_scenes,
        "scene_count": len(public_scenes),
        "raw_3dgs_included": False,
        "redistribution_authorized": False,
        "purpose": "private_internal_noncommercial_research_gpu_execution",
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(MANIFEST_NAME, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        for row in normalized:
            archive.write(row["_local_background_path"], row["background_filename"])
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
        for info in archive.infolist():
            path = PurePosixPath(info.filename)
            if path.is_absolute() or ".." in path.parts or info.file_size > MAX_INPUT_BYTES:
                raise ValueError("gpu_input_bundle_member_unsafe")
        manifest = json.loads(archive.read(MANIFEST_NAME).decode("utf-8"))
        if manifest.get("schema_version") != INPUT_SCHEMA_VERSION:
            raise ValueError("gpu_input_bundle_manifest_schema_invalid")
        scenes = manifest.get("scenes")
        if not isinstance(scenes, list) or not scenes:
            raise ValueError("gpu_input_bundle_scenes_invalid")
        filenames = {
            str(row.get("background_filename") or "")
            for row in scenes
            if isinstance(row, Mapping)
        }
        if (
            len(filenames) != len(scenes)
            or set(names) != {MANIFEST_NAME, *filenames}
            or len(names) != len(scenes) + 1
        ):
            raise ValueError("gpu_input_bundle_file_allowlist_mismatch")
        declared_manifest_sha = manifest.get("manifest_sha256")
        digest_payload = dict(manifest)
        digest_payload.pop("manifest_sha256", None)
        if declared_manifest_sha != canonical_sha256(digest_payload):
            raise ValueError("gpu_input_bundle_manifest_sha256_mismatch")
        extracted_rows = []
        output.mkdir(parents=True, exist_ok=True)
        for row in scenes:
            if not isinstance(row, Mapping):
                raise ValueError("gpu_input_bundle_scene_not_object")
            scene_id = str(row.get("source_scene_id") or "").strip()
            scene_kind = str(row.get("source_scene_kind") or "").strip()
            filename = str(row.get("background_filename") or "")
            if not scene_id or scene_kind not in SCENE_KINDS:
                raise ValueError("gpu_input_bundle_scene_identity_invalid")
            background_bytes = archive.read(filename)
            if hashlib.sha256(background_bytes).hexdigest() != row.get("background_sha256"):
                raise ValueError("gpu_input_background_sha256_mismatch")
            if len(background_bytes) != row.get("background_size_bytes"):
                raise ValueError("gpu_input_background_size_mismatch")
            background = output / Path(filename).name
            background.write_bytes(background_bytes)
            extracted_rows.append(
                {
                    "scene_id": scene_id,
                    "scene_kind": scene_kind,
                    "background_path": str(background),
                }
            )
    return {
        "manifest": manifest,
        "background_path": extracted_rows[0]["background_path"],
        "scene_backgrounds": extracted_rows,
    }


def _download_signed_input(
    url: str, destination: Path, *, max_bytes: int = MAX_INPUT_BYTES
) -> None:
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
            if total > max_bytes:
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
    execution_mode = os.getenv(
        "BLUEPRINT_OPENPI_EXECUTION_MODE", EXECUTION_MODE_FULL_CAMPAIGN
    ).strip()
    bundle = root / "policy-ranking-input.zip"
    extracted = root / "policy-ranking-input"
    campaign_output = root / "policy-ranking-output"
    extracted_input: dict[str, Any] = {}
    campaign: dict[str, Any] = {}
    failure_type = ""
    try:
        if not input_url or not output_url or len(input_sha) != 64:
            raise ValueError("signed_gpu_bootstrap_environment_missing")
        if execution_mode not in EXECUTION_MODES:
            raise ValueError("signed_gpu_bootstrap_execution_mode_invalid")
        _download_signed_input(
            input_url,
            bundle,
            max_bytes=(
                CANARY_MAX_INPUT_BYTES
                if execution_mode == EXECUTION_MODE_NEW_SITE_CANARY
                else (
                    FIVE_POLICY_MAX_INPUT_BYTES
                    if execution_mode == EXECUTION_MODE_IDENTITY_SMOKE
                    else MAX_INPUT_BYTES
                )
            ),
        )
        if execution_mode == EXECUTION_MODE_NEW_SITE_CANARY:
            extracted_input = extract_canary_input_bundle(
                bundle_path=bundle,
                expected_bundle_sha256=input_sha,
                output_dir=extracted,
            )
            campaign = run_skeleton_only_canary(
                protocol_path=extracted_input["protocol_path"],
                background_path=extracted_input["background_path"],
                cohort_path=(
                    "/opt/blueprint/frozen/warehouse_policy_cohort_v2_joint_position.json"
                ),
                checkpoint_inventory_path=(
                    "/opt/blueprint/frozen/openpi_polaris_checkpoint_inventory.json"
                ),
                menagerie_root="/opt/mujoco-menagerie/franka_emika_panda",
                output_dir=campaign_output,
                initial_camera_paths=extracted_input.get("initial_camera_paths") or None,
            )
        elif execution_mode == EXECUTION_MODE_IDENTITY_SMOKE:
            extracted_input = extract_five_policy_input_bundle(
                bundle_path=bundle,
                expected_bundle_sha256=input_sha,
                output_dir=extracted,
            )
            campaign = run_five_policy_identity_smoke(
                extracted=extracted_input,
                output_dir=campaign_output,
            )
        else:
            extracted_input = extract_private_input_bundle(
                bundle_path=bundle,
                expected_bundle_sha256=input_sha,
                output_dir=extracted,
            )
            campaign = run_openpi_policy_ranking_gpu_campaign(
                cohort_path=(
                    "/opt/blueprint/frozen/warehouse_policy_cohort_v2_joint_position.json"
                ),
                checkpoint_inventory_path=(
                    "/opt/blueprint/frozen/openpi_polaris_checkpoint_inventory.json"
                ),
                captured_site_background_path=extracted_input["background_path"],
                scene_backgrounds=extracted_input["scene_backgrounds"],
                menagerie_root="/opt/mujoco-menagerie/franka_emika_panda",
                output_dir=campaign_output,
                policy_ids=POLICY_IDS,
            )
    except Exception as exc:  # noqa: BLE001 - upload a secret-safe terminal envelope
        failure_type = type(exc).__name__
        campaign_output.mkdir(parents=True, exist_ok=True)
        is_canary = execution_mode == EXECUTION_MODE_NEW_SITE_CANARY
        is_identity_smoke = execution_mode == EXECUTION_MODE_IDENTITY_SMOKE
        failure_claim_boundary = (
            {
                "ranking_accuracy": False,
                "physical_success": False,
                "captured_site_transfer_validation": False,
                "phase_b_confirmation": False,
                "independently_attributable_matching_physical_outcomes_present": False,
                "result_type": "bounded_nonconfirmatory_technical_diagnostic",
            }
            if is_canary
            else (
                {
                    "real_checkpoint_inference_observed": False,
                    "actions_executed": False,
                    "policy_ranking": False,
                    "task_success": False,
                    "physical_robot_execution": False,
                }
                if is_identity_smoke
                else {
                    "learned_policy_simulator_execution": False,
                    "prospective_captured_site_ranking": False,
                    "prospective_controlled_warehouse_ranking": False,
                    "warehouse_ranking_is_independent_physical_answer_key": False,
                    "site_specific_physical_success_proven": False,
                    "physical_robot_endpoint_contacted": False,
                    "physical_robot_operated": False,
                    "wam_executed": False,
                }
            )
        )
        failure_manifest: dict[str, Any] = {
            "schema_version": (
                "new_site_diagnostic_canary_gpu.v1"
                if is_canary
                else (
                    "five_policy_identity_smoke_result.v1"
                    if is_identity_smoke
                    else "openpi_policy_ranking_gpu_job.v1"
                )
            ),
            "status": "blocked",
            "execution_mode": execution_mode,
            "blockers": [f"openpi_gpu_bootstrap_failed:{failure_type}"],
            "claim_boundary": failure_claim_boundary,
        }
        if is_canary:
            failure_manifest.update(
                {
                    "arm_id": (extracted_input.get("manifest") or {}).get("arm_id"),
                    "protocol_sha256": (
                        extracted_input.get("manifest") or {}
                    ).get("protocol_sha256"),
                    "canary": None,
                }
            )
        elif is_identity_smoke:
            failure_manifest.update(
                {
                    "expected_candidate_count": 5,
                    "completed_identity_query_count": 0,
                    "query_receipts": [],
                }
            )
        else:
            failure_manifest.update(
                {
                    "inputs": {
                        "scenes": list(
                            (extracted_input.get("manifest") or {}).get("scenes") or []
                        ),
                        "policy_ids": list(POLICY_IDS),
                    },
                    "policy_runs": [],
                    "rankings": {},
                }
            )
        failure_manifest["manifest_sha256"] = canonical_sha256(failure_manifest)
        write_json(
            campaign_output
            / (
                "new_site_diagnostic_canary_gpu.json"
                if is_canary
                else (
                    "five_policy_identity_smoke_result.json"
                    if is_identity_smoke
                    else "openpi_policy_ranking_gpu_job.json"
                )
            ),
            failure_manifest,
        )
        campaign = failure_manifest
    if not output_url:
        raise ValueError("signed_gpu_bootstrap_output_environment_missing")
    output_archive_base = root / "openpi-policy-ranking-output"
    archive_path = Path(
        shutil.make_archive(str(output_archive_base), "zip", root_dir=campaign_output)
    )
    upload_status = _upload_output(output_url, archive_path)
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed" if campaign.get("status") == "completed" else "blocked",
        "execution_mode": execution_mode,
        "input_bundle_sha256": input_sha,
        "input_manifest": extracted_input.get("manifest"),
        "campaign_manifest_sha256": campaign.get("manifest_sha256"),
        "failure_type": failure_type or None,
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
    build.add_argument(
        "--scene-kind",
        choices=tuple(sorted(SCENE_KINDS)),
        default="captured_3dgs",
    )
    multi = subparsers.add_parser("build-multi-input")
    multi.add_argument("--spec", required=True)
    multi.add_argument("--output", required=True)
    canary = subparsers.add_parser("build-canary-input")
    canary.add_argument("--protocol", required=True)
    canary.add_argument("--background", required=True)
    canary.add_argument("--output", required=True)
    canary.add_argument("--arm", choices=("skeleton_only",), required=True)
    canary.add_argument("--native-camera-canary-result")
    background = subparsers.add_parser("materialize-canary-background")
    background.add_argument("--source", required=True)
    background.add_argument("--output", required=True)
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
            source_scene_kind=args.scene_kind,
        )
        write_json(Path(args.output).with_suffix(".receipt.json"), receipt)
        return 0
    if args.command == "build-multi-input":
        spec = json.loads(Path(args.spec).expanduser().read_text(encoding="utf-8"))
        scenes = spec.get("scenes") if isinstance(spec, Mapping) else None
        if not isinstance(scenes, list):
            parser.error("multi-input spec requires a scenes list")
        receipt = build_multi_scene_private_input_bundle(
            scenes=scenes,
            output_zip=args.output,
        )
        write_json(Path(args.output).with_suffix(".receipt.json"), receipt)
        return 0
    if args.command == "build-canary-input":
        receipt = build_canary_input_bundle(
            protocol_path=args.protocol,
            background_path=args.background,
            output_zip=args.output,
            arm_id=args.arm,
            native_camera_canary_result_path=args.native_camera_canary_result,
        )
        if receipt.get("schema_version") != CANARY_INPUT_RECEIPT_SCHEMA_VERSION:
            raise RuntimeError("canary_input_receipt_schema_invalid")
        write_json(Path(args.output).with_suffix(".receipt.json"), receipt)
        return 0
    if args.command == "materialize-canary-background":
        receipt = materialize_canary_background(
            source_path=args.source,
            output_path=args.output,
        )
        write_json(Path(args.output).with_suffix(".receipt.json"), receipt)
        return 0
    result = run_signed_gpu_bootstrap(workspace=args.workspace)
    return 0 if result["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "build_canary_input_bundle",
    "build_multi_scene_private_input_bundle",
    "build_private_input_bundle",
    "extract_private_input_bundle",
]
