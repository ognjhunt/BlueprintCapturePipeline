"""Hash-bound GPU input bundle for the NVIDIA Warehouse camera canary."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import urllib.parse
import urllib.request
import zipfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from .common import write_json
from .nvidia_warehouse_native_camera_canary import (
    isaac_sim_6_backend,
    run_native_camera_canary,
)
from .nvidia_warehouse_workcell import (
    CANARY_SPEC_SCHEMA_VERSION,
    DATASET_REVISION,
    SCHEMA_VERSION as MATERIALIZATION_SCHEMA_VERSION,
)
from .policy_ranking_thesis import canonical_sha256, file_sha256


BUNDLE_SCHEMA_VERSION = "nvidia_warehouse_native_camera_gpu_bundle.v1"
RECEIPT_SCHEMA_VERSION = "nvidia_warehouse_native_camera_gpu_bundle_receipt.v1"
BUNDLE_MANIFEST_NAME = "bundle_manifest.json"
SPEC_NAME = "native_camera_canary_spec.json"
MATERIALIZATION_MANIFEST_NAME = "materialization_manifest.json"
ASSET_PREFIX = "assets"
MAX_BUNDLE_BYTES = 512 * 1024 * 1024
MAX_UNCOMPRESSED_BYTES = 768 * 1024 * 1024
MAX_MEMBERS = 256
MAX_WORKER_OUTPUT_BYTES = 128 * 1024 * 1024
# These constants name environment variables; they do not contain credentials.
INPUT_SECRET_URL_ENV = "BLUEPRINT_NVIDIA_WAREHOUSE_CAMERA_INPUT_URL"  # nosec B105
INPUT_SHA256_ENV = "BLUEPRINT_NVIDIA_WAREHOUSE_CAMERA_INPUT_SHA256"
OUTPUT_SECRET_PUT_URL_ENV = (  # nosec B105
    "BLUEPRINT_NVIDIA_WAREHOUSE_CAMERA_OUTPUT_PUT_URL"
)
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _read_object(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"nvidia_warehouse_gpu_bundle_json_not_object:{path}")
    return dict(value)


def require_clean_bundle_source_checkout(
    *, source_commit: str, repo_root: str | Path | None = None
) -> str:
    """Require the CLI bundle identity to equal the clean source checkout."""

    root = (
        Path(repo_root).expanduser().resolve()
        if repo_root is not None
        else Path(__file__).resolve().parents[2]
    )
    try:
        head = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--verify", "HEAD^{commit}"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip().lower()
        status = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain=v1"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError("nvidia_warehouse_gpu_bundle_source_checkout_unavailable") from exc
    declared = str(source_commit or "").strip().lower()
    if not _COMMIT.fullmatch(declared) or declared != head:
        raise ValueError("nvidia_warehouse_gpu_bundle_source_commit_not_checkout_head")
    if status.strip():
        raise ValueError("nvidia_warehouse_gpu_bundle_source_checkout_not_clean")
    return head


def _validated_identity(value: Mapping[str, Any], *, field: str) -> dict[str, Any]:
    payload = dict(value)
    declared = payload.pop(field, None)
    if declared != canonical_sha256(payload):
        raise ValueError(f"nvidia_warehouse_gpu_bundle_{field}_invalid")
    payload[field] = declared
    return payload


def _safe_member(name: str) -> str:
    member = PurePosixPath(name)
    if member.is_absolute() or ".." in member.parts or name.endswith("/"):
        raise ValueError(f"nvidia_warehouse_gpu_bundle_member_unsafe:{name}")
    normalized = member.as_posix()
    if not normalized or normalized == ".":
        raise ValueError("nvidia_warehouse_gpu_bundle_member_empty")
    return normalized


def build_native_camera_gpu_bundle(
    *,
    assets_root: str | Path,
    spec_path: str | Path,
    source_commit: str,
    output_zip: str | Path,
    receipt_path: str | Path | None = None,
) -> dict[str, Any]:
    """Package the exact materialized USD closure and prospective camera spec."""

    commit = str(source_commit).strip().lower()
    if not _COMMIT.fullmatch(commit):
        raise ValueError("nvidia_warehouse_gpu_bundle_source_commit_invalid")
    root = Path(assets_root).expanduser().resolve()
    if not root.is_dir() or root.is_symlink():
        raise ValueError("nvidia_warehouse_gpu_bundle_assets_root_invalid")
    materialization_path = root / MATERIALIZATION_MANIFEST_NAME
    materialization = _validated_identity(
        _read_object(materialization_path), field="manifest_sha256"
    )
    if (
        materialization.get("schema_version") != MATERIALIZATION_SCHEMA_VERSION
        or materialization.get("status") != "completed"
        or materialization.get("dataset_revision") != DATASET_REVISION
        or materialization.get("dataset_local_dependency_closure_complete") is not True
    ):
        raise ValueError("nvidia_warehouse_gpu_bundle_materialization_invalid")

    spec_file = Path(spec_path).expanduser().resolve()
    spec = _validated_identity(_read_object(spec_file), field="spec_sha256")
    if (
        spec.get("schema_version") != CANARY_SPEC_SCHEMA_VERSION
        or spec.get("dataset_revision") != DATASET_REVISION
        or spec.get("materialization_manifest_sha256")
        != materialization.get("manifest_sha256")
        or spec.get("label_free") is not True
        or spec.get("rankings_or_policy_outcomes_accessed") is not False
        or spec.get("paid_gpu_execution_admitted") is not False
    ):
        raise ValueError("nvidia_warehouse_gpu_bundle_spec_invalid")

    files_value = materialization.get("files")
    if not isinstance(files_value, list) or not files_value:
        raise ValueError("nvidia_warehouse_gpu_bundle_materialization_files_invalid")
    asset_rows: list[dict[str, Any]] = []
    source_paths: dict[str, Path] = {}
    for value in files_value:
        if not isinstance(value, Mapping):
            raise ValueError("nvidia_warehouse_gpu_bundle_materialization_file_invalid")
        relative = _safe_member(str(value.get("relative_path") or ""))
        path = (root / relative).resolve()
        if (
            not path.is_relative_to(root)
            or not path.is_file()
            or path.is_symlink()
            or value.get("sha256") != file_sha256(path)
            or value.get("size_bytes") != path.stat().st_size
        ):
            raise ValueError(f"nvidia_warehouse_gpu_bundle_asset_invalid:{relative}")
        member = f"{ASSET_PREFIX}/{relative}"
        source_paths[member] = path
        asset_rows.append(
            {
                "member": member,
                "relative_path": relative,
                "sha256": value["sha256"],
                "size_bytes": value["size_bytes"],
            }
        )

    manifest: dict[str, Any] = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "source_commit": commit,
        "dataset_revision": DATASET_REVISION,
        "materialization_manifest_sha256": materialization["manifest_sha256"],
        "materialization_manifest_file_sha256": file_sha256(materialization_path),
        "spec_sha256": spec["spec_sha256"],
        "spec_file_sha256": file_sha256(spec_file),
        "asset_count": len(asset_rows),
        "asset_size_bytes": sum(int(row["size_bytes"]) for row in asset_rows),
        "assets": sorted(asset_rows, key=lambda row: str(row["member"])),
        "label_free": True,
        "rankings_or_policy_outcomes_accessed": False,
        "purpose": "private_internal_nvidia_warehouse_native_camera_canary",
        "claim_boundary": {
            "camera_technical_canary_only": True,
            "policy_wam_loop_proven": False,
            "ranking_accuracy": False,
            "physical_success": False,
        },
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    destination = Path(output_zip).expanduser().resolve()
    if destination.exists():
        raise FileExistsError("nvidia_warehouse_gpu_bundle_output_exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".partial")
    temporary.unlink(missing_ok=True)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr(
                BUNDLE_MANIFEST_NAME,
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            )
            archive.write(spec_file, SPEC_NAME)
            archive.write(materialization_path, MATERIALIZATION_MANIFEST_NAME)
            for member, path in sorted(source_paths.items()):
                archive.write(path, member)
        if temporary.stat().st_size > MAX_BUNDLE_BYTES:
            raise ValueError("nvidia_warehouse_gpu_bundle_too_large")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)

    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "status": "completed",
        "bundle_path": str(destination),
        "bundle_sha256": file_sha256(destination),
        "bundle_size_bytes": destination.stat().st_size,
        "manifest": manifest,
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    if receipt_path is not None:
        receipt_destination = Path(receipt_path).expanduser().resolve()
        if receipt_destination.exists():
            raise FileExistsError("nvidia_warehouse_gpu_bundle_receipt_exists")
        write_json(receipt_destination, receipt)
    return receipt


def extract_native_camera_gpu_bundle(
    *, bundle_path: str | Path, expected_sha256: str, output_dir: str | Path
) -> dict[str, Any]:
    """Validate and extract a GPU bundle without trusting archive paths."""

    bundle = Path(bundle_path).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    if (
        not bundle.is_file()
        or bundle.is_symlink()
        or bundle.stat().st_size > MAX_BUNDLE_BYTES
        or file_sha256(bundle) != expected_sha256
    ):
        raise ValueError("nvidia_warehouse_gpu_bundle_missing_unsafe_or_sha_invalid")
    if output.exists():
        raise FileExistsError("nvidia_warehouse_gpu_bundle_extract_output_exists")
    with zipfile.ZipFile(bundle) as archive:
        infos = archive.infolist()
        if len(infos) > MAX_MEMBERS or len({info.filename for info in infos}) != len(infos):
            raise ValueError("nvidia_warehouse_gpu_bundle_member_inventory_invalid")
        names = {_safe_member(info.filename) for info in infos}
        if {BUNDLE_MANIFEST_NAME, SPEC_NAME, MATERIALIZATION_MANIFEST_NAME} - names:
            raise ValueError("nvidia_warehouse_gpu_bundle_required_member_missing")
        if any(
            name not in {BUNDLE_MANIFEST_NAME, SPEC_NAME, MATERIALIZATION_MANIFEST_NAME}
            and not name.startswith(f"{ASSET_PREFIX}/")
            for name in names
        ):
            raise ValueError("nvidia_warehouse_gpu_bundle_member_allowlist_invalid")
        if sum(info.file_size for info in infos) > MAX_UNCOMPRESSED_BYTES:
            raise ValueError("nvidia_warehouse_gpu_bundle_uncompressed_size_exceeded")
        manifest_value = json.loads(archive.read(BUNDLE_MANIFEST_NAME).decode("utf-8"))
        if not isinstance(manifest_value, Mapping):
            raise ValueError("nvidia_warehouse_gpu_bundle_manifest_not_object")
        manifest = _validated_identity(manifest_value, field="manifest_sha256")
        if (
            manifest.get("schema_version") != BUNDLE_SCHEMA_VERSION
            or manifest.get("label_free") is not True
            or manifest.get("rankings_or_policy_outcomes_accessed") is not False
        ):
            raise ValueError("nvidia_warehouse_gpu_bundle_manifest_invalid")
        rows_value = manifest.get("assets")
        rows = rows_value if isinstance(rows_value, list) else []
        expected_names = {
            BUNDLE_MANIFEST_NAME,
            SPEC_NAME,
            MATERIALIZATION_MANIFEST_NAME,
            *(str(row.get("member") or "") for row in rows if isinstance(row, Mapping)),
        }
        if names != expected_names or len(rows) != manifest.get("asset_count"):
            raise ValueError("nvidia_warehouse_gpu_bundle_manifest_inventory_mismatch")

        output.mkdir(parents=True)
        try:
            for info in infos:
                name = _safe_member(info.filename)
                data = archive.read(info)
                if name == SPEC_NAME:
                    if hashlib.sha256(data).hexdigest() != manifest.get("spec_file_sha256"):
                        raise ValueError("nvidia_warehouse_gpu_bundle_spec_file_sha_invalid")
                elif name == MATERIALIZATION_MANIFEST_NAME:
                    if hashlib.sha256(data).hexdigest() != manifest.get(
                        "materialization_manifest_file_sha256"
                    ):
                        raise ValueError(
                            "nvidia_warehouse_gpu_bundle_materialization_file_sha_invalid"
                        )
                elif name.startswith(f"{ASSET_PREFIX}/"):
                    row = next(
                        (
                            value
                            for value in rows
                            if isinstance(value, Mapping) and value.get("member") == name
                        ),
                        {},
                    )
                    if (
                        hashlib.sha256(data).hexdigest() != row.get("sha256")
                        or len(data) != row.get("size_bytes")
                    ):
                        raise ValueError(f"nvidia_warehouse_gpu_bundle_asset_sha_invalid:{name}")
                target = (output / name).resolve()
                if not target.is_relative_to(output):
                    raise ValueError("nvidia_warehouse_gpu_bundle_member_escaped_output")
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
        except Exception:
            # A failed extraction is never a rerunnable or admissible input.
            for child in sorted(output.rglob("*"), reverse=True):
                if child.is_file() or child.is_symlink():
                    child.unlink(missing_ok=True)
                elif child.is_dir():
                    child.rmdir()
            output.rmdir()
            raise
    return {
        "manifest": manifest,
        "spec_path": str(output / SPEC_NAME),
        "assets_root": str(output / ASSET_PREFIX),
        "materialization_manifest_path": str(output / MATERIALIZATION_MANIFEST_NAME),
    }


def run_native_camera_gpu_bundle(
    *,
    bundle_path: str | Path,
    expected_sha256: str,
    workspace: str | Path,
) -> dict[str, Any]:
    """Execute one extracted bundle through the native Isaac backend."""

    root = Path(workspace).expanduser().resolve()
    if root.exists():
        raise FileExistsError("nvidia_warehouse_gpu_bundle_workspace_exists")
    extracted = extract_native_camera_gpu_bundle(
        bundle_path=bundle_path,
        expected_sha256=expected_sha256,
        output_dir=root / "input",
    )
    return run_native_camera_canary(
        spec_path=extracted["spec_path"],
        assets_root=extracted["assets_root"],
        output_dir=root / "output",
        backend=isaac_sim_6_backend,
    )


def _validated_https_url(value: str, *, field: str) -> str:
    url = str(value or "").strip()
    parsed = urllib.parse.urlparse(url)
    if (
        parsed.scheme != "https"
        or not parsed.netloc
        or parsed.fragment
        or parsed.username is not None
        or parsed.password is not None
        or any(character.isspace() for character in url)
    ):
        raise ValueError(f"nvidia_warehouse_gpu_worker_{field}_invalid")
    return url


def _download_https(url: str, destination: Path) -> None:
    request = urllib.request.Request(url, method="GET")
    size = 0
    # _validated_https_url rejects non-HTTPS and credential-bearing URLs.
    with urllib.request.urlopen(  # nosec B310
        request, timeout=300
    ) as response, destination.open("wb") as handle:
        while chunk := response.read(1024 * 1024):
            size += len(chunk)
            if size > MAX_BUNDLE_BYTES:
                raise ValueError("nvidia_warehouse_gpu_worker_input_too_large")
            handle.write(chunk)


def _upload_https(url: str, source: Path) -> None:
    request = urllib.request.Request(
        url,
        data=source.read_bytes(),
        method="PUT",
        headers={"Content-Type": "application/zip"},
    )
    # _validated_https_url rejects non-HTTPS and credential-bearing URLs.
    with urllib.request.urlopen(  # nosec B310
        request, timeout=300
    ) as response:
        status = int(getattr(response, "status", 0) or 0)
    if not 200 <= status < 300:
        raise ValueError(f"nvidia_warehouse_gpu_worker_output_upload_failed:{status}")


def _archive_worker_output(source: Path, destination: Path) -> None:
    files = sorted(path for path in source.rglob("*") if path.is_file())
    if not files or len(files) > MAX_MEMBERS:
        raise ValueError("nvidia_warehouse_gpu_worker_output_inventory_invalid")
    if any(path.is_symlink() for path in files):
        raise ValueError("nvidia_warehouse_gpu_worker_output_symlink_forbidden")
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.relative_to(source).as_posix())
    if destination.stat().st_size > MAX_WORKER_OUTPUT_BYTES:
        destination.unlink(missing_ok=True)
        raise ValueError("nvidia_warehouse_gpu_worker_output_too_large")


def _write_worker_failure_output(
    *, output_dir: Path, phase: str, error_type: str
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    failure_dir = output_dir / "failure"
    failure_dir.mkdir(parents=True, exist_ok=True)
    failure_path = failure_dir / "worker_failure.json"
    failure = {
        "schema_version": "nvidia_warehouse_native_camera_worker_failure.v1",
        "status": "failed",
        "phase": str(phase),
        "error_type": str(error_type),
        "failure_before_frames": True,
        "label_free": True,
        "rankings_or_policy_outcomes_accessed": False,
        "raw_secret_values_recorded": False,
    }
    write_json(failure_path, failure)
    blocker = f"native_camera_worker_failure:{error_type}"
    result: dict[str, Any] = {
        "schema_version": "nvidia_warehouse_native_camera_canary_result.v1",
        "status": "failed",
        "blockers": [blocker],
        "assessment": {
            "status": "failed",
            "blockers": [blocker],
            "views": {},
        },
        "failure_evidence": {
            "phase": str(phase),
            "error_type": str(error_type),
            "failure_before_frames": True,
            "media": [
                {
                    "relative_path": "failure/worker_failure.json",
                    "sha256": file_sha256(failure_path),
                }
            ],
        },
        "label_free": True,
        "rankings_or_policy_outcomes_accessed": False,
        "paid_policy_or_wam_model_invoked": False,
        "claim_boundary": {
            "native_scene_and_camera_technical_canary_only": True,
            "policy_wam_loop_proven": False,
            "ranking_accuracy": False,
            "physical_success": False,
            "captured_site_transfer_validation": False,
            "phase_b_confirmation": False,
        },
    }
    result["result_sha256"] = canonical_sha256(result)
    write_json(output_dir / "native_camera_canary_result.json", result)
    return result


def run_native_camera_gpu_worker(
    *,
    workspace: str | Path,
    environment: Mapping[str, str] = os.environ,
    downloader: Callable[[str, Path], None] = _download_https,
    uploader: Callable[[str, Path], None] = _upload_https,
    bundle_runner: Callable[..., Mapping[str, Any]] = run_native_camera_gpu_bundle,
) -> dict[str, Any]:
    """Download, execute, and upload one provider worker job without retaining URLs."""

    root = Path(workspace).expanduser().resolve()
    if root.exists():
        raise FileExistsError("nvidia_warehouse_gpu_worker_workspace_exists")
    input_url = _validated_https_url(
        environment.get(INPUT_SECRET_URL_ENV, ""), field="input_url"
    )
    output_url = _validated_https_url(
        environment.get(OUTPUT_SECRET_PUT_URL_ENV, ""), field="output_url"
    )
    expected_sha256 = str(environment.get(INPUT_SHA256_ENV, "")).strip().lower()
    if not _SHA256.fullmatch(expected_sha256):
        raise ValueError("nvidia_warehouse_gpu_worker_input_sha256_invalid")
    root.mkdir(parents=True)
    phase = "download"
    try:
        bundle = root / "input.zip"
        downloader(input_url, bundle)
        if not bundle.is_file() or file_sha256(bundle) != expected_sha256:
            raise ValueError("nvidia_warehouse_gpu_worker_download_sha256_mismatch")
        phase = "native_camera_execution"
        result = dict(
            bundle_runner(
                bundle_path=bundle,
                expected_sha256=expected_sha256,
                workspace=root / "execution",
            )
        )
    except Exception as exc:
        result = _write_worker_failure_output(
            output_dir=root / "execution" / "output",
            phase=phase,
            error_type=type(exc).__name__,
        )
    output_zip = root / "output.zip"
    _archive_worker_output(root / "execution" / "output", output_zip)
    receipt: dict[str, Any] = {
        "schema_version": "nvidia_warehouse_native_camera_gpu_worker_receipt.v1",
        "status": "completed",
        "canary_status": result.get("status"),
        "input_bundle_sha256": expected_sha256,
        "output_zip_sha256": file_sha256(output_zip),
        "output_zip_size_bytes": output_zip.stat().st_size,
        "input_url_recorded": False,
        "output_url_recorded": False,
        "rankings_or_policy_outcomes_accessed": False,
        "physical_robot_operated": False,
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    write_json(root / "worker_receipt.json", receipt)
    uploader(output_url, output_zip)
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("--assets-root", required=True)
    build.add_argument("--spec", required=True)
    build.add_argument("--source-commit", required=True)
    build.add_argument("--output", required=True)
    build.add_argument("--receipt", required=True)
    run = commands.add_parser("run")
    run.add_argument("--bundle", required=True)
    run.add_argument("--expected-sha256", required=True)
    run.add_argument("--workspace", required=True)
    worker = commands.add_parser("worker")
    worker.add_argument("--workspace", required=True)
    args = parser.parse_args(argv)
    if args.command == "build":
        require_clean_bundle_source_checkout(source_commit=args.source_commit)
        result = build_native_camera_gpu_bundle(
            assets_root=args.assets_root,
            spec_path=args.spec,
            source_commit=args.source_commit,
            output_zip=args.output,
            receipt_path=args.receipt,
        )
    elif args.command == "run":
        result = run_native_camera_gpu_bundle(
            bundle_path=args.bundle,
            expected_sha256=args.expected_sha256,
            workspace=args.workspace,
        )
    else:
        result = run_native_camera_gpu_worker(workspace=args.workspace)
    print(json.dumps({"status": result["status"]}, sort_keys=True))
    return 0 if result["status"] in {"completed", "passed"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
