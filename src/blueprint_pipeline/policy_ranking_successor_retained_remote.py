"""Standalone digest-bound refresh controller embedded in the Cosmos bundle."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path
from typing import Any


ROOT = Path("/workspace/blueprint_vast_probe/cosmos3_retained")
IDENTITY_PATH = ROOT / "server_identity.json"
AUDIT_PATH = ROOT / "refresh_audit.jsonl"
IMAGE_DIGEST = "sha256:6d2630c7d637b699557573f2c3fee8df5d4d0cd718977aa22549ed6a6ef30587"
CHECKPOINT = "nvidia/Cosmos3-Nano"
CHECKPOINT_REVISION = "411f42a8fdfb8c5b2583cb8786e0938f49796eaa"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_identity() -> dict[str, Any]:
    try:
        value = json.loads(IDENTITY_PATH.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _healthy(identity: dict[str, Any]) -> bool:
    try:
        pid = int(identity.get("pid"))
        os.kill(pid, 0)
        with urllib.request.urlopen(  # nosec B310 - fixed loopback Cosmos health endpoint
            "http://127.0.0.1:8001/health", timeout=10
        ) as response:
            return 200 <= int(response.status) < 300
    except (OSError, TypeError, ValueError, urllib.error.URLError):
        return False


def _https_url(value: Any) -> str:
    candidate = str(value or "")
    parsed = urllib.parse.urlsplit(candidate)
    if (
        parsed.scheme != "https"
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
    ):
        raise ValueError("refresh_signed_url_invalid")
    return candidate


def _request() -> dict[str, Any]:
    raw = sys.stdin.buffer.read(65_537)
    if not raw or len(raw) > 65_536:
        raise ValueError("refresh_request_size_invalid")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("refresh_request_not_object")
    required = {
        "schema_version",
        "action",
        "bundle_url",
        "output_put_url",
        "source_commit",
        "dirty_state_declaration",
        "runtime_bundle_sha256",
        "authorization_receipt_sha256",
        "image_digest",
        "checkpoint",
        "checkpoint_revision",
        "provider_instance_id",
        "previous_bundle_sha256",
    }
    if set(value) != required or value.get("action") != "refresh":
        raise ValueError("refresh_request_contract_invalid")
    if value.get("schema_version") != "policy_ranking_successor_refresh_request.v1":
        raise ValueError("refresh_request_schema_invalid")
    for key in (
        "source_commit",
        "runtime_bundle_sha256",
        "authorization_receipt_sha256",
        "previous_bundle_sha256",
    ):
        if not re.fullmatch(
            r"[0-9a-f]{40}" if key == "source_commit" else r"[0-9a-f]{64}",
            str(value.get(key) or ""),
        ):
            raise ValueError(f"refresh_{key}_invalid")
    if value.get("dirty_state_declaration") not in {"clean_exact_commit", "declared_dirty_overlay"}:
        raise ValueError("refresh_dirty_state_declaration_invalid")
    if value.get("image_digest") != IMAGE_DIGEST:
        raise ValueError("refresh_image_digest_changed")
    if (
        value.get("checkpoint") != CHECKPOINT
        or value.get("checkpoint_revision") != CHECKPOINT_REVISION
    ):
        raise ValueError("refresh_checkpoint_binding_changed")
    value["bundle_url"] = _https_url(value["bundle_url"])
    value["output_put_url"] = _https_url(value["output_put_url"])
    return value


def _safe_members(archive: zipfile.ZipFile) -> None:
    for info in archive.infolist():
        path = Path(info.filename)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("refresh_bundle_path_traversal")


def run_refresh() -> dict[str, Any]:
    request = _request()
    ROOT.mkdir(parents=True, exist_ok=True)
    before = _read_identity()
    before_healthy = _healthy(before)
    with tempfile.TemporaryDirectory(prefix="successor-refresh-", dir=ROOT) as temporary:
        temporary_root = Path(temporary)
        bundle = temporary_root / "runtime_bundle.zip"
        urllib.request.urlretrieve(  # nosec B310 - _https_url validated signed URL
            request["bundle_url"], bundle
        )
        actual_sha256 = _sha256(bundle)
        if actual_sha256 != request["runtime_bundle_sha256"]:
            raise ValueError("refresh_bundle_sha256_mismatch")
        runtime_root = ROOT / f"runtime_bundle_{actual_sha256}"
        with zipfile.ZipFile(bundle) as archive:
            _safe_members(archive)
            manifest = json.loads(
                archive.read("provider_runtime/wam_provider_runtime_manifest.json")
            )
            if (
                manifest.get("public_image", "").split("@")[-1] != IMAGE_DIGEST
                or manifest.get("checkpoint") != CHECKPOINT
                or manifest.get("checkpoint_revision") != CHECKPOINT_REVISION
            ):
                raise ValueError("refresh_bundle_immutable_binding_changed")
            runtime_root.mkdir(parents=True, exist_ok=True)
            archive.extractall(runtime_root)
        output_dir = ROOT / f"runtime_output_{actual_sha256}"
        environment = dict(os.environ)
        environment.update(
            {
                "BLUEPRINT_RETAIN_COSMOS_SERVER": "true",
                "BLUEPRINT_COSMOS_RETAINED_ROOT": str(ROOT),
                "BLUEPRINT_VAST_PROVIDER_OUTPUT_DIR": str(output_dir),
                "BLUEPRINT_RUNTIME_BUNDLE_SHA256": actual_sha256,
            }
        )
        completed = subprocess.run(
            ["bash", str(runtime_root / "provider_runtime/run_wam_provider_runtime.sh")],
            check=False,
            env=environment,
        )
        output_zip = temporary_root / "runtime_output.zip"
        with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(output_dir.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(output_dir).as_posix())
        upload = urllib.request.Request(
            request["output_put_url"],
            data=output_zip.read_bytes(),
            method="PUT",
            headers={"Content-Type": "application/zip"},
        )
        with urllib.request.urlopen(  # nosec B310 - _https_url validated signed URL
            upload, timeout=120
        ) as response:
            upload_status = int(response.status)
    after = _read_identity()
    after_healthy = _healthy(after)
    server_remained_loaded = bool(
        before_healthy
        and after_healthy
        and before.get("pid") == after.get("pid")
        and before.get("process_start_ticks") == after.get("process_start_ticks")
    )
    row = {
        "schema_version": "policy_ranking_successor_refresh_audit.v1",
        "refresh_time_epoch": time.time(),
        "source_commit": request["source_commit"],
        "dirty_state_declaration": request["dirty_state_declaration"],
        "runtime_bundle_sha256": request["runtime_bundle_sha256"],
        "authorization_receipt_sha256": request["authorization_receipt_sha256"],
        "image_digest": request["image_digest"],
        "checkpoint": request["checkpoint"],
        "checkpoint_revision": request["checkpoint_revision"],
        "provider_instance_id": request["provider_instance_id"],
        "previous_bundle_sha256": request["previous_bundle_sha256"],
        "new_bundle_sha256": actual_sha256,
        "process_identity_before": before,
        "process_identity_after": after,
        "cosmos_server_remained_loaded": server_remained_loaded,
        "runner_exit_code": completed.returncode,
        "output_upload_http_status": upload_status,
        "signed_urls_stored": False,
    }
    row["audit_sha256"] = hashlib.sha256(
        json.dumps(row, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    with AUDIT_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return row


def main() -> int:
    try:
        result = run_refresh()
    except Exception as exc:  # noqa: BLE001 - standalone evidence path is fail closed
        print(json.dumps({"status": "blocked", "error_type": type(exc).__name__}))
        return 2
    print(
        json.dumps(
            {
                "status": "completed" if result["runner_exit_code"] == 0 else "blocked",
                "audit_sha256": result["audit_sha256"],
                "new_bundle_sha256": result["new_bundle_sha256"],
                "server_remained_loaded": result["cosmos_server_remained_loaded"],
            },
            sort_keys=True,
        )
    )
    return 0 if result["runner_exit_code"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
