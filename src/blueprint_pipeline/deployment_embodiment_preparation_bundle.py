"""Seal public source bytes and the installed release for an Isaac import batch."""
from __future__ import annotations

import hashlib
from http.client import HTTPException
from pathlib import Path, PurePosixPath
import shutil
import tempfile
import zipfile

from .deployment_embodiment_assets import materialize_sources, verify_source_bundle
from .deployment_embodiment_preparation_contract import (
    INPUT_FILENAME, INPUT_SCHEMA, MAX_OUTPUT_BYTES, RESULT_FILENAME, RESULT_SCHEMA,
    read, require, sealed, sha, validate_request, write_once,
)
from .native_task_runtime_source_packet import verify_native_task_runtime_source_packet
from .native_task_isaaclab_launch import NATIVE_TASK_ARENA_IMAGE
from .safe_outbound_http import pinned_api_policy, request as http_request
from .immutable_directory_publication import publish_staged_immutable_directory

SCHEMA = "embodiment_asset_preparation_bundle.v1"
RECEIPT = "embodiment_asset_preparation_bundle.v1.json"
MANIFEST_PATH = "provider_runtime/embodiment_asset_preparation_bundle.v1.json"
WHEELS = (
    ("xacro-2.1.1-py3-none-any.whl", "c3b330ebd984a3bce6d6482e0047eae5c5333fedd49b30b9b6df863a086b35f7",
     "https://files.pythonhosted.org/packages/93/6b/3fcfd8589d0e319f5fb56f105acbe791198fd36bb54469a91fbe49d828c4/xacro-2.1.1-py3-none-any.whl"),
    ("defusedxml-0.7.1-py2.py3-none-any.whl", "a352e7e428770286cc899e2542b6cdaedb2b4953ff269a210103ec58f6198a61",
     "https://files.pythonhosted.org/packages/07/6c/aa3f2f849e01cb6a001cd8554a88d4c77c5c1a31c95bdf1cf9301e6d9ef4/defusedxml-0.7.1-py2.py3-none-any.whl"),
    ("rfc8785-0.1.4-py3-none-any.whl", "520d690b448ecf0703691c76e1a34a24ddcd4fc5bc41d589cb7c58ec651bcd48",
     "https://files.pythonhosted.org/packages/4d/78/119878110660b2ad709888c8a1614fce7e2fab39080ab960656dc8605bf6/rfc8785-0.1.4-py3-none-any.whl"),
)

ENTRYPOINT = '''#!/usr/bin/env bash
set -euo pipefail
RUNTIME_DIR="$(cd "$(dirname "$0")" && pwd)"
OUT_DIR="${BLUEPRINT_ADP_ARENA_OUTPUT_DIR:-$RUNTIME_DIR/../runtime_output}"
export BLUEPRINT_EMBODIMENT_PROVISIONING_RECEIPT="$OUT_DIR/native_task_runtime_source_provisioning.v1.json"
mkdir -p "$OUT_DIR"
cd "$RUNTIME_DIR"
finish_batch() {
  result_code=$?
  trap - EXIT
  if [ ! -f "$OUT_DIR/embodiment_asset_preparation_result.v1.json" ]; then
    /isaac-sim/python.sh - "$OUT_DIR/embodiment_asset_preparation_result.v1.json" "$result_code" <<'PY'
import json, sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({
    "schema_version": "embodiment_asset_preparation_result.v1", "status": "blocked",
    "claim_ceiling": "robot_asset_import_only", "candidate_policy_queried": False,
    "task_executed": False, "provider_teardown_verified": False,
    "blockers": ["asset_preparation_entrypoint_failed"], "exit_code": int(sys.argv[2])}) + "\\n")
PY
  fi
  exit "$result_code"
}
trap finish_batch EXIT
/isaac-sim/python.sh -m pip install --no-index --no-deps --no-cache-dir "$RUNTIME_DIR"/import_wheels/*.whl
/isaac-sim/python.sh -m blueprint_pipeline.native_task_runtime_source_provision \
  --source-receipt "$RUNTIME_DIR/native_task_runtime_sources/native_task_runtime_source_packet.v1.json" \
  --source-packet "$RUNTIME_DIR/native_task_runtime_sources/native_task_runtime_sources.zip" \
  --extraction-dir "$RUNTIME_DIR/provisioned_runtime_sources" \
  --output "$OUT_DIR/native_task_runtime_source_provisioning.v1.json" --simulator-root /isaac-sim
/isaac-sim/python.sh -m blueprint_pipeline.deployment_embodiment_preparation_worker batch \
  --runtime "$RUNTIME_DIR" --output "$OUT_DIR"
'''


def _wheel_bytes(url):
    policy = pinned_api_policy("https://files.pythonhosted.org", max_response_bytes=2 * 1024**2)
    return http_request(url, method="GET", policy=policy, timeout_seconds=90,
                        max_response_bytes=2 * 1024**2).body


def _archive_verified(zip_out: zipfile.ZipFile, source: Path, row: dict) -> None:
    """Stream ``source`` into the archive; the stored bytes must be the ones verified.

    Stored, not deflated: the packet is itself a ZIP.
    """
    info = zipfile.ZipInfo.from_file(source, row["path"])
    digest = hashlib.sha256()
    with source.open("rb") as stream, zip_out.open(info, "w") as member:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
            member.write(chunk)
    require(info.file_size == row["size_bytes"] and "sha256:" + digest.hexdigest() == row["sha256"],
            "runtime_source_packet_changed")


def build_bundle(*, request: dict, runtime_source_receipt: Path, source_root: Path,
                 output: Path, materialize=materialize_sources, wheel_get=_wheel_bytes) -> dict:
    request = validate_request(request)
    if (output / RECEIPT).exists():
        existing = load_verified_bundle(output / RECEIPT, expected_implementation_commit=request["release_commit"])
        require(existing["request_digest"] == request["request_digest"], "bundle_request_conflict")
        return existing
    output.parent.mkdir(parents=True, exist_ok=True)
    source_root.mkdir(parents=True, exist_ok=True)
    runtime_packet = verify_native_task_runtime_source_packet(runtime_source_receipt)
    with tempfile.TemporaryDirectory(prefix=".robot-bundle-", dir=output.parent) as temporary:
        staged = Path(temporary)
        runtime = staged / "provider_runtime"
        runtime.mkdir()
        sources = []
        for identity in request["embodiment_ids"]:
            try:
                receipt = materialize(identity, source_root / identity)
                verify_source_bundle(source_root / identity)
                shutil.copytree(source_root / identity, runtime / "robot_sources" / identity)
                sources.append({"embodiment_id": identity, "status": "source_ready", "bundle_digest": receipt["bundle_digest"],
                                "recipe_digest": receipt["recipe_digest"],
                                "source_bytes": sum(row["size_bytes"] for row in receipt["files"])})
            except (OSError, ValueError, HTTPException) as exc:
                sources.append({"embodiment_id": identity, "status": "source_unavailable", "reason": str(exc)[:500]})
        source_summary = sealed({"request_digest": request["request_digest"], "results": sources}, "summary_digest")
        write_once(output.parent / ("source-preparation." + source_summary["summary_digest"][7:] + ".json"), source_summary)
        require(any(row["status"] == "source_ready" for row in sources), "no_source_ready_for_import")
        # Only program and catalog bytes are included; never captures, site media,
        # environment files, credentials, work directories or caller code.
        package = Path(__file__).resolve().parent
        shutil.copytree(package, runtime / "blueprint_pipeline", ignore=shutil.ignore_patterns(
            "__pycache__", "*.pyc", "*.pyo"))
        wheels = runtime / "import_wheels"
        wheels.mkdir()
        for name, expected, url in WHEELS:
            path = wheels / name
            path.write_bytes(wheel_get(url))
            require(sha(path) == "sha256:" + expected, "import_wheel_changed")
        native = runtime / "native_task_runtime_sources"
        native.mkdir()
        shutil.copyfile(runtime_source_receipt, native / "native_task_runtime_source_packet.v1.json")
        # The verified runtime packet (~4.3 GB) is archived from where it was
        # verified. Staging a copy first held it on disk twice beside the ZIP.
        packet_row = {"path": (native / "native_task_runtime_sources.zip").relative_to(staged).as_posix(),
                      "size_bytes": runtime_packet["packet_size_bytes"], "sha256": runtime_packet["packet_sha256"]}
        inputs = sealed({"schema_version": INPUT_SCHEMA, "request": request, "sources": sources,
                         "runtime_source_digest": runtime_packet["receipt_digest"], "maximum_output_bytes": MAX_OUTPUT_BYTES,
                         "claim_ceiling": "robot_asset_import_only", "candidate_policy_queried": False}, "inputs_digest")
        write_once(runtime / INPUT_FILENAME, inputs)
        entry = runtime / "run_adp_arena_provider_runtime.sh"
        entry.write_text(ENTRYPOINT, encoding="utf-8")
        entry.chmod(0o755)
        files = sorted([{"path": p.relative_to(staged).as_posix(), "size_bytes": p.stat().st_size, "sha256": sha(p)}
                        for p in staged.rglob("*") if p.is_file()] + [packet_row],
                       key=lambda row: PurePosixPath(row["path"]))
        manifest = sealed({"schema_version": SCHEMA, "status": "ready", "implementation_commit": request["release_commit"],
            "request_digest": request["request_digest"], "inputs_digest": inputs["inputs_digest"], "inputs": inputs,
            "container_image": NATIVE_TASK_ARENA_IMAGE,
            "expected_output_filename": RESULT_FILENAME, "expected_result_schema": RESULT_SCHEMA,
            "maximum_output_bytes": MAX_OUTPUT_BYTES, "files": files, "candidate_policy_queried": False,
            "runtime_source_packet": {"receipt_digest": runtime_packet["receipt_digest"],
                "packet_size_bytes": runtime_packet["packet_size_bytes"]}}, "manifest_digest")
        write_once(staged / MANIFEST_PATH, manifest)
        published = staged
        archive = published / "embodiment_asset_preparation.zip"
        with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=3) as zip_out:
            for row in files:
                if row is packet_row:
                    _archive_verified(zip_out, Path(runtime_packet["verified_packet_path"]), row)
                else:
                    zip_out.write(staged / row["path"], row["path"])
            zip_out.write(staged / MANIFEST_PATH, MANIFEST_PATH)
        # Copied retained sources preserve read-only directory modes. Only this
        # disposable staging copy is made writable for cleanup.
        for directory in (runtime, *(p for p in runtime.rglob("*") if p.is_dir())):
            directory.chmod(0o700)
        shutil.rmtree(runtime)
        result = sealed({**manifest, "bundle_path": str((output / archive.name).resolve()), "bundle_size_bytes": archive.stat().st_size,
                         "bundle_sha256": sha(archive)}, "bundle_digest")
        write_once(published / RECEIPT, result)
        publish_staged_immutable_directory(staging=published, destination=output, manifest_name=RECEIPT,
                                           output_exists_code="embodiment_preparation_bundle_destination_appeared")
    return result


def load_verified_bundle(receipt_path: str | Path, *, expected_implementation_commit: str, allow_expired: bool = False) -> dict:
    from .decision_evidence_contracts import canonical_digest
    value = read(Path(receipt_path), limit=32 * 1024**2)
    require(value.get("schema_version") == SCHEMA and value.get("status") == "ready"
            and value.get("implementation_commit") == expected_implementation_commit
            and value.get("container_image") == NATIVE_TASK_ARENA_IMAGE
            and value.get("bundle_digest") == canonical_digest(value, digest_field="bundle_digest"), "bundle_invalid")
    request = validate_request(value["inputs"]["request"], allow_expired=allow_expired)
    require(request["release_commit"] == expected_implementation_commit
            and request["request_digest"] == value["request_digest"], "bundle_release_mismatch")
    archive = Path(value["bundle_path"])
    require(not archive.is_symlink() and archive.stat().st_size == value["bundle_size_bytes"]
            and sha(archive) == value["bundle_sha256"], "bundle_bytes_changed")
    with zipfile.ZipFile(archive) as zipped:
        manifest = validate_archive(zipped)
        require(all(value[k] == v for k, v in manifest.items()), "bundle_manifest_changed")
    return value


def validate_archive(archive: zipfile.ZipFile) -> dict:
    """Streaming verification shared with provider preflight; no large read-all."""
    import json
    from .decision_evidence_contracts import canonical_digest
    require(archive.getinfo(MANIFEST_PATH).file_size <= 32 * 1024**2, "manifest_bound_exceeded")
    manifest = json.loads(archive.read(MANIFEST_PATH))
    require(manifest.get("schema_version") == SCHEMA and manifest.get("manifest_digest")
            == canonical_digest(manifest, digest_field="manifest_digest") and manifest.get("status") == "ready"
            and manifest.get("container_image") == NATIVE_TASK_ARENA_IMAGE, "manifest_invalid")
    validate_request(manifest["inputs"]["request"], allow_expired=True)
    names = archive.namelist()
    require(len(names) == len(set(names)) and set(names) == {r["path"] for r in manifest["files"]} | {MANIFEST_PATH},
            "bundle_inventory_changed")
    for row in manifest["files"]:
        path = Path(row["path"])
        info = archive.getinfo(row["path"])
        require(not path.is_absolute() and ".." not in path.parts and path.parts[0] == "provider_runtime"
                and info.file_size == row["size_bytes"] and (info.external_attr >> 16) & 0o170000 != 0o120000,
                "bundle_member_invalid")
        with archive.open(row["path"]) as stream:
            require("sha256:" + hashlib.file_digest(stream, "sha256").hexdigest() == row["sha256"], "bundle_member_changed")
    require(manifest["inputs"]["inputs_digest"] == canonical_digest(manifest["inputs"], digest_field="inputs_digest")
            and json.loads(archive.read("provider_runtime/" + INPUT_FILENAME)) == manifest["inputs"], "inputs_changed")
    return manifest


def main(argv: list[str] | None = None) -> int:
    """Rebuild a sealed preparation artifact without allocating a provider."""
    import argparse
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--runtime-source-receipt", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    result = build_bundle(request=read(args.request), runtime_source_receipt=args.runtime_source_receipt,
                          source_root=args.source_root, output=args.output)
    print(json.dumps({key: result[key] for key in (
        "status", "implementation_commit", "request_digest", "inputs_digest",
        "bundle_path", "bundle_sha256", "bundle_digest",
    )}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
