"""Seal one exact G1 candidate, native scene and current internal authority."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from pathlib import PurePosixPath
import stat
import tempfile
import zipfile

from .decision_evidence_contracts import canonical_digest
from .native_g1_beta_assets import selected_asset_inventory, needs_pi_tokenizer
from .native_g1_beta_session import (BUNDLE_SCHEMA, ENTRYPOINT, MANIFEST, REQUIRED_ENTRIES,
    RESULT_FILENAME, PROVIDER_KIND, validate_session_authority, transfer_quote)
from .native_g1_provider_bundle import (_contract_dependency, _files, _runtime_code_files,
    _review_g1_runtime_wheels, _verify_embedded_pi_tokenizer)
from .native_g1_publisher_source_stage import verify_g1_publisher_source, SOURCE_REPOSITORY
from .native_task_arena_bundle import verify_native_task_arena_packet
from .native_task_isaaclab_launch import NATIVE_TASK_ARENA_IMAGE
from .native_task_runtime_source_packet import verify_native_task_runtime_source_packet

BUNDLE_NAME = "native_g1_beta_provider_bundle.zip"


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _entrypoint():
    return '''#!/usr/bin/env bash
set -u
RUNTIME_DIR="$(cd "$(dirname "$0")" && pwd)"
OUT_DIR="${BLUEPRINT_ADP_ARENA_OUTPUT_DIR:-$RUNTIME_DIR/../runtime_output}"
mkdir -p "$OUT_DIR"
cd "$RUNTIME_DIR"
export BLUEPRINT_G1_PINNED_ISAAC_IMAGE="nvcr.io/nvidia/isaac-sim:6.0.1@sha256:b1c542b2ecc549b3d1ebb78c25664aa3bacba1709e6ad8e0a68e09426d57dedb"
/isaac-sim/python.sh -m blueprint_pipeline.native_task_runtime_source_provision \\
 --source-receipt "$RUNTIME_DIR/native_task_runtime_sources/native_task_runtime_source_packet.v1.json" \\
 --source-packet "$RUNTIME_DIR/native_task_runtime_sources/native_task_runtime_sources.zip" \\
 --extraction-dir "$RUNTIME_DIR/provisioned_runtime_sources" \\
 --output "$OUT_DIR/native_task_runtime_source_provisioning.v1.json" --simulator-root /isaac-sim
provision_rc=$?
if [ "$provision_rc" -eq 0 ]; then
 /isaac-sim/python.sh -m blueprint_pipeline.native_g1_beta_runtime --runtime-root "$RUNTIME_DIR" --output-dir "$OUT_DIR"
 runner_rc=$?
else
 runner_rc="$provision_rc"
fi
if [ ! -f "$OUT_DIR/native_g1_beta_session_result.v1.json" ]; then
 /isaac-sim/python.sh -m blueprint_pipeline.native_g1_beta_runtime --runtime-root "$RUNTIME_DIR" \\
   --output-dir "$OUT_DIR" --failure-only g1_beta_worker_failed_without_result
 runner_rc=2
fi
exit "$runner_rc"
'''


def build_bundle(*, job_dir: Path, authority, spec, implementation_commit: str,
                 tokenizer_dir: Path | None = None):
    from .beta_policy_screening_g1 import validate_execution_spec
    from .native_g1_beta_dependency_profile import artifact_manifest, DEPENDENCY_MANIFEST_FILENAME, DEPENDENCY_MANIFEST_SHA256
    from .beta_policy_screening_launch import _bound_path
    authority = validate_session_authority(authority)
    spec = validate_execution_spec(spec, candidate_id=authority["candidate_ids"][0])
    if (authority["implementation_commit"] != implementation_commit
            or authority["runtime_inputs_digest"] != spec["execution_spec_digest"]
            or authority["rights_authority"] != spec["rights_authority"]):
        raise ValueError("g1_beta_bundle_authority_mismatch")
    records = spec["native_g1"]
    packet, receipt, packet_files = verify_native_task_arena_packet(_bound_path(records["scene_packet"]).parent)
    source = _bound_path(records["publisher_source"]).parent
    publisher = verify_g1_publisher_source(source)
    publisher_receipt = json.loads(_bound_path(records["publisher_source"]).read_text())
    runtime_receipt = _bound_path(records["runtime_source"])
    runtime = verify_native_task_runtime_source_packet(runtime_receipt)
    if (runtime.get("runtime_profile") != "unitree_g1" or runtime.get("redistribution_permitted") is not True
            or receipt["arena_scene_plan_digest"] != spec["scene_plan_digest"]
            or json.loads((packet / "native_task_arena_packet_request.v1.json").read_text()).get("g1_scene_derivation", {}).get("source_packet_receipt_digest")
                != spec["rights_authority"]["source_packet_receipt_digest"]
            or publisher != publisher_receipt):
        raise ValueError("g1_beta_bundle_native_inputs_changed")
    package = Path(__file__).resolve().parent
    repository = package.parents[1]
    review = _review_g1_runtime_wheels(runtime, repository / "docs/runtime_dependency_license_policy.json")
    candidates = authority["candidate_ids"]
    assets = selected_asset_inventory(candidates)
    tokenizer = None
    if needs_pi_tokenizer(candidates):
        from .native_g1_beta_tokenizer import resolve_selected_tokenizer
        tokenizer_dir, tokenizer = resolve_selected_tokenizer(candidates, explicit_dir=tokenizer_dir)
    job = Path(job_dir)
    if not job.is_absolute() or job.exists() or job.is_symlink() or not job.parent.is_dir():
        raise ValueError("g1_beta_bundle_destination_invalid")
    sources = {}
    for row in packet_files:
        sources["provider_runtime/inputs/scene_packet/" + row["relative_path"]] = packet / row["relative_path"]
    # Keep only tracked source files and Git metadata needed to verify the exact
    # detached checkout. Never ship local hooks, logs or credential-bearing config.
    import subprocess
    tracked = subprocess.run(["git", "ls-files", "-z"], cwd=source / "source",
        check=True, capture_output=True).stdout.decode().split("\0")
    for relative in filter(None, tracked):
        sources["provider_runtime/publisher-source/source/" + relative] = source / "source" / relative
    for path in _files(source / "source" / ".git"):
        relative = path.relative_to(source / "source" / ".git").as_posix()
        if relative in {"HEAD", "shallow", "packed-refs"} or relative.startswith(("objects/", "refs/")):
            sources["provider_runtime/publisher-source/source/.git/" + relative] = path
    for path in _runtime_code_files(package):
        sources["provider_runtime/blueprint_pipeline/" + path.relative_to(package).as_posix()] = path
    dependency_profile = artifact_manifest()
    sources["provider_runtime/" + DEPENDENCY_MANIFEST_FILENAME] = package / "_vendor" / DEPENDENCY_MANIFEST_FILENAME
    dependency, contract_sources = _contract_dependency()
    for relative, path in contract_sources:
        sources["provider_runtime/" + relative] = path
    for name in ("g1_humanoidarena_checkpoint_inventory.v1.json", "g1_sonic_default_asset_inventory.v1.json",
                 "g1_humanoidarena_lerobot_pi_py312_linux_x86_64.requirements.txt"):
        sources["configs/" + name] = repository / "configs" / name
    if tokenizer is not None:
        sources["configs/g1_paligemma_tokenizer_inventory.v1.json"] = repository / "configs/g1_paligemma_tokenizer_inventory.v1.json"
        for row in tokenizer["files"]:
            sources["provider_runtime/inputs/pi_tokenizer/" + row["path"]] = tokenizer_dir / row["path"]
    for name in ("fetch_g1_humanoidarena_checkpoint.py", "fetch_g1_sonic_assets.py"):
        sources["provider_runtime/scripts/" + name] = repository / "scripts" / name
    model_review = _bound_path(records["model_terms_review"])
    sources["docs/arm_decision_proof_v1/manifests/" + model_review.name] = model_review
    sources["docs/runtime_dependency_license_policy.json"] = repository / "docs/runtime_dependency_license_policy.json"
    sources["provider_runtime/publisher-source/native_g1_publisher_source_stage.v1.json"] = _bound_path(records["publisher_source"])
    sources["provider_runtime/native_task_runtime_sources/native_task_runtime_source_packet.v1.json"] = runtime_receipt
    if "navigation_authority" in records:
        sources["provider_runtime/inputs/navigation_authority.json"] = _bound_path(records["navigation_authority"])
    literals = {
        "provider_runtime/inputs/session_authority.json": _json_bytes(authority),
        "provider_runtime/inputs/policy_execution_spec.json": _json_bytes(spec),
        "provider_runtime/publisher-source/source/.git/config": (
            "[core]\n\trepositoryformatversion = 0\n\tbare = false\n\tfilemode = false\n"
            "[remote \"origin\"]\n\turl = " + SOURCE_REPOSITORY + "\n").encode(),
        ENTRYPOINT: _entrypoint().encode(),
    }
    # A checkout without its index reports every tracked source file deleted.
    # Rebuild a minimal index from the verified commit, rather than exporting
    # the host's index extensions/stat cache or disabling the clean-tree check.
    with tempfile.TemporaryDirectory(prefix="g1-beta-git-index-") as temporary:
        index = Path(temporary) / "index"
        subprocess.run(["git", "read-tree", "HEAD"], cwd=source / "source",
            env={**os.environ, "GIT_INDEX_FILE": str(index)}, check=True, capture_output=True)
        literals["provider_runtime/publisher-source/source/.git/index"] = index.read_bytes()
    file_rows = []
    for relative, path in sources.items():
        if path.is_symlink() or not path.is_file():
            raise ValueError("g1_beta_bundle_source_missing_or_symlink")
        file_rows.append({"path": relative, "size_bytes": path.stat().st_size, "sha256": _sha(path)})
    file_rows += [{"path": key, "size_bytes": len(value), "sha256": "sha256:" + hashlib.sha256(value).hexdigest()}
                  for key, value in literals.items()]
    with zipfile.ZipFile(runtime["verified_packet_path"]) as runtime_archive:
        runtime_expanded_bytes = sum(info.file_size for info in runtime_archive.infolist())
        native_wheel_expanded_bytes = 0
        for row in runtime.get("runtime_dependency_wheels", []):
            with runtime_archive.open(row["archive_path"]) as incoming, zipfile.ZipFile(incoming) as wheel:
                native_wheel_expanded_bytes += sum(info.file_size for info in wheel.infolist())
    scene_packet_expanded_bytes = sum((packet / row["relative_path"]).stat().st_size for row in packet_files)
    bundle_expanded_bound_bytes = sum(row["size_bytes"] for row in file_rows) + 16 * 1024**2
    quoted_disk = transfer_quote(candidate_ids=candidates, runtime_packet_bytes=runtime["packet_size_bytes"],
        bundle_bytes=1, runtime_expanded_bytes=runtime_expanded_bytes,
        scene_packet_expanded_bytes=scene_packet_expanded_bytes, cell_count=len(authority["cells"]),
        bundle_expanded_bound_bytes=bundle_expanded_bound_bytes, native_wheel_expanded_bytes=native_wheel_expanded_bytes)["disk_gb"]
    manifest = {"schema_version": BUNDLE_SCHEMA, "status": "ready", "provider_bundle_kind": PROVIDER_KIND,
        "implementation_commit": implementation_commit, "container_image": NATIVE_TASK_ARENA_IMAGE,
        "candidate_ids": candidates, "cell_ids": [cell["cell_id"] for cell in authority["cells"]], "authority_digest": authority["authority_digest"],
        "runtime_inputs_digest": authority["runtime_inputs_digest"], "task_success_contract_digest": authority["task_success_contract_digest"],
        "expected_output_filename": RESULT_FILENAME, "runtime_entrypoint": ENTRYPOINT,
        "claim_ceiling": "development_only", "candidate_policy_queried": False,
        "provider_zero_required_after_return": True, "maximum_provider_allocations": 1, "retry_cap": 0,
        "runtime_source_packet": {"runtime_profile": "unitree_g1", "receipt_digest": runtime["receipt_digest"],
            "packet_sha256": runtime["packet_sha256"], "packet_size_bytes": runtime["packet_size_bytes"],
            "packet_path": runtime["verified_packet_path"], "transport": "content_addressed_external_layer.v1",
            "embedded_in_provider_bundle": False},
        "runtime_dependency_license_review": review, "selected_asset_inventory": assets,
        "dependency_profile": {"manifest_digest": dependency_profile["manifest_digest"], "manifest_sha256": DEPENDENCY_MANIFEST_SHA256},
        "runtime_expanded_bytes": runtime_expanded_bytes, "scene_packet_expanded_bytes": scene_packet_expanded_bytes,
        "bundle_expanded_bound_bytes": bundle_expanded_bound_bytes, "native_wheel_expanded_bytes": native_wheel_expanded_bytes,
        "provider_disk_gb": quoted_disk,
        "pi_tokenizer_assets": tokenizer, "contract_python_dependencies": [dependency],
        "files": sorted(file_rows, key=lambda row: row["path"])}
    manifest["manifest_digest"] = canonical_digest(manifest, digest_field="manifest_digest")
    if len(_json_bytes(manifest)) > 8 * 1024**2:
        raise ValueError("g1_beta_bundle_manifest_size_exceeded")
    job.mkdir()
    bundle = job / BUNDLE_NAME
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
        for relative, path in sources.items():
            archive.write(path, relative)
        for relative, data in literals.items():
            info = zipfile.ZipInfo(relative)
            info.external_attr = (stat.S_IFREG | (0o755 if relative == ENTRYPOINT else 0o600)) << 16
            archive.writestr(info, data)
        for relative in ("objects/info", "refs"):
            archive.writestr("provider_runtime/publisher-source/source/.git/" + relative + "/", b"")
        archive.writestr(MANIFEST, _json_bytes(manifest))
    receipt = {**manifest, "bundle_path": str(bundle), "bundle_size_bytes": bundle.stat().st_size,
        "bundle_sha256": _sha(bundle), "transfer_quote": transfer_quote(candidate_ids=candidates,
        runtime_packet_bytes=runtime["packet_size_bytes"], bundle_bytes=bundle.stat().st_size,
        runtime_expanded_bytes=runtime_expanded_bytes, scene_packet_expanded_bytes=scene_packet_expanded_bytes,
        cell_count=len(authority["cells"]), bundle_expanded_bound_bytes=bundle_expanded_bound_bytes,
        native_wheel_expanded_bytes=native_wheel_expanded_bytes)}
    if receipt["bundle_size_bytes"] > bundle_expanded_bound_bytes:
        raise ValueError("g1_beta_bundle_zip_size_exceeded")
    destination = job / (BUNDLE_SCHEMA + ".json")
    destination.write_bytes(_json_bytes(receipt))
    return load_verified_bundle(destination, expected_implementation_commit=implementation_commit)


def validate_bundle_archive(archive: zipfile.ZipFile, manifest):
    from .beta_policy_screening_g1 import validate_execution_spec
    from .native_g1_beta_dependency_profile import artifact_manifest, DEPENDENCY_MANIFEST_FILENAME, DEPENDENCY_MANIFEST_SHA256
    names = archive.namelist()
    profile = artifact_manifest()
    if ("provider_runtime/" + DEPENDENCY_MANIFEST_FILENAME not in names
            or "sha256:" + hashlib.sha256(archive.read("provider_runtime/" + DEPENDENCY_MANIFEST_FILENAME)).hexdigest() != DEPENDENCY_MANIFEST_SHA256
            or manifest.get("dependency_profile") != {"manifest_digest": profile["manifest_digest"], "manifest_sha256": DEPENDENCY_MANIFEST_SHA256}):
        raise ValueError("g1_beta_bundle_dependency_profile_changed")
    if len(names) != len(set(names)) or not set(REQUIRED_ENTRIES).issubset(names):
        raise ValueError("g1_beta_bundle_required_entries_invalid")
    if (manifest.get("schema_version") != BUNDLE_SCHEMA or manifest.get("status") != "ready"
            or manifest.get("provider_bundle_kind") != PROVIDER_KIND
            or manifest.get("container_image") != NATIVE_TASK_ARENA_IMAGE
            or manifest.get("expected_output_filename") != RESULT_FILENAME
            or manifest.get("runtime_entrypoint") != ENTRYPOINT
            or manifest.get("manifest_digest") != canonical_digest(manifest, digest_field="manifest_digest")):
        raise ValueError("g1_beta_bundle_manifest_invalid")
    rows = manifest.get("files") or []
    if len(rows) != len({row["path"] for row in rows}) or {r["path"] for r in rows} != {
            info.filename for info in archive.infolist() if not info.is_dir() and info.filename != MANIFEST}:
        raise ValueError("g1_beta_bundle_file_closure_changed")
    for row in rows:
        pure = PurePosixPath(row["path"])
        info = archive.getinfo(row["path"])
        if pure.is_absolute() or ".." in pure.parts or info.file_size != row["size_bytes"] or stat.S_ISLNK(info.external_attr >> 16):
            raise ValueError("g1_beta_bundle_member_invalid")
        digest = hashlib.sha256()
        with archive.open(info) as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        if "sha256:" + digest.hexdigest() != row["sha256"]:
            raise ValueError("g1_beta_bundle_file_bytes_changed")
    authority = validate_session_authority(json.loads(archive.read("provider_runtime/inputs/session_authority.json")))
    spec = validate_execution_spec(json.loads(archive.read("provider_runtime/inputs/policy_execution_spec.json")),
                                   candidate_id=authority["candidate_ids"][0])
    if (manifest["authority_digest"] != authority["authority_digest"]
            or manifest["runtime_inputs_digest"] != spec["execution_spec_digest"]
            or manifest["candidate_ids"] != authority["candidate_ids"]
            or manifest["cell_ids"] != [cell["cell_id"] for cell in authority["cells"]]
            or spec["rights_authority"] != authority["rights_authority"]
            or manifest["selected_asset_inventory"] != selected_asset_inventory(authority["candidate_ids"])):
        raise ValueError("g1_beta_bundle_selected_binding_changed")
    if needs_pi_tokenizer(authority["candidate_ids"]):
        _verify_embedded_pi_tokenizer(archive, manifest)
    elif manifest.get("pi_tokenizer_assets") is not None or any("/pi_tokenizer/" in name for name in names):
        raise ValueError("g1_beta_unselected_tokenizer_forbidden")
    return dict(manifest)


def load_verified_bundle(receipt_path: Path, *, expected_implementation_commit: str):
    path = Path(receipt_path)
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ValueError("g1_beta_bundle_receipt_missing")
    receipt = json.loads(path.read_text())
    bundle = Path(str(receipt.get("bundle_path") or ""))
    if (not bundle.is_absolute() or bundle.is_symlink() or not bundle.is_file()
            or receipt.get("implementation_commit") != expected_implementation_commit
            or bundle.stat().st_size != receipt.get("bundle_size_bytes") or _sha(bundle) != receipt.get("bundle_sha256")):
        raise ValueError("g1_beta_bundle_bytes_changed")
    manifest = {key: value for key, value in receipt.items()
                if key not in {"bundle_path", "bundle_size_bytes", "bundle_sha256", "transfer_quote"}}
    with zipfile.ZipFile(bundle) as archive:
        if json.loads(archive.read(MANIFEST)) != manifest:
            raise ValueError("g1_beta_bundle_receipt_changed")
        validate_bundle_archive(archive, manifest)
    if receipt["transfer_quote"] != transfer_quote(candidate_ids=receipt["candidate_ids"],
            runtime_packet_bytes=receipt["runtime_source_packet"]["packet_size_bytes"], bundle_bytes=receipt["bundle_size_bytes"],
            runtime_expanded_bytes=receipt["runtime_expanded_bytes"], scene_packet_expanded_bytes=receipt["scene_packet_expanded_bytes"],
            cell_count=len(receipt["cell_ids"]), bundle_expanded_bound_bytes=receipt["bundle_expanded_bound_bytes"],
            native_wheel_expanded_bytes=receipt["native_wheel_expanded_bytes"]):
        raise ValueError("g1_beta_bundle_transfer_quote_changed")
    if receipt["provider_disk_gb"] != receipt["transfer_quote"]["disk_gb"] or receipt["bundle_size_bytes"] > receipt["bundle_expanded_bound_bytes"]:
        raise ValueError("g1_beta_bundle_transfer_quote_changed")
    return receipt
