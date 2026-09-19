"""Standard-library bootstrap for an admitted website geometry worker.

Runs inside the allocator's exact base image. Install only the wheel and hashed
dependency files sealed into the input bundle, then enter the existing worker.
This file allocates no resources and accepts no caller-supplied shell commands.
"""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import urllib.request
import zipfile


def _sha(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return "sha256:" + value.hexdigest()


def _download(url, path, digest, limit):
    # URLs come only from the admitted allocator environment or pinned model
    # constants below. They are never printed or included in result artifacts.
    with urllib.request.urlopen(url, timeout=180) as response, path.open("xb") as target:
        size = 0
        for chunk in iter(lambda: response.read(1024 * 1024), b""):
            size += len(chunk)
            if size > limit:
                raise ValueError("website_worker_download_oversized")
            target.write(chunk)
    if _sha(path) != digest:
        raise ValueError("website_worker_download_digest_mismatch")


def install_runtime(root):
    root.mkdir(parents=True, exist_ok=False)
    receipt_path, bundle_path = root / "receipt.json", root / "inputs.zip"
    _download(os.environ["BLUEPRINT_RECONSTRUCTION_INPUT_RECEIPT_GET_URL"], receipt_path,
              os.environ["BLUEPRINT_RECONSTRUCTION_INPUT_RECEIPT_FILE_DIGEST"], 8 * 1024**2)
    receipt = json.loads(receipt_path.read_text())
    if (receipt.get("operation") != "website_mapanything"
            or receipt.get("source_commit_sha") != os.environ["BLUEPRINT_SOURCE_COMMIT"]
            or receipt.get("worker_image_digest") != os.environ["BLUEPRINT_CONTAINER_IMAGE_DIGEST"]
            or receipt.get("operation_request_digest") != os.environ["BLUEPRINT_RECONSTRUCTION_OPERATION_REQUEST_DIGEST"]):
        raise ValueError("website_worker_receipt_binding_mismatch")
    size = receipt.get("bundle_bytes")
    if type(size) is not int or not 0 < size <= 512 * 1024**2:
        raise ValueError("website_worker_input_size_invalid")
    _download(os.environ["BLUEPRINT_RECONSTRUCTION_INPUT_BUNDLE_GET_URL"], bundle_path,
              os.environ["BLUEPRINT_RECONSTRUCTION_INPUT_BUNDLE_DIGEST"], size)
    files = {}
    with zipfile.ZipFile(bundle_path) as archive:
        if len(archive.namelist()) != len(set(archive.namelist())):
            raise ValueError("website_worker_duplicate_bundle_member")
        for row in receipt["artifact_members"]:
            if row["role"] not in {"worker_wheel", "worker_dependencies"}:
                continue
            member = row["archive_path"]
            info = archive.getinfo(member)
            if info.file_size != row["bytes"] or info.file_size > 128 * 1024**2:
                raise ValueError("website_worker_runtime_size_invalid")
            destination = root / PurePosixPath(member).name
            if destination.exists():
                raise ValueError("website_worker_runtime_name_conflict")
            destination.write_bytes(archive.read(info))
            if _sha(destination) != row["digest"]:
                raise ValueError("website_worker_runtime_digest_mismatch")
            files.setdefault(row["role"], []).append(destination)
    wheels, dependencies = files.get("worker_wheel", []), files.get("worker_dependencies", [])
    if len(dependencies) != 1 or len(wheels) != 3 or any(path.suffix != ".whl" for path in wheels):
        raise ValueError("website_worker_runtime_missing")
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "--require-hashes",
                    "--no-cache-dir", "-r", str(dependencies[0])], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "--no-cache-dir", *map(str, wheels)], check=True)
    return root


def main():
    root = install_runtime(Path("/tmp/blueprint-website-worker"))
    from blueprint_pipeline.website_mapanything_operation import (
        MODEL_ROOT, MODEL_REVISION, MODEL_CHECKPOINT_DIGEST, MODEL_CONFIG_DIGEST,
    )
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    for name, digest, limit in (("model.safetensors", MODEL_CHECKPOINT_DIGEST, 5 * 1024**3),
                                ("config.json", MODEL_CONFIG_DIGEST, 1024**2)):
        target = MODEL_ROOT / name
        _download("https://huggingface.co/facebook/map-anything-apache/resolve/" + MODEL_REVISION + "/" + name,
                  target, digest, limit)
    # UniCeption calls DINOv2 through Torch Hub. Prepopulate its cache from an
    # exact source archive so that call cannot silently select newer code.
    revision = "7764ea0f912e53c92e82eb78a2a1631e92725fc8"
    archive_path = root / "dinov2.zip"
    _download("https://codeload.github.com/facebookresearch/dinov2/zip/" + revision, archive_path,
              "sha256:04276715cddb29d45d05bff3a6fc132224dc27749b279ac98ad2ce4620e20d48", 8 * 1024**2)
    os.environ["TORCH_HOME"] = str(root / "torch")
    cache = root / "torch" / "hub" / "facebookresearch_dinov2_main"
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            parts = PurePosixPath(member.filename).parts
            if not parts or parts[0] != "dinov2-" + revision or ".." in parts or member.file_size > 8 * 1024**2:
                raise ValueError("website_worker_encoder_archive_invalid")
            target = cache.joinpath(*parts[1:])
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.read(member))
    os.environ["HF_HUB_OFFLINE"] = "1"
    # Model acquisition above is explicit and hash-verified; inference is local.
    from blueprint_pipeline.reconstruction_gpu_operation_bootstrap import run_reconstruction_gpu_operation_bootstrap
    run_reconstruction_gpu_operation_bootstrap(environment=os.environ, work_root=root / "operation")


if __name__ == "__main__":
    main()
