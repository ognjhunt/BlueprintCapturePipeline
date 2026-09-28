"""ADP-011/day 7: package private, generation-bound model bytes for a worker.

The builder copies bytes; graph loading occurs only in the isolated worker.
This module does not provision a worker, grant scene access or qualify a task.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import secrets
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Callable, Mapping
from urllib.parse import urlsplit

from .decision_evidence_contracts import cross_runtime_canonical_digest
from .policy_model_onnx import MAX_MODEL_BYTES, validate_model_task_binding

SOURCE_FILES = (
    "policy_model_onnx.py", "policy_model_server.py", "company_policy_proxy.py",
    "company_policy_container_contract_v2.py", "decision_evidence_contracts.py",
)


def stage_model_build_context(*, artifact: Mapping[str, Any], contract: Mapping[str, Any],
                             output: Path, allowed_bucket: str, storage_client: Any,
                             source_root: Path) -> dict[str, Any]:
    validate_model_task_binding(artifact, contract)
    uri = urlsplit(str(artifact.get("uri", "")))
    generation = str(artifact.get("storage_generation", ""))
    size = artifact.get("size_bytes")
    if (uri.scheme != "gs" or uri.netloc != allowed_bucket or not allowed_bucket
            or not uri.path.startswith("/policy-models/") or uri.query or uri.fragment
            or not re.fullmatch(r"[1-9][0-9]*", generation)
            or type(size) is not int or not 0 < size <= MAX_MODEL_BYTES):
        raise ValueError("policy_model_storage_binding_invalid")
    blob = storage_client.bucket(allowed_bucket).blob(uri.path[1:], generation=int(generation))
    blob.reload(if_generation_match=int(generation))
    if blob.size != size:
        raise ValueError("policy_model_storage_size_mismatch")
    raw = blob.download_as_bytes(if_generation_match=int(generation), checksum="auto")
    if len(raw) != size or "sha256:" + hashlib.sha256(raw).hexdigest() != artifact.get("sha256"):
        raise ValueError("policy_model_storage_digest_mismatch")
    output.mkdir(mode=0o700, parents=False, exist_ok=False)
    inputs = output / "policy-model-build-input"
    inputs.mkdir(mode=0o700)
    # Never copy the private storage URI, generation, credentials or scene files
    # into the customer-policy image. Only graph bytes and approved I/O metadata.
    manifest = {key: artifact[key] for key in ("schema_version", "sha256", "size_bytes", "interface")}
    for name, content in (("policy.onnx", raw), ("artifact.json", json.dumps(manifest, allow_nan=False).encode())):
        path = inputs / name
        path.write_bytes(content)
        path.chmod(0o600)
    sources = output / "src" / "blueprint_pipeline"
    sources.mkdir(mode=0o700, parents=True)
    for name in SOURCE_FILES:
        shutil.copyfile(source_root / "src" / "blueprint_pipeline" / name, sources / name)
    shutil.copyfile(source_root / "deploy/docker/policy_model_cpu/Dockerfile", output / "Dockerfile")
    return {
        "schema_version": "blueprint.policy_model_build_context.v1",
        "artifact_id": artifact["artifact_id"], "artifact_sha256": artifact["sha256"],
        "storage_generation": generation,
        "interface_sha256": cross_runtime_canonical_digest(artifact["interface"]),
        "contract_digest": cross_runtime_canonical_digest(contract, digest_field="contract_digest"),
        "runner_profile": "onnx_state_mlp_cpu_v1", "model_loaded_on_builder": False,
        "task_execution_proven": False,
    }


def publish_model_image(*, context: Path, image_tag: str, allowed_registry: str) -> str:
    if (not allowed_registry or not re.fullmatch(r"[a-z0-9][a-z0-9._/:-]{1,250}", image_tag)
            or image_tag.split("/", 1)[0] != allowed_registry or ":" not in image_tag.rsplit("/", 1)[-1]):
        raise ValueError("policy_model_builder_registry_not_allowed")
    # Fixed commands, no shell and no customer-selected Dockerfile/build args.
    subprocess.run(["docker", "build", "--network=default", "--tag", image_tag, str(context)],
                   check=True, timeout=600, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(["docker", "push", image_tag], check=True, timeout=600,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    result = subprocess.run(["docker", "image", "inspect", "--format", "{{json .RepoDigests}}", image_tag],
                            check=True, timeout=30, capture_output=True)
    repository = image_tag.rsplit(":", 1)[0]
    refs = json.loads(result.stdout)
    images = [ref for ref in refs if re.fullmatch(re.escape(repository) + r"@sha256:[0-9a-f]{64}", ref)]
    if len(images) != 1:
        raise ValueError("policy_model_published_digest_missing")
    return images[0]


class PrivateModelImageBuilder:
    """Trusted dispatch hook; fixed bucket/repository and secret-clean context.

    Configure this on a dedicated image builder, never on the scene worker or
    WebApp. The existing broker handles a separate worker pull credential.
    """
    def __init__(self, *, allowed_bucket: str, allowed_registry: str, repository: str,
                 storage_client: Any, source_root: Path, scratch_root: Path,
                 retain_receipt: Callable[[Mapping[str, Any]], None]):
        if (repository.split("/", 1)[0] != allowed_registry
                or not re.fullmatch(r"[a-z0-9][a-z0-9._/-]+", repository)):
            raise ValueError("policy_model_builder_repository_invalid")
        self.allowed_bucket, self.allowed_registry, self.repository = allowed_bucket, allowed_registry, repository
        self.storage_client, self.source_root, self.scratch_root = storage_client, source_root, scratch_root
        self.retain_receipt = retain_receipt

    def __call__(self, *, artifact: Mapping[str, Any], contract: Mapping[str, Any],
                 job_request: Mapping[str, Any]) -> str:
        with tempfile.TemporaryDirectory(prefix="policy-model-", dir=self.scratch_root) as temporary:
            context = Path(temporary) / "context"
            receipt = stage_model_build_context(artifact=artifact, contract=contract, output=context,
                allowed_bucket=self.allowed_bucket, storage_client=self.storage_client, source_root=self.source_root)
            identity = {**receipt, "source_sha256": {
                name: "sha256:" + hashlib.sha256((context / "src/blueprint_pipeline" / name).read_bytes()).hexdigest()
                for name in SOURCE_FILES},
                "dockerfile_sha256": "sha256:" + hashlib.sha256((context / "Dockerfile").read_bytes()).hexdigest()}
            image_tag = self.repository + ":model-" + cross_runtime_canonical_digest(identity)[7:55] + "-" + secrets.token_hex(6)
            try:
                image = publish_model_image(context=context, image_tag=image_tag, allowed_registry=self.allowed_registry)
                self.retain_receipt({**identity, "image_ref": image, "job_id": job_request.get("job_id"),
                                     "status": "packaged_pending_isolated_runner_validation"})
                return image
            finally:
                # Remove only the local tag created for this build. Private
                # registry retention follows the uploaded model's access policy.
                subprocess.run(["docker", "image", "rm", image_tag], timeout=60,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allowed-bucket", required=True)
    parser.add_argument("--image-tag")
    parser.add_argument("--allowed-registry")
    args = parser.parse_args()
    from google.cloud import storage
    receipt = stage_model_build_context(
        artifact=json.loads(args.artifact.read_text()), contract=json.loads(args.contract.read_text()),
        output=args.output, allowed_bucket=args.allowed_bucket, storage_client=storage.Client(),
        source_root=Path(__file__).resolve().parents[2],
    )
    if args.image_tag:
        receipt["image_ref"] = publish_model_image(context=args.output, image_tag=args.image_tag,
                                                  allowed_registry=args.allowed_registry or "")
    path = args.output.parent / (args.output.name + "-receipt.json")
    with path.open("x") as stream:
        stream.write(json.dumps(receipt, allow_nan=False, indent=2))
    path.chmod(0o600)


if __name__ == "__main__":
    main()
