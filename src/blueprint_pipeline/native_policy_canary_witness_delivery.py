"""Private paired-cell witness delivery, streaming exact bytes in both directions."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import http.client
import json
import os
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlsplit
import zipfile

from .decision_evidence_contracts import canonical_digest
from .native_policy_canary_control_gate import _file
from .provider_signed_object_binding import signed_output_object_binding_sha256


def _connection(url: str):
    parsed = urlsplit(url)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        raise RuntimeError("strict_paired_private_delivery_capability_invalid")
    connection = http.client.HTTPSConnection(parsed.hostname, parsed.port or 443, timeout=120)
    target = parsed.path or "/"
    if parsed.query:
        target += "?" + parsed.query
    return connection, target


def _stream_roundtrip(archive: Path, put_url: str, get_url: str, binding_digest: str) -> None:
    connection, target = _connection(put_url)
    try:
        connection.putrequest("PUT", target)
        connection.putheader("Content-Length", str(archive.stat().st_size))
        connection.putheader("Content-Type", "application/zip")
        connection.putheader("x-amz-meta-blueprint-witness-binding", binding_digest)
        connection.endheaders()
        with archive.open("rb") as source:
            for chunk in iter(lambda: source.read(1024*1024), b""):
                connection.send(chunk)
        response = connection.getresponse()
        if response.status not in {200, 201, 204}:
            raise RuntimeError("strict_paired_witness_upload_failed")
        response.read(4096)
    finally:
        connection.close()
    connection, target = _connection(get_url)
    try:
        connection.request("GET", target)
        response = connection.getresponse()
        if response.status != 200:
            raise RuntimeError("strict_paired_witness_readback_failed")
        digest, total = hashlib.sha256(), 0
        for chunk in iter(lambda: response.read(1024*1024), b""):
            total += len(chunk)
            if total > archive.stat().st_size:
                raise RuntimeError("strict_paired_witness_readback_size_mismatch")
            digest.update(chunk)
        expected = _file(archive, archive.parent)
        if total != expected["size_bytes"] or "sha256:"+digest.hexdigest() != expected["sha256"]:
            raise RuntimeError("strict_paired_witness_readback_digest_mismatch")
    finally:
        connection.close()


def transfer_paired_witness(*, root: Path, output_root: Path, authority: Mapping[str, Any],
                            runtime_inputs_digest: str, runtime: Path) -> dict[str, Any]:
    def secret(name: str) -> str:
        value = os.environ.get(name)
        if not value:
            raise RuntimeError("strict_paired_delivery_private_capabilities_missing")
        return Path(value).read_text().strip()
    put_url = secret("BLUEPRINT_POLICY_CANARY_PAIRED_PUT_URL_FILE")
    get_url = secret("BLUEPRINT_POLICY_CANARY_PAIRED_GET_URL_FILE")
    binding = json.loads(secret("BLUEPRINT_POLICY_CANARY_PAIRED_DELIVERY_AUTHORITY_FILE"))
    manifest_identity = json.loads((runtime / "adp_arena_provider_manifest.json").read_text())
    provider_archive = Path(os.environ["BLUEPRINT_VAST_WORK_DIR"]) / "adp_arena_provider_runtime_bundle.zip"
    now = datetime.now(timezone.utc)
    generated = datetime.fromisoformat(binding["generated_at"].replace("Z", "+00:00"))
    expires = datetime.fromisoformat(binding["expires_at"].replace("Z", "+00:00"))
    if (binding.get("schema_version") != "native_task_arena_paired_delivery_authority.v1"
            or binding.get("binding_digest") != canonical_digest(binding, digest_field="binding_digest")
            or binding.get("run_id") != authority.get("run_id")
            or binding.get("authority_digest") != authority["authority_digest"]
            or binding.get("runtime_inputs_digest") != runtime_inputs_digest
            or binding.get("implementation_commit") != manifest_identity.get("implementation_commit")
            or binding.get("provider_bundle_sha256") != _file(provider_archive, provider_archive.parent)["sha256"]
            or binding.get("output_url_object_binding_sha256") != signed_output_object_binding_sha256(put_url, get_url)
            or not urlsplit(put_url).path.endswith(".paired-witness.zip")
            or binding.get("content_type") != "application/zip" or not generated <= now < expires):
        raise RuntimeError("strict_paired_delivery_authority_mismatch")
    files = [_file(path, root) for path in sorted(root.rglob("*")) if path.is_file()]
    manifest = {"schema_version": "policy_canary_paired_witness.v1", "run_id": authority.get("run_id"),
                "authority_digest": authority["authority_digest"], "runtime_inputs_digest": runtime_inputs_digest,
                "execution_release": authority.get("execution_release"), "files": files,
                "new_learned_episodes_executed": 0, "source": "retained_first_quick10_cell",
                "manifest_digest": ""}
    manifest["manifest_digest"] = canonical_digest(manifest, digest_field="manifest_digest")
    maximum = binding.get("maximum_archive_bytes")
    if not isinstance(maximum, int) or isinstance(maximum, bool) or not 0 < maximum <= 32*1024**3:
        raise RuntimeError("strict_paired_delivery_size_limit_missing")
    if sum(row["size_bytes"] for row in files) > maximum:
        raise RuntimeError("strict_paired_delivery_size_limit_exceeded")
    archive = output_root / "paired_witness.zip"
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_STORED, allowZip64=True) as bundle:
        bundle.writestr("paired_witness_manifest.v1.json", json.dumps(manifest, sort_keys=True))
        for record in files:
            bundle.write(root / record["relative_path"], record["relative_path"])
    if archive.stat().st_size > maximum:
        raise RuntimeError("strict_paired_delivery_size_limit_exceeded")
    _stream_roundtrip(archive, put_url, get_url, binding["binding_digest"])
    record = _file(archive, output_root)
    result = {"schema_version": "policy_canary_paired_delivery.v1", "status": "uploaded_and_readback_verified",
              "archive_sha256": record["sha256"], "archive_size_bytes": record["size_bytes"],
              "witness_manifest_digest": manifest["manifest_digest"], "authority_digest": authority["authority_digest"],
              "runtime_inputs_digest": runtime_inputs_digest, "run_id": authority.get("run_id"),
              "result_digest": ""}
    result["result_digest"] = canonical_digest(result, digest_field="result_digest")
    (output_root / "paired_delivery_receipt.v1.json").write_text(json.dumps(result, indent=2))
    (output_root / "paired_witness_manifest.v1.json").write_text(json.dumps(manifest, indent=2))
    # Original episode files remain; the verified remote ZIP is only a transfer
    # witness. Avoid nesting a second multi-GB copy in the final result archive.
    archive.unlink()
    return result
