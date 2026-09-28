from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json

import pytest

from blueprint_pipeline import native_policy_canary_witness_delivery as delivery
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.native_policy_canary_control_gate import _file
from blueprint_pipeline.provider_signed_object_binding import signed_output_object_binding_sha256


def _binding(tmp_path, monkeypatch):
    root = tmp_path / "pair"
    root.mkdir()
    (root / "episode.json").write_text('{"status":"completed"}')
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    (runtime / "adp_arena_provider_manifest.json").write_text(json.dumps({"implementation_commit": "a"*40}))
    archive = tmp_path / "adp_arena_provider_runtime_bundle.zip"
    archive.write_bytes(b"sealed-provider-code")
    put, get = "https://private.example/run.paired-witness.zip?put=secret", "https://private.example/run.paired-witness.zip?get=secret"
    authority = {"run_id": "fixture_run", "authority_digest": "sha256:"+"1"*64}
    inputs = "sha256:"+"2"*64
    now = datetime.now(timezone.utc)
    binding = {"schema_version": "native_task_arena_paired_delivery_authority.v1", **authority,
               "runtime_inputs_digest": inputs, "implementation_commit": "a"*40,
               "provider_bundle_sha256": _file(archive, tmp_path)["sha256"],
               "generated_at": (now-timedelta(minutes=1)).isoformat(), "expires_at": (now+timedelta(minutes=5)).isoformat(),
               "output_url_object_binding_sha256": signed_output_object_binding_sha256(put, get),
               "content_type": "application/zip", "maximum_archive_bytes": 100_000, "binding_digest": ""}
    binding["binding_digest"] = canonical_digest(binding, digest_field="binding_digest")
    values = {"BLUEPRINT_POLICY_CANARY_PAIRED_PUT_URL_FILE": put,
              "BLUEPRINT_POLICY_CANARY_PAIRED_GET_URL_FILE": get,
              "BLUEPRINT_POLICY_CANARY_PAIRED_DELIVERY_AUTHORITY_FILE": json.dumps(binding)}
    for key, value in values.items():
        path = tmp_path / key
        path.write_text(value)
        path.chmod(0o600)
        monkeypatch.setenv(key, str(path))
    monkeypatch.setenv("BLUEPRINT_VAST_WORK_DIR", str(tmp_path))
    return root, runtime, authority, inputs, binding


def test_witness_uses_separate_private_capabilities_and_binds_exact_runtime_bytes(tmp_path, monkeypatch):
    root, runtime, authority, inputs, binding = _binding(tmp_path, monkeypatch)
    sent = []
    monkeypatch.setattr(delivery, "_stream_roundtrip", lambda *args: sent.append(args))
    result = delivery.transfer_paired_witness(root=root, output_root=tmp_path, authority=authority,
                                             runtime_inputs_digest=inputs, runtime=runtime)
    assert result["status"] == "uploaded_and_readback_verified"
    assert len(sent) == 1 and sent[0][3] == binding["binding_digest"]
    assert "secret" not in json.dumps(result)
    assert result["archive_size_bytes"] > 0
    assert not (tmp_path / "paired_witness.zip").exists()
    assert (tmp_path / "paired_witness_manifest.v1.json").is_file()
    assert (root / "episode.json").is_file()


@pytest.mark.parametrize("mutation", ["commit", "bundle", "final_key", "limit"])
def test_witness_refuses_wrong_code_or_final_archive_key_before_network(tmp_path, monkeypatch, mutation):
    root, runtime, authority, inputs, binding = _binding(tmp_path, monkeypatch)
    if mutation == "commit":
        (runtime / "adp_arena_provider_manifest.json").write_text(json.dumps({"implementation_commit": "b"*40}))
    elif mutation == "bundle":
        (tmp_path / "adp_arena_provider_runtime_bundle.zip").write_bytes(b"different-code")
    elif mutation == "limit":
        binding["maximum_archive_bytes"] = 1
        binding["binding_digest"] = canonical_digest(binding, digest_field="binding_digest")
        (tmp_path / "BLUEPRINT_POLICY_CANARY_PAIRED_DELIVERY_AUTHORITY_FILE").write_text(json.dumps(binding))
    else:
        (tmp_path / "BLUEPRINT_POLICY_CANARY_PAIRED_PUT_URL_FILE").write_text("https://private.example/final.zip?secret")
    monkeypatch.setattr(delivery, "_stream_roundtrip", lambda *args: pytest.fail("must not send bytes"))
    with pytest.raises((RuntimeError, ValueError)):
        delivery.transfer_paired_witness(root=root, output_root=tmp_path, authority=authority,
                                         runtime_inputs_digest=inputs, runtime=runtime)


@pytest.mark.parametrize("corrupt", [False, True])
def test_upload_and_readback_stream_actual_bytes_in_bounded_chunks(tmp_path, monkeypatch, corrupt):
    path = tmp_path / "archive.zip"
    payload = b"0123456789"*220_000
    path.write_bytes(payload)
    sent, headers, methods = [], {}, []

    class Response:
        status = 200
        def __init__(self, data):
            self.data, self.offset = data, 0
        def read(self, size):
            result = self.data[self.offset:self.offset+size]
            self.offset += len(result)
            return result

    class Connection:
        def putrequest(self, method, target):
            methods.append(method)
        def putheader(self, key, value):
            headers[key] = value
        def endheaders(self):
            pass
        def send(self, chunk):
            sent.append(chunk)
        def request(self, method, target):
            methods.append(method)
        def getresponse(self):
            return Response(b"" if methods[-1] == "PUT" else b"wrong" if corrupt else payload)
        def close(self):
            pass

    monkeypatch.setattr(delivery, "_connection", lambda url: (Connection(), "/private"))
    if corrupt:
        with pytest.raises(RuntimeError, match="readback_digest_mismatch"):
            delivery._stream_roundtrip(path, "unused-put", "unused-get", "sha256:binding")
    else:
        delivery._stream_roundtrip(path, "unused-put", "unused-get", "sha256:binding")
    assert b"".join(sent) == payload
    assert max(map(len, sent)) <= 1024*1024
    assert methods == ["PUT", "GET"]
    assert headers["Content-Length"] == str(len(payload))
    assert headers["x-amz-meta-blueprint-witness-binding"] == "sha256:binding"
