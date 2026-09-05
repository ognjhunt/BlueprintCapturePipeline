from __future__ import annotations

import hashlib
import json
import os
import pwd
from pathlib import Path

import pytest

from blueprint_pipeline import task_evaluation_installed_source_bindings as binding
from blueprint_pipeline import task_evaluation_launch_preparation_worker as worker
from blueprint_pipeline.decision_evidence_contracts import canonical_digest

COMMIT = "a" * 40
REVISION = "b" * 40
URI = f"https://huggingface.co/datasets/publisher/scene/resolve/{REVISION}/scene/asset.ply"
ACCOUNT = pwd.getpwuid(os.geteuid()).pw_name


def _sha(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: dict, *, seal: bool = False) -> None:
    if seal:
        value["receipt_digest"] = canonical_digest(value, digest_field="receipt_digest")
    path.write_text(json.dumps(value))


@pytest.fixture
def packet(tmp_path):
    root = tmp_path / "host"
    installed = root / "installed"
    installed.mkdir(parents=True)
    source = installed / "inputs" / "asset.ply"
    source.parent.mkdir()
    source.write_bytes(b"immutable publisher source")
    installation_path = installed / "public_scene_host_input_installation_receipt.v1.json"
    installation = {
        "schema_version": "public_scene_host_input_installation_receipt.v1",
        "status": "installed", "scene_id": "example",
        "service_account": ACCOUNT, "service_readable": True,
        "source_commit_sha": COMMIT, "destination_root": str(installed),
        "files": [{"role": "appearance_3dgs", "relative_path": "inputs/asset.ply",
                   "sha256": _sha(source), "size_bytes": source.stat().st_size}],
    }
    _write(installation_path, installation, seal=True)
    publisher_path = root / "publisher.json"
    publisher = {
        "schema_version": "scene_example_publisher_source_intake.v1",
        "scene_id": "example",
        "status": "publisher_pinned_sources_verified_on_production",
        "publisher_direct_download": True, "source_uploaded_by_blueprint": False,
        "public_redistribution_allowed": False,
        "artifacts": [{"publisher_url": URI, "publisher_revision": REVISION,
                       "sha256": _sha(source), "size_bytes": source.stat().st_size}],
    }
    _write(publisher_path, publisher)
    config = {
        "installation_receipt_path": str(installation_path),
        "publisher_intake_path": str(publisher_path),
        "publisher_intake_sha256": _sha(publisher_path),
    }
    return root, source, installation_path, publisher_path, config


def _load(packet, *, commit=COMMIT, requested_uris=None):
    root, _, _, _, config = packet
    return binding.load_installed_source_bindings(
        expected_source_commit=commit, service_account=ACCOUNT,
        environment={binding.BINDINGS_ENV: json.dumps([config])},
        approved_roots=(root,), requested_uris=requested_uris,
    )


def _materialize(packet, output, *, uri=URI, digest=None, resolver=None):
    _, source, *_ = packet
    output.mkdir()
    def no_network(*args):
        pytest.fail("Raw source must never trigger network fetch or upload")
    return worker._materialize_reference_records(
        references=[{"uri": uri, "digest": digest or _sha(source),
                     "size_bytes": source.stat().st_size, "contract_path": "scene.appearance.representation"}],
        input_root=output, allowed_uri_prefixes=["s3://private/inputs/"],
        fetcher=no_network, installed_sources=resolver or _load(packet),
    )


def test_host_only_reference_copies_verified_installed_bytes_without_network(packet, tmp_path):
    rows, count = _materialize(packet, tmp_path / "output")
    assert count == 1
    assert Path(rows[0]["materialized_path"]).read_bytes() == packet[1].read_bytes()
    evidence = rows[0]["host_source_readback"]
    assert evidence["publisher_uri"] == URI
    assert evidence["network_fetch_performed"] is False
    assert evidence["raw_source_uploaded"] is False
    assert evidence["publisher_intake_sha256"] == packet[4]["publisher_intake_sha256"]


def test_empty_operator_configuration_retains_existing_behavior():
    assert not binding.load_installed_source_bindings(
        expected_source_commit="", service_account=ACCOUNT, environment={},
    ).sources


def test_stale_execution_commit_fails(packet):
    with pytest.raises(binding.InstalledSourceBindingError, match="installation_invalid"):
        _load(packet, commit="c" * 40)


def test_tampered_publisher_receipt_fails_its_operator_pin(packet):
    packet[3].write_text(packet[3].read_text() + "\n")
    with pytest.raises(binding.InstalledSourceBindingError, match="publisher_receipt_readback_mismatch"):
        _load(packet)


def test_installation_self_digest_is_required(packet):
    value = json.loads(packet[2].read_text())
    value["receipt_digest"] = "sha256:" + "0" * 64
    _write(packet[2], value)
    with pytest.raises(binding.InstalledSourceBindingError, match="installation_invalid"):
        _load(packet)


@pytest.mark.parametrize("field,value", [
    ("status", "prepared"), ("service_readable", False),
    ("service_account", "not-the-executing-account"), ("destination_root", "/tmp"),
])
def test_invalid_installation_claims_fail_even_with_self_digest(packet, field, value):
    receipt = json.loads(packet[2].read_text())
    receipt[field] = value
    _write(packet[2], receipt, seal=True)
    with pytest.raises(binding.InstalledSourceBindingError, match="installation_invalid"):
        _load(packet)


def test_unlisted_publisher_uri_does_not_fall_back_to_network(packet, tmp_path):
    with pytest.raises(worker.TaskEvaluationLaunchPreparationWorkerError, match="prefix_not_allowed"):
        _materialize(packet, tmp_path / "output", uri=URI + ".unlisted")


def test_listed_uri_with_wrong_request_digest_does_not_fall_back(packet, tmp_path):
    with pytest.raises(worker.TaskEvaluationLaunchPreparationWorkerError, match="request_identity_mismatch"):
        _materialize(packet, tmp_path / "output", digest="sha256:" + "0" * 64)


def test_symlink_member_is_rejected(packet, tmp_path):
    source = packet[1]
    saved = tmp_path / "outside.ply"
    saved.write_bytes(source.read_bytes())
    source.unlink()
    source.symlink_to(saved)
    with pytest.raises(binding.InstalledSourceBindingError, match="symlink_forbidden"):
        _load(packet)


def test_member_path_escape_is_rejected(packet):
    receipt = json.loads(packet[2].read_text())
    receipt["files"][0]["relative_path"] = "../publisher.json"
    _write(packet[2], receipt, seal=True)
    with pytest.raises(binding.InstalledSourceBindingError, match="member_path_invalid"):
        _load(packet)


def test_operator_receipt_path_must_be_under_host_roots(packet, tmp_path):
    outside = tmp_path / "outside.json"
    outside.write_bytes(packet[3].read_bytes())
    packet[4]["publisher_intake_path"] = str(outside)
    with pytest.raises(binding.InstalledSourceBindingError, match="outside_configured_roots"):
        _load(packet)


def test_tampered_source_is_rejected_even_if_cache_exists(packet, tmp_path):
    resolver = _load(packet)
    old_digest = _sha(packet[1])
    output = tmp_path / "output"
    output.mkdir()
    (output / old_digest.removeprefix("sha256:")).write_bytes(packet[1].read_bytes())
    packet[1].write_bytes(b"changed publisher source")
    with pytest.raises(worker.TaskEvaluationLaunchPreparationWorkerError, match="file_readback_mismatch"):
        worker._materialize_reference_records(
            references=[{"uri": URI, "digest": old_digest, "size_bytes": 26}],
            input_root=output, allowed_uri_prefixes=["s3://private/inputs/"],
            fetcher=lambda *args: pytest.fail("Network forbidden"), installed_sources=resolver,
        )


@pytest.mark.parametrize("revision,url", [
    ("main", URI.replace(REVISION, "main")),
    (REVISION, URI + "?token=forbidden"),
    (REVISION, URI.replace("huggingface.co", "other.example")),
])
def test_nonpinned_or_changed_publisher_identity_fails(packet, revision, url):
    publisher = json.loads(packet[3].read_text())
    publisher["artifacts"][0].update(publisher_revision=revision, publisher_url=url)
    _write(packet[3], publisher)
    packet[4]["publisher_intake_sha256"] = _sha(packet[3])
    with pytest.raises(binding.InstalledSourceBindingError, match="publisher_uri_not_pinned"):
        _load(packet)

def test_public_preparation_entrypoint_loads_operator_binding_after_commit_validation(
    packet, tmp_path, monkeypatch,
):
    from tests.test_task_evaluation_launch_preparation_contract import (
        test_configuration_request as configuration_request,
    )
    from tests.test_task_evaluation_launch_preparation_worker import request_with_fetchable_bytes

    request, payloads = request_with_fetchable_bytes(configuration_request())
    request["scene"]["appearance"]["representation"] = {
        "uri": URI, "digest": _sha(packet[1]), "size_bytes": packet[1].stat().st_size,
    }
    calls = []
    def load_operator_binding(**kwargs):
        calls.append(kwargs)
        return _load(packet)
    monkeypatch.setattr(worker, "load_installed_source_bindings", load_operator_binding)
    def fetch(uri, destination, limit):
        assert uri != URI, "Publisher source must remain on host"
        destination.write_bytes(payloads[uri])
    result = worker.materialize_preparation_references(
        request=request, input_root=tmp_path / "output",
        allowed_uri_prefixes=["s3://blueprint-production-inputs/"],
        service_account=ACCOUNT, source_commit=COMMIT, fetcher=fetch,
    )
    assert calls == [{
        "expected_source_commit": COMMIT, "service_account": ACCOUNT,
        "requested_uris": [row["uri"] for row in worker.collect_preparation_references(request)],
    }]
    assert any(row.get("host_source_readback") for row in result["references"])


@pytest.mark.parametrize("requested_uris", [[], ["s3://private/inputs/request.json"], [URI + ".other"]])
def test_unrelated_stale_installation_is_not_bound_to_current_request(packet, requested_uris):
    assert not _load(
        packet, commit="c" * 40, requested_uris=requested_uris,
    ).sources


def test_requested_stale_installation_still_fails_without_network_fallback(packet):
    with pytest.raises(binding.InstalledSourceBindingError, match="installation_invalid"):
        _load(packet, commit="c" * 40, requested_uris=[URI])


def test_requested_tampered_installation_still_fails(packet):
    packet[1].write_bytes(b"tampered source")
    with pytest.raises(binding.InstalledSourceBindingError, match="file_readback_mismatch"):
        _load(packet, requested_uris=[URI])


def test_publisher_pin_is_verified_before_unrelated_installation_is_skipped(packet):
    packet[3].write_text(packet[3].read_text() + "\n")
    with pytest.raises(binding.InstalledSourceBindingError, match="publisher_receipt_readback_mismatch"):
        _load(packet, commit="c" * 40, requested_uris=["s3://private/inputs/request.json"])


def test_unrelated_stale_binding_does_not_block_another_requested_binding(packet, tmp_path):
    second = globals()["packet"].__wrapped__(tmp_path / "second")
    new_uri = URI.replace("/scene/asset.ply", "/another/asset.ply")
    publisher = json.loads(second[3].read_text())
    publisher["artifacts"][0]["publisher_url"] = new_uri
    _write(second[3], publisher)
    second[4]["publisher_intake_sha256"] = _sha(second[3])
    installation = json.loads(packet[2].read_text())
    installation["source_commit_sha"] = "c" * 40
    _write(packet[2], installation, seal=True)
    resolver = binding.load_installed_source_bindings(
        expected_source_commit=COMMIT, service_account=ACCOUNT,
        environment={binding.BINDINGS_ENV: json.dumps([packet[4], second[4]])},
        approved_roots=(packet[0], second[0]), requested_uris=[new_uri],
    )
    assert set(resolver.sources) == {new_uri}
    assert resolver.resolve(new_uri, _sha(second[1]), second[1].stat().st_size) is not None


def test_s3_only_worker_request_ignores_stale_unrelated_installation(
    packet, tmp_path, monkeypatch,
):
    from tests.test_task_evaluation_launch_preparation_contract import (
        test_configuration_request as configuration_request,
    )
    from tests.test_task_evaluation_launch_preparation_worker import request_with_fetchable_bytes

    request, payloads = request_with_fetchable_bytes(configuration_request())
    installation = json.loads(packet[2].read_text())
    installation["source_commit_sha"] = "c" * 40
    _write(packet[2], installation, seal=True)
    def configured_loader(**kwargs):
        return binding.load_installed_source_bindings(
            **kwargs, approved_roots=(packet[0],),
            environment={binding.BINDINGS_ENV: json.dumps([packet[4]])},
        )
    monkeypatch.setattr(worker, "load_installed_source_bindings", configured_loader)
    original_collect = worker.collect_preparation_references
    collections = []
    def collect_once(value):
        collections.append(value)
        return original_collect(value)
    monkeypatch.setattr(worker, "collect_preparation_references", collect_once)
    def fetch(uri, destination, limit):
        destination.write_bytes(payloads[uri])
    result = worker.materialize_preparation_references(
        request=request, input_root=tmp_path / "s3-output",
        allowed_uri_prefixes=["s3://blueprint-production-inputs/"],
        service_account=ACCOUNT, source_commit=COMMIT, fetcher=fetch,
    )
    assert result["reference_count"] > 0
    assert len(collections) == 1
    assert all(not row.get("host_source_readback") for row in result["references"])
