"""Development-only authorization, recovery and per-body publication contracts."""
from __future__ import annotations

import os

from copy import deepcopy
import hashlib
from pathlib import Path, PurePosixPath
import time
from types import SimpleNamespace

import pytest

from blueprint_pipeline import deployment_embodiment_preparation_bundle as bundle_module
from blueprint_pipeline import deployment_embodiment_preparation_contract as contract
from blueprint_pipeline import deployment_embodiment_preparation_controller as controller
from blueprint_pipeline import deployment_embodiment_preparation_lane as lane

IDS = ["universal_robots_ur5e", "universal_robots_ur10e", "yaskawa_gp12"]
COMMIT = "1" * 40


def request(ids=None):
    return contract.sealed({"schema_version": contract.REQUEST_SCHEMA, "authorization_id": "development-test",
        "release_commit": COMMIT, "embodiment_ids": ids or IDS, "max_spend_usd": 4.,
        "max_hourly_rate_usd": .8, "ttl_seconds": 7200, "expires_at_epoch": time.time() + 86400,
        "authorization_reference": "test://development-only"}, "request_digest")


@pytest.mark.parametrize("field,value", [("authorization_id", 123), ("release_commit", 1),
    ("embodiment_ids", ["unknown"]), ("embodiment_ids", [IDS[0], IDS[0]]),
    ("max_spend_usd", 4.01), ("max_hourly_rate_usd", .81), ("max_spend_usd", 10**400),
    ("ttl_seconds", True), ("expires_at_epoch", float("nan"))])
def test_request_rejects_unbounded_or_repurposed_operator_grants(field, value):
    value_request = request()
    value_request[field] = value
    with pytest.raises(ValueError):
        contract.validate_request(value_request)


def test_stable_authorization_is_consumed_once_even_with_changed_digest_or_job(tmp_path, monkeypatch):
    monkeypatch.setattr(lane, "prepare_consumption_root", lambda: tmp_path)
    original = request()
    bundle = {"bundle_sha256": "sha256:" + "a" * 64, "inputs_digest": "sha256:" + "b" * 64}
    first = lane.consume_once(original, bundle, tmp_path / "job-a")
    changed = contract.sealed({**original, "authorization_reference": "test://different-reference"}, "request_digest")
    assert changed["request_digest"] != original["request_digest"]
    assert lane.resource_name(changed) == lane.resource_name(original)
    with pytest.raises(ValueError, match="already_consumed"):
        lane.consume_once(changed, bundle, tmp_path / "job-b")
    assert contract.read(lane.consumption_path(original)) == first


def test_door_enqueue_is_durable_idempotent_and_does_not_launch(tmp_path):
    grant = request()
    assert controller.enqueue(grant, root=tmp_path, commit=COMMIT)["provider_allocations_performed"] == 0
    assert controller.enqueue(grant, root=tmp_path, commit=COMMIT)["status"] == "queued"
    changed = contract.sealed({**grant, "authorization_reference": "test://changed"}, "request_digest")
    with pytest.raises(ValueError, match="existing_record_conflict"):
        controller.enqueue(changed, root=tmp_path, commit=COMMIT)
    assert contract.read(tmp_path / "authorizations/development-test.json") == grant
    assert not (tmp_path / "jobs").exists()
    with pytest.raises(ValueError, match="installed_release_mismatch"):
        controller.enqueue(grant, root=tmp_path, commit="2" * 40)


def _dispatch_fixture(tmp_path, monkeypatch):
    from blueprint_pipeline import rigid_robot_free_probe_lane
    grant = request()
    root = tmp_path / "state"
    controller.enqueue(grant, root=root, commit=COMMIT)
    consumption = tmp_path / "consumption"
    consumption.mkdir()
    monkeypatch.setattr(lane, "prepare_consumption_root", lambda: consumption)
    bundle = {"request_digest": grant["request_digest"], "bundle_sha256": "sha256:" + "a" * 64,
              "inputs_digest": "sha256:" + "b" * 64, "bundle_size_bytes": 4_300_000_000, "container_image": "pinned"}
    monkeypatch.setattr(lane, "load_verified_bundle", lambda *a, **k: bundle)
    monkeypatch.setattr(rigid_robot_free_probe_lane, "driver_ceiling_blockers", lambda: [])
    args = SimpleNamespace(provider="vast", embodiment_preparation_request=root / "authorizations/development-test.json",
        embodiment_preparation_bundle=root / "bundle.json", embodiment_preparation_job_dir=tmp_path / "bulk",
        admission_out=tmp_path / "admission.json", adapter_output=tmp_path / "output.json", execute=True)
    return grant, bundle, args, root


def test_allocator_consumes_before_transport_and_prices_all_in_envelope(tmp_path, monkeypatch):
    grant, bundle, args, root = _dispatch_fixture(tmp_path, monkeypatch)
    order = []
    def admit(*a, **k):
        order.append("admit")
        return "development-only-admission"
    monkeypatch.setattr(lane, "require_paid_resource_admission", admit)
    def transport(**kwargs):
        order.append("transport")
        assert contract.read(lane.consumption_path(grant))["bundle_sha256"] == bundle["bundle_sha256"]
        assert kwargs["paid_resource_admission_grant"] == "development-only-admission"
        assert kwargs["expected_provider_download_bytes"] == bundle["bundle_size_bytes"] + 64 * 1024**3
        assert kwargs["expected_provider_upload_bytes"] == 4 * 1024**3
        assert kwargs["hard_cap_usd"] == 4 and kwargs["hard_ttl_seconds"] == 7200
        assert kwargs["stale_offer_create_retry_limit"] == 0
        assert kwargs["candidate_policy_query_expected"] is False
        return {"status": "completed"}
    assert lane.dispatch(args, ([], {"orchestrator_source_commit": COMMIT}), transport=transport, trusted_root=root) == 0
    assert order == ["admit", "transport"]
    assert lane.dispatch(args, ([], {"orchestrator_source_commit": COMMIT}), transport=transport, trusted_root=root) == 2
    assert order == ["admit", "transport"]


def test_admission_or_release_refusal_never_consumes_or_calls_transport(tmp_path, monkeypatch):
    grant, _, args, root = _dispatch_fixture(tmp_path, monkeypatch)
    calls = []
    def blocked(*a, **k):
        raise lane.PaidResourceAdmissionBlocked(["monthly_budget_exhausted"])
    monkeypatch.setattr(lane, "require_paid_resource_admission", blocked)
    assert lane.dispatch(args, ([], {"orchestrator_source_commit": COMMIT}), transport=lambda **k: calls.append(k), trusted_root=root) == 2
    assert not lane.consumption_path(grant).exists() and not calls
    assert lane.dispatch(args, ([], {"orchestrator_source_commit": "2" * 40}), transport=lambda **k: calls.append(k), trusted_root=root) == 2
    assert not lane.consumption_path(grant).exists() and not calls


def test_dry_run_never_consumes_or_requests_paid_admission(tmp_path, monkeypatch):
    grant, _, args, root = _dispatch_fixture(tmp_path, monkeypatch)
    args.execute = False
    monkeypatch.setattr(lane, "require_paid_resource_admission", lambda *a, **k: pytest.fail("paid admission"))
    def transport(**kwargs):
        assert kwargs["execute"] is False and kwargs["authorization_consumption"] is None
        return {"status": "dry_run_ready"}
    assert lane.dispatch(args, ([], {"orchestrator_source_commit": COMMIT}), transport=transport, trusted_root=root) == 0
    assert not lane.consumption_path(grant).exists()


def _bundle_fixture(tmp_path, monkeypatch):
    from tests.test_deployment_embodiment_asset_import import _source
    source = _source(tmp_path, monkeypatch)
    monkeypatch.setattr(contract, "runtime_profiles", lambda: {"profiles": [{"embodiment_id": identity} for identity in IDS]})
    monkeypatch.setattr(contract, "asset_recipes", lambda: {"recipes": [{"embodiment_id": identity, "status": "source_available"} for identity in IDS]})
    packet = tmp_path / "native.zip"
    packet.write_bytes(b"development-only runtime packet")
    receipt = tmp_path / "native.json"
    receipt.write_text("{}")
    monkeypatch.setattr(bundle_module, "verify_native_task_runtime_source_packet", lambda *a: {
        "verified_packet_path": str(packet), "receipt_digest": "sha256:" + "c" * 64, "packet_size_bytes": packet.stat().st_size,
        "packet_sha256": contract.sha(packet)})
    monkeypatch.setattr(bundle_module, "WHEELS", (("fixture.whl", hashlib.sha256(b"wheel").hexdigest(), "https://fixture.invalid"),))
    def materialize(identity, output):
        if identity == IDS[1]:
            raise OSError("development-only public source unavailable")
        import shutil
        shutil.copytree(source, output, dirs_exist_ok=True)
        return contract.read(output / "source-bundle.json")
    return dict(request=request(), runtime_source_receipt=receipt, source_root=tmp_path / "source-cache",
                output=tmp_path / "bundle", materialize=materialize, wheel_get=lambda url: b"wheel")


def test_bundle_keeps_other_sources_and_can_retry_interrupted_publication(tmp_path, monkeypatch):
    params = _bundle_fixture(tmp_path, monkeypatch)
    publish = bundle_module.publish_staged_immutable_directory
    def interrupted(**kwargs):
        assert (kwargs["staging"] / bundle_module.RECEIPT).is_file()
        raise OSError("development-only crash before publication")
    monkeypatch.setattr(bundle_module, "publish_staged_immutable_directory", interrupted)
    with pytest.raises(OSError, match="crash before"):
        bundle_module.build_bundle(**params)
    assert not params["output"].exists()
    monkeypatch.setattr(bundle_module, "publish_staged_immutable_directory", publish)
    built = bundle_module.build_bundle(**params)
    assert [row["status"] for row in built["inputs"]["sources"]] == ["source_ready", "source_unavailable", "source_ready"]
    assert bundle_module.load_verified_bundle(params["output"] / bundle_module.RECEIPT, expected_implementation_commit=COMMIT) == built
    assert bundle_module.build_bundle(**params) == built
    with pytest.raises(ValueError, match="bundle_invalid"):
        bundle_module.load_verified_bundle(params["output"] / bundle_module.RECEIPT, expected_implementation_commit="2" * 40)


def test_bundle_streams_the_verified_runtime_packet_without_a_staged_copy(tmp_path, monkeypatch):
    import zipfile
    params = _bundle_fixture(tmp_path, monkeypatch)
    packet = tmp_path / "native.zip"
    copied = []
    copyfile = bundle_module.shutil.copyfile
    monkeypatch.setattr(bundle_module.shutil, "copyfile", lambda src, dst, *a, **k: (copied.append(Path(src)), copyfile(src, dst, *a, **k))[1])
    built = bundle_module.build_bundle(**params)
    assert packet not in copied
    name = "provider_runtime/native_task_runtime_sources/native_task_runtime_sources.zip"
    assert {"path": name, "size_bytes": packet.stat().st_size, "sha256": contract.sha(packet)} in built["files"]
    assert [row["path"] for row in built["files"]] == sorted((row["path"] for row in built["files"]), key=PurePosixPath)
    with zipfile.ZipFile(params["output"] / "embodiment_asset_preparation.zip") as archive:
        assert archive.read(name) == packet.read_bytes()
        assert archive.getinfo(name).compress_type == zipfile.ZIP_STORED
        bundle_module.validate_archive(archive)


def test_bundle_refuses_a_runtime_packet_that_changed_after_verification(tmp_path, monkeypatch):
    params = _bundle_fixture(tmp_path, monkeypatch)
    packet = tmp_path / "native.zip"
    verified = bundle_module.verify_native_task_runtime_source_packet()
    monkeypatch.setattr(bundle_module, "verify_native_task_runtime_source_packet", lambda *a: {
        **verified, "packet_sha256": "sha256:" + hashlib.sha256(b"the bytes that were verified").hexdigest()})
    with pytest.raises(ValueError, match="runtime_source_packet_changed"):
        bundle_module.build_bundle(**params)
    assert not params["output"].exists() and packet.is_file()


def test_no_available_source_has_individual_reasons_and_no_bundle(tmp_path, monkeypatch):
    params = _bundle_fixture(tmp_path, monkeypatch)
    def unavailable(*a):
        raise OSError("development-only offline")
    params["materialize"] = unavailable
    with pytest.raises(ValueError, match="no_source_ready"):
        bundle_module.build_bundle(**params)
    summary = contract.read(next(tmp_path.glob("source-preparation.*.json")))
    assert [row["embodiment_id"] for row in summary["results"]] == IDS
    assert all(row["status"] == "source_unavailable" for row in summary["results"])
    assert not params["output"].exists()


def test_bundle_cli_builds_verifiable_artifact_and_can_reuse_it(tmp_path, monkeypatch, capsys):
    import json

    params = _bundle_fixture(tmp_path, monkeypatch)
    request_path = tmp_path / "request.json"
    contract.write_once(request_path, params["request"])
    build = bundle_module.build_bundle
    monkeypatch.setattr(bundle_module, "build_bundle", lambda **kwargs: build(
        **kwargs, materialize=params["materialize"], wheel_get=params["wheel_get"]))
    argv = ["--request", str(request_path)]
    for key in ("runtime_source_receipt", "source_root", "output"):
        argv.extend(["--" + key.replace("_", "-"), str(params[key])])
    for _ in range(2):
        assert bundle_module.main(argv) == 0
        summary = json.loads(capsys.readouterr().out)
        verified = bundle_module.load_verified_bundle(params["output"] / bundle_module.RECEIPT,
                                                       expected_implementation_commit=COMMIT)
        assert summary["bundle_sha256"] == verified["bundle_sha256"]
        assert summary["request_digest"] == params["request"]["request_digest"]
        assert [row["status"] for row in verified["inputs"]["sources"]] == [
            "source_ready", "source_unavailable", "source_ready"]


def _queued(tmp_path):
    grant = request()
    controller.enqueue(grant, root=tmp_path, commit=COMMIT)
    job = tmp_path / "jobs" / grant["authorization_id"]
    job.mkdir(parents=True)
    return grant, job, tmp_path / "pending/development-test.json"


def test_interrupted_allocation_is_never_relaunched_and_accounts_for_every_body(tmp_path, monkeypatch):
    from blueprint_pipeline import rigid_robot_free_probe_runner
    grant, job, item = _queued(tmp_path)
    contract.write_once(job / "allocator_launch_intent.json", {"bundle_sha256": "sha256:bound"})
    monkeypatch.setattr(rigid_robot_free_probe_runner, "_allocator_alive", lambda *a: False)
    value = controller.process_entry(item, root=tmp_path, store=tmp_path / "models", commit=COMMIT,
        runner=lambda *a, **k: pytest.fail("relaunch"), build=lambda **k: pytest.fail("rebuild"))
    assert value["status"] == "reconciliation_required"
    assert [row["embodiment_id"] for row in value["results"]] == grant["embodiment_ids"]
    assert value["provider_relaunch_performed"] is False


def test_failed_admission_retains_individual_reasons(tmp_path):
    grant, job, item = _queued(tmp_path)
    contract.write_once(job / "allocator_result.json", {"status": "blocked", "blockers": ["price_exceeds_budget"],
        "continuing_spend_from_this_run": False})
    value = controller.process_entry(item, root=tmp_path, store=tmp_path / "models", commit=COMMIT)
    assert [row["embodiment_id"] for row in value["results"]] == grant["embodiment_ids"]
    assert all(row["details"] == ["price_exceeds_budget"] for row in value["results"])


def test_completed_transport_is_recovered_without_relaunch(tmp_path, monkeypatch):
    _, job, item = _queued(tmp_path)
    contract.write_once(job / "allocator_launch_intent.json", {"bundle_sha256": "sha256:bound"})
    recovered = {"status": "completed", "bundle_sha256": "sha256:bound", "continuing_spend_from_this_run": False}
    contract.write_once(controller.workspace_for(job) / "allocator/adp_arena_vast_result.json", recovered)
    monkeypatch.setattr(controller, "load_verified_bundle", lambda *a, **k: {"bundle_sha256": "sha256:bound"})
    monkeypatch.setattr(controller, "promote", lambda **k: {"status": "completed", "development_only": True})
    assert controller.process_entry(item, root=tmp_path, store=tmp_path / "models", commit=COMMIT,
        runner=lambda *a, **k: pytest.fail("relaunch"))["status"] == "completed"
    assert contract.read(job / "allocator_result.json") == recovered


def test_promotion_isolates_malformed_body_between_two_valid_bodies(tmp_path, monkeypatch):
    from blueprint_pipeline import deployment_embodiment_runtime_configuration as runtime
    from blueprint_pipeline import deployment_embodiment_asset_import as importer
    grant, job, _ = _queued(tmp_path)
    output = controller.workspace_for(job) / "allocator/output"
    rows = []
    model_sha = "sha256:" + "d" * 64
    binding = {"robot_embodiment": {"model": {"url": "https://robot-models.blueprint.internal/fixture/robot.usda"}}}
    for identity in IDS:
        body = output / "bodies" / identity
        contract.write_once(body / "binding-inputs.json", binding if identity != IDS[1] else {"robot_embodiment": []})
        contract.write_once(body / "body-result.json", {"development_only": True})
        (body / "configuration").mkdir()
        (body / "prepared").mkdir()
        (body / "prepared/robot.usda").write_bytes(b"development-only model bytes")
        rows.append({"embodiment_id": identity, "status": "imported", "configuration_digest": "configuration",
                     "import_digest": "import", "model_sha256": model_sha})
    inputs_digest = "sha256:" + "e" * 64
    result = contract.sealed({"schema_version": contract.RESULT_SCHEMA, "status": "completed",
        "claim_ceiling": "robot_asset_import_only", "candidate_policy_queried": False,
        "task_executed": False, "provider_teardown_verified": False, "inputs_digest": inputs_digest, "results": rows,
        "files": [{"path": p.relative_to(output).as_posix(), "size_bytes": p.stat().st_size, "sha256": contract.sha(p)}
                  for p in sorted(output.rglob("*")) if p.is_file()]}, "result_digest")
    contract.write_once(output / contract.RESULT_FILENAME, result)
    monkeypatch.setattr(runtime, "verify_runtime_configuration", lambda p: {"source_bundle_digest": "source", "configuration_digest": "configuration"})
    monkeypatch.setattr(importer, "verify_prepared_asset_directory", lambda p: {"import_digest": "import", "model": {"sha256": model_sha}})
    monkeypatch.setattr(runtime, "bind_prepared_configuration", lambda *a: deepcopy(binding))
    bundle = {"bundle_sha256": "sha256:bound", "inputs_digest": inputs_digest,
              "inputs": {"sources": [{"embodiment_id": identity, "bundle_digest": "source"} for identity in IDS]}}
    wrapper = {"status": "completed", "bundle_sha256": "sha256:bound", "continuing_spend_from_this_run": False,
               "native_control_result_path": str(output / contract.RESULT_FILENAME), "native_control_result_digest": result["result_digest"]}
    value = controller.promote(request=grant, bundle=bundle, allocator_result=wrapper, job=job, store=tmp_path / "models")
    assert [row["status"] for row in value["results"]] == ["retained", "import_verification_failed", "retained"]
    assert not (tmp_path / "models" / IDS[1]).exists()
    assert value["candidate_policy_queried"] is False and value["task_executed"] is False
    for row in (value["results"][0], value["results"][2]):
        assert Path(row["directory"]).is_dir()
        assert row["public_download_published"] is False


def test_queue_preparation_failure_still_accounts_for_every_id(tmp_path, monkeypatch):
    grant, _, _ = _queued(tmp_path)
    monkeypatch.setattr(controller, "source_commit", lambda: COMMIT)
    def fail(*a, **k):
        raise ValueError("source packet unavailable")
    monkeypatch.setattr(controller, "process_reserved_entry", fail)
    value = controller.run_queue(root=tmp_path, store=tmp_path / "models")
    assert [row["embodiment_id"] for row in value["results"][0]["results"]] == grant["embodiment_ids"]
    assert all(row["reason"] == "source packet unavailable" for row in value["results"][0]["results"])


def test_fresh_install_provisions_all_fixed_preparation_paths(tmp_path):
    import os
    import importlib.util
    spec = importlib.util.spec_from_file_location("deploy_preparation_fixture", Path(__file__).parents[1] / "scripts/deploy_control_plane_commit.py")
    deploy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(deploy)
    release = Path(__file__).parents[1]
    host = tmp_path / "host"
    (host / "mnt/blueprint-work").mkdir(parents=True)
    (host / "var/lib/blueprint").mkdir(parents=True)
    value = deploy._install_unit_sandbox_paths(release_path=release, root_prefix=host,
        units=["blueprint-embodiment-preparation.service"], owner_ids=(os.getuid(), os.getgid()))
    assert {row["path"] for row in value["created"]} == {
        str(contract.WORK_ROOT), str(contract.STATE_ROOT), str(contract.MODEL_ROOT)}
    from blueprint_pipeline.control_plane_storage_roots import classify_path
    for root in (contract.WORK_ROOT, contract.STATE_ROOT, contract.MODEL_ROOT):
        assert classify_path(str(root)).storage_class == "evidence_hot"
        assert (host / root.relative_to("/")).stat().st_uid == os.getuid()


def test_disk_reservation_accounts_for_actual_runtime_packet_before_build(tmp_path, monkeypatch):
    from contextlib import contextmanager
    from blueprint_pipeline import control_plane_disk_budget, native_task_runtime_source_packet
    grant, job, item = _queued(tmp_path)
    calls = []
    @contextmanager
    def reserve(role, **kwargs):
        calls.append((role, kwargs))
        yield SimpleNamespace(receipt=lambda: {"development_only": True})
    monkeypatch.setattr(control_plane_disk_budget, "reserve_control_plane_disk", reserve)
    monkeypatch.setattr(controller, "runtime_source_receipt", lambda: tmp_path / "runtime.json")
    packet_size = 4_287_162_924
    monkeypatch.setattr(native_task_runtime_source_packet, "verify_native_task_runtime_source_packet", lambda *a: {"packet_size_bytes": packet_size})
    def process(*a, **k):
        assert len(calls) == 2
        return {"status": "development_only"}
    monkeypatch.setattr(controller, "process_entry", process)
    controller.process_reserved_entry(item, root=tmp_path, store=tmp_path / "models", commit=COMMIT)
    assert calls[0][1]["expected_bytes"] == packet_size + 4 * 1024**3
    assert calls[1][1]["expected_bytes"] == contract.MAX_OUTPUT_BYTES
    evidence = contract.read(job / "disk-admission.json")
    assert evidence["authorization_id"] == grant["authorization_id"]
    assert evidence["workspace_path"] == str(controller.workspace_for(job))


def test_allocator_inherits_only_this_jobs_workspace_hold_as_enclosing(tmp_path, monkeypatch):
    from contextlib import contextmanager
    from blueprint_pipeline import control_plane_disk_budget, native_task_runtime_source_packet
    _, _job, item = _queued(tmp_path)
    tokens = iter(["a" * 32, "b" * 32])
    @contextmanager
    def reserve(role, **kwargs):
        token = next(tokens)
        yield SimpleNamespace(receipt=lambda: {"reservation_token": token})
    monkeypatch.setattr(control_plane_disk_budget, "reserve_control_plane_disk", reserve)
    monkeypatch.setattr(controller, "runtime_source_receipt", lambda: tmp_path / "runtime.json")
    monkeypatch.setattr(native_task_runtime_source_packet, "verify_native_task_runtime_source_packet",
                        lambda *a: {"packet_size_bytes": 1})
    monkeypatch.setenv(control_plane_disk_budget.ENCLOSING_RESERVATIONS_ENV, "c" * 32)
    seen = []
    def process(*a, **k):
        seen.append(os.environ[control_plane_disk_budget.ENCLOSING_RESERVATIONS_ENV])
        raise RuntimeError("development_only_stop")
    monkeypatch.setattr(controller, "process_entry", process)
    with pytest.raises(RuntimeError, match="development_only_stop"):
        controller.process_reserved_entry(item, root=tmp_path, store=tmp_path / "models", commit=COMMIT)
    # The workspace hold (first) is enclosing; the model-store hold is not.
    assert seen == ["c" * 32 + "," + "a" * 32]
    assert os.environ[control_plane_disk_budget.ENCLOSING_RESERVATIONS_ENV] == "c" * 32


def test_sealed_completed_recovery_does_not_need_live_registry_runtime_packet(tmp_path, monkeypatch):
    from contextlib import contextmanager
    from blueprint_pipeline import control_plane_disk_budget
    _, job, item = _queued(tmp_path)
    contract.write_once(job / "allocator_launch_intent.json", {"bundle_sha256": "sha256:bound"})
    contract.write_once(controller.workspace_for(job) / "allocator/adp_arena_vast_result.json", {
        "status": "completed", "bundle_sha256": "sha256:bound", "continuing_spend_from_this_run": False})
    @contextmanager
    def reserve(*a, **k):
        assert k["expected_bytes"] == contract.MAX_OUTPUT_BYTES
        yield SimpleNamespace(receipt=lambda: {"development_only": True})
    monkeypatch.setattr(control_plane_disk_budget, "reserve_control_plane_disk", reserve)
    monkeypatch.setattr(controller, "runtime_source_receipt", lambda: pytest.fail("live runtime dependency on recovery"))
    monkeypatch.setattr(controller, "load_verified_bundle", lambda *a, **k: {"bundle_sha256": "sha256:bound"})
    monkeypatch.setattr(controller, "promote", lambda **k: {"status": "completed", "development_only": True})
    assert controller.process_reserved_entry(item, root=tmp_path, store=tmp_path / "models", commit=COMMIT)["status"] == "completed"


def _diagnostic_publication(tmp_path, monkeypatch):
    """Fake native evidence tests publication integrity, never simulator behavior."""
    from blueprint_pipeline import deployment_embodiment_runtime_configuration as runtime
    from blueprint_pipeline import deployment_embodiment_asset_import as importer
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest
    identity, model_sha = IDS[0], "sha256:" + "d" * 64
    provenance = {"import_digest": "import", "source_bundle_digest": "source", "model": {"sha256": model_sha}}
    binding = {"configuration_digest": "configuration", "robot_embodiment": {
        "embodiment_digest": "embodiment", "model": {"url": "https://robot-models.blueprint.internal/fixture/robot.usda"}},
        "model_record": {"sha256": model_sha, "source_import_provenance": provenance}}
    monkeypatch.setattr(runtime, "verify_runtime_configuration", lambda p: {
        "source_bundle_digest": "source", "configuration_digest": "configuration"})
    monkeypatch.setattr(importer, "verify_prepared_asset_directory", lambda p: deepcopy(provenance))
    monkeypatch.setattr(runtime, "bind_prepared_configuration", lambda *a: deepcopy(binding))
    store = tmp_path / "models"

    def publish(authorization, *, changed=None, elapsed=1.):
        grant = contract.sealed({**request([identity]), "authorization_id": authorization}, "request_digest")
        job = tmp_path / "state/jobs" / authorization
        output = controller.workspace_for(job) / "allocator/output"
        body = output / "bodies" / identity
        contract.write_once(body / "binding-inputs.json", binding)
        (body / "configuration").mkdir()
        (body / "prepared").mkdir()
        (body / "prepared/robot.usda").write_bytes(b"development-only model bytes")
        native = {"schema_version": "embodiment_native_controller_smoke.v1", "status": "passed", "reasons": [],
            "claim_ceiling": "native_controller_diagnostic_only", "candidate_policy_queried": False,
            "task_executed": False, "physical_success_proven": False, "binding_digest": canonical_digest(binding),
            "embodiment_id": identity, "embodiment_digest": "embodiment", "model_sha256": model_sha,
            "source_bundle_digest": "source", "import_digest": "import", "configuration_digest": "configuration",
            "elapsed_wall_seconds": elapsed, "development_only": True}
        if changed == "model_binding":
            native["model_sha256"] = "sha256:" + "0" * 64
        native = contract.sealed(native, "smoke_digest")
        process = contract.sealed({"schema_version": "embodiment_native_controller_process.v1", "status": "passed",
            "candidate_policy_queried": False, "task_executed": False, "physical_success_proven": False,
            "reasons": [], "smoke_digest": native["smoke_digest"]}, "process_digest")
        reported = {key: process[key] for key in ("status", "process_digest", "smoke_digest", "reasons")}
        if changed == "summary":
            reported["process_digest"] = "sha256:" + "0" * 64
        elif changed == "native_digest":
            native["elapsed_wall_seconds"] += 1
        elif changed == "process_digest":
            process["reasons"] = ["changed"]
        elif changed == "summary_type":
            reported = ["invalid"]
        if changed != "missing_receipts":
            contract.write_once(body / "embodiment_native_controller_smoke.v1.json", native)
            contract.write_once(body / "native-controller-process.v1.json", process)
        row = {"embodiment_id": identity, "status": "imported", "configuration_digest": "configuration",
               "import_digest": "import", "model_sha256": model_sha, "native_controller_smoke": reported}
        contract.write_once(body / "body-result.json", row)
        inputs_digest = "sha256:" + "e" * 64
        result = contract.sealed({"schema_version": contract.RESULT_SCHEMA, "status": "completed",
            "claim_ceiling": "robot_asset_import_only", "candidate_policy_queried": False,
            "task_executed": False, "provider_teardown_verified": False, "inputs_digest": inputs_digest, "results": [row],
            "files": [{"path": p.relative_to(output).as_posix(), "size_bytes": p.stat().st_size, "sha256": contract.sha(p)}
                      for p in sorted(output.rglob("*")) if p.is_file()]}, "result_digest")
        contract.write_once(output / contract.RESULT_FILENAME, result)
        bundle = {"bundle_sha256": "sha256:bound", "inputs_digest": inputs_digest,
                  "inputs": {"sources": [{"embodiment_id": identity, "bundle_digest": "source"}]}}
        wrapper = {"status": "completed", "bundle_sha256": "sha256:bound", "continuing_spend_from_this_run": False,
                   "native_control_result_path": str(output / contract.RESULT_FILENAME),
                   "native_control_result_digest": result["result_digest"]}
        kwargs = dict(request=grant, bundle=bundle, allocator_result=wrapper, job=job, store=store)
        first = controller.promote(**kwargs)
        assert controller.promote(**kwargs) == first  # Recovery reuses exactly the retained diagnostics.
        return first["results"][0], native
    return publish, store


@pytest.mark.parametrize("changed", ["summary", "model_binding", "native_digest", "process_digest",
                                    "summary_type", "missing_receipts"])
def test_diagnostic_tampering_cannot_claim_success_or_discard_verified_model(tmp_path, monkeypatch, changed):
    publish, store = _diagnostic_publication(tmp_path, monkeypatch)
    row, _ = publish("tampered-diagnostic", changed=changed)
    assert row["status"] == "retained"
    assert Path(row["directory"], "robot.usda").is_file()
    diagnostic = row["native_controller_smoke"]
    assert diagnostic["status"] == "failed" and diagnostic["verification_status"] == "unverified"
    assert diagnostic["smoke_digest"] is None and diagnostic["process_digest"] is None
    assert diagnostic["reasons"]
    assert Path(diagnostic["diagnostic_path"]).is_relative_to(store / "batches/tampered-diagnostic" / IDS[0])


def test_repeated_model_publication_keeps_each_batches_actual_diagnostic(tmp_path, monkeypatch):
    publish, store = _diagnostic_publication(tmp_path, monkeypatch)
    first, first_native = publish("batch-one", elapsed=1.)
    second, second_native = publish("batch-two", elapsed=2.)
    assert first["directory"] == second["directory"]
    assert len(list((store / IDS[0]).iterdir())) == 1  # No repeated model closure.
    for row, native, authorization in ((first, first_native, "batch-one"), (second, second_native, "batch-two")):
        diagnostic = row["native_controller_smoke"]
        assert diagnostic["status"] == "passed" and diagnostic["verification_status"] == "verified"
        assert diagnostic["smoke_digest"] == native["smoke_digest"]
        path = Path(diagnostic["receipt_paths"]["embodiment_native_controller_smoke.v1.json"])
        assert path.is_relative_to(store / "batches" / authorization / IDS[0])
        assert contract.read(path) == native
        assert not (path.parent / "prepared").exists()
    assert first["native_controller_smoke"]["smoke_digest"] != second["native_controller_smoke"]["smoke_digest"]


def test_diagnostic_retention_failure_still_publishes_verified_model(tmp_path, monkeypatch):
    publish, _ = _diagnostic_publication(tmp_path, monkeypatch)
    def unavailable(**_kwargs):
        raise OSError("development-only diagnostic write failed")
    monkeypatch.setattr(controller, "retain_batch_diagnostic", unavailable)
    row, _ = publish("diagnostic-storage-failed")
    assert row["status"] == "retained" and Path(row["directory"], "robot.usda").is_file()
    assert row["native_controller_smoke"]["status"] == "failed"
    assert row["native_controller_smoke"]["reasons"][0].startswith("diagnostic_retention_failed:")
    assert "diagnostic_path" not in row["native_controller_smoke"]


def _uploaded(job, digest="sha256:" + "b" * 64):
    archive = controller.workspace_for(job) / "bundle" / controller.BUNDLE_ARCHIVE
    contract.write_once(archive.parent / controller.RECEIPT, {"bundle_sha256": digest})
    archive.write_bytes(b"development-only upload bytes")
    archive.parent.chmod(0o555)  # as publish_staged_immutable_directory leaves it
    return archive


@pytest.mark.parametrize("state,released", [
    ("published", True), ("not_completed", True), ("completed_unpromoted", False), ("running", False), ("built", False)])
def test_a_terminal_batch_releases_its_upload_archive_and_keeps_its_evidence(tmp_path, state, released):
    grant, job, _ = _queued(tmp_path)
    archive = _uploaded(job)
    if state == "published":
        contract.write_once(job / "publication.json", {"status": "completed"})
    elif state in {"not_completed", "completed_unpromoted"}:
        contract.write_once(job / "allocator_result.json", {
            "status": "blocked" if state == "not_completed" else "completed", "continuing_spend_from_this_run": False})
    elif state == "running":
        contract.write_once(job / "allocator_launch_intent.json", {"bundle_sha256": "sha256:" + "b" * 64})
    rows = controller.release_terminal_workspaces(tmp_path, commit=COMMIT)
    # Promotion and reconciliation still read the archive; a published or ended batch never does.
    assert archive.exists() is not released
    assert (archive.parent / controller.RECEIPT).is_file()
    assert archive.parent.stat().st_mode & 0o777 == 0o555
    if released:
        record = contract.read(job / controller.WORKSPACE_RELEASE)
        assert rows == [record] and record["bundle_sha256"] == "sha256:" + "b" * 64
        assert record["released_bytes"] == len(b"development-only upload bytes")
        assert controller.release_terminal_workspaces(tmp_path, commit=COMMIT) == []
    else:
        assert rows == [] and not (job / controller.WORKSPACE_RELEASE).exists()


def test_only_the_installed_releases_source_cache_is_kept(tmp_path):
    sources = tmp_path / "work" / "sources"
    for commit in (COMMIT, "2" * 40):
        cache = sources / commit / "universal_robots_ur5e"
        cache.mkdir(parents=True)
        (cache / "robot.urdf").write_text("development-only")
        cache.chmod(0o500)  # retained sources keep read-only modes
    rows = controller.release_terminal_workspaces(tmp_path, commit=COMMIT)
    assert rows == [{"source_cache_release": "2" * 40}]
    assert sorted(p.name for p in sources.iterdir()) == [COMMIT]
    assert (sources / COMMIT / "universal_robots_ur5e" / "robot.urdf").is_file()


def test_the_queue_pass_releases_terminal_workspaces_and_reports_them(tmp_path, monkeypatch):
    _, job, _ = _queued(tmp_path)
    archive = _uploaded(job)
    contract.write_once(job / "publication.json", {"status": "completed", "results": []})
    monkeypatch.setattr(controller, "source_commit", lambda: COMMIT)
    monkeypatch.setattr(controller, "process_reserved_entry", lambda item, **k: contract.read(job / "publication.json"))
    value = controller.run_queue(root=tmp_path, store=tmp_path / "models")
    assert not archive.exists() and [row["authorization_id"] for row in value["workspace_releases"]] == [job.name]


def test_a_release_that_cannot_unlink_is_reported_and_the_queue_still_runs(tmp_path, monkeypatch):
    _, job, _ = _queued(tmp_path)
    _uploaded(job)
    contract.write_once(job / "publication.json", {"status": "completed"})

    def refused(self, *args, **kwargs):
        raise PermissionError(13, "read-only")

    monkeypatch.setattr(controller.Path, "unlink", refused)
    assert controller.release_terminal_workspaces(tmp_path, commit=COMMIT) == [
        {"authorization_id": job.name, "release_blocked": "PermissionError"}]
