"""The transported selection reaches a worker only with its required binding."""

from __future__ import annotations

import json
import zipfile

from blueprint_pipeline import native_g1_team_provider_runtime as runtime
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from tests.test_native_g1_team_provider_bundle import _inputs, COMMIT


def _runtime(tmp_path, monkeypatch, *, endpoint=False):
    args, _ = _inputs(tmp_path, monkeypatch, endpoint=endpoint)
    from blueprint_pipeline import native_g1_team_provider_bundle as builder

    receipt = builder.build_g1_team_provider_bundle(**args)
    root = tmp_path / "provider"
    root.mkdir()
    with zipfile.ZipFile(receipt["bundle_path"]) as archive:
        archive.extractall(root)
    output = root / "runtime_output"
    output.mkdir()
    provision = {"status": "completed", "runtime_profile": "unitree_g1",
                 "source_packet_sha256": receipt["runtime_source_packet"]["packet_sha256"]}
    provision["receipt_digest"] = canonical_digest(provision, digest_field="receipt_digest")
    (output / "native_task_runtime_source_provisioning.v1.json").write_text(json.dumps(provision))
    monkeypatch.setattr(runtime, "verify_g1_publisher_source", builder.verify_g1_publisher_source)
    return root / "provider_runtime", output, receipt


def test_container_selection_requires_separate_runtime_and_never_calls_local_docker(tmp_path, monkeypatch):
    root, output, _ = _runtime(tmp_path, monkeypatch)
    def must_not_run(**kwargs):
        raise AssertionError("container mode bypassed paired runtime admission")
    monkeypatch.setattr(runtime, "run_supervised_g1_team_worker", must_not_run)
    result = runtime.run_g1_team_provider_runtime(runtime_root=root, output_dir=output)
    assert result["status"] == "blocked"
    assert result["blocker_code"] == "g1_team_provider_paired_policy_runtime_required"
    assert result["supervised_result_digest"] is None


def test_endpoint_requires_private_credential_then_reaches_selected_worker(tmp_path, monkeypatch):
    root, output, receipt = _runtime(tmp_path, monkeypatch, endpoint=True)
    credential = tmp_path / "credential"
    credential.write_text("private-test-value")
    credential.chmod(0o600)
    called = []
    def worker(**kwargs):
        called.append(kwargs)
        arguments = kwargs["worker_arguments"]
        assert arguments["expected_implementation_commit"] == COMMIT
        assert arguments["credential_file_path"] == credential
        assert arguments["scene_packet_root"] == root / "inputs/scene_packet"
        return {"status": "blocked", "result_digest": "sha256:" + "e" * 64}
    monkeypatch.setattr(runtime, "run_supervised_g1_team_worker", worker)
    result = runtime.run_g1_team_provider_runtime(
        runtime_root=root, output_dir=output, credential_file_path=credential,
    )
    assert len(called) == 1
    assert result["stage_reached"] == "supervised_worker"
    assert result["execution_packet_digest"] == receipt["execution_packet_digest"]
    assert result["status"] == "blocked"
    assert result["candidate_policy_queried"] is None
    assert "private-test-value" not in json.dumps(result)


def test_missing_credential_and_changed_scene_block_before_endpoint_contact(tmp_path, monkeypatch):
    root, output, _ = _runtime(tmp_path, monkeypatch, endpoint=True)
    def must_not_run(**kwargs):
        raise AssertionError("unbound endpoint received site observations")
    monkeypatch.setattr(runtime, "run_supervised_g1_team_worker", must_not_run)
    result = runtime.run_g1_team_provider_runtime(runtime_root=root, output_dir=output)
    assert result["status"] == "blocked"
    assert result["stage_reached"] == "policy_runtime_binding"
    other = tmp_path / "altered-output"
    plan = root / "inputs/scene_packet/native_task_arena_scene_plan.v1.json"
    plan.write_text("{}")
    result = runtime.run_g1_team_provider_runtime(runtime_root=root, output_dir=other)
    assert result["status"] == "blocked"
    assert result["stage_reached"] == "input_verification"
