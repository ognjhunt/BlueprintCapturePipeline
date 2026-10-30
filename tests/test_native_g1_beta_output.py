"""Provider reference envelopes hydrate only exact current-cell native bytes."""
from copy import deepcopy
import json
import zipfile

import pytest

from blueprint_pipeline import native_g1_beta_output as output
from blueprint_pipeline import native_g1_beta_runtime as runtime
from blueprint_pipeline import native_g1_beta_packet as packet
from blueprint_pipeline import native_g1_beta_session as session
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from tests.test_native_g1_beta_session import inputs
from tests.test_beta_policy_screening_g1 import session_result, DP


def seal(value, key):
    value[key] = canonical_digest(value, digest_field=key)
    return value


def fixture(tmp_path, monkeypatch, *, completed=False):
    plan, candidate, spec = inputs(tmp_path, monkeypatch)
    authority = session.build_session_authority(plan=plan, candidate=candidate, spec=spec)
    common = {"scene_id": spec["site_id"], "task_id": spec["source"]["task_id"]}
    task = {"task_success_contract_digest": spec["task_success_contract_digest"]}
    samples = []
    if completed:
        from tests.test_adp_task_scoring import _rigid_v2_spec, _rigid_v2_sample
        from blueprint_pipeline.adp_task_scoring import seal_rigid_task_success_contract
        task = _rigid_v2_spec()
        task.update(site_id=common["scene_id"], task_id=common["task_id"])
        task["task_success_contract"] = seal_rigid_task_success_contract(task_spec=task,
            site_id=common["scene_id"], task_id=common["task_id"], author_source="compatibility_default",
            author_id="hermetic-scoring-fixture", confirmation_status="confirmed")
        task["task_success_contract_digest"] = task["task_success_contract"]["contract_digest"]
        authority["task_success_contract_digest"] = task["task_success_contract_digest"]
        authority["rights_authority"]["task_success_contract_digest"] = task["task_success_contract_digest"]
        samples = [_rigid_v2_sample(i, [1.0, 2.0, .8]) for i in range(6)]
    request = seal({**common, "scenario": {"seed": 1}}, "request_digest")
    contract = seal({**common, "scenario": {"seed": 1}, "task_spec": task}, "contract_digest")
    scene = seal({**common, "scenario": {"seed": 1}, "task_spec": contract["task_spec"], "task_kind": "rigid_pick_place",
        "robot": {"robot_id": "unitree_g1"}, "runtime_contract_digest": contract["contract_digest"]}, "plan_digest")
    receipt = seal({"schema_version": "native_task_arena_packet_receipt.v1", "artifacts": [],
        "arena_scene_plan_digest": scene["plan_digest"]}, "receipt_digest")
    authority["rights_authority"]["scene_plan_digest"] = scene["plan_digest"]
    seal(authority["rights_authority"], "authority_digest")
    seal(authority, "authority_digest")
    session.validate_session_authority(authority)
    archive_path = tmp_path / "sealed.zip"
    manifest = {"authority_digest": authority["authority_digest"]}
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(session.MANIFEST, json.dumps(manifest))
        archive.writestr("provider_runtime/inputs/session_authority.json", json.dumps(authority))
        for name, value in ((packet.REQUEST, request), (packet.CONTRACT, contract), (packet.PLAN, scene), (packet.RECEIPT, receipt)):
            archive.writestr("provider_runtime/inputs/scene_packet/" + name, json.dumps(value))
    # Archive closure is tested against the real Vast validator separately.
    # Here only that already-verified boundary is replaced, not scene derivation,
    # retained artifact hashes, checkpoint load evidence, or outcome projection.
    monkeypatch.setattr(output, "validate_bundle_archive", lambda archive, value: value)
    bundle = {**manifest, "bundle_path": str(archive_path)}
    root = tmp_path / "returned"
    root.mkdir()
    cell = authority["cells"][0]
    derived = packet.derive_documents(base_request=request, base_contract=contract, base_scene=scene, cell=cell)
    query = deepcopy(session_result()[2]["episodes"][0]["query_checkpoints"][0])
    query["scene_plan_digest"] = derived["scene"]["plan_digest"]
    seal(query, "checkpoint_digest")
    cell_root = root / "cells" / cell["cell_id"]
    runtime._write(cell_root / "episode/query_checkpoints/query_0000.v1.json", query)
    if completed:
        from blueprint_pipeline.native_g1_beta_authority import derive_cell_authority
        from blueprint_pipeline.adp_task_scoring import score_task_episode_from_spec
        source_row = deepcopy(session_result()[2]["episodes"][0])
        trace, episode, worker = (source_row[key] for key in ("trace", "episode", "worker_result"))
        supervised = worker["supervised_episode"]
        lease = supervised["server_lease_receipt"]
        for artifact in (trace, episode, worker, supervised, lease):
            artifact["scene_plan_digest"] = derived["scene"]["plan_digest"]
        seal(lease, "receipt_digest")
        trace.update(seed=cell["seed"], initial_task_sample=samples[0],
            steps=[{"step_index": i, "task_sample": sample} for i, sample in enumerate(samples[1:], 1)],
            scene_step_count=len(samples) - 1, queries=[{**query["query"], "checkpoint_digest": query["checkpoint_digest"]}])
        seal(trace, "trace_digest")
        episode.update(trace_digest=trace["trace_digest"], score=score_task_episode_from_spec(task_spec=task, samples=samples))
        seal(episode, "result_digest")
        supervised["episode_result_digest"] = episode["result_digest"]
        seal(supervised, "result_digest")
        cell_receipt = packet.expected_cell_packet_receipt(receipt, derived, cell)
        cell_authority = derive_cell_authority(base_authority=authority["rights_authority"], cell=cell,
            scene_plan_digest=derived["scene"]["plan_digest"], packet_receipt_digest=cell_receipt["receipt_digest"])
        worker.update(request_digest=authority["runtime_inputs_digest"], rights_review_digest=cell_authority["authority_digest"])
        seal(worker, "result_digest")
        for filename, value in ((runtime.WORKER_RESULT, worker), ("episode/" + runtime.EPISODE, episode), ("episode/" + runtime.TRACE, trace)):
            runtime._write(cell_root / filename, value)
    row = runtime.collect_cell(root=root, cell_root=cell_root, cell=cell,
        scene_plan_digest=derived["scene"]["plan_digest"], candidate_id=DP, request_digest=authority["runtime_inputs_digest"],
        returncode=None, reason=None if completed else "g1_beta_native_worker_deadline")
    rows = [row, *[runtime._empty_cell(c) for c in authority["cells"][1:]]]
    wire = runtime._result(authority, rows, stage="native_worker", blockers=[])
    runtime._write(root / session.RESULT_FILENAME, wire)
    return root, bundle, wire


def test_reference_result_preserves_partial_actions_without_duplicate_arrays(tmp_path, monkeypatch):
    root, bundle, wire = fixture(tmp_path, monkeypatch)
    assert "query_checkpoints" not in wire["episodes"][0]
    assert wire["episodes"][0]["query_checkpoint_count"] == 1
    assert (root / session.RESULT_FILENAME).stat().st_size < 64 * 1024
    verified = output.verify_output(output_dir=root, bundle=bundle)
    assert verified["source_result_digest"] == wire["result_digest"]
    assert verified["result_digest"] != wire["result_digest"]
    assert verified["episodes"][0]["query_checkpoints"][0]["action_chunk"]
    assert verified["candidate_policy_queried"] is True
    assert verified["episodes"][0]["status"] == "blocked"
    assert "session_closeout" not in verified


def test_completed_native_chain_recomputes_the_negative_score(tmp_path, monkeypatch):
    root, bundle, wire = fixture(tmp_path, monkeypatch, completed=True)
    verified = output.verify_output(output_dir=root, bundle=bundle)
    assert verified["episodes"][0]["episode"]["score"]["task_succeeded"] is False
    assert verified["episodes"][0]["status"] == "completed"
    # Even a rehashed provider artifact cannot replace the deterministic score.
    record = next(r for r in wire["episodes"][0]["artifact_records"] if r["relative_path"].endswith(runtime.EPISODE))
    path = root / record["relative_path"]
    episode = json.loads(path.read_text())
    episode["score"]["task_succeeded"] = True
    seal(episode["score"], "report_digest")
    seal(episode, "result_digest")
    runtime._write(path, episode)
    record.update(size_bytes=path.stat().st_size, sha256=runtime._sha(path))
    seal(wire, "result_digest")
    runtime._write(root / session.RESULT_FILENAME, wire)
    with pytest.raises(ValueError, match="independent_score_changed"):
        output.verify_output(output_dir=root, bundle=bundle)


@pytest.mark.parametrize("case", ["changed_bytes", "wrong_current_scene", "unlisted_query", "inline_substitution", "completed_without_worker"])
def test_self_resealed_result_cannot_reassign_or_hide_query_evidence(tmp_path, monkeypatch, case):
    root, bundle, wire = fixture(tmp_path, monkeypatch)
    row = wire["episodes"][0]
    checkpoint = root / row["artifact_records"][0]["relative_path"]
    if case == "changed_bytes":
        checkpoint.write_text(checkpoint.read_text() + " ")
    elif case == "wrong_current_scene":
        row["scene_plan_digest"] = "sha256:" + "f" * 64
    elif case == "unlisted_query":
        checkpoint.with_name("query_0001.v1.json").write_bytes(checkpoint.read_bytes())
    elif case == "inline_substitution":
        row["query_checkpoints"] = [json.loads(checkpoint.read_text())]
    else:
        row["status"] = "completed"
    seal(wire, "result_digest")
    runtime._write(root / session.RESULT_FILENAME, wire)
    with pytest.raises(ValueError):
        output.verify_output(output_dir=root, bundle=bundle)
