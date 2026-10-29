"""Retained-byte adoption is diagnostic appearance, never another provider run."""
from copy import deepcopy
import hashlib
import json
import zipfile

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest, cross_runtime_canonical_digest
from blueprint_pipeline.owned_capture_atlas_reconstruction import (
    adopt_owned_atlas_reconstruction, project_owned_atlas_reconstruction,
)
from blueprint_pipeline.reconstruction_capability import (
    ReconstructionContractError, aggregate_reconstruction_costs, normalize_reconstruction_result,
)
from blueprint_pipeline.reconstruction_control_plane import (
    load_reconstruction_compilation_inputs, prepare_owned_capture_atlas_plan,
)
from tests.test_reconstruction_capability import _reconstruction_result
from tests.test_reconstruction_control_plane import _seed_owned_atlas_source


def sha(value):
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _write_retained_result(root, plan):
    digest = plan["source_video_digest"]
    directory = root / "download"
    (directory / "splat").mkdir(parents=True)
    data = b"ply\nformat ascii 1.0\nelement vertex 0\nend_header\n"
    (directory / "splat/draft.ply").write_bytes(data)
    manifest = {"schema_version": "atlas_download_bundle.v1", "source_video_digest": digest,
        "claim_ceiling": "development_only", "metric_measurement_proven": False,
        "simulation_qualified": False, "scale_status": "unscaled_provider_world_units",
        "splat_available": True, "assets": [{"path": "splat/draft.ply", "sha256": sha(data), "bytes": len(data)}]}
    manifest["digest"] = canonical_digest(manifest, digest_field="digest")
    (directory / "manifest.json").write_text(json.dumps(manifest))
    archive = root / "download.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        for name in ("manifest.json", "splat/draft.ply"):
            bundle.write(directory / name, name)
    result = {"schema_version": "website_atlas_pipeline_result.v1", "task_context_digest": plan["source_context"]["context_digest"], "source_video_digest": digest,
        "download": {"directory": str(directory), "zip_path": str(archive), "zip_sha256": sha(archive.read_bytes()),
                     "manifest_digest": manifest["digest"]}}
    result["digest"] = canonical_digest(result, digest_field="digest")
    result.update(status="ready", splat_available=True, mesh_available=False)
    result["digest"] = canonical_digest(result, digest_field="digest")
    (root / "result.json").write_text(json.dumps(result))
    return result, manifest


@pytest.fixture
def retained(tmp_path):
    store, state = tmp_path / "store", tmp_path / "state"
    session, intake, digest = _seed_owned_atlas_source(store)
    plan = prepare_owned_capture_atlas_plan(state_root=state, capture_store_root=store,
        capture_session_id=session, intake_id=intake, requested_claim_types=["navigation_clearance"],
        idempotency_key="retained-plan")
    root = state / "plans" / plan["plan_id"] / "atlas"
    result, manifest = _write_retained_result(root, plan)
    job = {"schema_version": "owned_capture_atlas_job.v1", "state": "ready", "plan_id": plan["plan_id"],
        "request_id": plan["plan_id"], "plan_digest": plan["plan_digest"], "capture_id": session,
        "intake_id": intake, "owner_user_id": "owner", "task_context_digest": result["task_context_digest"],
        "source_video_digest": digest, "result_digest": result["digest"], "authorization_digest": "sha256:" + "6" * 64,
        "enqueued_at_epoch": 10, "revised_at_epoch": 20, "updated_at_epoch": 20,
        "archive": {"digest": result["download"]["zip_sha256"], "manifest_digest": manifest["digest"],
                    "size_bytes": (root / "download.zip").stat().st_size, "filename": "capture-session-1.zip"},
        "stage": "archived", "reason": None, "revision": 2, "splat_available": True}
    (root / "owned_job.json").write_text(json.dumps(job))
    (root / "execution_binding.json").write_text(json.dumps({**job, "schema_version": "owned_capture_atlas_queue_binding.v1"}))
    (root / "result.json").write_text(json.dumps(result))
    return dict(state_root=state, capture_store_root=store, plan_id=plan["plan_id"]), plan, job, result, manifest, root


def project(retained):
    _, plan, job, result, manifest, _ = retained
    return project_owned_atlas_reconstruction(owned_plan=plan, job=job, result=result,
        manifest=manifest, implementation_digest="sha256:" + "1" * 64)


def test_real_loader_adopts_identical_bytes_without_provider_and_preserves_unknowns(retained, monkeypatch):
    from blueprint_pipeline import website_atlas_pipeline
    monkeypatch.setattr(website_atlas_pipeline, "run_atlas_pipeline", lambda **_: pytest.fail("provider dispatch"))
    args, _, job, _, _, root = retained
    first = adopt_owned_atlas_reconstruction(**args)
    execution = first["execution_result"]
    assert execution["cost_usd"] is None and execution["known_cost_subtotal_usd"] == 0.0
    assert execution["state"] == "partial"
    row = first["reconstruction_results"][0]
    assert row["outputs"] == ["appearance_layer"]
    assert row["uncertainty_map"]["generated_extent"] == "unknown"
    assert not row["claim_ceiling"]["collision_geometry"]
    assert not row["camera_solution"]["calibrated"]
    assert first["context"]["capture_session_id"] == "capture-session-1"
    assert first["capture_intake_envelope"]["known_task_specification"]["operations"][-1] == "stop_before_end_shelving"
    job.update(updated_at_epoch=900, provider_cleanup={"state": "complete"})
    (root / "owned_job.json").write_text(json.dumps(job))
    assert adopt_owned_atlas_reconstruction(**args) == first
    assert load_reconstruction_compilation_inputs(**args,
        execution_result_digest=execution["execution_result_digest"]) == first
    with pytest.raises(ValueError, match="execution_digest_mismatch"):
        load_reconstruction_compilation_inputs(**args, execution_result_digest="sha256:" + "f" * 64)


def test_projection_binds_source_result_and_manifest(retained):
    projection = project(retained)
    assert projection["reconstruction_results"][0]["provider_receipt"]["owned_result_digest"] == retained[3]["digest"]
    retained[2]["source_video_digest"] = "sha256:" + "f" * 64
    with pytest.raises(ValueError, match="binding_invalid"):
        project(retained)


def test_execution_digest_survives_javascript_integer_float_reserialization(retained):
    execution = project(retained)["execution_result"]

    def javascript_numbers(value):
        if isinstance(value, dict):
            return {key: javascript_numbers(item) for key, item in value.items()}
        if isinstance(value, list):
            return [javascript_numbers(item) for item in value]
        return int(value) if isinstance(value, float) and value.is_integer() else value

    transported = javascript_numbers(execution)
    assert transported["known_cost_subtotal_usd"] == 0
    assert cross_runtime_canonical_digest(transported, digest_field="execution_result_digest") == execution["execution_result_digest"]


@pytest.mark.parametrize("member", ["download.zip", "download/splat/draft.ply"])
def test_adoption_refuses_changed_retained_bytes(retained, member):
    (retained[-1] / member).write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        adopt_owned_atlas_reconstruction(**retained[0])


def test_adoption_refuses_current_source_revocation(retained):
    args = retained[0]
    marker = hashlib.sha256(b"capture-session-1\0intake-1").hexdigest()
    path = args["capture_store_root"] / "lifecycle_tombstones" / (marker + ".json")
    path.parent.mkdir(parents=True)
    path.write_text('{"revoked":true}')
    with pytest.raises(ValueError, match="capture_revoked_or_expired"):
        adopt_owned_atlas_reconstruction(**args)


def test_adoption_refuses_retargeted_job_owner(retained):
    retained[2]["owner_user_id"] = "another-owner"
    (retained[-1] / "owned_job.json").write_text(json.dumps(retained[2]))
    with pytest.raises(ValueError, match="execution_binding_invalid"):
        adopt_owned_atlas_reconstruction(**retained[0])


def test_zip_must_actually_contain_the_bound_manifest(retained):
    args, _, job, result, _, root = retained
    archive = root / "download.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("manifest.json", "{}")
        bundle.write(root / "download/splat/draft.ply", "splat/draft.ply")
    # Even a coherent outer archive/receipt digest cannot replace its manifest.
    result["download"]["zip_sha256"] = sha(archive.read_bytes())
    result["digest"] = canonical_digest(result, digest_field="digest")
    job["result_digest"] = result["digest"]
    job["archive"].update(digest=result["download"]["zip_sha256"], size_bytes=archive.stat().st_size)
    (root / "owned_job.json").write_text(json.dumps(job))
    (root / "result.json").write_text(json.dumps(result))
    with pytest.raises(ValueError, match="archive_manifest_changed"):
        adopt_owned_atlas_reconstruction(**args)


def test_existing_compiler_still_refuses_unapproved_task_and_missing_task_objects(retained):
    from blueprint_pipeline.site_task_testbed_compiler import (
        SiteTaskTestbedCompilerError, build_pipeline_owned_compilation_support,
    )
    from tests.test_site_task_testbed_compiler import _approved_task, _robot_binding
    adopted = adopt_owned_atlas_reconstruction(**retained[0])
    arguments = dict(testbed_id="testbed", version="1", capture_qa_report=adopted["capture_qa_report"],
        reconstruction_plan=adopted["reconstruction_plan"], robot_binding=_robot_binding())
    task = _approved_task()
    task["approval_status"] = "proposed_unconfirmed"
    task["approved_task_digest"] = canonical_digest(task, digest_field="approved_task_digest")
    with pytest.raises(SiteTaskTestbedCompilerError, match="not_approved"):
        build_pipeline_owned_compilation_support(approved_task_definition=task, **arguments)
    task = _approved_task()
    task["task"]["task_objects"] = []
    task["approved_task_digest"] = canonical_digest(task, digest_field="approved_task_digest")
    with pytest.raises(SiteTaskTestbedCompilerError, match="task_objects:missing"):
        build_pipeline_owned_compilation_support(approved_task_definition=task, **arguments)


def test_unknown_cost_preserves_legacy_digest_and_known_subtotal(retained):
    numeric = normalize_reconstruction_result(_reconstruction_result())
    assert normalize_reconstruction_result(numeric) == numeric
    assert "cost_status" not in numeric
    assert aggregate_reconstruction_costs([numeric]) == {"cost_usd": 0.0}
    raw = deepcopy(numeric)
    raw.pop("reconstruction_result_digest")
    raw["cost_usd"] = 2.5
    numeric = normalize_reconstruction_result(raw)
    unknown = project(retained)["reconstruction_results"][0]
    assert aggregate_reconstruction_costs([numeric, unknown]) == {"cost_usd": None,
        "cost_status": "unknown", "cost_reason": "provider_usage_unreported", "known_cost_subtotal_usd": 2.5}


@pytest.mark.parametrize("change", [{"cost_usd": 0}, {"cost_reason": ""},
    {"known_cost_subtotal_usd": float("inf")}, {"cost_status": "known"}])
def test_unknown_cost_metadata_must_be_coherent(retained, change):
    row = project(retained)["reconstruction_results"][0]
    row.pop("reconstruction_result_digest")
    row.update(change)
    with pytest.raises(ReconstructionContractError):
        normalize_reconstruction_result(row)
