"""Prefix bookkeeping is real; GPU evidence validators have separate coverage.

These tests never label their synthetic stage files scientific evidence. The
full scientific replay was also exercised read-only against retained R13 bytes.
"""
from copy import deepcopy
import json
from pathlib import Path

import pytest

import blueprint_pipeline.task_evaluation_sam31_prefix_adoption as adoption
from blueprint_pipeline import task_evaluation_sam31_prefix_evidence as evidence
from blueprint_pipeline.decision_evidence_contracts import canonical_digest, canonical_json
from blueprint_pipeline.task_evaluation_sam31_phase_queue import enqueue_sam31_phase
from blueprint_pipeline.task_evaluation_scene_configuration_sam31_plan import build_sam31_preparation_plan
from tests.test_sam31_camera_geometry import geometry_fixture
from tests.test_task_evaluation_launch_preparation_worker import production_request_with_fetchable_bytes


OLD, NEW = "a" * 40, "b" * 40


def test_automatic_selector_prefers_longest_compatible_and_keeps_rejections(monkeypatch):
    calls = []
    def materialize(**kwargs):
        calls.append((kwargs["through_phase"], kwargs["output_path"]))
        if kwargs["through_phase"] == "segment_cutout":
            raise ValueError("sam31_adoption_producer_code_changed")
        return {"through_phase": kwargs["through_phase"]}
    monkeypatch.setattr(adoption, "materialize_completed_prefix_adoption", materialize)
    result = adoption.select_completed_prefix_adoption(output_path="selected.json")
    assert result["through_phase"] == "sam31_tracking"
    assert result["rejected_candidates"][0]["blocker"] == "sam31_adoption_producer_code_changed"
    assert calls == [("segment_cutout", None), ("sam31_tracking", None), ("sam31_tracking", "selected.json")]


def test_automatic_selector_reports_no_compatible_prefix_without_publishing(monkeypatch):
    def materialize(**kwargs):
        assert kwargs["output_path"] is None
        raise ValueError("sam31_adoption_task_or_source_changed")
    monkeypatch.setattr(adoption, "materialize_completed_prefix_adoption", materialize)
    result = adoption.select_completed_prefix_adoption(output_path="must-not-exist.json")
    assert result["status"] == "no_reusable_prefix"
    assert len(result["rejected_candidates"]) == 3


def write(path, value, field=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if field:
        value[field] = canonical_digest(value, digest_field=field)
    if path.exists():
        path.chmod(0o600)  # Explicit mutation of this test-owned immutable fixture.
    path.write_text(canonical_json(value))
    return adoption.record(path)


@pytest.fixture
def prefix(tmp_path):
    task = {"expected_production_commit": OLD, "subject": {"source_instance_id": "115"}}
    host = {name: write(tmp_path / (name + ".json"), task if name == "task_request" else {"fixture": name})
            for name in ("task_request", "installation_receipt", "publisher_intake", "source_preparation_receipt", "interiorgs_terms")}
    provider = write(tmp_path / "sam-provider.json", {"model": "fixture-only"})
    profile = {"schema_version": adoption.PROFILE_SCHEMA, "source_commit": OLD,
               "review_model": "gpt-5.6-terra", "review_maximum_cost_usd": 1., "candidate_policy_queried": False,
               "artifact_references": {"sam31_provider_profile": provider}}
    profile_ref = write(tmp_path / "profile.json", profile, "profile_digest")
    request, _ = production_request_with_fetchable_bytes()
    request["expected_production_commit"] = OLD
    geometry = geometry_fixture(tmp_path / "geometry")
    lower, upper = geometry.pop("source_min"), geometry.pop("source_max")
    plan = build_sam31_preparation_plan(source_commit=OLD,
        task={"task_identity": request["task"]["identity"], "scene_identity": request["scene"]["identity"], "publisher_scene_id": "841757"},
        host_inputs={k: Path(v["path"]) for k, v in host.items()}, source_min=lower, source_max=upper,
        server_profile_path=Path(profile_ref["path"]), camera_geometry=geometry)
    plan_ref = write(tmp_path / "plan.json", plan)
    from blueprint_pipeline.task_evaluation_launch_preparation_contract import launch_preparation_request_digest
    request["runtime"]["mounts"].append({"source": {"uri": "s3://blueprint-production-inputs/adoption-plan.json",
        "digest": plan_ref["sha256"], "size_bytes": plan_ref["size_bytes"]}, "container_path": "/inputs/adoption-plan.json", "mode": "read_only"})
    parent_digest = launch_preparation_request_digest(request)
    envelope = {"request": request, "request_digest": parent_digest}
    parent_ref = write(tmp_path / "parents/blocked/parent.json", envelope, "envelope_digest")
    rows, inputs, artifacts = [], {**host, **profile["artifact_references"]}, {}
    names = {"source_selections": "task_selection", "standard_splat_conversion": "standard_splat_conversion_receipt",
             "calibrated_views": "calibrated_view_receipt", "sam31_inputs": "sam31_run_request", "sam31_tracking": "sam31_source_tracks"}
    for phase in adoption.PHASES[:5]:
        intake = enqueue_sam31_phase(queue_root=tmp_path / "queue", parent_preparation_id=request["preparation_id"],
            parent_request_digest=parent_digest, expected_source_commit=OLD, plan_ref=plan_ref, phase=phase, inputs=inputs)
        job_path = Path(intake["job_path"])
        job = json.loads(job_path.read_text())
        target = tmp_path / "queue/completed" / job_path.name
        job_path.rename(target)
        result_artifacts = {names[phase]: write(tmp_path / "artifacts" / f"{phase}.json", {"hermetic_phase": phase})}
        if phase == "standard_splat_conversion":
            result_artifacts["standard_splat"] = write(tmp_path / "artifacts/standard.json", {"not_real_splat": True})
        outcome = {"status": "completed", "stage_id": phase, "artifacts": result_artifacts}
        result = {"schema_version": "task_evaluation_sam31_preparation_execution_result.v1", "source_commit": OLD,
                  "status": "completed", "artifacts": result_artifacts,
                  **{key: job[key] for key in ("phase", "job_digest", "child_id", "parent_request_digest", "plan_digest")}}
        result_ref = write(Path(intake["result_path"]), result, "result_digest")
        receipt = {"schema_version": "task_evaluation_sam31_phase_execution_receipt.v1", "source_commit": OLD,
                   "job_digest": job["job_digest"], "phase": phase, "outcome": outcome}
        execution_ref = write(tmp_path / "executions" / job["child_id"] / "phase_execution_receipt.v1.json", receipt, "receipt_digest")
        rows.append({"phase": phase, "job": adoption.record(target), "result": result_ref, "execution_receipt": execution_ref})
        inputs.update(result_artifacts)
        artifacts.update(result_artifacts)
        if phase == "standard_splat_conversion":
            inputs["standard_splat_conversion"] = artifacts["standard_splat_conversion"] = artifacts["standard_splat_conversion_receipt"]
    value = {"schema_version": adoption.SCHEMA, "status": "verified_completed_prefix", "source_commit": NEW,
             "original_execution_commit": OLD, "original_parent_request_digest": parent_digest,
             "original_parent_envelope": parent_ref, "through_phase": "sam31_tracking", "source_plan": plan_ref,
             "source_profile": profile_ref, "phase_records": rows, "historical_receipts_modified": False,
             "paid_execution_performed": False, "candidate_policy_queried": False}
    return value, plan, profile, artifacts


def test_real_phase_chain_accepts_only_complete_three_or_five_prefixes(prefix, tmp_path):
    value, plan, profile, artifacts = prefix
    before = {row["result"]["path"]: Path(row["result"]["path"]).read_bytes() for row in value["phase_records"]}
    observed = adoption._phase_chain(value, (tmp_path,))
    assert observed[:3] == (plan, profile, artifacts)
    value["through_phase"] = "calibrated_views"
    value["phase_records"] = value["phase_records"][:3]
    assert set(adoption._phase_chain(value, (tmp_path,))[3]) == set(adoption.PHASES[:3])
    assert before == {path: Path(path).read_bytes() for path in before}


@pytest.mark.parametrize("fault", ["failed", "partial", "bytes", "job", "camera", "plan_digest"])
def test_real_prefix_chain_rejects_changed_evidence(prefix, tmp_path, fault):
    value, plan, _, _ = prefix
    row = value["phase_records"][2]
    if fault == "partial":
        value["phase_records"].pop(1)
    elif fault in {"camera", "plan_digest"}:
        plan["camera_policy"]["views"][0]["position_offset_m"][0] += .1
        if fault == "camera":
            plan["plan_digest"] = canonical_digest(plan, digest_field="plan_digest")
        value["source_plan"] = write(Path(value["source_plan"]["path"]), plan)
    elif fault == "job":
        job = json.loads(Path(row["job"]["path"]).read_text())
        job["expected_source_commit"] = NEW
        row["job"] = write(Path(row["job"]["path"]), job, "job_digest")
    else:
        result = json.loads(Path(row["result"]["path"]).read_text())
        if fault == "failed":
            result["status"] = "failed"
            row["result"] = write(Path(row["result"]["path"]), result, "result_digest")
        else:
            Path(next(iter(result["artifacts"].values()))["path"]).write_text("changed evidence bytes")
    with pytest.raises(ValueError):
        adoption._phase_chain(value, (tmp_path,))


def test_real_phase_chain_rejects_duplicate_child_owner_states(prefix, tmp_path):
    """A completed child is not reusable while another queue state owns its id."""
    value, _, _, _ = prefix
    completed = Path(value["phase_records"][0]["job"]["path"])
    processing = completed.parent.parent / "processing" / completed.name
    processing.parent.mkdir(parents=True, exist_ok=True)
    processing.write_bytes(completed.read_bytes())
    with pytest.raises(ValueError, match="sam31_adoption_job_identity_ambiguous"):
        adoption._phase_chain(value, (tmp_path,))


def test_adoption_retry_returns_immutable_record_after_fresh_zero(tmp_path, monkeypatch):
    """A retry rechecks live zero but never regenerates a new adoption digest."""
    zero = write(tmp_path / "zero.json", {
        "provider": "vast", "status": "observed", "api_confirmed": True,
        "name_prefix": "", "live_resource_count": 0, "resources": [],
        "http": 200, "observed_at_epoch": 1000.,
    })
    provider = write(tmp_path / "provider.json", {"profile": "current"})
    plan = write(tmp_path / "plan.json", {"plan": "retained"})
    profile = write(tmp_path / "profile.json", {"profile": "retained"})
    host = {"task_request": write(tmp_path / "task.json", {"task": "current"})}
    existing = {
        "schema_version": adoption.SCHEMA,
        "status": "verified_completed_prefix",
        "source_commit": NEW,
        "original_parent_request_digest": "sha256:" + "c" * 64,
        "current_release_root": str(tmp_path / "release"),
        "current_sam31_provider_profile": provider,
        "current_host_inputs": host,
        "source_plan": plan,
        "source_profile": profile,
        "through_phase": "calibrated_views",
        "adoption_digest": "",
    }
    output = tmp_path / "adoption.json"
    write(output, existing, "adoption_digest")
    before = output.read_bytes()
    checks = []
    monkeypatch.setattr(
        adoption, "validate_completed_prefix_adoption",
        lambda value, **kwargs: checks.append((value, kwargs)),
    )
    result = adoption.materialize_completed_prefix_adoption(
        source_plan_path=plan["path"], source_profile_path=profile["path"],
        parent_request_digest=existing["original_parent_request_digest"],
        through_phase="calibrated_views", current_host_inputs=host,
        current_provider_profile_path=provider["path"],
        current_repo_root=existing["current_release_root"], expected_source_commit=NEW,
        provider_zero_path=zero["path"], output_path=output, approved_roots=(tmp_path,),
        now_epoch=1001.,
    )
    assert result == json.loads(before.decode())
    assert output.read_bytes() == before and checks


@pytest.mark.parametrize("field", ["subject", "success", "support", "instruction", "destination"])
def test_task_science_does_not_allow_scientific_changes(field):
    before = {"expected_production_commit": OLD, "run_prefix": "old", field: {"value": 1}}
    administrative = {**before, "expected_production_commit": NEW, "run_prefix": "new"}
    assert evidence.task_science(before) == evidence.task_science(administrative)
    administrative[field] = {"value": 2}
    assert evidence.task_science(before) != evidence.task_science(administrative)


def test_model_science_keeps_checkpoint_code_image_and_parameters():
    model = {"source_commit_sha": OLD, "checkpoint_digest": "old", "runtime_image_identity": "image", "max_num_objects": 5}
    assert evidence.provider_science(model) == evidence.provider_science({**model, "source_commit_sha": NEW})
    for field in ("checkpoint_digest", "runtime_image_identity", "max_num_objects"):
        assert evidence.provider_science(model) != evidence.provider_science({**model, field: "changed"})


@pytest.mark.parametrize("fault", ["missing", "scoped", "live", "unconfirmed", "stale"])
def test_global_zero_is_authenticated_global_and_recent(tmp_path, fault):
    value = {"provider": "vast", "status": "observed", "api_confirmed": True, "name_prefix": "", "live_resource_count": 0,
             "resources": [], "http": 200, "observed_at_epoch": 1000.}
    path = tmp_path / "zero.json"
    write(path, value)
    adoption._zero(path, at=1001.)
    if fault == "missing":
        path.unlink()
    else:
        value.update({"scoped": {"name_prefix": "blueprint"}, "live": {"live_resource_count": 1},
                      "unconfirmed": {"api_confirmed": False}, "stale": {"observed_at_epoch": 1.}}[fault])
        write(path, value)
    with pytest.raises((OSError, ValueError)):
        adoption._zero(path, at=1001.)


def test_canonical_source_validator_rehashes_dataset_and_rights(tmp_path):
    from tests.test_public_scene_removal_selection import _source_fixture, SHA
    fixture = _source_fixture(tmp_path)
    host = {name: adoption.record(fixture[key]) for name, key in
            (("task_request", "task_request"), ("installation_receipt", "installation_receipt"),
             ("publisher_intake", "publisher_intake"), ("source_preparation_receipt", "source_preparation"))}
    _, context, _ = evidence.source_science(host, SHA)
    context["raw"]["appearance_3dgs"]["path"].chmod(0o600)
    context["raw"]["appearance_3dgs"]["path"].write_bytes(b"changed canonical appearance")
    with pytest.raises(ValueError):
        evidence.source_science(host, SHA)


def test_driver_adopts_old_five_stage_chain_and_only_enqueues_new_review(prefix, tmp_path, monkeypatch):
    """Test orchestration; scientific validators are isolated at their explicit seam."""
    from blueprint_pipeline import task_evaluation_scene_configuration_sam31_preparation_driver as driver
    value, old_plan, old_profile, _ = prefix
    _, _, artifacts, _, _ = adoption._phase_chain(value, (tmp_path,))
    current_host = deepcopy(old_plan["host_inputs"])
    current_host["task_request"] = write(tmp_path / "current-task.json", {"expected_production_commit": NEW, "subject": {"source_instance_id": "115"}})
    new_conversion = write(tmp_path / "new-conversion.json", {"commit": NEW})
    value.update(created_at_epoch=1001., current_host_inputs=current_host, current_release_root=str(tmp_path),
                 current_sam31_provider_profile=old_profile["artifact_references"]["sam31_provider_profile"],
                 retained_release_pin={"source_commit": OLD}, tracking_identity={"validated_fixture": True},
                 sam31_billing_source=write(tmp_path / "billing.json", {"fixture_only": True}),
                 provider_zero_at_adoption=write(tmp_path / "zero.json", {"provider": "vast", "status": "observed", "api_confirmed": True,
                    "name_prefix": "", "live_resource_count": 0, "resources": [], "http": 200, "observed_at_epoch": 1000.}))
    value["administrative_rebindings"] = {name: {"original": artifacts[name], "successor": successor} for name, successor in
        (("standard_splat", artifacts["standard_splat"]), ("standard_splat_conversion_receipt", new_conversion), ("standard_splat_conversion", new_conversion))}
    adoption_ref = write(tmp_path / "adoption.json", value, "adoption_digest")
    calls = []
    monkeypatch.setattr(adoption, "source_science", lambda host, commit: (json.loads(Path(host["task_request"]["path"]).read_text()), {}, {}))
    monkeypatch.setattr(adoption, "validate_current_rights", lambda *a: {"standard": artifacts["standard_splat"], "conversion": new_conversion})
    monkeypatch.setattr(adoption, "validate_render", lambda *a: calls.append("closed_render_validator") or {"source_commit": OLD})
    monkeypatch.setattr(adoption, "validate_tracking", lambda *a: calls.append("closed_tracking_validator") or {"validated_fixture": True})
    monkeypatch.setattr("blueprint_pipeline.public_scene_inpainting_inputs._git_identity", lambda _: {"commit": NEW})
    profile = {**old_profile, "source_commit": NEW, "completed_prefix_adoption": adoption_ref}
    profile_ref = write(tmp_path / "new-profile.json", profile, "profile_digest")
    plan = {**old_plan, "source_commit": NEW, "host_inputs": current_host, "server_profile_sha256": profile_ref["sha256"]}
    plan_ref = write(tmp_path / "new-plan.json", plan, "plan_digest")
    monkeypatch.setenv(driver.PROFILE_ENV, profile_ref["path"])
    monkeypatch.setenv(driver.CHILD_QUEUE_ENV, str(tmp_path / "new-queue"))
    expected = {"uri": "s3://blueprint-production-inputs/new-plan.json", "digest": plan_ref["sha256"], "size_bytes": plan_ref["size_bytes"]}
    context = {"expected_source_commit": NEW, "request_digest": "sha256:" + "c" * 64,
               "request": {"preparation_id": "successor", "scene": {"identity": plan["scene_identity"]}, "task": {"identity": plan["task_identity"]}},
               "stage_one_configuration": {"sam31_preparation_plan": expected},
               "materialized_references": [{**expected, "materialized_path": plan_ref["path"]}]}
    result = driver.advance_sam31_preparation(context, approved_roots=(tmp_path,))
    assert result["phase"] == "sam31_review" and result["status"] == "waiting_for_child"
    assert calls == ["closed_render_validator", "closed_tracking_validator"]
    pending = list((tmp_path / "new-queue/pending").glob("*.json"))
    assert len(pending) == 1
    job = json.loads(pending[0].read_text())
    assert job["expected_source_commit"] == NEW and job["inputs"]["standard_splat_conversion_receipt"] == new_conversion
    assert result["completed_prefix_adoption"]["original_execution_commit"] == OLD
    assert list((tmp_path / "new-queue/results").glob("*.json")) == []


def test_published_prefix_pin_is_recognized_by_canonical_retention(prefix, tmp_path):
    from blueprint_pipeline.task_evaluation_release_retention import _evidence_binding_protections
    value, _, _, _ = prefix
    value["retained_release_pin"] = {"source_commit": OLD, "path": "/immutable/releases/" + OLD, "tree": "d" * 40}
    ref = write(tmp_path / "adoption.json", value, "adoption_digest")
    bindings = tmp_path / "release-retention-bindings"
    bindings.mkdir()
    pin = adoption.publish_adoption_release_binding(ref["path"], binding_root=bindings)
    before = Path(pin["path"]).read_bytes()
    assert adoption.publish_adoption_release_binding(ref["path"], binding_root=bindings) == pin
    protected, _ = _evidence_binding_protections(bindings)
    assert OLD in protected
    assert Path(pin["path"]).read_bytes() == before
    assert json.loads(before)["evidence"] == ref


def test_producer_code_drift_is_tolerated_but_a_missing_file_still_refuses(tmp_path):
    """Piece 1 — content identity, not commit identity.

    A completed paid prefix must survive a deploy that changes producer-code bytes, as long
    as the retained OUTPUTS still pass the current validators (enforced elsewhere in
    validate_render/validate_completed_prefix_adoption). require_producer_files_present now
    only requires each producer file to EXIST under both trees, pinning the retained sha; it
    no longer refuses on a byte diff. A genuinely missing producer file still fails closed.
    """
    old_root, new_root = tmp_path / "old", tmp_path / "new"
    rel = "src/blueprint_pipeline/sam31_source_calibration_stage.py"
    for root, body in ((old_root, "# produced the retained prefix\n"), (new_root, "# owner-authority propagation diff\n")):
        (root / "src/blueprint_pipeline").mkdir(parents=True)
        (root / rel).write_text(body)

    # Drift (different bytes) is tolerated; the pinned sha is the retained (old-root) one.
    files = evidence.require_producer_files_present((rel,), old_root, new_root)
    assert files == [{"relative_path": rel, "sha256": evidence.sha(old_root / rel),
                      "size_bytes": (old_root / rel).stat().st_size}]
    assert evidence.sha(old_root / rel) != evidence.sha(new_root / rel)  # bytes really differ

    # A missing producer file (either tree) still fails closed.
    (new_root / rel).unlink()
    with pytest.raises(ValueError, match="sam31_adoption_producer_code_missing:" + rel):
        evidence.require_producer_files_present((rel,), old_root, new_root)
    (new_root / rel).write_text("restored\n")
    (old_root / rel).unlink()
    with pytest.raises(ValueError, match="sam31_adoption_producer_code_missing:" + rel):
        evidence.require_producer_files_present((rel,), old_root, new_root)
