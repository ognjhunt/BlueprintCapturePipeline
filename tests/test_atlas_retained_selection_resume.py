"""Retained selection reaches exact task recovery without resetting its allowance."""
from copy import deepcopy
from datetime import datetime, timezone
import json

import pytest

from blueprint_pipeline import atlas_video_inputs, paid_resource_allocator as allocator
from blueprint_pipeline import website_atlas_pipeline as pipeline
from blueprint_pipeline import website_task_context as control
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from tests.test_atlas_native_splat import native_case

pytest_plugins = ["tests.test_website_atlas_tasks"]


@pytest.fixture
def selected_resume(case, tmp_path, monkeypatch):
    task_root = tmp_path / "job/automation/refined/native"
    fake, request, _ = native_case(case, task_root)
    root = tmp_path / "job"
    (root / "survey").mkdir(parents=True)
    video = tmp_path / "source.mov"
    video.write_bytes(b"synthetic retained capture; no production media")
    source = pipeline._sha256_file(video)
    survey = {"schema_version": "atlas_video_survey.v1", "source_video_digest": source,
              "display_rotation_degrees": 0, "heldout_pixels_included": False, "frames": []}
    survey["digest"] = canonical_digest(survey, digest_field="digest")
    (root / "survey/atlas_video_survey.json").write_text(json.dumps(survey))
    (root / "selection_binding.json").write_text(json.dumps({
        "survey_relative_path": "survey/atlas_video_survey.json",
        "review": {"subject_origin": "captured", "automation": {"objects": [{"disposition": "remove"}]}},
        "planner_receipt": {"model": "fixture"}}))
    # Public rate values and remaining counters observed on the affected job;
    # all source, request, provider and authority records here are synthetic.
    policy = {"maximum_review_cost_usd": 4, "pose_input_cost_usd": .01,
              "rgbd_view_cost_usd": .05, "additional_rgbd_view_cost_usd": .01,
              "refined_splat_cost_usd": 5, "mesh_export_cost_usd": .5,
              "rate_card_reference": "fixture", "reviewed_by": "fixture",
              "reviewed_on": datetime.now(timezone.utc).date().isoformat()}
    saved = {"authority": {"provider_retention": "project_library_retention_approved"},
             "policy": policy, "remaining_cost_usd": 25, "remaining_requests": 32,
             pipeline.REVIEW_CAPS: [1, 4]}
    configuration = root / "configuration.json"
    configuration.write_text(json.dumps(saved))
    live = {**saved, "remaining_cost_usd": 12.588387, "remaining_requests": 1}
    configuration_reads = []

    def current(*_args, **kwargs):
        configuration_reads.append(kwargs)
        return deepcopy(live)

    monkeypatch.setattr(pipeline, "load_configuration", current)
    monkeypatch.setattr(atlas_video_inputs, "prepare_video_inputs", lambda **_: {"frames": []})
    monkeypatch.setattr(allocator, "run_sponsored_website_atlas_plan",
                        lambda **_: pytest.fail("retained selection must not buy another review"))
    clock = [datetime.now(timezone.utc).timestamp()]
    quote = {"schema_version": "website_atlas_task_price_review.v1", "task": "posedImages2Splats",
             "request_digest": canonical_digest(request), "maximum_cost_usd": 5,
             **{key: policy[key] for key in ("rate_card_reference", "reviewed_by", "reviewed_on")}}

    def task():
        return allocator.run_sponsored_website_atlas_task(task="posedImages2Splats", request=request,
            output_root=task_root, source_video_digest=source, task_context=fake.context,
            price_review=quote, transport=fake.transport, now=lambda: clock[0], sleep=lambda _: None,
            max_polls=1)

    observed = []

    def automation(**kwargs):
        assert kwargs["configuration"] == live
        result = task()  # Real canonical reservation/submission/retry/output verifier.
        observed.append(result)
        # Isolate task recovery; no new meshing or source-pose stage is run.
        return {"status": "processing", "stage": "refined_splat",
                "operation_id": result.get("operation_id", result.get("operation", {}).get("id"))}

    monkeypatch.setattr(allocator, "run_sponsored_website_atlas_automation", automation)

    def run(checkpoint=None):
        return pipeline.run_atlas_pipeline(source_video=video, output_root=root, task_context=fake.context,
                                           checkpoint=checkpoint)

    return {"fake": fake, "task": task, "run": run, "root": root, "task_root": task_root,
            "live": live, "observed": observed, "clock": clock,
            "configuration": configuration, "configuration_reads": configuration_reads}


@pytest.mark.parametrize("pending", [False, True])
@pytest.mark.parametrize("remaining", [12.588387, 0])
def test_retained_task_recovery_does_not_require_fresh_planning_budget(selected_resume, monkeypatch,
                                                                    pending, remaining):
    setup = selected_resume
    fake = setup["fake"]
    fake.pending = pending
    setup["task"]()
    admission = setup["task_root"] / "controller_admission.json"
    original_admission = admission.read_bytes()
    original_configuration = setup["configuration"].read_bytes()
    setup["live"].update(remaining_cost_usd=remaining, remaining_requests=0)
    monkeypatch.setattr(control, "website_webapp_request",
                        lambda **_: pytest.fail("retained operation must not request another reservation"))
    fake.pending = False
    assert setup["run"]()["stage"] == "refined_splat"
    assert setup["observed"][-1]["status"] == "completed"
    assert fake.reservations == fake.posts == 1
    assert admission.read_bytes() == original_admission
    assert setup["configuration"].read_bytes() == original_configuration
    assert setup["configuration_reads"][0] == {"create": False}


def test_failed_splat_capacity_retry_keeps_both_holds_and_exact_request(selected_resume, monkeypatch):
    setup, admissions, sent = selected_resume, {}, []
    fake = setup["fake"]
    transport = fake.transport

    def reserve(*, capture_id, operation, payload):
        assert operation == "preparation-spend"
        spend = payload["spend"]
        key = spend["allocation_binding_digest"]
        if key in admissions:
            return {**admissions[key], "status": "already_reserved"}
        assert len(admissions) < 2, "a third purchase is not authorized by retained work"
        fake.reservations += 1
        admissions[key] = {**spend, "schema_version": "paid_lane_admission.v1", "status": "admitted",
                           "blockers": [], "external_disclosure_allowed": True, "expires_at_epoch": 1e12}
        return admissions[key]

    def provider(method, url, headers, data):
        if url.endswith("/tasks:posedImages2Splats"):
            sent.append((headers["Idempotency-Key"], data))
            if len(sent) == 2:
                fake.posts += 1
                return 429, b'{"code":"RATE_LIMIT_EXCEEDED"}', {"Retry-After": "360"}
            status, body = transport(method, url, headers, data)
            return status, json.dumps({**json.loads(body), "id": f"op_task_{1 if len(sent) == 1 else 2}"}).encode()
        if url.endswith("/operations/op_task_1"):
            fake.polls += 1
            return 200, json.dumps({"id": "op_task_1", "done": True, "error": {"code": 13,
                "message": "fixture execution failure", "details": [
                    {"domain": "api.worldlabs.ai", "reason": "EXECUTION_FAILED"}]}}).encode()
        if url.endswith("/operations/op_task_2"):
            status, body = transport(method, url.replace("op_task_2", "op_task_1"), headers, data)
            return status, json.dumps({**json.loads(body), "id": "op_task_2"}).encode()
        return transport(method, url, headers, data)

    monkeypatch.setattr(control, "website_webapp_request", reserve)
    fake.transport = provider
    first = setup["task"]()
    assert first["waiting_for"] == "provider_capacity"
    assert len(admissions) == fake.reservations == fake.posts == 2
    assert sum(row["maximum_cost_usd"] for row in admissions.values()) == 10
    setup["live"].update(remaining_cost_usd=0, remaining_requests=0)
    original = setup["task_root"] / "atlas_pose/submission.json"
    failed_bytes = original.read_bytes()
    held_receipts = {path: path.read_bytes() for path in setup["task_root"].rglob("controller_admission.json")}
    setup["clock"][0] = first["retry_not_before_epoch"] - .01
    setup["run"]()
    assert setup["observed"][-1]["waiting_for"] == "provider_capacity" and fake.posts == 2
    assert original.read_bytes() == failed_bytes
    setup["clock"][0] = first["retry_not_before_epoch"]
    setup["run"]()
    assert setup["observed"][-1]["status"] == "completed"
    assert fake.posts == 3 and fake.reservations == len(admissions) == 2
    assert sent[1] == sent[2] and sent[0][0] != sent[1][0]
    failed, completed = json.loads(failed_bytes), json.loads(original.read_bytes())
    assert all(completed[key] == failed[key] for key in (
        "request_digest", "binding", "operation_id", "terminal_operation", "terminal_observed_at_epoch"))
    assert completed["completed_operation_id"] == "op_task_2"
    assert all(path.read_bytes() == data for path, data in held_receipts.items())
    assert setup["observed"][-1]["billing"]["total_reserved_usd"] == 10


def test_selected_resume_still_refuses_a_new_purchase_without_allowance(selected_resume, monkeypatch):
    setup = selected_resume
    setup["live"].update(remaining_cost_usd=0, remaining_requests=0)
    reservations = []

    def refuse(**kwargs):
        reservations.append(kwargs)
        raise ValueError("website_atlas_allowance_exhausted")

    monkeypatch.setattr(control, "website_webapp_request", refuse)
    with pytest.raises(ValueError, match="website_atlas_allowance_exhausted"):
        setup["run"]()
    assert len(reservations) == 1
    assert setup["fake"].posts == setup["fake"].reservations == 0
    assert not (setup["task_root"] / "atlas_pose/submission.json").exists()


def test_selected_resume_still_obeys_source_withdrawal(selected_resume):
    def withdrawn(_stage):
        raise ValueError("website_capture_withdrawn")

    with pytest.raises(ValueError, match="website_capture_withdrawn"):
        selected_resume["run"](checkpoint=withdrawn)
    assert selected_resume["fake"].posts == selected_resume["fake"].reservations == 0


def test_initial_selection_still_requires_the_full_current_forecast(selected_resume):
    (selected_resume["root"] / "selection_binding.json").unlink()
    with pytest.raises(ValueError, match="atlas_runtime_budget_insufficient"):
        selected_resume["run"]()
    assert selected_resume["fake"].posts == selected_resume["fake"].reservations == 0
