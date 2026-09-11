"""Production phase reuse preserves completed calls, costs, and exact inputs."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from blueprint_pipeline import task_evaluation_scene_configuration_astra_driver as driver
from blueprint_pipeline import task_evaluation_scene_configuration_astra_phase_adoption as adoption
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_evaluation_supervisor.inference_reservations import InferenceReservationAudit
from blueprint_pipeline.task_object_astra_authoring import AssetAuthoringError, VisualBrief
from tests.test_task_evaluation_scene_configuration_astra_driver import retained, component  # noqa: F401


def _save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def prior_runtime(root, request, package):
    prior = root / "prior-runtime"
    _save(prior / "authoring/request.json", request)
    for name in adoption.SOURCE_ARCHIVES:
        (prior / name).write_bytes((package / name).read_bytes())
    brief = VisualBrief(object_identity="book", observed_parts=["pages"], appearance_requirements=["opaque"],
        unknown_regions=["underside"], cad_brief_markdown="Exact supplied dimensions",
        proposed_material="paper", proposed_appearance="opaque")
    _save(prior / "authoring/source_analysis.json", {"request_digest": request["request_digest"],
        "model": "gpt-6-astra", "output": brief.model_dump(mode="json")})
    identity = {"run_id": request["run_id"], "capability": request["object_id"] + "_source_analysis",
        "model": "gpt-6-astra", "input_digest": "sha256:" + "a" * 64, "max_turns": 1, "max_output_tokens": 12000}
    policy = {"policy_digest": "sha256:" + "b" * 64}
    reservation = {**identity, "schema_version": "task_evaluation_inference_reservation.v1",
        "reservation_id": canonical_digest(identity), "projected_max_cost_usd": 1.4,
        "cache_policy": policy, "cache_policy_digest": policy["policy_digest"], "breakpoint_digests": []}
    # The cache identity is optional in historical reservations; new records bind it.
    identity["cache_policy_digest"] = policy["policy_digest"]
    reservation["reservation_id"] = canonical_digest(identity)
    reservation["inference_reservation_digest"] = canonical_digest(reservation, digest_field="inference_reservation_digest")
    completion = {"schema_version": "task_evaluation_inference_completion.v1",
        "reservation_id": reservation["reservation_id"], "run_id": request["run_id"],
        "capability": identity["capability"], "model": "gpt-6-astra", "provider": "openai",
        "cache_policy": policy, "breakpoint_digests": [], "projected_max_cost_usd": 1.4,
        "reconciled_actual_cost_usd": .1, "released_reservation_usd": 1.3,
        "structured_output_digest": canonical_digest(brief.model_dump(mode="json"))}
    completion["inference_completion_digest"] = canonical_digest(completion, digest_field="inference_completion_digest")
    audit = InferenceReservationAudit(run_root=prior / "inference", run_id=request["run_id"])
    audit.record_reservation(reservation)
    audit.record_completion(completion)
    return prior, brief


@pytest.fixture
def phase_fixture(retained, component):  # noqa: F811 - imported shared pytest fixtures
    request = driver.build_authoring_request(retained.input, retained.source, [retained.image], retained.rights).model_dump(mode="json")
    package = Path(component.environment[driver._PACKAGE_ENV])
    prior, brief = prior_runtime(retained.root, request, package)
    descriptor = adoption.materialize_phase_adoption(prior_runtime=prior, phases=["source_analysis"])
    return SimpleNamespace(prior=prior, brief=brief, request=request, package=package, descriptor=descriptor,
        budget=retained.root / "new-budget", retained=retained, component=component)


def test_restores_verified_completed_phase_and_full_inference_balance(phase_fixture):
    f = phase_fixture
    before = adoption._inventory(f.prior)
    result = adoption.prepare_phase_adoption(value=f.descriptor, request_value=f.request,
        package=f.package, budget_root=f.budget)
    assert result["authoring_kwargs"]["adopted_source_analysis"] == f.brief
    assert result["authoring_kwargs"]["adoption_record"]["new_provider_call"] is False
    adopted_phase = Path(result["authoring_kwargs"]["adoption_record"]["source_phase"]["path"])
    assert adopted_phase.is_relative_to(Path(result["retained_phase_root"]))
    assert adopted_phase.read_bytes() == (f.prior / "authoring/source_analysis.json").read_bytes()
    assert result["prior_call_count"] == 1
    audit = InferenceReservationAudit(run_root=f.budget, run_id=f.request["run_id"])
    assert audit.manifest()["reserved_max_cost_usd"] == .1
    assert adoption._inventory(f.prior) == before


@pytest.mark.parametrize("defect", ["phase_bytes", "source", "cad_source", "unknown_call", "run", "descriptor"])
def test_changed_or_unresolved_evidence_refuses_before_new_inference(phase_fixture, defect):
    f = phase_fixture
    if defect == "phase_bytes":
        (f.prior / "authoring/source_analysis.json").write_text("{}")
    elif defect == "source":
        f.request["owner_description"] = "Different object"
    elif defect == "cad_source":
        (f.package / adoption.SOURCE_ARCHIVES[0]).write_bytes(b"changed")
    elif defect == "unknown_call":
        next((f.prior / "inference/inference_reservations/completed").glob("*.json")).unlink()
        f.descriptor = adoption.materialize_phase_adoption(prior_runtime=f.prior, phases=["source_analysis"])
    elif defect == "run":
        f.request["run_id"] = "another-run"
    else:
        f.descriptor["adoption_digest"] = "sha256:" + "0" * 64
    with pytest.raises(AssetAuthoringError):
        adoption.prepare_phase_adoption(value=f.descriptor, request_value=f.request,
            package=f.package, budget_root=f.budget)
    assert not f.budget.exists()


def test_driver_supplies_verified_adoption_before_parent_gate(phase_fixture):
    f = phase_fixture
    value = json.loads(Path(f.component.environment[driver._INPUT_ENV]).read_text())
    value["configuration"]["astra_phase_adoption"] = f.descriptor
    _save(Path(f.component.environment[driver._INPUT_ENV]), value)
    original = f.component.kwargs["authoring_executor"]
    def authoring(**kwargs):
        assert kwargs["adopted_source_analysis"] == f.brief
        assert kwargs["invoker"].prior_calls == 1
        return original(**kwargs)
    driver.execute_astra_component(**{**f.component.kwargs, "authoring_executor": authoring})
    assert f.component.events.index("reserve") < f.component.events.index("sdk")


def test_prior_calls_consume_the_existing_request_ceiling():
    invoker = driver._StageInvoker(SimpleNamespace(invoke=lambda *args: pytest.fail("no new call allowed")),
                                   "same-run", 1, prior_calls=1)
    spec = SimpleNamespace(run_id="same-run", model="gpt-6-astra", max_turns=1, tool_bindings=(),
        max_output_tokens=12000, max_input_tokens=80000, reasoning_effort="high")
    with pytest.raises(driver.AstraStageError, match="inference_boundary_refused"):
        invoker.invoke(spec, "input")


def test_cad_or_later_phase_requires_the_completed_prefix(phase_fixture):
    with pytest.raises(AssetAuthoringError, match="descriptor_invalid"):
        adoption.materialize_phase_adoption(prior_runtime=phase_fixture.prior, phases=["cad_state"])
