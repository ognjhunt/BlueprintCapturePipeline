"""A newly frozen diagnostic rule must admit only witnessed candidate rejections."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import shutil

import numpy as np
import pytest

from blueprint_pipeline import native_task_arena_policy_canary_worker as worker
from blueprint_pipeline import policy_scientific_reset as reset_module
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.groot_n17_droid_policy_runtime import GrootN17DroidPolicyClient, GrootN17DroidPolicySpec
from blueprint_pipeline.groot_n17_wire_client import encode_wire_message, decode_wire_message
from blueprint_pipeline.native_policy_canary_diagnostic_continuation import (
    GATE_FILENAME, PROTOCOL_KEY, assess_diagnostic_first_cell, bind_diagnostic_continuation_protocol,
    validate_diagnostic_continuation_protocol,
)
from blueprint_pipeline.native_task_arena_policy_canary_session import validate_runtime_input_manifest
from blueprint_pipeline.openpi_droid_policy_runtime import OpenPIWebsocketDroidPolicyClient, OpenPIDroidPolicySpec
from blueprint_pipeline.policy_request_evidence import capture_request, restore_request
from tests.test_native_task_arena_policy_canary_lifecycle_rehearsal import (
    FakeIsaac, _OpenPIVendor, _GrootVendor, _rehearsal_runtime, _run_cell_in_process,
)
from tests.test_policy_canary_interrupted_cell_recovery import _stage


def _read(path):
    return json.loads(path.read_text())


def _write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _setup(tmp_path, *, opt_in=True):
    runtime, child, inputs, _task_digest = _stage(tmp_path)
    # Use the existing camera-start fixture, translated onto the lifecycle
    # fixture's task; keep the production final-camera admission predicate.
    from tests.test_native_task_camera_start_configuration import fixture as camera_fixture
    camera = camera_fixture()
    plan_path = runtime / "native_task_packet/native_task_arena_scene_plan.v1.json"
    plan = _read(plan_path)
    delta = np.asarray(plan["task_spec"]["start_pose_world"][:3]) - np.asarray(camera["task_spec"]["start_pose_world"][:3])
    plan["robot"] = camera["robot"]
    plan["robot"]["base_pose_world"]["position_world_m"] = (np.asarray(plan["robot"]["base_pose_world"]["position_world_m"]) + delta).tolist()
    plan["cameras"] = camera["cameras"]
    for view in plan["cameras"]:
        view["intrinsics"]["cx"] = (view["intrinsics"]["width"] - 1) / 2
        view["intrinsics"]["cy"] = (view["intrinsics"]["height"] - 1) / 2
        if view["pose_frame"] == "world":
            matrix = np.asarray(view["frame_from_camera_matrix"]).reshape(4, 4)
            matrix[:3, 3] += delta
            view["frame_from_camera_matrix"] = matrix.reshape(-1).tolist()
    binding = camera["policy_canary_camera_start_configuration"]
    binding["robot_base_pose_world"] = deepcopy(plan["robot"]["base_pose_world"])
    reference = binding["native_reference"]
    reference["robot_base_pose_world"]["position_world_m"] = (np.asarray(reference["robot_base_pose_world"]["position_world_m"]) + delta).tolist()
    matrix = np.asarray(reference["world_from_wrist_camera_opengl"])
    matrix[:3, 3] += delta
    reference["world_from_wrist_camera_opengl"] = matrix.tolist()
    binding["task_success_contract_digest"] = inputs["task_success_contract_digest"]
    binding["camera_plan_digest"] = canonical_digest({"cameras": plan["cameras"]})
    binding["configuration_digest"] = canonical_digest(binding, digest_field="configuration_digest")
    plan["policy_canary_camera_start_configuration"] = binding
    plan["task_spec"]["target_position_world_m"] = plan["task_spec"]["destination_position_world_m"]
    plan["plan_digest"] = canonical_digest(plan, digest_field="plan_digest")
    _write(plan_path, plan)
    omission = {"schema_version": "task_evaluation_diagnostic_control_omission_authority.v1",
        "run_kind": "internal_policy_canary", "claim_ceiling": "diagnostic_policy_execution",
        "authorized_by": "fixture_owner", "authorization_reference": "explicit_fixture_request",
        "omitted_controls": ["zero_action_negative", "deterministic_scripted_positive"],
        "source_task_success_contract_digest": "sha256:" + "a" * 64,
        "result_task_success_contract_digest": inputs["task_success_contract_digest"],
        "task_scoring_criteria_changed": False, "qualified_comparison_permitted": False}
    omission["authority_digest"] = canonical_digest(omission, digest_field="authority_digest")
    for cell in inputs["cells"]:
        cell["control_diagnostic"] = {"mode": "nonblocking_omitted_by_user", "typed_gap": "controls_omitted_by_user_request",
            "policy_execution_blocked": False, "omission_authority": omission}
    inputs["runtime_inputs_digest"] = canonical_digest(inputs, digest_field="runtime_inputs_digest")
    if opt_in:
        inputs = bind_diagnostic_continuation_protocol(inputs)
    input_path = runtime / "runtime_inputs/policy_canary_runtime_inputs.json"
    _write(input_path, inputs)
    authority_path = runtime / "runtime_inputs/policy_canary_session_authority.json"
    authority = _read(authority_path)
    authority["runtime_inputs_digest"] = inputs["runtime_inputs_digest"]
    authority["runtime_inputs"].update(size_bytes=input_path.stat().st_size,
        sha256="sha256:" + hashlib.sha256(input_path.read_bytes()).hexdigest())
    authority["authority_digest"] = canonical_digest(authority, digest_field="authority_digest")
    _write(authority_path, authority)
    manifest_path = runtime / "adp_arena_provider_manifest.json"
    manifest = _read(manifest_path)
    manifest["arena_scene_plan_digest"] = plan["plan_digest"]
    manifest.update(runtime_inputs_digest=inputs["runtime_inputs_digest"], authority_digest=authority["authority_digest"])
    manifest["input_digest"] = canonical_digest(manifest, digest_field="input_digest")
    _write(manifest_path, manifest)
    (runtime / "native_task_packet/native_task_arena_packet_request.v1.json").write_text("{}")
    return runtime, child, inputs


def _runtime_factory(*, failure="bounds"):
    def factory(isaac):
        runtime = _rehearsal_runtime(isaac)
        counter = {"queries": 0}

        class Wire:
            def __init__(self): self.response = b""
            def send(self, _message): pass
            def recv(self):
                if failure == "transport" and counter["queries"] == 3:
                    raise TimeoutError("fixture transport response unproven")
                return self.response
            def close(self): pass

        class OpenPIWire(_OpenPIVendor):
            def __init__(self, spec):
                super().__init__(spec)
                self._transport = Wire()
                self._ws = self._transport
            def infer(self, observation):
                self._ws.send(encode_wire_message(observation))
                response = super().infer(observation)
                counter["queries"] += 1
                if counter["queries"] == 3:
                    response["actions"][0, 3] = 100.0
                self._transport.response = encode_wire_message(response)
                return decode_wire_message(self._ws.recv())

        class GrootWire(_GrootVendor):
            def bind_request_evidence_sink(self, sink): self.sink = sink
            def get_action(self, request):
                wire = encode_wire_message(request)
                self.sink(capture_request(request, transport="groot_zmq",
                    scientific_wire_bytes=wire, decoded_wire_request=decode_wire_message(wire)))
                return super().get_action(request)

        def client(spec, *, groot_worker_identity_receipt=None):
            endpoint = spec["policy_endpoint"]
            if spec["candidate_id"] == "pi05_droid":
                policy_spec = OpenPIDroidPolicySpec(**spec["policy_spec"])
                return OpenPIWebsocketDroidPolicyClient(spec=policy_spec, host=endpoint["host"], port=endpoint["port"],
                    client_factory=lambda **kwargs: OpenPIWire(policy_spec), wire_decoder=decode_wire_message)
            return GrootN17DroidPolicyClient(spec=GrootN17DroidPolicySpec(**spec["policy_spec"]),
                worker_identity_receipt=groot_worker_identity_receipt, host=endpoint["host"], port=endpoint["port"],
                client_factory=lambda **kwargs: GrootWire())
        return worker.CellRuntime(**{**runtime.__dict__, "policy_client": client})
    return factory


def _complete_fixture_native_channels(monkeypatch):
    original = reset_module.read_native_reset_channels
    def read(built, environment):
        observed = original(built, environment)
        for name in reset_module.REQUIRED_CHANNELS:
            observed["observed"].setdefault(name, {"fixture_native_channel": name})
            observed["sources"].setdefault(name, "hermetic_native_boundary_fixture")
        observed["observed"]["robot"]["joint_limits"] = [environment.joint_limits()]
        observed["gaps"] = []
        return observed
    monkeypatch.setattr(reset_module, "read_native_reset_channels", read)


@pytest.fixture(scope="module")
def retained_pair(tmp_path_factory):
    root = tmp_path_factory.mktemp("diagnostic-continuation")
    runtime, child, _inputs = _setup(root)
    with pytest.MonkeyPatch.context() as monkeypatch:
        _complete_fixture_native_channels(monkeypatch)
        isaac = FakeIsaac(child / worker.PROVIDER_RESULT_FILENAME)
        with pytest.raises(SystemExit):
            worker._run_selected_cell(0, runtime_root=runtime, output_root=child,
                provider_output_root=child.parent.parent, cell_runtime=_runtime_factory()(isaac))
    result = _read(child / worker.PROVIDER_RESULT_FILENAME)
    assert result["episodes"][0].get("failure_type") == "DroidActionExecutionError", [
        (row.get("candidate_id"), row.get("failure_type"), row.get("failure_message")) for row in result["episodes"]]
    assert result["episodes"][1]["status"] == "completed", result["episodes"][1].get("failure_message")
    return root


def _copy_pair(tmp_path, retained_pair):
    destination = tmp_path / "copy"
    shutil.copytree(retained_pair, destination)
    return destination / "provider_runtime", destination / "runtime_output/cell_runs/00"


def _reseal_failure(child, mutate):
    path = next((child / "episodes").glob("*--pi05_droid.failure_evidence.json"))
    value = _read(path)
    mutate(value)
    # Refresh source-file descriptors after deliberate fixture mutations, so the
    # tests reach semantic bindings rather than merely failing a file checksum.
    for rows in (value["episode"]["policy_request_artifacts"], value["episode"]["media_artifacts"]):
        for row in rows:
            source = child / "episodes" / row["relative_path"]
            row.update(size_bytes=source.stat().st_size, sha256="sha256:" + hashlib.sha256(source.read_bytes()).hexdigest())
    value["gap_digest"] = canonical_digest(value, digest_field="gap_digest")
    _write(path, value)
    result_path = child / worker.PROVIDER_RESULT_FILENAME
    result = _read(result_path)
    result["episodes"][0].update(value)
    for row in result["artifact_inventory"]:
        source = child / row["relative_path"]
        row.update(size_bytes=source.stat().st_size, sha256="sha256:" + hashlib.sha256(source.read_bytes()).hexdigest())
    result["artifact_inventory_digest"] = canonical_digest({"value": result["artifact_inventory"]})
    result["result_digest"] = canonical_digest(result, digest_field="result_digest")
    result_path.chmod(0o640)  # This private adversarial copy is deliberately resealed by the test.
    _write(result_path, result)


def test_protocol_binds_new_inputs_without_changing_originals_or_action_contract(tmp_path):
    _runtime, _child, inputs = _setup(tmp_path, opt_in=False)
    original = deepcopy(inputs)
    bound = bind_diagnostic_continuation_protocol(inputs)
    assert inputs == original
    assert bound["runtime_inputs_digest"] != original["runtime_inputs_digest"]
    assert bound[PROTOCOL_KEY]["base_runtime_inputs_digest"] == original["runtime_inputs_digest"]
    assert bound[PROTOCOL_KEY]["action_admission_changed"] is False
    assert bound["cells"] == original["cells"]
    assert bound["task_success_contract"] == original["task_success_contract"]
    assert validate_runtime_input_manifest(bound) == bound
    assert bind_diagnostic_continuation_protocol(bound) == bound


def test_verified_rejection_advances_only_with_the_unchanged_independent_witness(tmp_path, retained_pair):
    runtime, child = _copy_pair(tmp_path, retained_pair)
    gate = assess_diagnostic_first_cell(runtime_root=runtime, child_root=child)
    assert gate["status"] == "passed", gate["blockers"]
    assert gate["result_digest"] == canonical_digest(gate, digest_field="result_digest")
    rejection, witness = gate["candidate_adjudications"]
    assert rejection["classification"] == "verified_candidate_joint_bound_rejection"
    assert rejection["prior_applied_action_count"] == 16
    assert rejection["rejected_query_applied_action_count"] == 0
    assert witness["classification"] == "paired_native_witness"
    assert witness["existing_approach_threshold_changed"] is False
    child_result = _read(child / worker.PROVIDER_RESULT_FILENAME)
    assert child_result["episodes"][0]["status"] == "blocked"
    assert child_result["episodes"][0]["episode"]["score"]["status"] == "not_scored"


@pytest.mark.parametrize("missing", ["prestart_readiness", "policy_inference_evidence"])
def test_v22_style_failure_without_new_bindings_cannot_enable_continuation(tmp_path, retained_pair, missing):
    runtime, child = _copy_pair(tmp_path, retained_pair)
    _reseal_failure(child, lambda value: value.pop(missing))
    gate = assess_diagnostic_first_cell(runtime_root=runtime, child_root=child)
    assert gate["status"] == "blocked"
    assert any("readiness_missing" in reason or "response_identity_unproven" in reason for reason in gate["blockers"])


def test_individually_valid_mismatched_wire_and_png_do_not_admit_continuation(tmp_path, retained_pair):
    runtime, child = _copy_pair(tmp_path, retained_pair)
    def mutate(value):
        reference = value["episode"]["policy_request_artifacts"][-1]
        path = child / "episodes" / reference["relative_path"]
        original = _read(path)
        request = restore_request(original["request"])
        request["observation/wrist_image_left"] = np.full_like(request["observation/wrist_image_left"], 17)
        wire = encode_wire_message(request)
        changed = capture_request(request, transport=original["transport"],
            scientific_wire_bytes=wire, decoded_wire_request=decode_wire_message(wire))
        changed["episode_binding"] = original["episode_binding"]
        changed["evidence_digest"] = canonical_digest(changed, digest_field="evidence_digest")
        _write(path, changed)
    _reseal_failure(child, mutate)
    gate = assess_diagnostic_first_cell(runtime_root=runtime, child_root=child)
    assert gate["status"] == "blocked"
    assert any("policy_pixels_mismatch" in reason for reason in gate["blockers"])


@pytest.mark.parametrize("fault", ["transport", "native_reset_gap", "response_identity", "rejected_query_applied", "duplicate_prior_command"])
def test_uncertain_transport_reset_identity_or_actuation_stays_blocked(tmp_path, retained_pair, fault):
    runtime, child = _copy_pair(tmp_path, retained_pair)
    def mutate(value):
        if fault == "transport":
            value["failure_type"] = value["typed_harness_failure"] = "TimeoutError"
        elif fault == "native_reset_gap":
            reset = value["scientific_reset"]
            reset["gaps"] = ["physics:TypeError"]
            reset["complete"] = False
            reset["receipt_digest"] = canonical_digest(reset, digest_field="receipt_digest")
        elif fault == "response_identity":
            value["policy_inference_evidence"]["server_identity_sha256"] = "0" * 64
        elif fault == "duplicate_prior_command":
            value["commanded_actions"][-1] = deepcopy(value["commanded_actions"][-2])
        else:
            value["commanded_actions"][-1]["query_index"] = 2
    _reseal_failure(child, mutate)
    gate = assess_diagnostic_first_cell(runtime_root=runtime, child_root=child)
    assert gate["status"] == "blocked"


@pytest.mark.parametrize("opt_in", [False, True])
def test_real_parent_reaches_cell01_only_for_fully_verified_opt_in_rejection(tmp_path, monkeypatch, opt_in):
    runtime, child, _inputs = _setup(tmp_path, opt_in=opt_in)
    output = child.parent.parent
    child.rmdir()
    _complete_fixture_native_channels(monkeypatch)
    calls, isaacs = [], []
    class ReachedCellOne(RuntimeError):
        pass
    def spawn(**kwargs):
        calls.append(kwargs["index"])
        if kwargs["index"] == 1:
            raise ReachedCellOne("bounded test stops before cell01 execution")
        return _run_cell_in_process(**kwargs, isaacs=isaacs, runtime_factory=_runtime_factory())
    expected = ReachedCellOne if opt_in else RuntimeError
    with pytest.raises(expected, match="cell01 execution" if opt_in else "embodiment_parity_probe_failed"):
        worker._run_isolated_cell_processes(runtime_root=runtime, output_root=output, run_cell_process=spawn)
    assert calls == ([0, 1] if opt_in else [0])
    if opt_in:
        assert _read(output / GATE_FILENAME)["status"] == "passed"
    else:
        assert not (output / GATE_FILENAME).exists()


def test_required_controls_and_unknown_protocols_are_not_admitted(tmp_path):
    _runtime, _child, inputs = _setup(tmp_path)
    altered = deepcopy(inputs)
    altered["task_success_contract"]["criteria"]["controls"] = {
        "mode": "required_per_cell", "control_ids": ["zero_action_negative", "deterministic_scripted_positive"]}
    with pytest.raises(ValueError, match="controls_mode_not_admitted"):
        validate_diagnostic_continuation_protocol(altered)
    altered = deepcopy(inputs)
    altered[PROTOCOL_KEY]["mode"] = "ignore_first_cell_failure"
    with pytest.raises(ValueError, match="protocol_binding_invalid"):
        validate_diagnostic_continuation_protocol(altered)
