from __future__ import annotations

import hashlib

import numpy as np
import pytest
from blueprint_pipeline.policy_model_onnx import OnnxStatePolicy

onnx = pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

def model_artifact(tmp_path):
    from onnx import TensorProto, helper, numpy_helper
    graph = helper.make_graph([
        helper.make_node("MatMul", ["state", "weights"], ["actions"]),
    ], "blueprint_owned_linear_policy", [helper.make_tensor_value_info("state", TensorProto.FLOAT, [1, 2])],
       [helper.make_tensor_value_info("actions", TensorProto.FLOAT, [1, 2])],
       initializer=[numpy_helper.from_array(np.eye(2, dtype=np.float32), name="weights")])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=10)
    raw = model.SerializeToString()
    path = tmp_path / "policy.onnx"
    path.write_bytes(raw)
    artifact = {
        "schema_version": "blueprint.policy_model_artifact.v1",
        "sha256": "sha256:" + hashlib.sha256(raw).hexdigest(), "size_bytes": len(raw),
        "interface": {
            "runner_profile": "onnx_state_mlp_cpu_v1", "input_name": "state", "output_name": "actions",
            "state_fields": [{"name": "joint_position", "width": 2}],
            "action_schema": {"chunk_rows": 1, "channels": [
                {"name": "joint_1", "raw_accepted_bounds": [-1.0, 1.0]},
                {"name": "gripper", "raw_accepted_bounds": [0.0, 1.0]},
            ]},
        },
    }
    return path, artifact


def test_real_onnx_inference_responds_to_observed_state(tmp_path):
    path, artifact = model_artifact(tmp_path)
    policy = OnnxStatePolicy(model_path=path, artifact=artifact)
    assert policy.infer({"state": {"joint_position": [0.25, 0.5]}}) == {"actions": [[0.25, 0.5]]}
    assert policy.infer({"state": {"joint_position": [-0.5, 0.75]}}) == {"actions": [[-0.5, 0.75]]}


def test_tampered_bytes_and_incompatible_interface_refuse_before_inference(tmp_path):
    path, artifact = model_artifact(tmp_path)
    path.write_bytes(path.read_bytes() + b"x")
    with pytest.raises(ValueError, match="digest_mismatch"):
        OnnxStatePolicy(model_path=path, artifact=artifact)
    path, artifact = model_artifact(tmp_path)
    artifact["interface"]["input_name"] = "wrong"
    with pytest.raises(ValueError, match="input_interface_mismatch"):
        OnnxStatePolicy(model_path=path, artifact=artifact)


def test_nonfinite_state_and_out_of_bounds_actions_refuse(tmp_path):
    path, artifact = model_artifact(tmp_path)
    policy = OnnxStatePolicy(model_path=path, artifact=artifact)
    with pytest.raises(ValueError, match="state_invalid"):
        policy.infer({"state": {"joint_position": [float("nan"), 0.5]}})
    with pytest.raises(ValueError, match="out_of_bounds"):
        policy.infer({"state": {"joint_position": [2.0, 0.5]}})


def test_custom_operator_and_external_weights_are_refused(tmp_path):
    path, artifact = model_artifact(tmp_path)
    graph = onnx.load_model_from_string(path.read_bytes())
    graph.graph.node[0].domain = "customer.code"
    raw = graph.SerializeToString()
    path.write_bytes(raw)
    artifact.update(sha256="sha256:" + hashlib.sha256(raw).hexdigest(), size_bytes=len(raw))
    with pytest.raises(ValueError, match="operator_unsupported"):
        OnnxStatePolicy(model_path=path, artifact=artifact)
    path, artifact = model_artifact(tmp_path)
    graph = onnx.load_model_from_string(path.read_bytes())
    graph.graph.initializer[0].data_location = onnx.TensorProto.EXTERNAL
    graph.graph.initializer[0].external_data.add(key="location", value="/private/scene.usd")
    raw = graph.SerializeToString()
    path.write_bytes(raw)
    artifact.update(sha256="sha256:" + hashlib.sha256(raw).hexdigest(), size_bytes=len(raw))
    with pytest.raises(ValueError, match="external_or_oversized"):
        OnnxStatePolicy(model_path=path, artifact=artifact)
