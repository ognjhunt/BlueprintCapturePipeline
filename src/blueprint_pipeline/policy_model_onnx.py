"""ADP-011: approved ONNX state/action runner for an isolated policy worker.

This is a deliberately bounded compatibility profile, not arbitrary ONNX or
native VLA compatibility. Only the dedicated worker imports the model. The
WebApp/control plane stores bytes and never loads customer inference graphs.
"""

from __future__ import annotations

import hashlib
import math
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .controlled_policy_actions import validate_action_response
from .company_policy_container_contract_v2 import validate_company_policy_container_contract_v2

PROFILE = "onnx_state_mlp_cpu_v1"
ONNX_VERSION = "1.23.0"
ORT_VERSION = "1.30.0"
MAX_MODEL_BYTES = 16 * 1024 * 1024
ALLOWED_OPERATORS = frozenset({"Identity", "Gemm", "MatMul", "Add", "Sub", "Mul", "Relu", "Tanh", "Sigmoid", "Reshape", "Clip"})


def validate_model_task_binding(artifact: Mapping[str, Any], contract: Mapping[str, Any]) -> None:
    """The frozen task owns units and action meaning; uploaded metadata cannot replace it."""
    trusted = validate_company_policy_container_contract_v2(contract)
    interface = artifact.get("interface", {})
    if any(len(field["shape"]) != 1 for field in trusted["observation_schema"]["state_fields"]):
        raise ValueError("policy_model_frozen_task_state_shape_unsupported")
    expected_fields = [
        {"name": field["name"], "width": math.prod(field["shape"]), "unit": field["unit"]}
        for field in trusted["observation_schema"]["state_fields"]
    ]
    expected_actions = {
        "chunk_rows": trusted["action_schema"]["chunk_rows"],
        "channels": [{key: channel[key] for key in ("name", "raw_accepted_bounds", "unit")}
                     for channel in trusted["action_schema"]["channels"]],
    }
    if (interface.get("state_fields") != expected_fields
            or interface.get("action_schema") != expected_actions
            or interface.get("preprocessing") != "embedded_in_model_graph"):
        raise ValueError("policy_model_frozen_task_interface_mismatch")


class OnnxStatePolicy:
    """One pinned, static float32 state vector to a fixed action chunk."""

    def __init__(self, *, model_path: Path, artifact: Mapping[str, Any]):
        if not (3, 11) <= sys.version_info[:2] < (3, 13):
            raise ValueError("policy_model_runner_python_version_unsupported")
        import onnx
        import onnxruntime as ort

        if onnx.__version__ != ONNX_VERSION or ort.__version__ != ORT_VERSION:
            raise ValueError("policy_model_runner_version_mismatch")
        interface = artifact.get("interface", {})
        if artifact.get("schema_version") != "blueprint.policy_model_artifact.v1" or interface.get("runner_profile") != PROFILE:
            raise ValueError("policy_model_runner_profile_unsupported")
        if (interface.get("schema_version") != "blueprint.policy_model_interface.v1"
                or interface.get("preprocessing") != "embedded_in_model_graph"):
            raise ValueError("policy_model_preprocessing_interface_invalid")
        if model_path.is_symlink() or not model_path.is_file() or not 0 < model_path.stat().st_size <= MAX_MODEL_BYTES:
            raise ValueError("policy_model_file_invalid")
        raw = model_path.read_bytes()
        digest = "sha256:" + hashlib.sha256(raw).hexdigest()
        if digest != artifact.get("sha256") or len(raw) != artifact.get("size_bytes"):
            raise ValueError("policy_model_content_digest_mismatch")
        fields = interface.get("state_fields")
        if not isinstance(fields, list) or not 1 <= len(fields) <= 32:
            raise ValueError("policy_model_state_interface_invalid")
        names = [field.get("name") for field in fields]
        widths = [field.get("width") for field in fields]
        if any(not isinstance(name, str) or not name for name in names) or len(set(names)) != len(names):
            raise ValueError("policy_model_state_interface_invalid")
        if any(not isinstance(field.get("unit"), str) or not field["unit"].strip() for field in fields):
            raise ValueError("policy_model_state_unit_required")
        if any(isinstance(width, bool) or not isinstance(width, int) or not 1 <= width <= 1024 for width in widths) or sum(widths) > 1024:
            raise ValueError("policy_model_state_interface_invalid")
        self.fields = fields
        self.action_schema = interface.get("action_schema", {})
        rows = self.action_schema.get("chunk_rows")
        channels = self.action_schema.get("channels", [])
        if isinstance(rows, bool) or not isinstance(rows, int) or not 1 <= rows <= 128 or not 1 <= len(channels) <= 128:
            raise ValueError("policy_model_action_interface_invalid")
        # Validate action bounds before constructing an inference session.
        validate_action_response({"actions": [[channel["raw_accepted_bounds"][0] for channel in channels]] * rows}, action_schema=self.action_schema)
        model = onnx.load_model_from_string(raw)  # never resolves external files
        if model.functions or model.training_info or model.graph.sparse_initializer or model.ir_version > 10:
            raise ValueError("policy_model_graph_profile_unsupported")
        if len(model.opset_import) != 1 or model.opset_import[0].domain not in ("", "ai.onnx") or not 7 <= model.opset_import[0].version <= 17:
            raise ValueError("policy_model_opset_unsupported")
        if not 1 <= len(model.graph.node) <= 1024:
            raise ValueError("policy_model_graph_size_invalid")
        for tensor in model.graph.initializer:
            if tensor.external_data or tensor.data_location == onnx.TensorProto.EXTERNAL or math.prod(tensor.dims) > 4_000_000:
                raise ValueError("policy_model_external_or_oversized_tensor_forbidden")
        for node in model.graph.node:
            if node.domain not in ("", "ai.onnx") or node.op_type not in ALLOWED_OPERATORS or any(a.type in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS, onnx.AttributeProto.TENSOR, onnx.AttributeProto.TENSORS) for a in node.attribute):
                raise ValueError("policy_model_operator_unsupported")
        onnx.checker.check_model(model)
        model = onnx.shape_inference.infer_shapes(model, strict_mode=True)
        for value in [*model.graph.input, *model.graph.output, *model.graph.value_info]:
            shape = [dim.dim_value for dim in value.type.tensor_type.shape.dim]
            if not shape or any(dim < 1 for dim in shape) or math.prod(shape) > 262_144:
                raise ValueError("policy_model_dynamic_or_oversized_shape_forbidden")
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        self.session = ort.InferenceSession(raw, sess_options=options, providers=["CPUExecutionProvider"])
        inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
        if len(inputs) != 1 or len(outputs) != 1:
            raise ValueError("policy_model_io_count_invalid")
        if inputs[0].name != interface.get("input_name") or inputs[0].shape != [1, sum(widths)] or inputs[0].type != "tensor(float)":
            raise ValueError("policy_model_input_interface_mismatch")
        if outputs[0].name != interface.get("output_name") or outputs[0].shape != [rows, len(channels)] or outputs[0].type != "tensor(float)":
            raise ValueError("policy_model_output_interface_mismatch")
        self.input_name, self.output_name = inputs[0].name, outputs[0].name
        self.model_sha256 = digest

    def infer(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        state = observation.get("state", {})
        vector: list[float] = []
        for field in self.fields:
            values = state.get(field["name"])
            if not isinstance(values, list) or len(values) != field["width"] or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in values):
                raise ValueError("policy_model_observation_state_invalid")
            vector.extend(values)
        tensor = np.asarray([vector], dtype=np.float32)
        if not np.isfinite(tensor).all():
            raise ValueError("policy_model_observation_float32_overflow")
        result = self.session.run([self.output_name], {self.input_name: tensor})[0]
        return validate_action_response({"actions": result.tolist()}, action_schema=self.action_schema)
