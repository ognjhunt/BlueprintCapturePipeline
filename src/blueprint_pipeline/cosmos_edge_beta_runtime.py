"""Pinned Cosmos Edge DROID execution through the ordinary beta simulator.

The historical WAM client retains a 16-row conditioning prefix. Simulator
execution instead retains all 32 native rows and executes eight at 15 Hz. The
three real policy cameras and the existing 224-pixel transport preprocessing
remain explicit; NVIDIA's pinned server composes and resizes them to 540x640.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

from .cosmos_edge_droid_policy_runtime import (
    COSMOS_FRAMEWORK_REVISION, MODEL_ID, MODEL_REVISION,
    CosmosEdgeDroidPolicySpec, canonical_sha256, validate_server_metadata,
)
from .decision_evidence_contracts import canonical_digest

CANDIDATE_ID = "cosmos3_edge_policy_droid"
INTERFACE_ID = "cosmos_edge_droid.v1"
CHECKPOINT_DIGEST = "sha256:f1da4f2c70e31b7c80c0069abdebf5d46d9a45a2e588b3aa84984a862f18568f"
SNAPSHOT_MANIFEST_SHA256 = "766c33e99945394406986495d3263596953ed8d696db29a63e91e41fae5d4c64"
SNAPSHOT_STAGED_NAME = "cosmos_edge_policy_snapshot_manifest.json"
MINIMUM_GPU_MEMORY_GB = 48  # Conservative policy+Isaac allocation floor, not a measured model minimum.


def snapshot_manifest_path() -> Path:
    return (Path(__file__).resolve().parents[2] / "docs/experiments/"
            "policy_ranking_cosmos3_edge_closed_loop_20260729/policy_snapshot_manifest_v1.json")


def verified_snapshot_manifest(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    body = {key: val for key, val in value.items() if key != "manifest_sha256"}
    if (value.get("manifest_sha256") != SNAPSHOT_MANIFEST_SHA256
            or canonical_sha256(body) != SNAPSHOT_MANIFEST_SHA256
            or value.get("model_id") != MODEL_ID or value.get("model_revision") != MODEL_REVISION):
        raise ValueError("cosmos_edge_beta_snapshot_manifest_mismatch")
    return value


def interface_contract() -> dict[str, Any]:
    return {
        "policy_interface_id": INTERFACE_ID,
        "robot_profile_id": "droid_franka_panda_robotiq_2f85_v1",
        "observation_profile_id": "droid_three_camera_robot_state_v1",
        "action_schema_id": "droid_absolute_joint_position_v1",
        "control_hz": 15, "action_horizon": 32, "execution_horizon": 8,
        "camera_roles": ["external", "wrist", "exterior_right"],
        "image_width": 224, "image_height": 224, "image_encoding": "rgb_uint8",
        "image_preprocessing": "resize_with_padding", "history_frames": 1,
        "normalization_digest": None,
    }


def candidate_execution_spec(*, scene_plan: Mapping[str, Any], contract: Mapping[str, Any],
                             source_commit: str) -> dict[str, Any]:
    from .adp009d_policy_episode import maximum_policy_queries_for_task_spec
    from .droid_policy_canary_embodiment import concrete_droid_task_instruction
    from .native_task_isaaclab_launch import NATIVE_TASK_ARENA_IMAGE

    roles = {row.get("role") for row in scene_plan.get("cameras", []) if row.get("policy_input") is True}
    if not set(interface_contract()["camera_roles"]).issubset(roles):
        raise ValueError("cosmos_edge_required_policy_camera_unavailable")
    verified_snapshot_manifest(snapshot_manifest_path())
    policy = CosmosEdgeDroidPolicySpec(snapshot_manifest_sha256=SNAPSHOT_MANIFEST_SHA256)
    policy.validate()
    from .cosmos_droid_bundle import runtime_contract
    runtime = {
        "provisioning_identity": runtime_contract(CANDIDATE_ID),
        "source_commit": source_commit, "container_image": NATIVE_TASK_ARENA_IMAGE,
        "candidate_source": {"repository": "https://github.com/NVIDIA/cosmos-framework",
                             "revision": COSMOS_FRAMEWORK_REVISION},
        "policy_identity": policy.server_metadata(), "checkpoint_inventory_digest": CHECKPOINT_DIGEST,
        "observation_adapter": interface_contract(), "action_adapter": "cosmos_native32_execute8_joint_position",
    }
    rights = {"scene_id": scene_plan["scene_id"], "task_id": scene_plan["task_id"],
              "scene_plan_digest": scene_plan["plan_digest"], "source_license": "OpenMDW-1.1",
              "checkpoint_provider_use_status": "allowed", "checkpoint_redistribution_status": "allowed",
              "rights_ready": True, "secret_material_recorded": False}
    rights["rights_receipt_digest"] = canonical_digest(rights, digest_field="rights_receipt_digest")
    spec = {
        "schema_version": "native_task_arena_policy_canary_execution_spec.v1",
        "candidate_id": CANDIDATE_ID, "policy_interface_id": INTERFACE_ID,
        "evaluation_backend": "robolab.v1", "execution_authority": "internal_policy_canary_unqualified",
        "claim_ceiling": "diagnostic_policy_execution", "scene_id": scene_plan["scene_id"],
        "task_id": scene_plan["task_id"], "scene_plan_digest": scene_plan["plan_digest"],
        "prompt": concrete_droid_task_instruction({**scene_plan["task_spec"], "task_id": scene_plan["task_id"]}),
        "policy_endpoint": {"host": "127.0.0.1", "port": 8000}, "policy_spec": asdict(policy),
        "candidate_rights_binding": rights, "checkpoint_digest": CHECKPOINT_DIGEST,
        "runtime_identity": runtime, "runtime_identity_digest": canonical_digest(runtime),
        "worker_identity_requirement": "none", "require_observed_eef_support": False,
        "max_policy_queries": maximum_policy_queries_for_task_spec(scene_plan["task_spec"], open_loop_horizon=8),
        "open_loop_horizon": 8, "ranking_permitted": False, "qualification_permitted": False,
        "scene_promotion_permitted": False, "scoring_authority": "deterministic_simulator_state",
        "task_success_contract": dict(contract), "task_success_contract_digest": contract["contract_digest"],
    }
    spec["execution_spec_digest"] = canonical_digest(spec, digest_field="execution_spec_digest")
    validate_execution_spec(spec)
    return spec


def validate_execution_spec(spec: Mapping[str, Any]) -> None:
    policy = CosmosEdgeDroidPolicySpec(**dict(spec["policy_spec"]))
    policy.validate()
    from .cosmos_droid_bundle import runtime_contract
    expected = CosmosEdgeDroidPolicySpec(snapshot_manifest_sha256=SNAPSHOT_MANIFEST_SHA256)
    if (spec.get("runtime_identity", {}).get("provisioning_identity") != runtime_contract(CANDIDATE_ID)
            or spec.get("candidate_id") != CANDIDATE_ID or spec.get("policy_interface_id") != INTERFACE_ID
            or spec.get("evaluation_backend") != "robolab.v1"
            or spec.get("checkpoint_digest") != CHECKPOINT_DIGEST
            or policy.snapshot_manifest_sha256 != SNAPSHOT_MANIFEST_SHA256
            or policy.server_metadata() != expected.server_metadata()
            or spec.get("policy_endpoint") != {"host": "127.0.0.1", "port": 8000}
            or spec.get("open_loop_horizon") != 8
            or spec.get("execution_spec_digest") != canonical_digest(spec, digest_field="execution_spec_digest")):
        raise ValueError("cosmos_edge_beta_execution_binding_invalid")


class CosmosEdgeSimulatorClient:
    """Use the verified native output, without treating predicted state as observation."""

    spec_type = CosmosEdgeDroidPolicySpec
    learned_policy = True
    policy_id = MODEL_ID
    action_space = "joint_position"
    action_chunk_rows = 32
    open_loop_horizon = 8

    def __init__(self, *, policy_spec: Mapping[str, Any], host: str, port: int, client_factory=None,
                 startup_timeout_seconds: float | None = None):
        # Only the pinned protocol wheels, staged beside the provider package;
        # never install a policy model or another NumPy into Isaac's interpreter.
        wheels = Path(__file__).resolve().parent.parent / "cosmos_policy_client"
        if client_factory is None and wheels.is_dir():
            from .cosmos_droid_client_dependencies import activate_client_dependencies
            activate_client_dependencies(wheels.parent)
        self._spec = self.spec_type(**dict(policy_spec))
        self._spec.validate()
        if host not in {"127.0.0.1", "localhost", "::1"} or port != 8000:
            raise ValueError("cosmos_edge_beta_endpoint_must_be_local")
        self._wire_decoder = None
        if client_factory is None:
            from openpi_client import msgpack_numpy, websocket_client_policy
            from .openpi_droid_policy_runtime import _bounded_openpi_client_type
            self._wire_decoder = msgpack_numpy.unpackb
            client_factory = _bounded_openpi_client_type(websocket_client_policy.WebsocketClientPolicy,
                                                         self._wire_decoder,
                **({"startup_timeout_seconds": startup_timeout_seconds} if startup_timeout_seconds is not None else {}))
        self._factory, self._host, self._port = client_factory, host, port
        self.required_policy_views = self._spec.required_policy_views
        self._client = None
        self._metadata = None
        self._inference_evidence = None
        self._request_evidence = None
        self._request_sink = None
        self._request_count = 0
        self.candidate_policy_queried = False

    def _open(self):
        self._client = self._factory(host=self._host, port=self._port, api_key=None)
        try:
            self._metadata = validate_server_metadata(self._client.get_server_metadata(), expected=self._spec)
        except BaseException:
            self.close()
            raise
        return self._client

    def bind_request_evidence_sink(self, sink):
        self._request_sink = sink

    def _retain_request(self, value):
        self._request_evidence = dict(value)
        if self._request_sink is not None:
            self._request_sink(value)

    def last_request_evidence(self):
        if self._request_evidence is None:
            raise ValueError("cosmos_edge_policy_request_evidence_missing")
        return json.loads(json.dumps(self._request_evidence))

    def infer(self, observation: Mapping[str, Any]) -> Any:
        import numpy as np
        from .adp009d_droid_action_execution import (
            BLOCKER_CHUNK_NONFINITE, BLOCKER_CHUNK_SHAPE, DroidActionExecutionError,
        )
        from .droid_policy_bridge import validate_droid_observation
        from .openpi_droid_policy_runtime import _action_payload_returned, _json_safe_vendor_response
        from .policy_request_evidence import ObservedWebsocket, capture_request

        self._inference_evidence = self._request_evidence = None
        self.candidate_policy_queried = False
        errors = validate_droid_observation(observation, required_views=self.required_policy_views)
        if errors:
            raise ValueError("cosmos_edge_policy_observation_invalid:" + errors[0])
        request = {key: observation[key] for key in (*self.required_policy_views,
                   "observation/joint_position", "observation/gripper_position", "prompt")}
        client = self._open()
        try:
            websocket = getattr(client, "_ws", None)
            if websocket is not None and self._wire_decoder is not None:
                client._ws = ObservedWebsocket(websocket, request=request, decoder=self._wire_decoder,
                                               sink=self._retain_request)
            else:
                self._retain_request(capture_request(request, transport="cosmos_injected_transport"))
            response = client.infer(request)
            raw = _json_safe_vendor_response(response)
            self.candidate_policy_queried = True
            self._request_count += 1
            self._inference_evidence = {
                "server_response_received": True, "server_identity_sha256": self._metadata["identity_sha256"],
                "wire_response_type": type(response).__name__, "raw_vendor_action_response": raw,
                "raw_vendor_action_response_digest": canonical_digest({"raw_vendor_action_response": raw}),
                "raw_vendor_action_response_role": "genuine_decoded_vendor_wire_response_before_candidate_normalization",
                "action_payload_returned": _action_payload_returned(response), "actions_extracted": False,
            }
            value = response.get("action", response.get("actions")) if isinstance(response, Mapping) else None
            try:
                actions = np.asarray(value, dtype=float)
            except (TypeError, ValueError) as exc:
                raise DroidActionExecutionError([BLOCKER_CHUNK_SHAPE]) from exc
            if actions.shape != (32, 8):
                raise DroidActionExecutionError([BLOCKER_CHUNK_SHAPE])
            if not np.isfinite(actions).all():
                raise DroidActionExecutionError([BLOCKER_CHUNK_NONFINITE])
            self._inference_evidence.update(actions_extracted=True, action_chunk_shape=[32, 8])
            return actions
        finally:
            self.close()

    def last_inference_evidence(self):
        if self._inference_evidence is None:
            raise ValueError("cosmos_edge_policy_inference_evidence_missing")
        return json.loads(json.dumps(self._inference_evidence, allow_nan=False))

    def preflight_readiness(self):
        prior = self.candidate_policy_queried
        try:
            self._open()
        finally:
            self.close()
        self.candidate_policy_queried = False
        self._inference_evidence = self._request_evidence = None
        return {"identity_verified": True, "server_metadata": self._metadata,
                "candidate_policy_queried": False, "candidate_inference_performed": False,
                "policy_state_advanced": False, "last_inference_evidence": None,
                "prior_candidate_policy_query_observed": prior,
                "readiness_method": "fresh_identity_handshake_without_inference"}

    def evidence_summary(self) -> dict[str, Any]:
        if self._metadata is None:
            self.preflight_readiness()
        return {"transport": "openpi_websocket_msgpack_numpy", "identity_verified": True,
                "server_metadata": self._metadata, "request_count": self._request_count,
                "last_inference_evidence": self._inference_evidence,
                "simulator_action_shape": [32, 8],
                "executed_prefix_steps": 8, "measured_state_required_each_query": True,
                "policy_history": "one_current_observation_deterministic_seed_no_episode_cache"}

    def close(self) -> None:
        client, self._client = self._client, None
        close = getattr(client, "close", None) or getattr(getattr(client, "_ws", None), "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass  # Teardown must not erase a returned action or its original failure.
