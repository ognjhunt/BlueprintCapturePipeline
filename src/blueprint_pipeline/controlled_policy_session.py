"""ADP-050/day 28: simulator-side execution for remote policies and adapters.

The environment and transport are Blueprint-owned adapters, never imports or
commands selected by a customer. Models and controller containers use the
qualified sandbox's Unix proxy; customer-hosted policies use approved HTTPS.
"""
from __future__ import annotations

import hashlib
import json
import secrets
import time
from typing import Any, Callable, Mapping, Protocol
from urllib.parse import urlsplit

from .company_policy_container_contract_v2 import validate_company_policy_container_contract_v2
from .controlled_policy_actions import validate_action_response
from .controlled_policy_observations import project_controlled_observation
from .core.security_controls import fetch_bounded_https


class ControlledEnvironment(Protocol):
    def read_policy_inputs(self) -> Mapping[str, Any]:
        """Return camera_pngs (bytes), robot_state (numbers), and prompt."""

    def apply_action_chunk(self, actions: list[list[float]], *, action_schema: Mapping[str, Any]) -> Mapping[str, Any]:
        """Translate the approved channels and independently observe execution."""

    def is_terminal(self) -> bool: ...

    def stop(self) -> None:
        """Stop the virtual controller after success, failure or timeout."""


class ControlledPolicyClient:
    def __init__(self, *, contract: Mapping[str, Any],
                 transport: Callable[[bytes, float], bytes], transport_name: str):
        self.contract = validate_company_policy_container_contract_v2(contract)
        if transport_name not in {"approved_https", "qualified_sandbox_unix_proxy"}:
            raise ValueError("controlled_policy_transport_invalid")
        self.transport = transport
        self.transport_name = transport_name
        self.request_sink: Callable[[Mapping[str, Any]], None] | None = None
        self.last_receipt: dict[str, Any] = {}

    def bind_request_evidence_sink(self, sink: Callable[[Mapping[str, Any]], None]) -> None:
        self.request_sink = sink

    def infer(self, observation: Mapping[str, Any]) -> list[list[float]]:
        wire = project_controlled_observation(
            contract=self.contract, request_id=secrets.token_hex(24),
            prompt=observation["prompt"], camera_pngs=observation["camera_pngs"],
            robot_state=observation["robot_state"], synthetic=observation.get("synthetic", False),
        )
        body = json.dumps(wire, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        # Retain the exact approved request before transport, including failed calls.
        if self.request_sink is None:
            raise ValueError("controlled_policy_evidence_sink_required")
        self.request_sink(wire)
        self.last_receipt = {"request_id": wire["request_id"], "transport": self.transport_name,
            "observation_sha256": "sha256:" + hashlib.sha256(body).hexdigest(),
            "scene_files_exported": False, "scoring_harness_exported": False,
            "action_payload_returned": False}
        timeout = min(30, self.contract["container"]["resources"]["request_timeout_ms"] / 1000)
        response = self.transport(body, timeout)
        if not isinstance(response, bytes) or len(response) > 65_536:
            raise ValueError("controlled_policy_response_size_invalid")
        def reject_constant(_value: str) -> None:
            raise ValueError("controlled_policy_response_nonfinite")
        actions = validate_action_response(json.loads(response, parse_constant=reject_constant),
                                           action_schema=self.contract["action_schema"])
        self.last_receipt.update(action_payload_returned=True,
            action_sha256="sha256:" + hashlib.sha256(response).hexdigest())
        return actions["actions"]

    def last_inference_evidence(self) -> Mapping[str, Any]:
        return dict(self.last_receipt)


def customer_hosted_client(*, endpoint: str, allowed_origins: tuple[str, ...],
                           contract: Mapping[str, Any], bearer_token: str | None = None) -> ControlledPolicyClient:
    parsed = urlsplit(endpoint)
    origin = f"{parsed.scheme}://{parsed.netloc}"
    if parsed.scheme != "https" or parsed.username or parsed.password or parsed.fragment or origin not in allowed_origins:
        raise ValueError("controlled_policy_origin_not_approved")
    if bearer_token is not None and (not bearer_token or any(c in bearer_token for c in "\r\n")):
        raise ValueError("controlled_policy_credential_invalid")
    def transport(body: bytes, timeout: float) -> bytes:
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if bearer_token is not None:
            headers["Authorization"] = "Bearer " + bearer_token
        return fetch_bounded_https(endpoint, method="POST", data=body, headers=headers,
            timeout_seconds=timeout, max_bytes=65_536, allowed_origins=allowed_origins,
            allowed_content_types=("application/json",), max_redirects=0).body
    return ControlledPolicyClient(contract=contract, transport=transport, transport_name="approved_https")


def run_controlled_policy_episode(*, environment: ControlledEnvironment,
                                 policy: ControlledPolicyClient, max_queries: int,
                                 deadline_seconds: float,
                                 retain: Callable[[Mapping[str, Any]], None]) -> dict[str, Any]:
    if type(max_queries) is not int or not 1 <= max_queries <= 10_000 or not 0 < deadline_seconds <= 3600:
        raise ValueError("controlled_policy_episode_budget_invalid")
    policy.bind_request_evidence_sink(lambda wire: retain({"kind": "policy_input", "wire": wire}))
    started = time.monotonic()
    queries = 0
    steps = 0
    status = "query_limit"
    try:
        for index in range(max_queries):
            if environment.is_terminal():
                status = "environment_terminal"
                break
            if time.monotonic() - started >= deadline_seconds:
                status = "deadline"
                break
            # Read each step afresh; never replay an old request over scenarios.
            actions = policy.infer(environment.read_policy_inputs())
            queries += 1
            if time.monotonic() - started >= deadline_seconds:
                status = "deadline"
                break
            observed = environment.apply_action_chunk(actions, action_schema=policy.contract["action_schema"])
            count = observed.get("executed_motor_steps")
            if type(count) is not int or not 0 <= count <= len(actions):
                raise ValueError("controlled_policy_execution_step_receipt_invalid")
            steps += count
            retain({"kind": "execution", "query_index": index, "actions": actions,
                    "policy_call": policy.last_inference_evidence(), "environment_receipt": dict(observed)})
        if environment.is_terminal():
            status = "environment_terminal"
    finally:
        environment.stop()
    return {"schema_version": "blueprint.controlled_policy_episode.v1", "status": status,
        "policy_queries": queries, "executed_motor_steps": steps,
        "task_success": None, "physical_success_proven": False,
        "outcome_evidence_required": True}
