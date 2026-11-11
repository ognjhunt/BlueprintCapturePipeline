"""Enter a policy Blueprint runs itself in screening as one registry entry, over the standard contract.

Plan 2.1 (``docs/launch-readiness/2026-11-11-evaluation-service-plan.md``). A
policy served on the session's own machine is entered by one operator registry
row under ``screening_candidates`` (``/etc/blueprint/team-evaluation-registry.json``,
``derived_profiles.defaults``), with no new shared code per policy::

    "pi05_droid_contract": {
      "session_served_policy": {
        "schema_version": "blueprint.session_served_policy.v1",
        "server": "openpi_droid_contract_server.v1",
        "native_candidate_id": "pi05_droid",
        "display_name": "π0.5 DROID (standard connection)",
        "minimum_gpu_memory_gb": 12,
        "startup_timeout_seconds": 900
      }
    }

``server`` names the command that starts the contract server on the GPU
(``SERVERS``). ``native_candidate_id`` is the checkpoint reference: the pinned
identity, checkpoint inventory and provisioning of a policy that already runs
natively; the session-served row serves that same checkpoint. The memory floor
is at least the native one. Everything else is derived:

- a catalog row with the native row's policy identity, licence and interface
  (``openpi_droid_https.v1`` in place of the native interface id);
- an execution spec whose ``policy_spec`` is the contract's, on
  ``http://127.0.0.1:<port>``, and whose ``session_policy_server`` block carries
  the native OpenPI spec the server loads.

Planning, funding, launch, settlement and publication then treat it like any
learned candidate. The only money is the GPU cap: there is no per-request fee
and no hosted field, so settlement charges exactly what a native candidate's
does. The session generates the server's bearer token itself; no operator
credential exists for it and none reaches the bundle.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import math
import re
from typing import Any

from .decision_evidence_contracts import canonical_digest

FIELD = "session_served_policy"
ENTRY_SCHEMA = "blueprint.session_served_policy.v1"
SPEC_FIELD = "session_policy_server"
INTERFACE_ID = "openpi_droid_https.v1"
TOKEN_ENV = "BLUEPRINT_SESSION_POLICY_TOKEN_FILE"
DEFAULT_PORT = 8765
# The native server worker's readiness allowance (``adp009d_policy_server_worker.READINESS_TIMEOUT_SECONDS``).
MAXIMUM_STARTUP_SECONDS = 900
MINIMUM_STARTUP_SECONDS = 60
SERVERS = {
    "openpi_droid_contract_server.v1": {
        "module": "blueprint_pipeline.openpi_droid_contract_server",
        "native_interface_id": "openpi_droid.v1",
        "native_candidates": ("pi05_droid", "pi0_droid_jointpos_polaris", "pi0_droid_jointpos_100k_polaris",
                              "pi0_fast_droid_jointpos_polaris"),
    },
}
_ENTRY_KEYS = frozenset({"schema_version", "server", "native_candidate_id", "display_name",
                         "minimum_gpu_memory_gb", "startup_timeout_seconds", "port"})
_CANDIDATE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


def validate_entry(entry: Any, *, candidate_id: str) -> dict[str, Any]:
    if (not isinstance(entry, Mapping) or set(entry) - _ENTRY_KEYS
            or entry.get("schema_version") != ENTRY_SCHEMA or entry.get("server") not in SERVERS):
        raise ValueError("session_served_policy_entry_invalid")
    if not _CANDIDATE.fullmatch(candidate_id) or entry.get("native_candidate_id") == candidate_id:
        raise ValueError("session_served_policy_candidate_id_invalid")
    if entry.get("native_candidate_id") not in SERVERS[entry["server"]]["native_candidates"]:
        raise ValueError("session_served_policy_native_candidate_unsupported")
    memory = entry.get("minimum_gpu_memory_gb")
    if memory is not None and (type(memory) not in (int, float) or not math.isfinite(memory) or not 0 < memory <= 1024):
        raise ValueError("session_served_policy_gpu_memory_invalid")
    startup = entry.get("startup_timeout_seconds", MAXIMUM_STARTUP_SECONDS)
    if type(startup) is not int or not MINIMUM_STARTUP_SECONDS <= startup <= MAXIMUM_STARTUP_SECONDS:
        raise ValueError("session_served_policy_startup_bound_invalid")
    port = entry.get("port", DEFAULT_PORT)
    if type(port) is not int or not 1024 <= port <= 65535 or port in {8000, 5555}:
        # 8000 and 5555 are the native OpenPI and GR00T servers' ports.
        raise ValueError("session_served_policy_port_invalid")
    display = entry.get("display_name", candidate_id)
    if not isinstance(display, str) or not display.strip() or len(display) > 200:
        raise ValueError("session_served_policy_display_name_invalid")
    return {**dict(entry), "startup_timeout_seconds": startup, "port": port, "display_name": display.strip()}


def registry_entries(profile: Mapping[str, Any]) -> dict[str, Any]:
    rows = profile.get("screening_candidates") or {}
    return {key: row[FIELD] for key, row in rows.items() if isinstance(row, Mapping) and FIELD in row}


def is_session_served(spec: Mapping[str, Any]) -> bool:
    return isinstance(spec.get(SPEC_FIELD), Mapping)


def session_served_spec(native_spec: Mapping[str, Any], *, candidate_id: str, entry: Mapping[str, Any]) -> dict[str, Any]:
    """The native execution spec, re-served over the contract on the session's loopback."""

    from .session_policy_client import loopback_origin

    entry = validate_entry(entry, candidate_id=candidate_id)
    server = SERVERS[entry["server"]]
    if (native_spec.get("candidate_id") != entry["native_candidate_id"]
            or native_spec.get("policy_interface_id") != server["native_interface_id"]):
        raise ValueError("session_served_policy_native_spec_mismatch")
    native_policy = dict(native_spec["policy_spec"])
    origin = loopback_origin(entry["port"])
    block = {"schema_version": ENTRY_SCHEMA, "server": entry["server"], "module": server["module"],
             "native_candidate_id": entry["native_candidate_id"], "native_policy_spec": native_policy,
             "native_runtime_identity_digest": native_spec["runtime_identity_digest"],
             "native_execution_spec_digest": native_spec["execution_spec_digest"],
             "port": entry["port"], "startup_timeout_seconds": entry["startup_timeout_seconds"],
             "token_env": TOKEN_ENV, "outbound_calls": "none"}
    spec = deepcopy(dict(native_spec))
    spec.pop("execution_spec_digest", None)
    spec.update(candidate_id=candidate_id, policy_interface_id=INTERFACE_ID,
                policy_endpoint={"credential_env": TOKEN_ENV}, **{SPEC_FIELD: block})
    spec["policy_spec"] = {"policy_id": candidate_id, "endpoint_url": origin + "/v1/infer",
                           "metadata_url": origin + "/v1/metadata", "allowed_origin": origin,
                           "model_identity_digest": native_spec["checkpoint_digest"],
                           "action_chunk_rows": native_policy["action_chunk_rows"],
                           "open_loop_horizon": native_spec["open_loop_horizon"]}
    spec["runtime_identity"] = {"native_runtime_identity": native_spec.get("runtime_identity"),
                                "native_runtime_identity_digest": native_spec["runtime_identity_digest"],
                                "policy_interface_id": INTERFACE_ID,
                                "session_policy_server": {key: block[key] for key in (
                                    "server", "module", "native_candidate_id", "port")}}
    spec["runtime_identity_digest"] = canonical_digest(spec["runtime_identity"])
    spec["execution_spec_digest"] = canonical_digest(spec, digest_field="execution_spec_digest")
    return validate_session_served_spec(spec)


def validate_session_served_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    """The block, the contract spec and the native spec agree; nothing else may vary."""

    from .openpi_droid_policy_runtime import OpenPIDroidPolicySpec
    from .session_policy_client import LoopbackDroidPolicySpec, loopback_origin

    block = spec.get(SPEC_FIELD)
    if (not isinstance(block, Mapping) or block.get("schema_version") != ENTRY_SCHEMA
            or block.get("server") not in SERVERS or block.get("module") != SERVERS[block["server"]]["module"]
            or block.get("native_candidate_id") not in SERVERS[block["server"]]["native_candidates"]
            or block.get("token_env") != TOKEN_ENV or block.get("outbound_calls") != "none"
            or spec.get("policy_interface_id") != INTERFACE_ID
            or spec.get("policy_endpoint") != {"credential_env": TOKEN_ENV}
            or type(block.get("startup_timeout_seconds")) is not int
            or not MINIMUM_STARTUP_SECONDS <= block["startup_timeout_seconds"] <= MAXIMUM_STARTUP_SECONDS):
        raise ValueError("session_served_policy_spec_invalid")
    native = OpenPIDroidPolicySpec(**block["native_policy_spec"])
    from .openpi_droid_policy_runtime import validate_arena_candidate_policy_binding
    validate_arena_candidate_policy_binding(candidate_id=block["native_candidate_id"], spec=native)
    policy = LoopbackDroidPolicySpec(**spec["policy_spec"])
    policy.validate()
    if (policy.allowed_origin != loopback_origin(block["port"]) or policy.policy_id != spec.get("candidate_id")
            or policy.model_identity_digest != spec.get("checkpoint_digest")
            or policy.action_chunk_rows != native.action_chunk_rows
            or policy.open_loop_horizon != spec.get("open_loop_horizon")
            or native.open_loop_horizon != spec.get("open_loop_horizon")):
        raise ValueError("session_served_policy_contract_native_mismatch")
    return dict(spec)


__all__ = ["DEFAULT_PORT", "ENTRY_SCHEMA", "FIELD", "INTERFACE_ID", "SERVERS", "SPEC_FIELD", "TOKEN_ENV",
           "is_session_served", "registry_entries", "session_served_spec", "validate_entry",
           "validate_session_served_spec"]
