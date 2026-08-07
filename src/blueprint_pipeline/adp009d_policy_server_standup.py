"""Fail-closed standup contract for an ADP-009D DROID policy server.

The Isaac runtime is launched by ``/isaac-sim/python.sh`` from a single provider
bundle entrypoint (``adp009d_native_microcheck_bundle.ENTRYPOINT``), and the Vast
transport runs that entrypoint in one container per instance
(``adp_isaac_lab_arena_vast`` uses ``vast_launch_mode="ssh_direct"``).  So the
policy server is not a separate container: it is a second *process* on the same
worker, and the only real question is which interpreter loads its weights.

That question is already answered in this repo, and the answer is a second
interpreter in the same container.  The shipped ``groot_oscar_closed_loop`` image
runs the GR00T policy server, the OSCAR client, and Isaac Sim in three separate
Pythons side by side, and ``scripts/verify_groot_oscar_thin_architecture`` fails
the build with ``foundation_groot_environment_not_isolated`` if anyone merges
them.  ``groot_oscar_worker_startup_script`` then raises
``groot_runtime_isolation_failed`` at runtime if a foreign path leaks onto the
policy interpreter, a gate added after ``accelerate`` resolved out of the wrong
environment.  This module is the ADP-009D receipt side of that precedent.

The isolation is about paths, not versions: those venvs are CPython 3.10 while
Isaac's interpreter is 3.12, in one container.  What differs per candidate is the
framework -- ``pi05_droid`` needs JAX (``openpi_policy_ranking_gpu_job`` calls
``jax.devices()``) and the GR00T and Cosmos candidates need PyTorch -- and JAX in
particular preallocates the device, so a co-resident JAX server has to be capped
before Isaac can survive next to it.

The contract's sharpest edge is readiness.  ``cosmos_edge_droid_policy_server``
writes ``"status": "model_loaded_ready_to_serve"`` *before* it calls
``serve_forever()``, so a loaded model is the only thing that claim proves.  A
loaded model that returns a wrong-shaped chunk, or that never binds its port,
would still look ready.  Here ``ready`` is refused unless the receipt carries a
completed inference round trip whose action chunk the ADP-009D action adapter
would accept.

Pure logic: no GPU, no network, and no heavyweight import at module import time.
"""

from __future__ import annotations

import json
import re
from typing import Any, Mapping, Sequence

from blueprint_pipeline.adp009d_droid_action_execution import (
    DROID_ACTION_WIDTH,
    DROID_CONTROL_HZ,
    DROID_OPEN_LOOP_HORIZON,
)
from blueprint_pipeline.adp009d_droid_observation import (
    DROID_OBSERVATION_SCHEMA_VERSION,
)
from blueprint_pipeline.adp009d_policy_candidate_admission import (
    EXPECTED_CANDIDATES,
    PROGRAM_ID,
)
from blueprint_pipeline.decision_evidence_contracts import canonical_digest


STANDUP_SCHEMA_VERSION = "adp009d_policy_server_standup.v1"

# The interpreter the Isaac runtime is actually launched with, quoted from the
# provider-bundle entrypoint in ``adp009d_native_microcheck_bundle.ENTRYPOINT``.
ISAAC_INTERPRETER = "/isaac-sim/python.sh"

# Isaac Sim 6.0.0-dev2 resolves to CPython 3.12: the worker installs
# ``h5py-3.16.0-cp312-cp312-...`` and ``msgpack-1.1.0-cp312-cp312-...`` wheels
# into it (``adp009d_native_microcheck_worker``).  Recorded as evidence, *not* as
# a version the policy environment must match: the shipped
# ``groot_oscar_closed_loop`` image already runs ``uv venv --python 3.10``
# environments beside this 3.12 interpreter in one container, so requiring a
# match would forbid the topology that is already in production.
ISAAC_PYTHON_VERSION = "3.12"

# Each candidate's transport is fixed by the client Blueprint already wrote for
# it, not chosen at standup time.  Serving GR00T over a websocket, or pi05 over
# ZMQ, would leave the corresponding client unable to speak to its own server.
CANDIDATE_TRANSPORTS: dict[str, str] = {
    # openpi_droid_policy_runtime.OpenPIWebsocketDroidPolicyClient.evidence_summary
    "pi05_droid": "openpi_websocket_msgpack_numpy",
    # groot_n17_droid_policy_runtime.GrootN17DroidPolicyClient.evidence_summary
    "groot_n17_droid": "nvidia_groot_zmq_msgpack",
    "groot_n16_droid": "nvidia_groot_zmq_msgpack",
    # cosmos_edge_droid_policy_runtime.CosmosEdgeDroidPolicyClient.evidence_summary
    "cosmos3_edge_policy_droid": "openpi_websocket_msgpack_numpy",
}

# The accelerator framework each server loads.  This is the fact that decides
# whether an interpreter can be shared, so it is recorded per candidate rather
# than treated as one property of "the policy server".
CANDIDATE_SERVER_FRAMEWORKS: dict[str, str] = {
    "pi05_droid": "jax",
    "groot_n17_droid": "torch",
    "groot_n16_droid": "torch",
    "cosmos3_edge_policy_droid": "torch",
}

# Topologies this programme is willing to run.  An unlisted topology fails
# closed: the point of the enumeration is that each entry has a known teardown
# and a known failure mode, not that the list is exhaustive in principle.
SUPPORTED_TOPOLOGIES = frozenset(
    {
        # One Vast instance, one container, Isaac under ``python.sh`` and the
        # policy server under its own interpreter.  The default.
        "shared_worker_separate_interpreter",
        # One Vast instance, one container, one interpreter for both.  Only
        # admissible with a recorded proof that no framework was co-installed.
        "shared_worker_shared_interpreter",
        # Two instances.  Costs a second teardown and leaves the loopback-only
        # rule behind, so it must be justified rather than defaulted into.
        "separate_worker",
    }
)

# The policy endpoint never leaves the worker.  ``openpi_droid_policy_runtime``
# already raises ``openpi_policy_server_must_be_loopback_only`` for the same
# reason: a routable policy port on a rented machine is an open inference
# endpoint serving weights Blueprint does not own.
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})

# The frozen startup order.  Two orderings matter and both are load-bearing.
#
# Checkpoint materialization precedes both servers because it is the long pole
# on a per-second-billed worker and because a checkpoint that fails its digest
# check should never cost a Kit boot.
#
# Isaac starts before the policy server because Isaac is the fragile consumer.
# Its failure mode under VRAM pressure is a native abort with no Python
# traceback (the ADP-009D runtime only catches ``Exception`` around ``_run``),
# whereas a policy server that cannot fit its weights raises an ordinary
# allocation error in its own process.  Claiming the remaining memory with the
# recoverable process makes co-residency measurable instead of fatal.
STARTUP_PHASES: tuple[str, ...] = (
    "worker_admitted",
    "policy_environment_created",
    "checkpoint_materialized",
    "checkpoint_verified",
    "isaac_runtime_started",
    "policy_server_started",
    "policy_server_endpoint_accepting",
    "identity_metadata_verified",
    "inference_round_trip_verified",
    "isaac_policy_query_enabled",
)

# Mirrors ``adp009d_policy_candidate_admission._FORBIDDEN_OUTCOME_KEYS``.  A
# standup receipt is written before any task outcome exists, so a caller
# asserting one here is asserting something it cannot know.  A test pins the two
# sets together so the vocabularies cannot drift apart silently.
FORBIDDEN_OUTCOME_KEYS = frozenset(
    {
        "candidate_result",
        "episode_success",
        "learned_outcome",
        "policy_result",
        "task_success",
    }
)

_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
_RFC3339_UTC = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
_PYTHON_MINOR = re.compile(r"^\d+\.\d+$")


class Adp009dPolicyServerStandupError(ValueError):
    """Stable fail-closed policy-server standup errors."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(sorted({str(error) for error in errors if str(error)}))
        super().__init__("; ".join(self.errors))


def _clone(value: Mapping[str, Any], *, error: str) -> dict[str, Any]:
    try:
        cloned = json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise Adp009dPolicyServerStandupError([error]) from exc
    if not isinstance(cloned, dict):
        raise Adp009dPolicyServerStandupError([error])
    return cloned


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _strings(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _forbidden_outcome_paths(value: Any, *, prefix: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if str(key).lower() in FORBIDDEN_OUTCOME_KEYS:
                found.append(path)
            found.extend(_forbidden_outcome_paths(child, prefix=path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_forbidden_outcome_paths(child, prefix=f"{prefix}[{index}]"))
    return found


def _positive_int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _positive_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if value > 0 else None


def describe_standup_plan(candidate_id: str) -> dict[str, Any]:
    """Report the fixed standup facts for one candidate, for the run receipt.

    Everything here is already determined by the candidate's own client and by
    the container the Isaac runtime executes in.  Reporting it separately means
    a standup can be reviewed against the plan without reading four runtimes.
    """

    if candidate_id not in EXPECTED_CANDIDATES:
        raise Adp009dPolicyServerStandupError(
            [f"policy_server_standup_unknown_candidate:{candidate_id}"]
        )
    expected = EXPECTED_CANDIDATES[candidate_id]
    return {
        "schema_version": STANDUP_SCHEMA_VERSION,
        "program_id": PROGRAM_ID,
        "candidate_id": candidate_id,
        "transport": CANDIDATE_TRANSPORTS[candidate_id],
        "server_framework": CANDIDATE_SERVER_FRAMEWORKS[candidate_id],
        "isaac_interpreter": ISAAC_INTERPRETER,
        "isaac_python_version": ISAAC_PYTHON_VERSION,
        "checkpoint_repository": expected["checkpoint_repository"],
        "checkpoint_revision": expected["checkpoint_revision"],
        "checkpoint_total_bytes": expected["checkpoint_total_bytes"],
        "checkpoint_snapshot_inventory_digest": expected["checkpoint_inventory_digest"],
        "startup_phases": list(STARTUP_PHASES),
        "endpoint_must_be_loopback": True,
        "checkpoint_materialized_on": "gpu_worker",
        "required_round_trip_action_width": DROID_ACTION_WIDTH,
        "required_round_trip_action_rows": DROID_OPEN_LOOP_HORIZON,
        "control_hz": DROID_CONTROL_HZ,
        "observation_adapter_schema_version": DROID_OBSERVATION_SCHEMA_VERSION,
    }


def _validate_checkpoint(receipt: Mapping[str, Any], candidate_id: str) -> list[str]:
    """Bind the materialized bytes to the frozen candidate checkpoint identity."""

    errors: list[str] = []
    expected = EXPECTED_CANDIDATES[candidate_id]
    checkpoint = _mapping(receipt.get("checkpoint"))
    identity = {
        "repository": expected["checkpoint_repository"],
        "revision": expected["checkpoint_revision"],
        "total_bytes": expected["checkpoint_total_bytes"],
        "snapshot_inventory_digest": expected["checkpoint_inventory_digest"],
    }
    for field, expected_value in identity.items():
        if checkpoint.get(field) != expected_value:
            errors.append(f"policy_server_standup_checkpoint_{field}_invalid")

    # The materialization digest covers the bytes that actually landed on the
    # worker.  Without it the receipt would prove which checkpoint was *named*,
    # not which one was served.
    if not _SHA256.fullmatch(str(checkpoint.get("materialization_digest") or "")):
        errors.append("policy_server_standup_checkpoint_materialization_digest_invalid")
    if checkpoint.get("all_files_verified") is not True:
        errors.append("policy_server_standup_checkpoint_files_unverified")

    # A truncated download still loads for some formats and then serves a model
    # that is not the frozen one, so a short byte count fails rather than warns.
    materialized_bytes = _positive_int(checkpoint.get("materialized_bytes"))
    if materialized_bytes is None or materialized_bytes != expected["checkpoint_total_bytes"]:
        errors.append("policy_server_standup_checkpoint_materialized_bytes_mismatch")

    # The orchestrating machine cannot hold these checkpoints; the smallest is
    # 6.5 GB and the host that compiles the bundle has single-digit GiB free.
    # Materialization is therefore a worker-side fact, and a receipt claiming
    # otherwise describes a run that could not have happened as designed.
    if checkpoint.get("materialized_on") != "gpu_worker":
        errors.append("policy_server_standup_checkpoint_not_materialized_on_worker")
    orchestrator_bytes = checkpoint.get("orchestrator_bytes")
    if orchestrator_bytes != 0:
        errors.append("policy_server_standup_checkpoint_orchestrator_bytes_nonzero")
    return errors


def _validate_topology(receipt: Mapping[str, Any], candidate_id: str) -> list[str]:
    """Validate the process topology and the interpreter-isolation proof."""

    errors: list[str] = []
    topology = _mapping(receipt.get("topology"))
    mode = str(topology.get("mode") or "")
    if mode not in SUPPORTED_TOPOLOGIES:
        errors.append(f"policy_server_standup_topology_unsupported:{mode or 'missing'}")
    if topology.get("isaac_interpreter") != ISAAC_INTERPRETER:
        errors.append("policy_server_standup_isaac_interpreter_invalid")

    policy_interpreter = str(topology.get("policy_interpreter") or "")
    if not policy_interpreter:
        errors.append("policy_server_standup_policy_interpreter_missing")

    python_version = str(topology.get("policy_python_version") or "")
    if not _PYTHON_MINOR.fullmatch(python_version):
        errors.append("policy_server_standup_policy_python_version_invalid")

    if mode == "shared_worker_separate_interpreter":
        # A "separate interpreter" that is the Isaac launcher is not separate.
        # The policy environment's minor version deliberately does *not* have to
        # match Isaac's: ``groot_oscar_closed_loop`` already ships 3.10 venvs
        # beside Isaac's 3.12 in one container.  Isolation is the requirement.
        if policy_interpreter == ISAAC_INTERPRETER:
            errors.append("policy_server_standup_policy_interpreter_not_separate")

        # ``python.sh`` exports Isaac's own environment, and a policy process
        # that picks up a foreign path imports the wrong framework.  This is not
        # hypothetical: ``groot_oscar_worker_startup_script`` raises
        # ``groot_runtime_isolation_failed`` on exactly these three checks
        # because ``accelerate`` once resolved out of the wrong venv.  The proof
        # is a measured ``sys.prefix`` and ``sys.path``, never an assumption.
        isolation = _mapping(topology.get("interpreter_isolation"))
        if isolation.get("isaac_site_packages_on_policy_sys_path") is not False:
            errors.append("policy_server_standup_policy_env_not_isolated")
        if isolation.get("policy_interpreter_prefix_exact") is not True:
            errors.append("policy_server_standup_policy_interpreter_prefix_inexact")
        if not _SHA256.fullmatch(str(isolation.get("policy_sys_path_digest") or "")):
            errors.append("policy_server_standup_policy_sys_path_digest_invalid")

    if mode != "separate_worker" and CANDIDATE_SERVER_FRAMEWORKS[candidate_id] == "jax":
        # JAX preallocates a fraction of the device at first use.  Blueprint's
        # standalone OpenPI image sets ``XLA_PYTHON_CLIENT_MEM_FRACTION=0.80``
        # (``deploy/docker/policy_ranking_openpi/Dockerfile``) because it owns
        # the whole GPU.  Co-resident with Isaac that setting is fatal, and a
        # smaller fraction only narrows the race rather than removing it, so
        # preallocation must be off outright.
        guard = _mapping(topology.get("accelerator_memory_guard"))
        if guard.get("xla_python_client_preallocate") is not False:
            errors.append("policy_server_standup_jax_preallocation_not_disabled")

    if mode == "shared_worker_shared_interpreter":
        # Sharing the interpreter means the candidate's framework had to already
        # be present.  Installing one alongside Isaac's pins is the failure this
        # branch refuses to allow silently.
        if topology.get("accelerator_framework_installed_into_isaac_interpreter") is not False:
            errors.append("policy_server_standup_framework_installed_into_isaac_interpreter")
        if policy_interpreter != ISAAC_INTERPRETER:
            errors.append("policy_server_standup_shared_interpreter_mismatch")

    if mode in {"shared_worker_separate_interpreter", "shared_worker_shared_interpreter"}:
        if topology.get("same_worker") is not True:
            errors.append("policy_server_standup_topology_worker_flag_inconsistent")
        # Co-residency is the claim under test, so the observed split has to be
        # recorded rather than inferred from the instance's advertised size.
        memory = _mapping(topology.get("gpu_memory_observed_mib"))
        for field in ("total", "isaac_peak", "policy_server_peak"):
            if _positive_int(memory.get(field)) is None:
                errors.append(f"policy_server_standup_gpu_memory_{field}_invalid")
        total = _positive_int(memory.get("total"))
        isaac_peak = _positive_int(memory.get("isaac_peak"))
        policy_peak = _positive_int(memory.get("policy_server_peak"))
        if None not in (total, isaac_peak, policy_peak) and isaac_peak + policy_peak > total:
            errors.append("policy_server_standup_gpu_memory_oversubscribed")
    elif mode == "separate_worker":
        if topology.get("same_worker") is not False:
            errors.append("policy_server_standup_topology_worker_flag_inconsistent")
        # A second instance is a second billing object and a second teardown.
        if topology.get("second_worker_teardown_required") is not True:
            errors.append("policy_server_standup_second_worker_teardown_unclaimed")

    if topology.get("server_framework") != CANDIDATE_SERVER_FRAMEWORKS[candidate_id]:
        errors.append("policy_server_standup_server_framework_mismatch")
    return errors


def _validate_endpoint(receipt: Mapping[str, Any], candidate_id: str) -> list[str]:
    """Validate the transport and the loopback-only endpoint."""

    errors: list[str] = []
    endpoint = _mapping(receipt.get("endpoint"))
    if endpoint.get("transport") != CANDIDATE_TRANSPORTS[candidate_id]:
        errors.append("policy_server_standup_transport_mismatch")
    host = str(endpoint.get("host") or "")
    if host not in LOOPBACK_HOSTS:
        errors.append("policy_server_standup_endpoint_not_loopback")
    port = _positive_int(endpoint.get("port"))
    if port is None or not 1 <= port <= 65535:
        errors.append("policy_server_standup_endpoint_port_invalid")
    # A bound socket is the difference between "the process started" and "the
    # client can reach it"; the Isaac side has no other way to tell them apart.
    if endpoint.get("listening_socket_confirmed") is not True:
        errors.append("policy_server_standup_endpoint_socket_unconfirmed")
    return errors


def _validate_readiness(receipt: Mapping[str, Any]) -> list[str]:
    """Refuse ``ready`` without a completed, shape-checked inference round trip.

    A loaded model is not a working server.  The existing Cosmos server writes
    its ``model_loaded_ready_to_serve`` startup record before ``serve_forever()``
    is ever called, so that record cannot distinguish a server that answers from
    one that never binds, answers with an empty metadata block, or returns a
    chunk the action adapter would reject.  The round trip is the only evidence
    that closes all three.
    """

    errors: list[str] = []
    readiness = _mapping(receipt.get("readiness"))
    if readiness.get("model_loaded") is not True:
        errors.append("policy_server_standup_model_not_loaded")
    # Identity verification is what the candidate clients already perform on
    # connect; recording it here keeps the standup and the client agreeing about
    # which weights answered.
    if readiness.get("server_identity_verified") is not True:
        errors.append("policy_server_standup_server_identity_unverified")
    if not _SHA256.fullmatch(str(readiness.get("server_identity_digest") or "")):
        errors.append("policy_server_standup_server_identity_digest_invalid")

    round_trip = _mapping(readiness.get("inference_round_trip"))
    if not round_trip:
        errors.append("policy_server_standup_inference_round_trip_missing")
    else:
        if round_trip.get("completed") is not True:
            errors.append("policy_server_standup_inference_round_trip_incomplete")
        for field in ("observation_digest", "action_digest"):
            if not _SHA256.fullmatch(str(round_trip.get(field) or "")):
                errors.append(f"policy_server_standup_round_trip_{field}_invalid")
        # The observation must have come from the ADP-009D adapter.  A round
        # trip on a hand-built array would prove the server answers something,
        # not that it answers what Isaac will actually send it.
        if (
            round_trip.get("observation_adapter_schema_version")
            != DROID_OBSERVATION_SCHEMA_VERSION
        ):
            errors.append("policy_server_standup_round_trip_observation_adapter_invalid")
        shape = round_trip.get("action_shape")
        if not isinstance(shape, list) or len(shape) != 2:
            errors.append("policy_server_standup_round_trip_action_shape_invalid")
        else:
            rows = _positive_int(shape[0])
            width = _positive_int(shape[1])
            if width != DROID_ACTION_WIDTH:
                errors.append("policy_server_standup_round_trip_action_width_invalid")
            # A chunk shorter than the open-loop horizon cannot drive the arm
            # for the interval the harness commits to before requerying.
            if rows is None or rows < DROID_OPEN_LOOP_HORIZON:
                errors.append("policy_server_standup_round_trip_action_rows_insufficient")
        if round_trip.get("action_finite") is not True:
            errors.append("policy_server_standup_round_trip_action_nonfinite")
        # Latency decides whether the loop can hold 15 Hz at all; an unmeasured
        # round trip leaves that unknown until the paid matrix is already running.
        if _positive_number(round_trip.get("latency_ms")) is None:
            errors.append("policy_server_standup_round_trip_latency_invalid")

    if readiness.get("ready") is not True:
        errors.append("policy_server_standup_not_ready")
    return errors


def _validate_lifecycle(receipt: Mapping[str, Any]) -> list[str]:
    """Validate startup ordering and the teardown the worker owes."""

    errors: list[str] = []
    if _strings(receipt.get("startup_phases_completed")) != list(STARTUP_PHASES):
        errors.append("policy_server_standup_startup_phase_order_invalid")

    teardown = _mapping(receipt.get("teardown"))
    if teardown.get("policy_server_process_terminated") is not True:
        errors.append("policy_server_standup_policy_process_not_terminated")
    # Nothing survives the instance, so the checkpoint must not have been copied
    # back: it is public, large, and re-materializable from its frozen digest.
    if teardown.get("checkpoint_retained_on_orchestrator") is not False:
        errors.append("policy_server_standup_checkpoint_retained_on_orchestrator")
    if teardown.get("provider_zero_required_after_return") is not True:
        errors.append("policy_server_standup_provider_zero_unclaimed")
    return errors


def validate_policy_server_standup(
    value: Mapping[str, Any],
    *,
    candidate: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate one policy-server standup receipt.

    ``candidate`` is an optional row from the ADP-009D candidate inventory.  When
    supplied, the receipt is additionally bound to that row's ``candidate_digest``
    so a standup cannot be reused against a re-audited candidate.
    """

    receipt = _clone(value, error="policy_server_standup_not_json_mapping")
    errors: list[str] = []

    if receipt.get("schema_version") != STANDUP_SCHEMA_VERSION:
        errors.append("policy_server_standup_schema_invalid")
    if receipt.get("program_id") != PROGRAM_ID:
        errors.append("policy_server_standup_program_invalid")
    if not _RFC3339_UTC.fullmatch(str(receipt.get("stood_up_at") or "")):
        errors.append("policy_server_standup_timestamp_invalid")

    candidate_id = str(receipt.get("candidate_id") or "")
    if candidate_id not in EXPECTED_CANDIDATES:
        # Everything downstream is keyed on the candidate, so an unknown one is
        # reported alone rather than compounded with derived failures.
        errors.append(f"policy_server_standup_unknown_candidate:{candidate_id or 'missing'}")
        raise Adp009dPolicyServerStandupError(errors)

    if candidate is not None:
        if candidate.get("candidate_id") != candidate_id:
            errors.append("policy_server_standup_candidate_id_mismatch")
        if receipt.get("candidate_digest") != candidate.get("candidate_digest"):
            errors.append("policy_server_standup_candidate_digest_mismatch")
    if not _SHA256.fullmatch(str(receipt.get("candidate_digest") or "")):
        errors.append("policy_server_standup_candidate_digest_invalid")

    errors.extend(_validate_checkpoint(receipt, candidate_id))
    errors.extend(_validate_topology(receipt, candidate_id))
    errors.extend(_validate_endpoint(receipt, candidate_id))
    errors.extend(_validate_readiness(receipt))
    errors.extend(_validate_lifecycle(receipt))

    # A standup happens before the matrix.  Any task outcome in this receipt is
    # asserted rather than observed.
    if receipt.get("task_outcomes_observed") is not False:
        errors.append("policy_server_standup_after_task_outcomes")
    if _forbidden_outcome_paths(receipt):
        errors.append("policy_server_standup_caller_asserted_outcome_forbidden")
    if _strings(receipt.get("blockers")):
        errors.append("policy_server_standup_has_blockers")

    if receipt.get("standup_digest") != canonical_digest(
        receipt, digest_field="standup_digest"
    ):
        errors.append("policy_server_standup_digest_mismatch")

    if errors:
        raise Adp009dPolicyServerStandupError(errors)
    return receipt


def seal_policy_server_standup(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Attach the self-describing digest to an otherwise complete receipt.

    Provided so a caller never hand-computes the digest field and accidentally
    seals a payload different from the one it validated.
    """

    sealed = _clone(receipt, error="policy_server_standup_not_json_mapping")
    sealed["standup_digest"] = canonical_digest(sealed, digest_field="standup_digest")
    return sealed


__all__ = [
    "Adp009dPolicyServerStandupError",
    "CANDIDATE_SERVER_FRAMEWORKS",
    "CANDIDATE_TRANSPORTS",
    "FORBIDDEN_OUTCOME_KEYS",
    "ISAAC_INTERPRETER",
    "ISAAC_PYTHON_VERSION",
    "LOOPBACK_HOSTS",
    "STANDUP_SCHEMA_VERSION",
    "STARTUP_PHASES",
    "SUPPORTED_TOPOLOGIES",
    "describe_standup_plan",
    "seal_policy_server_standup",
    "validate_policy_server_standup",
]
