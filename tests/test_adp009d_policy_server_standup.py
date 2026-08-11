from __future__ import annotations

from typing import Any

import pytest

from blueprint_pipeline.adp009d_droid_action_execution import (
    DROID_ACTION_WIDTH,
    DROID_OPEN_LOOP_HORIZON,
)
from blueprint_pipeline.adp009d_droid_observation import (
    DROID_OBSERVATION_SCHEMA_VERSION,
)
from blueprint_pipeline.adp009d_policy_candidate_admission import (
    EXPECTED_CANDIDATES,
    PROGRAM_ID,
)
from blueprint_pipeline.adp009d_policy_server_standup import (
    CANDIDATE_SERVER_FRAMEWORKS,
    CANDIDATE_TRANSPORTS,
    FORBIDDEN_OUTCOME_KEYS,
    ISAAC_INTERPRETER,
    ISAAC_PYTHON_VERSION,
    STANDUP_SCHEMA_VERSION,
    STARTUP_PHASES,
    Adp009dPolicyServerStandupError,
    describe_standup_plan,
    seal_policy_server_standup,
    validate_policy_server_standup,
)

_DIGEST_A = "sha256:" + "a" * 64
_DIGEST_B = "sha256:" + "b" * 64
_DIGEST_C = "sha256:" + "c" * 64
_DIGEST_D = "sha256:" + "d" * 64
_DIGEST_E = "sha256:" + "e" * 64


def _receipt(candidate_id: str = "groot_n17_droid", **overrides: Any) -> dict[str, Any]:
    """A standup that should pass, so every test below changes exactly one thing."""

    expected = EXPECTED_CANDIDATES[candidate_id]
    receipt: dict[str, Any] = {
        "schema_version": STANDUP_SCHEMA_VERSION,
        "program_id": PROGRAM_ID,
        "stood_up_at": "2026-08-07T12:00:00Z",
        "candidate_id": candidate_id,
        "candidate_digest": _DIGEST_A,
        "checkpoint": {
            "repository": expected["checkpoint_repository"],
            "revision": expected["checkpoint_revision"],
            "total_bytes": expected["checkpoint_total_bytes"],
            "snapshot_inventory_digest": expected["checkpoint_inventory_digest"],
            "materialization_digest": _DIGEST_B,
            "materialized_bytes": expected["checkpoint_total_bytes"],
            "materialized_on": "gpu_worker",
            "orchestrator_bytes": 0,
            "all_files_verified": True,
        },
        "topology": {
            "mode": "shared_worker_separate_interpreter",
            "same_worker": True,
            "isaac_interpreter": ISAAC_INTERPRETER,
            "policy_interpreter": "/opt/blueprint-policy-env/bin/python",
            # 3.10, not Isaac's 3.12: the shipped groot_oscar_closed_loop image
            # already runs 3.10 venvs beside Isaac's interpreter in one container.
            "policy_python_version": "3.10",
            "server_framework": CANDIDATE_SERVER_FRAMEWORKS[candidate_id],
            "interpreter_isolation": {
                "isaac_site_packages_on_policy_sys_path": False,
                "policy_interpreter_prefix_exact": True,
                "policy_sys_path_digest": _DIGEST_C,
            },
            "accelerator_memory_guard": {"xla_python_client_preallocate": False},
            "gpu_memory_observed_mib": {
                "total": 46_068,
                "isaac_peak": 18_400,
                "policy_server_peak": 21_100,
            },
        },
        "endpoint": {
            "transport": CANDIDATE_TRANSPORTS[candidate_id],
            "host": "127.0.0.1",
            "port": 5555,
            "listening_socket_confirmed": True,
        },
        "readiness": {
            "model_loaded": True,
            "server_identity_verified": True,
            "server_identity_digest": _DIGEST_D,
            "ready": True,
            "inference_round_trip": {
                "completed": True,
                "observation_digest": _DIGEST_E,
                "action_digest": _DIGEST_B,
                "observation_adapter_schema_version": DROID_OBSERVATION_SCHEMA_VERSION,
                "action_shape": [16, DROID_ACTION_WIDTH],
                "action_finite": True,
                "latency_ms": 148.5,
            },
        },
        "startup_phases_completed": list(STARTUP_PHASES),
        "teardown": {
            "policy_server_process_terminated": True,
            "checkpoint_retained_on_orchestrator": False,
            "provider_zero_required_after_return": True,
        },
        "task_outcomes_observed": False,
        "blockers": [],
    }
    for key, value in overrides.items():
        receipt[key] = value
    return seal_policy_server_standup(receipt)


def _errors(receipt: dict[str, Any], **kwargs: Any) -> tuple[str, ...]:
    with pytest.raises(Adp009dPolicyServerStandupError) as excinfo:
        validate_policy_server_standup(receipt, **kwargs)
    return excinfo.value.errors


def test_a_complete_standup_validates() -> None:
    """The happy path exists so every failure below is a single deliberate edit."""

    validated = validate_policy_server_standup(_receipt())
    assert validated["candidate_id"] == "groot_n17_droid"
    assert validated["readiness"]["ready"] is True


def test_a_loaded_model_alone_is_not_ready() -> None:
    """The whole point: the existing Cosmos server calls itself ready at load."""

    receipt = _receipt()
    receipt["readiness"].pop("inference_round_trip")
    assert receipt["readiness"]["model_loaded"] is True
    assert receipt["readiness"]["ready"] is True
    assert "policy_server_standup_inference_round_trip_missing" in _errors(
        seal_policy_server_standup(receipt)
    )


def test_an_attempted_round_trip_that_did_not_complete_is_not_ready() -> None:
    """A request that was issued but never returned proves nothing about the server."""

    receipt = _receipt()
    receipt["readiness"]["inference_round_trip"]["completed"] = False
    assert "policy_server_standup_inference_round_trip_incomplete" in _errors(
        seal_policy_server_standup(receipt)
    )


def test_a_round_trip_shorter_than_the_open_loop_horizon_fails() -> None:
    """A chunk the harness cannot execute for its commitment interval is unusable."""

    receipt = _receipt()
    receipt["readiness"]["inference_round_trip"]["action_shape"] = [
        DROID_OPEN_LOOP_HORIZON - 1,
        DROID_ACTION_WIDTH,
    ]
    assert "policy_server_standup_round_trip_action_rows_insufficient" in _errors(
        seal_policy_server_standup(receipt)
    )


def test_a_round_trip_with_the_wrong_action_width_fails() -> None:
    """Seven joints plus one gripper is the only width the Isaac adapter accepts."""

    receipt = _receipt()
    receipt["readiness"]["inference_round_trip"]["action_shape"] = [16, 7]
    assert "policy_server_standup_round_trip_action_width_invalid" in _errors(
        seal_policy_server_standup(receipt)
    )


def test_a_round_trip_on_a_hand_built_observation_fails() -> None:
    """The server must have answered what Isaac will send, not a convenient array."""

    receipt = _receipt()
    receipt["readiness"]["inference_round_trip"][
        "observation_adapter_schema_version"
    ] = "handmade.v0"
    assert "policy_server_standup_round_trip_observation_adapter_invalid" in _errors(
        seal_policy_server_standup(receipt)
    )


def test_a_nonfinite_or_unmeasured_round_trip_fails() -> None:
    """NaNs reach the arm; unmeasured latency hides whether 15 Hz is reachable."""

    receipt = _receipt()
    receipt["readiness"]["inference_round_trip"]["action_finite"] = False
    receipt["readiness"]["inference_round_trip"]["latency_ms"] = 0
    errors = _errors(seal_policy_server_standup(receipt))
    assert "policy_server_standup_round_trip_action_nonfinite" in errors
    assert "policy_server_standup_round_trip_latency_invalid" in errors


def test_a_partially_downloaded_checkpoint_fails() -> None:
    """Some formats load from a truncated tree and then serve the wrong weights."""

    receipt = _receipt()
    receipt["checkpoint"]["materialized_bytes"] = (
        EXPECTED_CANDIDATES["groot_n17_droid"]["checkpoint_total_bytes"] - 1
    )
    assert "policy_server_standup_checkpoint_materialized_bytes_mismatch" in _errors(
        seal_policy_server_standup(receipt)
    )


def test_a_checkpoint_naming_a_different_revision_fails() -> None:
    """The frozen inventory revision is the identity; a receipt cannot restate it."""

    receipt = _receipt()
    receipt["checkpoint"]["revision"] = "0" * 40
    assert "policy_server_standup_checkpoint_revision_invalid" in _errors(
        seal_policy_server_standup(receipt)
    )


def test_a_checkpoint_materialized_off_the_worker_fails() -> None:
    """The orchestrating host has single-digit GiB free; the smallest is 6.5 GB."""

    receipt = _receipt()
    receipt["checkpoint"]["materialized_on"] = "orchestrator"
    receipt["checkpoint"]["orchestrator_bytes"] = 6_914_267_987
    errors = _errors(seal_policy_server_standup(receipt))
    assert "policy_server_standup_checkpoint_not_materialized_on_worker" in errors
    assert "policy_server_standup_checkpoint_orchestrator_bytes_nonzero" in errors


def test_a_policy_env_that_inherits_isaac_site_packages_fails() -> None:
    """python.sh exports Isaac's PYTHONPATH; an inheriting venv imports Isaac's torch."""

    receipt = _receipt()
    receipt["topology"]["interpreter_isolation"][
        "isaac_site_packages_on_policy_sys_path"
    ] = True
    assert "policy_server_standup_policy_env_not_isolated" in _errors(
        seal_policy_server_standup(receipt)
    )


def test_a_separate_interpreter_that_is_isaacs_launcher_is_not_separate() -> None:
    """Naming the topology does not make it true."""

    receipt = _receipt()
    receipt["topology"]["policy_interpreter"] = ISAAC_INTERPRETER
    assert "policy_server_standup_policy_interpreter_not_separate" in _errors(
        seal_policy_server_standup(receipt)
    )


def test_a_policy_env_on_a_different_python_minor_is_allowed() -> None:
    """groot_oscar_closed_loop ships 3.10 venvs beside Isaac's 3.12 in one container.

    Requiring a version match would forbid the topology already in production, so
    the minor is recorded evidence rather than a gate. Isolation is the gate.
    """

    receipt = _receipt()
    receipt["topology"]["policy_python_version"] = "3.11"
    validated = validate_policy_server_standup(seal_policy_server_standup(receipt))
    assert validated["topology"]["policy_python_version"] == "3.11"
    assert validated["topology"]["policy_python_version"] != ISAAC_PYTHON_VERSION


def test_a_malformed_python_version_still_fails() -> None:
    """The minor is not a gate, but an unrecorded interpreter version is no evidence."""

    receipt = _receipt()
    receipt["topology"]["policy_python_version"] = "python3"
    assert "policy_server_standup_policy_python_version_invalid" in _errors(
        seal_policy_server_standup(receipt)
    )


def test_an_inexact_interpreter_prefix_fails() -> None:
    """This is the check that caught `accelerate` resolving out of the wrong venv."""

    receipt = _receipt()
    receipt["topology"]["interpreter_isolation"]["policy_interpreter_prefix_exact"] = False
    assert "policy_server_standup_policy_interpreter_prefix_inexact" in _errors(
        seal_policy_server_standup(receipt)
    )


def test_a_co_resident_jax_server_must_disable_preallocation() -> None:
    """The standalone OpenPI image sets MEM_FRACTION=0.80 because it owns the GPU."""

    receipt = _receipt("pi05_droid")
    receipt["endpoint"]["port"] = 8000  # openpi websocket default
    assert receipt["topology"]["server_framework"] == "jax"
    receipt["topology"]["accelerator_memory_guard"] = {
        "xla_python_client_preallocate": True
    }
    assert "policy_server_standup_jax_preallocation_not_disabled" in _errors(
        seal_policy_server_standup(receipt)
    )

    receipt["topology"]["accelerator_memory_guard"] = {
        "xla_python_client_preallocate": False
    }
    assert validate_policy_server_standup(seal_policy_server_standup(receipt))


def test_a_torch_candidate_needs_no_xla_guard() -> None:
    """The guard is framework-specific; GR00T and Cosmos do not preallocate this way."""

    receipt = _receipt("groot_n17_droid")
    receipt["topology"].pop("accelerator_memory_guard")
    assert validate_policy_server_standup(seal_policy_server_standup(receipt))


def test_sharing_isaacs_interpreter_requires_no_framework_install() -> None:
    """Co-installing JAX or a second torch resolves against Isaac's own pins."""

    receipt = _receipt()
    receipt["topology"] = {
        "mode": "shared_worker_shared_interpreter",
        "same_worker": True,
        "isaac_interpreter": ISAAC_INTERPRETER,
        "policy_interpreter": ISAAC_INTERPRETER,
        "policy_python_version": ISAAC_PYTHON_VERSION,
        "server_framework": CANDIDATE_SERVER_FRAMEWORKS["groot_n17_droid"],
        "accelerator_framework_installed_into_isaac_interpreter": True,
        "gpu_memory_observed_mib": {
            "total": 46_068,
            "isaac_peak": 18_400,
            "policy_server_peak": 21_100,
        },
    }
    assert "policy_server_standup_framework_installed_into_isaac_interpreter" in _errors(
        seal_policy_server_standup(receipt)
    )


def test_an_unknown_topology_fails_closed() -> None:
    """Each supported topology has a known teardown and a known failure mode."""

    receipt = _receipt()
    receipt["topology"]["mode"] = "kubernetes_sidecar"
    assert any(
        error.startswith("policy_server_standup_topology_unsupported")
        for error in _errors(seal_policy_server_standup(receipt))
    )


def test_oversubscribed_gpu_memory_fails() -> None:
    """Co-residency is the claim under test, so the arithmetic has to hold."""

    receipt = _receipt()
    receipt["topology"]["gpu_memory_observed_mib"] = {
        "total": 24_000,
        "isaac_peak": 18_400,
        "policy_server_peak": 21_100,
    }
    assert "policy_server_standup_gpu_memory_oversubscribed" in _errors(
        seal_policy_server_standup(receipt)
    )


def test_a_routable_policy_endpoint_fails() -> None:
    """A rented machine must not expose an inference port for third-party weights."""

    receipt = _receipt()
    receipt["endpoint"]["host"] = "0.0.0.0"
    assert "policy_server_standup_endpoint_not_loopback" in _errors(
        seal_policy_server_standup(receipt)
    )


def test_a_transport_the_candidates_client_cannot_speak_fails() -> None:
    """GR00T's client speaks ZMQ; serving it over a websocket strands the client."""

    receipt = _receipt("groot_n17_droid")
    receipt["endpoint"]["transport"] = "openpi_websocket_msgpack_numpy"
    assert "policy_server_standup_transport_mismatch" in _errors(
        seal_policy_server_standup(receipt)
    )


def test_an_unbound_socket_fails() -> None:
    """A started process and a reachable endpoint are different claims."""

    receipt = _receipt()
    receipt["endpoint"]["listening_socket_confirmed"] = False
    assert "policy_server_standup_endpoint_socket_unconfirmed" in _errors(
        seal_policy_server_standup(receipt)
    )


def test_starting_isaac_after_the_policy_server_fails() -> None:
    """Isaac aborts natively under VRAM pressure; the recoverable process goes second."""

    reordered = list(STARTUP_PHASES)
    isaac = reordered.index("isaac_runtime_started")
    server = reordered.index("policy_server_started")
    reordered[isaac], reordered[server] = reordered[server], reordered[isaac]
    receipt = _receipt()
    receipt["startup_phases_completed"] = reordered
    assert "policy_server_standup_startup_phase_order_invalid" in _errors(
        seal_policy_server_standup(receipt)
    )


def test_a_skipped_startup_phase_fails() -> None:
    """Every phase is evidence; a missing one is an unmeasured step, not a fast path."""

    receipt = _receipt()
    receipt["startup_phases_completed"] = [
        phase for phase in STARTUP_PHASES if phase != "checkpoint_verified"
    ]
    assert "policy_server_standup_startup_phase_order_invalid" in _errors(
        seal_policy_server_standup(receipt)
    )


def test_an_unterminated_server_or_retained_checkpoint_fails() -> None:
    """Nothing survives the instance, and the checkpoint is re-materializable."""

    receipt = _receipt()
    receipt["teardown"]["policy_server_process_terminated"] = False
    receipt["teardown"]["checkpoint_retained_on_orchestrator"] = True
    errors = _errors(seal_policy_server_standup(receipt))
    assert "policy_server_standup_policy_process_not_terminated" in errors
    assert "policy_server_standup_checkpoint_retained_on_orchestrator" in errors


def test_a_standup_asserting_a_task_outcome_fails() -> None:
    """A standup happens before the matrix, so any outcome here is asserted."""

    receipt = _receipt()
    receipt["readiness"]["inference_round_trip"]["task_success"] = True
    errors = _errors(seal_policy_server_standup(receipt))
    assert "policy_server_standup_caller_asserted_outcome_forbidden" in errors


def test_outcomes_observed_flag_must_be_false() -> None:
    """Standing a server up after outcomes exist would allow post-hoc swapping."""

    assert "policy_server_standup_after_task_outcomes" in _errors(
        _receipt(task_outcomes_observed=True)
    )


def test_a_tampered_receipt_fails_its_digest() -> None:
    """The digest is what makes the receipt quotable in a sealed protocol."""

    receipt = _receipt()
    receipt["endpoint"]["port"] = 5556
    assert "policy_server_standup_digest_mismatch" in _errors(receipt)


def test_blockers_and_readiness_cannot_coexist() -> None:
    """A receipt that names its own blocker is not a passing standup."""

    assert "policy_server_standup_has_blockers" in _errors(
        _receipt(blockers=["policy_server_oom"])
    )


def test_an_unknown_candidate_is_reported_alone() -> None:
    """Everything downstream is keyed on the candidate, so nothing else is derivable."""

    receipt = _receipt()
    receipt["candidate_id"] = "pi06_droid"
    errors = _errors(seal_policy_server_standup(receipt))
    assert errors == ("policy_server_standup_unknown_candidate:pi06_droid",)


def test_binding_to_an_inventory_candidate_row_catches_reuse() -> None:
    """A re-audited candidate gets a new digest; an old standup must not survive it."""

    receipt = _receipt()
    stale = {"candidate_id": "groot_n17_droid", "candidate_digest": _DIGEST_C}
    assert "policy_server_standup_candidate_digest_mismatch" in _errors(
        receipt, candidate=stale
    )
    fresh = {"candidate_id": "groot_n17_droid", "candidate_digest": _DIGEST_A}
    assert validate_policy_server_standup(receipt, candidate=fresh)["candidate_digest"] == (
        _DIGEST_A
    )


@pytest.mark.parametrize("candidate_id", sorted(EXPECTED_CANDIDATES))
def test_every_frozen_candidate_has_a_transport_and_framework(candidate_id: str) -> None:
    """A candidate the inventory admits but this module cannot serve is a silent gap."""

    plan = describe_standup_plan(candidate_id)
    assert plan["transport"] == CANDIDATE_TRANSPORTS[candidate_id]
    assert plan["server_framework"] in {"jax", "torch"}
    assert plan["checkpoint_total_bytes"] == (
        EXPECTED_CANDIDATES[candidate_id]["checkpoint_total_bytes"]
    )
    assert plan["startup_phases"] == list(STARTUP_PHASES)


def test_the_openpi_candidates_are_the_jax_ones() -> None:
    """pi05 is the only JAX server; the topology hinges on that split."""

    assert CANDIDATE_SERVER_FRAMEWORKS["pi05_droid"] == "jax"
    assert {
        candidate
        for candidate, framework in CANDIDATE_SERVER_FRAMEWORKS.items()
        if framework == "torch"
    } == {"groot_n17_droid", "groot_n16_droid", "cosmos3_edge_policy_droid"}


def test_the_forbidden_outcome_vocabulary_matches_the_admission_contract() -> None:
    """Two outcome vocabularies that drift apart would leave one surface permissive."""

    from blueprint_pipeline import adp009d_policy_candidate_admission as admission

    assert FORBIDDEN_OUTCOME_KEYS == frozenset(admission._FORBIDDEN_OUTCOME_KEYS)


def test_describe_standup_plan_rejects_an_unknown_candidate() -> None:
    """The plan is per-candidate measured fact, never a default."""

    with pytest.raises(Adp009dPolicyServerStandupError):
        describe_standup_plan("some_other_policy")
