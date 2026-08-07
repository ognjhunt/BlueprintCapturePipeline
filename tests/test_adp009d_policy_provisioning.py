from __future__ import annotations

import pytest

from blueprint_pipeline.adp009d_policy_candidate_admission import EXPECTED_CANDIDATES
from blueprint_pipeline.adp009d_policy_provisioning import (
    BLOCKER_CREDENTIALS,
    BLOCKER_JAX_PREALLOCATION,
    BLOCKER_NOT_LOOPBACK,
    BLOCKER_SHARED_INTERPRETER,
    ISAAC_INTERPRETER,
    PolicyProvisioningError,
    build_provisioning_script,
    describe_provisioning,
    validate_provisioning,
)


def test_the_policy_environment_is_built_beside_isaac_never_inside_it() -> None:
    """Pip-resolving against Isaac's own CPython is how Isaac stops starting."""

    script = build_provisioning_script("pi05_droid")

    assert "python3 -m venv" in script
    # Isaac's interpreter must never be the thing pip is pointed at.
    assert f'"{ISAAC_INTERPRETER}" -m pip' not in script
    assert "/opt/adp009d-policy-venv/bin/python" in script

    receipt = describe_provisioning("pi05_droid")
    assert receipt["policy_interpreter"] != ISAAC_INTERPRETER
    assert validate_provisioning(receipt) == []


def test_jax_preallocation_is_disabled_outright() -> None:
    """Co-resident, an 0.80 fraction kills Isaac as an uncatchable native abort."""

    script = build_provisioning_script("pi05_droid")
    assert 'export XLA_PYTHON_CLIENT_PREALLOCATE="false"' in script

    receipt = describe_provisioning("pi05_droid")
    assert receipt["jax_environment"]["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"

    # A merely smaller fraction narrows the race rather than removing it.
    narrowed = dict(receipt)
    narrowed["jax_environment"] = {"XLA_PYTHON_CLIENT_PREALLOCATE": "0.2"}
    assert BLOCKER_JAX_PREALLOCATION in validate_provisioning(narrowed)

    enabled = dict(receipt)
    enabled["jax_environment"] = {"XLA_PYTHON_CLIENT_PREALLOCATE": "true"}
    assert BLOCKER_JAX_PREALLOCATION in validate_provisioning(enabled)


def test_no_credential_is_forwarded_because_every_candidate_is_public() -> None:
    script = build_provisioning_script("groot_n17_droid")

    assert "unset HF_TOKEN" in script
    # The GCS fetch disables credential lookup explicitly.
    gcs = build_provisioning_script("pi05_droid")
    assert "gcloud storage cp -r -u" in gcs

    receipt = describe_provisioning("pi05_droid")
    leaked = dict(receipt)
    leaked["credentials_forwarded"] = True
    assert BLOCKER_CREDENTIALS in validate_provisioning(leaked)


def test_each_candidate_fetches_from_where_its_artifact_actually_lives() -> None:
    gcs = build_provisioning_script("pi05_droid")
    assert "gs://openpi-assets/checkpoints/pi05_droid" in gcs
    assert "huggingface_cli" not in gcs

    hub = build_provisioning_script("groot_n17_droid")
    assert "nvidia/GR00T-N1.7-DROID" in hub
    assert "05e7cc97e40dbd33b0890c35cc0214fcb0547ab5" in hub
    assert "gcloud storage" not in hub


def test_every_frozen_candidate_provisions_and_validates() -> None:
    for candidate_id in EXPECTED_CANDIDATES:
        script = build_provisioning_script(candidate_id)
        assert script.startswith("#!/usr/bin/env bash")
        assert "set -euo pipefail" in script
        receipt = describe_provisioning(candidate_id)
        assert validate_provisioning(receipt) == []
        assert receipt["materialize_on"] == "gpu_worker"


def test_a_non_loopback_endpoint_is_refused() -> None:
    receipt = dict(describe_provisioning("pi05_droid"))
    receipt["endpoint_host"] = "0.0.0.0"
    assert BLOCKER_NOT_LOOPBACK in validate_provisioning(receipt)


def test_sharing_isaacs_interpreter_is_refused() -> None:
    receipt = dict(describe_provisioning("pi05_droid"))
    receipt["policy_interpreter"] = ISAAC_INTERPRETER
    assert BLOCKER_SHARED_INTERPRETER in validate_provisioning(receipt)

    nested = dict(describe_provisioning("pi05_droid"))
    nested["policy_interpreter"] = "/isaac-sim/kit/python/bin/python3"
    assert BLOCKER_SHARED_INTERPRETER in validate_provisioning(nested)


def test_unknown_candidates_are_refused_everywhere() -> None:
    with pytest.raises(PolicyProvisioningError):
        build_provisioning_script("some_other_policy")
    with pytest.raises(PolicyProvisioningError):
        describe_provisioning("some_other_policy")
    assert validate_provisioning({"candidate_id": "some_other_policy"})


def test_provisioning_agrees_with_the_standup_and_materialization_contracts() -> None:
    """Three modules describe one worker; they must not disagree about it."""

    from blueprint_pipeline.adp009d_checkpoint_materialization import (
        plan_checkpoint_materialization,
    )
    from blueprint_pipeline.adp009d_policy_server_standup import describe_standup_plan

    for candidate_id in ("pi05_droid", "groot_n17_droid"):
        provisioning = describe_provisioning(candidate_id)
        materialization = plan_checkpoint_materialization(candidate_id)
        standup = describe_standup_plan(candidate_id)

        assert (
            provisioning["checkpoint_repository"]
            == materialization["checkpoint_repository"]
            == standup["checkpoint_repository"]
        )
        assert (
            provisioning["checkpoint_revision"]
            == materialization["checkpoint_revision"]
            == standup["checkpoint_revision"]
        )
        assert provisioning["materialize_on"] == standup["checkpoint_materialized_on"]
        assert provisioning["isaac_interpreter"] == standup["isaac_interpreter"]
