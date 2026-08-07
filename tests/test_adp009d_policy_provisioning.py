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

    assert "venv --python" in script
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
    # The GCS fetch goes over plain HTTPS with no SDK and no credential.
    gcs = build_provisioning_script("pi05_droid")
    assert "adp009d_checkpoint_fetch_worker.py" in gcs
    assert "gcloud" not in gcs

    receipt = describe_provisioning("pi05_droid")
    leaked = dict(receipt)
    leaked["credentials_forwarded"] = True
    assert BLOCKER_CREDENTIALS in validate_provisioning(leaked)


def test_each_candidate_fetches_from_where_its_artifact_actually_lives() -> None:
    gcs = build_provisioning_script("pi05_droid")
    assert "gs://openpi-assets/checkpoints/pi05_droid" in gcs
    assert "huggingface_cli" not in gcs
    assert "gcloud" not in gcs

    hub = build_provisioning_script("groot_n17_droid")
    assert "nvidia/GR00T-N1.7-DROID" in hub
    assert "05e7cc97e40dbd33b0890c35cc0214fcb0547ab5" in hub
    assert "gcloud" not in hub


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


def test_the_venv_is_built_from_the_measured_system_interpreter() -> None:
    """A worker run measured both interpreters; use the one outside Isaac's prefix."""

    from blueprint_pipeline.adp009d_policy_provisioning import (
        ISAAC_INTERPRETER,
        SYSTEM_INTERPRETER,
    )

    script = build_provisioning_script("pi05_droid")

    assert f'venv --python "{SYSTEM_INTERPRETER}"' in script
    assert SYSTEM_INTERPRETER == "/usr/bin/python3"
    # Never Isaac's own interpreter, and never its measured real path either.
    assert ISAAC_INTERPRETER not in script.split('# Prove')[0]
    assert "/isaac-sim/kit/python/bin/python3" not in script


def test_the_pinned_policy_source_is_verified_not_merely_cloned() -> None:
    """A moved branch must not silently change what runs."""

    script = build_provisioning_script("pi05_droid")
    revision = EXPECTED_CANDIDATES["pi05_droid"]["source_revision"]

    assert "github.com/Physical-Intelligence/openpi" in script
    assert f'origin "{revision}"' in script
    assert "checkout --detach FETCH_HEAD" in script
    # The checkout is asserted, so a fetch that landed elsewhere fails the run.
    assert f'rev-parse HEAD)" = "{revision}"' in script


def test_the_install_precedes_the_checkpoint_fetch() -> None:
    """A dependency failure should surface in seconds, not after 12.4 GB."""

    script = build_provisioning_script("pi05_droid")

    assert script.index("pip install -e") < script.index(
        "adp009d_checkpoint_fetch_worker.py"
    )


def test_every_candidate_installs_its_own_pinned_source() -> None:
    for candidate_id, expected in EXPECTED_CANDIDATES.items():
        script = build_provisioning_script(candidate_id)
        assert str(expected["source_repository"]) in script
        assert str(expected["source_revision"]) in script


def test_uv_creates_the_environment_because_the_image_lacks_ensurepip() -> None:
    """Two measured failures put uv on the primary path, not a fallback.

    This image's /usr/bin/python3 has no ensurepip, so a plain venv cannot be
    created at all; and pip could not resolve openpi, failing with
    resolution-too-deep after backtracking through tensorstore releases.
    openpi is itself a uv project, so uv installs it as packaged.
    """

    script = build_provisioning_script("pi05_droid")

    assert 'curl -LsSf https://astral.sh/uv/install.sh' in script
    assert '"$UV" venv --python' in script
    # pip is not used to create the environment at all.
    assert "-m venv" not in script


def test_the_venv_is_proven_real_and_not_isaacs_before_installing() -> None:
    """Installing into a venv that silently is Isaac's is the failure to avoid."""

    script = build_provisioning_script("pi05_droid")

    assert 'test -x "/opt/adp009d-policy-venv/bin/python"' in script
    assert "'isaac-sim' not in sys.prefix" in script
    # And the proof precedes any install.
    assert script.index("not in sys.prefix") < script.index("pip install -e")


def test_build_isolation_is_left_enabled() -> None:
    """--no-build-isolation is an instruction not to fetch the build backend.

    A live run disabled it and pip failed with
    BackendUnavailable: Cannot import 'hatchling.build' -- the backend openpi's
    pyproject declares.  Isolation must stay on so pip provisions it.
    """

    for candidate_id in EXPECTED_CANDIDATES:
        script = build_provisioning_script(candidate_id)
        assert "--no-build-isolation" not in script
        assert '"$UV" pip install -e' in script


def test_native_build_dependencies_are_installed_before_the_policy() -> None:
    """uv resolved openpi, then evdev failed to compile for want of kernel headers.

    The chain the failure named is openpi -> lerobot -> pynput -> evdev, pulled
    in for input-device handling inference never uses but which the graph still
    requires to build.
    """

    script = build_provisioning_script("pi05_droid")

    # The full standard C-extension build set, installed at once: iterating one
    # package per run costs a paid GPU run each time.  A live run cleared
    # linux/input.h and then failed on Python.h one package later.
    for package in (
        "linux-libc-dev",
        "build-essential",
        "python3-dev",
        "python3.12-dev",
        "pkg-config",
    ):
        assert package in script
    # Headers must be present before uv is asked to build anything.
    assert script.index("linux-libc-dev") < script.index("pip install -e")
    # And never fatal on their own: the apt step is best-effort.
    # Best-effort: a missing apt must not fail provisioning on its own.
    assert "|| true" in script[script.index("apt-get install") :][:400]
