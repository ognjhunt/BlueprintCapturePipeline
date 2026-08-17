"""Safety contract for the Windows GPU trainer lane on EC2.

Postshot has no Linux build and no service API, so its arm cannot use the
Linux/Docker bootstrap every other lane shares.  These tests pin the properties
that make a Windows trainer host safe to allocate rather than merely possible.
"""

from __future__ import annotations

import base64
from pathlib import Path

import pytest

from blueprint_pipeline.cloud_vm_render_providers import (
    WINDOWS_WORKER_PLATFORM,
    AWSRenderProvider,
    _windows_worker_bootstrap,
)
from blueprint_pipeline.gpu_render_providers import RenderLaunchSpec


def _spec(**env: str) -> RenderLaunchSpec:
    base = {
        "BLUEPRINT_WORKER_IMAGE_DIGEST": "blueprint-postshot-host@sha256:" + "a" * 64,
        "BLUEPRINT_WORKER_HARD_TTL_SECONDS": "5400",
        "BLUEPRINT_POSTSHOT_LICENCE_GET_URL": "https://example.invalid/signed-licence",
    }
    base.update(env)
    return RenderLaunchSpec(
        name="blueprint-postshot-primary-001",
        image="blueprint-postshot-host@sha256:" + "a" * 64,
        env=base,
        bootstrap_argv=["-lc", "run-arm"],
    )


def _aws_env(monkeypatch: pytest.MonkeyPatch, **overrides: str) -> None:
    values = {
        "BLUEPRINT_AWS_REGION": "us-east-1",
        "BLUEPRINT_AWS_ACCOUNT_ID": "111710313013",
        "BLUEPRINT_AWS_INSTANCE_TYPE": "g6.xlarge",
        "BLUEPRINT_AWS_AMI_ID": "ami-0ed0165f19a049904",
        "BLUEPRINT_AWS_SUBNET_ID": "subnet-abc123",
        "BLUEPRINT_AWS_SECURITY_GROUP_IDS": "sg-abc123",
        "BLUEPRINT_AWS_IAM_INSTANCE_PROFILE_ARN": "arn:aws:iam::111710313013:instance-profile/blueprint-worker",
        "BLUEPRINT_AWS_HOURLY_RATE_USD": "1.05",
        "BLUEPRINT_AWS_MAX_HOURLY_RATE_USD": "1.50",
        "BLUEPRINT_AWS_WORKER_PLATFORM": WINDOWS_WORKER_PLATFORM,
    }
    values.update(overrides)
    for key, value in values.items():
        monkeypatch.setenv(key, value)


# --------------------------------------------------------------------------
# The licence must never cross the UserData boundary
# --------------------------------------------------------------------------


def test_bootstrap_refuses_a_credential_instead_of_embedding_it() -> None:
    """UserData is readable over IMDS and via DescribeInstanceAttribute, so a
    credential must never reach it — and refusing beats silently dropping."""
    with pytest.raises(ValueError) as excinfo:
        _windows_worker_bootstrap(
            _spec(POSTSHOT_LOGIN_PASSWORD="correct-horse-battery-staple")
        )
    assert "refuses_credential_in_user_data" in str(excinfo.value)
    assert "correct-horse-battery-staple" not in str(excinfo.value)


def test_bootstrap_carries_only_the_signed_licence_fetch_url() -> None:
    script = _windows_worker_bootstrap(_spec())
    decoded = ""
    for token in script.split('"'):
        try:
            decoded += base64.b64decode(token).decode("utf-8", errors="ignore")
        except Exception:  # noqa: BLE001
            continue
    haystack = (script + decoded).lower()
    assert "signed-licence" in haystack
    for fragment in ("password", "private_key", "secret"):
        assert fragment not in haystack


def test_bootstrap_does_not_persist_user_data() -> None:
    """A persisted script would re-run the paid trainer on every boot."""
    assert "<persist>false</persist>" in _windows_worker_bootstrap(_spec())


# --------------------------------------------------------------------------
# Spend is bounded by the host itself, not only by the controller
# --------------------------------------------------------------------------


def test_bootstrap_arms_a_local_hard_deadline() -> None:
    script = _windows_worker_bootstrap(_spec(BLUEPRINT_WORKER_HARD_TTL_SECONDS="5400"))
    assert "blueprint-hard-deadline" in script
    assert "shutdown.exe" in script
    assert "5400" in script


def test_launch_request_terminates_on_instance_initiated_shutdown(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _aws_env(monkeypatch)
    request = AWSRenderProvider().build_request(_spec(), tmp_path)
    assert request["run_instances"]["InstanceInitiatedShutdownBehavior"] == "terminate"


def test_root_volume_is_encrypted_and_deleted_with_the_instance(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _aws_env(monkeypatch)
    ebs = AWSRenderProvider().build_request(_spec(), tmp_path)["run_instances"][
        "BlockDeviceMappings"
    ][0]["Ebs"]
    assert ebs["Encrypted"] is True
    assert ebs["DeleteOnTermination"] is True


# --------------------------------------------------------------------------
# Exactly-once allocation
# --------------------------------------------------------------------------


def test_two_identical_launches_share_one_client_token(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """EC2 deduplicates RunInstances by ClientToken, so a retried dispatch
    cannot allocate a second billable instance."""
    _aws_env(monkeypatch)
    provider = AWSRenderProvider()
    first = provider.build_request(_spec(), tmp_path)
    second = provider.build_request(_spec(), tmp_path)
    assert first["run_instances"]["ClientToken"] == second["run_instances"]["ClientToken"]


def test_a_different_run_gets_a_different_client_token(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _aws_env(monkeypatch)
    provider = AWSRenderProvider()
    first = provider.build_request(_spec(), tmp_path)
    other = RenderLaunchSpec(
        name="blueprint-postshot-primary-002",
        image=_spec().image,
        env=dict(_spec().env),
        bootstrap_argv=["-lc", "run-arm"],
    )
    assert (
        provider.build_request(other, tmp_path)["run_instances"]["ClientToken"]
        != first["run_instances"]["ClientToken"]
    )


# --------------------------------------------------------------------------
# Platform selection is explicit and fail-closed
# --------------------------------------------------------------------------


def test_windows_platform_selects_the_powershell_bootstrap(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _aws_env(monkeypatch)
    user_data = AWSRenderProvider().build_request(_spec(), tmp_path)["run_instances"][
        "UserData"
    ]
    assert user_data.startswith("<powershell>")
    assert "docker" not in user_data.lower()


def test_linux_platform_is_unchanged(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _aws_env(monkeypatch, BLUEPRINT_AWS_WORKER_PLATFORM="linux")
    user_data = AWSRenderProvider().build_request(_spec(), tmp_path)["run_instances"][
        "UserData"
    ]
    assert user_data.startswith("#!/bin/bash")


def test_unknown_platform_blocks_before_any_api_call(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _aws_env(monkeypatch, BLUEPRINT_AWS_WORKER_PLATFORM="freebsd")
    request = AWSRenderProvider().build_request(_spec(), tmp_path)
    assert "aws_worker_platform_invalid" in request["configuration_blockers"]


def test_windows_lane_refuses_a_registry_mode_it_cannot_honour(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """There is no container runtime on the trainer host."""
    _aws_env(monkeypatch, BLUEPRINT_AWS_REGISTRY_AUTH="aws_ecr")
    request = AWSRenderProvider().build_request(_spec(), tmp_path)
    assert "aws_windows_worker_registry_auth_unsupported" in request[
        "configuration_blockers"
    ]


def test_rate_above_the_authorized_ceiling_blocks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _aws_env(
        monkeypatch,
        BLUEPRINT_AWS_HOURLY_RATE_USD="4.00",
        BLUEPRINT_AWS_MAX_HOURLY_RATE_USD="1.50",
    )
    request = AWSRenderProvider().build_request(_spec(), tmp_path)
    assert "aws_hourly_rate_exceeds_cap" in request["configuration_blockers"]


def test_windows_lane_needs_no_instance_profile(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The trainer talks only to signed URLs, so a role is privilege it never uses."""
    _aws_env(monkeypatch)
    monkeypatch.delenv("BLUEPRINT_AWS_IAM_INSTANCE_PROFILE_ARN", raising=False)
    request = AWSRenderProvider().build_request(_spec(), tmp_path)
    assert request["configuration_blockers"] == []
    assert "IamInstanceProfile" not in request["run_instances"]


def test_linux_lane_still_requires_an_instance_profile(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _aws_env(monkeypatch, BLUEPRINT_AWS_WORKER_PLATFORM="linux")
    monkeypatch.delenv("BLUEPRINT_AWS_IAM_INSTANCE_PROFILE_ARN", raising=False)
    request = AWSRenderProvider().build_request(_spec(), tmp_path)
    assert "aws_iam_instance_profile_arn_missing" in request["configuration_blockers"]


def test_an_explicitly_configured_profile_is_still_attached(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Relaxing the requirement must not silently drop an operator's choice."""
    _aws_env(monkeypatch)
    request = AWSRenderProvider().build_request(_spec(), tmp_path)
    assert request["run_instances"]["IamInstanceProfile"]["Arn"].endswith(
        "instance-profile/blueprint-worker"
    )


def test_host_image_marker_is_verified_before_training(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A host that is not the admitted image must not start paid work."""
    script = _windows_worker_bootstrap(_spec())
    assert "blueprint_worker_image_marker_missing" in script
    assert "blueprint_worker_image_marker_mismatch" in script
