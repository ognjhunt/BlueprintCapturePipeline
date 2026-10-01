"""Safety contract for the Windows GPU trainer lane on EC2.

Postshot has no Linux build and no service API, so its arm cannot use the
Linux/Docker bootstrap every other lane shares.  These tests pin the properties
that make a Windows trainer host safe to allocate rather than merely possible.
"""

from __future__ import annotations

import base64
import gzip

import pytest

from blueprint_pipeline.cloud_vm_render_providers import (
    WINDOWS_WORKER_PLATFORM,
    _windows_worker_bootstrap,
)
from blueprint_pipeline.gpu_render_providers import RenderLaunchSpec


def _spec(**env: str) -> RenderLaunchSpec:
    base = {
        "BLUEPRINT_WORKER_IMAGE_DIGEST": "blueprint-postshot-host@sha256:" + "a" * 64,
        "BLUEPRINT_WORKER_HARD_TTL_SECONDS": "5400",
        "BLUEPRINT_POSTSHOT_LICENCE_GET_URL": "https://example.invalid/signed-licence",
        "BLUEPRINT_RECONSTRUCTION_INPUT_BUNDLE_GET_URL": "https://example.invalid/signed-bundle",
        "BLUEPRINT_RECONSTRUCTION_INPUT_BUNDLE_DIGEST": "sha256:" + "f" * 64,
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


def _decoded_user_data(script: str) -> str:
    decoded = ""
    for token in script.split('"'):
        try:
            raw = base64.b64decode(token)
            if raw.startswith(b"\x1f\x8b"):
                raw = gzip.decompress(raw)
            decoded += raw.decode("utf-8", errors="ignore")
        except Exception:  # noqa: BLE001
            continue
    return script + decoded


# --------------------------------------------------------------------------
# The licence must never cross the UserData boundary
# --------------------------------------------------------------------------


def test_bootstrap_refuses_a_credential_instead_of_embedding_it() -> None:
    """UserData is readable over IMDS and via DescribeInstanceAttribute, so a
    credential must never reach it — and refusing beats silently dropping."""
    with pytest.raises(ValueError) as excinfo:
        _windows_worker_bootstrap(_spec(POSTSHOT_LOGIN_PASSWORD="correct-horse-battery-staple"))
    assert "refuses_credential_in_user_data" in str(excinfo.value)
    assert "correct-horse-battery-staple" not in str(excinfo.value)


def test_bootstrap_carries_only_the_signed_licence_fetch_url() -> None:
    script = _windows_worker_bootstrap(_spec())
    haystack = _decoded_user_data(script).lower()
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






# --------------------------------------------------------------------------
# Exactly-once allocation
# --------------------------------------------------------------------------






# --------------------------------------------------------------------------
# Platform selection is explicit and fail-closed
# --------------------------------------------------------------------------












DRIVER_URL = "https://example.invalid/nvidia-datacenter.exe"
INSTALLER_URL = "https://example.invalid/Postshot.msi"
PYTHON_URL = "https://example.invalid/python-embed.zip"
NUMPY_URL = "https://example.invalid/numpy.whl"
DRIVER_SHA = "a" * 64
INSTALLER_SHA = "b" * 64
PYTHON_SHA = "c" * 64
NUMPY_SHA = "d" * 64


def _install_at_boot_spec(**extra: str) -> RenderLaunchSpec:
    from blueprint_pipeline.cloud_vm_render_providers import (
        WINDOWS_DRIVER_URL_ENV,
        WINDOWS_DRIVER_SHA256_ENV,
        WINDOWS_INSTALLER_SHA256_ENV,
        WINDOWS_INSTALLER_URL_ENV,
        WINDOWS_NUMPY_SHA256_ENV,
        WINDOWS_NUMPY_URL_ENV,
        WINDOWS_PYTHON_SHA256_ENV,
        WINDOWS_PYTHON_URL_ENV,
    )

    env = {
        WINDOWS_DRIVER_URL_ENV: DRIVER_URL,
        WINDOWS_DRIVER_SHA256_ENV: DRIVER_SHA,
        WINDOWS_INSTALLER_URL_ENV: INSTALLER_URL,
        WINDOWS_INSTALLER_SHA256_ENV: INSTALLER_SHA,
        WINDOWS_PYTHON_URL_ENV: PYTHON_URL,
        WINDOWS_PYTHON_SHA256_ENV: PYTHON_SHA,
        WINDOWS_NUMPY_URL_ENV: NUMPY_URL,
        WINDOWS_NUMPY_SHA256_ENV: NUMPY_SHA,
    }
    env.update(extra)
    return _spec(**env)


def test_without_installer_urls_the_host_must_already_be_baked() -> None:
    script = _windows_worker_bootstrap(_spec())
    assert "blueprint_worker_image_marker_missing" in script
    assert "msiexec" not in script








def test_install_at_boot_pins_the_installer_digest() -> None:
    """An unverified MSI decides which binary a paid instance executes."""
    script = _windows_worker_bootstrap(_install_at_boot_spec())
    assert INSTALLER_SHA in _decoded_user_data(script)
    assert "postshot_installer_digest_mismatch" in script


def test_installer_url_without_a_digest_is_refused() -> None:
    from blueprint_pipeline.cloud_vm_render_providers import (
        WINDOWS_INSTALLER_SHA256_ENV,
    )

    spec = _install_at_boot_spec()
    del spec.env[WINDOWS_INSTALLER_SHA256_ENV]
    with pytest.raises(ValueError) as excinfo:
        _windows_worker_bootstrap(spec)
    assert "runtime_digest_invalid" in str(excinfo.value)


def test_a_malformed_installer_digest_is_refused() -> None:
    with pytest.raises(ValueError) as excinfo:
        _windows_worker_bootstrap(
            _install_at_boot_spec(
                **{
                    "BLUEPRINT_WINDOWS_POSTSHOT_INSTALLER_SHA256": "not-a-digest",
                }
            )
        )
    assert "runtime_digest_invalid" in str(excinfo.value)


@pytest.mark.parametrize(
    "name",
    [
        "BLUEPRINT_WINDOWS_NVIDIA_DRIVER_SHA256",
        "BLUEPRINT_WINDOWS_PYTHON_EMBED_SHA256",
        "BLUEPRINT_WINDOWS_NUMPY_WHEEL_SHA256",
    ],
)
def test_every_executable_runtime_layer_requires_a_digest(name: str) -> None:
    spec = _install_at_boot_spec()
    del spec.env[name]
    with pytest.raises(ValueError, match="runtime_digest_invalid"):
        _windows_worker_bootstrap(spec)


def test_install_at_boot_still_arms_the_hard_deadline() -> None:
    """Provisioning eats the paid window; the deadline must still bound it."""
    script = _windows_worker_bootstrap(_install_at_boot_spec())
    assert "blueprint-hard-deadline" in script
    assert "shutdown.exe" in script








def test_host_image_marker_is_verified_before_training(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A host that is not the admitted image must not start paid work."""
    script = _windows_worker_bootstrap(_spec())
    assert "blueprint_worker_image_marker_missing" in script
    assert "blueprint_worker_image_marker_mismatch" in script
