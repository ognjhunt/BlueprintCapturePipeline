from __future__ import annotations

import os
import subprocess
from pathlib import Path

from blueprint_pipeline.vast_provider_transfer_upload import (
    EXPECTED_PROVIDER_UPLOAD_BYTES_ENV,
    provider_output_upload_shell_fragment,
)


def _run_upload_guard(
    *, tmp_path: Path, declared_bytes: str, payload: bytes
) -> subprocess.CompletedProcess[str]:
    archive = tmp_path / "provider-output.zip"
    archive.write_bytes(payload)
    env = dict(os.environ)
    env[EXPECTED_PROVIDER_UPLOAD_BYTES_ENV] = declared_bytes
    return subprocess.run(
        [
            "bash",
            "-c",
            provider_output_upload_shell_fragment()
            + 'blueprint_upload_put "https://unused.invalid/output.zip" "$1"; '
            + 'printf "UPLOAD_RC:%s\\n" "$?"',
            "upload-guard",
            str(archive),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )


def test_provider_output_upload_refuses_bytes_above_the_priced_ceiling(
    tmp_path: Path,
) -> None:
    """No provider may upload more bytes than admission priced."""

    result = _run_upload_guard(
        tmp_path=tmp_path,
        declared_bytes="4",
        payload=b"12345",
    )

    assert "provider_output_zip_exceeds_declared_transfer_ceiling" in result.stdout
    assert "UPLOAD_RC:86" in result.stdout


def test_provider_output_upload_refuses_an_invalid_declared_ceiling(
    tmp_path: Path,
) -> None:
    result = _run_upload_guard(
        tmp_path=tmp_path,
        declared_bytes="not-an-integer",
        payload=b"1234",
    )

    assert "provider_output_transfer_ceiling_invalid" in result.stdout
    assert "UPLOAD_RC:86" in result.stdout


def test_provider_output_upload_guard_is_valid_bash() -> None:
    subprocess.run(
        ["bash", "-n"],
        input=provider_output_upload_shell_fragment(),
        check=True,
        capture_output=True,
        text=True,
    )
