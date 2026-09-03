from __future__ import annotations

import os
import shlex
import subprocess
import time
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


def _install_fake_transport(tmp_path: Path) -> tuple[Path, Path, Path]:
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    attempts_path = tmp_path / "curl-attempts.txt"
    outcomes_path = tmp_path / "curl-outcomes.txt"
    curl_path = fake_bin / "curl"
    curl_path.write_text(
        """#!/usr/bin/env bash
set -eu
attempts_path=${BLUEPRINT_TEST_CURL_ATTEMPTS:?}
outcomes_path=${BLUEPRINT_TEST_CURL_OUTCOMES:?}
attempt=$(( $(wc -l < "$attempts_path") + 1 ))
outcome=$(sed -n "${attempt}p" "$outcomes_path")
rc=${outcome%% *}
status=${outcome#* }
upload_arg=""
url=""
expect_upload_path=0
for arg in "$@"; do
  if [ "$expect_upload_path" = 1 ]; then
    upload_arg=$arg
    expect_upload_path=0
    continue
  fi
  case "$arg" in
    --upload-file|-T) expect_upload_path=1 ;;
    @*) printf 'whole-body-upload-forbidden\n' >&2; exit 97 ;;
    https://*) url=$arg ;;
  esac
done
[ -n "$upload_arg" ] || { printf 'streaming-upload-path-missing\n' >&2; exit 98; }
upload_sha=$(sha256sum "$upload_arg" | cut -d" " -f1)
url_sha=$(printf '%s' "$url" | sha256sum | cut -d" " -f1)
printf '%s %s %s\n' "$attempt" "$upload_sha" "$url_sha" >> "$attempts_path"
if [ "${BLUEPRINT_TEST_MUTATE_AFTER_FIRST:-0}" = 1 ] && [ "$attempt" -eq 1 ]; then
  printf 'changed-after-first-attempt' > "$upload_arg"
fi
printf '%s' "$status"
exit "$rc"
""",
        encoding="utf-8",
    )
    curl_path.chmod(0o755)
    sleep_path = fake_bin / "sleep"
    sleep_path.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    sleep_path.chmod(0o755)
    attempts_path.write_text("", encoding="utf-8")
    return fake_bin, attempts_path, outcomes_path


def _run_upload_with_fake_transport(
    *,
    tmp_path: Path,
    outcomes: list[tuple[int, str]],
    mutate_after_first: bool = False,
    parent_deadline_epoch: int | float | None = None,
) -> tuple[subprocess.CompletedProcess[str], list[str]]:
    archive = tmp_path / "provider-output.zip"
    archive.write_bytes(b"immutable-provider-output")
    fake_bin, attempts_path, outcomes_path = _install_fake_transport(tmp_path)
    outcomes_path.write_text(
        "".join(f"{rc} {status}\n" for rc, status in outcomes),
        encoding="utf-8",
    )
    env = dict(os.environ)
    env[EXPECTED_PROVIDER_UPLOAD_BYTES_ENV] = str(archive.stat().st_size)
    env["BLUEPRINT_TEST_CURL_ATTEMPTS"] = str(attempts_path)
    env["BLUEPRINT_TEST_CURL_OUTCOMES"] = str(outcomes_path)
    env["BLUEPRINT_TEST_MUTATE_AFTER_FIRST"] = "1" if mutate_after_first else "0"
    env["PATH"] = f"{fake_bin}:{env['PATH']}"
    if parent_deadline_epoch is not None:
        env["BLUEPRINT_SCENE_CONFIGURATION_PARENT_DEADLINE_EPOCH"] = str(
            parent_deadline_epoch
        )
    command = (
        provider_output_upload_shell_fragment()
        + f"blueprint_upload_put {shlex.quote('https://signed.invalid/output.zip?secret=never-log')} \"$1\"; "
        + 'printf "UPLOAD_RC:%s\\n" "$?"'
    )
    result = subprocess.run(
        ["bash", "-c", command, "upload-transport", str(archive)],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    return result, attempts_path.read_text(encoding="utf-8").splitlines()


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


def test_provider_output_upload_retries_same_file_and_url_after_transient_transport(
    tmp_path: Path,
) -> None:
    result, attempts = _run_upload_with_fake_transport(
        tmp_path=tmp_path,
        outcomes=[(28, "000"), (0, "200")],
    )

    assert "BLUEPRINT_VAST_PROVIDER_OUTPUT_UPLOAD_TRANSPORT_RETRY:1" in result.stdout
    assert "UPLOAD_RC:0" in result.stdout
    assert len(attempts) == 2
    assert attempts[0].split()[1:] == attempts[1].split()[1:]
    assert "secret=never-log" not in result.stdout
    assert "secret=never-log" not in result.stderr


def test_provider_output_upload_retries_transient_http_failure(tmp_path: Path) -> None:
    result, attempts = _run_upload_with_fake_transport(
        tmp_path=tmp_path,
        outcomes=[(22, "503"), (0, "200")],
    )

    assert "UPLOAD_RC:0" in result.stdout
    assert len(attempts) == 2


def test_provider_output_upload_does_not_retry_nontransient_http_failure(
    tmp_path: Path,
) -> None:
    result, attempts = _run_upload_with_fake_transport(
        tmp_path=tmp_path,
        outcomes=[(22, "403"), (0, "200")],
    )

    assert "provider_output_upload_nontransient_failure:403:22" in result.stdout
    assert "UPLOAD_RC:22" in result.stdout
    assert len(attempts) == 1


def test_provider_output_upload_refuses_ambiguous_success_status(
    tmp_path: Path,
) -> None:
    result, attempts = _run_upload_with_fake_transport(
        tmp_path=tmp_path,
        outcomes=[(0, "000"), (0, "200")],
    )

    assert "provider_output_upload_nontransient_failure:000:0" in result.stdout
    assert "UPLOAD_RC:86" in result.stdout
    assert len(attempts) == 1


def test_provider_output_upload_stops_after_bounded_transient_retries(
    tmp_path: Path,
) -> None:
    result, attempts = _run_upload_with_fake_transport(
        tmp_path=tmp_path,
        outcomes=[(28, "000"), (22, "503"), (28, "000")],
    )

    assert "provider_output_upload_transient_retries_exhausted" in result.stdout
    assert "UPLOAD_RC:28" in result.stdout
    assert len(attempts) == 3


def test_provider_output_upload_refuses_changed_file_before_retry(
    tmp_path: Path,
) -> None:
    result, attempts = _run_upload_with_fake_transport(
        tmp_path=tmp_path,
        outcomes=[(28, "000"), (0, "200")],
        mutate_after_first=True,
    )

    assert "provider_output_zip_changed_during_upload_retry" in result.stdout
    assert "UPLOAD_RC:86" in result.stdout
    assert len(attempts) == 1


def test_provider_output_upload_refuses_expired_parent_deadline(
    tmp_path: Path,
) -> None:
    result, attempts = _run_upload_with_fake_transport(
        tmp_path=tmp_path,
        outcomes=[(0, "200")],
        parent_deadline_epoch=int(time.time()) + 59,
    )

    assert "provider_output_upload_deadline_exhausted" in result.stdout
    assert "UPLOAD_RC:86" in result.stdout
    assert attempts == []


def test_provider_output_upload_accepts_scene_watchdog_float_deadline(
    tmp_path: Path,
) -> None:
    result, attempts = _run_upload_with_fake_transport(
        tmp_path=tmp_path,
        outcomes=[(0, "200")],
        parent_deadline_epoch=time.time() + 120.5,
    )

    assert "provider_output_upload_deadline_invalid" not in result.stdout
    assert "UPLOAD_RC:0" in result.stdout
    assert len(attempts) == 1


def test_provider_output_upload_has_no_whole_file_python_fallback() -> None:
    fragment = provider_output_upload_shell_fragment()

    assert "handle.read()" not in fragment
    assert "urllib.request" not in fragment
    assert "provider_output_upload_transport_unavailable" in fragment
    assert "--http1.1" in fragment
    assert "--connect-timeout 30" in fragment
    assert '--max-time "$blueprint_upload_remaining"' in fragment
    assert "blueprint_upload_max_attempts=3" in fragment
    assert '--upload-file "$blueprint_upload_path"' in fragment
    assert "--data-binary" not in fragment


def test_provider_output_upload_streams_sparse_archive_larger_than_two_gib(
    tmp_path: Path,
) -> None:
    """The transport passes a path to curl and never materializes archive bytes."""

    archive = tmp_path / "provider-output-over-two-gib.zip"
    archive_size = 2_312_630_447
    with archive.open("wb") as handle:
        handle.truncate(archive_size)

    fake_bin, attempts_path, outcomes_path = _install_fake_transport(tmp_path)
    outcomes_path.write_text("0 200\n", encoding="utf-8")
    fake_sha256sum = fake_bin / "sha256sum"
    fake_sha256sum.write_text(
        "#!/usr/bin/env bash\n"
        "printf '0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef  %s\\n' \"$1\"\n",
        encoding="utf-8",
    )
    fake_sha256sum.chmod(0o755)
    env = dict(os.environ)
    env[EXPECTED_PROVIDER_UPLOAD_BYTES_ENV] = str(archive_size)
    env["BLUEPRINT_TEST_CURL_ATTEMPTS"] = str(attempts_path)
    env["BLUEPRINT_TEST_CURL_OUTCOMES"] = str(outcomes_path)
    env["BLUEPRINT_TEST_MUTATE_AFTER_FIRST"] = "0"
    env["PATH"] = f"{fake_bin}:{env['PATH']}"

    result = subprocess.run(
        [
            "bash",
            "-c",
            provider_output_upload_shell_fragment()
            + 'blueprint_upload_put "https://signed.invalid/output.zip" "$1"; '
            + 'printf "UPLOAD_RC:%s\\n" "$?"',
            "upload-transport",
            str(archive),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    assert "UPLOAD_RC:0" in result.stdout
    assert len(attempts_path.read_text(encoding="utf-8").splitlines()) == 1
