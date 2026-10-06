from __future__ import annotations

import os
import shlex
import subprocess
import time
from pathlib import Path

import pytest

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
            provider_output_upload_shell_fragment(scratch_root=str(tmp_path))
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
body_arg=""
url=""
expect_upload_path=0
expect_body_path=0
for arg in "$@"; do
  if [ "$expect_upload_path" = 1 ]; then
    upload_arg=$arg
    expect_upload_path=0
    continue
  fi
  if [ "$expect_body_path" = 1 ]; then
    body_arg=$arg
    expect_body_path=0
    continue
  fi
  case "$arg" in
    --upload-file|-T) expect_upload_path=1 ;;
    --output) expect_body_path=1 ;;
    @*) printf 'whole-body-upload-forbidden\n' >&2; exit 97 ;;
    https://*) url=$arg ;;
  esac
done
scratch_parent=$(dirname "$attempts_path")
scratch=""
for candidate in "$scratch_parent"/.blueprint-provider-upload.*; do
  if [ -d "$candidate" ]; then scratch=$candidate; fi
done
body_arg="$scratch/body"
if [ -n "${BLUEPRINT_TEST_SYMLINK_COMPLETION_TARGET:-}" ]; then
  mv "$scratch/blueprint_provider_upload_response.json" "$scratch/receipt.original"
  ln -s "$BLUEPRINT_TEST_SYMLINK_COMPLETION_TARGET" "$(dirname "$body_arg")/blueprint_provider_upload_response.json"
fi
if [ "${BLUEPRINT_TEST_BLOCK_COMPLETION:-0}" = 1 ]; then
  mv "$scratch/blueprint_provider_upload_response.json" "$scratch/receipt.original"
  mkdir "$(dirname "$body_arg")/blueprint_provider_upload_response.json"
fi
if [ "${BLUEPRINT_TEST_REPLACE_BODY:-0}" = 1 ]; then
  mv "$body_arg" "$body_arg.retained"
  printf foreign > "$body_arg"
fi
[ -n "$upload_arg" ] || { printf 'streaming-upload-path-missing\n' >&2; exit 98; }
upload_sha=$(sha256sum "$upload_arg" | cut -d" " -f1)
url_sha=$(printf '%s' "$url" | sha256sum | cut -d" " -f1)
printf '%s %s %s\n' "$attempt" "$upload_sha" "$url_sha" >> "$attempts_path"
if [ "${BLUEPRINT_TEST_INTERRUPT:-0}" = 1 ]; then
  kill -TERM "$PPID"
  exit 143
fi
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
    transport_environment: dict[str, str] | None = None,
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
    env.update(transport_environment or {})
    if parent_deadline_epoch is not None:
        env["BLUEPRINT_SCENE_CONFIGURATION_PARENT_DEADLINE_EPOCH"] = str(
            parent_deadline_epoch
        )
    command = (
        provider_output_upload_shell_fragment(scratch_root=str(tmp_path))
        + f"blueprint_upload_put {shlex.quote('https://signed.invalid/output.zip?secret=never-log')} \"$1\"; "
        + 'upload_rc=$?; if [ "$upload_rc" = 0 ]; then '
        + 'blueprint_upload_read_response || upload_rc=86; '
        + 'blueprint_upload_cleanup || upload_rc=86; fi; '
        + 'printf "UPLOAD_RC:%s\\n" "$upload_rc"'
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


def test_provider_output_upload_retries_timeout_after_interim_continue(
    tmp_path: Path,
) -> None:
    result, attempts = _run_upload_with_fake_transport(
        tmp_path=tmp_path,
        outcomes=[(28, "100"), (0, "200")],
    )

    assert "UPLOAD_RC:0" in result.stdout
    assert len(attempts) == 2
    assert attempts[0].split()[1:] == attempts[1].split()[1:]


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


def test_provider_output_upload_has_no_whole_file_python_fallback(tmp_path) -> None:
    fragment = provider_output_upload_shell_fragment(scratch_root=str(tmp_path))

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
            provider_output_upload_shell_fragment(scratch_root=str(tmp_path))
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


@pytest.mark.parametrize('dangling', [False, True])
def test_legacy_completion_symlink_is_never_followed_or_deleted(tmp_path, dangling):
    victim = tmp_path / 'keep'
    if not dangling:
        victim.write_bytes(b'private-canary')
    legacy = tmp_path / 'blueprint_provider_upload_response.json'
    legacy.symlink_to(victim)
    result, attempts = _run_upload_with_fake_transport(tmp_path=tmp_path, outcomes=[(0, '200')])
    assert 'UPLOAD_RC:0' in result.stdout
    assert len(attempts) == 1
    assert legacy.is_symlink()
    assert not victim.exists() if dangling else victim.read_bytes() == b'private-canary'
    assert len(list(tmp_path.glob('.blueprint-provider-upload.*'))) == 1


def test_completion_symlink_in_owned_scratch_refuses_without_touching_target(tmp_path):
    victim = tmp_path / 'keep'
    victim.write_bytes(b'private-canary')
    result, attempts = _run_upload_with_fake_transport(tmp_path=tmp_path, outcomes=[(0, '200')],
        transport_environment={'BLUEPRINT_TEST_SYMLINK_COMPLETION_TARGET': str(victim)})
    assert 'UPLOAD_RC:86' in result.stdout
    assert len(attempts) == 1
    assert victim.read_bytes() == b'private-canary'
    [scratch] = tmp_path.glob('.blueprint-provider-upload.*')
    assert (scratch / 'blueprint_provider_upload_response.json').is_symlink()


def test_foreign_replacement_is_preserved_and_cannot_publish_completion(tmp_path):
    result, attempts = _run_upload_with_fake_transport(tmp_path=tmp_path, outcomes=[(0, '200')],
        transport_environment={'BLUEPRINT_TEST_REPLACE_BODY': '1'})
    assert 'UPLOAD_RC:86' in result.stdout
    assert len(attempts) == 1
    [scratch] = tmp_path.glob('.blueprint-provider-upload.*')
    assert (scratch / 'body').read_bytes() == b'foreign'
    assert (scratch / 'blueprint_provider_upload_response.json').read_bytes() == b''


def test_repeated_success_uses_distinct_private_roots_and_closes_owned_descriptors(tmp_path):
    fake, attempts, outcomes = _install_fake_transport(tmp_path)
    outcomes.write_text('0 200\n0 200\n')
    source = tmp_path / 'payload.zip'
    source.write_bytes(b'unchanged')
    env = {**os.environ, 'PATH': f'{fake}:{os.environ["PATH"]}',
           'BLUEPRINT_TEST_CURL_ATTEMPTS': str(attempts), 'BLUEPRINT_TEST_CURL_OUTCOMES': str(outcomes)}
    program = provider_output_upload_shell_fragment(scratch_root=str(tmp_path)) + '''
for iteration in 1 2; do
  blueprint_upload_put https://fixture.invalid "$1" || exit 86
  printf 'COMPLETION:%s\n' "$BLUEPRINT_PROVIDER_UPLOAD_RESPONSE_FILE"
  python3 -c 'import os,stat,sys; p=sys.argv[1]; assert stat.S_IMODE(os.stat(p).st_mode)==0o600; assert stat.S_IMODE(os.stat(os.path.dirname(p)).st_mode)==0o700' "$BLUEPRINT_PROVIDER_UPLOAD_RESPONSE_FILE" || exit 87
  blueprint_upload_read_response || exit 88
  blueprint_upload_cleanup || exit 89
done
'''
    result = subprocess.run(['bash', '-c', program, 'test', str(source)], env=env, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr
    paths = [line.split(':', 1)[1] for line in result.stdout.splitlines() if line.startswith('COMPLETION:')]
    assert len(paths) == 2 and paths[0] != paths[1]
    roots = list(tmp_path.glob('.blueprint-provider-upload.*'))
    assert len(roots) == 2
    for root in roots:
        assert root.stat().st_mode & 0o777 == 0o700
        assert {p.name for p in root.iterdir()} == {'status', 'body', 'blueprint_provider_upload_response.json'}
        assert (root / 'status').stat().st_size == 3
        assert (root / 'body').stat().st_size == 0
        assert (root / 'blueprint_provider_upload_response.json').stat().st_size == 0
    assert source.read_bytes() == b'unchanged'
    assert len(attempts.read_text().splitlines()) == 2


def test_cleanup_closes_only_descriptors_and_preserves_replaced_root(tmp_path):
    fake, attempts, outcomes = _install_fake_transport(tmp_path)
    outcomes.write_text('0 200\n')
    source = tmp_path / 'payload.zip'
    source.write_bytes(b'unchanged')
    env = {**os.environ, 'PATH': f'{fake}:{os.environ["PATH"]}',
           'BLUEPRINT_TEST_CURL_ATTEMPTS': str(attempts), 'BLUEPRINT_TEST_CURL_OUTCOMES': str(outcomes)}
    program = provider_output_upload_shell_fragment(scratch_root=str(tmp_path)) + '''
blueprint_upload_put https://fixture.invalid "$1" || exit 85
original=$blueprint_upload_scratch
mv "$original" "$original.retained"
mkdir "$original"
printf keep > "$original/foreign"
blueprint_upload_cleanup
printf 'CLEANUP_RC:%s\n' "$?"
'''
    result = subprocess.run(['bash', '-c', program, 'test', str(source)], env=env, capture_output=True, text=True, timeout=20)
    assert 'CLEANUP_RC:0' in result.stdout
    foreign = list(tmp_path.glob('.blueprint-provider-upload.*/foreign'))
    assert len(foreign) == 1 and foreign[0].read_bytes() == b'keep'


def test_busy_descriptor_refuses_before_creating_scratch_or_transport(tmp_path):
    source = tmp_path / 'keep'
    source.write_bytes(b'unchanged')
    program = provider_output_upload_shell_fragment(scratch_root=str(tmp_path)) + '''
exec 10<"$1"
blueprint_upload_put https://fixture.invalid "$1"
printf 'UPLOAD_RC:%s\n' "$?"
cat <&10
'''
    result = subprocess.run(['bash', '-c', program, 'test', str(source)], capture_output=True, text=True, timeout=10)
    assert 'UPLOAD_RC:86' in result.stdout
    assert result.stdout.endswith('unchanged')
    assert list(tmp_path.glob('.blueprint-provider-upload.*')) == []


def test_interrupted_transport_releases_descriptors_and_retains_bounded_private_metadata(tmp_path):
    result, attempts = _run_upload_with_fake_transport(tmp_path=tmp_path, outcomes=[(0, '200')],
        transport_environment={'BLUEPRINT_TEST_INTERRUPT': '1'})
    assert 'UPLOAD_RC:143' in result.stdout
    assert len(attempts) == 1
    [scratch] = tmp_path.glob('.blueprint-provider-upload.*')
    assert sum(p.stat().st_size for p in scratch.iterdir()) <= 3
    assert (tmp_path / 'provider-output.zip').read_bytes() == b'immutable-provider-output'


def test_replaced_completion_is_never_reopened_or_read(tmp_path):
    fake, attempts, outcomes = _install_fake_transport(tmp_path)
    outcomes.write_text('0 200\n')
    source = tmp_path / 'payload.zip'
    source.write_bytes(b'unchanged')
    victim = tmp_path / 'foreign-secret'
    victim.write_bytes(b'NEVER-READ-FOREIGN-BYTES')
    env = {**os.environ, 'PATH': f'{fake}:{os.environ["PATH"]}',
           'BLUEPRINT_TEST_CURL_ATTEMPTS': str(attempts), 'BLUEPRINT_TEST_CURL_OUTCOMES': str(outcomes)}
    program = provider_output_upload_shell_fragment(scratch_root=str(tmp_path)) + '''
blueprint_upload_put https://fixture.invalid "$1" || exit 85
mv "$BLUEPRINT_PROVIDER_UPLOAD_RESPONSE_FILE" "$BLUEPRINT_PROVIDER_UPLOAD_RESPONSE_FILE.retained"
ln -s "$2" "$BLUEPRINT_PROVIDER_UPLOAD_RESPONSE_FILE"
blueprint_upload_read_response
printf 'READ_RC:%s\n' "$?"
blueprint_upload_cleanup
printf 'CLEANUP_RC:%s\n' "$?"
'''
    result = subprocess.run(['bash', '-c', program, 'test', str(source), str(victim)],
        env=env, capture_output=True, text=True, timeout=20)
    assert 'READ_RC:86' in result.stdout
    assert 'CLEANUP_RC:0' in result.stdout
    assert 'NEVER-READ-FOREIGN-BYTES' not in result.stdout
    assert victim.read_bytes() == b'NEVER-READ-FOREIGN-BYTES'
    assert source.read_bytes() == b'unchanged'


def test_post_transfer_completion_replacement_never_grants_receipt_authority(tmp_path):
    fake, attempts, outcomes = _install_fake_transport(tmp_path)
    outcomes.write_text('0 200\n')
    source = tmp_path / 'payload.zip'
    source.write_bytes(b'unchanged')
    env = {**os.environ, 'PATH': f'{fake}:{os.environ["PATH"]}',
           'BLUEPRINT_TEST_CURL_ATTEMPTS': str(attempts), 'BLUEPRINT_TEST_CURL_OUTCOMES': str(outcomes)}
    program = provider_output_upload_shell_fragment(scratch_root=str(tmp_path)) + '''
eval "$(declare -f blueprint_upload_transfer | sed '1s/blueprint_upload_transfer/blueprint_test_owned_transfer/')"
blueprint_upload_transfer() {
  blueprint_test_owned_transfer "$@"
  result=$?
  if [ "$result" = 0 ]; then
    mv "$blueprint_upload_completion_file" "$blueprint_upload_completion_file.original"
    printf FOREIGN-OTHER-ATTEMPT > "$blueprint_upload_completion_file"
  fi
  return "$result"
}
blueprint_upload_put https://fixture.invalid "$1"
printf 'UPLOAD_RC:%s\n' "$?"
blueprint_upload_read_response
printf 'READ_RC:%s\n' "$?"
printf 'ACTIVE_RECEIPT:%s\n' "$BLUEPRINT_PROVIDER_UPLOAD_RESPONSE_FILE"
'''
    result = subprocess.run(['bash', '-c', program, 'test', str(source)],
        env=env, capture_output=True, text=True, timeout=20)
    assert 'UPLOAD_RC:86' in result.stdout
    assert 'READ_RC:86' in result.stdout
    assert 'ACTIVE_RECEIPT:\n' in result.stdout
    assert 'FOREIGN-OTHER-ATTEMPT' not in result.stdout
    [scratch] = tmp_path.glob('.blueprint-provider-upload.*')
    assert (scratch / 'blueprint_provider_upload_response.json').read_bytes() == b'FOREIGN-OTHER-ATTEMPT'
    assert (scratch / 'blueprint_provider_upload_response.json.original').read_bytes() == b''
    assert len(attempts.read_text().splitlines()) == 1
    assert source.read_bytes() == b'unchanged'


def test_noclobber_creation_and_held_descriptor_are_atomic(tmp_path):
    fake, attempts, outcomes = _install_fake_transport(tmp_path)
    outcomes.write_text('0 200\n')
    source = tmp_path / 'payload.zip'
    source.write_bytes(b'unchanged')
    victim = tmp_path / 'foreign'
    victim.write_bytes(b'never-truncate')
    env = {**os.environ, 'PATH': f'{fake}:{os.environ["PATH"]}',
           'BLUEPRINT_TEST_CURL_ATTEMPTS': str(attempts), 'BLUEPRINT_TEST_CURL_OUTCOMES': str(outcomes)}
    program = provider_output_upload_shell_fragment(scratch_root=str(tmp_path)) + '''
victim=$2
mktemp() {
  newroot=$(command mktemp "$@") || return 86
  ln -s "$victim" "$newroot/blueprint_provider_upload_response.json" || return 86
  printf '%s' "$newroot"
}
blueprint_upload_put https://fixture.invalid "$1"
printf 'UPLOAD_RC:%s\n' "$?"
'''
    result = subprocess.run(['bash', '-c', program, 'test', str(source), str(victim)],
        env=env, capture_output=True, text=True, timeout=20)
    assert 'UPLOAD_RC:86' in result.stdout
    assert attempts.read_text() == ''
    assert victim.read_bytes() == b'never-truncate'
    [scratch] = tmp_path.glob('.blueprint-provider-upload.*')
    assert (scratch / 'blueprint_provider_upload_response.json').is_symlink()


@pytest.mark.parametrize('noclobber', [False, True])
@pytest.mark.parametrize('mask', ['022', '077'])
def test_setup_failure_restores_flags_umask_and_releases_only_owned_descriptors(tmp_path, noclobber, mask):
    victim = tmp_path / 'foreign'
    victim.write_bytes(b'never-truncate')
    source = tmp_path / 'payload.zip'
    source.write_bytes(b'unchanged')
    program = provider_output_upload_shell_fragment(scratch_root=str(tmp_path))
    program += f'umask {mask}; set {"-" if noclobber else "+"}o noclobber; '
    program += '''
victim=$2
before_mask=$(umask)
case "$-" in *C*) before_C=1;; *) before_C=0;; esac
mktemp() {
  newroot=$(command mktemp "$@") || return 86
  ln -s "$victim" "$newroot/blueprint_provider_upload_response.json" || return 86
  printf '%s' "$newroot"
}
for iteration in 1 2; do
  blueprint_upload_put https://fixture.invalid "$1"
  printf 'UPLOAD_RC:%s\n' "$?"
  [ "$(umask)" = "$before_mask" ] || exit 87
  case "$-" in *C*) after_C=1;; *) after_C=0;; esac
  [ "$after_C" = "$before_C" ] || exit 88
  for token in 10 11 12 13; do [ ! -e "/dev/fd/$token" ] || exit 89; done
  [ -z "$BLUEPRINT_PROVIDER_UPLOAD_RESPONSE_FILE" ] || exit 90
done
'''
    result = subprocess.run(['bash', '-c', program, 'test', str(source), str(victim)],
        capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.count('UPLOAD_RC:86') == 2
    assert victim.read_bytes() == b'never-truncate'
    assert source.read_bytes() == b'unchanged'
