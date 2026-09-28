"""Actual upload shell, tiny immutable payloads, byte-counted fake transport."""
import hashlib
import subprocess

import pytest

from blueprint_pipeline.vast_provider_transfer_upload import provider_output_upload_shell_fragment


@pytest.mark.parametrize("cut", [0, 5, 24])
def test_disconnect_restarts_same_immutable_stream(tmp_path, cut):
    payload = b"immutable-output-bytes!\n"
    source = tmp_path / "output.zip"
    source.write_bytes(payload)
    fake = tmp_path / "bin"
    fake.mkdir()
    curl = fake / "curl"
    curl.write_text('''#!/bin/bash
set -eu
while [ "$1" != "--upload-file" ]; do shift; done
source=$2
if [ ! -f "$TEST_ROOT/attempt" ]; then
  : > "$TEST_ROOT/partial"
  if [ "$CUT" -gt 0 ]; then head -c "$CUT" "$source" > "$TEST_ROOT/partial"; fi
  touch "$TEST_ROOT/attempt"
  printf 000
  exit 56
fi
cat "$source" > "$TEST_ROOT/delivered"
printf 200
''')
    curl.chmod(0o755)
    (fake / "date").write_text('#!/bin/bash\nprintf "%s\\n" "${BLUEPRINT_TEST_NOW:-102}"\n')
    (fake / "date").chmod(0o755)
    (fake / "sleep").write_text("#!/bin/bash\nexit 0\n")
    (fake / "sleep").chmod(0o755)
    env = {"PATH": f"{fake}:/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin", "TEST_ROOT": str(tmp_path), "CUT": str(cut)}
    fragment = provider_output_upload_shell_fragment(scratch_root=str(tmp_path))
    result = subprocess.run(["bash", "-c", fragment + 'blueprint_upload_put https://fixture.invalid "$1"',
        "test", str(source)], env=env, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (tmp_path / "partial").read_bytes() == payload[:cut]
    assert (tmp_path / "delivered").read_bytes() == source.read_bytes() == payload
    assert hashlib.sha256(source.read_bytes()).digest() == hashlib.sha256(payload).digest()
    assert result.stdout.count(b"TRANSPORT_RETRY") == 1
    assert (tmp_path / "blueprint_provider_upload_response.json").is_file()


def test_acknowledgement_without_durable_completion_is_not_success(tmp_path):
    from tests.test_vast_provider_transfer_upload import _run_upload_with_fake_transport
    (tmp_path / "blueprint_provider_upload_response.json").mkdir()
    result, attempts = _run_upload_with_fake_transport(tmp_path=tmp_path, outcomes=[(0, "200")])
    assert "UPLOAD_RC:86" in result.stdout
    assert len(attempts) == 1
    assert (tmp_path / "provider-output.zip").read_bytes() == b"immutable-provider-output"


def test_acknowledged_transfer_cannot_publish_mutated_source(tmp_path):
    from tests.test_vast_provider_transfer_upload import _run_upload_with_fake_transport
    result, attempts = _run_upload_with_fake_transport(tmp_path=tmp_path,
        outcomes=[(0, "200")], mutate_after_first=True)
    assert "UPLOAD_RC:86" in result.stdout
    assert len(attempts) == 1
    assert not (tmp_path / "blueprint_provider_upload_response.json").exists()


@pytest.mark.parametrize("status", ["400", "401", "404", "410", "422"])
def test_expired_url_and_nontransient_4xx_preserve_local_output(tmp_path, status):
    from tests.test_vast_provider_transfer_upload import _run_upload_with_fake_transport
    result, attempts = _run_upload_with_fake_transport(tmp_path=tmp_path, outcomes=[(22, status), (0, "200")])
    assert "UPLOAD_RC:22" in result.stdout
    assert len(attempts) == 1
    assert (tmp_path / "provider-output.zip").read_bytes() == b"immutable-provider-output"
    assert not (tmp_path / "blueprint_provider_upload_response.json").exists()


def test_deadline_expires_between_transport_attempts(tmp_path):
    from tests.test_vast_provider_transfer_upload import _install_fake_transport
    source = tmp_path / "output.zip"
    source.write_bytes(b"immutable-output")
    fake, attempts, outcomes = _install_fake_transport(tmp_path)
    outcomes.write_text("56 000\n0 200\n")
    (fake / "date").write_text('''#!/bin/bash
count=0
[ ! -f "$TEST_ROOT/date-count" ] || count=$(cat "$TEST_ROOT/date-count")
printf '%s' "$((count + 1))" > "$TEST_ROOT/date-count"
if [ "$count" -lt 2 ]; then printf 100; else printf 1301; fi
''')
    (fake / "date").chmod(0o755)
    env = {"PATH": f"{fake}:/usr/bin:/bin:/usr/sbin:/sbin", "TEST_ROOT": str(tmp_path),
           "BLUEPRINT_TEST_CURL_ATTEMPTS": str(attempts), "BLUEPRINT_TEST_CURL_OUTCOMES": str(outcomes)}
    result = subprocess.run(["bash", "-c", provider_output_upload_shell_fragment(scratch_root=str(tmp_path))
        + 'blueprint_upload_put https://fixture.invalid "$1"', "test", str(source)], env=env, capture_output=True)
    assert result.returncode == 86
    assert len(attempts.read_text().splitlines()) == 1
    assert source.read_bytes() == b"immutable-output"
    assert not (tmp_path / "blueprint_provider_upload_response.json").exists()
