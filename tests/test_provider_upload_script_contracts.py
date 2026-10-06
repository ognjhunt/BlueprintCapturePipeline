"""All generated provider consumers use the private upload receipt lifecycle."""
import os
import subprocess
import sys

import pytest

from blueprint_pipeline import vast_provider_adapter as adapter


@pytest.mark.parametrize('kind', sorted(adapter.VAST_PROVIDER_BUNDLE_KINDS))
def test_generated_provider_upload_consumers_read_then_clean_private_receipt(kind):
    script = adapter._probe_shell_script('https://fixture.invalid/heartbeat',
        enable_blueprint_bundle=True, provider_bundle_kind=kind,
        expected_provider_bundle_sha256='sha256:' + 'a' * 64)
    subprocess.run(['bash', '-n'], input=script, text=True, check=True, capture_output=True)
    assert 'cat /tmp/blueprint_provider_upload_response.json' not in script
    if 'echo BLUEPRINT_VAST_PROVIDER_OUTPUT_UPLOAD_OK;' in script:
        assert 'blueprint_upload_read_response || exit 86; blueprint_upload_cleanup || exit 86; echo BLUEPRINT_VAST_PROVIDER_OUTPUT_UPLOAD_OK;' in script
    if 'echo BLUEPRINT_VAST_PROVIDER_EARLY_DIAGNOSTIC_UPLOAD_OK;' in script:
        assert 'blueprint_upload_read_response || exit 86; blueprint_upload_cleanup || exit 86; echo BLUEPRINT_VAST_PROVIDER_EARLY_DIAGNOSTIC_UPLOAD_OK;' in script
    assert 'blueprint_upload_max_attempts=3' in script
    assert 'blueprint_upload_deadline=$(( $(date +%s) + 1200 ));' in script


def test_wam_early_diagnostic_releases_receipt_before_final_upload(tmp_path):
    from blueprint_pipeline.vast_provider_transfer_upload import provider_output_upload_shell_fragment
    from blueprint_pipeline.vast_wam_terminal_diagnostic import wam_terminal_diagnostic_shell_fragment
    from tests.test_vast_provider_transfer_upload import _install_fake_transport
    fake, attempts, outcomes = _install_fake_transport(tmp_path)
    outcomes.write_text('0 200\n0 200\n')
    work = tmp_path / 'work'
    work.mkdir()
    output = work / 'runtime_output'
    output.mkdir()
    env = {**os.environ, 'PATH': f'{fake}:{os.environ["PATH"]}',
        'BLUEPRINT_TEST_CURL_ATTEMPTS': str(attempts), 'BLUEPRINT_TEST_CURL_OUTCOMES': str(outcomes),
        'BLUEPRINT_VAST_WORK_DIR': str(work), 'BLUEPRINT_WAM_PROVIDER_OUTPUT_DIR': str(output)}
    program = provider_output_upload_shell_fragment(scratch_root=str(tmp_path)) + '''
provider_rc=17
RUNTIME_PY=$1
WORK_DIR=$2
OUTPUT_PUT_URL=https://fixture.invalid/output.zip
''' + wam_terminal_diagnostic_shell_fragment() + '''
"$RUNTIME_PY" - <<'PYTHON'
import os,zipfile
from pathlib import Path
output=Path(os.environ['BLUEPRINT_WAM_PROVIDER_OUTPUT_DIR'])
(output/'final.txt').write_text('retained provider diagnostic output')
with zipfile.ZipFile(Path(os.environ['BLUEPRINT_VAST_WORK_DIR'])/'wam_provider_runtime_output.zip','w') as archive:
    for path in sorted(output.iterdir()): archive.write(path,path.name)
PYTHON
blueprint_upload_put "$OUTPUT_PUT_URL" "$WORK_DIR/wam_provider_runtime_output.zip" || exit 85
blueprint_upload_read_response || exit 86
blueprint_upload_cleanup || exit 87
echo BLUEPRINT_TEST_FINAL_UPLOAD_OK
'''
    result = subprocess.run(['bash', '-c', program, 'test', sys.executable, str(work)],
        env=env,capture_output=True,text=True,timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'BLUEPRINT_VAST_PROVIDER_EARLY_DIAGNOSTIC_UPLOAD_OK' in result.stdout
    assert 'BLUEPRINT_TEST_FINAL_UPLOAD_OK' in result.stdout
    assert 'descriptor_busy' not in result.stdout
    wire = attempts.read_text().splitlines()
    assert len(wire) == 2
    assert wire[0].split()[1] != wire[1].split()[1]
    assert wire[0].split()[2] == wire[1].split()[2]
    roots = list(tmp_path.glob('.blueprint-provider-upload.*'))
    assert len(roots) == 2
    assert all(sum(p.stat().st_size for p in root.iterdir()) <= 3 for root in roots)
