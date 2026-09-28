"""Fixed report wrapper and preserve-on-upgrade owner metadata provisioning."""

# Covers (for impacted-test selection):
#   deploy/operator-door/door-owner-census.sh
#   deploy/operator-door/install.sh
import json
import os
import subprocess
from pathlib import Path

import pytest

DOOR = Path(__file__).parents[1] / "deploy/operator-door"


def run_wrapper(tmp_path, **extra):
    release = tmp_path / "release"
    release.mkdir(exist_ok=True)
    results = tmp_path / "results"
    results.mkdir(exist_ok=True)
    fake = tmp_path / "fake-python"
    fake.write_text(
        '#!/bin/bash\nprintf "%s\\n" "$@" > "$ARG_LOG"\n'
        'printf \'{"status":"owner_consent_observed"}\\n\' > "$DOOR_RESULTS_DIR/$DOOR_REQUEST_ID.outcome.json"\n'
        'printf \'{"mutations":0}\\n\' > "$DOOR_RESULTS_DIR/$DOOR_REQUEST_ID.owner-census.json"\n'
        'exit "${FAKE_RC:-0}"\n'
    )
    fake.chmod(0o700)
    env = {
        **os.environ,
        "DOOR_REQUEST_ID": "20260928T000000Z-owner-census-decision-deadbeef",
        "DOOR_RESULTS_DIR": str(results),
        "DOOR_VENV_PYTHON": str(fake),
        "DOOR_CONTROL_PLANE_REPO": str(release),
        "DOOR_CONSENT_ID": "a" * 32,
        "DOOR_CONSENT_SHA256": "sha256:" + "b" * 64,
        "DOOR_CONSENT_SIZE_BYTES": "100",
        "DOOR_CONFIG_PATH": "/etc/blueprint-operator-door/door.json",
        "ARG_LOG": str(tmp_path / "argv"),
        **extra,
    }
    done = subprocess.run(
        ["/bin/bash", str(DOOR / "door-owner-census.sh")],
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )
    return done, results, env


def test_fixed_wrapper_only_runs_active_report_module(tmp_path):
    done, results, env = run_wrapper(tmp_path)
    assert done.returncode == 0
    argv = Path(env["ARG_LOG"]).read_text().splitlines()
    assert argv[:3] == ["-m", "blueprint_pipeline.control_plane_lane_owner_consents", "report"]
    assert (
        "--door-config" in argv and argv[argv.index("--door-config") + 1] == env["DOOR_CONFIG_PATH"]
    )
    assert "--results-dir" in argv and "--request-id" in argv
    assert not any("apply" in a or "owner=" in a or "http" in a for a in argv)
    assert (
        json.loads((results / (env["DOOR_REQUEST_ID"] + ".outcome.json")).read_bytes())["status"]
        == "owner_consent_observed"
    )


@pytest.mark.parametrize(
    "extra",
    [
        {"DOOR_CONSENT_ID": "../foreign"},
        {"DOOR_CONSENT_SIZE_BYTES": "524289"},
        {"DOOR_CONSENT_SHA256": "SECRET"},
        {"DOOR_CONFIG_PATH": "/forged.json"},
    ],
)
def test_wrapper_refuses_before_running_leaf(tmp_path, extra):
    done, _, env = run_wrapper(tmp_path, **extra)
    assert done.returncode != 0 and not Path(env["ARG_LOG"]).exists()


def test_installer_provisions_only_disabled_metadata_and_retains_existing_paths():
    text = (DOOR / "install.sh").read_text()
    assert "door-owner-census.sh" in text
    assert "# OWNER CONSENT PROVISIONING BEGIN" in text
    block = text.split("# OWNER CONSENT PROVISIONING BEGIN", 1)[1].split(
        "# OWNER CONSENT PROVISIONING END", 1
    )[0]
    assert '"enabled": false' in block and '"principals": []' in block
    assert '[ ! -e "$owner_policy" ] && [ ! -L "$owner_policy" ]' in block
    assert '[ ! -e "$owner_store" ] && [ ! -L "$owner_store" ]' in block
    assert "0700" in block and "0600" in block and ".owner-consents.lock" in block
    unit = (DOOR.parents[0] / "systemd/blueprint-operator-door-runner.service").read_text()
    assert "ReadWritePaths=/var/lib/blueprint-operator-door/requests\n" in unit


def test_upgrade_owner_provisioning_preserves_policy_records_and_lock_inode(tmp_path):
    import sys

    state = tmp_path / "state"
    config = tmp_path / "config"
    config.mkdir()
    store = state / "requests" / "owner-consents"
    store.mkdir(parents=True, mode=0o700)
    policy = config / "lane-owner-policy.json"
    policy.write_bytes(b'{"enabled":true,"principals":[]}\n')
    policy.chmod(0o600)
    lock = store / ".owner-consents.lock"
    lock.write_bytes(b"")
    lock.chmod(0o600)
    record = store / ("a" * 32 + ".json")
    record.write_bytes(b"kept")
    record.chmod(0o600)
    before = (policy.read_bytes(), record.read_bytes(), lock.stat().st_ino)
    stubs = tmp_path / "bin"
    stubs.mkdir()
    # Simulate privileged ownership only; real file type/mode/inode stay observed.
    root_python = stubs / "python3"
    root_python.write_text(
        "#!" + sys.executable + "\nimport os,sys\nfrom types import SimpleNamespace\n"
        "original=os.lstat\ndef lstat(path):\n value=original(path)\n"
        ' fields={n:getattr(value,n) for n in dir(value) if n.startswith("st_")}\n'
        " fields.update(st_uid=0,st_gid=0)\n return SimpleNamespace(**fields)\n"
        "os.lstat=lstat\nsys.argv=sys.argv[1:]\nexec(sys.stdin.read())\n"
    )
    root_python.chmod(0o700)
    for name in ("chown",):
        p = stubs / name
        p.write_text("#!/bin/bash\nexit 0\n")
        p.chmod(0o700)
    block = (
        (DOOR / "install.sh")
        .read_text()
        .split("# OWNER CONSENT PROVISIONING BEGIN", 1)[1]
        .split("# OWNER CONSENT PROVISIONING END", 1)[0]
    )
    command = 'set -euo pipefail\nstate_root="$TEST_STATE"\nconfig_dir="$TEST_CONFIG"\n' + block
    done = subprocess.run(
        ["/bin/bash", "-c", command],
        env={
            **os.environ,
            "PATH": str(stubs) + ":" + os.environ["PATH"],
            "TEST_STATE": str(state),
            "TEST_CONFIG": str(config),
        },
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert done.returncode == 0, done.stderr
    assert (policy.read_bytes(), record.read_bytes(), lock.stat().st_ino) == before
    # The same real permission validation refuses an unsafe existing store,
    # without correcting/changing its policy or records.
    store.chmod(0o777)
    done = subprocess.run(
        ["/bin/bash", "-c", command],
        env={
            **os.environ,
            "PATH": str(stubs) + ":" + os.environ["PATH"],
            "TEST_STATE": str(state),
            "TEST_CONFIG": str(config),
        },
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert done.returncode != 0 and "owner_consent_provisioning_unsafe" in done.stderr
    assert (policy.read_bytes(), record.read_bytes(), lock.stat().st_ino) == before

    store.chmod(0o700)
    store.rename(store.with_name("kept-store"))
    foreign = tmp_path / "foreign"
    foreign.mkdir(mode=0o700)
    store.symlink_to(foreign)
    done = subprocess.run(
        ["/bin/bash", "-c", command],
        env={
            **os.environ,
            "PATH": str(stubs) + ":" + os.environ["PATH"],
            "TEST_STATE": str(state),
            "TEST_CONFIG": str(config),
        },
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert done.returncode != 0
    assert not (foreign / ".owner-consents.lock").exists(), (
        "created lock through unsafe store symlink"
    )


def test_leaf_failure_has_fixed_wrapper_outcome_without_echoing_private_stderr(tmp_path):
    done, results, env = run_wrapper(tmp_path, FAKE_RC="1")
    assert done.returncode == 1
    outcome = json.loads((results / (env["DOOR_REQUEST_ID"] + ".outcome.json")).read_bytes())
    assert outcome["status"] == "refused"
    assert outcome["code"] == "owner_consent_report_refused" and outcome["exit_code"] == 1
    assert len(json.dumps(outcome)) < 4096
