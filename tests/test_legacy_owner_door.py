# Covers (for impacted-test selection):
#   deploy/operator-door/door-legacy-owner-census.sh
#   deploy/operator-door/operator_door/requests.py
#   deploy/operator-door/operator_door/spool_runner.py
#   src/blueprint_pipeline/control_plane_lane_legacy_owner.py
"""The existing authenticated door exposes one bounded owner label report."""

import json
import os
import subprocess
from pathlib import Path

import pytest

from tests.test_owner_census_door import DEPLOYER, READER, _call, _json, door as _door_fixture
from tests.test_operator_door_runner import FakeRunner
from operator_door import requests, spool_runner
from operator_door.config import DoorConfig

door = _door_fixture


def test_legacy_report_request_has_no_caller_selected_path_or_action():
    request = {"kind": "legacy-owner-census"}
    assert requests.validate_request(request) == request
    assert requests.required_scope(request["kind"]) == "operate"
    assert requests.validate_request_id(requests.new_request_id(request["kind"]))
    for extra in ({"path": "/mnt/blueprint-work/lanes/old"}, {"apply": True},
                  {"owner": "alice"}, {"packet_id": "a" * 32}):
        with pytest.raises(requests.RequestRefused):
            requests.validate_request(request | extra)


def test_door_reports_all_candidates_and_keeps_unknown_fd_owner_label():
    from blueprint_pipeline.control_plane_lane_legacy_owner_door import _public_report

    labeled = dict(path="/work/lanes/diagnostics/old-1", family="diagnostics",
                   references=[], unreadable=0, allocated_bytes=4096, age_seconds=12,
                   classification="owner_review_reference_unknown", owner="owner",
                   approved_expiry=2000, process_fd_references="unknown",
                   gc_eligible=False, references_clear=False)
    unregistered = dict(path="/work/lanes/diagnostics/old-2", family="diagnostics",
                        references=[], unreadable=0, allocated_bytes=2048, age_seconds=20)
    observed = dict(status="incomplete", scan_errors=["process_inventory_unreadable"],
                    rows=[labeled, unregistered], candidate_count=2, observed_owner_count=1)
    report = _public_report(observed)
    assert report["status"] == "reference_incomplete"
    assert len(report["rows"]) == 2 and report["observed_owner_count"] == 1
    assert report["rows"][0]["classification"] == "owner_review_reference_unknown"
    assert report["rows"][0]["keep_reason"] == "process_fd_references_unknown"
    assert report["rows"][1]["classification"] == "unclassified"
    assert all(row["gc_eligible"] is False and row["references_clear"] is False
               and row["candidate_bytes"] is None for row in report["rows"])
    unsafe = observed | {"scan_errors": ["process_inventory_unreadable", "queue_inventory_unavailable"]}
    assert _public_report(unsafe)["rows"] == []


def test_legacy_report_door_requires_operate_scope_and_uses_existing_spool(door):
    assert _call(door, "/requests", token=READER,
                 body={"kind": "legacy-owner-census"})[0] == 403
    object.__setattr__(door["config"], "owner_census_decisions_enabled", 1)
    answer = _call(door, "/requests", token=DEPLOYER,
                   body={"kind": "legacy-owner-census"})
    assert answer[0] == 202
    request_id = _json(answer)["id"]
    spooled = door["state"] / "requests" / "pending" / (request_id + ".json")
    assert json.loads(spooled.read_bytes())["request"] == {"kind": "legacy-owner-census"}


def test_legacy_report_runner_is_read_only_and_finite(tmp_path):
    config = DoorConfig(state_root=str(tmp_path), owner_census_decisions_enabled=1)
    fake = FakeRunner()
    request_id = "20260929T000000Z-legacy-owner-census-deadbeef"
    assert spool_runner._act(config, fake, request_id,
                             {"kind": "legacy-owner-census"})["status"] == "launched"
    argv = fake.calls[0]
    assert argv[-1] == config.install_root + "/door-legacy-owner-census.sh"
    assert "--property=RuntimeMaxSec=5min" in argv
    assert "--property=ProtectSystem=strict" in argv
    assert "--property=PrivateNetwork=yes" in argv
    assert "--property=SystemCallFilter=~ptrace process_vm_readv process_vm_writev" in argv
    assert "--property=CapabilityBoundingSet=CAP_DAC_READ_SEARCH CAP_PERFMON" in argv
    assert "--property=AmbientCapabilities=CAP_DAC_READ_SEARCH CAP_PERFMON" in argv
    assert not any("CAP_SYS_PTRACE" in item for item in argv)
    hidden = next(item for item in argv if item.startswith("--property=InaccessiblePaths="))
    assert "/etc/blueprint/provider-secrets" in hidden
    assert "/var/lib/blueprint/spend-authority" in hidden
    hidden_paths = {part.lstrip("-") for part in hidden.split("=", 2)[2].split()}
    assert "/etc/blueprint-operator-door" not in hidden_paths
    assert "/etc/blueprint-operator-door/deploy-key" in hidden_paths
    assert DoorConfig().token_file in hidden_paths
    writable = [part for part in argv if part.startswith("--property=ReadWritePaths=")]
    assert writable == ["--property=ReadWritePaths=" + str(Path(config.spool_root) / "results")]
    assert not any("APPLY" in part or "OWNER=" in part or "PACKET" in part for part in argv)


@pytest.mark.slow
def test_legacy_report_wrapper_only_runs_fixed_report_with_discoverable_result(tmp_path):
    from tests.test_owner_census_wrapper import DOOR

    release, results = tmp_path / "release", tmp_path / "results"
    release.mkdir()
    results.mkdir()
    fake = tmp_path / "python"
    argv = tmp_path / "argv.json"
    fake.write_text(
        '#!/bin/bash\nprintf "%s\\n" "$@" > "$ARG_LOG"\n'
        'printf \'{"status":"complete","rows":[],"gc_eligible":false,"references_clear":false}\\n\' > "$DOOR_RESULTS_DIR/$DOOR_REQUEST_ID.legacy-owner-census.json"\n'
        'printf \'{"status":"legacy_owner_census_observed","result":"%s/%s.legacy-owner-census.json"}\\n\' "$DOOR_RESULTS_DIR" "$DOOR_REQUEST_ID" > "$DOOR_RESULTS_DIR/$DOOR_REQUEST_ID.outcome.json"\n'
    )
    fake.chmod(0o700)
    env = dict(os.environ, DOOR_REQUEST_ID="20260929T000000Z-legacy-owner-census-deadbeef",
               DOOR_RESULTS_DIR=str(results), DOOR_VENV_PYTHON=str(fake),
               DOOR_CONTROL_PLANE_REPO=str(release),
               DOOR_CONFIG_PATH="/etc/blueprint-operator-door/door.json", ARG_LOG=str(argv))
    done = subprocess.run(["/bin/bash", str(DOOR / "door-legacy-owner-census.sh")],
                          env=env, capture_output=True, text=True, timeout=10)
    assert done.returncode == 0
    args = argv.read_text().splitlines()
    assert args[:3] == ["-m", "blueprint_pipeline.control_plane_lane_legacy_owner_door", "report"]
    assert "--results-dir" in args and "--request-id" in args
    assert not any(part in {"packet", "approve", "apply"} for part in args)
    outcome = json.loads((results / (env["DOOR_REQUEST_ID"] + ".outcome.json")).read_bytes())
    assert outcome["status"] == "legacy_owner_census_observed"
    assert outcome["result"] == str(results / (env["DOOR_REQUEST_ID"] + ".legacy-owner-census.json"))
