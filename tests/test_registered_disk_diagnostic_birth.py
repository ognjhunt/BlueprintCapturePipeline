"""ADP-009D/day28: genuine ordinary diagnostic birth, CPU boundary only.

This fixture exercises issuer/birth code. It does not claim installed Linux
rights, completed producer authority, or permission for historical cleanup.
"""
import hashlib
import json
import os
from pathlib import Path

import pytest

from tests.test_registered_experiment_issuer import installation, issue, encoded  # noqa: F401
from tests.test_owner_target_version_publication import root_metadata  # noqa: F401
from tests.test_registered_experiment_birth import prepare, birth


def _selector(raw):
    return {"sha256": "sha256:" + hashlib.sha256(raw).hexdigest(), "size_bytes": len(raw)}


@pytest.mark.parametrize("profile,intent,cleanup", [
    ("root_disk_diagnostic_disposable.v1", "scratch", "delete"),
    ("root_disk_diagnostic_evidence.v1", "evidence", "offload"),
])
def test_actual_diagnostic_birth_never_transfers_writer_rights_to_blueprint(
    installation, monkeypatch, profile, intent, cleanup  # noqa: F811
):
    from blueprint_pipeline import control_plane_lane_experiment_birth as code
    from blueprint_pipeline import control_plane_lane_experiment_publication as publication
    public = prepare(installation)
    config, settings, store, _ = installation
    for root in (settings["lane_scratch_work_root"], settings["lane_scratch_inputs_root"]):
        (Path(root) / "diagnostics").mkdir(mode=0o750)
    request = store.parent / "diagnostic-request.json"
    from blueprint_pipeline import control_plane_lane_disk_diagnostic as producer
    request.write_bytes(encoded(producer.build_request(installed_config_path=config, run_ref='run1')))
    request.chmod(0o600)
    grant = issue(installation, participant_profile=profile,
                  request_records=((request, _selector(request.read_bytes())),))
    monkeypatch.setattr(code, "_blueprint_identity", lambda: (1234, 0))
    # Existing portable fixture cannot create a foreign UID. This observer
    # rejects that attempted transition; actual UID rights need Linux later.
    original = publication._BirthFiles.transition_owner
    transitions = []
    def root_only(files, fd, uid, gid, mode):
        transitions.append((uid, gid, mode))
        assert (uid, gid) == (0, 0), "diagnostic birth granted blueprint pathname writes"
        return original(files, fd, uid, gid, mode)
    monkeypatch.setattr(publication._BirthFiles, "transition_owner", root_only)
    monkeypatch.setattr(os, "fchown", lambda *_: None)
    result = birth(installation, grant)
    target = Path(result["path"])
    assert target.parent.name == "diagnostics"
    assert target.stat().st_mode & 0o777 == 0o700
    lease = json.loads((target / ".lane-scratch.v1.json").read_bytes())
    assert (lease["lane"], lease["class_intent"], lease["cleanup"]) == ("diagnostics", intent, cleanup)
    head = json.loads((public / "HEAD.json").read_bytes())
    current = json.loads((public / head["record_name"]).read_bytes())["enrollments"][0]
    assert current["lane"] == "diagnostics" and current["completion"] is None
    assert all(uid == gid == 0 for uid, gid, _ in transitions)


@pytest.mark.parametrize("invalid", ["missing_request", "arbitrary_command", "changed_config"])
def test_diagnostic_issuance_refuses_before_intent_or_payload(installation, invalid):  # noqa: F811
    from blueprint_pipeline import control_plane_lane_disk_diagnostic as producer
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest
    prepare(installation)
    config, _, store, _ = installation
    value = producer.build_request(installed_config_path=config, run_ref="run1")
    if invalid == "arbitrary_command":
        value["command"] = "caller-selected-code"
    elif invalid == "changed_config":
        value["config"]["sha256"] = "sha256:" + "0" * 64
    value["request_digest"] = canonical_digest(value, digest_field="request_digest")
    request = store.parent / "diagnostic-request.json"
    request.write_bytes(encoded(value))
    request.chmod(0o600)
    before = {path.name: path.read_bytes() for path in store.iterdir()}
    records = () if invalid == "missing_request" else ((request, _selector(request.read_bytes())),)
    with pytest.raises(ValueError, match="experiment_creation_invalid|diagnostic_request_"):
        issue(installation, participant_profile="root_disk_diagnostic_disposable.v1",
              request_records=records)
    assert {path.name: path.read_bytes() for path in store.iterdir()} == before
    assert not any(path.name.startswith("registered-") for path in config.parent.rglob("*"))
