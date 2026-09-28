"""Actual G1 pair/worker payload is protected by authenticated current target SH."""
# Covers (for impacted-test selection):
#   src/blueprint_pipeline/control_plane_lane_experiment_consumer.py
#   src/blueprint_pipeline/native_g1_development_pair.py
#   src/blueprint_pipeline/native_g1_development_worker.py

import fcntl
import hashlib
import json
import os
from pathlib import Path

import pytest

from tests.test_owner_target_version_publication import root_metadata  # noqa: F401
from tests.test_registered_experiment_issuer import installation, issue  # noqa: F401
from tests.test_registered_experiment_birth import birth, prepare
from tests.test_native_g1_development_pair import _paired_requests


@pytest.mark.parametrize("caller", ["pair", "worker"])
def test_registered_missing_use_refuses_before_request_or_payload_read(tmp_path, monkeypatch, caller):
    from blueprint_pipeline import native_g1_development_pair as pair
    from blueprint_pipeline import native_g1_development_worker as worker
    root = tmp_path / "lanes"
    target = root / "g1" / ("registered-" + "a" * 32)
    monkeypatch.setattr(pair, "LANE_SCRATCH_ROOTS", (root,))
    from blueprint_pipeline import control_plane_scratch_lifetime as lifetime
    monkeypatch.setattr(lifetime, "LANE_ROOTS", (root,))
    monkeypatch.setattr(Path, "read_text", lambda *a, **kw: pytest.fail("unadmitted payload read"))
    with pytest.raises(ValueError, match="experiment_consumer_authority_required"):
        if caller == "pair":
            pair.run_g1_development_pair(request_paths=[tmp_path / "absent"], output_dir=target)
        else:
            worker.run_g1_development_worker(request={}, output_dir=target / "candidate")


def test_actual_registered_pair_and_worker_preflight_output_holds_target_sh(installation, tmp_path, monkeypatch):  # noqa: F811
    from blueprint_pipeline import control_plane_lane_experiment_birth as issuer
    from blueprint_pipeline import control_plane_lane_experiment_consumer as consumer
    from blueprint_pipeline import native_g1_development_pair as pair
    from blueprint_pipeline import native_g1_development_worker as worker
    prepare(installation)
    paths, _ = _paired_requests(tmp_path)
    selectors = tuple((path, {"sha256": "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(),
                             "size_bytes": path.stat().st_size}) for path in paths)
    grant = issue(installation, participant_profile="g1_local_prelaunch_block.v1", request_records=selectors)
    monkeypatch.setattr(issuer, "_blueprint_identity", lambda: (0, 0))
    monkeypatch.setattr(os, "fchown", lambda *a: None)
    born = birth(installation, grant)
    root = Path(installation[1]["lane_scratch_work_root"])
    monkeypatch.setattr(consumer, "LANE_ROOTS", (root, Path(installation[1]["lane_scratch_inputs_root"])))
    monkeypatch.setattr(consumer, "AUTHORITY_ROOT", installation[2].parents[1] / "experiment-authority")
    monkeypatch.setattr(consumer, "_blueprint_gid", lambda: 0)
    monkeypatch.setattr(pair, "LANE_SCRATCH_ROOTS", (root,))
    use = consumer.RegisteredExperimentUse.admit(Path(born["path"]), expected_birth=born["birth"],
        expected_generation=born["generation"], now=lambda: 1200)
    entered = []
    def preflight(request):
        entered.append(True)
        fd = os.open(born["path"], os.O_RDONLY | os.O_DIRECTORY)
        try:
            with pytest.raises(BlockingIOError):
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            os.close(fd)
        raise ValueError("tiny_fixture_preflight_refused")
    monkeypatch.setattr(worker, "_preflight_inputs", preflight)
    result = pair.run_g1_development_pair(request_paths=paths, output_dir=Path(born["path"]),
        mode="local", _registered_use=use)
    assert result["status"] == "blocked" and entered == [True]
    attempt = result["attempts"][0]
    receipt = json.loads(Path(attempt["worker_result_path"]).read_bytes())
    assert receipt["phase_reached"] == "preflight" and receipt["teardown"] == {"environment": "not_started", "simulator": "not_started"}
    assert use._closed
    fd = os.open(born["path"], os.O_RDONLY | os.O_DIRECTORY)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        os.close(fd)
