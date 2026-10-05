"""Independent, provider-free review regressions for retained beta journals."""
import json

import pytest

from blueprint_pipeline import agent_run_executor as executor
from tests.test_agent_run_executor import FakeClient, _row


@pytest.mark.parametrize("mismatch", ["owner", "digest", "missing_owner", "matching"])
@pytest.mark.parametrize("state", ["staged", "native_execution_in_progress"])
def test_resumed_native_launch_rechecks_current_owner_and_admission(tmp_path, monkeypatch, mismatch, state):
    from blueprint_pipeline import controlled_native_queue as native

    row = _row(tmp_path)
    journal_dir = tmp_path / "journals"
    journal_dir.mkdir()
    (journal_dir / "run-1.json").write_text(json.dumps({
        "state": state, "run_id": "run-1", "pipeline_run_id": "attempt-1", "row": row,
        "canonical_job_id": "canonical-job-1", "capture_root": str(tmp_path),
        "execution_admission_digest": row["execution_admission_digest"],
    }))
    observed = {
        "run_id": "run-1", "state": "requested", "money_resolved": False,
        "cancellation_requested": False,
        "dispatch": {"pipeline_run_id": "attempt-1"},
        "execution_admission_digest": row["execution_admission_digest"],
    }
    if mismatch == "owner":
        observed["dispatch"]["pipeline_run_id"] = "other-attempt"
    elif mismatch == "missing_owner":
        observed["dispatch"] = None
    elif mismatch == "digest":
        observed["execution_admission_digest"] = "sha256:" + "f" * 64
    client = FakeClient([])
    client.get_run = lambda _: observed
    monkeypatch.setattr(native, "routes_controlled_request", lambda _: True)
    launches = []
    monkeypatch.setattr(native, "execute_staged_controlled_request", lambda **kwargs: launches.append(kwargs))
    executor.poll_once(client=client, capture_root=tmp_path, journal_dir=journal_dir,
                       terminal_reader=lambda **_: {"status": "pending"})
    assert len(launches) == (1 if mismatch == "matching" else 0), "current WebApp owner and admission must bind every resumed native launch"
