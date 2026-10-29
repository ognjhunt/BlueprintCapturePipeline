"""Native current-input publication cannot overwrite a reclaimed generation."""

from datetime import datetime, timedelta, timezone
import json

from blueprint_pipeline import pubsub_handoff_listener as listener
from blueprint_pipeline import website_native_appearance as appearance
from blueprint_pipeline import website_scene_runtime_inputs as runtime
from blueprint_pipeline.core.handoff_lease import HandoffLeaseError, bind_handoff_lease
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.website_task_preparation import compile_website_scene_preparation
from tests.test_website_task_preparation import _arguments


def test_actual_runtime_inputs_preserve_b_after_loss_during_appearance(tmp_path, monkeypatch):
    args = _arguments(tmp_path)
    capture = tmp_path / "capture"
    capture.mkdir()
    args["output_root"] = capture / "pipeline/website_scene_preparation"
    preparation = compile_website_scene_preparation(**args)
    output = args["output_root"] / "native"
    kwargs = dict(preparation=preparation, base_scene=args["base_scene"],
        source_geometry=args["source_geometry"], task_masks=args["task_masks"], output_root=output)
    baseline = runtime.prepare_website_runtime_inputs(**kwargs)
    context = args["task_context"]

    def claim(owner):
        state, value = listener._claim_job_lease(capture, scene_id=context["scene_id"],
            capture_id=context["capture_id"], owner=owner, lease_seconds=60)
        assert state == "claimed"
        return value

    lease = claim("review-a")
    heartbeat = listener._JobLeaseHeartbeat(capture_root=capture, owner="review-a",
        token=lease["lease_token"], lease_seconds=60)
    current = output / "runtime_inputs.json"
    ledger_path = capture / listener.JOB_LEDGER_FILENAME
    newer = {}

    def delayed_appearance(**_kwargs):
        with listener._locked_job_ledger(capture) as ledger:
            listener.write_json(ledger_path, {**ledger,
                "lease_expires_at": (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()})
        claim("review-b")
        value = {**baseline, "owner_generation": "review-b"}
        value["digest"] = canonical_digest(value, digest_field="digest")
        current.write_text(json.dumps(value))
        newer.update(current=current.read_bytes(), ledger=ledger_path.read_bytes())
        raise ValueError("retained_appearance_pending")

    monkeypatch.setattr(appearance, "prepare_native_appearance", delayed_appearance)
    with bind_handoff_lease(heartbeat.checkpoint, heartbeat.effect):
        try:
            runtime.prepare_website_runtime_inputs(**kwargs)
        except HandoffLeaseError:
            pass
    assert newer
    assert ledger_path.read_bytes() == newer["ledger"]
    assert current.read_bytes() == newer["current"], "stale A replaced B native/runtime_inputs.json"
