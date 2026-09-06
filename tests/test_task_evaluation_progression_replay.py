"""Real retained queue consumers gate activation before any external mutation."""
import json

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_evaluation_progression_replay import replay_progression_admission
from tests.test_task_evaluation_scene_configuration_activation_automation import (
    _preparation, _intent, _advance,
)


def test_actual_parent_consumers_pass_and_report_binds_retained_bytes(tmp_path):
    path = _preparation(tmp_path)
    report = replay_progression_admission(result_path=path, queue_root=path.parent.parent,
        replay_root=tmp_path / "replay", child_queue_root=tmp_path / "children")
    assert report["status"] == "accepted"
    assert len(report["next_consumer_admission"]) == 2
    saved = json.loads(open(report["report_path"]).read())
    assert saved["report_digest"] == canonical_digest(saved, digest_field="report_digest")
    assert saved["provider_mutation_performed"] is False


def test_activation_never_publishes_when_next_consumer_refuses(tmp_path):
    path = _preparation(tmp_path)
    result = json.loads(path.read_text())
    result["references"] = []
    result["result_digest"] = canonical_digest(result, digest_field="result_digest")
    path.write_text(json.dumps(result))
    intent_root, _ = _intent(tmp_path)
    observed, lineage, window = _advance(tmp_path, path, intent_root)
    assert observed["status"] == "scene_configuration_lookahead_blocked"
    assert any("configured_controls_provisioning_reference_missing" in b for b in observed["blockers"])
    assert not list((tmp_path / "activations").rglob("*.json"))
    assert observed["provider_mutation_performed"] is False
    assert lineage.published == window.published == {}
