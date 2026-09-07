"""Real retained queue consumers gate activation before any external mutation."""
import json
from pathlib import Path

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_evaluation_progression_replay import replay_progression_admission
from tests.test_task_evaluation_scene_configuration_activation_automation import (
    _preparation, _intent, _advance,
)


def test_owned_queue_overrides_legacy_input_default(tmp_path, monkeypatch):
    from blueprint_pipeline.task_evaluation_sam31_parent_evidence import configured_parent_route
    path = _preparation(tmp_path)
    queue = path.parent.parent
    envelope = json.loads(next((queue / "materialized").glob("*.json")).read_text())
    inputs = tmp_path / "owned-inputs"
    config = {"schema_version": "task_evaluation_scene_progression_config.v1",
              "preparation_queue_root": str(queue),
              "preparation_worker": {"input_root": str(inputs)}}
    config["config_digest"] = canonical_digest(config, digest_field="config_digest")
    config_path = tmp_path / "scene.json"
    config_path.write_text(json.dumps(config))
    monkeypatch.setenv("BLUEPRINT_TASK_EVALUATION_SCENE_PROGRESSION_CONFIG", str(config_path))
    job = {"parent_preparation_id": envelope["request"]["preparation_id"],
           "parent_request_digest": envelope["request_digest"]}
    # Both entry routes must select the operator-bound CAS, including when the
    # caller already resolved the owned queue but retained the legacy input default.
    for incoming in (queue, tmp_path / "legacy-queue"):
        assert configured_parent_route(job, Path(incoming), tmp_path / "legacy-inputs") == (queue, inputs)


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
