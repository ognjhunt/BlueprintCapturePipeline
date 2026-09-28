"""An unsupported inventory must be refused before real source work or spend."""
import json

from blueprint_pipeline import task_evaluation_scene_progression as engine
from blueprint_pipeline.task_evaluation_scene_policy_capability import supported_policy_candidates
from blueprint_pipeline.decision_evidence_contracts import cross_runtime_canonical_digest
from tests.test_task_evaluation_completed_scene_progression import _config


def test_unknown_checkpoint_inventory_refuses_before_source_or_reservation(tmp_path, monkeypatch):
    config, intent_id, intents, now = _config(tmp_path, monkeypatch)
    path = intents / intent_id / "intent.json"
    intent = json.loads(path.read_text())
    intent["request"]["execution"]["policy_candidates"][0]["artifact_digest"] = "sha256:" + "0" * 64
    intent["intent_digest"] = cross_runtime_canonical_digest(intent, digest_field="intent_digest")
    # The intake writer intentionally seals intent.json read-only.  This test
    # models an independently re-sealed retained request, rather than bypassing
    # the file mode and accidentally testing a permission failure.
    path.chmod(0o640)
    try:
        path.write_text(json.dumps(intent))
    finally:
        path.chmod(0o440)
    def forbidden(*args, **kwargs):
        raise AssertionError("source resolution must not run for unavailable checkpoint")
    monkeypatch.setattr(engine, "_source", forbidden)
    result = engine.process_scene_intents(config_path=config, now=now)
    assert result["results"][0]["status"] == "needs_input"
    assert result["results"][0]["blockers"] == ["scene_policy_checkpoint_capability_unavailable"]
    assert not list((intents / intent_id / "attempts").glob("*.json"))
    assert not list((intents / intent_id / "preparation-attempts").glob("*.json"))
    assert json.loads(path.read_text()) == intent


def test_admitted_inventory_uses_same_digests_as_real_canary_setup(tmp_path, monkeypatch):
    from blueprint_pipeline.task_evaluation_policy_canary_scene_setup import EXPECTED_CANDIDATES, CANDIDATE_IDS
    assert supported_policy_candidates() == [{"id": name, "artifact_digest":
        EXPECTED_CANDIDATES[name]["checkpoint_inventory_digest"]} for name in CANDIDATE_IDS]
    config, _, _, now = _config(tmp_path, monkeypatch)
    result = engine.process_scene_intents(config_path=config, now=now)
    assert result["results"][0]["phase"] == "publication_ready", result
