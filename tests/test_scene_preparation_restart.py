"""Restart and administrative release changes preserve source and paid limits."""
import json
from pathlib import Path

import pytest

from blueprint_pipeline import public_scene_host_input_intake
from blueprint_pipeline import task_evaluation_scene_progression as engine
from blueprint_pipeline import task_evaluation_scene_configuration_submission_publication as publication
from blueprint_pipeline import task_evaluation_launch_preparation_worker as worker
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_evaluation_scene_preparation_service import run_preparation_service
from blueprint_pipeline.task_evaluation_public_scene_attempt_factory import record
from tests.test_scene_preparation_installation import _installed
from tests.test_task_evaluation_completed_scene_progression import _config
from tests.test_task_evaluation_scene_configuration_submission_publication import Store
from tests.test_task_evaluation_scene_configuration_submission import SHA

pytestmark = pytest.mark.slow


def test_release_change_reuses_completed_mesh_source_without_debiting_paid_limits(tmp_path, monkeypatch):
    _, installation, config, intent_id = _installed(tmp_path, monkeypatch)
    class ReadbackStore(Store):
        def get_object(self, *, Bucket, Key):
            return {**super().get_object(Bucket=Bucket, Key=Key), "ContentLength": len(self.objects[Key])}
    store = ReadbackStore()
    monkeypatch.setattr(publication, "_s3_client", lambda _: store)
    monkeypatch.setattr(worker, "_s3_client", lambda _: store)
    monkeypatch.setattr(publication, "_verified_checkout_head", lambda: SHA)
    path = installation["config"]["path"]
    assert run_preparation_service(config_path=path)["scene_progression"]["results"][0]["phase"] == "construction_prepared"
    intent = Path(config["intent_root"]) / intent_id
    normalizations = {str(p): (record(p), p.stat().st_mtime_ns)
                      for p in Path(config["factory_output_root"]).rglob("mesh_normalization.v1.json")
                      if "normalized-sources" in p.parts}
    new_commit = "b" * 40
    receipts = Path(config["deployment_receipt_root"])
    old = json.loads(next(receipts.glob("*.json")).read_text())
    provenance = json.loads(Path(old["release_provenance"]["path"]).read_text())
    provenance["git_sha"] = new_commit
    proof_path = receipts.parent / new_commit / "deploy-release-provenance.json"
    proof_path.parent.mkdir()
    proof_path.write_text(json.dumps(provenance))
    new = {**old, "source_commit": new_commit,
        "intake_runtime": {"commit_proven": True, "source_commit": new_commit},
        "release_provenance": {**old["release_provenance"], **record(proof_path), "git_sha": new_commit}}
    (receipts / "next.json").write_text(json.dumps(new))
    for kind in ("scene-configuration", "splat-render"):
        root = Path(config["runtime_publication_root"]) / kind
        value = json.loads((root / (SHA + ".publication.v1.json")).read_text())
        value["source_commit"] = new_commit
        value["receipt_digest"] = canonical_digest(value, digest_field="receipt_digest")
        (root / (new_commit + ".publication.v1.json")).write_text(json.dumps(value))
    monkeypatch.setattr(public_scene_host_input_intake, "_verified_checkout_head", lambda: new_commit)
    monkeypatch.setattr(publication, "_verified_checkout_head", lambda: new_commit)
    result = run_preparation_service(config_path=path)
    assert result["scene_progression"]["results"][0]["phase"] == "construction_prepared", result
    assert len(list((intent / "preparation-attempts").glob("*.json"))) == 2
    assert not list((intent / "attempts").glob("*.json"))
    assert normalizations == {str(p): (record(p), p.stat().st_mtime_ns)
        for p in Path(config["factory_output_root"]).rglob("mesh_normalization.v1.json")
        if "normalized-sources" in p.parts}
    state = json.loads((intent / "progression.json").read_text())["state"]
    assert state["release_predecessors"][0]["basis"] == "preparation_only_no_execution_authority_issued"


def test_interrupted_submission_build_restarts_on_the_same_free_preparation(tmp_path, monkeypatch):
    from blueprint_pipeline.task_evaluation_scene_configuration_submission_inputs import Staging
    config, intent_id, intents, now = _config(tmp_path, monkeypatch, source_kind="mesh",
        extra={"activation_enabled": False})
    original = Staging.json
    def fail_after_staging(self, relative, value):
        result = original(self, relative, value)
        if relative == "configuration/task_template.v1.json":
            raise ValueError("interrupted_fixture_build")
        return result
    monkeypatch.setattr(Staging, "json", fail_after_staging)
    first = engine.process_scene_intents(config_path=config, now=now)
    assert first["results"][0]["status"] == "blocked"
    attempts = list((intents / intent_id / "preparation-attempts").glob("*.json"))
    monkeypatch.setattr(Staging, "json", original)
    second = engine.process_scene_intents(config_path=config, now=now)
    assert second["results"][0]["phase"] == "publication_ready", second
    assert list((intents / intent_id / "preparation-attempts").glob("*.json")) == attempts
    assert not list((intents / intent_id / "attempts").glob("*.json"))
