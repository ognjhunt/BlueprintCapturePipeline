import json
import pytest
from blueprint_pipeline import website_publication_recovery as worker
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from tests.test_scene_preparation_completion import prepared as prepared_fixture

prepared = prepared_fixture


def arrange(prepared, monkeypatch):
    args, run, result_path = prepared
    args["intent"]["request"]["task"]["task_id"] = "website-object-development"
    args["release"] = {"source_commit": "a" * 40}
    launch_path = run / "launch_receipt.json"
    launch = json.loads(launch_path.read_text())
    launch["status"] = "blocked"
    launch_path.write_text(json.dumps(launch))
    result = json.loads(result_path.read_text())
    result.update(
        configured_scene_published=False, execution_result_path=str(run / "provider.json")
    )
    result_path.write_text(json.dumps(result))
    profile_path = run / "launch_profile.json"
    profile = json.loads(profile_path.read_text())
    profile["allocator"] = {
        "argv": ["--scene-configuration-bundle-receipt", str(run / "bundle.json")]
    }
    profile["profile_digest"] = canonical_digest(profile, digest_field="profile_digest")
    profile_path.write_text(json.dumps(profile))
    launch["launch_profile_digest"] = profile["profile_digest"]
    launch_path.write_text(json.dumps(launch))
    monkeypatch.setenv(
        "BLUEPRINT_TASK_EVALUATION_SCENE_CONSTRUCTION_QUEUE_ROOT", str(run / "queue")
    )
    calls = []

    def recover(**kwargs):
        calls.append(kwargs)
        kwargs["output_root"].mkdir()
        (kwargs["output_root"] / worker.RECOVERED_LAUNCH_RECEIPT_FILENAME).write_text("{}")

    monkeypatch.setattr(worker, "recover_completed_configuration_publication", recover)
    return args, run, result_path, calls


def test_retries_delivery_without_repeating_completed_build_or_publication(prepared, monkeypatch):
    args, run, _, calls = arrange(prepared, monkeypatch)

    def fail(**kwargs):
        raise ValueError("temporary_website_delivery_failure")

    monkeypatch.setattr(worker, "activate_recovered_launch_receipt", fail)
    assert worker.reconcile_website_publication(**args)["status"] == "blocked"
    monkeypatch.setattr(
        worker, "activate_recovered_launch_receipt", lambda **kw: {"status": "activated"}
    )
    assert worker.reconcile_website_publication(**args)["status"] == "awaiting_execution"
    assert len(calls) == 1
    assert calls[0]["original_launch_receipt_path"] == run / "launch_receipt.json"


@pytest.mark.parametrize("fault", ["incomplete", "owner", "profile", "attempt"])
def test_does_not_recover_incomplete_or_unbound_output(prepared, monkeypatch, fault):
    args, run, result_path, calls = arrange(prepared, monkeypatch)
    if fault == "incomplete":
        d = json.loads(result_path.read_text())
        d["configuration_completed"] = False
        result_path.write_text(json.dumps(d))
    elif fault == "owner":
        args["intent"]["intent_digest"] = "sha256:" + "f" * 64
    elif fault == "profile":
        p = run / "launch_profile.json"
        d = json.loads(p.read_text())
        d["profile_digest"] = "bad"
        p.write_text(json.dumps(d))
    else:
        p = next(
            (
                worker.Path(args["config"]["intent_root"])
                / args["intent"]["intent_id"]
                / "attempts"
            ).glob("*.json")
        )
        d = json.loads(p.read_text())
        d["attempt_digest"] = "bad"
        p.chmod(0o600)
        p.write_text(json.dumps(d))
    worker.reconcile_website_publication(**args)
    assert calls == []


def test_publication_failure_is_bounded_per_release(prepared, monkeypatch):
    args, _, _, calls = arrange(prepared, monkeypatch)

    def fail(**kwargs):
        calls.append(kwargs)
        raise ValueError("invalid_artifact")

    monkeypatch.setattr(worker, "recover_completed_configuration_publication", fail)
    for _ in range(2):
        assert worker.reconcile_website_publication(**args)["blockers"] == ["invalid_artifact"]
    assert len(calls) == 1


def test_owner_attempt_uses_intake_cross_runtime_digest(prepared, monkeypatch):
    from blueprint_pipeline.decision_evidence_contracts import cross_runtime_canonical_digest

    args, _, _, calls = arrange(prepared, monkeypatch)
    path = next(
        (
            worker.Path(args["config"]["intent_root"]) / args["intent"]["intent_id"] / "attempts"
        ).glob("*.json")
    )
    value = json.loads(path.read_text())
    value["maximum_spend_usd"] = 2.0
    value["reserved_at_epoch"] = 101.125
    value["attempt_digest"] = cross_runtime_canonical_digest(value, digest_field="attempt_digest")
    assert value["attempt_digest"] != canonical_digest(value, digest_field="attempt_digest")
    path.chmod(0o600)
    path.write_text(json.dumps(value))
    monkeypatch.setattr(worker, "activate_recovered_launch_receipt", lambda **kw: {})
    assert worker.reconcile_website_publication(**args)["status"] == "awaiting_execution"
    assert len(calls) == 1
