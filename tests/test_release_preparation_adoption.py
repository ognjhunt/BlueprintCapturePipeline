"""A deploy adopts a $0 website preparation made by a retained earlier release (dark launch).

Before: every deploy retired the website preparation in flight and rebuilt it
on the new release (one more preparation attempt per deploy), and the new
release's workers refused the old request by its commit name.
Now, with ``BLUEPRINT_RELEASE_PREPARATION_ADOPTION=enabled`` on the host, the
new release keeps the preparation when the producer code's contract identity
and the inputs are unchanged, and only the commit-name comparisons of the
preparation worker, the activation staging and the activation worker accept it.
With the switch off, every check refuses exactly as before.
"""
from __future__ import annotations

import json
import os
import pwd
from pathlib import Path

import pytest

from blueprint_pipeline import public_scene_host_input_intake
from blueprint_pipeline import release_preparation_adoption as rule
from blueprint_pipeline import task_evaluation_scene_configuration_submission_publication as publication
from blueprint_pipeline import task_evaluation_scene_progression as engine
from blueprint_pipeline import task_evaluation_scene_release_adoption as progression_rule
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_evaluation_launch_preparation_worker import process_launch_preparation_queue

pytestmark = pytest.mark.slow


# ----------------------------------------------------------------- contract identity

def test_contract_identity_follows_the_producer_code_and_nothing_else(tmp_path):
    package = tmp_path / "pkg"
    package.mkdir()
    (package / "root.py").write_text("def f():\n    from .lazy import x\n")
    (package / "lazy.py").write_text("from .deep import y\nx = 1\n")
    (package / "deep.py").write_text("y = 1\n")
    (package / "unrelated.py").write_text("z = 1\n")
    roots = ("root",)
    assert rule.producer_closure(package, roots) == ["deep.py", "lazy.py", "root.py"]
    before = rule.contract_identity(package, roots)
    (package / "unrelated.py").write_text("z = 2\n")
    rule._identity.cache_clear()
    assert rule.contract_identity(package, roots) == before
    (package / "deep.py").write_text("y = 2\n")
    rule._identity.cache_clear()
    assert rule.contract_identity(package, roots) != before


def test_the_real_closure_covers_the_preparation_producers():
    closure = rule.producer_closure()
    for name in ("website_scene_dispatch.py", "website_native_submission.py",
                 "task_evaluation_launch_preparation_worker.py", "release_preparation_adoption.py"):
        assert name in closure
    assert rule.contract_identity() == rule.contract_identity()


def test_the_switch_is_off_unless_the_host_names_it(monkeypatch):
    monkeypatch.delenv(rule.SWITCH_ENV, raising=False)
    assert rule.enabled() is False
    for value in ("1", "true", "on", ""):
        assert rule.enabled({rule.SWITCH_ENV: value}) is False
    assert rule.enabled({rule.SWITCH_ENV: "enabled"}) is True


def test_an_adopted_activation_binds_only_a_retained_toolchain(tmp_path):
    running, prepared = "b" * 40, "a" * 40
    current = tmp_path / "scene-configuration" / running
    current.mkdir(parents=True)
    assert rule.adopted_toolchain(current, None) == current
    adoption = {"adopting_commit": running, "prepared_commit": prepared}
    with pytest.raises(ValueError, match="not_retained"):
        rule.adopted_toolchain(current, adoption)
    (tmp_path / "scene-configuration" / prepared).mkdir()
    assert rule.adopted_toolchain(current, adoption) == tmp_path / "scene-configuration" / prepared
    with pytest.raises(ValueError, match="layout_unknown"):
        rule.adopted_toolchain(tmp_path / "elsewhere", adoption)


# ------------------------------------------------------------- a website job across deploys

def _website(tmp_path, monkeypatch):
    from tests.test_website_native_submission import prepare_website_construction
    job = prepare_website_construction(tmp_path, monkeypatch, False, run_worker=False)
    config_path = job["config_path"]
    config = json.loads(config_path.read_text())
    config["activation_enabled"] = False  # stop at the $0 boundary: nothing paid can be reserved
    config["config_digest"] = canonical_digest(config, digest_field="config_digest")
    config_path.write_text(json.dumps(config))
    intent = next((tmp_path / "intents").glob("scene-*"))
    state = json.loads((intent / "progression.json").read_text())["state"]
    assert state["preparation_contract"] == rule.contract_identity()
    return job, config, intent, state["attempt_commit"]


def _deploy(config, monkeypatch, commit):
    path = Path(config["release_binding_path"])
    release = json.loads(path.read_text())
    release.update(source_commit=commit, runtime_digest="sha256:" + commit[:1] * 64)
    release["release_digest"] = canonical_digest(release, digest_field="release_digest")
    path.write_text(json.dumps(release))
    monkeypatch.setattr(public_scene_host_input_intake, "_verified_checkout_head", lambda: commit)
    monkeypatch.setattr(publication, "_verified_checkout_head", lambda: commit)


def _tick(job):
    [row] = engine.process_scene_intents(config_path=job["config_path"], publisher=job["publish"])["results"]
    return row


def _worker(tmp_path, job, commit):
    return process_launch_preparation_queue(queue_root=job["queue"], input_root=tmp_path / "worker-inputs",
        allowed_uri_prefixes=["s3://blueprint/task-evaluation/production-inputs/"],
        service_account=pwd.getpwuid(os.geteuid()).pw_name, source_commit=commit, fetcher=job["fetch"],
        construction_queue_root=tmp_path / "construction")["results"][0]


def _attempts(intent):
    return sorted(p.name for p in (intent / "preparation-attempts").glob("*.json"))


def _state(intent):
    return json.loads((intent / "progression.json").read_text())["state"]


def test_switch_off_keeps_todays_rebuild_and_todays_refusals(tmp_path, monkeypatch):
    monkeypatch.delenv(rule.SWITCH_ENV, raising=False)
    job, config, intent, prepared = _website(tmp_path, monkeypatch)
    _deploy(config, monkeypatch, "b" * 40)
    row = _tick(job)
    assert row["status"] == "blocked", row  # the rebuild on B, as before
    assert len(_attempts(intent)) == 2
    assert [entry["basis"] for entry in _state(intent)["release_predecessors"]] == [
        "preparation_only_no_execution_authority_issued"]
    assert not (job["queue"] / rule.DIRECTORY).exists()
    refused = _worker(tmp_path, job, "b" * 40)
    assert refused["status"] == "blocked"
    assert refused["blockers"] == ["launch_preparation_worker_source_commit_mismatch"]


def test_switch_on_adopts_a_preparation_still_running_and_finishes_it_without_a_second(tmp_path, monkeypatch):
    monkeypatch.setenv(rule.SWITCH_ENV, rule.SWITCH_ON)
    job, config, intent, prepared = _website(tmp_path, monkeypatch)
    pending = sorted(p.name for p in (job["queue"] / "pending").glob("*.json"))
    _deploy(config, monkeypatch, "b" * 40)
    row = _tick(job)
    assert (row["status"], row["phase"]) == ("running", "source_preparation"), row
    assert len(_attempts(intent)) == 1
    adoption = _state(intent)["release_adoption"]
    assert (adoption["prepared_commit"], adoption["adopting_commit"]) == (prepared, "b" * 40)
    assert sorted(p.name for p in (job["queue"] / "pending").glob("*.json")) == pending  # no second request
    result = _worker(tmp_path, job, "b" * 40)
    assert result["status"] == "queued_for_production_scene_configuration", result.get("blockers")
    assert result["source_commit"] == prepared
    assert result["release_adoption"]["worker_source_commit"] == "b" * 40
    assert result["release_adoption"]["prepared_commit"] == prepared
    row = _tick(job)
    assert (row["status"], row["phase"]) == ("awaiting_execution", "construction_prepared"), row
    assert len(_attempts(intent)) == 1
    assert not list((intent / "attempts").glob("*.json"))  # no paid reservation


@pytest.mark.parametrize("switch", ["on", "off"])
def test_n_deploys_make_one_preparation_and_a_contract_change_exactly_one_more(tmp_path, monkeypatch, switch):
    if switch == "on":
        monkeypatch.setenv(rule.SWITCH_ENV, rule.SWITCH_ON)
    else:
        monkeypatch.delenv(rule.SWITCH_ENV, raising=False)
    job, config, intent, _prepared = _website(tmp_path, monkeypatch)
    commits = ["b" * 40, "c" * 40, "d" * 40]
    for commit in commits:
        _deploy(config, monkeypatch, commit)
        _tick(job)
    assert len(_attempts(intent)) == (1 if switch == "on" else 1 + len(commits))
    if switch == "off":
        return
    changed = "sha256:" + "9" * 64
    monkeypatch.setattr(progression_rule, "contract_identity", lambda: changed)
    _deploy(config, monkeypatch, "e" * 40)
    _tick(job)
    assert len(_attempts(intent)) == 2
    assert _state(intent)["attempt_commit"] == "e" * 40
    _tick(job)
    assert len(_attempts(intent)) == 2


def test_an_adopted_preparation_refused_by_the_new_worker_is_rebuilt_once(tmp_path, monkeypatch):
    monkeypatch.setenv(rule.SWITCH_ENV, rule.SWITCH_ON)
    job, config, intent, _prepared = _website(tmp_path, monkeypatch)
    _deploy(config, monkeypatch, "b" * 40)
    _tick(job)
    original = rule.contract_identity
    monkeypatch.setattr(rule, "contract_identity", lambda *a, **k: "sha256:" + "8" * 64)  # the worker's code differs
    assert _worker(tmp_path, job, "b" * 40)["blockers"] == ["launch_preparation_worker_source_commit_mismatch"]
    monkeypatch.setattr(rule, "contract_identity", original)
    row = _tick(job)
    assert (row["phase"], row["blockers"]) == ("previous_release", ["release_adoption_refused"]), row
    _tick(job)
    _tick(job)
    assert len(_attempts(intent)) == 2
    assert len(_state(intent)["release_predecessors"]) == 1


# ------------------------------------------------------------------ row 9: activation staging

def test_activation_staging_accepts_only_a_recorded_adoption(tmp_path, monkeypatch):
    from tests.test_task_evaluation_scene_configuration_activation_automation import (
        COMMIT, PREPARATION_ID, _advance, _intent, _preparation,
    )
    other = "f" * 40
    intent_root, _ = _intent(tmp_path)
    result_path = _preparation(tmp_path)
    monkeypatch.setenv(rule.SWITCH_ENV, rule.SWITCH_ON)
    observed, _, _ = _advance(tmp_path, result_path, intent_root, running_commit=other)
    assert observed["status"] == "preparation_bound_to_superseded_release"  # no record: today's answer
    rule.record_adoption(queue_root=tmp_path / "preparations",
        request_digest=rule.request_digest_from_queue_name(result_path.name), preparation_id=PREPARATION_ID,
        prepared_commit=COMMIT, adopting_commit=other, identity=rule.contract_identity(),
        intent_id="scene-fixture", attempt_id="source-fixture")
    monkeypatch.delenv(rule.SWITCH_ENV)
    observed, _, _ = _advance(tmp_path, result_path, intent_root, running_commit=other)
    assert observed["status"] == "preparation_bound_to_superseded_release"  # switch off: today's answer
    monkeypatch.setenv(rule.SWITCH_ENV, rule.SWITCH_ON)
    observed, _, _ = _advance(tmp_path, result_path, intent_root, running_commit=other)
    assert observed["status"] == "scene_configuration_activation_queued", observed
    assert observed["expected_production_commit"] == COMMIT
    assert observed["release_adoption"]["prepared_commit"] == COMMIT
    assert observed["release_adoption"]["worker_source_commit"] == other


def test_a_refused_adopted_activation_is_reported_for_one_rebuild(tmp_path):
    results = tmp_path / "activations" / "results"
    results.mkdir(parents=True)
    assert rule.blocked_activation(tmp_path / "activations", "activation-x") is None
    (results / "activation-x-1.json").write_text(json.dumps({
        "activation_id": "activation-x", "status": "blocked", "provider_mutation_performed": False,
        "blockers": ["launch_activation_scene_configuration_toolchain_invalid"]}))
    assert rule.blocked_activation(tmp_path / "activations", "activation-x") == [
        "launch_activation_scene_configuration_toolchain_invalid"]
    state = {"release_adoption": {"prepared_commit": "a" * 40}}
    progression_rule.refuse_adoption(state, ["x"])
    assert state["release_adoption"]["refused"] == ["x"]


# ------------------------------------------------------------------ row 10: activation worker

@pytest.mark.parametrize("adopted", [False, True])
def test_activation_worker_accepts_another_commit_only_under_a_recorded_adoption(tmp_path, monkeypatch, adopted):
    import tests.test_task_evaluation_launch_activation_worker as fixture
    from datetime import datetime, timedelta, timezone
    from blueprint_pipeline.task_evaluation_launch_activation_queue import stage_launch_activation_request
    from blueprint_pipeline.task_evaluation_launch_activation_worker import process_launch_activation_queue
    from blueprint_pipeline.task_evaluation_launch_preparation_contract import launch_preparation_request_digest

    preparation, preparation_result, payloads, preparation_queue, input_root = (
        fixture._stage_verified_preparation(tmp_path))
    lane = "native_task_arena_construction"
    request = fixture.activation_request(lane=lane)
    request["activation_id"] = "activation-scene-841007-adoption"
    request["preparation"] = {"preparation_id": preparation["preparation_id"],
                              "request_digest": launch_preparation_request_digest(preparation),
                              "result_digest": preparation_result["result_digest"]}
    now = datetime.now(timezone.utc)
    request["authorization"]["authorized_on"] = now.isoformat()
    request["authorization"]["standing_authorization_expires_at"] = (now + timedelta(hours=1)).isoformat()
    for name, reference in request["lineage"].items():
        if name != "kind":
            content = json.dumps({"kind": name}).encode()
            reference.update(fixture._reference(str(reference["uri"]), content))
            payloads[str(reference["uri"])] = content
    window_bytes = fixture._release_window(request, now)
    request["release_window"] = fixture._reference(
        "s3://blueprint-production-inputs/coordinator-release-windows/release-window.json", window_bytes)
    payloads[request["release_window"]["uri"]] = window_bytes
    queue = tmp_path / "activation-queue"
    stage_launch_activation_request(value=request, queue_root=queue, submitted_by="blueprint-webapp")
    running = "f" * 40
    monkeypatch.setenv(rule.SWITCH_ENV, rule.SWITCH_ON)
    if adopted:
        rule.record_adoption(queue_root=preparation_queue, request_digest=request["preparation"]["request_digest"],
            preparation_id=preparation["preparation_id"], prepared_commit=request["expected_production_commit"],
            adopting_commit=running, identity=rule.contract_identity(), intent_id="scene-x", attempt_id="source-x")
    authorization_dir = tmp_path / "standing-authorizations"
    prepared: list[dict] = []

    def prepare(**kwargs):
        context = json.loads(Path(kwargs["context_path"]).read_text())
        prepared.append(context)
        set_root = Path(context["operations"]["set_root"])
        set_root.mkdir(parents=True)
        profile = {"profile_id": "scene-841007-construction-r1", "profile_digest": "sha256:" + "d" * 64}
        steps = []
        for step_id, name, value in (("live_profile", "live_profile-r1.v1.json", profile),
                                     ("profile_publication", "publication.json", {"status": "published"})):
            path = set_root / name
            path.write_text(json.dumps(value))
            steps.append({"step_id": step_id, "artifact_path": str(path),
                          "artifact_sha256": fixture._bytes_digest(path.read_bytes())})
        authorization_dir.mkdir(parents=True)
        grant = authorization_dir / "scene-841007-construction-r1.json"
        grant.write_text(json.dumps({**profile, "max_launches": 1,
            "max_total_spend_usd": preparation["spend"]["hard_cap_usd"], "provider_mutation_performed": False}))
        steps.append({"step_id": "standing_authorization", "artifact_path": str(grant),
                      "artifact_sha256": fixture._bytes_digest(grant.read_bytes())})
        receipt = {"schema_version": "paid_lane_launch_preparation.v1", "status": "prepared",
                   "source_commit": request["expected_production_commit"], "completed_steps": steps,
                   "provider_allocation_performed": False, "paid_inference_performed": False}
        Path(kwargs["receipt_path"]).write_text(json.dumps(receipt))
        return receipt

    def fetch(uri, destination, maximum_bytes):
        destination.write_bytes(payloads[uri])

    run = process_launch_activation_queue(queue_root=queue, preparation_queue_root=preparation_queue,
        preparation_input_root=input_root, activation_root=tmp_path / "activations",
        allowed_uri_prefixes=["s3://blueprint-production-inputs/"], service_account=fixture.SERVICE_ACCOUNT,
        service_group=pwd.getpwuid(os.geteuid()).pw_name, repository_root=tmp_path,
        destination_prefix="s3://blueprint-production-inputs/activated",
        release_window_prefix="s3://blueprint-production-inputs/coordinator-release-windows/",
        profile_dir=tmp_path / "profiles", webapp_catalog=tmp_path / "catalog.json",
        standing_authorization_dir=authorization_dir, source_commit=running, fetcher=fetch, preparer=prepare)
    [result] = run["results"]
    if not adopted:
        assert result["blockers"] == ["launch_activation_worker_source_commit_mismatch"]
        assert prepared == []
        return
    assert result["status"] == "profile_authority_materialized_no_execution", result
    assert result["provider_mutation_performed"] is False and result["paid_execution_requested"] is False
    assert result["release_adoption"]["prepared_commit"] == request["expected_production_commit"]
    assert result["release_adoption"]["worker_source_commit"] == running
    assert result["result_digest"] == canonical_digest(result, digest_field="result_digest")
    assert [context["operations"]["source_commit"] for context in prepared] == [request["expected_production_commit"]]
