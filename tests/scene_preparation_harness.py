"""Hermetic run of the scene-preparation stage from a website intent to `completed/scene_prepared`.

Real code: website compile + runtime inputs + intake `stage_scene_intent`, `register_website_preparation`,
`run_preparation_service` (progression pass, owned preparation worker, second pass) with
`activation_enabled: true`, the paid reservation, project-spend refresh, activation provisioning, the
activation automation (provider-zero gate, lookahead replay, release window, standing-authorization and
spend gates, launch submission), `reconcile_launches`, the completion join and `scene_intent_status`.

Faked (machine boundary only): the object store (in-memory `Store`), the Vast inventory (a canned
provider-zero receipt), the WebApp launch submitter and receipt sync, the GPU launch itself (a canned
dispatcher/allocator run with a guard report), the robot-free readiness read of the published revision.
The launch-activation worker (`aw`) is not run: it needs the scene-configuration toolchain archive and the
lane graph; its published profile, standing authorization and result are canned as the existing tests do.
"""
from __future__ import annotations

import contextlib
import json
import os
import pwd
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from blueprint_pipeline import control_plane_capacity_controller as capacity
from blueprint_pipeline import control_plane_disk_budget as disk
from blueprint_pipeline import public_scene_host_input_intake
from blueprint_pipeline import robot_free_readiness_retention as retention
from blueprint_pipeline import task_evaluation_launch_preparation_worker as worker
from blueprint_pipeline import task_evaluation_launch_webapp_sync as webapp_sync
from blueprint_pipeline import task_evaluation_scene_configuration_activation_automation as automation
from blueprint_pipeline import task_evaluation_scene_configuration_submission_publication as publication
from blueprint_pipeline import task_evaluation_scene_intake as intake
from blueprint_pipeline import task_evaluation_scene_progression as progression
from blueprint_pipeline import team_scene_packet_worker as packets
from blueprint_pipeline.common import write_json
from blueprint_pipeline.decision_evidence_contracts import canonical_digest, cross_runtime_canonical_json
from blueprint_pipeline.task_evaluation_launch_preparation_queue import ensure_launch_preparation_queue_root
from blueprint_pipeline.task_evaluation_launch_reconciler import reconcile_launches
from blueprint_pipeline.task_evaluation_launch_terminal_evidence import _scene_configuration_terminal_projection
from blueprint_pipeline.task_evaluation_public_scene_attempt_factory import RELEASE_SCHEMA, record
from blueprint_pipeline.task_evaluation_scene_execution_authority import bind_scene_attempt
from blueprint_pipeline.task_evaluation_scene_preparation_service import run_preparation_service
from blueprint_pipeline.website_scene_dispatch import register_website_preparation
from tests import test_website_native_submission as native
from tests.test_project_spend_reconciliation import _human_baseline
from tests.test_task_evaluation_launch_reconciler_stale_profile import _zero_guard
from tests.test_task_evaluation_scene_configuration_activation_automation import _provider_zero, _publisher
from tests.test_task_evaluation_scene_configuration_submission import SHA
from tests.test_task_evaluation_scene_configuration_submission_publication import Store

_real_time = time.time
CLOCK = datetime(2026, 11, 15, 12, 0, tzinfo=timezone.utc).timestamp()
RELEASE = SHA
ACCOUNT = pwd.getpwuid(os.geteuid()).pw_name
PREFIX = "s3://blueprint/task-evaluation/production-inputs/"
HARD_STOPS = (None, "revoked", "policy_candidate", "unregistered")


class _ReadbackStore(Store):
    def get_object(self, *, Bucket, Key):
        return {**super().get_object(Bucket=Bucket, Key=Key), "ContentLength": len(self.objects[Key])}


def _sealed(path: Path, value: dict, field: str) -> dict:
    value[field] = canonical_digest(value, digest_field=field)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, value)
    return value


def _sha_file(path: Path) -> str:
    import hashlib
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _config(root: Path, kwargs: dict, release_commit: str) -> Path:
    """The installed service config (inst:231-302) with the website source roots of the native fixture."""
    _sealed(root / "machinery.json", {"schema_version": "task_evaluation_completed_scene_machinery.v1",
        "maximum_preparation_spend_usd": 0, "provider": "control_plane"}, "machinery_digest")
    _sealed(root / "release.json", {"schema_version": RELEASE_SCHEMA, "source_commit": release_commit,
        "runtime_digest": "sha256:" + "f" * 64, "repo_root": str(root / "repo"),
        "runtime_publication_root": str(kwargs["runtime_publication_root"]),
        "namespace_timestamp": kwargs["namespace_timestamp"], "release_admission_mode": "promoted",
        **{key: record(kwargs[key + "_path"]) for key in ("deploy_receipt", "release_provenance", "release_environment")}},
        "release_digest")
    state = root / "state"
    queue = state / "task-evaluation-owned-scene-preparations"
    ensure_launch_preparation_queue_root(queue)
    for name in ("activation-intents", "launch-runs", "scene-project-spend", "service", "disk", "pins"):
        (state / name).mkdir(parents=True, exist_ok=True)
    baseline, _ = _human_baseline(root / "spend-baseline.json")
    from blueprint_pipeline.project_spend_reconciliation import materialize_project_spend_reconciliation
    seed = root / "spend-seed.json"
    materialize_project_spend_reconciliation(baseline_authority_path=baseline, posted_reconciliation_paths=[],
        expected_coverage_ids=[], completeness_reference=str(baseline), authorized_by="fixture-owner",
        authorized_on=datetime.fromtimestamp(CLOCK, timezone.utc).isoformat(), output_path=seed)
    monitor = _sealed(state / "scene-project-spend/monitor.json", {
        "schema_version": "task_evaluation_scene_project_spend_monitor.v1", "scene_root": str(root / "intents"),
        "seed_reconciliation_path": str(seed), "output_root": str(state / "scene-project-spend/outputs"),
        "current_path": str(state / "scene-project-spend/current.json")}, "config_digest")
    path = root / "config.json"
    _sealed(path, {"schema_version": progression.CONFIG_SCHEMA, "intent_root": str(root / "intents"),
        "public_source_binding_root": str(root / "unused"), "website_source_binding_root": str(root / "bindings"),
        "completed_source_machinery_path": str(root / "machinery.json"),
        "release_binding_path": str(root / "release.json"), "factory_output_root": str(root / "factories"),
        "trusted_clients": ["blueprint-webapp"], "maximum_intents_per_pass": 16,
        "maximum_http_submission_attempts": 2, "submission_enabled": True,
        "submission_transport": "local_owned_queue", "service_account": ACCOUNT,
        "preparation_queue_root": str(queue), "publication_lock_root": str(state / "locks"),
        "service_status_path": str(state / "service/latest.json"),
        "launch_execution_root": str(state / "launch-runs"),
        "activation_enabled": True, "activation_intent_root": str(state / "activation-intents"),
        "project_spend_current_path": monitor["current_path"],
        "preparation_worker": {"input_root": str(root / "inputs/prepared-references"),
            "construction_queue_root": str(state / "constructions"), "disk_reservation_root": str(state / "disk"),
            "storage_pins_root": str(state / "pins"), "max_messages": 1, "allowed_uri_prefixes": [PREFIX]}},
        "config_digest")
    os.environ["BLUEPRINT_SCENE_PROJECT_SPEND_CONFIG"] = str(state / "scene-project-spend/monitor.json")
    return path


def _launch_environment(mp: pytest.MonkeyPatch, root: Path) -> None:
    roots = {"CONFIGURED_CONTROLS_STATE_ROOT": "configured-controls", "LAUNCH_ACTIVATION_QUEUE_ROOT": "activations",
             "LAUNCH_PROFILE_DIR": "profiles", "STANDING_AUTHORIZATION_DIR": "standing",
             "CONTROL_PLANE_REPO": "repo", "WEBAPP_SUBMISSION_SECRET_FILE": "secret"}
    for name, value in roots.items():
        mp.setenv("BLUEPRINT_TASK_EVALUATION_" + name, str(root / "state" / value))
    mp.setenv("BLUEPRINT_TASK_EVALUATION_WEBAPP_SUBMISSION_URL",
              "https://tryblueprint.io/api/internal/task-evaluation-launch-submissions")
    mp.setenv(packets.QUEUE_ROOT_ENV, str(root / "state/team-scene-packets"))


def _canned_activation(root: Path, row: dict, attempt: dict, task_id: str, release_commit: str, now: datetime) -> None:
    """What the launch-activation worker publishes (as tests/test_website_scene_configuration_launch_in_progression)."""
    state = root / "state"
    profile_id = f"task-evaluation-scene-configuration-{release_commit}-r1-binding-0a1b2c3d4e5f"
    profile = _sealed(state / "profiles" / f"{profile_id}.json", {
        "schema_version": "task_evaluation_launch_profile.v1", "profile_id": profile_id,
        "source_commit": release_commit, "claim_ceiling": "development_only", **bind_scene_attempt(attempt),
        "task_evaluation_run": {"run_mode": "scene_configuration", "configuration_run_id": "harness-configuration",
                                "task_id": task_id},
        "required_controls": {"provider_zero_required": True},
        "reconciliation": {"required_providers": ["vast"], "max_guard_age_seconds": 300},
        "webapp_sync": {"max_attempts": 2},
        "evaluation_run_spec": {"uri": PREFIX + "ns/evaluation_run_spec.json",
                                "digest": "sha256:" + "e" * 64, "size_bytes": 512},
        "allocator": {"max_spend_usd": 11.0, "retry_cap": 0, "hard_ttl_seconds": 27_000}}, "profile_digest")
    write_json(state / "standing" / f"{profile_id}.json", {
        "schema_version": "task_evaluation_standing_launch_authorization.v1",
        "authorization_reference": "scene intent", "authorized_by": "owner-1",
        "issued_at": now.isoformat(), "expires_at": (now + timedelta(hours=1)).isoformat(),
        "max_launches": 1, "max_total_spend_usd": 11.0, "profile_id": profile_id,
        "profile_digest": profile["profile_digest"], "provider_mutation_performed": False})
    _sealed(state / "activations/results" / f"{row['activation_id']}-{'f' * 64}.json", {
        "schema_version": "task_evaluation_launch_activation_result.v1",
        "status": "profile_authority_materialized_no_execution", "activation_id": row["activation_id"],
        "preparation_id": row["preparation_id"], "lane": automation.LANE, "source_commit": release_commit,
        "profile_id": profile_id, "profile_digest": profile["profile_digest"],
        "provider_mutation_performed": False, "paid_execution_requested": False, "blockers": []}, "result_digest")


def _canned_launch_run(root: Path, submission: dict, release_commit: str, now: datetime) -> Path:
    """The dispatcher/allocator outcome of one completed scene-configuration launch (S8, not run here)."""
    state = root / "state"
    run = state / "launch-runs" / submission["launch_id"]
    profile = json.loads(next((state / "profiles").glob("*.json")).read_text())
    write_json(run / "launch_profile.json", profile)
    allocator = state / "allocator" / submission["launch_id"]
    teardown = allocator / "vast_teardown_manifest.json"
    write_json(teardown, {"schema_version": "vast_teardown_manifest.v1", "status": "completed",
        "generated_at": (now + timedelta(minutes=5)).isoformat(), "continuing_spend_from_this_run": False,
        "vast_instance_ids": [1], "teardown_actions_performed": ["destroy"]})
    reference = {"uri": "s3://blueprint/harness/qualified-scene", "digest": "sha256:" + "c" * 64, "size_bytes": 100}
    revision_digest = "sha256:" + "d" * 64
    offering = {"schema_version": "task_evaluation_configured_scene_offering.v1",
        "status": "configured_controls_pending", "configuration_run_id": "harness-configuration",
        "catalog_visibility": "team_only", "presentation": {"task_thumbnail": reference, "selection_receipt": reference},
        "evaluation_preparation_binding": {"configured_scene_revision": reference,
            "configured_scene_revision_digest": revision_digest, "configured_scene_bundle": reference}}
    offering["offering_digest"] = canonical_digest(offering, digest_field="offering_digest")
    finalization = {"schema_version": "task_evaluation_scene_construction_finalization.v1", "status": "completed",
        "queue_state": "completed", "finalization_performed": True, "run_id": "harness-configuration",
        "source_commit": release_commit}
    finalization["result_digest"] = canonical_digest(finalization, digest_field="result_digest")
    write_json(allocator / "publication.json", {"schema_version": "test.publication.v1"})
    write_json(allocator / "revision.json", {"schema_version": "test.revision.v1"})
    result = _sealed(allocator / "result.json", {
        "schema_version": "task_evaluation_scene_configuration_vast_result.v1", "run_id": "harness-configuration",
        "publication_result_path": str(allocator / "publication.json"),
        "configured_scene_revision_path": str(allocator / "revision.json"), "status": "completed",
        "source_commit": release_commit, "configuration_completed": True, "configured_scene_published": True,
        "full_byte_service_account_readback_passed": True, "configured_scene_revision_reference": reference,
        "configured_scene_bundle_reference": reference, "configured_scene_revision_digest": revision_digest,
        "publication_result_digest": reference["digest"], "task_thumbnail_reference": reference,
        "task_thumbnail_selection_receipt_reference": reference, "configured_scene_offering": offering,
        "scene_construction_queue_finalization": finalization}, "result_digest")
    write_json(run / "launch_started.json", {"launch_id": submission["launch_id"], "started_at": now.isoformat()})
    _sealed(run / "launch_receipt.json", {"schema_version": "task_evaluation_launch_receipt.v1",
        "status": "completed", "launch_id": submission["launch_id"], "source_commit": release_commit,
        "run_id": submission["run_id"], "request_digest": "sha256:" + "1" * 64,
        "launch_profile_digest": profile["profile_digest"], "execute_requested": True,
        "provider_mutation_attempted": True,
        "terminal_evidence": {"status": "passed",
            "result": {"path": str(allocator / "result.json"), "exists": True,
                       "digest": _sha_file(allocator / "result.json")},
            "artifacts": {"teardown_manifest_path": {"path": str(teardown), "exists": True,
                                                     "digest": _sha_file(teardown)}},
            "scene_configuration": _scene_configuration_terminal_projection(result)[0]}}, "receipt_digest")
    write_json(state / "gpu_spend_guard/latest.json", _zero_guard(observed_at=now + timedelta(minutes=6)))
    return run


def _webapp_receipt_sync(*, receipt, **_):
    """The WebApp's answer to the receipt POST (task_evaluation_launch_webapp_sync), as the WebApp returns it."""
    offering = receipt["terminal_evidence"]["scene_configuration"].get("configured_scene_offering") or {}
    bound = {"launch_id": receipt["launch_id"], "run_id": receipt["run_id"],
             "request_digest": receipt["request_digest"], "receipt_digest": receipt["receipt_digest"],
             "configured_scene_offering_digest": offering.get("offering_digest"),
             "configured_scene_offering_status": offering.get("status")}
    return {"schema_version": "task_evaluation_launch_webapp_sync_result.v1", "status": "succeeded", **bound,
            "response": {"schema_version": "task_evaluation_launch_web_sync_receipt.v1", "status": "completed",
                         "already_exists": False, **bound}}


# ----------------------------------------------------------------------------------------- normalisation

_EPOCH_KEYS = re.compile(r".*(_epoch|_at|_iso|observed_at|generated_at|issued|expires).*")


def _normal(value, root: str):
    if isinstance(value, dict):
        return {k: ("<time>" if _EPOCH_KEYS.fullmatch(k) and isinstance(v, (int, float, str)) and v is not None
                    and not isinstance(v, bool) else _normal(v, root)) for k, v in sorted(value.items())}
    if isinstance(value, list):
        return [_normal(v, root) for v in value]
    if isinstance(value, str) and root in value:
        return value.replace(root, "<root>")
    return value


def _pick(value: dict | None, *keys):
    return None if value is None else {k: value.get(k) for k in keys}


# ----------------------------------------------------------------------------------------- the run


def run_website_intent(root: Path, *, task_kind: str = "pick_and_place", release_commit: str = RELEASE,
                       clock: float = CLOCK, hard_stop: str | None = None) -> dict:
    """Drive one website intent through the old stage and return the normalised structural record."""
    assert task_kind in ("pick_and_place", "articulated_open_close") and hard_stop in HARD_STOPS
    assert release_commit == SHA, "the release fixtures (production_fixture) are sealed for SHA"
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    record_out: dict = {"task_kind": task_kind, "hard_stop": hard_stop, "ticks": []}
    with pytest.MonkeyPatch.context() as mp, _saved_environment():
        # The old stage reads the wall clock in places that take no `now` (owner authority reopen,
        # submission inventory): a shifted wall clock that keeps running from `clock`.
        started = _real_time()
        mp.setattr(time, "time", lambda: clock + (_real_time() - started))
        # The intent's own times (accepted, expiry) are the pinned clock, not the running one.
        mp.setattr(native, "time", SimpleNamespace(time=lambda: clock))
        kwargs, accepted = native.setup(root, mp, articulated=task_kind == "articulated_open_close")
        intent_root = root / "intents"
        intent_dir = intent_root / accepted["intent_id"]
        intent_path = intent_dir / "intent.json"
        record_out["intent"] = {"strategy": json.loads(intent_path.read_text())["request"]["task"]["strategy"]}
        if hard_stop == "policy_candidate":
            request = json.loads(intent_path.read_text())["request"]
            request["submission_id"] += "-policy"
            request["execution"]["policy_candidates"] = ["openvla-7b"]
            try:
                intake.stage_scene_intent(value=request, queue_root=intent_root, authenticated_client="blueprint-webapp",
                                          trusted_clients={"blueprint-webapp"}, now=time.time())
                record_out["intake"] = {"status": "accepted"}
            except ValueError as exc:
                record_out["intake"] = {"status": "refused", "code": str(exc).split(":", 1)[0]}
            record_out["intents"] = sorted(p.name == accepted["intent_id"] for p in intent_root.glob("scene-*"))
            record_out["source_work"] = sorted(p.name for p in root.iterdir() if p.name in ("bindings", "factories"))
            return _normal(record_out, str(root))
        task = kwargs["task"]
        if hard_stop != "unregistered":
            register_website_preparation(preparation_path=task["preparation"]["path"],
                runtime_inputs_path=task["runtime_inputs"]["path"], task_context_path=task["task_context"]["path"],
                root=root / "bindings", now=clock)
        # The WebApp forwards ECMAScript JSON: whole-number floats become integers.
        intent = json.loads(intent_path.read_text())
        intent["request"] = json.loads(cross_runtime_canonical_json(intent["request"]))
        write_json(intent_path, intent)
        if hard_stop == "revoked":
            intake.revoke_scene_intent(queue_root=intent_root, intent_id=intent["intent_id"],
                                       intent_digest=intent["intent_digest"], owner=intent["request"]["owner"], now=time.time())
        config_path = _config(root, kwargs, release_commit)
        _launch_environment(mp, root)
        store = _ReadbackStore()
        submissions: list[dict] = []
        mp.setattr(publication, "_s3_client", lambda _: store)
        mp.setattr(worker, "_s3_client", lambda _: store)
        for module in (publication, public_scene_host_input_intake):
            mp.setattr(module, "_verified_checkout_head", lambda: release_commit)
        mp.setattr(packets, "current_release_commit", lambda: release_commit)
        mp.setattr(automation, "running_release_commit", lambda: release_commit)
        mp.setattr(capacity, "measure_mount", lambda *a, **kw: {"status": "measured", "available_bytes": 64 * 1024**3})
        real_reserve = disk.reserve_control_plane_disk
        roomy = SimpleNamespace(total=100 * disk.GIB, free=64 * disk.GIB, used=36 * disk.GIB)
        mp.setattr(disk, "reserve_control_plane_disk", lambda *a, **kw: real_reserve(*a, **kw, disk_usage=lambda _p: roomy))
        mp.setattr(worker, "reserve_control_plane_disk", disk.reserve_control_plane_disk)
        real_owned = automation.advance_owned_scene_configuration
        moment = {"now": clock}

        def owned(**kw):
            now = datetime.fromtimestamp(moment["now"], timezone.utc)
            return real_owned(**kw, provider_zero_collector=lambda: _provider_zero(now - timedelta(seconds=20)),
                lineage_publisher_factory=lambda: _publisher("scene-configuration-activation-lineage"),
                release_window_publisher_factory=lambda: _publisher("coordinator-release-windows"),
                submitter=lambda request: submissions.append(dict(request)) or {
                    "status": "submitted", "launch_id": request["launch_id"],
                    "provider_mutation_performed_inside_web_request": False}, now=now)
        mp.setattr(automation, "advance_owned_scene_configuration", owned)
        passes: list[dict] = []
        real_process = progression.process_scene_intents

        def counted(**kw):
            run = real_process(**kw)
            passes.append([_pick(row, "status", "phase", "blockers") for row in run["results"]])
            return run
        mp.setattr(progression, "process_scene_intents", counted)

        def tick(at: float) -> dict:
            moment["now"] = at
            passes.clear()
            service = run_preparation_service(config_path=config_path, now=at)
            progress = json.loads((intent_dir / "progression.json").read_text())
            row = {"passes": list(passes), "worker": [_pick(r, "status", "blockers")
                                                      for r in service["preparation_worker"].get("results", [])],
                   "progression": _pick(progress, "status", "phase", "blockers")}
            record_out["ticks"].append(row)
            return progress

        t0 = time.time()  # the running clock: a little after `clock`
        progress = tick(t0)
        staged = (progress.get("state") or {}).get("scene_configuration_launch")
        if staged and staged.get("status") == "awaiting_scene_configuration_authority":
            attempt = json.loads(next((intent_dir / "attempts").glob("*.json")).read_text())
            _canned_activation(root, staged, attempt, intent["request"]["task"]["task_id"], release_commit,
                               datetime.fromtimestamp(t0 + 60, timezone.utc))
            tick(t0 + 60)
        record_out["launch_submissions"] = [{"profile_digest": s["profile_digest"], "rights": s.get("rights"),
                                             "spend": {"max_spend_usd": s["spend"]["max_spend_usd"]}}
                                            for s in submissions]
        if submissions:
            launched = datetime.fromtimestamp(t0 + 120, timezone.utc)
            _canned_launch_run(root, submissions[0], release_commit, launched)
            mp.setattr(retention, "retain_robot_free_readiness",
                       lambda **_: {"status": "probe_unavailable", "retained": True})
            mp.setattr(webapp_sync, "sync_launch_receipt_to_webapp", _webapp_receipt_sync)
            report = reconcile_launches(queue_root=root / "state/task-evaluation-launches",
                state_root=root / "state/launch-runs", guard_report_path=root / "state/gpu_spend_guard/latest.json",
                profile_dir=root / "state/profiles", now=launched + timedelta(minutes=7), publish_progress=False)
            record_out["reconciler"] = {"status": report["status"], "rows": [
                _pick(r, "status", "provider_zero_confirmed", "blockers")
                for r in (*report["launches"], *report["terminal_provider_zero"], *report["webapp_sync"])]}
            tick(t0 + 600)
        _collect(record_out, root, intent_dir, intent, config_path, t0 + 660)
    return _normal(record_out, str(root))


@contextlib.contextmanager
def fixed_root(case: str):
    """Paths enter the digests (source binding -> attempt ids -> every later id), so a pinned record needs one
    root: `/tmp/blueprint-scene-preparation-harness/<case>`, emptied and held under a lock for the run."""
    import fcntl
    import shutil
    base = Path("/tmp/blueprint-scene-preparation-harness")
    base.mkdir(exist_ok=True)
    with open(base / f"{case}.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        shutil.rmtree(base / case, ignore_errors=True)
        try:
            yield base / case
        finally:
            shutil.rmtree(base / case, ignore_errors=True)


@contextlib.contextmanager
def _saved_environment():
    saved = dict(os.environ)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(saved)


def _collect(out: dict, root: Path, intent_dir: Path, intent: dict, config_path: Path, now: float) -> None:
    out["outcome"] = _pick(_read_optional(intent_dir / "outcome.json"), "kind", "terminal", "reason_code")
    out["attempts"] = [_pick(json.loads(p.read_text()), "attempt_id", "maximum_spend_usd", "provider",
                             "counts_as_paid_attempt", "status")
                       for p in sorted([*intent_dir.glob("preparation-attempts/*.json"), *intent_dir.glob("attempts/*.json")])]
    config = json.loads(config_path.read_text())
    queue = Path(config["preparation_queue_root"])
    envelopes = sorted((queue / "materialized").glob("*.json"))
    if envelopes:
        request = _normal(json.loads(envelopes[0].read_text())["request"], str(root))
        out["preparation_request"] = {"normalised_digest": canonical_digest(request),
            "preparation_id": request["preparation_id"], "scene_id": request["scene"]["identity"]["id"],
            "task_id": request["task"]["identity"]["id"], "hard_cap_usd": request["spend"]["hard_cap_usd"]}
    results = sorted((queue / "results").glob("*.json"))
    if results:
        result = json.loads(results[0].read_text())
        out["preparation_result"] = _pick(result, "status", "run_mode", "construction_recipe_digest",
                                          "configuration_render_inputs_result_digest")
    activation = sorted((root / "state/activations").glob("pending/*.json"))
    if activation:
        # Minus times: the lineage objects and the release window carry observation times inside them.
        request = json.loads(activation[0].read_text())["request"]
        out["activation_request"] = {**_pick(request, "schema_version", "activation_id", "lane",
            "expected_production_commit", "team_namespace", "requested_mutations"),
            "authorization": _pick(request["authorization"], "authorized_by", "profile_revision", "reference"),
            "preparation": _pick(request["preparation"], "preparation_id", "request_digest"),
            "lineage": sorted(request["lineage"]), "release_window": sorted(request["release_window"])}
    out["status"] = _pick(intake.scene_intent_status(queue_root=intent_dir.parent, intent_id=intent["intent_id"],
                                                     now=now), "status", "phase", "blockers")
    team = sorted((root / "state/team-scene-packets").rglob("*.json"))
    out["team_scene_requests"] = len(team)


def _read_optional(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None
