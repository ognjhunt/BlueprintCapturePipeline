"""Three more proofs that a storage pin protects nothing, released only once the owner opts in.

On 2026-09-27 the derived phase kept 142 directories only because a pin named
them: 70 preparation, 58 compilation and 14 activation pins. The original two
proofs release an activation pin only when its run is archived or sealed cold
without a result registry, so a registry run, a prepared activation that never
launched, and a preparation or compilation nothing consumes all waited out the
30-day TTL.
"""

# Covers (for impacted-test selection):
#   src/blueprint_pipeline/control_plane_terminal_cache_pins.py
#   src/blueprint_pipeline/control_plane_storage_pins.py
#   src/blueprint_pipeline/task_evaluation_result_artifact_store.py
#   src/blueprint_pipeline/task_evaluation_launch_activation_worker.py
#   src/blueprint_pipeline/task_evaluation_launch_activation_queue.py
#   src/blueprint_pipeline/control_plane_storage_gc.py
#   deploy/systemd/pipeline-control-plane.env.example
#   deploy/systemd/blueprint-control-plane-storage-gc.service

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from blueprint_pipeline import completed_replay_cache_retention as retention
from blueprint_pipeline import control_plane_replay_cache_gc as replay_gc
from blueprint_pipeline import control_plane_storage_gc as gc_module
from blueprint_pipeline import control_plane_terminal_cache_pins as terminal_pins
from blueprint_pipeline.control_plane_storage_gc import RUN_ACK, run_storage_gc
from blueprint_pipeline.control_plane_storage_pins import load_storage_pins, write_storage_pin
from blueprint_pipeline.control_plane_storage_roots import require_storage_class
from blueprint_pipeline.control_plane_terminal_cache_pins import reconcile_terminal_cache_pins
from blueprint_pipeline.decision_evidence_contracts import canonical_digest, cross_runtime_canonical_digest
from blueprint_pipeline.task_evaluation_result_delivery import REGISTRY_SCHEMA_VERSION

NOW = 30_000_000.0
DAY = 86_400
# A shared mutation window lives at most a week; the proofs wait a day longer.
LAPSE = 604_800 + 86_400
PREPARED_REFERENCES = "/var/lib/blueprint/task-evaluation-inputs/prepared-references"
COMPILED_EPISODES = "/var/lib/blueprint/task-evaluation-inputs/compiled-episodes"


def _classifier(path, *, expected, code):
    """The real storage classes for production paths; the test's own temporary roots are accepted."""

    if not str(path).startswith(("/var/lib/blueprint", "/etc/")):
        return None
    return require_storage_class(str(path), expected=expected, code=code)


def _args(tmp_path: Path, **overrides) -> dict:
    evidence = tmp_path / "evidence"
    evidence.mkdir(parents=True, exist_ok=True)
    return {"pins_root": tmp_path / "storage-pins", "queue_roots": [tmp_path / "queue"],
            "evidence_roots": [evidence], "now": NOW, "reference_checker": lambda _path: False,
            "classifier": _classifier, "hot_window_seconds": 2 * DAY, **overrides}


def _pin(args: dict, kind: str, owner: str, *, age: float, paths=None, depends_on=()) -> dict:
    default = {"preparation": PREPARED_REFERENCES, "compilation": COMPILED_EPISODES}.get(
        kind, "/var/lib/blueprint/task-evaluation-inputs/launch-activations")
    return write_storage_pin(pins_root=args["pins_root"], kind=kind, owner_id=owner,
                             paths=paths or [f"{default}/{owner}"], depends_on=list(depends_on),
                             now=lambda: NOW - age)


def _states(args: dict) -> dict:
    return {(pin["kind"], pin["owner_id"]): pin["status"]
            for pin in load_storage_pins(args["pins_root"], now=lambda: NOW)}


def _kept(result: dict) -> dict:
    return {(row["kind"], row["owner_id"]): row["reason"] for row in result["kept"]}


RUNNING = "b" * 40
SUPERSEDED = "a" * 40


def _preparation_queue(tmp_path: Path, args: dict) -> Path:
    """The preparation queue, configured as a queue root as it is on the host, and the release the tick runs."""

    root = tmp_path / "task-evaluation-launch-preparations"
    root.mkdir(exist_ok=True)
    args["queue_roots"] = [*args["queue_roots"], root]
    args["preparation_queue_root"] = root
    args["running_commit"] = RUNNING
    return root


def _prepared(queue: Path, preparation_id: str, *, commit: str = SUPERSEDED, state: str = "materialized") -> Path:
    """The sealed envelope a preparation leaves in ``state``, bound to the release ``commit``."""

    request_digest = "sha256:" + hashlib.sha256(preparation_id.encode()).hexdigest()
    envelope = {"schema_version": "task_evaluation_launch_preparation_envelope.v1", "request_digest": request_digest,
                "request": {"preparation_id": preparation_id, "expected_production_commit": commit},
                "envelope_digest": ""}
    envelope["envelope_digest"] = canonical_digest(envelope, digest_field="envelope_digest")
    path = queue / state / f"{preparation_id}-{request_digest.removeprefix('sha256:')}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(envelope), encoding="utf-8")
    return path


def _queue_row(args: dict, text: str) -> None:
    pending = Path(args["queue_roots"][0]) / "pending"
    pending.mkdir(parents=True, exist_ok=True)
    (pending / f"row-{len(list(pending.iterdir()))}.json").write_text(text, encoding="utf-8")


def test_unconsumed_stale_preparation_pin_is_released(tmp_path) -> None:
    """A preparation or compilation no live pin depends on and no activation can take, eight days old, is released.

    The activation worker takes only a preparation whose envelope sits in
    ``materialized/`` bound to the running release, and it verifies, never
    re-fetches, the materialized inputs. A preparation bound to a superseded
    release, or one that ended blocked, can never be activated.
    """

    args = _args(tmp_path)
    queue = _preparation_queue(tmp_path, args)
    _prepared(queue, "prep-orphan")
    _prepared(queue, "prep-compiled")
    _prepared(queue, "prep-blocked", commit=RUNNING, state="blocked")
    _pin(args, "preparation", "prep-orphan", age=LAPSE + DAY)
    _pin(args, "preparation", "prep-blocked", age=LAPSE + DAY)
    _pin(args, "preparation", "prep-compiled", age=LAPSE + DAY)
    _pin(args, "compilation", "prep-compiled", age=LAPSE + DAY,
         depends_on=[{"kind": "preparation", "owner_id": "prep-compiled"}])

    planned = reconcile_terminal_cache_pins(**args, extended_proofs_enabled=True)

    assert planned["enabled"] is True and planned["status"] == "dry_run"
    assert [(row["kind"], row["owner_id"], row["proof"]["kind"], row["enabled"]) for row in planned["candidates"]] == [
        ("preparation", "prep-blocked", "unconsumed_stale_pin", True),
        ("preparation", "prep-orphan", "unconsumed_stale_pin", True),
        ("compilation", "prep-compiled", "unconsumed_stale_pin", True),
    ]
    assert [(row["proof"]["preparation_state"], row["proof"]["preparation_commit"]) for row in planned["candidates"]] == [
        ("blocked", RUNNING), ("materialized", SUPERSEDED), ("materialized", SUPERSEDED)]
    # The compilation still consumes its preparation, which goes with it.
    assert _kept(planned) == {("preparation", "prep-compiled"): "depended_on"}
    assert planned["released"] == [] and planned["released_count"] == 0
    assert set(_states(args).values()) == {"live"}

    applied = reconcile_terminal_cache_pins(**args, apply=True, extended_proofs_enabled=True)

    assert applied["status"] == "applied"
    assert applied["released_count_by_kind"] == {"compilation": 1, "preparation": 3}
    assert applied["released_count"] == 4
    assert applied["candidate_count_by_proof"] == {"unconsumed_stale_pin": 3}
    assert set(_states(args).values()) == {"released"}
    assert applied["cache_or_evidence_bytes_removed"] is False


@pytest.mark.parametrize("reason", ["depended", "young", "recent", "queued", "parked", "process", "path_class"])
def test_depended_or_young_or_queued_preparation_pin_is_kept(tmp_path, reason) -> None:
    args = _args(tmp_path)
    preparations = _preparation_queue(tmp_path, args)
    _prepared(preparations, "prep-x")
    age = {"young": LAPSE - DAY, "recent": 3600}.get(reason, LAPSE + DAY)
    paths = ["/var/lib/blueprint/pipeline-control-plane/task-evaluation-launches/prep-x"] if reason == "path_class" else None
    _pin(args, "preparation", "prep-x", age=age, paths=paths)
    if reason == "depended":
        _pin(args, "activation", "act-x", age=DAY, depends_on=[{"kind": "preparation", "owner_id": "prep-x"}])
    if reason == "queued":
        _queue_row(args, json.dumps({"preparation_id": "prep-x"}))
    if reason == "parked":
        # A row parked in a state that will still run, here one naming prep-x, is
        # still in flight, though it sits in neither pending nor processing.
        (preparations / "awaiting_source_preparation").mkdir(parents=True)
        (preparations / "awaiting_source_preparation" / f"prep-child-{'0' * 64}.json").write_text(
            json.dumps({"preparation_id": "prep-child", "parent_preparation_id": "prep-x"}), encoding="utf-8")
    if reason == "process":
        args["reference_checker"] = lambda path: path.name == "prep-x"

    result = reconcile_terminal_cache_pins(**args, apply=True, extended_proofs_enabled=True)

    assert _kept(result)[("preparation", "prep-x")] == {
        "depended": "depended_on", "young": "pin_not_stale", "recent": "pin_young", "queued": "active_reference",
        "parked": "active_reference", "process": "active_reference", "path_class": "path_class_invalid"}[reason]
    assert not any(row["owner_id"] == "prep-x" for row in result["candidates"])
    assert result["released"] == [] and _states(args)[("preparation", "prep-x")] == "live"


def test_a_consumer_pinned_before_the_mutation_edge_keeps_the_stale_pin(tmp_path) -> None:
    """The stale-pin proof rests on the ledger, so it is derived again, under the pin lock, before a release.

    Here an activation pins the preparation after the tick planned to release it.
    """

    args = _args(tmp_path)
    _prepared(_preparation_queue(tmp_path, args), "prep-x")
    _pin(args, "preparation", "prep-x", age=LAPSE + DAY)
    arrived: list[dict] = []

    def consumer_arrives(_path) -> bool:
        if not arrived:
            arrived.append(_pin(args, "activation", "act-late", age=0,
                                depends_on=[{"kind": "preparation", "owner_id": "prep-x"}]))
        return False

    args["reference_checker"] = consumer_arrives

    result = reconcile_terminal_cache_pins(**args, apply=True, extended_proofs_enabled=True)

    assert [row["owner_id"] for row in result["candidates"]] == ["prep-x"] and arrived
    assert _kept(result) == {("preparation", "prep-x"): "reference_changed"}
    assert result["released"] == [] and _states(args)[("preparation", "prep-x")] == "live"


@pytest.mark.parametrize("reason", ["current", "missing", "unknown_commit", "unconfigured", "tampered", "ambiguous"])
def test_a_preparation_the_running_release_can_still_activate_keeps_its_pin(tmp_path, reason) -> None:
    """Review of 2026-09-27: a materialized preparation waits for its activation intent with no age limit.

    Its pin may be old (it is written when the preparation first parks on its
    source), and the activation worker verifies the materialized inputs rather
    than re-fetching them, so a stale pin alone proves nothing. Only positive
    evidence that no activation can take the preparation releases it.
    """

    args = _args(tmp_path)
    preparations = _preparation_queue(tmp_path, args)
    if reason != "missing":
        envelope = _prepared(preparations, "prep-x", commit=RUNNING if reason == "current" else SUPERSEDED)
    if reason == "unknown_commit":
        args["running_commit"] = ""
    if reason == "unconfigured":
        args["preparation_queue_root"] = None
    if reason == "tampered":
        value = json.loads(envelope.read_text(encoding="utf-8"))
        value["request"]["expected_production_commit"] = "c" * 40
        envelope.write_text(json.dumps(value), encoding="utf-8")
    if reason == "ambiguous":
        _prepared(preparations, "prep-x", state="blocked")
    _pin(args, "preparation", "prep-x", age=LAPSE + DAY)
    _pin(args, "compilation", "prep-x", age=LAPSE + DAY, depends_on=[{"kind": "preparation", "owner_id": "prep-x"}])

    result = reconcile_terminal_cache_pins(**args, apply=True, extended_proofs_enabled=True)

    expected = {"current": "preparation_release_current", "missing": "preparation_envelope_missing",
                "unknown_commit": "running_commit_unknown", "unconfigured": "preparation_queue_unconfigured",
                "tampered": "preparation_envelope_invalid", "ambiguous": "preparation_envelope_ambiguous"}[reason]
    assert _kept(result) == {("preparation", "prep-x"): "depended_on", ("compilation", "prep-x"): expected}
    assert result["candidates"] == [] and set(_states(args).values()) == {"live"}


def _archived_case(tmp_path: Path, args: dict) -> None:
    """The original archived-run proof: an activation whose run sits behind a verified pointer."""

    _pin(args, "preparation", "prep-a", age=DAY)
    _pin(args, "compilation", "prep-a", age=DAY, depends_on=[{"kind": "preparation", "owner_id": "prep-a"}])
    _pin(args, "activation", "archived", age=DAY, depends_on=[
        {"kind": "compilation", "owner_id": "prep-a"}, {"kind": "preparation", "owner_id": "prep-a"}])
    pointer = {"schema_version": "control_plane_evidence_offload_pointer.v1", "status": "offloaded",
               "directory": "archived", "evidence_deleted": False, "terminal_receipt": "dispatch_receipt.json",
               "size_bytes": 1024, "digest": "sha256:" + "a" * 64,
               "uri": "s3://blueprint-task-evaluation-artifacts-prod/retained/evidence.tar"}
    pointer["pointer_digest"] = canonical_digest(pointer, digest_field="pointer_digest")
    (tmp_path / "evidence" / "archived.offloaded.v1.json").write_text(json.dumps(pointer), encoding="utf-8")


def _sealed_cold_case(tmp_path: Path, args: dict) -> Path:
    """The original sealed-cold-run proof: a blocked launch with a terminal receipt and no registry."""

    run = tmp_path / "evidence" / "cold"
    (run / "allocator").mkdir(parents=True)
    (run / "launch_receipt.json").write_text(json.dumps({"status": "blocked"}), encoding="utf-8")
    for path in (run / "launch_receipt.json", run / "allocator", run):
        os.utime(path, (NOW - 5 * DAY, NOW - 5 * DAY))
    _pin(args, "activation", "cold", age=7 * DAY, paths=[run])
    return run


@pytest.mark.parametrize("enabled", [False, True])
def test_existing_proofs_unchanged_without_the_flag(tmp_path, enabled) -> None:
    """The archived-run and sealed-cold-run proofs release exactly what they did, with or without the opt-in."""

    args = _args(tmp_path)
    _archived_case(tmp_path, args)
    run = _sealed_cold_case(tmp_path, args)
    # A malformed pin an extended proof cannot read costs only itself.
    corrupt = args["pins_root"] / "preparation" / "corrupt.json"
    corrupt.write_text(json.dumps({
        "schema_version": "control_plane_storage_pin.v1", "kind": "preparation", "owner_id": "corrupt",
        "depends_on": [], "created_at_epoch": NOW - LAPSE - DAY, "expires_at_epoch": NOW + DAY,
        "released_at_epoch": None}), encoding="utf-8")

    planned = reconcile_terminal_cache_pins(**args, extended_proofs_enabled=enabled)
    applied = reconcile_terminal_cache_pins(**args, apply=True, extended_proofs_enabled=enabled)

    assert planned["enabled"] is enabled
    for report in (planned, applied):
        assert [(row["owner_id"], row["proof"]["kind"], row["enabled"]) for row in report["candidates"]] == [
            ("archived", "archived_run", True), ("cold", "sealed_cold_run", True)]
    assert [row for row in planned["kept"] if row["owner_id"] == "corrupt"] == [
        {"kind": "preparation", "owner_id": "corrupt", "reason": "proof_error", "error_type": "KeyError"}]
    assert {row["owner_id"]: sorted((pin["kind"], pin["owner_id"]) for pin in row["released"])
            for row in applied["released"]} == {
        "archived": [("activation", "archived"), ("compilation", "prep-a"), ("preparation", "prep-a")],
        "cold": [("activation", "cold")],
    }
    assert _states(args) == {("preparation", "prep-a"): "released", ("preparation", "corrupt"): "live",
                             ("compilation", "prep-a"): "released", ("activation", "archived"): "released",
                             ("activation", "cold"): "released"}
    assert run.is_dir()
    # A pin the original proofs close still refuses a path outside its classes, as it always did.
    other = _args(tmp_path / "unclassified")
    _sealed_cold_case(tmp_path / "unclassified", other)
    write_storage_pin(pins_root=other["pins_root"], kind="activation", owner_id="stray",
                      paths=["/etc/blueprint/stray"], now=lambda: NOW - 7 * DAY)
    stray = tmp_path / "unclassified" / "evidence" / "stray"
    (stray / "allocator").mkdir(parents=True)
    (stray / "launch_receipt.json").write_text("{}", encoding="utf-8")
    for path in (stray / "launch_receipt.json", stray / "allocator", stray):
        os.utime(path, (NOW - 5 * DAY, NOW - 5 * DAY))
    with pytest.raises(ValueError, match="terminal_cache_pin_path_class_invalid"):
        reconcile_terminal_cache_pins(**other, extended_proofs_enabled=enabled)


def _registry_run(evidence: Path, name: str, *, status: str = "completed_unqualified", idle: float = 5 * DAY,
                  receipt: bool = True) -> Path:
    """A run whose result registry the artifact store accepts as sealed, the registry idle ``idle`` seconds.

    With ``receipt`` it also carries its terminal launch receipt.
    """

    run = evidence / name
    closeout = run / "closeout"
    closeout.mkdir(parents=True)
    if receipt:
        (run / "launch_receipt.json").write_text(json.dumps({"status": status}), encoding="utf-8")
    records, reproducibility = [], {}
    for part in ("billing", "teardown", "provider_zero"):
        data = json.dumps({"part": part, "status": "closed"}).encode()
        (closeout / f"{part}.json").write_bytes(data)
        digest = "sha256:" + hashlib.sha256(data).hexdigest()
        records.append({"artifact_id": part, "role": f"closure_{part}", "relative_path": f"{part}.json",
                        "evidence_root": str(closeout), "sha256": digest, "size_bytes": len(data)})
        reproducibility[f"{part}_receipt"] = {"artifact_id": part, "digest": digest, "size_bytes": len(data)}
    reproducibility["provider_zero_receipt"]["provider_zero_verified"] = True
    delivery = {"schema_version": "task_evaluation_result_delivery.v2", "run_id": name, "result_status": status,
                "reproducibility": reproducibility, "delivery_digest": ""}
    delivery["delivery_digest"] = cross_runtime_canonical_digest(delivery, digest_field="delivery_digest")
    registry = {"schema_version": REGISTRY_SCHEMA_VERSION, "run_id": name, "artifacts": records,
                "delivery_digest": delivery["delivery_digest"], "registry_digest": ""}
    registry["registry_digest"] = canonical_digest(registry, digest_field="registry_digest")
    result_delivery = run / "artifacts" / "result_delivery"
    result_delivery.mkdir(parents=True)
    (result_delivery / "delivery.json").write_text(json.dumps(delivery), encoding="utf-8")
    (result_delivery / "artifact_registry.json").write_text(json.dumps(registry), encoding="utf-8")
    os.utime(result_delivery / "artifact_registry.json", (NOW - idle, NOW - idle))
    return run


def test_sealed_registry_run_releases_its_activation_pin_when_enabled(tmp_path) -> None:
    """A registry run its closeout sealed, idle past the hot window, no longer needs its activation pin.

    Whole-run offload never archives a registry run, so neither original proof
    could ever release one: the pin waited out its TTL.
    """

    args = _args(tmp_path)
    evidence = tmp_path / "evidence"
    run = _registry_run(evidence, "act-registry")
    auto = "website-example-20260920t204229z-activation-auto"
    _registry_run(evidence, auto + "-launch", status="blocked")
    _pin(args, "activation", "act-registry", age=7 * DAY)
    _pin(args, "activation", auto, age=7 * DAY)
    registry = run / "artifacts" / "result_delivery" / "artifact_registry.json"
    before = registry.read_bytes()

    listed = reconcile_terminal_cache_pins(**args, apply=True)

    assert [(row["owner_id"], row["proof"]["kind"], row["proof"]["delivery_status"], row["enabled"])
            for row in listed["candidates"]] == [
        ("act-registry", "sealed_registry_run", "completed_unqualified", False),
        (auto, "sealed_registry_run", "blocked", False)]
    assert listed["released"] == [] and set(_states(args).values()) == {"live"}

    applied = reconcile_terminal_cache_pins(**args, apply=True, extended_proofs_enabled=True)

    assert applied["released_count_by_kind"] == {"activation": 2}
    assert set(_states(args).values()) == {"released"}
    proof = applied["candidates"][0]["proof"]
    assert (proof["run_directory"], proof["registry_mtime_epoch"]) == ("act-registry", NOW - 5 * DAY)
    assert proof["registry_digest"] == json.loads(before)["registry_digest"]
    # A proof names the run, never a host path, and releasing the pin removes nothing.
    assert not any("/" in str(value) for value in proof.values())
    assert registry.read_bytes() == before and (run / "closeout" / "billing.json").is_file()


@pytest.mark.parametrize("reason", [
    "hot", "tampered", "not_terminal", "closeout_changed", "pointer", "linked", "no_receipt"])
def test_hot_or_invalid_registry_keeps_its_pin(tmp_path, reason) -> None:
    """A registry run keeps its pin until it is sealed, closed out and cold, with no pointer.

    Without its terminal receipt the canary dispatcher can still recover a
    stranded delivery after a deploy, and that recovery re-reads the
    activation's launch set, which the pin keeps.
    """

    args = _args(tmp_path)
    evidence = tmp_path / "evidence"
    if reason == "linked":
        (evidence / "act-x").symlink_to(_registry_run(tmp_path / "elsewhere", "act-x"))
    else:
        run = _registry_run(evidence, "act-x", idle=DAY if reason == "hot" else 5 * DAY,
                            status="running" if reason == "not_terminal" else "completed_unqualified",
                            receipt=reason != "no_receipt")
        registry = run / "artifacts" / "result_delivery" / "artifact_registry.json"
    if reason == "tampered":
        value = json.loads(registry.read_text(encoding="utf-8"))
        value["artifacts"] = value["artifacts"][:2]
        registry.write_text(json.dumps(value), encoding="utf-8")
        os.utime(registry, (NOW - 5 * DAY, NOW - 5 * DAY))
    if reason == "closeout_changed":
        (run / "closeout" / "billing.json").write_bytes(b'{"part": "billing", "status": "reopened"}')
    if reason == "pointer":
        (evidence / "act-x.offloaded.v1.json").write_text("{}", encoding="utf-8")
    _pin(args, "activation", "act-x", age=7 * DAY)

    result = reconcile_terminal_cache_pins(**args, apply=True, extended_proofs_enabled=True)

    assert _kept(result)[("activation", "act-x")] == {
        "hot": "registry_hot", "tampered": "registry_unsealed", "not_terminal": "registry_unsealed",
        "closeout_changed": "registry_unsealed", "pointer": "run_pointer_present", "linked": "run_path_unsafe",
        "no_receipt": "run_not_sealed"}[reason]
    assert result["candidates"] == [] and _states(args)[("activation", "act-x")] == "live"


AUTO = "website-example-20260920t204229z-activation-auto"


def _activation_queue(tmp_path: Path, args: dict) -> Path:
    """The activation queue, configured as a queue root as it is on the host."""

    root = tmp_path / "task-evaluation-launch-activations"
    (root / "results").mkdir(parents=True, exist_ok=True)
    args["queue_roots"] = [*args["queue_roots"], root]
    args["activation_queue_root"] = root
    return root


def _activation_result(queue: Path, owner: str, *, status: str = "profile_authority_materialized_no_execution",
                       age: float = LAPSE + DAY, authorization_expires: float | None = NOW - LAPSE) -> Path:
    """The result the activation worker seals beside its queue, named for the activation's queue envelope.

    The envelope it prepared goes to ``prepared/`` under the same name, its
    request carrying the standing authorization's expiry
    (``authorization_expires``; None leaves the field out).
    """

    from datetime import datetime, timezone

    from blueprint_pipeline.task_evaluation_launch_activation_queue import _queue_filename

    value = {"schema_version": "task_evaluation_launch_activation_result.v1", "status": status,
             "activation_id": owner, "blockers": [], "provider_mutation_performed": False, "result_digest": ""}
    value["result_digest"] = canonical_digest(value, digest_field="result_digest")
    request_digest = "sha256:" + hashlib.sha256(owner.encode()).hexdigest()
    path = queue / "results" / _queue_filename(activation_id=owner, request_digest=request_digest)
    path.write_text(json.dumps(value), encoding="utf-8")
    os.utime(path, (NOW - age, NOW - age))
    authorization = {} if authorization_expires is None else {
        "standing_authorization_expires_at": datetime.fromtimestamp(authorization_expires, tz=timezone.utc).isoformat()}
    envelope = {"schema_version": "task_evaluation_launch_activation_envelope.v1", "request_digest": request_digest,
                "request": {"activation_id": owner, "authorization": authorization}, "envelope_digest": ""}
    envelope["envelope_digest"] = canonical_digest(envelope, digest_field="envelope_digest")
    (queue / "prepared").mkdir(exist_ok=True)
    (queue / "prepared" / path.name).write_text(json.dumps(envelope), encoding="utf-8")
    return path


def test_expired_unlaunched_activation_releases_its_pin(tmp_path) -> None:
    """A prepared activation that never launched, a day past the longest mutation window, needs nothing.

    Such an activation has no run directory, so neither original proof ever
    looked at it; launch re-validates the window, so it can no longer start.
    """

    args = _args(tmp_path)
    queue = _activation_queue(tmp_path, args)
    _activation_result(queue, "act-profile")
    _activation_result(queue, "act-policy", status="policy_campaign_queue_materialized_no_execution")
    _pin(args, "preparation", "prep-u", age=LAPSE + 2 * DAY)
    _pin(args, "compilation", "prep-u", age=LAPSE + 2 * DAY, depends_on=[{"kind": "preparation", "owner_id": "prep-u"}])
    for owner in ("act-policy", "act-profile"):
        _pin(args, "activation", owner, age=LAPSE + DAY, depends_on=[
            {"kind": "compilation", "owner_id": "prep-u"}, {"kind": "preparation", "owner_id": "prep-u"}])

    listed = reconcile_terminal_cache_pins(**args, apply=True)

    assert [(row["owner_id"], row["proof"]["kind"], row["proof"]["result_status"], row["enabled"])
            for row in listed["candidates"]] == [
        ("act-policy", "activation_expired_unlaunched", "policy_campaign_queue_materialized_no_execution", False),
        ("act-profile", "activation_expired_unlaunched", "profile_authority_materialized_no_execution", False)]
    assert set(_states(args).values()) == {"live"}

    applied = reconcile_terminal_cache_pins(**args, apply=True, extended_proofs_enabled=True)

    # The shared preparation and compilation go with the last activation that consumed them.
    assert applied["released_count_by_kind"] == {"activation": 2, "compilation": 1, "preparation": 1}
    assert set(_states(args).values()) == {"released"}
    proof = applied["candidates"][1]["proof"]
    assert proof["result_name"] == f"act-profile-{hashlib.sha256(b'act-profile').hexdigest()}.json"
    assert (proof["result_mtime_epoch"], proof["authorization_expires_epoch"]) == (NOW - LAPSE - DAY, NOW - LAPSE)
    assert not any("/" in str(value) for value in proof.values())


@pytest.mark.parametrize("reason", ["young", "launched", "archived", "referenced", "process"])
def test_young_or_launched_or_referenced_unlaunched_activation_keeps_its_pin(tmp_path, reason) -> None:
    second = tmp_path / "policy-canaries"
    second.mkdir()
    args = _args(tmp_path)
    args["evidence_roots"] = [*args["evidence_roots"], second]
    queue = _activation_queue(tmp_path, args)
    _activation_result(queue, AUTO, age=LAPSE - DAY if reason == "young" else LAPSE + DAY)
    _pin(args, "activation", AUTO, age=LAPSE + DAY)
    if reason == "launched":
        # Any evidence name in any evidence root is a launch: here the website launch run.
        (second / f"{AUTO}-launch" / "allocator").mkdir(parents=True)
    if reason == "archived":
        (second / f"{AUTO}-launch.offloaded.v1.json").write_text("{}", encoding="utf-8")
    if reason == "referenced":
        dispatches = tmp_path / "task-evaluation-policy-canary-dispatches"
        (dispatches / "processing").mkdir(parents=True)
        (dispatches / "processing" / "row.json").write_text(json.dumps({"activation_id": AUTO}), encoding="utf-8")
        args["queue_roots"] = [*args["queue_roots"], dispatches]
    if reason == "process":
        args["reference_checker"] = lambda path: path.name == AUTO

    result = reconcile_terminal_cache_pins(**args, apply=True, extended_proofs_enabled=True)

    assert _kept(result)[("activation", AUTO)] == {
        "young": "activation_result_not_stale", "launched": "run_not_sealed", "archived": "run_pointer_present",
        "referenced": "active_reference", "process": "active_reference"}[reason]
    assert result["candidates"] == [] and _states(args)[("activation", AUTO)] == "live"


@pytest.mark.parametrize("reason", [
    "blocked", "tampered", "missing", "other_owner", "duplicate", "linked", "unconfigured", "root_linked",
    "root_missing", "results_missing", "results_linked"])
def test_unprepared_or_unreadable_activation_result_keeps_its_pin(tmp_path, reason) -> None:
    """Only one exact, sealed, prepared result is proof; anything less, or an unreadable root, keeps the pin."""

    args = _args(tmp_path)
    queue = _activation_queue(tmp_path, args)
    if reason not in ("missing", "other_owner", "linked"):
        result_path = _activation_result(queue, "act-u", status="blocked" if reason == "blocked" else
                                         "profile_authority_materialized_no_execution")
    if reason == "tampered":
        value = json.loads(result_path.read_text(encoding="utf-8"))
        value["status"] = "policy_campaign_queue_materialized_no_execution"
        result_path.write_text(json.dumps(value), encoding="utf-8")
        os.utime(result_path, (NOW - LAPSE - DAY, NOW - LAPSE - DAY))
    if reason == "other_owner":
        # A longer activation id that merely starts with this one is someone else's result.
        _activation_result(queue, "act-u-retry")
    if reason == "duplicate":
        (queue / "results" / f"act-u-{'0' * 64}.json").write_bytes(result_path.read_bytes())
    if reason == "linked":
        elsewhere = _activation_result(_activation_queue(tmp_path / "elsewhere", {"queue_roots": []}), "act-u")
        (queue / "results" / elsewhere.name).symlink_to(elsewhere)
    if reason == "unconfigured":
        args["activation_queue_root"] = None
    if reason == "root_linked":
        (tmp_path / "linked-evidence").symlink_to(args["evidence_roots"][0])
        args["evidence_roots"] = [tmp_path / "linked-evidence"]
    if reason == "root_missing":
        # An unmounted or renamed root cannot show that nothing launched into it.
        args["evidence_roots"] = [*args["evidence_roots"], tmp_path / "unmounted-evidence"]
    if reason in ("results_missing", "results_linked"):
        (queue / "results").rename(tmp_path / "moved-results")
        if reason == "results_linked":
            (queue / "results").symlink_to(tmp_path / "moved-results")
    _pin(args, "activation", "act-u", age=LAPSE + DAY)

    result = reconcile_terminal_cache_pins(**args, apply=True, extended_proofs_enabled=True)

    assert _kept(result)[("activation", "act-u")] == {
        "blocked": "activation_result_not_prepared", "tampered": "activation_result_invalid",
        "missing": "activation_result_missing", "other_owner": "activation_result_missing",
        "duplicate": "activation_result_ambiguous", "linked": "activation_result_invalid",
        "unconfigured": "activation_queue_unconfigured", "root_linked": "evidence_root_unavailable",
        "root_missing": "evidence_root_unavailable", "results_missing": "activation_queue_unavailable",
        "results_linked": "activation_queue_unavailable"}[reason]
    assert result["candidates"] == [] and _states(args)[("activation", "act-u")] == "live"


@pytest.mark.parametrize("reason", [
    "in_force", "lapsed_hours_ago", "no_expiry", "envelope_missing", "envelope_tampered", "other_activation"])
def test_a_standing_authorization_still_in_force_keeps_an_unlaunched_activation_pinned(tmp_path, reason) -> None:
    """Launch admission checks the standing authorization the activation published, not the mutation window.

    Its expiry comes from the activation request, with no maximum, so an
    activation can launch long after its window lapsed; the proof waits a day
    past that expiry too, read from the sealed envelope the worker prepared.
    """

    args = _args(tmp_path)
    queue = _activation_queue(tmp_path, args)
    expires = {"in_force": NOW + DAY, "lapsed_hours_ago": NOW - 3600, "no_expiry": None}.get(reason, NOW - LAPSE)
    result_path = _activation_result(queue, "act-u", authorization_expires=expires)
    envelope = queue / "prepared" / result_path.name
    if reason == "envelope_missing":
        envelope.unlink()
    if reason in ("envelope_tampered", "other_activation"):
        value = json.loads(envelope.read_text(encoding="utf-8"))
        value["request"]["activation_id"] = "act-other"
        if reason == "other_activation":
            value["envelope_digest"] = canonical_digest(value, digest_field="envelope_digest")
        envelope.write_text(json.dumps(value), encoding="utf-8")
    _pin(args, "activation", "act-u", age=LAPSE + DAY)

    result = reconcile_terminal_cache_pins(**args, apply=True, extended_proofs_enabled=True)

    assert _kept(result)[("activation", "act-u")] == {
        "in_force": "activation_authorization_not_lapsed", "lapsed_hours_ago": "activation_authorization_not_lapsed",
        "no_expiry": "activation_envelope_invalid", "envelope_missing": "activation_envelope_missing",
        "envelope_tampered": "activation_envelope_invalid", "other_activation": "activation_envelope_invalid"}[reason]
    assert result["candidates"] == [] and _states(args)[("activation", "act-u")] == "live"


@pytest.mark.parametrize(("suffix", "run", "reason"), [
    ("-controls", "registry_hot", "registry_hot"),
    ("-destination", "registry_hot", "registry_hot"),
    ("-construction", "registry_hot", "registry_hot"),
    ("-controls", "cold_without_registry", "run_without_registry"),
    ("-controls", "pointer", "run_pointer_present"),
])
def test_launched_configured_controls_activation_is_not_unlaunched(tmp_path, suffix, run, reason) -> None:
    """2026-09-27 review: configured-controls activations launch as ``<id>-launch`` too.

    With a prepared result nine days old and a sealed registry run under that
    name touched a day ago, the pin was released as never launched, while the
    -activation-auto equivalent was kept.
    """

    owner = "run-x" + suffix
    args = _args(tmp_path)
    queue = _activation_queue(tmp_path, args)
    _activation_result(queue, owner)
    _pin(args, "activation", owner, age=LAPSE + DAY)
    evidence = tmp_path / "evidence"
    if run == "registry_hot":
        _registry_run(evidence, owner + "-launch", idle=DAY)
    if run == "cold_without_registry":
        cold = evidence / (owner + "-launch")
        (cold / "allocator").mkdir(parents=True)
        (cold / "launch_receipt.json").write_text(json.dumps({"status": "completed"}), encoding="utf-8")
        for path in (cold / "launch_receipt.json", cold / "allocator", cold):
            os.utime(path, (NOW - 5 * DAY, NOW - 5 * DAY))
    if run == "pointer":
        (evidence / f"{owner}-launch.offloaded.v1.json").write_text("{}", encoding="utf-8")

    result = reconcile_terminal_cache_pins(**args, apply=True, extended_proofs_enabled=True)

    assert _kept(result)[("activation", owner)] == reason
    assert result["candidates"] == [] and _states(args)[("activation", owner)] == "live"


def test_a_run_under_the_shortened_launch_id_keeps_a_long_activation_pinned(tmp_path) -> None:
    """A long activation id launches under the bounded id its launch path derives, not ``<id>-launch``.

    Its queue result is found under the queue's own long-id name, so without a
    run the activation is proven unlaunched; a run under the bounded launch id
    keeps it.
    """

    from blueprint_pipeline.task_evaluation_configured_controls_progression import _bounded_launch_id

    owner = "scene-" + "x" * 176 + "-controls"
    launch_id = _bounded_launch_id(owner)
    assert launch_id != owner + "-launch" and launch_id.endswith("-launch")
    args = _args(tmp_path)
    queue = _activation_queue(tmp_path, args)
    result_path = _activation_result(queue, owner)
    assert result_path.name.startswith("activation-")
    _pin(args, "activation", owner, age=LAPSE + DAY)

    unlaunched = reconcile_terminal_cache_pins(**args)
    (tmp_path / "evidence" / launch_id / "allocator").mkdir(parents=True)
    launched = reconcile_terminal_cache_pins(**args, apply=True, extended_proofs_enabled=True)

    assert [(row["owner_id"], row["proof"]["kind"]) for row in unlaunched["candidates"]] == [
        (owner, "activation_expired_unlaunched")]
    assert _kept(launched)[("activation", owner)] == "run_not_sealed"
    assert launched["candidates"] == [] and _states(args)[("activation", owner)] == "live"


def test_activation_auto_launch_names_are_unchanged(tmp_path) -> None:
    """A website activation's run is still found under ``<id>-launch``, and the original proofs read what they did."""

    assert terminal_pins._evidence_names(AUTO) == (AUTO, AUTO + "-launch")
    assert terminal_pins._evidence_names("run-x-controls") == ("run-x-controls",)
    assert terminal_pins._launch_evidence_names(AUTO) == (AUTO, AUTO + "-launch")
    args = _args(tmp_path)
    queue = _activation_queue(tmp_path, args)
    _activation_result(queue, AUTO)
    _registry_run(tmp_path / "evidence", AUTO + "-launch", idle=DAY)
    _pin(args, "activation", AUTO, age=LAPSE + DAY)

    result = reconcile_terminal_cache_pins(**args, apply=True, extended_proofs_enabled=True)

    assert _kept(result)[("activation", AUTO)] == "registry_hot"
    assert _states(args)[("activation", AUTO)] == "live"


def test_sealed_registry_run_finds_a_configured_controls_launch_run(tmp_path) -> None:
    args = _args(tmp_path)
    _registry_run(tmp_path / "evidence", "run-y-controls-launch")
    _pin(args, "activation", "run-y-controls", age=7 * DAY)

    result = reconcile_terminal_cache_pins(**args, apply=True, extended_proofs_enabled=True)

    assert [(row["owner_id"], row["proof"]["kind"], row["proof"]["run_directory"]) for row in result["candidates"]] == [
        ("run-y-controls", "sealed_registry_run", "run-y-controls-launch")]
    assert _states(args)[("activation", "run-y-controls")] == "released"


def test_the_pin_report_caps_its_rows_and_keeps_its_counts(tmp_path, monkeypatch) -> None:
    """About 142 live pins in production: each row list stops at 200, and every count still covers every pin."""

    assert terminal_pins._MAX_ROWS == 200
    monkeypatch.setattr(terminal_pins, "_MAX_ROWS", 2)
    args = _args(tmp_path)
    preparations = _preparation_queue(tmp_path, args)
    for index in range(3):
        _prepared(preparations, f"prep-stale-{index}")
        _pin(args, "preparation", f"prep-stale-{index}", age=LAPSE + DAY)
    for index in range(4):
        _pin(args, "preparation", f"prep-young-{index}", age=3600)

    result = reconcile_terminal_cache_pins(**args, apply=True, extended_proofs_enabled=True)

    assert (len(result["candidates"]), result["omitted_candidates_count"], result["candidate_count"]) == (2, 1, 3)
    assert (len(result["kept"]), result["omitted_kept_count"]) == (2, 2)
    assert result["retained_counts"] == {"pin_young": 4}
    assert result["candidate_count_by_proof"] == {"unconsumed_stale_pin": 3}
    # The cap bounds the report, never the work: every stale pin was released.
    assert result["released_count_by_kind"] == {"preparation": 3}


def test_every_run_an_activation_owns_must_be_sealed(tmp_path) -> None:
    """A website activation owns ``<id>`` and ``<id>-launch``: one sealed registry run does not cover an unsealed other."""

    args = _args(tmp_path)
    _registry_run(tmp_path / "evidence", AUTO)
    (tmp_path / "evidence" / (AUTO + "-launch") / "allocator").mkdir(parents=True)
    _pin(args, "activation", AUTO, age=7 * DAY)

    result = reconcile_terminal_cache_pins(**args, apply=True, extended_proofs_enabled=True)

    assert _kept(result) == {("activation", AUTO): "run_not_sealed"}
    assert result["candidates"] == [] and _states(args)[("activation", AUTO)] == "live"


def test_a_launch_between_the_plan_and_the_mutation_edge_keeps_the_activation_pinned(tmp_path) -> None:
    """Both activation proofs are derived again at the mutation edge: a run that appears after the plan keeps the pin."""

    args = _args(tmp_path)
    queue = _activation_queue(tmp_path, args)
    _activation_result(queue, "run-z-controls")
    _pin(args, "activation", "run-z-controls", age=LAPSE + DAY)
    launched: list[Path] = []

    def launch_starts(_path) -> bool:
        if not launched:
            launched.append(tmp_path / "evidence" / "run-z-controls-launch")
            (launched[0] / "allocator").mkdir(parents=True)
        return False

    args["reference_checker"] = launch_starts

    result = reconcile_terminal_cache_pins(**args, apply=True, extended_proofs_enabled=True)

    assert [row["proof"]["kind"] for row in result["candidates"]] == ["activation_expired_unlaunched"] and launched
    assert _kept(result) == {("activation", "run-z-controls"): "reference_changed"}
    assert result["released"] == [] and _states(args)[("activation", "run-z-controls")] == "live"


def test_a_queue_row_naming_a_dependency_keeps_the_extended_candidate(tmp_path) -> None:
    """The closure is checked whole: a row naming the preparation an activation depends on keeps the activation."""

    args = _args(tmp_path)
    queue = _activation_queue(tmp_path, args)
    _activation_result(queue, "act-u")
    _pin(args, "preparation", "prep-shared", age=DAY)
    _pin(args, "activation", "act-u", age=LAPSE + DAY, depends_on=[{"kind": "preparation", "owner_id": "prep-shared"}])
    _queue_row(args, json.dumps({"preparation_id": "prep-shared"}))

    result = reconcile_terminal_cache_pins(**args, apply=True, extended_proofs_enabled=True)

    assert _kept(result)[("activation", "act-u")] == "active_reference"
    assert set(_states(args).values()) == {"live"}


def test_prepared_statuses_are_exactly_those_the_activation_worker_writes() -> None:
    """The proof reads the worker's result, so its schema and prepared statuses must stay the worker's own."""

    import ast
    import inspect

    from blueprint_pipeline import task_evaluation_launch_activation_queue as activation_queue
    from blueprint_pipeline import task_evaluation_launch_activation_worker as worker

    written = set()
    for node in ast.walk(ast.parse(inspect.getsource(worker))):
        if isinstance(node, ast.Dict):
            fields = {key.value: value for key, value in zip(node.keys, node.values) if isinstance(key, ast.Constant)}
            if (isinstance(fields.get("schema_version"), ast.Name) and fields["schema_version"].id == "RESULT_SCHEMA_VERSION"
                    and isinstance(fields.get("status"), ast.Constant)):
                written.add(fields["status"].value)

    assert terminal_pins.ACTIVATION_RESULT_SCHEMA_VERSION == activation_queue.RESULT_SCHEMA_VERSION
    assert terminal_pins.ACTIVATION_ENVELOPE_SCHEMA_VERSION == activation_queue.ENVELOPE_SCHEMA_VERSION
    from blueprint_pipeline import task_evaluation_launch_preparation_queue as preparation_queue

    assert terminal_pins.PREPARATION_ENVELOPE_SCHEMA_VERSION == preparation_queue.ENVELOPE_SCHEMA_VERSION
    assert written - {"blocked"} == terminal_pins.PREPARED_ACTIVATION_STATUSES
    assert terminal_pins.activation_queue_root_of(["/q/task-evaluation-launches", "/q/task-evaluation-launch-activations/"]) == Path(
        "/q/task-evaluation-launch-activations")
    assert terminal_pins.activation_queue_root_of(["/q/task-evaluation-launches"]) is None
    assert terminal_pins.activation_queue_root_of(["/a/task-evaluation-launch-activations",
                                                "/b/task-evaluation-launch-activations"]) is None


def test_extended_pin_proofs_only_list_candidates_until_enabled(tmp_path) -> None:
    """Every extended proof lists its candidates on every tick; only the opt-in lets one release a pin."""

    args = _args(tmp_path)
    queue = _activation_queue(tmp_path, args)
    _prepared(_preparation_queue(tmp_path, args), "prep-stale")
    _pin(args, "preparation", "prep-stale", age=LAPSE + DAY)
    _registry_run(tmp_path / "evidence", "act-registry")
    _pin(args, "activation", "act-registry", age=7 * DAY)
    _activation_result(queue, "act-unlaunched")
    _pin(args, "activation", "act-unlaunched", age=LAPSE + DAY)
    _sealed_cold_case(tmp_path, args)
    extended = {"preparation": "unconsumed_stale_pin", "act-registry": "sealed_registry_run",
                "act-unlaunched": "activation_expired_unlaunched"}

    listed = reconcile_terminal_cache_pins(**args, apply=True)

    assert listed["enabled"] is False and listed["candidate_count"] == 4
    assert {row["owner_id"]: (row["proof"]["kind"], row["enabled"]) for row in listed["candidates"]} == {
        "prep-stale": (extended["preparation"], False), "act-registry": (extended["act-registry"], False),
        "act-unlaunched": (extended["act-unlaunched"], False), "cold": ("sealed_cold_run", True)}
    assert listed["candidate_count_by_proof"] == {
        "activation_expired_unlaunched": 1, "sealed_cold_run": 1, "sealed_registry_run": 1, "unconsumed_stale_pin": 1}
    # Only the original proof released anything, and the next tick lists the same three again.
    assert _states(args) == {("preparation", "prep-stale"): "live", ("activation", "act-registry"): "live",
                             ("activation", "act-unlaunched"): "live", ("activation", "cold"): "released"}
    again = reconcile_terminal_cache_pins(**args, apply=True)
    assert [(row["owner_id"], row["enabled"]) for row in again["candidates"]] == [
        ("prep-stale", False), ("act-registry", False), ("act-unlaunched", False)]
    assert again["released"] == []

    applied = reconcile_terminal_cache_pins(**args, apply=True, extended_proofs_enabled=True)

    assert applied["enabled"] is True and applied["released_count"] == 3
    assert applied["released_count_by_kind"] == {"activation": 2, "preparation": 1}
    assert set(_states(args).values()) == {"released"}


@pytest.mark.parametrize("value", [None, "", "1", "true", " YES ", "0", "false", "No", "sometimes", "2"])
def test_extended_pin_proofs_setting_parses_like_the_other_opt_ins(value) -> None:
    name = terminal_pins.EXTENDED_PIN_PROOFS_ENV
    environ = {} if value is None else {name: value}
    scene = {} if value is None else {gc_module.SCENE_WORKSPACE_RETIREMENT_ENV: value}

    enabled, alert = terminal_pins.extended_pin_proofs_setting(environ)
    scene_enabled, scene_alert = gc_module.scene_workspace_retirement_setting(scene)

    assert enabled == scene_enabled
    assert alert == (None if scene_alert is None else "extended_pin_proofs_setting_invalid")
    assert (enabled, alert) == replay_gc._truthy_setting(environ, name, "extended_pin_proofs_setting_invalid")


def test_the_command_line_reads_the_extended_pin_proofs_opt_in(tmp_path, monkeypatch, capsys) -> None:
    seen: list[dict] = []

    def run(**kwargs):
        seen.append(kwargs)
        return {"schema_version": gc_module.RUN_SCHEMA_VERSION, "report_digest": "sha256:0"}

    monkeypatch.setattr(gc_module, "run_storage_gc", run)
    # Every other opt-in on: none of them turns this one on.
    for other in (gc_module.EVIDENCE_OFFLOAD_ENV, gc_module.SCENE_WORKSPACE_RETIREMENT_ENV,
                  replay_gc.REPLAY_CACHE_RETENTION_ENV):
        monkeypatch.setenv(other, "1")
    monkeypatch.delenv(terminal_pins.EXTENDED_PIN_PROOFS_ENV, raising=False)
    for value in (None, "sometimes", "1"):
        if value is not None:
            monkeypatch.setenv(terminal_pins.EXTENDED_PIN_PROOFS_ENV, value)
        assert gc_module.main(["run", "--pins-root", str(tmp_path / "pins")]) == 0

    assert [(call["extended_pin_proofs_enabled"], call["extended_pin_proofs_alert"]) for call in seen] == [
        (False, None), (False, "extended_pin_proofs_setting_invalid"), (True, None)]
    assert "storage_gc_alert:extended_pin_proofs_setting_invalid" in capsys.readouterr().err


def test_a_tick_passes_the_opt_in_and_the_activation_queue_to_the_pin_pass(tmp_path, monkeypatch) -> None:
    # No process on this host references anything.
    monkeypatch.setattr(retention, "process_reference", lambda _root, **_kwargs: None)
    args = _args(tmp_path)
    queue = _activation_queue(tmp_path, args)
    _prepared(_preparation_queue(tmp_path, args), "prep-stale")
    _activation_result(queue, "act-unlaunched")
    _pin(args, "activation", "act-unlaunched", age=LAPSE + DAY)
    _pin(args, "preparation", "prep-stale", age=LAPSE + DAY)
    common = {"content_store_roots": [], "derived_roots": [], "queue_roots": args["queue_roots"],
              "pins_root": args["pins_root"], "evidence_roots": args["evidence_roots"], "now": lambda: NOW,
              "classifier": _classifier, "apply": True, "ack": RUN_ACK, "running_commit": RUNNING}
    alert = "extended_pin_proofs_setting_invalid"

    listed = run_storage_gc(**common, extended_pin_proofs_alert=alert)

    assert (listed["opt_in"]["extended_pin_proofs"], listed["alerts"]) == (False, [alert])
    phase = listed["terminal_cache_pins"]
    assert (phase["enabled"], phase["alerts"], phase["released"]) == (False, [alert], [])
    # The unlaunched activation and the superseded preparation are found only through the queues the tick
    # derives from its queue roots, and the stale pin only through the release the tick runs.
    assert {row["owner_id"]: row["proof"]["kind"] for row in phase["candidates"]} == {
        "prep-stale": "unconsumed_stale_pin", "act-unlaunched": "activation_expired_unlaunched"}

    applied = run_storage_gc(**common, extended_pin_proofs_enabled=True)

    assert applied["opt_in"]["extended_pin_proofs"] is True and "alerts" not in applied
    assert applied["terminal_cache_pins"]["released_count_by_kind"] == {"activation": 1, "preparation": 1}
    assert set(_states(args).values()) == {"released"}


def test_extended_pin_proofs_stay_an_operator_opt_in() -> None:
    deploy = Path(__file__).resolve().parents[1] / "deploy" / "systemd"
    unit = (deploy / "blueprint-control-plane-storage-gc.service").read_text(encoding="utf-8")
    example = (deploy / "pipeline-control-plane.env.example").read_text(encoding="utf-8").splitlines()
    name = terminal_pins.EXTENDED_PIN_PROOFS_ENV

    assert f"Environment={name}=" not in unit
    assert f"# {name}=1" in example
    assert not any(line.startswith(f"{name}=") for line in example), "it stays plan-only by default"
    # The unit's queue roots name the activation queue, so the unlaunched proof finds its results.
    queue_roots = next(line.split("=", 2)[2] for line in unit.splitlines()
                       if line.startswith(f"Environment={gc_module.QUEUE_ROOTS_ENV}="))
    assert terminal_pins.activation_queue_root_of(queue_roots.split(":")) == Path(
        "/var/lib/blueprint/pipeline-control-plane/task-evaluation-launch-activations")
    assert terminal_pins.preparation_queue_root_of(queue_roots.split(":")) == Path(
        "/var/lib/blueprint/pipeline-control-plane/task-evaluation-launch-preparations")
