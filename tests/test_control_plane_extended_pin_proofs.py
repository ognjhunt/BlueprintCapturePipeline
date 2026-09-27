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

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

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


def _queue_row(args: dict, text: str) -> None:
    pending = Path(args["queue_roots"][0]) / "pending"
    pending.mkdir(parents=True, exist_ok=True)
    (pending / f"row-{len(list(pending.iterdir()))}.json").write_text(text, encoding="utf-8")


def test_unconsumed_stale_preparation_pin_is_released(tmp_path) -> None:
    """A preparation or compilation no live pin depends on, eight days old, is released.

    Its content is reproducible and re-fetched by digest, so releasing the pin
    only lets the derived phase recheck the directory as it does any other.
    """

    args = _args(tmp_path)
    _pin(args, "preparation", "prep-orphan", age=LAPSE + DAY)
    _pin(args, "preparation", "prep-compiled", age=LAPSE + DAY)
    _pin(args, "compilation", "prep-compiled", age=LAPSE + DAY,
         depends_on=[{"kind": "preparation", "owner_id": "prep-compiled"}])

    planned = reconcile_terminal_cache_pins(**args, extended_proofs_enabled=True)

    assert planned["enabled"] is True and planned["status"] == "dry_run"
    assert [(row["kind"], row["owner_id"], row["proof"]["kind"], row["enabled"]) for row in planned["candidates"]] == [
        ("preparation", "prep-orphan", "unconsumed_stale_pin", True),
        ("compilation", "prep-compiled", "unconsumed_stale_pin", True),
    ]
    # The compilation still consumes its preparation, which goes with it.
    assert _kept(planned) == {("preparation", "prep-compiled"): "depended_on"}
    assert planned["released"] == [] and planned["released_count"] == 0
    assert set(_states(args).values()) == {"live"}

    applied = reconcile_terminal_cache_pins(**args, apply=True, extended_proofs_enabled=True)

    assert applied["status"] == "applied"
    assert applied["released_count_by_kind"] == {"compilation": 1, "preparation": 2}
    assert applied["released_count"] == 3
    assert applied["candidate_count_by_proof"] == {"unconsumed_stale_pin": 2}
    assert set(_states(args).values()) == {"released"}
    assert applied["cache_or_evidence_bytes_removed"] is False


@pytest.mark.parametrize("reason", ["depended", "young", "recent", "queued", "process", "path_class"])
def test_depended_or_young_or_queued_preparation_pin_is_kept(tmp_path, reason) -> None:
    args = _args(tmp_path)
    age = {"young": LAPSE - DAY, "recent": 3600}.get(reason, LAPSE + DAY)
    paths = ["/var/lib/blueprint/pipeline-control-plane/task-evaluation-launches/prep-x"] if reason == "path_class" else None
    _pin(args, "preparation", "prep-x", age=age, paths=paths)
    if reason == "depended":
        _pin(args, "activation", "act-x", age=DAY, depends_on=[{"kind": "preparation", "owner_id": "prep-x"}])
    if reason == "queued":
        _queue_row(args, json.dumps({"preparation_id": "prep-x"}))
    if reason == "process":
        args["reference_checker"] = lambda path: path.name == "prep-x"

    result = reconcile_terminal_cache_pins(**args, apply=True, extended_proofs_enabled=True)

    assert _kept(result)[("preparation", "prep-x")] == {
        "depended": "depended_on", "young": "pin_not_stale", "recent": "pin_young", "queued": "active_reference",
        "process": "active_reference", "path_class": "path_class_invalid"}[reason]
    assert not any(row["owner_id"] == "prep-x" for row in result["candidates"])
    assert result["released"] == [] and _states(args)[("preparation", "prep-x")] == "live"


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


def _registry_run(evidence: Path, name: str, *, status: str = "completed_unqualified", idle: float = 5 * DAY) -> Path:
    """A run whose result registry the artifact store accepts as sealed, the registry idle ``idle`` seconds."""

    run = evidence / name
    closeout = run / "closeout"
    closeout.mkdir(parents=True)
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


@pytest.mark.parametrize("reason", ["hot", "tampered", "not_terminal", "closeout_changed", "pointer", "linked"])
def test_hot_or_invalid_registry_keeps_its_pin(tmp_path, reason) -> None:
    args = _args(tmp_path)
    evidence = tmp_path / "evidence"
    if reason == "linked":
        (evidence / "act-x").symlink_to(_registry_run(tmp_path / "elsewhere", "act-x"))
    else:
        run = _registry_run(evidence, "act-x", idle=DAY if reason == "hot" else 5 * DAY,
                            status="running" if reason == "not_terminal" else "completed_unqualified")
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
        "closeout_changed": "registry_unsealed", "pointer": "run_pointer_present", "linked": "run_path_unsafe"}[reason]
    assert result["candidates"] == [] and _states(args)[("activation", "act-x")] == "live"
