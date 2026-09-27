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

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from blueprint_pipeline.control_plane_storage_pins import load_storage_pins, write_storage_pin
from blueprint_pipeline.control_plane_storage_roots import require_storage_class
from blueprint_pipeline.control_plane_terminal_cache_pins import reconcile_terminal_cache_pins
from blueprint_pipeline.decision_evidence_contracts import canonical_digest

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
