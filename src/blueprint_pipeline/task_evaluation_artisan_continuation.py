"""Continue one closed attempt's paid ARTISAN work in the next attempt for the same scene.

capture-447ae4f4, 2026-11-02: ARTISAN designed, built and exported a candidate
tray, then stopped in its review phase when the object's frozen authoring cap
was spent ($13.76 of $15, ``artisan_shared_stage_authority_exhausted``). The
next attempt would repeat every paid upstream phase.

This module selects such a source, carries the source's exact upstream
workspace into the successor bundle, restores it at its original absolute root
and lets the ARTISAN driver resume the upstream run at its saved phase.

It grants no spend. The successor's own configuration supplies the cap, and the
cap's scope (``whole_authoring_stage_cumulative_including_resume``) counts the
carried spend: the successor's meter and its official cost reservation receive
only ``cap - carried``. The carried spend stays settled under the source
attempt, so nothing is counted twice.

Lineage, closure, provider-zero and archive authentication reuse the partial
Astra transport's checks unchanged (``task_evaluation_partial_astra_transport``).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import tempfile
from collections.abc import Mapping
from typing import Any
import zipfile

from .decision_evidence_contracts import canonical_digest
from .control_plane_disk_budget import reserve_control_plane_disk
from . import task_evaluation_partial_astra_transport as transport

SCHEMA_VERSION = "task_evaluation_artisan_continuation_adoption.v1"
TRANSPORT_SCHEMA = "task_evaluation_artisan_continuation_transport.v1"
FIELD = "artisan_continuation"
PREFIX = "stages/stage-3/producer/artisan_runtime/"
STAGE_INPUT = "stages/stage-3/producer/stage_production_input.v1.json"
CANDIDATES_PREFIX = "stages/stage-3/producer/candidates/"
ORIGINAL_ROOT = ("/workspace/task_evaluation_scene_configuration_provider_bundle/runtime_output/"
                 + PREFIX.rstrip("/"))
#: Run-bound records of the source: carried for evidence under ``continuation/source/``,
#: never at the paths the successor's own run writes.
RELOCATED = ("blueprint_request.v1.json", "bridge_context.v1.json", "blueprint_usage.json",
             "terminal.v1.json", "last_failure.json", "prompt.txt")
RELOCATED_ROOT = "continuation/source/"
#: Scratch and per-run state the successor regenerates; never carried.
_DROPPED_PARTS = {"blueprint_sandbox_ipc", "tmp", "xdg", "official_openai_cost", "sandbox_preflight",
                  "packaged_blender"}
_SECRET_PARTS = {".env", "secrets", "credentials", "runtime_secrets", "openai_api_key"}
ARTISAN_BACKEND = "artisan_v1"


class ArtisanContinuationError(ValueError):
    """A fixed predicate refused the continuation; never artifact contents."""


def _require(condition: Any, reason: str) -> None:
    if not condition:
        raise ArtisanContinuationError("artisan_continuation_" + reason)


# ---------------------------------------------------------------------------
# Semantics: what must be identical between the source and the successor.


def _without_cap(budget: Any) -> Any:
    if not isinstance(budget, Mapping):
        return budget
    return {key: value for key, value in budget.items() if key != "maximum_cost_usd"}


def semantic_configuration(configuration: Mapping[str, Any]) -> dict[str, Any]:
    """The stage-3 configuration with only its cap amount removed."""

    value = json.loads(json.dumps(configuration))
    value["authoring_budget"] = _without_cap(value.get("authoring_budget"))
    return value


def semantic_request(request: Mapping[str, Any]) -> dict[str, Any]:
    """The construction request without run identity, cap amount or per-run URIs.

    Frames, source candidate, dimensions, physical evidence, constraints and
    models must be identical. capture-447ae4f4's two stage-3 requests differed
    only in these run-scoped fields (2026-11-01 12:27Z vs 23:26Z).
    """

    value = json.loads(json.dumps(request))
    for key in ("run_id", "request_digest", "source_commit", "source_envelope_digest", "configuration_sha256"):
        value.pop(key, None)
    value["authoring_budget"] = _without_cap(value.get("authoring_budget"))
    if isinstance(value.get("configuration"), dict):
        value["configuration"] = semantic_configuration(value["configuration"])
    placement = value.get("support_and_placement")
    if isinstance(placement, dict):
        placement.pop("recipe_id", None)
        placement.pop("recipe_digest", None)
        for stage in placement.get("stage_sequence") or []:
            configuration = stage.get("configuration") if isinstance(stage, dict) else None
            if isinstance(configuration, dict):
                configuration.pop("uri", None)
                if (stage.get("adapter") or {}).get("id") == "content_agents_rigid_replacement":
                    # The ARTISAN stage's own configuration digest binds its cap.
                    configuration.pop("digest", None)
                    configuration.pop("size_bytes", None)
    return value


def carried_spend(usage: Mapping[str, Any]) -> dict[str, Any]:
    """Every metered call of the source, which must all be priced."""

    calls = usage.get("calls")
    _require(usage.get("schema_version") == "artisan_upstream_usage.v1" and isinstance(calls, list),
             "carried_usage_invalid")
    total, largest = 0.0, 0.0
    for row in calls:
        cost = row.get("cost_usd") if isinstance(row, Mapping) else None
        _require(isinstance(row, Mapping) and row.get("status") == "priced"
                 and isinstance(cost, (int, float)) and not isinstance(cost, bool)
                 and math.isfinite(cost) and cost >= 0, "unknown_charge_requires_reconciliation")
        total += float(cost)
        largest = max(largest, float(cost))
    return {"cost_usd": round(total, 6), "call_count": len(calls), "largest_call_usd": round(largest, 6)}


def frozen_cap(configuration: Mapping[str, Any]) -> float:
    from .task_object_authoring_backend import frozen_authoring_cap
    return frozen_authoring_cap(configuration)


# ---------------------------------------------------------------------------
# Selection on the control plane (activation time).


def _classify(relative: str) -> str | None:
    """``keep`` at the original root, ``relocate`` under continuation/source, or drop."""

    parts = PurePosixPath(relative).parts
    _require(parts and not PurePosixPath(relative).is_absolute() and ".." not in parts, "archive_path_invalid")
    _require(not any(part in _SECRET_PARTS or part.endswith((".pem", ".p12", ".key")) for part in parts),
             "runtime_secret_forbidden")
    if relative in RELOCATED:
        return "relocate"
    if relative.endswith(".lock") or any(part in _DROPPED_PARTS for part in parts):
        return None
    if parts[0] == "workspace":
        return "keep"
    return None


def _members(archive: zipfile.ZipFile) -> list[tuple[str, str, zipfile.ZipInfo]]:
    infos = archive.infolist()
    _require(len(infos) <= transport.MAX_MEMBERS and len({i.filename for i in infos}) == len(infos),
             "archive_members_invalid")
    selected = []
    for info in infos:
        if not info.filename.startswith(PREFIX) or info.is_dir():
            continue
        relative = info.filename[len(PREFIX):]
        kind = _classify(relative)
        if kind is None:
            continue
        _require(stat.S_IFMT(info.external_attr >> 16) in (0, stat.S_IFREG) and info.file_size >= 0,
                 "archive_entry_invalid")
        destination = relative if kind == "keep" else RELOCATED_ROOT + relative
        selected.append((destination, relative, info))
    _require(sum(i.file_size for _, _, i in selected) <= transport.MAX_EXPANDED_BYTES, "archive_expansion_exceeded")
    return sorted(selected)


def _archive_json(archive: zipfile.ZipFile, name: str) -> dict[str, Any]:
    info = archive.getinfo(name)
    _require(info.file_size <= 16 * 1024**2, "archive_json_too_large")
    value = json.loads(archive.read(info))
    _require(isinstance(value, dict), "archive_json_invalid")
    return value


def _successor_configuration(envelope: Mapping[str, Any]) -> dict[str, Any]:
    stages, references = envelope["recipe"]["stage_sequence"], envelope["stage_configuration_references"]
    _require(len(stages) == len(references), "configuration_reference_count_invalid")
    rows = [(index, row) for index, (stage, row) in enumerate(zip(stages, references, strict=True))
            if stage.get("stage_id") == "stage-3"]
    _require(len(rows) == 1, "configuration_stage_invalid")
    index, row = rows[0]
    _require(row.get("contract_path") == f"construction.recipe.stage_sequence.{index}.configuration",
             "configuration_reference_slot_invalid")
    path = transport._safe(str(row.get("materialized_path") or ""))
    _require(path.is_file() and transport._sha(path) == row.get("digest"), "configuration_reference_changed")
    value = json.loads(path.read_text())
    _require(isinstance(value, dict), "configuration_invalid")
    return value


def successor_is_artisan(envelope: Mapping[str, Any]) -> bool:
    """Whether the successor's stage 3 runs ARTISAN (and so may continue a source)."""

    try:
        return _successor_configuration(envelope).get("authoring_backend") == ARTISAN_BACKEND
    except (ArtisanContinuationError, OSError, KeyError, TypeError, ValueError):
        return False


def _source_state(archive: zipfile.ZipFile, *, source_run_id: str) -> dict[str, Any]:
    """The source's resumable upstream state and carried spend, or a typed refusal."""

    names = set(archive.namelist())
    _require(not any(name.startswith(CANDIDATES_PREFIX) for name in names), "source_completed")
    request = _archive_json(archive, PREFIX + "blueprint_request.v1.json")
    _require(request.get("request_digest") == canonical_digest(request, digest_field="request_digest")
             and request.get("run_id") == source_run_id
             and request.get("backend") == ARTISAN_BACKEND, "source_request_invalid")
    upstream_run_id = "pl_" + request["request_digest"][7:39]
    terminal = _archive_json(archive, PREFIX + "terminal.v1.json")
    _require(terminal.get("schema_version") == "blueprint_artisan_terminal.v1"
             and terminal.get("run_id") == upstream_run_id
             and terminal.get("registered") is False and terminal.get("phase") != "DONE"
             and terminal.get("status") == "running" and not terminal.get("termination_reason"),
             "upstream_not_resumable")
    state = f"{PREFIX}workspace/data/cache/runs/{upstream_run_id}/pipeline_state.json"
    _require(state in names, "upstream_checkpoint_missing")
    usage = _archive_json(archive, PREFIX + "blueprint_usage.json")
    spend = carried_spend(usage)
    source_input = _archive_json(archive, STAGE_INPUT)
    configuration = source_input.get("configuration")
    _require(isinstance(configuration, dict) and configuration.get("authoring_backend") == ARTISAN_BACKEND
             and request.get("configuration") == configuration
             and not str(configuration.get("schema_version", "")).startswith("articulated_"),
             "source_configuration_invalid")
    return {"request": request, "upstream_run_id": upstream_run_id, "terminal_phase": terminal["phase"],
            "spend": spend, "usage_sha256": "sha256:" + hashlib.sha256(
                archive.read(PREFIX + "blueprint_usage.json")).hexdigest(),
            "configuration": configuration}


def select_artisan_continuation_source(*, owner_attempt_path, envelope, output_root,
                                       intent_root=None, launch_root=None, activation_request=None,
                                       activation_root=None, pins_root=None, on_pin_created=None):
    """Freeze the newest closed same-owner/scene/task ARTISAN source that can resume.

    Returns the selection path, or ``None`` when no source qualifies (the
    successor then runs ARTISAN from its start, as before).
    """

    from . import task_evaluation_scene_intake as intake
    current = transport._read(owner_attempt_path, "owner_attempt_digest")
    transport.validate_envelope_owner(envelope, current)
    configuration = _successor_configuration(envelope)
    if configuration.get("authoring_backend") != ARTISAN_BACKEND:
        return None
    target = transport._safe(output_root)
    root = transport._safe(intent_root or os.environ[intake.ROOT_ENV])
    if target.exists():
        retained = transport._read(target / "selection.json", "transport_digest")
        _require(retained.get("schema_version") == TRANSPORT_SCHEMA
                 and retained["authority_evidence"]["successor_owner_attempt"] == current
                 and retained["verified_lineage"]["successor_run_id"] == envelope["run_id"], "retained_selection_changed")
        lineage, proof = transport._source_identity(Path(retained["source_launch_root"]), current,
            intent_root=root, successor_run_id=envelope["run_id"])
        _require(proof == retained["authority_evidence"] and lineage == retained["verified_lineage"]
                 and transport._file(Path(retained["source_archive"]["path"])) == retained["source_archive"],
                 "retained_source_changed")
        transport._pin_source_metadata(proof=proof, activation_request=activation_request,
            activation_root=activation_root, pins_root=pins_root, on_pin_created=on_pin_created)
        return target / "selection.json"
    transport._canonical_owner(current, root)
    launches = transport._safe(launch_root or os.getenv("BLUEPRINT_TASK_EVALUATION_LAUNCH_STATE_ROOT")
                               or root.parent / "task-evaluation-launch-runs")
    if not launches.exists():
        return None
    sources = []
    for index, source in enumerate(launches.iterdir()):
        if index >= transport.MAX_SCAN:
            break
        zero = source / "post_teardown_provider_zero_receipt.json"
        if not (source.is_dir() and not source.is_symlink() and zero.is_file() and not zero.is_symlink()):
            continue
        try:
            profile = transport._read(source / "launch_profile.json")
            scope = profile.get("task_evaluation_run") or {}
            if ((profile.get("scene_attempt_binding") or {}).get("intent_digest") == current["scene_intent_digest"]
                    and scope.get("scene_id") == current["scene_id"] and scope.get("task_id") == current["task_id"]):
                sources.append((zero.stat().st_mtime_ns, source))
        except (OSError, ValueError, KeyError, TypeError):
            continue
    cap = frozen_cap(configuration)
    rejected = []
    for _, source in sorted(sources, reverse=True)[:transport.MAX_CANDIDATES]:
        created = False
        try:
            lineage, proof = transport._source_identity(source, current, intent_root=root,
                                                        successor_run_id=envelope["run_id"])
            archive_path = transport._safe(proof["source_result"]["provider_runtime_output_zip_path"])
            _require(archive_path.is_file() and archive_path.stat().st_size <= transport.MAX_ARCHIVE_BYTES,
                     "source_archive_invalid")
            original = transport._file(archive_path)
            _require(original["sha256"] == proof["source_result"]["provider_runtime_output_zip_sha256"],
                     "source_archive_download_changed")
            with zipfile.ZipFile(archive_path) as archive:
                if PREFIX + "terminal.v1.json" not in set(archive.namelist()):
                    rejected.append({"source_launch_id": source.name, "reason": "artisan_continuation_no_upstream_run"})
                    continue
                state = _source_state(archive, source_run_id=lineage["source_run_id"])
                _require(semantic_configuration(state["configuration"]) == semantic_configuration(configuration),
                         "configuration_semantics_changed")
                # A continuation that cannot afford one call as large as the source
                # already made would rent a GPU only to exhaust again.
                _require(cap - state["spend"]["cost_usd"] >= state["spend"]["largest_call_usd"] > 0,
                         "cumulative_cap_below_one_source_call")
                members = _members(archive)
                _require(any(dest.endswith(f"/{state['upstream_run_id']}/pipeline_state.json")
                             for dest, _, _ in members), "upstream_checkpoint_missing")
                transport._pin_source_metadata(proof=proof, activation_request=activation_request,
                    activation_root=activation_root, pins_root=pins_root, on_pin_created=on_pin_created)
                with reserve_control_plane_disk(
                    "semantic_pretraining", target_root=target.parent,
                    expected_bytes=sum(i.file_size for _, _, i in members) + 16 * 1024**2,
                ):
                    target.mkdir(parents=True, exist_ok=False)
                    created = True
                    packed = target / "retained_runtime.zip"
                    inventory = []
                    with zipfile.ZipFile(packed, "x", compression=zipfile.ZIP_STORED) as output:
                        for destination, _, info in members:
                            digest = hashlib.sha256()
                            with archive.open(info) as stream, output.open(destination, "w") as sink:
                                while chunk := stream.read(1024 * 1024):
                                    digest.update(chunk)
                                    sink.write(chunk)
                            inventory.append({"relative_path": destination, "sha256": "sha256:" + digest.hexdigest(),
                                              "size_bytes": info.file_size})
            _require(transport._file(archive_path) == original
                     and packed.stat().st_size <= transport.MAX_ARCHIVE_BYTES, "source_archive_changed")
            descriptor = {
                "schema_version": SCHEMA_VERSION,
                "source_run_id": lineage["source_run_id"], "successor_run_id": envelope["run_id"],
                "original_runtime_root": ORIGINAL_ROOT,
                "source_request_digest": state["request"]["request_digest"],
                "source_upstream_run_id": state["upstream_run_id"],
                "source_terminal_phase": state["terminal_phase"],
                "semantic_request_digest": canonical_digest(semantic_request(state["request"])),
                "semantic_configuration_digest": canonical_digest(semantic_configuration(configuration)),
                "carried_spend": {**state["spend"], "usage_sha256": state["usage_sha256"]},
                "owner_intent_lineage": lineage,
                "retained_runtime_archive": {k: v for k, v in transport._file(packed).items() if k != "path"},
                "retained_files": inventory,
                "new_spend_authorized": False,
            }
            descriptor["adoption_digest"] = canonical_digest(descriptor, digest_field="adoption_digest")
            descriptor_path = target / "descriptor.json"
            transport._write(descriptor_path, descriptor)
            selection = {"schema_version": TRANSPORT_SCHEMA, "verified_lineage": lineage,
                         "authority_evidence": proof, "source_launch_root": str(source), "intent_root": str(root),
                         "source_archive": original, "descriptor": transport._file(descriptor_path),
                         "runtime_archive": transport._file(packed), "rejected_sources": rejected,
                         "new_spend_authorized": False}
            selection["transport_digest"] = canonical_digest(selection, digest_field="transport_digest")
            selection_path = target / "selection.json"
            transport._write(selection_path, selection)
            return selection_path
        except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile) as exc:
            rejected.append({"source_launch_id": source.name, "reason": str(exc)[:240]})
            if created:
                shutil.rmtree(target)  # only this function's own incomplete directory
    return None


def select_stage_three_continuation(operations: dict, *, envelope: Mapping[str, Any], activation_root: Path,
                                    activation_request: Mapping[str, Any] | None, created_storage_pins) -> None:
    """Freeze the stage-3 continuation an activation carries, if any, into ``operations``.

    An ARTISAN successor may continue a resumable ARTISAN source
    (``artisan_continuation_selection``); otherwise the partial Astra transport
    selects as before (``partial_astra_successor_selection``). Selection grants
    no spend. Typed refusals stop the activation.
    """

    pinned = activation_request if created_storage_pins is not None else None
    on_pin = created_storage_pins.append if created_storage_pins is not None else None
    common = dict(owner_attempt_path=operations["scene_owner_attempt"], envelope=envelope,
                  activation_request=pinned, activation_root=activation_root, on_pin_created=on_pin)
    if successor_is_artisan(envelope):
        selection = select_artisan_continuation_source(output_root=Path(activation_root) / "artisan_continuation",
                                                       **common)
        if selection is not None:
            operations["artisan_continuation_selection"] = str(selection)
            return
    selection = transport.select_partial_astra_source(output_root=Path(activation_root) / "partial_astra_successor",
                                                      **common)
    if selection is not None:
        operations["partial_astra_successor_selection"] = str(selection)


#: What ``select_stage_three_continuation`` refuses with.
StageThreeContinuationError = (ArtisanContinuationError, transport.PartialAstraTransportError)


def validate_artisan_transport(value: Mapping[str, Any], descriptor: Mapping[str, Any]) -> None:
    """The partial-Astra transport's owner-record chain, under this transport's schema."""

    _require(descriptor.get("schema_version") == SCHEMA_VERSION
             and descriptor.get("new_spend_authorized") is False, "transport_schema_invalid")
    transport.validate_transport(value, descriptor, schema=TRANSPORT_SCHEMA)


def stage_artisan_continuation(*, selection_path, runtime, successor_run_id, owner_attempt_path, envelope):
    """Reopen authorities and original bytes, then copy only the sealed transport into the bundle."""

    value = transport._read(selection_path, "transport_digest")
    descriptor = transport._read(value["descriptor"]["path"], "adoption_digest")
    validate_artisan_transport(value, descriptor)
    actual_owner = transport._read(owner_attempt_path, "owner_attempt_digest")
    transport.validate_envelope_owner(envelope, actual_owner)
    _require(envelope["run_id"] == successor_run_id == descriptor["successor_run_id"], "successor_run_changed")
    _require(actual_owner == value["authority_evidence"]["successor_owner_attempt"], "successor_owner_changed")
    lineage, proof = transport._source_identity(Path(value["source_launch_root"]),
        value["authority_evidence"]["successor_owner_attempt"], intent_root=value["intent_root"],
        successor_run_id=successor_run_id)
    _require(lineage == value["verified_lineage"] and proof == value["authority_evidence"]
             and transport._file(Path(value["source_archive"]["path"])) == value["source_archive"],
             "source_identity_changed")
    result: dict[str, Any] = {"verified_lineage": lineage}
    base = Path(runtime) / "input/artisan_continuation"
    for name, filename in (("descriptor", "descriptor.json"), ("runtime_archive", "retained_runtime.zip")):
        source = Path(value[name]["path"])
        _require(transport._file(source) == value[name], "transport_file_changed")
        destination = base / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        copied = transport._file(destination)
        _require(copied["sha256"] == value[name]["sha256"], "transport_copy_changed")
        result[name] = {"path": destination.relative_to(runtime).as_posix(), "digest": copied["sha256"],
                        "size_bytes": copied["size_bytes"]}
    proof_path = base / "transport.json"
    transport._write(proof_path, value)
    result["transport"] = {"path": proof_path.relative_to(runtime).as_posix(),
                           "digest": transport._sha(proof_path), "size_bytes": proof_path.stat().st_size}
    return result


# ---------------------------------------------------------------------------
# Provider side: restore, then adopt under the successor's own authority.


def _inventory(root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(p for p in root.rglob("*") if p.is_file() or p.is_symlink()):
        _require(not path.is_symlink(), "restored_symlink_forbidden")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
        rows.append({"relative_path": path.relative_to(root).as_posix(), "sha256": "sha256:" + digest.hexdigest(),
                     "size_bytes": path.stat().st_size})
    return rows


def verify_descriptor(descriptor: Mapping[str, Any], *, verified_lineage: Mapping[str, Any],
                      successor_run_id: str) -> None:
    claimed = descriptor.get("owner_intent_lineage")
    _require(descriptor.get("schema_version") == SCHEMA_VERSION
             and descriptor.get("adoption_digest") == canonical_digest(descriptor, digest_field="adoption_digest")
             and descriptor.get("original_runtime_root") == ORIGINAL_ROOT
             and descriptor.get("successor_run_id") == successor_run_id
             and descriptor.get("source_run_id") != successor_run_id
             and descriptor.get("new_spend_authorized") is False
             and isinstance(claimed, Mapping) and claimed == verified_lineage
             and claimed.get("source_run_id") == descriptor.get("source_run_id")
             and claimed.get("successor_run_id") == successor_run_id, "descriptor_invalid")


def restore_artisan_continuation(*, descriptor: Mapping[str, Any], archive_path: Path, runtime_root: Path,
                                 verified_lineage: Mapping[str, Any], successor_run_id: str) -> bool:
    """Restore exact archived bytes at the original root; ``False`` if this run already restored them.

    A same-run retry of the successor finds its own sealed adoption record and
    does not restore again; any other existing root is refused.
    """

    verify_descriptor(descriptor, verified_lineage=verified_lineage, successor_run_id=successor_run_id)
    _require(str(runtime_root) == descriptor["original_runtime_root"], "original_root_changed")
    record = descriptor["retained_runtime_archive"]
    actual = transport._file(archive_path)
    _require(actual["sha256"] == record.get("sha256") and actual["size_bytes"] == record.get("size_bytes")
             and actual["size_bytes"] <= transport.MAX_ARCHIVE_BYTES, "archive_invalid")
    adoption = runtime_root / RELOCATED_ROOT.rstrip("/") / "adoption.v1.json"
    if runtime_root.exists():
        _require(not runtime_root.is_symlink() and adoption.is_file() and not adoption.is_symlink()
                 and json.loads(adoption.read_text()).get("adoption_digest") == descriptor["adoption_digest"],
                 "original_root_occupied")
        return False
    expected = {}
    for row in descriptor.get("retained_files") or []:
        relative = PurePosixPath(row["relative_path"])
        _require(not relative.is_absolute() and ".." not in relative.parts and str(relative) == row["relative_path"]
                 and relative.parts and str(relative) not in expected and type(row["size_bytes"]) is int
                 and row["size_bytes"] >= 0, "inventory_invalid")
        expected[str(relative)] = row
    _require(expected and sum(row["size_bytes"] for row in expected.values()) <= transport.MAX_EXPANDED_BYTES,
             "inventory_invalid")
    runtime_root.parent.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix=".artisan-continuation-", dir=runtime_root.parent))
    try:
        with zipfile.ZipFile(archive_path) as bundle:
            entries = [entry for entry in bundle.infolist() if not entry.is_dir()]
            _require(len(entries) == len(expected) and {e.filename for e in entries} == set(expected),
                     "archive_inventory_mismatch")
            for entry in entries:
                row = expected[entry.filename]
                _require(stat.S_IFMT(entry.external_attr >> 16) in {0, stat.S_IFREG}
                         and entry.file_size == row["size_bytes"], "archive_entry_invalid")
                target = scratch / entry.filename
                target.parent.mkdir(parents=True, exist_ok=True)
                digest = hashlib.sha256()
                with bundle.open(entry) as source, target.open("xb") as output:
                    while chunk := source.read(1024 * 1024):
                        digest.update(chunk)
                        output.write(chunk)
                _require("sha256:" + digest.hexdigest() == row["sha256"], "archive_entry_changed")
        _require(_inventory(scratch) == sorted(descriptor["retained_files"], key=lambda r: r["relative_path"]),
                 "restored_inventory_mismatch")
        os.rename(scratch, runtime_root)
    finally:
        if scratch.exists():
            shutil.rmtree(scratch)
    return True


def adopt_artisan_continuation(*, descriptor: Mapping[str, Any], runtime_root: Path, request: Mapping[str, Any],
                               cumulative_cap: float, attempt_allocation: float) -> dict[str, Any]:
    """Bind the restored source to the successor's request and split the cumulative cap.

    ``cumulative_cap`` is the configuration's frozen whole-stage cap, which the
    carried spend counts against. ``attempt_allocation`` is this attempt's
    signed allocation, which bounds new spend only. Returns the sealed adoption
    record: ``increment_usd`` is the only new spend this attempt's meter and
    official cost reservation may carry, and ``upstream_cap_usd`` is the
    cumulative cap the upstream resume, which keeps its own running tally, sees.
    """

    relocated = runtime_root / RELOCATED_ROOT
    previous = json.loads((relocated / "blueprint_request.v1.json").read_text())
    usage = json.loads((relocated / "blueprint_usage.json").read_text())
    spend = carried_spend(usage)
    usage_sha = "sha256:" + hashlib.sha256((relocated / "blueprint_usage.json").read_bytes()).hexdigest()
    carried = descriptor["carried_spend"]
    _require(previous.get("request_digest") == descriptor["source_request_digest"]
             and previous.get("run_id") == descriptor["source_run_id"]
             and canonical_digest(semantic_request(previous)) == descriptor["semantic_request_digest"],
             "source_request_changed")
    _require(semantic_request(previous) == semantic_request(request)
             and request.get("run_id") == descriptor["successor_run_id"]
             and canonical_digest(semantic_configuration(request["configuration"]))
                 == descriptor["semantic_configuration_digest"], "scientific_inputs_changed")
    _require(spend == {k: carried[k] for k in ("cost_usd", "call_count", "largest_call_usd")}
             and usage_sha == carried["usage_sha256"],
             "carried_usage_changed")
    _require(all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and v > 0
                 for v in (cumulative_cap, attempt_allocation)), "cap_invalid")
    increment = round(min(float(attempt_allocation), float(cumulative_cap) - spend["cost_usd"]), 6)
    _require(increment > 0, "cumulative_cap_exhausted")
    checkpoint = (runtime_root / "workspace/data/cache/runs" / descriptor["source_upstream_run_id"]
                  / "pipeline_state.json")
    _require(checkpoint.is_file() and not checkpoint.is_symlink(), "upstream_checkpoint_missing")
    adoption_path = relocated / "adoption.v1.json"
    record = {"schema_version": "task_evaluation_artisan_continuation_record.v1",
              "adoption_digest": descriptor["adoption_digest"],
              "source_run_id": descriptor["source_run_id"], "successor_run_id": descriptor["successor_run_id"],
              "source_request_digest": descriptor["source_request_digest"],
              "successor_request_digest": request["request_digest"],
              "upstream_run_id": descriptor["source_upstream_run_id"],
              "cumulative_cap_usd": float(cumulative_cap), "attempt_allocation_usd": float(attempt_allocation),
              "carried_cost_usd": spend["cost_usd"], "carried_call_count": spend["call_count"],
              "increment_usd": increment, "upstream_cap_usd": round(spend["cost_usd"] + increment, 6),
              "carried_spend_settled_under_source_attempt": True, "new_spend_authorized": False}
    record["record_digest"] = canonical_digest(record, digest_field="record_digest")
    if adoption_path.exists():
        _require(json.loads(adoption_path.read_text()) == record, "adoption_record_changed")
    else:
        adoption_path.write_text(json.dumps(record, sort_keys=True) + "\n")
    return record


__all__ = [
    "ArtisanContinuationError",
    "StageThreeContinuationError",
    "select_stage_three_continuation",
    "FIELD",
    "ORIGINAL_ROOT",
    "adopt_artisan_continuation",
    "carried_spend",
    "restore_artisan_continuation",
    "select_artisan_continuation_source",
    "semantic_configuration",
    "semantic_request",
    "stage_artisan_continuation",
    "successor_is_artisan",
    "validate_artisan_transport",
]
