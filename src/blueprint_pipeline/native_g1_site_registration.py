"""Register a website site's G1 humanoid candidates by the service itself, under one standing approval.

The owner approved on 2026-11-06 (beta launch tracker, "G1 humanoid on website screening", answer 2)
that the service registers each site, with no per-site person. The screening worker calls
:func:`ensure_site_registration` once per site pass, before it reads the site's profile. It writes the
service's own overlay row, exactly the shape of the root registry's ``site_screening_candidates.<site_id>``,
at ``<root>/<site_id>.json``; ``team_candidate_registry_reader`` reads it when the root-owned registry has
no current row for the site. The root registry and the door's manual install stay as they are.

Inputs resolve as the door's builder resolves them (``native_g1_registration_install``): the site's base
packet and task contract from the team registry and the completed source queue
(``native_g1_registration_contract.site_inputs``); the shared publisher source, runtime source and Arena
robot USD from the controller-pinned ``config/native_g1_shared_inputs.v1.json`` instead of the historical
ADP registry. Before it prepares a new packet, and again before it writes any row, it reads the site's
consent live (``beta_policy_screening_worker.require_current_source``); ``prepare_native_g1_beta_packet``
writes the packet under the site's own ``native-g1/<base>/<source>/`` directory. A refusal (a consent
refusal, :data:`CONSENT_REFUSALS`) prepares nothing, writes no row and removes the site's existing row;
the status file then records only that reason. An unanswered check writes nothing new and is tried
again on the next pass. Registration grants no spend and claims nothing: planning re-derives every
packet and source (``beta_policy_screening_g1.candidate_execution_spec``) and re-checks consent live
(``compose_beta_authority``). Withdrawal deletion removes the row, its status and the packets
(``website_capture_withdrawal_native_g1``).

Only privacy, rights and money stop work (AGENTS.md). A G1 problem here is a fixed reason in
``<root>/<site_id>.status.json``, never an exception. ``BLUEPRINT_NATIVE_G1_AUTO_REGISTRATION=0`` turns
the registration, and the reading of its rows, off.
"""
from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any

from . import team_candidate_registry_reader as reader
from .beta_policy_screening_g1_catalog import CANDIDATES
from .decision_evidence_contracts import canonical_digest
from .native_g1_beta_authority import SOURCE_FIELDS

MANIFEST_PATH = Path(__file__).resolve().parents[2] / "config" / "native_g1_shared_inputs.v1.json"
MANIFEST_SCHEMA = "native_g1_shared_inputs.v1"
STATUS_SCHEMA = "native_g1_site_registration_status.v1"
PACKET_RECEIPT = "native_task_arena_packet_receipt.v1.json"
PUBLISHER_STAGE_SCHEMA = "native_g1_publisher_source_stage.v1"
# The goal authority names the door's registration looks for (native_g1_registration_contract).
NAVIGATION_AUTHORITY_NAMES = frozenset({"navigation_authority.json", "native_g1_navigation_goal_authority.v1.json"})
# A restore on the host, or a preparation already running, clears these with no new source, pin or
# release: tried again on the next pass. Any other reason waits for one of those three to change.
RETRY_EVERY_PASS = frozenset({
    "g1_shared_publisher_source_missing", "g1_shared_publisher_source_changed",
    "g1_shared_runtime_source_missing", "g1_shared_runtime_source_changed",
    "g1_shared_robot_asset_missing", "g1_beta_preparation_in_progress",
})
# Final consent refusals besides every ``beta_screening_*`` code of ``require_current_source``: a capture
# withdrawn on this host (``resolve_capture_root``) and retained consent the site's inputs refuse
# (``native_g1_registration_contract.site_inputs``). Each removes the site's row and keeps no customer data.
CONSENT_REFUSALS = frozenset({"team_scene_capture_withdrawn", "native_g1_install_retained_task_consent_invalid"})
_CODE = re.compile(r"[a-z][a-z0-9_]{1,127}")


class _NotRegistered(Exception):
    def __init__(self, reason: str, *, retry: bool = False):
        super().__init__(reason)
        self.reason, self.retry = reason, retry or reason in RETRY_EVERY_PASS


def _code(exc: BaseException, fallback: str) -> str:
    return str(exc) if _CODE.fullmatch(str(exc)) else fallback


def consent_refused(reason: Any) -> bool:
    """Whether a registration reason is a final consent refusal (the row goes; only the reason is kept)."""
    return isinstance(reason, str) and (reason.startswith("beta_screening_") or reason in CONSENT_REFUSALS)


def _require_current_consent(current: Mapping[str, Any], site_id: str) -> None:
    """Privacy: read the site's consent live; a refusal is final, an unanswered check is tried again."""
    from .beta_policy_screening_worker import require_current_source
    try:
        require_current_source(current, site_id=site_id)
    except Exception as exc:  # noqa: BLE001 - a refusal is final; an unanswered check is tried again
        code = str(exc) if isinstance(exc, ValueError) else ""
        if _CODE.fullmatch(code) and consent_refused(code):
            raise _NotRegistered(code) from exc
        raise _NotRegistered("g1_current_consent_unverified", retry=True) from exc


def load_shared_inputs(path: str | Path = MANIFEST_PATH) -> dict[str, Any]:
    """The controller's pinned shared G1 inputs, checked against their own digest."""
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        paths = (value["publisher_source"]["path"], value["runtime_source"]["path"],
                 value["robot_asset"]["manipulation_packet_dir"])
        if (value.get("schema_version") != MANIFEST_SCHEMA
                or value.get("manifest_digest") != canonical_digest(value, digest_field="manifest_digest")
                or not all(isinstance(item, str) and Path(item).is_absolute() for item in paths)
                or not value["candidates"] or not set(value["candidates"]) <= set(CANDIDATES)
                or not isinstance(value["authorization_reference"], str)
                or not value["authorization_reference"].strip()):
            raise ValueError("manifest")
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        raise _NotRegistered("g1_shared_inputs_manifest_invalid") from exc
    return value


def _contract():
    # The door builder's own stdlib resolution, loaded as the builder loads it (read only here).
    from .native_g1_registration_install import contract
    return contract


def _shared_records(manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """The pinned publisher and runtime source receipts, their sha and size measured now."""
    contract, records = _contract(), {}
    for key in ("publisher_source", "runtime_source"):
        pinned = manifest[key]
        try:
            record = contract.file_record(Path(pinned["path"]))
            if key == "publisher_source":
                stage = contract.common.read(Path(pinned["path"]))
                changed = (stage.get("schema_version") != PUBLISHER_STAGE_SCHEMA
                           or stage.get("receipt_digest") != pinned["receipt_digest"]
                           or canonical_digest(stage, digest_field="receipt_digest") != pinned["receipt_digest"])
            else:
                changed = (record["sha256"], record["size_bytes"]) != (pinned["sha256"], pinned["size_bytes"])
        except FileNotFoundError as exc:
            raise _NotRegistered(f"g1_shared_{key}_missing") from exc
        except (OSError, ValueError) as exc:
            raise _NotRegistered(f"g1_shared_{key}_changed") from exc
        if changed:
            raise _NotRegistered(f"g1_shared_{key}_changed")
        records[key] = record
    return records


def _robot_record(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """The exact Arena robot USD, found as the builder finds it: the old scene plan's ``robot.usd_path``."""
    contract, pinned = _contract(), manifest["robot_asset"]
    try:
        root = contract.common.safe_path(Path(pinned["manipulation_packet_dir"]))
        old = contract.common.read(root / "native_task_arena_scene_plan.v1.json")
        raw = Path(old["robot"]["usd_path"])
        asset = raw if raw.is_absolute() else contract._child(root, str(raw))
        return contract.bound(asset, {"sha256": pinned["sha256"], "size_bytes": pinned["size_bytes"]})
    except FileNotFoundError as exc:
        raise _NotRegistered("g1_shared_robot_asset_missing") from exc
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise _NotRegistered("g1_shared_robot_asset_changed") from exc


def _records_verify(row: Mapping[str, Any], manifest_path: Path) -> bool:
    """Every record still names its file's bytes, and the shared ones are still the pinned ones."""
    contract, seen = _contract(), {}
    try:
        manifest = load_shared_inputs(manifest_path)
        pinned = manifest["runtime_source"]
        for item in row["screening_candidates"].values():
            records = item["native_g1"]
            if (records["publisher_source"]["path"] != manifest["publisher_source"]["path"]
                    or records["runtime_source"] != {key: pinned[key] for key in ("path", "sha256", "size_bytes")}):
                return False
            for record in records.values():
                path = record["path"]
                if path not in seen:
                    seen[path] = contract.file_record(Path(path))
                if seen[path] != dict(record):
                    return False
    except (_NotRegistered, OSError, ValueError, KeyError, TypeError, AttributeError):
        return False
    return bool(seen)


def _current_base_hex(site_id: str, current: Mapping[str, Any], source_queue: Path) -> str | None:
    """The site's current base packet receipt digest (hex), or ``None`` when it cannot be resolved now."""
    try:
        registry = reader.load_registry()
        if registry is None:
            return None
        _root, _inputs, base_receipt, _request, _scene = _contract().site_inputs(
            {"source": dict(current), "site_id": site_id}, registry, source_queue=source_queue)
        return str(base_receipt["receipt_digest"])[7:]
    except Exception:  # noqa: BLE001 - unresolvable now: the full path records why
        return None


def _built_from_base(row: Mapping[str, Any], base_hex: str | None) -> bool:
    """Every packet in the row was prepared from this base packet (its ``native-g1/<base>/`` directory).

    A recompiled team scene keeps its six-field source but gets a new base receipt; the row prepared from
    the old one would fail ``g1_beta_source_scene_derivation_mismatch`` at planning, so it is replaced.
    """
    marker = f"/native-g1/{base_hex}/"
    return base_hex is not None and all(marker in item["native_g1"]["scene_packet"]["path"]
                                        for item in row["screening_candidates"].values())


def _measure_spawns(row: Mapping[str, Any] | None) -> None:
    """Measure each registered packet's G1 spawn once (``native_g1_spawn_clearance``); never raises.

    Planning funds no G1 run from a packet whose report reads overlap and flags one without a report as
    unmeasured; a packet prepared before the check existed has none, so this writes it beside the packet.
    A failure leaves it missing: planning then flags the row and funds it.
    """
    from .native_g1_spawn_clearance import ensure_packet_spawn_clearance

    try:
        packets = {str(Path(item["native_g1"]["scene_packet"]["path"]).parent)
                   for item in (row or {}).get("screening_candidates", {}).values()}
    except (AttributeError, KeyError, TypeError):
        return
    for packet in sorted(packets):
        try:
            ensure_packet_spawn_clearance(packet)
        except Exception:  # noqa: BLE001 - a G1 problem never stops the site's other candidates
            continue


def _navigation_authority(site_root: Path, packet_dir: Path) -> dict[str, Any] | None:
    """Exactly one goal authority for this G1 scene in the site's own tree, else None."""
    contract = _contract()
    plan = contract.common.read(packet_dir / "native_task_arena_scene_plan.v1.json")
    matches, visited = [], 0
    for directory, directories, files in os.walk(site_root, followlinks=False):
        directories[:] = sorted(d for d in directories if not (Path(directory) / d).is_symlink())
        visited += len(files) + len(directories)
        if visited > 20_000:
            return None
        for name in sorted(set(files) & NAVIGATION_AUTHORITY_NAMES):
            try:
                path = Path(directory) / name
                if (contract.common.read(path).get("scope") or {}).get("scene_plan_digest") == plan.get("plan_digest"):
                    matches.append(contract.file_record(path))
            except (OSError, ValueError, AttributeError):
                continue
    return matches[0] if len(matches) == 1 else None


def _write(path: Path, value: Mapping[str, Any]) -> None:
    """Replace one file atomically (temporary file, then ``os.replace``), mode 0640."""
    if not path.is_absolute() or any(part.is_symlink() for part in path.parents):
        raise ValueError("g1_site_registration_root_invalid")
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o750)
    descriptor, temporary = tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write((json.dumps(value, indent=2, sort_keys=True) + "\n").encode())
            os.fchmod(stream.fileno(), 0o640)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _remove(path: Path) -> None:
    """Remove one row by its own name (a link is removed, never followed); never through a linked root."""
    if not path.is_absolute() or any(part.is_symlink() for part in path.parents):
        raise ValueError("g1_site_registration_root_invalid")
    try:
        path.unlink()
    except FileNotFoundError:
        return
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _read_status(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8")) if not path.is_symlink() else {}
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _record_status(path: Path, status: dict[str, Any]) -> dict[str, Any]:
    """Write the site's status when it says something new (``generated_at`` aside)."""
    prior = _read_status(path)
    if {k: v for k, v in prior.items() if k != "generated_at"} != {k: v for k, v in status.items() if k != "generated_at"}:
        _write(path, {**status, "generated_at": datetime.now(timezone.utc).isoformat()})
    return status


def _register(*, site_id: str, current: dict[str, Any], source_queue: Path, release: str | None,
              path: Path, status_path: Path, manifest_path: Path) -> tuple[dict[str, Any], bool]:
    """The site's status, and whether it is worth recording."""
    status: dict[str, Any] = {"schema_version": STATUS_SCHEMA, "site_id": site_id, "registered_by": "service_automatic",
        "authorization_reference": None, "source": current, "shared_inputs_digest": None, "release": release,
        "execution_authority_granted": False, "provider_allocations_performed": 0}
    candidates: list[str] = list(CANDIDATES)
    try:
        manifest = load_shared_inputs(manifest_path)
        candidates = list(manifest["candidates"])
        status.update(authorization_reference=manifest["authorization_reference"],
                      shared_inputs_digest=manifest["manifest_digest"])
        status["attempt_key"] = canonical_digest({"source": current, "shared_inputs_digest": manifest["manifest_digest"],
                                                  "release": release})
        try:
            registry = reader.load_registry()
        except reader.TeamCandidateRegistryError as exc:
            raise _NotRegistered(_code(exc, "team_candidate_registry_invalid"), retry=True) from exc
        if registry is None:
            return {**status, "status": "not_registered", "reason": "team_candidate_registry_not_configured"}, False
        root_row = (registry.get("site_screening_candidates") or {}).get(site_id)
        if isinstance(root_row, Mapping) and root_row.get("source") == current:
            _measure_spawns(root_row)
            return {**status, "status": "root_registry_row_current", "reason": None, "candidates": {}}, True
        shared = _shared_records(manifest)
        prior = _read_status(status_path)
        if (prior.get("status") == "not_registered" and prior.get("retry") == "on_source_pin_or_release_change"
                and prior.get("attempt_key") == status["attempt_key"]):
            return prior, False  # Nothing it depends on changed: no new resolution, preparation or write.
        contract = _contract()
        try:
            site_root, inputs, base_receipt, _request, base_scene = contract.site_inputs(
                {"source": current, "site_id": site_id}, registry, source_queue=source_queue)
        except ValueError as exc:
            if str(exc) == "native_g1_install_current_site_changed":
                # Another revision is the site's current one; its own pass registers it.
                return {**status, "status": "not_current", "reason": "native_g1_install_current_site_changed"}, False
            raise _NotRegistered(_code(exc, "g1_current_site_inputs_unavailable")) from exc
        except Exception as exc:  # noqa: BLE001 - recorded by name, never raised
            raise _NotRegistered("g1_current_site_inputs_unavailable") from exc
        if base_scene.get("task_kind") != "rigid_pick_place":
            raise _NotRegistered("g1_beta_native_rigid_task_required")
        packet_dir = site_root / "native-g1" / base_receipt["receipt_digest"][7:] / canonical_digest(current)[7:]
        if not (packet_dir / PACKET_RECEIPT).exists():
            robot = _robot_record(manifest)
            _require_current_consent(current, site_id)  # Privacy: no new derived scene data once withdrawn.
            from .beta_policy_screening_g1_prepare import prepare_native_g1_beta_packet
            try:  # Exactly the door builder's call (native_g1_registration_install._prepare_current_packet).
                prepare_native_g1_beta_packet(base_packet_record=inputs["base_native_packet"], source=current,
                    site_id=site_id, output_root=site_root, robot_asset_record=robot,
                    task_success_contract=inputs["task_success_contract"])
            except Exception as exc:  # noqa: BLE001 - USD, stance and packet failures alike are recorded
                raise _NotRegistered(_code(exc, "g1_beta_preparation_failed")) from exc
        try:
            records = {**shared, "scene_packet": contract.file_record(packet_dir / PACKET_RECEIPT)}
            goal = (base_scene.get("task_spec") or {}).get("g1_navigation_goal") is not None
            authority = _navigation_authority(site_root, packet_dir) if goal else None
        except (OSError, ValueError) as exc:
            raise _NotRegistered("g1_current_scene_packet_missing") from exc
        # Privacy: a row (new or replacing one) only while the site's consent stands now, after any
        # preparation above, however long it took.
        _require_current_consent(current, site_id)
    except Exception as exc:  # noqa: BLE001 - a G1 problem never stops the site's other candidates
        refused = exc if isinstance(exc, _NotRegistered) else _NotRegistered("g1_site_registration_failed")
        if consent_refused(refused.reason):
            # No row (ensure_site_registration removes the site's old one) and only the refusal is kept:
            # no source, inputs or candidates. The attempt key (a digest) holds it until something changes.
            return {"schema_version": STATUS_SCHEMA, "site_id": site_id, "registered_by": "service_automatic",
                    "status": "not_registered", "reason": refused.reason,
                    "retry": "on_source_pin_or_release_change", "attempt_key": status.get("attempt_key")}, True
        if not isinstance(exc, _NotRegistered):
            from .bounded_error import error_description

            status["error"] = error_description(exc)
        retry = "next_pass" if refused.retry else "on_source_pin_or_release_change"
        return {**status, "status": "not_registered", "reason": refused.reason, "retry": retry,
                "candidates": {key: {"status": "unavailable", "reason": refused.reason} for key in candidates}}, True
    registrations, results = {}, {}
    for key in candidates:
        if key.endswith("_vision_navi"):
            if authority is None:  # Planning's own reason; the door's registration gives the same one.
                results[key] = {"status": "unavailable", "reason": "g1_current_navigation_goal_authority_missing"}
                continue
            registrations[key] = {"native_g1": {**records, "navigation_authority": authority}}
        elif goal:
            results[key] = {"status": "unavailable", "reason": "g1_beta_manipulation_policy_task_mismatch"}
            continue
        else:
            registrations[key] = {"native_g1": dict(records)}
        results[key] = {"status": "registered"}
    if not registrations:
        return {**status, "status": "not_registered", "reason": "g1_no_candidate_registered",
                "retry": "on_source_pin_or_release_change", "candidates": results}, True
    row = {"source": current, "screening_candidates": registrations}
    _write(path, row)
    _measure_spawns(row)  # A packet prepared above has its report already; an older one gets it here.
    return {**status, "status": "registered", "reason": None, "candidates": results}, True


def ensure_site_registration(*, site_id: str, source: Mapping[str, Any], source_queue: str | Path,
                             release: str | None = None, overlay_root: str | Path | None = None,
                             manifest_path: str | Path = MANIFEST_PATH) -> dict[str, Any]:
    """Register the site's G1 candidates once for its current source; never raises.

    Does nothing while the site's row already names the current source and its files still verify.
    Otherwise prepares the site's G1 packet and writes a new row; another site's file is never written.
    A consent refusal removes the site's row instead, and keeps removing it while the refusal stands.
    """
    if not reader.auto_registration_enabled():
        return {"status": "disabled"}
    path = reader.site_registration_path(site_id, overlay_root)
    if path is None or not isinstance(source, Mapping) or set(source) != set(SOURCE_FIELDS):
        return {"status": "not_registered", "reason": "g1_site_identity_invalid"}
    current = {key: source[key] for key in SOURCE_FIELDS}
    status_path = path.with_name(site_id + ".status.json")
    overlay = reader.site_registration_overlay(site_id, overlay_root)  # None for any unusable row
    if (overlay is not None and not consent_refused(_read_status(status_path).get("reason"))
            and overlay["source"] == current and _records_verify(overlay, Path(manifest_path))
            and _built_from_base(overlay, _current_base_hex(site_id, current, Path(source_queue)))):
        _measure_spawns(overlay)
        return {"status": "already_registered"}
    try:
        status, record = _register(site_id=site_id, current=current, source_queue=Path(source_queue),
            release=release, path=path, status_path=status_path, manifest_path=Path(manifest_path))
    except Exception as exc:  # noqa: BLE001 - a G1 problem never stops the site's other candidates
        from .bounded_error import error_description

        status, record = {"schema_version": STATUS_SCHEMA, "site_id": site_id, "registered_by": "service_automatic",
            "source": current, "status": "not_registered", "reason": "g1_site_registration_failed",
            "retry": "next_pass", "error": error_description(exc)}, True
    if consent_refused(status.get("reason")):
        try:  # A site whose consent is refused has no G1 registration (its status stays, reason only).
            _remove(path)
        except (OSError, ValueError) as exc:
            status = {**status, "registration_removal_error": type(exc).__name__}
    if record:
        try:
            _record_status(status_path, status)
        except (OSError, ValueError) as exc:
            status = {**status, "status_file_error": type(exc).__name__}
    return {key: status[key] for key in ("status", "reason", "candidates", "status_file_error",
                                         "registration_removal_error") if key in status}


__all__ = ["CONSENT_REFUSALS", "MANIFEST_PATH", "RETRY_EVERY_PASS", "consent_refused", "ensure_site_registration",
           "load_shared_inputs"]
