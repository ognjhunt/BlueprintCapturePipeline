"""Measured admission for a website scene configuration's provider output.

Plan 13a.1, step 3a1.0. Under ``ceiling`` (the default, today's path byte for
byte) a paid run needs 5U + 512 MiB of raw free space before staging, where U
is the provider's upload ceiling: U for the returned zip and 4U for the
largest extraction that zip may declare. Under ``measured``:

1. the output holds U + 512 MiB on the disk ledger (role
   ``scene_configuration_output``) from before the paid allocation until the
   result is sealed, bound to the job directory so its history learns real
   footprints. A CPU prefix that runs on this host first leaves archives in
   the job directory, bounded by one unpacked bundle. When it also reserves on
   the output's volume, admission checks the larger of its own need and the
   hold plus those archives up front, and the hold is taken only after that
   prefix released its own reservation; otherwise one reservation holds the
   output and those archives from the start. A hold
   refused then is sealed like a refusal before staging: typed, zero provider
   mutations, and retried by capacity recovery only when the attempt recorded
   no paid external spend. API pretraining that ran, or a CPU prefix whose own
   cost records show or cannot rule out OpenAI or Anthropic spend, withholds
   the retry (``recovery_withheld``), since a retry never repeats paid work;
2. the local zip, from whichever source, is published to B2 before anything is
   extracted, on every outcome;
3. the extraction is sized from the zip's own central directory, and whatever
   the hold no longer covers is taken as a growth reservation or refused. A
   refused extraction leaves the output durable (B2 and the local zip), but
   nothing recovers it automatically yet: publication recovery requires
   ``configuration_completed``, which stays false until readers can fetch
   archive members on demand (plan 13a.1, PR C).

Only production website runs are measured. Diagnostic and warm-session runs
carry their checkpoint subtree into the next run and non-website runs are
unmeasured, so they stay on ``ceiling``. An unknown mode refuses before
staging. Nothing here allocates, grants or mutates a provider.
"""

from __future__ import annotations

import functools
import json
import os
import re
import shutil
import zipfile
from collections.abc import Callable, Mapping
from pathlib import Path, PurePosixPath
from typing import Any

from .decision_evidence_contracts import canonical_digest
from .control_plane_disk_budget import (
    DEFAULT_RESERVATION_ROOT,
    ControlPlaneDiskBudgetError,
    DiskReservation,
    disk_headroom,
    reserve_control_plane_disk,
    target_device,
)
from .control_plane_disk_usage import tree_usage
from .task_evaluation_scene_configuration_provider_artifacts import (
    PROVIDER_OUTPUT_MAXIMUM_EXPANSION_RATIO,
    PROVIDER_OUTPUT_MAXIMUM_MEMBER_COUNT,
    PROVIDER_OUTPUT_OPERATIONAL_RESERVE_BYTES,
    TaskEvaluationSceneConfigurationVastError,
    _provider_output_disk_requirements,
    _publish_provider_output_archive,
    _sha256,
)


OUTPUT_ADMISSION_ENV = "BLUEPRINT_SCENE_CONFIGURATION_OUTPUT_ADMISSION"
CEILING_MODE = "ceiling"
MEASURED_MODE = "measured"
OUTPUT_ROLE = "scene_configuration_output"
ADMISSION_SCHEMA_VERSION = "scene_configuration_provider_output_disk_admission.v1"
EXTRACTION_SCHEMA_VERSION = "scene_configuration_provider_output_extraction_admission.v1"
MODE_INVALID_BLOCKER = "scene_configuration_output_admission_mode_invalid"
BUDGET_EXCEEDED_BLOCKER = "scene_configuration_provider_output_disk_budget_exceeded"
ADMISSION_UNAVAILABLE_BLOCKER = "scene_configuration_provider_output_disk_admission_unavailable"
EXTRACTION_BUDGET_EXCEEDED_BLOCKER = (
    "scene_configuration_provider_output_extraction_budget_exceeded"
)
RESERVATION_ROOT_ENV = "BLUEPRINT_CONTROL_PLANE_DISK_RESERVATION_ROOT"
#: Footprint samples are labelled by lane.
WORKLOAD = "website_scene_configuration"
#: Both CPU phases that may run on this host before the GPU reserve 3 x the
#: bundle's unpacked bytes + 512 MiB: the prefix in
#: ``task_evaluation_scene_configuration_cpu_prestage`` and the ArtiFixer
#: semantic preparation in ``task_evaluation_artifixer_pretraining``.
CPU_PREFIX_EXPANSION = 3
CPU_PREFIX_OVERHEAD_BYTES = 512 * 1024**2
#: What a CPU phase leaves in the job directory once it released its own
#: reservation is bounded by one unpacked bundle: the API pretraining capsule
#: re-archives the extracted bundle with its prepared frames, and the prestage
#: keeps its capsule and output archive, both archives of its stage outputs.
#: Each phase's own 3 x unpacked estimate already budgets its outgoing archive
#: at one unpacked share, so admission counts that share beside the output.
CPU_PREFIX_RESIDUE_SHARES = 1
PREALLOCATION_PHASE = "before_allocation_and_staging"
#: Where a hold deferred behind a CPU prefix sharing the volume is taken.
DEFERRED_HOLD_PHASE = "after_cpu_prefix"
#: Why a typed refusal after the prefix is not retried automatically: a retry
#: never repeats paid external work, so recorded or unprovable spend holds it.
API_PRETRAINING_CONSUMED = "api_pretraining_consumed"
PREFIX_SPEND_RECORDED = "prefix_external_spend_recorded"
PREFIX_SPEND_UNPROVEN = "prefix_spend_unproven"
PREFIX_SPEND_NONE = "no_external_spend_recorded"
#: The prestage's retained output, the cost reservations of failed stages included.
CPU_PREFIX_OUTPUT_ARCHIVE = "cpu_prestage_output.zip"
#: The lane's result schema; measured runs are never diagnostic.
LANE_RESULT_SCHEMA_VERSION = "task_evaluation_scene_configuration_vast_result.v1"
_HOLD_DEFERRED = "deferred_until_after_cpu_prefix"
_LEDGER_EXCEEDED = "control_plane_disk_budget_exceeded:"
_LEDGER_NUMBERS = re.compile(r"(need|available|free|floor|reserved)_bytes=(\d+)")
_READ_ERRORS = (
    EOFError, OSError, RuntimeError, ValueError, zipfile.BadZipFile, zipfile.LargeZipFile,
)
#: Admissions holding ledger entries, keyed by job directory, until the lane
#: seals that job's terminal result (``release_scene_configuration_output``).
_HELD: dict[str, SceneConfigurationOutputAdmission] = {}


class SceneConfigurationOutputHoldRefused(TaskEvaluationSceneConfigurationVastError):
    """The hold deferred behind a CPU prefix no longer fits; nothing was allocated."""


def configured_output_admission_mode(environment: Mapping[str, str] | None = None) -> str | None:
    """``ceiling`` when unset or empty, ``measured`` when asked for, else None."""

    raw = (os.environ if environment is None else environment).get(OUTPUT_ADMISSION_ENV)
    if raw is None or raw == "":
        return CEILING_MODE
    return raw if raw in (CEILING_MODE, MEASURED_MODE) else None


def bundle_unpacked_bytes(bundle_path: Path) -> int:
    with zipfile.ZipFile(bundle_path) as archive:
        return sum(member.file_size for member in archive.infolist())


def cpu_prefix_peak_bytes(bundle_path: Path) -> int:
    """What a CPU phase over this bundle reserves before the GPU is rented."""

    return CPU_PREFIX_EXPANSION * bundle_unpacked_bytes(bundle_path) + CPU_PREFIX_OVERHEAD_BYTES


def output_volume_requirement(hold_bytes: int, phases: list[Mapping[str, Any]]) -> int:
    """Bytes the output's volume must have available above its floor.

    ``phases`` are the pre-GPU CPU phases in the order they run. Each leaves
    archives in the job directory after releasing its own reservation, so a
    phase that shares the volume needs its peak beside what earlier phases
    left, and the output hold then needs to fit beside everything they left.
    """

    required, residue = hold_bytes, 0
    for phase in phases:
        if phase["shares_output_volume"]:
            required = max(required, int(phase["peak_bytes"]) + residue)
        residue += int(phase["residue_bytes"])
    return max(required, hold_bytes + residue)


def extraction_requirement(archive_path: Path, *, maximum_archive_bytes: int) -> dict[str, Any]:
    """The bytes extracting this exact zip needs, read from its central directory.

    A zip the extractor refuses before writing any member -- absent,
    unreadable, above the upload ceiling, over the member cap or beyond the
    expansion bound -- needs only the operational reserve, and the extractor
    keeps its own typed refusal.
    """

    requirement: dict[str, Any] = {
        "archive_name": archive_path.name,
        "archive_present": archive_path.is_file(),
        "archive_bytes": None,
        "member_count": None,
        "expanded_bytes": None,
        "extraction_bytes": 0,
    }
    if requirement["archive_present"]:
        size = archive_path.stat().st_size
        requirement["archive_bytes"] = size
        try:
            with zipfile.ZipFile(archive_path) as archive:
                members = archive.infolist()
        except _READ_ERRORS:
            members = None
        if members is not None:
            expanded = sum(member.file_size for member in members)
            requirement.update(member_count=len(members), expanded_bytes=expanded)
            if (
                size <= maximum_archive_bytes
                and len(members) <= PROVIDER_OUTPUT_MAXIMUM_MEMBER_COUNT
                and expanded <= maximum_archive_bytes * PROVIDER_OUTPUT_MAXIMUM_EXPANSION_RATIO
            ):
                requirement["extraction_bytes"] = expanded
    requirement["operational_reserve_bytes"] = PROVIDER_OUTPUT_OPERATIONAL_RESERVE_BYTES
    requirement["required_bytes"] = (
        requirement["extraction_bytes"] + PROVIDER_OUTPUT_OPERATIONAL_RESERVE_BYTES
    )
    return requirement


def cpu_prefix_spend(archive_path: Path) -> dict[str, Any]:
    """What the CPU prefix's own records show it reserved against OpenAI or Anthropic.

    The prestage keeps its runtime output, a failed stage's cost reservations
    included, as ``cpu_prestage_output.zip``. Every paid path reserves before
    its first call: OpenAI through receipts in an ``*official_openai_cost/``
    directory, Anthropic and Agents reviews through ``inference_reservations/``
    files and their audit manifest. Zero spend is proven only by an archive
    complete as its writer seals it (its exclusions manifest the first member,
    its own retained-checkpoint manifest the last) holding no such
    reservation. A missing, unreadable or partial archive, or an audit that
    does not verify, proves nothing and counts as spent.
    """

    from .task_evaluation_scene_configuration_output_archive import EXCLUDED_PARTS
    from .task_evaluation_supervisor.inference_reservations import (
        INFERENCE_RESERVATION_MANIFEST_SCHEMA_VERSION as MANIFEST_SCHEMA,
    )

    evidence: dict[str, Any] = {
        "archive_name": archive_path.name,
        "archive_sha256": None,
        "cap_record_count": 0,
        "status": PREFIX_SPEND_UNPROVEN,
    }
    try:
        if archive_path.is_symlink() or not archive_path.is_file():
            return evidence
        evidence["archive_sha256"] = _sha256(archive_path)
        recorded, unverified = 0, False
        with zipfile.ZipFile(archive_path) as archive:
            names = archive.namelist()

            def read(name: str) -> Any:
                if archive.getinfo(name).file_size > 2 * 1024**2:
                    raise ValueError("prefix_spend_record_oversized")
                return json.loads(archive.read(name))

            # The writer opens with its exclusions and closes with its own
            # retained-checkpoint manifest; a crash in between leaves neither
            # at its place, so a partial archive cannot pass for a whole one.
            closing = read(names[-1]) if names else None
            if (
                len(names) != len(set(names))
                or "runtime_output_missing.json" in names
                or names[0] != "provider_output_zip_exclusions.json"
                or names[-1] != "retained_training_checkpoints.json"
                or not isinstance(closing, dict)
                or closing.get("schema_version")
                != "scene_configuration_retained_training_checkpoints.v1"
                or not isinstance(closing.get("checkpoints"), list)
                or read(names[0]) != {
                    "schema_version": (
                        "task_evaluation_scene_configuration_provider_output_zip_exclusions.v1"
                    ),
                    "excluded_directory_names": sorted(EXCLUDED_PARTS),
                }
            ):
                return evidence
            for name in names:
                parts = PurePosixPath(name).parts
                if any(part.endswith("official_openai_cost") for part in parts[:-1]) or (
                    "inference_reservations" in parts and parts[-1] != "manifest.json"
                ):
                    recorded += 1
                elif parts[-1] == "inference_audit.json" or parts[-2:] == (
                    "inference_reservations", "manifest.json"
                ):
                    manifest = read(name)
                    if (
                        not isinstance(manifest, dict)
                        or manifest.get("schema_version") != MANIFEST_SCHEMA
                        or manifest.get("inference_reservation_manifest_digest")
                        != canonical_digest(
                            manifest, digest_field="inference_reservation_manifest_digest"
                        )
                    ):
                        unverified = True
                    elif any(
                        manifest.get(key) != 0
                        for key in (
                            "reservation_count", "in_flight_unknown_count", "reserved_max_cost_usd",
                        )
                    ):
                        recorded += 1
        evidence["cap_record_count"] = recorded
        evidence["status"] = (
            PREFIX_SPEND_RECORDED if recorded
            else PREFIX_SPEND_UNPROVEN if unverified
            else PREFIX_SPEND_NONE
        )
    except (*_READ_ERRORS, LookupError, TypeError):
        evidence["status"] = PREFIX_SPEND_UNPROVEN
    return evidence


def _ledger_numbers(refusal: str) -> dict[str, int]:
    """The ledger's own numbers from its typed refusal (never a path)."""

    return {f"{name}_bytes": int(value) for name, value in _LEDGER_NUMBERS.findall(refusal)}


def _refusal_blocker(refusal: str, exceeded: str) -> str:
    return exceeded if refusal.startswith(_LEDGER_EXCEEDED) else ADMISSION_UNAVAILABLE_BLOCKER


def _output_projection(
    target: Path, *, reservation_root: str | Path, disk_usage: Callable[[Path], Any]
) -> dict[str, int]:
    """Free, floor, live reservations and available bytes as this role's admission sees them."""

    headroom = disk_headroom(
        target_root=target, reservation_root=reservation_root, disk_usage=disk_usage
    )
    row = next(row for row in headroom["targets"] if row["role"] == OUTPUT_ROLE)
    return {
        key: int(row[key])
        for key in ("free_bytes", "floor_bytes", "reserved_bytes", "available_bytes")
    }


def _is_website_request(envelope: Mapping[str, Any]) -> bool:
    request = envelope.get("request") if isinstance(envelope, Mapping) else None
    scene = request.get("scene") if isinstance(request, Mapping) else None
    return isinstance(scene, Mapping) and bool(scene.get("website_native_inputs"))


class SceneConfigurationOutputAdmission:
    """One lane run's output admission. Every method passes through under ``ceiling``."""

    def __init__(
        self,
        *,
        mode: str | None,
        measured: bool,
        job: Path,
        receipt: Mapping[str, Any],
        expected_upload_bytes: int,
        cpu_prefix_phases: list[tuple[str, Path]],
        reservation_root: str | Path,
        disk_usage: Callable[[Path], Any],
    ) -> None:
        self.mode = mode
        self.measured = measured
        self.job = job
        self.receipt = receipt
        self.maximum_archive_bytes = expected_upload_bytes
        self.hold_bytes = expected_upload_bytes + PROVIDER_OUTPUT_OPERATIONAL_RESERVE_BYTES
        self.record: dict[str, Any] = {}
        self.hold: DiskReservation | None = None
        self.growth: DiskReservation | None = None
        self.archive_durable = False
        self._cpu_prefix_phases = cpu_prefix_phases
        self._reservation_root = reservation_root
        self._disk_usage = disk_usage
        self._publication: tuple[dict[str, Any], dict[str, Any], Path] | None = None
        self._publication_error: Exception | None = None
        self._deferred_hold_refused = False
        #: What the hold already covered when the output's footprint began.
        self._covered_before_output = 0

    # -- before allocation -------------------------------------------------

    def before_allocation(
        self, legacy_check: Callable[..., dict[str, Any]], **legacy_arguments: Any
    ) -> dict[str, Any]:
        """The pre-staging capacity record; ``status`` is ``ready`` or ``blocked``."""

        if self.mode is None:
            return {
                "schema_version": ADMISSION_SCHEMA_VERSION,
                "status": "blocked",
                "phase": PREALLOCATION_PHASE,
                "mode": None,
                "blockers": [MODE_INVALID_BLOCKER],
            }
        if not self.measured:
            return legacy_check(**legacy_arguments)
        self.record = {
            "schema_version": ADMISSION_SCHEMA_VERSION,
            "status": "blocked",
            "phase": PREALLOCATION_PHASE,
            "mode": MEASURED_MODE,
            "role": OUTPUT_ROLE,
            "measurement_path": str(self.job),
            "maximum_archive_bytes": self.maximum_archive_bytes,
            "operational_reserve_bytes": PROVIDER_OUTPUT_OPERATIONAL_RESERVE_BYTES,
            "hold_bytes": self.hold_bytes,
            "sequential_phases": [],
            "required_available_bytes": self.hold_bytes,
            "free_bytes": None,
            "floor_bytes": None,
            "reserved_bytes": None,
            "available_bytes": None,
            "hold": "not_taken",
            "hold_phase": None,
            "hold_reservation": None,
            "blockers": [],
        }
        record = self.record
        try:
            phases = self._cpu_prefix_phase_rows()
            record["required_available_bytes"] = output_volume_requirement(
                self.hold_bytes, phases
            )
            deferred = any(row["shares_output_volume"] for row in phases)
            if deferred:
                # A prefix reserves on this volume first and releases before the
                # hold is taken, so the two are checked in sequence, not summed.
                record.update(
                    _output_projection(
                        self.job, reservation_root=self._reservation_root,
                        disk_usage=self._disk_usage,
                    )
                )
        except ControlPlaneDiskBudgetError as exc:
            record.update(hold_phase=PREALLOCATION_PHASE, ledger_refusal=str(exc))
            return self._refuse(ADMISSION_UNAVAILABLE_BLOCKER)
        except _READ_ERRORS as exc:
            record.update(hold_phase=PREALLOCATION_PHASE, measurement_error_type=type(exc).__name__)
            return self._refuse(ADMISSION_UNAVAILABLE_BLOCKER)
        if not deferred:
            # One reservation now, covering the output and whatever a prefix
            # running elsewhere will leave in this job directory.
            blocker = self._take_hold(
                PREALLOCATION_PHASE, expected_bytes=record["required_available_bytes"]
            )
            return self._refuse(blocker) if blocker else record
        if record["available_bytes"] < record["required_available_bytes"]:
            record["hold_phase"] = PREALLOCATION_PHASE
            return self._refuse(BUDGET_EXCEEDED_BLOCKER)
        record.update(
            status="ready",
            hold=_HOLD_DEFERRED,
            hold_phase=DEFERRED_HOLD_PHASE,
            projection_before_staging={
                key: record[key]
                for key in ("free_bytes", "floor_bytes", "reserved_bytes", "available_bytes")
            },
        )
        return record

    def hold_before_allocation(self) -> None:
        """Start the output's footprint here, taking a hold deferred behind a CPU prefix.

        A deferred hold that no longer fits raises ``SceneConfigurationOutputHoldRefused``
        before the adapter is entered; the lane then seals ``refused_hold_result``.
        """

        if not self.measured:
            return None
        if self.hold is not None:
            # Held since admission: staging and any prefix archives written
            # since then stay covered by the hold, but the footprint the role's
            # history learns is the output's alone, from here on.
            self._covered_before_output = max(
                0, tree_usage(self.job).allocated_bytes - self.hold.baseline_bytes
            )
            self.hold.bind_workspace(self.job, fresh=True)
            return None
        if self.record.get("hold") != _HOLD_DEFERRED:
            return None
        # What the prefix left behind is on the volume now; the ledger sees it.
        blocker = self._take_hold(DEFERRED_HOLD_PHASE, expected_bytes=self.hold_bytes)
        if blocker:
            self._deferred_hold_refused = True
            self._refuse(blocker)
            raise SceneConfigurationOutputHoldRefused(blocker)
        return None

    @property
    def hold_refused(self) -> bool:
        """The hold deferred behind a CPU prefix was refused; no provider was touched."""

        return self._deferred_hold_refused

    def refused_hold_result(
        self,
        *,
        authority: Mapping[str, Any],
        consumption: Mapping[str, Any],
        api_pretraining: Mapping[str, Any],
        cpu_prestage: Mapping[str, Any],
        expected_download_bytes: int,
        cleanup: Mapping[str, Any],
        cleanup_blockers: list[str],
        watchdog_close: Mapping[str, Any],
    ) -> dict[str, Any]:
        """The terminal result of a hold refused after the CPU prefix.

        It has the shape of a refusal before staging, so capacity recovery
        re-opens it the same way, and adds what the attempt already did: the
        consumed authority, the prefix receipts, the staging cleanup and the
        watchdog closed as no allocation. Automatic recovery never repeats
        paid external work, so it is withheld when API pretraining ran or when
        the CPU prefix's own records show, or cannot rule out, external spend.
        """

        record = self.record
        prefix_spend = (
            cpu_prefix_spend(self.job / CPU_PREFIX_OUTPUT_ARCHIVE) if cpu_prestage else None
        )
        if prefix_spend is not None:
            record["prefix_spend"] = prefix_spend
        record["recovery_withheld"] = (
            API_PRETRAINING_CONSUMED if api_pretraining
            else prefix_spend["status"]
            if prefix_spend is not None and prefix_spend["status"] != PREFIX_SPEND_NONE
            else None
        )
        blockers = [*record["blockers"], *cleanup_blockers]
        if cleanup.get("all_objects_absent") is not True:
            blockers.append("object_store_provider_zero_not_proven")
        if watchdog_close.get("status") not in {"provider_terminal", "cancelled_no_allocation"}:
            blockers.append("independent_watchdog_not_closed")
        return {
            "schema_version": LANE_RESULT_SCHEMA_VERSION,
            "status": "blocked",
            "run_id": self.receipt["run_id"],
            "source_commit": self.receipt["source_commit"],
            "bundle_sha256": self.receipt["bundle_sha256"],
            "authority_digest": authority["authority_digest"],
            "authorization_consumption": dict(consumption),
            "api_pretraining": dict(api_pretraining) or None,
            "cpu_prestage": dict(cpu_prestage) or None,
            "provider_mutations_performed": 0,
            "retry_cap": 0,
            "continuing_spend_from_this_run": False,
            "expected_provider_download_bytes": expected_download_bytes,
            "expected_provider_upload_bytes": self.maximum_archive_bytes,
            "provider_output_disk_requirements": _provider_output_disk_requirements(
                self.maximum_archive_bytes
            ),
            "provider_output_disk_capacity": record,
            "all_staged_objects_absent": cleanup.get("all_objects_absent"),
            "object_store_cleanup": dict(cleanup),
            "independent_watchdog": dict(watchdog_close),
            "runtime_secret_cleanup_completed": not cleanup_blockers,
            "blockers": sorted(set(blockers)),
        }

    def _cpu_prefix_phase_rows(self) -> list[dict[str, Any]]:
        """The pre-GPU CPU phases in run order: their peak, residue and volume."""

        if not self._cpu_prefix_phases:
            return []
        unpacked = bundle_unpacked_bytes(Path(str(self.receipt["bundle_path"])))
        output_device = target_device(self.job)
        for phase, target in self._cpu_prefix_phases:
            self.record["sequential_phases"].append({
                "phase": phase,
                "peak_bytes": CPU_PREFIX_EXPANSION * unpacked + CPU_PREFIX_OVERHEAD_BYTES,
                "residue_bytes": CPU_PREFIX_RESIDUE_SHARES * unpacked,
                "shares_output_volume": target_device(target) == output_device,
            })
        return self.record["sequential_phases"]

    def _take_hold(self, hold_phase: str, *, expected_bytes: int) -> str | None:
        """Reserve the job's output hold; return a typed blocker on refusal."""

        record = self.record
        try:
            self.hold = reserve_control_plane_disk(
                OUTPUT_ROLE,
                target_root=self.job,
                expected_bytes=expected_bytes,
                reservation_root=self._reservation_root,
                disk_usage=self._disk_usage,
                workspace=self.job,
                workload=WORKLOAD,
                # Rebased on the output alone right before the adapter, so its
                # growth from there is this output's whole footprint.
                fresh=True,
            )
        except ControlPlaneDiskBudgetError as exc:
            # The ledger's own numbers at refusal time, whichever phase refused.
            numbers = _ledger_numbers(str(exc))
            numbers.pop("need_bytes", None)
            record.update(hold="refused", hold_phase=hold_phase, ledger_refusal=str(exc), **numbers)
            return _refusal_blocker(str(exc), BUDGET_EXCEEDED_BLOCKER)
        except OSError as exc:
            record.update(
                hold="refused", hold_phase=hold_phase, measurement_error_type=type(exc).__name__
            )
            return ADMISSION_UNAVAILABLE_BLOCKER
        _HELD[str(self.job)] = self
        receipt = self.hold.receipt()
        record.update(
            status="ready",
            hold="held",
            hold_phase=hold_phase,
            hold_reservation=receipt,
            free_bytes=receipt["free_bytes_at_admission"],
            floor_bytes=receipt["floor_bytes"],
            reserved_bytes=receipt["reserved_bytes_before_admission"],
            available_bytes=receipt["available_bytes_before_admission"],
        )
        return None

    def _refuse(self, blocker: str) -> dict[str, Any]:
        self.record.update(status="blocked", blockers=[blocker])
        return self.record

    # -- after the provider returned ----------------------------------------

    def extract(
        self,
        legacy_extract: Callable[..., tuple[dict[str, Any], list[str], dict[str, Any]]],
        archive_path: Path,
        destination: Path,
        **arguments: Any,
    ) -> tuple[dict[str, Any], list[str], dict[str, Any]]:
        """Make the zip durable, then extract it within the bytes it declares."""

        if not self.measured:
            return legacy_extract(archive_path, destination, **arguments)
        self._publish(archive_path)
        maximum_archive_bytes = arguments["maximum_archive_bytes"]
        requirement = extraction_requirement(
            archive_path, maximum_archive_bytes=maximum_archive_bytes
        )
        held = self.hold.expected_bytes if self.hold is not None else None
        used = (
            None if self.hold is None
            else self._covered_before_output + int(self.hold.sample() or 0)
        )
        remaining = 0 if held is None else max(0, held - int(used or 0))
        # A zip the extractor refuses before writing a member needs no room.
        writes = requirement["extraction_bytes"] > 0
        growth = max(0, requirement["required_bytes"] - remaining) if writes else 0
        record: dict[str, Any] = {
            "schema_version": EXTRACTION_SCHEMA_VERSION,
            "status": "ready",
            "phase": "before_extraction",
            "mode": MEASURED_MODE,
            "role": OUTPUT_ROLE,
            "measurement_path": str(destination.parent),
            **requirement,
            "hold_bytes": held,
            "hold_bytes_used": used,
            "hold_bytes_remaining": remaining,
            "growth_bytes": growth,
            "growth_reservation": None,
            "observed_free_bytes": None,
            "archive_durable": self.archive_durable,
            "blockers": [],
        }
        blocker = self._take_growth(growth, record) if growth else None
        if writes and blocker is None:
            # The ledger can only account for ledger writers: re-read the
            # volume itself, even when the hold alone covers the extraction.
            blocker = self._recheck_free(destination.parent, record)
        if blocker:
            # Nothing is extracted; the durable zip stays here for recovery.
            record.update(status="blocked", blockers=[blocker])
            return {}, [blocker], record
        result, blockers = arguments["extractor"](
            archive_path,
            destination,
            maximum_archive_bytes=maximum_archive_bytes,
            diagnostic_only=bool(arguments.get("diagnostic_only", False)),
        )
        return result, blockers, record

    def _recheck_free(self, directory: Path, record: dict[str, Any]) -> str | None:
        try:
            free = self._disk_usage(directory).free
        except (OSError, TypeError, ValueError) as exc:
            record["measurement_error_type"] = type(exc).__name__
            return ADMISSION_UNAVAILABLE_BLOCKER
        if type(free) is not int or free < 0:
            return ADMISSION_UNAVAILABLE_BLOCKER
        record["observed_free_bytes"] = free
        return EXTRACTION_BUDGET_EXCEEDED_BLOCKER if free < record["required_bytes"] else None

    def _take_growth(self, growth: int, record: dict[str, Any]) -> str | None:
        try:
            self.growth = reserve_control_plane_disk(
                OUTPUT_ROLE,
                target_root=self.job,
                expected_bytes=growth,
                reservation_root=self._reservation_root,
                disk_usage=self._disk_usage,
                workload=WORKLOAD,
            )
        except ControlPlaneDiskBudgetError as exc:
            record["ledger_refusal"] = str(exc)
            return _refusal_blocker(str(exc), EXTRACTION_BUDGET_EXCEEDED_BLOCKER)
        except OSError as exc:
            record["measurement_error_type"] = type(exc).__name__
            return ADMISSION_UNAVAILABLE_BLOCKER
        _HELD[str(self.job)] = self
        record["growth_reservation"] = self.growth.receipt()
        return None

    def _publish(self, archive_path: Path) -> None:
        if not archive_path.is_file():
            return
        try:
            self._publication = _publish_provider_output_archive(
                output_zip=archive_path, job=self.job, receipt=self.receipt
            )
        except Exception as exc:  # noqa: BLE001 - object-store clients vary; the lane records it
            self._publication_error = exc
            return
        _reference, index, _path = self._publication
        self.archive_durable = bool(
            index.get("status") == "completed"
            and index.get("all_artifacts_remote_verified") is True
        )

    def publishes(self, execution: Mapping[str, Any]) -> bool:
        """``ceiling`` publishes after an extraction; ``measured`` already did, on every outcome."""

        return True if self.measured else bool(execution)

    def durable_archive(
        self, publish: Callable[..., tuple[dict[str, Any], dict[str, Any], Path]], **arguments: Any
    ) -> tuple[dict[str, Any], dict[str, Any], Path]:
        """The durable reference: published now under ``ceiling``, before extraction otherwise."""

        if not self.measured:
            return publish(**arguments)
        if self._publication is None and self._publication_error is None:
            self._publish(Path(arguments["output_zip"]))
        if self._publication_error is not None:
            raise self._publication_error
        if self._publication is None:
            raise TaskEvaluationSceneConfigurationVastError(
                "scene_configuration_provider_output_archive_missing"
            )
        return self._publication

    def result_fields(self) -> dict[str, Any]:
        if not self.measured:
            return {}
        return {
            "provider_output_admission_mode": MEASURED_MODE,
            "provider_output_archive_durable": self.archive_durable,
        }

    def release(self, outcome: str) -> None:
        for reservation in (self.growth, self.hold):
            if reservation is not None:
                try:
                    reservation.release(outcome=outcome)
                except OSError:
                    pass  # the ledger drops an entry whose pid has exited


def open_scene_configuration_output_admission(
    *,
    job: Path,
    receipt: Mapping[str, Any],
    read_envelope: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    expected_upload_bytes: int,
    diagnostic_only: bool,
    retain_warm_session: bool,
    api_pretraining: bool,
    cpu_prestage: bool,
    disk_usage_provider: Callable[[Path], Any] | None = None,
    environment: Mapping[str, str] | None = None,
) -> SceneConfigurationOutputAdmission:
    """Resolve this run's mode; only a measured run reads its envelope or the ledger."""

    values = os.environ if environment is None else environment
    mode = configured_output_admission_mode(values)
    measured = bool(
        mode == MEASURED_MODE
        and not diagnostic_only
        and not retain_warm_session
        and _is_website_request(read_envelope(receipt))
    )
    phases: list[tuple[str, Path]] = []
    if measured and api_pretraining:
        from .task_evaluation_artifixer_pretraining import LOGICAL_ROOT  # noqa: PLC0415

        phases.append(("semantic_pretraining", Path(LOGICAL_ROOT)))
    if measured and cpu_prestage:
        from .task_evaluation_scene_configuration_cpu_prestage import (  # noqa: PLC0415
            DEFAULT_WORK_DIR,
            WORK_DIR_ENV,
        )

        phases.append(("cpu_prestage", Path(values.get(WORK_DIR_ENV) or DEFAULT_WORK_DIR)))
    return SceneConfigurationOutputAdmission(
        mode=mode,
        measured=measured,
        job=Path(job),
        receipt=receipt,
        expected_upload_bytes=expected_upload_bytes,
        cpu_prefix_phases=phases,
        reservation_root=values.get(RESERVATION_ROOT_ENV) or DEFAULT_RESERVATION_ROOT,
        disk_usage=disk_usage_provider or shutil.disk_usage,
    )


def release_scene_configuration_output(
    job: Path, sealed: dict[str, Any]
) -> dict[str, Any]:
    """Release a job's output reservations once its terminal result is sealed."""

    admission = _HELD.pop(str(job), None)
    if admission is not None:
        admission.release("completed" if sealed.get("status") == "completed" else "blocked")
    return sealed


def releases_output_on_exit(
    lane: Callable[..., dict[str, Any]]
) -> Callable[..., dict[str, Any]]:
    """Wrap the lane so no exit leaves its output reservations behind.

    Sealing a terminal result releases them; this only acts when the lane
    raised after admission (outcome ``failed``, with its footprint sample) or
    returned without a live seal. Otherwise the hold would stay live until the
    allocator process exited or the role's TTL passed.
    """

    @functools.wraps(lane)
    def run(**arguments: Any) -> dict[str, Any]:
        outcome = "failed"
        try:
            result = lane(**arguments)
            outcome = "completed" if result.get("status") == "completed" else "blocked"
            return result
        finally:
            key = str(Path(arguments["job_dir"]).expanduser().resolve())
            admission = _HELD.pop(key, None)
            if admission is not None:
                admission.release(outcome)

    return run


def recovery_withheld(result: Mapping[str, Any]) -> bool:
    """A measured refusal that capacity recovery must not retry: it keeps its
    typed blocker, but a retry would repeat paid external work.

    That covers API pretraining that already ran, as for a credit refusal, and
    a CPU prefix unless the sealed record carries its proof of zero external
    spend; an older or incomplete record counts as spent.
    """

    record = measured_admission_record(result)
    if record is None:
        return False
    if record.get("recovery_withheld"):
        return True
    if record.get("hold_phase") != DEFERRED_HOLD_PHASE:
        return False
    proof = record.get("prefix_spend")
    return result.get("api_pretraining") is not None or (
        result.get("cpu_prestage") is not None
        and not (isinstance(proof, Mapping) and proof.get("status") == PREFIX_SPEND_NONE)
    )


def recorded_preallocation_refusal(
    result: Mapping[str, Any], *, maximum_archive_bytes: int, job_dir: Path
) -> dict[str, Any] | None:
    """The measured refusal this job sealed before allocation, or None if it is not one.

    Refused before staging, or after a CPU prefix sharing the volume when no
    paid API pretraining ran; the ledger's own numbers must show the refusal.
    """

    record = result.get("provider_output_disk_capacity")
    if not isinstance(record, Mapping):
        return None
    required = recorded_output_requirement(record, maximum_archive_bytes=maximum_archive_bytes)
    available = record.get("available_bytes")
    valid = bool(
        required is not None
        and result.get("blockers") == [BUDGET_EXCEEDED_BLOCKER]
        and result.get("expected_provider_upload_bytes") == maximum_archive_bytes
        and record.get("schema_version") == ADMISSION_SCHEMA_VERSION
        and record.get("mode") == MEASURED_MODE
        and record.get("role") == OUTPUT_ROLE
        and record.get("phase") == PREALLOCATION_PHASE
        and record.get("hold_phase") in (PREALLOCATION_PHASE, DEFERRED_HOLD_PHASE)
        and not recovery_withheld(result)
        and record.get("status") == "blocked"
        and record.get("blockers") == [BUDGET_EXCEEDED_BLOCKER]
        and record.get("measurement_path") == str(job_dir)
        and record.get("maximum_archive_bytes") == maximum_archive_bytes
        and type(available) is int
        and 0 <= available < required
    )
    return dict(record) if valid else None


def recorded_output_requirement(
    record: Mapping[str, Any], *, maximum_archive_bytes: int
) -> int | None:
    """The requirement a measured record states, when its own hold and phases give it.

    Capacity recovery re-checks with the admission formula re-derived from the
    record, never with a number it cannot re-derive.
    """

    hold = maximum_archive_bytes + PROVIDER_OUTPUT_OPERATIONAL_RESERVE_BYTES
    phases = record.get("sequential_phases")
    if (
        record.get("hold_bytes") != hold
        or not isinstance(phases, list)
        or not all(
            isinstance(row, Mapping)
            and all(type(row.get(key)) is int and row[key] >= 0
                    for key in ("peak_bytes", "residue_bytes"))
            and type(row.get("shares_output_volume")) is bool
            for row in phases
        )
    ):
        return None
    required = output_volume_requirement(hold, phases)
    return required if record.get("required_available_bytes") == required else None


def measured_admission_record(result: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """The measured pre-allocation record a sealed lane result carries, or None.

    A refusal carries it as ``provider_output_disk_capacity``; a run that got
    past admission nests it under ``before_allocation_and_staging``. None
    means the run was admitted by the ceiling formula.
    """

    capacity = result.get("provider_output_disk_capacity")
    if not isinstance(capacity, Mapping):
        return None
    record = (
        capacity if "schema_version" in capacity
        else capacity.get("before_allocation_and_staging")
    )
    if (
        isinstance(record, Mapping)
        and record.get("schema_version") == ADMISSION_SCHEMA_VERSION
        and record.get("mode") == MEASURED_MODE
    ):
        return record
    return None


def output_role_projection(
    *,
    output_path: Path,
    reservation_root: str | Path,
    required_available_bytes: int,
    disk_usage: Callable[[Path], Any],
) -> dict[str, Any]:
    """Re-measure the output volume exactly as this role's admission will: free
    bytes must cover the bulk floor, every live reservation and the need."""

    projection = _output_projection(
        output_path, reservation_root=reservation_root, disk_usage=disk_usage
    )
    return {
        "role": OUTPUT_ROLE,
        **projection,
        "required_available_bytes": required_available_bytes,
        "required_free_bytes": (
            projection["floor_bytes"] + projection["reserved_bytes"] + required_available_bytes
        ),
    }


__all__ = [
    "ADMISSION_SCHEMA_VERSION",
    "ADMISSION_UNAVAILABLE_BLOCKER",
    "BUDGET_EXCEEDED_BLOCKER",
    "CEILING_MODE",
    "EXTRACTION_BUDGET_EXCEEDED_BLOCKER",
    "EXTRACTION_SCHEMA_VERSION",
    "MEASURED_MODE",
    "MODE_INVALID_BLOCKER",
    "OUTPUT_ADMISSION_ENV",
    "OUTPUT_ROLE",
    "SceneConfigurationOutputAdmission",
    "SceneConfigurationOutputHoldRefused",
    "configured_output_admission_mode",
    "cpu_prefix_peak_bytes",
    "cpu_prefix_spend",
    "extraction_requirement",
    "measured_admission_record",
    "open_scene_configuration_output_admission",
    "output_role_projection",
    "output_volume_requirement",
    "recorded_output_requirement",
    "bundle_unpacked_bytes",
    "recorded_preallocation_refusal",
    "recovery_withheld",
    "release_scene_configuration_output",
    "releases_output_on_exit",
]
