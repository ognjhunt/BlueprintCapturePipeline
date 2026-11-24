"""Join prepared website captures to the scene their Atlas job reconstructed.

ADP-030/040, day 28: retain the task placement and authoring inputs as soon as
the Atlas job finishes. A held job or missing execution authority must not
discard completed preparation or label the scene simulator-ready.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from .core.handoff_lease import (
    HandoffLeaseError, guarded_handoff_write_json as write_json, guarded_handoff_write_track_json as write_track_json,
    handoff_effect, handoff_lease_checkpoint,
)
from .decision_evidence_contracts import canonical_digest, cross_runtime_canonical_digest
from .website_task_preparation import (
    GEOMETRY_ONLY_BLOCKER, compile_website_scene_preparation, native_task_preparation_gap,
    native_task_subject_gap, native_task_subject_profile, subject_rigidity,
)
from .website_scene_runtime_inputs import prepare_website_runtime_inputs


def execute_prepared_task_family(*, preparation_path: Path, runtime: Any, policy: Any,
                                output_dir: Path, episode_id: str,
                                maximum_steps: int, control_frequency_hz: float) -> dict[str, Any]:
    """Consume the compiled family route using an already admitted native runtime.

    The operator transfers runtime ownership through its existing execution
    lane. Every preflight refusal closes it; the episode executor owns closure
    once called. This function buys nothing and cannot admit a provider.
    Runtime readback must bind the exact compiled scene, not only its engine.
    """
    executor_owns_runtime = False
    try:
        # ``source_targets`` carries every target's full-video source track: read and hashed compact.
        from .website_track_mask_codec import canonical_track_digest, load_track_json
        preparation = load_track_json(preparation_path)
        if (preparation.get("digest") != canonical_track_digest(preparation, digest_field="digest")
                or preparation.get("status") not in {"task_evaluation_ready", "intake_ready"} or preparation.get("blockers") != []):
            raise ValueError("website_task_family_preparation_not_ready")
        inputs = preparation["task_evaluation_inputs"]
        from .task_evaluation_task_family_native_adapter import (
            execute_task_family_episode, validate_task_family_native_runtime_contract,
        )
        native = validate_task_family_native_runtime_contract(inputs["native_runtime_contract"])
        from .task_family_native_contracts import family_source_blockers
        if family_source_blockers(native=native, source_bindings=inputs["source_bindings"], source_scene=inputs["source_scene"]):
            raise ValueError("website_task_family_source_binding_incomplete")
        if (getattr(runtime, "native_runtime_contract_digest", None) != native["contract_digest"]
                or native["task_spec"]["task_success_contract"] != inputs["task_success_contract"]
                or inputs["scope"]["task_context_digest"] != preparation["binding"]["task_context_digest"]):
            raise ValueError("website_task_family_runtime_scene_mismatch")
        from .local_reconstruction_adapters import _sha256_file
        source = inputs["source_scene"]
        for key in ("splat", "collision_mesh"):
            if _sha256_file(Path(source[key + "_path"])) != source[key + "_digest"]:
                raise ValueError("website_task_family_source_scene_changed")
        executor_owns_runtime = True
        return execute_task_family_episode(runtime=runtime, policy=policy, contract=inputs["task_success_contract"],
            native_runtime_contract=native,
            maximum_steps=maximum_steps, control_frequency_hz=control_frequency_hz,
            output_dir=output_dir, episode_id=episode_id, seed=native["seed"])
    finally:
        if not executor_owns_runtime:
            runtime.close()


def _robot_fit_table(*, preparation: Mapping[str, Any], verdict: Mapping[str, Any], collision_path: Path,
                     native_root: Path) -> dict[str, Any]:
    """Which catalog robot models fit the job by spec sheet and, for the arms that have one, their own
    kinematic chain (``website_robot_fit``), beside the verdict.

    Informational: it protects no privacy, rights or money, so it fails open. A failure is recorded
    and the handoff goes on with its gating unchanged; only a lost handoff lease, which stops every
    write, stops it here too.
    """
    try:
        from .robot_arm_chains import load_robot_arm_chains
        from .robot_spec_catalog import load_robot_spec_catalog
        from .website_robot_fit import FILENAME, robot_fit_table
        table = robot_fit_table(preparation=preparation, collision_path=collision_path, verdict=verdict,
                                catalog=load_robot_spec_catalog(), arm_chains=load_robot_arm_chains())
        write_json(native_root / FILENAME, table)
        return {"path": str(native_root / FILENAME), "digest": table["table_digest"], "counts": table["counts"]}
    except HandoffLeaseError:
        raise
    except Exception as exc:  # noqa: BLE001 - only privacy, rights and money stop a job; this is recorded
        return {"status": "failed", "error": type(exc).__name__}


def _rigid_verdict_and_fit(*, preparation: Mapping[str, Any], collision_path: Path,
                           native_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """The capability verdict (written beside the native inputs) and the robot fit table, CPU only.

    One read scope holds both: a USDA text collider (one normalized before 2026-11-03) is read once, into
    a binary copy every search of both opens, where each search parsed it again at about 3.6 GB
    (``usd_binary_read_cache``). A binary collider is opened as it is.
    """
    from .usd_binary_read_cache import read_scope
    from .website_rigid_feasibility import rigid_task_verdict

    with read_scope():
        verdict = rigid_task_verdict(preparation=preparation, collision_path=collision_path)
        write_json(native_root / "capability_verdict.json", verdict)
        return verdict, _robot_fit_table(preparation=preparation, verdict=verdict, native_root=native_root,
                                         collision_path=collision_path)


def geometry_only_fit_check(*, preparation: Mapping[str, Any], base_scene: Mapping[str, Any],
                            native_root: Path) -> dict[str, Any]:
    """Where a geometry-only job ends: the capability verdict and the robot fit table, CPU only.

    The scene keeps its captured subject (``website_atlas_pipeline.ROUTE_ENV``), its box cut
    from the collider at compile (``website_subject_excision``), so the job is never
    registered or enqueued: nothing paid follows. Only a rigid pick-and-place preparation
    held for nothing but this check, with no uncut subject, is assessed, as the full route
    assesses only an intake-ready one; anything else is recorded as not run.
    """
    if list(preparation.get("blockers") or []) != [GEOMETRY_ONLY_BLOCKER]:
        return {"status": "not_run", "reason": "preparation_blocked"}
    if preparation["intake_request"]["task"]["strategy"] != "pick_and_place":
        return {"status": "not_run", "reason": "task_strategy_not_modelled"}
    if (preparation["subject"].get("geometry_origin") != "synthetic_design"
            and (preparation.get("subject_excision") or {}).get("status") != "excised"):
        # Every robot check would meet the subject itself.
        return {"status": "not_run", "reason": "subject_not_excised"}
    from .local_reconstruction_adapters import _sha256_file
    from .website_scene_runtime_inputs import normalize_runtime_collision
    if (_sha256_file(Path(base_scene["collision_mesh_path"])) != base_scene["collision_mesh_digest"]
            or base_scene["collision_mesh_digest"] != preparation["binding"]["collision_mesh_digest"]):
        raise ValueError("website_native_background_changed")
    collision_root, normalized = normalize_runtime_collision(preparation=preparation, base_scene=base_scene,
                                                             output_root=native_root)
    collision = collision_root / normalized["output"]["relative_path"]
    verdict, fit = _rigid_verdict_and_fit(preparation=preparation, collision_path=collision, native_root=native_root)
    return {"status": "completed",
            "collision": {"path": str(collision), "digest": normalized["output"]["sha256"]},
            "capability_verdict": {"status": verdict["status"], "blockers": verdict["blockers"],
                                   **({"risks": verdict["risks"]} if "risks" in verdict else {}),
                                   "path": str(native_root / "capability_verdict.json"),
                                   "digest": verdict["verdict_digest"]},
            "robot_fit_table": fit}


def name_anthropic_for_site(rights: Mapping[str, Any], capture_root: Path | None) -> bool:
    """Before a rights admission that lets frames and task evidence go to Anthropic is written,
    name Anthropic in the site's own withdrawal ledger (the folder the withdrawal planner reads)."""
    if "anthropic_provider_terms_reference" not in rights:
        return False
    from .website_capture_withdrawal import _site
    from .website_capture_withdrawal_provider_copies import record_provider_admission
    if capture_root is None:
        raise ValueError("website_anthropic_admission_site_unknown")
    record_provider_admission(_site(Path(capture_root)), "anthropic")
    return True


def artisan_provider_choice_for(*, authority: Mapping[str, Any], root: Path, capture_root: Path,
                                geometry_only: bool = False) -> dict[str, Any] | None:
    """The host's once-per-grant choice of ARTISAN's text provider, or ``None`` (today's path).

    Only on a host set to prefer Anthropic, for a both-providers grant of an ARTISAN job; every
    other job runs today's path untouched, writing nothing. When Anthropic is chosen, the site's
    withdrawal ledger names it before the choice is returned, so no site text or frame reaches
    Anthropic before that entry exists.
    """
    import os

    from .artisan_provider_choice import PREFER_ANTHROPIC_ENV, both_providers_granted, retained_or_new_choice
    from .task_object_authoring_backend import ARTISAN_BACKEND, default_backend
    if (os.environ.get(PREFER_ANTHROPIC_ENV) != "1" or geometry_only or not both_providers_granted(authority)
            or (authority.get("replacement_authoring_backend") or default_backend()) != ARTISAN_BACKEND):
        return None
    choice = retained_or_new_choice(root=root, grant=authority)
    if choice["provider"] == "anthropic":
        from .website_capture_withdrawal import _site
        from .website_capture_withdrawal_provider_copies import record_provider_admission
        record_provider_admission(_site(Path(capture_root)), "anthropic")
    return choice


def prepare_rigid_native_inputs(*, result: dict[str, Any], preparation: Mapping[str, Any], preparation_path: Path,
                                native_root: Path, task_context: Mapping[str, Any], now: float,
                                runtime_inputs: Mapping[str, Any] | None = None,
                                base_scene: Mapping[str, Any] | None = None,
                                source_geometry: Mapping[str, Any] | None = None,
                                task_masks: Mapping[str, Any] | None = None,
                                capture_root: Path | None = None) -> bool:
    """The rigid native tail of a compiled preparation, CPU only: it buys nothing.

    Runtime inputs (unless the caller already made them), the collision and
    appearance stages, the construction inputs, the rights admission and the
    capability verdict, with the informational robot fit table beside it. The
    handoff and a renewal re-issue
    (``website_preparation_renewal``) share it, so the inputs bound to a
    preparation's digest cannot drift between them. Records into ``result``
    and returns whether the preparation may be registered: intake ready and
    appearance authored. The capability verdict and the support record are
    recorded and never hold the job.
    """
    if runtime_inputs is None:
        runtime_inputs = prepare_website_runtime_inputs(preparation=preparation, base_scene=base_scene,
            source_geometry=source_geometry, task_masks=task_masks, output_root=native_root)
    result["runtime_inputs"] = {"status": runtime_inputs["status"], "digest": runtime_inputs["digest"],
                                "path": str(native_root / "runtime_inputs.json")}
    pad = (runtime_inputs.get("collision") or {}).get("support_pad")
    support_hold = None
    if isinstance(pad, Mapping):
        from .website_support_pad import hold_reason
        support_hold = hold_reason(pad)
        result["task_support_pad"] = {key: pad[key] for key in ("status", "reason", "scene_revision_collider")
                                      if key in pad}
        result["task_support_pad"]["hold_reason"] = support_hold
    from .website_native_background import prepare_collision_stage, prepare_appearance_stage, prepare_construction_stages
    collision_stage = prepare_collision_stage(native_root / "runtime_inputs.json")
    write_json(native_root / "collision_stage_inputs.json", collision_stage)
    result["runtime_inputs"]["collision_stage_inputs_path"] = str(native_root / "collision_stage_inputs.json")
    if runtime_inputs["appearance"]["status"] != "native_appearance_authored":
        return False
    appearance_stage = prepare_appearance_stage(native_root / "runtime_inputs.json")
    write_json(native_root / "appearance_stage_inputs.json", appearance_stage)
    result["runtime_inputs"]["appearance_stage_inputs_path"] = str(native_root / "appearance_stage_inputs.json")
    construction = prepare_construction_stages(runtime_inputs_path=native_root / "runtime_inputs.json",
                                                preparation_path=preparation_path)
    if preparation["status"] == "intake_ready":
        from .website_native_background import construction_rights_admission
        from .website_object_observations import _record
        rights = construction_rights_admission(preparation=preparation, task_context=task_context, now=now)
        name_anthropic_for_site(rights, capture_root)
        rights_path = native_root / "rights_admission.json"
        write_json(rights_path, rights)
        construction["references"].append({"contract_path": "scene.rights.admission", **_record(rights_path)})
    write_json(native_root / "construction_inputs.json", construction)
    result["runtime_inputs"]["construction_inputs_path"] = str(native_root / "construction_inputs.json")
    if preparation["status"] == "intake_ready":
        from .usd_binary_read_cache import read_scope
        collision = Path(runtime_inputs["collision"]["path"])
        verdict = None
        # One read scope for every search below: a USDA text collider is read once (``usd_binary_read_cache``).
        with read_scope():
            # A job whose rule is withheld has no destination the site described: no verdict, no fit table.
            withheld = "rule_withheld" in result or sealed_rule_withheld(capture_root) is not None
            if preparation["intake_request"]["task"]["strategy"] == "pick_and_place" and not withheld:
                # Abstain before the owner is asked to pay when no enabled embodiment can carry the
                # task; one the gate cannot model is unassessed and never refused.
                verdict, fit = _rigid_verdict_and_fit(preparation=preparation, collision_path=collision,
                                                      native_root=native_root)
                result["capability_verdict"] = {"status": verdict["status"], "blockers": verdict["blockers"],
                    **({"risks": verdict["risks"]} if "risks" in verdict else {}),
                    "path": str(native_root / "capability_verdict.json"), "digest": verdict["verdict_digest"]}
                result["robot_fit_table"] = fit
                # An abstained verdict (no enabled arm, no clear stance found, or the object beyond reach
                # from every stance it found) is a judgement about the outcome: recorded, never a stop. It
                # travels as ``scene_estimate:capability_unassessed`` (``atlas_scene_estimates``).
        # A support on which the object cannot rest everywhere it is counted as placed is recorded, never a
        # stop (owner, 2026-11-16): the package says so (``website_company_simulation``), and the paid
        # robot-free probe still refuses it (``rigid_support_pad.half_support_hold``).
    return preparation["status"] == "intake_ready"


#: Where a run keeps the records a base scene from an earlier Atlas result named (``retain_superseded_records``).
SUPERSEDED_DIRECTORY = "superseded_atlas_results"
#: The workspace and its directories whose records each run rewrites in place; what they name is content-addressed.
_RECORD_DIRECTORIES = ("", "native", "fit_check", "development_test")


def retain_superseded_records(*, capture_root: Path, root: Path, atlas_result_digest: Any,
                              levelling_digest: Any = None) -> Path | None:
    """Copy the records a base scene from another Atlas result named, before this run replaces them.

    The levelled scene, the navigation scene half and every file they name sit in content-addressed
    directories, and a new Atlas result writes new ones beside them. The records naming them (the handoff,
    the base scene, the compiled specs, the native inputs) sit at fixed paths each run rewrites. When the
    base scene this run replaces came from another result (the job's Atlas result was published again,
    ``website_held_handoff_replay``), each is copied once, as it is, under
    ``superseded_atlas_results/<that result>/``, so evidence made on the old scene still verifies against
    the records it named. Nothing is deleted. Returns that directory, or None when nothing was superseded.
    """
    import os
    import re
    import shutil

    try:
        previous = json.loads((root / "base_scene.json").read_text())
    except (OSError, ValueError):
        return None
    built = previous.get("atlas_result_digest") if isinstance(previous, dict) else None
    levelled = (previous.get("levelling") or {}).get("digest") if isinstance(previous, dict) else None
    if not isinstance(built, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", built):
        return None
    preparation = capture_root / "pipeline" / "website_scene_preparation"
    if built != atlas_result_digest:
        archive = preparation / SUPERSEDED_DIRECTORY / built[7:23]
    elif (levelling_digest is not None and isinstance(levelled, str)
          and re.fullmatch(r"sha256:[0-9a-f]{64}", levelled) and levelled != levelling_digest):
        # The same Atlas result levelled again by a newer levelling revision (``superseded_levelling``).
        archive = preparation / SUPERSEDED_DIRECTORY / f"{built[7:23]}-levelling-{levelled[7:23]}"
    else:
        return None
    records = [preparation / "handoff.json", *sorted(path for name in _RECORD_DIRECTORIES
                                                     for path in (root / name).glob("*"))]
    for path in dict.fromkeys(records):
        target = archive / path.relative_to(preparation)
        if path.is_symlink() or not path.is_file() or target.exists():
            continue
        # A whole copy or none: a crash leaves only the staging name, and the next run copies again.
        staging = target.with_name(target.name + ".partial")
        with handoff_effect("website_scene_handoff_superseded_records"):
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, staging)
            os.replace(staging, target)
    return archive


#: Owner direction 2026-11-16: any job is accepted; what is missing is estimated and flagged, and the job keeps
#: going. A job the robot does by moving itself (navigation), and one Blueprint has no task profile, no installed
#: runtime or no one-object model for, gets the room package: the room as reconstructed, the job as the site wrote
#: it and, for navigation, the goal region with an assumed rule. The company builder reads it from the handoff.
ROOM_PACKAGE = "room_package_written"
ROOM_RECORD = "room_package_inputs.v1.json"
#: A routed job's room record, the package floor while its object is not built (``website_company_simulation``).
ROOM_FLOOR, ROOM_FLOOR_FLAG = "room_floor", "object_not_built"
#: Beside the floor's half: when this floor (by its record's digest) was first written. A handoff that runs again
#: writes the same record and keeps it, so the wait before an unstarted object path is listed never restarts.
ROOM_FLOOR_FIRST_WRITTEN = "first_written.json"
#: On the handoff's room floor of a preparation stopped on a blocker that is no wall (``floor_ships``): its blockers.
#: The company pass ships that floor at once, flagged with them, while the job keeps retrying (owner, 2026-11-23).
ROOM_FLOOR_BLOCKERS = "blockers"


def write_room_floor(*, now: float, output_root: Path, **room: Any) -> dict[str, Any]:
    """The room floor's record (``write_room_package``), and the time a record with its digest was first written."""
    record = write_room_package(output_root=output_root, **room)
    first = output_root / ROOM_FLOOR_FIRST_WRITTEN
    try:
        kept = json.loads(first.read_text())
    except (OSError, ValueError):
        kept = None
    if not isinstance(kept, dict) or kept.get("inputs_digest") != record["inputs_digest"]:
        write_json(first, {"inputs_digest": record["inputs_digest"], "first_written_epoch": int(now)})
    return record


#: The navigation rule's two assumed numbers: how near the goal centre the robot's base must stop, and for how long.
GOAL_RADIUS_M, GOAL_DWELL_SECONDS = 0.5, 2.0
#: The handoff gaps that mean "no task profile": ``website_task_preparation.profile_preparation_gap`` and an operation
#: reading naming no family (``website_operation_coverage``).
ROOM_GAPS = frozenset({"website_native_task_profile_preparation_required", "website_task_operation_coverage_unresolved"})
#: The other "no model for this job" findings that ship the room package, each flagged on its record by its code.
MISMATCH = "website_task_profile_mechanism_mismatch"
#: The video analysis found nothing the task is about (``clean_plate_stage``): the room as filmed, flagged.
NOT_FOUND = "task_relevant_target_not_observed"
#: The analysis found the job's object but the clean plate could not outline it (``clean_plate_stage.NOT_LOCATED``).
NOT_LOCATED = "task_target_not_located"
#: The clean plate's flags that end the job as the room package, as filmed.
ROOM_AS_FILMED = (NOT_FOUND, NOT_LOCATED)
RUNTIME_GAP = "website_task_family_runtime_preparation_required"
#: Owner direction 2026-11-17: accept every job, then decide what is needed. A preparation blocker that is no wall
#: ships the room package, flagged, and buys nothing more (``goes_on``). Since 2026-11-22 only codes that protect
#: privacy or money keep the job waiting for good; any other bad record or input ships, flagged, once its first
#: retry cycle is spent, like every other blocker that is no wall.
KEEP_HOLDING = frozenset()
#: Read from the codes' own definitions. Privacy: a digest or binding that changed or does not match, or a path
#: outside the capture, may be another job's or capture's record; a step still pending covers the clean plate's
#: privacy check (``website_scene_preparation_pending``). Money: a receipt that is invalid or missing, a paid
#: result that is missing, a call that needs reconciling or a settlement that is invalid is spend not yet proven.
#: Every such code on the preparation path ends in one of these (``tests/test_website_scene_handoff``, which reads
#: them from the source).
KEEP_HOLDING_SUFFIXES = ("_changed", "_mismatch", "_binding_invalid", "_outside_capture", "_receipt_invalid", "_receipts_invalid", "_receipt_missing",
                         "_result_missing", "_requires_reconciliation", "_settlement_invalid", "_pending")
KEEP_HOLDING_PREFIXES: tuple[str, ...] = ()


#: Written beside the handoff just before the prepared scene is sent to the website (``prepared-scene``), whose
#: outbox entry is forwarded into paid preparation, and settled by the answer: ``sent`` (a receipt), ``refused`` (a
#: typed answer that commits nothing), or left ``sending`` (no answer, or one that may have committed). An earlier
#: release's receipt on the handoff is kept here too (``keep_outbox_receipt``), so held runs never lose it.
OUTBOX_MARKER = "prepared_scene_outbox.json"
#: The website's typed refusals of the write that commit nothing: a 4xx other than a conflict with an entry.
# Unlike other conflicts, this exact prepared-scene refusal is returned before an outbox insert.
_PREPARED_SCENE_CAP_REFUSAL = "website_control_prepared-scene_http_409:website_scene_preparation_budget_exhausted"
_REFUSED_WRITE = re.compile(r"website_control_prepared-scene_http_4(?!09)[0-9]{2}(:|$)")


def _outbox_marker(capture_root: Path) -> Path:
    return Path(capture_root) / "pipeline" / "website_scene_preparation" / OUTBOX_MARKER


def keep_outbox_receipt(capture_root: Path) -> None:
    """An earlier handoff's prepared-scene receipt, kept as the marker before this run rewrites the handoff."""
    preparation = Path(capture_root) / "pipeline" / "website_scene_preparation"
    try:
        earlier = json.loads((preparation / "handoff.json").read_text())
    except (OSError, ValueError):
        return
    receipt = earlier.get("website_intake_outbox") if isinstance(earlier, dict) else None
    if isinstance(receipt, Mapping) and not _outbox_marker(capture_root).exists():
        write_json(_outbox_marker(capture_root), {"state": "sent", "basis": "earlier_handoff_receipt",
                                                  "outbox_id": receipt.get("id"),
                                                  "request_digest": receipt.get("request_digest"),
                                                  "task_context_digest": None})


def _unsettled_outbox_evidence(capture_root: Path) -> dict[str, Any] | None:
    """The marker as it stands before a new send, unless it is absent or settled ``refused``: an earlier send that may
    have committed (``sending``), one that did (``sent``), or a marker that cannot be read."""
    try:
        prior = json.loads(_outbox_marker(capture_root).read_text())
    except FileNotFoundError:
        return None
    except (OSError, ValueError):
        return {"state": "unreadable", "task_context_digest": None}
    if not isinstance(prior, dict):
        return {"state": "unreadable", "task_context_digest": None}
    return None if prior.get("state") == "refused" else prior


def prepared_scene_may_exist(capture_root: Path, *, task_context_digest: str | None = None) -> bool:
    """Whether a prepared-scene outbox entry for this job may exist: a marker for this job's task context (or one
    that names none) that is not settled ``refused``."""
    try:
        marker = json.loads(_outbox_marker(capture_root).read_text())
    except FileNotFoundError:
        return False
    except (OSError, ValueError):
        return True  # unreadable: it may exist
    if not isinstance(marker, dict):
        return True
    if marker.get("state") == "refused":
        return False
    recorded = marker.get("task_context_digest")
    return recorded is None or task_context_digest is None or recorded == task_context_digest


def keeps_holding(code: str) -> bool:
    """Whether the room package would be wrong for ``code`` (``KEEP_HOLDING``), judged on every ``:``-separated
    segment: a website refusal carries its own code after its operation
    (``website_control_preparation-spend_http_409:website_scene_sponsorship_binding_invalid``)."""
    return any(part in KEEP_HOLDING or part.endswith(KEEP_HOLDING_SUFFIXES) or part.startswith(KEEP_HOLDING_PREFIXES)
               for part in code.split(":"))


#: Raised codes another run clears by itself (it holds the same paid call), waited on like an outage.
TRANSIENT = frozenset({"website_gemini_request_in_progress"})
#: A typed blocker: a refusal raised under its own code (``word_word[:detail]``, the website's typed answers
#: ``website_control_<operation>_http_<status>:<code>`` among them). Anything else raised (a KeyError, a
#: TypeError, an untyped message) is Blueprint's defect, not a fact about the job.
_TYPED = re.compile(r"[a-z][a-z0-9-]*(?:_[a-z0-9-]+)+(?::|$)")
#: A defect is recorded under this code with its exception type (``blocker_code``). It never ships at once: the
#: room package is final, and a release that fixes the defect must still find the job. It waits like an outage,
#: the owner is paged (``preparation_fault_alert``), and at the window's end it ships, the code named.
DEFECT = "website_preparation_defect"
#: Codes whose "budget" counts tokens or bytes of one provider request (``authoring_frame_budget``), never money.
REQUEST_SIZE_CODES = ("authoring_frame_bytes_exceed_budget:", "authoring_frames_token_budget_exceeded")
#: Blockers that leave out Blueprint's success rule rather than the object or its destination.
RULE_CODES = frozenset({"website_owner_success_criteria_translation_required",
                        # The job named a place to put something, but it is one that opens and closes in place.
                        "website_articulated_task_has_no_destination"})
#: For a job with a grant, the longest wait between two retries of an outage or a money wait
#: (``website_gemini_receipts.MONEY_RECHECK_SECONDS``): the last run before the window ends falls within twice it.
LAST_CALL_SECONDS = 2 * 3600


def piece_of(code: str) -> str:
    """The piece of the job a blocker leaves out, as the room package's README names it."""
    head = code.split(":", 1)[0]
    return "success rule" if head in RULE_CODES else "destination" if "destination" in head else "object"


def ships_at_once(code: str) -> bool:
    """Whether ``code`` is a fact about the job's own scene or wording that compile found
    (``website_task_preparation.SCENE_FACT_BLOCKERS``), or the end of a provider's bound of unanswered sends, which
    no later run can change (``website_gemini_receipts.EXHAUSTED``, #3636), or a fit or facing stop found once the
    room was built (``website_recapture_required.ROOM_BUILT_CODES``), which the same video always repeats. Judged on its head: a coverage blocker
    may name its target after a ``:``."""
    from .website_gemini_receipts import EXHAUSTED
    from .website_recapture_required import ROOM_BUILT_CODES
    from .website_task_preparation import SCENE_FACT_BLOCKERS
    head = code.split(":", 1)[0]
    return head in SCENE_FACT_BLOCKERS or head in ROOM_BUILT_CODES or code == EXHAUSTED


def defect(error: BaseException) -> bool:
    """Whether ``error`` is Blueprint's defect: raised under no typed code, and no outage either."""
    from .website_preparation_retry import OUTAGE, classify
    return not ((isinstance(error, ValueError) and _TYPED.match(str(error))) or str(error) in TRANSIENT
                or classify(f"{type(error).__name__}: {error}") == OUTAGE)


def spending_cap_code(code: str) -> bool:
    """Whether ``code`` is the website's refusal at the job's spending cap (``hard_stops.JOB_SPEND_CAP``), bare or
    as a website control answer (``website_control_<operation>_http_409:<code>``)."""
    from .hard_stops import JOB_SPEND_CAP
    return code == JOB_SPEND_CAP or code.endswith(f"_http_409:{JOB_SPEND_CAP}")


def shipped_code(code: str) -> str:
    """The code a room package ships for: the plain ``spending_cap_reached`` at the cap, else ``code``."""
    from .website_job_spend_cap import SPENDING_CAP_REACHED
    return SPENDING_CAP_REACHED if spending_cap_code(code) else code


def blocker_code(error: BaseException) -> str:
    """The code a raised failure is recorded under: its own, or ``DEFECT`` with its type."""
    return f"{DEFECT}:{type(error).__name__}" if defect(error) else str(error)


def first_retry_cycle_spent(capture_root: Path, now: float) -> bool:
    """Whether the attempt now running is the last of the job's first retry cycle (``job_retry_policy``): its ledger
    shows at least as many failed attempts in a row as the policy has timed retries, the first of them failed at least
    the whole schedule (about 46 hours) ago. An outage, retried more often, reaches it after as long. An unreadable
    ledger or time: no (the window's end still ships the job)."""
    from .handoff_job_state import _read_job_ledger
    from .job_retry_policy import REFUSAL_RETRY_DELAYS_SECONDS
    from .website_preparation_retry import consecutive_failures
    from .website_preparation_terminal import _parse_utc
    history = _read_job_ledger(Path(capture_root)).get("attempt_history")
    if type(history) is not list:
        return False
    failures = consecutive_failures(history)
    if failures < len(REFUSAL_RETRY_DELAYS_SECONDS):
        return False
    first = history[-failures]
    ended = _parse_utc(first.get("failed_at")) if isinstance(first, Mapping) else None
    return ended is not None and now - ended.timestamp() >= sum(REFUSAL_RETRY_DELAYS_SECONDS)


def goes_on(codes: Sequence[str], *, task_context: Mapping[str, Any], now: float,
            window_end: Any = lambda: None, error: BaseException | None = None,
            retries_spent: Any = lambda: False) -> bool:
    """Whether a preparation stopped on ``codes`` ships the room package (owner, 2026-11-17), or keeps waiting.

    At the job's spending cap it never holds (owner, 2026-11-19): the refusal already stopped the purchase, so the
    job ships what it has, flagged ``spending_cap_reached`` (``shipped_code``), and waits for no one. Otherwise
    it holds on a wall (``hard_stops.is_wall``, or a code the retry policy reads as privacy, rights or money, the
    authoring admissions among them, or as the owner's closing of the run: ``website_preparation_retry.classify``),
    on the job's sentence still to come, on a stop only a new video clears (``website_recapture_required``, a fit
    or facing stop found once the room was built excepted: that one ships at once), on a
    code the room package would be wrong for (``keeps_holding``, any segment), and in a development test, which
    places the object on its own surface. Only facts about the job's own scene or wording ship at once
    (``ships_at_once``). Everything else, an unknown code included (a configuration fault, a missing key or
    profile, a website refusal, an outage, a defect), ships once this run is the last of its first retry cycle
    (``retries_spent()``: the six timed retries of ``job_retry_policy``), or once the job's window, read when
    asked (``window_end()``), ends within ``LAST_CALL_SECONDS``; until then a fix or the next run can still
    prepare the job. A later fix rebuilds it (``website_held_handoff_replay``). Faults ours to fix are paged (``preparation_fault_alert``). A code
    whose "budget" is a request's size (``REQUEST_SIZE_CODES``) is not read as money.
    """
    from .hard_stops import is_wall
    from .job_retry_policy import MONEY, PRIVACY_RIGHTS
    from .website_development_test import enabled
    from .website_preparation_retry import AWAITING_JOB_SENTENCE, OWNER_CLOSED, classify
    from .website_recapture_required import CODES, ROOM_BUILT_CODES
    RECAPTURE_CODES = CODES - ROOM_BUILT_CODES  # only these still need a new video: no room was built for them
    codes = [str(code) for code in codes]
    at_cap = bool(codes) and all(spending_cap_code(code) for code in codes)
    if not at_cap and (not codes or (error is not None and is_wall(error))
            or any(not code.startswith(REQUEST_SIZE_CODES)
                   and classify(code) in {PRIVACY_RIGHTS, MONEY, OWNER_CLOSED, AWAITING_JOB_SENTENCE} for code in codes)
            or any(keeps_holding(code) or any(part in RECAPTURE_CODES for part in code.split(":"))
                   for code in codes)):
        return False
    try:
        if enabled(task_context["context_digest"]):
            return False
    except (ValueError, KeyError):
        return False  # an unreadable test allowlist: the preparation is kept as before
    if at_cap:
        return True
    if all(ships_at_once(code) for code in codes) and (error is None or not defect(error)):
        return True
    if retries_spent():
        return True
    window = window_end()
    return window is not None and window - now <= LAST_CALL_SECONDS


def floor_ships(codes: Sequence[str], *, task_context: Mapping[str, Any], error: BaseException | None = None) -> bool:
    """Whether a preparation stopped on ``codes``, its room built, ships its room floor flagged now (owner,
    2026-11-23: "once the room is built and the sharing rights hold, the room floor ships flagged immediately;
    retries continue, and a better package replaces it"): exactly when its first retry cycle, once spent, would
    ship the room package (``goes_on``), so a wall, a privacy or money hold or a development test never does."""
    return goes_on(codes, task_context=task_context, now=0.0, error=error, retries_spent=lambda: True)


def sealed_rule_withheld(capture_root: Path | None) -> dict[str, Any] | None:
    """Why the capture's sealed handoff withheld Blueprint's rule (``rule_withheld``), or None."""
    try:
        handoff = json.loads((Path(capture_root) / "pipeline/website_scene_preparation/handoff.json").read_text())
    except (OSError, ValueError, TypeError):
        return None
    withheld = handoff.get("rule_withheld") if isinstance(handoff, dict) else None
    if not isinstance(withheld, dict) or handoff.get("digest") != canonical_digest(handoff, digest_field="digest"):
        return None
    return dict(withheld)


def _sim_frame(*, base_scene: Mapping[str, Any], source_geometry: Mapping[str, Any],
               focus: Mapping[str, Any] | None) -> tuple[float, Any, Any]:
    """Metres per runtime unit, runtime-to-simulator (Z up) and the source-to-runtime registration (or None).

    As compile frames a scene (``compile_website_scene_preparation``): the base scene's declared scale, else the
    registration's. Registration runs only when a goal must be placed or no scale is declared; a goal that does not
    register is left out (None), a scene with neither scale is refused.
    """
    import numpy as np

    from .website_task_preparation import _UP_INDEX, register_source_to_runtime
    mpu = base_scene.get("meters_per_unit")
    matrix = None
    if focus is not None or mpu is None:
        anchor = base_scene.get("anchor")
        anchor = ({**anchor, "meters_per_unit": mpu, "up_axis": base_scene["up_axis"],
                   "ground_plane_offset_m": base_scene.get("ground_plane_offset_m")}
                  if isinstance(anchor, Mapping) and mpu is not None else None)
        try:
            registration = register_source_to_runtime(
                source_geometry=source_geometry, collision_mesh_path=Path(base_scene["collision_mesh_path"]),
                anchor=anchor, focus_bounds=focus, up_axis=base_scene["up_axis"])
        except ValueError:
            if mpu is None:
                raise
        else:
            matrix = np.asarray(registration["source_to_runtime"])
            mpu = 1.0 / registration["scale"] if mpu is None else mpu
    up_sign = -1 if base_scene["up_axis"] == "-Y" else 1
    runtime_to_sim = np.eye(4)
    runtime_to_sim[:3, :3] = float(mpu) * (np.array([[1, 0, 0], [0, 0, -up_sign], [0, up_sign, 0]])
                                           if _UP_INDEX[base_scene["up_axis"]] == 1 else np.eye(3))
    return float(mpu), runtime_to_sim, matrix


def room_flags(flags: Sequence[str], clean_plate: Mapping[str, Any]) -> list[str]:
    """A room package's flags: the caller's, then the clean plate's room-as-filmed flags (``ROOM_AS_FILMED``)
    it raised; in that order, each once."""
    found = [code for code in ROOM_AS_FILMED if code in (clean_plate.get("flags") or [])]
    return list(dict.fromkeys([*flags, *found]))


def planner_removals_applied(capture_root: Path) -> int:
    """How many objects the capture's Atlas job took out of the room scan as its planner asked: its sealed
    decision (``website_atlas_pipeline.JOB_OBJECT_RECORD``) did not apply the as-filmed rule. Else 0."""
    from .website_atlas_pipeline import JOB_OBJECT_RECORD
    try:
        decided = json.loads((Path(capture_root) / "pipeline" / "atlas" / JOB_OBJECT_RECORD).read_text())
    except (OSError, ValueError):
        return 0
    if (not isinstance(decided, dict) or decided.get("applied") is not False
            or decided.get("digest") != canonical_digest(decided, digest_field="digest")):
        return 0
    return int(decided.get("planner_remove_count") or 0)


def write_room_package(*, task_context: Mapping[str, Any], base_scene: Mapping[str, Any],
                       source_geometry: Mapping[str, Any], removal: Mapping[str, Any], output_root: Path,
                       task_profile: str | None, task_masks: Mapping[str, Any] | None = None,
                       objects_in_scan: bool = False, flags: Sequence[str] = (), taken_out: int = 0,
                       not_prepared: Sequence[Mapping[str, str]] = ()) -> dict[str, Any]:
    """The room package's half (``scene-package/``: the room's collision and look, the plan) and its sealed record.

    CPU only, buys nothing. A navigation job (``task_masks`` given) whose reading names exactly one goal mask gets
    a goal region: that mask's visible bounds lifted onto the scene by the registration compile uses, its footprint
    centred at its lowest point (the floor it stands on), and the rule "stop within ``GOAL_RADIUS_M`` of it for
    ``GOAL_DWELL_SECONDS``", both numbers assumed. Any other job ships no rule of Blueprint's: the record says so,
    and the site's words stay on the job's page, the one place they are shown redacted. The job's objects are counted, never placed: taken out of the scan by the
    clean plate, or part of it on a geometry-only route; ``taken_out`` counts those the Atlas job's planner took out
    when the clean plate removed none (``planner_removals_applied``).
    """
    import shutil

    import numpy as np

    from .local_reconstruction_adapters import _sha256_file
    from .scene_package_stage import placements
    from .team_scene_robot_absent import DIRECTORY, PLAN_FILENAME, PLAN_SCHEMA, PROBE_FIELD, PROBE_NOT_QUEUED
    from .website_native_appearance import prepare_native_appearance
    from .website_scene_runtime_inputs import normalize_runtime_collision
    from .website_task_preparation import _runtime_bounds
    for key in ("splat", "collision_mesh"):
        if _sha256_file(Path(base_scene[key + "_path"])) != base_scene[key + "_digest"]:
            raise ValueError("website_base_scene_changed")
    goals = [row for row in (task_masks or {}).get("targets") or []
             if row.get("target_role") == "destination" and isinstance(row.get("estimated_visible_bounds"), Mapping)]
    focus = goals[0]["estimated_visible_bounds"] if task_masks is not None and len(goals) == 1 else None
    mpu, runtime_to_sim, matrix = _sim_frame(base_scene=base_scene, source_geometry=source_geometry, focus=focus)
    room = {"binding": {key + "_digest": base_scene[key + "_digest"] for key in ("splat", "collision_mesh")},
            "coordinate_frame": {"declared_meters_per_unit": mpu, "declared_up_axis": base_scene["up_axis"],
                                 "physical_scale_measured": False, "runtime_to_simulator": runtime_to_sim.tolist()}}
    room["digest"] = canonical_digest(room, digest_field="digest")
    collision_root, normalized = normalize_runtime_collision(preparation=room, base_scene=base_scene,
                                                             output_root=output_root)
    appearance = prepare_native_appearance(preparation=room, base_scene=base_scene, output_root=output_root / "appearance")
    goal = None
    if focus is not None and matrix is not None:
        low, high = (np.asarray(corner) @ runtime_to_sim[:3, :3].T for corner in _runtime_bounds(focus, matrix))
        low, high = np.minimum(low, high), np.maximum(low, high)
        goal = {"basis": "goal_mask_bounds_registered_footprint_centre_at_its_lowest_point",
                "centre_world_m": [float((low[0] + high[0]) / 2), float((low[1] + high[1]) / 2), float(low[2])],
                "radius_m": GOAL_RADIUS_M, "dwell_seconds": GOAL_DWELL_SECONDS, "radius_and_dwell": "assumed"}
    elif task_masks is not None:
        goal = {"basis": "goal_not_placed:" + ("no_single_goal_mask" if focus is None else "goal_not_registered")}
    packet = output_root / "half" / DIRECTORY
    if packet.exists():
        shutil.rmtree(packet)
    rows = []
    for role, source, visible in (("scene_collision", collision_root / normalized["output"]["relative_path"], False),
                                  ("scene_appearance", Path(appearance["artifact"]["path"]), True)):
        target = packet / "assets" / f"{role}{source.suffix}"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        rows.append({"name": role, "semantic_role": role, "usd_path": f"assets/{target.name}",
                     "sha256": _sha256_file(target), "size_bytes": target.stat().st_size, "visible": visible,
                     "pose_world": {"position_world_m": [0.0, 0.0, 0.0], "orientation_xyzw": [0.0, 0.0, 0.0, 1.0]}})
    ruled = goal is not None and "centre_world_m" in goal
    task_kind = "goal_region_navigation" if task_masks is not None else "not_defined_by_blueprint"
    criteria = ({"surface_target": {"surface_position_world_m": goal["centre_world_m"], "radius_m": GOAL_RADIUS_M},
                 "settling": {"stable_seconds": GOAL_DWELL_SECONDS}} if ruled else None)
    plan = {"schema_version": PLAN_SCHEMA, "scene_id": task_context.get("scene_id"),
            "task_id": "website-" + task_context["context_digest"][7:27], "task_kind": task_kind,
            "task_spec": {"task_kind": task_kind, **({"task_success_contract": {"criteria": criteria}} if ruled else {})},
            "asset_directory": "assets", "objects": rows,
            "appearance_frame_alignment": {"representation": "particlefield_3d_gaussian_splat"},
            "robot": None, "cameras": [], "stage_5": None, "stage_4": None, "plan_digest": ""}
    plan["plan_digest"] = canonical_digest(plan, digest_field="plan_digest")
    placements(plan)
    write_json(packet / PLAN_FILENAME, plan)
    removed = [row for row in removal.get("entries") or [] if row.get("task_effect") == "manipulated"]
    count, objects_in_scan = (len(removed), objects_in_scan) if removed else (taken_out, False)
    record = {"schema_version": "blueprint.room_package_inputs.v1",
              "task_context_digest": task_context["context_digest"], "task_kind": task_kind,
              "task_profile": task_profile or "none", "success_rule_defined": ruled, "success_rule_assumed": ruled,
              **({"goal_region": goal} if goal is not None else {}),
              "job_objects": {"count": count, "in_package": "part_of_room_scan" if objects_in_scan
                              else "taken_out_of_room_scan"} if count else None,
              **({"flags": list(flags)} if flags else {}),
              **({"not_prepared": [dict(row) for row in not_prepared]} if not_prepared else {}),
              PROBE_FIELD: PROBE_NOT_QUEUED,
              "robot_included": False, "simulator_run": False, "plan_digest": plan["plan_digest"],
              "inputs_digest": ""}
    record["inputs_digest"] = canonical_digest(record, digest_field="inputs_digest")
    write_json(output_root / "half" / ROOM_RECORD, record)
    return record


class BuildChoiceUnsatisfiable(Exception):
    """An honoured build choice compile cannot build as chosen: not a ValueError, so no step records it as a stop."""


def build_choice_root(capture_root: Path) -> Path:
    """Beside every revision of the workspace, so a target correction that leaves the menu as it was finds the same
    receipt."""
    return Path(capture_root) / "pipeline" / "website_scene_preparation" / "build_choice"


def prepare_website_scene_handoff(*, descriptor: Mapping[str, Any], clean_plate: Mapping[str, Any],
                                 capture_root: Path, now: float,
                                 allow_operation_coverage_review: bool = False,
                                 allow_static_target_recovery: bool = False,
                                 allow_build_choice: bool = False) -> dict[str, Any]:
    """``allow_build_choice``: the job's build is the model's choice from the pipeline's menu where it answers validly
    (``website_build_choice``), one retained paid call; without it, or on any fallback, today's rules decide.

    A choice compile cannot build as chosen is rejected and this run goes on under the rules, never held; the next
    run decides again with that build off the menu. The decision's record keeps what shipped."""
    from .website_build_choice import record_shipped, reject_choice
    arguments = dict(descriptor=descriptor, clean_plate=clean_plate, capture_root=capture_root, now=now,
                     allow_operation_coverage_review=allow_operation_coverage_review,
                     allow_static_target_recovery=allow_static_target_recovery)
    decided: list[dict[str, Any]] = []
    try:
        result = _prepare_website_scene_handoff(**arguments, allow_build_choice=allow_build_choice, decided=decided)
    except BuildChoiceUnsatisfiable as exc:
        reject_choice(build_choice_root(capture_root), decided[-1], code=str(exc))
        result = _prepare_website_scene_handoff(**arguments, allow_build_choice=False, decided=None)
        record_shipped(build_choice_root(capture_root), decided[-1], result, rejected=str(exc))
        return result
    if decided:
        record_shipped(build_choice_root(capture_root), decided[-1], result)
    return result


def _prepare_website_scene_handoff(*, descriptor: Mapping[str, Any], clean_plate: Mapping[str, Any],
                                   capture_root: Path, now: float, allow_operation_coverage_review: bool,
                                   allow_static_target_recovery: bool, allow_build_choice: bool,
                                   decided: list[dict[str, Any]] | None) -> dict[str, Any]:
    handoff_lease_checkpoint("website_scene_handoff_start")
    from .website_preparation_revision import workspace, bind_preparation
    from .website_preparation_terminal import retained_window_end
    root = workspace(capture_root)
    root.mkdir(parents=True, exist_ok=True)
    handoff_path = capture_root / "pipeline" / "website_scene_preparation" / "handoff.json"
    result: dict[str, Any] = {
        "schema_version": "website_scene_handoff.v1", "status": "awaiting_reconstruction",
        "blockers": [], "claim_ceiling": "development_only", "simulator_ready": False,
        "provider_mutation_performed": False,
    }
    metadata = descriptor.get("metadata") or {}
    operation_review_requested = False
    result["capture_id"] = descriptor["capture_id"]
    result["scene_id"] = descriptor["scene_id"]

    def retain_gap(gap: Mapping[str, Any]) -> dict[str, Any]:
        result.update(status="needs_input", blockers=[gap["code"]], native_task_preparation_gap=dict(gap))
        return finish()

    def scan_taken_out() -> int:
        # Whether the scan that ships lost anything: the Atlas result is the one source of truth. A scene kept
        # as filmed (``preserves_capture``: a preserved origin, C1's rule, or the editor's floor taken after the
        # sealed decision said the planner's removals would apply) erased nothing.
        from .atlas_automation_plan import preserves_capture
        return 0 if preserves_capture(atlas) else planner_removals_applied(capture_root)

    def package_flags(flags: Sequence[str]) -> list[str]:
        # Every room record's flags, the room package's and its floor's alike: the caller's, then the clean
        # plate's (``room_flags``), the planner's removals not applied, the scene source's own last; each once.
        flags = room_flags(flags, clean_plate)
        from .atlas_automation_plan import JOB_OBJECT_RULE
        if atlas.get("job_object_rule") and JOB_OBJECT_RULE not in flags:
            # The planner's removals were not applied: nothing was taken out of the room scan.
            flags = [*flags, JOB_OBJECT_RULE]
        # The scene source's own flags last (the gap views not filled in), each once.
        return list(dict.fromkeys([*flags, *(atlas.get("flags") or [])]))

    def room_package(profile: str | None, masks: Mapping[str, Any] | None = None,
                     flags: Sequence[str] = (), taken_out: int = 0,
                     not_prepared: Sequence[str] = ()) -> dict[str, Any]:
        # Not held: the room package goes to the company builder (``write_room_package``); no object is built.
        # ``not_prepared``: the blockers it ships for, each with the piece of the job it leaves out (``piece_of``).
        result.pop("rule_withheld", None)
        flags = package_flags(flags)
        record = write_room_package(task_context=context, base_scene=base, source_geometry=atlas["source_geometry"],
                                    removal=removal, output_root=root / "room_package", task_profile=profile,
                                    task_masks=masks, objects_in_scan=geometry_only or in_scan, flags=flags,
                                    taken_out=taken_out, not_prepared=[
                                        {"piece": piece_of(code), "code": code}
                                        # The code alone, never its detail (a builder's or an exception's words).
                                        for code in dict.fromkeys(code.split(":", 1)[0] for code in not_prepared)])
        result.update(status=ROOM_PACKAGE, blockers=[], job_disposition="accepted",
                      geometry_backend=atlas["source_geometry"]["geometry_backend"],
                      source_geometry_digest=atlas["source_geometry"]["digest"],
                      room_package={"path": str(root / "room_package" / "half" / ROOM_RECORD),
                                    **{key: record[key] for key in ("inputs_digest", "task_kind", "task_profile",
                                                                    "success_rule_defined", "success_rule_assumed")}})
        result["digest"] = canonical_digest(result, digest_field="digest")
        write_json(handoff_path, result)
        return result

    def room_floor(blockers: Sequence[str] = ()) -> None:
        # Kept beside the object's own route, CPU only: the company builder ships it, flagged, while that route
        # runs or once it stops (``website_company_simulation.object_path_stopped``). Never a stop of its own.
        # ``blockers``: the preparation stopped on them, none a wall (``floor_ships``); flagged and named so.
        try:
            record = write_room_floor(now=now, task_context=context, base_scene=base,
                                      source_geometry=atlas["source_geometry"], removal=removal,
                                      output_root=root / ROOM_FLOOR, task_profile=None,
                                      objects_in_scan=geometry_only,
                                      flags=package_flags([ROOM_FLOOR_FLAG, *map(shipped_code, blockers)]),
                                      not_prepared=[{"piece": piece_of(code), "code": code} for code in dict.fromkeys(
                                          shipped_code(code).split(":", 1)[0] for code in blockers)])
        except (ValueError, KeyError, TypeError, OSError) as exc:
            result[ROOM_FLOOR] = {"status": "not_written", "reason": str(exc)[:200]}
            return
        result[ROOM_FLOOR] = {"path": str(root / ROOM_FLOOR / "half" / ROOM_RECORD),
                              "inputs_digest": record["inputs_digest"],
                              **({ROOM_FLOOR_BLOCKERS: list(blockers)} if blockers else {})}

    def finish() -> dict[str, Any]:
        # Every handoff this run writes itself. The room is built (``room_ready``) and the preparation stopped on no
        # wall: its room floor ships flagged at once, the job keeps retrying, and the package a later run writes
        # replaces it (owner, 2026-11-23).
        if (room_ready and result["status"] not in ("intake_ready", ROOM_PACKAGE) and result.get("blockers")
                and ROOM_FLOOR not in result and may_fall_back()
                and floor_ships(result["blockers"], task_context=metadata.get("site_task_context") or {},
                                error=failure)):
            room_floor([str(code) for code in result["blockers"]])
        result["digest"] = canonical_digest(result, digest_field="digest")
        write_json(handoff_path, result)
        return result

    def room_for(gap: Mapping[str, Any]) -> dict[str, Any]:
        # A gap that says Blueprint has no model for the job, by code: the job's declared profile, flagged.
        declared = gap.get("task_family") if gap["code"] == MISMATCH else gap.get("inferred_profile_id")
        return room_package(declared, flags=[gap["code"]])

    def window_end() -> float | None:
        return retained_window_end(capture_root, scene_id=descriptor["scene_id"], capture_id=descriptor["capture_id"])

    def retries_spent() -> bool:
        # This run is the last of the first retry cycle: if it fails too, the cycle is spent.
        return first_retry_cycle_spent(capture_root, now)

    def may_fall_back() -> bool:
        # Preparation and compile only: never once the prepared scene is being, or may have been, sent.
        return not dispatching and not prepared_scene_may_exist(
            capture_root, task_context_digest=(metadata.get("site_task_context") or {}).get("context_digest"))

    def send(request: Mapping[str, Any]) -> dict[str, Any]:
        # The prepared scene to the website, marked before the write and settled by its answer.
        nonlocal dispatching
        from .website_task_context import enqueue_website_prepared_scene
        dispatching = True
        marker = {"request_digest": cross_runtime_canonical_digest(request),
                  "task_context_digest": context.get("context_digest")}
        prior = _unsettled_outbox_evidence(capture_root)
        write_json(_outbox_marker(capture_root), {**marker, "state": "sending"})
        try:
            receipt = enqueue_website_prepared_scene(task_context=context, request=request)
        except ValueError as exc:
            cap_refused = str(exc) == _PREPARED_SCENE_CAP_REFUSAL
            if cap_refused or _REFUSED_WRITE.search(str(exc)):
                # This send committed nothing, but an earlier one that may have (or did) still may be forwarding into
                # paid preparation: its evidence stands, and only a marker with none behind it settles ``refused``.
                write_json(_outbox_marker(capture_root),
                           {**prior, "later_refusal": str(exc)[:200]} if prior is not None
                           else {**marker, "state": "refused", "code": str(exc)[:200]})
                if cap_refused and prior is None:
                    dispatching = False  # The settled refusal permits the room package.
            raise
        write_json(_outbox_marker(capture_root), {**marker, "state": "sent", "outbox_id": receipt.get("id")})
        return receipt

    room_ready = dispatching = False
    failure: BaseException | None = None
    keep_outbox_receipt(capture_root)
    try:
        if clean_plate.get("privacy_verified") is not True or clean_plate.get("status") not in {"noop", "objects_removed"}:
            raise ValueError("website_scene_preparation_pending")
        context = metadata.get("site_task_context") or {}
        if context.get("capture_id") != descriptor["capture_id"] or context.get("scene_id") != descriptor["scene_id"]:
            raise ValueError("website_scene_task_identity_mismatch")
        # The capture's Atlas job is its one paid reconstruction lane: its
        # object-removed background is the scene and its source pose the
        # geometry. Reopening it never reconstructs or purchases geometry.
        from .website_atlas_scene_source import atlas_scene_source
        atlas = atlas_scene_source(capture_root=capture_root, output_root=root / "atlas")
        handoff_lease_checkpoint("website_scene_handoff_after_source")
        if atlas["status"] != "ready":
            result.update(status="awaiting_reconstruction" if atlas["status"] == "pending" else "needs_input",
                blockers=[atlas["reason"]], geometry_backend=None, provider_mutation_performed=False,
                visual_reconstruction_ready=False,
                atlas_scene_source={key: atlas.get(key) for key in ("status", "reason", "stage", "job_reason")},
                native_geometry_requirements={
                    "source_aligned_depth_and_camera_poses": "required_for_mask_lifting_and_scene_registration",
                    "measured_scale": "unknown", "atlas_required": True})
            return finish()
        base = dict(atlas["base_scene"])
        # A base scene the job's earlier Atlas result built is replaced: its records are kept first.
        retain_superseded_records(capture_root=capture_root, root=root,
                                  atlas_result_digest=base.get("atlas_result_digest"),
                                  levelling_digest=(base.get("levelling") or {}).get("digest"))
        write_json(root / "base_scene.json", base)
        result.update(base_scene_path=str(root / "base_scene.json"), visual_reconstruction_ready=True)
        # A geometry-only job keeps its subject in the scene: it ends at the robot fit
        # check (geometry_only_fit_check) and is never registered or enqueued.
        geometry_only = atlas.get("atlas_route") == "geometry_only"
        if geometry_only:
            result["atlas_route"] = "geometry_only"
        removal_path = Path(clean_plate["removal_manifest_path"])
        if not removal_path.resolve().is_relative_to(capture_root.resolve() / "pipeline"):
            raise ValueError("website_scene_removal_manifest_outside_capture")
        removal = json.loads(removal_path.read_text())
        # The build plan took one object out of the scan: the job's other objects stay in it as filmed (owner, 2026-11-23).
        from .website_build_plan import apply_to_removal, sealed_decision
        removal, planned = apply_to_removal(removal, capture_root), sealed_decision(capture_root)
        if planned is not None:
            result["build_plan"] = {"digest": planned["build_plan"]["digest"],
                                    "built": (planned.get("built") or {}).get("target_id"),
                                    "kept_in_scan": len(planned.get("kept_in_scan_target_ids") or [])}
        # Capture truth applies to held families too. Reading this retained
        # manifest performs no mask, authoring or provider work.
        # The editor could not take the job's object out cleanly, or the job's objects are several or soft, so the
        # scene kept them as filmed (``website_atlas_pipeline.background_floor``, ``job_object_review``): they are
        # part of the room scan, never also built on their own.
        kept_rule = atlas.get("job_object_rule") or {}
        in_scan = (bool(kept_rule.get("background_not_cleaned") or kept_rule.get("job_objects_kept")
                        or atlas.get("job_objects_kept"))
                   and any(row.get("task_effect") == "manipulated" for row in removal.get("entries") or []))
        room_ready = True  # from here a room package can be written (``room_package``)
        # A subject-origin conflict protects no privacy, rights or money: once the first retry cycle is spent it
        # ships the room flagged (``goes_on``), so it is raised once the room can be packaged.
        if (atlas.get("subject_origin") == "synthetic_design") != ("synthetic_subject" in removal):
            raise ValueError("website_subject_origin_conflict")
        from .website_job_spend_cap import SPENDING_CAP_REACHED
        if SPENDING_CAP_REACHED in (atlas.get("flags") or []):
            if not may_fall_back():
                raise ValueError(SPENDING_CAP_REACHED)
            return room_package(context.get("task_family"), not_prepared=[SPENDING_CAP_REACHED])
        if in_scan:
            return room_package(context.get("task_family"))
        from .website_task_semantics import load_task_semantics
        source_video_digest = (atlas["source_geometry"].get("binding") or {}).get("source_video_digest")
        semantics = (load_task_semantics(job_dir=capture_root / "pipeline" / "atlas", task_context=context,
                                        source_video_digest=source_video_digest)
                     if context.get("task_family") == "general_task" and source_video_digest else None)
        if semantics is not None and allow_operation_coverage_review:
            from .website_operation_coverage import clarification_request
            if clarification_request(semantics, task_context=context, source_video_digest=source_video_digest) is not None:
                # A completed saved Atlas job need not re-enter its controller.
                # Clarify its operation reading through the same capped source
                # allowance; no scene/pose/selection or owner fact is rewritten.
                from .website_operation_coverage_review import clarify_operation_coverage
                operation_review_requested = True
                result.pop("provider_mutation_performed", None)
                handoff_lease_checkpoint("website_scene_handoff_operation_review")
                clarified = clarify_operation_coverage(job_dir=capture_root / "pipeline" / "atlas",
                    task_context=context, source_video_digest=source_video_digest)
                semantics = clarified["task_semantics"]
        semantics_options = {"task_semantics": semantics, "source_video_digest": source_video_digest}
        # The model's object build, honoured in place of today's judgement rules (``website_build_choice``).
        override = None
        # The numbers of the chosen rule the site's words state, from the same answer, honoured or not.
        stated_success = None
        if allow_build_choice:
            from .website_build_choice import (RECORD, ROOM_FLAG, ROOM_ONLY, RULE_WITHHELD, RULE_WITHHELD_CODE,
                                               decide_build, rules_route)
            found_targets = (clean_plate.get("task_masks") or {}).get("targets") or []
            rules = rules_route(task_context=context, removal=removal, targets=found_targets,
                                flags=clean_plate.get("flags") or [], geometry_only=geometry_only, **semantics_options)
            if rules is not None:
                video = clean_plate.get("input_video_path")
                key_video = Path(video) if isinstance(video, str) else None
                if key_video is not None and (key_video.is_symlink() or not key_video.is_file()
                                              or not key_video.resolve().is_relative_to(capture_root.resolve())):
                    key_video = None
                handoff_lease_checkpoint("website_scene_handoff_build_choice")
                decision = decide_build(task_context=context, removal=removal, targets=found_targets, rules=rules,
                                        output_root=build_choice_root(capture_root), source_video=key_video,
                                        task_semantics=semantics, source_video_digest=source_video_digest)
                if decision is not None and decided is not None:
                    decided.append(decision)
                if decision is not None and (decision.get("choice") or {}).get("success_rule"):
                    # Only numbers the words state: with none, the rule and the job's output are today's.
                    stated_success = {"build": decision["choice"]["build"],
                                      "fields": dict(decision["choice"]["success_rule"]),
                                      "request_binding_digest": decision["request_binding_digest"]}
                if decision is not None and decision["honoured"]:
                    chosen = decision["choice"]
                    result["build_choice"] = {"build": chosen["build"], "rules_build": rules["build"],
                                              "source": "model",
                                              "record_path": str(build_choice_root(capture_root) / RECORD),
                                              "request_binding_digest": decision["request_binding_digest"]}
                    if chosen["build"] == ROOM_ONLY:
                        from .task_profile_contracts import task_profile_for
                        from .website_early_task_hold import task_family_runtime_gap
                        from .website_task_semantics import effective_task_profile
                        own = task_profile_for(effective_task_profile(task_context=context, **semantics_options))
                        missing = task_family_runtime_gap(context, **semantics_options) is not None
                        return room_package(own.profile_id if own is not None else None,
                                            flags=[ROOM_FLAG, *([RUNTIME_GAP] if missing else [])])
                    override = {"build": chosen["build"], "destination_target_id": chosen["destination_target_id"],
                                "request_binding_digest": decision["request_binding_digest"]}
                    if chosen["build"] == RULE_WITHHELD:
                        result["rule_withheld"] = {"code": RULE_WITHHELD_CODE,
                                                   "task_id": "website-" + context["context_digest"][7:27]}
        gap = None if override is not None else native_task_preparation_gap(context, **semantics_options)
        if gap is not None and gap["code"] in ROOM_GAPS:
            from .website_operation_coverage import clarification_request
            if semantics is None or clarification_request(semantics, task_context=context,
                                                          source_video_digest=source_video_digest) is None:
                # No task profile, nor one a clarification could still find: never turned into pick/place.
                # One removed object is built on the rigid path with Blueprint's rule withheld (named and sealed
                # here); anything else ships as the room package.
                manipulated = [row for row in removal.get("entries", []) if row.get("task_effect") == "manipulated"]
                found = {row["target_id"]: row for row in (clean_plate.get("task_masks") or {}).get("targets", [])}
                if (len(manipulated) != 1 or geometry_only or removal.get("synthetic_subject") is not None
                        or manipulated[0].get("target_id") not in found or native_task_subject_profile(
                            manipulated[0], found[manipulated[0]["target_id"]])[1] != "rigid_pick_place"):
                    return room_package(None)  # several objects, none, or a part that opens
                withheld = {"code": gap["code"], "task_id": "website-" + context["context_digest"][7:27]}
                result["rule_withheld"], gap = withheld, None
        if gap is not None:
            # Retain the completed reconstruction and original task. Missing
            # adapter/scoring support cannot turn a different job into pick/place.
            result.update(geometry_backend=atlas["source_geometry"]["geometry_backend"],
                          source_geometry_digest=atlas["source_geometry"]["digest"])
            return retain_gap(gap)
        from .task_profile_contracts import task_profile_for
        from .website_task_semantics import effective_task_profile
        resolved = task_profile_for(effective_task_profile(task_context=context, **semantics_options))
        navigation = resolved is not None and not resolved.requires_manipulation
        if set(ROOM_AS_FILMED) & set(clean_plate.get("flags") or []):
            # Nothing of the task in the video, or the object seen but not outlined: no object, mask or goal to
            # read. The room as filmed, flagged; a navigation job keeps its kind, with no goal region. If the
            # Atlas job had already sealed its planner's removals, the job's objects are out of the scan too.
            return room_package(resolved.profile_id if resolved is not None else None,
                                {"targets": []} if navigation else None, taken_out=scan_taken_out())
        from .website_task_masks import bind_task_masks_to_geometry, complete_retained_static_task_masks
        from .task_family_scoring import canonical_task_family
        from .website_task_semantics import effective_task_profile
        family = canonical_task_family(effective_task_profile(task_context=context, **semantics_options))
        result.pop("provider_mutation_performed", None)
        if family is None and (clean_plate.get("task_masks") or {}).get("deferred_target_ids"):
            result["deferred_masks_controller_invoked"] = True
            plan_path = Path(clean_plate["removal_plan_path"])
            if not plan_path.resolve().is_relative_to(capture_root.resolve() / "pipeline"):
                raise ValueError("website_scene_removal_plan_outside_capture")
            recovery_options = {}
            video = clean_plate.get("input_video_path")
            if allow_static_target_recovery and isinstance(video, str):
                source_video = Path(video)
                if (not source_video.is_symlink()
                        and source_video.resolve().is_relative_to(capture_root.resolve())):
                    recovery_options = {"task_context": context, "source_video": source_video,
                                        "recovery_geometry": atlas["source_geometry"]}
            handoff_lease_checkpoint("website_scene_handoff_static_masks")
            # Kept surfaces' full-video source tracks stay compact (``RunLengths``) through
            # binding, coverage, compile and the native inputs; each writes and hashes them
            # with the codec, to the bytes and digests the plain lists give.
            masks = complete_retained_static_task_masks(plan=json.loads(plan_path.read_text()),
                source_geometry=clean_plate["source_frames"], task_masks=clean_plate["task_masks"],
                output_root=plan_path.parent / "task_masks",
                view_plan_root=plan_path.parent / "mask_view_plan", compact_runs=True, **recovery_options)
            clean_plate = {**clean_plate, "task_masks": masks}
        geometry = atlas["source_geometry"]
        result.update(geometry_backend=geometry["geometry_backend"],
                      atlas_scene_source={key: atlas[key] for key in (
                          "status", "source_pose_digest", "background_pose_digest", "alignment")})
        masks = bind_task_masks_to_geometry(task_masks=clean_plate["task_masks"], source_geometry=geometry)
        write_track_json(root / "task_masks.geometry.json", masks)
        clean_plate = {**clean_plate, "source_geometry": geometry, "task_masks": masks}
        from .website_target_background_worker import queue_correction
        correction = queue_correction(capture_root=capture_root, task_context=context, task_masks=masks, now=now)
        if correction['status'] != 'ready':
            result.update(status='awaiting_reconstruction' if correction['status'] == 'pending' else 'needs_input',
                blockers=[correction['reason']], visual_reconstruction_ready=False,
                atlas_scene_source={'status': correction['status'], 'reason': correction['reason'],
                                    'stage': 'target_correction'})
            # No room floor here: this room was cleaned around the object the site has since corrected, so a
            # floor would tell a company the wrong object was the job's. The corrected room ships when it is built.
            result["digest"] = canonical_digest(result, digest_field="digest")
            write_json(handoff_path, result)
            return result
        if navigation:
            # The robot moves itself: the room, and the goal region lifted from the reading's goal mask.
            return room_package(resolved.profile_id, masks)
        if family is not None:
            from .website_early_task_hold import task_family_runtime_gap
            if task_family_runtime_gap(context, **semantics_options) is not None:
                # No runtime for the family is installed here: its route cannot run, so the room package instead.
                return room_package(family, flags=[RUNTIME_GAP])
        if family is not None and geometry_only:
            # The fit check models one rigid subject; a family task is recorded, never assessed.
            result.update(status="needs_input", blockers=[GEOMETRY_ONLY_BLOCKER],
                          fit_check={"status": "not_run", "reason": "task_family_not_modelled"})
            return finish()
        if family is not None:
            # Family tasks compile all retained targets and their native
            # measurements. The single rigid-subject authoring path cannot
            # represent fluid, residue, cloth or an ordered set of operations.
            preparation = compile_website_scene_preparation(task_context=context, task_masks=masks,
                removal_manifest=removal, source_geometry=geometry, base_scene=base, output_root=root,
                spend=metadata.get("website_scene_execution_authority") or {}, now=now,
                evaluation_inputs=metadata.get("native_task_evaluation"),
                **({"task_semantics": semantics} if semantics is not None else {}))
            preparation = bind_preparation(preparation, output_root=root, capture_root=capture_root,
                authority=metadata.get("website_scene_execution_authority") or {})
            result.update(status=preparation["status"], blockers=preparation["blockers"],
                job_disposition="accepted", task_evaluation_inputs=preparation["task_evaluation_inputs"],
                preparation_path=str(root / "preparation.json"), preparation_digest=preparation["digest"],
                source_geometry_digest=geometry["digest"],
                dispatch_blockers=preparation.get("dispatch_blockers", []))
            if not operation_review_requested:
                result["provider_mutation_performed"] = False
            if preparation["status"] == "intake_ready":
                room_floor()
                dispatching = True  # from here a failure keeps the job waiting, as before
                from .website_scene_dispatch import binding_root, register_website_preparation
                native_path = root / "native_runtime_contract.json"
                write_json(native_path, preparation["task_evaluation_inputs"]["native_runtime_contract"])
                context_path = root / "task_context.json"
                write_json(context_path, context)
                result["source_registration"] = register_website_preparation(
                    preparation_path=root / "preparation.json", runtime_inputs_path=native_path,
                    task_context_path=context_path, root=binding_root(), now=now)
                if (metadata.get("website_scene_execution_authority") or {}).get("schema_version") == "website_scene_sponsorship.v1":
                    result["website_intake_outbox"] = send(preparation["intake_request"])
            return finish()
        targets = {row["target_id"]: row for row in masks.get("targets", [])}
        subjects = [row for row in removal.get("entries", []) if row.get("task_effect") == "manipulated"]
        synthetic = removal.get("synthetic_subject")
        inferred = None
        if synthetic is not None:
            if (not isinstance(synthetic, Mapping) or subjects
                    or synthetic.get("start_support_target_id") not in targets):
                raise ValueError("website_synthetic_subject_binding_invalid")
            _, inferred = native_task_subject_profile({}, targets[synthetic["start_support_target_id"]],
                                                      synthetic_subject=synthetic)
        elif len(subjects) == 1 and subjects[0].get("target_id") in targets:
            _, inferred = native_task_subject_profile(subjects[0], targets[subjects[0]["target_id"]])
        if inferred is not None and override is None:
            gap = native_task_preparation_gap(context, inferred_profile_id=inferred, **semantics_options)
            if gap is not None and gap["code"] != (result.get("rule_withheld") or {}).get("code"):
                # The reading and the object disagree (the only gap left here): fitted to neither, the room
                # package, flagged, before coverage or spec callbacks buy anything.
                return room_for(gap)
        if synthetic is None:
            # Several task objects, or a soft one, under a one-rigid-object or one-opening
            # profile: held here, before coverage, spec research and authoring buy anything for it.
            # The model's honoured build decided this already (one object, as it chose).
            gap = None if override is not None else native_task_subject_gap(
                context, subjects=subjects, targets=targets, **semantics_options)
            if gap is not None:
                # Not narrowed to one rigid object: the room package, the objects counted, no rule of Blueprint's.
                return room_for(gap)
            # Rigid or unknown goes on; the reading, its source and an unknown's flag stay on the record.
            rigidity = subject_rigidity(context, subjects=subjects, targets=targets, **semantics_options)
            if rigidity is not None:
                result["task_object_rigidity"] = rigidity
                if "flag" in rigidity:
                    result["flags"] = [rigidity["flag"]]
        # Each articulated task target needs views of its whole assembly from
        # the full track, not only the geometry frames. Retained receipts and
        # decoded frames make a restart free.
        from .website_assembly_coverage import attach_assembly_coverage
        source_video = Path(clean_plate.get("input_video_path") or "")
        handoff_lease_checkpoint("website_scene_handoff_assembly_coverage")
        masks = attach_assembly_coverage(task_masks=clean_plate["task_masks"],
            source_geometry=clean_plate["source_geometry"], removal_manifest=removal, task_context=context,
            source_video=(source_video if source_video.is_file()
                          and source_video.resolve().is_relative_to(capture_root.resolve()) else None),
            output_root=root / "assembly_coverage")
        # Published size and mass of an identified object reach compile before
        # the builder. An unknown identity or closed agent gate records not_run
        # and spends nothing; retained receipts make restarts free.
        from .website_object_spec_research import attach_object_specs
        handoff_lease_checkpoint("website_scene_handoff_object_specs")
        masks = attach_object_specs(task_masks=masks, removal_manifest=removal, task_context=context,
                                    output_root=root / "object_spec")
        if masks != clean_plate["task_masks"]:
            write_track_json(root / "task_masks.coverage.json", masks)
            clean_plate = {**clean_plate, "task_masks": masks}
        # The plan's object from a licensed library first, before a shape or a paid model (owner, 2026-11-23).
        from .website_object_library import attach_library_subject
        handoff_lease_checkpoint("website_scene_handoff_object_library")
        removal = attach_library_subject(removal, capture_root=capture_root, task_context=context,
            output_root=root / "object_library", source_video_digest=source_video_digest,
            source_video=source_video if source_video.is_file() and source_video.resolve().is_relative_to(
                capture_root.resolve()) else None)
        authority = metadata.get("website_scene_execution_authority") or {}
        if not all(key in authority for key in ("max_total_spend_usd", "max_paid_attempts", "expires_at_epoch")):
            # Compile geometry even while execution is held. Zero authority
            # cannot validate as a paid intake request.
            authority = {"max_total_spend_usd": 0, "max_paid_attempts": 0, "expires_at_epoch": now}
        def choose():  # called by compile only once it found no gap: a room package never chooses
            return artisan_provider_choice_for(authority=authority, root=root, capture_root=capture_root,
                                               geometry_only=geometry_only)
        from .website_object_local_frame import REGISTRATION_REFUSALS
        from .website_recapture_required import CODES as RECAPTURE_CODES
        try:
            preparation = compile_website_scene_preparation(
                task_context=context, task_masks=clean_plate["task_masks"],
                removal_manifest=removal, source_geometry=clean_plate["source_geometry"],
                base_scene=base, output_root=root, spend=authority, now=now,
                artisan_provider_choice=choose,
                **({"task_semantics": semantics} if semantics is not None else {}),
                **({"atlas_route": "geometry_only"} if geometry_only else {}),
                **({"rule_withheld": result["rule_withheld"]} if "rule_withheld" in result else {}),
                **({"build_choice": override} if override is not None else {}),
                **({"success_rule": stated_success} if stated_success is not None else {}),
            )
        except ValueError as exc:
            from .website_build_choice import BINDING_INVALID
            if override is not None and str(exc) == BINDING_INVALID:
                raise BuildChoiceUnsatisfiable(str(exc)) from exc  # the rules decide this run; never a hold
            if "rule_withheld" in result and str(exc) in REGISTRATION_REFUSALS | RECAPTURE_CODES:
                # The object does not fit the scene or the video: left out, never a hold or a new video.
                return room_package(None, flags=[str(exc)])
            raise
        if (((preparation.get("collider_conditioning") or {}).get("status") == "conditioned"
                or (preparation.get("destination_surface_estimate") or {}).get("status") == "estimated")
                and (preparation.get("subject_excision") or {}).get("status") != "excised"):
            # Compile flattened the surfaces objects rest on, or added one the scan missed: it is still the room as
            # filmed, so every package reads it. A cut collider has the job's own object taken out of it, which a
            # room package keeping the objects in its scan must not read: that one is read only after the fallback.
            base = json.loads((root / "base_scene.json").read_text())
        if (preparation["status"] != "intake_ready" and not preparation.get("native_task_preparation_gap")
                and may_fall_back()
                and ("rule_withheld" in result and not set(preparation["blockers"]) & REGISTRATION_REFUSALS
                     or not geometry_only and goes_on(preparation["blockers"], task_context=context, now=now,
                                                      window_end=window_end, retries_spent=retries_spent))):
            # Anything else that stops the object (no internal place, no support, a size the published figures
            # refute, an authoring admission unmet: nothing was spent or sent for it): the room package, as main
            # ships this job, the object left out and the reason a flag. A registration refusal is a blocker only
            # in development-test mode, which places the object on its estimated frame instead. Any other job
            # whose blockers are no wall ships the same room package (``goes_on``); a geometry-only job ends at its
            # fit check instead.
            shipped = [shipped_code(str(code)) for code in preparation["blockers"]]
            return room_package(None, flags=shipped, not_prepared=shipped)
        preparation = bind_preparation(preparation, output_root=root, capture_root=capture_root, authority=authority)
        if ((preparation.get("collider_conditioning") or {}).get("status") == "conditioned"
                or (preparation.get("subject_excision") or {}).get("status") == "excised"):
            # Compile sealed a conditioned or cut collider into the base scene; later stages read that one.
            base = json.loads((root / "base_scene.json").read_text())
        result.update(status=preparation["status"], blockers=preparation["blockers"],
                      preparation_path=str(root / "preparation.json"), preparation_digest=preparation["digest"],
                      thumbnail=preparation["thumbnail"], base_scene_path=str(root / "base_scene.json"))
        if isinstance(preparation.get("native_task_preparation_gap"), Mapping):
            # Later source-bound evidence can also reveal a mismatch: no native intake, so no runtime preparation.
            # Every gap compile writes says only that Blueprint has no model for the job: the room package, flagged.
            return room_for(preparation["native_task_preparation_gap"])
        if geometry_only:
            result["fit_check"] = geometry_only_fit_check(preparation=preparation, base_scene=base,
                                                          native_root=root / "fit_check")
            return finish()
        if preparation["status"] == "intake_ready":
            room_floor()
        # A missing execution authority must not stop this CPU-only geometry
        # conversion. The object stays in its separate authoring lane.
        try:
            from .website_development_test import enabled, prepare_development_test
            development = enabled(context["context_digest"])
            if development:
                root = root / "development_test"
                preparation, runtime_inputs = prepare_development_test(preparation=preparation,
                    source_geometry=clean_plate["source_geometry"], task_masks=clean_plate["task_masks"],
                    output_root=root)
                result["development_test"] = {**preparation["development_test"],
                    "preparation_path": str(root / "preparation.json")}
                native_root = root
            else:
                native_root = root / "native"
                runtime_inputs = None
            if prepare_rigid_native_inputs(result=result, preparation=preparation,
                                           preparation_path=root / "preparation.json", native_root=native_root,
                                           task_context=context, now=now, runtime_inputs=runtime_inputs,
                                           base_scene=base, source_geometry=clean_plate["source_geometry"],
                                           task_masks=clean_plate["task_masks"], capture_root=capture_root):
                dispatching = True  # from here a failure keeps the job waiting, as before
                from .website_scene_dispatch import binding_root, register_website_preparation
                context_path = root / "task_context.json"
                write_json(context_path, context)
                result["source_registration"] = register_website_preparation(
                    preparation_path=root / "preparation.json", runtime_inputs_path=native_root / "runtime_inputs.json",
                    task_context_path=context_path, root=binding_root(), now=now)
                if authority.get("schema_version") == "website_scene_sponsorship.v1":
                    result["website_intake_outbox"] = send(preparation["intake_request"])
        except (ValueError, KeyError, TypeError, OSError, ImportError) as exc:
            code = blocker_code(exc)
            if may_fall_back() and goes_on([*preparation["blockers"], code], task_context=context, now=now,
                                           error=exc, window_end=window_end, retries_spent=retries_spent):
                # The object's inputs cannot be made, and nothing the preparation holds is a wall: the room
                # package, flagged (``goes_on``).
                return room_package(None, flags=[shipped_code(code)], not_prepared=[shipped_code(code)])
            # Preserve finished CPU outputs when a later handoff is held.
            result.setdefault("runtime_inputs", {}).update(status="awaiting_inputs", blockers=[code],
                **({"defect": f"{type(exc).__name__}: {exc}"[:500]} if code != str(exc) else {}))
    except (ValueError, KeyError, TypeError, OSError) as exc:
        from .website_gemini_receipts import CreditDepleted, last_money_recheck
        code = blocker_code(exc)
        stopped_task = metadata.get("site_task_context") or {}
        if room_ready and may_fall_back() and (
                goes_on([code], task_context=stopped_task, now=now, error=exc, window_end=window_end,
                        retries_spent=retries_spent)
                or isinstance(exc, CreditDepleted) and last_money_recheck(exc, now, window_end())):
            # A step of this preparation failed for no wall (``goes_on``): the job goes on as the room package,
            # flagged, and buys nothing. Gemini's answer that Blueprint's credits are depleted is a money wait
            # until the job's window, read now, ends before it can be asked again (``CreditDepleted``).
            try:
                return room_package(None, flags=[shipped_code(code)], not_prepared=[shipped_code(code)])
            except (ValueError, KeyError, TypeError, OSError):
                pass  # the room cannot be packaged either: the step's code is kept, and the job retries
        if code != str(exc):
            # A defect waits, recorded under its code (the owner is paged) with its message for whoever fixes it.
            result["defect"] = f"{type(exc).__name__}: {exc}"[:500]
        from .website_recapture_required import CODES as RECAPTURE_CODES, FIELD as RECAPTURE_FIELD, record
        stopped_context = metadata.get("site_task_context") or {}
        if (isinstance(exc, ValueError) and str(exc) in RECAPTURE_CODES
                and isinstance(stopped_context.get("context_digest"), str)):
            # Only a new video clears it: the stage ends as a hold that asks the site for one, never retried.
            result.update(status="needs_input", blockers=[str(exc)], **{RECAPTURE_FIELD: record(
                str(exc), task_context_digest=stopped_context["context_digest"])})
        else:
            result.update(status="awaiting_inputs", blockers=[code])
        failure = exc
    return finish()
