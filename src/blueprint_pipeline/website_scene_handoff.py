"""Join prepared website captures to the scene their Atlas job reconstructed.

ADP-030/040, day 28: retain the task placement and authoring inputs as soon as
the Atlas job finishes. A held job or missing execution authority must not
discard completed preparation or label the scene simulator-ready.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

from .core.handoff_lease import (
    HandoffLeaseError, guarded_handoff_write_json as write_json, guarded_handoff_write_track_json as write_track_json,
    handoff_effect, handoff_lease_checkpoint,
)
from .decision_evidence_contracts import canonical_digest
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


def _robot_fit_precheck(*, preparation: Mapping[str, Any], verdict: Mapping[str, Any] | None,
                        collision_path: Path, native_root: Path) -> dict[str, Any]:
    """Whether a robot the screening catalog runs can do the task in this room (``website_robot_fit_precheck``).

    It fails open: a failure is recorded as ``unknown`` and the handoff goes on, since only a ``no_fit``
    the late checks would also make holds the job. A lost handoff lease stops it, as it stops every write.
    """
    try:
        from .website_robot_fit_hold import team_registry_state
        from .website_robot_fit_precheck import FILENAME, registered_team_bodies, robot_fit_precheck
        registry, registry_digest = team_registry_state()
        try:
            bodies, flags = (registered_team_bodies(registry) if registry is not None else {}), []
        except (OSError, ValueError, KeyError, TypeError):
            bodies, registry_digest = {}, "unreadable"
        if registry_digest == "unreadable":
            # The operator's team registry did not read: its bodies stay unknown, never refused.
            flags = ["team_registry_unreadable"]
        value = robot_fit_precheck(preparation=preparation, collision_path=collision_path, verdict=verdict,
                                   team_bodies=bodies, team_registry_digest=registry_digest)
        write_json(native_root / FILENAME, value)
        # What a hold was decided against rides on the handoff, so a change reruns it (website_robot_fit_hold).
        return {"path": str(native_root / FILENAME), "digest": value["precheck_digest"], "status": value["status"],
                "hold_reason": value["hold_reason"], "counts": value["counts"],
                "recheck_inputs": value["recheck_inputs"], **({"flags": flags} if flags else {})}
    except HandoffLeaseError:
        raise
    except Exception as exc:  # noqa: BLE001 - unknown never holds a job; this is recorded
        return {"status": "unknown", "hold_reason": None, "error": type(exc).__name__}


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


def _navigation_consent_refusal(*, capture_root: Path, task_context: Mapping[str, Any],
                                authority: Mapping[str, Any], now: float) -> str | None:
    """Why the job's own consent records stop its navigation scene half, or None while they allow it.

    Privacy and rights, read from the records the website gave this job: a withdrawal on this host
    (``website_atlas_preview_store.capture_withdrawn``, the tombstone or a revoked consent), the task
    context's capture rights (``website_task_context.json``) and the end of its sponsorship
    (``website_scene_sponsorship.json``, ``expires_at_epoch``, the website's ``consent_expired``).
    """
    from .website_atlas_preview_store import capture_withdrawn

    if capture_withdrawn(capture_root):
        return "website_capture_withdrawn"
    rights = task_context.get("capture_rights") if isinstance(task_context.get("capture_rights"), Mapping) else {}
    if rights.get("derived_scene_generation_allowed") is not True or rights.get("consent_revoked") is not False:
        return "website_scene_processing_rights_required"
    expiry = authority.get("expires_at_epoch") if isinstance(authority, Mapping) else None
    if isinstance(expiry, bool) or not isinstance(expiry, (int, float)) or not math.isfinite(expiry):
        return "website_scene_execution_authority_required"
    return "website_scene_consent_expired" if expiry <= now else None


def _navigation_scene_inputs(*, result: dict[str, Any], capture_root: Path, root: Path,
                             navigation: Mapping[str, Any], assets: Mapping[str, Any],
                             task_context: Mapping[str, Any], authority: Mapping[str, Any],
                             now: float) -> dict[str, Any] | None:
    """The G1 lane's scene half for a navigation job, on the controller and with no spend.

    The base packet with the pinned G1 standing at the compiled start, its robot-free probe receipt and
    the sealed record (``website_navigation_scene_inputs``), kept in the capture's own workspace so the
    site's withdrawal deletes them, and the team registry's derived row naming the record. The robot is
    the G1 registration's own pinned Arena USD (``native_g1_site_registration``). Withdrawn, revoked or
    expired consent writes neither. Returns the written record (``{"path", "record"}``) once both exist,
    else None; anything else is recorded and is this job's missing piece, never a stop. A lost handoff
    lease stops it, as it stops every write.
    """
    from .website_navigation_scene_inputs import OUTPUT_DIRECTORY

    refusal = _navigation_consent_refusal(capture_root=capture_root, task_context=task_context,
                                          authority=authority, now=now)
    if refusal is not None:
        result["navigation_scene_inputs"] = {"status": "not_written", "reason": refusal}
        return None
    try:
        from . import native_g1_site_registration as registration
        from .team_candidate_quick10_registry import write_derived_profile
        from .website_navigation_scene_inputs import derived_profile_row, write_navigation_scene_inputs

        handoff_lease_checkpoint("website_scene_handoff_navigation_scene_inputs")
        robot = registration._robot_record(registration.load_shared_inputs())
        written = write_navigation_scene_inputs(navigation_task_spec=navigation, scene_assets=assets,
                                                robot_record=robot, output_root=root / OUTPUT_DIRECTORY,
                                                capture_root=capture_root)
        handoff_lease_checkpoint("website_scene_handoff_navigation_registration")
        profile = write_derived_profile(derived_profile_row(written, capture_root=capture_root))
    except HandoffLeaseError:
        raise
    except Exception as exc:  # noqa: BLE001 - only privacy, rights and money stop a job; this is its missing piece
        reason = str(exc) if str(exc) else type(exc).__name__
        result["navigation_scene_inputs"] = {"status": "failed", "error": reason[:300]}
        return None
    record = written["record"]
    result["navigation_scene_inputs"] = {
        "status": "written" if profile.get("status") == "written" else "not_registered",
        "path": written["path"], "digest": record["scene_inputs_digest"],
        "scene_revision_digest": record["scene_revision_digest"],
        "robot_free_probe_status": record["robot_free_probe"]["status"], "registry": dict(profile)}
    return dict(written) if profile.get("status") == "written" else None


def _navigation_intake(*, result: dict[str, Any], written: Mapping[str, Any], capture_root: Path,
                       base_scene: Mapping[str, Any], task_context: Mapping[str, Any],
                       authority: Mapping[str, Any]) -> tuple[bool, str | None]:
    """The job's website navigation intake, once its scene half is written: (receipt exists, refusal name).

    The request is the record's digests with the job's own sponsorship: its owner, consent and rights
    exactly as a rigid intake carries them, and the website's own ``task_confirmed`` reading of the brief
    (``website_navigation_intake``). The website reserves $0 for it and grants no construction allowance.
    A typed refusal (an unconfirmed brief is ``website_task_context_not_confirmed``) is recorded and
    returned, and the job holds by that name. Any other failure is recorded and is this job's missing
    piece. Once the Pipeline has staged the intent (seen by a later handoff), its receipt is bound here.
    A lost handoff lease stops it, as it stops every write.
    """
    if (authority.get("schema_version") != "website_scene_sponsorship.v1"
            or not isinstance(authority.get("owner"), Mapping) or not isinstance(authority.get("consent"), Mapping)):
        result["navigation_intake"] = {"status": "not_requested", "reason": "website_scene_execution_authority_required"}
        return False, None
    try:
        from .website_navigation_intake import bind_navigation_intake_receipt, enqueue_navigation_intake

        handoff_lease_checkpoint("website_scene_handoff_navigation_intake")
        sent = enqueue_navigation_intake(record_path=written["path"], record=written["record"],
                                         capture_root=capture_root, task_context=task_context,
                                         authority=authority, base_scene=base_scene)
        if sent["status"] == "refused":
            result["navigation_intake"] = dict(sent)
            return False, sent["reason"]
        receipt = bind_navigation_intake_receipt(record_path=written["path"], capture_root=capture_root)
    except HandoffLeaseError:
        raise
    except Exception as exc:  # noqa: BLE001 - only privacy, rights and money stop a job; this is its missing piece
        reason = str(exc) if str(exc) else type(exc).__name__
        result["navigation_intake"] = {"status": "failed", "error": reason[:300]}
        return False, None
    result["navigation_intake"] = {**sent, "receipt": None if receipt is None else {
        "receipt_digest": receipt["receipt_digest"], "source": receipt["source"]}}
    return receipt is not None, None


def prepare_rigid_native_inputs(*, result: dict[str, Any], preparation: Mapping[str, Any], preparation_path: Path,
                                native_root: Path, task_context: Mapping[str, Any], now: float,
                                runtime_inputs: Mapping[str, Any] | None = None,
                                base_scene: Mapping[str, Any] | None = None,
                                source_geometry: Mapping[str, Any] | None = None,
                                task_masks: Mapping[str, Any] | None = None) -> bool:
    """The rigid native tail of a compiled preparation, CPU only: it buys nothing.

    Runtime inputs (unless the caller already made them), the collision and
    appearance stages, the construction inputs, the rights admission and the
    capability verdict, with the informational robot fit table beside it, and
    the robot fit precheck (``website_robot_fit_precheck``). The
    handoff and a renewal re-issue
    (``website_preparation_renewal``) share it, so the inputs bound to a
    preparation's digest cannot drift between them. Records into ``result``
    and returns whether the preparation may be registered: intake ready,
    appearance authored, no abstention and no ``website_robot_fit_none`` hold.
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
        rights_path = native_root / "rights_admission.json"
        write_json(rights_path, rights)
        construction["references"].append({"contract_path": "scene.rights.admission", **_record(rights_path)})
    write_json(native_root / "construction_inputs.json", construction)
    result["runtime_inputs"]["construction_inputs_path"] = str(native_root / "construction_inputs.json")
    held = False
    if preparation["status"] == "intake_ready":
        from .usd_binary_read_cache import read_scope
        collision = Path(runtime_inputs["collision"]["path"])
        verdict = None
        # One read scope for every search below: a USDA text collider is read once (``usd_binary_read_cache``).
        with read_scope():
            if preparation["intake_request"]["task"]["strategy"] == "pick_and_place":
                # Abstain before the owner is asked to pay when no enabled embodiment can carry the
                # task; one the gate cannot model is unassessed and never refused.
                verdict, fit = _rigid_verdict_and_fit(preparation=preparation, collision_path=collision,
                                                      native_root=native_root)
                result["capability_verdict"] = {"status": verdict["status"], "blockers": verdict["blockers"],
                    **({"risks": verdict["risks"]} if "risks" in verdict else {}),
                    "path": str(native_root / "capability_verdict.json"), "digest": verdict["verdict_digest"]}
                result["robot_fit_table"] = fit
                if verdict["status"] == "abstained":
                    result.update(status="needs_input", blockers=list(verdict["blockers"]))
            # Before ARTISAN and the GPU scene build: hold, never fail, when no catalog robot can do the
            # task here by a refusal the late checks also make; an unknown is flagged and goes on.
            precheck = _robot_fit_precheck(preparation=preparation, verdict=verdict, collision_path=collision,
                                           native_root=native_root)
        result["robot_fit_precheck"] = precheck
        if precheck.get("status") == "no_fit" and isinstance(precheck.get("hold_reason"), str):
            held = True
            prior = list(result.get("blockers") or []) if result.get("status") == "needs_input" else []
            result.update(status="needs_input", blockers=prior + [precheck["hold_reason"]])
        if isinstance(support_hold, str):
            # The object cannot rest where it is scored on the collider the scene would seal: no stage of it
            # can be scored, so nothing is registered, enqueued or paid for (``website_support_pad``).
            held = True
            prior = list(result.get("blockers") or []) if result.get("status") == "needs_input" else []
            result.update(status="needs_input", blockers=prior + [support_hold])
    return (preparation["status"] == "intake_ready" and not held
            and result.get("capability_verdict", {}).get("status") != "abstained")


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


def prepare_website_scene_handoff(*, descriptor: Mapping[str, Any], clean_plate: Mapping[str, Any],
                                 capture_root: Path, now: float,
                                 allow_operation_coverage_review: bool = False,
                                 allow_static_target_recovery: bool = False) -> dict[str, Any]:
    handoff_lease_checkpoint("website_scene_handoff_start")
    from .website_preparation_revision import workspace, bind_preparation
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
        result["digest"] = canonical_digest(result, digest_field="digest")
        write_json(handoff_path, result)
        return result

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
            # The Atlas lane held the task after its review, before buying the scene.
            from .website_early_task_hold import retain_task_preparation_hold
            video = clean_plate.get("input_video_path")
            held = retain_task_preparation_hold(capture_root=capture_root, task_context=context,
                                                source_video=Path(video) if isinstance(video, str) else None)
            if held is not None:
                return held
            result.update(status="awaiting_reconstruction" if atlas["status"] == "pending" else "needs_input",
                blockers=[atlas["reason"]], geometry_backend=None, provider_mutation_performed=False,
                visual_reconstruction_ready=False,
                atlas_scene_source={key: atlas.get(key) for key in ("status", "reason", "stage", "job_reason")},
                native_geometry_requirements={
                    "source_aligned_depth_and_camera_poses": "required_for_mask_lifting_and_scene_registration",
                    "measured_scale": "unknown", "atlas_required": True})
            result["digest"] = canonical_digest(result, digest_field="digest")
            write_json(handoff_path, result)
            return result
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
        # Capture truth applies to held families too. Reading this retained
        # manifest performs no mask, authoring or provider work.
        if (atlas.get("subject_origin") == "synthetic_design") != ("synthetic_subject" in removal):
            raise ValueError("website_subject_origin_conflict")
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
        gap = native_task_preparation_gap(context, **semantics_options)
        if gap is not None:
            # Retain the completed reconstruction and original task. Missing
            # adapter/scoring support cannot turn a different job into pick/place.
            result.update(geometry_backend=atlas["source_geometry"]["geometry_backend"],
                          source_geometry_digest=atlas["source_geometry"]["digest"])
            return retain_gap(gap)
        from .task_profile_contracts import task_profile_for
        from .website_task_semantics import effective_task_profile
        resolved = task_profile_for(effective_task_profile(task_context=context, **semantics_options))
        if resolved is not None and not resolved.requires_manipulation:
            # The robot moves itself: the scene stays as filmed, and the task compiles from
            # its own facts (floor, obstacles, the filmed route). Robot and policy are chosen
            # by the spec's requirements; with none runnable here the hold names each piece.
            from .task_spec_compiler import requirements as task_requirements
            from .website_locomotion_task import SPEC_FILE, compile_locomotion_spec, runnable_policy_gap, scene_facts
            need = task_requirements(locomote=resolved.requires_locomotion, manipulate=False)
            facts = scene_facts(base_scene=base, camera_path_levelled_m=atlas.get("camera_path_levelled_m") or [])
            spec = compile_locomotion_spec(task_context=context, facts=facts, requirements=need)
            write_json(root / SPEC_FILE, spec)
            if need["locomote"]:
                # The G1 checkpoints' goal and scene task, made from the compiled spec
                # (website_navigation_packet); the hold below names what is still unbuilt.
                from .website_locomotion_task import NAVIGATION_TASK_FILE
                from .website_navigation_packet import navigation_task_spec
                navigation = navigation_task_spec(spec, task_context=context)
                write_json(root / NAVIGATION_TASK_FILE, navigation)
                result.update(navigation_task_spec_path=str(root / NAVIGATION_TASK_FILE),
                              navigation_task_spec_digest=navigation["task_spec_digest"])
                # The scene as filmed, as Isaac files, on the controller and with no spend
                # (website_navigation_scene_assets). A failure is this job's missing piece.
                from .website_locomotion_task import navigation_pieces_built
                from .website_navigation_scene_assets import FILE as ASSETS_FILE, navigation_scene_assets
                outputs = {"navigation_task_spec": True, "navigation_scene_assets": False,
                           "navigation_scene_inputs": False, "navigation_intake_receipt": False}
                intake_refusal = None
                try:
                    assets = navigation_scene_assets(base_scene=base, output_root=root / "navigation_scene")
                except (ValueError, OSError) as exc:
                    result["navigation_scene_assets_error"] = str(exc)[:300]
                else:
                    write_json(root / ASSETS_FILE, assets)
                    result.update(navigation_scene_assets_path=str(root / ASSETS_FILE),
                                  navigation_scene_assets_digest=assets["digest"])
                    outputs["navigation_scene_assets"] = True
                    # The G1 lane's scene half: the G1 at the compiled start, its probe receipt and
                    # the registry row (website_navigation_scene_inputs). A failure is this job's
                    # missing piece (pipeline:g1_navigation_preparation).
                    authority = metadata.get("website_scene_execution_authority") or {}
                    written = _navigation_scene_inputs(
                        result=result, capture_root=capture_root, root=root, navigation=navigation, assets=assets,
                        task_context=context, authority=authority, now=now)
                    outputs["navigation_scene_inputs"] = written is not None
                    if written is not None:
                        # Its website intake and, once the intent is staged, the receipt binding its
                        # screening source (pipeline:navigation_source_identity, for this job only).
                        outputs["navigation_intake_receipt"], intake_refusal = _navigation_intake(
                            result=result, written=written, capture_root=capture_root, base_scene=base,
                            task_context=context, authority=authority)
                # The pieces built for this job are the outputs it has.
                built = navigation_pieces_built(outputs)
            result.update(geometry_backend=atlas["source_geometry"]["geometry_backend"],
                          source_geometry_digest=atlas["source_geometry"]["digest"],
                          compiled_task_spec_path=str(root / SPEC_FILE), compiled_task_spec_digest=spec["spec_digest"])
            from .website_locomotion_task import runnable_options
            options = runnable_options(need, built_pieces=frozenset(built)) if need["locomote"] else None
            if need["locomote"] and intake_refusal is not None:
                # The website refused the job's intake by name (an unconfirmed brief is
                # website_task_context_not_confirmed): the job holds by that name, never passes.
                from .website_locomotion_hold import navigation_intake_refusal_hold
                return retain_gap(navigation_intake_refusal_hold(
                    task_context=context, requirements=need, spec_digest=spec["spec_digest"],
                    refusal=intake_refusal, options=options, task_semantics=semantics_options["task_semantics"],
                    source_video_digest=source_video_digest))
            held = runnable_policy_gap(task_context=context, requirements=need, spec_digest=spec["spec_digest"],
                                       task_semantics=semantics_options["task_semantics"],
                                       source_video_digest=source_video_digest, options=options)
            if held is None and need["locomote"] and outputs["navigation_scene_inputs"] \
                    and outputs["navigation_intake_receipt"]:
                # Every runnable option is a G1 navigation checkpoint and its lane inputs exist: the
                # scene-inputs record (registered) and this job's intake receipt. Screening enters from
                # the registry row, so the job is handed off, development_only, claiming no result.
                from .website_locomotion_hold import navigation_preparation
                handed = navigation_preparation(
                    task_context=context, spec_digest=spec["spec_digest"], options=options,
                    scene_inputs=result["navigation_scene_inputs"], receipt=result["navigation_intake"]["receipt"],
                    task_semantics=semantics_options["task_semantics"], source_video_digest=source_video_digest)
                if handed is not None:
                    result.update(status="intake_ready", blockers=[], job_disposition="accepted",
                                  navigation_preparation=handed)
                    result["digest"] = canonical_digest(result, digest_field="digest")
                    write_json(handoff_path, result)
                    return result
            if held is None:
                # A runnable policy exists but its execution route is not built (a body with no lane
                # here): say so, typed.
                raise ValueError("website_locomotion_task_execution_route_missing")
            return retain_gap(held)
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
            result['digest'] = canonical_digest(result, digest_field='digest')
            write_json(handoff_path, result)
            return result
        if family is not None and geometry_only:
            # The fit check models one rigid subject; a family task is recorded, never assessed.
            result.update(status="needs_input", blockers=[GEOMETRY_ONLY_BLOCKER],
                          fit_check={"status": "not_run", "reason": "task_family_not_modelled"})
            result["digest"] = canonical_digest(result, digest_field="digest")
            write_json(handoff_path, result)
            return result
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
                from .website_scene_dispatch import binding_root, register_website_preparation
                native_path = root / "native_runtime_contract.json"
                write_json(native_path, preparation["task_evaluation_inputs"]["native_runtime_contract"])
                context_path = root / "task_context.json"
                write_json(context_path, context)
                result["source_registration"] = register_website_preparation(
                    preparation_path=root / "preparation.json", runtime_inputs_path=native_path,
                    task_context_path=context_path, root=binding_root(), now=now)
                if (metadata.get("website_scene_execution_authority") or {}).get("schema_version") == "website_scene_sponsorship.v1":
                    from .website_task_context import enqueue_website_prepared_scene
                    result["website_intake_outbox"] = enqueue_website_prepared_scene(
                        task_context=context, request=preparation["intake_request"])
            result["digest"] = canonical_digest(result, digest_field="digest")
            write_json(handoff_path, result)
            return result
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
        if inferred is not None:
            gap = native_task_preparation_gap(context, inferred_profile_id=inferred, **semantics_options)
            if gap is not None:
                # Catch a contradictory profile before coverage/spec callbacks.
                # Earlier mask work may have called a provider, so do not reset
                # provider_mutation_performed to false on this later path.
                result["source_geometry_digest"] = geometry["digest"]
                return retain_gap(gap)
        if synthetic is None:
            # Several task objects, or a soft one, under a one-rigid-object or one-opening
            # profile: held here, before coverage, spec research and authoring buy anything for it.
            gap = native_task_subject_gap(context, subjects=subjects, targets=targets, **semantics_options)
            if gap is not None:
                result["source_geometry_digest"] = geometry["digest"]
                return retain_gap(gap)
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
        authority = metadata.get("website_scene_execution_authority") or {}
        if not all(key in authority for key in ("max_total_spend_usd", "max_paid_attempts", "expires_at_epoch")):
            # Compile geometry even while execution is held. Zero authority
            # cannot validate as a paid intake request.
            authority = {"max_total_spend_usd": 0, "max_paid_attempts": 0, "expires_at_epoch": now}
        preparation = compile_website_scene_preparation(
            task_context=context, task_masks=clean_plate["task_masks"],
            removal_manifest=removal, source_geometry=clean_plate["source_geometry"],
            base_scene=base, output_root=root, spend=authority, now=now,
            **({"task_semantics": semantics} if semantics is not None else {}),
            **({"atlas_route": "geometry_only"} if geometry_only else {}),
        )
        preparation = bind_preparation(preparation, output_root=root, capture_root=capture_root, authority=authority)
        if ((preparation.get("collider_conditioning") or {}).get("status") == "conditioned"
                or (preparation.get("subject_excision") or {}).get("status") == "excised"):
            # Compile sealed a conditioned or cut collider into the base scene; later stages read that one.
            base = json.loads((root / "base_scene.json").read_text())
        result.update(status=preparation["status"], blockers=preparation["blockers"],
                      preparation_path=str(root / "preparation.json"), preparation_digest=preparation["digest"],
                      thumbnail=preparation["thumbnail"], base_scene_path=str(root / "base_scene.json"))
        if isinstance(preparation.get("native_task_preparation_gap"), Mapping):
            # Later source-bound evidence can also reveal a mismatch. A held
            # compiler output has no native intake and must not enter runtime preparation.
            return retain_gap(preparation["native_task_preparation_gap"])
        if geometry_only:
            result["fit_check"] = geometry_only_fit_check(preparation=preparation, base_scene=base,
                                                          native_root=root / "fit_check")
            result["digest"] = canonical_digest(result, digest_field="digest")
            write_json(handoff_path, result)
            return result
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
                                           task_masks=clean_plate["task_masks"]):
                from .website_scene_dispatch import binding_root, register_website_preparation
                context_path = root / "task_context.json"
                write_json(context_path, context)
                result["source_registration"] = register_website_preparation(
                    preparation_path=root / "preparation.json", runtime_inputs_path=native_root / "runtime_inputs.json",
                    task_context_path=context_path, root=binding_root(), now=now)
                if authority.get("schema_version") == "website_scene_sponsorship.v1":
                    from .website_task_context import enqueue_website_prepared_scene
                    result["website_intake_outbox"] = enqueue_website_prepared_scene(
                        task_context=context, request=preparation["intake_request"])
        except (ValueError, KeyError, TypeError, OSError, ImportError) as exc:
            # Preserve finished CPU outputs when a later handoff is held.
            result.setdefault("runtime_inputs", {}).update(status="awaiting_inputs", blockers=[str(exc)])
    except (ValueError, KeyError, TypeError, OSError) as exc:
        result.update(status="awaiting_inputs", blockers=[str(exc)])
    result["digest"] = canonical_digest(result, digest_field="digest")
    write_json(handoff_path, result)
    return result
