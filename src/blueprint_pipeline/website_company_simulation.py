"""The simulation package a deployment company downloads: one zip per job, made when the job's simulation exists.

Owner decision 2026-11-12: the simulation is core, because a deployment company cannot say "can do" without
testing on the twin; it downloads the simulation first and tests in its own tools (connecting its software to
Blueprint's simulation comes later and is not built here). This makes the package and publishes it exactly as the
company clip is published (``website_company_clip``):

- the source is the job's compiled team scene half, the simulation the trial itself runs: the team scene packet
  queue's ``status/<key>.json`` (``completed``, ``outcome: compiled``) names the half, whose
  ``scene/native-task-packet/`` holds ``native_task_arena_scene_plan.v1.json`` and the staged USD files under
  ``assets/``: the scene's look (``scene_appearance``), its collision (``scene_collision``) and the job's object
  (``task_object``) with its start pose, and the plan's ``task_spec.task_success_contract.criteria`` (the half's
  sealed record holds them when the plan does not, ``_with_sealed_rule``); a website half written without
  Blueprint's idle arm holds the same under ``scene/scene-package/`` (``team_scene_robot_absent``), and a half
  blocked only for its robot-team offer is a source all the same (owner decision 2026-11-15);
- the package holds ``scene.usda``, written by ``scene_package_stage`` from the plan: every part referenced by a
  relative path and placed at the plan's pose under a neutral prim name (``/World/Room``, ``/World/Object``, ...),
  in metres with the plan's up axis; the parts (``scene/room.*``, ``scene/room_collision.*``, ``object.*``), what
  each needs (renamed neutrally under ``scene/files/``), ``success_rule.json`` (the criteria and run facts, neutral
  only) and ``README.txt``. No record, manifest, ledger or receipt goes in, the plan included; no file name carries
  an id. Every outside or absolute USD reference is rewritten to a path inside the package, or the package is
  refused; a plan row that cannot be placed refuses it, naming the row's role;
- a look the website authored from a splat (``website_native_appearance``, found by ``look_source``) is written
  again from that source splat with today's writer, in the job's own frame, and ships twice: as that USD and as the
  standard 3DGS PLY common splat viewers open (``scene/room.ply``, ``particlefield_usd.write_placed_viewer_ply``),
  both from the same source; scene.usda does not use the PLY. A look whose source cannot be written again ships as
  its half holds it, with no PLY, and the README says so. Over the size limit, the PLY goes first, then the look
  (``build_package``, ``SIZE_STEPS``, carried between passes as ``size_over``);
- every file is read with the shared outbound scan (``outbound_file_scan``: ids in both forms, account ids,
  keys, host paths, storage, file and web links, bare web addresses, and the text form of every binary USD
  layer) before the zip is made; a finding refuses the package (``refused``), naming the file and the kind;
- the zip is deterministic (names in order, fixed modes, every member dated by the capture's own date, never the
  build's: ``website_company_member_date``), so the same inputs give the same digest; it
  goes to the private artifact store by digest under a name for the capture (withdrawal deletion finds every
  version by that name), the WebApp is told its identity through the signed ``company-simulation`` control
  operation and reads it through the signed ``/api/live-pipeline/website-company-simulations`` route.

Rights are the company clip's: not withdrawn, the delivery-time takedown gate allows it, the WebApp's live task
context grants consent, and ``sharing_statement.sharing_basis`` gives a basis. The not-permitted live read is
shared with the clip (its ``statement_check.json``), so a capture that may not be shared is read once a window,
not once per package.

It runs in the Atlas preview pass after the clip, before and after the paid Atlas steps (``company_simulation_tick``):
local files and ``pxr`` only, no model, provider or paid resource. Nothing here raises out of the pass or holds a
job: every outcome is a returned status, retried on the next pass.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import logging
import math
import os
import re
import shutil
import stat
import time
import zipfile
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import Any

from . import company_share_permission as permission
from . import website_company_clip as clip
from .atlas_automation_plan import JOB_OBJECT_RULE
from .atlas_unscaled_room import SCALE_NOT_MEASURED
from .common import write_json
from .decision_evidence_contracts import canonical_digest, cross_runtime_canonical_digest
from .outbound_file_scan import capture_site_identity
from .website_atlas_preview_store import capture_name_matches_request, capture_withdrawn
from . import website_half_preparation as half_preparation
from . import website_destination_unheld as unheld
from . import website_library_notes as library_notes
from . import website_object_variations as variations
from .website_half_preparation import generated_patch, look_views, surface_estimated
from .website_build_plan_notes import build_plan_notes
from .website_company_look import look_ply, look_source, reauthor_look
from .website_company_member_date import ZIP_EPOCH, member_date

logger = logging.getLogger(__name__)

MANIFEST_SCHEMA = "website_company_simulation_manifest.v1"
PACKAGE_SCHEMA = "website_company_simulation.v1"
ACCEPTANCE_SCHEMA = "website_company_simulation_acceptance.v1"
DIRECTORY = ("pipeline", "company_simulation")
MANIFEST_FILE = "company_simulation.json"
PACKAGE_FILE = "simulation.zip"
STAGING = ".staging"
#: Withdrawal deletion holds this lock, so no pass publishes between its listing and deletion.
LOCK_FILE = "company_simulation.lock"
ARTIFACT_KIND = "website-company-simulation"
FORMAT = "zip_usd_scene_object_success_rule"
CONTENT_TYPE = "application/zip"
#: The one real compiled half on the host (capture-447, 2026-11-12) stages 211 MiB of USD (appearance 124 MiB,
#: collision 83 MiB, object 3.8 MiB); its configured bundle's text collision alone is 278 MiB. 512 MiB is about
#: 2.4 times the real scene, the size the first-site team runbook already allows for one flattened USD, and a
#: quarter of what the website accepts (2 GiB). Measured on the zip, the file a company downloads: the WebApp
#: checks the same number (``size_bytes``, the zip's) when it accepts the package.
MAX_PACKAGE_BYTES = 512 * 1024 * 1024
#: Source files above this are refused unbuilt (copying, scanning and zipping them could not end within one build
#: and could not come under ``MAX_PACKAGE_BYTES``); below it the zip decides. Why: a room whose look is our own
#: draft scan (``depth_fused_draft_fallback``, 6.2 million points, no bound) ran to 562 MB of source files, which
#: the old rule, measured on them, refused without zipping; built, its zip is 342 MB.
MAX_SOURCE_BYTES = 2 * 1024 * 1024 * 1024
#: Recorded in every manifest this builder writes: a size refusal without it was made under the old rule (source
#: files over ``MAX_PACKAGE_BYTES``) and is built once more under this one; a size refusal with it stands.
SIZE_RULE = "zip_bytes.v1"
#: When the zip with the room's look is over the limit, the package is built once more without the look, with the
#: collision file shown in its place, and says so.
LOOK_LEFT_OUT = ("The look file is not in this package: with it the download would have been larger than "
                 "Blueprint's size limit. The collision file, which holds the room's shape, is shown in its place.")
#: A look written from its source splat (``look_source``) also ships as the standard 3DGS PLY common splat viewers
#: open. It is the first thing a zip over the limit gives up: built once more without it, then without the look.
PLY_LEFT_OUT = ("The look file is in this package as USD only: with a splat PLY of it as well, the download would "
                "have been larger than Blueprint's size limit.")
#: A look whose source splat could not be written again here (missing, changed, over ``website_company_look.REAUTHOR_MAX_SOURCE_BYTES``
#: or refused by the writer) ships as its half holds it, with no PLY. Said as what the reader gets, not how Blueprint
#: wrote it.
LOOK_AS_FIRST_WRITTEN = "The look file is included as USD only, in an earlier version."
#: ... and, when that first writing predates 2026-11-18 (``look_source``'s ``first_rotations_scrambled``):
ROTATIONS_STORED_WRONG = ("In that version each Gaussian's rotation is stored in the wrong order, so the look may not "
                          "show correctly.")
#: Moves whenever what the builder writes changes, so a package built or refused by the earlier builder is built once
#: more at the first pass after the release. v2 (2026-11-16): the layer's flatten note naming a host file is dropped
#: (``place``); a package v1 refused for that note waited six hours for its retry. v3 (2026-11-18): a look is written
#: again from its source splat with the fixed writer and also ships as a standard 3DGS PLY. v4 (2026-11-21): a plain
#: README. v5 (2026-11-25): plain zip headers; ZIP64 in the local header alone made macOS refuse the zip (Error 640).
#: v6 (2026-11-25): members dated by the capture, not 1980; an object placed where its box meets the room says so;
#: the object's simulated variations (``variations.usda``, ``variations.json``); the mass range's basis in words.
#: v7: variations seeded only by what the revision covers, so v6's samples are drawn once more.
BUILDER_REVISION = "website_company_simulation_builder.v7"
OVER_CAP = "website_company_simulation_over_size_cap"
BUILD_TIMED_OUT = "website_company_simulation_build_timed_out"
#: One build's deadline. A real compiled half (211 MiB of USD) copies, scans by spec walk, proves and zips in
#: well under this; one that does not is recorded and tried again after the live-statement window.
BUILD_DEADLINE_SECONDS = 900
#: The pass starts no capture after this. It runs after the clip's pass (up to 900 s plus one encode of at most
#: 900 s), before the Atlas steps and again after them; 600 s plus one build keeps each run inside the unit's 2 h.
TICK_BUDGET_SECONDS = 600
QUEUE_ROOT_ENV = "BLUEPRINT_TEAM_SCENE_PACKET_QUEUE_ROOT"
DEFAULT_QUEUE_ROOT = "/var/lib/blueprint/pipeline-control-plane/team-scene-packets"
PACKET_DIRECTORIES = ("native-task-packet", "native-task-packet-three-view")
PLAN_FILE = "native_task_arena_scene_plan.v1.json"
#: A website half written without Blueprint's idle arm (``team_scene_robot_absent``): its own folder and plan.
SCENE_PACKAGE = ("scene-package", "scene_package_plan.v1.json")
#: A half compiled and recorded whose robot-team offer post failed (``team_scene_packet_worker``) is blocked only
#: for that offer, which no company package needs: the package is made from it all the same.
OFFER_ONLY_BLOCK = ["team_scene_offer_publication_failed"]
#: What ``README.txt`` says was checked (``physics_check``), and the other plain notes it may carry: the one place
#: for that wording. A scene whose stage 5 ran in Isaac Sim, or whose stage 5 cannot be read, keeps the README
#: packages always had, which names no check. One whose stage 5 ran no simulator says so, and says the object's
#: static USD check passed only when the revision's own stage-4 record shows it passed (``static_check_passed``).
STATIC_CHECK_PASSED = ("The object's file passed Blueprint's static USD physics check (one rigid body, collision "
                       "shapes, mass and friction within set ranges).")
NO_SIMULATOR = "Blueprint did not run this scene in Isaac Sim or any other simulator."
#: A jointed object whose stage 5 ran no simulator: its joint is exactly as authored.
JOINT_AS_AUTHORED = ("The joint's range is from the object's authored file and the success interval is the "
                     "job's share of that range; no simulator moved the joint.")
#: The same, for a half whose rule is withheld (``RULE_WITHHELD_REASONS``): no rule, so no success interval.
JOINT_AS_AUTHORED_NO_RULE = "The joint's range is from the object's authored file; no simulator moved the joint."
PHYSICS_CHECK_WORDS = {
    "isaac_sim": (), "not_run": (NO_SIMULATOR,),
    "not_run_static_check_passed": (STATIC_CHECK_PASSED, NO_SIMULATOR),
    "not_run_joint_as_authored": (NO_SIMULATOR, JOINT_AS_AUTHORED),
    "not_run_joint_as_authored_no_rule": (NO_SIMULATOR, JOINT_AS_AUTHORED_NO_RULE),
}
#: ``scene_packet.FLATTEN_PROVENANCE``, copied (``test_scene_packet_parity`` pins them equal): the import graph
#: ``test_live_pipeline_import_isolation`` walks reaches the hot lane through ``scene_packet``.
FLATTEN_PROVENANCE = "Generated from Composed Stage of root layer "
#: The stage-4 record of a jointed object (``task_evaluation_articulated_open_close_native_adapter``).
ARTICULATED_STATIC_SCHEMA = "task_evaluation_articulated_replacement_static_qualification.v1"
#: ``{count}`` movable items the site agreed to move out of the arm's way (``robot_access_removal``): a count only,
#: said in the singular for one (``_counted``).
FURNITURE_REMOVED = ("{count} movable items the site agreed to move are not in the collision file; they still "
                     "appear in the look file.")
FURNITURE_REMOVED_ONE = ("1 movable item the site agreed to move is not in the collision file; it still appears in "
                         "the look file.")
#: The same, for a package built without the look file (``LOOK_LEFT_OUT``).
FURNITURE_REMOVED_NO_LOOK = "{count} movable items the site agreed to move are not in the collision file."
FURNITURE_REMOVED_NO_LOOK_ONE = "1 movable item the site agreed to move is not in the collision file."
#: Where a generated estimate of a surface the video did not show sits (``generated_patch``, read from the base scene
#: the job's scene handoff recorded). The collision mesh is trimmed after registration and nothing checks that the
#: generated samples survive it, so each sentence names the scan the collision file was made from.
#: ``collision_only`` (owner 2026-11-17): the look is Atlas's refined splat of the filmed frames.
GENERATED_COLLISION_ONLY_NOTE = ("The collision file was made from a scan that also holds a generated estimate of "
                                 "a surface the video did not show; that estimate was not filmed, and it is not in "
                                 "the look file.")
#: ``look_and_collision``: no refined splat could be had, so the look is that scan itself.
GENERATED_LOOK_AND_COLLISION_NOTE = ("The look file, and the scan the collision file was made from, hold a "
                                     "generated estimate of a surface the video did not show; that estimate was not "
                                     "filmed.")
#: Either, for a package built without the look file (``LOOK_LEFT_OUT``).
GENERATED_NO_LOOK_NOTE = ("The collision file was made from a scan that holds a generated estimate of a surface the "
                          "video did not show; that estimate was not filmed.")
#: The destination's surface did not show in the scan and was estimated from its visible edges, then added to the
#: collision file only (``website_unscanned_surface``; read from the base scene the job's scene handoff recorded).
SURFACE_ESTIMATED_NOTE = ("The destination shelf or surface the job's object is placed on did not show in the scan "
                          "(glass and other see-through or shiny surfaces often do not); it was estimated from its "
                          "visible edges in the video. That estimate is in the collision file, not in the look file.")
#: The same, for a package built without the look file (``LOOK_LEFT_OUT``).
SURFACE_ESTIMATED_NO_LOOK_NOTE = ("The destination shelf or surface the job's object is placed on did not show in "
                                  "the scan (glass and other see-through or shiny surfaces often do not); it was "
                                  "estimated from its visible edges in the video. That estimate is in the collision "
                                  "file.")
#: The same in a room package (``website_scene_handoff.write_room_package``), which reads the base scene compile sealed
#: with the estimate (``website_scene_handoff``) but holds no object to place on it.
SURFACE_ESTIMATED_ROOM_NOTE = ("The job's destination shelf or surface did not show in the scan (glass and other "
                               "see-through or shiny surfaces often do not); it was estimated from its visible edges in "
                               "the video. That estimate is in the collision file, not in the look file.")
SURFACE_ESTIMATED_ROOM_NO_LOOK_NOTE = ("The job's destination shelf or surface did not show in the scan (glass and "
                                       "other see-through or shiny surfaces often do not); it was estimated from its "
                                       "visible edges in the video. That estimate is in the collision file.")
#: For the destination piece of an estimated surface left out only for ``task_destination_surface_contact_required``,
#: with why in ``website_destination_unheld``'s words where compile recorded it (job move-the-lamp-2, 24 Nov).
NOT_PLACED_ON_ESTIMATED_SURFACE = "Blueprint could not place the job's object on that estimated surface."
DESTINATION_CONTACT_CODE = "task_destination_surface_contact_required"
#: The job's object was sized from its own points across the frames, its masks' spill left out (``website_object_extent``,
#: read from the preparation the job's scene handoff names). A room package holds no object and says nothing of it.
SIZE_FROM_ITS_POINTS_NOTE = ("The job's object's size was estimated from the video's points of the object that the "
                             "frames agree on: points its masks also took in from what lies under or behind it were "
                             "left out.")
#: No surface at the destination holds the object's whole footprint: its centre is over the surface, part of it over
#: the edge (``website_destination_overhang``, read with the size notes; only when something does hang over). Never
#: in a room package.
DESTINATION_OVERHANG_NOTE = ("The job's object's footprint is larger than the supported part of the surface where it "
                             "is placed: its centre is over the surface and part of its footprint hangs over the "
                             "edge.")
#: The object's exact model has a published size that disagrees with the video's estimate; the file keeps the estimate.
SIZE_DISAGREES_WITH_PUBLISHED_NOTE = ("The published size of the job's object's exact model disagrees with the size "
                                      "estimated from the video; the object's file keeps the video's estimate.")
#: The look's refined splat also holds generated views (``atlas_look_views``, ``look_views``): never for a package
#: built without the look file.
GENERATED_VIEWS_LOOK_NOTE = ("Parts of the look file were made from generated views of areas the video saw from "
                             "few angles, checked against the filmed frames and their depth.")
#: The look file failed its quality gate (``team_scene_robot_absent.APPEARANCE_QUALITY``) and ships as it is.
APPEARANCE_QUALITY_NOT_PASSED = ("The look file did not pass Blueprint's image quality check; it is included as "
                                 "it is.")
#: The half's support record (``task_support_pad``, ``rigid_support_pad.support_rest_check``, read on the CPU) is
#: ``held``, chosen by its typed reasons: only a corner of the destination region fails
#: (``rigid_support_goal_region_rest_outside_success_window``), or else the start or the goal centre does. A CPU
#: reading of the collider, never a simulator's.
SUPPORT_REGION_NOT_HELD = ("At part of the destination area the surface in the collision file does not hold the "
                           "object where the success rule counts it as placed: it would tip off an edge or rest "
                           "higher or lower than the rule allows. Blueprint did not test this.")
SUPPORT_NOT_HELD = ("Where the object starts or is placed, the surface in the collision file does not hold it as "
                    "the success rule expects. Blueprint did not test this.")
#: A room package (``website_scene_handoff.write_room_package``): a job with no object of its own in the package. Its
#: README says what the room is, whether the rule is Blueprint's assumption or there is none, and where the job's
#: objects are. The site's own words are not in it: companies read them on the job's page, where the WebApp removes
#: the site's name, address and contact details first (``companyJobPage.ts``), which the Pipeline cannot repeat.
ROOM_NOT_RUN = "The room is as reconstructed from the video; Blueprint ran no robot, policy or simulator in it."
GOAL_ESTIMATED = ("The goal region was estimated from the video; its radius and how long the robot must stay in it "
                  "are assumed, not stated by the site.")
NO_RULE = ("This package includes no success rule from Blueprint for this job; the job as the site described it is "
           "on the job's page.")
OBJECTS_TAKEN_OUT = "{count} objects the job involves were taken out of the room scan and are not in this package."
OBJECTS_TAKEN_OUT_ONE = "1 object the job involves was taken out of the room scan and is not in this package."
OBJECTS_IN_SCAN = "The job's objects are part of the room scan; they are not separate files."
NOT_FOUND = "Blueprint did not find what the job is about in the video."
NOT_LOCATED = ("Blueprint saw the job's object in the video but could not locate it precisely, so it is not a "
               "separate file in this package.")
#: The job's object could not be prepared (``website_scene_handoff``): a Gemini step that reads it from the video was
#: never answered within its bound, or Blueprint's credits with Gemini were depleted when the job's window ended.
OBJECT_NOT_PREPARED = ("Blueprint could not prepare the job's object as a separate file: a step that reads it from "
                       "the video could not be completed.")
OBJECT_NOT_PREPARED_FLAGS = frozenset({"website_gemini_unanswered_dispatches_exhausted",
                                       "website_gemini_provider_credit_depleted"})
#: The job's object has no surface to start on in the scan (compile, ``website_task_preparation``).
OBJECT_NOT_PLACED = ("Blueprint did not place the job's object: no surface to start it on was found in the "
                     "room scan.")
OBJECT_NOT_PLACED_FLAGS = frozenset({"support_surface_not_found_under_subject",
                                     "website_synthetic_subject_start_unplaceable"})
#: Any other blocker a room package shipped for (``website_scene_handoff.goes_on``), from the record's
#: ``not_prepared``: the piece of the job it left out, never its code (the record keeps the codes) or a reason.
NOT_PREPARED = "Blueprint could not prepare the job's {piece}."
NOT_PREPARED_PIECES = ("object", "destination", "success rule")
#: The same, when the job's spending cap is why (``website_scene_handoff.shipped_code`` ships the plain
#: ``spending_cap_reached``, ``website_job_spend_cap.SPENDING_CAP_REACHED``, and ``piece_of`` names the piece): every
#: code the piece left out for is the cap (AGENTS.md: the README says "the job reached its limit on paid work").
NOT_PREPARED_AT_CAP = "Blueprint could not prepare the job's {piece}: the job reached its limit on paid work."
SPENDING_CAP_REACHED = "spending_cap_reached"
#: How many fields of the criteria the allow-list left out (``scene_package_stage.neutral_criteria``).
FIELDS_LEFT_OUT_LINES = ["  criteria_fields_left_out counts the fields Blueprint left out of criteria: free text, which",
                         "  could carry the site's own words, and any field outside the names it shares."]
#: The room is as filmed: the planner's removals were not applied (``website_atlas_pipeline.job_object_review``).
NOTHING_TAKEN_OUT = "Nothing was taken out of the room scan."
NOT_FILLED_IN = "Parts of the room the video did not show were not filled in."
ROOM_RECORD_SCHEMA = "blueprint.room_package_inputs.v1"
#: A one-object job no task profile models (``website_scene_handoff``; the gaps of its ``ROOM_GAPS``, pinned equal
#: by a test): its object is built on the rigid path and its half's sealed record says the rule is withheld, and
#: why (``team_scene_packet``: no rule in the plan or the record). Only such a half, with its object, ships
#: without a rule, and says so (``NO_RULE``); a rule beside the reason refuses it.
RULE_WITHHELD_REASONS = frozenset({"website_native_task_profile_preparation_required",
                                   "website_task_operation_coverage_unresolved",
                                   # The model chose the object built with no rule (``website_build_choice``).
                                   "website_build_choice_rule_withheld"})
#: The stage-4 record a rigid website object passes (``task_object_artisan_candidate``, ``_usd_findings``).
RIGID_STATIC_SCHEMA = "task_evaluation_rigid_replacement_static_qualification.v1"
_USD = frozenset({".usd", ".usda", ".usdc", ".usdz"})
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
_RIGHTS_REFUSALS = re.compile(r"(source_revoked|consent_revoked|consent_expired|website_capture_withdrawn"
                              r"|capture_revoked_or_expired|website_company_simulation_not_permitted"
                              r"|website_company_clip_not_permitted)\b")
#: Record fields whose string values are account or owner ids, and those that are keys.
_ACCOUNT_KEYS = frozenset({"accepted_by", "account_id", "user_id", "uid", "owner_id", "owner_uid", "created_by",
                           "submitted_by", "approved_by", "requested_by", "buyer_account_id", "organization_id"})
_KEY_KEYS = frozenset({"delivery_key", "producer_delivery_key", "upload_token", "token", "session_token"})


class CompanySimulationUnavailable(ValueError):
    """No archived package this caller may read; never says why."""


class CompanySimulationBusy(RuntimeError):
    """Another pass (or withdrawal deletion) holds this capture's package lock."""


class PackageRefused(ValueError):
    """The package may not leave: a scan finding, an outside reference or an unusable scene file."""

    def __init__(self, reason: str, detail: str = ""):
        super().__init__(reason)
        self.reason, self.detail = reason, detail


class ShapeRefused(PackageRefused):
    """The half's own scene cannot be staged (a ``StageRefused``), or its object may not leave (a library model's
    credit cannot be read: ``website_library_notes.refusal``): the job's room floor ships instead, flagged."""


def package_filename(capture_id: str, request_id: str) -> str:
    """The archive name: the walkthrough's own, or a continuation scoped to its job."""
    if not capture_name_matches_request(request_id, capture_id):
        raise CompanySimulationUnavailable("website_company_simulation_unavailable")
    return f"{capture_id}.zip" if capture_id.startswith("walkthrough-") else f"{request_id}-{capture_id}.zip"


def _read(path: Path) -> dict[str, Any]:
    return permission.read_json(Path(path))


def _sha256_file(path: Path) -> str:
    return permission.sha256_file(Path(path))


# --- where the job's simulation is -------------------------------------------------------------------------------

def _queue_root() -> Path:
    return Path(os.environ.get(QUEUE_ROOT_ENV) or DEFAULT_QUEUE_ROOT)


def _inside(base: Path, path: Path) -> Path | None:
    """``path`` resolved, when it is a regular file under ``base`` reached through no link."""
    try:
        resolved = path.resolve(strict=True)
    except OSError:
        return None
    if any(item.is_symlink() for item in (path, *path.parents) if item.is_relative_to(base)) \
            or not resolved.is_relative_to(base.resolve()) or not resolved.is_file():
        return None
    return resolved


#: The team scene packet worker's two outcomes of a completed status that name a half
#: (``team_scene_packet_worker``: ``_compiled_status`` writes ``compiled``, a reused half ``already_compiled``).
SUCCESS_OUTCOMES = frozenset({"compiled", "already_compiled"})


def simulation_source(capture_root: Path, *, queue_root: Path | None = None) -> dict[str, Any] | None:
    """The newest compiled team scene half of this capture, as its plan and staged files name it, or None.

    None while no half is compiled (the job is never held for it). A half whose plan or files do not read is
    None too, and is tried again on the next pass.
    """
    root, queue = Path(capture_root), Path(queue_root or _queue_root())
    identity = permission.capture_identity(root)
    statuses = []
    try:
        candidates = sorted((queue / "status").glob("*.json"))
    except OSError:
        return None
    for path in candidates:
        status = _read(path)
        # A reused half (a duplicate request, or a retry that found its half) is written as ``already_compiled``;
        # either way the half itself is checked below (``_half``), never trusted for the word.
        if ((status.get("status") == "completed" or (status.get("status") == "blocked"
                                                       and status.get("blockers") == OFFER_ONLY_BLOCK))
                and status.get("outcome") in SUCCESS_OUTCOMES
                and status.get("capture_id") == identity["capture_id"]
                and Path(str(status.get("capture_root") or "")) == root
                and isinstance(status.get("team_scene_inputs_path"), str)):
            statuses.append((path.stat().st_mtime_ns, path.name, status))
    for _mtime, _name, status in sorted(statuses, reverse=True):
        source = _half(Path(status["team_scene_inputs_path"]).parent, status, capture=root)
        if source is not None:
            return source
    # Else the room package the scene handoff wrote (``website_scene_handoff.write_room_package``), when sealed.
    handoff = _read(root / "pipeline" / "website_scene_preparation" / "handoff.json")
    path = (handoff.get("room_package") or {}).get("path")
    record = (_inside(root, Path(path)) if isinstance(path, str) and handoff.get("status") == "room_package_written"
              and handoff.get("digest") == canonical_digest(handoff, digest_field="digest") else None)
    source = (_half(record.parent, {"team_scene_inputs_path": str(record)},
                    generated=generated_patch(root), look=look_views(root), capture=root) if record is not None else None)
    return source if source is not None and source.get("room") is not None else None


#: The room record a routed job's handoff seals for its package floor (``website_scene_handoff``), flagged so.
FLOOR_FLAG = "object_not_built"
#: The floor's reason while the job's object is still being built: the room ships now, flagged, not after the build.
OBJECT_IN_PROGRESS = FLOOR_FLAG + ":object_build_in_progress"
#: The scene intents (``task_evaluation_scene_intake``); tests point it elsewhere.
INTENT_ROOT: Path | None = None
DEFAULT_INTENT_ROOT = "/var/lib/blueprint/pipeline-control-plane/task-evaluation-scene-intents"
#: The scene stage's hold store (``scene_stage_holds``: its queue root, ``holds/``, one file per scene intent), read
#: here as a file: that module and the team-scene worker reach the GPU lane, which this builder never imports.
SCREENING_ROOT: Path | None = None
SCREENING_ROOT_ENV, DEFAULT_SCREENING_ROOT = ("BLUEPRINT_BETA_POLICY_SCREENING_QUEUE_ROOT",
                                              "/var/lib/blueprint/pipeline-control-plane/beta-policy-screening")


def _scene_hold(intent_id: str) -> dict[str, Any]:
    """The hold the scene stage told the website for this intent (``scene_stage_holds.settle_hold``), or {}."""
    import hashlib
    root = Path(SCREENING_ROOT or os.environ.get(SCREENING_ROOT_ENV) or DEFAULT_SCREENING_ROOT)
    name = "hold-" + hashlib.sha256(intent_id.encode("utf-8")).hexdigest()[:32] + ".json"
    publication = _read(root / "holds" / name).get("publication")  # one file per intent; released has no reason
    return (dict(publication) if isinstance(publication, Mapping) and publication.get("hold_digest")
            == cross_runtime_canonical_digest(publication, digest_field="hold_digest") else {})


def object_path_stopped(root: Path, handoff: Mapping[str, Any], *, queue_root: Path, now: float) -> str | None:
    """Why the job's object is not in its package (``object_not_built:<code>``: its room floor ships, flagged), or
    None when privacy, rights or consent ended its path, or ``floor_unclassified:<code>`` when its records cannot be
    read or tied to this job (nothing ships; the pass names the capture).

    Owner decision 2026-11-23: once the room is built and the sharing rights hold, the room floor ships flagged at
    once, whatever the object path says or fails to say; retries continue and a package with the object replaces it
    (``package_source``). So an object still being built (``OBJECT_IN_PROGRESS``), an intent not made yet, a half
    neither queued nor started, and a block with no hold recorded (its timed retries may still run) all ship the
    floor now; ``<code>`` names what the records show, never a guessed reason. Consent, withdrawal, takedown and
    the live sharing rights are checked separately at publication (``_publish``) and stop it there.

    Read through the job's own scene intent, the one its handoff's intake receipt names and whose request it
    sealed, never the capture's other intents (a release retires and replaces them). An intent bound to another
    request, or an outcome that does not read, may hide a withdrawal or a consent ending for this job: nothing ships.
    ``now`` is unused (the floor no longer waits); it stays for the callers.
    """
    from .task_evaluation_scene_outcome import SceneOutcomeError, read_outcome

    del now
    receipt = handoff.get("website_intake_outbox")
    intent_id = receipt.get("id") if isinstance(receipt, Mapping) else None
    directory = Path(INTENT_ROOT or DEFAULT_INTENT_ROOT) / str(intent_id)
    if not isinstance(intent_id, str) or not intent_id.startswith("scene-") or not (directory / "intent.json").is_file():
        return FLOOR_FLAG + ":intent_missing"  # not made yet, never made or refused at intake: the room ships
    intent = _read(directory / "intent.json")
    if (intent.get("intent_id") != intent_id or not isinstance(intent.get("request"), Mapping)
            or cross_runtime_canonical_digest(intent["request"]) != receipt.get("request_digest")):
        return "floor_unclassified:intent_unbound"
    try:
        outcome = read_outcome(directory, intent)
    except SceneOutcomeError as exc:
        return "floor_unclassified:" + str(exc)
    if outcome is not None:
        # Privacy, rights and consent never ship it; any other ending, or a wait on a person, does.
        if outcome["kind"] == "withdrawn" or outcome["reason_code"] == "site_consent_required":
            return None
        return FLOOR_FLAG + ":" + outcome["reason_code"]
    progression = _read(directory / "progression.json")
    if progression.get("status") != "completed":
        return OBJECT_IN_PROGRESS  # queued, running, a capacity wait or a receipt still pending
    asked = ((progression.get("state") or {}).get("scene_preparation_completion") or {}).get("team_scene_request")
    asked = asked if isinstance(asked, Mapping) else {}
    if asked.get("status") in ("not_configured", "not_a_website_scene", "not_enqueued"):
        return FLOOR_FLAG + ":" + str(asked.get("reason") or asked["status"])
    key = asked.get("request_key")
    status = _read(Path(queue_root) / "status" / f"{key}.json") if isinstance(key, str) else {}
    if not status:
        if any((Path(queue_root) / state / f"{key}.json").is_file() for state in ("pending", "processing")):
            return OBJECT_IN_PROGRESS  # queued for the worker, which writes its status when it claims the request
        return FLOOR_FLAG + ":half_not_enqueued"
    if status.get("status") in ("pending", "processing"):
        return OBJECT_IN_PROGRESS
    if status.get("status") == "skipped":
        return FLOOR_FLAG + ":" + str(status.get("reason") or "team_scene_skipped")
    if status.get("status") == "blocked":
        # Held for good under the running release: the scene stage has told the website so (``final_block``). With
        # no hold its timed retries may still run; the room ships meanwhile and the half replaces it if it comes.
        hold = _scene_hold(intent_id)
        return FLOOR_FLAG + ":" + str(hold.get("reason_code") or "team_scene_blocked_unheld")
    return FLOOR_FLAG + ":team_scene_" + str(status.get("status"))


def _floor_source(root: Path, *, queue_root: Path, now: float, reason: str | None = None,
                  notes: dict[str, Any] | None = None) -> dict[str, Any] | None:
    """The job's room floor as a source, when its object path stopped (``reason`` given: its half was refused)."""
    handoff = _read(root / "pipeline" / "website_scene_preparation" / "handoff.json")
    floor = handoff.get("room_floor") if isinstance(handoff.get("room_floor"), Mapping) else {}
    blocked = floor.get("blockers") if handoff.get("status") != "intake_ready" and floor.get("blockers") == handoff.get("blockers") else None  # ``floor_ships``
    record = (_inside(root, Path(str(floor.get("path") or ""))) if (handoff.get("status") == "intake_ready" or blocked)
              and handoff.get("digest") == canonical_digest(handoff, digest_field="digest") else None)
    if record is None or _read(record).get("inputs_digest") != floor.get("inputs_digest"):
        return None
    reason = reason or (blocked and f"{FLOOR_FLAG}:{','.join(map(str, blocked))}") or object_path_stopped(root, handoff, queue_root=queue_root, now=now)
    if reason is None or not reason.startswith("object_not_built:"):
        if reason is not None and notes is not None:
            notes["reason"] = reason  # cannot be read or bound to this job: nothing ships, the pass says why
        return None
    source = _half(record.parent, {"team_scene_inputs_path": str(record)},
                   generated=generated_patch(root), look=look_views(root), capture=root)
    if source is None or source.get("room") is None or FLOOR_FLAG not in (_read(record).get("flags") or []):
        return None
    return {**source, "source_kind": "floor", "floor_reason": reason}


def _delivered_half(manifest: Mapping[str, Any]) -> bool:
    """The package now with the companies is a compiled half's (older manifests: it ships the job's object)."""
    kind = manifest.get("source_kind") or ("half" if "Object" in ((manifest.get("source") or {}).get("files") or {})
                                           else "room")
    return kind == "half" and bool(manifest.get("package")) and bool(manifest.get("delivered"))


def package_source(root: Path, *, queue_root: Path | None = None, manifest: Mapping[str, Any] | None = None,
                   max_bytes: int = MAX_PACKAGE_BYTES, now: float | None = None,
                   notes: dict[str, Any] | None = None) -> dict[str, Any] | None:
    """What this pass builds: the compiled half, else the handoff's room package, else the room floor.

    A half the last build refused for its own shape (``refused_half``, that revision) gives way to the floor
    until a new half revision arrives; a delivered half is never replaced by a floor.
    """
    queue = Path(queue_root or _queue_root())
    manifest = manifest or {}
    source = simulation_source(root, queue_root=queue)
    if source is not None and source.get("room") is not None:
        return {**source, "source_kind": "room"}
    refused = manifest.get("refused_half") if isinstance(manifest.get("refused_half"), Mapping) else {}
    if refused.get("reason") == OVER_CAP and refused.get("size_rule") != SIZE_RULE and not manifest.get("delivered"):
        refused = {}  # refused for size under the old rule, and its floor not delivered: the half is built once more
    at = refused.get("refused_at_epoch")
    if refused.get("reason") == BUILD_TIMED_OUT and not (
            isinstance(at, int) and 0 <= (time.time() if now is None else now) - at
            < permission.LIVE_STATEMENT_RECHECK_SECONDS):
        # A timeout is about the host that day, not the files: the floor stands in for the half within the window,
        # and the half is tried again once a window (the window ``_publish`` gives every timed-out build).
        refused = {}
    if source is not None and refused.get("simulation_revision") != simulation_revision(source, max_bytes=max_bytes):
        return {**source, "source_kind": "half"}
    if _delivered_half(manifest):
        return None
    floor = _floor_source(root, queue_root=queue, now=time.time() if now is None else now, notes=notes,
                          reason=("object_not_built:" + str(refused.get("reason"))) if source is not None else None)
    # No floor to stand in for a refused half (a job routed before floors, or none written): its refusal stands.
    return floor if floor is not None or source is None else {**source, "source_kind": "half"}


def object_estimates(capture_root: Path) -> Mapping[str, Any] | None:
    """The object's mass and friction ranges in the preparation the job's handoff names (a half reads its own:
    ``website_half_preparation``)."""
    return half_preparation.object_estimates(half_preparation.preparation(capture_root))


def success_basis(capture_root: Path) -> dict[str, str] | None:
    """Where each number of the rule came from, in the preparation the job's handoff names (a half reads its own)."""
    return half_preparation.success_basis(half_preparation.preparation(capture_root))


def size_notes(capture_root: Path) -> dict[str, bool]:
    """What the preparation the job's handoff names says of the object's size (a half reads its own)."""
    return half_preparation.size_notes(half_preparation.preparation(capture_root))


def _half(scene: Path, status: Mapping[str, Any], *, generated: str | None = None, look: bool = False,
          capture: Path | None = None) -> dict[str, Any] | None:
    """The half's plan and shipped files; a row that cannot be placed is kept as ``refusal`` naming its role. The
    look's row carries the splat it was authored from (``look_source``) when ``capture``'s records name one. With
    ``capture``, a half's notes are read from the preparation it was built from (``website_half_preparation``); a
    room's are its handoff's (``generated``, ``look``)."""
    from .scene_package_stage import StageRefused, neutral_rule, placements, up_axis
    if not scene.is_absolute() or scene.is_symlink() or not scene.is_dir():
        return None
    record, notes = _sealed_record(scene, status), _record_notes(scene, status)
    told = half_preparation.half_disclosures(capture, record) if capture is not None and "room" not in notes else {}
    estimates, basis = told.pop("estimates", None), told.pop("basis", None)
    assumed = "room" not in notes and half_preparation.thresholds_assumed(record)  # the site confirmed none of them
    # ... when it was compiled. The site may have confirmed exactly those numbers since: its status says so.
    confirmed = assumed and half_preparation.thresholds_confirmed_later(record, status)
    for name, plan_file in [*((name, PLAN_FILE) for name in PACKET_DIRECTORIES), SCENE_PACKAGE]:
        packet = scene / name
        plan_path = _inside(scene, packet / plan_file)
        if plan_path is None:
            continue
        plan = _read(plan_path)
        if not plan:
            continue
        source: dict[str, Any] = {"packet_dir": packet, "plan_path": plan_path, "plan": plan,
                                  "plan_digest": plan.get("plan_digest"), "status": dict(status), "files": {},
                                  "refusal": None, **notes, **({"generated_patch": generated} if generated else {}),
                                  **({"look_views": True} if look else {}),
                                  **({"surface_estimated": True} if capture is not None and "room" in notes
                                     and surface_estimated(capture) else {}), **told,
                                  **({"thresholds_assumed": True} if assumed else {}),
                                  **({"member_date": member_date(capture)} if capture is not None else {}),
                                  **({"thresholds_confirmed_later": True} if confirmed else {}),
                                  **(build_plan_notes(capture, told.get("further_objects"))
                                     if capture is not None else {}),
                                  "physics_check": physics_check(
                                      packet, plan, rule_withheld="rule_withheld" in notes)}
        try:
            rows = placements(plan)
            axis = up_axis(plan, packet=packet)
            rule = half_preparation.with_assumed_basis(neutral_rule(
                _with_sealed_rule(plan, scene, status), estimates=estimates, packet=packet, basis=basis), assumed,
                confirmed=confirmed)
            room = source.get("room")
            if room is not None and (
                    name != SCENE_PACKAGE[0] or room["plan_digest"] != plan.get("plan_digest")
                    or (rule is not None) != room["success_rule_defined"]
                    or any(isinstance(row, Mapping) and row.get("semantic_role") == "task_object"
                           for row in plan.get("objects") or [])):
                # Only a real room package ships with no rule and no /World/Object: its own plan (the one its
                # sealed record names), no object row, and a rule exactly when its record says it has one.
                raise StageRefused("scene_stage_room_record_mismatch")
            if source.get("rule_withheld") and (
                    rule is not None or _sealed_on(plan, scene, _sealed_record(scene, status)) is None
                    or not any(isinstance(row, Mapping) and row.get("semantic_role") == "task_object"
                               for row in plan.get("objects") or [])):
                # The one more shape: the job's object, and no rule, as its own sealed record says and why.
                raise StageRefused("scene_stage_rule_withheld_mismatch")
            if rule is None and room is None and not source.get("rule_withheld"):
                raise StageRefused("scene_stage_success_rule_missing")
            for placement, raw in zip(rows, plan["objects"], strict=True):
                role = placement["role"]
                path = _inside(packet, packet.joinpath(*PurePosixPath(placement["usd_path"]).parts))
                if (path is None or path.stat().st_size != raw.get("size_bytes")
                        or not _DIGEST.fullmatch(str(raw.get("sha256") or ""))):
                    raise StageRefused("scene_stage_asset_not_shipped", role)
                source["files"][placement["prim"]] = {"path": path, "role": role, "sha256": raw["sha256"], "size_bytes": int(raw["size_bytes"]),
                                         "suffix": PurePosixPath(placement["usd_path"]).suffix.lower(),
                                         "placement": {key: value for key, value in placement.items()
                                                       if key != "usd_path"}}
                found = look_source(capture, raw["sha256"]) if capture is not None and role == "scene_appearance" else None
                if found is not None:
                    source["files"][placement["prim"]]["look_source"] = found
            source.update(axis=axis, rule=rule, assets_dir=packet / str(plan.get("asset_directory") or "assets"),
                          appearance_representation=(plan.get("appearance_frame_alignment") or {}).get(
                              "representation"))
        except StageRefused as error:
            source["refusal"] = "website_company_simulation_" + str(error).removeprefix("scene_stage_")
        return source
    return None


def _with_sealed_rule(plan: Mapping[str, Any], scene: Path, status: Mapping[str, Any]) -> Mapping[str, Any]:
    """The plan, with the half's sealed success contract when the plan carries none.

    A half sealed on the strategy's default contract writes it in its record (``team_scene_inputs.v1.json``),
    not in the plan. The record is read only when it is this scene's and seals itself, and only for the task it
    was sealed on: the plan it names (this one, or this one's sibling with one more camera) holds the same task.
    """
    task_spec = plan.get("task_spec")
    if not isinstance(task_spec, Mapping) or task_spec.get("task_success_contract") is not None:
        return plan
    record = _sealed_record(scene, status)
    contract = record.get("task_success_contract")
    if not isinstance(contract, Mapping):
        return plan
    if _sealed_on(plan, scene, record) is None:
        return plan
    return {**plan, "task_spec": {**task_spec, "task_success_contract": dict(contract)}}


def _sealed_on(plan: Mapping[str, Any], scene: Path, record: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """The plan the record was sealed on, when it holds this plan's task: this one, or its sibling with one more
    camera."""
    sealed_on = plan if record.get("plan_digest") == plan.get("plan_digest") else next(
        (other for other in (_read(path) for path in (_inside(scene, scene / name / PLAN_FILE)
                                                      for name in PACKET_DIRECTORIES) if path is not None)
         if other and other.get("plan_digest") == record.get("plan_digest")), None)
    return sealed_on if sealed_on is not None and sealed_on.get("task_spec") == plan.get("task_spec") else None


def _sealed_record(scene: Path, status: Mapping[str, Any]) -> Mapping[str, Any]:
    """The half's own record (``team_scene_inputs.v1.json``) when it is this scene's and seals itself, else empty."""
    record_path = _inside(scene, scene / Path(str(status.get("team_scene_inputs_path") or "")).name)
    record = _read(record_path) if record_path is not None else {}
    return record if record.get("inputs_digest") == canonical_digest(record, digest_field="inputs_digest") else {}


def _record_notes(scene: Path, status: Mapping[str, Any]) -> dict[str, Any]:
    """What the half's sealed record says the package must tell: furniture moved for the arm, a look file that
    failed its quality gate, a support that does not hold the object. Empty for a half that has none of them, so
    its package is as before."""
    record = _sealed_record(scene, status)
    access = record.get("robot_access_removal") if isinstance(record.get("robot_access_removal"), Mapping) else {}
    items = access.get("items") if access.get("status") == "applied" and access.get("station_after") else None
    absent = record.get("robot_station_absent") if isinstance(record.get("robot_station_absent"), Mapping) else {}
    support = record.get("task_support_pad") if isinstance(record.get("task_support_pad"), Mapping) else {}
    reasons = [str(reason) for reason in support.get("reasons") or []] if support.get("status") == "held" else []
    objects = record.get("job_objects") if isinstance(record.get("job_objects"), Mapping) else {}
    room = ({"plan_digest": record.get("plan_digest"), "success_rule_defined": record.get("success_rule_defined") is True,
             "success_rule_assumed": record.get("success_rule_assumed") is True,
             **({"objects": int(objects["count"]), "objects_in_scan": objects.get("in_package") == "part_of_room_scan"}
                if isinstance(objects.get("count"), int) and objects["count"] > 0 else {}),
             **({"not_found": True} if "task_relevant_target_not_observed" in (record.get("flags") or []) else {}),
             **({"not_located": True} if "task_target_not_located" in (record.get("flags") or []) else {}),
             **({"object_not_prepared": True} if OBJECT_NOT_PREPARED_FLAGS & set(record.get("flags") or []) else {}),
             **({"object_not_placed": True} if OBJECT_NOT_PLACED_FLAGS & set(record.get("flags") or []) else {}),
             **({"not_prepared": not_prepared} if (not_prepared := [
                 [row["piece"], row["code"]] for row in record.get("not_prepared") or []
                 if isinstance(row, Mapping) and row.get("piece") in NOT_PREPARED_PIECES
                 and isinstance(row.get("code"), str)
                 and row["code"] not in OBJECT_NOT_PREPARED_FLAGS | OBJECT_NOT_PLACED_FLAGS]) else {}),
             **{"destination_unheld": row["cause"] for row in record.get("not_prepared") or [] if isinstance(row, Mapping)
                and row.get("code") == DESTINATION_CONTACT_CODE and row.get("cause") in unheld.CAUSES},  # typed only
             **({"nothing_taken_out": True} if JOB_OBJECT_RULE in (record.get("flags") or []) else {}),
             **({"scale_not_measured": True} if SCALE_NOT_MEASURED in (record.get("flags") or []) else {}),
             **({"not_filled_in": True} if "scene_completion_not_admitted" in (record.get("flags") or []) else {})}
            if record.get("schema_version") == ROOM_RECORD_SCHEMA else None)
    return {**({"removed_items": len(items)} if isinstance(items, list) and items else {}),
            **({"appearance_quality": "not_passed"} if absent.get("appearance_quality") == "not_passed" else {}),
            **({"support_not_held": "region" if set(reasons) == {"rigid_support_goal_region_rest_outside_success_window"}
                else "start_or_goal"} if support.get("status") == "held" else {}),
            **({"room": room} if room is not None else {}),
            **({"rule_withheld": record["rule_withheld"]["reason"]} if isinstance(record.get("rule_withheld"), Mapping)
               and record["rule_withheld"].get("reason") in RULE_WITHHELD_REASONS else {})}


def _stage_documents(packet: Path, plan: Mapping[str, Any]) -> tuple[Any, Any]:
    """The revision's stage-5 and stage-4 records as the half carries them: an arm-free plan's own copies, else the
    adapted task's source documents in the packet's request (``configured_task_template_adapter``)."""
    if isinstance(plan.get("stage_5"), Mapping):
        return plan.get("stage_5"), plan.get("stage_4")
    request = _read(packet / "native_task_arena_packet_request.v1.json")
    documents = (((request.get("configured_task_template_adapter") or {}).get("source_documents") or {})
                 .get("documents") or {})
    # The rigid adapter names them short; the articulated one by their revision contract paths.
    prefix = "scene.configured_revision.replacement."
    return (documents.get("native_import_qualification", documents.get(prefix + "native_import_qualification")),
            documents.get("static_qualification", documents.get(prefix + "static_qualification")))


def static_check_passed(static: Any) -> bool:
    """The revision's stage-4 record shows the object's static USD check passed (``_usd_findings``: no finding)."""
    return (isinstance(static, Mapping) and static.get("schema_version") == RIGID_STATIC_SCHEMA
            and static.get("status") == "authored_structure_statically_qualified"
            and static.get("authored_structure_statically_qualified") is True
            and static.get("structural_findings") == []
            and static.get("result_digest") == canonical_digest(static, digest_field="result_digest"))


def physics_check(packet: Path, plan: Mapping[str, Any], *, rule_withheld: bool = False) -> str | None:
    """What the README says was checked: ``isaac_sim``, ``not_run``, ``not_run_static_check_passed`` or
    ``not_run_joint_as_authored`` (a jointed object; ``..._no_rule`` when its rule is withheld), or None.

    Stage 5 is read by its own reader (``website_scene_without_simulator.stage_five_receipt_kind``): a passed Isaac
    import, or the record that no simulator ran, each sealed; anything else is None (today's README).
    """
    from .website_scene_without_simulator import stage_five_receipt_kind

    stage, static = _stage_documents(packet, plan)
    kind = stage_five_receipt_kind(stage)
    if kind == "qualified":
        return "isaac_sim"
    if kind == "not_run" and (static or {}).get("schema_version") == ARTICULATED_STATIC_SCHEMA:
        return "not_run_joint_as_authored_no_rule" if rule_withheld else "not_run_joint_as_authored"
    if kind == "not_run":
        return "not_run_static_check_passed" if static_check_passed(static) else "not_run"
    return None


def simulation_revision(source: Mapping[str, Any], *, max_bytes: int = MAX_PACKAGE_BYTES) -> str:
    """Names this package: the files' digests, their placements, the success rule, the builder and the cap."""
    return canonical_digest({"builder": BUILDER_REVISION, "format": FORMAT, "max_bytes": int(max_bytes),
        "files": {prim: {key: row[key] for key in ("sha256", "size_bytes", "suffix", "placement")}
                  for prim, row in sorted(source["files"].items())},
        "rule": source.get("rule"), "axis": source.get("axis"), "refusal": source.get("refusal"),
        "plan_digest": source.get("plan_digest") if source.get("refusal") else None,
        # Only a wording that differs from the packages before it names a new revision.
        **({"physics_check": source["physics_check"]} if PHYSICS_CHECK_WORDS.get(source.get("physics_check")) else {}),
        **{key: source[key] for key in ("removed_items", "appearance_quality", "support_not_held", "room", "plan_notes",
                                        "generated_patch", "look_views", "surface_estimated", "not_filled_in",
                                        "size_from_its_points",
                                        "size_disagrees_with_published", "destination_overhang",
                                        "destination_box_meets_room", "member_date", "library_models",
                                        "library_refusal", "thresholds_assumed", "thresholds_confirmed_later",
                                        "object_source", "object_shape", "variation_evidence", "mass_basis",
                                        "further_objects")
           if source.get(key)},
        # The splat a look is written again from (``look_source``): a look found later, or moved, is a new package.
        **({"looks": looks} if (looks := {prim: {key: row["look_source"][key] for key in ("sha256", "runtime_to_simulator")}
                                          for prim, row in sorted(source["files"].items())
                                          if row.get("look_source")}) else {})})


# --- what must never leave --------------------------------------------------------------------------------------

def _walk_values(value: Any, keys: frozenset[str]) -> Iterator[str]:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key in keys and isinstance(item, str):
                yield item
            else:
                yield from _walk_values(item, keys)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_values(item, keys)


def secret_literals(capture_root: Path, source: Mapping[str, Any]) -> list[tuple[str, str]]:
    """The job's own ids, account ids and keys, each with the kind printed in a finding (never the value)."""
    root = Path(capture_root)
    identity = permission.capture_identity(root)
    status = source.get("status") or {}
    records = [_read(root / name) for name in (
        "capture_descriptor.json", "raw/manifest.json", "raw/capture_upload_complete.json",
        "pipeline_staging_manifest.json", "pipeline_job_ledger.json", "pipeline/website_task_context.json")]
    records.append(_read(Path(source["plan_path"])))
    found = [(identity[key], "a job id") for key in ("request_id", "scene_id", "capture_id")]
    core = re.search(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", identity["request_id"])
    if core:
        found += [(core.group(0), "a job id"), (core.group(0).replace("-", ""), "a job id")]
    found += [(str(status.get(key) or ""), "a job id") for key in ("task_id", "request_key")]
    for record in records:
        found += [(value, "a job id") for value in _walk_values(record, frozenset({"site_submission_id", "intent_id",
                                                                                     "scene_intent_id", "site_id"}))]
        found += [(value, "an account id") for value in _walk_values(record, _ACCOUNT_KEYS)]
        found += [(value, "a delivery key") for value in _walk_values(record, _KEY_KEYS)]
    # Short words ("system", "operator") are roles, not ids: an id here is at least eight characters.
    return [(value, kind) for value, kind in found if len(value) >= 8 and not re.fullmatch(r"[a-z_]+", value)]


# --- making the package ----------------------------------------------------------------------------------------

def _open_layer(path: str) -> Any:
    """The layer, or None when ``pxr`` cannot open it (it raises for a broken file)."""
    from pxr import Sdf
    try:
        return Sdf.Layer.OpenAsAnonymous(path)
    except Exception:  # noqa: BLE001 - any failure to open is a refusal by the caller
        return None


def _asset_paths(layer: Any) -> list[str]:
    from pxr import UsdUtils
    found: list[str] = []

    def collect(path: str) -> str:
        found.append(path)
        return path
    UsdUtils.ModifyAssetPaths(layer, collect)
    return found


def _outside(path: str) -> bool:
    pure = PurePosixPath(path.replace("\\", "/"))
    return ("://" in path or path.startswith(("file:", "/", "\\")) or re.match(r"[A-Za-z]:", path) is not None
            or ".." in pure.parts)


def _check_usdz(path: Path, label: str) -> None:
    """A usdz is kept as it is: every reference inside it must stay inside it."""
    try:
        with zipfile.ZipFile(path) as archive:
            names = [info.filename for info in archive.infolist() if Path(info.filename).suffix.lower() in _USD]
    except (OSError, zipfile.BadZipFile):
        raise PackageRefused("website_company_simulation_scene_unreadable", label) from None
    if not names:
        raise PackageRefused("website_company_simulation_scene_unreadable", label)
    for name in names:
        layer = _open_layer(f"{path}[{name}]")
        if layer is None:
            raise PackageRefused("website_company_simulation_scene_unreadable", label)
        for reference in [*_asset_paths(layer), *(value for _spec, value in _file_strings(layer))]:
            if reference and _outside(reference):
                raise PackageRefused("website_company_simulation_outside_reference", label)


#: A string or token value with one of these endings names a file: it is shipped and rewritten like an asset path.
FILE_SUFFIXES = frozenset({".usd", ".usda", ".usdc", ".usdz", ".png", ".jpg", ".jpeg", ".exr", ".hdr", ".tga",
                           ".tif", ".tiff", ".bmp", ".ktx", ".ktx2", ".dds", ".mdl", ".obj", ".ply", ".glb", ".gltf",
                           ".vdb", ".nvdb", ".mtl", ".wav", ".mp4", ".mov"})
_STRING_TYPES = frozenset({"string", "token", "string[]", "token[]"})


def _names_file(value: Any) -> bool:
    return isinstance(value, str) and PurePosixPath(value.strip().replace("\\", "/")).suffix.lower() in FILE_SUFFIXES


def _file_strings(layer: Any) -> list[tuple[Any, str]]:
    """Every (attribute spec, value) whose string or token value names a file, in defaults and time samples."""
    from pxr import Sdf
    found: list[tuple[Any, str]] = []
    paths: list = []
    layer.Traverse(Sdf.Path.absoluteRootPath, paths.append)
    for path in paths:
        spec = layer.GetAttributeAtPath(path) if path.IsPropertyPath() else None
        if spec is None or str(spec.typeName) not in _STRING_TYPES:
            continue
        values = [spec.default] if spec.HasDefaultValue() else []
        values += [layer.QueryTimeSample(path, sample) for sample in layer.ListTimeSamplesForPath(path)]
        for value in values:
            for item in (list(value) if not isinstance(value, str) and value is not None else [value]):
                if _names_file(item):
                    found.append((spec, str(item)))
    return found


def _rewrite_strings(layer: Any, rewrite: Callable[[str], str]) -> None:
    """Rewrite every string or token value that names a file, in defaults and time samples."""
    from pxr import Sdf
    paths: list = []
    layer.Traverse(Sdf.Path.absoluteRootPath, paths.append)

    def fixed(value: Any) -> Any:
        if isinstance(value, str):
            return rewrite(value) if _names_file(value) else value
        if value is not None and not isinstance(value, str):
            items = list(value)
            new = [rewrite(item) if _names_file(item) else item for item in items]
            return new if new != items else value
        return value
    for path in paths:
        spec = layer.GetAttributeAtPath(path) if path.IsPropertyPath() else None
        if spec is None or str(spec.typeName) not in _STRING_TYPES:
            continue
        if spec.HasDefaultValue():
            new = fixed(spec.default)
            if new is not spec.default and new != spec.default:
                spec.default = new
        for sample in layer.ListTimeSamplesForPath(path):
            value = layer.QueryTimeSample(path, sample)
            new = fixed(value)
            if new is not value and new != value:
                layer.SetTimeSample(path, sample, new)


_UDIM = "<UDIM>"
#: The closed list of kinds a scene may ship (``outbound_file_scan``, Rule 1), by ending and by content.
_KIND_BY_SUFFIX = {".usd": ("usdc", "usda"), ".usda": ("usda",), ".usdc": ("usdc",), ".usdz": ("usdz",),
                   ".png": ("png",), ".jpg": ("jpeg",), ".jpeg": ("jpeg",)}
KIND_NOT_SHIPPED = "website_company_simulation_file_kind_not_shipped"


def _shippable(path: Path, label: str) -> None:
    """Refuse, naming the role and the kind, a file a scene depends on that is not on the closed list."""
    from .outbound_file_scan import content_kind
    suffix = path.suffix.lower()
    allowed = _KIND_BY_SUFFIX.get(suffix)
    try:
        actual = content_kind(path)
    except OSError:
        actual = None
    if allowed is None or actual not in allowed:
        raise PackageRefused(KIND_NOT_SHIPPED, f"{label} {suffix or 'no ending'}")


def _write(source: Path, target: Path, label: str) -> None:
    """Every file is written by Blueprint from what it read, never copied (``outbound_file_writer``)."""
    from .outbound_file_writer import WriteRefused, write_file
    try:
        write_file(source, target)
    except WriteRefused as error:
        raise PackageRefused(KIND_NOT_SHIPPED, f"{label} {error.detail}") from None


class _Localizer:
    """Copies each USD file into the package, rewriting every reference to a relative path inside it.

    Asset paths (sublayers, references, payloads, asset-valued attributes) and string or token values that name a
    file are rewritten; sublayer time offsets are kept; every tile of a UDIM texture inside the scene's own folder
    is shipped. A reference that is a link, points outside the scene's files or names nothing refuses the package.
    """

    def __init__(self, assets_dir: Path, staging: Path):
        self.assets, self.staging = assets_dir.resolve(), staging
        self.placed: dict[Path, Path] = {}
        self.count = 0

    def _target(self, suffix: str) -> Path:
        self.count += 1
        return self.staging / "scene" / "files" / f"file-{self.count:03d}{suffix}"

    def place(self, source: Path, target: Path, label: str) -> None:
        source = source.resolve()
        _shippable(source, label)
        self.placed[source] = target
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.suffix.lower() == ".usdz":
            _check_usdz(source, label)
            _write(source, target, label)
            return
        if source.suffix.lower() not in _USD:
            _write(source, target, label)
            return
        layer = _open_layer(str(source))
        if layer is None:
            raise PackageRefused("website_company_simulation_scene_unreadable", label)
        changed: list[bool] = []
        failure: list[str] = []

        def inside(candidate: Path) -> Path | None:
            try:
                resolved = candidate.resolve(strict=True)
            except OSError:
                return None
            return resolved if resolved.is_relative_to(self.assets) and resolved.is_file() else None

        def rewrite(reference: str) -> str:
            if not reference or failure:
                return reference
            if "://" in reference or reference.startswith("file:"):
                failure.append("website_company_simulation_outside_reference")
                return reference
            base = Path(reference) if Path(reference).is_absolute() else source.parent / reference
            if _UDIM in reference:
                return udim(reference, base)
            try:
                base.resolve(strict=True)
            except OSError:
                failure.append("website_company_simulation_reference_missing")
                return reference
            resolved = inside(base)
            if resolved is None:
                failure.append("website_company_simulation_outside_reference")
                return reference
            placed = self.placed.get(resolved)
            if placed is None:
                placed = self._target(resolved.suffix.lower())
                self.place(resolved, placed, label)
            relative = "./" + Path(os.path.relpath(placed, target.parent)).as_posix()
            if relative != reference:
                changed.append(True)
            return relative

        def udim(reference: str, base: Path) -> str:
            head, _, tail = base.name.partition(_UDIM)
            pattern = re.compile(re.escape(head) + r"(1[0-9]{3})" + re.escape(tail))
            try:
                tiles = sorted(item for item in base.parent.iterdir() if pattern.fullmatch(item.name))
            except OSError:
                tiles = []
            if not tiles or any(inside(tile) is None for tile in tiles):
                failure.append("website_company_simulation_reference_missing" if not tiles
                               else "website_company_simulation_outside_reference")
                return reference
            stem = self._target("")
            for tile in tiles:
                _shippable(tile.resolve(), label)
                number = pattern.fullmatch(tile.name).group(1)
                tile_target = stem.with_name(f"{stem.name}.{number}{tail.lower() or tile.suffix.lower()}")
                tile_target.parent.mkdir(parents=True, exist_ok=True)
                _write(tile.resolve(), tile_target, label)
                self.placed.setdefault(tile.resolve(), tile_target)
            changed.append(True)
            return "./" + Path(os.path.relpath(stem.with_name(f"{stem.name}.{_UDIM}{tail.lower()}"),
                                                target.parent)).as_posix()
        from pxr import UsdUtils
        offsets = list(layer.subLayerOffsets)
        UsdUtils.ModifyAssetPaths(layer, rewrite)
        for index, offset in enumerate(offsets):
            if index < len(layer.subLayerPaths):
                layer.subLayerOffsets[index] = offset
        if not failure:
            _rewrite_strings(layer, rewrite)
        if failure:
            raise PackageRefused(failure[0], label)
        # Flattening's note names the host file the object was made from (every articulated object has
        # one); it says nothing about the object, and the identity scan would refuse the file for it.
        layer.documentation = "\n".join(line for line in layer.documentation.split("\n")
                                        if not line.startswith(FLATTEN_PROVENANCE)).rstrip("\n")
        # Written by Sdf from the opened layer, changed or not: never copied (``outbound_file_writer``).
        if not layer.Export(str(target)):
            raise PackageRefused("website_company_simulation_scene_unreadable", label)


def _layer_units(path: Path) -> tuple[str | None, float | None]:
    layer = _open_layer(str(path))
    root = layer.pseudoRoot if layer is not None else None
    up = root.GetInfo("upAxis") if root is not None and root.HasInfo("upAxis") else None
    metres = root.GetInfo("metersPerUnit") if root is not None and root.HasInfo("metersPerUnit") else None
    return (str(up) if up else None), (float(metres) if metres is not None else None)


def _root_offset(path: Path) -> dict[str, list[float]] | None:
    """The part file's own root translation and rotation, when either is not identity (1 mm, 0.001 rad)."""
    from pxr import Gf, Usd, UsdGeom
    try:
        stage = Usd.Stage.Open(str(path), Usd.Stage.LoadNone)
        prim = stage.GetDefaultPrim() or next(iter(stage.GetPseudoRoot().GetChildren()), None)
        if prim is None or not prim.IsA(UsdGeom.Xformable):
            return None
        matrix = UsdGeom.Xformable(prim).GetLocalTransformation(Usd.TimeCode.Default())
        transform = Gf.Transform(matrix)
    except Exception:  # noqa: BLE001 - an unreadable part is refused elsewhere; no note
        return None
    translation = [float(value) for value in transform.GetTranslation()]
    quat = transform.GetRotation().GetQuat()
    xyzw = [float(value) for value in quat.GetImaginary()] + [float(quat.GetReal())]
    angle = transform.GetRotation().GetAngle()
    if max(abs(value) for value in translation) <= 1e-3 and abs(angle) <= math.degrees(1e-3):
        return None
    return {"translation_m": [round(value, 6) for value in translation],
            "rotation_xyzw": [round(value, 6) for value in xyzw]}


def without_look(source: Mapping[str, Any]) -> dict[str, Any] | None:
    """The same source with the room's look left out and its collision shown in its place (``LOOK_LEFT_OUT``), or
    None when it has no look or no collision to show instead."""
    files = source.get("files") or {}
    roles = [row["role"] for row in files.values()]
    if "scene_appearance" not in roles or "scene_collision" not in roles:
        return None
    kept = {prim: ({**row, "placement": {**row["placement"], "visible": True}} if row["role"] == "scene_collision"
                   else row) for prim, row in files.items() if row["role"] != "scene_appearance"}
    # What the README would say of the appearance file goes with it: its quality note and its representation.
    return {**{key: value for key, value in source.items()
               if key not in ("appearance_quality", "appearance_representation")}, "files": kept, "look_left_out": True}


#: The packages tried in turn while the zip is over the limit: whole, without the look's PLY, without the look.
SIZE_STEPS = ("whole", "without_ply", "without_look")


def build_package(source: Mapping[str, Any], output: Path, *, literals: list[tuple[str, str]],
                  words: list[tuple[bytes, str]] | None = None, max_bytes: int = MAX_PACKAGE_BYTES,
                  deadline: float | None = None, known_over: Sequence[str] = (),
                  over: list[str] | None = None) -> dict[str, Any]:
    """Stage, rewrite, compose, scan and zip the package to ``output``; ``PackageRefused`` or ``OVER_CAP`` otherwise.

    The limit is on the zip. Source files above ``MAX_SOURCE_BYTES`` are refused unbuilt. A zip over the limit is
    built once more as the next of ``SIZE_STEPS``: without the look's PLY (``PLY_LEFT_OUT``; skipped when the zip
    held none), then without the room's look (``without_look``); only when that one is over too is it
    ``OVER_CAP``. Each step found over is appended to ``over``, and steps in ``known_over`` (an earlier pass's, for
    this revision) are not built again, so a package that runs out of time starts the next pass where it stopped.
    It holds no lock: every file it writes is its own (a staging folder and a partial zip named for this build).
    ``deadline`` (a ``time.monotonic`` value) is checked between steps and files: past it, ``BUILD_TIMED_OUT``.
    """
    over = [] if over is None else over
    if source.get("refusal"):
        raise ShapeRefused(source["refusal"])
    if library_notes.refusal(source):  # an object built from a library model never leaves without its credit
        raise ShapeRefused(library_notes.refusal(source), "task_object")  # the room floor still ships
    if sum(row["size_bytes"] for row in source["files"].values()) > MAX_SOURCE_BYTES:
        raise ValueError(OVER_CAP)
    options = {"literals": literals, "words": words, "max_bytes": max_bytes, "deadline": deadline}
    for step in SIZE_STEPS:
        if step in known_over or step in over:
            continue
        value = (source if step == "whole" else {**source, "ply_left_out": True} if step == "without_ply"
                 else without_look(source))
        if value is None:
            break  # no look to leave out, or no collision to show in its place
        try:
            built = _build_once(value, output, **options)
        except ValueError as error:
            if str(error) != OVER_CAP:
                raise
            over.append(step)
            if step == "whole" and not getattr(error, "plys", False):
                over.append("without_ply")  # it held no PLY: without one it is the same package
            continue
        return {**built, **({"ply_left_out": True} if step == "without_ply" else
                            {"look_left_out": True} if step == "without_look" else {})}
    raise ValueError(OVER_CAP)


class _OverCap(ValueError):
    """``OVER_CAP``, saying whether the zip held a splat PLY of the look (``plys``), which goes first."""

    def __init__(self, plys: bool):
        super().__init__(OVER_CAP)
        self.plys = plys


def _build_once(source: Mapping[str, Any], output: Path, *, literals: list[tuple[str, str]],
                words: list[tuple[bytes, str]] | None, max_bytes: int, deadline: float | None) -> dict[str, Any]:
    def on_time() -> None:
        if deadline is not None and time.monotonic() >= deadline:
            raise ValueError(BUILD_TIMED_OUT)
    import tempfile

    from .outbound_file_scan import literal_forms, scan_tree
    from .scene_package_stage import StageRefused, write_stage
    from .website_company_readme import readme
    output.parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=STAGING + "-", dir=output.parent))
    staging, scratch, partial = work / "package", work / "scan", work / "simulation.zip.partial"
    staging.mkdir(mode=0o750)
    scratch.mkdir(mode=0o700)
    try:
        localizer = _Localizer(Path(source["assets_dir"]), staging)
        written: dict[str, Mapping[str, Any]] = {}  # looks written again from their source splat
        first_written: dict[str, Mapping[str, Any]] = {}  # looks with a source that could not be
        not_written: dict[str, str] = {}
        placed: dict[str, str] = {}
        units: dict[str, tuple[str | None, float | None]] = {}
        offsets: dict[str, dict[str, list[float]]] = {}
        rows = []
        for prim, row in source["files"].items():
            on_time()
            if _sha256_file(row["path"]) != row["sha256"]:
                raise PackageRefused("website_company_simulation_source_changed", row["role"])
            relative = row["placement"]["name"] + row["suffix"]
            look = row.get("look_source")
            why = reauthor_look(look, staging / relative) if look else None
            if look and why is None:
                written[prim] = look
            else:
                if why is not None:
                    first_written[prim] = look
                    not_written[prim] = why
                    logger.warning("website_company_simulation_look_not_written_again reason=%s", why)
                localizer.place(Path(row["path"]), staging / relative, row["role"])
            placed[prim] = relative
            units[prim] = _layer_units(staging / relative)
            offset = _root_offset(staging / relative)
            if offset is not None:
                offsets[prim] = offset
            rows.append({**row["placement"], "file": relative})
        library_notes.place(source, staging, _write)  # a library model ships as its own file, with its credit
        plys: dict[str, str] = {}
        for prim, look in written.items():
            if source.get("ply_left_out"):
                continue
            on_time()
            relative = source["files"][prim]["placement"]["name"] + ".ply"
            why = look_ply(look, staging / relative)
            if why is None:
                plys[prim] = relative
            else:
                not_written[prim] = why
                logger.warning("website_company_simulation_look_ply_not_written reason=%s", why)
        try:
            write_stage(staging / "scene.usda", rows, axis=source["axis"])
        except StageRefused as error:
            raise ShapeRefused("website_company_simulation_" + str(error).removeprefix("scene_stage_")) from None
        if source["rule"] is not None:
            rule = dict(source["rule"])
            if offsets:
                rule["part_notes"] = {f"/World/{prim}": {"own_root_offset": offset}
                                      for prim, offset in sorted(offsets.items())}
            (staging / "success_rule.json").write_text(json.dumps(rule, indent=2, sort_keys=True) + "\n",
                                                       encoding="utf-8")
        varied = variations.write(source, staging, placed)  # each object's simulated variations, said to be so
        names = sorted(path.relative_to(staging).as_posix() for path in staging.rglob("*") if path.is_file())
        credited = library_notes.credited(source, staging, placed)  # every file of each model, by its credit
        (staging / "README.txt").write_text(readme({**source, **credited, **varied}, placed, [*names, "README.txt"],
                                                   units, offsets, plys, first_written), encoding="utf-8")
        on_time()
        _resolves_inside(staging, room=source.get("room") is not None)
        findings = scan_tree(staging, [*literal_forms(literals), *(words or [])], scratch=scratch, before_each=on_time,
                             words_except=("README.txt", "scene.usda"), closed=True)
        on_time()
        from .outbound_file_scan import NOT_SHIPPED
        if findings and findings[0].kind == NOT_SHIPPED:
            raise PackageRefused(KIND_NOT_SHIPPED, findings[0].file)
        if findings:
            raise PackageRefused("website_company_simulation_file_names_the_site", findings[0].sentence())
        names = sorted(path.relative_to(staging).as_posix() for path in staging.rglob("*") if path.is_file())
        with zipfile.ZipFile(partial, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for name in names:
                info = zipfile.ZipInfo(name, date_time=tuple(source.get("member_date") or ZIP_EPOCH))
                info.compress_type, info.create_system, info.external_attr = zipfile.ZIP_DEFLATED, 3, 0o100644 << 16
                info.file_size = (staging / name).stat().st_size
                with (staging / name).open("rb") as stream, archive.open(info, "w") as member:
                    shutil.copyfileobj(stream, member, 8 << 20)
        size = partial.stat().st_size
        if size > max_bytes:
            raise _OverCap(bool(plys))
        os.replace(partial, output)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return {"format": FORMAT, "content_type": CONTENT_TYPE, "digest": _sha256_file(output), "size_bytes": size,
            "files": names, "max_bytes": int(max_bytes),
            **({"look_not_written": dict(sorted(not_written.items()))} if not_written else {})}


def _resolves_inside(staging: Path, *, room: bool = False) -> None:
    """``scene.usda`` opens, and every layer and file it uses resolves inside the package; it places the job's
    object, unless it is a room package (no object of its own)."""
    from pxr import Usd, UsdUtils
    scene = staging / "scene.usda"
    try:
        layers, assets, unresolved = UsdUtils.ComputeAllDependencies(str(scene))
        stage = Usd.Stage.Open(str(scene), Usd.Stage.LoadNone)
    except Exception:  # noqa: BLE001 - any failure is a refusal
        raise PackageRefused("website_company_simulation_scene_unreadable", "scene.usda") from None
    if stage is None or unresolved or not (room or stage.GetPrimAtPath("/World/Object").IsValid()):
        raise PackageRefused("website_company_simulation_reference_missing", "scene.usda")
    for item in [*(layer.realPath for layer in layers), *assets]:
        path = Path(str(item).split("[", 1)[0])
        if not path.resolve().is_relative_to(staging.resolve()):
            raise PackageRefused("website_company_simulation_outside_reference", "scene.usda")


# --- publishing -------------------------------------------------------------------------------------------------

#: The neutral prim of the job's object (``scene_package_stage.ROLE_PRIMS``): in the package's placed parts exactly
#: when the job's object ships in it. A room package, or a floor standing in for a refused half, has none.
OBJECT_PRIM = "Object"
#: The WebApp's answer while its reader of ``website_company_simulation.v1`` is strict and predates
#: ``object_included`` (its route's schema failure, a 409 with this code).
RECORD_REFUSED = "website_company_simulation_invalid"


def object_included(manifest: Mapping[str, Any]) -> bool:
    """Whether the job's object is in the package the manifest names: its placed parts hold the object's prim."""
    files = (manifest.get("source") or {}).get("files")
    return isinstance(files, Mapping) and OBJECT_PRIM in files


def package_record(*, identity: Mapping[str, str], manifest: Mapping[str, Any],
                   with_object_included: bool = True) -> dict[str, Any]:
    """What the WebApp may store: the package's identity, never a path, object key or person.

    ``object_included`` (2026-11-24, added compatibly) is true only when the job's object is in the package. A
    website whose strict reader predates it refuses the record (``RECORD_REFUSED``); the same record without it is
    then sent (``_publish_built``), so the package still reaches the companies and nothing waits for a deploy.
    """
    built, basis = manifest["package"], manifest["permission"]
    value = {"schema_version": PACKAGE_SCHEMA, "capture_id": identity["capture_id"],
             "simulation_revision": manifest["simulation_revision"], "state": "ready", "format": built["format"],
             "content_type": built["content_type"], "digest": built["digest"], "size_bytes": built["size_bytes"],
             "files": list(built["files"]), "permission_basis": basis["permission_basis"],
             "statement_version": basis["statement_version"], "updated_at_epoch": int(manifest["built_at_epoch"]),
             **({"object_included": object_included(manifest)} if with_object_included else {})}
    value["simulation_digest"] = cross_runtime_canonical_digest(value)
    return value


def _record_refused(error: BaseException) -> bool:
    """The website's strict reader refused the record's shape (not its rights, its revision or its digest)."""
    return f"_http_409:{RECORD_REFUSED}" in str(error)


@contextmanager
def package_lock(capture_root: Path) -> Iterator[None]:
    directory = Path(capture_root).joinpath(*DIRECTORY)
    if not directory.parent.is_dir() or directory.parent.is_symlink():
        raise ValueError("website_company_simulation_capture_changed")  # a retired capture is never re-created
    directory.mkdir(mode=0o750, exist_ok=True)
    descriptor = os.open(directory / LOCK_FILE, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600)
    with os.fdopen(descriptor, "r+") as lock:
        if not stat.S_ISREG(os.fstat(lock.fileno()).st_mode):
            raise ValueError("website_company_simulation_lock_unsafe")
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise CompanySimulationBusy("website_company_simulation_busy") from None
        yield


def _archive(path: Path, *, identity: Mapping[str, str]) -> dict[str, Any]:
    """The private artifact store only, never the legacy fallback, with its bytes proven."""
    from .task_evaluation_configured_scene_object_store import (
        publish_configured_scene_artifact, verify_dedicated_artifact_store)
    verify_dedicated_artifact_store()
    filename = package_filename(identity["capture_id"], identity["request_id"])
    reference = publish_configured_scene_artifact(path=path, artifact_kind=ARTIFACT_KIND, filename=filename)
    return {"uri": reference["uri"], "digest": reference["digest"], "size_bytes": reference["size_bytes"],
            "filename": filename}


def _deliver(record: Mapping[str, Any], *, identity: Mapping[str, str]) -> None:
    from .website_task_context import website_webapp_request
    acceptance = website_webapp_request(capture_id=identity["capture_id"], operation="company-simulation", payload={
        "request_id": identity["request_id"], "scene_id": identity["scene_id"], "simulation": dict(record)})
    if (acceptance.get("schema_version") != ACCEPTANCE_SCHEMA or acceptance.get("accepted") is not True
            or acceptance.get("simulation_revision") != record["simulation_revision"]
            or acceptance.get("simulation_digest") != record["simulation_digest"]):
        raise ValueError("website_company_simulation_acceptance_invalid")


def _current(manifest: Mapping[str, Any], revision: str, package: Path) -> bool:
    built, archive = manifest.get("package") or {}, manifest.get("archive") or {}
    return (manifest.get("schema_version") == MANIFEST_SCHEMA and manifest.get("simulation_revision") == revision
            and not package.is_symlink() and package.is_file() and package.stat().st_size == built.get("size_bytes")
            and archive.get("digest") == built.get("digest") and not permission.hidden(archive)
            and (manifest.get("delivered") or {}).get("simulation_digest") == manifest.get("simulation_digest"))


def _take_back_if_archived(root: Path, manifest_path: Path, retract: Callable[[str], None]) -> None:
    if not manifest_path.is_file():
        return
    with package_lock(root):
        manifest = _read(manifest_path)
        if manifest:
            permission.take_back(manifest, manifest_path, retract)


def upload_sha256(capture_root: Path) -> str | None:
    """The digest of the upload the scene was built from: the
    original normalization recorded, else the one the company clip recorded for the raw upload."""
    recorded = clip.recorded_video_sha256(capture_root)
    if recorded:
        return recorded
    value = _read(Path(capture_root).joinpath(*clip.DIRECTORY) / clip.MANIFEST_FILE).get("upload_sha256")
    return value if isinstance(value, str) and _DIGEST.fullmatch(value) else None


def publish_company_simulation(capture_root: Path, *, now: Callable[[], float] = time.time,
                               check_rights: Callable[[Mapping[str, str]], Mapping[str, Any]] = permission.live_rights,
                               archive: Callable[..., dict[str, Any]] = _archive,
                               retract: Callable[[str], None] = permission.retract_object,
                               deliver: Callable[..., None] = _deliver, queue_root: Path | None = None,
                               max_bytes: int = MAX_PACKAGE_BYTES,
                               build_seconds: float = BUILD_DEADLINE_SECONDS) -> dict[str, Any]:
    """One idempotent pass for one capture: make, archive and deliver what is not yet current, and take back
    what may no longer be shared (``company_share_permission``, shared with the company clip).

    Returns ``{"status": ...}``: ``current``, ``published``, ``not_permitted`` (no basis; any archived copy is
    taken back), ``no_source`` (no compiled simulation yet), ``over_size_cap``, ``build_timed_out`` (tried again
    after the window), ``withdrawn``, ``refused`` (rights, the takedown gate, the website, or the package may not
    leave, with a reason), ``busy`` or ``failed`` (retried by the next pass). It never raises.
    """
    try:
        return _publish(Path(capture_root), now, check_rights, archive, retract, deliver, queue_root, max_bytes,
                        build_seconds)
    except CompanySimulationBusy:
        return {"status": "busy"}
    except Exception as error:  # noqa: BLE001 - one capture never stops a pass; retried next time
        logger.warning("website_company_simulation_failed capture=%s type=%s", Path(capture_root).name,
                       type(error).__name__)
        return {"status": "failed", "reason": _own_reason(error)}


def _own_reason(error: BaseException | str) -> str:
    """The reason in the simulation's own name: the shared rights checks speak in the clip's."""
    reason = error if isinstance(error, str) else permission.reason_code(error)
    return "website_company_simulation_" + reason.removeprefix("website_company_clip_") \
        if reason.startswith("website_company_clip_") else reason


def _own(result: dict[str, Any] | None) -> dict[str, Any] | None:
    if result is not None and isinstance(result.get("reason"), str):
        result = {**result, "reason": _own_reason(result["reason"])}
    return result


def _refused_live(check, manifest, manifest_path, retract):
    return _own(permission.refused_live(check, manifest, manifest_path, retract,
                                        unanswered="website_company_simulation_rights_unanswered"))


#: Build outcomes recorded for a revision and tried again only after the window, not every pass.
_RETRIED_AFTER_WINDOW = re.compile(r"website_company_simulation_[a-z_]+")


def _kept_or_taken_back(root, identity, manifest, manifest_path, check_rights, retract, now):
    """None while the published package keeps its basis, else its take-back's answer: the live rights at least
    once a window (and at once after a recheck mark). A definite "no" takes it back."""
    check = permission.read_check(root)
    if permission.live_due(check, now(), root):
        check = permission.record_live(root, identity, check_rights, now())
        if check["outcome"] == "rights_refused":
            return _refused_live(check, manifest, manifest_path, retract)
    if check.get("outcome") == "rights_refused":  # recorded by this pass or the clip's: any basis stops
        return _refused_live(check, manifest, manifest_path, retract)
    basis = (manifest.get("permission") or {}).get("permission_basis")
    if ((basis == "site_sharing_statement" and permission.live_grant_holds(check, now()))
            or (basis == "internal_test" and permission.internal_test_holds(check, now()))):
        return None
    permission.take_back(manifest, manifest_path, retract)
    return {"status": "not_permitted"}


def _publish(root, now, check_rights, archive, retract, deliver, queue_root, max_bytes, build_seconds):
    identity = permission.capture_identity(root)
    directory = root.joinpath(*DIRECTORY)
    manifest_path, package_path = directory / MANIFEST_FILE, directory / PACKAGE_FILE
    if permission.retraction_pending(_read(manifest_path)):
        with package_lock(root):
            manifest = _read(manifest_path)
            if permission.retraction_pending(manifest) and not permission.take_back(manifest, manifest_path, retract):
                return {"status": "failed", "reason": "website_company_simulation_retraction_pending"}
    if permission.retraction_pending(_read(manifest_path), permission.SUPERSEDED):
        # The copy the website read before its new package: retired after that delivery, and again if it failed.
        with package_lock(root):
            manifest = _read(manifest_path)
            if (permission.retraction_pending(manifest, permission.SUPERSEDED)
                    and not permission.retire(manifest, manifest_path, retract)):
                return {"status": "failed", "reason": "website_company_simulation_retraction_pending"}
    if capture_withdrawn(root):
        return {"status": "withdrawn"}
    if not permission.delivery_gate_allows(root, surface="website_company_simulation"):
        _take_back_if_archived(root, manifest_path, retract)
        return {"status": "refused", "reason": "website_company_simulation_takedown_gate_closed"}
    if not permission.may_be_permitted(root, identity, now()):
        _take_back_if_archived(root, manifest_path, retract)
        return {"status": "not_permitted"}
    notes: dict[str, Any] = {}
    if package_source(root, queue_root=queue_root, manifest=_read(manifest_path), max_bytes=max_bytes, now=now(),
                      notes=notes) is None:
        # Nothing to build (a floor whose object path moves again, a half whose status went): what is out still
        # follows its rights on this pass.
        with package_lock(root):
            manifest = _read(manifest_path)
            archived = manifest.get("archive")
            if isinstance(archived, Mapping) and not permission.hidden(archived):
                stop = _kept_or_taken_back(root, identity, manifest, manifest_path, check_rights, retract, now)
                if stop is not None:
                    return stop
        return {"status": "no_source", **notes}
    # 1. Under the lock: decide. 2. Without it: build (minutes for a real scene; withdrawal deletion is never held).
    # 3. Under the lock again: check nothing changed, then publish.
    with package_lock(root):
        manifest = _read(manifest_path)
        source = package_source(root, queue_root=queue_root, manifest=manifest, max_bytes=max_bytes, now=now())
        if source is None:
            return {"status": "no_source"}
        revision = simulation_revision(source, max_bytes=max_bytes)
        upload = upload_sha256(root)
        check = permission.read_check(root)
        if _current(manifest, revision, package_path):
            return (_kept_or_taken_back(root, identity, manifest, manifest_path, check_rights, retract, now)
                    or {"status": "current", "simulation_revision": revision})
        refusal = _refusal(manifest, revision)
        refused, refused_at = refusal.get("refused"), refusal.get("refused_at_epoch")
        # A size refusal stands for its revision; one made under the old rule (no ``size_rule``) is built once more.
        if refused == OVER_CAP and refusal.get("size_rule") == SIZE_RULE:
            return {"status": "over_size_cap"}
        if (isinstance(refused, str) and refused != OVER_CAP
                and isinstance(refused_at, int) and _RETRIED_AFTER_WINDOW.fullmatch(refused)
                and 0 <= now() - refused_at < permission.LIVE_STATEMENT_RECHECK_SECONDS):
            return ({"status": "build_timed_out"} if refused == BUILD_TIMED_OUT
                    else {"status": "refused", "reason": refused})
        # A definite "no" from this window's live read stands until the window passes: no request every pass.
        if (not permission.live_due(check, now(), root)
                and check.get("outcome") in ("not_sharing", "rights_refused")):
            if manifest:
                permission.take_back(manifest, manifest_path, retract)
            return ({"status": "not_permitted"} if check["outcome"] == "not_sharing"
                    else {"status": "refused", "reason": _own_reason(check.get("reason") or permission.RIGHTS_REFUSED)})
        # A build or a delivery: always on a fresh live answer, recorded whatever it says.
        check = permission.record_live(root, identity, check_rights, now())
        stopped = _refused_live(check, manifest, manifest_path, retract)
        if stopped is not None:
            return stopped
        basis = permission.decided_basis(check)
        if basis is None:
            if manifest:
                permission.take_back(manifest, manifest_path, retract)
            return {"status": "not_permitted"}
        # The website refused this package: nothing is uploaded again until the live rights differ from those at
        # the refusal, the window passes or a new revision arrives. A recheck only makes the live read due.
        memory = manifest.get("refused_by_webapp")
        if (isinstance(memory, Mapping) and memory.get("simulation_revision") == revision
                and memory.get("rights_fingerprint") == check.get("rights_fingerprint")
                and isinstance(memory.get("at_epoch"), int)
                and 0 <= now() - memory["at_epoch"] < permission.LIVE_STATEMENT_RECHECK_SECONDS):
            return {"status": "refused", "reason": str(memory.get("reason"))}
        built = manifest.get("package") or {}
        needs_build = (manifest.get("simulation_revision") != revision or package_path.is_symlink()
                       or not package_path.is_file() or _sha256_file(package_path) != built.get("digest"))
        # The package out now stays readable through the build: the new zip is built beside it, the website is
        # moved to the new one by its delivery, and only then is the old one retired (``_publish_built``). A
        # rebuild that gives way (shape, size, time) leaves it exactly as it was.
        known_over = _size_over(manifest, revision)
    base = {"schema_version": MANIFEST_SCHEMA, **identity, "simulation_revision": revision,
            "source": {"plan_digest": source.get("plan_digest"),
                       "files": {prim: row["sha256"] for prim, row in sorted(source["files"].items())}},
            "upload_sha256": upload, "permission": basis, "built_at_epoch": int(now()), "archive": None,
            "delivered": None, "source_kind": source["source_kind"], "size_rule": SIZE_RULE,
            # Outside the revision: why a floor ships, and the refused half it stands for (never rebuilt as it is).
            **({"floor_reason": source["floor_reason"]} if source.get("floor_reason") else {}),
            **({"refused_half": manifest["refused_half"]} if source["source_kind"] == "floor"
               and isinstance(manifest.get("refused_half"), Mapping) else {})}
    candidate, outcome, shape = directory / f".candidate-{revision[7:23]}.zip", None, False
    over: list[str] = []
    if needs_build:
        try:
            built = build_package(source, candidate, literals=secret_literals(root, source),
                                  words=capture_site_identity(root), max_bytes=max_bytes,
                                  deadline=time.monotonic() + build_seconds, known_over=known_over, over=over)
        except PackageRefused as error:
            outcome = {**base, "package": None, "refused": error.reason, "detail": error.detail,
                       "refused_at_epoch": int(now())}
            shape = isinstance(error, ShapeRefused)
        except ValueError as error:
            if str(error) not in (OVER_CAP, BUILD_TIMED_OUT):
                candidate.unlink(missing_ok=True)
                raise
            outcome = {**base, "package": None, "refused": str(error), "refused_at_epoch": int(now())}
            shape = True
        # The steps already found over the limit: the next try of this revision starts after them.
        size_over = [step for step in SIZE_STEPS if step in {*known_over, *over}]
        if outcome is not None and size_over:
            outcome["size_over"] = size_over
        if outcome is not None and shape and source["source_kind"] == "half":
            # Its own scene or its size: the room floor ships instead until a new half revision arrives. Its build
            # time: the floor ships meanwhile, and the half is tried again once a window (``package_source``).
            outcome["refused_half"] = {"simulation_revision": revision, "reason": outcome["refused"],
                                       "size_rule": SIZE_RULE, "refused_at_epoch": outcome["refused_at_epoch"],
                                       **({"size_over": size_over} if size_over else {})}

    def stop(result: dict[str, Any]) -> dict[str, Any]:
        # The package kept readable through the build is taken back whenever the build itself is stopped.
        if needs_build:
            current = _read(manifest_path)
            if current and not permission.take_back(current, manifest_path, retract):
                return {"status": "failed", "reason": "website_company_simulation_retraction_pending"}
        return result

    try:
        with package_lock(root):
            if capture_withdrawn(root):
                return stop({"status": "withdrawn"})
            # The whole decision again before any upload: a takedown, a recorded "no" or removal from Blueprint's
            # list that landed while the package was being built stops it here.
            if not permission.delivery_gate_allows(root, surface="website_company_simulation"):
                return stop({"status": "refused", "reason": "website_company_simulation_takedown_gate_closed"})
            if not permission.basis_holds(root, identity["capture_id"], basis["permission_basis"], now()):
                recorded = permission.read_check(root)
                if recorded.get("outcome") == "rights_refused":
                    return stop({"status": "refused",
                                 "reason": _own_reason(recorded.get("reason") or permission.RIGHTS_REFUSED)})
                return stop({"status": "not_permitted"})
            if needs_build:
                fresh = package_source(root, queue_root=queue_root, manifest=manifest, max_bytes=max_bytes, now=now())
                if fresh is None or simulation_revision(fresh, max_bytes=max_bytes) != revision:
                    return {"status": "failed", "reason": "website_company_simulation_changed_during_build"}
                current = _read(manifest_path)
                if outcome is not None:
                    logger.warning("website_company_simulation_not_built capture=%s reason=%s",
                                   identity["capture_id"], outcome["refused"])
                    if _kept_copies(current):
                        # A rebuild refused, out of time or over the limit leaves what is out exactly as it was, still
                        # read; its refusal is kept beside it for its revision (and a half's, for the floor rule).
                        write_json(manifest_path, {**current, "rebuild_refused": {
                            key: outcome[key] for key in ("simulation_revision", "refused", "detail", "refused_at_epoch",
                                                          "size_rule", "size_over") if key in outcome},
                            **({"refused_half": outcome["refused_half"]} if outcome.get("refused_half") else {})})
                        return _not_built(outcome)
                    package_path.unlink(missing_ok=True)
                    write_json(manifest_path, outcome)
                    return _not_built(outcome)
                superseded = _kept_copies(current, besides=built["digest"])
                os.replace(candidate, package_path)
                manifest = {**base, "package": {**built, "path": PACKAGE_FILE},
                            **({permission.SUPERSEDED: superseded} if superseded else {}),
                            **({"refused_by_webapp": manifest["refused_by_webapp"]}
                               if isinstance(manifest.get("refused_by_webapp"), Mapping) else {})}
            return _publish_built(root, identity, manifest, manifest_path, package_path, basis, upload, built,
                                  revision, check, now, archive, retract, deliver)
    finally:
        candidate.unlink(missing_ok=True)


def _refusal(manifest: Mapping[str, Any], revision: str) -> Mapping[str, Any]:
    """This revision's recorded refusal: the manifest's own, or the failed rebuild's kept beside a package out."""
    for record in (manifest, manifest.get("rebuild_refused")):
        if isinstance(record, Mapping) and record.get("simulation_revision") == revision and record.get("refused"):
            return record
    return {}


def _size_over(manifest: Mapping[str, Any], revision: str) -> list[str]:
    """The ``SIZE_STEPS`` an earlier pass found over the limit for this revision (its own record, a failed rebuild's
    kept beside a package out, or the half's a floor stands for)."""
    for record in (manifest, manifest.get("rebuild_refused"), manifest.get("refused_half")):
        if (isinstance(record, Mapping) and record.get("simulation_revision") == revision
                and isinstance(record.get("size_over"), list)):
            return [step for step in SIZE_STEPS if step in record["size_over"]]
    return []


def _kept_copies(current: Mapping[str, Any], *, besides: str | None = None) -> list[dict[str, Any]]:
    """Every archived copy a company may still be reading: the package out and each one kept since the last the
    website is known to have taken (a delivery whose answer was lost may have moved it to any of them), never
    the one named ``besides``. Retired only once a newer package is taken (``_publish_built``)."""
    kept: dict[str, dict[str, Any]] = {}
    for record in [*(current.get(permission.SUPERSEDED) or []), current.get("archive")]:
        if (isinstance(record, Mapping) and isinstance(record.get("uri"), str) and not permission.hidden(record)
                and record.get("digest") != besides):
            kept[str(record.get("digest"))] = dict(record)
    return list(kept.values())


def _not_built(outcome: Mapping[str, Any]) -> dict[str, Any]:
    if outcome["refused"] == OVER_CAP:
        return {"status": "over_size_cap"}
    if outcome["refused"] == BUILD_TIMED_OUT:
        return {"status": "build_timed_out"}
    return {"status": "refused", "reason": outcome["refused"], "detail": outcome["detail"]}


def _publish_built(root, identity, manifest, manifest_path, package_path, basis, upload, built, revision, check,
                   now, archive, retract, deliver):
    manifest["permission"], manifest["upload_sha256"] = basis, upload
    record = package_record(identity=identity, manifest=manifest)
    manifest["simulation_digest"] = record["simulation_digest"]
    write_json(manifest_path, manifest)
    stored = manifest.get("archive") or {}
    if stored.get("digest") != built["digest"] or permission.hidden(stored):
        manifest["archive"] = archive(package_path, identity=identity)
        if manifest["archive"]["digest"] != built["digest"]:
            raise ValueError("website_company_simulation_archive_mismatch")
        write_json(manifest_path, manifest)
    if capture_withdrawn(root):
        permission.take_back(manifest, manifest_path, retract)
        return {"status": "withdrawn"}
    try:
        try:
            deliver(record, identity=identity)
        except Exception as error:  # only the record's shape is retried here
            if "object_included" not in record or not _record_refused(error):
                raise
            # A website that predates ``object_included``: the same package, its record without the field. The
            # digest that is out is this one, so a later pass finds it current and never sends it again.
            record = package_record(identity=identity, manifest=manifest, with_object_included=False)
            manifest["simulation_digest"] = record["simulation_digest"]
            write_json(manifest_path, manifest)
            deliver(record, identity=identity)
    except Exception as error:  # noqa: BLE001 - classified below; the archive stands only with rights
        reason = _own_reason(error)
        if permission.RIGHTS_REFUSALS.search(str(error)):
            permission.take_back(manifest, manifest_path, retract)
            manifest["refused_by_webapp"] = {"simulation_revision": revision, "reason": reason,
                                             "at_epoch": int(now()), "rights_fingerprint": check.get("rights_fingerprint")}
            write_json(manifest_path, manifest)
            return {"status": "refused", "reason": reason}
        manifest["last_delivery_error"] = reason
        write_json(manifest_path, manifest)
        return {"status": "failed", "reason": reason}
    manifest["delivered"] = {"simulation_digest": record["simulation_digest"], "at_epoch": int(now())}
    manifest.pop("last_delivery_error", None)
    manifest.pop("refused_by_webapp", None)
    # The website reads the new package now: the copies it may have read before are marked for retirement in this
    # same write, so a stop before their deletion leaves them pending for the next pass, which retires them before
    # anything else (``retraction_pending``), the already-current path included.
    permission.mark_retiring(manifest)
    write_json(manifest_path, manifest)
    permission.retire(manifest, manifest_path, retract)
    return {"status": "published", "simulation_revision": revision, "simulation_digest": record["simulation_digest"]}


def tick(storage_root: Path, *, stop_reason: Callable[[], str | None] | None = None,
         budget_seconds: float = TICK_BUDGET_SECONDS, clock: Callable[[], float] = time.monotonic,
         **options: Any) -> dict[str, int]:
    """One pass over every capture the clip pass covers; counts of each outcome.

    As the clip's pass: between captures it stops for a hold, a quiesce or a new release (``stop_reason``) and
    starts no capture after ``budget_seconds``; the captures left are counted ``deferred``. One build also has
    its own deadline (``BUILD_DEADLINE_SECONDS``).
    """
    counts: dict[str, int] = {}
    started = clock()
    captures = clip.capture_roots(storage_root)
    for index, capture in enumerate(captures):
        reason = stop_reason() if stop_reason is not None else None
        if reason or clock() - started >= budget_seconds:
            counts["deferred"] = len(captures) - index
            counts["stopped_" + (reason or "time_budget")] = 1
            break
        result = publish_company_simulation(capture, **options)
        status = result["status"]
        # A floor the pass could not decide for (``object_path_stopped``: ``floor_unclassified:<code>``) is named
        # like a package that should exist and does not, under its own count.
        key = "floor_unclassified" if status == "no_source" and result.get("reason") else status
        counts[key] = counts.get(key, 0) + 1
        if status in NOT_PUBLISHED_ALOUD or key == "floor_unclassified":
            # Each capture whose package is not out, and why, in the unit's journal: the counts alone hide whose.
            logger.warning("website_company_simulation_outcome capture=%s status=%s reason=%s", Path(capture).name,
                           status, result.get("reason") or "")
    return counts


#: Outcomes a pass names by capture (``tick``): a package that should exist and does not. ``no_source`` and
#: ``not_permitted`` are the job's state, not a fault, and stay counts only, except a ``no_source`` whose floor
#: could not be decided, which is named too.
NOT_PUBLISHED_ALOUD = frozenset({"failed", "refused", "over_size_cap", "build_timed_out"})


#: The route's clock (the pass takes ``now`` from its caller); a test can set it.
def _wall_clock() -> float:
    return time.time()


def open_company_simulation(*, request_id: str, capture_id: str, digest: str, storage_root: Path | None,
                            range_header: str | None = None, client: Any = None, bucket: str | None = None,
                            now: float | None = None) -> tuple[Any, int, tuple[int, int] | None]:
    """The archived package body, its full size and the range served, for a read the WebApp authorized.

    It fails closed: with no storage root, a withdrawn site (found without the workspace), no capture
    workspace, a closed takedown gate, a manifest that does not name this digest as archived and not taken back,
    or a basis that no longer holds at ``now`` (the last live grant is younger than the bound and nothing has
    said no since), nothing is served.
    """
    if (storage_root is None or not capture_name_matches_request(request_id, capture_id)
            or not _DIGEST.fullmatch(str(digest)) or permission.site_withdrawn(storage_root, request_id)):
        raise CompanySimulationUnavailable("website_company_simulation_unavailable")
    roots = [root for root in Path(storage_root).glob(f"*/scenes/site-{request_id}/captures/{capture_id}")
             if root.is_dir() and not root.is_symlink()]
    if len(roots) != 1:
        raise CompanySimulationUnavailable("website_company_simulation_unavailable")
    manifest = _read(roots[0].joinpath(*DIRECTORY) / MANIFEST_FILE)
    # The package out now, or one the website may still read while a newer one is delivered (``SUPERSEDED``).
    stored = next((record for record in (manifest.get("archive"), *(manifest.get(permission.SUPERSEDED) or []))
                   if isinstance(record, Mapping) and record.get("digest") == digest), None) or {}
    moment = _wall_clock() if now is None else now
    if (not permission.delivery_gate_allows(roots[0], surface="website_company_simulation")
            or manifest.get("schema_version") != MANIFEST_SCHEMA or stored.get("digest") != digest
            or permission.hidden(stored)
            or not permission.basis_still_holds(roots[0], capture_id, manifest, moment)):
        raise CompanySimulationUnavailable("website_company_simulation_unavailable")
    from .task_evaluation_configured_scene_object_keys import LARGE_ARTIFACT_KEY_PREFIX
    from .task_evaluation_configured_scene_object_store import _artifact_object_store_client, _object_missing
    if client is None or bucket is None:
        client, bucket = _artifact_object_store_client()
    key = f"{LARGE_ARTIFACT_KEY_PREFIX}/{ARTIFACT_KIND}/sha256/{digest[7:]}/{package_filename(capture_id, request_id)}"
    try:
        head = client.head_object(Bucket=bucket, Key=key)
    except Exception as error:  # noqa: BLE001 - provider exception shapes vary
        if _object_missing(error):
            raise CompanySimulationUnavailable("website_company_simulation_unavailable") from None
        raise
    size = int(head["ContentLength"])
    if (head.get("Metadata") or {}).get("sha256") != digest[7:] or not 0 < size <= MAX_PACKAGE_BYTES:
        raise CompanySimulationUnavailable("website_company_simulation_unavailable")
    served = clip.parse_range(range_header, size)
    extra = {"Range": f"bytes={served[0]}-{served[1]}"} if served else {}
    return client.get_object(Bucket=bucket, Key=key, **extra)["Body"], size, served


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    commands = parser.add_subparsers(dest="command", required=True)
    one = commands.add_parser("publish", help="make and publish one capture's simulation package if it is not current")
    one.add_argument("--capture-root", type=Path, required=True)
    every = commands.add_parser("tick", help="make and publish every capture's simulation package that is not current")
    every.add_argument("--storage-root", type=Path, required=True)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    result = (publish_company_simulation(args.capture_root) if args.command == "publish"
              else tick(args.storage_root))
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
