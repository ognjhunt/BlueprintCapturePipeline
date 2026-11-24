"""Every code the website is sent about a job's simulation, derived from the code that sends it.

The WebApp words each code it reads (``server/utils/taskPreparationOutcome.ts``). A code it has no words for shows the
site a general "stopped" or "still working" card and alerts staff (``atlasPreviewReasonKnown``,
``websitePreparationStatus.unknownPreparationCodes``). On 2026-11-24 the stress tests found such codes one at a time:
``atlas_completion_visual_review_required``, ``atlas_preview_step_interrupted``, the scene planner's review refusals and
``atlas_source_rotation_unsupported``. This test lists them all at once, so a change that adds one fails here until the
fixture names it, and the WebApp's copy of the fixture fails there until it has words for it.

``docs/webapp_handoff/website-pipeline-codes.v1.json`` holds the list; Blueprint-WebApp
``contracts/pipeline/website-pipeline-codes.v1.json`` is a byte-for-byte copy (``test_webapp_contract_mirrors``, and the
WebApp's ``npm run pipeline:mirrors:verify`` against Pipeline main), and the WebApp's
``server/tests/pipeline-codes-fixture.test.ts`` checks that each code in it has words. Regenerate it with
``BLUEPRINT_REGENERATE_PIPELINE_CODES_GOLDEN=1`` after changing a code, and copy it to the WebApp.

What is listed, for the Atlas preview's ``reason`` (state ``needs_input`` or ``failed``), by the class the worker gives it
(``website_atlas_preview._failed_step`` and ``advance_preparation_step``):

- ``site_input``: a new video or the site's input moves it on (``_NEEDS_NEW_CAPTURE``, ``_NEEDS_TASK_INPUT``, the scene
  review's abstentions, ``website_task_not_named``).
- ``recheck``: rechecked until an owner, an operator or Blueprint's own setup lifts it (``_AWAITING_AUTHORITY``).
- ``deterministic``: the same inputs and release repeat it (``deterministic_refusal``), raised in the Atlas lane.
- ``wall``: raised in the Atlas lane as a wall, by its type (``hard_stops.MoneyStop``, ``PrivacyRightsStop``, the
  website's own wall refusals) or by its words (``job_retry_policy.hard_stop_kind``); and the paid admission's blockers
  (``admission_refusal_code``). It can be stored as either state: its words, and where it is raised, decide which.
- ``ended``: the worker's own end of a job (its deadline, its source, the owner's closing, the spending cap, a step
  killed each time it ran, a request no resend can make succeed).
- ``parked``: the worker's own name for a stop with no code of its own (``advance_preparation_step``), and the
  completion review's own stops (``COMPLETION_FLOOR_REASONS``).

Each row's ``wall`` is ``money``, ``privacy_rights`` or null: the wall the Pipeline reads the code as, by its type or
its words, whatever its class. A code's words can name a wall it guards none of (a token count's "budget"); the row says
what the Pipeline reads, and the website says what is true.

Not listed: any other code raised in the lane. It takes the refusal schedule and, once spent, ends the job ``failed``;
the website's general card for an ended job says what is true of it ("a step on our side stopped before it finished"),
and its alert is how staff learn of a code nobody listed. The Atlas lane is the ``atlas_*``, ``website_atlas_*`` and
``website_target_background_*`` modules and ``LANE_EXTRA``; a code raised elsewhere is not found here.

For the preparation status (``website_preparation_status``): the outcome's reasons, categories and causes (the one
blocker ``website_preparation_not_possible`` names), and the terminal reasons.
"""
from __future__ import annotations

import ast
import json
import os
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
PACKAGE = REPO / "src" / "blueprint_pipeline"
GOLDEN = REPO / "docs" / "webapp_handoff" / "website-pipeline-codes.v1.json"
SCHEMA = "website_pipeline_codes.v1"
LANE_GLOBS = ("atlas_*.py", "website_atlas_*.py", "website_target_background_*.py")
#: Modules outside those names whose raises reach the Atlas preview step.
LANE_EXTRA = ("website_task_context", "website_job_spend_cap", "website_image_completion", "meta_sam31",
              "paid_resource_allocator", "website_task_preparation")
#: Lane-named modules that never run in the worker's step: replay and operator tools, and the delivery transport.
LANE_EXCLUDED = frozenset({"atlas_preview_replay", "atlas_reconstruction_stage_replay", "website_atlas_operator",
                           "website_atlas_preview_http"})
#: The paid admission's own blockers (``paid_resource_admission``, ``gcp_monthly_budget_admission``).
ADMISSION_MODULES = ("paid_resource_admission", "gcp_monthly_budget_admission")
#: The account's monthly spend check names its blockers by this prefix; the website words the prefix.
ACCOUNT_PREFIX = "account_monthly_spend_"
_CODE = re.compile(r"[a-z][a-z0-9_]{0,99}")


def _code(text: object) -> str | None:
    """The code a raise names, as ``atlas_preparation_job_control._reason`` reads it: before any ``:``."""
    if not isinstance(text, str):
        return None
    head = text.partition(":")[0].strip().lower().replace("-", "_")
    return head if _CODE.fullmatch(head) and "_" in head else None


def _texts(node: ast.AST, constants: dict[str, str]) -> list[str]:
    """The literal text of an argument: a string, a module constant, ``"code:" + detail``, or either branch of an if."""
    if isinstance(node, ast.IfExp):
        return _texts(node.body, constants) + _texts(node.orelse, constants)
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return [node.value]
    if isinstance(node, ast.Name) and node.id in constants:
        return [constants[node.id]]
    if isinstance(node, ast.JoinedStr) and node.values and isinstance(node.values[0], ast.Constant):
        first = str(node.values[0].value)
        return [first] if ":" in first else []
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return [text for text in _texts(node.left, constants) if ":" in text]
    return []


def _module(name: str) -> tuple[ast.Module, dict[str, str]]:
    tree = ast.parse((PACKAGE / f"{name}.py").read_text(encoding="utf-8"))
    constants = {node.targets[0].id: node.value.value for node in tree.body
                 if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                 and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)}
    return tree, constants


def lane_modules() -> list[str]:
    names = {path.stem for pattern in LANE_GLOBS for path in PACKAGE.glob(pattern)} | set(LANE_EXTRA)
    return sorted(names - LANE_EXCLUDED)


def lane_raises() -> dict[str, dict[str, set[str]]]:
    """Each code raised in the lane, by the call that raises it, and the modules raising it."""
    found: dict[str, dict[str, set[str]]] = {}
    for name in lane_modules():
        tree, constants = _module(name)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not node.args:
                continue
            callee = (node.func.id if isinstance(node.func, ast.Name)
                      else node.func.attr if isinstance(node.func, ast.Attribute) else None)
            if callee not in {"ValueError", "RuntimeError", "MoneyStop", "PrivacyRightsStop", "DeterministicRefusal",
                              "_review_refusal"}:
                continue
            for text in _texts(node.args[0], constants):
                code = _code(text)
                if code is not None:
                    found.setdefault(code, {}).setdefault(callee, set()).add(name)
    return found


def admission_blockers() -> set[str]:
    """The paid admission's literal blockers: what ``blockers`` is given, and the lists its checks return."""
    found: set[str] = set()
    for name in ADMISSION_MODULES:
        tree, constants = _module(name)
        for node in ast.walk(tree):
            values: list[ast.AST] = []
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name) and node.func.value.id == "blockers"
                    and node.func.attr == "append"):
                values = list(node.args)
            elif isinstance(node, ast.Return) and isinstance(node.value, ast.List):
                values = list(node.value.elts)
            elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                  and node.func.id == "PaidResourceAdmissionBlocked" and node.args
                  and isinstance(node.args[0], ast.List)):
                values = list(node.args[0].elts)
            for value in values:
                for text in _texts(value, constants):
                    if _CODE.fullmatch(text) and not text.startswith(ACCOUNT_PREFIX):
                        found.add(text)
    from blueprint_pipeline.paid_spend_meter import JOURNAL_UNWRITABLE  # imported by name where it is raised
    return found | {JOURNAL_UNWRITABLE, "paid_resource_admission_blocked"}


def atlas_preview_reasons() -> dict[str, dict[str, object]]:
    from blueprint_pipeline import atlas_preparation_job_control as control
    from blueprint_pipeline import website_atlas_preview as preview
    from blueprint_pipeline.atlas_automation_plan import COMPLETION_FLOOR_REASONS
    from blueprint_pipeline.hard_stops import JOB_SPEND_CAP, WEBAPP_MONEY, WEBAPP_PRIVACY_RIGHTS
    from blueprint_pipeline.job_retry_policy import (
        _UNSENDABLE_REQUEST_CODES, MONEY, OWNER_CLOSED_CODE, PRIVACY_RIGHTS, hard_stop_kind, provider_says_retry_later,
        request_cannot_succeed)
    from blueprint_pipeline.website_job_spend_cap import SPENDING_CAP_REACHED
    from blueprint_pipeline.website_recapture_required import ABSTENTION_CODES

    reasons: dict[str, dict[str, object]] = {}
    walls = {PRIVACY_RIGHTS: "privacy_rights", MONEY: "money"}

    def add(code: str, kind: str, states: tuple[str, ...], *, wall: str | None = None,
            raised_in: set[str] = frozenset()) -> None:
        # The first class a code is given is the one the worker reads first (``_failed_step``), and its states are that
        # class's; a wall it is read as anywhere stays with it, privacy and rights before money (``hard_stop_kind``).
        row = reasons.setdefault(code, {"class": kind, "states": set(), "wall": None, "raised_in": set()})
        if row["class"] == kind:
            row["states"] |= set(states)
        row["raised_in"] |= set(raised_in)
        if wall is not None and row["wall"] != "privacy_rights":
            row["wall"] = wall

    both = ("needs_input", "failed")
    add(OWNER_CLOSED_CODE, "ended", ("failed",), wall="privacy_rights")
    for code in sorted(preview._AWAITING_AUTHORITY - {"website_task_not_named"}):
        add(code, "recheck", ("needs_input",))
    for code in sorted(control._NEEDS_NEW_CAPTURE | control._NEEDS_TASK_INPUT | set(ABSTENTION_CODES.values())
                       | {"atlas_scene_planner_abstained", "website_task_not_named"}):
        add(code, "site_input", ("needs_input",))
    raised = lane_raises()
    for code, callers in sorted(raised.items()):
        if "DeterministicRefusal" in callers or "_review_refusal" in callers:
            add(code, "deterministic", ("needs_input",), raised_in=set().union(*callers.values()))
    # The worker's own ends: its deadline, a source it never had, a step killed each time it ran, the spending cap,
    # and a request no resend can make succeed (``request_cannot_succeed``).
    for code in sorted(control._NOT_RETRYABLE | _UNSENDABLE_REQUEST_CODES) + ["atlas_preview_step_interrupted"]:
        add(code, "ended", ("failed",))
    add(SPENDING_CAP_REACHED, "ended", ("failed",), wall="money")
    # The worker's names for a stop with no code of its own, and the completion review's own stops: the floor now ships
    # past those (``website_atlas_pipeline.completion_floor``), and a job that stopped on one before keeps its code.
    for code in sorted({"atlas_background_review_required", "atlas_preserved_completion_not_admitted"}
                       | COMPLETION_FLOOR_REASONS):
        add(code, "parked", ("needs_input",))
    for code, callers in sorted(raised.items()):
        if code in control._WITHDRAWN or code == JOB_SPEND_CAP:
            continue
        # By its type, else by its words where the worker reads them (not for a request that cannot succeed).
        words = (None if request_cannot_succeed(code) or provider_says_retry_later(code)
                 else walls.get(hard_stop_kind(code)))
        kind = "privacy_rights" if "PrivacyRightsStop" in callers else "money" if "MoneyStop" in callers else words
        if kind is not None:
            add(code, "wall", both, wall=kind, raised_in=set().union(*callers.values()))
            if words is not None and words != kind:
                add(code, "wall", both, wall=words)
    for code in sorted(WEBAPP_PRIVACY_RIGHTS - control._WITHDRAWN - {OWNER_CLOSED_CODE}):
        add(code, "wall", both, wall="privacy_rights")
    for code in sorted(WEBAPP_MONEY - {JOB_SPEND_CAP}):
        add(code, "wall", both, wall="money")
    for code in sorted(admission_blockers()):
        add(code, "wall", ("needs_input",), wall="money")
    assert not set(reasons) & control._WITHDRAWN, "a withdrawal is its own state, never a reason"
    return {code: {"class": row["class"], "states": sorted(row["states"]), "wall": row["wall"],
                   "raised_in": sorted(row["raised_in"])} for code, row in sorted(reasons.items())}


def preparation_status_codes() -> dict[str, list[str]]:
    """The outcome's reasons, categories and causes, and the terminal reasons, as the status can send them."""
    from blueprint_pipeline import website_preparation_outcome as outcome
    from blueprint_pipeline.website_preparation_terminal import REASONS
    from blueprint_pipeline.website_recapture_required import CODES
    from blueprint_pipeline.website_task_context_contracts import TASK_NOT_NAMED

    tree, constants = _module("website_preparation_outcome")
    reasons, categories = set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values):
                if isinstance(key, ast.Constant) and key.value in {"reason", "category"}:
                    texts = [text for text in _texts(value, constants) if _CODE.fullmatch(text)]
                    (reasons if key.value == "reason" else categories).update(texts)
    reasons |= set(CODES) | {TASK_NOT_NAMED} | set(outcome.NEW_REASON_FALLBACK.values())
    return {"outcome_reasons": sorted(reasons), "outcome_categories": sorted(categories),
            "outcome_causes": sorted(outcome._MISSING), "terminal_reasons": sorted(REASONS)}


def codes() -> dict[str, object]:
    prefixes = {ACCOUNT_PREFIX: {"class": "wall", "states": ["needs_input"], "wall": "money", "raised_in": []}}
    return {"schema_version": SCHEMA, "atlas_preview": {"reasons": atlas_preview_reasons(), "prefixes": prefixes},
            "preparation_status": preparation_status_codes()}


# Literal entries only: the impacted lane reads them from this source (``impacted_pinned_sets``). They are LANE_GLOBS,
# LANE_EXTRA, ADMISSION_MODULES and the modules whose sets and rules ``codes`` reads.
@pytest.mark.pins("src/blueprint_pipeline/atlas_*.py", "src/blueprint_pipeline/website_atlas_*.py",
                  "src/blueprint_pipeline/website_target_background_*.py", "blueprint_pipeline.website_task_context",
                  "blueprint_pipeline.website_job_spend_cap", "blueprint_pipeline.website_image_completion",
                  "blueprint_pipeline.meta_sam31", "blueprint_pipeline.paid_resource_allocator",
                  "blueprint_pipeline.website_task_preparation", "blueprint_pipeline.paid_resource_admission",
                  "blueprint_pipeline.gcp_monthly_budget_admission", "blueprint_pipeline.paid_spend_meter",
                  "blueprint_pipeline.hard_stops", "blueprint_pipeline.job_retry_policy",
                  "blueprint_pipeline.website_preparation_outcome", "blueprint_pipeline.website_preparation_terminal",
                  "blueprint_pipeline.website_recapture_required", "blueprint_pipeline.website_task_context_contracts")
def test_the_fixture_lists_every_code_the_website_is_sent():
    value = codes()
    if os.environ.get("BLUEPRINT_REGENERATE_PIPELINE_CODES_GOLDEN") == "1":
        GOLDEN.write_text(json.dumps(value, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    added = sorted(set(value["atlas_preview"]["reasons"]) - set(golden["atlas_preview"]["reasons"]))
    removed = sorted(set(golden["atlas_preview"]["reasons"]) - set(value["atlas_preview"]["reasons"]))
    assert golden == value, (f"{GOLDEN.name} is out of date (Atlas reasons added {added}, removed {removed}): "
                             "regenerate it with BLUEPRINT_REGENERATE_PIPELINE_CODES_GOLDEN=1, give the WebApp words "
                             "for each new code and copy the file to its server/tests/fixtures/")


def test_the_scan_finds_the_codes_the_stress_tests_found_one_at_a_time():
    reasons = codes()["atlas_preview"]["reasons"]
    assert reasons["atlas_source_rotation_unsupported"]["class"] == "site_input"
    assert reasons["atlas_completion_visual_review_required"]["class"] == "parked"
    assert reasons["atlas_preview_step_interrupted"] == {"class": "ended", "states": ["failed"], "wall": None,
                                                         "raised_in": []}
    assert reasons["atlas_scene_planner_unsafe_frames_invalid"]["class"] == "deterministic"
    assert reasons["atlas_scene_planner_unsafe_frames_invalid"]["raised_in"] == ["atlas_scene_planner"]
    # A typed wall the word backstop misses is still a wall here, and a recheck keeps the wall it is.
    assert reasons["website_atlas_pose_disclosure_not_authorized"]["wall"] == "privacy_rights"
    assert reasons["website_atlas_pose_admission_expired"]["wall"] == "money"
    assert reasons["atlas_runtime_price_review_expired"]["class"] == "recheck"
    assert reasons["atlas_runtime_price_review_expired"]["wall"] == "money"
    # A request that cannot succeed ends the job at once, whatever its words ("ceiling").
    assert reasons["meta_sam_inline_clip_exceeds_request_ceiling"] == {"class": "ended", "states": ["failed"],
                                                                       "wall": None, "raised_in": []}
    # The website's own wall refusals, and the paid admission's blockers.
    assert reasons["website_atlas_source_owner_mismatch"]["wall"] == "privacy_rights"
    assert reasons["gcp_monthly_budget_policy_invalid"] == {"class": "wall", "states": ["needs_input"],
                                                            "wall": "money", "raised_in": []}
    status = codes()["preparation_status"]
    assert {"website_preparation_not_possible", "website_preparation_retries_spent", "robot_fit_none",
            "website_capture_frames_unusable"} <= set(status["outcome_reasons"])
    assert status["terminal_reasons"] == ["budget_exhausted", "capacity_blocked", "out_of_scope", "permanent_invalid",
                                          "retry_exhausted"]
