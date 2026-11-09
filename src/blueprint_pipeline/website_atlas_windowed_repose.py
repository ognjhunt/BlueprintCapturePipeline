"""A new job's one windowed re-pose, before its refined splat: the term it was quoted under and its outcome.

Owner decision 2026-11-09 15:27:24Z: "yes". When a new job's first source pose comes
back without World Labs' pose refinement (``poseRefinementAccepted`` not true) and
the motion analysis on that pose finds footage to leave out (a pose jump, a
stationary run, views that disagree; ``atlas_splat_rebuild_window``), the job buys
one more pose of only its consistent moving footage before it buys the refined
splat. This is the same windowed pose a rebuild's retry makes, made before the
splat is bought rather than after.

The money:

- **The term.** A job records it once, before its first purchase
  (``TERM_FILE``). It names the owner's decision, the per-view price and the input
  bound (``SPLAT_POSE_INPUT_CAPACITY``, the bound the first pose's selection
  uses), and the most the extra pose can cost. The extra pose is conditional, so
  the job's expected cost leaves it out and its maximum includes it. The selection
  budget the job's scene review records holds that maximum back, so the first
  pose's selection never spends the re-pose's money. A job whose allowance cannot
  hold it back records the term disabled and goes on exactly as before.
- **In-flight jobs.** A job that bought anything before this release has no term
  and never re-poses: it keeps the quote it was admitted under. A recorded term is
  read back, never re-decided.
- **Before the re-pose.** Its inputs must fit the term's bound and the first
  pose's own count, its price must be the term's, and the job's remaining
  dollars and Atlas requests must both cover it and the final outputs still to buy
  (``atlas_pricing.headroom``, the forecast every optional paid step uses). The pose itself goes
  through the job's paid admission like any pose. It buys no scale review: its
  scale comes from the same measurement as any pose without one, and the re-pose
  is used only when that gives an estimated scale.
- **Its outcome** is recorded once beside it (``OUTCOME_FILE``). Only an accepted
  re-pose replaces the first pose for the splat. Anything else (no window, a
  refused selection, money, a pose World Labs again did not refine) leaves the
  job on its first pose, exactly as before. There is never a third pose.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .common import write_json
from .decision_evidence_contracts import canonical_digest

TERM_FILE = "windowed_repose.json"
TERM_SCHEMA = "website_atlas_windowed_repose_term.v1"
#: The re-pose's own pose root, beside the job's first (``pose``), so it shares the job's clean plate.
ROOT = "pose-windowed"
OUTCOME_FILE = "repose_outcome.json"
OUTCOME_SCHEMA = "website_atlas_windowed_repose_outcome.v1"
OWNER_DECISION = {"decided_by": "owner", "decided_at": "2026-11-09T15:27:24Z", "answer": "yes",
                  "question": "one windowed re-pose before the refined splat when a new job's first pose "
                              "is not refinement-accepted and the motion analysis finds a jump or a "
                              "no-parallax run"}
CONDITIONS = ("first_pose_refinement_not_accepted", "motion_analysis_window_found",
              "inputs_within_term_bound", "allowance_covers_repose_and_final_outputs")
#: The price review fields a term records, so its price is the one the job's rate card reviewed.
_PRICE_REVIEW = ("rate_card_reference", "reviewed_by", "reviewed_on")


def _read(path: Path) -> dict[str, Any] | None:
    if path.is_symlink() or not path.is_file():
        return None
    value = json.loads(path.read_text())
    if not isinstance(value, dict) or value.get("digest") != canonical_digest(value, digest_field="digest"):
        raise ValueError("website_atlas_windowed_repose_record_invalid")
    return value


def read_term(job_dir: Path) -> dict[str, Any] | None:
    """The job's recorded term, or None for a job that has none (one that bought before this release)."""
    value = _read(Path(job_dir) / TERM_FILE)
    if value is not None and value.get("schema_version") != TERM_SCHEMA:
        raise ValueError("website_atlas_windowed_repose_record_invalid")
    return value


def decide_term(job_dir: Path, *, policy: Mapping[str, Any], purchased: bool, available_usd: float,
                minimum_usd: float) -> dict[str, Any] | None:
    """The term this job was quoted under: read back, or decided once before its first purchase.

    ``available_usd`` is what the job's selection could spend before the term
    (its allowance less the reviews and final outputs held back); ``minimum_usd``
    the least a selection needs. A job that already bought, or whose route buys no
    refined splat, gets no term.
    """
    from .atlas_refined_splat_forms import SPLAT_POSE_INPUT_CAPACITY

    recorded = read_term(job_dir)
    if recorded is not None or purchased or "refined_splat_cost_usd" not in policy:
        return recorded
    price = float(policy["pose_input_cost_usd"])
    maximum = round(SPLAT_POSE_INPUT_CAPACITY * price, 6)
    enabled = available_usd - maximum >= minimum_usd
    value = {"schema_version": TERM_SCHEMA, "owner_decision": OWNER_DECISION, "enabled": enabled,
             **({} if enabled else {"reason": "allowance_cannot_hold_back_the_repose"}),
             "per_view_cost_usd": price, "max_input_images": SPLAT_POSE_INPUT_CAPACITY,
             # Conditional: the expected cost leaves it out, the maximum includes it.
             "expected_cost_usd": 0.0, "maximum_cost_usd": maximum if enabled else 0.0,
             "conditional_on": list(CONDITIONS),
             "price_review": {key: policy.get(key) for key in _PRICE_REVIEW},
             "policy_digest": canonical_digest(dict(policy))}
    value["digest"] = canonical_digest(value, digest_field="digest")
    write_json(Path(job_dir) / TERM_FILE, value)
    return value


def held_back_usd(term: Mapping[str, Any] | None) -> float:
    """What a job's selection budget holds back for its re-pose: the term's maximum, when enabled."""
    return float(term["maximum_cost_usd"]) if term and term.get("enabled") else 0.0


def read_outcome(job_dir: Path, binding: Mapping[str, Any]) -> dict[str, Any] | None:
    value = _read(Path(job_dir) / ROOT / OUTCOME_FILE)
    if value is not None and (value.get("schema_version") != OUTCOME_SCHEMA
                              or any(value.get(key) != item for key, item in binding.items())):
        raise ValueError("website_atlas_windowed_repose_record_invalid")
    return value


def record_outcome(job_dir: Path, binding: Mapping[str, Any], status: str, **details: Any) -> dict[str, Any]:
    """Record once what the re-pose came to; the job's splat reads it on every later tick."""
    value = {"schema_version": OUTCOME_SCHEMA, **binding, "status": status, **details}
    value["digest"] = canonical_digest(value, digest_field="digest")
    path = Path(job_dir) / ROOT / OUTCOME_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise ValueError("website_atlas_windowed_repose_outcome_exists")
    write_json(path, value)
    return value


def disclosure(outcome: Mapping[str, Any], window: Mapping[str, Any] | None) -> dict[str, Any]:
    """What a job's result says about its re-pose: whether it was used, and the footage it left out."""
    value = {"status": outcome["status"], "outcome_digest": outcome["digest"],
             **{key: outcome[key] for key in ("reason", "retry_pose_result_digest") if outcome.get(key)}}
    if window is not None and window.get("status") == "windowed":
        value.update(window_digest=window["digest"], intervals_seconds=window["intervals_seconds"],
                     left_out_required_frame_ids=window["left_out_required_frame_ids"])
    return value
