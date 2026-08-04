"""Reusable positive/negative control gate for Franka DROID simulation runs."""

from __future__ import annotations

import argparse
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .adp_prospective_design import validate_episode_evidence_contract
from .common import write_json
from .decision_evidence_contracts import canonical_digest
from .franka_droid_closed_loop import (
    ScriptedDroidJointPositionOracleClient,
    StationaryDroidJointPositionClient,
    prepare_franka_droid_runtime,
    run_franka_droid_closed_loop,
)


SCHEMA_VERSION = "franka_droid_control_preflight.v1"
PINNED_MENAGERIE_REVISION = "71f066ad0be9cd271f7ed58c030243ef157af9f4"


def _verified_menagerie_revision(root: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    revision = result.stdout.strip() if result.returncode == 0 else ""
    if revision != PINNED_MENAGERIE_REVISION:
        raise ValueError("franka_menagerie_revision_mismatch")
    return revision


def _episode_summary(result: Mapping[str, Any], admission: Mapping[str, Any]) -> dict[str, Any]:
    metrics = dict(result.get("metrics", {}))
    visual = dict(result.get("visual_evidence", {}))
    video = dict(visual.get("video", {}))
    return {
        "episode_id": result.get("episode_id"),
        "status": result.get("status"),
        "task_success": result.get("task_success"),
        "action_steps_executed": result.get("action_steps_executed"),
        "policy_query_count": result.get("policy_query_count"),
        "lift_delta_m": metrics.get("lift_delta_m"),
        "contained_in_tray_interior": metrics.get("contained_in_tray_interior"),
        "frame_manifest_digest": visual.get("frame_manifest_digest"),
        "review_video_sha256": video.get("sha256"),
        "episode_admission_digest": admission.get("episode_admission_digest"),
        "episode_result_digest": canonical_digest(result),
    }


def run_franka_droid_control_preflight(
    *, menagerie_root: str | Path, output_dir: str | Path
) -> dict[str, Any]:
    """Require a passing oracle and failing stationary control with full media."""

    root = Path(menagerie_root).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    revision = _verified_menagerie_revision(root)
    runtime = prepare_franka_droid_runtime(
        menagerie_root=root,
        output_dir=output / "runtime",
    )
    positive = run_franka_droid_closed_loop(
        runtime=runtime,
        policy_client=ScriptedDroidJointPositionOracleClient(
            runtime["targets"],
            initial_joint_target=runtime["targets"]["pregrasp"],
        ),
        output_dir=output / "positive",
    )
    negative = run_franka_droid_closed_loop(
        runtime=runtime,
        policy_client=StationaryDroidJointPositionClient(),
        output_dir=output / "negative",
        max_action_steps=24,
    )
    positive_admission = validate_episode_evidence_contract(positive)
    negative_admission = validate_episode_evidence_contract(negative)
    blockers: list[str] = []
    if positive.get("status") != "completed" or positive.get("task_success") is not True:
        blockers.append("scripted_positive_control_did_not_succeed")
    if negative.get("status") != "completed" or negative.get("task_success") is not False:
        blockers.append("stationary_negative_control_did_not_fail")
    if positive_admission.get("status") != "admitted":
        blockers.append("scripted_positive_control_evidence_not_admitted")
    if negative_admission.get("status") != "admitted":
        blockers.append("stationary_negative_control_evidence_not_admitted")
    receipt: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if not blockers else "blocked",
        "menagerie": {
            "repository": "google-deepmind/mujoco_menagerie",
            "revision": revision,
            "asset_path": "franka_emika_panda",
        },
        "positive_control": _episode_summary(positive, positive_admission),
        "negative_control": _episode_summary(negative, negative_admission),
        "blockers": blockers,
        "production_policy_episode_executed": False,
        "paid_compute_used": False,
        "physical_robot_moved": False,
    }
    receipt["preflight_digest"] = canonical_digest(receipt, digest_field="preflight_digest")
    write_json(output / "franka_droid_control_preflight.json", receipt)
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--menagerie-root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    receipt = run_franka_droid_control_preflight(
        menagerie_root=args.menagerie_root,
        output_dir=args.output_dir,
    )
    return 0 if receipt["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "PINNED_MENAGERIE_REVISION",
    "SCHEMA_VERSION",
    "run_franka_droid_control_preflight",
]
