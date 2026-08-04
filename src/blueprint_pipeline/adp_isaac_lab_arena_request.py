"""Immutable Isaac Lab-Arena worker request for the founder sim-only protocol.

This module compiles the approved Blueprint schedule into Arena-native logical
jobs.  It does not import Arena, download assets, launch Isaac Sim, contact a
policy server, or authorize paid compute.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .adp_founder_sim_protocol import (
    ALTERNATIVE_ID,
    BASELINE_ID,
    build_founder_sim_protocol,
)
from .adp_prospective_design import validate_schedule_for_execution
from .common import write_json
from .decision_evidence_contracts import canonical_digest


SCHEMA_VERSION = "adp_isaac_lab_arena_worker_request.v1"


def _policy_binding(candidate_id: str) -> dict[str, Any]:
    if candidate_id == BASELINE_ID:
        return {
            "candidate_id": candidate_id,
            "arena_policy_type": ("isaaclab_arena_openpi.policy.pi0_remote_policy.Pi0RemotePolicy"),
            "policy_variant": "pi05",
            "openpi_embodiment_adapter": "droid",
            "remote_protocol": "websocket",
            "expected_checkpoint_identity": BASELINE_ID,
        }
    if candidate_id == ALTERNATIVE_ID:
        return {
            "candidate_id": candidate_id,
            "arena_policy_type": (
                "isaaclab_arena_gr00t.policy.gr00t_remote_closedloop_policy."
                "Gr00tRemoteClosedloopPolicy"
            ),
            "policy_config_yaml_path": (
                "isaaclab_arena_gr00t/policy/config/droid_manip_gr00t_closedloop_config.yaml"
            ),
            "remote_protocol": "zmq",
            "expected_checkpoint_identity": ALTERNATIVE_ID,
            "arena_native_groot_droid_adapter_required": True,
        }
    raise ValueError(f"arena_candidate_not_frozen:{candidate_id}")


def build_arena_worker_request(
    protocol: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compile the canonical protocol into exact, non-executable Arena jobs."""

    canonical = build_founder_sim_protocol()
    supplied = dict(protocol) if protocol is not None else canonical
    if supplied != canonical:
        raise ValueError("arena_worker_request_protocol_not_canonical")
    schedule = dict(canonical["schedule"])
    schedule_admission = validate_schedule_for_execution(schedule)
    condition = dict(canonical["conditions"][0])
    jobs: list[dict[str, Any]] = []
    for row_value in schedule["rows"]:
        row = dict(row_value)
        jobs.append(
            {
                "trial_id": row["trial_id"],
                "execution_order": row["execution_order"],
                "candidate_role": row["candidate_role"],
                "policy": _policy_binding(row["candidate_id"]),
                "environment": {
                    "type": condition["environment"],
                    "embodiment": condition["embodiment"],
                    "pick_up_object": condition["pick_up_object"],
                    "destination_location": condition["destination_location"],
                    "hdr": condition["hdr"],
                    "light_intensity": condition["light_intensity"],
                    "additional_table_objects": [],
                    "variations": {},
                    "enable_cameras": True,
                },
                "rollout": {
                    "num_envs": 1,
                    "num_episodes": 1,
                    "seed": row["seed"],
                    "language_instruction": canonical["task"]["instruction"],
                    "record_camera_video": True,
                },
                "reset_digest": row["reset_digest"],
                "required_blueprint_overlay": {
                    "lossless_policy_input_frames": True,
                    "terminal_observation_frame": True,
                    "frame_manifest": True,
                    "derived_review_video": True,
                    "independent_arena_success_provenance": True,
                },
            }
        )
    request: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "frozen_not_authorized_for_execution",
        "protocol_id": canonical["protocol_id"],
        "protocol_digest": canonical["protocol_digest"],
        "schedule_digest": schedule["schedule_digest"],
        "schedule_admission_digest": schedule_admission["admission_digest"],
        "runtime_identity": canonical["scene"]["simulator_stack"],
        "controls_required_before_candidate_jobs": [
            {
                "control_id": "arena_zero_action_negative",
                "arena_policy_type": "zero_action",
                "expected_task_success": False,
            },
            {
                "control_id": "arena_replay_or_scripted_positive",
                "expected_task_success": True,
                "action_fixture_digest": "required_before_execution",
            },
        ],
        "jobs": jobs,
        "job_count": len(jobs),
        "candidate_trials_interleaved_in_matched_pairs": True,
        "scenario_cousins_enabled": False,
        "execution_requires_separate_founder_approval_receipt": True,
        "paid_compute_authorized": False,
        "production_simulation_started": False,
        "physical_execution_authorized": False,
    }
    request["worker_request_digest"] = canonical_digest(
        request, digest_field="worker_request_digest"
    )
    return request


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    request = build_arena_worker_request()
    if args.output:
        write_json(Path(args.output).expanduser().resolve(), request)
    else:
        print(json.dumps(request, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["SCHEMA_VERSION", "build_arena_worker_request"]
