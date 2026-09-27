"""Run one approved team policy in the retained native G1 task scene.

The paid controller supplies a private execution packet, scene packet, pinned
SONIC assets, and (for HTTPS only) a protected credential file. This worker
does not allocate a GPU or claim provider teardown, billing, or public rights.
It retains a preclose receipt because Isaac may exit its interpreter in close().
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import (
    canonical_digest,
    cross_runtime_canonical_digest,
)
from .native_g1_development_worker import (
    _build_scene,
    _launch_scene,
    _to_tensor,
    _verify_packet,
)
from .native_g1_official_sonic_target_bridge import _require_artifact, require_pinned_sonic_source
from .native_g1_runtime_assembly import build_pinned_g1_team_sonic_bridge
from .native_g1_team_policy_approval import validate_g1_team_policy_approval
from .native_g1_team_policy_execution_packet import SCHEMA as PACKET_SCHEMA
from .native_g1_team_policy_run_request import validate_g1_team_policy_run_request
from .native_g1_team_supervised_episode import run_g1_team_supervised_episode
from .task_evaluation_packet_planning_setup import validate_packet_planning_setup


SCHEMA = "native_g1_team_policy_worker_result.v1"
FILENAME = SCHEMA + ".json"
PRECLOSE_SCHEMA = "native_g1_team_policy_worker_preclose.v1"
PRECLOSE_FILENAME = PRECLOSE_SCHEMA + ".json"
_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
_INTENT_ID = re.compile(r"g1-team-policy-[0-9a-f]{64}\Z")


def _execution_packet(path: Path, expected_commit: str) -> dict[str, Any]:
    if (
        not path.is_absolute() or path.is_symlink() or not path.is_file()
        or path.stat().st_size > 1024 * 1024
        or not isinstance(expected_commit, str)
        or _COMMIT.fullmatch(expected_commit) is None
    ):
        raise ValueError("g1_team_worker_execution_packet_unavailable")
    value = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != PACKET_SCHEMA
        or value.get("status") != "approved_input_not_executed"
        or value.get("implementation_commit") != expected_commit
        or value.get("packet_digest")
        != cross_runtime_canonical_digest(value, digest_field="packet_digest")
        or value.get("credential_value_included") is not False
        or value.get("artifact_bytes_included") is not False
        or value.get("provider_mutation_performed") is not False
        or value.get("claim_ceiling") != "development_only"
        or not isinstance(value.get("request"), dict)
        or not isinstance(value.get("trusted_setup"), dict)
        or not isinstance(value.get("operator_approval"), dict)
    ):
        raise ValueError("g1_team_worker_execution_packet_invalid")
    setup = validate_packet_planning_setup(value["trusted_setup"])
    request = validate_g1_team_policy_run_request(
        value["request"], trusted_setup=setup,
        authenticated_owner=value["request"].get("owner"),
    )
    approval = validate_g1_team_policy_approval(
        value["operator_approval"], profile=request["policy_profile"],
        trusted_setup=setup, authenticated_owner=request["owner"],
        objective_id=request["objective_id"],
    )
    if (
        not isinstance(value.get("intent_id"), str)
        or _INTENT_ID.fullmatch(value["intent_id"]) is None
        or not isinstance(value.get("intent_digest"), str)
        or _DIGEST.fullmatch(value["intent_digest"]) is None
        or value.get("policy_profile_digest") != request["policy_profile"]["profile_digest"]
        or value.get("source_packet_receipt_digest") != setup["source_packet_receipt_digest"]
        or value.get("objective_id") != request["objective_id"]
        or value.get("delivery_mode") != request["policy_profile"]["delivery"]["mode"]
        or approval["profile_digest"] != value["policy_profile_digest"]
    ):
        raise ValueError("g1_team_worker_execution_binding_invalid")
    return value


def _credential(path: Path | None, *, required: bool) -> str | None:
    if not required:
        if path is not None:
            raise ValueError("g1_team_worker_unexpected_credential_file")
        return None
    if (
        path is None or not path.is_absolute() or path.is_symlink()
        or not path.is_file() or not 0 < path.stat().st_size <= 16384
        or path.stat().st_mode & 0o077
    ):
        raise ValueError("g1_team_worker_credential_file_invalid")
    value = path.read_text(encoding="utf-8").strip()
    if not value or "\n" in value or "\r" in value:
        raise ValueError("g1_team_worker_credential_file_invalid")
    return value


def run_g1_team_policy_worker(
    *,
    execution_packet_path: Path,
    expected_implementation_commit: str,
    scene_packet_root: Path,
    runtime_provisioning_receipt_path: Path,
    sonic_provider_source: Path,
    sonic_encoder: Path,
    sonic_encoder_sha256: str,
    sonic_decoder: Path,
    sonic_decoder_sha256: str,
    output_dir: Path,
    credential_file_path: Path | None = None,
    max_steps: int = 3000,
) -> dict[str, Any]:
    """Run one scored scene and seal child teardown, even on worker failure."""

    root = Path(output_dir)
    if (
        not root.is_absolute() or root.exists() or root.is_symlink()
        or type(max_steps) is not int or not 1 <= max_steps <= 3000
    ):
        raise ValueError("g1_team_worker_output_or_step_bound_invalid")
    root.mkdir(parents=True, mode=0o700)
    phase = "execution_packet"
    packet = None
    scene_receipt = None
    app = None
    built = None
    launch = None
    device_binding = None
    episode = None
    failure: BaseException | None = None
    teardown = {"environment": "not_started", "simulator": "not_started"}
    try:
        packet = _execution_packet(Path(execution_packet_path), expected_implementation_commit)
        request = packet["request"]
        setup = packet["trusted_setup"]
        approval = packet["operator_approval"]
        profile = request["policy_profile"]
        scene_root = Path(scene_packet_root)
        phase = "scene_packet"
        if not scene_root.is_absolute() or scene_root.is_symlink():
            raise ValueError("g1_team_worker_scene_packet_path_invalid")
        scene_receipt = _verify_packet(scene_root)
        plan_path = scene_root / "native_task_arena_scene_plan.v1.json"
        if plan_path.is_symlink() or not plan_path.is_file():
            raise ValueError("g1_team_worker_scene_plan_missing")
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        if (
            not isinstance(plan, dict)
            or plan.get("plan_digest") != canonical_digest(plan, digest_field="plan_digest")
            or plan["plan_digest"] != scene_receipt["arena_scene_plan_digest"]
            or plan.get("scene_id") != setup["scene_id"]
            or plan.get("task_id") != setup["task_id"]
            or plan.get("task_kind") != "rigid_pick_place"
            or (plan.get("robot") or {}).get("robot_id") != "unitree_g1"
            or (plan.get("task_spec") or {}).get("task_success_contract_digest")
            != setup["task_success_contract_digest"]
        ):
            raise ValueError("g1_team_worker_scene_binding_invalid")
        phase = "sonic_asset_identity"
        require_pinned_sonic_source(sonic_provider_source)
        _require_artifact(sonic_encoder, sonic_encoder_sha256)
        _require_artifact(sonic_decoder, sonic_decoder_sha256)
        credential = _credential(
            credential_file_path,
            required=profile["delivery"]["mode"] == "authenticated_endpoint",
        )
        phase = "simulator_launch"
        print("BLUEPRINT_G1_TEAM_WORKER_PHASE:simulator_launch", flush=True)
        app, launch = _launch_scene(request={
            "runtime_provisioning_receipt_path": str(runtime_provisioning_receipt_path),
            "device": "cuda:0",
        }, plan=plan)
        phase = "scene_build"
        print("BLUEPRINT_G1_TEAM_WORKER_PHASE:scene_build", flush=True)
        built, device_binding = _build_scene(
            plan=plan, bundle_root=scene_root, device="cuda:0",
            dependency_receipt_path=root / "native_task_dependency_matrix.v1.json",
        )
        phase = "sonic_bridge"
        bridge = build_pinned_g1_team_sonic_bridge(
            built=built, source_path=sonic_provider_source,
            encoder_path=sonic_encoder, encoder_sha256=sonic_encoder_sha256,
            decoder_path=sonic_decoder, decoder_sha256=sonic_decoder_sha256,
        )
        phase = "scored_episode"
        print("BLUEPRINT_G1_TEAM_WORKER_PHASE:scored_episode", flush=True)
        import torch

        episode = run_g1_team_supervised_episode(
            built=built, profile=profile, trusted_setup=setup,
            authenticated_owner=request["owner"], operator_approval=approval,
            sonic_bridge=bridge, objective_id=request["objective_id"],
            max_steps=max_steps, output_dir=root / "episode",
            to_tensor=_to_tensor, make_action_tensor=torch.tensor,
            credential=credential,
        )
        if episode.get("status") != "completed_development_only":
            raise ValueError("g1_team_worker_episode_incomplete")
    except BaseException as exc:  # noqa: BLE001 - paid attempt retains terminal evidence
        failure = exc
    finally:
        if built is not None:
            try:
                built.env.close()
                teardown["environment"] = "closed"
            except BaseException as exc:  # noqa: BLE001
                teardown["environment"] = "close_failed:" + type(exc).__name__
                failure = failure or exc
        core = {
            "schema_version": SCHEMA,
            "claim_ceiling": "development_only",
            "phase_reached": phase,
            "execution_packet_digest": packet.get("packet_digest") if packet else None,
            "operator_approval_digest": (
                packet["operator_approval"]["approval_digest"] if packet else None
            ),
            "scene_packet_receipt_digest": scene_receipt.get("receipt_digest") if scene_receipt else None,
            "profile_digest": packet.get("policy_profile_digest") if packet else None,
            "objective_id": packet.get("objective_id") if packet else None,
            "delivery_mode": packet.get("delivery_mode") if packet else None,
            "isaaclab_launch": launch,
            "device_binding": device_binding,
            "supervised_episode_result_digest": episode.get("result_digest") if episode else None,
            "policy_query_count": episode.get("policy_query_count") if episode else 0,
            "blocker_type": type(failure).__name__ if failure else None,
            "ranking_eligible": False,
            "physical_outcome_claimed": False,
            "public_redistribution_authorized": False,
            "provider_teardown_verified": False,
            "official_billing_reconciled": False,
        }
        if app is not None:
            preclose = {
                **core,
                "schema_version": PRECLOSE_SCHEMA,
                "status": "awaiting_simulator_close",
                "teardown": {**teardown, "simulator": "close_requested"},
            }
            preclose["preclose_digest"] = canonical_digest(preclose, digest_field="preclose_digest")
            (root / PRECLOSE_FILENAME).write_text(
                json.dumps(preclose, indent=2, sort_keys=True, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            try:
                print("BLUEPRINT_G1_TEAM_WORKER_PHASE:simulator_close", flush=True)
                app.close()
                teardown["simulator"] = "closed"
            except SystemExit as exc:
                if exc.code in (None, 0):
                    teardown["simulator"] = "closed"
                else:
                    teardown["simulator"] = "close_failed:SystemExit"
                    failure = failure or exc
            except BaseException as exc:  # noqa: BLE001
                teardown["simulator"] = "close_failed:" + type(exc).__name__
                failure = failure or exc
        completed = (
            failure is None
            and episode is not None
            and episode.get("status") == "completed_development_only"
            and type(episode.get("policy_query_count")) is int
            and episode["policy_query_count"] > 0
            and teardown == {"environment": "closed", "simulator": "closed"}
        )
        result = {
            **core,
            "status": "completed_development_only" if completed else "blocked",
            "blocker_type": type(failure).__name__ if failure else None,
            "teardown": teardown,
        }
        result["result_digest"] = canonical_digest(result, digest_field="result_digest")
        (root / FILENAME).write_text(
            json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    return result


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "execution-packet", "scene-packet-root", "runtime-provisioning-receipt",
        "sonic-provider-source", "sonic-encoder", "sonic-decoder", "output-dir",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--expected-implementation-commit", required=True)
    parser.add_argument("--sonic-encoder-sha256", required=True)
    parser.add_argument("--sonic-decoder-sha256", required=True)
    parser.add_argument("--credential-file", type=Path)
    parser.add_argument("--max-steps", type=int, default=3000)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _argument_parser()
    args = parser.parse_args(argv)
    result = run_g1_team_policy_worker(
        execution_packet_path=args.execution_packet,
        expected_implementation_commit=args.expected_implementation_commit,
        scene_packet_root=args.scene_packet_root,
        runtime_provisioning_receipt_path=args.runtime_provisioning_receipt,
        sonic_provider_source=args.sonic_provider_source,
        sonic_encoder=args.sonic_encoder,
        sonic_encoder_sha256=args.sonic_encoder_sha256,
        sonic_decoder=args.sonic_decoder,
        sonic_decoder_sha256=args.sonic_decoder_sha256,
        output_dir=args.output_dir,
        credential_file_path=args.credential_file,
        max_steps=args.max_steps,
    )
    print(json.dumps({"status": result["status"], "result_digest": result["result_digest"]}))
    return 0 if result["status"] == "completed_development_only" else 1


if __name__ == "__main__":
    sys.exit(main())
