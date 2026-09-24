"""Run one G1 development episode from the sealed native task/site packet.

This worker is callable in a local Isaac installation or the pinned Arena
container. It uses the same scene builder as Franka and owns Isaac, the G1
policy server, and SONIC teardown. Its result cannot qualify a policy ranking.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest
from .native_g1_policy_server_supervisor import PINNED_SOURCE_REVISION
from .native_g1_runtime_assembly import run_g1_supervised_built_scene_episode
from .native_g1_run_preflight import preflight_g1_shared_scene_run
from .native_g1_shared_scene_episode import G1_BOX_CANDIDATES


REQUEST_SCHEMA = "native_g1_development_episode_request.v1"
RESULT_SCHEMA = "native_g1_development_worker_result.v1"
RESULT_FILENAME = RESULT_SCHEMA + ".json"
RIGHTS_SCHEMA = "native_g1_development_rights_review.v1"
PATH_FIELDS = (
    "bundle_root", "inventory_path", "checkpoint_root", "policy_server_source",
    "sonic_provider_source", "sonic_encoder", "sonic_decoder",
    "python_executable", "runtime_provisioning_receipt_path",
)


def _request(value: Mapping[str, Any]) -> dict[str, Any]:
    try:
        request = json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError("g1_worker_request_invalid") from exc
    if not isinstance(request, dict):
        raise ValueError("g1_worker_request_invalid")
    if (
        request.get("schema_version") != REQUEST_SCHEMA
        or request.get("candidate_id") not in G1_BOX_CANDIDATES
        or request.get("device") != "cuda:0"
        or isinstance(request.get("port"), bool)
        or not isinstance(request.get("port"), int)
        or not 1 <= request["port"] <= 65535
        or isinstance(request.get("max_steps"), bool)
        or not isinstance(request.get("max_steps"), int)
        or not 1 <= request["max_steps"] <= 3000
        or any(not isinstance(request.get(field), str) or not request[field].strip() for field in PATH_FIELDS)
        or request.get("request_digest") != canonical_digest(request, digest_field="request_digest")
    ):
        raise ValueError("g1_worker_request_invalid")
    for field in ("sonic_encoder_sha256", "sonic_decoder_sha256"):
        digest = request.get(field)
        if (
            not isinstance(digest, str) or len(digest) != 71
            or not digest.startswith("sha256:")
            or any(char not in "0123456789abcdef" for char in digest[7:])
        ):
            raise ValueError("g1_worker_model_digest_invalid")
    return request


def _preflight_inputs(request: Mapping[str, Any]) -> dict[str, Any]:
    paths = {field: Path(request[field]).expanduser() for field in PATH_FIELDS}
    root = paths["bundle_root"]
    if root.is_symlink() or not root.is_dir():
        raise ValueError("g1_worker_bundle_root_invalid")
    scene = root / "native_task_arena_scene_plan.v1.json"
    if scene.is_symlink() or not scene.is_file():
        raise ValueError("g1_worker_scene_plan_missing")
    return {
        "scene_plan_path": scene,
        "bundle_root": root,
        "inventory_path": paths["inventory_path"],
        "candidate_id": request["candidate_id"],
        "checkpoint_root": paths["checkpoint_root"],
        "policy_server_source": paths["policy_server_source"],
        "sonic_provider_source": paths["sonic_provider_source"],
        "sonic_encoder": paths["sonic_encoder"],
        "sonic_encoder_sha256": request["sonic_encoder_sha256"],
        "sonic_decoder": paths["sonic_decoder"],
        "sonic_decoder_sha256": request["sonic_decoder_sha256"],
    }


def _verify_packet(bundle_root: Path) -> dict[str, Any]:
    from .native_task_arena_bundle import _verified_packet

    _, receipt, _ = _verified_packet(bundle_root)
    return receipt


def _rights_review(value: Any, *, preflight: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("g1_worker_rights_review_missing")
    receipt = dict(value)
    if (
        receipt.get("schema_version") != RIGHTS_SCHEMA
        or receipt.get("status") != "approved_for_development_simulation"
        or receipt.get("candidate_id") != preflight.get("candidate_id")
        or receipt.get("scene_plan_digest") != preflight.get("scene_plan_digest")
        or receipt.get("inventory_file_sha256") != preflight.get("inventory_file_sha256")
        or receipt.get("source_revision") != PINNED_SOURCE_REVISION
        or not str(receipt.get("human_reviewer") or "").strip()
        or receipt.get("checkpoint_terms_reviewed") is not True
        or receipt.get("source_and_sonic_terms_reviewed") is not True
        or receipt.get("rights_review_digest")
        != canonical_digest(receipt, digest_field="rights_review_digest")
    ):
        raise ValueError("g1_worker_rights_review_invalid")
    return receipt


def _to_tensor(value: Any) -> Any:
    if hasattr(value, "detach"):
        return value
    module = type(value).__module__
    if module == "warp" or module.startswith("warp."):
        import warp as wp

        return wp.to_torch(value)
    raise TypeError("g1_worker_sim_array_unsupported")


def _launch_scene(
    *, request: Mapping[str, Any], plan: Mapping[str, Any]
) -> tuple[Any, dict[str, Any]]:
    from .native_task_isaaclab_launch import launch_native_task_isaaclab
    from .native_task_nurec_render_setup import appearance_render_path_from_plan

    return launch_native_task_isaaclab(
        request["runtime_provisioning_receipt_path"],
        device=request["device"],
        appearance_render_path=appearance_render_path_from_plan(plan),
    )


def _build_scene(
    *, plan: Mapping[str, Any], bundle_root: Path, device: str
) -> tuple[Any, dict[str, Any]]:
    from .native_task_arena_construction_worker import preflight_native_dependency_matrix
    from .native_task_arena_device_readback import read_native_task_arena_device_binding
    from .native_task_arena_preconstruction import prepare_native_task_arena_preconstruction
    from .native_task_arena_runtime import build_native_task_arena_environment

    dependencies = preflight_native_dependency_matrix(robot_id="unitree_g1")
    if dependencies.get("all_required_available") is not True:
        raise ValueError("g1_worker_dependency_preflight_failed:" + ",".join(dependencies.get("blockers") or []))
    preconstruction = prepare_native_task_arena_preconstruction(expected_device=device)
    if preconstruction.get("passed") is not True:
        raise ValueError("g1_worker_preconstruction_failed:" + ",".join(preconstruction.get("blockers") or []))
    built = build_native_task_arena_environment(
        plan, device=device, bundle_root=bundle_root,
        preconstruction_receipt=preconstruction,
    )
    try:
        binding = read_native_task_arena_device_binding(built, expected_device=device)
        if binding.get("passed") is not True:
            raise ValueError("g1_worker_device_binding_failed:" + ",".join(binding.get("blockers") or []))
    except BaseException:
        try:
            built.env.close()
        finally:
            raise
    return built, binding


def run_g1_development_worker(
    *, request: Mapping[str, Any], output_dir: Path
) -> dict[str, Any]:
    """Run one attempt and retain its own terminal receipt on every failure."""

    sealed = _request(request)
    if not isinstance(output_dir, Path) or output_dir.exists() or output_dir.is_symlink():
        raise ValueError("g1_worker_output_directory_exists")
    output_dir.mkdir(parents=True)
    phase = "preflight"
    app = None
    built = None
    preflight = None
    packet_receipt = None
    rights = None
    launch = None
    device_binding = None
    episode = None
    failure: BaseException | None = None
    teardown: dict[str, Any] = {"environment": "not_started", "simulator": "not_started"}
    try:
        inputs = _preflight_inputs(sealed)
        phase = "packet_verification"
        packet_receipt = _verify_packet(inputs["bundle_root"])
        phase = "preflight"
        preflight = preflight_g1_shared_scene_run(**inputs)
        if (
            preflight.get("status") != "staged_inputs_verified"
            or preflight.get("scene_plan_digest")
            != packet_receipt.get("arena_scene_plan_digest")
        ):
            raise ValueError("g1_worker_preflight_incomplete")
        phase = "rights_review"
        rights = _rights_review(sealed.get("rights_review"), preflight=preflight)
        plan = json.loads(inputs["scene_plan_path"].read_text(encoding="utf-8"))
        if plan.get("plan_digest") != preflight.get("scene_plan_digest"):
            raise ValueError("g1_worker_scene_changed_after_preflight")
        phase = "simulator_launch"
        app, launch = _launch_scene(request=sealed, plan=plan)
        phase = "scene_build"
        built, device_binding = _build_scene(
            plan=plan, bundle_root=inputs["bundle_root"], device=sealed["device"]
        )
        phase = "episode"
        import torch

        episode = run_g1_supervised_built_scene_episode(
            built=built,
            candidate_id=sealed["candidate_id"],
            preflight_inputs=inputs,
            python_executable=Path(sealed["python_executable"]),
            port=sealed["port"], device=sealed["device"],
            max_steps=sealed["max_steps"], output_dir=output_dir / "episode",
            to_tensor=_to_tensor, make_action_tensor=torch.tensor,
        )
        if episode.get("status") != "completed_development_only":
            raise ValueError("g1_worker_supervised_episode_incomplete")
    except BaseException as exc:  # noqa: BLE001 - terminal evidence for failed attempts
        failure = exc
    finally:
        if built is not None:
            try:
                built.env.close()
                teardown["environment"] = "closed"
            except BaseException as exc:  # noqa: BLE001
                teardown["environment"] = "close_failed:" + type(exc).__name__
                if failure is None:
                    failure = exc
        if app is not None:
            try:
                app.close()
                teardown["simulator"] = "closed"
            except BaseException as exc:  # noqa: BLE001
                teardown["simulator"] = "close_failed:" + type(exc).__name__
                if failure is None:
                    failure = exc
        supervised_path = output_dir / "episode/native_g1_supervised_built_scene_episode.v1.json"
        try:
            supervised = json.loads(supervised_path.read_text()) if supervised_path.is_file() else episode
        except (OSError, ValueError) as exc:
            supervised = None
            if failure is None:
                failure = exc
        result = {
            "schema_version": RESULT_SCHEMA,
            "status": (
                "completed_development_only"
                if failure is None and episode is not None
                and episode.get("status") == "completed_development_only"
                and teardown == {"environment": "closed", "simulator": "closed"}
                else "blocked"
            ),
            "phase_reached": phase,
            "request_digest": sealed["request_digest"],
            "packet_receipt_digest": packet_receipt.get("receipt_digest") if packet_receipt else None,
            "scene_plan_digest": preflight.get("scene_plan_digest") if preflight else None,
            "candidate_id": sealed["candidate_id"],
            "preflight_receipt_digest": canonical_digest(preflight) if preflight else None,
            "rights_review_digest": rights.get("rights_review_digest") if rights else None,
            "isaaclab_launch": launch,
            "device_binding": device_binding,
            "supervised_episode": supervised,
            "teardown": teardown,
            "blocker": {"type": type(failure).__name__, "message": str(failure)} if failure else None,
            "ranking_eligible": False,
            "physical_outcome_claimed": False,
        }
        result["result_digest"] = canonical_digest(result, digest_field="result_digest")
        (output_dir / RESULT_FILENAME).write_text(
            json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    request = json.loads(args.request.read_text(encoding="utf-8"))
    result = run_g1_development_worker(request=request, output_dir=args.output_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "completed_development_only" else 1


if __name__ == "__main__":
    sys.exit(main())
