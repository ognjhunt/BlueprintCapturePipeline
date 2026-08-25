"""Execute one frozen learned candidate after native construction and controls."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import traceback
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


RESULT_SCHEMA_VERSION = "native_task_arena_policy_result.v1"
RESULT_FILENAME = "native_task_arena_policy_result.v1.json"
DIAGNOSTIC_RESULT_SCHEMA_VERSION = "native_task_arena_policy_diagnostic_result.v1"
DIAGNOSTIC_RESULT_FILENAME = "native_task_arena_policy_diagnostic_result.v1.json"
DIAGNOSTIC_EXECUTION_AUTHORITY = (
    "development_only_unqualified_controls_canonical_diagnostic"
)
DIAGNOSTIC_CLAIM_CEILING = (
    "development_only_policy_motion_diagnostic_not_scoring_not_ranking_"
    "not_qualification"
)
GROOT_RUNTIME_IDENTITY_FILENAME = (
    "adp009d_groot_worker_identity.groot_n17_droid.json"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _persist(path: Path, result: dict[str, Any]) -> None:
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest

    # Normalise before digesting. This runs from a `finally`, and BOTH
    # canonical_digest and json.dumps refuse values json cannot encode -- a
    # stray warp array or Path would raise *inside* the handler, replace the
    # real exception and leave a paid run with no receipt at all. Hardening only
    # the write is not enough, because the digest is computed first.
    normalised = json.loads(json.dumps(result, default=str))
    normalised["result_digest"] = canonical_digest(
        normalised, digest_field="result_digest"
    )
    result["result_digest"] = normalised["result_digest"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(normalised, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _to_tensor(value: Any) -> Any:
    if hasattr(value, "detach"):
        return value
    module = type(value).__module__
    if module == "warp" or module.startswith("warp."):
        import warp as wp

        return wp.to_torch(value)
    raise TypeError(f"unsupported_sim_array:{module}.{type(value).__name__}")


def _inputs(runtime: Path, manifest: Mapping[str, Any]) -> dict[str, Path]:
    verified: dict[str, Path] = {}
    for row in manifest.get("bound_runtime_inputs") or []:
        relative = str(row.get("relative_path") or "")
        path = runtime / relative
        if (
            not relative.startswith("runtime_inputs/")
            or not path.is_file()
            or path.stat().st_size != row.get("size_bytes")
            or _sha256(path) != row.get("sha256")
        ):
            raise RuntimeError(f"native_task_policy_input_identity_mismatch:{relative}")
        verified[Path(relative).name] = path
    required = {
        "native_task_arena_construction_result.v1.json",
        "native_task_arena_control_result.v1.json",
        "native_task_arena_policy_execution_spec.v1.json",
    }
    # The pi05 bundle freezes the exact checkpoint inventory that its server
    # will materialize.  The bundle verifier already requires this fourth
    # input, so dropping it here makes every otherwise-valid pi05 provider run
    # fail before Isaac or the policy server starts.  GR00T has no equivalent
    # inventory input; keep its three-file contract exact rather than allowing
    # arbitrary extras for both candidates.
    if manifest.get("policy_candidate_id") == "pi05_droid":
        required.add("openpi_polaris_checkpoint_inventory.json")
    if set(verified) != required:
        raise RuntimeError("native_task_policy_inputs_incomplete")
    return verified


def _bound_digest(value: Any) -> bool:
    """A digest relation only holds when both sides actually carry a digest."""
    return isinstance(value, str) and value.startswith("sha256:") and len(value) > 7


def _typed_media_gap_for_blocked_result(
    *, output_root: Path, result: Mapping[str, Any]
) -> dict[str, Any] | None:
    """A failure before the first observation must retain a typed media gap.

    The doctrine refuses completed episodes without lossless media and refuses
    pre-observation failures without an explicit typed gap; a bare blocked
    receipt is indistinguishable from lost media.  The discriminator here is
    observable truth rather than a phase label: whether any episode media byte
    was retained under this run's media root.
    """

    if (
        result.get("status") == "completed"
        or "episode" in result
        or "visual_evidence" in result
    ):
        return None
    episodes_root = output_root / "episodes"
    if episodes_root.is_dir() and any(
        path.is_file() for path in episodes_root.rglob("*")
    ):
        return None
    reason = next(iter(result.get("blockers") or []), "")
    if not reason:
        reason = f"failed_at_{result.get('phase_reached') or 'unknown'}"
    return {
        "status": "unavailable_before_first_observation",
        "media_gap": {
            "type": "before_first_observation",
            "reason": str(reason),
        },
    }


def _seal_post_observation_failure_media(
    *,
    output_root: Path,
    result: dict[str, Any],
    media_progress: Mapping[str, Any],
) -> None:
    """Attach a sealed visual index when execution fails after observation."""

    exact_frames = media_progress.get("candidate_exact_policy_input_frames")
    if result.get("status") == "completed" or not isinstance(exact_frames, list):
        return
    if not exact_frames:
        return
    reason = next(iter(result.get("blockers") or []), "") or (
        f"failed_at_{result.get('phase_reached') or 'unknown'}"
    )
    try:
        from blueprint_pipeline.decision_evidence_contracts import canonical_digest
        from blueprint_pipeline.episode_visual_evidence import (
            finalize_failed_policy_visual_evidence,
        )

        index = finalize_failed_policy_visual_evidence(
            output_dir=output_root / "episodes",
            episode_id=str(media_progress["episode_id"]),
            identity={
                "candidate_id": str(media_progress["candidate_id"]),
                "prompt": str(media_progress["prompt"]),
                "episode_status": "failed_after_first_observation",
            },
            exact_policy_input_frames=exact_frames,
            multicamera_policy_input_observations=list(
                media_progress.get("multicamera_policy_input_observations") or []
            ),
            review_observations=list(
                media_progress.get("review_observations") or []
            ),
            failure_reason=str(reason),
        )
        result["visual_evidence"] = {
            **index["visual_evidence"],
            "visual_index": {
                "relative_path": index["relative_path"],
                "sha256": index["sha256"],
                "visual_index_digest": index["visual_index_digest"],
            },
        }
        result["media_artifacts"] = index["media_artifacts"]
        result["candidate_exact_policy_input_frames"] = exact_frames
        result["candidate_exact_policy_input_manifest_digest"] = canonical_digest(
            {"frames": exact_frames}
        )
    except (KeyError, OSError, TypeError, ValueError, RuntimeError) as exc:
        result["visual_evidence"] = {
            "status": "post_observation_evidence_sealing_failed",
            "media_gap": {
                "type": "after_first_observation_evidence_sealing_failed",
                "reason": f"{type(exc).__name__}:{exc}",
            },
            "lossless_policy_input_files_retained": True,
            "terminal_observation_invented": False,
        }
        result["blockers"].append(
            "native_task_policy_post_observation_media_seal_failed:"
            f"{type(exc).__name__}:{exc}"
        )


def _admission_binding_mismatches(
    *,
    manifest: Mapping[str, Any],
    spec: Mapping[str, Any],
    construction: Mapping[str, Any],
    controls: Mapping[str, Any],
    scene_plan: Mapping[str, Any],
    diagnostic: bool = False,
) -> list[str]:
    """Name every disagreeing admission relation instead of one opaque blocker.

    This gate stands between a frozen candidate and a paid provider run, so a
    refusal has to say which relation broke. Absent fields are refusals, never
    agreements: two missing digests must not compare equal.
    """

    pair = controls.get("control_pair") or {}
    mismatched: list[str] = []
    candidate = spec.get("candidate_id")
    if not candidate or candidate != manifest.get("policy_candidate_id"):
        mismatched.append("execution_spec_candidate_id_vs_manifest")
    digests = (
        (
            "construction_result_digest_vs_execution_spec",
            construction.get("result_digest"),
            spec.get("construction_result_digest"),
        ),
        (
            "control_result_digest_vs_execution_spec",
            controls.get("result_digest"),
            spec.get("control_result_digest"),
        ),
        (
            "control_pair_digest_vs_execution_spec",
            pair.get("pair_digest"),
            spec.get("control_pair_digest"),
        ),
        (
            "scene_plan_digest_vs_execution_spec",
            scene_plan.get("plan_digest"),
            spec.get("scene_plan_digest"),
        ),
    )
    mismatched.extend(
        relation
        for relation, left, right in digests
        if not _bound_digest(left) or not _bound_digest(right) or left != right
    )
    qualifications = [
        (
            "construction_gate_qualified",
            construction.get("construction_gate_qualified"),
        ),
    ]
    if diagnostic:
        qualifications.extend(
            [
                ("diagnostic_controls_unqualified", controls.get("controls_qualified") is False),
                (
                    "diagnostic_control_pair_not_admitted",
                    pair.get("cell_admitted_for_policy_execution") is False,
                ),
                (
                    "diagnostic_execution_authority",
                    spec.get("execution_authority") == DIAGNOSTIC_EXECUTION_AUTHORITY,
                ),
                (
                    "diagnostic_claim_ceiling",
                    spec.get("claim_ceiling") == DIAGNOSTIC_CLAIM_CEILING,
                ),
                ("diagnostic_canonical_reset", spec.get("initial_state") == "canonical_scene_reset"),
                ("diagnostic_scoring_forbidden", spec.get("scientific_scoring_permitted") is False),
                ("diagnostic_ranking_forbidden", spec.get("ranking_permitted") is False),
                ("diagnostic_qualification_forbidden", spec.get("qualification_permitted") is False),
            ]
        )
        zero_rows = pair.get("controls") or []
        qualifications.append(
            (
                "diagnostic_zero_action_negative_separate",
                spec.get("zero_action_negative_bound_separately") is True
                and any(
                    isinstance(row, Mapping)
                    and row.get("control_id") == "zero_action_negative"
                    and row.get("control_passed") is True
                    and row.get("observed_outcome") == "never_moved"
                    for row in zero_rows
                ),
            )
        )
    else:
        qualifications.extend(
            [
                ("controls_qualified", controls.get("controls_qualified")),
                (
                    "control_pair_cell_admitted_for_policy_execution",
                    pair.get("cell_admitted_for_policy_execution"),
                ),
            ]
        )
    mismatched.extend(
        relation for relation, value in qualifications if value is not True
    )
    task_spec = scene_plan.get("task_spec") or {}
    expected_prompt = str(task_spec.get("prompt") or "")
    if not expected_prompt or spec.get("prompt") != expected_prompt:
        mismatched.append("execution_spec_prompt_vs_task_spec")
    try:
        from blueprint_pipeline.adp009d_policy_episode import (
            maximum_policy_queries_for_task_spec,
        )

        expected_queries = maximum_policy_queries_for_task_spec(
            task_spec,
            open_loop_horizon=int(spec.get("open_loop_horizon")),
        )
    except (TypeError, ValueError):
        expected_queries = None
    if (
        expected_queries is None
        or spec.get("max_policy_queries") != expected_queries
    ):
        mismatched.append("execution_spec_query_budget_vs_task_spec")
    return mismatched


def _runtime_groot_worker_identity(
    *, output_root: Path, spec: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read the identity measured from the checkpoint that serves this run."""

    from blueprint_pipeline.decision_evidence_contracts import canonical_digest
    from blueprint_pipeline.groot_n17_droid_policy_runtime import (
        GrootN17DroidPolicySpec,
        validate_worker_identity_receipt,
    )

    path = output_root / GROOT_RUNTIME_IDENTITY_FILENAME
    if not path.is_file():
        raise RuntimeError("groot_runtime_worker_identity_receipt_missing")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("groot_runtime_worker_identity_receipt_invalid") from exc
    if not isinstance(value, Mapping):
        raise RuntimeError("groot_runtime_worker_identity_receipt_invalid")
    try:
        validated = validate_worker_identity_receipt(
            value,
            expected=GrootN17DroidPolicySpec(**spec["policy_spec"]),
        )
    except (TypeError, ValueError) as exc:
        raise RuntimeError("groot_runtime_worker_identity_receipt_invalid") from exc
    evidence = {
        "source": "runtime_provisioning_measurement",
        "relative_path": GROOT_RUNTIME_IDENTITY_FILENAME,
        "file_sha256": _sha256(path),
        "receipt_digest": canonical_digest(validated),
        "receipt": validated,
    }
    return validated, evidence


def _policy_client(
    spec: Mapping[str, Any],
    *,
    groot_worker_identity_receipt: Mapping[str, Any] | None = None,
) -> Any:
    endpoint = spec["policy_endpoint"]
    secret = os.environ.get(str(endpoint["credential_env"]))
    if spec["candidate_id"] == "pi05_droid":
        from blueprint_pipeline.openpi_droid_policy_runtime import (
            OpenPIDroidPolicySpec,
            OpenPIWebsocketDroidPolicyClient,
        )

        return OpenPIWebsocketDroidPolicyClient(
            spec=OpenPIDroidPolicySpec(**spec["policy_spec"]),
            host=str(endpoint["host"]),
            port=int(endpoint["port"]),
            api_key=secret,
        )
    from blueprint_pipeline.groot_n17_droid_policy_runtime import (
        GrootN17DroidPolicyClient,
        GrootN17DroidPolicySpec,
    )

    if groot_worker_identity_receipt is None:
        raise RuntimeError("groot_runtime_worker_identity_receipt_missing")
    return GrootN17DroidPolicyClient(
        spec=GrootN17DroidPolicySpec(**spec["policy_spec"]),
        worker_identity_receipt=groot_worker_identity_receipt,
        host=str(endpoint["host"]),
        port=int(endpoint["port"]),
        api_token=secret,
    )


class _PolicyQueryTracker:
    """Remember a completed server query even if later action handling fails."""

    def __init__(self, client: Any) -> None:
        self._client = client
        self.candidate_policy_queried = False

    def infer(self, observation: Mapping[str, Any]) -> Any:
        try:
            response = self._client.infer(observation)
        except BaseException:  # noqa: BLE001 - preserve paid-run query truth
            self.candidate_policy_queried = bool(
                getattr(self._client, "candidate_policy_queried", False)
            )
            raise
        self.candidate_policy_queried = True
        return response

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


def main(argv: Sequence[str] | None = None) -> int:
    del argv
    runtime = Path(__file__).resolve().parent
    output_root = Path(
        os.environ.get("BLUEPRINT_ADP_ARENA_OUTPUT_DIR")
        or runtime.parent / "runtime_output"
    ).resolve()
    try:
        initial_manifest = json.loads(
            (runtime / "adp_arena_provider_manifest.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError):
        initial_manifest = {}
    diagnostic = initial_manifest.get("execution_mode") == "policy_diagnostic"
    output = output_root / (
        DIAGNOSTIC_RESULT_FILENAME if diagnostic else RESULT_FILENAME
    )
    result: dict[str, Any] = {
        "schema_version": (
            DIAGNOSTIC_RESULT_SCHEMA_VERSION if diagnostic else RESULT_SCHEMA_VERSION
        ),
        "status": "blocked",
        "blockers": [],
        "phase_reached": "start",
        "candidate_policy_queried": False,
        "policy_outcome_interpretable": False,
        "scientific_outcome_admitted": False,
        "ranking_eligible": False,
        "provider_zero_required_after_return": True,
        "simulator_execution_is_not_physical_truth": True,
        "execution_mode": "policy_diagnostic" if diagnostic else "policy",
        "claim_ceiling": DIAGNOSTIC_CLAIM_CEILING if diagnostic else None,
    }
    simulation_app = None
    policy_query_tracker = None
    media_progress: dict[str, Any] = {}

    def record_media_progress(progress: Mapping[str, Any]) -> None:
        media_progress.clear()
        media_progress.update(progress)

    try:
        from blueprint_pipeline.decision_evidence_contracts import canonical_digest
        manifest = initial_manifest
        if (
            manifest.get("schema_version") != "native_task_arena_provider_bundle.v1"
            or manifest.get("execution_mode")
            != ("policy_diagnostic" if diagnostic else "policy")
            or manifest.get("policy_candidate_id") not in {
                "pi05_droid",
                "groot_n17_droid",
            }
            or manifest.get("candidate_policy_queried") is not False
            or manifest.get("input_digest")
            != canonical_digest(manifest, digest_field="input_digest")
        ):
            raise RuntimeError("native_task_policy_manifest_invalid")
        inputs = _inputs(runtime, manifest)
        spec = json.loads(
            inputs["native_task_arena_policy_execution_spec.v1.json"].read_text()
        )
        if (
            spec.get("schema_version")
            != "native_task_arena_policy_execution_spec.v1"
            or spec.get("candidate_id") not in {"pi05_droid", "groot_n17_droid"}
            or spec.get("execution_spec_digest")
            != canonical_digest(spec, digest_field="execution_spec_digest")
        ):
            raise RuntimeError("native_task_policy_execution_spec_invalid")
        construction = json.loads(
            inputs["native_task_arena_construction_result.v1.json"].read_text()
        )
        controls = json.loads(
            inputs["native_task_arena_control_result.v1.json"].read_text()
        )
        packet = runtime / "native_task_packet"
        scene_plan = json.loads(
            (packet / "native_task_arena_scene_plan.v1.json").read_text()
        )
        admission_mismatches = _admission_binding_mismatches(
            manifest=manifest,
            spec=spec,
            construction=construction,
            controls=controls,
            scene_plan=scene_plan,
            diagnostic=diagnostic,
        )
        if admission_mismatches:
            raise RuntimeError(
                "native_task_policy_admission_binding_mismatch:"
                + ",".join(sorted(admission_mismatches))
            )
        result["phase_reached"] = "inputs_verified"

        from blueprint_pipeline.native_task_isaaclab_launch import (
            NATIVE_TASK_ARENA_DEVICE,
            launch_native_task_isaaclab,
        )

        simulation_app, launch = launch_native_task_isaaclab(
            output_root / "native_task_runtime_source_provisioning.v1.json",
            device=NATIVE_TASK_ARENA_DEVICE,
        )
        result["isaaclab_launch"] = launch
        import torch

        from blueprint_pipeline.adp009d_droid_action_execution import (
            GripperConvention,
        )
        from blueprint_pipeline.adp009d_policy_episode import run_policy_episode
        from blueprint_pipeline.native_franka_pose_servo import (
            NativeFrankaDifferentialIkServo,
        )
        from blueprint_pipeline.native_task_arena_construction_worker import (
            _gripper_convention_probe,
            preflight_native_dependency_matrix,
        )
        from blueprint_pipeline.native_task_arena_device_readback import (
            read_native_task_arena_device_binding,
        )
        from blueprint_pipeline.native_task_arena_preconstruction import (
            prepare_native_task_arena_preconstruction,
        )
        from blueprint_pipeline.native_task_arena_readback import (
            NativeArticulatedTaskArenaReadback,
        )
        from blueprint_pipeline.native_task_arena_runtime import (
            build_native_task_arena_environment,
        )
        from blueprint_pipeline.native_task_episode_environment import (
            build_native_task_episode_environment,
        )

        dependencies = preflight_native_dependency_matrix(
            robot_id=str(scene_plan["robot"]["robot_id"])
        )
        if not dependencies["all_required_available"]:
            result["blockers"].extend(dependencies["blockers"])
            raise RuntimeError("native_task_policy_dependency_preflight_failed")
        preconstruction = prepare_native_task_arena_preconstruction(
            expected_device=NATIVE_TASK_ARENA_DEVICE
        )
        if not preconstruction["passed"]:
            result["blockers"].extend(preconstruction["blockers"])
            raise RuntimeError("native_task_policy_preconstruction_failed")
        built = build_native_task_arena_environment(
            scene_plan,
            device=NATIVE_TASK_ARENA_DEVICE,
            bundle_root=packet,
            preconstruction_receipt=preconstruction,
        )
        device = read_native_task_arena_device_binding(built, expected_device=NATIVE_TASK_ARENA_DEVICE)
        if not device["passed"]:
            result["blockers"].extend(device["blockers"])
            raise RuntimeError("native_task_policy_device_binding_failed")
        env = built.env
        seed = int(scene_plan["scenario"]["seed"])
        env.reset(seed=seed)
        robot = env.unwrapped.scene["robot"]
        gripper = _gripper_convention_probe(env=env, robot=robot, seed=seed, torch=torch)
        if gripper["status"] != "measured":
            result["blockers"].extend(gripper["blockers"])
            raise RuntimeError("native_task_policy_gripper_unresolved")
        env.reset(seed=seed)
        servo = NativeFrankaDifferentialIkServo(
            env=env, robot=robot, gripper_convention=gripper
        )
        # The same sealed-reset measurement the construction worker retains:
        # which frame the controlled body is actually in, read back from the
        # finger bodies rather than assumed from a convention.  Taken here,
        # before any control has moved the arm.
        result["gripper_frame_axis_readback"] = (
            servo.current_gripper_frame_axis_readback()
        )
        task_readback = (
            NativeArticulatedTaskArenaReadback(
                built,
                grasp_frame_pose_callback=servo.current_grasp_frame_pose_world,
            )
            if scene_plan["task_kind"] == "articulated_open_close"
            else None
        )
        episode_environment, environment_receipt = build_native_task_episode_environment(
            built=built,
            gripper_convention=gripper,
            servo=servo,
            task_readback=task_readback,
            to_tensor=_to_tensor,
        )
        result["episode_environment"] = environment_receipt
        groot_worker_identity_receipt = None
        if spec["candidate_id"] == "groot_n17_droid":
            (
                groot_worker_identity_receipt,
                result["policy_runtime_identity"],
            ) = _runtime_groot_worker_identity(output_root=output_root, spec=spec)
        policy_query_tracker = _PolicyQueryTracker(
            _policy_client(
                spec,
                groot_worker_identity_receipt=groot_worker_identity_receipt,
            )
        )
        result["phase_reached"] = "policy_client_verified"
        episode_id = f"{scene_plan['task_id']}--{spec['cell_id']}--{spec['candidate_id']}"
        episode = run_policy_episode(
            environment=episode_environment,
            policy=policy_query_tracker,
            candidate_id=spec["candidate_id"],
            prompt=spec["prompt"],
            task_spec=scene_plan["task_spec"],
            max_policy_queries=spec["max_policy_queries"],
            settle_window_samples=int(scene_plan["task_spec"]["settle_window_samples"]),
            open_loop_horizon=spec["open_loop_horizon"],
            gripper=GripperConvention(
                closed_command=float(gripper["closed_command"]),
                open_command=float(gripper["open_command"]),
                measured_by_probe=True,
            ),
            media_output_dir=output_root / "episodes",
            episode_id=episode_id,
            scoring_authorized=not diagnostic,
            media_progress_callback=record_media_progress,
        )
        result["episode"] = episode
        if diagnostic:
            result["policy_outcome_interpretable"] = False
            result["scientific_outcome_admitted"] = False
            result["ranking_eligible"] = False
            result["diagnostic_motion_observed"] = bool(
                episode["motion_evidence"]["arm_moved"]
            )
        else:
            result["policy_outcome_interpretable"] = bool(
                episode["motion_evidence"]["policy_outcome_interpretable"]
            )
            result["scientific_outcome_admitted"] = bool(
                result["policy_outcome_interpretable"]
                and episode["score"].get("status") == "scored"
            )
            result["ranking_eligible"] = result["scientific_outcome_admitted"]
        result["status"] = "completed"
        result["phase_reached"] = "episode_complete"
    except BaseException as exc:  # noqa: BLE001 - retain every paid failure
        result["exception"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "phase": result["phase_reached"],
            "traceback": traceback.format_exc(),
        }
        result["blockers"].append(
            f"native_task_policy_failed_at_{result['phase_reached']}:"
            f"{type(exc).__name__}:{exc}"
        )
    finally:
        if policy_query_tracker is not None:
            result["candidate_policy_queried"] = (
                policy_query_tracker.candidate_policy_queried
            )
        _seal_post_observation_failure_media(
            output_root=output_root,
            result=result,
            media_progress=media_progress,
        )
        result["blockers"] = sorted(set(result["blockers"]))
        media_gap = _typed_media_gap_for_blocked_result(
            output_root=output_root, result=result
        )
        if media_gap is not None:
            result["visual_evidence"] = media_gap
        result["completed_at_unix_ns"] = time.time_ns()
        _persist(output, result)
        if simulation_app is not None:
            try:
                simulation_app.close()
            except Exception:  # noqa: BLE001
                pass
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
