"""One-shot GPU campaign for the frozen OpenPI captured-site policy cohort.

The worker downloads public checkpoints, verifies every byte against the frozen
generation inventory, runs three simulator-only variants per policy, writes a
deterministic prospective ranking or abstention, and exits. It never exposes a
physical-robot endpoint and never treats simulator outcomes as physical truth.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import subprocess
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .captured_site_policy_ranking import aggregate_policy_rankings
from .common import write_json
from .franka_can_tray_feasibility import _CAN_INITIAL
from .franka_droid_closed_loop import (
    DEFAULT_LEARNED_MAX_ACTION_STEPS,
    prepare_franka_droid_runtime,
    run_franka_droid_closed_loop,
)
from .openpi_droid_policy_runtime import (
    OpenPIDroidPolicySpec,
    load_policy_spec,
    verify_local_checkpoint,
)
from .policy_ranking_thesis import canonical_sha256


SCHEMA_VERSION = "openpi_policy_ranking_gpu_job.v1"
FROZEN_VARIANTS = (
    ("center", (0.0, 0.0, 0.0)),
    ("left_2cm", (0.0, 0.02, 0.0)),
    ("right_2cm", (0.0, -0.02, 0.0)),
)
SCENE_KINDS = {"captured_3dgs", "controlled_nvidia_usd"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_revision(root: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else ""


class LocalOpenPIDroidPolicyClient:
    """In-process policy adapter carrying verified checkpoint attribution."""

    learned_policy = True

    def __init__(
        self,
        *,
        spec: OpenPIDroidPolicySpec,
        policy: Any,
        local_verification: Mapping[str, Any],
    ) -> None:
        spec.validate()
        if local_verification.get("local_checkpoint_verified") is not True:
            raise ValueError("local_checkpoint_not_verified_before_policy_load")
        self.policy_id = spec.policy_id
        self.action_space = spec.action_space
        self.action_chunk_rows = spec.action_chunk_rows
        self.open_loop_horizon = spec.open_loop_horizon
        self._policy = policy
        self._evidence = {
            "transport": "in_process_openpi_policy_inference",
            "identity_verified": True,
            "policy_identity": spec.server_metadata(),
            "local_checkpoint_verification": dict(local_verification),
        }

    def infer(self, observation: Mapping[str, Any]) -> Any:
        return self._policy.infer(dict(observation))

    def evidence_summary(self) -> dict[str, Any]:
        return dict(self._evidence)


def _default_openpi_loader(spec: OpenPIDroidPolicySpec, checkpoint: Path) -> Any:
    try:
        from openpi.policies import policy_config
        from openpi.training import config as training_config
    except ImportError as exc:  # pragma: no cover - exercised in GPU image
        raise RuntimeError("openpi_gpu_runtime_not_installed") from exc
    config = training_config.get_config(spec.config_name)
    if int(config.model.action_horizon) != spec.action_chunk_rows:
        raise ValueError("openpi_config_action_horizon_mismatch")
    return policy_config.create_trained_policy(config, checkpoint)


def _default_checkpoint_downloader(uri: str) -> Path:
    try:
        from openpi.shared import download
    except ImportError as exc:  # pragma: no cover - exercised in GPU image
        raise RuntimeError("openpi_gpu_runtime_not_installed") from exc
    return Path(download.maybe_download(uri)).expanduser().resolve()


def _gpu_runtime_evidence() -> dict[str, Any]:
    evidence: dict[str, Any] = {"jax_imported": False, "gpu_device_present": False}
    try:
        import jax

        devices = [
            {
                "platform": str(device.platform),
                "device_kind": str(device.device_kind),
                "id": int(device.id),
            }
            for device in jax.devices()
        ]
        evidence.update(
            {
                "jax_imported": True,
                "jax_version": str(jax.__version__),
                "devices": devices,
                "gpu_device_present": any(row["platform"] == "gpu" for row in devices),
            }
        )
    except Exception as exc:  # noqa: BLE001 - becomes explicit admission blocker
        evidence["error_type"] = type(exc).__name__
    return evidence


def run_openpi_policy_ranking_gpu_campaign(
    *,
    cohort_path: str | Path,
    checkpoint_inventory_path: str | Path,
    captured_site_background_path: str | Path,
    menagerie_root: str | Path,
    output_dir: str | Path,
    policy_ids: Sequence[str],
    scene_backgrounds: Sequence[Mapping[str, Any]] | None = None,
    max_action_steps: int = DEFAULT_LEARNED_MAX_ACTION_STEPS,
    checkpoint_downloader: Callable[[str], Path] = _default_checkpoint_downloader,
    policy_loader: Callable[[OpenPIDroidPolicySpec, Path], Any] = _default_openpi_loader,
) -> dict[str, Any]:
    """Execute the complete frozen cohort and persist a fail-closed campaign."""
    cohort = Path(cohort_path).expanduser().resolve()
    inventory = Path(checkpoint_inventory_path).expanduser().resolve()
    background = Path(captured_site_background_path).expanduser().resolve()
    menagerie = Path(menagerie_root).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    scene_rows = list(scene_backgrounds or [])
    if not scene_rows:
        scene_rows = [
            {
                "scene_id": "captured_site",
                "scene_kind": "captured_3dgs",
                "background_path": str(background),
            }
        ]
    normalized_scenes: list[dict[str, Any]] = []
    seen_scene_ids: set[str] = set()
    for row in scene_rows:
        scene_id = str(row.get("scene_id") or "").strip()
        scene_kind = str(row.get("scene_kind") or "").strip()
        scene_background = Path(str(row.get("background_path") or "")).expanduser().resolve()
        if not scene_id or scene_id in seen_scene_ids:
            raise ValueError("gpu_campaign_scene_id_missing_or_duplicate")
        if scene_kind not in SCENE_KINDS:
            raise ValueError("gpu_campaign_scene_kind_invalid")
        seen_scene_ids.add(scene_id)
        normalized_scenes.append(
            {
                "scene_id": scene_id,
                "scene_kind": scene_kind,
                "background_path": scene_background,
            }
        )
    gpu = _gpu_runtime_evidence()
    blockers: list[str] = []
    if not gpu.get("gpu_device_present"):
        blockers.append("jax_gpu_device_not_present")
    if len(policy_ids) != 4 or len(set(policy_ids)) != 4:
        blockers.append("frozen_policy_cohort_must_contain_four_unique_policies")
    for label, path in (
        ("cohort", cohort),
        ("checkpoint_inventory", inventory),
    ):
        if not path.is_file() or path.is_symlink():
            blockers.append(f"{label}_missing_or_unsafe")
    for scene in normalized_scenes:
        path = scene["background_path"]
        if not path.is_file() or path.is_symlink():
            blockers.append(f"scene_background_missing_or_unsafe:{scene['scene_id']}")
    if not menagerie.is_dir() or menagerie.is_symlink():
        blockers.append("menagerie_root_missing_or_unsafe")
    episodes_by_scene: dict[str, dict[str, list[dict[str, Any]]]] = {
        scene["scene_id"]: {} for scene in normalized_scenes
    }
    policy_runs: list[dict[str, Any]] = []
    if not blockers:
        for policy_id in policy_ids:
            run_summary: dict[str, Any] = {"policy_id": policy_id, "status": "blocked"}
            policy = None
            try:
                spec = load_policy_spec(cohort, policy_id=policy_id)
                checkpoint = checkpoint_downloader(spec.checkpoint_uri)
                local_verification = verify_local_checkpoint(
                    spec=spec,
                    checkpoint_dir=checkpoint,
                    checkpoint_inventory_path=inventory,
                )
                policy = policy_loader(spec, checkpoint)
                client = LocalOpenPIDroidPolicyClient(
                    spec=spec,
                    policy=policy,
                    local_verification=local_verification,
                )
                episode_records: list[dict[str, Any]] = []
                scene_runs: list[dict[str, Any]] = []
                for scene in normalized_scenes:
                    scene_id = scene["scene_id"]
                    episodes: list[dict[str, Any]] = []
                    scene_episode_records: list[dict[str, Any]] = []
                    for variant_id, offset in FROZEN_VARIANTS:
                        episode_output = output / policy_id / scene_id / variant_id
                        runtime = prepare_franka_droid_runtime(
                            menagerie_root=menagerie,
                            output_dir=episode_output,
                        )
                        initial = tuple(
                            float(base + delta)
                            for base, delta in zip(_CAN_INITIAL, offset, strict=True)
                        )
                        episode = run_franka_droid_closed_loop(
                            runtime=runtime,
                            policy_client=client,
                            output_dir=episode_output,
                            max_action_steps=max_action_steps,
                            captured_site_background_path=scene["background_path"],
                            external_background_kind=scene["scene_kind"],
                            external_background_scene_id=scene_id,
                            initial_can_position_m=initial,
                        )
                        episodes.append(episode)
                        record = {
                            "scene_id": scene_id,
                            "scene_kind": scene["scene_kind"],
                            "variant_id": variant_id,
                            "initial_can_offset_m": list(offset),
                            "episode_manifest_sha256": episode["manifest_sha256"],
                        }
                        scene_episode_records.append(record)
                        episode_records.append(record)
                    episodes_by_scene[scene_id][policy_id] = episodes
                    scene_runs.append(
                        {
                            "scene_id": scene_id,
                            "scene_kind": scene["scene_kind"],
                            "episode_records": scene_episode_records,
                        }
                    )
                run_summary.update(
                    {
                        "status": "completed",
                        "checkpoint_dir": str(checkpoint),
                        "local_checkpoint_verification": local_verification,
                        "episode_manifest_sha256s": [
                            row["episode_manifest_sha256"] for row in episode_records
                        ],
                        "episode_records": episode_records,
                        "scene_runs": scene_runs,
                    }
                )
            except Exception as exc:  # noqa: BLE001 - policy failure is experimental evidence
                reason = f"policy_campaign_failed:{policy_id}:{type(exc).__name__}:{exc}"
                blockers.append(reason)
                run_summary["blocker"] = reason
            finally:
                policy = None
                gc.collect()
                try:
                    import jax

                    jax.clear_caches()
                except Exception:  # noqa: BLE001 - cleanup best effort, execution evidence remains
                    pass
            policy_runs.append(run_summary)
    rankings = {
        scene["scene_id"]: aggregate_policy_rankings(episodes_by_scene[scene["scene_id"]])
        for scene in normalized_scenes
        if not blockers and len(episodes_by_scene[scene["scene_id"]]) == len(policy_ids)
    }
    captured_scene_id = next(
        (
            scene["scene_id"]
            for scene in normalized_scenes
            if scene["scene_kind"] == "captured_3dgs"
        ),
        None,
    )
    ranking = rankings.get(captured_scene_id) if captured_scene_id else None
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": (
            "completed"
            if not blockers and len(rankings) == len(normalized_scenes)
            else "blocked"
        ),
        "gpu_runtime": gpu,
        "inputs": {
            "cohort_path": str(cohort),
            "cohort_sha256": _sha256(cohort) if cohort.is_file() else None,
            "checkpoint_inventory_path": str(inventory),
            "checkpoint_inventory_file_sha256": _sha256(inventory) if inventory.is_file() else None,
            "captured_site_background_path": str(background),
            "captured_site_background_sha256": _sha256(background) if background.is_file() else None,
            "scenes": [
                {
                    "scene_id": scene["scene_id"],
                    "scene_kind": scene["scene_kind"],
                    "background_path": str(scene["background_path"]),
                    "background_sha256": _sha256(scene["background_path"]),
                }
                for scene in normalized_scenes
            ],
            "menagerie_root": str(menagerie),
            "menagerie_git_revision": _git_revision(menagerie),
            "policy_ids": list(policy_ids),
            "max_action_steps": int(max_action_steps),
            "variants": [
                {"variant_id": variant_id, "initial_can_offset_m": list(offset)}
                for variant_id, offset in FROZEN_VARIANTS
            ],
        },
        "policy_runs": policy_runs,
        "ranking": ranking,
        "rankings": rankings,
        "blockers": blockers,
        "claim_boundary": {
            "learned_policy_simulator_execution": bool(not blockers and policy_runs),
            "prospective_captured_site_ranking": bool(
                ranking and ranking.get("status") == "completed"
            ),
            "prospective_controlled_warehouse_ranking": any(
                scene["scene_kind"] == "controlled_nvidia_usd"
                and rankings.get(scene["scene_id"], {}).get("status") == "completed"
                for scene in normalized_scenes
            ),
            "warehouse_ranking_is_independent_physical_answer_key": False,
            "site_specific_physical_success_proven": False,
            "physical_robot_endpoint_contacted": False,
            "physical_robot_operated": False,
            "wam_executed": False,
        },
    }
    result["manifest_sha256"] = canonical_sha256(result)
    write_json(output / "openpi_policy_ranking_gpu_job.json", result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--checkpoint-inventory", required=True)
    parser.add_argument("--captured-site-background", required=True)
    parser.add_argument("--menagerie-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--policy-id", action="append", required=True)
    parser.add_argument("--max-action-steps", type=int, default=DEFAULT_LEARNED_MAX_ACTION_STEPS)
    args = parser.parse_args(argv)
    result = run_openpi_policy_ranking_gpu_campaign(
        cohort_path=args.cohort,
        checkpoint_inventory_path=args.checkpoint_inventory,
        captured_site_background_path=args.captured_site_background,
        menagerie_root=args.menagerie_root,
        output_dir=args.output_dir,
        policy_ids=args.policy_id,
        max_action_steps=args.max_action_steps,
    )
    return 0 if result["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["LocalOpenPIDroidPolicyClient", "run_openpi_policy_ranking_gpu_campaign"]
