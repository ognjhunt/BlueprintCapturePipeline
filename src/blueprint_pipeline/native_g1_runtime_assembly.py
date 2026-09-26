"""Assemble a pinned G1 policy, SONIC controller, and shared scene episode.

The caller supplies the existing built Arena task/site scene. This module owns
the policy server for one development episode and retains teardown truth. It
does not turn a simulator score into qualified ranking or physical evidence.
"""

from __future__ import annotations

import importlib
import inspect
import json
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib.parse import urlparse

from .decision_evidence_contracts import canonical_digest
from .native_g1_official_sonic_target_bridge import (
    NativeG1OfficialSonicTargetBridge,
    SonicWxyzEnvironmentView,
    _require_artifact,
    require_pinned_sonic_source,
)
from .native_g1_policy_server_supervisor import (
    _source_revision,
    start_g1_policy_server,
)
from .native_g1_sonic_cuda_runtime import require_sonic_cuda_runtime
from .native_g1_shared_scene_episode import run_g1_built_scene_policy_episode


def _require_official_module_closure(root: Path) -> None:
    """Refuse a stale import that would bypass the pinned upstream checkout."""

    for name, module in tuple(sys.modules.items()):
        if not (
            name == "action_provider" or name.startswith("action_provider.")
            or name == "pico_server" or name.startswith("pico_server.")
            or name == "common_env_objects" or name == "tools.get_reward"
        ):
            continue
        source = getattr(module, "__file__", None)
        if source and not Path(source).resolve().is_relative_to(root):
            raise ValueError("g1_sonic_upstream_import_origin_mismatch:" + name)


def build_pinned_g1_sonic_bridge(
    *,
    built: Any,
    source_path: Path,
    encoder_path: Path,
    encoder_sha256: str,
    decoder_path: Path,
    decoder_sha256: str,
    server_url: str,
) -> NativeG1OfficialSonicTargetBridge:
    """Construct the official provider without its physics-stepping API."""

    plan = getattr(built, "plan", None)
    env = getattr(getattr(built, "env", None), "unwrapped", getattr(built, "env", None))
    limits = (plan.get("robot") or {}).get("joint_position_limits_rad") if isinstance(plan, Mapping) else None
    try:
        parsed = urlparse(server_url)
        port = parsed.port
    except ValueError as exc:
        raise ValueError("g1_sonic_runtime_configuration_invalid") from exc
    if (
        not isinstance(plan, Mapping)
        or (plan.get("robot") or {}).get("robot_id") != "unitree_g1"
        or env is None or not hasattr(env, "scene") or not hasattr(env, "device")
        or not isinstance(limits, Mapping)
        or parsed.scheme != "http" or parsed.hostname != "127.0.0.1"
        or port is None or parsed.path not in {"", "/"}
        or parsed.username is not None or parsed.password is not None
        or parsed.query or parsed.fragment
    ):
        raise ValueError("g1_sonic_runtime_configuration_invalid")
    require_pinned_sonic_source(source_path)
    _require_artifact(encoder_path, encoder_sha256)
    _require_artifact(decoder_path, decoder_sha256)
    _source_revision(source_path, expected_parent="action_provider")
    root = source_path.resolve().parents[1]
    _require_official_module_closure(root)
    require_sonic_cuda_runtime()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    module = importlib.import_module("action_provider.action_provider_sonic")
    _require_official_module_closure(root)
    provider_class = getattr(module, "SonicActionProvider", None)
    if (
        provider_class is None
        or inspect.getsourcefile(provider_class) is None
        or Path(inspect.getsourcefile(provider_class)).resolve() != source_path.resolve()
    ):
        raise ValueError("g1_sonic_provider_source_import_mismatch")
    arguments = SimpleNamespace(
        input_source="vla",
        gmt_backend="sonic_joint29",
        sonic_vla_action_format="semantic_v3",
        robot_type="unitree_g1_refpose_v3_1",
        sonic_encoder_path=str(encoder_path.resolve()),
        sonic_decoder_path=str(decoder_path.resolve()),
        lerobot_server_url=server_url,
        enable_dex3_dds=False,
        enable_dex1_dds=False,
    )
    provider = provider_class(SonicWxyzEnvironmentView(env), arguments)
    return NativeG1OfficialSonicTargetBridge(
        provider=provider,
        source_path=source_path,
        encoder_sha256=encoder_sha256,
        decoder_sha256=decoder_sha256,
        joint_limits=limits,
    )


def run_g1_supervised_built_scene_episode(
    *,
    built: Any,
    candidate_id: str,
    preflight_inputs: Mapping[str, Any],
    python_executable: Path,
    port: int,
    device: str,
    max_steps: int,
    output_dir: Path,
    to_tensor: Callable[[Any], Any],
    make_action_tensor: Callable[..., Any],
) -> dict[str, Any]:
    """Run one bounded development episode and always retain child teardown."""

    plan = getattr(built, "plan", None)
    if (
        not isinstance(plan, Mapping)
        or (plan.get("robot") or {}).get("robot_id") != "unitree_g1"
        or preflight_inputs.get("candidate_id") != candidate_id
        or not isinstance(output_dir, Path)
        or output_dir.exists()
    ):
        raise ValueError("g1_supervised_episode_scene_or_candidate_invalid")
    output_dir.mkdir(parents=True, exist_ok=True)
    lease = None
    episode = None
    failure: BaseException | None = None
    teardown: dict[str, Any] | None = None
    try:
        lease = start_g1_policy_server(
            preflight_inputs=preflight_inputs,
            python_executable=python_executable,
            port=port,
            device=device,
            log_path=output_dir / "g1_policy_server.log",
        )
        if (
            lease.receipt.get("candidate_id") != candidate_id
            or lease.receipt.get("scene_plan_digest") != plan.get("plan_digest")
        ):
            raise ValueError("g1_supervised_episode_server_binding_invalid")
        bridge = build_pinned_g1_sonic_bridge(
            built=built,
            source_path=Path(preflight_inputs["sonic_provider_source"]),
            encoder_path=Path(preflight_inputs["sonic_encoder"]),
            encoder_sha256=str(preflight_inputs["sonic_encoder_sha256"]),
            decoder_path=Path(preflight_inputs["sonic_decoder"]),
            decoder_sha256=str(preflight_inputs["sonic_decoder_sha256"]),
            server_url=lease.client.base_url,
        )
        episode = run_g1_built_scene_policy_episode(
            built=built,
            policy_client=lease.client,
            sonic_bridge=bridge,
            candidate_id=candidate_id,
            max_steps=max_steps,
            output_dir=output_dir,
            preflight_inputs=preflight_inputs,
            to_tensor=to_tensor,
            make_action_tensor=make_action_tensor,
        )
    except BaseException as exc:  # noqa: BLE001 - preserve failed GPU attempts
        failure = exc
    finally:
        if lease is not None:
            try:
                teardown = lease.close()
            except BaseException as exc:  # noqa: BLE001 - teardown is terminal evidence
                teardown = {"status": "teardown_failed", "error_type": type(exc).__name__}
                if failure is None:
                    failure = exc
        result = {
            "schema_version": "native_g1_supervised_built_scene_episode.v1",
            "status": (
                "completed_development_only"
                if failure is None and episode is not None and (teardown or {}).get("status") == "child_exited"
                else "blocked"
            ),
            "scene_plan_digest": plan.get("plan_digest"),
            "candidate_id": candidate_id,
            "server_lease_receipt": lease.receipt if lease is not None else None,
            "episode_result_digest": episode.get("result_digest") if episode is not None else None,
            "server_teardown": teardown,
            "blocker": type(failure).__name__ if failure is not None else None,
            "ranking_eligible": False,
            "physical_outcome_claimed": False,
        }
        result["result_digest"] = canonical_digest(result, digest_field="result_digest")
        with (output_dir / "native_g1_supervised_built_scene_episode.v1.json").open(
            "x", encoding="utf-8"
        ) as stream:
            json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
    if failure is not None:
        raise failure
    return result
