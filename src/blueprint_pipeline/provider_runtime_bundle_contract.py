"""Shared static runtime contracts for paid provider bundles."""

from __future__ import annotations


PROVIDER_RUNTIME_BUNDLE_KINDS = (
    "isaac",
    "wam",
    "unitree_unifolm",
    "unitree_groot_n17_sonic",
)


def provider_runtime_contract_blockers(
    *,
    provider_bundle_kind: str,
    entrypoint_text: str,
    runner_text: str,
) -> list[str]:
    """Return stable fail-closed runtime-contract blockers for a provider bundle.

    Bundle-specific admission paths call this helper before paid authority is
    consumed. Provider adapters call it again immediately before any mutation.
    """
    if provider_bundle_kind not in PROVIDER_RUNTIME_BUNDLE_KINDS:
        raise ValueError(f"unsupported_provider_bundle_kind:{provider_bundle_kind}")
    if provider_bundle_kind == "isaac":
        entrypoint_valid = (
            "write_missing_result" in entrypoint_text
            and "isaac_runner_process_exited_without_runtime_result" in entrypoint_text
            and "blocked_isaac_process_exited_without_result" in entrypoint_text
        )
        runner_valid = "SimulationApp" in runner_text
        runner_blocker = "provider_runner_missing_isaac_simulation_app_smoke"
    elif provider_bundle_kind == "unitree_unifolm":
        entrypoint_valid = (
            "unitree_unifolm_provider_runner_failed_without_runtime_result" in entrypoint_text
            and "blocked_unitree_unifolm_process_exited_without_result" in entrypoint_text
        )
        runner_valid = (
            "unitree_unifolm_policy_provider_output.json" in runner_text
            and "unitree_unifolm_model_executed" in runner_text
            and "unitree_unifolm_policy_action_command_ran" in runner_text
        )
        runner_blocker = "provider_runner_missing_unitree_unifolm_runtime_contract"
    elif provider_bundle_kind == "unitree_groot_n17_sonic":
        entrypoint_valid = (
            "unitree_groot_n17_sonic_provider_runner_failed_without_runtime_result"
            in entrypoint_text
            and "blocked_unitree_groot_n17_sonic_process_exited_without_result" in entrypoint_text
        )
        runner_valid = (
            "unitree_groot_n17_sonic_policy_provider_output.json" in runner_text
            and "unitree_groot_n17_sonic_model_executed" in runner_text
            and "unitree_groot_n17_sonic_policy_action_command_ran" in runner_text
        )
        runner_blocker = "provider_runner_missing_unitree_groot_n17_sonic_runtime_contract"
    else:
        entrypoint_valid = (
            "write_missing_result" in entrypoint_text
            and "wam_runner_process_exited_without_runtime_result" in entrypoint_text
            and "blocked_wam_process_exited_without_result" in entrypoint_text
        )
        runner_valid = (
            "wam_runtime_result.json" in runner_text
            and ("OSCAR-2B" in runner_text or "Cosmos3-Nano" in runner_text)
            and "action_conditioned_video_rollout_generated" in runner_text
        )
        runner_blocker = "provider_runner_missing_wam_runtime_contract"
    blockers: list[str] = []
    if not entrypoint_valid:
        blockers.append("provider_entrypoint_missing_runtime_result_crash_fallback")
    if not runner_valid:
        blockers.append(runner_blocker)
    return blockers
