"""Prove a fresh retained-worker SSH session inherits no runtime credentials."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .gpu_render_providers import enroll_vast_ssh_host_key


def probe_fresh_ssh_secret_environment_absent(
    connection: Mapping[str, Any], *, attempt_dir: str | Path
) -> dict[str, Any]:
    try:
        enrollment = enroll_vast_ssh_host_key(
            connection, attempt_dir=attempt_dir, timeout_seconds=15
        )
    except Exception:  # noqa: BLE001 - any uncertainty forbids warm retention
        return {
            "status": "blocked",
            "fresh_ssh_runtime_secret_environment_absent": False,
            "blockers": ["fresh_ssh_runtime_secret_environment_probe_failed"],
            "raw_secret_values_recorded": False,
        }
    if enrollment.get("status") != "enrolled":
        return {
            "status": "blocked",
            "fresh_ssh_runtime_secret_environment_absent": False,
            "blockers": list(enrollment.get("blockers") or []),
            "raw_secret_values_recorded": False,
        }
    from .native_task_arena_warm_vast import _run_pinned_ssh  # noqa: PLC0415

    command = r"""
count=0
while IFS= read -r name; do
  case "$name" in
    BLUEPRINT_VAST_RUNTIME_SECRET_B64_*|OPENAI_API_KEY|OPENAI_API_KEY_FILE|OPENAI_ADMIN_API_KEY_FILE|BLUEPRINT_OPENAI_ADMIN_KEY|OPENAI_ARTIFIXER_SEMANTIC_TEACHER_API_KEY_FILE|OPENAI_ARTIFIXER_VISUAL_REVIEW_API_KEY_FILE|OPENAI_CONTENT_AGENTS_API_KEY_FILE|BLUEPRINT_OPENAI_*_COST_SCOPE_ATTESTATION_FILE|HF_TOKEN|HUGGING_FACE_HUB_TOKEN)
      count=$((count + 1)) ;;
  esac
done < <(compgen -e)
if [ "$count" -eq 0 ]; then
  printf 'RUNTIME_SECRET_ENVIRONMENT:absent\n'
  exit 0
fi
printf 'RUNTIME_SECRET_ENVIRONMENT:present\n'
exit 75
"""
    try:
        observation = _run_pinned_ssh(
            session=connection,
            known_hosts_file=str(enrollment.get("known_hosts_file") or ""),
            remote_argv=["/bin/bash", "-c", command],
            timeout_seconds=30,
        )
    except Exception:  # noqa: BLE001 - never retain on an ambiguous probe
        observation = {"status": "blocked", "stdout": ""}
    absent = bool(
        observation.get("status") == "completed"
        and str(observation.get("stdout") or "")
        == "RUNTIME_SECRET_ENVIRONMENT:absent\n"
    )
    return {
        "status": "completed" if absent else "blocked",
        "fresh_ssh_runtime_secret_environment_absent": absent,
        "strict_host_key_checking": True,
        "name_only_probe": True,
        "raw_secret_values_recorded": False,
        "blockers": [] if absent else ["fresh_ssh_runtime_secret_environment_present"],
    }


__all__ = ["probe_fresh_ssh_secret_environment_absent"]
