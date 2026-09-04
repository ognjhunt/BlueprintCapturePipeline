"""Run one sealed Isaac Lab control sweep on an already-owned Vast worker."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

from .decision_evidence_contracts import canonical_json
from .task_evaluation_control_search_funnel import (
    validate_control_search_funnel_plan,
    validate_control_search_sweep_result,
)
from .task_evaluation_curobo_candidate_generator import (
    _enroll_warm_host_key,
    _run_warm_ssh,
)
from .task_evaluation_isaaclab_control_sweep import (
    validate_isaaclab_control_sweep_schedule,
)


class RemoteIsaacLabControlSweepError(ValueError):
    """The retained worker transport or result could not be trusted."""


class RemoteIsaacLabControlSweepRunner:
    """Execute the provider worker without allocating or replacing the GPU."""

    def __init__(
        self,
        *,
        warm_session: Mapping[str, Any],
        local_transport_root: str | Path,
        remote_python_package_root: str | None = None,
        identity_file: str | Path | None = None,
    ) -> None:
        self._session = json.loads(
            json.dumps(dict(warm_session), allow_nan=False)
        )
        remote_work_dir = str(self._session.get("remote_work_dir") or "")
        if (
            self._session.get("status") != "ready"
            or self._session.get("continuing_spend") is not True
            or remote_work_dir
            not in {"/workspace", "/tmp/blueprint_vast_work"}  # nosec B108 - remote roots
        ):
            raise RemoteIsaacLabControlSweepError(
                "control_search_remote_warm_session_invalid"
            )
        expected_package_root = (
            remote_work_dir + "/adp_arena_provider_bundle/provider_runtime"
        )
        package_root = remote_python_package_root or expected_package_root
        pure = PurePosixPath(package_root)
        if (
            package_root != expected_package_root
            or not pure.is_absolute()
            or ".." in pure.parts
            or re.fullmatch(
                r"/(workspace|tmp/blueprint_vast_work)/[A-Za-z0-9_./-]+",
                package_root,
            )
            is None
        ):
            raise RemoteIsaacLabControlSweepError(
                "control_search_remote_runtime_root_invalid"
            )
        self._work_dir = remote_work_dir
        self._package_root = package_root.rstrip("/")
        self._root = Path(local_transport_root).expanduser().resolve()
        self._root.mkdir(parents=True, exist_ok=True)
        self._identity_file = identity_file
        enrollment = _enroll_warm_host_key(
            self._session,
            attempt_dir=self._root / "ssh-trust",
            timeout_seconds=15.0,
        )
        if enrollment.get("status") != "enrolled":
            raise RemoteIsaacLabControlSweepError(
                "control_search_remote_ssh_unavailable"
            )
        self._known_hosts = str(enrollment["known_hosts_file"])

    def _ssh(
        self,
        argv: list[str],
        *,
        stdin: bytes | None = None,
        timeout_seconds: float,
        maximum_timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        if maximum_timeout_seconds is not None:
            kwargs["maximum_timeout_seconds"] = maximum_timeout_seconds
        result = _run_warm_ssh(
            session=self._session,
            known_hosts_file=self._known_hosts,
            identity_file=self._identity_file,
            remote_argv=argv,
            stdin=stdin,
            timeout_seconds=timeout_seconds,
            **kwargs,
        )
        if result.get("status") != "completed":
            raise RemoteIsaacLabControlSweepError(
                "control_search_remote_process_failed:"
                + ":".join(str(item) for item in result.get("blockers") or [])
            )
        return result

    def _upload(self, *, path: str, value: Mapping[str, Any]) -> None:
        payload = (canonical_json(dict(value)) + "\n").encode("utf-8")
        script = (
            "import os,sys,pathlib;"
            "p=pathlib.Path(sys.argv[1]);p.parent.mkdir(parents=True,exist_ok=True);"
            "d=sys.stdin.buffer.read();t=p.with_suffix('.tmp');"
            "t.write_bytes(d);os.chmod(t,0o600);os.replace(t,p)"
        )
        self._ssh(
            ["/isaac-sim/python.sh", "-c", script, path],
            stdin=payload,
            timeout_seconds=30.0,
        )

    def execute(
        self,
        *,
        plan: Mapping[str, Any],
        schedule: Mapping[str, Any],
        candidate_inventory: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Run one sweep on the bound worker; performs zero allocations."""

        frozen_plan = validate_control_search_funnel_plan(plan)
        frozen_schedule = validate_isaaclab_control_sweep_schedule(
            schedule, plan=frozen_plan
        )
        if candidate_inventory.get("inventory_digest") != frozen_schedule.get(
            "candidate_inventory_digest"
        ):
            raise RemoteIsaacLabControlSweepError(
                "control_search_remote_inventory_invalid"
            )
        key = str(frozen_schedule["schedule_digest"]).removeprefix("sha256:")
        remote = f"{self._work_dir}/blueprint-control-sweep/{key}"
        plan_path = f"{remote}/plan.json"
        schedule_path = f"{remote}/schedule.json"
        inventory_path = f"{remote}/inventory.json"
        result_path = f"{remote}/result.json"
        self._upload(path=plan_path, value=frozen_plan)
        self._upload(path=schedule_path, value=frozen_schedule)
        self._upload(path=inventory_path, value=candidate_inventory)
        packet_root = self._package_root + "/blueprint_pipeline/native_task_packet"
        scene_plan = packet_root + "/native_task_arena_scene_plan.v1.json"
        script = f"""set -euo pipefail
mapfile -t receipts < <(find {self._work_dir} -type f -name native_task_runtime_source_provisioning.v1.json -print)
test "${{#receipts[@]}}" -eq 1
exec env PYTHONPATH={self._package_root} /isaac-sim/python.sh -m blueprint_pipeline.native_task_arena_control_sweep_worker \
  --plan {plan_path} \
  --schedule {schedule_path} \
  --candidate-inventory {inventory_path} \
  --scene-plan {scene_plan} \
  --packet-root {packet_root} \
  --provisioning-receipt "${{receipts[0]}}" \
  --output {result_path}
"""
        self._ssh(
            ["/bin/bash", "-c", script],
            timeout_seconds=1_800.0,
            maximum_timeout_seconds=1_900.0,
        )
        downloaded = self._ssh(
            ["cat", "--", result_path], timeout_seconds=30.0
        )
        if (downloaded.get("stdout_truncation") or {}).get("truncated") is True:
            raise RemoteIsaacLabControlSweepError(
                "control_search_remote_result_too_large"
            )
        try:
            result = json.loads(str(downloaded.get("stdout") or ""))
        except json.JSONDecodeError as exc:
            raise RemoteIsaacLabControlSweepError(
                "control_search_remote_result_invalid"
            ) from exc
        if not isinstance(result, Mapping):
            raise RemoteIsaacLabControlSweepError(
                "control_search_remote_result_invalid"
            )
        return validate_control_search_sweep_result(result, plan=frozen_plan)


__all__ = [
    "RemoteIsaacLabControlSweepError",
    "RemoteIsaacLabControlSweepRunner",
]
