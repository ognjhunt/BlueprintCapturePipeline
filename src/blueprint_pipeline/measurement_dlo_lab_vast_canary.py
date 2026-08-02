"""Grant-gated Vast lifecycle for the exact-source DLO-Lab CUDA canary."""

from __future__ import annotations

import json
import math
import os
import time
import urllib.error
from pathlib import Path
from typing import Any, Callable, Mapping

from .common import ensure_dir, utc_now_iso, write_json
from .decision_evidence_contracts import canonical_digest
from .gpu_render_providers import GpuRenderProvider, RenderLaunchSpec
from .measurement_adapter_execution import validate_measurement_adapter_execution_bundle
from .measurement_dlo_lab_cable_adapter import (
    EXPECTED_DISTRIBUTION_VERSION,
    EXPECTED_SOURCE_COMMIT,
)
from .measurement_dlo_lab_runtime_release import (
    PYTHON_CONDA_SPEC,
    PYTHON_VERSION,
    PYTORCH_VERSION,
    PYTORCH_WHEEL_SHA256,
    PYTORCH_WHEEL_URL,
    QUADRANTS_VERSION,
    QUADRANTS_WHEEL_SHA256,
    QUADRANTS_WHEEL_URL,
    RUNTIME_IMAGE,
)
from .measurement_dlo_lab_vast_bundle import (
    validate_measurement_dlo_lab_input_bundle_receipt,
)
from .paid_lane_guard import (
    bind_pending_teardown_instance,
    cancel_pending_teardown,
    close_pending_teardown,
    load_pending_teardowns,
    mark_pending_teardown_ambiguous,
    open_pending_teardown,
)
from .paid_provider_lane_lease import (
    acquire_paid_provider_lane_lease,
    build_paid_provider_lane_reconciliation,
    release_paid_provider_lane_lease,
)
from .paid_resource_admission import (
    PaidResourceAdmissionGrant,
    require_paid_resource_admission_grant,
)
from .safe_outbound_http import presigned_transfer_policy, request as safe_http_request
from .watchdog_owner_teardown_contract import (
    WATCHDOG_EVIDENCE_NAME,
    write_owner_teardown_cancel_request,
)


RUNTIME_RESULT_SCHEMA_VERSION = "measurement_dlo_lab_cuda_vast_runtime_result.v1"
EXECUTION_SCHEMA_VERSION = "measurement_dlo_lab_cuda_vast_execution.v1"
TEARDOWN_SCHEMA_VERSION = "measurement_dlo_lab_cuda_vast_teardown.v1"
PROVIDER_ZERO_SCHEMA_VERSION = "measurement_dlo_lab_cuda_vast_provider_zero.v1"
OPERATION = "measurement_dlo_lab_canary"
PAID_LANE = "measurement_dlo_lab_gpu_canary"
NAME_PREFIX = "blueprint-measurement-dlo-"
MAX_RESULT_BYTES = 64 * 1024**2

INPUT_GET_ENV = "BLUEPRINT_MEASUREMENT_DLO_INPUT_GET_URL"
OUTPUT_PUT_ENV = "BLUEPRINT_MEASUREMENT_DLO_OUTPUT_PUT_URL"
WATCHDOG_STARTED_INSTANCE_ID_NAME = "started_vast_instance_id.txt"


class MeasurementDloLabVastCanaryError(ValueError):
    pass


def _default_result_fetcher(url: str) -> Mapping[str, Any]:
    try:
        response = safe_http_request(
            url,
            method="GET",
            timeout_seconds=30,
            policy=presigned_transfer_policy(url, max_response_bytes=MAX_RESULT_BYTES),
            max_response_bytes=MAX_RESULT_BYTES,
        )
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            raise FileNotFoundError("measurement_dlo_lab_output_http:404") from exc
        raise
    if response.status != 200:
        raise FileNotFoundError(f"measurement_dlo_lab_output_http:{response.status}")
    value = json.loads(response.body)
    if not isinstance(value, Mapping):
        raise MeasurementDloLabVastCanaryError("measurement_dlo_lab_output_not_object")
    return dict(value)


def _watchdog_valid(
    watchdog: Mapping[str, Any], *, now_epoch: float, hard_ttl_seconds: int
) -> bool:
    try:
        pid = int(watchdog.get("pid") or 0)
        deadline = float(watchdog.get("deadline_epoch") or 0)
    except (TypeError, ValueError):
        return False
    prefix = str(watchdog.get("name_prefix") or watchdog.get("pod_name_prefix") or "")
    if (
        watchdog.get("status") != "armed"
        or watchdog.get("independent_process") is not True
        or prefix != NAME_PREFIX
        or pid <= 0
        or deadline < now_epoch + hard_ttl_seconds
    ):
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    root_value = str(watchdog.get("watchdog_out_dir") or "").strip()
    if not root_value:
        return False
    declared_root = Path(root_value).expanduser()
    root = declared_root.resolve()
    evidence_path = root / WATCHDOG_EVIDENCE_NAME
    try:
        root_stat = declared_root.lstat()
        evidence_stat = evidence_path.lstat()
        persisted = json.loads(evidence_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False
    if (
        declared_root.is_symlink()
        or not root.is_dir()
        or evidence_path.is_symlink()
        or not evidence_path.is_file()
        or root_stat.st_uid != os.getuid()
        or evidence_stat.st_uid != os.getuid()
        or evidence_stat.st_size > 64 * 1024
        or not isinstance(persisted, Mapping)
    ):
        return False
    return all(
        persisted.get(key) == watchdog.get(key)
        for key in (
            "status",
            "independent_process",
            "pid",
            "deadline_epoch",
            "provider",
            "pod_name_prefix",
            "watchdog_out_dir",
        )
    )


def _watchdog_root(watchdog: Mapping[str, Any]) -> Path | None:
    value = str(watchdog.get("watchdog_out_dir") or "").strip()
    return Path(value).expanduser().resolve() if value else None


def _record_watchdog_instance_id(*, watchdog: Mapping[str, Any], instance_id: str) -> None:
    root = _watchdog_root(watchdog)
    if root is None:
        return
    path = root / WATCHDOG_STARTED_INSTANCE_ID_NAME
    if path.exists() or path.is_symlink():
        raise MeasurementDloLabVastCanaryError(
            "measurement_dlo_lab_watchdog_instance_id_already_exists"
        )
    path.write_text(instance_id, encoding="utf-8")
    os.chmod(path, 0o600)


def _bootstrap_script() -> str:
    script = r"""set -euo pipefail
work="$(mktemp -d "${TMPDIR:-/tmp}/blueprint-measurement-dlo.XXXXXX")"
archive="$work/input.zip"
bundle="$work/bundle"
dlo="$work/DLO-Lab"
dlo_env="$work/python-env"
dlo_python="$dlo_env/bin/python"
result="$work/result.json"
mkdir -p "$bundle"
python - "$archive" "$bundle" <<'PY'
import hashlib, json, os, stat, sys, urllib.request, zipfile
from pathlib import Path
archive = Path(sys.argv[1])
bundle = Path(sys.argv[2])
url = os.environ["BLUEPRINT_MEASUREMENT_DLO_INPUT_GET_URL"]
with urllib.request.urlopen(url, timeout=300) as response:
    payload = response.read(134217729)
if len(payload) > 134217728:
    raise SystemExit("measurement_dlo_lab_input_oversized")
expected = os.environ["BLUEPRINT_MEASUREMENT_DLO_INPUT_BUNDLE_DIGEST"].removeprefix("sha256:")
if hashlib.sha256(payload).hexdigest() != expected:
    raise SystemExit("measurement_dlo_lab_input_digest_mismatch")
archive.write_bytes(payload)
total = 0
with zipfile.ZipFile(archive) as source:
    for info in source.infolist():
        parts = Path(info.filename).parts
        mode = info.external_attr >> 16
        total += info.file_size
        if (
            not parts or Path(info.filename).is_absolute() or ".." in parts
            or stat.S_ISLNK(mode) or info.file_size > 67108864 or total > 268435456
        ):
            raise SystemExit("measurement_dlo_lab_input_member_unsafe")
    source.extractall(bundle)
manifest = json.loads((bundle / "bundle_manifest.json").read_text())
for row in manifest.get("source_files", []):
    path = bundle / row["path"]
    observed = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    if observed != row["digest"]:
        raise SystemExit("measurement_dlo_lab_source_digest_mismatch")
PY
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends git ca-certificates libgl1 libglib2.0-0
rm -rf /var/lib/apt/lists/*
git clone --filter=blob:none --no-checkout \
  https://github.com/UMass-Embodied-AGI/DLO-Lab.git "$dlo"
git -C "$dlo" checkout --detach "$BLUEPRINT_MEASUREMENT_DLO_SOURCE_UPSTREAM_COMMIT"
observed_commit="$(git -C "$dlo" rev-parse HEAD^{commit})"
test "$observed_commit" = "$BLUEPRINT_MEASUREMENT_DLO_SOURCE_UPSTREAM_COMMIT"
conda create --yes --prefix "$dlo_env" --override-channels --channel conda-forge \
  "__PYTHON_CONDA_SPEC__" "pip=25.1"
"$dlo_python" -m pip install --no-cache-dir \
  "__PYTORCH_WHEEL_URL__#sha256=__PYTORCH_WHEEL_SHA256__"
"$dlo_python" -m pip install --no-cache-dir \
  "__QUADRANTS_WHEEL_URL__#sha256=__QUADRANTS_WHEEL_SHA256__"
"$dlo_python" -m pip install --no-cache-dir -e "$dlo"
"$dlo_python" - <<'PY'
import importlib.metadata, platform
expected = {
    "python": "__PYTHON_VERSION__",
    "torch": "__PYTORCH_VERSION__",
    "quadrants": "__QUADRANTS_VERSION__",
}
observed = {
    "python": platform.python_version(),
    "torch": importlib.metadata.version("torch"),
    "quadrants": importlib.metadata.version("quadrants"),
}
if observed != expected:
    raise SystemExit("measurement_dlo_lab_runtime_identity_mismatch")
PY
export PYTHONPATH="$bundle/src"
set +e
"$dlo_python" "$bundle/scripts/run_measurement_dlo_lab_bundle.py" \
  --bundle-root "$bundle" --output "$result"
worker_status=$?
set -e
if [ ! -s "$result" ]; then
  exit "$worker_status"
fi
"$dlo_python" - "$result" <<'PY'
import os, sys, urllib.request
from pathlib import Path
payload = Path(sys.argv[1]).read_bytes()
request = urllib.request.Request(
    os.environ["BLUEPRINT_MEASUREMENT_DLO_OUTPUT_PUT_URL"],
    data=payload,
    method="PUT",
    headers={"Content-Type": "application/zip"},
)
with urllib.request.urlopen(request, timeout=300) as response:
    if response.status not in {200, 201, 204}:
        raise SystemExit(f"measurement_dlo_lab_output_upload_failed:{response.status}")
PY
exit "$worker_status"
"""
    return (
        script.replace("__PYTHON_CONDA_SPEC__", PYTHON_CONDA_SPEC)
        .replace("__PYTHON_VERSION__", PYTHON_VERSION)
        .replace("__PYTORCH_VERSION__", PYTORCH_VERSION)
        .replace("__PYTORCH_WHEEL_URL__", PYTORCH_WHEEL_URL)
        .replace("__PYTORCH_WHEEL_SHA256__", PYTORCH_WHEEL_SHA256.removeprefix("sha256:"))
        .replace("__QUADRANTS_VERSION__", QUADRANTS_VERSION)
        .replace("__QUADRANTS_WHEEL_URL__", QUADRANTS_WHEEL_URL)
        .replace(
            "__QUADRANTS_WHEEL_SHA256__",
            QUADRANTS_WHEEL_SHA256.removeprefix("sha256:"),
        )
    )


def _finite_number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def validate_measurement_dlo_lab_vast_runtime_result(
    value: Mapping[str, Any],
    *,
    bound_request: Mapping[str, Any],
    bundle_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    result = json.loads(json.dumps(dict(value)))
    blockers: list[str] = []
    if result.get("runtime_result_digest") != canonical_digest(
        result, digest_field="runtime_result_digest"
    ):
        blockers.append("measurement_dlo_lab_runtime_result_digest_mismatch")
    expected = {
        "schema_version": RUNTIME_RESULT_SCHEMA_VERSION,
        "status": "passed",
        "source_commit_sha": bound_request.get("source_commit_sha"),
        "runtime_image_digest": RUNTIME_IMAGE,
        "runtime_release_digest": bound_request.get(
            "measurement_dlo_lab_runtime_release_digest"
        ),
        "input_bundle_digest": bundle_receipt.get("input_bundle_digest"),
        "bundle_manifest_digest": bundle_receipt.get("bundle_manifest_digest"),
        "dlo_lab_source_commit": EXPECTED_SOURCE_COMMIT,
        "execution_bundle_count": 2,
        "development_only": True,
        "synthetic_fixture": True,
        "held_out": False,
        "physical_measurements_included": False,
        "qualification_created": False,
        "r5_evidence": False,
        "r6_decision": False,
        "r7_admission": False,
        "production_route_eligible": False,
        "physical_success_established": False,
        "comparative_policy_ranking_verdict": "thesis_not_supported",
        "raw_secret_values_recorded": False,
        "proof_effect": "development_execution_only",
        "claim_ceiling": "dlo_lab_cuda_cable_development",
    }
    for key, expected_value in expected.items():
        if result.get(key) != expected_value:
            blockers.append(f"measurement_dlo_lab_runtime_{key}_mismatch")
    bundles = result.get("execution_bundles")
    if not isinstance(bundles, list) or len(bundles) != 2:
        blockers.append("measurement_dlo_lab_runtime_execution_bundles_invalid")
    else:
        for raw_bundle in bundles:
            try:
                bundle = validate_measurement_adapter_execution_bundle(raw_bundle)
            except ValueError:
                blockers.append("measurement_dlo_lab_runtime_execution_bundle_invalid")
                continue
            runtime = bundle["receipt"].get("runtime_observations", {})
            if (
                bundle["receipt"].get("status") != "completed"
                or runtime.get("engine_version") != EXPECTED_DISTRIBUTION_VERSION
                or runtime.get("distribution_version") != EXPECTED_DISTRIBUTION_VERSION
                or runtime.get("source_commit") != EXPECTED_SOURCE_COMMIT
                or runtime.get("python_version") != PYTHON_VERSION
                or runtime.get("torch_version") != PYTORCH_VERSION
                or runtime.get("torch_distribution_version") != PYTORCH_VERSION
                or runtime.get("quadrants_distribution_version") != QUADRANTS_VERSION
                or runtime.get("cuda_available") is not True
                or not isinstance(runtime.get("cuda_device_count"), int)
                or runtime.get("cuda_device_count", 0) < 1
                or runtime.get("cpu_fallback_used") is not False
                or runtime.get("deterministic_replay_match") is not True
                or runtime.get("q_dlo_qualification_created") is not False
                or runtime.get("r5_evidence_created") is not False
                or runtime.get("r6_decision_created") is not False
                or runtime.get("r7_admission_created") is not False
                or runtime.get("physical_success_established") is not False
            ):
                blockers.append("measurement_dlo_lab_runtime_execution_observation_invalid")
    metrics = result.get("aggregate_metrics")
    if (
        not isinstance(metrics, Mapping)
        or metrics.get("case_count") != 2
        or not _finite_number(metrics.get("maximum_tip_displacement_m"))
        or not _finite_number(metrics.get("mean_tip_displacement_m"))
        or not isinstance(metrics.get("within_envelope_case_count"), int)
        or not 0 <= metrics.get("within_envelope_case_count", -1) <= 2
    ):
        blockers.append("measurement_dlo_lab_runtime_aggregate_metrics_invalid")
    if result.get("blockers") != []:
        blockers.append("measurement_dlo_lab_runtime_reported_blockers")
    if blockers:
        raise MeasurementDloLabVastCanaryError(";".join(sorted(set(blockers))))
    return result


def run_measurement_dlo_lab_vast_canary(
    *,
    bound_request: Mapping[str, Any],
    bundle_receipt: Mapping[str, Any],
    preflight: Mapping[str, Any],
    job_dir: str | Path,
    input_bundle_get_url: str,
    output_put_url: str,
    output_get_url: str,
    provider: GpuRenderProvider,
    paid_resource_admission_grant: PaidResourceAdmissionGrant | None,
    result_fetcher: Callable[[str], Mapping[str, Any]] = _default_result_fetcher,
    sleeper: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.time,
    watchdog_validator: Callable[[Mapping[str, Any], float, int], bool] | None = None,
) -> dict[str, Any]:
    """Execute one bounded paid canary and always attempt exact teardown."""

    require_paid_resource_admission_grant(
        paid_resource_admission_grant, resource_class="gpu_render"
    )
    request = json.loads(json.dumps(dict(bound_request)))
    receipt = validate_measurement_dlo_lab_input_bundle_receipt(bundle_receipt)
    if request.get("bound_request_digest") != canonical_digest(
        request, digest_field="bound_request_digest"
    ):
        raise MeasurementDloLabVastCanaryError(
            "measurement_dlo_lab_bound_request_digest_mismatch"
        )
    if (
        request.get("bound_provider") != "vast"
        or request.get("provider_mutation_authorized") is not True
        or request.get("operation") != OPERATION
        or provider.name != "vast"
        or request.get("worker_image_digest") != RUNTIME_IMAGE
        or request.get("operation_input_bundle_digest") != receipt["input_bundle_digest"]
        or request.get("source_commit_sha") != receipt["source_commit_sha"]
        or request.get("measurement_dlo_lab_runtime_release_digest")
        != receipt["runtime_release_digest"]
    ):
        raise MeasurementDloLabVastCanaryError(
            "measurement_dlo_lab_bound_request_not_executable"
        )
    hard_ttl = int(request.get("hard_ttl_seconds") or 0)
    max_spend = float(request.get("max_spend_usd") or 0)
    retry_cap = int(request.get("retry_cap") or 0)
    if retry_cap != 0 or hard_ttl <= 0 or max_spend <= 0:
        raise MeasurementDloLabVastCanaryError(
            "measurement_dlo_lab_execution_bounds_invalid"
        )

    root = Path(job_dir)
    pending_dir = root / "pending_teardowns"
    lease_dir = root / "leases"
    ensure_dir(root)
    ensure_dir(pending_dir)
    ensure_dir(lease_dir)
    request_digest = str(request["request_digest"])
    name = f"{NAME_PREFIX}{request_digest.removeprefix('sha256:')[:12]}"
    started_at = float(clock())
    watchdog = preflight.get("watchdog")
    watchdog = watchdog if isinstance(watchdog, Mapping) else {}
    validator = watchdog_validator or (
        lambda value, now, ttl: _watchdog_valid(
            value, now_epoch=now, hard_ttl_seconds=ttl
        )
    )
    if not validator(watchdog, started_at, hard_ttl):
        raise MeasurementDloLabVastCanaryError(
            "measurement_dlo_lab_independent_watchdog_not_live"
        )
    scoped_before = provider.billable_inventory(name_prefix=NAME_PREFIX)
    global_before = provider.billable_inventory(name_prefix="")
    if not all(
        row.get("api_confirmed") is True and row.get("live_resource_count") == 0
        for row in (scoped_before, global_before)
    ):
        raise MeasurementDloLabVastCanaryError(
            "measurement_dlo_lab_provider_not_zero_before_launch"
        )
    reconciliation = build_paid_provider_lane_reconciliation(
        provider="vast",
        lane=PAID_LANE,
        provider_inventory=global_before,
        open_pending_teardowns=load_pending_teardowns(registry_dir=pending_dir),
    )
    lease = acquire_paid_provider_lane_lease(
        provider="vast",
        lane=PAID_LANE,
        job_dir=str(root),
        ttl_seconds=hard_ttl,
        lease_dir=lease_dir,
        reconciliation=reconciliation,
    )
    if lease.get("status") != "acquired":
        raise MeasurementDloLabVastCanaryError(
            "measurement_dlo_lab_paid_lane_not_acquired"
        )
    pending = open_pending_teardown(
        provider="vast",
        lane=PAID_LANE,
        run_id=request_digest,
        resource_name=name,
        job_dir=root,
        max_age_seconds=hard_ttl + 1800,
        registry_dir=pending_dir,
    )
    pending_path = Path(str(pending["path"]))
    instance_id: str | None = None
    launch_result: dict[str, Any] = {}
    validated_result: dict[str, Any] | None = None
    blockers: list[str] = []
    provider_mutations = 0
    try:
        env = {
            INPUT_GET_ENV: input_bundle_get_url,
            OUTPUT_PUT_ENV: output_put_url,
            "BLUEPRINT_MEASUREMENT_DLO_SOURCE_COMMIT": str(request["source_commit_sha"]),
            "BLUEPRINT_MEASUREMENT_DLO_SOURCE_UPSTREAM_COMMIT": EXPECTED_SOURCE_COMMIT,
            "BLUEPRINT_MEASUREMENT_DLO_RUNTIME_IMAGE": RUNTIME_IMAGE,
            "BLUEPRINT_MEASUREMENT_DLO_RUNTIME_RELEASE_DIGEST": str(
                request["measurement_dlo_lab_runtime_release_digest"]
            ),
            "BLUEPRINT_MEASUREMENT_DLO_INPUT_BUNDLE_DIGEST": str(
                receipt["input_bundle_digest"]
            ),
        }
        spec = RenderLaunchSpec(
            name=name,
            image=RUNTIME_IMAGE,
            env=env,
            bootstrap_argv=["-lc", _bootstrap_script()],
            entrypoint=["bash"],
            container_disk_gb=max(
                100, int(preflight.get("container_disk_bytes") or 0) // 1024**3
            ),
            volume_gb=0,
            max_hourly_rate_usd=float(
                preflight.get("on_demand_price_usd_per_hour") or 0
            ),
            min_gpu_ram_mb=max(
                24_000, int(preflight.get("gpu_memory_bytes") or 0) // 1_000_000
            ),
            requires_rtx=False,
            vast_launch_mode="args",
        )
        provider_request = provider.build_request(spec, root)
        vast_preferences = request.get("vast_preferred_gpu_keywords")
        if isinstance(vast_preferences, list) and vast_preferences:
            provider_request["preferred_gpu_keywords"] = [
                str(item).strip() for item in vast_preferences
            ]
        provider_request["prelaunch_spend_guard"] = {
            "schema_version": "measurement_dlo_lab_gpu_prelaunch_spend_guard.v1",
            "required_before_provider_launch": True,
            "can_launch": True,
            "blockers": [],
            "max_spend_usd": max_spend,
            "hard_ttl_seconds": hard_ttl,
            "retry_cap": retry_cap,
            "request_digest": request_digest,
        }
        launch_result = dict(
            provider.launch(
                root,
                provider_request,
                cold=True,
                allow_cold_fallback=False,
                paid_resource_admission_grant=paid_resource_admission_grant,
            )
        )
        if launch_result.get("allocation_outcome_ambiguous") is True:
            provider_mutations += 1
            mark_pending_teardown_ambiguous(
                pending_path,
                reason="measurement_dlo_lab_vast_create_outcome_ambiguous",
                evidence={"blockers": list(launch_result.get("blockers") or [])},
            )
            blockers.append("measurement_dlo_lab_vast_create_outcome_ambiguous")
        elif launch_result.get("status") != "launched" or not launch_result.get(
            "instance_id"
        ):
            cancel_pending_teardown(
                pending_path,
                reason="provider_confirmed_no_allocation",
                evidence={"status": launch_result.get("status")},
            )
            blockers.append("measurement_dlo_lab_vast_instance_not_created")
        else:
            instance_id = str(launch_result["instance_id"])
            provider_mutations += 1
            bind_pending_teardown_instance(pending_path, instance_id)
            _record_watchdog_instance_id(watchdog=watchdog, instance_id=instance_id)
            raw_result: dict[str, Any] | None = None
            while float(clock()) - started_at <= hard_ttl:
                try:
                    raw_result = dict(result_fetcher(output_get_url))
                    break
                except (FileNotFoundError, TimeoutError):
                    if float(clock()) - started_at >= hard_ttl:
                        break
                    sleeper(
                        min(
                            10.0,
                            max(0.0, hard_ttl - (float(clock()) - started_at)),
                        )
                    )
                except Exception as exc:  # noqa: BLE001 - preserve teardown evidence
                    blockers.append(
                        f"measurement_dlo_lab_output_fetch_failed:{type(exc).__name__}"
                    )
                    break
            if raw_result is None:
                if not any(
                    blocker.startswith("measurement_dlo_lab_output_fetch_failed:")
                    for blocker in blockers
                ):
                    blockers.append("measurement_dlo_lab_output_timeout")
            else:
                write_json(root / "provider_runtime_result.json", raw_result)
                try:
                    validated_result = validate_measurement_dlo_lab_vast_runtime_result(
                        raw_result,
                        bound_request=request,
                        bundle_receipt=receipt,
                    )
                except MeasurementDloLabVastCanaryError as exc:
                    blockers.extend(str(exc).split(";"))
    finally:
        terminate_result: dict[str, Any] = {
            "status": "not_required" if instance_id is None else "not_attempted"
        }
        if instance_id is not None:
            terminate_result = dict(provider.terminate(instance_id))
            provider_mutations += 1
        scoped_after = provider.billable_inventory(name_prefix=NAME_PREFIX)
        global_after = provider.billable_inventory(name_prefix="")
        provider_zero = all(
            row.get("api_confirmed") is True and row.get("live_resource_count") == 0
            for row in (scoped_after, global_after)
        )
        teardown_passed = bool(
            provider_zero
            and (
                instance_id is None
                or terminate_result.get("status") in {"stopped", "terminated", "deleted"}
            )
        )
        teardown = {
            "schema_version": TEARDOWN_SCHEMA_VERSION,
            "status": "PASS" if teardown_passed else "FAIL",
            "provider": "vast",
            "request_digest": request_digest,
            "bound_request_digest": request.get("bound_request_digest"),
            "worker_image_digest": RUNTIME_IMAGE,
            "instance_id": instance_id,
            "terminate_result": terminate_result,
            "provider_zero_verified": provider_zero,
            "timestamp": utc_now_iso(),
        }
        teardown["teardown_receipt_digest"] = canonical_digest(
            teardown, digest_field="teardown_receipt_digest"
        )
        write_json(root / "teardown_receipt.json", teardown)
        if instance_id is not None and teardown_passed:
            close_pending_teardown(pending_path, teardown)
        elif (
            instance_id is None
            and launch_result.get("allocation_outcome_ambiguous") is True
            and teardown_passed
        ):
            cancel_pending_teardown(
                pending_path,
                reason="provider_zero_resolved_ambiguous_create",
                evidence={"teardown_digest": teardown["teardown_receipt_digest"]},
            )
        reconciliation_after = build_paid_provider_lane_reconciliation(
            provider="vast",
            lane=PAID_LANE,
            provider_inventory=global_after,
            open_pending_teardowns=load_pending_teardowns(registry_dir=pending_dir),
        )
        release = release_paid_provider_lane_lease(
            lease,
            reason="measurement_dlo_lab_canary_terminal",
            provider_mutation_started=instance_id is not None,
            terminal_reconciliation=reconciliation_after,
            lease_dir=lease_dir,
        )
        if not teardown_passed:
            blockers.append("measurement_dlo_lab_teardown_verification_failed")
        if instance_id is not None and release.get("released") is not True:
            blockers.append("measurement_dlo_lab_paid_lane_release_blocked")
        provider_zero_receipt = {
            "schema_version": PROVIDER_ZERO_SCHEMA_VERSION,
            "status": "PASS" if provider_zero else "FAIL",
            "provider": "vast",
            "request_digest": request_digest,
            "bound_request_digest": request.get("bound_request_digest"),
            "scoped_live_resource_count": scoped_after.get("live_resource_count"),
            "global_live_resource_count": global_after.get("live_resource_count"),
            "api_confirmed": bool(
                scoped_after.get("api_confirmed") is True
                and global_after.get("api_confirmed") is True
            ),
            "timestamp": utc_now_iso(),
        }
        provider_zero_receipt["provider_zero_digest"] = canonical_digest(
            provider_zero_receipt, digest_field="provider_zero_digest"
        )
        write_json(root / "provider_zero_verification.json", provider_zero_receipt)
        watchdog_root = _watchdog_root(watchdog)
        if instance_id is not None and teardown_passed and watchdog_root is not None:
            write_owner_teardown_cancel_request(
                root=watchdog_root,
                pod_name_prefix=NAME_PREFIX,
                provider_name="vast",
                instance_id=instance_id,
            )

    duration = max(0.0, float(clock()) - started_at)
    hourly = float(preflight.get("on_demand_price_usd_per_hour") or 0)
    cost = hourly * duration / 3600.0 if instance_id else 0.0
    if cost > max_spend:
        blockers.append("measurement_dlo_lab_budget_exhausted")
    result = {
        "schema_version": EXECUTION_SCHEMA_VERSION,
        "status": "completed" if validated_result is not None and not blockers else "failed",
        "request_digest": request_digest,
        "bound_request_digest": request.get("bound_request_digest"),
        "source_commit_sha": request.get("source_commit_sha"),
        "worker_image_digest": RUNTIME_IMAGE,
        "input_bundle_digest": receipt["input_bundle_digest"],
        "provider": "vast",
        "instance_id": instance_id,
        "runtime_result_digest": (
            validated_result.get("runtime_result_digest") if validated_result else None
        ),
        "duration_seconds": duration,
        "cost_usd": cost,
        "provider_mutations_performed": provider_mutations,
        "provider_mutation_outcome_ambiguous": bool(
            launch_result.get("allocation_outcome_ambiguous") is True
        ),
        "blockers": sorted(set(blockers)),
        "teardown_receipt_digest": teardown["teardown_receipt_digest"],
        "provider_zero_digest": provider_zero_receipt["provider_zero_digest"],
        "provider_zero_verified": provider_zero_receipt["status"] == "PASS",
        "development_execution_completed": validated_result is not None,
        "qualification_created": False,
        "r7_admission_created": False,
        "physical_success_established": False,
        "comparative_policy_ranking_verdict": "thesis_not_supported",
        "raw_secret_values_recorded": False,
        "proof_effect": "development_execution_only" if validated_result else "none",
        "claim_ceiling": (
            "dlo_lab_cuda_cable_development"
            if validated_result
            else "provider_execution_evidence_only"
        ),
    }
    result["execution_result_digest"] = canonical_digest(
        result, digest_field="execution_result_digest"
    )
    write_json(root / "measurement_dlo_lab_vast_execution.json", result)
    return result


__all__ = [
    "EXECUTION_SCHEMA_VERSION",
    "MeasurementDloLabVastCanaryError",
    "NAME_PREFIX",
    "OPERATION",
    "PAID_LANE",
    "RUNTIME_RESULT_SCHEMA_VERSION",
    "run_measurement_dlo_lab_vast_canary",
    "validate_measurement_dlo_lab_vast_runtime_result",
]
