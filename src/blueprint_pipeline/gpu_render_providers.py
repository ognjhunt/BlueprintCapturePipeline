"""Provider-agnostic GPU launch for the splat render job.

The render job builds ONE provider-neutral :class:`RenderLaunchSpec` (image + env +
bootstrap command + GPU sizing). A provider launcher translates that spec into its own
API body and runs a cold launch, returning a uniform
``{status, instance_id, mode, attempts}``. The bundle/output transport is itself
provider-neutral — signed GET/PUT URLs are carried inside ``spec.env`` and the bootstrap
fetches/uploads through them — so the watch+collect loop in the render job stays
provider-independent; only *launch* and *stop* differ per provider.

Providers:

* ``runpod`` — REST pods. Warm-host restart first (cheapest, no image pull), else cold
  on-demand create.
* ``vast`` — search offers (RT-capable GPU, under hourly rate) -> create an instance from
  the chosen ask -> ``args`` onstart running the same bootstrap. Reuses the proven Vast
  API mechanics in :mod:`blueprint_pipeline.vast_provider_adapter` so offer selection and
  request shaping live in one place.
* ``digitalocean`` — explicitly configured GPU Droplet.
* ``gcp`` — explicitly configured Compute Engine GPU VM.
* ``aws`` — explicitly configured EC2 GPU VM.

This deliberately does NOT depend on the heavy ``robot_eval_gpu_provider_launch_request``
schema used by :mod:`runpod_provider_adapter` / :mod:`vast_provider_adapter`; it is a thin
launch layer scoped to the splat render bundle contract.

Secrets are file-based under ``~/.blueprint-secrets`` and never logged.
"""
from __future__ import annotations

import json
import math
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from blueprint_pipeline import safe_outbound_http

SCHEMA_VERSION = "gpu_render_providers.v1"
SECRETS = Path.home() / ".blueprint-secrets"
RUNPOD_API = "https://rest.runpod.io/v1"
RUNPOD_GRAPHQL_API = "https://api.runpod.io/graphql"
_RUNPOD_API_POLICY = safe_outbound_http.pinned_api_policy(RUNPOD_API)
_RUNPOD_GRAPHQL_POLICY = safe_outbound_http.pinned_api_policy(RUNPOD_GRAPHQL_API)


def _read_secret(name: str) -> str | None:
    p = SECRETS / name
    return p.read_text().strip() if p.is_file() else None


# ----------------------------- neutral launch spec -----------------------------

# Price-aware but capability-gated review-lane priority: A40 first when truly
# allocatable ($0.44/hr), RTX A6000 next ($0.49/hr, live-proven 2026-07-10 on
# ONE machine at driver 570.211.01 — one machine's driver is never a guarantee
# for the whole GPU model), then L40/L40S/RTX 6000 Ada. H100/H200 stay excluded
# from the RTX review lane (no RT cores) but remain valid for explicitly
# compute-only policy/model workers.
DEFAULT_RUNPOD_RENDER_GPU_TYPES: tuple = (
    "NVIDIA A40", "NVIDIA RTX A6000", "NVIDIA L40",
    "NVIDIA L40S", "NVIDIA RTX 6000 Ada Generation",
    "NVIDIA RTX PRO 6000 Blackwell Server Edition",
)


def _runpod_gpu_types_from_env() -> tuple:
    import os

    raw = (os.getenv("BLUEPRINT_RUNPOD_GPU_TYPES") or "").strip()
    if raw:
        types = tuple(t.strip() for t in raw.split(",") if t.strip())
        if types:
            return types
    return DEFAULT_RUNPOD_RENDER_GPU_TYPES


@dataclass
class RenderLaunchSpec:
    """A provider-neutral GPU render launch request.

    ``env`` already carries the signed bundle GET url (``BLUEPRINT_EVAL_MANIFEST_URI``)
    and output PUT url (``BLUEPRINT_WORKER_RUNTIME_MANIFEST_SIGNED_PUT_URL``), so every
    provider just forwards it. ``bootstrap_argv`` is the bash command
    (``["-lc", <script>]``): RunPod uses it verbatim as ``dockerStartCmd``; Vast runs the
    script body as its ``args`` onstart.
    """

    name: str
    image: str
    env: dict
    bootstrap_argv: list
    entrypoint: list = field(default_factory=lambda: ["bash"])
    container_disk_gb: int = 140
    volume_gb: int = 80
    volume_mount_path: str = "/workspace"
    # RunPod GPU selection (provider GPU type ids, in priority order). All RTX-capable (RT cores
    # required for Isaac RTX / splat rendering — A100/H100 excluded). Datacenter tier ONLY by
    # default: the GeForce 4090 pool produced ~10 dud nodes on 2026-07-02 (never-started
    # containers, driver-550 boot segfaults, wedged workers) while L40S/driver-580 nodes rendered
    # fine. BLUEPRINT_RUNPOD_GPU_TYPES (comma-separated) re-adds types for deliberate experiments.
    gpu_types: tuple = field(default_factory=lambda: _runpod_gpu_types_from_env())
    gpu_count: int = 1
    min_vcpu: int = 8
    min_ram_gb: int = 32
    # Vast offer selection (RT-capable GPU under this hourly rate / VRAM floor).
    max_hourly_rate_usd: float = 5.0
    min_gpu_ram_mb: int = 24000
    # Render capability is separate from CUDA support and VRAM size. Hopper
    # compute GPUs can be valid model-inference workers without satisfying the
    # RT-core contract for Isaac RTX review media.
    requires_rtx: bool = True

    @property
    def bootstrap_script(self) -> str:
        """The shell script body (last element of bootstrap_argv)."""
        return self.bootstrap_argv[-1] if self.bootstrap_argv else ""


# ----------------------------- provider base -----------------------------

class GpuRenderProvider:
    """Uniform launch/stop surface. ``build_request`` produces the provider-native body
    (also used for the no-spend plan); ``launch``/``stop`` perform the paid calls."""

    name = "base"

    def available(self) -> dict:
        raise NotImplementedError

    def build_request(self, spec: RenderLaunchSpec, job_dir: Path) -> dict:
        raise NotImplementedError

    def launch(
        self,
        job_dir: Path,
        request: dict,
        *,
        cold: bool = False,
        allow_cold_fallback: bool = True,
    ) -> dict:
        raise NotImplementedError

    def stop(self, instance_id: str) -> dict:
        raise NotImplementedError

    def inspect(self, instance_id: str) -> dict:
        return {
            "status": "unavailable",
            "reason": "provider_inspect_not_implemented",
            "instance_id": instance_id,
        }

    def capacity_preflight(self, request: Mapping[str, Any] | None = None) -> dict:
        """Optional free provider capacity probe before staging large bundles.

        Providers that do not expose a useful read-only capacity API can leave
        this as not implemented; callers must treat that as non-blocking.
        """
        return {"status": "not_implemented", "provider": self.name}

    def billable_inventory(self, *, name_prefix: str) -> dict:
        """Return provider-API inventory scoped to one orchestrator name prefix.

        Startup supervision must not infer zero inventory from its own process
        memory: a prior crashed process may still own a billable resource.  A
        provider without an authoritative list API fails closed.
        """
        return {
            "status": "not_implemented",
            "provider": self.name,
            "name_prefix": str(name_prefix),
            "live_resource_count": None,
            "resources": [],
            "api_confirmed": False,
            "blockers": ["provider_billable_inventory_not_implemented"],
        }

    def terminate(self, instance_id: str) -> dict:
        """Permanently delete the instance and provider-managed storage."""
        return self.stop(instance_id)


# ----------------------------- RunPod -----------------------------

def _runpod_call(method: str, path: str, body: dict | None, *, key: str, timeout: int = 90):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        RUNPOD_API + path, data=data, method=method,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    try:
        r = safe_outbound_http.open_request(
            req, policy=_RUNPOD_API_POLICY, timeout_seconds=timeout
        )
        raw = r.body.decode()
        return r.status, (json.loads(raw) if raw.strip() else {})
    except urllib.error.HTTPError as e:
        return e.code, {"error": e.read().decode()[:400]}
    except Exception as e:  # noqa: BLE001
        return 0, {"error": repr(e)[:300]}


def _runpod_graphql_call(query: str, *, key: str, timeout: int = 60):
    """Execute a read-only RunPod GraphQL query without recording the API key."""
    request = urllib.request.Request(
        RUNPOD_GRAPHQL_API,
        data=json.dumps({"query": query}).encode("utf-8"),
        method="POST",
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "User-Agent": "BlueprintCapturePipeline/1.0",
        },
    )
    try:
        response = safe_outbound_http.open_request(
            request, policy=_RUNPOD_GRAPHQL_POLICY, timeout_seconds=timeout
        )
        raw = response.body.decode("utf-8")
        return response.status, json.loads(raw) if raw.strip() else {}
    except urllib.error.HTTPError as exc:
        return exc.code, {"error": "runpod_graphql_http_error"}
    except Exception as exc:  # noqa: BLE001
        return 0, {"error": type(exc).__name__}


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return []
    return [str(item) for item in value if str(item).strip()]


def _record_started_id(path: Path, instance_id: str) -> dict[str, Any]:
    try:
        path.write_text(str(instance_id), encoding="utf-8")
        return {"status": "recorded", "path": str(path)}
    except OSError as exc:
        # The provider allocation is already known. Never hide its id merely
        # because local evidence persistence failed.
        return {
            "status": "write_failed",
            "path": str(path),
            "error_type": type(exc).__name__,
        }


def _normalize_provider_instance_id(
    value: Any, *, numeric_only: bool = False
) -> str | None:
    """Accept only provider IDs whose wire type and shape are trustworthy.

    Stringifying a bool, mapping, or other malformed success value can produce
    a fake ID whose later 404 incorrectly looks like teardown proof. RunPod
    IDs are opaque URL-safe strings; Vast and DigitalOcean IDs are positive
    integers (or their canonical decimal string representation).
    """
    if type(value) is int:
        if not numeric_only or value <= 0:
            return None
        return str(value)
    if type(value) is not str:
        return None
    candidate = value.strip()
    if candidate != value or not candidate or len(candidate) > 256:
        return None
    if numeric_only:
        if not candidate.isascii() or not candidate.isdigit():
            return None
        return candidate if int(candidate) > 0 else None
    if not candidate.isascii() or any(
        not (character.isalnum() or character in "-_.")
        for character in candidate
    ):
        return None
    return candidate


def _runpod_render_prelaunch_guard_blockers(request: Mapping[str, Any]) -> list[str]:
    return _render_prelaunch_guard_blockers(request, provider_name="runpod")


def _runpod_create_capacity_unavailable(status: int, response: Any) -> bool:
    if int(status or 0) not in {409, 429, 500, 503}:
        return False
    error = str(_mapping(response).get("error") or "").lower()
    return any(
        marker in error
        for marker in (
            "does not have the resources to deploy your pod",
            "no available machine",
            "insufficient capacity",
            "capacity unavailable",
        )
    )


def _runpod_mutation_definitively_rejected(status: int, response: Any) -> bool:
    """Return true only when the provider explicitly says no mutation occurred."""
    if status in {401, 403, 404}:
        return True
    if status not in {400, 409, 422}:
        return False
    error = str(_mapping(response).get("error") or "").lower()
    return any(
        marker in error
        for marker in (
            "not startable",
            "invalid update",
            "invalid request",
            "does not exist",
            "not found",
        )
    )


def validate_runpod_restart_storage_contract(
    *, container_disk_sentinel: str | Path, volume_sentinel: str | Path
) -> dict[str, Any]:
    """Prove warm-restart state is on persistent volume, not container disk."""
    container_path = Path(container_disk_sentinel)
    volume_path = Path(volume_sentinel)
    blockers: list[str] = []
    if container_path.exists():
        blockers.append("runpod_container_disk_sentinel_unexpectedly_survived_restart")
    if not volume_path.is_file():
        blockers.append("runpod_persistent_volume_sentinel_missing_after_restart")
    return {
        "schema_version": "runpod_restart_storage_contract.v1",
        "status": "passed" if not blockers else "blocked",
        "blockers": blockers,
        "container_disk": {
            "path": str(container_path),
            "temporary_wiped_on_restart": not container_path.exists(),
        },
        "persistent_volume": {
            "path": str(volume_path),
            "survived_restart": volume_path.is_file(),
        },
        "resumable_state_must_use_persistent_volume_or_redownload": True,
    }


def _render_prelaunch_guard_blockers(
    request: Mapping[str, Any], *, provider_name: str
) -> list[str]:
    prefix = f"{provider_name}_render_prelaunch_spend_guard"
    guard = _mapping(request.get("prelaunch_spend_guard"))
    if not guard:
        return [f"{prefix}_missing"]
    if guard.get("required_before_provider_launch") is not True:
        return [f"{prefix}_not_required"]
    if guard.get("can_launch") is not True:
        return [
            f"{prefix}_not_passed",
            *_string_list(guard.get("blockers")),
        ]
    return []


class RunPodRenderProvider(GpuRenderProvider):
    name = "runpod"

    def __init__(self, warm_candidates: Sequence[str] = ()) -> None:
        self.warm_candidates = tuple(warm_candidates)

    def _key(self) -> str | None:
        return _read_secret("runpod_api_key")

    def available(self) -> dict:
        key = self._key()
        return {"provider": self.name, "available": bool(key),
                "reason": None if key else "runpod_api_key_missing"}

    def capacity_preflight(self, request: Mapping[str, Any] | None = None) -> dict:
        """Read-only RTX stock and price probe for the requested RunPod pool."""

        key = self._key()
        if not key:
            return {
                "status": "blocked",
                "provider": self.name,
                "blockers": ["runpod_api_key_missing"],
            }
        req = _mapping(request)
        cloud_type = str(req.get("cloudType") or "SECURE").strip().upper()
        secure_cloud = cloud_type != "COMMUNITY"
        secure_literal = "true" if secure_cloud else "false"
        requested_types = tuple(
            _string_list(req.get("gpuTypeIds")) or _runpod_gpu_types_from_env()
        )
        requested_data_centers = _string_list(req.get("dataCenterIds"))
        requested_cuda_versions = _string_list(req.get("allowedCudaVersions"))
        data_center_filter = (
            f", dataCenterId: {json.dumps(','.join(requested_data_centers))}"
            if requested_data_centers
            else ""
        )
        cuda_filter = (
            f", allowedCudaVersions: {json.dumps(requested_cuda_versions)}"
            if requested_cuda_versions
            else ""
        )
        min_gpu_ram_mb = _positive_int(req.get("min_gpu_ram_mb")) or 0
        requires_rtx = req.get("requires_rtx") is not False
        query = f"""
        query BlueprintRenderCapacity {{
          gpuTypes {{
            id
            displayName
            memoryInGb
            secureCloud
            communityCloud
            lowestPrice(input: {{gpuCount: 1, secureCloud: {secure_literal}{data_center_filter}{cuda_filter}}}) {{
              stockStatus
              uninterruptablePrice
              availableGpuCounts
            }}
          }}
        }}
        """
        status, payload = _runpod_graphql_call(query, key=key, timeout=60)
        data = _mapping(_mapping(payload).get("data"))
        rows = [
            dict(row)
            for row in data.get("gpuTypes", [])
            if isinstance(row, Mapping)
        ]
        if status != 200 or not rows or _mapping(payload).get("errors"):
            return {
                "status": "blocked",
                "provider": self.name,
                "blockers": ["runpod_capacity_probe_failed"],
                "http": status,
                "requested_gpu_types": list(requested_types),
                "raw_provider_response_recorded": False,
            }
        by_id = {str(row.get("id") or ""): row for row in rows}
        considered: list[dict[str, Any]] = []
        viable: list[dict[str, Any]] = []
        for gpu_type in requested_types:
            row = _mapping(by_id.get(gpu_type))
            price = _mapping(row.get("lowestPrice"))
            pool_capable = (
                row.get("secureCloud") is True
                if secure_cloud
                else row.get("communityCloud") is True
            )
            memory_gb = _positive_int(row.get("memoryInGb")) or 0
            stock = str(price.get("stockStatus") or "None")
            counts = [
                int(value)
                for value in (price.get("availableGpuCounts") or [])
                if isinstance(value, int) and not isinstance(value, bool)
            ]
            single_gpu_count_known = bool(counts)
            # ``lowestPrice`` was queried with ``gpuCount: 1``. RunPod now
            # returns ``availableGpuCounts: null`` for otherwise valid exact
            # datacenter/CUDA offers, so that nullable catalog field cannot be
            # required as duplicate proof. A non-None stock label plus an
            # on-demand price is the provider's advisory one-GPU offer; only
            # the subsequent create response remains authoritative.
            single_gpu_offer_available = bool(
                stock.lower() != "none"
                and _positive_float(price.get("uninterruptablePrice")) is not None
                and (not counts or 1 in counts)
            )
            record = {
                "gpu_type_id": gpu_type,
                "display_name": row.get("displayName"),
                "memory_in_gb": memory_gb or None,
                "secure_cloud": row.get("secureCloud") is True,
                "community_cloud": row.get("communityCloud") is True,
                "cloud_type": cloud_type,
                "capacity_data_center_ids": requested_data_centers,
                "capacity_data_center_id": (
                    requested_data_centers[0]
                    if len(requested_data_centers) == 1
                    else None
                ),
                "capacity_allowed_cuda_versions": requested_cuda_versions,
                "requested_pool_capable": pool_capable,
                "stock_status": stock,
                "catalog_reported_stock": stock,
                "available_gpu_counts": counts,
                "single_gpu_count_known": single_gpu_count_known,
                "single_gpu_offer_requested": True,
                "single_gpu_offer_available": single_gpu_offer_available,
                "reservation_proven": False,
                "on_demand_price_usd_per_hour": price.get("uninterruptablePrice"),
                "rtx_required": requires_rtx,
            }
            blockers: list[str] = []
            if not row:
                blockers.append("gpu_type_not_listed")
            if not pool_capable:
                blockers.append(
                    "secure_cloud_unavailable"
                    if secure_cloud
                    else "community_cloud_unavailable"
                )
            if min_gpu_ram_mb and memory_gb * 1000 < min_gpu_ram_mb:
                blockers.append("below_min_gpu_ram")
            if requires_rtx and gpu_type not in DEFAULT_RUNPOD_RENDER_GPU_TYPES:
                blockers.append("rtx_render_capability_unregistered")
            if stock.lower() == "none" or (counts and 1 not in counts):
                blockers.append("single_gpu_stock_unavailable")
            if _positive_float(price.get("uninterruptablePrice")) is None:
                blockers.append("on_demand_price_missing")
            record["blockers"] = blockers
            if blockers:
                record["capacity_confidence"] = "unavailable"
            elif single_gpu_offer_available:
                record["capacity_confidence"] = "advisory"
            else:
                # The read-only probe is never a reservation. Keep any shape
                # that does not constitute the exact one-GPU offer unknown and
                # let the paid admission seam fail closed.
                record["capacity_confidence"] = "unknown"
            if not blockers:
                viable.append(record)
            considered.append(record)
        confidences = {row["capacity_confidence"] for row in viable}
        overall_confidence = (
            "advisory"
            if "advisory" in confidences
            else "unknown" if "unknown" in confidences else "unavailable"
        )
        return {
            "status": "available" if viable else "blocked",
            "provider": self.name,
            "cloud_type": cloud_type,
            "blockers": [] if viable else ["runpod_secure_rtx_capacity_unavailable"],
            "requested_gpu_types": list(requested_types),
            "requested_data_center_ids": requested_data_centers,
            "requested_allowed_cuda_versions": requested_cuda_versions,
            "min_gpu_ram_mb": min_gpu_ram_mb,
            "requires_rtx": requires_rtx,
            "viable_gpu_types": viable,
            "considered_gpu_types": considered,
            "reservation_proven": False,
            "capacity_confidence": overall_confidence,
            "authoritative_capacity_source": "provider_create_response",
            "raw_provider_response_recorded": False,
            "claim_boundary": (
                "This is a read-only Secure Cloud stock/price snapshot. It is "
                "advisory only: it does not reserve capacity, and only the "
                "provider create response is authoritative. It proves nothing "
                "about pod creation, image startup, or rendering. A create "
                "failure that allocates no pod is a capacity outcome, not a "
                "startup failure and not spend."
            ),
        }

    def build_request(self, spec: RenderLaunchSpec, job_dir: Path) -> dict:
        environment = dict(spec.env)
        environment.update(
            {
                "BLUEPRINT_RUNPOD_CONTAINER_DISK_EPHEMERAL": "1",
                "BLUEPRINT_RESUMABLE_STATE_ROOT": str(
                    Path(spec.volume_mount_path) / "blueprint-resumable"
                ),
                "BLUEPRINT_REDOWNLOAD_BUNDLE_AFTER_RESTART": "1",
            }
        )
        return {
            "name": spec.name, "imageName": spec.image,
            "gpuTypeIds": list(spec.gpu_types), "gpuTypePriority": "availability",
            "cloudType": "SECURE", "gpuCount": spec.gpu_count,
            "containerDiskInGb": spec.container_disk_gb, "volumeInGb": spec.volume_gb,
            "volumeMountPath": spec.volume_mount_path,
            "minVCPUPerGPU": spec.min_vcpu, "minRAMPerGPU": spec.min_ram_gb,
            "max_hourly_rate_usd": spec.max_hourly_rate_usd,
            "env": environment, "dockerEntrypoint": list(spec.entrypoint),
            "dockerStartCmd": list(spec.bootstrap_argv),
            "blueprintStorageContract": {
                "container_disk": "temporary_wiped_on_restart",
                "persistent_volume": "survives_restart_bills_while_stopped",
                "resumable_state_root": str(
                    Path(spec.volume_mount_path) / "blueprint-resumable"
                ),
                "terminal_delete_pod_and_volume_required": True,
            },
        }

    def launch(
        self,
        job_dir: Path,
        request: dict,
        *,
        cold: bool = False,
        allow_cold_fallback: bool = True,
    ) -> dict:
        """Warm-host restart first (no image pull); else cold on-demand create."""
        key = self._key()
        if not key:
            return {"status": "blocked", "blockers": ["runpod_api_key_missing"]}
        prelaunch_blockers = _runpod_render_prelaunch_guard_blockers(request)
        if prelaunch_blockers:
            return {
                "status": "blocked",
                "blockers": prelaunch_blockers,
                "prelaunch_spend_guard": _mapping(request.get("prelaunch_spend_guard"))
                or None,
            }
        launch_request = dict(request)
        rate_cap = _positive_float(
            _mapping(request.get("prelaunch_spend_guard")).get(
                "max_hourly_rate_usd"
            )
        ) or _positive_float(request.get("max_hourly_rate_usd"))
        # Internal admission/cleanup evidence is consumed locally and is not
        # part of RunPod's public Pod request schema.
        launch_request.pop("prelaunch_spend_guard", None)
        launch_request.pop("pending_teardown_record", None)
        launch_request.pop("blueprintStorageContract", None)
        launch_request.pop("max_hourly_rate_usd", None)
        # Local capability filters shape ``capacity_preflight`` only. They are
        # not part of RunPod's public Pod-create schema and must never leak
        # into the paid mutation request.
        launch_request.pop("min_gpu_ram_mb", None)
        launch_request.pop("requires_rtx", None)
        attempts: list[dict] = []
        if not cold and self.warm_candidates:
            upd = {k: launch_request[k] for k in (
                "imageName", "containerDiskInGb", "volumeInGb", "volumeMountPath",
                "env", "dockerEntrypoint", "dockerStartCmd") if k in launch_request}
            for pid in self.warm_candidates:
                attempt: dict = {"pod_id": pid}
                s, get_body = _runpod_call("GET", f"/pods/{pid}", None, key=key)
                attempt["get_status"] = s
                if isinstance(get_body, dict):
                    attempt["desiredStatus"] = get_body.get("desiredStatus")
                    if get_body.get("costPerHr") is not None:
                        attempt["costPerHr"] = get_body.get("costPerHr")
                    if get_body.get("error"):
                        attempt["get_error"] = get_body.get("error")
                if s != 200:
                    attempts.append(attempt)
                    continue
                if rate_cap is not None:
                    observed_rate = _positive_float(
                        _mapping(get_body).get("costPerHr")
                    )
                    if observed_rate is None or observed_rate > rate_cap:
                        attempt["rate_cap_status"] = "blocked"
                        attempt["rate_cap_usd_per_hour"] = rate_cap
                        attempt["rate_cap_blocker"] = (
                            "runpod_warm_hourly_rate_unverifiable"
                            if observed_rate is None
                            else "runpod_warm_hourly_rate_exceeds_spend_cap"
                        )
                        attempts.append(attempt)
                        continue
                us, update_body = _runpod_call("POST", f"/pods/{pid}/update", upd, key=key)
                attempt["update_status"] = us
                if isinstance(update_body, dict) and update_body.get("error"):
                    attempt["update_error"] = update_body.get("error")
                if not (200 <= us < 300) and not _runpod_mutation_definitively_rejected(
                    us, update_body
                ):
                    attempts.append(attempt)
                    return {
                        "status": "blocked",
                        "blockers": ["runpod_warm_update_outcome_ambiguous"],
                        "instance_id": pid,
                        "attempts": attempts,
                        "allocation_outcome_ambiguous": True,
                    }
                if not (200 <= us < 300):
                    attempts.append(attempt)
                    continue
                ss, start_body = _runpod_call("POST", f"/pods/{pid}/start", {}, key=key)
                attempt["start_status"] = ss
                if isinstance(start_body, dict):
                    if start_body.get("desiredStatus") is not None:
                        attempt["start_desiredStatus"] = start_body.get("desiredStatus")
                    if start_body.get("error"):
                        attempt["start_error"] = start_body.get("error")
                attempts.append(attempt)
                if not (200 <= ss < 300) and not _runpod_mutation_definitively_rejected(
                    ss, start_body
                ):
                    return {
                        "status": "blocked",
                        "blockers": ["runpod_warm_start_outcome_ambiguous"],
                        "instance_id": pid,
                        "attempts": attempts,
                        "allocation_outcome_ambiguous": True,
                    }
                if 200 <= ss < 300:
                    started_id_record = _record_started_id(
                        job_dir / "started_pod_id.txt", pid
                    )
                    return {"status": "launched", "instance_id": pid,
                            "mode": "warm_restart", "attempts": attempts,
                            "started_id_record": started_id_record}
        if not cold and self.warm_candidates and not allow_cold_fallback:
            rate_blockers = [
                str(attempt.get("rate_cap_blocker"))
                for attempt in attempts
                if attempt.get("rate_cap_blocker")
            ]
            return {
                "status": "blocked",
                "blockers": rate_blockers
                or ["warm_restart_failed_cold_fallback_disabled"],
                "attempts": attempts,
                "allocation_created": False,
            }
        if rate_cap is not None:
            rate_preflight = self.capacity_preflight(launch_request)
            rows = [
                row
                for row in rate_preflight.get("viable_gpu_types", [])
                if isinstance(row, Mapping)
                and (_positive_float(row.get("on_demand_price_usd_per_hour")) or math.inf)
                <= rate_cap
            ]
            eligible_ids = [
                str(row.get("gpu_type_id"))
                for row in rows
                if str(row.get("gpu_type_id") or "").strip()
            ]
            attempts.append(
                {
                    "pre_mutation_rate_cap_status": (
                        "passed" if eligible_ids else "blocked"
                    ),
                    "rate_cap_usd_per_hour": rate_cap,
                    "eligible_gpu_type_ids": eligible_ids,
                    "capacity_status": rate_preflight.get("status"),
                }
            )
            if not eligible_ids:
                return {
                    "status": "blocked",
                    "blockers": ["runpod_pre_mutation_rate_cap_unverified"],
                    "attempts": attempts,
                    "allocation_created": False,
                    "rate_preflight": rate_preflight,
                }
            launch_request["gpuTypeIds"] = eligible_ids
        s, r = _runpod_call("POST", "/pods", launch_request, key=key)
        created_pid = _normalize_provider_instance_id(
            r.get("id") if isinstance(r, dict) else None
        )
        attempts.append({"cold_create_status": s, "pod_id": created_pid,
                         "error": r.get("error") if isinstance(r, dict) else None})
        if created_pid:
            started_id_record = _record_started_id(
                job_dir / "started_pod_id.txt", created_pid
            )
            return {"status": "launched", "instance_id": created_pid,
                    "mode": "cold_create", "attempts": attempts,
                    "started_id_record": started_id_record}
        if s == 0:
            return {
                "status": "blocked",
                "blockers": ["runpod_create_outcome_ambiguous"],
                "attempts": attempts,
                "allocation_outcome_ambiguous": True,
                "spend_occurred": None,
            }
        capacity_outcome = _runpod_create_capacity_unavailable(s, r)
        explicit_rejection = bool(
            capacity_outcome or s in {400, 401, 403, 404, 409, 422, 429}
        )
        if not explicit_rejection:
            return {
                "status": "blocked",
                "blockers": ["runpod_create_outcome_ambiguous"],
                "attempts": attempts,
                "allocation_outcome_ambiguous": True,
                "spend_occurred": None,
            }
        blockers = ["no_pod_started"]
        if capacity_outcome:
            blockers.insert(0, "runpod_secure_cloud_create_capacity_unavailable")
        return {
            "status": "blocked",
            "blockers": blockers,
            "attempts": attempts,
            # A create failure with no pod is a capacity outcome, not a startup
            # failure: nothing was allocated and nothing billed.
            "capacity_outcome": capacity_outcome,
            "allocation_created": False,
            "spend_occurred": False,
        }

    def stop(self, instance_id: str) -> dict:
        key = self._key()
        if not key:
            return {"status": "blocked", "blockers": ["runpod_api_key_missing"]}
        s, _ = _runpod_call("POST", f"/pods/{instance_id}/stop", {}, key=key)
        if s == 404:
            return {"status": "stopped", "http": s, "already_gone": True}
        return {"status": "stopped" if s in (200, 201, 204) else "stop_failed", "http": s}

    def inspect(self, instance_id: str) -> dict:
        key = self._key()
        if not key:
            return {
                "status": "blocked",
                "blockers": ["runpod_api_key_missing"],
                "instance_id": instance_id,
            }
        s, body = _runpod_call("GET", f"/pods/{instance_id}", None, key=key, timeout=30)
        if not isinstance(body, dict):
            return {
                "status": "unavailable",
                "http": s,
                "instance_id": instance_id,
                "raw_provider_response_recorded": False,
            }
        runtime = body.get("runtime")
        public_ip = str(body.get("publicIp") or "").strip()
        return {
            "status": "observed" if s == 200 else "unavailable",
            "http": s,
            "instance_id": instance_id,
            "desiredStatus": body.get("desiredStatus"),
            "runtime_present": runtime is not None,
            "public_ip_present": bool(public_ip),
            "machineId": body.get("machineId"),
            "costPerHr": body.get("costPerHr"),
            "createdAt": body.get("createdAt"),
            "lastStartedAt": body.get("lastStartedAt"),
            "lastStatusChange": body.get("lastStatusChange"),
            "imageName": body.get("imageName"),
            "error": body.get("error"),
            "raw_provider_response_recorded": False,
        }

    def billable_inventory(self, *, name_prefix: str) -> dict:
        observed_at_epoch = time.time()
        key = self._key()
        if not key:
            return {
                "status": "blocked",
                "provider": self.name,
                "name_prefix": str(name_prefix),
                "live_resource_count": None,
                "resources": [],
                "api_confirmed": False,
                "observed_at_epoch": observed_at_epoch,
                "blockers": ["runpod_api_key_missing"],
            }
        status, body = _runpod_call("GET", "/pods", None, key=key, timeout=30)
        rows = body if isinstance(body, list) else _mapping(body).get("pods")
        if status != 200 or not isinstance(rows, list):
            return {
                "status": "blocked",
                "provider": self.name,
                "name_prefix": str(name_prefix),
                "live_resource_count": None,
                "resources": [],
                "api_confirmed": False,
                "observed_at_epoch": observed_at_epoch,
                "blockers": ["runpod_billable_inventory_failed"],
                "http": status,
                "raw_provider_response_recorded": False,
            }
        prefix = str(name_prefix or "")
        warm_candidate_ids = {
            str(candidate).strip() for candidate in self.warm_candidates
            if str(candidate).strip()
        }
        resources = []
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            name = str(row.get("name") or "")
            instance_id = str(row.get("id") or "")
            if prefix and not name.startswith(prefix) and instance_id not in warm_candidate_ids:
                continue
            resources.append(
                {
                    "instance_id": instance_id,
                    "name": name,
                    "desired_status": row.get("desiredStatus"),
                    "cost_per_hour": row.get("costPerHr"),
                }
            )
        return {
            "status": "observed",
            "provider": self.name,
            "name_prefix": prefix,
            "explicit_warm_candidate_ids_checked": sorted(warm_candidate_ids),
            "live_resource_count": len(resources),
            "resources": resources,
            "api_confirmed": True,
            "observed_at_epoch": observed_at_epoch,
            "http": status,
            "raw_provider_response_recorded": False,
        }

    def terminate(self, instance_id: str) -> dict:
        """DELETE the terminal pod and its remaining provider-managed storage.

        RunPod container disk is temporary and is wiped on restart. A persistent
        Pod volume survives restart and incurs stopped storage cost. Terminal
        deletion remains mandatory, but resumable state must live on the volume
        (or be re-downloaded), never be inferred from container-disk survival.
        """
        key = self._key()
        if not key:
            return {"status": "blocked", "blockers": ["runpod_api_key_missing"]}
        s, _ = _runpod_call("DELETE", f"/pods/{instance_id}", None, key=key)
        if s == 404:
            return {"status": "terminated", "http": s, "already_gone": True}
        return {"status": "terminated" if s in (200, 201, 204) else "terminate_failed", "http": s}


# ----------------------------- Vast.ai -----------------------------

class VastRenderProvider(GpuRenderProvider):
    name = "vast"

    def _key(self) -> str | None:
        return _read_secret("vast_api_key")

    def available(self) -> dict:
        key = self._key()
        return {"provider": self.name, "available": bool(key),
                "reason": None if key else "vast_api_key_missing"}

    def build_request(self, spec: RenderLaunchSpec, job_dir: Path) -> dict:
        # Reuse the proven Vast request builders so offer-search + create-instance shaping
        # stays consistent with the rest of the repo's Vast tooling.
        from .vast_provider_adapter import (
            VAST_API_BASE, _create_payload, _search_payload,
        )
        search_payload = _search_payload(limit=100, max_hourly_rate=spec.max_hourly_rate_usd)
        create_payload = _create_payload(
            image=spec.image, label=spec.name, launch_mode="args",
            probe_script=spec.bootstrap_script, disk_gb=spec.container_disk_gb,
            env=dict(spec.env),
        )
        return {
            "provider": "vast",
            "api_base": VAST_API_BASE,
            "search_endpoint": "POST /bundles/",
            "search_payload": search_payload,
            "create_endpoint": "PUT /asks/{ask_contract_id}/",
            "create_payload": create_payload,
            "image": spec.image, "disk": spec.container_disk_gb,
            "min_gpu_ram_mb": spec.min_gpu_ram_mb,
            "max_hourly_rate_usd": spec.max_hourly_rate_usd,
        }

    def launch(
        self,
        job_dir: Path,
        request: dict,
        *,
        cold: bool = False,
        allow_cold_fallback: bool = True,
    ) -> dict:
        """Search RT-capable offers under rate, then create an instance from the cheapest.
        Vast is always on-demand cold create; ``cold`` is accepted for a uniform signature."""
        key = self._key()
        if not key:
            return {"status": "blocked", "blockers": ["vast_api_key_missing"]}
        prelaunch_blockers = _render_prelaunch_guard_blockers(
            request, provider_name="vast"
        )
        if prelaunch_blockers:
            return {
                "status": "blocked",
                "blockers": prelaunch_blockers,
                "prelaunch_spend_guard": _mapping(request.get("prelaunch_spend_guard"))
                or None,
            }
        from .vast_provider_adapter import (
            _api_json, _offers_from_response, _select_offer,
        )
        attempts: list[dict] = []
        search_payload = request.get("search_payload") or {}
        max_rate = float(request.get("max_hourly_rate_usd") or 2.0)
        min_ram = int(request.get("min_gpu_ram_mb") or 0)
        try:
            s, resp = _api_json(method="POST", path="/bundles/", api_key=key,
                                payload=search_payload, timeout_seconds=45)
        except Exception as e:  # noqa: BLE001
            return {"status": "blocked", "blockers": ["vast_offer_search_failed"],
                    "error": repr(e)[:200]}
        offers = _offers_from_response(resp)
        attempts.append({"offer_search_status": s, "offer_count": len(offers)})
        create_payload = request.get("create_payload") or {}
        # Offers go stale between search and create (bundle staging can take minutes),
        # and a stale ask 400s. Walk up to 3 candidate offers before giving up so one
        # expired ask can't dud the whole provider in a race.
        remaining = list(offers)
        last_blocker = "no_vast_offer_matching_rate_and_gpu_memory"
        for _try in range(3):
            offer = _select_offer(remaining, max_hourly_rate=max_rate, min_gpu_ram_mb=min_ram)
            if not offer:
                break
            remaining = [o for o in remaining if o is not offer]
            ask_id = offer.get("ask_contract_id")
            try:
                cs, cresp = _api_json(method="PUT", path=f"/asks/{ask_id}/", api_key=key,
                                      payload=create_payload, timeout_seconds=45)
            except urllib.error.HTTPError as e:
                body = ""
                try:
                    body = (e.read() or b"")[:300].decode("utf-8", "replace")
                except Exception:  # noqa: BLE001
                    pass
                attempts.append({"create_http_status": e.code, "ask_id": ask_id,
                                 "create_error_body": body,
                                 "gpu_name": offer.get("gpu_name")})
                if e.code not in {400, 404, 409, 422}:
                    return {
                        "status": "blocked",
                        "blockers": ["vast_create_outcome_ambiguous"],
                        "attempts": attempts,
                        "allocation_outcome_ambiguous": True,
                    }
                last_blocker = f"vast_create_http_error:{e.code}"
                continue
            except Exception as e:  # noqa: BLE001
                attempts.append({"create_error": repr(e)[:200], "ask_id": ask_id})
                return {
                    "status": "blocked",
                    "blockers": ["vast_create_outcome_ambiguous"],
                    "attempts": attempts,
                    "allocation_outcome_ambiguous": True,
                }
            iid = None
            if isinstance(cresp, dict):
                for k in ("new_contract", "instance_id", "id"):
                    iid = _normalize_provider_instance_id(
                        cresp.get(k), numeric_only=True
                    )
                    if iid:
                        break
            attempts.append({"create_status": cs, "instance_id": iid,
                             "gpu_name": offer.get("gpu_name"),
                             "hourly_rate_usd": offer.get("hourly_rate_usd")})
            if iid:
                started_id_record = _record_started_id(
                    job_dir / "started_vast_instance_id.txt", iid
                )
                return {"status": "launched", "instance_id": iid, "mode": "vast_on_demand",
                        "attempts": attempts, "started_id_record": started_id_record}
            if cs not in {400, 404, 409, 422}:
                return {
                    "status": "blocked",
                    "blockers": ["vast_create_outcome_ambiguous"],
                    "attempts": attempts,
                    "allocation_outcome_ambiguous": True,
                }
            last_blocker = "vast_instance_not_created"
        return {
            "status": "blocked",
            "blockers": [last_blocker],
            "attempts": attempts,
            "allocation_created": False,
        }

    def stop(self, instance_id: str) -> dict:
        # Vast has no warm-preserving stopped state here: DELETE /instances destroys
        # the instance. Do not route warm-pool preservation through this provider.
        key = self._key()
        if not key:
            return {"status": "blocked", "blockers": ["vast_api_key_missing"]}
        from .vast_provider_adapter import _api_json
        try:
            s, _ = _api_json(method="DELETE", path=f"/instances/{instance_id}/",
                             api_key=key, timeout_seconds=30)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return {"status": "stopped", "http": e.code, "already_gone": True}
            return {"status": "stop_failed", "http": e.code}
        except Exception as e:  # noqa: BLE001
            return {"status": "stop_failed", "error": repr(e)[:200]}
        if s == 404:
            return {"status": "stopped", "http": s, "already_gone": True}
        return {"status": "stopped" if s in (200, 201, 204) else "stop_failed", "http": s}


# ----------------------------- registry -----------------------------

# ----------------------------- DigitalOcean GPU Droplets -----------------------------

DO_API = "https://api.digitalocean.com/v2"
DEFAULT_DO_GPU_SIZE = "gpu-6000adax1-48gb"   # RTX 6000 Ada: 3rd-gen RT cores + 48GB, $1.57/hr
DEFAULT_DO_GPU_REGION = "atl1"               # GPU droplet regions: nyc2, tor1, atl1, ric1, ams3
DO_GPU_BASE_IMAGE = "gpu-h100x1-base"        # "NVIDIA AI/ML Ready": Ubuntu + drivers + docker
DEFAULT_DO_GPU_SIZE_CANDIDATES = (
    DEFAULT_DO_GPU_SIZE,
    "gpu-l40sx1-48gb",
    "gpu-h100x1-80gb",
    "gpu-h200x1-141gb",
    "gpu-4000adax1-20gb",
)
DEFAULT_DO_GPU_REGION_CANDIDATES = (DEFAULT_DO_GPU_REGION, "nyc2", "tor1", "ric1", "ams3")
DO_GPU_HOURLY_RATE_USD = {
    "gpu-4000adax1-20gb": 0.76,
    "gpu-l40sx1-48gb": 1.57,
    "gpu-6000adax1-48gb": 1.57,
    "gpu-mi300x1-192gb": 1.99,
    "gpu-h100x1-80gb": 3.39,
    "gpu-h200x1-141gb": 3.44,
}
DO_GPU_SIZE_VRAM_MB = {
    "gpu-4000adax1-20gb": 20000,
    "gpu-l40sx1-48gb": 48000,
    "gpu-6000adax1-48gb": 48000,
    "gpu-mi300x1-192gb": 192000,
    "gpu-h100x1-80gb": 80000,
    "gpu-h200x1-141gb": 141000,
}
DO_GPU_SIZE_RTX_CAPABLE = {
    "gpu-4000adax1-20gb": True,
    "gpu-l40sx1-48gb": True,
    "gpu-6000adax1-48gb": True,
    "gpu-mi300x1-192gb": False,
    "gpu-h100x1-80gb": False,
    "gpu-h200x1-141gb": False,
}
DEFAULT_DO_MAX_HOURLY_RATE_USD = 1.75
DO_GPU_SIZE_CANDIDATES_ENV = "BLUEPRINT_DO_GPU_SIZES"
DO_GPU_REGION_CANDIDATES_ENV = "BLUEPRINT_DO_GPU_REGIONS"
DO_GPU_MAX_HOURLY_RATE_ENV = "BLUEPRINT_DO_MAX_HOURLY_RATE_USD"
DO_SSH_KEY_IDS_ENV = "BLUEPRINT_DO_SSH_KEY_IDS"
DO_SSH_KEY_IDS_FILE_ENV = "BLUEPRINT_DO_SSH_KEY_IDS_FILE"
DEFAULT_DO_SSH_KEY_IDS_FILE = "~/.blueprint-secrets/digitalocean_ssh_key_ids"


_DO_API_POLICY = safe_outbound_http.pinned_api_policy(DO_API)


def _do_call(method: str, path: str, body: dict | None = None, *, token: str, timeout: int = 90):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        DO_API + path, data=data, method=method,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
    )
    try:
        r = safe_outbound_http.open_request(
            req, policy=_DO_API_POLICY, timeout_seconds=timeout
        )
        raw = r.body.decode()
        return r.status, (json.loads(raw) if raw.strip() else {})
    except urllib.error.HTTPError as e:
        raw = ""
        try:
            raw = (e.read() or b"")[:400].decode("utf-8", "replace")
        except Exception:  # noqa: BLE001
            pass
        return e.code, {"error": raw}


def _do_user_data(spec: RenderLaunchSpec) -> str:
    """Cloud-init script: run the worker container on the NVIDIA-ready droplet.

    Env values (presigned URLs) and the bootstrap script ride base64 so no shell
    quoting can corrupt them; the container runs exactly like RunPod's
    dockerEntrypoint/dockerStartCmd pair.
    """
    import base64
    import shlex

    env_file = "\n".join(f"{k}={v}" for k, v in spec.env.items())
    env_b64 = base64.b64encode(env_file.encode()).decode()
    argv_b64 = base64.b64encode(json.dumps(list(spec.bootstrap_argv)).encode()).decode()
    entrypoint = spec.entrypoint[0] if spec.entrypoint else "bash"
    image_json = json.dumps(spec.image)
    entrypoint_json = json.dumps(entrypoint)
    return f"""#!/bin/bash
set -euo pipefail
echo {env_b64} | base64 -d > /root/blueprint_worker.env
echo {argv_b64} | base64 -d > /root/blueprint_argv.json
mkdir -p /root/blueprint-workspace/out
chmod 0777 /root/blueprint-workspace /root/blueprint-workspace/out
python3 - <<'PY'
import json
argv = json.load(open("/root/blueprint_argv.json"))
open("/root/blueprint_argv_decoded.json", "w").write(json.dumps(argv, indent=2))
PY
docker pull {shlex.quote(spec.image)}
python3 - <<'PY'
import json, subprocess
argv = json.load(open("/root/blueprint_argv.json"))
cmd = [
    "docker", "run", "-d",
    "--gpus", "all",
    "--name", "blueprint-worker",
    "--user", "0:0",
    "--env-file", "/root/blueprint_worker.env",
    "-v", "/root/blueprint-workspace:/workspace",
    "--workdir", "/workspace",
    "--shm-size=8g",
    "--entrypoint", {entrypoint_json},
    {image_json},
    *argv,
]
open("/root/blueprint_docker_cmd.json", "w").write(json.dumps(cmd, indent=2))
subprocess.check_call(cmd)
PY
"""


def _parse_do_ssh_key_ids(raw: str) -> list[int | str]:
    keys: list[int | str] = []
    for part in raw.replace("\n", ",").split(","):
        value = part.strip()
        if not value:
            continue
        keys.append(int(value) if value.isdigit() else value)
    return keys


def _parse_csv_values(raw: str) -> list[str]:
    return [part.strip() for part in raw.replace("\n", ",").split(",") if part.strip()]


def _ordered_unique(values: Sequence[str]) -> list[str]:
    ordered: list[str] = []
    for value in values:
        if value and value not in ordered:
            ordered.append(value)
    return ordered


def _do_size_candidates(initial: str) -> list[str]:
    import os

    configured = _parse_csv_values(os.getenv(DO_GPU_SIZE_CANDIDATES_ENV) or "")
    if configured:
        return _ordered_unique(configured)
    return _ordered_unique([initial, *DEFAULT_DO_GPU_SIZE_CANDIDATES])


def _do_region_candidates(initial: str) -> list[str]:
    import os

    configured = _parse_csv_values(os.getenv(DO_GPU_REGION_CANDIDATES_ENV) or "")
    if configured:
        return _ordered_unique(configured)
    return _ordered_unique([initial, *DEFAULT_DO_GPU_REGION_CANDIDATES])


def _positive_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) and parsed > 0 else None


def _positive_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _do_max_hourly_rate_usd(request: Mapping[str, Any] | None = None) -> tuple[float, str]:
    req = _mapping(request)
    requested = _positive_float(req.get("max_hourly_rate_usd"))
    if requested is not None:
        return requested, "request:max_hourly_rate_usd"
    import os

    raw = (os.getenv(DO_GPU_MAX_HOURLY_RATE_ENV) or "").strip()
    env_value = _positive_float(raw)
    if env_value is not None:
        return env_value, DO_GPU_MAX_HOURLY_RATE_ENV
    return DEFAULT_DO_MAX_HOURLY_RATE_USD, "default"


def _filter_do_size_candidates_by_budget(
    size_candidates: Sequence[str],
    request: Mapping[str, Any] | None = None,
) -> tuple[list[str], dict]:
    max_hourly, source = _do_max_hourly_rate_usd(request)
    allowed: list[str] = []
    rejected: list[dict] = []
    for size in size_candidates:
        hourly = DO_GPU_HOURLY_RATE_USD.get(size)
        if hourly is None:
            rejected.append({
                "size": size,
                "reason": "unknown_hourly_rate",
            })
            continue
        if hourly > max_hourly:
            rejected.append({
                "size": size,
                "hourly_rate_usd": hourly,
                "reason": "over_max_hourly_rate",
            })
            continue
        allowed.append(size)
    return allowed, {
        "max_hourly_rate_usd": max_hourly,
        "max_hourly_rate_source": source,
        "max_hourly_rate_env": DO_GPU_MAX_HOURLY_RATE_ENV,
        "allowed_size_candidates": allowed,
        "rejected_size_candidates": rejected,
    }


def _requested_do_min_gpu_ram_mb(request: Mapping[str, Any] | None = None) -> int:
    req = _mapping(request)
    return _positive_int(req.get("min_gpu_ram_mb")) or 0


def _filter_do_size_candidates_by_gpu_ram(
    size_candidates: Sequence[str],
    request: Mapping[str, Any] | None = None,
) -> tuple[list[str], dict]:
    min_gpu_ram_mb = _requested_do_min_gpu_ram_mb(request)
    if min_gpu_ram_mb <= 0:
        unrestricted = list(size_candidates)
        return unrestricted, {
            "min_gpu_ram_mb": 0,
            "allowed_size_candidates": unrestricted,
            "rejected_size_candidates": [],
        }
    allowed: list[str] = []
    rejected: list[dict] = []
    for size in size_candidates:
        gpu_ram_mb = DO_GPU_SIZE_VRAM_MB.get(size)
        if gpu_ram_mb is None:
            rejected.append({
                "size": size,
                "min_gpu_ram_mb": min_gpu_ram_mb,
                "reason": "unknown_gpu_ram",
            })
            continue
        if gpu_ram_mb < min_gpu_ram_mb:
            rejected.append({
                "size": size,
                "gpu_ram_mb": gpu_ram_mb,
                "min_gpu_ram_mb": min_gpu_ram_mb,
                "reason": "below_min_gpu_ram",
            })
            continue
        allowed.append(size)
    return allowed, {
        "min_gpu_ram_mb": min_gpu_ram_mb,
        "allowed_size_candidates": allowed,
        "rejected_size_candidates": rejected,
    }


def _filter_do_size_candidates_by_render_capability(
    size_candidates: Sequence[str],
    request: Mapping[str, Any] | None = None,
) -> tuple[list[str], dict]:
    req = _mapping(request)
    requires_rtx = req.get("requires_rtx") is not False
    if not requires_rtx:
        unrestricted = list(size_candidates)
        return unrestricted, {
            "requires_rtx": False,
            "allowed_size_candidates": unrestricted,
            "rejected_size_candidates": [],
        }
    allowed: list[str] = []
    rejected: list[dict] = []
    for size in size_candidates:
        if DO_GPU_SIZE_RTX_CAPABLE.get(size) is True:
            allowed.append(size)
        else:
            rejected.append(
                {
                    "size": size,
                    "rtx_capable": DO_GPU_SIZE_RTX_CAPABLE.get(size),
                    "reason": "rtx_render_capability_missing",
                }
            )
    return allowed, {
        "requires_rtx": True,
        "allowed_size_candidates": allowed,
        "rejected_size_candidates": rejected,
        "claim_boundary": (
            "RTX capability is a render-lane requirement, separate from CUDA model "
            "inference support and GPU memory capacity."
        ),
    }


def _do_size_region_unavailable(body: dict) -> bool:
    text = json.dumps(body, sort_keys=True).lower() if isinstance(body, dict) else ""
    return "size is not available in this region" in text


def _configured_do_ssh_key_ids() -> tuple[list[int | str], dict]:
    import os

    raw = (os.getenv(DO_SSH_KEY_IDS_ENV) or "").strip()
    if raw:
        return _parse_do_ssh_key_ids(raw), {
            "source": DO_SSH_KEY_IDS_ENV,
            "account_lookup_performed": False,
        }
    path = Path(
        (os.getenv(DO_SSH_KEY_IDS_FILE_ENV) or "").strip()
        or DEFAULT_DO_SSH_KEY_IDS_FILE
    ).expanduser()
    if path.is_file():
        configured = path.read_text(encoding="utf-8").strip()
        if configured:
            return _parse_do_ssh_key_ids(configured), {
                "source": DO_SSH_KEY_IDS_FILE_ENV
                if os.getenv(DO_SSH_KEY_IDS_FILE_ENV)
                else "default_file:digitalocean_ssh_key_ids",
                "account_lookup_performed": False,
            }
    return [], {
        "source": None,
        "account_lookup_performed": False,
    }


def _account_do_ssh_key_ids(token: str) -> tuple[list[int | str], dict]:
    status, body = _do_call("GET", "/account/keys?per_page=200", token=token, timeout=45)
    if status != 200 or not isinstance(body, dict):
        return [], {
            "source": "account_keys_api",
            "account_lookup_performed": True,
            "account_lookup_status": status,
            "raw_provider_response_recorded": False,
            "blocker": "digitalocean_ssh_key_lookup_failed",
        }
    keys: list[int | str] = []
    for item in body.get("ssh_keys") or []:
        if not isinstance(item, dict):
            continue
        key_id = item.get("id") or item.get("fingerprint")
        if key_id:
            keys.append(key_id)
    return keys[:1], {
        "source": "account_keys_api_first_available",
        "account_lookup_performed": True,
        "account_lookup_status": status,
        "account_key_count": len(keys),
        "raw_provider_response_recorded": False,
    }


def _resolve_do_ssh_key_ids(token: str) -> tuple[list[int | str], dict]:
    keys, detail = _configured_do_ssh_key_ids()
    if keys:
        detail["configured_key_count"] = len(keys)
        return keys, detail
    keys, detail = _account_do_ssh_key_ids(token)
    if keys:
        detail["configured_key_count"] = len(keys)
        return keys, detail
    detail.setdefault("blocker", "digitalocean_ssh_key_missing")
    detail["configured_key_count"] = 0
    return [], detail


class DigitalOceanRenderProvider(GpuRenderProvider):
    """GPU Droplets (Paperspace's successor inside DigitalOcean) as a render leg.

    VM-based: the NVIDIA AI/ML-ready image boots with drivers + docker, and
    cloud-init user_data starts the same worker container RunPod runs. Slower
    first boot than a container host (VM provision + image pull) but a
    dedicated droplet, not a marketplace node lottery. Droplets bill until
    DESTROYED — ``stop`` powers off but keeps billing; prefer ``terminate``.
    """

    name = "digitalocean"

    def _token(self) -> str | None:
        import os

        path = Path(
            (os.getenv("DIGITALOCEAN_TOKEN_FILE") or "").strip()
            or "~/.blueprint-secrets/digitalocean_api_token"
        ).expanduser()
        try:
            return path.read_text().strip() or None
        except OSError:
            return None

    def available(self) -> dict:
        ok = self._token() is not None
        return {"provider": self.name, "available": ok,
                "reason": None if ok else "digitalocean_token_missing"}

    def build_request(self, spec: RenderLaunchSpec, job_dir: Path) -> dict:
        import os

        return {
            "name": spec.name,
            "region": (os.getenv("BLUEPRINT_DO_GPU_REGION") or "").strip() or DEFAULT_DO_GPU_REGION,
            "size": (os.getenv("BLUEPRINT_DO_GPU_SIZE") or "").strip() or DEFAULT_DO_GPU_SIZE,
            "image": DO_GPU_BASE_IMAGE,
            "user_data": _do_user_data(spec),
            "env": dict(spec.env),
            "_blueprint_worker_image": spec.image,
            "_blueprint_bootstrap_argv": list(spec.bootstrap_argv),
            "_blueprint_entrypoint": list(spec.entrypoint),
            "min_gpu_ram_mb": int(spec.min_gpu_ram_mb),
            "requires_rtx": bool(spec.requires_rtx),
            "tags": ["blueprint-isaac-render"],
            "ipv6": False,
            "monitoring": False,
        }

    def capacity_preflight(self, request: Mapping[str, Any] | None = None) -> dict:
        """Read-only GPU size/region preflight.

        DigitalOcean's sizes API reports the regions each size can currently be
        created in for this account. If every requested NVIDIA GPU size has an
        empty/non-overlapping region list, a paid launch would only fail at
        ``POST /droplets`` after the caller has staged large bundles. This probe
        lets higher-level jobs fail before that staging work while still making
        no billable provider call.
        """
        token = self._token()
        if not token:
            return {
                "status": "unknown",
                "provider": self.name,
                "blockers": ["digitalocean_token_missing"],
            }
        req = _mapping(request)
        initial_region = str(req.get("region") or DEFAULT_DO_GPU_REGION)
        initial_size = str(req.get("size") or DEFAULT_DO_GPU_SIZE)
        size_candidates, budget_policy = _filter_do_size_candidates_by_budget(
            _do_size_candidates(initial_size),
            req,
        )
        size_candidates, gpu_ram_policy = _filter_do_size_candidates_by_gpu_ram(
            size_candidates,
            req,
        )
        size_candidates, render_capability_policy = (
            _filter_do_size_candidates_by_render_capability(size_candidates, req)
        )
        region_candidates = _do_region_candidates(initial_region)
        if not size_candidates:
            blockers = ["digitalocean_gpu_size_below_min_vram"]
            if budget_policy.get("allowed_size_candidates") == []:
                blockers = ["digitalocean_gpu_size_over_budget"]
            elif gpu_ram_policy.get("allowed_size_candidates"):
                blockers = ["digitalocean_gpu_size_not_rtx_capable"]
            return {
                "status": "blocked",
                "provider": self.name,
                "blockers": blockers,
                "budget_policy": budget_policy,
                "gpu_ram_policy": gpu_ram_policy,
                "render_capability_policy": render_capability_policy,
                "size_candidates": [],
                "region_candidates": region_candidates,
                "raw_provider_response_recorded": False,
            }
        status, body = _do_call("GET", "/sizes?per_page=200", token=token, timeout=60)
        if status != 200 or not isinstance(body, dict):
            return {
                "status": "unknown",
                "provider": self.name,
                "blockers": ["digitalocean_gpu_capacity_probe_failed"],
                "http": status,
                "budget_policy": budget_policy,
                "gpu_ram_policy": gpu_ram_policy,
                "render_capability_policy": render_capability_policy,
                "size_candidates": size_candidates,
                "region_candidates": region_candidates,
                "raw_provider_response_recorded": False,
            }
        by_slug = {
            str(item.get("slug") or ""): item
            for item in body.get("sizes") or []
            if isinstance(item, Mapping)
        }
        considered: list[dict[str, Any]] = []
        viable: list[dict[str, Any]] = []
        for size in size_candidates:
            record = _mapping(by_slug.get(size))
            available_regions = _string_list(record.get("regions"))
            matching_regions = [r for r in region_candidates if r in available_regions]
            row = {
                "size": size,
                "provider_available": bool(record.get("available")),
                "provider_regions": available_regions,
                "matching_regions": matching_regions,
                "price_hourly": record.get("price_hourly"),
                "memory_mb": record.get("memory"),
                "gpu_ram_mb": DO_GPU_SIZE_VRAM_MB.get(size),
                "vcpus": record.get("vcpus"),
            }
            if not record:
                row["blocker"] = "digitalocean_gpu_size_not_listed"
            elif record.get("available") is not True:
                row["blocker"] = "digitalocean_gpu_size_not_available"
            elif not matching_regions:
                row["blocker"] = "digitalocean_gpu_size_region_unavailable"
            else:
                viable.append(row)
            considered.append(row)
        if viable:
            return {
                "status": "available",
                "provider": self.name,
                "blockers": [],
                "budget_policy": budget_policy,
                "gpu_ram_policy": gpu_ram_policy,
                "render_capability_policy": render_capability_policy,
                "size_candidates": size_candidates,
                "region_candidates": region_candidates,
                "viable_size_regions": viable,
                "considered_size_regions": considered,
                "raw_provider_response_recorded": False,
                "claim_boundary": (
                    "Capacity preflight is a read-only size/region check. It "
                    "does not reserve capacity or prove droplet creation will "
                    "succeed."
                ),
            }
        return {
            "status": "blocked",
            "provider": self.name,
            "blockers": ["digitalocean_gpu_size_region_unavailable"],
            "budget_policy": budget_policy,
            "gpu_ram_policy": gpu_ram_policy,
            "render_capability_policy": render_capability_policy,
            "size_candidates": size_candidates,
            "region_candidates": region_candidates,
            "considered_size_regions": considered,
            "raw_provider_response_recorded": False,
            "claim_boundary": (
                "Capacity preflight is a read-only size/region check. Empty or "
                "non-overlapping provider region lists predict a no-allocation "
                "create failure; no billable droplet was requested."
            ),
        }

    def launch(
        self,
        job_dir: Path,
        request: dict,
        *,
        cold: bool = False,
        allow_cold_fallback: bool = True,
    ) -> dict:
        token = self._token()
        if not token:
            return {"status": "blocked", "blockers": ["digitalocean_token_missing"]}
        prelaunch_blockers = _render_prelaunch_guard_blockers(
            request, provider_name="digitalocean"
        )
        if prelaunch_blockers:
            return {
                "status": "blocked",
                "blockers": prelaunch_blockers,
                "prelaunch_spend_guard": _mapping(request.get("prelaunch_spend_guard"))
                or None,
            }
        launch_request = dict(request)
        launch_request.pop("prelaunch_spend_guard", None)
        worker_env = launch_request.pop("env", None)
        worker_image = launch_request.pop("_blueprint_worker_image", None)
        bootstrap_argv = launch_request.pop("_blueprint_bootstrap_argv", None)
        entrypoint = launch_request.pop("_blueprint_entrypoint", None)
        min_gpu_ram_mb = _requested_do_min_gpu_ram_mb(launch_request)
        launch_request.pop("min_gpu_ram_mb", None)
        requires_rtx = launch_request.pop("requires_rtx", True) is not False
        budget_request = {
            "max_hourly_rate_usd": launch_request.pop("max_hourly_rate_usd", None)
        }
        size_filter_request = {
            **budget_request,
            "min_gpu_ram_mb": min_gpu_ram_mb,
            "requires_rtx": requires_rtx,
        }
        if (
            isinstance(worker_env, dict)
            and worker_image
            and isinstance(bootstrap_argv, list)
        ):
            launch_request["user_data"] = _do_user_data(
                RenderLaunchSpec(
                    name=str(launch_request.get("name") or "blueprint-isaac-g1-kitchen-parity"),
                    image=str(worker_image),
                    env=dict(worker_env),
                    bootstrap_argv=[str(item) for item in bootstrap_argv],
                    entrypoint=[str(item) for item in entrypoint]
                    if isinstance(entrypoint, list)
                    else ["bash"],
                )
            )
        ssh_key_detail: dict = {
            "source": "request",
            "account_lookup_performed": False,
            "configured_key_count": len(launch_request.get("ssh_keys") or []),
        }
        if not launch_request.get("ssh_keys"):
            ssh_keys, ssh_key_detail = _resolve_do_ssh_key_ids(token)
            if not ssh_keys:
                return {
                    "status": "blocked",
                    "blockers": [ssh_key_detail.get("blocker") or "digitalocean_ssh_key_missing"],
                    "ssh_key_configuration": ssh_key_detail,
                }
            launch_request["ssh_keys"] = ssh_keys
        create_attempts: list[dict] = []
        initial_region = str(launch_request.get("region") or DEFAULT_DO_GPU_REGION)
        initial_size = str(launch_request.get("size") or DEFAULT_DO_GPU_SIZE)
        size_candidates, budget_policy = _filter_do_size_candidates_by_budget(
            _do_size_candidates(initial_size),
            size_filter_request,
        )
        size_candidates, gpu_ram_policy = _filter_do_size_candidates_by_gpu_ram(
            size_candidates,
            size_filter_request,
        )
        size_candidates, render_capability_policy = (
            _filter_do_size_candidates_by_render_capability(
                size_candidates,
                size_filter_request,
            )
        )
        if not size_candidates:
            blockers = ["digitalocean_gpu_size_below_min_vram"]
            if budget_policy.get("allowed_size_candidates") == []:
                blockers = ["digitalocean_gpu_size_over_budget"]
            elif gpu_ram_policy.get("allowed_size_candidates"):
                blockers = ["digitalocean_gpu_size_not_rtx_capable"]
            return {
                "status": "blocked",
                "blockers": blockers,
                "budget_policy": budget_policy,
                "gpu_ram_policy": gpu_ram_policy,
                "render_capability_policy": render_capability_policy,
                "ssh_key_configuration": ssh_key_detail,
            }
        for size in size_candidates:
            for region in _do_region_candidates(initial_region):
                attempt_request = dict(launch_request, size=size, region=region)
                s, body = _do_call("POST", "/droplets", attempt_request, token=token)
                droplet = _mapping(_mapping(body).get("droplet"))
                iid = _normalize_provider_instance_id(
                    droplet.get("id"), numeric_only=True
                )
                if s in (201, 202) and iid:
                    started_id_record = _record_started_id(
                        Path(job_dir) / "started_do_droplet_id.txt", iid
                    )
                    create_attempts.append({
                        "create_status": s,
                        "droplet_id": iid,
                        "size": size,
                        "region": region,
                    })
                    return {
                        "status": "launched",
                        "instance_id": iid,
                        "mode": "do_gpu_droplet",
                        "ssh_key_configuration": ssh_key_detail,
                        "budget_policy": budget_policy,
                        "gpu_ram_policy": gpu_ram_policy,
                        "render_capability_policy": render_capability_policy,
                        "attempts": create_attempts,
                        "started_id_record": started_id_record,
                    }
                attempt = {
                    "create_status": s,
                    "size": size,
                    "region": region,
                    "retryable_region_capacity_error": _do_size_region_unavailable(
                        body if isinstance(body, dict) else {}
                    ),
                }
                if isinstance(body, dict) and body.get("error"):
                    attempt["error"] = str(body.get("error"))[:300]
                create_attempts.append(attempt)
                ambiguous_create = bool(
                    not attempt["retryable_region_capacity_error"]
                    and (s == 0 or 200 <= s < 300 or s >= 500)
                )
                if ambiguous_create:
                    return {
                        "status": "blocked",
                        "blockers": ["digitalocean_create_outcome_ambiguous"],
                        "attempts": create_attempts,
                        "allocation_outcome_ambiguous": True,
                        "budget_policy": budget_policy,
                        "gpu_ram_policy": gpu_ram_policy,
                        "render_capability_policy": render_capability_policy,
                        "ssh_key_configuration": ssh_key_detail,
                    }
                if not attempt["retryable_region_capacity_error"]:
                    return {
                        "status": "blocked",
                        "blockers": [f"do_droplet_create_http_{s}"],
                        "attempts": create_attempts,
                        "budget_policy": budget_policy,
                        "gpu_ram_policy": gpu_ram_policy,
                        "render_capability_policy": render_capability_policy,
                        "ssh_key_configuration": ssh_key_detail,
                        "allocation_created": False,
                    }
        return {
            "status": "blocked",
            "blockers": ["digitalocean_gpu_size_region_unavailable"],
            "attempts": create_attempts,
            "budget_policy": budget_policy,
            "gpu_ram_policy": gpu_ram_policy,
            "render_capability_policy": render_capability_policy,
            "ssh_key_configuration": ssh_key_detail,
            "allocation_created": False,
        }

    def inspect(self, instance_id: str) -> dict:
        token = self._token()
        if not token:
            return {"status": "unavailable", "reason": "digitalocean_token_missing",
                    "instance_id": instance_id}
        s, body = _do_call("GET", f"/droplets/{instance_id}", token=token)
        droplet = (body or {}).get("droplet") if isinstance(body, dict) else None
        return {"status": "ok" if s == 200 else "unavailable", "http": s,
                "instance_id": instance_id,
                "droplet_status": (droplet or {}).get("status"),
                "raw_provider_response_recorded": False}

    def billable_inventory(self, *, name_prefix: str) -> dict:
        token = self._token()
        if not token:
            return {
                "status": "blocked",
                "provider": self.name,
                "name_prefix": str(name_prefix),
                "live_resource_count": None,
                "resources": [],
                "api_confirmed": False,
                "blockers": ["digitalocean_token_missing"],
            }
        status, body = _do_call("GET", "/droplets?per_page=200", token=token)
        rows = _mapping(body).get("droplets")
        if status != 200 or not isinstance(rows, list):
            return {
                "status": "blocked",
                "provider": self.name,
                "name_prefix": str(name_prefix),
                "live_resource_count": None,
                "resources": [],
                "api_confirmed": False,
                "blockers": ["digitalocean_billable_inventory_failed"],
                "http": status,
                "raw_provider_response_recorded": False,
            }
        prefix = str(name_prefix or "")
        resources = []
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            name = str(row.get("name") or "")
            if prefix and not name.startswith(prefix):
                continue
            resources.append(
                {
                    "instance_id": str(row.get("id") or ""),
                    "name": name,
                    "status": row.get("status"),
                    "size_slug": row.get("size_slug"),
                    "region": _mapping(row.get("region")).get("slug"),
                }
            )
        return {
            "status": "observed",
            "provider": self.name,
            "name_prefix": prefix,
            "live_resource_count": len(resources),
            "resources": resources,
            "api_confirmed": True,
            "http": status,
            "raw_provider_response_recorded": False,
        }

    def stop(self, instance_id: str) -> dict:
        token = self._token()
        if not token:
            return {"status": "blocked", "blockers": ["digitalocean_token_missing"]}
        s, _body = _do_call("POST", f"/droplets/{instance_id}/actions",
                            {"type": "power_off"}, token=token)
        return {"status": "stopped" if s in (200, 201) else "stop_failed", "http": s,
                "warning": "powered-off droplets keep full billing until destroyed; "
                           "use terminate() to stop spend"}

    def terminate(self, instance_id: str) -> dict:
        token = self._token()
        if not token:
            return {"status": "blocked", "blockers": ["digitalocean_token_missing"]}
        s, _body = _do_call("DELETE", f"/droplets/{instance_id}", token=token)
        if s in (204, 404):
            return {"status": "terminated", "http": s, "already_gone": s == 404}
        return {"status": "terminate_failed", "http": s}


def get_render_provider(name: str | None, *, warm_candidates: Sequence[str] = ()) -> GpuRenderProvider:
    key = (name or "runpod").strip().lower()
    if key == "runpod":
        return RunPodRenderProvider(warm_candidates=warm_candidates)
    if key == "vast":
        return VastRenderProvider()
    if key == "digitalocean":
        return DigitalOceanRenderProvider()
    if key == "gcp":
        from .cloud_vm_render_providers import GCPRenderProvider
        return GCPRenderProvider()
    if key == "aws":
        from .cloud_vm_render_providers import AWSRenderProvider
        return AWSRenderProvider()
    raise ValueError(
        f"unknown_render_provider:{name!r} "
        "(known: runpod, vast, digitalocean, gcp, aws)"
    )


def list_render_providers() -> list[dict]:
    """Report each provider and whether its credentials are present in this env."""
    from .cloud_vm_render_providers import AWSRenderProvider, GCPRenderProvider
    return [
        RunPodRenderProvider().available(),
        VastRenderProvider().available(),
        DigitalOceanRenderProvider().available(),
        GCPRenderProvider().available(),
        AWSRenderProvider().available(),
    ]
