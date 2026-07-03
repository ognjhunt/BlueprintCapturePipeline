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

This deliberately does NOT depend on the heavy ``robot_eval_gpu_provider_launch_request``
schema used by :mod:`runpod_provider_adapter` / :mod:`vast_provider_adapter`; it is a thin
launch layer scoped to the splat render bundle contract.

Secrets are file-based under ``~/.blueprint-secrets`` and never logged.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

SCHEMA_VERSION = "gpu_render_providers.v1"
SECRETS = Path.home() / ".blueprint-secrets"
RUNPOD_API = "https://rest.runpod.io/v1"


def _read_secret(name: str) -> str | None:
    p = SECRETS / name
    return p.read_text().strip() if p.is_file() else None


# ----------------------------- neutral launch spec -----------------------------

DEFAULT_RUNPOD_RENDER_GPU_TYPES: tuple = (
    "NVIDIA L40S", "NVIDIA RTX 6000 Ada Generation", "NVIDIA RTX A6000",
    "NVIDIA L40", "NVIDIA A40",
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

    def terminate(self, instance_id: str) -> dict:
        """Permanently delete the instance (releases its disk too). Defaults to stop(); RunPod
        overrides because a stopped pod keeps billing for its container disk."""
        return self.stop(instance_id)


# ----------------------------- RunPod -----------------------------

def _runpod_call(method: str, path: str, body: dict | None, *, key: str, timeout: int = 90):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        RUNPOD_API + path, data=data, method=method,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read().decode()
            return r.status, (json.loads(raw) if raw.strip() else {})
    except urllib.error.HTTPError as e:
        return e.code, {"error": e.read().decode()[:400]}
    except Exception as e:  # noqa: BLE001
        return 0, {"error": repr(e)[:300]}


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

    def build_request(self, spec: RenderLaunchSpec, job_dir: Path) -> dict:
        return {
            "name": spec.name, "imageName": spec.image,
            "gpuTypeIds": list(spec.gpu_types), "gpuTypePriority": "availability",
            "cloudType": "SECURE", "gpuCount": spec.gpu_count,
            "containerDiskInGb": spec.container_disk_gb, "volumeInGb": spec.volume_gb,
            "volumeMountPath": spec.volume_mount_path,
            "minVCPUPerGPU": spec.min_vcpu, "minRAMPerGPU": spec.min_ram_gb,
            "env": dict(spec.env), "dockerEntrypoint": list(spec.entrypoint),
            "dockerStartCmd": list(spec.bootstrap_argv),
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
        attempts: list[dict] = []
        if not cold and self.warm_candidates:
            upd = {k: request[k] for k in (
                "imageName", "containerDiskInGb", "volumeInGb", "volumeMountPath",
                "env", "dockerEntrypoint", "dockerStartCmd") if k in request}
            for pid in self.warm_candidates:
                attempt: dict = {"pod_id": pid}
                s, get_body = _runpod_call("GET", f"/pods/{pid}", None, key=key)
                attempt["get_status"] = s
                if isinstance(get_body, dict):
                    attempt["desiredStatus"] = get_body.get("desiredStatus")
                    if get_body.get("error"):
                        attempt["get_error"] = get_body.get("error")
                if s != 200:
                    attempts.append(attempt)
                    continue
                us, update_body = _runpod_call("POST", f"/pods/{pid}/update", upd, key=key)
                attempt["update_status"] = us
                if isinstance(update_body, dict) and update_body.get("error"):
                    attempt["update_error"] = update_body.get("error")
                if us not in (200, 201, 204):
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
                if ss in (200, 201):
                    (job_dir / "started_pod_id.txt").write_text(pid)
                    return {"status": "launched", "instance_id": pid,
                            "mode": "warm_restart", "attempts": attempts}
        if not cold and self.warm_candidates and not allow_cold_fallback:
            return {
                "status": "blocked",
                "blockers": ["warm_restart_failed_cold_fallback_disabled"],
                "attempts": attempts,
            }
        s, r = _runpod_call("POST", "/pods", request, key=key)
        pid = r.get("id") if isinstance(r, dict) else None
        attempts.append({"cold_create_status": s, "pod_id": pid,
                         "error": r.get("error") if isinstance(r, dict) else None})
        if pid:
            (job_dir / "started_pod_id.txt").write_text(pid)
            return {"status": "launched", "instance_id": pid,
                    "mode": "cold_create", "attempts": attempts}
        return {"status": "blocked", "blockers": ["no_pod_started"], "attempts": attempts}

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

    def terminate(self, instance_id: str) -> dict:
        """DELETE the pod — a stopped RunPod pod still bills for its 140GB+ container disk, so
        render pods must be deleted, not just stopped, to avoid runaway storage charges."""
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
                last_blocker = f"vast_create_http_error:{e.code}"
                continue
            except Exception as e:  # noqa: BLE001
                attempts.append({"create_error": repr(e)[:200], "ask_id": ask_id})
                last_blocker = "vast_create_failed"
                continue
            iid = None
            if isinstance(cresp, dict):
                for k in ("new_contract", "instance_id", "id"):
                    if cresp.get(k):
                        iid = str(cresp[k])
                        break
            attempts.append({"create_status": cs, "instance_id": iid,
                             "gpu_name": offer.get("gpu_name"),
                             "hourly_rate_usd": offer.get("hourly_rate_usd")})
            if iid:
                (job_dir / "started_vast_instance_id.txt").write_text(iid)
                return {"status": "launched", "instance_id": iid, "mode": "vast_on_demand",
                        "attempts": attempts}
            last_blocker = "vast_instance_not_created"
        return {"status": "blocked", "blockers": [last_blocker], "attempts": attempts}

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
DEFAULT_DO_MAX_HOURLY_RATE_USD = 1.75
DO_GPU_SIZE_CANDIDATES_ENV = "BLUEPRINT_DO_GPU_SIZES"
DO_GPU_REGION_CANDIDATES_ENV = "BLUEPRINT_DO_GPU_REGIONS"
DO_GPU_MAX_HOURLY_RATE_ENV = "BLUEPRINT_DO_MAX_HOURLY_RATE_USD"
DO_SSH_KEY_IDS_ENV = "BLUEPRINT_DO_SSH_KEY_IDS"
DO_SSH_KEY_IDS_FILE_ENV = "BLUEPRINT_DO_SSH_KEY_IDS_FILE"
DEFAULT_DO_SSH_KEY_IDS_FILE = "~/.blueprint-secrets/digitalocean_ssh_key_ids"


def _do_call(method: str, path: str, body: dict | None = None, *, token: str, timeout: int = 90):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        DO_API + path, data=data, method=method,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read().decode()
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
    entrypoint = shlex.quote(spec.entrypoint[0] if spec.entrypoint else "bash")
    return f"""#!/bin/bash
set -x
echo {env_b64} | base64 -d > /root/blueprint_worker.env
echo {argv_b64} | base64 -d > /root/blueprint_argv.json
python3 - <<'PY'
import json, shlex
argv = json.load(open("/root/blueprint_argv.json"))
open("/root/blueprint_run.sh", "w").write(" ".join(shlex.quote(a) for a in argv))
PY
docker pull {shlex.quote(spec.image)}
docker run -d --gpus all --name blueprint-worker \\
  --env-file /root/blueprint_worker.env \\
  --shm-size=8g \\
  --entrypoint {entrypoint} \\
  {shlex.quote(spec.image)} $(cat /root/blueprint_run.sh)
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


def _do_max_hourly_rate_usd() -> float:
    import os

    raw = (os.getenv(DO_GPU_MAX_HOURLY_RATE_ENV) or "").strip()
    if raw:
        try:
            value = float(raw)
            if value > 0:
                return value
        except ValueError:
            pass
    return DEFAULT_DO_MAX_HOURLY_RATE_USD


def _filter_do_size_candidates_by_budget(size_candidates: Sequence[str]) -> tuple[list[str], dict]:
    max_hourly = _do_max_hourly_rate_usd()
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
        "max_hourly_rate_env": DO_GPU_MAX_HOURLY_RATE_ENV,
        "allowed_size_candidates": allowed,
        "rejected_size_candidates": rejected,
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
            "tags": ["blueprint-isaac-render"],
            "ipv6": False,
            "monitoring": False,
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
        launch_request = dict(request)
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
            _do_size_candidates(initial_size)
        )
        if not size_candidates:
            return {
                "status": "blocked",
                "blockers": ["digitalocean_gpu_size_over_budget"],
                "budget_policy": budget_policy,
                "ssh_key_configuration": ssh_key_detail,
            }
        for size in size_candidates:
            for region in _do_region_candidates(initial_region):
                attempt_request = dict(launch_request, size=size, region=region)
                s, body = _do_call("POST", "/droplets", attempt_request, token=token)
                if s in (201, 202) and isinstance(body, dict) and body.get("droplet"):
                    iid = str(body["droplet"].get("id"))
                    (Path(job_dir) / "started_do_droplet_id.txt").write_text(iid)
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
                        "attempts": create_attempts,
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
                if not attempt["retryable_region_capacity_error"]:
                    return {
                        "status": "blocked",
                        "blockers": [f"do_droplet_create_http_{s}"],
                        "attempts": create_attempts,
                        "budget_policy": budget_policy,
                        "ssh_key_configuration": ssh_key_detail,
                    }
        return {
            "status": "blocked",
            "blockers": ["digitalocean_gpu_size_region_unavailable"],
            "attempts": create_attempts,
            "budget_policy": budget_policy,
            "ssh_key_configuration": ssh_key_detail,
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
    raise ValueError(f"unknown_render_provider:{name!r} (known: runpod, vast, digitalocean)")


def list_render_providers() -> list[dict]:
    """Report each provider and whether its credentials are present in this env."""
    return [RunPodRenderProvider().available(), VastRenderProvider().available(),
            DigitalOceanRenderProvider().available()]
