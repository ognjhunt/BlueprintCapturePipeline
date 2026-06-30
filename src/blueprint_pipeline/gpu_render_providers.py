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
    # required for Isaac RTX / splat rendering — A100/H100 excluded). Broad list to find capacity.
    gpu_types: tuple = (
        "NVIDIA L40S", "NVIDIA RTX 6000 Ada Generation", "NVIDIA RTX A6000",
        "NVIDIA L40", "NVIDIA A40", "NVIDIA GeForce RTX 4090",
    )
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
        offer = _select_offer(offers, max_hourly_rate=max_rate, min_gpu_ram_mb=min_ram)
        attempts.append({"offer_search_status": s, "offer_count": len(offers),
                         "selected": bool(offer)})
        if not offer:
            return {"status": "blocked",
                    "blockers": ["no_vast_offer_matching_rate_and_gpu_memory"],
                    "attempts": attempts}
        ask_id = offer.get("ask_contract_id")
        create_payload = request.get("create_payload") or {}
        try:
            cs, cresp = _api_json(method="PUT", path=f"/asks/{ask_id}/", api_key=key,
                                  payload=create_payload, timeout_seconds=45)
        except urllib.error.HTTPError as e:
            return {"status": "blocked", "blockers": [f"vast_create_http_error:{e.code}"],
                    "attempts": attempts}
        except Exception as e:  # noqa: BLE001
            return {"status": "blocked", "blockers": ["vast_create_failed"],
                    "error": repr(e)[:200], "attempts": attempts}
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
        return {"status": "blocked", "blockers": ["vast_instance_not_created"],
                "attempts": attempts}

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

def get_render_provider(name: str | None, *, warm_candidates: Sequence[str] = ()) -> GpuRenderProvider:
    key = (name or "runpod").strip().lower()
    if key == "runpod":
        return RunPodRenderProvider(warm_candidates=warm_candidates)
    if key == "vast":
        return VastRenderProvider()
    raise ValueError(f"unknown_render_provider:{name!r} (known: runpod, vast)")


def list_render_providers() -> list[dict]:
    """Report each provider and whether its credentials are present in this env."""
    return [RunPodRenderProvider().available(), VastRenderProvider().available()]
