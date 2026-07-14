"""First-class GCP Compute Engine and AWS EC2 GPU render providers.

Unlike marketplace providers, these adapters never invent account infrastructure.
Every account/project, location, VM shape, image, network, identity, registry mode,
and price is explicit configuration.  Missing or unverifiable configuration fails
closed before a mutating API call.  Provider responses are normalized to the
``GpuRenderProvider`` launch/inventory/terminate contract.
"""
from __future__ import annotations

import base64
import json
import os
import re
import subprocess
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

from blueprint_pipeline import safe_outbound_http

from .gpu_render_providers import (
    GpuRenderProvider,
    RenderLaunchSpec,
    _mapping,
    _positive_float,
    _record_started_id,
    _render_prelaunch_guard_blockers,
    _string_list,
)

GCP_COMPUTE_API = "https://compute.googleapis.com/compute/v1"
_GCP_COMPUTE_POLICY = safe_outbound_http.pinned_api_policy(GCP_COMPUTE_API)
GCP_SERVICE_USAGE_API = "https://serviceusage.googleapis.com/v1beta1"
_GCP_SERVICE_USAGE_POLICY = safe_outbound_http.pinned_api_policy(GCP_SERVICE_USAGE_API)
GCP_CREDENTIALS_FILE_ENV = "GOOGLE_APPLICATION_CREDENTIALS"
AWS_CREDENTIALS_FILE_ENV = "AWS_SHARED_CREDENTIALS_FILE"

_NAME_RE = re.compile(r"^[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?$")
_AWS_ID_RE = re.compile(r"^i-[0-9a-f]{8,32}$")


class _AccessTokenCredentials:
    """Minimal credential carrier for an already-issued short-lived token."""

    def __init__(self, token: str) -> None:
        self.token = token
        self.valid = True


def _env(name: str) -> str:
    return (os.getenv(name) or "").strip()


def _csv(name: str) -> list[str]:
    return [item.strip() for item in _env(name).split(",") if item.strip()]


def _positive_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _required_config(values: Mapping[str, Any], names: Sequence[str], prefix: str) -> list[str]:
    return [f"{prefix}_{name}_missing" for name in names if not str(values.get(name) or "").strip()]


def _worker_cloud_init(
    spec: RenderLaunchSpec,
    *,
    provider: str,
    registry_auth: str,
    registry_host: str | None,
    aws_region: str | None = None,
) -> str:
    """Build a provider-neutral startup script for a pre-baked GPU host.

    Signed transport values are base64 encoded to preserve bytes, not to claim
    secrecy. Instance metadata is therefore limited to the explicitly scoped VM.

    Production customer startup deliberately performs no registry login or
    image pull.  The provider host image must already contain the exact worker
    digest and an identity marker written by the host-image build.  This keeps
    a 40+ GB transfer out of both the customer path and asynchronous cold boot.
    """
    del provider, registry_auth, registry_host, aws_region
    env_b64 = base64.b64encode(
        "\n".join(f"{key}={value}" for key, value in spec.env.items()).encode()
    ).decode()
    argv_b64 = base64.b64encode(json.dumps(list(spec.bootstrap_argv)).encode()).decode()
    image = json.dumps(spec.image)
    entrypoint = json.dumps(spec.entrypoint[0] if spec.entrypoint else "bash")
    return f"""#!/bin/bash
set -euo pipefail
umask 077
printf '%s' '{env_b64}' | base64 -d > /root/blueprint_worker.env
printf '%s' '{argv_b64}' | base64 -d > /root/blueprint_argv.json
mkdir -p /workspace/out
test -f /etc/blueprint/worker-image-ref
test "$(cat /etc/blueprint/worker-image-ref)" = {image}
docker image inspect {image} >/dev/null
python3 - <<'PY'
import json, subprocess
argv = json.load(open('/root/blueprint_argv.json'))
cmd = ['docker', 'run', '-d', '--gpus', 'all', '--name', 'blueprint-worker',
       '--env-file', '/root/blueprint_worker.env', '-v', '/workspace:/workspace',
       '--workdir', '/workspace', '--shm-size=8g', '--entrypoint', {entrypoint},
       {image}, *argv]
subprocess.check_call(cmd)
PY
"""


class GCPRenderProvider(GpuRenderProvider):
    """Compute Engine GPU VM adapter using Application Default Credentials."""

    name = "gcp"

    def _config(self) -> dict[str, Any]:
        return {
            "project": _env("BLUEPRINT_GCP_PROJECT"),
            "auth_mode": _env("BLUEPRINT_GCP_AUTH_MODE") or "application_default",
            "zone": _env("BLUEPRINT_GCP_ZONE"),
            "machine_type": _env("BLUEPRINT_GCP_MACHINE_TYPE"),
            "source_image": _env("BLUEPRINT_GCP_SOURCE_IMAGE"),
            "network": _env("BLUEPRINT_GCP_NETWORK"),
            "subnetwork": _env("BLUEPRINT_GCP_SUBNETWORK"),
            "service_account": _env("BLUEPRINT_GCP_SERVICE_ACCOUNT"),
            "accelerator_type": _env("BLUEPRINT_GCP_ACCELERATOR_TYPE"),
            "accelerator_count": _positive_int(_env("BLUEPRINT_GCP_ACCELERATOR_COUNT")) or 0,
            "gpu_quota_metric": _env("BLUEPRINT_GCP_GPU_QUOTA_METRIC"),
            "gpu_quota_units": _positive_float(_env("BLUEPRINT_GCP_GPU_QUOTA_UNITS")) or 1.0,
            "boot_disk_gb": _positive_int(_env("BLUEPRINT_GCP_BOOT_DISK_GB")) or 200,
            "boot_disk_type": _env("BLUEPRINT_GCP_BOOT_DISK_TYPE"),
            "private_egress_ready": _env("BLUEPRINT_GCP_PRIVATE_EGRESS_READY").lower()
            == "true",
            "fractional_vgpu_driver_ready": _env(
                "BLUEPRINT_GCP_FRACTIONAL_VGPU_DRIVER_READY"
            ).lower()
            == "true",
            "provisioning_model": _env("BLUEPRINT_GCP_PROVISIONING_MODEL") or "STANDARD",
            "max_hourly_rate_usd": _positive_float(_env("BLUEPRINT_GCP_MAX_HOURLY_RATE_USD")),
            "configured_hourly_rate_usd": _positive_float(_env("BLUEPRINT_GCP_HOURLY_RATE_USD")),
            "registry_auth": _env("BLUEPRINT_GCP_REGISTRY_AUTH") or "public",
            "registry_host": _env("BLUEPRINT_GCP_REGISTRY_HOST"),
        }

    def _credentials(self) -> tuple[Any | None, str | None]:
        try:
            if self._config()["auth_mode"] == "gcloud_cli":
                result = subprocess.run(
                    ["gcloud", "auth", "print-access-token"],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                token = result.stdout.strip()
                if not token:
                    return None, "GcloudAccessTokenEmpty"
                return _AccessTokenCredentials(token), None
            if self._config()["auth_mode"] != "application_default":
                return None, "GcpAuthModeInvalid"
            import google.auth
            from google.auth.transport.requests import Request

            credentials, _ = google.auth.default(
                scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )
            if not credentials.valid:
                credentials.refresh(Request())
            return credentials, None
        except Exception as exc:  # noqa: BLE001
            return None, type(exc).__name__

    def _call(
        self, method: str, path: str, body: Mapping[str, Any] | None = None, *, timeout: int = 90
    ) -> tuple[int, dict[str, Any]]:
        credentials, error = self._credentials()
        if credentials is None:
            return 0, {"error": f"gcp_credentials_unavailable:{error}"}
        request = urllib.request.Request(
            GCP_COMPUTE_API + path,
            data=json.dumps(dict(body)).encode() if body is not None else None,
            method=method,
            headers={"Authorization": f"Bearer {credentials.token}", "Content-Type": "application/json"},
        )
        try:
            response = safe_outbound_http.open_request(
                request,
                policy=_GCP_COMPUTE_POLICY,
                timeout_seconds=timeout,
            )
            raw = response.body.decode()
            return response.status, json.loads(raw) if raw.strip() else {}
        except urllib.error.HTTPError as exc:
            return exc.code, {"error": "gcp_compute_http_error"}
        except Exception as exc:  # noqa: BLE001
            return 0, {"error": type(exc).__name__}

    def _service_usage_call(self, path: str, *, timeout: int = 60) -> tuple[int, dict[str, Any]]:
        credentials, error = self._credentials()
        if credentials is None:
            return 0, {"error": f"gcp_credentials_unavailable:{error}"}
        request = urllib.request.Request(
            GCP_SERVICE_USAGE_API + path,
            method="GET",
            headers={"Authorization": f"Bearer {credentials.token}"},
        )
        try:
            response = safe_outbound_http.open_request(
                request,
                policy=_GCP_SERVICE_USAGE_POLICY,
                timeout_seconds=timeout,
            )
            raw = response.body.decode()
            return response.status, json.loads(raw) if raw.strip() else {}
        except urllib.error.HTTPError as exc:
            return exc.code, {"error": "gcp_service_usage_http_error"}
        except Exception as exc:  # noqa: BLE001
            return 0, {"error": type(exc).__name__}

    def available(self) -> dict:
        config = self._config()
        missing = _required_config(
            config,
            ("project", "zone", "machine_type", "source_image", "network", "subnetwork", "service_account", "gpu_quota_metric"),
            "gcp",
        )
        credentials, error = self._credentials()
        if credentials is None:
            missing.append("gcp_application_default_credentials_missing")
        return {
            "provider": self.name,
            "available": not missing,
            "reason": missing[0] if missing else None,
            "blockers": missing,
            "project": config["project"] or None,
            "zone": config["zone"] or None,
            "credentials_source": config["auth_mode"] if credentials else None,
            "credential_error_type": error,
            "raw_secret_values_recorded": False,
        }

    def build_request(self, spec: RenderLaunchSpec, job_dir: Path) -> dict:
        config = self._config()
        registry_auth = str(config["registry_auth"])
        blockers = _required_config(
            config,
            ("project", "zone", "machine_type", "source_image", "network", "subnetwork", "service_account", "gpu_quota_metric"),
            "gcp",
        )
        if registry_auth not in {"public", "gcp_artifact_registry"}:
            blockers.append("gcp_registry_auth_invalid")
        provisioning_model = str(config["provisioning_model"]).upper()
        if provisioning_model not in {"STANDARD", "SPOT"}:
            blockers.append("gcp_provisioning_model_invalid")
        if registry_auth == "gcp_artifact_registry" and not config["registry_host"]:
            blockers.append("gcp_registry_host_missing")
        # The provider intentionally creates no external IP.  Pulling the worker
        # and uploading artifacts therefore requires a pre-existing, verified
        # private egress path (Cloud NAT and/or Private Google Access as relevant).
        if not config["private_egress_ready"]:
            blockers.append("gcp_private_egress_unverified")
        machine_type = str(config["machine_type"])
        fractional_g4 = machine_type in {
            "g4-standard-6",
            "g4-standard-12",
            "g4-standard-24",
        }
        if fractional_g4 and not config["fractional_vgpu_driver_ready"]:
            blockers.append("gcp_fractional_vgpu_driver_unverified")
        if config["configured_hourly_rate_usd"] is None:
            blockers.append("gcp_hourly_rate_unconfigured")
        if config["max_hourly_rate_usd"] is None:
            blockers.append("gcp_max_hourly_rate_unconfigured")
        elif (
            config["configured_hourly_rate_usd"] is not None
            and config["configured_hourly_rate_usd"] > config["max_hourly_rate_usd"]
        ):
            blockers.append("gcp_hourly_rate_exceeds_cap")
        name = spec.name.lower().replace("_", "-")[:63].rstrip("-")
        if not _NAME_RE.fullmatch(name):
            blockers.append("gcp_instance_name_invalid")
        project, zone = config["project"], config["zone"]
        network_interface: dict[str, Any] = {
            "network": f"projects/{project}/global/networks/{config['network']}",
        }
        if config["subnetwork"]:
            network_interface["subnetwork"] = str(config["subnetwork"])
        body: dict[str, Any] = {
            "name": name,
            "machineType": f"zones/{zone}/machineTypes/{config['machine_type']}",
            "deletionProtection": False,
            "disks": [{
                "boot": True,
                "autoDelete": True,
                "initializeParams": {
                    "sourceImage": config["source_image"],
                    "diskSizeGb": str(max(spec.container_disk_gb, int(config["boot_disk_gb"]))),
                    "diskType": (
                        f"zones/{zone}/diskTypes/"
                        + (
                            str(config["boot_disk_type"])
                            or ("hyperdisk-balanced" if machine_type.startswith("g4-") else "pd-balanced")
                        )
                    ),
                },
            }],
            "networkInterfaces": [network_interface],
            "serviceAccounts": [{
                "email": config["service_account"],
                "scopes": ["https://www.googleapis.com/auth/cloud-platform"],
            }],
            "metadata": {"items": [{
                "key": "startup-script",
                "value": _worker_cloud_init(
                    spec,
                    provider="gcp",
                    registry_auth=registry_auth,
                    registry_host=config["registry_host"],
                ),
            }]},
            "labels": {"blueprint-managed": "true", "blueprint-name-prefix": name[:40]},
            "scheduling": {"onHostMaintenance": "TERMINATE", "automaticRestart": False},
        }
        if provisioning_model == "SPOT":
            body["scheduling"].update(
                {"provisioningModel": "SPOT", "instanceTerminationAction": "DELETE"}
            )
        if config["accelerator_type"] and config["accelerator_count"]:
            body["guestAccelerators"] = [{
                "acceleratorType": f"zones/{zone}/acceleratorTypes/{config['accelerator_type']}",
                "acceleratorCount": config["accelerator_count"],
            }]
        return {
            "provider": self.name,
            "project": project,
            "zone": zone,
            "instance_name": name,
            "instance_body": body,
            "configured_hourly_rate_usd": config["configured_hourly_rate_usd"],
            "max_hourly_rate_usd": config["max_hourly_rate_usd"],
            "registry_auth": registry_auth,
            "gpu_quota_units": config["gpu_quota_units"],
            "fractional_g4": fractional_g4,
            "provisioning_model": provisioning_model,
            "configuration_blockers": blockers,
            "idempotency_request_id": str(
                uuid.uuid5(uuid.NAMESPACE_URL, f"gcp://{project}/{zone}/{name}")
            ),
        }

    def capacity_preflight(self, request: Mapping[str, Any] | None = None) -> dict:
        req = _mapping(request)
        blockers = list(_string_list(req.get("configuration_blockers")))
        project = str(req.get("project") or self._config()["project"])
        zone = str(req.get("zone") or self._config()["zone"])
        body = _mapping(req.get("instance_body"))
        machine_type = str(body.get("machineType") or "").rsplit("/", 1)[-1]
        if blockers:
            return {"status": "blocked", "provider": self.name, "blockers": blockers, "api_confirmed": False}
        checks: dict[str, Any] = {}
        for label, path in (
            ("machine_type", f"/projects/{project}/zones/{zone}/machineTypes/{machine_type}"),
            ("zone", f"/projects/{project}/zones/{zone}"),
            ("network", f"/projects/{project}/global/networks/{str(_mapping((body.get('networkInterfaces') or [{}])[0]).get('network') or '').rsplit('/', 1)[-1]}"),
            ("subnetwork", "/" + str(_mapping((body.get("networkInterfaces") or [{}])[0]).get("subnetwork") or "")),
        ):
            status, payload = self._call("GET", path, timeout=45)
            checks[label] = {"http": status, "verified": status == 200}
            if status != 200:
                blockers.append(f"gcp_{label}_preflight_failed")
        source_image = str(
            _mapping(_mapping((body.get("disks") or [{}])[0]).get("initializeParams")).get("sourceImage") or ""
        )
        image_path = "/" + source_image if source_image.startswith("projects/") else ""
        if not image_path:
            blockers.append("gcp_source_image_reference_invalid")
        else:
            status, _ = self._call("GET", image_path, timeout=45)
            checks["source_image"] = {"http": status, "verified": status == 200}
            if status != 200:
                blockers.append("gcp_source_image_preflight_failed")
        accelerator = (_mapping((body.get("guestAccelerators") or [{}])[0]).get("acceleratorType")
                       if body.get("guestAccelerators") else None)
        if accelerator:
            status, _ = self._call("GET", f"/projects/{project}/zones/{zone}/acceleratorTypes/{str(accelerator).rsplit('/', 1)[-1]}", timeout=45)
            checks["accelerator_type"] = {"http": status, "verified": status == 200}
            if status != 200:
                blockers.append("gcp_accelerator_type_preflight_failed")
        region = zone.rsplit("-", 1)[0]
        status, region_payload = self._call("GET", f"/projects/{project}/regions/{region}", timeout=45)
        quota_rows = region_payload.get("quotas") if isinstance(region_payload, Mapping) else None
        quota_metric = self._config()["gpu_quota_metric"]
        quota_row = next(
            (dict(row) for row in (quota_rows or []) if isinstance(row, Mapping) and row.get("metric") == quota_metric),
            {},
        )
        quota_limit = _positive_float(quota_row.get("limit"))
        quota_usage = float(quota_row.get("usage") or 0) if quota_row else 0.0
        required_gpu_count = float(req.get("gpu_quota_units") or 1.0)
        quota_verified = bool(
            status == 200
            and isinstance(quota_rows, list)
            and quota_limit is not None
            and quota_usage + required_gpu_count <= quota_limit
        )
        quota_source = "compute_region"
        service_usage_http = None
        if not quota_verified:
            metric_id = str(quota_metric)
            if not metric_id.startswith("compute.googleapis.com/"):
                metric_id = "compute.googleapis.com/" + metric_id.lower()
            encoded_metric = metric_id.replace("/", "%2F")
            service_usage_http, metric_payload = self._service_usage_call(
                f"/projects/{project}/services/compute.googleapis.com/"
                f"consumerQuotaMetrics/{encoded_metric}?view=FULL"
            )
            candidates: list[dict[str, Any]] = []
            for limit in metric_payload.get("consumerQuotaLimits") or []:
                if not isinstance(limit, Mapping):
                    continue
                for bucket in limit.get("quotaBuckets") or []:
                    if isinstance(bucket, Mapping):
                        candidates.append(dict(bucket))
            specific = next(
                (
                    row
                    for row in candidates
                    if _mapping(row.get("dimensions")).get("region") == region
                ),
                {},
            )
            fallback = next((row for row in candidates if not row.get("dimensions")), {})
            selected = specific or fallback
            service_limit = _positive_float(selected.get("effectiveLimit"))
            if service_usage_http == 200 and service_limit is not None:
                quota_limit = service_limit
                quota_usage = 0.0
                quota_verified = required_gpu_count <= service_limit
                quota_source = "service_usage"
        checks["regional_quota"] = {
            "http": service_usage_http if quota_source == "service_usage" else status,
            "verified": quota_verified,
            "metric": quota_metric,
            "limit": quota_limit,
            "usage": quota_usage if quota_row else None,
            "required_gpu_count": required_gpu_count,
            "quota_row_count": len(quota_rows or []),
            "source": quota_source,
        }
        if not quota_verified:
            blockers.append("gcp_gpu_quota_unverified")
        return {
            "status": "available" if not blockers else "blocked",
            "provider": self.name,
            "project": project,
            "zone": zone,
            "checks": checks,
            "quota_verified": quota_verified,
            "capacity_reservation_proven": False,
            "blockers": blockers,
            "api_confirmed": not blockers,
            "raw_provider_response_recorded": False,
        }

    def launch(self, job_dir: Path, request: dict, *, cold: bool = False, allow_cold_fallback: bool = True) -> dict:
        blockers = [
            *_string_list(request.get("configuration_blockers")),
            *_render_prelaunch_guard_blockers(request, provider_name="gcp"),
        ]
        if blockers:
            return {"status": "blocked", "blockers": list(dict.fromkeys(blockers)), "allocation_created": False}
        preflight = self.capacity_preflight(request)
        if preflight.get("status") != "available":
            return {"status": "blocked", "blockers": preflight.get("blockers") or ["gcp_preflight_failed"], "allocation_created": False, "preflight": preflight}
        project, zone, name = request["project"], request["zone"], request["instance_name"]
        request_id = str(request.get("idempotency_request_id") or "")
        status, response = self._call("POST", f"/projects/{project}/zones/{zone}/instances?requestId={request_id}", _mapping(request.get("instance_body")))
        if status in {200, 201} and response.get("name"):
            record = _record_started_id(Path(job_dir) / "started_gcp_instance_name.txt", name)
            return {"status": "launched", "instance_id": name, "mode": "gcp_compute_engine", "operation_name": response.get("name"), "started_id_record": record}
        if status == 0 or status >= 500 or 200 <= status < 300:
            return {"status": "blocked", "blockers": ["gcp_create_outcome_ambiguous"], "allocation_outcome_ambiguous": True, "http": status}
        return {"status": "blocked", "blockers": [f"gcp_instance_create_http_{status}"], "allocation_created": False, "http": status}

    def inspect(self, instance_id: str) -> dict:
        if not _NAME_RE.fullmatch(str(instance_id)):
            return {"status": "unavailable", "instance_id": instance_id, "reason": "gcp_instance_name_invalid"}
        config = self._config()
        status, body = self._call("GET", f"/projects/{config['project']}/zones/{config['zone']}/instances/{instance_id}", timeout=45)
        return {"status": "observed" if status == 200 else "unavailable", "http": status, "instance_id": instance_id, "instance_status": body.get("status"), "raw_provider_response_recorded": False}

    def billable_inventory(self, *, name_prefix: str) -> dict:
        config = self._config()
        if not config["project"] or not config["zone"]:
            return {"status": "blocked", "provider": self.name, "name_prefix": name_prefix, "live_resource_count": None, "resources": [], "api_confirmed": False, "blockers": ["gcp_inventory_scope_unconfigured"]}
        status, body = self._call("GET", f"/projects/{config['project']}/zones/{config['zone']}/instances", timeout=60)
        rows = body.get("items", []) if isinstance(body, Mapping) else None
        if status != 200 or not isinstance(rows, list):
            return {"status": "blocked", "provider": self.name, "name_prefix": name_prefix, "live_resource_count": None, "resources": [], "api_confirmed": False, "blockers": ["gcp_billable_inventory_failed"], "http": status}
        resources = [{"instance_id": str(row.get("name") or ""), "name": row.get("name"), "status": row.get("status"), "machine_type": str(row.get("machineType") or "").rsplit("/", 1)[-1], "zone": config["zone"], "created_at": row.get("creationTimestamp"), "cost_per_hour": config["configured_hourly_rate_usd"]} for row in rows if isinstance(row, Mapping) and str(row.get("name") or "").startswith(name_prefix) and row.get("status") != "TERMINATED"]
        return {"status": "observed", "provider": self.name, "name_prefix": name_prefix, "live_resource_count": len(resources), "resources": resources, "api_confirmed": True, "http": status, "raw_provider_response_recorded": False}

    def stop(self, instance_id: str) -> dict:
        config = self._config()
        status, _ = self._call("POST", f"/projects/{config['project']}/zones/{config['zone']}/instances/{instance_id}/stop", {})
        return {"status": "stopped" if status in {200, 201} else "stop_failed", "http": status, "warning": "stopped Compute Engine disks can continue billing; use terminate()"}

    def terminate(self, instance_id: str) -> dict:
        if not _NAME_RE.fullmatch(str(instance_id)):
            return {"status": "terminate_failed", "reason": "gcp_instance_name_invalid"}
        config = self._config()
        status, _ = self._call("DELETE", f"/projects/{config['project']}/zones/{config['zone']}/instances/{instance_id}")
        if status in {200, 204, 404}:
            return {"status": "terminated", "http": status, "already_gone": status == 404}
        return {"status": "terminate_failed", "http": status}


class AWSRenderProvider(GpuRenderProvider):
    """EC2 GPU VM adapter using the standard boto3 credential chain."""

    name = "aws"
    _G_QUOTA_CODE = "L-DB2E81BA"
    _P_QUOTA_CODE = "L-417A185B"

    def _config(self) -> dict[str, Any]:
        return {
            "region": _env("BLUEPRINT_AWS_REGION") or _env("AWS_REGION") or _env("AWS_DEFAULT_REGION"),
            "account_id": _env("BLUEPRINT_AWS_ACCOUNT_ID"),
            "instance_type": _env("BLUEPRINT_AWS_INSTANCE_TYPE"),
            "ami_id": _env("BLUEPRINT_AWS_AMI_ID"),
            "subnet_id": _env("BLUEPRINT_AWS_SUBNET_ID"),
            "security_group_ids": _csv("BLUEPRINT_AWS_SECURITY_GROUP_IDS"),
            "iam_instance_profile_arn": _env("BLUEPRINT_AWS_IAM_INSTANCE_PROFILE_ARN"),
            "key_name": _env("BLUEPRINT_AWS_KEY_NAME"),
            "boot_disk_gb": _positive_int(_env("BLUEPRINT_AWS_BOOT_DISK_GB")) or 200,
            "max_hourly_rate_usd": _positive_float(_env("BLUEPRINT_AWS_MAX_HOURLY_RATE_USD")),
            "configured_hourly_rate_usd": _positive_float(_env("BLUEPRINT_AWS_HOURLY_RATE_USD")),
            "registry_auth": _env("BLUEPRINT_AWS_REGISTRY_AUTH") or "public",
            "registry_host": _env("BLUEPRINT_AWS_REGISTRY_HOST"),
        }

    def _session(self) -> Any:
        import boto3

        return boto3.Session(profile_name=_env("AWS_PROFILE") or None, region_name=self._config()["region"] or None)

    def _ec2(self) -> Any:
        return self._session().client("ec2")

    def _service_quotas(self) -> Any:
        return self._session().client("service-quotas")

    def _iam(self) -> Any:
        return self._session().client("iam")

    def _sts(self) -> Any:
        return self._session().client("sts")

    def available(self) -> dict:
        config = self._config()
        missing = _required_config(config, ("account_id", "region", "instance_type", "ami_id", "subnet_id", "iam_instance_profile_arn"), "aws")
        if not config["security_group_ids"]:
            missing.append("aws_security_group_ids_missing")
        credential_error = None
        try:
            credentials = self._session().get_credentials()
            if credentials is None:
                missing.append("aws_credentials_missing")
        except Exception as exc:  # noqa: BLE001
            credential_error = type(exc).__name__
            missing.append("aws_credentials_missing")
        return {"provider": self.name, "available": not missing, "reason": missing[0] if missing else None, "blockers": missing, "region": config["region"] or None, "credential_chain": "boto3_standard", "credential_error_type": credential_error, "raw_secret_values_recorded": False}

    def build_request(self, spec: RenderLaunchSpec, job_dir: Path) -> dict:
        config = self._config()
        blockers = _required_config(config, ("account_id", "region", "instance_type", "ami_id", "subnet_id", "iam_instance_profile_arn"), "aws")
        if not config["security_group_ids"]:
            blockers.append("aws_security_group_ids_missing")
        if config["registry_auth"] not in {"public", "aws_ecr"}:
            blockers.append("aws_registry_auth_invalid")
        if config["registry_auth"] == "aws_ecr" and not config["registry_host"]:
            blockers.append("aws_registry_host_missing")
        if config["configured_hourly_rate_usd"] is None:
            blockers.append("aws_hourly_rate_unconfigured")
        if config["max_hourly_rate_usd"] is None:
            blockers.append("aws_max_hourly_rate_unconfigured")
        elif config["configured_hourly_rate_usd"] is not None and config["configured_hourly_rate_usd"] > config["max_hourly_rate_usd"]:
            blockers.append("aws_hourly_rate_exceeds_cap")
        name = spec.name[:255]
        body: dict[str, Any] = {
            "ImageId": config["ami_id"],
            "InstanceType": config["instance_type"],
            "MinCount": 1,
            "MaxCount": 1,
            "SubnetId": config["subnet_id"],
            "SecurityGroupIds": config["security_group_ids"],
            "IamInstanceProfile": {"Arn": config["iam_instance_profile_arn"]},
            "UserData": _worker_cloud_init(spec, provider="aws", registry_auth=config["registry_auth"], registry_host=config["registry_host"], aws_region=config["region"]),
            "BlockDeviceMappings": [{"DeviceName": "/dev/sda1", "Ebs": {"VolumeSize": max(spec.container_disk_gb, int(config["boot_disk_gb"])), "VolumeType": "gp3", "DeleteOnTermination": True, "Encrypted": True}}],
            "TagSpecifications": [{"ResourceType": "instance", "Tags": [{"Key": "Name", "Value": name}, {"Key": "blueprint-managed", "Value": "true"}, {"Key": "blueprint-name-prefix", "Value": name[:128]}]}],
            "MetadataOptions": {"HttpTokens": "required", "HttpEndpoint": "enabled", "HttpPutResponseHopLimit": 1},
            "InstanceInitiatedShutdownBehavior": "terminate",
            "ClientToken": str(
                uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    f"aws://{config['account_id']}/{config['region']}/{name}",
                )
            ),
        }
        if config["key_name"]:
            body["KeyName"] = config["key_name"]
        return {"provider": self.name, "account_id": config["account_id"], "region": config["region"], "instance_name": name, "run_instances": body, "configured_hourly_rate_usd": config["configured_hourly_rate_usd"], "max_hourly_rate_usd": config["max_hourly_rate_usd"], "registry_auth": config["registry_auth"], "configuration_blockers": blockers}

    def capacity_preflight(self, request: Mapping[str, Any] | None = None) -> dict:
        req = _mapping(request)
        blockers = list(_string_list(req.get("configuration_blockers")))
        body = _mapping(req.get("run_instances"))
        instance_type = str(body.get("InstanceType") or self._config()["instance_type"])
        if blockers:
            return {"status": "blocked", "provider": self.name, "blockers": blockers, "api_confirmed": False}
        checks: dict[str, Any] = {}
        try:
            ec2 = self._ec2()
            caller_account = str(self._sts().get_caller_identity().get("Account") or "")
            checks["account"] = caller_account == str(req.get("account_id") or "")
            if not checks["account"]:
                blockers.append("aws_account_id_mismatch")
            types = ec2.describe_instance_types(InstanceTypes=[instance_type]).get("InstanceTypes", [])
            offerings = ec2.describe_instance_type_offerings(LocationType="region", Filters=[{"Name": "instance-type", "Values": [instance_type]}]).get("InstanceTypeOfferings", [])
            image = ec2.describe_images(ImageIds=[body["ImageId"]]).get("Images", [])
            subnets = ec2.describe_subnets(SubnetIds=[body["SubnetId"]]).get("Subnets", [])
            groups = ec2.describe_security_groups(GroupIds=list(body["SecurityGroupIds"])).get("SecurityGroups", [])
            profile_arn = str(_mapping(body.get("IamInstanceProfile")).get("Arn") or "")
            profile_name = profile_arn.rsplit("/", 1)[-1]
            profile = self._iam().get_instance_profile(InstanceProfileName=profile_name).get("InstanceProfile", {})
            checks.update({"instance_type": bool(types), "regional_offering": bool(offerings), "ami": bool(image), "subnet": bool(subnets), "security_groups": len(groups) == len(body["SecurityGroupIds"]), "iam_instance_profile": bool(profile) and profile.get("Arn") == profile_arn})
            for key, passed in list(checks.items()):
                if key == "account":
                    continue
                if not passed:
                    blockers.append(f"aws_{key}_preflight_failed")
            family = instance_type.split(".", 1)[0].lower()
            quota_code = self._G_QUOTA_CODE if family.startswith("g") else self._P_QUOTA_CODE if family.startswith("p") else None
            quota_value: float | None = None
            required_vcpus = 0
            if quota_code is None:
                blockers.append("aws_gpu_instance_family_quota_mapping_missing")
                quota = None
            else:
                quota = self._service_quotas().get_service_quota(ServiceCode="ec2", QuotaCode=quota_code).get("Quota", {})
                required_vcpus = int(_mapping(types[0].get("VCpuInfo")).get("DefaultVCpus") or 0) if types else 0
                quota_value = _positive_float(quota.get("Value"))
                if quota_value is None or required_vcpus <= 0 or quota_value < required_vcpus:
                    blockers.append("aws_gpu_quota_unverified")
            checks["quota"] = {
                "quota_code": quota_code,
                "value": quota.get("Value") if quota else None,
                "required_vcpus": required_vcpus if quota_code else None,
                "verified": bool(quota is not None and quota_value is not None and required_vcpus > 0 and quota_value >= required_vcpus),
            }
        except Exception as exc:  # noqa: BLE001
            blockers.append("aws_capacity_preflight_api_failed")
            checks["error_type"] = type(exc).__name__
        return {"status": "available" if not blockers else "blocked", "provider": self.name, "region": req.get("region") or self._config()["region"], "checks": checks, "quota_verified": bool(_mapping(checks.get("quota")).get("verified")), "capacity_reservation_proven": False, "blockers": list(dict.fromkeys(blockers)), "api_confirmed": not blockers, "raw_provider_response_recorded": False}

    def launch(self, job_dir: Path, request: dict, *, cold: bool = False, allow_cold_fallback: bool = True) -> dict:
        blockers = [*_string_list(request.get("configuration_blockers")), *_render_prelaunch_guard_blockers(request, provider_name="aws")]
        if blockers:
            return {"status": "blocked", "blockers": list(dict.fromkeys(blockers)), "allocation_created": False}
        preflight = self.capacity_preflight(request)
        if preflight.get("status") != "available":
            return {"status": "blocked", "blockers": preflight.get("blockers") or ["aws_preflight_failed"], "allocation_created": False, "preflight": preflight}
        try:
            response = self._ec2().run_instances(**_mapping(request.get("run_instances")))
            instances = response.get("Instances", [])
            instance_id = instances[0].get("InstanceId") if instances and isinstance(instances[0], Mapping) else None
            if isinstance(instance_id, str) and _AWS_ID_RE.fullmatch(instance_id):
                record = _record_started_id(Path(job_dir) / "started_aws_instance_id.txt", instance_id)
                return {"status": "launched", "instance_id": instance_id, "mode": "aws_ec2", "started_id_record": record}
            return {"status": "blocked", "blockers": ["aws_create_outcome_ambiguous"], "allocation_outcome_ambiguous": True}
        except Exception as exc:  # noqa: BLE001
            code = str(getattr(exc, "response", {}).get("Error", {}).get("Code", ""))
            definitive = code in {"InsufficientInstanceCapacity", "InstanceLimitExceeded", "VcpuLimitExceeded", "InvalidAMIID.NotFound", "InvalidParameterValue", "UnauthorizedOperation"}
            return {"status": "blocked", "blockers": [f"aws_instance_create_{code or 'outcome_ambiguous'}"], "allocation_created": False if definitive else None, "allocation_outcome_ambiguous": not definitive, "error_type": type(exc).__name__}

    def _describe(self, **kwargs: Any) -> list[dict[str, Any]]:
        response = self._ec2().describe_instances(**kwargs)
        return [dict(instance) for reservation in response.get("Reservations", []) if isinstance(reservation, Mapping) for instance in reservation.get("Instances", []) if isinstance(instance, Mapping)]

    def inspect(self, instance_id: str) -> dict:
        if not _AWS_ID_RE.fullmatch(str(instance_id)):
            return {"status": "unavailable", "instance_id": instance_id, "reason": "aws_instance_id_invalid"}
        try:
            rows = self._describe(InstanceIds=[instance_id])
            row = rows[0] if rows else {}
            return {"status": "observed" if row else "unavailable", "instance_id": instance_id, "instance_status": _mapping(row.get("State")).get("Name"), "instance_type": row.get("InstanceType"), "raw_provider_response_recorded": False}
        except Exception as exc:  # noqa: BLE001
            return {"status": "unavailable", "instance_id": instance_id, "error_type": type(exc).__name__}

    def billable_inventory(self, *, name_prefix: str) -> dict:
        try:
            rows = self._describe(Filters=[{"Name": "tag:blueprint-managed", "Values": ["true"]}, {"Name": "instance-state-name", "Values": ["pending", "running", "stopping", "stopped"]}])
        except Exception as exc:  # noqa: BLE001
            return {"status": "blocked", "provider": self.name, "name_prefix": name_prefix, "live_resource_count": None, "resources": [], "api_confirmed": False, "blockers": ["aws_billable_inventory_failed"], "error_type": type(exc).__name__}
        resources = []
        for row in rows:
            tags = {str(tag.get("Key")): str(tag.get("Value")) for tag in row.get("Tags", []) if isinstance(tag, Mapping)}
            name = tags.get("Name", "")
            if name.startswith(name_prefix):
                launch_time = row.get("LaunchTime")
                isoformat = getattr(launch_time, "isoformat", None)
                created_at = isoformat() if callable(isoformat) else launch_time
                resources.append({"instance_id": row.get("InstanceId"), "name": name, "status": _mapping(row.get("State")).get("Name"), "instance_type": row.get("InstanceType"), "region": self._config()["region"], "created_at": created_at, "cost_per_hour": self._config()["configured_hourly_rate_usd"]})
        return {"status": "observed", "provider": self.name, "name_prefix": name_prefix, "live_resource_count": len(resources), "resources": resources, "api_confirmed": True, "raw_provider_response_recorded": False}

    def stop(self, instance_id: str) -> dict:
        try:
            self._ec2().stop_instances(InstanceIds=[instance_id])
            return {"status": "stopped", "warning": "stopped EC2 EBS volumes continue billing; use terminate()"}
        except Exception as exc:  # noqa: BLE001
            return {"status": "stop_failed", "error_type": type(exc).__name__}

    def terminate(self, instance_id: str) -> dict:
        if not _AWS_ID_RE.fullmatch(str(instance_id)):
            return {"status": "terminate_failed", "reason": "aws_instance_id_invalid"}
        try:
            self._ec2().terminate_instances(InstanceIds=[instance_id])
            return {"status": "terminated", "provider_absence_verified": False, "verification_required": "billable_inventory"}
        except Exception as exc:  # noqa: BLE001
            return {"status": "terminate_failed", "error_type": type(exc).__name__}
