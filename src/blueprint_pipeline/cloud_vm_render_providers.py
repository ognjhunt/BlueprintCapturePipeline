"""GCP Compute Engine GPU render provider and retired AWS compatibility marker.

Unlike marketplace providers, these adapters never invent account infrastructure.
Every account/project, location, VM shape, image, network, identity, registry mode,
and price is explicit configuration.  Missing or unverifiable configuration fails
closed before a mutating API call.  Provider responses are normalized to the
``GpuRenderProvider`` launch/inventory/terminate contract.
"""
from __future__ import annotations

import base64
import gzip
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
from blueprint_pipeline.paid_resource_admission import (
    PaidResourceAdmissionBlocked,
    PaidResourceAdmissionGrant,
    require_paid_resource_admission_grant,
)

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

_NAME_RE = re.compile(r"^[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?$")


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
) -> str:
    """Build a provider-neutral startup script for a pre-baked GPU host.

    Signed transport values are base64 encoded to preserve bytes, not to claim
    secrecy. Instance metadata is therefore limited to the explicitly scoped VM.

    Production customer startup deliberately performs no registry login or
    image pull.  The provider host image must already contain the exact worker
    digest and an identity marker written by the host-image build.  This keeps
    a 40+ GB transfer out of both the customer path and asynchronous cold boot.
    """
    del provider, registry_auth, registry_host
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
import json, pathlib, subprocess
argv = json.load(open('/root/blueprint_argv.json'))
worker_env = dict(
    line.rstrip('\n').split('=', 1)
    for line in open('/root/blueprint_worker.env', encoding='utf-8')
    if '=' in line
)
cmd = ['docker', 'run', '-d', '--gpus', 'all', '--name', 'blueprint-worker',
       '--env-file', '/root/blueprint_worker.env', '-v', '/workspace:/workspace',
       '--workdir', '/workspace', '--shm-size=8g', '--entrypoint', {entrypoint},
]
if worker_env.get('BLUEPRINT_GROOT_OSCAR_MODEL_CACHE'):
    model_root = pathlib.Path('/var/lib/blueprint/models')
    container_cache = pathlib.PurePosixPath(worker_env['BLUEPRINT_GROOT_OSCAR_MODEL_CACHE'])
    try: cache_relative = container_cache.relative_to('/models')
    except ValueError: raise RuntimeError('container_external_model_cache_must_be_under_models')
    if not (model_root / cache_relative / 'groot_oscar_model_cache_manifest.json').is_file():
        raise RuntimeError('host_external_model_cache_manifest_missing')
    cmd.extend(['-v', str(model_root) + ':/models:ro'])
cmd.extend([{image}, *argv])
subprocess.check_call(cmd)
PY
"""


#: Trainers that exist only as Windows executables.  Postshot ships
#: ``postshot-cli.exe`` and has no Linux build and no service API, so its arm
#: cannot run through the Linux/Docker bootstrap every other lane uses.
WINDOWS_WORKER_PLATFORM = "windows"


#: Env keys that switch the Windows host from "already baked" to "build me at
#: boot".  Both installers are operator-supplied signed URLs with pinned
#: digests: the driver is large and the Postshot MSI is licensed, so neither
#: can be fetched from an arbitrary location.
WINDOWS_DRIVER_URL_ENV = "BLUEPRINT_WINDOWS_NVIDIA_DRIVER_GET_URL"
WINDOWS_DRIVER_SHA256_ENV = "BLUEPRINT_WINDOWS_NVIDIA_DRIVER_SHA256"
WINDOWS_INSTALLER_URL_ENV = "BLUEPRINT_WINDOWS_POSTSHOT_INSTALLER_GET_URL"
WINDOWS_INSTALLER_SHA256_ENV = "BLUEPRINT_WINDOWS_POSTSHOT_INSTALLER_SHA256"
WINDOWS_PYTHON_URL_ENV = "BLUEPRINT_WINDOWS_PYTHON_EMBED_GET_URL"
WINDOWS_PYTHON_SHA256_ENV = "BLUEPRINT_WINDOWS_PYTHON_EMBED_SHA256"
WINDOWS_NUMPY_URL_ENV = "BLUEPRINT_WINDOWS_NUMPY_WHEEL_GET_URL"
WINDOWS_NUMPY_SHA256_ENV = "BLUEPRINT_WINDOWS_NUMPY_WHEEL_SHA256"


def _validated_sha256(spec: RenderLaunchSpec, name: str) -> str:
    value = str(spec.env.get(name) or "").lower().removeprefix("sha256:")
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"windows_worker_runtime_digest_invalid:{name}")
    return value


def _windows_provisioning_block(spec: RenderLaunchSpec, *, marker: str) -> str:
    """Verify a baked host, or build one at boot when none exists yet.

    A baked AMI is the better steady state: it keeps a multi-GB driver download
    and an MSI install out of every paid window.  But that image has to be
    created before it can be used, so the first run has nowhere to start.
    Install-at-boot removes that chicken-and-egg at the cost of ~30-45 minutes
    of each paid window.

    The installer digest is pinned either way.  An unverified MSI would decide
    which binary a paid instance executes.
    """

    driver_url = str(spec.env.get(WINDOWS_DRIVER_URL_ENV) or "")
    installer_url = str(spec.env.get(WINDOWS_INSTALLER_URL_ENV) or "")
    python_url = str(spec.env.get(WINDOWS_PYTHON_URL_ENV) or "")
    numpy_url = str(spec.env.get(WINDOWS_NUMPY_URL_ENV) or "")

    if not (driver_url or installer_url):
        return f"""# The baked host image must already carry the exact worker identity.
$markerPath = "C:\\blueprint\\worker-image-ref"
if (-not (Test-Path $markerPath)) {{ throw "blueprint_worker_image_marker_missing" }}
$marker = (Get-Content $markerPath -Raw).Trim()
if ($marker -ne {marker}) {{ throw "blueprint_worker_image_marker_mismatch" }}"""

    if not (driver_url and installer_url and python_url and numpy_url):
        raise ValueError("windows_worker_install_at_boot_requires_all_runtime_urls")
    for digest_name in (
        WINDOWS_DRIVER_SHA256_ENV,
        WINDOWS_INSTALLER_SHA256_ENV,
        WINDOWS_PYTHON_SHA256_ENV,
        WINDOWS_NUMPY_SHA256_ENV,
    ):
        _validated_sha256(spec, digest_name)

    return f"""# No baked image yet: provision this host in the paid window.
Invoke-WebRequest -Uri $workerEnv["{WINDOWS_DRIVER_URL_ENV}"] -OutFile C:\\work\\nvidia.exe -UseBasicParsing -TimeoutSec 900
$driverHash = (Get-FileHash C:\\work\\nvidia.exe -Algorithm SHA256).Hash.ToLower()
if ($driverHash -ne $workerEnv["{WINDOWS_DRIVER_SHA256_ENV}"].Replace("sha256:", "")) {{ throw "nvidia_driver_digest_mismatch" }}
$d = Start-Process -FilePath C:\\work\\nvidia.exe -ArgumentList "-s","-noreboot" -PassThru
Wait-Process -Id $d.Id -Timeout 1800 -ErrorAction SilentlyContinue | Out-Null
if (-not $d.HasExited) {{ Stop-Process -Id $d.Id -Force; throw "nvidia_driver_install_timeout" }}
if ($d.ExitCode -ne 0 -and $d.ExitCode -ne 3010) {{ throw "nvidia_driver_install_exit_$($d.ExitCode)" }}
if (-not (Test-Path "C:\\Windows\\System32\\nvidia-smi.exe")) {{ throw "nvidia_driver_install_failed" }}

Invoke-WebRequest -Uri $workerEnv["{WINDOWS_INSTALLER_URL_ENV}"] -OutFile C:\\work\\postshot.msi -UseBasicParsing -TimeoutSec 900
$hash = (Get-FileHash C:\\work\\postshot.msi -Algorithm SHA256).Hash.ToLower()
if ($hash -ne $workerEnv["{WINDOWS_INSTALLER_SHA256_ENV}"].Replace("sha256:", "")) {{ throw "postshot_installer_digest_mismatch" }}
$m = Start-Process -FilePath msiexec.exe -ArgumentList "/i","C:\\work\\postshot.msi","/qn","/norestart" -PassThru
Wait-Process -Id $m.Id -Timeout 900 -ErrorAction SilentlyContinue | Out-Null
if (-not $m.HasExited) {{ Stop-Process -Id $m.Id -Force; throw "msiexec_timeout" }}
if ($m.ExitCode -ne 0 -and $m.ExitCode -ne 3010) {{ throw "msiexec_exit_$($m.ExitCode)" }}
if (-not (Test-Path "$Env:ProgramFiles\\Jawset Postshot\\bin\\postshot-cli.exe")) {{ throw "postshot_cli_not_found" }}

# Bootstrap an exact Python 3.12 runtime without a package index or mutable
# resolver. The official embeddable runtime and the only binary dependency are
# content-addressed before either can execute.
Invoke-WebRequest -Uri $workerEnv["{WINDOWS_PYTHON_URL_ENV}"] -OutFile C:\\work\\python-embed.zip -UseBasicParsing -TimeoutSec 900
$pythonHash = (Get-FileHash C:\\work\\python-embed.zip -Algorithm SHA256).Hash.ToLower()
if ($pythonHash -ne $workerEnv["{WINDOWS_PYTHON_SHA256_ENV}"].Replace("sha256:", "")) {{ throw "python_embed_digest_mismatch" }}
New-Item -ItemType Directory -Force -Path C:\\blueprint\\python\\Lib\\site-packages | Out-Null
Expand-Archive -LiteralPath C:\\work\\python-embed.zip -DestinationPath C:\\blueprint\\python -Force
$pth = Get-ChildItem C:\\blueprint\\python -Filter "python*._pth"
if (@($pth).Count -ne 1) {{ throw "python_embed_path_file_invalid" }}
$pthText = (Get-Content $pth.FullName -Raw).Replace("#import site", "import site")
if ($pthText -notmatch "Lib\\\\site-packages") {{ $pthText += "`r`nLib\\site-packages`r`n" }}
Set-Content -LiteralPath $pth.FullName -Value $pthText -Encoding ASCII -NoNewline

Invoke-WebRequest -Uri $workerEnv["{WINDOWS_NUMPY_URL_ENV}"] -OutFile C:\\work\\numpy.whl -UseBasicParsing -TimeoutSec 900
$numpyHash = (Get-FileHash C:\\work\\numpy.whl -Algorithm SHA256).Hash.ToLower()
if ($numpyHash -ne $workerEnv["{WINDOWS_NUMPY_SHA256_ENV}"].Replace("sha256:", "")) {{ throw "numpy_wheel_digest_mismatch" }}
Copy-Item C:\\work\\numpy.whl C:\\work\\numpy.zip
Expand-Archive -LiteralPath C:\\work\\numpy.zip -DestinationPath C:\\blueprint\\python\\Lib\\site-packages -Force

# The exact promoted-SHA Blueprint wheel already travels inside the immutable
# input bundle. Verify the whole bundle before extracting that wheel.
$bundleUrl = $workerEnv["BLUEPRINT_RECONSTRUCTION_INPUT_BUNDLE_GET_URL"]
$bundleDigest = $workerEnv["BLUEPRINT_RECONSTRUCTION_INPUT_BUNDLE_DIGEST"].Replace("sha256:", "")
if (-not $bundleUrl -or $bundleDigest.Length -ne 64) {{ throw "bootstrap_transport_identity_missing" }}
Invoke-WebRequest -Uri $bundleUrl -OutFile C:\\work\\bootstrap-transport.zip -UseBasicParsing -TimeoutSec 900
$bundleHash = (Get-FileHash C:\\work\\bootstrap-transport.zip -Algorithm SHA256).Hash.ToLower()
if ($bundleHash -ne $bundleDigest) {{ throw "bootstrap_transport_digest_mismatch" }}
Expand-Archive -LiteralPath C:\\work\\bootstrap-transport.zip -DestinationPath C:\\work\\bootstrap-transport -Force
$workerWheel = Get-ChildItem C:\\work\\bootstrap-transport\\worker -Filter "*.whl" -File
if (@($workerWheel).Count -ne 1) {{ throw "bootstrap_worker_wheel_count_invalid" }}
Copy-Item $workerWheel.FullName C:\\work\\blueprint-worker.zip
Expand-Archive -LiteralPath C:\\work\\blueprint-worker.zip -DestinationPath C:\\blueprint\\python\\Lib\\site-packages -Force
& C:\\blueprint\\python\\python.exe -I -c "import numpy, blueprint_pipeline.windows_worker_entrypoint"
if ($LASTEXITCODE -ne 0) {{ throw "blueprint_python_runtime_import_failed" }}"""


def _windows_worker_bootstrap(spec: RenderLaunchSpec) -> str:
    """Build the PowerShell EC2 UserData for a pre-baked Windows GPU host.

    Three properties matter more than convenience here:

    * **No credential ever enters UserData.**  EC2 UserData is readable from
      the instance itself over IMDS and from the account via
      ``DescribeInstanceAttribute``, so the trainer licence is fetched at run
      time from a single signed URL and the remote object is deleted on
      acknowledgement.  Only the fetch URL crosses this boundary.
    * **The host image is already complete.**  Startup verifies the baked
      worker marker instead of installing a driver or trainer, keeping a
      multi-GB download out of the paid window.
    * **The instance ends itself.**  A local deadline plus
      ``InstanceInitiatedShutdownBehavior=terminate`` bounds spend even if the
      controller dies.  This is a backstop, never a replacement for the
      independent watchdog and provider-zero proof.
    """

    # Refuse rather than filter.  Silently dropping a credential would surface
    # later as an opaque "licence missing" failure on a paid instance; refusing
    # here fails closed while the mistake is still free to fix.
    smuggled = sorted(
        key
        for key in spec.env
        if any(
            fragment in str(key).lower()
            for fragment in ("password", "secret", "token", "private_key", "credential")
        )
    )
    if smuggled:
        raise ValueError(
            "windows_worker_bootstrap_refuses_credential_in_user_data:"
            + ",".join(smuggled)
        )

    env_gzip_b64 = base64.b64encode(
        gzip.compress(
            "\n".join(f"{key}={value}" for key, value in spec.env.items()).encode(),
            mtime=0,
        )
    ).decode()
    argv_b64 = base64.b64encode(json.dumps(list(spec.bootstrap_argv)).encode()).decode()
    marker = json.dumps(spec.image)
    deadline_seconds = int(spec.env.get("BLUEPRINT_WORKER_HARD_TTL_SECONDS") or 0)
    provision = _windows_provisioning_block(spec, marker=marker)
    return f"""<powershell>
$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force -Path C:\\work\\out | Out-Null

# Bound the paid window locally even if the controller never returns.
$deadline = {deadline_seconds}
if ($deadline -gt 0) {{
  $action = New-ScheduledTaskAction -Execute "shutdown.exe" -Argument "/s /t 0 /f"
  $trigger = New-ScheduledTaskTrigger -Once -At (Get-Date).AddSeconds($deadline)
  Register-ScheduledTask -TaskName "blueprint-hard-deadline" -Action $action `
    -Trigger $trigger -User "SYSTEM" -RunLevel Highest -Force | Out-Null
}}

$compressedEnv = [Convert]::FromBase64String("{env_gzip_b64}")
$compressedStream = New-Object IO.MemoryStream(,$compressedEnv)
$gzipStream = New-Object IO.Compression.GzipStream(
  $compressedStream, [IO.Compression.CompressionMode]::Decompress)
$environmentStream = New-Object IO.MemoryStream
$gzipStream.CopyTo($environmentStream)
[IO.File]::WriteAllBytes("C:\\work\\blueprint_worker.env", $environmentStream.ToArray())
$gzipStream.Dispose()
$compressedStream.Dispose()
$environmentStream.Dispose()
[IO.File]::WriteAllBytes("C:\\work\\blueprint_argv.json",
  [Convert]::FromBase64String("{argv_b64}"))

$workerEnv = @{{}}
Get-Content C:\\work\\blueprint_worker.env | ForEach-Object {{
  $parts = $_.Split('=', 2)
  if ($parts.Count -eq 2) {{ $workerEnv[$parts[0]] = $parts[1] }}
}}

{provision}

$env:BLUEPRINT_WORKER_ENV_FILE = "C:\\work\\blueprint_worker.env"
$env:BLUEPRINT_WORKER_ARGV_FILE = "C:\\work\\blueprint_argv.json"
& "C:\\blueprint\\python\\python.exe" -I -m blueprint_pipeline.windows_worker_entrypoint
$workerExit = $LASTEXITCODE
# The output upload is synchronous. Once it returns, terminate immediately;
# the scheduled deadline remains only the crash/hang backstop.
shutdown.exe /s /t 0 /f
exit $workerExit
</powershell>
<persist>false</persist>
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
            metric_match = re.fullmatch(
                r"(?:compute[.]googleapis[.]com/)?([a-z0-9_.-]+)",
                str(quota_metric).lower(),
            )
            if metric_match is None:
                blockers.append("gcp_gpu_quota_metric_invalid")
                metric_name = "invalid"
            else:
                metric_name = metric_match.group(1)
            # Construct the provider namespace from a strict allowlisted metric
            # name; never carry an arbitrary URL substring into the request.
            encoded_metric = "compute.googleapis.com%2F" + metric_name
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

    def launch(self, job_dir: Path, request: dict, *, cold: bool = False, allow_cold_fallback: bool = True, paid_resource_admission_grant: PaidResourceAdmissionGrant | None = None) -> dict:
        try:
            require_paid_resource_admission_grant(paid_resource_admission_grant, resource_class="gpu_render")
        except PaidResourceAdmissionBlocked as exc:
            return {"status": "blocked", "blockers": ["legacy_gpu_render_provider_launch_disabled", *exc.blockers], "allocation_created": False}
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
    """Retained import compatibility; AWS account access has been removed.

    No credential discovery, inventory, launch, or teardown is performed.
    Historical resources remain unverified rather than being reported absent.
    """

    name = "aws"

    def available(self) -> dict[str, Any]:
        return {"provider": self.name, "available": False,
                "reason": "aws_provider_integration_removed",
                "blockers": ["aws_provider_integration_removed"]}

    def billable_inventory(self, **kwargs: Any) -> dict[str, Any]:
        return {"provider": self.name, "status": "unavailable",
                "live_resource_count": None, "api_confirmed": False,
                "blockers": ["aws_provider_integration_removed"]}

    def _refuse(self, *args: Any, **kwargs: Any) -> Any:
        raise ValueError("aws_provider_integration_removed")

    build_request = capacity_preflight = launch = poll = stop = terminate = _refuse
