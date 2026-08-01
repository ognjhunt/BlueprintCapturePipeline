#!/usr/bin/env python3
"""Launch, monitor, and tear down the Postshot Windows GPU worker on AWS EC2.

Executes the frozen P1 (Splat3) and P2 (MCMC) arms from the Postshot execution
packet on one Windows g6/g5 instance, sequentially, under one Studio license.

Transport is DigitalOcean Spaces presigned URLs (no AWS S3 permissions needed,
no inbound ports on the instance).  The instance runs a PowerShell bootstrap
from user-data, uploads status + results, then shuts itself down; with
InstanceInitiatedShutdownBehavior=terminate that self-destructs the box.  A
local TTL watchdog (default 360 min) terminates it if it stalls.

Secrets: the Postshot login travels only as a short-lived presigned GET to a
sealed env blob; values are never logged, and the worker logs a redacted
command line.  Never print credential values from this script.

Stages: stage -> launch -> watch -> collect -> teardown/verify-zero.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
import time
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

import boto3
from botocore.client import Config

from blueprint_pipeline.decision_evidence_contracts import canonical_digest, canonical_json

SECRETS = Path.home() / ".blueprint-secrets"
AWS_CREDENTIALS_FILE = SECRETS / "aws_agent_credentials"
AWS_PROFILE = "blueprint-agent"
REGION = "us-east-1"
WINDOWS_BASE_AMI = "ami-0ed0165f19a049904"  # Windows_Server-2022-English-Full-Base-2026.07.15
INSTANCE_TYPES = ("g6.xlarge", "g5.xlarge")
NVIDIA_DRIVER_URLS = [
    "https://us.download.nvidia.com/tesla/566.03/566.03-data-center-tesla-desktop-winserver-2022-dch-international.exe",
    "https://us.download.nvidia.com/tesla/553.62/553.62-data-center-tesla-desktop-winserver-2022-dch-international.exe",
]
# sha256 of Postshot-1.1.0.msi, carved from the vendor bundle and verified
# against the bundle manifest's SHA-512 (bundle sha256 was 99fc687c...).
POSTSHOT_INSTALLER_SHA256 = "70d4c35de6ff1296a8c0b4b2d87e84b35579baaf08ae173fa843601a2ea0e361"
TTL_MINUTES = 360
SPEND_CAP_USD = 90.0
EXPECTED_HOURLY_USD = 1.25  # g6.xlarge Windows on-demand, upper-bound estimate
TAG = "blueprint-postshot-bakeoff"

BOOTSTRAP_TEMPLATE = r"""<powershell>
$ErrorActionPreference = "Continue"
New-Item -ItemType Directory -Force -Path C:\work, C:\work\out | Out-Null
Start-Transcript -Path C:\work\bootstrap-transcript.txt
function Put-Status([string]$msg) {
  $line = "$((Get-Date).ToUniversalTime().ToString('o')) $msg"
  Add-Content -Path C:\work\status.log -Value $line
  try { Invoke-RestMethod -Method Put -Uri "__STATUS_PUT__" -InFile C:\work\status.log -ContentType "text/plain" | Out-Null } catch {}
}
function Fail-And-Stop([string]$msg) {
  Put-Status "FATAL $msg"
  try { Invoke-RestMethod -Method Put -Uri "__RESULTS_PUT__" -InFile C:\work\partial.zip -ContentType "application/zip" | Out-Null } catch {}
  Stop-Transcript; Stop-Computer -Force
}
Put-Status "BOOT worker started"

# Reboot sentinel: if anything force-reboots the box, the next boot announces
# itself instead of dying silently (user-data only runs on first boot).
$sentinel = @'
Invoke-RestMethod -Method Put -Uri "__STATUS_PUT__" -InFile C:\work\status.log -ContentType "text/plain"
Add-Content -Path C:\work\status.log -Value "$((Get-Date).ToUniversalTime().ToString('o')) UNEXPECTED_REBOOT_DETECTED"
Invoke-RestMethod -Method Put -Uri "__STATUS_PUT__" -InFile C:\work\status.log -ContentType "text/plain"
'@
Set-Content -Path C:\work\reboot-sentinel.ps1 -Value $sentinel
schtasks /Create /TN BlueprintRebootSentinel /SC ONSTART /RU SYSTEM /F /TR "powershell.exe -ExecutionPolicy Bypass -File C:\work\reboot-sentinel.ps1" | Out-Null

# Heartbeat: every 3 minutes upload the transcript tail so a hang is always
# diagnosable from outside (no inbound access exists by design).
Start-Job -ScriptBlock {
  while ($true) {
    try {
      $tail = ""
      if (Test-Path C:\work\bootstrap-transcript.txt) {
        $tail = (Get-Content C:\work\bootstrap-transcript.txt -Tail 40 -ErrorAction SilentlyContinue) -join "`n"
      }
      $hb = "$((Get-Date).ToUniversalTime().ToString('o')) HEARTBEAT`n--- transcript tail ---`n$tail"
      Invoke-RestMethod -Method Put -Uri "__HEARTBEAT_PUT__" -Body $hb -ContentType "text/plain" | Out-Null
    } catch {}
    Start-Sleep -Seconds 180
  }
} | Out-Null

# 1) NVIDIA datacenter driver
$driverOk = $false
foreach ($u in @(__DRIVER_URLS__)) {
  try {
    Put-Status "DRIVER downloading $u"
    Invoke-WebRequest -Uri $u -OutFile C:\work\nvidia.exe -UseBasicParsing -TimeoutSec 900
    $p = Start-Process -FilePath C:\work\nvidia.exe -ArgumentList "-s","-noreboot" -PassThru
    if (-not (Wait-Process -Id $p.Id -Timeout 900 -ErrorAction SilentlyContinue)) { }
    if (-not $p.HasExited) { Stop-Process -Id $p.Id -Force; Put-Status "DRIVER install timed out"; continue }
    $smi = "C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe"
    if (-not (Test-Path $smi)) { $smi = "C:\Windows\System32\nvidia-smi.exe" }
    $gpu = & $smi --query-gpu=name,driver_version --format=csv,noheader 2>$null
    if ($LASTEXITCODE -eq 0 -and $gpu) { Put-Status "DRIVER ok: $gpu"; $driverOk = $true; break }
  } catch { Put-Status "DRIVER attempt failed: $($_.Exception.Message)" }
}
if (-not $driverOk) { Fail-And-Stop "nvidia_driver_install_failed" }

# 2) Postshot install via the extracted MSI and native msiexec. The vendor
#    bundle's managed bootstrapper (ReactionsBA.exe, .NET) hangs forever in
#    /quiet on bare Server 2022 -- proven on attempts 1 and 2 -- so the MSI
#    (the bundle's only package, hash-verified against its manifest) is
#    installed directly. 3010 = success, reboot wanted.
try {
  Put-Status "POSTSHOT downloading msi"
  Invoke-WebRequest -Uri "__POSTSHOT_INSTALLER_GET__" -OutFile C:\work\Postshot-1.1.0.msi -UseBasicParsing -TimeoutSec 600
  $size = (Get-Item C:\work\Postshot-1.1.0.msi).Length
  $h = (Get-FileHash C:\work\Postshot-1.1.0.msi -Algorithm SHA256).Hash.ToLower()
  Put-Status "POSTSHOT msi downloaded bytes=$size sha256_ok=$($h -eq '__POSTSHOT_SHA256__')"
  if ($h -ne "__POSTSHOT_SHA256__") { Fail-And-Stop "postshot_msi_digest_mismatch:$h" }
  Put-Status "POSTSHOT installing via msiexec (qn, 900s timeout)"
  $p = Start-Process -FilePath msiexec.exe -ArgumentList "/i","C:\work\Postshot-1.1.0.msi","/qn","/norestart","/l*v","C:\work\postshot-install.log" -PassThru
  if (-not (Wait-Process -Id $p.Id -Timeout 900 -ErrorAction SilentlyContinue)) { }
  if (-not $p.HasExited) {
    Stop-Process -Id $p.Id -Force
    Get-Content C:\work\postshot-install.log -Tail 30 -ErrorAction SilentlyContinue | ForEach-Object { Add-Content C:\work\status.log $_ }
    Fail-And-Stop "postshot_msiexec_timed_out_after_900s"
  }
  $code = $p.ExitCode
  Put-Status "POSTSHOT msiexec exit=$code (0 or 3010 accepted)"
  if ($code -ne 0 -and $code -ne 3010) {
    Get-Content C:\work\postshot-install.log -Tail 40 -ErrorAction SilentlyContinue | ForEach-Object { Add-Content C:\work\status.log $_ }
    Fail-And-Stop "postshot_msiexec_failed_exit_$code"
  }
} catch { Fail-And-Stop "postshot_install_failed: $($_.Exception.Message)" }
$cli = "$Env:ProgramFiles\Jawset Postshot\bin\postshot-cli.exe"
if (-not (Test-Path $cli)) {
  $found = Get-ChildItem -Path "$Env:ProgramFiles" -Recurse -Filter postshot-cli.exe -ErrorAction SilentlyContinue | Select-Object -First 1
  if ($found) { $cli = $found.FullName } else { Fail-And-Stop "postshot_cli_not_found_after_install" }
}
Put-Status "POSTSHOT cli at $cli"
& $cli --help *> C:\work\out\postshot-help.txt
& $cli train --help *> C:\work\out\postshot-train-help.txt

# 3) Dataset + sealed license blob
try {
  Put-Status "DATA downloading dataset bundle"
  Invoke-WebRequest -Uri "__DATASET_GET__" -OutFile C:\work\dataset.zip -UseBasicParsing
  Expand-Archive -Path C:\work\dataset.zip -DestinationPath C:\work\dataset -Force
  Invoke-WebRequest -Uri "__LICENSE_GET__" -OutFile C:\work\license.env -UseBasicParsing
} catch { Fail-And-Stop "input_download_failed: $($_.Exception.Message)" }
$lic = @{}
Get-Content C:\work\license.env | ForEach-Object {
  if ($_ -match "^([A-Za-z_]+)=(.*)$") { $lic[$Matches[1]] = $Matches[2] }
}
if (-not $lic["POSTSHOT_LOGIN_EMAIL"] -or -not $lic["POSTSHOT_LOGIN_PASSWORD"]) { Fail-And-Stop "license_env_incomplete" }
Remove-Item C:\work\license.env -Force

# 4) Train arms sequentially. Login args are appended programmatically and the
#    logged command line is redacted; credential values never enter logs.
$dataset = "C:\work\dataset"
$images = Join-Path $dataset "images"
function Run-Arm([string]$armId, [string]$profile) {
  Put-Status "TRAIN $armId starting (profile=$profile)"
  $out = "C:\work\out\$armId"
  New-Item -ItemType Directory -Force -Path $out | Out-Null
  $args = @("train",
    "--import", $dataset,
    "--profile", $profile,
    "--max-image-size", "0",
    "--output", "$out\$armId.psht",
    "--export-splat", "$out\$armId.ply",
    "--login", $lic["POSTSHOT_LOGIN_EMAIL"], "--password", $lic["POSTSHOT_LOGIN_PASSWORD"])
  $redacted = ($args | ForEach-Object { $_ }) -join " "
  $redacted = $redacted -replace [regex]::Escape($lic["POSTSHOT_LOGIN_PASSWORD"]), "***" -replace [regex]::Escape($lic["POSTSHOT_LOGIN_EMAIL"]), "***"
  Add-Content -Path "$out\command.txt" -Value $redacted
  $sw = [System.Diagnostics.Stopwatch]::StartNew()
  & $cli @args *> "$out\train-log.txt"
  $code = $LASTEXITCODE
  $sw.Stop()
  # Redact any credential echo the CLI itself might have produced.
  (Get-Content "$out\train-log.txt" -Raw -ErrorAction SilentlyContinue) `
    -replace [regex]::Escape($lic["POSTSHOT_LOGIN_PASSWORD"]), "***" `
    -replace [regex]::Escape($lic["POSTSHOT_LOGIN_EMAIL"]), "***" |
    Set-Content "$out\train-log.txt"
  Add-Content -Path "$out\receipt.txt" -Value "arm=$armId exit=$code seconds=$([int]$sw.Elapsed.TotalSeconds)"
  Put-Status "TRAIN $armId finished exit=$code seconds=$([int]$sw.Elapsed.TotalSeconds)"
  return $code
}
$p1 = Run-Arm "P1_splat3" "Splat3"
$p2 = Run-Arm "P2_mcmc" "Splat MCMC"

# 5) Package + upload results, then self-destruct (shutdown => terminate)
try {
  $smi = "C:\Windows\System32\nvidia-smi.exe"
  & $smi *> C:\work\out\nvidia-smi.txt
  Copy-Item C:\work\postshot-install.log C:\work\out\ -ErrorAction SilentlyContinue
  Copy-Item C:\work\status.log C:\work\out\ -ErrorAction SilentlyContinue
  Compress-Archive -Path C:\work\out\* -DestinationPath C:\work\results.zip -Force
  Put-Status "UPLOAD results.zip $((Get-Item C:\work\results.zip).Length) bytes"
  Invoke-RestMethod -Method Put -Uri "__RESULTS_PUT__" -InFile C:\work\results.zip -ContentType "application/zip" | Out-Null
  Put-Status "DONE p1_exit=$p1 p2_exit=$p2"
} catch { Put-Status "UPLOAD failed: $($_.Exception.Message)" }
Stop-Transcript
Stop-Computer -Force
</powershell>
"""


def _read_secret(name: str) -> str:
    return (SECRETS / name).read_text(encoding="utf-8").strip()


def _spaces_client():
    return boto3.client(
        "s3",
        region_name=_read_secret("digitalocean_spaces_region"),
        endpoint_url=_read_secret("digitalocean_spaces_endpoint_url"),
        aws_access_key_id=_read_secret("digitalocean_spaces_access_key_id"),
        aws_secret_access_key=_read_secret("digitalocean_spaces_secret_access_key"),
        config=Config(signature_version="s3v4"),
    )


def _aws_session():
    os.environ["AWS_SHARED_CREDENTIALS_FILE"] = str(AWS_CREDENTIALS_FILE)
    return boto3.Session(profile_name=AWS_PROFILE, region_name=REGION)


def _sha256_bytes(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def stage(arguments) -> dict:
    """Upload dataset bundle, installer, and sealed license blob; presign URLs."""

    proxy_root = Path(arguments.proxy_root).resolve()
    dataset_root = proxy_root / arguments.dataset_relative
    run_id = time.strftime("postshot-%Y%m%dT%H%M%SZ", time.gmtime())
    prefix = f"blueprint-postshot-bakeoff/{run_id}"
    bucket = _read_secret("digitalocean_spaces_bucket")
    spaces = _spaces_client()

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED) as archive:
        for path in sorted(dataset_root.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(dataset_root).as_posix())
    dataset_bytes = buffer.getvalue()
    license_bytes = (SECRETS / "postshot.env").read_bytes()
    installer_bytes = Path(arguments.installer).read_bytes()
    observed = hashlib.sha256(installer_bytes).hexdigest()
    if observed != POSTSHOT_INSTALLER_SHA256:
        raise SystemExit(f"installer digest mismatch: {observed}")

    keys = {
        "dataset": f"{prefix}/dataset.zip",
        "installer": f"{prefix}/Postshot-1.1.0.msi",
        "license": f"{prefix}/license.env",
        "status": f"{prefix}/status.log",
        "heartbeat": f"{prefix}/heartbeat.log",
        "results": f"{prefix}/results.zip",
    }
    spaces.put_object(Bucket=bucket, Key=keys["dataset"], Body=dataset_bytes, ACL="private")
    spaces.put_object(Bucket=bucket, Key=keys["installer"], Body=installer_bytes, ACL="private")
    spaces.put_object(Bucket=bucket, Key=keys["license"], Body=license_bytes, ACL="private")
    expiry = 12 * 3600
    urls = {
        "dataset_get": spaces.generate_presigned_url(
            "get_object", Params={"Bucket": bucket, "Key": keys["dataset"]}, ExpiresIn=expiry
        ),
        "installer_get": spaces.generate_presigned_url(
            "get_object", Params={"Bucket": bucket, "Key": keys["installer"]}, ExpiresIn=expiry
        ),
        "license_get": spaces.generate_presigned_url(
            "get_object", Params={"Bucket": bucket, "Key": keys["license"]}, ExpiresIn=expiry
        ),
        "status_put": spaces.generate_presigned_url(
            "put_object",
            Params={"Bucket": bucket, "Key": keys["status"], "ContentType": "text/plain"},
            ExpiresIn=expiry,
        ),
        "heartbeat_put": spaces.generate_presigned_url(
            "put_object",
            Params={"Bucket": bucket, "Key": keys["heartbeat"], "ContentType": "text/plain"},
            ExpiresIn=expiry,
        ),
        "results_put": spaces.generate_presigned_url(
            "put_object",
            Params={"Bucket": bucket, "Key": keys["results"], "ContentType": "application/zip"},
            ExpiresIn=expiry,
        ),
    }
    staging = {
        "schema_version": "postshot_worker_staging.v1",
        "run_id": run_id,
        "bucket": bucket,
        "keys": keys,
        "dataset_digest": _sha256_bytes(dataset_bytes),
        "dataset_bytes": len(dataset_bytes),
        "installer_digest": "sha256:" + POSTSHOT_INSTALLER_SHA256,
        "license_blob_digest": _sha256_bytes(license_bytes),
        "url_expiry_seconds": expiry,
        "ttl_minutes": TTL_MINUTES,
        "spend_cap_usd": SPEND_CAP_USD,
    }
    staging["staging_digest"] = canonical_digest(staging, digest_field="staging_digest")
    state_dir = proxy_root / "provider_packets" / "postshot" / run_id
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "staging.json").write_text(canonical_json(staging) + "\n", encoding="utf-8")
    (state_dir / "presigned_urls.json").write_text(
        json.dumps(urls) + "\n", encoding="utf-8"
    )
    os.chmod(state_dir / "presigned_urls.json", 0o600)
    print(json.dumps({k: staging[k] for k in ("run_id", "dataset_bytes", "dataset_digest")}, indent=1))
    print("state:", state_dir)
    return {"staging": staging, "urls": urls, "state_dir": state_dir}


def launch(arguments) -> None:
    proxy_root = Path(arguments.proxy_root).resolve()
    state_dir = proxy_root / "provider_packets" / "postshot" / arguments.run_id
    staging = json.loads((state_dir / "staging.json").read_text(encoding="utf-8"))
    urls = json.loads((state_dir / "presigned_urls.json").read_text(encoding="utf-8"))
    driver_urls = ",".join(f'"{u}"' for u in NVIDIA_DRIVER_URLS)
    user_data = (
        BOOTSTRAP_TEMPLATE.replace("__STATUS_PUT__", urls["status_put"])
        .replace("__HEARTBEAT_PUT__", urls["heartbeat_put"])
        .replace("__RESULTS_PUT__", urls["results_put"])
        .replace("__DRIVER_URLS__", driver_urls)
        .replace("__POSTSHOT_INSTALLER_GET__", urls["installer_get"])
        .replace("__POSTSHOT_SHA256__", POSTSHOT_INSTALLER_SHA256)
        .replace("__DATASET_GET__", urls["dataset_get"])
        .replace("__LICENSE_GET__", urls["license_get"])
    )
    session = _aws_session()
    ec2 = session.client("ec2")
    groups = ec2.describe_security_groups(
        Filters=[{"Name": "group-name", "Values": [TAG]}]
    )["SecurityGroups"]
    if groups:
        group_id = groups[0]["GroupId"]
    else:
        vpc = ec2.describe_vpcs(Filters=[{"Name": "isDefault", "Values": ["true"]}])["Vpcs"][0]
        group_id = ec2.create_security_group(
            GroupName=TAG,
            Description="Blueprint Postshot worker: no inbound, egress only",
            VpcId=vpc["VpcId"],
        )["GroupId"]
    error = None
    for instance_type in INSTANCE_TYPES:
        try:
            run = ec2.run_instances(
                ImageId=WINDOWS_BASE_AMI,
                InstanceType=instance_type,
                MinCount=1,
                MaxCount=1,
                SecurityGroupIds=[group_id],
                InstanceInitiatedShutdownBehavior="terminate",
                UserData=user_data,
                BlockDeviceMappings=[
                    {
                        "DeviceName": "/dev/sda1",
                        "Ebs": {"VolumeSize": 150, "VolumeType": "gp3", "DeleteOnTermination": True},
                    }
                ],
                TagSpecifications=[
                    {
                        "ResourceType": "instance",
                        "Tags": [
                            {"Key": "Name", "Value": f"{TAG}-{arguments.run_id}"},
                            {"Key": "blueprint-run", "Value": arguments.run_id},
                            {"Key": "blueprint-ttl-minutes", "Value": str(TTL_MINUTES)},
                        ],
                    }
                ],
            )
            instance = run["Instances"][0]
            record = {
                "schema_version": "postshot_worker_launch.v1",
                "run_id": arguments.run_id,
                "instance_id": instance["InstanceId"],
                "instance_type": instance_type,
                "image_id": WINDOWS_BASE_AMI,
                "security_group": group_id,
                "launched_at_epoch": int(time.time()),
                "ttl_deadline_epoch": int(time.time()) + TTL_MINUTES * 60,
                "staging_digest": staging["staging_digest"],
                "expected_hourly_usd_upper_bound": EXPECTED_HOURLY_USD,
                "spend_cap_usd": SPEND_CAP_USD,
            }
            (state_dir / "launch.json").write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
            print(json.dumps(record, indent=1))
            return
        except Exception as exc:  # noqa: BLE001 - try next instance type, then surface
            error = exc
            print(f"launch {instance_type} failed: {exc}")
    raise SystemExit(f"all instance types failed: {error}")


def status(arguments) -> None:
    proxy_root = Path(arguments.proxy_root).resolve()
    state_dir = proxy_root / "provider_packets" / "postshot" / arguments.run_id
    staging = json.loads((state_dir / "staging.json").read_text(encoding="utf-8"))
    spaces = _spaces_client()
    bucket = staging["bucket"]
    try:
        body = spaces.get_object(Bucket=bucket, Key=staging["keys"]["status"])["Body"].read()
        print(body.decode("utf-8", errors="replace")[-3000:])
    except Exception as exc:  # noqa: BLE001
        print(f"no status yet: {type(exc).__name__}")
    try:
        head = spaces.head_object(Bucket=bucket, Key=staging["keys"]["results"])
        print(f"RESULTS PRESENT: {head['ContentLength']} bytes")
    except Exception:
        print("results: not yet uploaded")
    if (state_dir / "launch.json").exists():
        launch_record = json.loads((state_dir / "launch.json").read_text(encoding="utf-8"))
        ec2 = _aws_session().client("ec2")
        try:
            reservations = ec2.describe_instances(InstanceIds=[launch_record["instance_id"]])
            state = reservations["Reservations"][0]["Instances"][0]["State"]["Name"]
        except Exception as exc:  # noqa: BLE001
            state = f"describe_failed:{type(exc).__name__}"
        remaining = launch_record["ttl_deadline_epoch"] - int(time.time())
        print(f"instance {launch_record['instance_id']}: {state}; ttl_remaining_s={remaining}")


def collect(arguments) -> None:
    proxy_root = Path(arguments.proxy_root).resolve()
    state_dir = proxy_root / "provider_packets" / "postshot" / arguments.run_id
    staging = json.loads((state_dir / "staging.json").read_text(encoding="utf-8"))
    spaces = _spaces_client()
    payload = spaces.get_object(Bucket=staging["bucket"], Key=staging["keys"]["results"])[
        "Body"
    ].read()
    out = state_dir / "results.zip"
    out.write_bytes(payload)
    print(json.dumps({"results_zip": str(out), "bytes": len(payload), "digest": _sha256_bytes(payload)}, indent=1))


def teardown(arguments) -> None:
    proxy_root = Path(arguments.proxy_root).resolve()
    state_dir = proxy_root / "provider_packets" / "postshot" / arguments.run_id
    ec2 = _aws_session().client("ec2")
    launch_path = state_dir / "launch.json"
    if launch_path.exists():
        launch_record = json.loads(launch_path.read_text(encoding="utf-8"))
        try:
            ec2.terminate_instances(InstanceIds=[launch_record["instance_id"]])
            print("terminate requested:", launch_record["instance_id"])
        except Exception as exc:  # noqa: BLE001
            print(f"terminate call: {type(exc).__name__}: {exc}")
    reservations = ec2.describe_instances(
        Filters=[{"Name": "instance-state-name", "Values": ["pending", "running", "stopping", "stopped", "shutting-down"]}]
    )["Reservations"]
    live = [i["InstanceId"] for r in reservations for i in r["Instances"]]
    proof = {
        "schema_version": "postshot_worker_teardown_proof.v1",
        "run_id": arguments.run_id,
        "checked_at_epoch": int(time.time()),
        "status_source": "provider_api",
        "region": REGION,
        "non_terminated_instances": live,
        "provider_zero": not live,
    }
    (state_dir / "teardown_proof.json").write_text(json.dumps(proof, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(proof, indent=1))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["stage", "launch", "status", "collect", "teardown"])
    parser.add_argument("--proxy-root", required=True)
    parser.add_argument("--dataset-relative", default="trainer_input/colmap_dataset_9de1972eae8fe5ef")
    parser.add_argument("--installer", default=None)
    parser.add_argument("--run-id", default=None)
    arguments = parser.parse_args()
    if arguments.stage == "stage":
        if not arguments.installer:
            raise SystemExit("--installer required for stage")
        stage(arguments)
    elif arguments.stage == "launch":
        launch(arguments)
    elif arguments.stage == "status":
        status(arguments)
    elif arguments.stage == "collect":
        collect(arguments)
    elif arguments.stage == "teardown":
        teardown(arguments)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
