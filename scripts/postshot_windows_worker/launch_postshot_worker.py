#!/usr/bin/env python3
"""Safe control plane for the bounded Postshot Windows reconstruction worker.

The command surface is intentionally explicit:

``stage -> admit -> launch -> watch/status -> collect -> abort/teardown``

``inventory`` and ``reconcile`` are read-only/reporting operations.  The paid
launch path requires a digest-bound admission receipt created from the exact
authorization line; no command in this module infers paid authority.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

import boto3  # noqa: E402
from botocore.exceptions import ClientError  # noqa: E402
from botocore.client import Config  # noqa: E402

from blueprint_pipeline.common import write_json  # noqa: E402
from blueprint_pipeline.decision_evidence_contracts import canonical_digest, canonical_json  # noqa: E402
from blueprint_pipeline.postshot_worker_contracts import (  # noqa: E402
    ATTEMPT_LEDGER_SCHEMA_VERSION,
    EXTERNAL_WATCHDOG_SCHEMA_VERSION,
    PHASE_LIMITS_SECONDS,
    PULSE_INTERVAL_SECONDS,
    WatchDecision,
    assert_secret_free,
    build_attempt_ledger,
    build_deletion_receipt,
    build_external_watchdog_record,
    build_live_cost_estimate,
    build_provider_zero_proof,
    build_reconciled_cost,
    derive_phase_started_epoch,
    evaluate_canary_gate,
    evaluate_pulses,
    parse_timestamp,
    sanitize_text,
    sha256_bytes,
    sha256_file,
    validate_pulse,
    utc_now_iso,
)  # noqa: E402

SECRETS = Path.home() / ".blueprint-secrets"
AWS_CREDENTIALS_FILE = SECRETS / "aws_agent_credentials"
AWS_PROFILE = "blueprint-agent"
REGION = "us-east-1"
WINDOWS_BASE_AMI = "ami-0ed0165f19a049904"
INSTANCE_TYPES = ("g6.xlarge", "g5.xlarge")
NVIDIA_DRIVER_URLS = [
    "https://us.download.nvidia.com/tesla/566.03/566.03-data-center-tesla-desktop-winserver-2022-dch-international.exe",
    "https://us.download.nvidia.com/tesla/553.62/553.62-data-center-tesla-desktop-winserver-2022-dch-international.exe",
]
POSTSHOT_INSTALLER_SHA256 = "70d4c35de6ff1296a8c0b4b2d87e84b35579baaf08ae173fa843601a2ea0e361"
ATTEMPT_5_INCREMENTAL_CAP_USD = 15.0
ATTEMPT_5_TTL_MINUTES = 300
EXPECTED_HOURLY_USD = 1.25
EXPECTED_EBS_HOURLY_USD = 0.01
TAG = "blueprint-postshot-bakeoff"
RUN_TAG_KEY = "blueprint-run"
AUTHORIZATION_LINE = "AUTHORIZE_POSTSHOT_ATTEMPT_5 MAX_INCREMENTAL_USD=15 TTL_MINUTES=300"
RUN_ID_RE = re.compile(r"^postshot-[0-9]{8}T[0-9]{6}Z$")

# Per-object expiries sized to each object's last legitimate use, measured
# from stage time.  Inputs are consumed by ~launch + boot(10m) + driver(15m)
# + install(10m) + retrieval(15m); two hours absorbs launch latency without
# reviving the old everything-lives-12h posture.  Telemetry/result PUT URLs
# are re-signed for the same key on every upload, and the worker keeps using
# them until the 300-minute TTL plus collection, so they must outlive TTL.
URL_EXPIRIES_SECONDS = {
    "dataset_get": 2 * 3600,
    "installer_get": 2 * 3600,
    "license_get": 2 * 3600,
    "license_delete": 3 * 3600,
    "status_put": 7 * 3600,
    "pulse_put": 7 * 3600,
    "results_put": 7 * 3600,
    "canary_results_put": 7 * 3600,
    "canary_approval_get": 7 * 3600,
}


# This template deliberately contains no transcript capture.  The hard TTL
# task and the independent pulse process are created before driver/install
# work begins.  All fields uploaded by the pulse agent are curated.
BOOTSTRAP_TEMPLATE = r'''<powershell>
$ErrorActionPreference = "Continue"
$runId = "__RUN_ID__"
$attempt = __ATTEMPT__
$ttlDeadlineUtc = [DateTime]::Parse("__TTL_DEADLINE_UTC__").ToUniversalTime()
$pulseIntervalSeconds = __PULSE_INTERVAL_SECONDS__
$phaseLimits = '__PHASE_LIMITS_JSON__' | ConvertFrom-Json
New-Item -ItemType Directory -Force -Path C:\work, C:\work\out, C:\work\pulses | Out-Null

function Put-CuratedStatus([string]$phase, [string]$message) {
  $safe = $message -replace 'https?://[^\s]+', '[REDACTED_URL]' -replace '(?i)(password|passwd|secret|token|authorization|email)\s*[:=]\s*[^\s,;]+', '$1=[REDACTED_SECRET]'
  $line = [ordered]@{
    schema_version = "worker_status.v2"
    run_id = $runId
    attempt = $attempt
    observed_at_utc = (Get-Date).ToUniversalTime().ToString('o')
    phase = $phase
    message = $safe
  } | ConvertTo-Json -Compress
  Add-Content -Path C:\work\status.jsonl -Value $line
  try { Invoke-RestMethod -Method Put -Uri "__STATUS_PUT__" -InFile C:\work\status.jsonl -ContentType "application/x-ndjson" | Out-Null } catch {}
}

function Set-WorkerState([string]$phase, [string]$arm, [string]$logPath) {
  $phaseStarted = (Get-Date).ToUniversalTime()
  [ordered]@{
    schema_version = "postshot_worker_state.v2"
    run_id = $runId
    attempt = $attempt
    phase = $phase
    arm = $arm
    log_path = $logPath
    phase_started_at_utc = $phaseStarted.ToString('o')
    startup_grace_until_utc = $phaseStarted.AddSeconds(600).ToString('o')
  } | ConvertTo-Json -Compress | Set-Content -Path C:\work\worker-state.json -Encoding UTF8
}

function Stop-Worker([string]$reason) {
  Put-CuratedStatus "abort" $reason
  try { Stop-Computer -Force } catch {}
}

function Redact-Line([string]$line) {
  return ($line -replace 'https?://[^\s]+', '[REDACTED_URL]' -replace '(?i)(--password\s+|--login\s+|password|passwd|secret|token|authorization|email)\s*[:=]?\s*[^\s,;]+', '$1[REDACTED_SECRET]')
}

Set-WorkerState "windows_boot" "none" "C:\work\status.jsonl"

# Hard watchdog: this task is independent of the conversational session and
# the Postshot process.  It enforces the exact UTC deadline and every phase
# deadline.  InstanceInitiatedShutdownBehavior=terminate is set in launch().
$hardWatchdog = @'
$runId = "__RUN_ID__"
$deadline = [DateTime]::Parse("__TTL_DEADLINE_UTC__").ToUniversalTime()
$phaseLimits = '__PHASE_LIMITS_JSON__' | ConvertFrom-Json
while ($true) {
  $now = (Get-Date).ToUniversalTime()
  if ($now -ge $deadline) {
    Add-Content -Path C:\work\watchdog.log -Value "$($now.ToString('o')) HARD_TTL_EXPIRED run=$runId"
    Stop-Computer -Force
    break
  }
  try {
    $state = Get-Content C:\work\worker-state.json -Raw | ConvertFrom-Json
    $limit = $phaseLimits.($state.phase)
    if ($limit -and $state.phase_started_at_utc) {
      $phaseDeadline = ([DateTime]::Parse($state.phase_started_at_utc).ToUniversalTime()).AddSeconds([double]$limit)
      if ($now -ge $phaseDeadline) {
        Add-Content -Path C:\work\watchdog.log -Value "$($now.ToString('o')) PHASE_TIMEOUT phase=$($state.phase) run=$runId"
        Stop-Computer -Force
        break
      }
    }
  } catch {}
  Start-Sleep -Seconds 15
}
'@
$hardWatchdog = $hardWatchdog.Replace('__RUN_ID__', $runId).Replace('__TTL_DEADLINE_UTC__', $ttlDeadlineUtc.ToString('o')).Replace('__PHASE_LIMITS_JSON__', ('__PHASE_LIMITS_JSON__'))
Set-Content -Path C:\work\hard-watchdog.ps1 -Value $hardWatchdog -Encoding UTF8
$watchdogAction = New-ScheduledTaskAction -Execute "PowerShell.exe" -Argument "-NoProfile -ExecutionPolicy Bypass -File C:\work\hard-watchdog.ps1"
$watchdogTrigger = New-ScheduledTaskTrigger -Once -At $ttlDeadlineUtc.ToLocalTime()
Register-ScheduledTask -TaskName "BlueprintPostshotHardWatchdog-$runId" -Action $watchdogAction -Trigger $watchdogTrigger -User "SYSTEM" -RunLevel Highest -Force | Out-Null
Start-ScheduledTask -TaskName "BlueprintPostshotHardWatchdog-$runId"

# Independent pulse process.  It writes a local cumulative JSONL series and
# replaces the remote telemetry object with the latest complete series.  The
# series contains no transcript and no credential-bearing URL.
$pulseAgent = @'
$runId = "__RUN_ID__"
$attempt = __ATTEMPT__
$localInstanceId = try { (Invoke-RestMethod -Uri "http://169.254.169.254/latest/meta-data/instance-id" -TimeoutSec 2) } catch { "unknown" }
$localInstanceType = try { (Invoke-RestMethod -Uri "http://169.254.169.254/latest/meta-data/instance-type" -TimeoutSec 2) } catch { "unknown" }
$ttlDeadlineUtc = [DateTime]::Parse("__TTL_DEADLINE_UTC__").ToUniversalTime()
$pulseIntervalSeconds = __PULSE_INTERVAL_SECONDS__
$phaseLimits = '__PHASE_LIMITS_JSON__' | ConvertFrom-Json
$seqPath = "C:\work\pulse-seq.txt"
$seriesPath = "C:\work\pulse-series.jsonl"
function Next-Sequence {
  $n = 0
  try { $n = [int](Get-Content $seqPath -Raw) } catch {}
  $n += 1
  Set-Content -Path $seqPath -Value $n -Encoding ASCII
  return $n
}
function Sha256([string]$path) {
  try { return (Get-FileHash -Path $path -Algorithm SHA256).Hash.ToLower() } catch { return $null }
}
function Safe-Tail([string]$path) {
  try {
    return Redact-Line (((Get-Content -Path $path -Tail 40 -ErrorAction Stop) -join "`n"))
  } catch { return "" }
}
function Safe-Name([string]$name) {
  return ([regex]::Replace([IO.Path]::GetFileName($name), '[^A-Za-z0-9._-]', '_')).Trim('.')
}
function Number([string]$value) {
  try { return [double]$value } catch { return $null }
}
while ($true) {
  try {
    $state = Get-Content C:\work\worker-state.json -Raw | ConvertFrom-Json
    $now = (Get-Date).ToUniversalTime()
    $previous = $null
    if (Test-Path $seriesPath) {
      try { $previous = Get-Content $seriesPath -Tail 1 | ConvertFrom-Json } catch {}
    }
    $proc = Get-Process -Name postshot-cli -ErrorAction SilentlyContinue | Select-Object -First 1
    $procStart = $null
    if ($proc) { try { $procStart = $proc.StartTime.ToUniversalTime().ToString('o') } catch {} }
    $logPath = if ($state.log_path) { $state.log_path } else { "C:\work\status.jsonl" }
    $log = Get-Item $logPath -ErrorAction SilentlyContinue
    $logBytes = if ($log) { [int64]$log.Length } else { 0 }
    $oldLogBytes = if ($previous -and $previous.postshot_log) { [int64]$previous.postshot_log.byte_count } else { 0 }
    $logGrowth = $logBytes - $oldLogBytes
    $gpuRaw = & "C:\Windows\System32\nvidia-smi.exe" --query-gpu=name,driver_version,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits 2>$null
    $gpu = if ($gpuRaw) { @($gpuRaw -split ',') | ForEach-Object { $_.Trim() } } else { @() }
    $gpuUtil = if ($gpu.Count -gt 2) { Number $gpu[2] } else { $null }
    $gpuMemory = if ($gpu.Count -gt 3) { Number $gpu[3] } else { $null }
    $gpuActive = (($gpuUtil -ne $null -and $gpuUtil -gt 0) -or ($gpuMemory -ne $null -and $gpuMemory -gt 0))
    $outputs = @()
    $oldOutputSizes = @{}
    if ($previous -and $previous.outputs) { foreach ($oldOutput in @($previous.outputs)) { $oldOutputSizes[[string]$oldOutput.path] = [int64]$oldOutput.bytes } }
    foreach ($file in Get-ChildItem -Path "C:\work\out" -Recurse -File -ErrorAction SilentlyContinue) {
      $growth = [int64]$file.Length - [int64]($oldOutputSizes[$file.Name] | ForEach-Object { if ($_ -ne $null) { $_ } else { 0 } })
      $outputs += [ordered]@{ path = (Safe-Name $file.Name); bytes = [int64]$file.Length; growth_bytes = $growth; digest = (Sha256 $file.FullName) }
    }
    $outputProgress = @($outputs | Where-Object { [int64]$_.growth_bytes -gt 0 }).Count -gt 0
    $phaseProgress = [bool]($previous -and [string]$previous.phase -ne [string]$state.phase)
    $graceUntil = if ($state.startup_grace_until_utc) { [DateTime]::Parse($state.startup_grace_until_utc).ToUniversalTime() } else { $now }
    $startupGraceActive = $now -lt $graceUntil
    $credible = (-not $startupGraceActive) -and ($logGrowth -gt 0 -or $outputProgress -or $phaseProgress -or ([bool]$proc -and $gpuActive))
    $lastCredible = if ($credible) { $now.ToString('o') } elseif ($previous) { $previous.last_credible_progress_at_utc } else { $null }
    $elapsedHours = [Math]::Max(0, ($now - [DateTime]::Parse("__WORKER_START_UTC__").ToUniversalTime()).TotalHours)
    $liveCost = [Math]::Round(($elapsedHours * (__INSTANCE_HOURLY_USD__ + __EBS_HOURLY_USD__)), 6)
    $phaseKill = $ttlDeadlineUtc
    $phaseLimit = $phaseLimits.($state.phase)
    if ($phaseLimit) {
      $candidatePhaseKill = ([DateTime]::Parse($state.phase_started_at_utc).ToUniversalTime()).AddSeconds([double]$phaseLimit)
      if ($candidatePhaseKill -lt $phaseKill) { $phaseKill = $candidatePhaseKill }
    }
    $credentialState = if (Test-Path C:\work\credential-deletion.json) { (Get-Content C:\work\credential-deletion.json -Raw | ConvertFrom-Json).state } else { "not_acknowledged" }
    $resultState = if (Test-Path C:\work\result-upload-state.txt) { Get-Content C:\work\result-upload-state.txt -Raw } else { "not_started" }
    $exitCode = $null
    $receiptPath = "C:\work\out\$($state.arm)\receipt.txt"
    if (Test-Path $receiptPath) {
      $exitMatch = [regex]::Match((Get-Content $receiptPath -Raw), 'exit=(-?\d+)')
      if ($exitMatch.Success) { $exitCode = [int]$exitMatch.Groups[1].Value }
    }
    $pulse = [ordered]@{
      schema_version = "worker_pulse.v2"
      pulse_digest_encoding = "sha256:json_utf8_noncanonical"
      run_id = $runId
      attempt = $attempt
      arm = $state.arm
      phase = $state.phase
      phase_started_at_utc = $state.phase_started_at_utc
      sequence = (Next-Sequence)
      observed_at_utc = (Get-Date).ToUniversalTime().ToString('o')
      instance = [ordered]@{ id = $localInstanceId; type = $localInstanceType; state = "running" }
      postshot_process = [ordered]@{ pid = if ($proc) { $proc.Id } else { $null }; start_time_utc = $procStart; alive = [bool]$proc; exit_code = $exitCode }
      postshot_log = [ordered]@{ tail = (Safe-Tail $logPath); byte_count = $logBytes; digest = if ($log) { Sha256 $log.FullName } else { $null }; growth_bytes = $logGrowth }
      gpu = [ordered]@{ name = if ($gpu.Count -gt 0) { $gpu[0] } else { $null }; driver_version = if ($gpu.Count -gt 1) { $gpu[1] } else { $null }; utilization_percent = $gpuUtil; memory_used_mib = $gpuMemory; memory_total_mib = if ($gpu.Count -gt 4) { Number $gpu[4] } else { $null }; temperature_c = if ($gpu.Count -gt 5) { Number $gpu[5] } else { $null }; power_w = if ($gpu.Count -gt 6) { Number $gpu[6] } else { $null } }
      outputs = @($outputs)
      disk_free_bytes = (Get-PSDrive -Name C).Free
      last_credible_progress_at_utc = $lastCredible
      live_cost_estimate_usd = $liveCost
      incremental_cap_usd = __INCREMENTAL_CAP_USD__
      ttl_deadline_utc = "__TTL_DEADLINE_UTC__"
      next_automatic_kill_deadline_utc = $phaseKill.ToString('o')
      startup_grace_until_utc = $state.startup_grace_until_utc
      result_upload_state = $resultState.Trim()
      credential_object_deletion_state = $credentialState
      progress = [ordered]@{ os_alive = $true; postshot_process_alive = [bool]$proc; gpu_active = $gpuActive; log_progress = ($logGrowth -gt 0); output_progress = $outputProgress; phase_progress = $phaseProgress; startup_grace_active = $startupGraceActive; credible_training_progress = $credible }
    }
    $json = $pulse | ConvertTo-Json -Depth 12 -Compress
    $pulse["pulse_digest"] = "sha256:" + ([BitConverter]::ToString(([Security.Cryptography.SHA256]::Create()).ComputeHash([Text.Encoding]::UTF8.GetBytes($json))).Replace('-', '').ToLower())
    $json = $pulse | ConvertTo-Json -Depth 12 -Compress
    Add-Content -Path $seriesPath -Value $json -Encoding UTF8
    $pulsePath = "C:\work\pulses\pulse-$('{0:D8}' -f $pulse.sequence).json"
    Set-Content -Path $pulsePath -Value $json -Encoding UTF8
    Invoke-RestMethod -Method Put -Uri "__PULSE_PUT__" -InFile $seriesPath -ContentType "application/x-ndjson" | Out-Null
  } catch {
    Add-Content -Path C:\work\watchdog.log -Value "$(Get-Date -Format o) PULSE_AGENT_ERROR $($_.Exception.GetType().Name)"
  }
  Start-Sleep -Seconds $pulseIntervalSeconds
}
'@
$pulseAgent = $pulseAgent.Replace('__RUN_ID__', $runId).Replace('__ATTEMPT__', [string]$attempt).Replace('__TTL_DEADLINE_UTC__', $ttlDeadlineUtc.ToString('o')).Replace('__INCREMENTAL_CAP_USD__', [string]__INCREMENTAL_CAP_USD__).Replace('__WORKER_START_UTC__', (Get-Date).ToUniversalTime().ToString('o')).Replace('__INSTANCE_HOURLY_USD__', '__INSTANCE_HOURLY_USD__').Replace('__EBS_HOURLY_USD__', '__EBS_HOURLY_USD__')
Set-Content -Path C:\work\pulse-agent.ps1 -Value $pulseAgent -Encoding UTF8
Start-Process -FilePath PowerShell.exe -ArgumentList "-NoProfile","-ExecutionPolicy","Bypass","-File","C:\work\pulse-agent.ps1" -WindowStyle Hidden

function Download-LicenseAndDelete {
  try {
    Invoke-WebRequest -Uri "__LICENSE_GET__" -OutFile C:\work\license.env -UseBasicParsing -TimeoutSec 240
    $lic = @{}
    Get-Content C:\work\license.env | ForEach-Object { if ($_ -match "^([A-Za-z_]+)=(.*)$") { $lic[$Matches[1]] = $Matches[2] } }
    if (-not $lic["POSTSHOT_LOGIN_EMAIL"] -or -not $lic["POSTSHOT_LOGIN_PASSWORD"]) { throw "license_env_incomplete" }
    Put-CuratedStatus "dataset_license_retrieval" "license acknowledged; deleting remote credential object"
    $response = Invoke-WebRequest -Method Delete -Uri "__LICENSE_DELETE__" -UseBasicParsing
    if ($response.StatusCode -lt 200 -or $response.StatusCode -ge 300) { throw "license_delete_not_acknowledged" }
    [ordered]@{ schema_version = "postshot_credential_deletion.v1"; state = "deleted"; object_key = "__LICENSE_KEY__"; acknowledged_at_utc = (Get-Date).ToUniversalTime().ToString('o'); raw_secret_values_recorded = $false } | ConvertTo-Json -Compress | Set-Content C:\work\credential-deletion.json -Encoding UTF8
    return $lic
  } finally {
    Remove-Item C:\work\license.env -Force -ErrorAction SilentlyContinue
  }
}

Put-CuratedStatus "windows_boot" "worker_started"
Set-WorkerState "nvidia_driver" "none" "C:\work\status.jsonl"
$driverOk = $false
foreach ($driverUrl in @(__DRIVER_URLS__)) {
  try {
    Put-CuratedStatus "nvidia_driver" "driver_download_started"
    Invoke-WebRequest -Uri $driverUrl -OutFile C:\work\nvidia.exe -UseBasicParsing -TimeoutSec 600
    $p = Start-Process -FilePath C:\work\nvidia.exe -ArgumentList "-s","-noreboot" -PassThru
    Wait-Process -Id $p.Id -Timeout 900 -ErrorAction SilentlyContinue | Out-Null
    if (-not $p.HasExited) { Stop-Process -Id $p.Id -Force; continue }
    $gpu = & "C:\Windows\System32\nvidia-smi.exe" --query-gpu=name,driver_version --format=csv,noheader 2>$null
    if ($LASTEXITCODE -eq 0 -and $gpu) { Put-CuratedStatus "nvidia_driver" "driver_verified"; $driverOk = $true; break }
  } catch { Put-CuratedStatus "nvidia_driver" "driver_attempt_failed:$($_.Exception.GetType().Name)" }
}
if (-not $driverOk) { Stop-Worker "nvidia_driver_install_failed"; exit 20 }

Set-WorkerState "msi_download_install" "none" "C:\work\status.jsonl"
try {
  Put-CuratedStatus "msi_download_install" "msi_download_started"
  Invoke-WebRequest -Uri "__INSTALLER_GET__" -OutFile C:\work\Postshot-1.1.0.msi -UseBasicParsing -TimeoutSec 600
  $hash = (Get-FileHash C:\work\Postshot-1.1.0.msi -Algorithm SHA256).Hash.ToLower()
  if ($hash -ne "__INSTALLER_SHA256__") { throw "installer_digest_mismatch" }
  $p = Start-Process -FilePath msiexec.exe -ArgumentList "/i","C:\work\Postshot-1.1.0.msi","/qn","/norestart","/l*v","C:\work\postshot-install.log" -PassThru
  Wait-Process -Id $p.Id -Timeout 600 -ErrorAction SilentlyContinue | Out-Null
  if (-not $p.HasExited) { Stop-Process -Id $p.Id -Force; throw "msiexec_timeout" }
  if ($p.ExitCode -ne 0 -and $p.ExitCode -ne 3010) { throw "msiexec_exit_$($p.ExitCode)" }
  Put-CuratedStatus "msi_download_install" "msi_install_verified"
} catch { Stop-Worker "postshot_install_failed:$($_.Exception.GetType().Name)"; exit 21 }

$cli = "$Env:ProgramFiles\Jawset Postshot\bin\postshot-cli.exe"
if (-not (Test-Path $cli)) { $found = Get-ChildItem -Path "$Env:ProgramFiles" -Recurse -Filter postshot-cli.exe -ErrorAction SilentlyContinue | Select-Object -First 1; if ($found) { $cli = $found.FullName } }
if (-not (Test-Path $cli)) { Stop-Worker "postshot_cli_not_found"; exit 22 }
Set-WorkerState "cli_activation_canary" "none" "C:\work\status.jsonl"
& $cli --help *> C:\work\out\postshot-help.txt
& $cli train --help *> C:\work\out\postshot-train-help.txt

Set-WorkerState "dataset_license_retrieval" "none" "C:\work\status.jsonl"
try {
  Invoke-WebRequest -Uri "__DATASET_GET__" -OutFile C:\work\dataset.zip -UseBasicParsing -TimeoutSec 600
  Expand-Archive -Path C:\work\dataset.zip -DestinationPath C:\work\dataset -Force
  $lic = Download-LicenseAndDelete
} catch { Stop-Worker "input_download_failed:$($_.Exception.GetType().Name)"; exit 23 }

$dataset = "C:\work\dataset"
Set-WorkerState "tiny_training_canary" "CANARY" "C:\work\out\canary\train-log.txt"
try {
    $canaryOut = "C:\work\out\canary"
    New-Item -ItemType Directory -Force -Path $canaryOut | Out-Null
  Put-CuratedStatus "tiny_training_canary" "canary_started_train_steps_limit_1_max_image_size_256_max_num_splats_100"
  $canaryArgs = @("--login",$lic["POSTSHOT_LOGIN_EMAIL"],"--password",$lic["POSTSHOT_LOGIN_PASSWORD"],"train","--import",$dataset,"--profile","Splat3","--no-recenter-points","--max-image-size","256","--train-steps-limit","1","--max-num-splats","100","--output","$canaryOut\C0_canary_splat3.psht","--export-splat","$canaryOut\C0_canary_splat3.ply")
  & $cli @canaryArgs 2>&1 | ForEach-Object { Redact-Line ([string]$_) } | Set-Content "$canaryOut\train-log.txt" -Encoding UTF8
  $canaryExit = $LASTEXITCODE
  Set-Content -Path "$canaryOut\receipt.txt" -Value "arm=CANARY exit=$canaryExit" -Encoding UTF8
  (Get-Content "$canaryOut\train-log.txt" -Raw -ErrorAction SilentlyContinue) -replace [regex]::Escape($lic["POSTSHOT_LOGIN_PASSWORD"]), "***" -replace [regex]::Escape($lic["POSTSHOT_LOGIN_EMAIL"]), "***" | Set-Content "$canaryOut\train-log.txt" -Encoding UTF8
  if ($canaryExit -ne 0 -or -not (Test-Path "$canaryOut\C0_canary_splat3.psht") -or -not (Test-Path "$canaryOut\C0_canary_splat3.ply")) { throw "tiny_canary_failed_exit_$canaryExit" }
  Put-CuratedStatus "tiny_training_canary" "canary_verified"
} catch { Stop-Worker "tiny_training_canary_failed:$($_.Exception.GetType().Name)"; exit 24 }

function Run-Arm([string]$armId, [string]$profile) {
  $out = "C:\work\out\$armId"
  New-Item -ItemType Directory -Force -Path $out | Out-Null
  Set-WorkerState $armId $armId "$out\train-log.txt"
  Put-CuratedStatus $armId "training_started"
  $args = @("--login",$lic["POSTSHOT_LOGIN_EMAIL"],"--password",$lic["POSTSHOT_LOGIN_PASSWORD"],"train","--import",$dataset,"--profile",$profile,"--no-recenter-points","--max-image-size","0","--output","$out\$armId.psht","--export-splat","$out\$armId.ply")
  $redacted = ($args | ForEach-Object { $_ -replace [regex]::Escape($lic["POSTSHOT_LOGIN_PASSWORD"]), "***" -replace [regex]::Escape($lic["POSTSHOT_LOGIN_EMAIL"]), "***" }) -join " "
  Set-Content -Path "$out\command.txt" -Value $redacted -Encoding UTF8
  $timer = [Diagnostics.Stopwatch]::StartNew()
  & $cli @args 2>&1 | ForEach-Object { Redact-Line ([string]$_) } | Set-Content "$out\train-log.txt" -Encoding UTF8
  $code = $LASTEXITCODE
  $timer.Stop()
  (Get-Content "$out\train-log.txt" -Raw -ErrorAction SilentlyContinue) -replace [regex]::Escape($lic["POSTSHOT_LOGIN_PASSWORD"]), "***" -replace [regex]::Escape($lic["POSTSHOT_LOGIN_EMAIL"]), "***" | Set-Content "$out\train-log.txt" -Encoding UTF8
  Add-Content -Path "$out\receipt.txt" -Value "arm=$armId exit=$code seconds=$([int]$timer.Elapsed.TotalSeconds)"
  Put-CuratedStatus $armId "training_finished_exit_$code"
  return $code
}

$p1 = Run-Arm "P1" "Splat3"
$p2 = Run-Arm "P2" "Splat MCMC"
try {
  Set-Content -Path C:\work\result-upload-state.txt -Value "upload_started" -Encoding UTF8
  Copy-Item C:\work\postshot-install.log C:\work\out\ -ErrorAction SilentlyContinue
  Compress-Archive -Path C:\work\out\* -DestinationPath C:\work\results.zip -Force
  Put-CuratedStatus "result_upload" "results_upload_started"
  Invoke-RestMethod -Method Put -Uri "__RESULTS_PUT__" -InFile C:\work\results.zip -ContentType "application/zip" | Out-Null
  Set-Content -Path C:\work\result-upload-state.txt -Value "upload_completed" -Encoding UTF8
  Put-CuratedStatus "result_upload" "results_upload_completed_p1_$p1`_p2_$p2"
} catch { Set-Content -Path C:\work\result-upload-state.txt -Value "upload_failed" -Encoding UTF8; Put-CuratedStatus "result_upload" "results_upload_failed:$($_.Exception.GetType().Name)" }
Stop-Computer -Force
</powershell>
'''


def _read_secret(name: str) -> str:
    return (SECRETS / name).read_text(encoding="utf-8").strip()


def _secret_values() -> list[str]:
    values: list[str] = []
    for name in ("postshot.env", "aws_agent_credentials", "digitalocean_spaces_access_key_id", "digitalocean_spaces_secret_access_key"):
        path = SECRETS / name
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                value = line.split("=", 1)[1].strip().strip('"').strip("'")
                if value:
                    values.append(value)
    return values


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


def _safe_run_id(run_id: str) -> str:
    if not RUN_ID_RE.fullmatch(run_id):
        raise ValueError("invalid_postshot_run_id")
    return run_id


def _state_dir(proxy_root: str | Path, run_id: str) -> Path:
    return Path(proxy_root).resolve() / "provider_packets" / "postshot" / _safe_run_id(run_id)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected_json_object:{path.name}")
    return value


def _write_artifact(path: Path, value: Mapping[str, Any]) -> None:
    assert_secret_free(value, _secret_values())
    write_json(path, dict(value))
    os.chmod(path, 0o600)


def _is_not_found(exc: Exception) -> bool:
    if isinstance(exc, ClientError):
        code = str(exc.response.get("Error", {}).get("Code", ""))
        return code in {"404", "NoSuchKey", "NoSuchObject", "NotFound", "InvalidInstanceID.NotFound"}
    return False


def _packet_path(proxy_root: Path, explicit: str | None) -> Path:
    if explicit:
        return Path(explicit).resolve()
    return proxy_root / "provider_packets" / "postshot" / "postshot_execution_packet.v1.json"


def _git_snapshot() -> dict[str, Any]:
    def run(args: list[str]) -> str:
        result = subprocess.run(args, cwd=REPO_ROOT, check=False, capture_output=True, text=True)  # noqa: S603
        return result.stdout.strip()

    return {
        "head": run(["git", "rev-parse", "HEAD"]),
        "status_porcelain": run(["git", "status", "--porcelain=v1"]),
    }


def _input_digests(packet: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "source_capture_digest": packet.get("source_capture_digest"),
        "candidate_pose_only_dataset_digest": packet.get("pose_only_dataset_digest"),
        "frozen_split_digest": packet.get("frozen_split_digest"),
        "hidden_images_included": packet.get("hidden_images_included"),
        "provider_sees_hidden_views": packet.get("provider_sees_hidden_views"),
    }


def _deterministic_zip_bytes(root: Path) -> bytes:
    import io

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED) as archive:
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(root).as_posix()
            info = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o600 << 16
            archive.writestr(info, path.read_bytes())
    return buffer.getvalue()


def _stage_inputs(arguments: argparse.Namespace) -> dict[str, Any]:
    proxy_root = Path(arguments.proxy_root).resolve()
    dataset_root = proxy_root / arguments.dataset_relative
    if not dataset_root.is_dir():
        raise SystemExit(f"dataset_missing:{dataset_root}")
    installer = Path(arguments.installer).resolve()
    if not installer.is_file():
        raise SystemExit(f"installer_missing:{installer}")
    observed = sha256_file(installer).removeprefix("sha256:")
    if observed != POSTSHOT_INSTALLER_SHA256:
        raise SystemExit(f"installer_digest_mismatch:{observed}")
    run_id = _safe_run_id(arguments.run_id or time.strftime("postshot-%Y%m%dT%H%M%SZ", time.gmtime()))
    prefix = f"{TAG}/{run_id}"
    bucket = _read_secret("digitalocean_spaces_bucket")
    spaces = _spaces_client()
    # A deterministic ZIP is required so the candidate digest is reproducible.
    dataset_bytes = _deterministic_zip_bytes(dataset_root)
    license_bytes = (SECRETS / "postshot.env").read_bytes()
    installer_bytes = installer.read_bytes()
    keys = {
        "dataset": f"{prefix}/dataset.zip",
        "installer": f"{prefix}/Postshot-1.1.0.msi",
        "license": f"{prefix}/license.env",
        "status": f"{prefix}/status.jsonl",
        "pulse": f"{prefix}/pulse-series.jsonl",
        "results": f"{prefix}/results.zip",
    }
    spaces.put_object(Bucket=bucket, Key=keys["dataset"], Body=dataset_bytes, ACL="private")
    spaces.put_object(Bucket=bucket, Key=keys["installer"], Body=installer_bytes, ACL="private")
    spaces.put_object(Bucket=bucket, Key=keys["license"], Body=license_bytes, ACL="private")
    urls: dict[str, str] = {}
    urls["dataset_get"] = spaces.generate_presigned_url("get_object", Params={"Bucket": bucket, "Key": keys["dataset"]}, ExpiresIn=URL_EXPIRIES_SECONDS["dataset_get"])
    urls["installer_get"] = spaces.generate_presigned_url("get_object", Params={"Bucket": bucket, "Key": keys["installer"]}, ExpiresIn=URL_EXPIRIES_SECONDS["installer_get"])
    urls["license_get"] = spaces.generate_presigned_url("get_object", Params={"Bucket": bucket, "Key": keys["license"]}, ExpiresIn=URL_EXPIRIES_SECONDS["license_get"])
    urls["license_delete"] = spaces.generate_presigned_url("delete_object", Params={"Bucket": bucket, "Key": keys["license"]}, ExpiresIn=URL_EXPIRIES_SECONDS["license_delete"])
    urls["status_put"] = spaces.generate_presigned_url("put_object", Params={"Bucket": bucket, "Key": keys["status"], "ContentType": "application/x-ndjson"}, ExpiresIn=URL_EXPIRIES_SECONDS["status_put"])
    urls["pulse_put"] = spaces.generate_presigned_url("put_object", Params={"Bucket": bucket, "Key": keys["pulse"], "ContentType": "application/x-ndjson"}, ExpiresIn=URL_EXPIRIES_SECONDS["pulse_put"])
    urls["results_put"] = spaces.generate_presigned_url("put_object", Params={"Bucket": bucket, "Key": keys["results"], "ContentType": "application/zip"}, ExpiresIn=URL_EXPIRIES_SECONDS["results_put"])
    staging = {
        "schema_version": "postshot_worker_staging.v2",
        "run_id": run_id,
        "bucket": bucket,
        "keys": keys,
        "dataset_digest": sha256_bytes(dataset_bytes),
        "dataset_bytes": len(dataset_bytes),
        "installer_digest": "sha256:" + POSTSHOT_INSTALLER_SHA256,
        "license_blob_digest": sha256_bytes(license_bytes),
        "url_expiry_seconds": dict(URL_EXPIRIES_SECONDS),
        "ttl_minutes": ATTEMPT_5_TTL_MINUTES,
        "incremental_cap_usd": ATTEMPT_5_INCREMENTAL_CAP_USD,
        "raw_secret_values_recorded": False,
    }
    staging["staging_digest"] = canonical_digest(staging, digest_field="staging_digest")
    state_dir = _state_dir(proxy_root, run_id)
    state_dir.mkdir(parents=True, exist_ok=True)
    _write_artifact(state_dir / "staging.json", staging)
    (state_dir / "presigned_urls.json").write_text(json.dumps(urls) + "\n", encoding="utf-8")
    os.chmod(state_dir / "presigned_urls.json", 0o600)
    safe = {key: staging[key] for key in ("run_id", "dataset_bytes", "dataset_digest", "installer_digest", "url_expiry_seconds")}
    print(json.dumps(safe, indent=2))
    print(f"state: {state_dir}")
    return {"staging": staging, "urls": urls, "state_dir": state_dir}


def _authorization(arguments: argparse.Namespace) -> tuple[bool, str]:
    candidate = os.environ.get("POSTSHOT_ATTEMPT_5_AUTHORIZATION", "")
    if arguments.authorization_file:
        path = Path(arguments.authorization_file).resolve()
        if path.is_file():
            candidate = path.read_text(encoding="utf-8").strip()
    return candidate == AUTHORIZATION_LINE, "explicit_user_authorization_line" if candidate == AUTHORIZATION_LINE else "missing_exact_authorization_line"


def admit(arguments: argparse.Namespace) -> dict[str, Any]:
    proxy_root = Path(arguments.proxy_root).resolve()
    state_dir = _state_dir(proxy_root, arguments.run_id)
    staging = _read_json(state_dir / "staging.json")
    packet = _read_json(_packet_path(proxy_root, arguments.execution_packet))
    git = _git_snapshot()
    authorized, authorization_reason = _authorization(arguments)
    input_digests = _input_digests(packet)
    checks = {
        "source_commit_present": bool(git["head"]),
        "worktree_clean": git["status_porcelain"] == "",
        "candidate_split_digest_frozen": bool(input_digests.get("frozen_split_digest")),
        "hidden_images_excluded": packet.get("hidden_images_included") is False and packet.get("provider_sees_hidden_views") is False,
        "one_worker_license_constraint": True,
        "staging_digest_present": bool(staging.get("staging_digest")),
        "focused_test_receipt_present": bool(arguments.focused_test_receipt and Path(arguments.focused_test_receipt).is_file()),
    }
    receipt = {
        "schema_version": "postshot_worker_admission.v2",
        "run_id": arguments.run_id,
        "attempt": 5,
        "generated_at_utc": utc_now_iso(),
        "source_commit": git["head"],
        "worktree_status": git["status_porcelain"],
        "staging_digest": staging.get("staging_digest"),
        "input_digests": input_digests,
        "checks": checks,
        "authorization": {"authorized": authorized, "reason": authorization_reason, "line_required": AUTHORIZATION_LINE},
        "provider_mutations_performed": 0,
        "launch_allowed": bool(authorized and all(checks.values())),
        "claim_ceiling": "execution_observability_only_until_independent_evaluation",
        "raw_secret_values_recorded": False,
    }
    receipt["admission_digest"] = canonical_digest(receipt, digest_field="admission_digest")
    _write_artifact(state_dir / "admission.json", receipt)
    print(json.dumps({"run_id": arguments.run_id, "launch_allowed": receipt["launch_allowed"], "checks": checks, "authorization": receipt["authorization"]}, indent=2))
    return receipt


def _phase_deadlines_json() -> str:
    return json.dumps(PHASE_LIMITS_SECONDS, separators=(",", ":"))


def _render_user_data(*, run_id: str, staging: Mapping[str, Any], urls: Mapping[str, str], instance_type: str) -> str:
    now = time.time()
    deadline = utc_now_iso(now + ATTEMPT_5_TTL_MINUTES * 60)
    rendered = BOOTSTRAP_TEMPLATE
    replacements = {
        "__RUN_ID__": run_id,
        "__ATTEMPT__": "5",
        "__TTL_DEADLINE_UTC__": deadline,
        "__PULSE_INTERVAL_SECONDS__": str(PULSE_INTERVAL_SECONDS),
        "__PHASE_LIMITS_JSON__": _phase_deadlines_json(),
        "__INSTANCE_ID__": "__INSTANCE_ID__",
        "__INSTANCE_TYPE__": instance_type,
        "__INCREMENTAL_CAP_USD__": str(ATTEMPT_5_INCREMENTAL_CAP_USD),
        "__INSTANCE_HOURLY_USD__": str(EXPECTED_HOURLY_USD),
        "__EBS_HOURLY_USD__": str(EXPECTED_EBS_HOURLY_USD),
        "__DRIVER_URLS__": ",".join(f'"{url}"' for url in NVIDIA_DRIVER_URLS),
        "__STATUS_PUT__": urls["status_put"],
        "__PULSE_PUT__": urls["pulse_put"],
        "__RESULTS_PUT__": urls["results_put"],
        "__LICENSE_GET__": urls["license_get"],
        "__LICENSE_DELETE__": urls["license_delete"],
        "__LICENSE_KEY__": str(staging["keys"]["license"]),
        "__INSTALLER_GET__": urls["installer_get"],
        "__INSTALLER_SHA256__": POSTSHOT_INSTALLER_SHA256,
        "__DATASET_GET__": urls["dataset_get"],
    }
    # The user-data string necessarily carries short-lived transport URLs to
    # the worker.  It is never printed or persisted in a local evidence file.
    for token, value in replacements.items():
        rendered = rendered.replace(token, value)
    return rendered


def _spawn_external_watchdog(*, state_dir: Path, run_id: str, ttl_deadline_utc: str) -> dict[str, Any]:
    log_path = state_dir / "external-watchdog.log"
    log_handle = log_path.open("a", encoding="utf-8")
    command = [sys.executable, str(Path(__file__).resolve()), "watch", "--proxy-root", str(state_dir.parents[2]), "--run-id", run_id, "--daemon", "--interval-seconds", str(PULSE_INTERVAL_SECONDS)]
    try:
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log_handle, stderr=subprocess.STDOUT, text=True, start_new_session=True, close_fds=True)  # noqa: S603
    except OSError:
        log_handle.close()
        raise
    finally:
        if not log_handle.closed:
            log_handle.close()
    record = build_external_watchdog_record(run_id=run_id, instance_id="", pid=process.pid, started_at_utc=utc_now_iso(), ttl_deadline_utc=ttl_deadline_utc, log_path=str(log_path), command_digest=sha256_bytes(canonical_json({"command": command}).encode("utf-8")))
    _write_artifact(state_dir / "external_watchdog.json", record)
    return record


def launch(arguments: argparse.Namespace) -> None:
    """Keep the legacy AWS mutation surface hard-disabled."""

    raise SystemExit(
        "legacy_postshot_windows_worker_launch_disabled_use_paid_resource_allocator"
    )


def _safe_tags(tags: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    return {sanitize_text(x.get("Key")): sanitize_text(x.get("Value")) for x in tags if x.get("Key")}


def _exact_instance(ec2: Any, run_id: str, instance_id: str) -> dict[str, Any]:
    try:
        response = ec2.describe_instances(InstanceIds=[instance_id])
    except Exception as exc:
        if _is_not_found(exc):
            return {"id": instance_id, "state": "not_found", "identity_valid": True, "volumes": []}
        return {"id": instance_id, "state": "describe_failed", "identity_valid": False, "error_type": type(exc).__name__, "volumes": []}
    rows = [item for reservation in response.get("Reservations", []) for item in reservation.get("Instances", [])]
    if not rows:
        return {"id": instance_id, "state": "not_found", "identity_valid": True, "volumes": []}
    row = rows[0]
    tags = _safe_tags(row.get("Tags", []))
    return {
        "id": row.get("InstanceId"),
        "state": row.get("State", {}).get("Name"),
        "type": row.get("InstanceType"),
        "launch_time": row.get("LaunchTime").isoformat() if row.get("LaunchTime") else None,
        "identity_valid": tags.get(RUN_TAG_KEY) == run_id,
        "tags": tags,
        "volumes": [mapping.get("Ebs", {}).get("VolumeId") for mapping in row.get("BlockDeviceMappings", []) if mapping.get("Ebs", {}).get("VolumeId")],
    }


def _inventory_exact(*, run_id: str, launch: Mapping[str, Any] | None = None) -> dict[str, Any]:
    session = _aws_session()
    ec2 = session.client("ec2")
    instance_id = str((launch or {}).get("instance_id", ""))
    instances: list[dict[str, Any]] = []
    if instance_id:
        instances.append(_exact_instance(ec2, run_id, instance_id))
    tagged = ec2.describe_instances(Filters=[{"Name": f"tag:{RUN_TAG_KEY}", "Values": [run_id]}]).get("Reservations", [])
    known = {str(item.get("id")) for item in instances}
    for reservation in tagged:
        for row in reservation.get("Instances", []):
            if row.get("InstanceId") in known:
                continue
            instances.append(_exact_instance(ec2, run_id, row["InstanceId"]))
    volumes: list[dict[str, Any]] = []
    volume_ids = sorted({str(item) for row in instances for item in row.get("volumes", []) if item})
    volume_response = ec2.describe_volumes(Filters=[{"Name": f"tag:{RUN_TAG_KEY}", "Values": [run_id]}]).get("Volumes", [])
    volume_ids.extend(str(row["VolumeId"]) for row in volume_response if row.get("VolumeId"))
    for volume_id in sorted(set(volume_ids)):
        try:
            response = ec2.describe_volumes(VolumeIds=[volume_id]).get("Volumes", [])
        except Exception as exc:
            if _is_not_found(exc):
                continue
            volumes.append({"id": volume_id, "state": "describe_failed", "error_type": type(exc).__name__})
            continue
        for row in response:
            volumes.append({"id": row.get("VolumeId"), "state": row.get("State"), "size": row.get("Size"), "attachments": [x.get("InstanceId") for x in row.get("Attachments", [])], "tags": _safe_tags(row.get("Tags", []))})
    snapshots = []
    for row in ec2.describe_snapshots(OwnerIds=[_aws_session().client("sts").get_caller_identity()["Account"]], Filters=[{"Name": f"tag:{RUN_TAG_KEY}", "Values": [run_id]}]).get("Snapshots", []):
        snapshots.append({"id": row.get("SnapshotId"), "state": row.get("State"), "tags": _safe_tags(row.get("Tags", []))})
    images = []
    for row in ec2.describe_images(Owners=[_aws_session().client("sts").get_caller_identity()["Account"]], Filters=[{"Name": f"tag:{RUN_TAG_KEY}", "Values": [run_id]}]).get("Images", []):
        images.append({"id": row.get("ImageId"), "state": row.get("State"), "name": sanitize_text(row.get("Name", "")), "tags": _safe_tags(row.get("Tags", []))})
    elastic_ips = []
    exact_instance_ids = {str(item.get("id")) for item in instances}
    for row in ec2.describe_addresses().get("Addresses", []):
        tags = _safe_tags(row.get("Tags", []))
        if tags.get(RUN_TAG_KEY) == run_id or row.get("InstanceId") in exact_instance_ids:
            elastic_ips.append({"allocation_id": row.get("AllocationId"), "association_id": row.get("AssociationId"), "instance_id": row.get("InstanceId"), "tags": tags})
    security_groups = []
    group_id = str((launch or {}).get("security_group_id", ""))
    if group_id:
        for row in ec2.describe_security_groups(GroupIds=[group_id]).get("SecurityGroups", []):
            security_groups.append({"id": row.get("GroupId"), "name": row.get("GroupName"), "vpc_id": row.get("VpcId"), "tags": _safe_tags(row.get("Tags", []))})
    return {
        "schema_version": "postshot_worker_inventory.v2",
        "run_id": run_id,
        "region": REGION,
        "scope": "exact_run_tag_account_region",
        "checked_at_utc": utc_now_iso(),
        "instances": instances,
        "volumes": volumes,
        "snapshots": snapshots,
        "images": images,
        "elastic_ips": elastic_ips,
        "security_groups": security_groups,
        "security_groups_are_non_billable_separate_state": True,
    }


def _terminate_exact(*, run_id: str, launch: Mapping[str, Any]) -> dict[str, Any]:
    instance_id = str(launch.get("instance_id", ""))
    if not instance_id:
        return {"requested": False, "reason": "exact_instance_id_missing"}
    ec2 = _aws_session().client("ec2")
    identity = _exact_instance(ec2, run_id, instance_id)
    if identity.get("identity_valid") is not True:
        return {"requested": False, "reason": "exact_instance_tag_not_verified", "instance_id": instance_id}
    if identity.get("state") in {"terminated", "not_found"}:
        return {"requested": False, "reason": "instance_already_terminal", "instance_id": instance_id, "state": identity.get("state")}
    ec2.terminate_instances(InstanceIds=[instance_id])
    return {"requested": True, "instance_id": instance_id, "state_before": identity.get("state"), "target_scope": "exact_instance_id_after_tag_verification"}


def _get_object_body(spaces: Any, *, bucket: str, key: str) -> bytes | None:
    try:
        return spaces.get_object(Bucket=bucket, Key=key)["Body"].read()
    except Exception as exc:
        if _is_not_found(exc):
            return None
        raise


def _load_remote_pulses(state_dir: Path, staging: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    body: bytes | None = None
    pulse_key = staging.get("keys", {}).get("pulse") if isinstance(staging.get("keys", {}), Mapping) else None
    if not pulse_key:
        return [], ["pulse_key_missing"]
    try:
        body = _get_object_body(_spaces_client(), bucket=staging["bucket"], key=pulse_key)
    except Exception as exc:
        return [], [f"pulse_read_failed:{type(exc).__name__}"]
    if body is None:
        local = state_dir / "pulse-series.jsonl"
        body = local.read_bytes() if local.is_file() else None
    if body is None:
        return [], []
    pulses: list[dict[str, Any]] = []
    errors: list[str] = []
    previous: Mapping[str, Any] | None = None
    for line_number, line in enumerate(body.decode("utf-8", errors="replace").splitlines(), 1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            errors.append(f"pulse_line_{line_number}:malformed_json")
            continue
        if not isinstance(value, dict):
            errors.append(f"pulse_line_{line_number}:not_object")
            continue
        if value.get("run_id") != staging.get("run_id"):
            errors.append(f"pulse_line_{line_number}:run_id_mismatch")
        errors.extend(f"pulse_line_{line_number}:{error}" for error in validate_pulse(value, previous))
        if previous is not None and value.get("run_id") != previous.get("run_id"):
            errors.append(f"pulse_line_{line_number}:run_id_changed")
        previous = value
        pulses.append(value)
    return pulses, sorted(set(errors))


def _cost_for_launch(launch: Mapping[str, Any], now_epoch: float) -> dict[str, Any]:
    elapsed_hours = max(0.0, now_epoch - float(launch.get("launched_at_epoch", now_epoch))) / 3600.0
    return build_live_cost_estimate(as_of_utc=utc_now_iso(now_epoch), instance_usd=elapsed_hours * EXPECTED_HOURLY_USD, ebs_usd=elapsed_hours * EXPECTED_EBS_HOURLY_USD, transfer_usd=0.0, object_storage_usd=0.0, license_increment_usd=0.0)


def _safe_status(*, run_id: str, launch: Mapping[str, Any] | None, pulses: Sequence[Mapping[str, Any]], pulse_errors: Sequence[str], now_epoch: float) -> dict[str, Any]:
    latest = pulses[-1] if pulses else {}
    observed = parse_timestamp(latest.get("observed_at_utc")) if latest else None
    progress = latest.get("progress", {}) if isinstance(latest.get("progress", {}), Mapping) else {}
    outputs = latest.get("outputs", []) if isinstance(latest.get("outputs", []), list) else []
    return {
        "schema_version": "postshot_worker_status.v2",
        "run_id": run_id,
        "instance_id": launch.get("instance_id") if launch else None,
        "phase": latest.get("phase") if latest else "unknown",
        "arm": latest.get("arm") if latest else "unknown",
        "postshot_process_alive": bool(progress.get("postshot_process_alive")) if latest else None,
        "gpu_active": bool(progress.get("gpu_active")) if latest else None,
        "training_progress_observable": bool(progress.get("credible_training_progress")) if latest else False,
        "log_growth_bytes": latest.get("postshot_log", {}).get("growth_bytes") if latest else None,
        "output_growth_bytes": sum(int(item.get("growth_bytes", 0) or 0) for item in outputs if isinstance(item, Mapping)),
        "heartbeat_age_seconds": None if observed is None else round(max(0.0, now_epoch - observed), 3),
        "live_cost_estimate_usd": latest.get("live_cost_estimate_usd") if latest else None,
        "incremental_cap_usd": latest.get("incremental_cap_usd") if latest else (launch.get("incremental_cap_usd") if launch else None),
        "kill_deadline": latest.get("next_automatic_kill_deadline_utc") if latest else (launch.get("ttl_deadline_utc") if launch else None),
        "pulse_count": len(pulses),
        "pulse_errors": list(pulse_errors),
        "training_progress_state": "training progress unobservable" if not latest or not progress.get("credible_training_progress") else "credible progress observed",
    }


def watch_once(arguments: argparse.Namespace) -> dict[str, Any]:
    proxy_root = Path(arguments.proxy_root).resolve()
    state_dir = _state_dir(proxy_root, arguments.run_id)
    launch_path = state_dir / "launch.json"
    if not launch_path.is_file():
        result = {"run_id": arguments.run_id, "action": "continue", "reason": "waiting_for_exact_launch_identity", "independent_process": True}
        print(json.dumps(result, indent=2))
        return result
    launch = _read_json(launch_path)
    staging = _read_json(state_dir / "staging.json")
    pulses, pulse_errors = _load_remote_pulses(state_dir, staging)
    now_epoch = time.time()
    inventory = _inventory_exact(run_id=arguments.run_id, launch=launch)
    identity_invalid = any(row.get("identity_valid") is False for row in inventory.get("instances", []))
    if identity_invalid:
        decision = WatchDecision("abort", "exact_instance_tag_mismatch")
    elif pulse_errors:
        decision = WatchDecision("terminate", "pulse_contract_invalid")
    else:
        launched_epoch = float(launch.get("launched_at_epoch", now_epoch))
        # Phase timeouts anchor at the current phase's start; anchoring at
        # launch would fire the 15-minute driver limit mid-install and kill
        # P2 at minute ~150 even when it just started.
        phase_started_epoch = derive_phase_started_epoch(pulses, launched_epoch=launched_epoch)
        decision = evaluate_pulses(pulses, now_epoch=now_epoch, phase_started_epoch=phase_started_epoch, launched_epoch=launched_epoch, live_cost_estimate_usd=float(_cost_for_launch(launch, now_epoch)["total_usd"]), incremental_cap_usd=float(launch.get("incremental_cap_usd", ATTEMPT_5_INCREMENTAL_CAP_USD)))
    action: dict[str, Any] = decision.as_dict()
    if decision.action in {"abort", "terminate"}:
        action.update(_terminate_exact(run_id=arguments.run_id, launch=launch))
        action["acted_at_utc"] = utc_now_iso()
        _write_artifact(state_dir / "watchdog_action.json", {"schema_version": "postshot_watchdog_action.v2", "run_id": arguments.run_id, **action})
    status = _safe_status(run_id=arguments.run_id, launch=launch, pulses=pulses, pulse_errors=pulse_errors, now_epoch=now_epoch)
    result = {"decision": action, "status": status, "provider_inventory": {"active_instance_count": sum(1 for row in inventory.get("instances", []) if row.get("state") in {"pending", "running", "stopping", "stopped", "shutting-down"})}}
    print(json.dumps(result, indent=2))
    return result


def watch(arguments: argparse.Namespace) -> None:
    state_dir = _state_dir(arguments.proxy_root, arguments.run_id)
    if arguments.daemon:
        previous = _read_json(state_dir / "external_watchdog.json") if (state_dir / "external_watchdog.json").is_file() else {}
        previous_pid = previous.get("pid")
        if previous_pid and int(previous_pid) != os.getpid():
            try:
                os.kill(int(previous_pid), 0)
            except (OSError, TypeError, ValueError):
                pass
            else:
                print(json.dumps({"run_id": arguments.run_id, "status": "watchdog_already_running", "pid": int(previous_pid)}, indent=2))
                return
        record = {
            "schema_version": EXTERNAL_WATCHDOG_SCHEMA_VERSION,
            "run_id": arguments.run_id,
            "instance_id": previous.get("instance_id", ""),
            "pid": os.getpid(),
            "started_at_utc": utc_now_iso(),
            "ttl_deadline_utc": previous.get("ttl_deadline_utc"),
            "log_path": str(state_dir / "external-watchdog.log"),
            "command_digest": previous.get("command_digest"),
            "status": "armed",
            "independent_process": True,
            "reattached_from_pid": previous.get("pid"),
            "raw_secret_values_recorded": False,
        }
        record["record_digest"] = canonical_digest(record, digest_field="record_digest")
        _write_artifact(state_dir / "external_watchdog.json", record)
    while True:
        result = watch_once(arguments)
        if not arguments.daemon or arguments.once:
            return
        action = result.get("decision", {}) if isinstance(result, Mapping) else {}
        if action.get("action") in {"abort", "terminate"}:
            return
        time.sleep(max(1, int(arguments.interval_seconds)))


def status(arguments: argparse.Namespace) -> None:
    state_dir = _state_dir(arguments.proxy_root, arguments.run_id)
    launch = _read_json(state_dir / "launch.json") if (state_dir / "launch.json").is_file() else None
    staging = _read_json(state_dir / "staging.json") if (state_dir / "staging.json").is_file() else None
    if staging:
        pulses, errors = _load_remote_pulses(state_dir, staging)
    else:
        pulses, errors = [], ["staging_missing"]
    print(json.dumps(_safe_status(run_id=arguments.run_id, launch=launch, pulses=pulses, pulse_errors=errors, now_epoch=time.time()), indent=2))


def collect(arguments: argparse.Namespace) -> None:
    state_dir = _state_dir(arguments.proxy_root, arguments.run_id)
    staging = _read_json(state_dir / "staging.json")
    spaces = _spaces_client()
    body = _get_object_body(spaces, bucket=staging["bucket"], key=staging["keys"]["results"])
    if body is None:
        raise SystemExit("results_not_uploaded")
    temporary = state_dir / "results.zip.part"
    temporary.write_bytes(body)
    with zipfile.ZipFile(temporary) as archive:
        bad_names = [name for name in archive.namelist() if ".." in Path(name).parts or Path(name).is_absolute()]
        if bad_names:
            temporary.unlink(missing_ok=True)
            raise SystemExit("result_archive_path_traversal")
    os.replace(temporary, state_dir / "results.zip")
    pulses, pulse_errors = _load_remote_pulses(state_dir, staging)
    if pulses:
        (state_dir / "pulse-series.jsonl").write_text("\n".join(canonical_json(item) for item in pulses) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": "postshot_worker_collection.v2",
        "run_id": arguments.run_id,
        "collected_at_utc": utc_now_iso(),
        "results_bytes": len(body),
        "results_digest": sha256_bytes(body),
        "pulse_count": len(pulses),
        "pulse_errors": pulse_errors,
        "raw_secret_values_recorded": False,
    }
    manifest["collection_digest"] = canonical_digest(manifest, digest_field="collection_digest")
    _write_artifact(state_dir / "collection.json", manifest)
    print(json.dumps(manifest, indent=2))


def _delete_staged_inputs(*, state_dir: Path, staging: Mapping[str, Any]) -> dict[str, Any]:
    spaces = _spaces_client()
    objects = []
    for kind in ("dataset", "installer", "license"):
        key = staging["keys"][kind]
        requested = False
        try:
            spaces.delete_object(Bucket=staging["bucket"], Key=key)
            requested = True
        except Exception as exc:
            if not _is_not_found(exc):
                objects.append({"key": key, "object_kind": kind, "delete_requested": False, "absent_verified": False, "error_type": type(exc).__name__})
                continue
        absent = False
        try:
            spaces.head_object(Bucket=staging["bucket"], Key=key)
        except Exception as exc:
            absent = _is_not_found(exc)
        objects.append({"key": key, "object_kind": kind, "delete_requested": requested, "absent_verified": absent})
    receipt = build_deletion_receipt(run_id=staging["run_id"], checked_at_utc=utc_now_iso(), objects=objects, secrets=_secret_values())
    _write_artifact(state_dir / "deletion_receipt.json", receipt)
    return receipt


def teardown(arguments: argparse.Namespace, *, abort_reason: str | None = None) -> dict[str, Any]:
    state_dir = _state_dir(arguments.proxy_root, arguments.run_id)
    launch = _read_json(state_dir / "launch.json") if (state_dir / "launch.json").is_file() else {}
    staging = _read_json(state_dir / "staging.json")
    action = _terminate_exact(run_id=arguments.run_id, launch=launch) if launch else {"requested": False, "reason": "launch_record_missing"}
    if abort_reason:
        action["abort_reason"] = sanitize_text(abort_reason, _secret_values())
    wait_deadline = time.monotonic() + max(0, int(arguments.wait_seconds))
    while launch and time.monotonic() < wait_deadline:
        inventory = _inventory_exact(run_id=arguments.run_id, launch=launch)
        states = [row.get("state") for row in inventory.get("instances", [])]
        if not any(state in {"pending", "running", "stopping", "stopped", "shutting-down"} for state in states):
            break
        time.sleep(5)
    inventory = _inventory_exact(run_id=arguments.run_id, launch=launch)
    proof = build_provider_zero_proof(run_id=arguments.run_id, region=REGION, instances=inventory.get("instances", []), volumes=inventory.get("volumes", []), snapshots=inventory.get("snapshots", []), images=inventory.get("images", []), elastic_ips=inventory.get("elastic_ips", []), security_groups=inventory.get("security_groups", []), checked_at_utc=utc_now_iso())
    proof["termination_action"] = action
    _write_artifact(state_dir / "teardown_proof.json", proof)
    deletion = _delete_staged_inputs(state_dir=state_dir, staging=staging)
    result = {"teardown_proof": proof, "deletion_receipt": deletion}
    print(json.dumps({"provider_zero": proof["provider_zero"], "blockers": proof["blockers"], "all_staged_inputs_absent": deletion["all_absent_verified"], "termination_action": action}, indent=2))
    return result


def abort(arguments: argparse.Namespace) -> None:
    teardown(arguments, abort_reason=arguments.reason or "operator_abort")


def inventory(arguments: argparse.Namespace) -> None:
    state_dir = _state_dir(arguments.proxy_root, arguments.run_id)
    launch = _read_json(state_dir / "launch.json") if (state_dir / "launch.json").is_file() else {}
    result = _inventory_exact(run_id=arguments.run_id, launch=launch)
    _write_artifact(state_dir / "inventory.json", result)
    print(json.dumps(result, indent=2))


def reconcile(arguments: argparse.Namespace) -> None:
    state_dir = _state_dir(arguments.proxy_root, arguments.run_id)
    launch = _read_json(state_dir / "launch.json") if (state_dir / "launch.json").is_file() else {}
    live = _cost_for_launch(launch, time.time()) if launch else {"kind": "live_estimate", "reconciled": False, "total_usd": None}
    result: dict[str, Any] = {"schema_version": "postshot_worker_cost_reconciliation.v1", "run_id": arguments.run_id, "live_estimate": live, "reconciled_billing": None, "status": "reconciliation_input_required"}
    if arguments.billing_json:
        billing = _read_json(Path(arguments.billing_json).resolve())
        result["reconciled_billing"] = build_reconciled_cost(source=str(Path(arguments.billing_json).resolve()), reconciled_at_utc=utc_now_iso(), total_usd=float(billing["total_usd"]), line_items=billing.get("line_items", {}))
        result["status"] = "reconciled"
    result["reconciliation_digest"] = canonical_digest(result, digest_field="reconciliation_digest")
    _write_artifact(state_dir / "cost_reconciliation.json", result)
    print(json.dumps(result, indent=2))


def _historical_attempts(proxy_root: Path) -> list[dict[str, Any]]:
    base = proxy_root / "provider_packets" / "postshot"
    data = [
        (1, "postshot-20260801T195240Z", "i-0dda3d1290611be90", "g5.xlarge", "infrastructure_bootstrap_failure", "vendor bootstrapper became silent after driver installation; monitoring was inadequate"),
        (2, "postshot-20260801T210457Z", "i-0a6a8c699195ba723", "g6.xlarge", "vendor_wrapper_incompatibility", "WiX/ReactionsBA quiet-mode installer hang reproduced; 900-second fence fired"),
        (3, "postshot-20260801T213413Z", "i-0768611be1d5888d0", "g6.xlarge", "cli_invocation_failure", "direct MSI install and CLI discovery succeeded; corrected global flag ordering had not yet been applied and exit 109 was observed"),
        (4, "postshot-20260801T215521Z", "i-037c684e4a5384bd1", "g6.xlarge", "monitoring_evidence_unavailable", "corrected invocation started, but no independent process/GPU/log/output progress was observable; run was manually terminated"),
    ]
    attempts = []
    for number, run_id, instance_id, instance_type, classification, finding in data:
        root = base / run_id
        attempts.append({
            "attempt": number,
            "run_id": run_id,
            "immutable_local_root": str(root),
            "local_root_present": root.is_dir(),
            "instance_id": instance_id,
            "instance_type": instance_type,
            "observed_evidence": [finding, "local staging/launch artifacts retained", "provider-native output accepted=false"],
            "inference": "none beyond the classification stated above",
            "classification": classification,
            "live_cost_estimate": {"status": "not_reconciled", "usd": None},
            "reconciled_billing": {"status": "required_not_observed", "usd": None},
            "accepted_reconstruction_result": False,
        })
    return attempts


def ledger(arguments: argparse.Namespace) -> None:
    proxy_root = Path(arguments.proxy_root).resolve()
    result = build_attempt_ledger(attempts=_historical_attempts(proxy_root), historical_bakeoff_budget_usd=250.0, historical_postshot_spend_estimate_usd=3.80, generated_at_utc=utc_now_iso())
    path = proxy_root / "provider_packets" / "postshot" / "attempt_ledger.v1.json"
    _write_artifact(path, result)
    print(json.dumps({"ledger": str(path), "schema_version": ATTEMPT_LEDGER_SCHEMA_VERSION, "attempt_count": len(result["attempts"]), "historical_spend_reconciliation_status": result["historical_spend_reconciliation_status"]}, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=["stage", "admit", "launch", "watch", "status", "collect", "abort", "teardown", "inventory", "reconcile", "ledger"])
    parser.add_argument("--proxy-root", required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--dataset-relative", default="trainer_input/colmap_dataset_9de1972eae8fe5ef")
    parser.add_argument("--installer")
    parser.add_argument("--execution-packet")
    parser.add_argument("--authorization-file")
    parser.add_argument("--focused-test-receipt")
    parser.add_argument("--instance-type", choices=INSTANCE_TYPES)
    parser.add_argument("--daemon", action="store_true")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--interval-seconds", type=int, default=PULSE_INTERVAL_SECONDS)
    parser.add_argument("--wait-seconds", type=int, default=120)
    parser.add_argument("--reason")
    parser.add_argument("--billing-json")
    arguments = parser.parse_args()
    if arguments.operation == "stage":
        if not arguments.installer:
            raise SystemExit("--installer required for stage")
        _stage_inputs(arguments)
        return 0
    if not arguments.run_id:
        raise SystemExit("--run-id required for this operation")
    if arguments.operation == "admit":
        admit(arguments)
    elif arguments.operation == "launch":
        launch(arguments)
    elif arguments.operation == "watch":
        watch(arguments)
    elif arguments.operation == "status":
        status(arguments)
    elif arguments.operation == "collect":
        collect(arguments)
    elif arguments.operation == "abort":
        abort(arguments)
    elif arguments.operation == "teardown":
        teardown(arguments)
    elif arguments.operation == "inventory":
        inventory(arguments)
    elif arguments.operation == "reconcile":
        reconcile(arguments)
    elif arguments.operation == "ledger":
        ledger(arguments)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
