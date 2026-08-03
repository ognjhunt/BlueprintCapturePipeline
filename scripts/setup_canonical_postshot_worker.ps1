param(
  [Parameter(Mandatory = $true)]
  [string]$WorkerWheel,
  [Parameter(Mandatory = $true)]
  [string]$PostshotVersion,
  [string]$VenvRoot = "C:\work\blueprint-canonical-3dgs-venv",
  [string]$PythonCommand = "python",
  [string]$PostshotCliPath = "C:\Program Files\Jawset Postshot\bin\postshot-cli.exe"
)

$ErrorActionPreference = "Stop"

$wheel = (Resolve-Path -LiteralPath $WorkerWheel).Path
if (-not $wheel.EndsWith(".whl", [System.StringComparison]::OrdinalIgnoreCase)) {
  throw "canonical_worker_wheel_required"
}
$postshotCli = (Resolve-Path -LiteralPath $PostshotCliPath).Path

& $PythonCommand -m venv $VenvRoot
$venvPython = Join-Path $VenvRoot "Scripts\python.exe"
$venvPip = Join-Path $VenvRoot "Scripts\pip.exe"
$transportCli = Join-Path $VenvRoot "Scripts\blueprint-canonical-3dgs-transport.exe"
$workerCli = Join-Path $VenvRoot "Scripts\blueprint-run-canonical-3dgs-arm.exe"

& $venvPython -m pip install --upgrade "pip==26.1.2"
& $venvPip install "numpy==2.4.6" "Pillow==12.3.0"
& $venvPip install --no-deps $wheel
& $transportCli --help | Out-Null
& $workerCli --help | Out-Null

[ordered]@{
  schema_version = "canonical_postshot_worker_runtime_preflight.v1"
  worker_venv = $VenvRoot
  postshot_cli_path = $postshotCli
  trainer_runtime_version = $PostshotVersion
  trainer_runtime_digest = "sha256:$((Get-FileHash -LiteralPath $postshotCli -Algorithm SHA256).Hash.ToLowerInvariant())"
  training_started = $false
} | ConvertTo-Json -Compress
