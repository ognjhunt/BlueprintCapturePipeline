#!/usr/bin/env bash
set -euo pipefail

out=/var/lib/blueprint/host-self-test.json
mkdir -p "$(dirname "$out")"
started="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
driver="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
gpu="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
docker_version="$(docker version --format '{{.Server.Version}}')"
docker info --format '{{json .Runtimes}}' | grep -q 'nvidia'
toolkit="$(nvidia-ctk --version | head -1)"
python3 - "$out" "$started" "$driver" "$gpu" "$docker_version" "$toolkit" <<'PY'
import json, sys
from pathlib import Path
payload = {
    "schema_version": "blueprint_g4_host_self_test.v1",
    "status": "passed",
    "started_at": sys.argv[2],
    "completed_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
    "nvidia_driver": sys.argv[3],
    "gpu": sys.argv[4],
    "docker_version": sys.argv[5],
    "nvidia_container_toolkit": sys.argv[6],
    "docker_nvidia_runtime_configured": True,
    "application_or_model_code_baked": False,
}
Path(sys.argv[1]).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
