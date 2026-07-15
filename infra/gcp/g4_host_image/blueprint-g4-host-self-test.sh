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
contract=/etc/blueprint/host-image-contract
test -f "$contract"
worker_ref="$(sed -n 's/^preloaded_worker_image_ref=//p' "$contract")"
worker_digest="$(sed -n 's/^preloaded_worker_image_digest=//p' "$contract")"
worker_source_sha="$(sed -n 's/^worker_source_sha=//p' "$contract")"
outside_worker="$(sed -n 's/^application_or_model_code_outside_worker_image=//p' "$contract")"
[[ "$worker_ref" == *@"$worker_digest" ]]
[[ "$worker_digest" =~ ^sha256:[0-9a-f]{64}$ ]]
[[ "$worker_source_sha" =~ ^[0-9a-f]{40}$ ]]
[[ "$outside_worker" == false ]]
docker image inspect "$worker_ref" >/dev/null
docker image inspect --format '{{join .RepoDigests "\n"}}' "$worker_ref" \
  | grep -Fx -- "$worker_ref" >/dev/null
python3 - "$out" "$started" "$driver" "$gpu" "$docker_version" "$toolkit" \
  "$worker_ref" "$worker_digest" "$worker_source_sha" <<'PY'
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
    "application_or_model_code_outside_worker_image": False,
    "preloaded_worker_image_ref": sys.argv[7],
    "preloaded_worker_image_digest": sys.argv[8],
    "worker_source_sha": sys.argv[9],
    "image_present_before_allocation": True,
    "local_digest_inspect_passed": True,
    "cold_pull_required_during_campaign": False,
}
Path(sys.argv[1]).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
