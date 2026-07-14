#!/usr/bin/env bash
set -euo pipefail

image_ref="${BLUEPRINT_WORKER_IMAGE_REF:-}"
if [[ ! "$image_ref" =~ ^[^[:space:]@]+@sha256:[0-9a-f]{64}$ ]]; then
  echo "BLUEPRINT_WORKER_IMAGE_REF must be digest pinned" >&2
  exit 2
fi

# The selected source image must already contain the provider-specific GPU
# driver. A CPU image builder cannot validate hardware attachment, but it can
# fail before publishing if the driver payload is absent. The promoted image
# still requires a fresh target-GPU canary.
command -v nvidia-smi >/dev/null

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends docker.io nvidia-container-toolkit python3 ca-certificates
nvidia-ctk runtime configure --runtime=docker
systemctl enable docker
systemctl restart docker

# Registry authentication must come from the short-lived builder identity or a
# credential helper. The script never accepts a registry password and removes
# any generated Docker credential file before the disk image is captured.
docker pull "$image_ref"
docker image inspect "$image_ref" >/dev/null

install -d -m 0755 /etc/blueprint /var/lib/blueprint/isaac-cache
printf '%s' "$image_ref" > /etc/blueprint/worker-image-ref
chmod 0444 /etc/blueprint/worker-image-ref

python3 - <<'PY'
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

image = os.environ["BLUEPRINT_WORKER_IMAGE_REF"]
payload = {
    "schema_version": "blueprint_gpu_host_image_manifest.v1",
    "built_at": datetime.now(timezone.utc).isoformat(),
    "worker_image_ref": image,
    "worker_image_ref_sha256": hashlib.sha256(image.encode()).hexdigest(),
    "host_components_baked": {
        "nvidia_driver_payload": True,
        "docker_engine": True,
        "nvidia_container_runtime": True,
        "worker_image_layers": True,
    },
    "runtime_network_download_required": False,
    "claim_boundary": {
        "image_build_is_not_target_gpu_canary": True,
        "image_build_is_not_warm_worker_readiness": True,
        "image_build_is_not_customer_latency_proof": True,
    },
}
Path("/etc/blueprint/gpu-host-image-manifest.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY
chmod 0444 /etc/blueprint/gpu-host-image-manifest.json

rm -f /root/.docker/config.json
apt-get clean
rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# The cached digest is the release payload. Never run docker image prune here.
docker image inspect "$image_ref" >/dev/null
echo "BLUEPRINT_GPU_HOST_IMAGE_BAKE_READY"
