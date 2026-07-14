#!/usr/bin/env bash
set -euo pipefail

: "${NVIDIA_DRIVER_URL:?required}"
: "${NVIDIA_DRIVER_SHA256:?required}"
: "${NVIDIA_CONTAINER_TOOLKIT_VERSION:?required}"
: "${WORKER_IMAGE_DIGEST_REF:?required}"
: "${WORKER_SOURCE_SHA:?required}"

if [[ ! "$WORKER_IMAGE_DIGEST_REF" =~ ^[^[:space:]]+@sha256:[0-9a-f]{64}$ ]]; then
  echo "WORKER_IMAGE_DIGEST_REF must be immutable" >&2
  exit 2
fi
if [[ ! "$WORKER_SOURCE_SHA" =~ ^[0-9a-f]{40}$ ]]; then
  echo "WORKER_SOURCE_SHA must be a full lowercase git SHA" >&2
  exit 2
fi

sudo apt-get update
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
  "linux-headers-$(uname -r)" \
  build-essential \
  ca-certificates \
  curl \
  dkms \
  gnupg

driver=/tmp/nvidia-driver.run
driver_curl_args=(--fail --location --proto '=https' --tlsv1.2)
if [[ "$NVIDIA_DRIVER_URL" == https://storage.googleapis.com/gce-nvidia-vgpu-drivers/* ]]; then
  metadata_token="$(curl --fail --silent --show-error \
    -H 'Metadata-Flavor: Google' \
    'http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token' \
    | python3 -c 'import json,sys; print(json.load(sys.stdin)["access_token"])')"
  driver_curl_args+=(-H "Authorization: Bearer $metadata_token")
fi
curl "${driver_curl_args[@]}" "$NVIDIA_DRIVER_URL" -o "$driver"
unset metadata_token driver_curl_args
printf '%s  %s\n' "$NVIDIA_DRIVER_SHA256" "$driver" | sha256sum --check --strict
chmod 0700 "$driver"
sudo "$driver" --silent --dkms
rm -f "$driver"

curl --fail --location --proto '=https' --tlsv1.2 \
  https://nvidia.github.io/libnvidia-container/gpgkey \
  | gpg --dearmor | sudo tee /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg >/dev/null
curl --fail --location --proto '=https' --tlsv1.2 \
  https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
sudo apt-get install -y --no-install-recommends \
  docker.io \
  "nvidia-container-toolkit=${NVIDIA_CONTAINER_TOOLKIT_VERSION}" \
  "nvidia-container-toolkit-base=${NVIDIA_CONTAINER_TOOLKIT_VERSION}" \
  "libnvidia-container-tools=${NVIDIA_CONTAINER_TOOLKIT_VERSION}" \
  "libnvidia-container1=${NVIDIA_CONTAINER_TOOLKIT_VERSION}"
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl enable docker
sudo systemctl start docker

# Keep the exact worker closure in Docker's content-addressed store. No files
# from the image are copied onto the host filesystem or installed independently.
sudo docker pull "$WORKER_IMAGE_DIGEST_REF"
sudo docker image inspect "$WORKER_IMAGE_DIGEST_REF" >/dev/null
resolved_digest="${WORKER_IMAGE_DIGEST_REF##*@}"
sudo docker image inspect --format '{{join .RepoDigests "\n"}}' "$WORKER_IMAGE_DIGEST_REF" \
  | grep -Fx -- "$WORKER_IMAGE_DIGEST_REF" >/dev/null

sudo install -D -m 0755 /tmp/blueprint-g4-host-self-test.sh \
  /usr/local/sbin/blueprint-g4-host-self-test
sudo tee /etc/systemd/system/blueprint-g4-host-self-test.service >/dev/null <<'UNIT'
[Unit]
Description=Blueprint pinned G4 host startup self-test
After=docker.service
Requires=docker.service

[Service]
Type=oneshot
ExecStart=/usr/local/sbin/blueprint-g4-host-self-test
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
UNIT
sudo systemctl enable blueprint-g4-host-self-test.service

sudo mkdir -p /etc/blueprint
sudo tee /etc/blueprint/host-image-contract >/dev/null <<EOF
application_or_model_code_outside_worker_image=false
preloaded_worker_image_ref=$WORKER_IMAGE_DIGEST_REF
preloaded_worker_image_digest=$resolved_digest
worker_source_sha=$WORKER_SOURCE_SHA
EOF
