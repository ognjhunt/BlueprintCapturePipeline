#!/usr/bin/env bash
set -euo pipefail

: "${NVIDIA_DRIVER_URL:?required}"
: "${NVIDIA_DRIVER_SHA256:?required}"
: "${NVIDIA_CONTAINER_TOOLKIT_VERSION:?required}"

driver=/tmp/nvidia-driver.run
curl --fail --location --proto '=https' --tlsv1.2 "$NVIDIA_DRIVER_URL" -o "$driver"
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
sudo apt-get update
sudo apt-get install -y --no-install-recommends \
  docker.io \
  "nvidia-container-toolkit=${NVIDIA_CONTAINER_TOOLKIT_VERSION}" \
  "nvidia-container-toolkit-base=${NVIDIA_CONTAINER_TOOLKIT_VERSION}" \
  "libnvidia-container-tools=${NVIDIA_CONTAINER_TOOLKIT_VERSION}" \
  "libnvidia-container1=${NVIDIA_CONTAINER_TOOLKIT_VERSION}"
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl enable docker

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

# The host closure contains only driver/container infrastructure. Application,
# model, task, and capture bytes remain exclusively in immutable worker images
# or explicitly hashed runtime bundles.
sudo mkdir -p /etc/blueprint
printf '%s\n' 'application_or_model_code_baked=false' \
  | sudo tee /etc/blueprint/host-image-contract >/dev/null
