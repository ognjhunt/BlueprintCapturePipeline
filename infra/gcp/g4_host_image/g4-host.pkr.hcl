packer {
  required_plugins {
    googlecompute = {
      version = "= 1.2.1"
      source  = "github.com/hashicorp/googlecompute"
    }
  }
}

variable "project_id" { type = string }
variable "zone" { type = string }
variable "source_image" {
  type        = string
  description = "Exact Ubuntu image name; image families are deliberately forbidden."
  validation {
    condition     = can(regex("^ubuntu-[0-9]+-[a-z]+-v[0-9]{8}$", var.source_image))
    error_message = "source_image must be a date-pinned Ubuntu image, never a family."
  }
}
variable "nvidia_driver_url" { type = string }
variable "nvidia_driver_sha256" {
  type = string
  validation {
    condition     = can(regex("^[0-9a-f]{64}$", var.nvidia_driver_sha256))
    error_message = "nvidia_driver_sha256 must be an exact SHA-256."
  }
}
variable "nvidia_container_toolkit_version" {
  type    = string
  default = "1.17.8-1"
}
variable "image_name" {
  type        = string
  description = "Immutable Blueprint host image identity for this closure."
}

source "googlecompute" "g4_host" {
  project_id              = var.project_id
  zone                    = var.zone
  source_image            = var.source_image
  source_image_project_id = "ubuntu-os-cloud"
  image_name              = var.image_name
  machine_type            = "n1-standard-4"
  disk_size               = 100
  disk_type               = "pd-balanced"
  ssh_username            = "packer"
  tags                    = ["blueprint-image-build"]
}

build {
  sources = ["source.googlecompute.g4_host"]

  provisioner "file" {
    source      = "infra/gcp/g4_host_image/install-pinned-host.sh"
    destination = "/tmp/install-pinned-host.sh"
  }
  provisioner "file" {
    source      = "infra/gcp/g4_host_image/blueprint-g4-host-self-test.sh"
    destination = "/tmp/blueprint-g4-host-self-test.sh"
  }
  provisioner "shell" {
    environment_vars = [
      "NVIDIA_DRIVER_URL=${var.nvidia_driver_url}",
      "NVIDIA_DRIVER_SHA256=${var.nvidia_driver_sha256}",
      "NVIDIA_CONTAINER_TOOLKIT_VERSION=${var.nvidia_container_toolkit_version}",
    ]
    script = "infra/gcp/g4_host_image/install-pinned-host.sh"
  }

  post-processor "manifest" {
    output     = "output/g4-host-packer-manifest.json"
    strip_path = false
  }
}
