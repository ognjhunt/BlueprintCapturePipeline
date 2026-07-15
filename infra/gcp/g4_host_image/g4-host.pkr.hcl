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
    condition     = can(regex("^ubuntu-[0-9]+-[a-z0-9-]+-v[0-9]{8}$", var.source_image))
    error_message = "Source image must be a date-pinned Ubuntu image, never a family."
  }
}
variable "nvidia_driver_url" { type = string }
variable "nvidia_driver_sha256" {
  type = string
  validation {
    condition     = can(regex("^[0-9a-f]{64}$", var.nvidia_driver_sha256))
    error_message = "NVIDIA driver SHA-256 must be an exact lowercase digest."
  }
}
variable "nvidia_container_toolkit_version" {
  type    = string
  default = "1.19.1-1"
}
variable "image_name" {
  type        = string
  description = "Immutable Blueprint host image identity for this closure."
}
variable "worker_image_digest_ref" {
  type        = string
  description = "Exact worker registry reference to preload into Docker's immutable content store."
  validation {
    condition     = can(regex("^[^[:space:]]+@sha256:[0-9a-f]{64}$", var.worker_image_digest_ref))
    error_message = "Worker image digest ref must be an exact registry digest reference."
  }
}
variable "worker_source_sha" {
  type        = string
  description = "Exact protected-main source revision bound to the preloaded worker."
  validation {
    condition     = can(regex("^[0-9a-f]{40}$", var.worker_source_sha))
    error_message = "Worker source SHA must be a full lowercase git SHA."
  }
}

source "googlecompute" "g4_host" {
  project_id              = var.project_id
  zone                    = var.zone
  source_image            = var.source_image
  source_image_project_id = ["ubuntu-os-cloud"]
  image_name              = var.image_name
  machine_type            = "n1-standard-4"
  disk_size               = 300
  disk_type               = "pd-balanced"
  ssh_username            = "packer"
  tags                    = ["blueprint-image-build"]
  scopes                  = ["https://www.googleapis.com/auth/cloud-platform"]
  labels = {
    blueprint-managed = "true"
    blueprint-lane    = "g4-host-image-build"
  }
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
      "WORKER_IMAGE_DIGEST_REF=${var.worker_image_digest_ref}",
      "WORKER_SOURCE_SHA=${var.worker_source_sha}",
    ]
    script = "infra/gcp/g4_host_image/install-pinned-host.sh"
  }

  post-processor "manifest" {
    output     = "output/g4-host-packer-manifest.json"
    strip_path = false
  }
}
