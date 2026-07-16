"""Narrow trust boundary for RunPod Serverless network-volume inputs."""

from __future__ import annotations

import os
import urllib.parse
from pathlib import Path


NETWORK_VOLUME_ROOT = Path("/runpod-volume")


def provider_runtime_local_path(uri: str, *, source: str) -> Path:
    """Resolve a provider-local input only inside the attached RunPod volume."""

    enabled = os.getenv("BLUEPRINT_RUNPOD_SERVERLESS_NETWORK_VOLUME_RUNTIME", "")
    if enabled.strip().lower() not in {"1", "true", "yes", "on"}:
        raise ValueError(f"local {source} sources are disabled in provider runtime")
    if urllib.parse.urlparse(uri).scheme:
        raise ValueError(f"local {source} sources are disabled in provider runtime")
    root = NETWORK_VOLUME_ROOT.resolve()
    candidate = Path(uri).expanduser().resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError(f"local {source} sources are disabled in provider runtime")
    return candidate
