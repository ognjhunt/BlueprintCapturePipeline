"""Existing CPU-only provider credential lookup shared with account readers."""
from __future__ import annotations

from pathlib import Path

PROVIDER_SECRETS_DIR_ENV = "BLUEPRINT_GPU_PROVIDER_SECRETS_DIR"


DEFAULT_SECRETS_DIR_NAME = ".blueprint-secrets"


def _developer_secrets_dir() -> Path | None:
    """The interactive fallback, which a hardened service will not have.

    ``ProtectHome`` can make home resolution fail outright, so a missing home
    must read as "no secret here" rather than crashing the provider adapter.
    """
    try:
        return Path.home() / DEFAULT_SECRETS_DIR_NAME
    except (RuntimeError, OSError):
        return None


def _read_secret(name: str) -> str | None:
    """Resolve a provider secret for a service first, a developer shell second.

    Every control-plane unit runs as ``blueprint`` with ``ProtectHome=true`` and
    home ``/nonexistent``, so resolving only ``~/.blueprint-secrets`` -- and
    resolving it once at import time -- meant a hardened service could never
    read a provider key. The units already name the locations explicitly, so
    honour them: the per-secret ``<NAME>_FILE`` override is the most specific
    instruction, then the configured secrets directory, then the developer home.
    """
    import os

    explicit = str(os.getenv(f"{name.upper()}_FILE") or "").strip()
    candidates: list[Path] = [Path(explicit)] if explicit else []

    configured = str(os.getenv(PROVIDER_SECRETS_DIR_ENV) or "").strip()
    if configured:
        # A configured directory is authoritative. Falling through to a
        # developer home from a configured environment would let a local key
        # stand in for one the deployment never installed, which reads as
        # success on the machine that has it and fails everywhere else.
        candidates.append(Path(configured) / name)
    else:
        developer = _developer_secrets_dir()
        if developer is not None:
            candidates.append(developer / name)

    for candidate in candidates:
        try:
            if candidate.is_file():
                return candidate.read_text().strip()
        except OSError:
            continue
    return None

