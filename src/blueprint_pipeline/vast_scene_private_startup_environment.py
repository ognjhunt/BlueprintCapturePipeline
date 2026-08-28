"""Credential-scoped startup environment for retained Vast scene workers."""

from __future__ import annotations

import re
import shlex
from collections.abc import Mapping

from . import vast_runtime_environment_contract as vrec


VAST_RUNTIME_SECRET_BOOTSTRAP_PREFIX = "BLUEPRINT_VAST_RUNTIME_SECRET_B64_"
VAST_SCENE_CONFIGURATION_PRIVATE_STARTUP_ENV_NAMES = frozenset(
    {
        "BLUEPRINT_EVAL_MANIFEST_URI",
        "BLUEPRINT_WORKER_RUNTIME_MANIFEST_SIGNED_PUT_URL",
        "BLUEPRINT_RUNTIME_DEPENDENCY_URI",
        "BLUEPRINT_VAST_PROVIDER_BUNDLE_BASE64",
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
    }
)
SENSITIVE_KEY_MARKERS = (
    "KEY",
    "TOKEN",
    "SECRET",
    "PASSWORD",
    "CREDENTIAL",
    "LOGIN",
    "JUPYTER",
)


def scene_configuration_startup_environments(
    env: Mapping[str, str],
) -> tuple[dict[str, str], dict[str, str]]:
    """Partition credentials out of the container-wide environment."""

    container_env: dict[str, str] = {}
    private_startup_env: dict[str, str] = {}
    for name, value in env.items():
        private = (
            name.startswith(VAST_RUNTIME_SECRET_BOOTSTRAP_PREFIX)
            or name in VAST_SCENE_CONFIGURATION_PRIVATE_STARTUP_ENV_NAMES
            or (
                any(marker in name.upper() for marker in SENSITIVE_KEY_MARKERS)
                and not vrec.is_public_openai_identity_name(name)
            )
        )
        (private_startup_env if private else container_env)[name] = value
    return container_env, private_startup_env


def private_startup_environment_script(env: Mapping[str, str]) -> str:
    """Build the one-shot shell preamble that owns private startup values."""

    if not env:
        return ""
    exports: list[str] = []
    for name, value in sorted(env.items()):
        if re.fullmatch(r"[A-Z][A-Z0-9_]{1,160}", name) is None:
            raise ValueError("invalid_vast_private_startup_environment_name")
        if not isinstance(value, str) or not value or "\x00" in value:
            raise ValueError("invalid_vast_private_startup_environment_value")
        exports.append(f"export {name}={shlex.quote(value)}")
    return (
        "if [ -e /root/onstart.sh ]; then "
        "if [ -L /root/onstart.sh ] || [ ! -f /root/onstart.sh ]; then "
        "echo BLUEPRINT_VAST_PRIVATE_STARTUP_BLOCKED:onstart_path_invalid; exit 91; fi; "
        "rm -f /root/onstart.sh || exit 92; fi; "
        "if [ -e /root/onstart.sh ]; then "
        "echo BLUEPRINT_VAST_PRIVATE_STARTUP_BLOCKED:onstart_path_persisted; exit 93; fi; "
        + "; ".join(exports)
        + "; echo BLUEPRINT_VAST_PRIVATE_STARTUP_ENV_READY; "
    )


__all__ = [
    "SENSITIVE_KEY_MARKERS",
    "VAST_RUNTIME_SECRET_BOOTSTRAP_PREFIX",
    "private_startup_environment_script",
    "scene_configuration_startup_environments",
]
