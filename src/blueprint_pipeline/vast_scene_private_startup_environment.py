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
        "BLUEPRINT_ARTIFIXER_PRETRAINING_CAPSULE_URL",
        # Signed read link to the control-plane stage prefix: site-derived
        # meshes and records. Its sha256 and byte size are not secrets.
        "BLUEPRINT_SCENE_CONFIGURATION_STAGE_PREFIX_CAPSULE_URL",
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
#: Any value that is itself a link (``*_URL``/``*_URI``, or an http(s) value)
#: travels privately too: a signed object-store link gives access to a site's
#: data, and a name added later must not reach the container environment by
#: default.
_LINK_NAME_SUFFIXES = ("_URL", "_URI")
_LINK_VALUE_PREFIXES = ("http://", "https://")


def _is_link(name: str, value: object) -> bool:
    return name.upper().endswith(_LINK_NAME_SUFFIXES) or (
        isinstance(value, str) and value.strip().lower().startswith(_LINK_VALUE_PREFIXES)
    )


def scene_configuration_startup_environments(
    env: Mapping[str, str],
) -> tuple[dict[str, str], dict[str, str]]:
    """Partition credentials and links out of the container-wide environment.

    The container environment is readable for the whole life of the instance
    (provider console, ``docker inspect``, any process in it). Signed links and
    credentials go only to the one-shot private start-up preamble, which the
    on-start shell exports to the processes it starts.
    """

    container_env: dict[str, str] = {}
    private_startup_env: dict[str, str] = {}
    for name, value in env.items():
        private = (
            name.startswith(VAST_RUNTIME_SECRET_BOOTSTRAP_PREFIX)
            or name in VAST_SCENE_CONFIGURATION_PRIVATE_STARTUP_ENV_NAMES
            or _is_link(name, value)
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
