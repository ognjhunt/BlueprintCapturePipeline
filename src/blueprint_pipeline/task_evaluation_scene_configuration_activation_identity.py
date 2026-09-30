"""Stable activation and launch identities, independent of execution services."""
from __future__ import annotations

import hashlib
import re
from typing import Any

_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,191}")



class SceneConfigurationActivationAutomationError(RuntimeError):
    """A join between preparation, authority, and launch could not be made."""



def _identifier(value: Any) -> str:
    text = str(value or "")
    if _IDENTIFIER.fullmatch(text) is None:
        raise SceneConfigurationActivationAutomationError(
            "scene_configuration_activation_identifier_invalid"
        )
    return text



def _activation_id(preparation_id: str) -> str:
    stem = preparation_id.removesuffix("-preparation")
    return _identifier(f"{stem}-activation-auto")



def _bounded_launch_id(activation_id: str) -> str:
    readable = activation_id + "-launch"
    if _IDENTIFIER.fullmatch(readable) is not None:
        return readable
    prefix = activation_id[:150].rstrip("._-")
    token = hashlib.sha256(activation_id.encode("utf-8")).hexdigest()[:24]
    return _identifier(f"{prefix}-{token}-launch")

