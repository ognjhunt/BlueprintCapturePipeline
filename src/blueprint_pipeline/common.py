"""Compatibility imports for common helpers now canonical in :mod:`core`."""

from .core import common as _common
from .core.common import *  # noqa: F403

# Preserve the established test/diagnostic patch surface without polluting the
# canonical module's public helper list.
os = _common.os
