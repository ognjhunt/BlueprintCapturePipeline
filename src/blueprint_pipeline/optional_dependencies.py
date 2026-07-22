"""Compatibility imports for optional dependency helpers.

New code should import from :mod:`blueprint_pipeline.core.optional_dependencies`.
"""

from .core.optional_dependencies import install_extra_hint, log_missing_optional_dependency

__all__ = ["install_extra_hint", "log_missing_optional_dependency"]
