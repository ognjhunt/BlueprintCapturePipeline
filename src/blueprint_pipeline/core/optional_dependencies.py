"""Helpers for consistent optional dependency messaging."""

from __future__ import annotations

import logging


def install_extra_hint(extra: str) -> str:
    return (
        f"Install optional `{extra}` dependencies with "
        f"`uv sync --extra {extra}` or `pip install -e .[{extra}]`."
    )


def log_missing_optional_dependency(
    logger: logging.Logger,
    *,
    feature: str,
    package: str,
    extra: str,
) -> str:
    message = f"{feature} requires optional dependency `{package}`. {install_extra_hint(extra)}"
    logger.warning(message)
    return message
