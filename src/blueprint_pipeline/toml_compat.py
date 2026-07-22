"""Import-safe TOML parsing for minimal Python 3.10 runtime images."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

try:
    import tomllib as _tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    try:
        import tomli as _tomllib
    except ModuleNotFoundError:  # pragma: no cover - sealed minimal runtime
        _tomllib = None  # type: ignore[assignment]


class TOMLParserUnavailable(ValueError):
    """Raised only when a caller actually needs unavailable TOML parsing."""


def load_toml(path: Path) -> Mapping[str, Any]:
    if _tomllib is None:
        raise TOMLParserUnavailable("TOML parser unavailable")
    return _tomllib.loads(path.read_text(encoding="utf-8"))
