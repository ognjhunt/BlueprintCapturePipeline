"""Safe local env-file loading for launch gates.

The helper intentionally returns only file paths and key names. Secret values are
loaded into the process environment for child commands but are never included in
the summary.
"""

from __future__ import annotations

import os
import re
import shlex
from pathlib import Path
from typing import Iterable, Mapping


DEFAULT_ENV_FILENAMES = (".env", ".env.local", ".env.alpha.local")
_ENV_KEY_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_PLACEHOLDER_MARKERS = (
    "REPLACE_ME",
    "replace_me",
    "example.com",
    "your-project",
    "your-backend",
    "your-token",
    "your-api-key",
)
_LIVE_TEST_ENV_PREFIXES = (
    "PIPELINE_SYNC_",
    "PRIVACY_",
    "WORLDLABS_",
    "BLUEPRINT_LAUNCH_",
    "SITE_WORLD_RUNTIME_",
    "VIDEO_TO_WORLD_",
)
_LIVE_TEST_ENV_KEYS = {
    "BLUEPRINT_PREVIEW_PROVIDER",
    "BLUEPRINT_ALLOW_RAW_WORLDLABS_BYPASS",
    "GCS_ROOT",
    "GEMINI_API_KEY",
    "GOOGLE_GENAI_API_KEY",
    "PIPELINE_ALPHA_EXPECT_HOSTED_RUNTIME",
    "PIPELINE_BUCKET",
    "PIPELINE_PROJECT_ID",
    "PIPELINE_REGION",
}


def _parse_value(raw_value: str) -> str:
    lexer = shlex.shlex(raw_value, posix=True)
    lexer.whitespace_split = True
    lexer.commenters = "#"
    parts = list(lexer)
    if not parts:
        return ""
    return parts[0]


def _parse_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line.removeprefix("export ").strip()
        if "=" not in line:
            continue
        key, raw_value = line.split("=", 1)
        key = key.strip()
        if not _ENV_KEY_PATTERN.match(key):
            continue
        values[key] = _parse_value(raw_value.strip())
    return values


def _is_placeholder_value(value: str) -> bool:
    return any(marker in value for marker in _PLACEHOLDER_MARKERS)


def load_env_files(
    roots: Iterable[Path],
    *,
    filenames: Iterable[str] = DEFAULT_ENV_FILENAMES,
    environ: Mapping[str, str] | None = None,
) -> dict[str, list[str]]:
    """Load local env files into ``os.environ`` while preserving exported vars.

    Files are read in the supplied root order and filename order. Later files
    override earlier files in the local overlay, but variables that were already
    exported in the process before loading are not overwritten.
    """

    original_env = dict(os.environ if environ is None else environ)
    overlay: dict[str, str] = {}
    loaded_files: list[str] = []
    for root in roots:
        for filename in filenames:
            path = Path(root).expanduser().resolve() / filename
            if not path.is_file():
                continue
            loaded_files.append(str(path))
            overlay.update(_parse_env_file(path))

    loaded_keys: list[str] = []
    skipped_existing_keys: list[str] = []
    skipped_placeholder_keys: list[str] = []
    for key, value in overlay.items():
        if _is_placeholder_value(value):
            skipped_placeholder_keys.append(key)
            continue
        if key in original_env:
            skipped_existing_keys.append(key)
            continue
        os.environ[key] = value
        loaded_keys.append(key)

    return {
        "files": loaded_files,
        "loaded_keys": sorted(loaded_keys),
        "skipped_existing_keys": sorted(skipped_existing_keys),
        "skipped_placeholder_keys": sorted(skipped_placeholder_keys),
    }


def contract_test_env() -> dict[str, str]:
    """Return an env for hermetic local contract tests.

    Launch scripts may load live/local env files for real checks, but pytest
    contract suites must not silently call live privacy, provider, runtime, or
    WebApp sync endpoints.
    """

    env = os.environ.copy()
    for key in list(env):
        if key in _LIVE_TEST_ENV_KEYS or any(key.startswith(prefix) for prefix in _LIVE_TEST_ENV_PREFIXES):
            env.pop(key, None)
    return env
