"""Which money model a release runs for website jobs: one switch, read in one place.

Owner decision 2026-11-11 ("yes replace them - $300 monthly limit, $100 fuse"). The code for the
new model ships switched off and is turned on by a reviewed one-line change to the committed
config file ``config/website_job_limits.v1.json`` (a normal pull request and release), so turning
it off again is a config change too, never a code rollback.

``nested`` (the shipped value): every amount, plan, site limit, quote ceiling and message is as it
was before 2026-11-11: a website plan budgets the smaller of $15 and what the job's ceiling leaves,
a site's screening limit is $30, and no start check runs.

``fuse``: a website plan budgets everything the job's ceiling leaves; a site's limit is the
largest job ceiling its campaigns recorded (``beta_policy_screening_site_limit``); a row the
ceiling cannot fund names the ceiling; a new job starts its paid work only when the month has
room for it (``website_job_start_room``).

Always on, in both modes, so a release is a safe one to switch back to: one site past its limit
holds only itself, and a ledger that recorded job ceilings above $30 reads without raising (in
``nested`` such a site counts at $30, so it is held and commits nothing more).

The file is found and read like the account limit's (``account_monthly_spend.config_path``): the deploy
installs it with the promoted source tree, read at ``INSTALLED_CONFIG_PATH`` (the active release's own
``config/``), unless ``BLUEPRINT_WEBSITE_JOB_LIMITS_CONFIG`` names another file (tests); a regular file,
never a symlink, small, strictly validated. A missing, unreadable or unknown value raises
``WebsiteJobLimitsInvalid`` wherever the mode is asked, in one line naming the file: money never silently
picks a mode. Kept free of package imports so any lane can ask it.
"""
from __future__ import annotations

import json
import os
import stat
from pathlib import Path

SCHEMA = "blueprint.website_job_limits.v1"
CONFIG_ENV = "BLUEPRINT_WEBSITE_JOB_LIMITS_CONFIG"
CONFIG_FILENAME = "website_job_limits.v1.json"
#: Where the deploy installs it, exactly as ``account_monthly_spend.INSTALLED_CONFIG_PATH``.
INSTALLED_CONFIG_PATH = Path("/opt/blueprint/task-evaluation-control-plane/config") / CONFIG_FILENAME
#: The committed file in this source tree (what a release carries to ``INSTALLED_CONFIG_PATH``).
SHIPPED_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / CONFIG_FILENAME
NESTED, FUSE = "nested", "fuse"
MODES = frozenset({NESTED, FUSE})
_MAX_BYTES = 16 * 1024
_KEYS = {"schema_version", "mode", "decision"}
_cache: dict[tuple[str, int, int], str] = {}


class WebsiteJobLimitsInvalid(ValueError):
    """The switch file is missing, unreadable or names no known mode."""


def config_path() -> Path:
    configured = str(os.environ.get(CONFIG_ENV) or "").strip()
    return Path(configured) if configured else INSTALLED_CONFIG_PATH


def limits_mode() -> str:
    """``nested`` or ``fuse``; raises ``WebsiteJobLimitsInvalid`` for anything else."""
    path = config_path()
    try:
        info = os.lstat(path)
        if not stat.S_ISREG(info.st_mode) or info.st_size > _MAX_BYTES:
            raise WebsiteJobLimitsInvalid(f"website_job_limits_config_unsafe: {path}")
        key = (str(path), info.st_mtime_ns, info.st_size)
        if key in _cache:
            return _cache[key]
        value = json.loads(path.read_text(encoding="utf-8"))
    except WebsiteJobLimitsInvalid:
        raise
    except FileNotFoundError as exc:
        raise WebsiteJobLimitsInvalid(f"website_job_limits_config_missing: {path}") from exc
    except (OSError, ValueError, UnicodeDecodeError) as exc:
        raise WebsiteJobLimitsInvalid(f"website_job_limits_config_unreadable: {path}") from exc
    if (not isinstance(value, dict) or set(value) - _KEYS or value.get("schema_version") != SCHEMA
            or value.get("mode") not in MODES or not isinstance(value.get("decision", {}), dict)):
        raise WebsiteJobLimitsInvalid(f"website_job_limits_mode_invalid: {path}")
    _cache.clear()
    _cache[key] = value["mode"]
    return value["mode"]


def fuse() -> bool:
    """Whether the 2026-11-11 model (testing up to each job's ceiling, the start check) is on."""
    return limits_mode() == FUSE


__all__ = ["CONFIG_ENV", "CONFIG_FILENAME", "INSTALLED_CONFIG_PATH", "FUSE", "MODES", "NESTED", "SHIPPED_CONFIG_PATH", "WebsiteJobLimitsInvalid", "config_path",
           "fuse", "limits_mode"]
