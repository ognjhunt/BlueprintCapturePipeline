"""One number: the most one website job may spend (owner decision 2026-11-19, "One cap at $40").

Everything a website job spends counts against it: every paid provider (Gemini, Meta SAM, World Labs,
OpenAI including ARTISAN, Vast) and every retry. The account's monthly limit
(``account_monthly_spend``) still applies on top. Nothing raises the cap: there are no owner grants and
no automatic extensions.

The figure lives in one committed file, ``config/website_job_spend_cap.v1.json``. Blueprint-WebApp
mirrors it byte for byte (``contracts/pipeline/website-job-spend-cap.v1.json``) and enforces it on the
job's ledger; the Pipeline enforces it on each scene intent. Changing it is a money decision for the
owner, made in a reviewed commit to this file and its mirror.

A file that is missing or does not validate raises ``website_job_spend_cap_unavailable``: money fails
closed, so nothing that asks for the cap can admit spend without it.

Standard library only.
"""
from __future__ import annotations

import json
import re
from decimal import Decimal
from functools import lru_cache
from pathlib import Path

SCHEMA_VERSION = "blueprint.website_job_spend_cap.v1"
CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "website_job_spend_cap.v1.json"
#: The plain flag a job carries once a paid step was refused at the cap and the job went on without it.
SPENDING_CAP_REACHED = "spending_cap_reached"
UNAVAILABLE = "website_job_spend_cap_unavailable"
_DOLLARS = re.compile(r"[1-9][0-9]{0,3}(\.[0-9]{1,2})?")
_KEYS = frozenset({"schema_version", "per_job_cap_usd", "scope", "mirrors", "decision"})


def load(path: Path = CONFIG_PATH) -> Decimal:
    """The cap in dollars, read from ``path``; anything unreadable raises ``UNAVAILABLE``."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(UNAVAILABLE) from exc
    figure = value.get("per_job_cap_usd") if isinstance(value, dict) else None
    if (not isinstance(value, dict) or set(value) != _KEYS or value["schema_version"] != SCHEMA_VERSION
            or not isinstance(figure, str) or not _DOLLARS.fullmatch(figure)):
        raise ValueError(UNAVAILABLE)
    return Decimal(figure)


@lru_cache(maxsize=1)
def per_job_cap_usd() -> Decimal:
    """The committed cap, read once per process."""
    return load()
