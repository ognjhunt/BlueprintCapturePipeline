"""The date every member of a company's ``simulation.zip`` carries: the capture's own, read from the job's inputs.

Beta journey step: the company package (``website_company_simulation``) zips the simulation of the site. Its members
were dated 1 January 1980, the first date a zip can hold, so that the same inputs always gave the same bytes (the
package's digest is compared); Finder showed every file as made that day (job move-the-lamp-2, 25 Nov). Members now
carry the capture's own date, which never changes for a capture, so the bytes are still the same for the same inputs:

- the time the website recorded for the video, ``capture_start_epoch_ms`` in the capture's ``raw/manifest.json``, else
  in its delivery's ``capture_descriptor.json`` (``metadata.media_metadata``). For a browser upload the WebApp records
  when the upload was received (``capture_start_epoch_ms_basis``: ``server_upload_received``), the phone's own
  recording time being unknown; it is the capture's date either way;
- else when the site's rights for the capture were recorded (``pipeline/website_task_context.json``,
  ``capture_rights.recorded_at_iso``).

Each in UTC, to the second (a zip keeps every second second). One the zip format cannot hold (before 1980 or after
2107), or none, leaves ``ZIP_EPOCH``. Never the time of the build. Reads only; a file it cannot read is passed over.
"""
from __future__ import annotations

import json
import math
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

#: The zip format's first date: what members carried before, and the date where no input gives one.
ZIP_EPOCH = (1980, 1, 1, 0, 0, 0)
#: The years a zip member's date can hold (MS-DOS date: 1980 plus seven bits).
FIRST_YEAR, LAST_YEAR = 1980, 2107


def _read(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8")) if path.is_file() and not path.is_symlink() else {}
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, Mapping) else {}


def _from_epoch_ms(value: Any) -> datetime | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    try:
        return datetime.fromtimestamp(value / 1000.0, tz=timezone.utc)
    except (OverflowError, OSError, ValueError):
        return None


def _from_iso(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        moment = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    return moment.astimezone(timezone.utc) if moment.tzinfo is not None else None


def capture_moment(capture_root: Path) -> datetime | None:
    """The capture's own time from its records, in UTC, or None (the order the module's docstring gives)."""
    root = Path(capture_root)
    media = _read(root / "capture_descriptor.json").get("metadata")
    media = media.get("media_metadata") if isinstance(media, Mapping) else None
    rights = _read(root / "pipeline" / "website_task_context.json").get("capture_rights")
    for moment in (_from_epoch_ms(_read(root / "raw" / "manifest.json").get("capture_start_epoch_ms")),
                   _from_epoch_ms(media.get("capture_start_epoch_ms") if isinstance(media, Mapping) else None),
                   _from_iso(rights.get("recorded_at_iso") if isinstance(rights, Mapping) else None)):
        if moment is not None and FIRST_YEAR <= moment.year <= LAST_YEAR:
            return moment
    return None


def member_date(capture_root: Path | None) -> tuple[int, int, int, int, int, int]:
    """The ``zipfile.ZipInfo.date_time`` of every member: the capture's moment, else ``ZIP_EPOCH``."""
    moment = capture_moment(capture_root) if capture_root is not None else None
    if moment is None:
        return ZIP_EPOCH
    return (moment.year, moment.month, moment.day, moment.hour, moment.minute, moment.second)
