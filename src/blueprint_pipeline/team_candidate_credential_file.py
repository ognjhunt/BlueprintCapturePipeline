"""Check an operator's private team-policy credential file without opening it.

The hosted lane hands this file to the provider's private runtime-secret
channel; the effective catalog reports the same finding before any plan. Only
file metadata is inspected; the token is never read here.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path


def credential_file_blockers(path: str | Path | None) -> list[str]:
    """The private token file the lane hands to the provider channel: regular, 0600, owned, bounded."""

    if not path:
        return ["team_candidate_credential_file_missing"]
    candidate = Path(path)
    try:
        status = candidate.lstat()
    except OSError:
        return ["team_candidate_credential_file_missing"]
    if (not candidate.is_absolute() or stat.S_ISLNK(status.st_mode) or not stat.S_ISREG(status.st_mode)
            or status.st_mode & 0o077 or status.st_uid != os.getuid() or not 0 < status.st_size <= 65_536):
        return ["team_candidate_credential_file_not_private"]
    return []


__all__ = ["credential_file_blockers"]
