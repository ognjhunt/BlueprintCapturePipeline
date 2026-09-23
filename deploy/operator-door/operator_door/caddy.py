"""Add the operator-door route to a Caddyfile without rewriting anything else.

The host's live Caddyfile names its site literally and has drifted from the
repository copy, so the installer patches the live file in place: one
``handle /api/live-pipeline/operator/*`` block inserted just before the intake's
``handle /api/live-pipeline/*`` block, with matching indentation.
"""

from __future__ import annotations

import re

ROUTE = "handle /api/live-pipeline/operator/*"
UPSTREAM = "127.0.0.1:8767"
_ANCHOR = re.compile(r"^(\s*)handle /api/live-pipeline/\*")


class CaddyPatchError(ValueError):
    pass


def patch_caddyfile(text: str) -> str | None:
    """Return the patched text, or ``None`` when the route is already present."""

    if ROUTE in text:
        return None
    lines = text.splitlines(keepends=True)
    for index, line in enumerate(lines):
        match = _ANCHOR.match(line)
        if match is None:
            continue
        indent = match.group(1)
        step = "\t" if (indent.startswith("\t") or not indent) else " " * max(len(indent), 4)
        block = [f"{indent}{ROUTE} {{\n", f"{indent}{step}reverse_proxy {UPSTREAM}\n", f"{indent}}}\n"]
        return "".join(lines[:index] + block + lines[index:])
    raise CaddyPatchError("caddy_anchor_missing")
