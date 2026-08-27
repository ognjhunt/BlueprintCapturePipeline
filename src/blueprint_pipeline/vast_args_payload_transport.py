"""Bound Vast args-mode startup programs below the observed API ceiling."""

from __future__ import annotations

import base64
import gzip
import shlex

# Vast accepted run ``...-6e9b81ed-...`` at 15,538 bytes, then returned an empty
# HTTP 400 for the first 16,412-byte command after CUDA provisioning. Vast's
# CLI guidance says long scripts must be gzip/base64 encoded; keep headroom
# below that observed boundary instead of sending an ever-growing program.
VAST_ARGS_STR_SAFE_MAX_BYTES = 16_000
VAST_ARGS_GZIP_BASE64_MARKER = "BLUEPRINT_VAST_ARGS_GZIP_BASE64_V1="


def args_mode_command(wrapped_script: str) -> str:
    """Return a raw or compressed fail-closed ``bash -lc`` command."""

    raw_args_str = "bash -lc " + shlex.quote(wrapped_script)
    if len(raw_args_str.encode("utf-8")) <= VAST_ARGS_STR_SAFE_MAX_BYTES:
        return raw_args_str

    # Runtime secrets remain in Vast's separate environment map. Deterministic
    # gzip keeps the request small while pipefail preserves decode and script
    # failures. Do not move this into ``onstart``: args mode is what preserves
    # the Isaac image entrypoint.
    encoded_script = base64.b64encode(
        gzip.compress(wrapped_script.encode("utf-8"), mtime=0)
    ).decode("ascii")
    compressed_wrapper = (
        "set -o pipefail; "
        + VAST_ARGS_GZIP_BASE64_MARKER
        + shlex.quote(encoded_script)
        + '; printf %s "$BLUEPRINT_VAST_ARGS_GZIP_BASE64_V1" '
        "| base64 -d | gzip -dc | bash"
    )
    compressed_args_str = "bash -lc " + shlex.quote(compressed_wrapper)
    if len(compressed_args_str.encode("utf-8")) > VAST_ARGS_STR_SAFE_MAX_BYTES:
        raise ValueError("vast_args_str_exceeds_safe_inline_command_size")
    return compressed_args_str


__all__ = [
    "VAST_ARGS_GZIP_BASE64_MARKER",
    "VAST_ARGS_STR_SAFE_MAX_BYTES",
    "args_mode_command",
]
