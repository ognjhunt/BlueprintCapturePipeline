"""Bound Vast startup programs below the observed API ceiling."""

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


def args_mode_command(wrapped_script: str, *, force_compression: bool = False) -> str:
    """Return a raw or compressed fail-closed ``bash -lc`` command."""

    raw_args_str = "bash -lc " + shlex.quote(wrapped_script)
    if (
        not force_compression
        and len(raw_args_str.encode("utf-8")) <= VAST_ARGS_STR_SAFE_MAX_BYTES
    ):
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



def onstart_mode_script(script: str, *, force_compression: bool = False) -> str:
    """Return a raw or compressed onstart script under the same ceiling.

    Vast applies the 16384-byte args limit to onstart too: run
    ``...-2ad2f8f6-...-154844Z`` selected offer 39678103 in ssh_direct mode and
    was refused with ``error 400/3471: Invalid args: len(image) > 1024, or
    len(args) > 16384, or len(label) > 256`` while its args_str stayed bounded
    -- the oversized member was the uncompressed onstart branch, which the
    args-mode fix did not touch. Unlike args mode there is no ``bash -lc``
    wrapper: onstart is already executed as a script, so the compressed form is
    itself a script that decodes and runs the real one under pipefail.
    """

    if not force_compression and len(script.encode("utf-8")) <= VAST_ARGS_STR_SAFE_MAX_BYTES:
        return script
    encoded_script = base64.b64encode(
        gzip.compress(script.encode("utf-8"), mtime=0)
    ).decode("ascii")
    compressed = (
        "set -o pipefail; "
        + VAST_ARGS_GZIP_BASE64_MARKER
        + shlex.quote(encoded_script)
        + '; printf %s "$BLUEPRINT_VAST_ARGS_GZIP_BASE64_V1" '
        "| base64 -d | gzip -dc | bash"
    )
    if len(compressed.encode("utf-8")) > VAST_ARGS_STR_SAFE_MAX_BYTES:
        raise ValueError("vast_onstart_exceeds_safe_inline_command_size")
    return compressed


__all__ = [
    "VAST_ARGS_GZIP_BASE64_MARKER",
    "VAST_ARGS_STR_SAFE_MAX_BYTES",
    "args_mode_command",
    "onstart_mode_script",
]
