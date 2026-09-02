"""Provider-side SHA-256 guard fragments for downloaded Vast bundles."""

from __future__ import annotations

import shlex


def provider_bundle_digest_guard(
    expected_sha256: str | None,
    bundle_expression: str,
    mismatch_marker: str,
    success_marker: str,
    *,
    emit_downloaded_marker: bool = False,
) -> str:
    downloaded = (
        "echo BLUEPRINT_VAST_PROVIDER_BUNDLE_DOWNLOADED; "
        if emit_downloaded_marker
        else ""
    )
    if expected_sha256 is None:
        return downloaded + "bundle_digest_rc=0; "
    return (
        f'actual_bundle_sha="sha256:$(sha256sum {bundle_expression} | cut -d" " -f1)"; '
        f"if [ \"$actual_bundle_sha\" != {shlex.quote(expected_sha256)} ]; then "
        f"echo BLUEPRINT_VAST_PROVIDER_BUNDLE_BLOCKED:{mismatch_marker}; "
        "bundle_digest_rc=86; else "
        f"{downloaded}echo {success_marker}; bundle_digest_rc=0; fi; "
    )


__all__ = ["provider_bundle_digest_guard"]
