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


#: Opt-in for ``blueprint_download_url``'s curl tier: when the caller names a
#: file in ``blueprint_download_report``, curl writes its last pass's HTTP status
#: and exit code there (``%output`` needs curl 8.3+; the scene image installs
#: 8.5). Unset (every bundle download) it expands to nothing, so curl runs
#: exactly as before.
DOWNLOAD_REPORT_OPTION = (
    '${blueprint_download_report:+--write-out "%output{$blueprint_download_report}%{http_code} %{exitcode}"}'
)
#: Where the scene-configuration on-start puts the control-plane stage prefix.
STAGE_PREFIX_CAPSULE_FILENAME = "task_evaluation_scene_configuration_stage_prefix_capsule.zip"


def stage_prefix_capsule_step() -> str:
    """Fetch and check the stage-prefix capsule like the bundle, then open the runner's step.

    Runs in the scene-configuration on-start after the bundle is unpacked.

    - The binding is checked first, with no network: an absent sha256 or size
      (an older host) stops as ``..._binding_incomplete``; a sha256 that is not
      ``sha256:`` and 64 lower-case hex digits, or a size that is not a positive
      decimal, stops as ``..._binding_invalid``.
    - Any earlier file at the capsule path is removed, so curl never resumes a
      stale partial. The same ``blueprint_download_url`` helper (curl over
      HTTP/1.1 with resume and retries) downloads it into the work dir.
    - A failed transfer stops as ``..._download_failed:<helper rc>:http_<status>:curl_<exit>``
      from curl's own report (digits only, never the link; ``unknown`` when a
      non-curl tier ran) and the partial file is removed.
    - Size and sha256 are compared in shell; a mismatch stops as
      ``..._digest_mismatch`` and removes the file. Only then is the local
      path exported for the runner, which reads that file and never the network.

    Every stop happens before the runner, as a failed bundle download does. The
    lines around the transfer are plain (no ``_PROGRESS:``), so the no-progress
    watchdog counts them without switching to structured-marker tracking
    before the entrypoint's ticker.

    Leaves an ``if ... else`` open around the entrypoint: the else branch
    exports ``PYTHONDONTWRITEBYTECODE=1`` (root ignores the sealed toolchain's
    read-only modes) and the caller closes it with one more ``fi``.
    """

    from .task_evaluation_scene_configuration_cpu_prestage import (
        PREFIX_BYTES_ENV,
        PREFIX_PATH_ENV,
        PREFIX_SHA_ENV,
        PREFIX_URL_ENV,
    )

    return (
        "prefix_capsule_rc=0; prefix_capsule_blocker=''; "
        f'if [ -n "${{{PREFIX_URL_ENV}:-}}" ]; then '
        f'PREFIX_CAPSULE_FILE="$WORK_DIR/{STAGE_PREFIX_CAPSULE_FILENAME}"; '
        'PREFIX_CAPSULE_TRANSFER="$PREFIX_CAPSULE_FILE.transfer"; '
        'rm -f "$PREFIX_CAPSULE_FILE" "$PREFIX_CAPSULE_TRANSFER"; '
        f'prefix_capsule_expected_sha="${{{PREFIX_SHA_ENV}:-}}"; '
        f'prefix_capsule_expected_bytes="${{{PREFIX_BYTES_ENV}:-}}"; '
        'prefix_capsule_hex="${prefix_capsule_expected_sha#sha256:}"; '
        'if [ -z "$prefix_capsule_expected_sha" ] || [ -z "$prefix_capsule_expected_bytes" ]; then '
        "prefix_capsule_rc=86; prefix_capsule_blocker=scene_configuration_stage_prefix_binding_incomplete; "
        'elif [ "$prefix_capsule_expected_sha" != "sha256:$prefix_capsule_hex" ] || [ "${#prefix_capsule_hex}" -ne 64 ]; then '
        "prefix_capsule_rc=86; prefix_capsule_blocker=scene_configuration_stage_prefix_binding_invalid; "
        "fi; "
        'case "$prefix_capsule_hex" in *[!0-9a-f]*) prefix_capsule_rc=86; '
        "prefix_capsule_blocker=${prefix_capsule_blocker:-scene_configuration_stage_prefix_binding_invalid};; esac; "
        'case "$prefix_capsule_expected_bytes" in *[!0-9]*|0*) prefix_capsule_rc=86; '
        "prefix_capsule_blocker=${prefix_capsule_blocker:-scene_configuration_stage_prefix_binding_invalid};; esac; "
        "if [ $prefix_capsule_rc -eq 0 ]; then "
        "echo BLUEPRINT_VAST_SCENE_CONFIGURATION_STAGE_PREFIX_DOWNLOAD_STARTED; "
        'blueprint_download_report="$PREFIX_CAPSULE_TRANSFER"; '
        f'blueprint_download_url "${PREFIX_URL_ENV}" "$PREFIX_CAPSULE_FILE"; prefix_capsule_rc=$?; '
        "blueprint_download_report=''; "
        "echo BLUEPRINT_VAST_SCENE_CONFIGURATION_STAGE_PREFIX_DOWNLOAD_EXIT_CODE:$prefix_capsule_rc; "
        'prefix_capsule_transfer=$(cat "$PREFIX_CAPSULE_TRANSFER" 2>/dev/null); rm -f "$PREFIX_CAPSULE_TRANSFER"; '
        'prefix_capsule_http="${prefix_capsule_transfer%% *}"; prefix_capsule_curl="${prefix_capsule_transfer##* }"; '
        'case "$prefix_capsule_http" in ""|*[!0-9]*) prefix_capsule_http=unknown;; esac; '
        'case "$prefix_capsule_curl" in ""|*[!0-9]*) prefix_capsule_curl=unknown;; esac; '
        "if [ $prefix_capsule_rc -ne 0 ]; then "
        "prefix_capsule_blocker=scene_configuration_stage_prefix_download_failed:$prefix_capsule_rc:"
        "http_$prefix_capsule_http:curl_$prefix_capsule_curl; "
        'rm -f "$PREFIX_CAPSULE_FILE"; '
        "else "
        "prefix_capsule_actual_bytes=$(wc -c < \"$PREFIX_CAPSULE_FILE\" | tr -d '[:space:]'); "
        'prefix_capsule_actual_sha="sha256:$(sha256sum "$PREFIX_CAPSULE_FILE" | cut -d" " -f1)"; '
        'if [ "$prefix_capsule_actual_bytes" != "$prefix_capsule_expected_bytes" ] '
        '|| [ "$prefix_capsule_actual_sha" != "$prefix_capsule_expected_sha" ]; then '
        "prefix_capsule_rc=86; prefix_capsule_blocker=scene_configuration_stage_prefix_digest_mismatch; "
        'rm -f "$PREFIX_CAPSULE_FILE"; '
        "else "
        f'export {PREFIX_PATH_ENV}="$PREFIX_CAPSULE_FILE"; '
        "echo BLUEPRINT_VAST_SCENE_CONFIGURATION_STAGE_PREFIX_SHA256_VERIFIED; "
        "fi; fi; fi; fi; "
        "if [ $prefix_capsule_rc -ne 0 ]; then "
        "echo BLUEPRINT_VAST_PROVIDER_BUNDLE_BLOCKED:$prefix_capsule_blocker; "
        "else export PYTHONDONTWRITEBYTECODE=1; "
    )


__all__ = [
    "DOWNLOAD_REPORT_OPTION",
    "STAGE_PREFIX_CAPSULE_FILENAME",
    "provider_bundle_digest_guard",
    "stage_prefix_capsule_step",
]
