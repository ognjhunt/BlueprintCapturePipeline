"""Fail-closed provider-output upload transport shared by Vast lanes."""

from __future__ import annotations


EXPECTED_PROVIDER_UPLOAD_BYTES_ENV = (
    "BLUEPRINT_VAST_EXPECTED_PROVIDER_UPLOAD_BYTES"
)


def provider_output_upload_shell_fragment() -> str:
    """Return the bounded, fail-closed provider-output upload function.

    A provider output is an immutable object: retrying the same bytes to the
    same signed PUT URL is idempotent, while rerunning the provider workload is
    not.  Keep retries here at the transport boundary and admit only explicit
    transient curl failures or transient HTTP statuses.
    """

    return (
        "blueprint_upload_put() { "
        'blueprint_upload_url="$1"; blueprint_upload_path="$2"; '
        f'blueprint_upload_limit="${{{EXPECTED_PROVIDER_UPLOAD_BYTES_ENV}:-0}}"; '
        'if [ ! -f "$blueprint_upload_path" ]; then '
        "echo BLUEPRINT_VAST_PROVIDER_BUNDLE_BLOCKED:provider_output_zip_missing; return 86; fi; "
        'blueprint_upload_bytes=$(wc -c < "$blueprint_upload_path" | tr -d \'[:space:]\'); '
        'case "$blueprint_upload_limit" in \'\'|*[!0-9]*) '
        "echo BLUEPRINT_VAST_PROVIDER_BUNDLE_BLOCKED:provider_output_transfer_ceiling_invalid; return 86;; esac; "
        'case "$blueprint_upload_bytes" in \'\'|*[!0-9]*) '
        "echo BLUEPRINT_VAST_PROVIDER_BUNDLE_BLOCKED:provider_output_zip_size_invalid; return 86;; esac; "
        'if [ "$blueprint_upload_limit" -gt 0 ] && [ "$blueprint_upload_bytes" -gt "$blueprint_upload_limit" ]; then '
        "echo BLUEPRINT_VAST_PROVIDER_BUNDLE_BLOCKED:provider_output_zip_exceeds_declared_transfer_ceiling; return 86; fi; "
        "if ! command -v curl >/dev/null 2>&1; then "
        "echo BLUEPRINT_VAST_PROVIDER_BUNDLE_BLOCKED:provider_output_upload_transport_unavailable; return 127; fi; "
        "if ! command -v sha256sum >/dev/null 2>&1; then "
        "echo BLUEPRINT_VAST_PROVIDER_BUNDLE_BLOCKED:provider_output_upload_digest_tool_unavailable; return 127; fi; "
        'blueprint_upload_sha256=$(sha256sum "$blueprint_upload_path" | cut -d" " -f1); '
        'case "$blueprint_upload_sha256" in \'\'|*[!0-9a-f]*) '
        "echo BLUEPRINT_VAST_PROVIDER_BUNDLE_BLOCKED:provider_output_upload_digest_invalid; return 86;; esac; "
        'if [ "${#blueprint_upload_sha256}" -ne 64 ]; then '
        "echo BLUEPRINT_VAST_PROVIDER_BUNDLE_BLOCKED:provider_output_upload_digest_invalid; return 86; fi; "
        'blueprint_upload_deadline=$(( $(date +%s) + 1200 )); '
        'blueprint_parent_deadline="${BLUEPRINT_SCENE_CONFIGURATION_PARENT_DEADLINE_EPOCH:-}"; '
        'if [ -n "$blueprint_parent_deadline" ]; then case "$blueprint_parent_deadline" in '
        '*.*) blueprint_parent_deadline_integer=${blueprint_parent_deadline%%.*}; '
        'blueprint_parent_deadline_fraction=${blueprint_parent_deadline#*.}; '
        'case "$blueprint_parent_deadline_integer" in \'\'|*[!0-9]*) '
        'echo BLUEPRINT_VAST_PROVIDER_BUNDLE_BLOCKED:provider_output_upload_deadline_invalid; return 86;; esac; '
        'case "$blueprint_parent_deadline_fraction" in \'\'|*[!0-9]*) '
        'echo BLUEPRINT_VAST_PROVIDER_BUNDLE_BLOCKED:provider_output_upload_deadline_invalid; return 86;; esac;; '
        '*[!0-9]*) echo BLUEPRINT_VAST_PROVIDER_BUNDLE_BLOCKED:provider_output_upload_deadline_invalid; return 86;; '
        '*) blueprint_parent_deadline_integer="$blueprint_parent_deadline";; esac; '
        'blueprint_parent_upload_deadline=$((blueprint_parent_deadline_integer - 60)); '
        'if [ "$blueprint_parent_upload_deadline" -lt "$blueprint_upload_deadline" ]; then '
        'blueprint_upload_deadline="$blueprint_parent_upload_deadline"; fi; fi; '
        'blueprint_upload_attempt=1; blueprint_upload_max_attempts=3; blueprint_upload_last_rc=86; '
        'blueprint_upload_status_file="/tmp/blueprint_provider_upload_http_status.$$"; '
        'blueprint_upload_body_file="/tmp/blueprint_provider_upload_response_body.$$"; '
        'rm -f "$blueprint_upload_status_file" "$blueprint_upload_body_file"; '
        'while [ "$blueprint_upload_attempt" -le "$blueprint_upload_max_attempts" ]; do '
        'blueprint_upload_current_bytes=$(wc -c < "$blueprint_upload_path" | tr -d \'[:space:]\'); '
        'blueprint_upload_current_sha256=$(sha256sum "$blueprint_upload_path" | cut -d" " -f1); '
        'if [ "$blueprint_upload_current_bytes" != "$blueprint_upload_bytes" ] || '
        '[ "$blueprint_upload_current_sha256" != "$blueprint_upload_sha256" ]; then '
        'rm -f "$blueprint_upload_status_file" "$blueprint_upload_body_file"; '
        "echo BLUEPRINT_VAST_PROVIDER_BUNDLE_BLOCKED:provider_output_zip_changed_during_upload_retry; return 86; fi; "
        'blueprint_upload_now=$(date +%s); blueprint_upload_remaining=$((blueprint_upload_deadline - blueprint_upload_now)); '
        'if [ "$blueprint_upload_remaining" -le 0 ]; then '
        'rm -f "$blueprint_upload_status_file" "$blueprint_upload_body_file"; '
        "echo BLUEPRINT_VAST_PROVIDER_BUNDLE_BLOCKED:provider_output_upload_deadline_exhausted; return 86; fi; "
        ': > "$blueprint_upload_status_file"; : > "$blueprint_upload_body_file"; '
        'curl --http1.1 --silent --show-error --fail -X PUT -H \'Content-Type: application/zip\' '
        '--connect-timeout 30 --max-time "$blueprint_upload_remaining" --speed-limit 1024 --speed-time 60 '
        '--output "$blueprint_upload_body_file" --write-out \'%{http_code}\' '
        '--data-binary @"$blueprint_upload_path" "$blueprint_upload_url" >"$blueprint_upload_status_file"; '
        'blueprint_upload_rc=$?; blueprint_upload_last_rc="$blueprint_upload_rc"; '
        'blueprint_upload_http_status=$(tr -d \'[:space:]\' < "$blueprint_upload_status_file"); '
        'case "$blueprint_upload_http_status" in \'\'|*[!0-9]*) blueprint_upload_http_status=000;; esac; '
        'if [ "$blueprint_upload_rc" -eq 0 ]; then case "$blueprint_upload_http_status" in 2??) '
        ': > /tmp/blueprint_provider_upload_response.json; '
        'rm -f "$blueprint_upload_status_file" "$blueprint_upload_body_file"; return 0;; esac; fi; '
        'blueprint_upload_transient=0; '
        'case "$blueprint_upload_http_status" in 408|425|429|500|502|503|504) blueprint_upload_transient=1;; esac; '
        'if [ "$blueprint_upload_http_status" = 000 ]; then case "$blueprint_upload_rc" in '
        '5|6|7|18|28|35|47|52|55|56|92) blueprint_upload_transient=1;; esac; fi; '
        'if [ "$blueprint_upload_transient" -ne 1 ]; then '
        'rm -f "$blueprint_upload_status_file" "$blueprint_upload_body_file"; '
        'echo "BLUEPRINT_VAST_PROVIDER_BUNDLE_BLOCKED:provider_output_upload_nontransient_failure:${blueprint_upload_http_status}:${blueprint_upload_rc}"; '
        'if [ "$blueprint_upload_rc" -eq 0 ]; then return 86; fi; return "$blueprint_upload_rc"; fi; '
        'if [ "$blueprint_upload_attempt" -ge "$blueprint_upload_max_attempts" ]; then '
        'rm -f "$blueprint_upload_status_file" "$blueprint_upload_body_file"; '
        "echo BLUEPRINT_VAST_PROVIDER_BUNDLE_BLOCKED:provider_output_upload_transient_retries_exhausted; "
        'if [ "$blueprint_upload_last_rc" -eq 0 ]; then return 86; fi; return "$blueprint_upload_last_rc"; fi; '
        'echo "BLUEPRINT_VAST_PROVIDER_OUTPUT_UPLOAD_TRANSPORT_RETRY:${blueprint_upload_attempt}"; '
        'sleep "$blueprint_upload_attempt"; blueprint_upload_attempt=$((blueprint_upload_attempt + 1)); '
        'done; rm -f "$blueprint_upload_status_file" "$blueprint_upload_body_file"; return 86; '
        "}; "
    )


__all__ = [
    "EXPECTED_PROVIDER_UPLOAD_BYTES_ENV",
    "provider_output_upload_shell_fragment",
]
