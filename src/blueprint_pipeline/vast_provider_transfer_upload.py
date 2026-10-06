"""Fail-closed provider-output upload transport shared by Vast lanes."""

from __future__ import annotations

import shlex


EXPECTED_PROVIDER_UPLOAD_BYTES_ENV = (
    "BLUEPRINT_VAST_EXPECTED_PROVIDER_UPLOAD_BYTES"
)


def provider_output_upload_shell_fragment(*, scratch_root: str = "/tmp") -> str:
    """Return the bounded, fail-closed provider-output upload function.

    A provider output is an immutable object: retrying the same bytes to the
    same signed PUT URL is idempotent, while rerunning the provider workload is
    not.  Keep retries here at the transport boundary and admit only explicit
    transient curl failures or transient HTTP statuses.

    A successful call exposes BLUEPRINT_PROVIDER_UPLOAD_RESPONSE_FILE in its
    fresh private directory. Consumers call blueprint_upload_read_response, then
    blueprint_upload_cleanup. The empty receipt descriptor is held before
    transport; readback validates its identity and emits no file contents.
    Descriptors 10-13 are reserved only for that lifetime; an occupied token
    refuses before mutation. Cleanup closes only owned descriptors and clears
    active receipt state. Private zero-byte receipt/body plus at most three
    status bytes remain until the existing worker/session filesystem is disposed;
    no pathname deletion can race into a foreign inode or directory. No shared
    completion sentinel is adopted or deleted.
    """

    return (
        "blueprint_upload_same_file() { "
        'if [ -d /proc/self/fd ]; then [ "$1" -ef "/proc/self/fd/$2" ]; '
        "else python3 -c 'import os,sys; a=os.stat(sys.argv[1],follow_symlinks=False); b=os.fstat(int(sys.argv[2])); sys.exit((a.st_dev,a.st_ino)!=(b.st_dev,b.st_ino))' \"$1\" \"$2\"; fi; "
        "}; "
        "blueprint_upload_read_response() { "
        'if [ "${blueprint_upload_completion_open:-0}" != 1 ] || '
        '[ -L "$blueprint_upload_scratch" ] || ! blueprint_upload_same_file "$blueprint_upload_scratch" 10 || '
        '[ -L "$blueprint_upload_completion_file" ] || ! blueprint_upload_same_file "$blueprint_upload_completion_file" 13; then '
        'echo BLUEPRINT_VAST_PROVIDER_BUNDLE_BLOCKED:provider_output_upload_completion_identity_unproven; return 86; fi; '
        'return 0; '
        "}; "
        "blueprint_upload_cleanup() { "
        'if [ "${blueprint_upload_active:-0}" = 1 ]; then '
        'exec 13<&-; exec 12<&-; exec 11<&-; exec 10<&-; fi; '
        'blueprint_upload_active=0; blueprint_upload_completion_open=0; '
        'BLUEPRINT_PROVIDER_UPLOAD_RESPONSE_FILE=""; return 0; '
        "}; "
        "blueprint_upload_put() { "
        'for blueprint_upload_fd in 10 11 12 13; do '
        'if [ -e "/dev/fd/$blueprint_upload_fd" ]; then '
        'echo BLUEPRINT_VAST_PROVIDER_BUNDLE_BLOCKED:provider_output_upload_scratch_descriptor_busy; return 86; fi; done; '
        'BLUEPRINT_PROVIDER_UPLOAD_RESPONSE_FILE=""; blueprint_upload_active=1; '
        f'blueprint_upload_parent={shlex.quote(scratch_root)}; '
        'if [ -L "$blueprint_upload_parent" ] || [ ! -d "$blueprint_upload_parent" ]; then '
        'echo BLUEPRINT_VAST_PROVIDER_BUNDLE_BLOCKED:provider_output_upload_scratch_parent_invalid; blueprint_upload_cleanup; return 86; fi; '
        'blueprint_upload_scratch=$(umask 077; mktemp -d "$blueprint_upload_parent/.blueprint-provider-upload.XXXXXXXX") || { blueprint_upload_cleanup; return 86; }; '
        'blueprint_upload_completion_open=0; '
        'blueprint_upload_status_file="$blueprint_upload_scratch/status"; '
        'blueprint_upload_body_file="$blueprint_upload_scratch/body"; '
        'blueprint_upload_completion_file="$blueprint_upload_scratch/blueprint_provider_upload_response.json"; '
        'exec 10<"$blueprint_upload_scratch" || { blueprint_upload_cleanup; return 86; }; '
        'blueprint_upload_old_umask=$(umask); '
        'case "$-" in *C*) blueprint_upload_old_noclobber=1;; *) blueprint_upload_old_noclobber=0;; esac; '
        'umask 077; set -o noclobber; '
        'if { exec 11>"$blueprint_upload_status_file" && exec 12>"$blueprint_upload_body_file" && exec 13>"$blueprint_upload_completion_file"; }; then blueprint_upload_setup_rc=0; else blueprint_upload_setup_rc=86; fi; '
        'umask "$blueprint_upload_old_umask"; '
        'if [ "$blueprint_upload_old_noclobber" = 0 ]; then set +o noclobber; fi; '
        'if [ "$blueprint_upload_setup_rc" -ne 0 ]; then blueprint_upload_cleanup; return 86; fi; '
        'blueprint_upload_transfer "$@"; blueprint_upload_result=$?; '
        'if [ "$blueprint_upload_result" -ne 0 ]; then '
        'blueprint_upload_cleanup; return "$blueprint_upload_result"; fi; '
        'if [ -L "$blueprint_upload_completion_file" ] || ! blueprint_upload_same_file "$blueprint_upload_completion_file" 13; then '
        'echo BLUEPRINT_VAST_PROVIDER_BUNDLE_BLOCKED:provider_output_upload_completion_identity_unproven; blueprint_upload_cleanup; return 86; fi; '
        'blueprint_upload_completion_open=1; '
        'BLUEPRINT_PROVIDER_UPLOAD_RESPONSE_FILE="$blueprint_upload_completion_file"; return 0; '
        "}; "
        "blueprint_upload_transfer_exit() { "
        'blueprint_upload_exit_code=$?; '
        'case "${blueprint_upload_http_status:-}" in [0-9][0-9][0-9]) printf \'%s\' "$blueprint_upload_http_status" >&11 || blueprint_upload_exit_code=86;; esac; '
        'if [ "$blueprint_upload_exit_code" -ne 0 ]; then '
        'blueprint_upload_cleanup || blueprint_upload_exit_code=86; fi; '
        'exit "$blueprint_upload_exit_code"; '
        "}; "
        "blueprint_upload_transfer() ( "
        "trap blueprint_upload_transfer_exit EXIT; "
        "trap 'exit 143' TERM; trap 'exit 130' INT; trap 'exit 129' HUP; "
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
        'while [ "$blueprint_upload_attempt" -le "$blueprint_upload_max_attempts" ]; do '
        'blueprint_upload_current_bytes=$(wc -c < "$blueprint_upload_path" | tr -d \'[:space:]\'); '
        'blueprint_upload_current_sha256=$(sha256sum "$blueprint_upload_path" | cut -d" " -f1); '
        'if [ "$blueprint_upload_current_bytes" != "$blueprint_upload_bytes" ] || '
        '[ "$blueprint_upload_current_sha256" != "$blueprint_upload_sha256" ]; then '
        "echo BLUEPRINT_VAST_PROVIDER_BUNDLE_BLOCKED:provider_output_zip_changed_during_upload_retry; return 86; fi; "
        'blueprint_upload_now=$(date +%s); blueprint_upload_remaining=$((blueprint_upload_deadline - blueprint_upload_now)); '
        'if [ "$blueprint_upload_remaining" -le 0 ]; then '
        "echo BLUEPRINT_VAST_PROVIDER_BUNDLE_BLOCKED:provider_output_upload_deadline_exhausted; return 86; fi; "
        'if [ -L "$blueprint_upload_scratch" ] || ! blueprint_upload_same_file "$blueprint_upload_scratch" 10 || '
        '[ -L "$blueprint_upload_status_file" ] || ! blueprint_upload_same_file "$blueprint_upload_status_file" 11 || '
        '[ -L "$blueprint_upload_body_file" ] || ! blueprint_upload_same_file "$blueprint_upload_body_file" 12; then '
        'echo BLUEPRINT_VAST_PROVIDER_BUNDLE_BLOCKED:provider_output_upload_scratch_identity_unproven; return 86; fi; '
        'blueprint_upload_http_status=$(curl --disable --http1.1 --silent --show-error --fail -X PUT -H \'Content-Type: application/zip\' '
        '--connect-timeout 30 --max-time "$blueprint_upload_remaining" --speed-limit 1024 --speed-time 60 '
        '--output /dev/null --write-out \'%{http_code}\' '
        # --upload-file streams from the file descriptor.  --data-binary
        # constructs a request body and curl may allocate the complete archive;
        # a real 2.31 GiB policy evidence archive exhausted provider memory that
        # way after all episodes had already completed.
        '--upload-file "$blueprint_upload_path" "$blueprint_upload_url"); '
        'blueprint_upload_rc=$?; blueprint_upload_last_rc="$blueprint_upload_rc"; '
        'case "$blueprint_upload_http_status" in [0-9][0-9][0-9]) ;; *) blueprint_upload_http_status=000;; esac; '
        'if [ "$blueprint_upload_rc" -eq 0 ]; then case "$blueprint_upload_http_status" in 2??) '
        'blueprint_upload_final_sha256=$(sha256sum "$blueprint_upload_path" | cut -d" " -f1); '
        'if [ "$blueprint_upload_final_sha256" != "$blueprint_upload_sha256" ]; then '
        'echo BLUEPRINT_VAST_PROVIDER_BUNDLE_BLOCKED:provider_output_zip_changed_during_upload_retry; return 86; fi; '
        'if [ -L "$blueprint_upload_scratch" ] || ! blueprint_upload_same_file "$blueprint_upload_scratch" 10 || '
        '[ -L "$blueprint_upload_status_file" ] || ! blueprint_upload_same_file "$blueprint_upload_status_file" 11 || '
        '[ -L "$blueprint_upload_body_file" ] || ! blueprint_upload_same_file "$blueprint_upload_body_file" 12; then '
        'echo BLUEPRINT_VAST_PROVIDER_BUNDLE_BLOCKED:provider_output_upload_scratch_identity_unproven; return 86; fi; '
        'if [ -L "$blueprint_upload_completion_file" ] || ! blueprint_upload_same_file "$blueprint_upload_completion_file" 13; then '
        'echo BLUEPRINT_VAST_PROVIDER_BUNDLE_BLOCKED:provider_output_upload_completion_identity_unproven; return 86; fi; '
        'return 0;; esac; fi; '
        'blueprint_upload_transient=0; '
        'case "$blueprint_upload_http_status" in 408|425|429|500|502|503|504) blueprint_upload_transient=1;; esac; '
        # curl can report the interim HTTP 100 response when the final PUT
        # times out; it is still a transport failure eligible for one retry.
        'if [ "$blueprint_upload_http_status" = 000 ] || [ "$blueprint_upload_http_status" = 100 ]; then case "$blueprint_upload_rc" in '
        '5|6|7|18|28|35|47|52|55|56|92) blueprint_upload_transient=1;; esac; fi; '
        'if [ "$blueprint_upload_transient" -ne 1 ]; then '
        'echo "BLUEPRINT_VAST_PROVIDER_BUNDLE_BLOCKED:provider_output_upload_nontransient_failure:${blueprint_upload_http_status}:${blueprint_upload_rc}"; '
        'if [ "$blueprint_upload_rc" -eq 0 ]; then return 86; fi; return "$blueprint_upload_rc"; fi; '
        'if [ "$blueprint_upload_attempt" -ge "$blueprint_upload_max_attempts" ]; then '
        "echo BLUEPRINT_VAST_PROVIDER_BUNDLE_BLOCKED:provider_output_upload_transient_retries_exhausted; "
        'if [ "$blueprint_upload_last_rc" -eq 0 ]; then return 86; fi; return "$blueprint_upload_last_rc"; fi; '
        'echo "BLUEPRINT_VAST_PROVIDER_OUTPUT_UPLOAD_TRANSPORT_RETRY:${blueprint_upload_attempt}"; '
        'sleep "$blueprint_upload_attempt"; blueprint_upload_attempt=$((blueprint_upload_attempt + 1)); '
        'done; return 86; '
        "); "
    )


__all__ = [
    "EXPECTED_PROVIDER_UPLOAD_BYTES_ENV",
    "provider_output_upload_shell_fragment",
]
