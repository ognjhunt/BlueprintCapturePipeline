"""Fail-closed provider-output upload transport shared by Vast lanes."""

from __future__ import annotations


EXPECTED_PROVIDER_UPLOAD_BYTES_ENV = (
    "BLUEPRINT_VAST_EXPECTED_PROVIDER_UPLOAD_BYTES"
)


def provider_output_upload_shell_fragment() -> str:
    """Return the upload function, including the admission-priced byte guard."""

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
        "if command -v curl >/dev/null 2>&1; then curl -fsS -X PUT -H 'Content-Type: application/zip' "
        '--data-binary @"$blueprint_upload_path" "$blueprint_upload_url" >/tmp/blueprint_provider_upload_response.json; return $?; fi; '
        'blueprint_upload_py="${PY_NET:-${RUNTIME_PY:-}}"; '
        'if [ -n "$blueprint_upload_py" ]; then '
        'BLUEPRINT_UPLOAD_URL="$blueprint_upload_url" BLUEPRINT_UPLOAD_PATH="$blueprint_upload_path" "$blueprint_upload_py" - <<\'PY\' >/tmp/blueprint_provider_upload_response.json\n'
        "import os\n"
        "import sys\n"
        "import urllib.request\n"
        "url = os.environ.get('BLUEPRINT_UPLOAD_URL', '')\n"
        "path = os.environ.get('BLUEPRINT_UPLOAD_PATH', '')\n"
        "try:\n"
        "    with open(path, 'rb') as handle:\n"
        "        data = handle.read()\n"
        "    request = urllib.request.Request(url, data=data, method='PUT', headers={'Content-Type': 'application/zip', 'User-Agent': 'BlueprintVastProbe/1.0'})\n"
        "    with urllib.request.urlopen(request, timeout=120) as response:\n"
        "        sys.stdout.buffer.write(response.read())\n"
        "except Exception as exc:\n"
        "    print('BLUEPRINT_VAST_PY_UPLOAD_ERROR:%s' % type(exc).__name__)\n"
        "    raise SystemExit(1)\n"
        "PY\n"
        "return $?; "
        "fi; "
        "return 127; "
        "}; "
    )


__all__ = [
    "EXPECTED_PROVIDER_UPLOAD_BYTES_ENV",
    "provider_output_upload_shell_fragment",
]
