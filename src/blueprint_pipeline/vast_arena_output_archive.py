"""Generate the self-contained provider-side Arena evidence archive script."""

from __future__ import annotations


_RESULT_NAME = "native_task_arena_policy_canary_session_result.v1.json"
_ARCHIVE_SCRIPT = """import json
import os
import zipfile
from pathlib import Path
output_dir = Path(os.environ.get('BLUEPRINT_ADP_ARENA_OUTPUT_DIR', '/workspace/adp_arena_provider_bundle/runtime_output'))
work_dir = Path(os.environ.get('BLUEPRINT_VAST_WORK_DIR', '/tmp/blueprint_vast_work'))
output_zip = work_dir / 'adp_arena_provider_runtime_output.zip'
required_result_max_bytes = 512 * 1024 * 1024
if required_result_name is not None:
    required_result = output_dir / required_result_name
    if required_result.is_symlink() or not required_result.is_file():
        raise SystemExit('policy_canary_required_terminal_result_missing_or_not_regular')
    required_result_size = required_result.stat().st_size
    if not 0 < required_result_size <= required_result_max_bytes:
        raise SystemExit('policy_canary_required_terminal_result_size_invalid:%d' % required_result_size)
with zipfile.ZipFile(output_zip, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
    if output_dir.is_dir():
        for path in sorted(output_dir.rglob('*')):
            if path.is_file():
                size = path.stat().st_size
                relative_name = path.relative_to(output_dir).as_posix()
                size_limit = required_result_max_bytes if relative_name == required_result_name else 100_000_000
                if size <= size_limit:
                    archive.write(path, relative_name)
    else:
        archive.writestr('runtime_output_missing.json', json.dumps({'status': 'blocked', 'blockers': ['runtime_output_directory_missing']}, indent=2))
    if required_result_name is not None:
        if archive.getinfo(required_result_name).file_size != required_result_size:
            raise SystemExit('policy_canary_required_terminal_result_archive_size_mismatch')
print('BLUEPRINT_VAST_PROVIDER_OUTPUT_ZIP_WRITTEN:%d' % output_zip.stat().st_size)
"""


def arena_output_archive_script(*, policy_canary: bool) -> str:
    """Keep V25-sized terminal aggregates without enlarging generic file limits."""

    return f"required_result_name = {(_RESULT_NAME if policy_canary else None)!r}\n" + _ARCHIVE_SCRIPT
