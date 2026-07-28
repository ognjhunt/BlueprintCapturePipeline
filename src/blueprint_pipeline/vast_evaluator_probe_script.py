"""Shell fragment for Vast evaluator-only provider bundles."""

from __future__ import annotations


EVALUATOR_REQUIRED_ENTRIES = {
    "provider_runtime/evaluator_provider_runtime_runner.py",
    "provider_runtime/run_evaluator_provider_runtime.sh",
    "provider_runtime/evaluator_provider_runtime_manifest.json",
    "provider_runtime/evaluator_input_manifest.json",
}


def evaluator_provider_probe_script(common_start: str) -> str:
    """Return the evaluator bundle execution and upload shell fragment."""
    return (
        common_start + "RUNTIME_PY=''; "
        "if command -v apt-get >/dev/null 2>&1 && "
        "{ ! command -v python3 >/dev/null 2>&1 || ! command -v curl >/dev/null 2>&1 || "
        "! command -v unzip >/dev/null 2>&1 || ! command -v ffmpeg >/dev/null 2>&1; }; then "
        "apt-get update >/tmp/blueprint_vast_apt_update.log 2>&1 && "
        "DEBIAN_FRONTEND=noninteractive apt-get install -y python3 curl unzip ffmpeg >/tmp/blueprint_vast_apt_install.log 2>&1; "
        "fi; "
        "if [ -x /opt/conda/bin/python ]; then RUNTIME_PY=/opt/conda/bin/python; "
        "elif [ -x /usr/local/bin/python ]; then RUNTIME_PY=/usr/local/bin/python; "
        "elif command -v python3 >/dev/null 2>&1; then RUNTIME_PY=$(command -v python3); "
        "elif command -v python >/dev/null 2>&1; then RUNTIME_PY=$(command -v python); fi; "
        'if [ -z "$RUNTIME_PY" ]; then echo BLUEPRINT_VAST_PROVIDER_BUNDLE_BLOCKED:python_missing; '
        "else "
        'rm -rf "$WORK_DIR/evaluator_provider_bundle" "$WORK_DIR/evaluator_provider_runtime_bundle.zip" "$WORK_DIR/evaluator_provider_runtime_output.zip"; '
        'blueprint_download_url "$BUNDLE_URL" "$WORK_DIR/evaluator_provider_runtime_bundle.zip"; dl=$?; '
        "if [ $dl -ne 0 ]; then echo BLUEPRINT_VAST_PROVIDER_BUNDLE_BLOCKED:download_failed:$dl; "
        "else "
        "echo BLUEPRINT_VAST_PROVIDER_BUNDLE_DOWNLOADED; "
        '$RUNTIME_PY -m zipfile -e "$WORK_DIR/evaluator_provider_runtime_bundle.zip" "$WORK_DIR/evaluator_provider_bundle"; unzip_rc=$?; '
        "if [ $unzip_rc -ne 0 ]; then echo BLUEPRINT_VAST_PROVIDER_BUNDLE_BLOCKED:unzip_failed:$unzip_rc; "
        'elif [ ! -f "$WORK_DIR/evaluator_provider_bundle/provider_runtime/run_evaluator_provider_runtime.sh" ]; then echo BLUEPRINT_VAST_PROVIDER_BUNDLE_BLOCKED:entrypoint_missing; '
        "else "
        'export BLUEPRINT_EVALUATOR_PROVIDER_PYTHON="$RUNTIME_PY"; '
        'export BLUEPRINT_EVALUATOR_PROVIDER_OUTPUT_DIR="$WORK_DIR/evaluator_provider_bundle/runtime_output"; '
        'export BLUEPRINT_EVALUATOR_PROVIDER_BUNDLE_DIR="$WORK_DIR/evaluator_provider_bundle"; '
        'export BLUEPRINT_EVALUATOR_INPUT="$WORK_DIR/evaluator_provider_bundle/provider_runtime/evaluator_input_manifest.json"; '
        "echo BLUEPRINT_VAST_PROVIDER_ENTRYPOINT_STARTED; "
        'bash "$WORK_DIR/evaluator_provider_bundle/provider_runtime/run_evaluator_provider_runtime.sh"; provider_rc=$?; '
        "echo BLUEPRINT_VAST_PROVIDER_ENTRYPOINT_EXIT_CODE:$provider_rc; "
        "$RUNTIME_PY - <<'PY'\n"
        "import json\n"
        "import os\n"
        "import zipfile\n"
        "from pathlib import Path\n"
        "output_dir = Path(os.environ.get('BLUEPRINT_EVALUATOR_PROVIDER_OUTPUT_DIR', '/workspace/evaluator_provider_bundle/runtime_output'))\n"
        "work_dir = Path(os.environ.get('BLUEPRINT_VAST_WORK_DIR', '/tmp/blueprint_vast_work'))\n"
        "output_zip = work_dir / 'evaluator_provider_runtime_output.zip'\n"
        "with zipfile.ZipFile(output_zip, 'w', compression=zipfile.ZIP_DEFLATED) as archive:\n"
        "    if output_dir.is_dir():\n"
        "        for path in sorted(output_dir.rglob('*')):\n"
        "            if path.is_file():\n"
        "                archive.write(path, path.relative_to(output_dir).as_posix())\n"
        "    else:\n"
        "        archive.writestr('runtime_output_missing.json', json.dumps({'status': 'blocked', 'blockers': ['runtime_output_directory_missing']}, indent=2))\n"
        "print('BLUEPRINT_VAST_PROVIDER_OUTPUT_ZIP_WRITTEN:%d' % output_zip.stat().st_size)\n"
        "PY\n"
        "zip_rc=$?; "
        "if [ $zip_rc -ne 0 ]; then echo BLUEPRINT_VAST_PROVIDER_BUNDLE_BLOCKED:output_zip_failed:$zip_rc; "
        'elif blueprint_upload_put "$OUTPUT_PUT_URL" "$WORK_DIR/evaluator_provider_runtime_output.zip"; then '
        "echo BLUEPRINT_VAST_PROVIDER_OUTPUT_UPLOAD_OK; cat /tmp/blueprint_provider_upload_response.json; "
        "else upload_rc=$?; echo BLUEPRINT_VAST_PROVIDER_BUNDLE_BLOCKED:output_upload_failed:$upload_rc; fi; "
        "echo BLUEPRINT_VAST_PROVIDER_BUNDLE_COMPLETED_OR_BLOCKED; "
        "fi; fi; fi; fi; "
    )
