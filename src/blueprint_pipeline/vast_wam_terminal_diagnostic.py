"""Shell fragment that preserves WAM provider terminal diagnostics early."""

from __future__ import annotations


def wam_terminal_diagnostic_shell_fragment() -> str:
    """Return a fail-closed fragment that uploads diagnostics before full output."""
    return (
        'export BLUEPRINT_WAM_PROVIDER_ENTRYPOINT_RC="$provider_rc"; '
        "$RUNTIME_PY - <<'PY'\n"
        "import json\n"
        "import os\n"
        "from pathlib import Path\n"
        "output_dir = Path(os.environ.get('BLUEPRINT_WAM_PROVIDER_OUTPUT_DIR', '/workspace/wam_provider_bundle/runtime_output'))\n"
        "output_dir.mkdir(parents=True, exist_ok=True)\n"
        "provider_rc = int(os.environ.get('BLUEPRINT_WAM_PROVIDER_ENTRYPOINT_RC', '255'))\n"
        "diagnostic = {\n"
        "    'schema_version': 'wam_provider_entrypoint_diagnostic.v1',\n"
        "    'status': 'completed' if provider_rc == 0 else 'blocked',\n"
        "    'provider_entrypoint_exit_code': provider_rc,\n"
        "    'provider_entrypoint_terminated_by_signal': provider_rc - 128 if provider_rc >= 128 else None,\n"
        "    'runtime_output_directory_present': output_dir.is_dir(),\n"
        "    'claim_boundary': 'engineering terminal diagnostic only; not model or task success',\n"
        "}\n"
        "(output_dir / 'provider_entrypoint_diagnostic.json').write_text(json.dumps(diagnostic, indent=2, sort_keys=True) + '\\n', encoding='utf-8')\n"
        "print('BLUEPRINT_VAST_PROVIDER_ENTRYPOINT_DIAGNOSTIC_WRITTEN')\n"
        "PY\n"
        "diagnostic_rc=$?; "
        "if [ $diagnostic_rc -ne 0 ]; then echo BLUEPRINT_VAST_PROVIDER_BUNDLE_BLOCKED:entrypoint_diagnostic_write_failed:$diagnostic_rc; "
        "elif [ $provider_rc -ne 0 ]; then "
        "$RUNTIME_PY - <<'PY'\n"
        "import os\n"
        "import zipfile\n"
        "from pathlib import Path\n"
        "output_dir = Path(os.environ.get('BLUEPRINT_WAM_PROVIDER_OUTPUT_DIR', '/workspace/wam_provider_bundle/runtime_output'))\n"
        "work_dir = Path(os.environ.get('BLUEPRINT_VAST_WORK_DIR', '/tmp/blueprint_vast_work'))\n"
        "diagnostic = output_dir / 'provider_entrypoint_diagnostic.json'\n"
        "with zipfile.ZipFile(work_dir / 'wam_provider_runtime_output.zip', 'w', compression=zipfile.ZIP_DEFLATED) as archive:\n"
        "    archive.write(diagnostic, diagnostic.name)\n"
        "print('BLUEPRINT_VAST_PROVIDER_EARLY_DIAGNOSTIC_ZIP_WRITTEN')\n"
        "PY\n"
        "early_zip_rc=$?; "
        'if [ $early_zip_rc -eq 0 ] && blueprint_upload_put "$OUTPUT_PUT_URL" "$WORK_DIR/wam_provider_runtime_output.zip"; then '
        "echo BLUEPRINT_VAST_PROVIDER_EARLY_DIAGNOSTIC_UPLOAD_OK; "
        "else early_upload_rc=$?; echo BLUEPRINT_VAST_PROVIDER_BUNDLE_BLOCKED:early_diagnostic_upload_failed:$early_upload_rc; fi; "
        "fi; "
    )
