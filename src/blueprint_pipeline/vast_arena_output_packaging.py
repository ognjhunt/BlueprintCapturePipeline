"""Bounded output packaging for the shared Isaac provider shell envelope."""

from __future__ import annotations

from .provider_runtime_bundle_contract import g1_provider_output_contract


def arena_provider_output_packaging_script(provider_bundle_kind: str) -> str:
    """Keep every required artifact and refuse output above the lane's quote."""
    return (
        "$RUNTIME_PY - <<'PY'\n"
        "import json\n"
        "import os\n"
        "import zipfile\n"
        "from pathlib import Path\n"
        "output_dir = Path(os.environ.get('BLUEPRINT_ADP_ARENA_OUTPUT_DIR', '/workspace/adp_arena_provider_bundle/runtime_output'))\n"
        "work_dir = Path(os.environ.get('BLUEPRINT_VAST_WORK_DIR', '/tmp/blueprint_vast_work'))\n"
        "output_zip = work_dir / 'adp_arena_provider_runtime_output.zip'\n"
        # The twenty-row policy canary aggregate exceeded the generic
        # 100 MB cap in V25. Keep this required record with a bounded
        # allowance; refusing retention must prevent a success upload.
        "required_result_name = "
        + repr(
            "native_task_arena_policy_canary_session_result.v1.json"
            if provider_bundle_kind == "native_task_arena_policy_canary_session"
            else g1_provider_output_contract(provider_bundle_kind)[0]
        )
        + "\n"
        "preserve_all_output = "
        + repr(g1_provider_output_contract(provider_bundle_kind)[1])
        + "\n"
        # The pinned checkpoint and SONIC receipts live directly under
        # models/. Their downloaded weights are execution inputs, not
        # review evidence, and exceed the bounded output PUT ceiling.
        "g1_transient_runtime_prefixes = "
        + repr(
            g1_provider_output_contract(provider_bundle_kind)[2]
        )
        + "\n"
        "required_result_max_bytes = 512 * 1024 * 1024\n"
        "asset_preparation_output = "
        + repr(provider_bundle_kind == "embodiment_asset_preparation")
        + "\n"
        "g1_beta_output = "
        + repr(provider_bundle_kind == "native_g1_beta_screening")
        + "\n"
        "if g1_beta_output:\n"
        "    required_result_max_bytes = 128 * 1024**2\n"
        "    for path in output_dir.rglob('*'):\n"
        "        if path.is_symlink() or not (path.is_file() or path.is_dir()):\n"
        "            raise SystemExit('g1_beta_output_not_regular')\n"
        "if asset_preparation_output:\n"
        "    total_output_bytes = 0\n"
        "    for path in output_dir.rglob('*'):\n"
        "        if path.is_symlink() or not (path.is_file() or path.is_dir()):\n"
        "            raise SystemExit('embodiment_preparation_output_not_regular')\n"
        "        if path.is_file():\n"
        "            total_output_bytes += path.stat().st_size\n"
        "    if total_output_bytes > 4 * 1024**3:\n"
        "        raise SystemExit('embodiment_preparation_output_bound_exceeded')\n"
        "required_evidence_names = "
        + repr(("native_continuation_stage_output.zip",)
               if provider_bundle_kind == "scene_configuration_native_continuation" else ())
        + "\n"
        "for name in required_evidence_names:\n"
        "    evidence = output_dir / name\n"
        "    if evidence.is_symlink() or not evidence.is_file():\n"
        "        raise SystemExit('provider_required_evidence_missing_or_not_regular:' + name)\n"
        "    if not 0 < evidence.stat().st_size <= required_result_max_bytes:\n"
        "        raise SystemExit('provider_required_evidence_size_invalid:' + name)\n"
        "if required_result_name is not None:\n"
        "    required_result = output_dir / required_result_name\n"
        "    if required_result.is_symlink() or not required_result.is_file():\n"
        "        raise SystemExit('policy_canary_required_terminal_result_missing_or_not_regular')\n"
        "    required_result_size = required_result.stat().st_size\n"
        "    if not 0 < required_result_size <= required_result_max_bytes:\n"
        "        raise SystemExit('policy_canary_required_terminal_result_size_invalid:%d' % required_result_size)\n"
        "with zipfile.ZipFile(output_zip, 'w', compression=zipfile.ZIP_DEFLATED) as archive:\n"
        "    if output_dir.is_dir():\n"
        "        for path in sorted(output_dir.rglob('*')):\n"
        "            if path.is_file():\n"
        "                size = path.stat().st_size\n"
        "                relative_name = path.relative_to(output_dir).as_posix()\n"
        "                if relative_name.startswith(g1_transient_runtime_prefixes):\n"
        "                    continue\n"
        "                size_limit = required_result_max_bytes if relative_name == required_result_name else 100_000_000\n"
        "                if preserve_all_output or size <= size_limit:\n"
        "                    archive.write(path, relative_name)\n"
        "    else:\n"
        "        archive.writestr('runtime_output_missing.json', json.dumps({'status': 'blocked', 'blockers': ['runtime_output_directory_missing']}, indent=2))\n"
        "    if required_result_name is not None:\n"
        "        if archive.getinfo(required_result_name).file_size != required_result_size:\n"
        "            raise SystemExit('policy_canary_required_terminal_result_archive_size_mismatch')\n"
        "if asset_preparation_output and output_zip.stat().st_size > 4 * 1024**3:\n"
        "    raise SystemExit('embodiment_preparation_archive_bound_exceeded')\n"
        "if g1_beta_output and output_zip.stat().st_size > 4 * 1024**3:\n"
        "    output_zip.unlink()\n"
        "    raise SystemExit('g1_beta_archive_bound_exceeded')\n"
        "print('BLUEPRINT_VAST_PROVIDER_OUTPUT_ZIP_WRITTEN:%d' % output_zip.stat().st_size)\n"
        "PY\n"
    )
