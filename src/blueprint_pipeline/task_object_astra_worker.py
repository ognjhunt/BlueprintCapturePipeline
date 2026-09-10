"""Production CPU entrypoint for bounded Astra/CAD/Blender asset authoring.

One request per object, one durable shared budget per batch. No GPU allocation,
policy execution, scene mutation, or native qualification happens here.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any

from .asset_authoring_sandbox import SandboxedAssetRunner
from .astra_cad_skill_runtime import execute_mac_candidate
from .task_object_astra_authoring import (
    AssetAuthoringError, budgeted_invoker, execute_asset_authoring, file_record,
    save_json, validate_request,
)


def verify_execution_commit(expected: str, repo: Path) -> None:
    head = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    status = subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain'], text=True).strip()
    if head != expected or status:
        raise AssetAuthoringError('authoring_requires_exact_clean_immutable_commit')


def author_one(*, request_path: Path, output_root: Path, budget_root: Path,
               cad_source_root: Path, mac_source_root: Path, blender: Path,
               maximum_cost_usd: float = 15.0) -> dict[str, Any]:
    value = json.loads(request_path.read_text())
    request = validate_request(value)
    repo = Path(__file__).resolve().parents[2]
    verify_execution_commit(request.expected_production_commit, repo)
    if platform.system() == 'Linux':
        from .production_blender_runtime import validate_runtime
        toolchain_root = Path(os.environ['BLUEPRINT_BLENDER_RUNTIME_ROOT'])
        receipt = validate_runtime(toolchain_root)
        if Path(receipt['executable']).resolve() != blender.resolve():
            raise AssetAuthoringError('authoring_blender_release_mismatch')
    budget_root.mkdir(parents=True, exist_ok=True)
    if budget_root.resolve().is_relative_to(output_root.resolve()):
        raise AssetAuthoringError('authoring_budget_inside_untrusted_write_root')
    # One write root contains both CAD and Blender attempts. Model credentials
    # never enter generated-program environments or mounted source roots.
    python_roots = [Path(sys.prefix).resolve(), Path(sys.base_prefix).resolve()]
    python_roots += [Path(p).resolve() for p in sys.path if p and Path(p).is_dir()
                     and ('site-packages' in p or p.endswith('/mac-deps'))]
    blender_root = blender.parents[2] if platform.system() == 'Darwin' else blender.parent
    read_roots = list(dict.fromkeys([*python_roots, cad_source_root.resolve(),
        mac_source_root.resolve(), repo / 'src', blender_root.resolve()]))
    scratch = output_root
    scratch.mkdir(parents=True, exist_ok=False)
    runner = SandboxedAssetRunner(read_roots=read_roots, write_root=scratch,
                                 executable_roots=[*python_roots, blender_root.resolve()])
    runner.preflight()
    invoker, audit = budgeted_invoker(root=budget_root, run_id=request.run_id,
                                    maximum_cost_usd=maximum_cost_usd)
    # Every charged HTTP attempt is individually reserved. SDK transport
    # retries after a timeout could otherwise duplicate an unknown charge.
    from agents import set_default_openai_client
    from openai import AsyncOpenAI
    secret_path = Path(os.environ['OPENAI_API_KEY_FILE']).expanduser()
    if secret_path.is_symlink() or not secret_path.is_file() or secret_path.stat().st_mode & 0o077:
        raise AssetAuthoringError('authoring_canonical_secret_file_invalid')
    set_default_openai_client(AsyncOpenAI(
        api_key=secret_path.read_text().strip(), base_url='https://api.openai.com/v1',
        max_retries=0, timeout=240), use_for_tracing=False)
    def cad_executor(*, brief, output_root, dimensions_m):
        receipt = execute_mac_candidate(
            brief, output_root, mac_source_root, cad_source_root, invoker,
            expected_dimensions_mm=tuple(v * 1000 for v in dimensions_m),
            subprocess_runner=runner, run_id=request.run_id, object_label=request.object_id,
            dimension_tolerance_mm=request.maximum_export_error_m * 1000,
        )
        return {**receipt, 'stl': file_record(Path(receipt['stl_path'])),
                'step': file_record(Path(receipt['step_path']))}
    try:
        return execute_asset_authoring(request_value=value, output_root=output_root,
            invoker=invoker, mac_executor=cad_executor, blender_runner=runner,
            blender_executable=str(blender))
    finally:
        save_json(budget_root / 'budget_manifest.json', audit.write_manifest())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--budget-root', type=Path, required=True)
    parser.add_argument('--cad-source-root', type=Path, required=True)
    parser.add_argument('--mac-source-root', type=Path, required=True)
    parser.add_argument('--blender', type=Path, required=True)
    parser.add_argument('--maximum-cost-usd', type=float, default=15.0)
    args = parser.parse_args(argv)
    result = author_one(request_path=args.request, output_root=args.output_root,
        budget_root=args.budget_root, cad_source_root=args.cad_source_root,
        mac_source_root=args.mac_source_root, blender=args.blender,
        maximum_cost_usd=args.maximum_cost_usd)
    print(json.dumps({'status': result['status'], 'object_id': result['object_id'],
                      'result_digest': result['result_digest']}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
