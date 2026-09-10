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
    AssetAuthoringError, VisualBrief, budgeted_invoker, execute_asset_authoring, file_record,
    save_json, validate_request,
)
from .decision_evidence_contracts import canonical_digest


def verify_execution_commit(expected: str, repo: Path) -> None:
    head = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    status = subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain'], text=True).strip()
    if head != expected or status:
        raise AssetAuthoringError('authoring_requires_exact_clean_immutable_commit')


def author_one(*, request_path: Path, output_root: Path, budget_root: Path,
               cad_source_root: Path, mac_source_root: Path, blender: Path,
               maximum_cost_usd: float = 15.0,
               adopt_source_analysis_from: Path | None = None) -> dict[str, Any]:
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
    adopted = adoption = None
    if adopt_source_analysis_from is not None:
        adopted, adoption = verify_source_analysis_adoption(
            prior_root=adopt_source_analysis_from, request_value=value, budget_root=budget_root)
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
            blender_executable=str(blender), adopted_source_analysis=adopted,
            adoption_record=adoption,
            authoring_instructions=(cad_source_root / 'skills/cad/SKILL.md').read_text())
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
    parser.add_argument('--adopt-source-analysis-from', type=Path)
    args = parser.parse_args(argv)
    result = author_one(request_path=args.request, output_root=args.output_root,
        budget_root=args.budget_root, cad_source_root=args.cad_source_root,
        mac_source_root=args.mac_source_root, blender=args.blender,
        maximum_cost_usd=args.maximum_cost_usd,
        adopt_source_analysis_from=args.adopt_source_analysis_from)
    print(json.dumps({'status': result['status'], 'object_id': result['object_id'],
                      'result_digest': result['result_digest']}))
    return 0


def verify_source_analysis_adoption(*, prior_root: Path, request_value: dict,
                                    budget_root: Path):
    """Reuse only an exact completed model output bound to unchanged source inputs."""
    prior_request = json.loads((prior_root / 'request.json').read_text())
    validate_request(prior_request)
    def relevant(value):
        return {k: v for k, v in value.items()
                if k not in {'request_digest', 'expected_production_commit'}}
    if relevant(prior_request) != relevant(request_value):
        raise AssetAuthoringError('authoring_adoption_source_inputs_changed')
    phase = prior_root / 'source_analysis.json'
    record = json.loads(phase.read_text())
    if record.get('request_digest') != prior_request['request_digest'] or record.get('model') != 'gpt-6-astra':
        raise AssetAuthoringError('authoring_adoption_phase_binding_invalid')
    output = VisualBrief.model_validate(record['output'])
    output_digest = canonical_digest(output.model_dump(mode='json'))
    matching = []
    for path in (budget_root / 'inference_reservations/completed').glob('*.json'):
        completion = json.loads(path.read_text())
        if (completion.get('run_id') == request_value['run_id']
            and completion.get('capability') == request_value['object_id'] + '_source_analysis'
            and completion.get('structured_output_digest') == output_digest
            and completion.get('inference_completion_digest') == canonical_digest(
                completion, digest_field='inference_completion_digest')):
            matching.append(path)
    if len(matching) != 1:
        raise AssetAuthoringError('authoring_adoption_completed_response_missing')
    return output, {'schema_version': 'asset_source_analysis_adoption.v1',
        'output_digest': output_digest, 'source_phase': file_record(phase),
        'source_request_digest': prior_request['request_digest'],
        'source_production_commit': prior_request['expected_production_commit'],
        'new_request_digest': request_value['request_digest'],
        'completed_provider_response': file_record(matching[0]), 'new_provider_call': False}


if __name__ == '__main__':
    raise SystemExit(main())
