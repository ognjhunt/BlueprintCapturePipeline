"""Archive scene evidence, excluding reproducible tool installations."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import zipfile

EXCLUDED_PARTS = frozenset({'.artifixer-venv', '.hf_home', '.venv', '.ovrtx_venv',
    '.ovrtx_native_venv', '.ovphysx_venv', '.git', '__pycache__', 'artifixer_bundle',
    'artifixer_execution', 'artifixer_output', 'content_agents_source', 'packaged_blender'})


def write_output_archive(output_dir: Path, output_zip: Path, *, completed_stages=None):
    """Keep strict symlink refusal outside disposable runtime directories."""
    with zipfile.ZipFile(output_zip, 'w', compression=zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
        archive.writestr('provider_output_zip_exclusions.json', json.dumps({
            'schema_version': 'task_evaluation_scene_configuration_provider_output_zip_exclusions.v1',
            'excluded_directory_names': sorted(EXCLUDED_PARTS)}, sort_keys=True))
        if completed_stages is not None:
            archive.writestr('completed_stage_checkpoint.json', json.dumps({
                'schema_version': 'scene_configuration_completed_stage_checkpoint.v1',
                'status': 'completed_prefix_only', 'completed_stage_ids': list(completed_stages),
                'whole_run_completed': False, 'qualification_authority_granted': False}, sort_keys=True))
        if not output_dir.is_dir():
            archive.writestr('runtime_output_missing.json', json.dumps({
                'status': 'blocked', 'blockers': ['runtime_output_directory_missing']}))
            return
        for path in sorted(output_dir.rglob('*')):
            relative = path.relative_to(output_dir)
            if not EXCLUDED_PARTS.isdisjoint(relative.parts):
                continue
            # Only uncompleted stage directories are omitted; the resume
            # binding beside them travels, so another root can adopt the prefix.
            if (completed_stages is not None and relative.parts[0] == 'stages'
                    and len(relative.parts) > 2 and relative.parts[1] not in completed_stages):
                continue
            if path.is_symlink():
                raise RuntimeError('scene_configuration_provider_output_symlink_forbidden:' + relative.as_posix())
            if path.is_file():
                archive.write(path, relative.as_posix())

        # Runtime scratch is omitted, but the exact completed optimizer/model
        # checkpoint is irreplaceable work. Keep only receipt-bound weights.
        weights = []
        pattern = 'stages/stage-*/producer/released_artifixer_runtime/artifixer_candidate_round_*/artifixer_output/public_scene_artifixer3d_runtime_result.json'
        for receipt_path in sorted(output_dir.glob(pattern)):
            stage_id = receipt_path.relative_to(output_dir).parts[1]
            if completed_stages is not None and stage_id not in completed_stages:
                continue
            # Preserve the child's own result even when its enclosing component
            # times out before constructing the post-training reuse checkpoint.
            if receipt_path.resolve(strict=True) != receipt_path:
                raise RuntimeError('scene_configuration_training_result_symlink_forbidden')
            result = json.loads(receipt_path.read_text())
            archive.write(receipt_path, receipt_path.relative_to(output_dir).as_posix())
            for task in result.get('tasks', []):
                record = task.get('artifixer3d_checkpoint')
                if not record:
                    continue
                source = Path(record['path'])
                resolved = source.resolve(strict=True)
                if (source != resolved or not resolved.is_relative_to(output_dir.resolve())
                        or source.suffix != '.pt' or not source.name.startswith('ckpt_')
                        or source.stat().st_size != record.get('size_bytes')):
                    raise RuntimeError('scene_configuration_training_checkpoint_path_invalid')
                with source.open('rb') as stream:
                    digest = 'sha256:' + hashlib.file_digest(stream, 'sha256').hexdigest()
                if digest != record.get('sha256'):
                    raise RuntimeError('scene_configuration_training_checkpoint_digest_invalid')
                relative = source.relative_to(output_dir).as_posix()
                archive.write(source, relative)
                weights.append({'path': relative, 'sha256': digest, 'size_bytes': source.stat().st_size})
        archive.writestr('retained_training_checkpoints.json', json.dumps({
            'schema_version': 'scene_configuration_retained_training_checkpoints.v1',
            'checkpoints': weights}, sort_keys=True))


def preserve_stage_prefix(*, output_root: Path, completed_results, checkpoint_path: Path):
    """Atomically replace the previous local checkpoint before advancing stages."""
    stages = [row['stage_id'] for row in completed_results]
    temporary = checkpoint_path.with_suffix('.partial.zip')
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    write_output_archive(output_root, temporary, completed_stages=stages)
    os.replace(temporary, checkpoint_path)
    print('BLUEPRINT_SCENE_CONFIGURATION_STAGE_CHECKPOINT:' + json.dumps({
        'completed_stage_ids': stages, 'size_bytes': checkpoint_path.stat().st_size,
        'status': 'completed_prefix_only'}, sort_keys=True), flush=True)
