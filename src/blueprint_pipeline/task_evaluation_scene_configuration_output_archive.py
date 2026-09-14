"""Archive scene evidence, excluding reproducible tool installations."""
from __future__ import annotations

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
            if (completed_stages is not None and relative.parts[0] == 'stages'
                    and len(relative.parts) > 1 and relative.parts[1] not in completed_stages):
                continue
            if path.is_symlink():
                raise RuntimeError('scene_configuration_provider_output_symlink_forbidden:' + relative.as_posix())
            if path.is_file():
                archive.write(path, relative.as_posix())


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
