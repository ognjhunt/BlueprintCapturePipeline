import hashlib
import json
import zipfile

import pytest

from blueprint_pipeline.task_evaluation_scene_configuration_output_archive import write_output_archive


def test_declared_completed_weights_survive_scratch_exclusion(tmp_path):
    output = tmp_path/'output'
    scratch = output/'stages/stage-1/producer/released_artifixer_runtime/artifixer_candidate_round_0/artifixer_output'
    weights = scratch/'tasks/object/artifixer3d/runs/task/ours_30000/ckpt_30000.pt'
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b'irreplaceable-completed-model')
    (scratch/'disposable.bin').write_bytes(b'rebuildable')
    result = {'tasks':[{'artifixer3d_checkpoint':{'path':str(weights),
        'sha256':'sha256:'+hashlib.sha256(weights.read_bytes()).hexdigest(), 'size_bytes':weights.stat().st_size}}]}
    receipt = scratch/'public_scene_artifixer3d_runtime_result.json'
    receipt.write_text(json.dumps(result))
    destination = tmp_path/'output.zip'
    write_output_archive(output, destination)
    with zipfile.ZipFile(destination) as archive:
        assert archive.read(weights.relative_to(output).as_posix()) == weights.read_bytes()
        assert archive.read(receipt.relative_to(output).as_posix()) == receipt.read_bytes()
        assert not any(p.endswith('disposable.bin') for p in archive.namelist())
        assert len(json.loads(archive.read('retained_training_checkpoints.json'))['checkpoints']) == 1
    saved = scratch/'saved-result.json'
    receipt.rename(saved)
    receipt.symlink_to(saved)
    with pytest.raises(RuntimeError, match='training_result_symlink_forbidden'):
        write_output_archive(output, destination)
    receipt.unlink()
    saved.rename(receipt)
    weights.write_bytes(b'tampered')
    with pytest.raises(RuntimeError, match='training_checkpoint_path_invalid'):
        write_output_archive(output, destination)
