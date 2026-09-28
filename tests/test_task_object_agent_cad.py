"""The agent's CAD tool uses the pinned compiler and independent readback."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from blueprint_pipeline import task_object_agent_cad as cad
from blueprint_pipeline.task_object_astra_authoring import AssetAuthoringError


@pytest.mark.parametrize('failure', [None, 'compile', 'dimensions', 'source_changed'])
def test_compiler_exports_and_readback_own_acceptance(tmp_path, monkeypatch, failure):
    checked, commands = [], []
    def sources(*args):
        checked.append(True)
        return {'digest': 'changed' if failure == 'source_changed' and len(checked) > 1 else 'pinned'}
    monkeypatch.setattr(cad, 'verify_cad_sources', sources)
    def readback(path, dimensions, tolerance):
        assert Path(path).name == 'candidate.step'
        assert dimensions == (180, 220, 150) and tolerance == .01
        return {'passed': failure != 'dimensions', 'measured_dimensions_mm': list(dimensions)}
    monkeypatch.setattr(cad, '_read_step', readback)
    class Sandbox:
        def preflight(self): commands.append('preflight')
        def __call__(self, argv, **kwargs):
            commands.append(argv)
            root = Path(kwargs['cwd'])
            assert '--stl' in argv and argv[argv.index('--stl') + 1] == 'candidate.stl'
            assert set(kwargs['env']) == {'PYTHONPATH'}
            (root / 'candidate.step').write_text('deterministic fixture')
            (root / 'candidate.stl').write_text('deterministic fixture')
            return SimpleNamespace(returncode=int(failure == 'compile'), stdout='', stderr='actual compiler error')
    kwargs = dict(program='def gen_step():\n    return Part()\n', output_root=tmp_path / 'candidate',
        request=SimpleNamespace(dimensions_m=(.18, .22, .15), maximum_export_error_m=.00001),
        cad_root=tmp_path / 'pinned-cad', mac_root=tmp_path / 'pinned-mac', sandbox=Sandbox())
    if failure:
        with pytest.raises(AssetAuthoringError, match={'compile': 'actual compiler error',
                'dimensions': 'geometry rejected', 'source_changed': 'source_changed'}[failure]):
            cad.execute_cad_program(**kwargs)
        assert not (kwargs['output_root'] / 'candidate-receipt.json').exists()
    else:
        result = cad.execute_cad_program(**kwargs)
        assert result['passed'] and result['measured_dimensions_m'] == [.18, .22, .15]
        assert result['program']['sha256'] and result['step']['sha256'] and result['stl']['sha256']
        assert commands[0] == 'preflight'
        assert commands[1][1] == str(kwargs['cad_root'] / 'skills/cad/scripts/step')
