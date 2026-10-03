import sys,json,tempfile,pathlib
sys.path.insert(0,str(pathlib.Path.cwd() / 'tests'))
from tests import test_scene_preparation_installation as t
from pytest import MonkeyPatch
from blueprint_pipeline import task_evaluation_scene_progression as progression
with tempfile.TemporaryDirectory(prefix='entrypoint-audit-',dir='/private/tmp') as d, MonkeyPatch.context() as mp:
 p=pathlib.Path(d).resolve()
 old_path,*_=t._config(p,mp,source_kind='mesh',real_destination=True)
 old=json.loads(old_path.read_text()); machinery=json.loads(pathlib.Path(old['completed_source_machinery_path']).read_text())
 receipt=t._install_bootstrap(p,machinery,old['capture_store_root'],True,p/'authorized')
 config=json.loads(pathlib.Path(receipt['config']['path']).read_text())
 print('activation_enabled',config['activation_enabled'],'preparation_worker_present','preparation_worker' in config)
 try:
  progression.main(['--config',receipt['config']['path']])
 except Exception as exc:
  print('production_main_error',type(exc).__name__,str(exc))
