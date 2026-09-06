import sys,tempfile,pathlib,json
sys.path.insert(0,str(pathlib.Path.cwd() / 'tests'))
from tests import test_task_evaluation_public_scene_intent_provisioner as t
from pytest import MonkeyPatch
for mode in ['wrong_terms','conversion_drift_real_fields','dependency_conflict']:
 with tempfile.TemporaryDirectory(prefix='fixed-source-audit-',dir='/private/tmp') as d, MonkeyPatch.context() as mp:
  p=pathlib.Path(d).resolve(); retained,owner,extras=t.retained.__wrapped__(p,mp)
  if mode=='wrong_terms': owner['consent']['provider_terms_reference']='terms-v1'
  if mode=='conversion_drift_real_fields':
   conversion=json.loads(retained['standard_splat_conversion_receipt'].read_text()); conversion['repository']['commit']='a'*40; conversion['raw_source_uploaded']=False
   t._write(retained['standard_splat_conversion_receipt'],conversion,'receipt_digest')
   seed=json.loads(retained['accepted_task_request'].read_text())
   for purpose,ref in seed['human_authority']['full_source_provider_disclosure_authorities'].items():
    path=pathlib.Path(ref['path']); auth=json.loads(path.read_text()); auth['source_commit']='a'*40
    t._write(path,auth,'authorization_digest'); seed['human_authority']['full_source_provider_disclosure_authorities'][purpose]=t._ref(path)
   t._write(retained['accepted_task_request'],seed)
  if mode=='dependency_conflict':
   path=p/'config'/'task-evaluation-public-scene-machinery.json'; path.parent.mkdir(); path.write_text('{}')
  try:
   result,intents,bindings,machinery=t._provision(p,retained,owner,now=extras['now'])
   t._bind_runtime_env(mp,intents,extras['commit'])
   config=t._progression_config(p,intents,bindings,machinery,retained['release_binding'])
   print(mode,json.dumps(t.engine.process_scene_intents(config_path=config)))
  except Exception as exc:
   print(mode,type(exc).__name__,str(exc),'intent_count',len(list((p/'intents').glob('*/intent.json'))))
