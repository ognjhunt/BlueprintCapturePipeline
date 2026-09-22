import sys, tempfile, pathlib, json
sys.path.insert(0, str(pathlib.Path.cwd() / 'tests'))
import test_task_evaluation_public_scene_intent_provisioner as t
from pytest import MonkeyPatch

def exercise(mode):
 with tempfile.TemporaryDirectory(prefix='source-audit-case-',dir='/private/tmp') as d, MonkeyPatch.context() as mp:
  p=pathlib.Path(d).resolve(); retained,owner,extras=t.retained.__wrapped__(p,mp)
  if mode == 'conversion_release_drift':
   conversion=json.loads(retained['standard_splat_conversion_receipt'].read_text())
   conversion['repository']['commit']='a'*40
   t._write(retained['standard_splat_conversion_receipt'],conversion,'receipt_digest')
  original=owner['consent']['provider_terms_reference']
  result,intents,bindings,machinery=t._provision(p,retained,owner,now=extras['now'])
  intent=json.loads((intents/result['intent_id']/'intent.json').read_text())
  t._bind_runtime_env(mp,intents,extras['commit'])
  if mode == 'discovered_prefix':
   # Isolate the boundary after discovery: production resolver output has no provider_zero.
   mp.setattr(t.factory,'_prefix_candidates',lambda *a:[{'retained_candidate':True}])
  config=t._progression_config(p,intents,bindings,machinery,retained['release_binding'])
  run=t.engine.process_scene_intents(config_path=config)
  print(json.dumps({'case':mode,'provision_status':result['status'],
    'consent_input_reference':original,
    'consent_staged_reference':intent['request']['consent']['provider_terms_reference'],
    'consent_reference_overwritten':original!=intent['request']['consent']['provider_terms_reference'],
    'worker':run},sort_keys=True))
for mode in ('baseline_terms_substitution','conversion_release_drift','discovered_prefix'):
 exercise(mode)
