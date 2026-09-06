from pathlib import Path
from tempfile import TemporaryDirectory
from tests.test_task_evaluation_scene_terminal_reconciler import _env,_receipts,_write,OTHER_COMMIT,_blocked_projection
from tests.test_task_evaluation_scene_terminal_result_index import _source_outputs,_index
from blueprint_pipeline import task_evaluation_scene_terminal_reconciler as r
from blueprint_pipeline import task_evaluation_scene_progression as engine
from blueprint_pipeline import task_evaluation_scene_progression_state as state
from blueprint_pipeline import task_evaluation_scene_intake as intake
import json
for mode in ['wrong_commit','failed_notification','nonexistent_uri_via_real_index','invalid_sync_poisons_retry','expired_blocked_status']:
 with TemporaryDirectory() as d:
  base=Path(d).resolve(); e=_env(base)
  if mode in ['nonexistent_uri_via_real_index','invalid_sync_poisons_retry']:
   src,projection,pub=_source_outputs(base,e)
   if mode=='nonexistent_uri_via_real_index':
    pub['uri']='https://unpublished.invalid/never-published.json'; pub['size_bytes']=123456
    _index(e,src,pub)
   else:
    path=src/'policy_canary_webapp_sync.json'; good=json.loads(path.read_text()); bad=dict(good,status='skipped'); _write(path,bad)
    print(mode,'first_index',_index(e,src,pub)['status'])
    _write(path,good)
    try: _index(e,src,pub)
    except Exception as ex: print(mode,'corrected_retry',str(ex))
    continue
  else: _receipts(e,projection=_blocked_projection() if mode=='expired_blocked_status' else None)
  root=Path(e['config']['terminal_result_root'])/e['intent_id']
  if mode=='wrong_commit':
   p=json.loads((root/'launch_profile.json').read_text()); p['source_commit']=OTHER_COMMIT; _write(root/'launch_profile.json',p,'profile_digest'); p=json.loads((root/'launch_profile.json').read_text())
   q=json.loads((root/'launch_request.json').read_text()); q['source_commit']=OTHER_COMMIT; q['launch_profile_digest']=p['profile_digest']; _write(root/'launch_request.json',q)
   c=json.loads((root/'provider_zero_closure.json').read_text()); c['launch_profile_digest']=p['profile_digest']; _write(root/'provider_zero_closure.json',c,'provider_zero_receipt_digest')
  if mode=='failed_notification':
   p=json.loads((root/'policy_canary_webapp_sync.json').read_text()); p['notification_delivery']['status']='failed'; _write(root/'policy_canary_webapp_sync.json',p)
  if mode=='expired_blocked_status':
   state.advance(e['directory'],e['intent'],None,status='awaiting_execution',phase='scene_configuration',state={'activation':{'provider_allocation_performed':False},'attempt_id':'source-1'},now=e['now'])
   expired=e['intent']['request']['execution']['expires_at_epoch']+1
   config=dict(e['config'],factory_output_root=str(base/'factory'))
   progress=engine._advance_intent(e['directory'],e['intent'],config,e['release'],resolver=None,publisher=None,submitter=None,status_reader=None,activation_provisioner=None,now=expired)
   status=intake.scene_intent_status(queue_root=e['config']['intent_root'],intent_id=e['intent_id'],now=expired)
   print(mode,'progress',progress['status'],progress['blockers'],'website_status',status['status'],status['blockers']); continue
  result=r.reconcile_terminal_owner_result(intent=e['intent'],config=e['config'],release=e['release'],now=e['now'])
  print(mode,None if result is None else (result['status'],result['blockers'],result['result_reference']))
