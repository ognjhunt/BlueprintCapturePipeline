from pathlib import Path
from tempfile import TemporaryDirectory
from tests.test_task_evaluation_scene_terminal_reconciler import _env, _receipts, _write, OTHER_COMMIT
from blueprint_pipeline import task_evaluation_scene_terminal_reconciler as r
import json
for mode in ['wrong_commit', 'failed_notification', 'fabricated_publication']:
 with TemporaryDirectory() as d:
  e = _env(Path(d).resolve()); _receipts(e)
  root = Path(e['config']['terminal_result_root']) / e['intent_id']
  if mode == 'wrong_commit':
   p = json.loads((root/'launch_profile.json').read_text()); p['source_commit'] = OTHER_COMMIT
   _write(root/'launch_profile.json', p, 'profile_digest'); p = json.loads((root/'launch_profile.json').read_text())
   q = json.loads((root/'launch_request.json').read_text()); q['source_commit'] = OTHER_COMMIT; q['launch_profile_digest'] = p['profile_digest']; _write(root/'launch_request.json', q)
   c = json.loads((root/'provider_zero_closure.json').read_text()); c['launch_profile_digest'] = p['profile_digest']; _write(root/'provider_zero_closure.json', c, 'provider_zero_receipt_digest')
  elif mode == 'failed_notification':
   p = json.loads((root/'policy_canary_webapp_sync.json').read_text()); p['notification_delivery']['status'] = 'failed'; _write(root/'policy_canary_webapp_sync.json', p)
  else:
   p = json.loads((root/'terminal_result_publication.json').read_text()); p['schema_version']='wrong'; p['run_id']='unrelated-run'; p['uri']='https://unrelated.invalid/missing.json'; p.pop('provider_allocated'); _write(root/'terminal_result_publication.json', p)
  result = r.reconcile_terminal_owner_result(intent=e['intent'], config=e['config'], release=e['release'], now=e['now'])
  print(mode, result['status'], result['blockers'])
