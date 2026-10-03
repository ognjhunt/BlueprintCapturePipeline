import json
from pathlib import Path
from tempfile import TemporaryDirectory
from pytest import MonkeyPatch
from tests.test_task_evaluation_controls_autoprovision import _configured_scene
from blueprint_pipeline import task_evaluation_scene_progression as e
from blueprint_pipeline import task_evaluation_scene_intake as intake
from tests.test_task_evaluation_configured_controls_continuation_provisioning import NOW, COMMIT
with TemporaryDirectory(prefix='activation-link-audit-') as d, MonkeyPatch.context() as mp:
 root=Path(d).resolve(); mp.setenv(intake.CLIENTS_ENV,'webapp'); args=_configured_scene(root)
 directory=args['scene_root']/args['intent_id']; intent=json.loads((directory/'intent.json').read_text())
 envelope_path=next(p for p in (args['preparation_queue_root']/'materialized').glob('*.json') if json.loads(p.read_text())['request'].get('scene_intent_digest')==intent['intent_digest'])
 envelope=json.loads(envelope_path.read_text()); observed={'request':envelope['request'],'request_digest':envelope['request_digest'],'result_filename':envelope_path.name}
 config={'activation_enabled':False,'intent_root':str(args['scene_root'])}
 ref=e._link(intent=intent,attempt={'source_commit':COMMIT},observed=observed,directory=directory,config=config,now=NOW.timestamp())
 link=json.loads(Path(ref['path']).read_text()); print('prepared link has scene_configuration_attempt:', 'scene_configuration_attempt' in link)
 output=root/'activation-output'; output.mkdir(); spend=args['catalog']['bindings']['franka-droid']['project_spend_reconciliation']['path']
 e._put(output/'activation_inputs.json',intake._seal({'spend':e.record(Path(spend)),'issued_at_epoch':NOW.timestamp(),'link_digest':link['link_digest']},'receipt_digest'))
 config['activation_enabled']=True
 try: e._activation(intent=intent,link=link,config=config,output=output,now=NOW.timestamp(),provisioner=lambda **kw: (_ for _ in ()).throw(AssertionError('unreachable')))
 except Exception as exc: print('same retained link after activation authorized:',type(exc).__name__,str(exc))
